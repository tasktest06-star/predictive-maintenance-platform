"""The WDCNN architecture family, and the two upstream models, as classifiers.

What is reproduced here, and what is not
----------------------------------------
``wdcnn``
    Zhang et al., *A New Deep Learning Model for Fault Diagnosis with Good
    Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals*, Sensors
    17(2):425, 2017. Reproduced faithfully in shape: 2048 raw samples in; a wide
    first layer (64 taps, stride 16) whose stated purpose is to act as a learned
    band-pass bank at a coarse time resolution; five conv stages with
    BN + ReLU + max-pool; FC-100; softmax. 54 k parameters.
``ticnn``
    Zhang et al., *Deep Convolutional Neural Networks with Wide First-layer
    Kernels for Fault Diagnosis under Varying Working Conditions*, MSSP 100:439,
    2018. The architectural delta over WDCNN that this file implements is the one
    the paper's claims rest on: **dropout applied to the raw input** plus
    train-time amplitude and additive-noise augmentation. TICNN's "ensemble of
    small-batch models" is NOT implemented.
``upstream_1dcnn``
    ``impl/ml-supervised-1dcnn``'s ``BearingFaultCNN1D``, transcribed layer for
    layer from ``model.py`` on that branch, including its preprocessing: the
    input is an rfft magnitude zero-padded back to 2048 and per-window
    standardised. 1.32 M parameters. Transcribed so the comparison is like for
    like; the branch's own PyTorch code is also run directly, see
    ``prototype/supervised_cnn/run_upstream.py``.

Where this deviates from the papers, stated once
------------------------------------------------
* **Optimiser and schedule.** Adam at 1e-3 with cosine decay, batch 64, and an
  epoch count set by this host's CPU budget rather than by the papers (WDCNN's
  original schedule is not published in enough detail to reproduce anyway).
  Under-training suppresses the *random-split* accuracy more than the grouped
  one, so it works against this branch's own hypothesis rather than for it --
  which is the direction an honest shortcut should point.
* **No ensembling, no early stopping.** Early stopping needs a validation fold,
  and in CWRU every fold that is not the test fold shares a physical bearing with
  it. The upstream branches early-stop on the test fold itself; this does not.
* **Per-window z-score normalisation is the default**, because it is what the
  literature does. It also discards the overall vibration level, which is the
  single most diagnostic scalar a condition-monitoring engineer has. The
  ``normalise="global"`` option keeps it, and the matrix reports both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nn import (Adam, DTYPE, BatchNorm1d, Conv1d, Dense, Dropout, Flatten, MaxPool1d,
                ReLU, Sequential, softmax_cross_entropy)

__all__ = ["CnnConfig", "build_model", "CnnClassifier", "ARCHITECTURES"]

ARCHITECTURES = ("wdcnn", "ticnn", "upstream_1dcnn")


@dataclass
class CnnConfig:
    arch: str = "wdcnn"
    input_length: int = 2048
    n_classes: int = 10
    epochs: int = 24
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    #: "window" = per-window z-score (literature default, discards level);
    #: "global" = train-set mean/std per sample position-independent (keeps level).
    normalise: str = "window"
    #: TICNN: dropout probability on the raw input, and train-time augmentation.
    input_dropout: float = 0.0
    augment_amplitude: float = 0.0
    augment_noise_snr_db: Optional[float] = None
    class_weight: bool = True
    seed: int = 0

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def build_model(cfg: CnnConfig) -> Sequential:
    """Construct the architecture. Parameter counts are asserted in the tests."""
    s = cfg.seed
    if cfg.arch in ("wdcnn", "ticnn"):
        layers: List = []
        if cfg.input_dropout > 0:
            layers.append(Dropout(cfg.input_dropout, seed=s + 90))
        # Stage 1: the wide first kernel. 64 taps at stride 16 over 2048 samples.
        layers += [Conv1d(1, 16, 64, stride=16, padding="same", seed=s + 1),
                   BatchNorm1d(16), ReLU(), MaxPool1d(2)]
        for i, (ci, co) in enumerate(((16, 32), (32, 64), (64, 64))):
            layers += [Conv1d(ci, co, 3, stride=1, padding="same", seed=s + 10 + i),
                       BatchNorm1d(co), ReLU(), MaxPool1d(2)]
        layers += [Conv1d(64, 64, 3, stride=1, padding="valid", seed=s + 20),
                   BatchNorm1d(64), ReLU(), MaxPool1d(2), Flatten()]
        # 2048 -> 128 -> 64 -> 32 -> 16 -> 8 -> (valid 3 taps) 6 -> pool 3
        flat = 64 * 3
        layers += [Dense(flat, 100, seed=s + 30), ReLU(), Dense(100, cfg.n_classes, seed=s + 31)]
        return Sequential(layers)

    if cfg.arch == "upstream_1dcnn":
        # Transcribed from impl/ml-supervised-1dcnn:impl/ml_supervised_1dcnn/model.py
        # Conv1DBlock = Conv1d(bias=False) + BN + ReLU + MaxPool(2) + Dropout(0.25)
        layers = []
        spec = ((1, 64, 64, 2), (64, 128, 32, 1), (128, 256, 16, 1), (256, 256, 8, 1))
        for i, (ci, co, k, st) in enumerate(spec):
            layers += [Conv1d(ci, co, k, stride=st, padding="same", seed=s + 40 + i),
                       BatchNorm1d(co), ReLU(), MaxPool1d(2), Dropout(0.25, seed=s + 50 + i)]
        layers += [_GlobalAvgPool1d(), Dense(256, 128, seed=s + 60), ReLU(),
                   Dropout(0.5, seed=s + 61), Dense(128, cfg.n_classes, seed=s + 62)]
        return Sequential(layers)

    raise ValueError(f"unknown arch {cfg.arch!r}; expected one of {ARCHITECTURES}")


class _GlobalAvgPool1d(Flatten):
    """Mean over the length axis: ``(N, C, L) -> (N, C)``."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.mean(axis=2)

    def backward(self, g: np.ndarray) -> np.ndarray:
        n, c, length = self._shape
        return np.repeat(g[:, :, None], length, axis=2) / length


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _rfft_magnitude_padded(X: np.ndarray) -> np.ndarray:
    """The upstream branch's ``segment_and_fft``, reproduced exactly.

    Hann window, rfft magnitude (1025 bins for a 2048 window), zero-padded back
    to 2048, then per-window standardised. Note the consequence: **half of the
    model's 2048-long input is structurally zero**, and the wide first kernel --
    designed in the WDCNN paper to act as a band-pass bank over *raw time
    samples* -- is being slid over a magnitude spectrum instead.
    """
    n = X.shape[1]
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(X * w, n=n, axis=1))
    out = np.zeros((X.shape[0], n), dtype=np.float64)
    out[:, : mag.shape[1]] = mag
    return (out - out.mean(axis=1, keepdims=True)) / (out.std(axis=1, keepdims=True) + 1e-8)


@dataclass
class _Normaliser:
    mode: str
    mu: float = 0.0
    sd: float = 1.0

    def fit(self, X: np.ndarray) -> "_Normaliser":
        if self.mode == "global":
            self.mu, self.sd = float(X.mean()), float(X.std()) or 1.0
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "window":
            return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
        if self.mode == "global":
            return (X - self.mu) / self.sd
        if self.mode == "none":
            return X
        raise ValueError(f"unknown normalise mode {self.mode!r}")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
@dataclass
class CnnClassifier:
    """Fit / predict wrapper with the repo's ``score`` convention bolted on.

    ``predict_proba`` is the classifier interface; ``score(X)`` returns
    ``1 - P(healthy)`` so the object is interchangeable with a
    ``pdmcore.models.anomaly`` detector in the "higher = more anomalous"
    direction the health chain and ``pdmedge`` assume.
    """

    cfg: CnnConfig = field(default_factory=CnnConfig)
    classes_: Optional[np.ndarray] = None
    model: Optional[Sequential] = None
    history_: List[float] = field(default_factory=list)
    fit_seconds_: float = 0.0
    healthy_class_: str = "healthy"

    def _prep(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
        if self.cfg.arch == "upstream_1dcnn":
            Z = _rfft_magnitude_padded(np.asarray(X, dtype=np.float64))
        else:
            if fitting:
                self._norm = _Normaliser(self.cfg.normalise).fit(np.asarray(X, dtype=np.float64))
            Z = self._norm.apply(np.asarray(X, dtype=np.float64))
        return np.ascontiguousarray(Z[:, None, :], dtype=DTYPE)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CnnClassifier":
        t0 = time.time()
        y = np.asarray([str(v) for v in y])
        self.classes_ = np.unique(y)
        cfg = self.cfg
        if cfg.n_classes != self.classes_.size:
            cfg = CnnConfig(**{**cfg.as_dict(), "n_classes": int(self.classes_.size)})
            self.cfg = cfg
        yi = np.searchsorted(self.classes_, y)
        Z = self._prep(X, fitting=True)
        self.model = build_model(cfg)
        opt = Adam(self.model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)
        cw = None
        if cfg.class_weight:
            counts = np.array([np.count_nonzero(yi == k) for k in range(cfg.n_classes)], dtype=float)
            cw = (counts.sum() / (cfg.n_classes * np.maximum(counts, 1)))
        n = Z.shape[0]
        self.history_ = []
        for ep in range(cfg.epochs):
            opt.lr = cfg.lr * 0.5 * (1.0 + np.cos(np.pi * ep / max(1, cfg.epochs)))
            self.model.train()
            order = rng.permutation(n)
            tot = 0.0
            for b0 in range(0, n, cfg.batch_size):
                idx = order[b0 : b0 + cfg.batch_size]
                if idx.size < 2:
                    continue  # batch norm on a single sample has zero variance
                xb = Z[idx]
                if cfg.augment_amplitude > 0:
                    amp = 1.0 + cfg.augment_amplitude * rng.standard_normal((idx.size, 1, 1))
                    xb = xb * amp.astype(DTYPE)
                if cfg.augment_noise_snr_db is not None:
                    sig = np.sqrt(np.mean(xb**2, axis=2, keepdims=True))
                    nstd = sig * np.asarray(10 ** (-cfg.augment_noise_snr_db / 20.0), dtype=DTYPE)
                    xb = xb + nstd * rng.standard_normal(xb.shape).astype(DTYPE)
                logits = self.model.forward(xb)
                loss, g = softmax_cross_entropy(logits, yi[idx], class_weight=cw)
                self.model.backward(g)
                opt.step()
                tot += loss * idx.size
            self.history_.append(tot / n)
        self.model.eval()
        self.fit_seconds_ = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray, batch: int = 256) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CnnClassifier is not fitted")
        self.model.eval()
        Z = self._prep(X, fitting=False)
        out = []
        for b0 in range(0, Z.shape[0], batch):
            logits = self.model.forward(Z[b0 : b0 + batch])
            z = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(z)
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.vstack(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous: ``1 - P(healthy)``.

        If the training label space had no healthy class this cannot be formed,
        and it raises rather than returning something that looks like a score.
        """
        assert self.classes_ is not None
        where = np.flatnonzero(self.classes_ == self.healthy_class_)
        if where.size != 1:
            raise RuntimeError(
                f"no unique {self.healthy_class_!r} class in {list(self.classes_)}; "
                "a one-vs-rest anomaly score is not defined for this label space"
            )
        return 1.0 - self.predict_proba(X)[:, int(where[0])]

    @property
    def n_params(self) -> int:
        return 0 if self.model is None else self.model.n_params

    def state(self) -> Dict[str, object]:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("CnnClassifier is not fitted")
        st: Dict[str, object] = {
            "cfg": self.cfg.as_dict(),
            "classes": [str(c) for c in self.classes_],
            "layers": self.model.state(),
        }
        if self.cfg.arch != "upstream_1dcnn":
            st["normaliser"] = {"mode": self._norm.mode, "mu": self._norm.mu, "sd": self._norm.sd}
        return st

    @classmethod
    def from_state(cls, st: Dict) -> "CnnClassifier":
        cfg = CnnConfig(**st["cfg"])
        obj = cls(cfg=cfg, classes_=np.asarray(st["classes"]))
        obj.model = build_model(cfg).load_state(st["layers"]).eval()
        nz = st.get("normaliser")
        obj._norm = _Normaliser(nz["mode"], nz["mu"], nz["sd"]) if nz else _Normaliser("window")
        return obj
