"""The mandatory comparison: ``pdmcore``'s 63 engineered features + a shallow head.

The honest question in this field is not "which CNN wins" but "does any of them
beat good features at all", and the only way to answer it is to run both on
identical windows. Every model in ``run_matrix.py`` sees the same 2048-sample
windows under the same split; the only difference is what happens to them next.

What the feature path gets that the CNN does not, stated plainly
---------------------------------------------------------------
``pdmcore.features.FeatureExtractor`` is handed two pieces of prior knowledge the
raw-waveform models have no access to:

1. **The shaft rate of every window**, which the datasets publish and which a
   real deployment reads from a tacho or a VFD setpoint. Order features and
   bearing-fault bands are meaningless without it -- ``extract`` raises rather
   than guessing.
2. **The bearing geometry**, which sets where BPFO / BPFI / BSF / FTF are. A real
   deployment knows this from the asset register.

That is an asymmetry, and it is the asymmetry a plant actually has. It is also
*why* engineered features might transfer across rigs where a CNN cannot: the
kinematic normalisation is done analytically instead of being learned from one
rig's resonance map. :func:`extract` therefore takes a ``geometry`` argument so
the cross-dataset arm can also be run with the **source** domain's geometry --
the ablation that separates "engineered features transfer" from "known kinematics
transfer". Both numbers are in the matrix.

Heads
-----
``logreg``   multinomial logistic regression, L2. A linear decision on 63
             features: 640 parameters for 10 classes, 84x smaller than WDCNN and
             24 000x smaller than the upstream CNN.
``mlp``      ``sklearn.neural_network.MLPClassifier`` with (64, 32) hidden units.
             Shallow, ~6 k parameters.

Both are standardised by ``pdmcore.models.anomaly.StandardScaler`` -- median/MAD
with a log transform on the heavy-tailed strictly-positive features, and the same
20-sigma clip the detectors use -- rather than by sklearn's mean/std, so the
feature treatment matches what the deployed one-class detectors already do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from pdmcore.dsp.bearing import BEARING_CATALOG, BearingGeometry
from pdmcore.features import FeatureConfig, FeatureExtractor
from pdmcore.models.anomaly import StandardScaler

from cwru import WindowSet

__all__ = ["HEADS", "feature_config", "extract", "FeatureClassifier"]

HEADS = ("logreg", "mlp")

#: Fixed demodulation band for the envelope analysis. Explicit rather than
#: chosen per acquisition by ``select_demodulation_band``: the kurtogram search
#: costs more than the rest of the vector put together, and -- more importantly
#: -- a band chosen per window is a per-window decision the firmware cannot make,
#: so a result obtained with it would not be a result about the deployed path.
#: 2-5 kHz covers the CWRU drive-end resonances and sits inside Nyquist at 12 kHz.
DEMOD_BAND_HZ = (2000.0, 5000.0)


def feature_config(fs_hz: float, geometry: BearingGeometry) -> FeatureConfig:
    """The extractor configuration used for every feature result on this branch."""
    return FeatureConfig(
        fs_hz=fs_hz,
        n_fft=2048,
        window_name="hann",
        bearing=geometry,
        demod_band_hz=DEMOD_BAND_HZ,
        envelope_n_fft=1024,
        envelope_decimate_to_hz=2000.0,
        velocity_band_hz=(10.0, 1000.0),
    )


def extract(ws: WindowSet, *, geometry: Optional[BearingGeometry] = None,
            bearing_name: str = "SKF6205") -> Tuple[np.ndarray, Tuple[str, ...]]:
    """``(n_windows, 63)`` feature matrix plus the feature names.

    One 2048-point FFT and one 1024-point envelope FFT per window, which is
    exactly what the firmware does per acquisition -- so the cost reported in
    ``production/supervised_cnn/benchmark.py`` is the cost of the deployed path.

    Caveat recorded here because it affects the numbers: 2048 samples at 12 kHz
    is 171 ms, about 5 shaft revolutions, and the envelope spectrum after
    decimation to 2 kHz with a 1024-point FFT has 1.95 Hz resolution. BPFO is
    107 Hz, so roughly 18 impacts fall in the window. That is thin for harmonic
    SNR estimation and it handicaps the bearing block of the vector. The window
    is kept at 2048 anyway because the CNNs' published input length is 2048 and a
    comparison in which one model sees more signal than the other is not a
    comparison.
    """
    geom = geometry if geometry is not None else BEARING_CATALOG[bearing_name]
    ex = FeatureExtractor(feature_config(ws.fs_hz, geom))
    rows = np.empty((len(ws), len(ex.names)), dtype=np.float64)
    for i in range(len(ws)):
        rows[i] = ex.extract(ws.X[i], float(ws.shaft_hz[i])).vector
    return rows, ex.names


@dataclass
class FeatureClassifier:
    """63 engineered features -> robust standardiser -> shallow head.

    Exposes ``score(X)`` on **feature vectors**, matching the
    ``pdmcore.models.anomaly`` convention, so it drops into the same place in
    ``pdmedge`` as a one-class detector does.
    """

    head: str = "logreg"
    seed: int = 0
    max_iter: int = 2000
    feature_names: Tuple[str, ...] = ()
    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: object = None
    classes_: Optional[np.ndarray] = None
    healthy_class_: str = "healthy"

    def _build(self):
        if self.head == "logreg":
            return LogisticRegression(
                max_iter=self.max_iter, C=1.0, class_weight="balanced",
                random_state=self.seed,
            )
        if self.head == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=self.max_iter, alpha=1e-3,
                random_state=self.seed, early_stopping=False,
            )
        raise ValueError(f"unknown head {self.head!r}; expected one of {HEADS}")

    def fit(self, F: np.ndarray, y: Sequence) -> "FeatureClassifier":
        F = np.asarray(F, dtype=np.float64)
        y = np.asarray([str(v) for v in y])
        self.classes_ = np.unique(y)
        Z = self.scaler.fit_transform(F)
        self.clf = self._build()
        self.clf.fit(Z, y)
        return self

    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("FeatureClassifier is not fitted")
        Z = self.scaler.transform(np.asarray(F, dtype=np.float64))
        p = self.clf.predict_proba(Z)
        # sklearn orders columns by its own classes_; realign to ours.
        order = np.searchsorted(self.classes_, np.asarray([str(c) for c in self.clf.classes_]))
        out = np.zeros_like(p)
        out[:, order] = p
        return out

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_proba(F), axis=1)]

    def score(self, F: np.ndarray) -> np.ndarray:
        """Higher = more anomalous: ``1 - P(healthy)``."""
        assert self.classes_ is not None
        where = np.flatnonzero(self.classes_ == self.healthy_class_)
        if where.size != 1:
            raise RuntimeError(
                f"no unique {self.healthy_class_!r} class in {list(self.classes_)}; "
                "a one-vs-rest anomaly score is not defined for this label space"
            )
        return 1.0 - self.predict_proba(F)[:, int(where[0])]

    @property
    def n_params(self) -> int:
        if self.clf is None:
            return 0
        if self.head == "logreg":
            return int(self.clf.coef_.size + self.clf.intercept_.size)
        return int(sum(w.size for w in self.clf.coefs_) + sum(b.size for b in self.clf.intercepts_))

    def state(self) -> Dict[str, object]:
        if self.clf is None or self.classes_ is None:
            raise RuntimeError("FeatureClassifier is not fitted")
        if self.head != "logreg":
            raise NotImplementedError(
                "only the logreg head is serialisable here; the MLP head is a "
                "prototype-only comparison and is not exported, because an "
                "exported model that cannot be reloaded is worse than none"
            )
        return {
            "head": self.head,
            "classes": [str(c) for c in self.classes_],
            "feature_names": list(self.feature_names),
            "scaler": self.scaler.to_dict(),
            "coef": np.asarray(self.clf.coef_).tolist(),
            "intercept": np.asarray(self.clf.intercept_).tolist(),
            "sklearn_classes": [str(c) for c in self.clf.classes_],
        }
