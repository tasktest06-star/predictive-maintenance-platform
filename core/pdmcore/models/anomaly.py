"""Unsupervised detectors -- the only ones deployable on day one.

A new customer has thousands of healthy hours and zero labelled failures. Every
supervised model in ``models/`` is therefore a *later* capability, trained once
field failures accumulate (or transferred from public datasets and then
validated). The cold-start path is: fit a normal model per asset during a
commissioning window, alarm on deviation, and let the first few confirmed
failures bootstrap the classifier.

Detectors here, in the order they should be tried:

``MahalanobisDetector``   Full-covariance Gaussian on standardised features.
                          Cheap, closed-form, and surprisingly hard to beat on
                          60-dimensional engineered features. Supports shrinkage
                          because with 200 commissioning samples and 60
                          features the empirical covariance is near-singular.
``PCAResidualDetector``   Reconstruction error from the top-k principal
                          subspace. Interpretable: the residual decomposes per
                          feature, which gives "why" for free.
``RobustThresholdDetector`` Per-feature robust z-scores (median/MAD) combined by
                          max. Dumb, transparent, and the right default when a
                          maintenance team must be able to explain every alarm.
``DeepSVDDDetector``      One-class deep SVDD, pure NumPy with Adam. Beats
                          reconstruction autoencoders on the public acoustic
                          benchmarks at a fraction of the parameters, which is
                          what makes it interesting for an MCU.

All detectors share the same interface -- ``fit(X)``, ``score(X)`` (higher =
more anomalous), ``threshold_at_fpr`` -- so they are interchangeable in the
pipeline, and all expose ``explain(x)`` for per-feature attribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StandardScaler",
    "AnomalyDetector",
    "MahalanobisDetector",
    "PCAResidualDetector",
    "RobustThresholdDetector",
    "DeepSVDDDetector",
    "build_detector",
]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
@dataclass
class StandardScaler:
    """Robust standardiser: median and MAD, not mean and std.

    Commissioning windows are never perfectly clean -- a technician bumps the
    machine, a VFD trips, somebody runs the asset dry for a minute. A single
    outlier at 50x nominal inflates the std enough to hide every subsequent real
    anomaly. Median/MAD is unaffected by up to 50 % contamination, at the cost
    of ~25 % efficiency on genuinely clean Gaussian data. That trade is correct
    for field data.

    Features whose MAD is zero (constant during commissioning -- e.g. an ISO
    zone index that never left A) get unit scale instead of exploding.
    """

    centre: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    use_log: Optional[np.ndarray] = None

    #: MAD -> sigma for a normal distribution.
    MAD_TO_SIGMA = 1.4826

    def fit(self, X: np.ndarray, *, log_skew_threshold: float = 2.0) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("need a 2-D array with >= 2 rows")
        # Heavy-tailed, strictly positive features (band RMS, SNR ratios) are
        # log-transformed first: their distribution is roughly log-normal, and a
        # z-score on the raw value calls every busy hour an anomaly.
        pos = np.all(X > 0, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            med = np.median(X, axis=0)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            skew = np.where(std > 0, np.mean((X - mean) ** 3, axis=0) / np.maximum(std, 1e-12) ** 3, 0.0)
        self.use_log = pos & (np.abs(skew) > log_skew_threshold)
        Xt = self._apply_log(X)
        self.centre = np.median(Xt, axis=0)
        mad = np.median(np.abs(Xt - self.centre), axis=0) * self.MAD_TO_SIGMA
        fallback = np.std(Xt, axis=0)
        mad = np.where(mad > 1e-12, mad, fallback)
        self.scale = np.where(mad > 1e-12, mad, 1.0)
        del med
        return self

    def _apply_log(self, X: np.ndarray) -> np.ndarray:
        if self.use_log is None or not np.any(self.use_log):
            return np.asarray(X, dtype=np.float64)
        Xt = np.asarray(X, dtype=np.float64).copy()
        cols = np.where(self.use_log)[0]
        Xt[:, cols] = np.log(np.maximum(Xt[:, cols], 1e-12))
        return Xt

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.centre.size == 0:
            raise RuntimeError("scaler is not fitted")
        if X.shape[1] != self.centre.size:
            raise ValueError(f"expected {self.centre.size} features, got {X.shape[1]}")
        return (self._apply_log(X) - self.centre) / self.scale

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.fit(X, **kwargs).transform(X)

    def to_dict(self) -> Dict:
        return {
            "centre": self.centre.tolist(),
            "scale": self.scale.tolist(),
            "use_log": self.use_log.tolist() if self.use_log is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StandardScaler":
        return cls(
            centre=np.asarray(d["centre"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            use_log=np.asarray(d["use_log"], dtype=bool) if d.get("use_log") is not None else None,
        )


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Common interface. ``score`` is "higher = more anomalous", always."""

    kind = "base"

    def __init__(self, feature_names: Optional[Sequence[str]] = None):
        self.feature_names: Tuple[str, ...] = tuple(feature_names or ())
        self.scaler = StandardScaler()
        self._fitted = False
        self.train_scores_: Optional[np.ndarray] = None

    # -- to implement ------------------------------------------------------
    def _fit(self, Z: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _score(self, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- public ------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = self.scaler.fit_transform(X)
        self._fit(Z)
        self._fitted = True
        self.train_scores_ = self._score(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return self._score(self.scaler.transform(np.atleast_2d(np.asarray(X, dtype=np.float64))))

    def threshold_at_fpr(self, fpr: float = 0.01) -> float:
        """Threshold giving ``fpr`` false alarms on the *training* (healthy) set.

        This is the honest way to set an industrial alarm: the customer agrees a
        tolerable nuisance-alarm rate (typically "no more than one per sensor per
        month", i.e. ~0.1 % of hourly acquisitions), and the threshold follows
        from the healthy distribution rather than from a number someone liked.
        With fewer than ``1/fpr`` training samples the quantile is extrapolated
        from the tail by a Gumbel fit, because the empirical quantile simply does
        not exist.
        """
        if self.train_scores_ is None:
            raise RuntimeError("detector is not fitted")
        if not 0.0 < fpr < 1.0:
            raise ValueError("fpr must be in (0, 1)")
        s = np.sort(self.train_scores_)
        n = s.size
        if n * fpr >= 1.0:
            return float(np.quantile(s, 1.0 - fpr))
        tail = s[max(0, n - max(10, n // 5)) :]
        mu = float(np.mean(tail))
        sigma = float(np.std(tail))
        if sigma <= 0:
            return float(s[-1])
        # Gumbel upper-tail extrapolation.
        beta = sigma * np.sqrt(6.0) / np.pi
        loc = mu - 0.5772 * beta
        return float(loc - beta * np.log(-np.log(1.0 - fpr)))

    def explain(self, x: np.ndarray, top_k: int = 5, *, normalise: bool = True) -> List[Tuple[str, float]]:
        """Per-feature contribution to the score, largest first.

        With ``normalise=True`` (the default) each contribution is a **fraction
        of the total score**, so the numbers are bounded, comparable between
        detectors, and directly presentable: "72 % of this alarm is band 10 RMS".
        Raw contributions are in whatever units the detector's internal
        statistic uses -- for an ill-conditioned Mahalanobis precision matrix
        those can reach 1e6 and look like a bug, which is why normalising is the
        default.

        Operations teams do not act on "anomaly score 7.3"; they act on "band 10
        RMS and outer-race SNR carry 85 % of the deviation", which is the same
        information in a form a millwright can take to the machine.
        """
        z = self.scaler.transform(np.atleast_2d(x))[0]
        contrib = np.asarray(self._contributions(z), dtype=np.float64)
        if normalise:
            total = float(np.sum(np.abs(contrib)))
            contrib = contrib / total if total > 0 else contrib
        names = self.feature_names or tuple(f"f{i}" for i in range(z.size))
        order = np.argsort(-np.abs(contrib))[:top_k]
        return [(names[i], float(contrib[i])) for i in order]

    def _contributions(self, z: np.ndarray) -> np.ndarray:
        """Default attribution: squared standardised deviation."""
        return z**2


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class MahalanobisDetector(AnomalyDetector):
    """Gaussian with shrunk covariance; score is the Mahalanobis distance.

    Ledoit-Wolf-style shrinkage toward a scaled identity keeps the covariance
    invertible when ``n_samples`` is only a few times ``n_features`` -- the
    normal situation during a two-week commissioning window.
    """

    kind = "mahalanobis"

    def __init__(self, feature_names: Optional[Sequence[str]] = None, shrinkage: Optional[float] = None):
        super().__init__(feature_names)
        self.shrinkage = shrinkage
        self.precision_: Optional[np.ndarray] = None
        self.shrinkage_used_: float = 0.0
        self.samples_per_feature_: float = 0.0

    #: Below this samples-per-feature ratio the empirical covariance is not
    #: usable and shrinkage is forced upward. 3 is the conventional floor; at
    #: n == d the sample covariance is singular by construction.
    MIN_SAMPLES_PER_FEATURE = 3.0

    def _fit(self, Z: np.ndarray) -> None:
        n, d = Z.shape
        self.samples_per_feature_ = float(n) / float(max(1, d))
        cov = np.cov(Z, rowvar=False)
        cov = np.atleast_2d(cov)
        if self.shrinkage is None:
            # Ledoit-Wolf optimal shrinkage intensity.
            mu = float(np.trace(cov) / d)
            target = mu * np.eye(d)
            delta = float(np.sum((cov - target) ** 2))
            beta_sq = 0.0
            for i in range(n):
                zi = Z[i : i + 1]
                si = zi.T @ zi
                beta_sq += float(np.sum((si - cov) ** 2))
            beta_sq /= n * n
            alpha = 0.0 if delta <= 0 else float(np.clip(beta_sq / delta, 0.0, 1.0))
        else:
            alpha = float(np.clip(self.shrinkage, 0.0, 1.0))
            mu = float(np.trace(cov) / d)
            target = mu * np.eye(d)
        # Guard against the small-n regime regardless of how alpha was chosen.
        # With 50 commissioning samples and 63 features the Ledoit-Wolf estimate
        # is itself noisy, and an under-shrunk precision matrix turns a few
        # near-collinear features into enormous spurious distances.
        if self.samples_per_feature_ < self.MIN_SAMPLES_PER_FEATURE:
            deficit = 1.0 - self.samples_per_feature_ / self.MIN_SAMPLES_PER_FEATURE
            alpha = float(np.clip(max(alpha, 0.2 + 0.7 * deficit), 0.0, 0.95))
        self.shrinkage_used_ = alpha
        shrunk = (1.0 - alpha) * cov + alpha * target
        # Jitter guarantees positive definiteness even for duplicated features.
        shrunk += 1e-9 * np.eye(d)
        self.precision_ = np.linalg.pinv(shrunk)

    def _score(self, Z: np.ndarray) -> np.ndarray:
        assert self.precision_ is not None
        q = np.einsum("ij,jk,ik->i", Z, self.precision_, Z)
        return np.sqrt(np.maximum(q, 0.0))

    def _contributions(self, z: np.ndarray) -> np.ndarray:
        assert self.precision_ is not None
        return z * (self.precision_ @ z)


class PCAResidualDetector(AnomalyDetector):
    """Score = norm of the residual outside the top-k principal subspace.

    Picks ``k`` from an explained-variance target rather than a fixed number,
    because the intrinsic dimension of a machine's healthy feature cloud depends
    on how many operating points it has (a single-speed pump: 3-5; a VFD
    compressor sweeping 30-60 Hz: 10-15).
    """

    kind = "pca_residual"

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        n_components: Optional[int] = None,
        explained_variance: float = 0.95,
    ):
        super().__init__(feature_names)
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.components_: Optional[np.ndarray] = None
        self.n_components_: int = 0

    def _fit(self, Z: np.ndarray) -> None:
        # Economy SVD of the centred (already standardised) matrix.
        _u, s, vt = np.linalg.svd(Z, full_matrices=False)
        var = s**2
        total = float(np.sum(var))
        if self.n_components is not None:
            k = int(min(max(1, self.n_components), vt.shape[0]))
        elif total <= 0:
            k = 1
        else:
            cum = np.cumsum(var) / total
            k = int(np.searchsorted(cum, self.explained_variance) + 1)
            k = int(min(max(1, k), vt.shape[0]))
        self.n_components_ = k
        self.components_ = vt[:k]

    def _score(self, Z: np.ndarray) -> np.ndarray:
        assert self.components_ is not None
        proj = Z @ self.components_.T @ self.components_
        return np.linalg.norm(Z - proj, axis=1)

    def _contributions(self, z: np.ndarray) -> np.ndarray:
        assert self.components_ is not None
        proj = self.components_.T @ (self.components_ @ z)
        return (z - proj) ** 2


class RobustThresholdDetector(AnomalyDetector):
    """Max robust z-score across features. Fully explainable by construction.

    This is the detector to ship first. It cannot find a correlation-structure
    anomaly, but every alarm it raises reduces to one sentence -- "feature X is
    N sigma above its commissioning distribution" -- and that is what gets a
    pilot converted into a contract.
    """

    kind = "robust_threshold"

    def __init__(self, feature_names: Optional[Sequence[str]] = None, soft_k: float = 0.0):
        super().__init__(feature_names)
        self.soft_k = soft_k

    def _fit(self, Z: np.ndarray) -> None:
        return None

    def _score(self, Z: np.ndarray) -> np.ndarray:
        a = np.abs(Z)
        if self.soft_k > 0:
            # Soft-max over features: less brittle to a single noisy feature.
            k = self.soft_k
            return np.log(np.sum(np.exp(k * a), axis=1)) / k
        return np.max(a, axis=1)

    def _contributions(self, z: np.ndarray) -> np.ndarray:
        return np.abs(z)


class DeepSVDDDetector(AnomalyDetector):
    """One-class deep SVDD in pure NumPy.

    Maps features through a small MLP and minimises the squared distance of
    every healthy sample to a fixed centre ``c``; the score is that distance.
    Two implementation details decide whether this works at all:

    * **No bias terms and no bounded activations.** With a bias the network can
      collapse to the constant ``c`` and reach zero loss having learned nothing
      ("hypersphere collapse", Ruff et al. 2018). Biases are omitted and LeakyReLU
      is used throughout.
    * **``c`` is fixed from the initial forward pass**, never trained, for the
      same reason.

    Chosen for the edge because it is tiny: a 60-24-12 MLP is 1.7 k parameters,
    6.9 kB as float32 and 1.7 kB quantised to int8 -- it fits in the RAM budget
    left over after the FFT buffers on every kit in this repository.
    """

    kind = "deep_svdd"

    def __init__(
        self,
        feature_names: Optional[Sequence[str]] = None,
        hidden: Sequence[int] = (24, 12),
        epochs: int = 300,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        leak: float = 0.1,
        seed: int = 0,
    ):
        super().__init__(feature_names)
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.leak = float(leak)
        self.seed = int(seed)
        self.weights_: List[np.ndarray] = []
        self.centre_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []

    # -- plumbing ----------------------------------------------------------
    def _forward(self, Z: np.ndarray, cache: bool = False):
        acts = [Z]
        h = Z
        for i, w in enumerate(self.weights_):
            pre = h @ w
            h = pre if i == len(self.weights_) - 1 else np.where(pre > 0, pre, self.leak * pre)
            if cache:
                acts.append(pre)
        return (h, acts) if cache else h

    def _fit(self, Z: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [Z.shape[1], *self.hidden]
        self.weights_ = [
            rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i])
            for i in range(len(dims) - 1)
        ]
        # Fix the centre from the initial mapping, and nudge any near-zero
        # coordinate away from 0 -- a coordinate at exactly 0 is a free direction
        # along which the network can collapse.
        h0 = self._forward(Z)
        c = np.mean(h0, axis=0)
        c[np.abs(c) < 1e-3] = 1e-1
        self.centre_ = c

        m = [np.zeros_like(w) for w in self.weights_]
        v = [np.zeros_like(w) for w in self.weights_]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n = Z.shape[0]
        self.loss_history_ = []
        for step in range(1, self.epochs + 1):
            h, acts = self._forward(Z, cache=True)
            diff = h - self.centre_
            loss = float(np.mean(np.sum(diff**2, axis=1)))
            self.loss_history_.append(loss)
            # Backward pass.
            grad = 2.0 * diff / n
            grads: List[np.ndarray] = [np.zeros_like(w) for w in self.weights_]
            for i in range(len(self.weights_) - 1, -1, -1):
                a_in = acts[i] if i == 0 else np.where(acts[i] > 0, acts[i], self.leak * acts[i])
                grads[i] = a_in.T @ grad + self.weight_decay * self.weights_[i]
                if i > 0:
                    grad = grad @ self.weights_[i].T
                    grad = grad * np.where(acts[i] > 0, 1.0, self.leak)
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1**step)
                vh = v[i] / (1 - b2**step)
                self.weights_[i] -= self.lr * mh / (np.sqrt(vh) + eps)

    def _score(self, Z: np.ndarray) -> np.ndarray:
        assert self.centre_ is not None
        h = self._forward(Z)
        return np.sqrt(np.sum((h - self.centre_) ** 2, axis=1))

    def _contributions(self, z: np.ndarray) -> np.ndarray:
        """Gradient-times-input attribution of the distance w.r.t. the input."""
        assert self.centre_ is not None
        h, acts = self._forward(np.atleast_2d(z), cache=True)
        diff = h - self.centre_
        grad = 2.0 * diff
        for i in range(len(self.weights_) - 1, -1, -1):
            grad = grad @ self.weights_[i].T
            if i > 0:
                grad = grad * np.where(acts[i] > 0, 1.0, self.leak)
        return (grad[0] * z) ** 1

    # -- export ------------------------------------------------------------
    def to_dict(self) -> Dict:
        if self.centre_ is None:
            raise RuntimeError("detector is not fitted")
        return {
            "kind": self.kind,
            "hidden": list(self.hidden),
            "leak": self.leak,
            "centre": self.centre_.tolist(),
            "weights": [w.tolist() for w in self.weights_],
            "scaler": self.scaler.to_dict(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DeepSVDDDetector":
        det = cls(feature_names=d.get("feature_names"), hidden=d["hidden"], leak=d.get("leak", 0.1))
        det.weights_ = [np.asarray(w, dtype=np.float64) for w in d["weights"]]
        det.centre_ = np.asarray(d["centre"], dtype=np.float64)
        det.scaler = StandardScaler.from_dict(d["scaler"])
        det._fitted = True
        return det

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


_REGISTRY = {
    "mahalanobis": MahalanobisDetector,
    "pca_residual": PCAResidualDetector,
    "robust_threshold": RobustThresholdDetector,
    "deep_svdd": DeepSVDDDetector,
}


def build_detector(kind: str, feature_names: Optional[Sequence[str]] = None, **kwargs) -> AnomalyDetector:
    """Factory by name -- lets a YAML config pick the detector."""
    try:
        cls = _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"unknown detector {kind!r}; expected one of {sorted(_REGISTRY)}") from exc
    return cls(feature_names=feature_names, **kwargs)  # type: ignore[arg-type]
