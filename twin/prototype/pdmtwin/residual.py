"""The learned residual, and the trap it sets.

A hybrid twin is a physics prediction plus a learned correction:

.. math:: \\hat{z} = h_{\\text{phys}}(x;\\theta) + \\hat{r}(u)

The correction is what makes the hybrid work -- no lumped-parameter model of a
machine is right to better than a few percent, and a few percent of a band RMS
is larger than an incipient fault. It is also the branch's central hazard:

    **A residual model with enough capacity absorbs the fault signature itself,
    which is exactly the thing you are trying to detect.**

The absorption has three distinct mechanisms, and they are not equally obvious:

1. **Contaminated training.** The residual is refitted online so the twin
   "tracks ageing". Any fault slower than the refit window is learned as normal.
   This is the mechanism everybody warns about and the easiest to avoid: do not
   refit online. ``pdmcore.health.index`` already states the rule -- adapt the
   baseline only while the asset is in ``NORMAL``.
2. **Endogenous inputs.** The residual is a function of inputs that themselves
   move with the fault. Even trained *only* on healthy data, such a model
   extrapolates along the fault direction and predicts away part of the
   residual it was never shown. If the inputs include the measured channels, it
   predicts away nearly all of it.
3. **Capacity against a finite healthy window.** With enough effective degrees
   of freedom relative to the number of healthy frames, the residual model fits
   the healthy *noise*, which shrinks the apparent healthy residual spread. The
   detection threshold is calibrated on that shrunken spread, so the nominal
   false-alarm rate is kept -- but the fault-time residual is now compared
   against a model that interpolates confidently in a region it never saw, and
   the deflection-to-spread ratio falls.

``run_capacity.py`` measures all three on one axis: effective degrees of
freedom, which for kernel ridge is

.. math:: \\mathrm{edf} = \\operatorname{tr}\\bigl(K (K + \\alpha I)^{-1}\\bigr)

a real number between 0 and ``n_train``, not a hyperparameter name. Reporting
edf rather than "we used an RBF kernel with alpha=1e-3" is what makes "how much
capacity is too much" a transferable answer.

Kernel ridge rather than an MLP: it is the only choice here whose capacity is
*measurable* in closed form. A small MLP's effective capacity depends on the
optimiser, the initialisation and the step count, so a sweep over hidden width
does not give a number anyone can reuse. ``MLPResidual`` is provided for
comparison and its width sweep is reported alongside, with that caveat stated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

__all__ = [
    "ResidualInputs",
    "NullResidual",
    "KernelRidgeResidual",
    "MLPResidual",
    "build_residual",
]


class ResidualInputs(str, Enum):
    """What the residual model is allowed to look at.

    ``EXOGENOUS`` only sees quantities that cannot move with the fault: shaft
    speed, and the ambient-independent part of the operating point. This is the
    defensible choice.

    ``STATE`` also sees the filter's own state estimate. Tempting -- the state
    is where load lives, and load is the dominant nuisance -- and the route by
    which a healthy-only-trained residual still absorbs fault signal, because
    the state estimate rises with the fault.

    ``MEASURED`` also sees the measured channels. This is what "let the network
    correct the physics using the rest of the telemetry" means in practice, and
    it is a catastrophe: the residual model can reconstruct the measurement it
    is supposed to be explaining.
    """

    EXOGENOUS = "exogenous"
    STATE = "state"
    MEASURED = "measured"


def _design(mode: ResidualInputs, *, shaft_hz: np.ndarray, state: np.ndarray,
            z: np.ndarray, f_nom: float) -> np.ndarray:
    """Assemble the residual model's input matrix."""
    shaft_hz = np.atleast_1d(np.asarray(shaft_hz, dtype=np.float64))
    cols: List[np.ndarray] = [np.log(shaft_hz / f_nom)]
    if mode in (ResidualInputs.STATE, ResidualInputs.MEASURED):
        cols.extend(np.atleast_2d(state).T)
    if mode is ResidualInputs.MEASURED:
        cols.extend(np.atleast_2d(z).T)
    return np.column_stack(cols)


@dataclass
class NullResidual:
    """No correction: the physics-only twin, and the ``alpha -> inf`` limit.

    Present as a first-class model rather than an ``if`` in the caller, because
    it is the control the whole capacity sweep is measured against. If the
    residual never beats this, the hybrid structure is not earning its keep.
    """

    n_channels: int = 1
    inputs: ResidualInputs = ResidualInputs.EXOGENOUS
    edf: float = 0.0
    n_train: int = 0

    def fit(self, *_args, **_kwargs) -> "NullResidual":
        return self

    def predict(self, *_args, **_kwargs) -> np.ndarray:
        return np.zeros((1, self.n_channels))

    def predict_for(self, shaft_hz, state, z, f_nom) -> np.ndarray:  # noqa: ANN001
        n = np.atleast_2d(np.asarray(state, dtype=np.float64)).shape[0]
        return np.zeros((n, self.n_channels))

    def report(self) -> Dict[str, float]:
        return {"kind": "none", "edf": 0.0, "edf_per_frame": 0.0, "n_train": self.n_train,
                "inputs": self.inputs.value}


@dataclass
class KernelRidgeResidual:
    """RBF kernel ridge correction with a measured effective complexity.

    ``alpha`` is the capacity knob and ``edf`` is what it buys. ``gamma`` is set
    by the median-heuristic on the training inputs unless given, so that the
    sweep over ``alpha`` is a sweep over capacity rather than over whether the
    kernel happens to be degenerate.
    """

    n_channels: int
    inputs: ResidualInputs = ResidualInputs.EXOGENOUS
    alpha: float = 1.0
    gamma: Optional[float] = None
    max_train: int = 600
    seed: int = 0
    _model: Optional[KernelRidge] = field(default=None, init=False, repr=False)
    _mu: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False, repr=False)
    _sd: np.ndarray = field(default_factory=lambda: np.ones(0), init=False, repr=False)
    edf: float = field(default=0.0, init=False)
    n_train: int = field(default=0, init=False)
    gamma_used: float = field(default=0.0, init=False)
    _xfit: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False, repr=False)
    _dual: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), init=False, repr=False)
    _sqfit: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False, repr=False)

    def fit(self, U: np.ndarray, Rlog: np.ndarray) -> "KernelRidgeResidual":
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Rlog, dtype=np.float64))
        if U.shape[0] != Y.shape[0]:
            raise ValueError("inputs and residuals must have the same number of rows")
        if U.shape[0] > self.max_train:
            # Subsample deterministically: kernel ridge is O(n^3) and the point
            # of the sweep is capacity, not training-set size. The subsample size
            # is reported, because edf only means something relative to it.
            idx = np.random.default_rng(self.seed).choice(U.shape[0], self.max_train, replace=False)
            U, Y = U[np.sort(idx)], Y[np.sort(idx)]
        self._mu = np.mean(U, axis=0)
        self._sd = np.std(U, axis=0)
        self._sd = np.where(self._sd > 1e-12, self._sd, 1.0)
        Us = (U - self._mu) / self._sd
        if self.gamma is None:
            # Median heuristic on pairwise distances.
            d2 = np.sum((Us[:, None, :] - Us[None, :, :]) ** 2, axis=-1)
            med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
            self.gamma_used = 1.0 / max(med, 1e-9)
        else:
            self.gamma_used = float(self.gamma)
        self._model = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma_used)
        self._model.fit(Us, Y)
        # Effective degrees of freedom from the kernel eigenvalues.
        sq = np.sum(Us**2, axis=1)
        K = np.exp(-self.gamma_used * (sq[:, None] + sq[None, :] - 2.0 * Us @ Us.T))
        ev = np.linalg.eigvalsh(0.5 * (K + K.T))
        ev = np.clip(ev, 0.0, None)
        self.edf = float(np.sum(ev / (ev + self.alpha)))
        self.n_train = int(Us.shape[0])
        # Keep the fitted dual form for a fast prediction path. sklearn fits
        # the model; sklearn's `predict` is not used in the filter loop because
        # 70 % of its cost is input validation and kernel dispatch, measured:
        # 3.79 ms per filter step with it, 1.2 ms without, for a kernel
        # evaluation that is one matrix product. The same code is what the
        # production variant would ship, so the latency budget in
        # `twin/production/` is a budget for code that exists.
        self._xfit = np.asarray(self._model.X_fit_, dtype=np.float64)
        self._dual = np.atleast_2d(np.asarray(self._model.dual_coef_, dtype=np.float64))
        if self._dual.shape[0] != self._xfit.shape[0]:
            self._dual = self._dual.T
        self._sqfit = np.sum(self._xfit**2, axis=1)
        return self

    def predict(self, U: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("residual model is not fitted")
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        Q = (U - self._mu) / self._sd
        d2 = np.sum(Q**2, axis=1)[:, None] + self._sqfit[None, :] - 2.0 * (Q @ self._xfit.T)
        return np.exp(-self.gamma_used * np.maximum(d2, 0.0)) @ self._dual

    def predict_sklearn(self, U: np.ndarray) -> np.ndarray:
        """The reference path, kept so the fast one can be tested against it."""
        if self._model is None:
            raise RuntimeError("residual model is not fitted")
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        return np.atleast_2d(self._model.predict((U - self._mu) / self._sd))

    def predict_for(self, shaft_hz, state, z, f_nom) -> np.ndarray:  # noqa: ANN001
        return self.predict(_design(self.inputs, shaft_hz=shaft_hz, state=state, z=z, f_nom=f_nom))

    def report(self) -> Dict[str, float]:
        return {
            "kind": "kernel_ridge", "alpha": float(self.alpha), "gamma": float(self.gamma_used),
            "edf": float(self.edf), "n_train": int(self.n_train),
            "edf_per_frame": float(self.edf / max(1, self.n_train)),
            "inputs": self.inputs.value,
        }


@dataclass
class MLPResidual:
    """Small MLP correction, for comparison with the kernel-ridge sweep.

    Included because "a small neural network corrects the physics" is the
    commonest form of the hybrid in the literature, and because it shows the
    limit of the capacity story: an MLP's *effective* capacity is not its
    parameter count. Width 64 with early stopping can behave like far fewer
    degrees of freedom than width 8 trained to convergence. The parameter count
    is reported, labelled as the wrong axis.
    """

    n_channels: int
    inputs: ResidualInputs = ResidualInputs.EXOGENOUS
    hidden: Tuple[int, ...] = (16,)
    alpha: float = 1e-4
    max_iter: int = 2000
    seed: int = 0
    _model: Optional[MLPRegressor] = field(default=None, init=False, repr=False)
    _mu: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False, repr=False)
    _sd: np.ndarray = field(default_factory=lambda: np.ones(0), init=False, repr=False)
    n_train: int = field(default=0, init=False)
    edf: float = field(default=float("nan"), init=False)

    def fit(self, U: np.ndarray, Rlog: np.ndarray) -> "MLPResidual":
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Rlog, dtype=np.float64))
        self._mu, sd = np.mean(U, axis=0), np.std(U, axis=0)
        self._sd = np.where(sd > 1e-12, sd, 1.0)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
            random_state=self.seed, learning_rate_init=3e-3, tol=1e-7, n_iter_no_change=50,
        )
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._model.fit((U - self._mu) / self._sd, Y)
        self.n_train = int(U.shape[0])
        return self

    def predict(self, U: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("residual model is not fitted")
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        return np.atleast_2d(self._model.predict((U - self._mu) / self._sd))

    def predict_for(self, shaft_hz, state, z, f_nom) -> np.ndarray:  # noqa: ANN001
        return self.predict(_design(self.inputs, shaft_hz=shaft_hz, state=state, z=z, f_nom=f_nom))

    @property
    def n_params(self) -> int:
        if self._model is None:
            return 0
        return int(sum(w.size for w in self._model.coefs_) + sum(b.size for b in self._model.intercepts_))

    def report(self) -> Dict[str, float]:
        return {
            "kind": "mlp", "hidden": list(self.hidden), "alpha": float(self.alpha),
            "n_params": self.n_params, "n_train": int(self.n_train),
            "edf": float("nan"), "edf_per_frame": float("nan"),
            "inputs": self.inputs.value,
            "note": "parameter count is not effective capacity; see the module docstring",
        }


def build_residual(kind: str, n_channels: int, **kwargs):
    """Factory by name, so a sweep configuration can be a dict."""
    if kind in ("none", "null"):
        return NullResidual(n_channels=n_channels, inputs=kwargs.get("inputs", ResidualInputs.EXOGENOUS))
    if kind in ("kernel_ridge", "krr"):
        return KernelRidgeResidual(n_channels=n_channels, **kwargs)
    if kind == "mlp":
        return MLPResidual(n_channels=n_channels, **kwargs)
    raise ValueError(f"unknown residual model {kind!r}")


def design_matrix(mode: ResidualInputs, *, shaft_hz, state, z, f_nom) -> np.ndarray:  # noqa: ANN001
    """Public wrapper over the input assembly, used by the twin and the tests."""
    return _design(mode, shaft_hz=shaft_hz, state=state, z=z, f_nom=f_nom)
