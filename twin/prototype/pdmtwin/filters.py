"""State estimation that couples the twin to the measurements.

Two filters over the same model, because one of them is not trustworthy on its
own:

:class:`UnscentedTwinFilter`
    A scaled unscented Kalman filter over the twin's state. Cheap, gives an
    innovation covariance for free (which the CUSUM test needs), and is the
    standard answer. It is also the one that can diverge, so it is instrumented:
    every Cholesky repair, every covariance condition number and the normalised
    innovation squared are recorded, and :meth:`UnscentedTwinFilter.diagnostics`
    reports them whether they are flattering or not.

:class:`BootstrapTwinFilter`
    A bootstrap particle filter in the style of
    :class:`pdmcore.health.rul.ParticleFilterRUL` -- Student-t likelihood so one
    bumped acquisition cannot annihilate the posterior, ESS-triggered
    resampling, regularisation jitter with an absolute floor. Slower by ~50x and
    used as a cross-check: when the UKF and the particle filter disagree about
    the state, the UKF is wrong, because the particle filter does not assume the
    posterior is Gaussian and this posterior is not.

On divergence, up front
-----------------------
A UKF on a stiff nonlinear model diverges, and a report that does not mention it
is hiding it. What happens *here*, measured in ``run_filters.py``:

* With the 2-state (line + floor) model the filter is stable and consistent:
  mean NIS/C close to 1 and no Cholesky repairs over the healthy verification
  window.
* With the damage state added, the filter is **not** stable at low severity.
  Damage is unobservable while the impact level sits under the structural floor,
  so the posterior on it is flat, the covariance grows until the sigma points
  straddle the ``d >= 0`` boundary, and the clipped sigma points bias the mean
  upward. The symptom is not a blow-up -- it is a damage estimate that drifts
  away from zero on a perfectly healthy machine, which is worse, because it
  looks like an early detection.

Both behaviours are numbers in ``twin/prototype/results/filters.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ProcessModel",
    "UnscentedTwinFilter",
    "BootstrapTwinFilter",
    "FilterTrace",
]


@dataclass
class ProcessModel:
    """Discrete-time process model between acquisitions.

    The two nuisance states are Ornstein-Uhlenbeck: a plant's load wanders and
    comes back, it does not random-walk to infinity, and a pure random walk
    gives a filter that slowly forgets the commissioning reference -- i.e. it
    adapts to a developing fault, which is the one thing a detector must not do.
    ``revert`` is the per-step pull toward 0 (the commissioning median).

    The damage state, when present, is a non-negative random walk with drift:
    damage does not revert.
    """

    n_state: int = 2
    revert: Tuple[float, ...] = (0.25, 0.25, 0.0)
    #: Process std per step. 0.10 on the line level at a 6 h cadence matches the
    #: measured healthy log spread of the line channels (~0.16) after the OU
    #: pull; the floor state is given less because the floor channels move by
    #: ~0.04. Setting these from the data rather than by taste is what makes the
    #: innovation covariance meaningful.
    sigma: Tuple[float, ...] = (0.10, 0.04, 0.02)
    damage_drift: float = 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        rev = np.asarray(self.revert[: self.n_state])
        out = x * (1.0 - rev)
        if self.n_state >= 3:
            out[:, 2] = x[:, 2] + self.damage_drift
        return out

    @property
    def Q(self) -> np.ndarray:
        return np.diag(np.asarray(self.sigma[: self.n_state]) ** 2)


@dataclass
class FilterTrace:
    """Per-frame filter output, in time order."""

    x: np.ndarray                     # (n, n_state) posterior mean
    P: np.ndarray                     # (n, n_state, n_state)
    innovation: np.ndarray            # (n, C) z - h(x_pred)
    whitened: np.ndarray              # (n, C) S^{-1/2} innovation
    nis: np.ndarray                   # (n,) innovation' S^-1 innovation
    n_repairs: int = 0
    max_cond: float = 0.0
    n_clipped: int = 0

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def _chol_psd(A: np.ndarray, jitter0: float = 1e-12) -> Tuple[np.ndarray, int]:
    """Cholesky with escalating jitter. Returns the factor and repair count.

    A filter that silently symmetrises and jitters its covariance until the
    factorisation succeeds is a filter whose divergence is invisible. The repair
    count is returned so it can be reported.
    """
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    jitter = 0.0
    for k in range(8):
        try:
            return np.linalg.cholesky(A + jitter * np.eye(n)), k
        except np.linalg.LinAlgError:
            jitter = jitter0 * 10.0**k if jitter == 0.0 else jitter * 10.0
    # Last resort: eigenvalue flooring, counted as a repair.
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-12 * max(1.0, float(np.max(w))))
    return V @ np.diag(np.sqrt(w)), 8


class UnscentedTwinFilter:
    """Scaled unscented Kalman filter over the twin's state.

    Parameters
    ----------
    h
        Measurement function ``(X (m, n_state), aux) -> (m, C)`` in log space.
        The residual model is folded into ``h`` by the caller, so the filter
        does not know whether it is tracking physics or physics-plus-learning.
    R
        Measurement covariance in log space, from the healthy fit window.
    alpha, beta, kappa
        Scaled unscented transform parameters. ``alpha = 0.3`` keeps the sigma
        points close to the mean, which matters here because the measurement
        function saturates: a sigma point far below the commissioning level
        predicts a sensor-noise-dominated channel, and the mean of the
        transformed points is then biased by the saturation rather than by the
        state.
    state_box
        Hard validity box on the state. Sigma points outside it are clipped and
        counted; the count is a divergence indicator, not housekeeping.
    """

    def __init__(
        self,
        h: Callable[[np.ndarray, Dict], np.ndarray],
        R: np.ndarray,
        process: ProcessModel,
        *,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
        state_box: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.h = h
        self.R = np.atleast_2d(np.asarray(R, dtype=np.float64))
        self.process = process
        n = process.n_state
        self.n = n
        self.x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).ravel()
        self.P = (np.diag(np.asarray(process.sigma[:n]) ** 2) * 4.0) if P0 is None else np.asarray(P0)
        self.alpha, self.beta, self.kappa = float(alpha), float(beta), float(kappa)
        lam = self.alpha**2 * (n + self.kappa) - n
        self.lam = lam
        self.wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.wc = self.wm.copy()
        self.wm[0] = lam / (n + lam)
        self.wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)
        if state_box is None:
            lo = np.array([-3.0, -3.0, 0.0])[:n]
            hi = np.array([3.0, 3.0, 50.0])[:n]
            state_box = (lo, hi)
        self.box = state_box
        self.n_repairs = 0
        self.max_cond = 0.0
        self.n_clipped = 0

    # -- internals ---------------------------------------------------------
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        L, reps = _chol_psd((self.n + self.lam) * P)
        self.n_repairs += reps
        pts = np.vstack([x, x + L.T, x - L.T])
        lo, hi = self.box
        clipped = np.clip(pts, lo, hi)
        self.n_clipped += int(np.count_nonzero(np.any(clipped != pts, axis=1)))
        return clipped

    # -- public ------------------------------------------------------------
    def step(self, z: np.ndarray, aux: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """One predict/update cycle. Returns ``(innovation, whitened, nis)``."""
        z = np.asarray(z, dtype=np.float64).ravel()
        # --- predict
        pts = self._sigma_points(self.x, self.P)
        pts = self.process.predict(pts)
        x_pred = self.wm @ pts
        dx = pts - x_pred
        P_pred = (dx * self.wc[:, None]).T @ dx + self.process.Q
        # --- measurement
        pts2 = self._sigma_points(x_pred, P_pred)
        Z = self.h(pts2, aux)
        z_pred = self.wm @ Z
        dz = Z - z_pred
        dx2 = pts2 - x_pred
        S = (dz * self.wc[:, None]).T @ dz + self.R
        C = (dx2 * self.wc[:, None]).T @ dz
        Ls, reps = _chol_psd(S)
        self.n_repairs += reps
        self.max_cond = max(self.max_cond, float(np.linalg.cond(S)))
        innov = z - z_pred
        # Whitened innovation: S^{-1/2} nu via the Cholesky factor, so the test
        # statistic downstream is in units of standard deviations of what the
        # filter itself expected -- which is the only scale on which a CUSUM
        # threshold transfers between assets.
        wht = np.linalg.solve(Ls, innov)
        nis = float(wht @ wht)
        K = np.linalg.solve(S.T, C.T).T
        self.x = x_pred + K @ innov
        self.P = P_pred - K @ S @ K.T
        self.x = np.clip(self.x, self.box[0], self.box[1])
        return innov, wht, nis

    def run(self, Z: np.ndarray, aux_list: Sequence[Dict]) -> FilterTrace:
        """Run over a trajectory in time order."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        xs, Ps, inn, wh, nis = [], [], [], [], []
        for k in range(Z.shape[0]):
            i, w, q = self.step(Z[k], aux_list[k])
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
            inn.append(i)
            wh.append(w)
            nis.append(q)
        return FilterTrace(
            x=np.array(xs), P=np.array(Ps), innovation=np.array(inn),
            whitened=np.array(wh), nis=np.array(nis),
            n_repairs=self.n_repairs, max_cond=self.max_cond, n_clipped=self.n_clipped,
        )

    def diagnostics(self, trace: FilterTrace) -> Dict[str, float]:
        """Consistency and divergence summary.

        ``nis_over_dof`` is the number to read. A consistent filter sits at 1.0:
        its innovations are as large as its own covariance says they should be.
        Below 1 it is under-confident (it will be slow to detect); above 1 it is
        over-confident, which on healthy data means false alarms and is the
        practical face of divergence.
        """
        dof = max(1, trace.innovation.shape[1])
        return {
            "nis_over_dof_mean": float(np.mean(trace.nis) / dof),
            "nis_over_dof_median": float(np.median(trace.nis) / dof),
            "n_frames": int(trace.n),
            "n_chol_repairs": int(trace.n_repairs),
            "n_sigma_clipped": int(trace.n_clipped),
            "max_innovation_cov_cond": float(trace.max_cond),
            "state_drift": [float(v) for v in (trace.x[-1] - trace.x[0])],
        }


class BootstrapTwinFilter:
    """Bootstrap particle filter over the same model, as a cross-check.

    Deliberately the same shape as :class:`pdmcore.health.rul.ParticleFilterRUL`
    -- Student-t likelihood with nu=4, resample when the effective sample size
    halves, regularisation jitter with a proportional term *and* an absolute
    floor. The floor is not defensive: proportional-only jitter cannot
    re-explore after the posterior collapses, and the symptom is a
    zero-width credible interval, which is the most dangerous possible output
    because it looks maximally confident. That lesson is already written down in
    ``rul.py``; it is reused here rather than relearned.
    """

    def __init__(
        self,
        h: Callable[[np.ndarray, Dict], np.ndarray],
        R: np.ndarray,
        process: ProcessModel,
        *,
        n_particles: int = 1000,
        seed: int = 0,
        nu: float = 4.0,
        jitter: float = 0.04,
    ):
        self.h = h
        self.R = np.atleast_2d(np.asarray(R, dtype=np.float64))
        self.sigma = np.sqrt(np.diag(self.R))
        self.process = process
        self.n_particles = int(n_particles)
        self.nu = float(nu)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        n = process.n_state
        self.n = n
        self.particles = self.rng.normal(0.0, np.asarray(process.sigma[:n]) * 2.0, (self.n_particles, n))
        if n >= 3:
            self.particles[:, 2] = np.abs(self.particles[:, 2])
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.n_resamples = 0

    def step(self, z: np.ndarray, aux: Dict) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64).ravel()
        p = self.process.predict(self.particles)
        p = p + self.rng.normal(0.0, np.asarray(self.process.sigma[: self.n]), p.shape)
        if self.n >= 3:
            p[:, 2] = np.maximum(p[:, 2], 0.0)
        Zp = self.h(p, aux)
        resid = (z[None, :] - Zp) / self.sigma[None, :]
        logw = np.sum(-0.5 * (self.nu + 1.0) * np.log1p(resid**2 / self.nu), axis=1)
        logw -= np.max(logw)
        w = self.weights * np.exp(logw)
        total = float(np.sum(w))
        if total <= 0 or not np.isfinite(total):
            self.particles = p
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
            return self.mean(), self.cov()
        w /= total
        ess = 1.0 / float(np.sum(w**2))
        if ess < self.n_particles / 2.0:
            idx = self.rng.choice(self.n_particles, size=self.n_particles, p=w)
            p = p[idx]
            spread = np.std(p, axis=0) * self.jitter
            floor = np.maximum(np.abs(np.mean(p, axis=0)) * 0.02, 1e-4)
            p = p + self.rng.normal(0.0, np.maximum(spread, floor), p.shape)
            if self.n >= 3:
                p[:, 2] = np.maximum(p[:, 2], 0.0)
            w = np.full(self.n_particles, 1.0 / self.n_particles)
            self.n_resamples += 1
        self.particles, self.weights = p, w
        return self.mean(), self.cov()

    def mean(self) -> np.ndarray:
        return self.weights @ self.particles

    def cov(self) -> np.ndarray:
        d = self.particles - self.mean()
        return (d * self.weights[:, None]).T @ d

    def run(self, Z: np.ndarray, aux_list: Sequence[Dict]) -> Dict[str, np.ndarray]:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        xs, ps = [], []
        for k in range(Z.shape[0]):
            m, c = self.step(Z[k], aux_list[k])
            xs.append(m.copy())
            ps.append(np.sqrt(np.maximum(np.diag(c), 0.0)))
        return {"x": np.array(xs), "sd": np.array(ps), "n_resamples": self.n_resamples}
