"""Remaining useful life with honest uncertainty.

Design position
---------------
A single RUL number is worse than no RUL number. "This bearing has 23 days left"
is acted on as if it were a fact; when the machine fails in 9 days the programme
loses credibility permanently. Everything here therefore returns a
**distribution**: a median plus a credible interval, and an explicit
"insufficient evidence" state that is returned far more often than a point
estimate.

Method
------
A bootstrap particle filter over a two-parameter exponential degradation model,
which is the standard stochastic model for fatigue-driven damage growth:

.. math:: h(t) = h_0 + a\\,\\bigl(e^{b t} - 1\\bigr) + \\varepsilon

Each particle carries ``(h0, a, b)``. On each new health observation the
particles are reweighted by the observation likelihood and resampled, then the
parameters are jittered slightly (regularised particle filter) to avoid
degeneracy. RUL is the distribution of first-crossing times of the failure
threshold, propagated forward from the posterior.

Why a particle filter instead of fitting a curve: the posterior over ``b`` is
strongly skewed and the first-crossing time is a nonlinear function of it, so a
least-squares fit plus a delta-method interval produces intervals that are both
too narrow and mis-centred. The particle filter gets the skew right, which is
the whole point of reporting an interval.

Limits, stated plainly: this is a data-driven model with no knowledge of the
physics of the specific asset. It needs the degradation to have *started* -- it
cannot predict when a healthy bearing will begin to fail, only how long one that
is already degrading will last. :meth:`ParticleFilterRUL.estimate` returns
``status="insufficient_evidence"`` until the trend is detectably positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["RULEstimate", "ParticleFilterRUL", "fit_exponential_trend"]


@dataclass
class RULEstimate:
    """RUL posterior summary."""

    status: str                      # "ok" | "insufficient_evidence" | "already_failed"
    median_days: Optional[float] = None
    p10_days: Optional[float] = None
    p90_days: Optional[float] = None
    mean_days: Optional[float] = None
    prob_failure_within_7d: Optional[float] = None
    prob_failure_within_30d: Optional[float] = None
    health_now: Optional[float] = None
    threshold: Optional[float] = None
    n_observations: int = 0
    doubling_time_days: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        def r(v):
            return None if v is None else round(float(v), 3)

        return {
            "status": self.status,
            "median_days": r(self.median_days),
            "p10_days": r(self.p10_days),
            "p90_days": r(self.p90_days),
            "mean_days": r(self.mean_days),
            "prob_failure_within_7d": r(self.prob_failure_within_7d),
            "prob_failure_within_30d": r(self.prob_failure_within_30d),
            "health_now": r(self.health_now),
            "threshold": r(self.threshold),
            "n_observations": self.n_observations,
            "doubling_time_days": r(self.doubling_time_days),
            "note": self.note,
        }

    @property
    def interval_width_days(self) -> Optional[float]:
        if self.p10_days is None or self.p90_days is None:
            return None
        return float(self.p90_days - self.p10_days)


def fit_exponential_trend(t_days: Sequence[float], health: Sequence[float]) -> Dict[str, float]:
    """Least-squares exponential fit, used to seed the particle filter.

    Returns ``a``, ``b``, ``h0``, ``r2`` and ``doubling_time_days``. A quick
    sanity check and a cheap fallback: when the filter has too few observations
    this fit is reported with a wide interval rather than nothing.
    """
    t = np.asarray(t_days, dtype=np.float64)
    h = np.asarray(health, dtype=np.float64)
    if t.size < 3:
        raise ValueError("need >= 3 observations")
    h0 = float(np.min(h))
    y = h - h0 + 1e-6
    mask = y > 0
    if np.count_nonzero(mask) < 3:
        return {"a": 0.0, "b": 0.0, "h0": h0, "r2": 0.0, "doubling_time_days": float("inf")}
    logy = np.log(y[mask])
    tm = t[mask]
    A = np.vstack([np.ones_like(tm), tm]).T
    coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
    log_a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "a": float(np.exp(log_a)),
        "b": b,
        "h0": h0,
        "r2": float(r2),
        "doubling_time_days": float(np.log(2.0) / b) if b > 1e-9 else float("inf"),
    }


@dataclass
class ParticleFilterRUL:
    """Bootstrap particle filter over exponential degradation.

    Parameters
    ----------
    failure_threshold
        Health-index value defining failure. Set it from the alarm policy
        (e.g. :attr:`AlarmConfig.raise_critical`) so the RUL answers
        "how long until this asset reaches the state at which we stop it",
        which is a question operations can act on, rather than the unanswerable
        "when will it physically break".
    n_particles
        1000 is plenty for a 3-parameter model and keeps the update under a
        millisecond, so this can run on a gateway.
    obs_noise
        Standard deviation of the health observation. Estimated from the data
        when left as ``None``.
    min_observations, min_b, min_r2
        Gates before any number is reported. These exist to make the
        "insufficient evidence" answer the default rather than the exception.
    """

    failure_threshold: float = 4.0
    n_particles: int = 1000
    obs_noise: Optional[float] = None
    min_observations: int = 12
    min_b: float = 5e-3
    min_r2: float = 0.30
    #: The asset must be out of its normal band before any RUL is quoted. RUL
    #: for a machine sitting at the healthy median is not a long number, it is
    #: an undefined one: there is no degradation signal to extrapolate, so the
    #: fit is extrapolating noise. Without this gate the filter happily reports
    #: "3 days" off eight noisy healthy samples whose exponential fit happened
    #: to pass R2 by luck -- the single most damaging failure mode a prognostics
    #: feature can have.
    min_health: float = 1.0
    #: Only the most recent observations are used for the trend gate.
    #: Degradation is a local phenomenon; a 90-day history dominated by a flat
    #: healthy stage drags the fitted growth rate toward zero and suppresses a
    #: real alarm that started last week.
    trend_window: int = 30
    jitter: float = 0.04
    seed: int = 0
    horizon_days: float = 730.0

    _t: List[float] = field(default_factory=list, init=False)
    _h: List[float] = field(default_factory=list, init=False)
    _particles: Optional[np.ndarray] = field(default=None, init=False)
    _weights: Optional[np.ndarray] = field(default=None, init=False)
    _rng: Optional[np.random.Generator] = field(default=None, init=False)

    # -- internals ---------------------------------------------------------
    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def _init_particles(self) -> None:
        rng = self._ensure_rng()
        try:
            seed_fit = fit_exponential_trend(self._t, self._h)
        except ValueError:
            seed_fit = {"a": 1e-3, "b": 0.02, "h0": float(np.min(self._h))}
        a0 = max(1e-6, float(seed_fit["a"]))
        b0 = max(1e-4, float(seed_fit["b"]))
        h00 = float(seed_fit["h0"])
        n = self.n_particles
        # Log-normal priors on a and b: both are strictly positive and span
        # orders of magnitude across assets.
        self._particles = np.column_stack([
            h00 + rng.normal(0.0, 0.05, n),
            a0 * np.exp(rng.normal(0.0, 1.0, n)),
            b0 * np.exp(rng.normal(0.0, 0.7, n)),
        ])
        self._weights = np.full(n, 1.0 / n)

    def _predict_health(self, particles: np.ndarray, t: float) -> np.ndarray:
        h0, a, b = particles[:, 0], particles[:, 1], particles[:, 2]
        return h0 + a * (np.exp(np.clip(b * t, -50.0, 50.0)) - 1.0)

    def _sigma(self) -> float:
        if self.obs_noise is not None:
            return float(self.obs_noise)
        if len(self._h) < 4:
            return 0.2
        d = np.diff(np.asarray(self._h, dtype=np.float64))
        # Robust scale of the first difference, halved (differencing doubles
        # variance), floored so a flat series does not give zero noise.
        mad = float(np.median(np.abs(d - np.median(d))) * 1.4826)
        return max(0.05, mad / np.sqrt(2.0))

    # -- public ------------------------------------------------------------
    def update(self, t_days: float, health: float) -> None:
        """Add one health observation."""
        if self._t and t_days < self._t[-1]:
            raise ValueError("observations must arrive in time order")
        self._t.append(float(t_days))
        self._h.append(float(health))
        if len(self._t) < 3:
            return
        if self._particles is None:
            self._init_particles()
        assert self._particles is not None and self._weights is not None
        rng = self._ensure_rng()
        sigma = self._sigma()
        pred = self._predict_health(self._particles, float(t_days))
        resid = (float(health) - pred) / sigma
        # Student-t likelihood (nu=4): one anomalous acquisition -- a bumped
        # sensor, a transient load -- must not annihilate every particle, which
        # a Gaussian likelihood would do.
        logw = -2.5 * np.log1p(resid**2 / 4.0)
        logw -= np.max(logw)
        w = self._weights * np.exp(logw)
        total = float(np.sum(w))
        if total <= 0 or not np.isfinite(total):
            self._init_particles()
            return
        w /= total
        # Resample when the effective sample size drops below half.
        ess = 1.0 / float(np.sum(w**2))
        if ess < self.n_particles / 2.0:
            idx = rng.choice(self.n_particles, size=self.n_particles, p=w)
            self._particles = self._particles[idx]
            # Regularisation jitter, scaled per-dimension so it adapts to the
            # current posterior spread instead of adding a fixed absolute kick.
            # Jitter needs an absolute floor as well as a proportional term.
            # Proportional-only jitter is self-defeating: once the posterior has
            # collapsed its std is ~0, so the jitter is ~0, so it can never
            # re-explore. The symptom is a zero-width credible interval -- an
            # RUL of "3 days [3, 3]" -- which is the most dangerous possible
            # output because it looks maximally confident.
            spread = np.std(self._particles, axis=0) * self.jitter
            floor = np.abs(np.mean(self._particles, axis=0)) * 0.02
            scale = np.maximum(spread, np.maximum(floor, 1e-6))
            self._particles = self._particles + rng.normal(0.0, scale, self._particles.shape)
            self._particles[:, 1:] = np.maximum(self._particles[:, 1:], 1e-9)
            self._weights = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            self._weights = w

    def estimate(self) -> RULEstimate:
        """Current RUL posterior, or an explicit "not enough evidence"."""
        n_obs = len(self._t)
        health_now = self._h[-1] if self._h else None
        if health_now is not None and health_now >= self.failure_threshold:
            return RULEstimate(
                status="already_failed",
                median_days=0.0,
                p10_days=0.0,
                p90_days=0.0,
                health_now=health_now,
                threshold=self.failure_threshold,
                n_observations=n_obs,
                note="health index is already at or past the failure threshold",
            )
        if n_obs < self.min_observations or self._particles is None:
            return RULEstimate(
                status="insufficient_evidence",
                health_now=health_now,
                threshold=self.failure_threshold,
                n_observations=n_obs,
                note=f"need >= {self.min_observations} observations, have {n_obs}",
            )
        if health_now is not None and health_now < self.min_health:
            return RULEstimate(
                status="insufficient_evidence",
                health_now=health_now,
                threshold=self.failure_threshold,
                n_observations=n_obs,
                note=(
                    f"health index {health_now:.2f} is inside the normal band "
                    f"(< {self.min_health:.2f}); no degradation signal to extrapolate"
                ),
            )
        w = max(self.min_observations, int(self.trend_window))
        try:
            fit = fit_exponential_trend(self._t[-w:], self._h[-w:])
        except ValueError:
            fit = {"b": 0.0, "r2": 0.0, "doubling_time_days": float("inf")}
        if fit["b"] < self.min_b or fit["r2"] < self.min_r2:
            return RULEstimate(
                status="insufficient_evidence",
                health_now=health_now,
                threshold=self.failure_threshold,
                n_observations=n_obs,
                doubling_time_days=fit.get("doubling_time_days"),
                note=(
                    f"no detectable degradation trend (growth rate {fit['b']:.4f}/day, "
                    f"R2 {fit['r2']:.2f}); RUL is not identifiable from a flat trend"
                ),
            )

        assert self._weights is not None
        t_now = self._t[-1]
        h0 = self._particles[:, 0]
        a = self._particles[:, 1]
        b = self._particles[:, 2]
        # First-crossing time: solve h0 + a(e^{bt}-1) = threshold for t.
        target = (self.failure_threshold - h0) / np.maximum(a, 1e-12) + 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            t_fail = np.where(target > 1.0, np.log(np.maximum(target, 1.0 + 1e-12)) / np.maximum(b, 1e-12),
                              np.inf)
        rul = t_fail - t_now
        rul = np.where(np.isfinite(rul), rul, self.horizon_days)
        rul = np.clip(rul, 0.0, self.horizon_days)
        w = self._weights / float(np.sum(self._weights))

        def wq(q: float) -> float:
            order = np.argsort(rul)
            cw = np.cumsum(w[order])
            return float(rul[order][int(np.searchsorted(cw, q))])

        median = wq(0.5)
        censored = float(np.sum(w[rul >= self.horizon_days * 0.999]))
        note = ""
        if censored > 0.2:
            note = (
                f"{censored * 100:.0f}% of the posterior exceeds the {self.horizon_days:.0f}-day "
                "horizon; the upper bound is censored, not estimated"
            )
        return RULEstimate(
            status="ok",
            median_days=median,
            p10_days=wq(0.1),
            p90_days=wq(0.9),
            mean_days=float(np.sum(w * rul)),
            prob_failure_within_7d=float(np.sum(w[rul <= 7.0])),
            prob_failure_within_30d=float(np.sum(w[rul <= 30.0])),
            health_now=health_now,
            threshold=self.failure_threshold,
            n_observations=n_obs,
            doubling_time_days=float(fit["doubling_time_days"]),
            note=note,
        )

    def posterior_samples(self, n: int = 500) -> np.ndarray:
        """Draw ``(h0, a, b)`` samples from the current posterior, for plotting."""
        if self._particles is None or self._weights is None:
            raise RuntimeError("filter has no particles yet")
        rng = self._ensure_rng()
        idx = rng.choice(self.n_particles, size=n, p=self._weights / np.sum(self._weights))
        return self._particles[idx]

    def predict_band(self, t_future_days: Sequence[float], n: int = 500) -> Dict[str, np.ndarray]:
        """Predictive median and 10/90 band of the health index over time."""
        samples = self.posterior_samples(n)
        t = np.asarray(t_future_days, dtype=np.float64)
        curves = np.array([self._predict_health(samples, float(ti)) for ti in t])  # (T, n)
        return {
            "t_days": t,
            "median": np.median(curves, axis=1),
            "p10": np.quantile(curves, 0.1, axis=1),
            "p90": np.quantile(curves, 0.9, axis=1),
        }
