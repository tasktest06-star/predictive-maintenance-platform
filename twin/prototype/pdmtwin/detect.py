"""Change detection on the twin residual -- and the same machinery applied to
the baseline, because that is the only way the comparison is fair.

Everything in this module operates on a **standardised per-frame deviation**
``q_k``, so that the twin and the one-class detector can be driven through
identical sequential logic:

* twin:      ``q_k = (||nu_k||^2 - C) / sqrt(2C)``, the whitened innovation
             energy referred to its own chi-square mean and standard deviation.
* baseline:  ``q_k = (s_k - median) / (1.4826 * MAD)`` of the detector score on
             the calibration window.

Both are dimensionless, both have (approximately) zero mean and unit scale on
healthy data, and neither carries any information about how it was produced.
A CUSUM that cannot tell which arm it is running on cannot favour one.

The trap this module is built around
------------------------------------
``docs/08_branch_catalogue.md`` records that *a gate expressed as a ratio of
rates is really a gate on a count, and at small counts the ratio is the wrong
parameterisation.* That trap is wide open in a CUSUM threshold, in a specific
way worth spelling out:

:func:`pdmcore.eval.metrics.alarms_per_day` estimates the nuisance rate as
``p**N * acquisitions_per_day`` where ``p`` is the per-frame exceedance
probability and ``N`` the confirmation length. That formula assumes consecutive
exceedances are **independent**. A CUSUM statistic is a running sum: once it is
above the threshold it tends to stay above, and once it has been reset it takes
many frames to climb back. Its exceedances are about as dependent as a time
series gets, so ``p**N`` is wrong by orders of magnitude in *both* directions
depending on the drift parameter -- which means a CUSUM calibrated to an
alarms/day budget computed that way has an unknown false-alarm rate.

So nothing here is calibrated on a rate. Thresholds are set to a target
**average run length to false alarm** (ARL0), measured by running the test over
healthy trajectories with reset-on-alarm and *counting alarms*, and verified on
a third, disjoint window against a Poisson band on the count,
``allowed = lambda + 3*sqrt(lambda) + 3`` -- the same fix the Jetson branch
applied to its own ratio gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "energy_deviation",
    "score_deviation",
    "CusumTest",
    "GlrTest",
    "ThresholdTest",
    "robust_centre_scale",
    "measure_arl0",
    "threshold_for_arl0",
    "poisson_allowance",
    "first_alarm",
    "naive_alarms_per_day",
]


def energy_deviation(whitened: np.ndarray) -> np.ndarray:
    """Whitened innovation energy, referred to its chi-square moments.

    ``||nu||^2`` for a consistent filter is chi-square with ``C`` degrees of
    freedom: mean ``C``, variance ``2C``. Subtracting and dividing gives a
    statistic whose healthy distribution does not depend on how many channels
    the twin observes, which is what lets a threshold transfer between assets
    with different gated channel sets.
    """
    W = np.atleast_2d(np.asarray(whitened, dtype=np.float64))
    c = W.shape[1]
    return (np.sum(W**2, axis=1) - c) / np.sqrt(2.0 * c)


def score_deviation(scores: np.ndarray, *, centre: float, scale: float) -> np.ndarray:
    """Robustly standardised detector score. Median/MAD, as elsewhere here."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    return (s - centre) / max(scale, 1e-12)


def robust_centre_scale(scores: np.ndarray) -> Tuple[float, float]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    if mad <= 1e-12:
        mad = float(np.std(s))
    return med, max(mad, 1e-12)


@dataclass
class CusumTest:
    """One-sided CUSUM on the standardised deviation.

    ``drift`` (the classical ``k``) is the deviation size the test is tuned for,
    in units of ``q``'s healthy standard deviation. It is not a free parameter
    to be optimised against the test set: it is the smallest shift worth
    detecting, and it is fixed in the protocol *before* any result, at 0.5 --
    half a sigma of per-frame deviation, which at a 6 h cadence is a change a
    maintenance engineer would want a week's warning of.

    ``reset_on_alarm`` restarts the accumulator after an alarm. Without it a
    single excursion leaves the statistic pinned above the threshold forever and
    the measured ARL0 is meaningless.
    """

    drift: float = 0.5
    threshold: float = 5.0
    reset_on_alarm: bool = True
    consecutive_required: int = 2

    def run(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        """Returns the statistic and the confirmed-alarm indicator per frame."""
        q = np.asarray(q, dtype=np.float64).ravel()
        stat = np.zeros(q.size)
        alarm = np.zeros(q.size, dtype=bool)
        c = 0.0
        run_len = 0
        for i, qi in enumerate(q):
            c = max(0.0, c + qi - self.drift)
            stat[i] = c
            over = c > self.threshold
            run_len = run_len + 1 if over else 0
            if run_len >= self.consecutive_required:
                alarm[i] = True
                run_len = 0
                if self.reset_on_alarm:
                    c = 0.0
        return {"stat": stat, "alarm": alarm}


@dataclass
class ThresholdTest:
    """Per-frame threshold with an N-consecutive confirmation rule.

    What the platform ships today, expressed in the same interface as the
    sequential tests so that the only difference between the arms is the test,
    not the harness. ``reset_on_alarm`` is accepted and ignored: a memoryless
    test has nothing to reset, and accepting the argument is what lets the
    calibration and ARL0 code be shared rather than duplicated.
    """

    threshold: float = 3.0
    consecutive_required: int = 2
    reset_on_alarm: bool = True
    drift: float = 0.0

    def run(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).ravel()
        alarm = np.zeros(q.size, dtype=bool)
        run_len = 0
        for i, qi in enumerate(q):
            run_len = run_len + 1 if qi > self.threshold else 0
            if run_len >= self.consecutive_required:
                alarm[i] = True
                run_len = 0
        return {"stat": q.copy(), "alarm": alarm}


@dataclass
class GlrTest:
    """Windowed generalised likelihood ratio for an unknown-size mean shift.

    For Gaussian ``q`` with unit variance the GLR statistic over a window of
    unknown length ``w <= W`` is ``max_w (sum of the last w)^2 / (2w)`` when the
    shift sign is unknown, and the one-sided version drops the square for
    negative sums. Reported alongside the CUSUM because the CUSUM is optimal
    only for the shift size it is tuned to, and a degradation knee is not one
    shift size -- so if the CUSUM's fixed ``drift`` were quietly costing the
    twin its advantage, the GLR would show it.
    """

    window: int = 24
    threshold: float = 5.0
    consecutive_required: int = 2

    def run(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).ravel()
        stat = np.zeros(q.size)
        alarm = np.zeros(q.size, dtype=bool)
        run_len = 0
        csum = np.concatenate([[0.0], np.cumsum(q)])
        for i in range(q.size):
            best = 0.0
            for w in range(1, min(self.window, i + 1) + 1):
                s = csum[i + 1] - csum[i + 1 - w]
                if s > 0:
                    best = max(best, s * s / (2.0 * w))
            stat[i] = best
            run_len = run_len + 1 if best > self.threshold else 0
            if run_len >= self.consecutive_required:
                alarm[i] = True
                run_len = 0
        return {"stat": stat, "alarm": alarm}


def measure_arl0(q_by_traj: Sequence[Sequence[float]], test) -> Dict[str, float]:
    """Average run length to false alarm, **measured by counting alarms**.

    Healthy trajectories only. The returned ``arl0_frames`` is total healthy
    frames divided by total confirmed alarms, which is the estimator whose
    sampling distribution is Poisson in the numerator of alarms -- so a
    confidence statement about it is a statement about a count, not about a
    ratio of rates.
    """
    n_frames = 0
    n_alarms = 0
    for q in q_by_traj:
        out = test.run(q)
        n_frames += int(len(q))
        n_alarms += int(np.count_nonzero(out["alarm"]))
    return {
        "n_frames": float(n_frames),
        "n_alarms": float(n_alarms),
        "arl0_frames": float(n_frames / n_alarms) if n_alarms else float("inf"),
        "alarms_per_frame": float(n_alarms / max(1, n_frames)),
    }


def threshold_for_arl0(
    q_by_traj: Sequence[Sequence[float]],
    test,
    target_arl0_frames: float,
    *,
    lo: float = 0.05,
    hi: float = 400.0,
    n_iter: int = 48,
) -> Tuple[float, Dict[str, float]]:
    """Bisect the threshold until the measured ARL0 hits the target.

    Monotone by construction (a higher threshold cannot produce more alarms), so
    bisection is safe. The measured ARL0 at the chosen threshold is returned
    with it, because on a finite calibration window the target is not reachable
    exactly and pretending otherwise is how a comparison stops being matched.
    """
    def arl_at(h: float) -> float:
        t = _clone_with_threshold(test, h)
        return measure_arl0(q_by_traj, t)["arl0_frames"]

    a, b = lo, hi
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        if arl_at(m) < target_arl0_frames:
            a = m
        else:
            b = m
    h = 0.5 * (a + b)
    t = _clone_with_threshold(test, h)
    return h, {**measure_arl0(q_by_traj, t), "threshold": float(h),
               "target_arl0_frames": float(target_arl0_frames)}


def _clone_with_threshold(test, h: float):
    import dataclasses
    return dataclasses.replace(test, threshold=float(h))


def poisson_allowance(expected_count: float, sigma: float = 3.0, pad: float = 3.0) -> float:
    """``lambda + 3*sqrt(lambda) + 3``: the allowed alarm count on a window.

    Copied, with its justification, from the fix on ``hw/jetson-orin-nano``. At
    an expected count of 1.8 a 3-sigma Poisson band spans 0 to 6, so a gate on
    the *ratio* of observed to expected rate refuses stationary data about a
    third of the time -- and in a plant that means re-running calibration until
    it passes, which selects for whichever window looked quiet. The additive
    pad covers the small-lambda regime where ``sqrt(lambda)`` alone is still too
    tight.
    """
    lam = max(0.0, float(expected_count))
    return lam + sigma * np.sqrt(lam) + pad


def first_alarm(q: Sequence[float], test) -> Optional[int]:
    """Index of the first confirmed alarm, or ``None``.

    Reset-on-alarm is disabled for this measurement: the question is when the
    test first fires, and a reset after the answer cannot change it.
    """
    t = test
    try:
        import dataclasses
        t = dataclasses.replace(test, reset_on_alarm=False)
    except TypeError:
        pass
    out = t.run(q)
    idx = np.flatnonzero(out["alarm"])
    return int(idx[0]) if idx.size else None


def naive_alarms_per_day(
    q_by_traj: Sequence[Sequence[float]],
    test,
    *,
    acquisitions_per_day: float = 4.0,
) -> Dict[str, float]:
    """The ``p**N`` estimate, kept so the error can be quoted.

    This is what :func:`pdmcore.eval.metrics.alarms_per_day` computes when it is
    handed a CUSUM statistic: the per-frame exceedance probability raised to the
    confirmation length. It is reported next to the measured rate purely to
    quantify how wrong the independence assumption is on a cumulative statistic.
    """
    exceed = 0
    total = 0
    measured_alarms = 0
    for q in q_by_traj:
        out = test.run(q)
        exceed += int(np.count_nonzero(out["stat"] > test.threshold))
        total += int(len(q))
        measured_alarms += int(np.count_nonzero(out["alarm"]))
    p = exceed / max(1, total)
    n = int(getattr(test, "consecutive_required", 1))
    naive = (p**max(1, n)) * acquisitions_per_day
    measured = measured_alarms / max(1, total) * acquisitions_per_day
    return {
        "p_exceed": float(p),
        "naive_alarms_per_day": float(naive),
        "measured_alarms_per_day": float(measured),
        "ratio_measured_over_naive": float(measured / naive) if naive > 0 else float("inf"),
        "consecutive_required": float(n),
    }
