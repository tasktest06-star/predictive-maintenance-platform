"""The deployable detector, shaped by what the measured data actually broke.

The prototype's evaluation on CWRU and MFPT produced one result that matters
more for deployment than every accuracy number in it:

    CWRU, 63 features + Mahalanobis, four motor loads
      leave-one-load-out grouped split   AUC 1.000   realised FPR 50.5 %
      leaky (random) split               AUC 1.000   realised FPR  0.8 %

**The ranking transfers across operating points. The threshold does not.** A
threshold calibrated at one load and deployed at another realised 50.5 % false
positives against a 1 % target -- 1 212 alarms/day/100 assets against an alarm
budget of 24 -- while AUC stayed at exactly 1.000 in both conditions. A reviewer
reading AUC, which is what the literature reports, would see nothing wrong.

That is not a model problem and it cannot be fixed by a better model. It is a
*calibration-scope* problem, so this module fixes it structurally:

1. **Thresholds are per operating point**, keyed the way
   `pdmcore.health.index.OperatingBaseline` already keys healthy statistics --
   coarse (speed bin, load bin) cells, so a variable-duty asset is compared
   against itself at the same duty.
2. **An operating point that was never calibrated is refused, not guessed.**
   `score_frame` returns a verdict with `scored=False` and a reason. The
   alternative -- scoring against the nearest or the global threshold -- is
   precisely the experiment that produced 50.5 %.
3. **The realised alarm count is governed against its budget.** If the detector
   alarms more than a Poisson band around its target allows, the calibration no
   longer describes the asset, and the right response is to say so and stop
   raising work orders rather than to keep flooding. Counted, not rated: at a
   1 % target over 175 frames the expected count is 1.8 and noise alone spans
   0-6, so a gate on the *rate* would fire on healthy data.

What this module deliberately does not do: pick a different model. On CWRU the
63-feature Mahalanobis detector, PCA residual, and the plain window RMS with no
model at all all score AUC 1.000 -- the dataset is saturated and cannot rank
detectors. Choosing between heads needs MFPT (AUC 0.986, and RMS alone 0.348,
*below chance*) or field data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pdmcore.health.index import OperatingBaseline
from pdmcore.io.payload import FrameFlags

#: Poisson band on the alarm count, matching the convention used elsewhere in
#: this repository (3 sigma with a floor so the band does not collapse at small
#: expected counts). See `pdmcore.decide` and the Jetson branch for the same
#: parameterisation and the reason a ratio-of-rates gate is wrong here.
ALARM_SIGMA = 3.0
ALARM_COUNT_FLOOR = 3.0

#: Minimum healthy frames in an operating-point cell before its threshold is
#: believed. `AnomalyDetector.calibrate` itself refuses below 10; this is the
#: deployment-side floor and is deliberately higher, because a threshold
#: calibrated on 12 frames of one duty is the thing that produced 50.5 % FPR.
MIN_CALIBRATION_FRAMES = 40


class CalibrationScopeError(ValueError):
    """Raised when a detector is asked to operate outside its calibration."""


@dataclass
class Verdict:
    """One scoring decision, including the decision not to score."""

    scored: bool
    op_key: Optional[Tuple[int, int]] = None
    score: float = float("nan")
    threshold: float = float("nan")
    alarm: bool = False
    reason: str = ""
    #: Set when the governor has decided the calibration no longer describes
    #: the asset. Scores still flow (they are useful for trending) but alarms
    #: are advisory and must not raise a work order.
    advisory_only: bool = False

    @property
    def raises_work_order(self) -> bool:
        return bool(self.scored and self.alarm and not self.advisory_only)


@dataclass
class AlarmBudget:
    """Governs the realised alarm count against what the target implies.

    The measured failure this exists for: 1 212 alarms/day/100 assets against a
    budget of 24. A detector in that state is not detecting anything, it is
    emitting noise, and the only useful behaviour is to notice and say so.
    """

    target_fpr: float = 0.01
    window: int = 500
    #: Alarms/day/100 assets the programme can absorb. The figure a maintenance
    #: planner gives you; 24 is one alarm per hour per 100 assets.
    budget_per_day_per_100: float = 24.0
    _decisions: List[bool] = field(default_factory=list)
    tripped: bool = False
    trip_reason: str = ""

    def observe(self, alarmed: bool) -> None:
        self._decisions.append(bool(alarmed))
        if len(self._decisions) > self.window:
            del self._decisions[: len(self._decisions) - self.window]
        n = len(self._decisions)
        if n < 50:
            return                      # too few to judge; see ALARM_COUNT_FLOOR
        alarms = sum(self._decisions)
        lam = n * self.target_fpr
        allowed = lam + ALARM_SIGMA * math.sqrt(lam) + ALARM_COUNT_FLOOR
        if alarms > allowed:
            self.tripped = True
            self.trip_reason = (
                f"{alarms} alarms in the last {n} frames against {lam:.1f} "
                f"expected and {allowed:.1f} allowed at {ALARM_SIGMA:.0f} sigma. "
                f"The calibration no longer describes this asset: either the "
                f"operating point has moved outside what was calibrated, or the "
                f"asset has genuinely changed. Alarms are advisory until "
                f"recalibration."
            )
        else:
            # Recovery is allowed -- an asset that was briefly outside its
            # calibration should not be permanently downgraded.
            self.tripped = False
            self.trip_reason = ""

    @property
    def realised_fpr(self) -> float:
        n = len(self._decisions)
        return (sum(self._decisions) / n) if n else float("nan")

    def alarms_per_day_per_100(self, interval_s: float) -> float:
        """Realised alarm rate in the units a planner budgets in."""
        if not self._decisions or interval_s <= 0:
            return float("nan")
        per_asset_per_day = self.realised_fpr * 86400.0 / interval_s
        return per_asset_per_day * 100.0


class DeployedDetector:
    """A calibrated detector that knows the limits of its own calibration.

    `detector` is any object with `.score(X)` and `.calibrate(X, fpr)` -- the
    convention `pdmcore.models.anomaly` and `pdmedge` already use, so this works
    with Mahalanobis, PCA residual, robust threshold or a quantised export
    without change.
    """

    def __init__(
        self,
        detector: object,
        *,
        target_fpr: float = 0.01,
        speed_bin_hz: float = 5.0,
        load_bins: Sequence[float] = (0.0, 0.4, 0.7, 1.01),
        interval_s: float = 3600.0,
        budget_per_day_per_100: float = 24.0,
    ) -> None:
        self.detector = detector
        self.target_fpr = float(target_fpr)
        self.interval_s = float(interval_s)
        # Reuse the platform's binning rather than inventing a second scheme:
        # the health index already compares an asset against itself at the same
        # duty, and a detector that binned differently would disagree with it.
        self._baseline = OperatingBaseline(speed_bin_hz=speed_bin_hz,
                                           load_bins=load_bins)
        self.thresholds: Dict[Tuple[int, int], float] = {}
        self.n_calibration: Dict[Tuple[int, int], int] = {}
        self.budget = AlarmBudget(target_fpr=target_fpr,
                                  budget_per_day_per_100=budget_per_day_per_100)

    # -- calibration ------------------------------------------------------
    def op_key(self, speed_hz: float, load_frac: float) -> Tuple[int, int]:
        return self._baseline.key(speed_hz, load_frac)

    def calibrate(
        self,
        X_healthy: np.ndarray,
        speed_hz: Sequence[float],
        load_frac: Sequence[float],
    ) -> Dict[Tuple[int, int], float]:
        """Calibrate one threshold per operating point from healthy frames.

        Cells with fewer than `MIN_CALIBRATION_FRAMES` get **no threshold**, and
        therefore get refused at scoring time. That is the whole point: a
        half-calibrated cell is worse than an uncalibrated one, because it
        produces a number.
        """
        X = np.atleast_2d(np.asarray(X_healthy, dtype=np.float64))
        speed = np.asarray(speed_hz, dtype=np.float64)
        load = np.asarray(load_frac, dtype=np.float64)
        if not (X.shape[0] == speed.size == load.size):
            raise ValueError(
                f"X has {X.shape[0]} rows but {speed.size} speeds and "
                f"{load.size} loads were given"
            )
        groups: Dict[Tuple[int, int], List[int]] = {}
        for i in range(X.shape[0]):
            groups.setdefault(self.op_key(float(speed[i]), float(load[i])), []).append(i)

        self.thresholds.clear()
        self.n_calibration.clear()
        skipped = []
        for key, idx in sorted(groups.items()):
            self.n_calibration[key] = len(idx)
            if len(idx) < MIN_CALIBRATION_FRAMES:
                skipped.append((key, len(idx)))
                continue
            thr = float(self.detector.calibrate(X[idx], fpr=self.target_fpr))
            self.thresholds[key] = thr
        if not self.thresholds:
            raise CalibrationScopeError(
                f"no operating-point cell reached {MIN_CALIBRATION_FRAMES} healthy "
                f"frames; cells seen: "
                f"{[(k, n) for k, n in sorted(self.n_calibration.items())]}. "
                f"Commission for longer, or widen the bins -- but do not deploy "
                f"a single global threshold: on CWRU that realised 50.5 % false "
                f"positives against a 1 % target."
            )
        self._skipped = skipped
        return dict(self.thresholds)

    @property
    def calibrated_points(self) -> List[Tuple[int, int]]:
        return sorted(self.thresholds)

    # -- scoring ----------------------------------------------------------
    def score_frame(
        self,
        features: Sequence[float],
        speed_hz: float,
        load_frac: float,
        *,
        flags: FrameFlags = FrameFlags.NONE,
    ) -> Verdict:
        """Score one frame, or refuse and say why."""
        if FrameFlags(flags).blocks_scoring:
            return Verdict(scored=False, reason=(
                f"frame flags {FrameFlags(flags)!r} block scoring: a clipped or "
                f"overrun acquisition produces a confident anomaly on an "
                f"artefact"))
        if not self.thresholds:
            raise CalibrationScopeError("score_frame before calibrate")

        key = self.op_key(float(speed_hz), float(load_frac))
        thr = self.thresholds.get(key)
        if thr is None:
            nearest = min(self.calibrated_points,
                          key=lambda k: abs(k[0] - key[0]) + abs(k[1] - key[1]))
            return Verdict(scored=False, op_key=key, reason=(
                f"operating point {key} was never calibrated (calibrated: "
                f"{self.calibrated_points}). Refusing rather than borrowing the "
                f"threshold from {nearest}: on CWRU, scoring one motor load "
                f"against another load's threshold realised 50.5 % false "
                f"positives while AUC stayed at 1.000."))

        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        score = float(np.asarray(self.detector.score(x)).ravel()[0])
        alarm = bool(score > thr)
        self.budget.observe(alarm)
        return Verdict(scored=True, op_key=key, score=score, threshold=thr,
                       alarm=alarm, advisory_only=self.budget.tripped,
                       reason=self.budget.trip_reason if self.budget.tripped else "")

    # -- reporting --------------------------------------------------------
    def status(self) -> Dict[str, object]:
        return {
            "calibrated_points": [list(k) for k in self.calibrated_points],
            "n_calibration": {str(k): v for k, v in sorted(self.n_calibration.items())},
            "uncalibrated_cells_seen": [
                [list(k), n] for k, n in getattr(self, "_skipped", [])
            ],
            "target_fpr": self.target_fpr,
            "realised_fpr": self.budget.realised_fpr,
            "alarms_per_day_per_100": self.budget.alarms_per_day_per_100(self.interval_s),
            "budget_per_day_per_100": self.budget.budget_per_day_per_100,
            "advisory_only": self.budget.tripped,
            "advisory_reason": self.budget.trip_reason,
        }

    def report(self) -> str:
        s = self.status()
        lines = [
            f"deployed detector: {len(self.calibrated_points)} calibrated "
            f"operating point(s), target FPR {self.target_fpr:.3f}",
        ]
        for key in self.calibrated_points:
            lines.append(f"  OP {key}  threshold {self.thresholds[key]:8.3f}  "
                         f"from {self.n_calibration[key]:4d} healthy frames")
        for key, n in getattr(self, "_skipped", []):
            lines.append(f"  OP {key}  NOT CALIBRATED -- only {n} healthy frames "
                         f"(need {MIN_CALIBRATION_FRAMES}); frames at this duty "
                         f"will be refused")
        realised = s["realised_fpr"]
        if isinstance(realised, float) and realised == realised:
            lines.append(f"  realised FPR {realised:.3f}, "
                         f"{float(s['alarms_per_day_per_100']):.0f} alarms/day/100 "
                         f"against a budget of {s['budget_per_day_per_100']:.0f}")
        if s["advisory_only"]:
            lines.append(f"  ADVISORY ONLY: {s['advisory_reason']}")
        return "\n".join(lines)
