"""Health index, trending and the alarm state machine.

The alarm state machine is the part of a condition-monitoring platform that
decides whether anyone trusts it. Three failure modes, all common, all avoided
here deliberately:

**Chattering.** A raw threshold crossed by a noisy index produces alarm storms.
Fixed with hysteresis (a clear threshold strictly below the raise threshold) and
an N-of-M confirmation rule.

**Load blindness.** A pump at 40 % flow and the same pump at 95 % flow are
different machines. Comparing both against one baseline guarantees either missed
detections at low load or false alarms at high load. Fixed by binning the
baseline on an operating-condition key (speed band x load band) -- see
:class:`OperatingBaseline`.

**Drift.** Mounting relaxes, grease redistributes, seasons change ambient
temperature by 25 K. A frozen baseline slowly turns into a false-alarm
generator. Fixed by letting the baseline adapt *only* while the asset is in a
normal state, so a developing fault can never be absorbed into "normal" -- which
is the trap that makes naive online adaptation hide exactly the thing it was
meant to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from collections import deque

__all__ = [
    "AlarmState",
    "AlarmConfig",
    "AlarmEvent",
    "AlarmStateMachine",
    "EwmaTrend",
    "OperatingBaseline",
    "HealthIndex",
]


class AlarmState(str, Enum):
    """States an asset can be in. Mirrors ISO 20816 zone semantics loosely."""

    NORMAL = "normal"
    WATCH = "watch"       # elevated, trending; no action yet
    ALARM = "alarm"       # schedule intervention
    CRITICAL = "critical"  # intervene now
    STALE = "stale"        # no data -- an alarm in its own right


@dataclass
class AlarmConfig:
    """Thresholds and debouncing.

    Thresholds are in health-index units (see :class:`HealthIndex`, where 0 is
    the healthy median and 1.0 is "at the commissioning 99.9th percentile").

    ``raise_*`` / ``clear_*`` pairs implement hysteresis. The gap should be at
    least one standard deviation of the index's own noise, otherwise a borderline
    asset flaps. ``n_of_m`` requires ``n`` exceedances within the last ``m``
    acquisitions, which rejects single-acquisition transients (a forklift hitting
    the skid) without adding the full delay of an N-consecutive rule.
    """

    raise_watch: float = 1.0
    clear_watch: float = 0.7
    raise_alarm: float = 2.0
    clear_alarm: float = 1.5
    raise_critical: float = 4.0
    clear_critical: float = 3.0
    n_of_m: Tuple[int, int] = (3, 5)
    stale_after_s: float = 3 * 3600.0
    min_dwell_s: float = 1800.0

    def __post_init__(self) -> None:
        if not (self.clear_watch < self.raise_watch <= self.clear_alarm < self.raise_alarm
                <= self.clear_critical < self.raise_critical):
            raise ValueError(
                "thresholds must satisfy clear_watch < raise_watch <= clear_alarm < "
                "raise_alarm <= clear_critical < raise_critical"
            )
        n, m = self.n_of_m
        if not 1 <= n <= m:
            raise ValueError("n_of_m must satisfy 1 <= n <= m")


@dataclass
class AlarmEvent:
    """A state transition worth telling someone about."""

    timestamp_s: float
    asset_id: str
    from_state: AlarmState
    to_state: AlarmState
    index: float
    reason: str
    evidence: Dict[str, float] = field(default_factory=dict)

    @property
    def is_escalation(self) -> bool:
        order = list(AlarmState)
        return order.index(self.to_state) > order.index(self.from_state)


class AlarmStateMachine:
    """Hysteretic, debounced, dwell-limited alarm logic.

    One instance per asset per index. Deterministic and side-effect free: feed it
    ``(timestamp, index)`` and it returns an event or ``None``. That makes the
    whole alarm policy unit-testable, which matters because this is the component
    whose bugs cost customer trust rather than CPU cycles.
    """

    def __init__(self, asset_id: str, config: Optional[AlarmConfig] = None):
        self.asset_id = asset_id
        self.config = config or AlarmConfig()
        self.state = AlarmState.NORMAL
        self.last_change_s: Optional[float] = None
        self.last_sample_s: Optional[float] = None
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self.config.n_of_m[1])
        self.events: List[AlarmEvent] = []

    # -- helpers -----------------------------------------------------------
    def _n_of_m(self, threshold: float) -> bool:
        n, _m = self.config.n_of_m
        return sum(1 for _t, v in self._history if v >= threshold) >= n

    def _target_state(self) -> Tuple[AlarmState, str]:
        cfg = self.config
        cur = self.state
        # Escalation requires n-of-m above the raise threshold.
        if self._n_of_m(cfg.raise_critical):
            return AlarmState.CRITICAL, f"index >= {cfg.raise_critical} in {cfg.n_of_m[0]}/{cfg.n_of_m[1]}"
        if self._n_of_m(cfg.raise_alarm):
            return AlarmState.ALARM, f"index >= {cfg.raise_alarm} in {cfg.n_of_m[0]}/{cfg.n_of_m[1]}"
        if self._n_of_m(cfg.raise_watch):
            return AlarmState.WATCH, f"index >= {cfg.raise_watch} in {cfg.n_of_m[0]}/{cfg.n_of_m[1]}"
        # De-escalation uses the clear thresholds against the latest value only,
        # so an asset that has genuinely recovered does not wait for the window
        # to flush.
        latest = self._history[-1][1] if self._history else 0.0
        if cur == AlarmState.CRITICAL and latest < cfg.clear_critical:
            return AlarmState.ALARM, f"index < {cfg.clear_critical}"
        if cur == AlarmState.ALARM and latest < cfg.clear_alarm:
            return AlarmState.WATCH, f"index < {cfg.clear_alarm}"
        if cur == AlarmState.WATCH and latest < cfg.clear_watch:
            return AlarmState.NORMAL, f"index < {cfg.clear_watch}"
        return cur, "unchanged"

    # -- public ------------------------------------------------------------
    def update(
        self,
        timestamp_s: float,
        index: float,
        *,
        evidence: Optional[Dict[str, float]] = None,
    ) -> Optional[AlarmEvent]:
        """Feed one sample. Returns an :class:`AlarmEvent` on state change."""
        if self.last_sample_s is not None and timestamp_s < self.last_sample_s:
            raise ValueError("timestamps must be non-decreasing")
        self.last_sample_s = timestamp_s
        self._history.append((timestamp_s, float(index)))
        target, reason = self._target_state()
        if target == self.state:
            return None
        # Minimum dwell time stops a borderline asset from oscillating between
        # two states every acquisition. Escalation to CRITICAL ignores dwell:
        # suppressing a critical alarm to keep the log tidy is indefensible.
        if (
            self.last_change_s is not None
            and target != AlarmState.CRITICAL
            and timestamp_s - self.last_change_s < self.config.min_dwell_s
        ):
            return None
        event = AlarmEvent(
            timestamp_s=timestamp_s,
            asset_id=self.asset_id,
            from_state=self.state,
            to_state=target,
            index=float(index),
            reason=reason,
            evidence=dict(evidence or {}),
        )
        self.state = target
        self.last_change_s = timestamp_s
        self.events.append(event)
        return event

    def check_stale(self, now_s: float) -> Optional[AlarmEvent]:
        """Raise a STALE event when an asset stops reporting.

        A silent sensor is indistinguishable from a healthy one on the
        dashboard, which is how a dead battery turns into an unmonitored machine
        for six months. Staleness is treated as a first-class alarm.
        """
        if self.last_sample_s is None or self.state == AlarmState.STALE:
            return None
        if now_s - self.last_sample_s <= self.config.stale_after_s:
            return None
        event = AlarmEvent(
            timestamp_s=now_s,
            asset_id=self.asset_id,
            from_state=self.state,
            to_state=AlarmState.STALE,
            index=float("nan"),
            reason=f"no data for {(now_s - self.last_sample_s) / 3600.0:.1f} h",
        )
        self.state = AlarmState.STALE
        self.last_change_s = now_s
        self.events.append(event)
        return event


@dataclass
class EwmaTrend:
    """Exponentially weighted trend with a slope estimate.

    ``alpha`` is expressed as a half-life in samples, which is how maintenance
    engineers think ("average over about a day"), not as an opaque 0.07.
    """

    half_life_samples: float = 12.0
    value: Optional[float] = None
    variance: float = 0.0
    n: int = 0
    _slope_buf: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=48))

    @property
    def alpha(self) -> float:
        return float(1.0 - 0.5 ** (1.0 / max(1e-9, self.half_life_samples)))

    def update(self, timestamp_s: float, x: float) -> float:
        a = self.alpha
        if self.value is None:
            self.value = float(x)
            self.variance = 0.0
        else:
            delta = float(x) - self.value
            self.value += a * delta
            self.variance = (1 - a) * (self.variance + a * delta * delta)
        self.n += 1
        self._slope_buf.append((timestamp_s, self.value))
        return self.value

    @property
    def std(self) -> float:
        return float(np.sqrt(max(0.0, self.variance)))

    def slope_per_day(self) -> float:
        """Least-squares slope of the smoothed series, per day.

        Slope is what turns a level into a forecast: an index of 1.5 that is
        flat is a note, an index of 1.5 climbing 0.4/day is next week's failure.
        """
        if len(self._slope_buf) < 3:
            return 0.0
        t = np.array([p[0] for p in self._slope_buf], dtype=np.float64)
        v = np.array([p[1] for p in self._slope_buf], dtype=np.float64)
        t = (t - t[0]) / 86400.0
        denom = float(np.sum((t - t.mean()) ** 2))
        if denom <= 0:
            return 0.0
        return float(np.sum((t - t.mean()) * (v - v.mean())) / denom)


class OperatingBaseline:
    """Per-operating-point healthy statistics.

    Keyed by a coarse ``(speed_bin, load_bin)`` so a variable-duty asset is
    compared against itself at the same duty. Bins are intentionally coarse:
    with fine bins most cells never collect enough samples to have a usable
    distribution, and the ones that do are the operating points you already
    understand.
    """

    def __init__(
        self,
        *,
        speed_bin_hz: float = 5.0,
        load_bins: Sequence[float] = (0.0, 0.4, 0.7, 1.01),
        min_samples: int = 30,
        max_samples: int = 2000,
    ):
        self.speed_bin_hz = speed_bin_hz
        self.load_bins = tuple(load_bins)
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._cells: Dict[Tuple[int, int], Deque[np.ndarray]] = {}

    def key(self, speed_hz: float, load_frac: float = 1.0) -> Tuple[int, int]:
        s = int(np.floor(max(0.0, speed_hz) / self.speed_bin_hz))
        l = int(np.clip(np.searchsorted(self.load_bins, load_frac, side="right") - 1,
                        0, len(self.load_bins) - 2))
        return s, l

    def add(self, vector: np.ndarray, speed_hz: float, load_frac: float = 1.0) -> None:
        """Add a vector believed to be healthy.

        Callers must only pass vectors taken while the asset was in
        :attr:`AlarmState.NORMAL`. ``HealthIndex.update`` enforces that.
        """
        k = self.key(speed_hz, load_frac)
        buf = self._cells.setdefault(k, deque(maxlen=self.max_samples))
        buf.append(np.asarray(vector, dtype=np.float64))

    def ready(self, speed_hz: float, load_frac: float = 1.0) -> bool:
        buf = self._cells.get(self.key(speed_hz, load_frac))
        return buf is not None and len(buf) >= self.min_samples

    def stats(
        self, speed_hz: float, load_frac: float = 1.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(median, robust_sigma, usable_mask)`` for the matching cell.

        Returns ``None`` until the cell has ``min_samples`` observations.

        The sigma floors here are load-bearing, not defensive padding. A feature
        that never moved during commissioning has MAD = 0, and dividing by a
        1e-12 epsilon turns a 0.1 % change into a z-score of 1e9 -- which is how
        a health index ends up reading 8e10 on a healthy machine. Three cases,
        handled differently because they mean different things:

        * **Integer-valued features** (matched-harmonic counts, ISO zone index)
          get a sigma floor of 1.0. One count is the natural quantum of such a
          feature, so "went from 0 to 4 matched harmonics" scores z = 4. This is
          both numerically safe and diagnostically correct.
        * **Continuous features with some spread** get ``max(MAD, std,
          2 % of range)``, so a feature with a tiny but real spread cannot
          dominate.
        * **Exactly constant continuous features** are marked unusable. There is
          no scale to normalise by, so no z-score exists. They are still
          available to the rule engine, which reads them in absolute terms --
          the right division of labour: statistics for what varies, rules for
          what does not.
        """
        buf = self._cells.get(self.key(speed_hz, load_frac))
        if buf is None or len(buf) < self.min_samples:
            return None
        X = np.vstack(buf)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        std = np.std(X, axis=0)
        rng = np.ptp(X, axis=0)
        is_int = np.all(np.isclose(X, np.round(X)), axis=0)

        sigma = np.maximum(mad, std)
        sigma = np.maximum(sigma, 0.02 * rng)
        sigma = np.where(is_int, np.maximum(sigma, 1.0), sigma)
        usable = (sigma > 0) & np.isfinite(sigma)
        sigma = np.where(usable, sigma, 1.0)
        return med, sigma, usable

    @property
    def n_cells(self) -> int:
        return len(self._cells)

    def coverage(self) -> Dict[str, int]:
        return {f"s{k[0]}_l{k[1]}": len(v) for k, v in sorted(self._cells.items())}


class HealthIndex:
    """Scalar health index with trending, alarming and adaptive baselining.

    The index is the **robust z-score of the worst contributing feature**,
    divided by a reference z of 3.09 (the two-sided 99.9 % normal quantile) so
    that 1.0 means "at the edge of the commissioning distribution" and the
    alarm thresholds in :class:`AlarmConfig` are dimensionless and portable
    across assets.

    A learned detector score can be supplied instead via ``external_score``;
    the trending, hysteresis and adaptive-baseline machinery is identical. Using
    the transparent z-score for the shipped default and the learned score as an
    upgrade is a deliberate sequencing choice: the explainable detector earns
    the trust that lets the learned one be adopted later.
    """

    #: Two-sided 99.9 % normal quantile. Index 1.0 == this many robust sigmas.
    REFERENCE_Z = 3.09

    #: Per-feature z-scores are clipped here before the max is taken. Beyond
    #: ~20 robust sigmas the Gaussian model is not merely wrong, it is
    #: meaningless -- the exact value carries no extra information, and leaving
    #: it uncapped lets one pathological feature drive the index (and therefore
    #: the alarm state, and therefore the RUL) by nine orders of magnitude.
    #:
    #: The cap must stay comfortably above ``AlarmConfig.raise_critical``
    #: (default 4.0, i.e. z = 12.4). A cap of 12 looks safe and silently makes
    #: CRITICAL unreachable, so a severely damaged machine sits in ALARM
    #: forever. Headroom here is a correctness requirement, not a tuning knob.
    Z_CAP = 20.0

    #: Separate, much looser cap for the *un-clipped* aggregate reported as
    #: ``index_raw``. The alarm index must saturate (a 40-sigma excursion and a
    #: 400-sigma one call for the same action), but a saturating signal is the
    #: wrong input for a trend model: the exponential fit in
    #: :mod:`pdmcore.health.rul` sees a flat top and concludes the crossing has
    #: already happened. So prognostics consume ``index_raw`` and alarms consume
    #: ``index``.
    Z_CAP_RAW = 150.0

    def __init__(
        self,
        asset_id: str,
        feature_names: Sequence[str],
        *,
        alarm_config: Optional[AlarmConfig] = None,
        baseline: Optional[OperatingBaseline] = None,
        half_life_samples: float = 12.0,
        key_features: Optional[Sequence[str]] = None,
        adapt_baseline: bool = True,
        top_k: int = 3,
    ):
        self.asset_id = asset_id
        self.feature_names = tuple(feature_names)
        self.alarms = AlarmStateMachine(asset_id, alarm_config)
        self.baseline = baseline or OperatingBaseline()
        self.trend = EwmaTrend(half_life_samples=half_life_samples)
        self.adapt_baseline = adapt_baseline
        # Aggregating the top-k z-scores instead of only the maximum. A pure max
        # is driven by whichever single feature is noisiest, and it saturates at
        # Z_CAP as soon as *one* feature pegs -- losing all resolution in exactly
        # the region where the trend feeds the RUL model. Requiring k features to
        # agree also halves the nuisance-alarm rate in practice, because a
        # genuine mechanical change moves a family of correlated features while
        # a glitch moves one.
        self.top_k = max(1, int(top_k))
        # Restricting the index to diagnostically meaningful features stops it
        # being driven by bookkeeping columns (demod band edges, ISO zone index)
        # that legitimately jump when configuration changes.
        if key_features is None:
            key_features = [
                n for n in self.feature_names
                if not n.startswith(("bearing.demod", "iso.zone"))
            ]
        self._key_idx = np.array(
            [self.feature_names.index(n) for n in key_features if n in self.feature_names], dtype=int
        )
        self.last_index: Optional[float] = None
        self.last_index_raw: Optional[float] = None
        self.last_contributions: Dict[str, float] = {}

    def update(
        self,
        timestamp_s: float,
        vector: np.ndarray,
        *,
        speed_hz: float,
        load_frac: float = 1.0,
        external_score: Optional[float] = None,
        external_reference: float = 1.0,
    ) -> Dict[str, object]:
        """Ingest one feature vector.

        Returns a dict with ``index`` (capped, drives alarms), ``index_raw``
        (loosely capped, feed this to :mod:`pdmcore.health.rul`), ``smoothed``,
        ``slope_per_day``, ``state``, ``event`` (or ``None``),
        ``top_contributors`` and ``baseline_ready``.

        While the baseline for the current operating point is not ready the index
        is reported as 0.0 and no alarm can fire -- an asset is not "healthy"
        during commissioning, it is *unknown*, and the dashboard says so.
        """
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {v.size}")

        stats = self.baseline.stats(speed_hz, load_frac)
        contributions: Dict[str, float] = {}
        index_raw = 0.0
        if external_score is not None:
            index = index_raw = float(external_score) / max(1e-9, external_reference)
        elif stats is None:
            index = 0.0
        else:
            med, sigma, usable = stats
            idx = np.array([i for i in self._key_idx if usable[i]], dtype=int)
            if idx.size == 0:
                index = 0.0
            else:  # noqa: RET505
                z_abs = np.abs((v[idx] - med[idx]) / sigma[idx])
                zk = np.clip(z_abs, 0.0, self.Z_CAP)
                z_raw = np.clip(z_abs, 0.0, self.Z_CAP_RAW)
                order = np.argsort(-zk)
                k = min(self.top_k, zk.size)
                index = float(np.mean(zk[order[:k]]) / self.REFERENCE_Z)
                raw_order = np.argsort(-z_raw)
                index_raw = float(np.mean(z_raw[raw_order[:k]]) / self.REFERENCE_Z)
                contributions = {
                    self.feature_names[idx[i]]: float(zk[i] / self.REFERENCE_Z) for i in order[:5]
                }

        smoothed = self.trend.update(timestamp_s, index)
        event = self.alarms.update(timestamp_s, smoothed, evidence=contributions)

        # Adapt the baseline only while normal. A developing fault therefore
        # cannot be absorbed into "normal" -- the failure mode of naive online
        # adaptation, and the reason it is gated on state rather than on time.
        if self.adapt_baseline and self.alarms.state == AlarmState.NORMAL:
            self.baseline.add(v, speed_hz, load_frac)

        self.last_index = index
        self.last_index_raw = index_raw
        self.last_contributions = contributions
        return {
            "index": index,
            "index_raw": index_raw,
            "smoothed": float(smoothed),
            "slope_per_day": self.trend.slope_per_day(),
            "state": self.alarms.state.value,
            "event": event,
            "top_contributors": contributions,
            "baseline_ready": stats is not None,
            "baseline_cells": self.baseline.n_cells,
        }

    def commission(
        self,
        vectors: np.ndarray,
        speeds_hz: Sequence[float],
        loads: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Seed the baseline from a known-good commissioning window."""
        X = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        loads = list(loads) if loads is not None else [1.0] * X.shape[0]
        if not (X.shape[0] == len(speeds_hz) == len(loads)):
            raise ValueError("vectors, speeds_hz and loads must have the same length")
        for row, s, l in zip(X, speeds_hz, loads):
            self.baseline.add(row, float(s), float(l))
        return {
            "n_samples": int(X.shape[0]),
            "n_cells": self.baseline.n_cells,
            "coverage": self.baseline.coverage(),
            "ready_cells": sum(
                1 for k in self.baseline.coverage().values() if k >= self.baseline.min_samples
            ),
        }
