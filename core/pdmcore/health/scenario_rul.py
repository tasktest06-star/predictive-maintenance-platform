"""Scenario-conditional RUL: "how long if we keep running like this?"

What this adds to :mod:`pdmcore.health.rul`
-------------------------------------------
:class:`~pdmcore.health.rul.ParticleFilterRUL` answers one question: how long
does this asset have, *if the future looks like the past*. That is the question
a trend can answer, and the filter abstains until the trend can answer it.

A twin's distinctive capability is the conditional question:

* "How long if we keep it at 100 % load?"
* "How long if we back it off to 70 %?"
* "If we defer to the planned shutdown in six weeks, what is the chance it does
  not get there?"

The last of those is the one maintenance planning actually runs on, and it is
*easier* than the first two: it needs only the posterior mass before a date, not
a median and an interval. That asymmetry is the thing the value-of-information
analysis on this branch turns on.

Method: a damage clock
----------------------
The filter's particles carry ``(h0, a, b)`` for

.. math:: h(t) = h_0 + a\\,(e^{b t} - 1)

in *calendar* time at whatever load the observed trend accumulated under. Load
is introduced as a **time change** rather than as a new parameter. Define a
damage clock

.. math:: \\tau(t) = \\int_0^t \\lambda\\bigl(u(s)\\bigr)\\,ds,
          \\qquad \\lambda(u) = (u/u_{\\mathrm{ref}})^{p}

so that the existing health model holds unchanged in ``tau``, and the only new
quantity is the rate multiplier ``lambda``. The first-crossing time in damage
time is exactly the filter's own crossing time; the calendar answer is obtained
by inverting ``tau``. For a constant load that is one division; for a duty
profile it is a walk over the segments. Nothing about the posterior is
re-estimated, so at ``u == u_ref`` the conditional answer is *identical* to the
unconditional one -- which is asserted as a test, because a scenario model that
quietly changes the baseline answer is worse than none.

The exponent ``p`` is the load-life exponent: ISO 281 gives 3 for ball bearings
and 10/3 for roller bearings on the basic rating life ``L10 ~ (C/P)^p``. It is
**sampled per particle** from a prior rather than fixed, so the conditional
interval widens with distance from the observed load, and does not widen at all
at the observed load. The fraction of the interval width attributable to the
load model rather than to the degradation posterior is reported
(:attr:`ScenarioRULEstimate.load_model_width_share`) -- if that share is large,
the number is a statement about an assumed exponent wearing the clothes of a
measurement.

Abstention, and why it gets *stricter* here
-------------------------------------------
A conditional RUL from insufficient data is more dangerous than an
unconditional one, because it looks like it answers a specific question. Four
gates, all of which can fire:

1. The unconditional estimate must be ``ok``. Everything
   :class:`ParticleFilterRUL` refuses, this refuses, with the same note.
2. More observations than the unconditional estimate needs
   (``extra_observations``, default +6). Extrapolating a *rate* in a new regime
   needs more evidence than extrapolating a level.
3. The load history must be known. If the loads under which the observed trend
   accumulated were not recorded, there is no ``u_ref``, so "70 % load" names
   nothing. Refuses rather than assuming 100 %, which is the assumption that
   would be made silently and is wrong on every variable-duty asset.
4. The scenario load must be within ``max_load_ratio`` of the observed load.
   Asking about 100 % from a history entirely at 40 % is not a conditional
   prediction, it is an extrapolation of an assumed exponent over a factor of
   2.5; at ``p = 3`` that is a 16x rate change with no supporting observation.

What this is not
----------------
There is no physics here beyond the load-life exponent. This module does not
know the asset's thermal state, its lubrication regime or its actual contact
loads; a physics twin (see the sibling branch ``twin/motor-physics-core``) would
replace ``lambda(u)`` with something derived rather than assumed. The boundary
is deliberately at this function: everything upstream of ``lambda`` is physics,
everything downstream is decision theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rul import ParticleFilterRUL, RULEstimate

__all__ = [
    "LoadScenario",
    "ScenarioRULEstimate",
    "ScenarioRUL",
    "BALL_BEARING_EXPONENT",
    "ROLLER_BEARING_EXPONENT",
]

#: ISO 281 basic rating life exponents for ``L10 ~ (C/P)^p``.
BALL_BEARING_EXPONENT = 3.0
ROLLER_BEARING_EXPONENT = 10.0 / 3.0


@dataclass(frozen=True)
class LoadScenario:
    """A future duty profile.

    ``segments`` is a sequence of ``(duration_days, load_frac)``. The final
    segment extends indefinitely, whatever duration it declares, because a RUL
    question has no end date. A constant-load scenario is one segment.

    ``load_frac`` is the same quantity the health index bins its baseline on
    (:class:`pdmcore.health.index.OperatingBaseline`) -- fraction of rated duty,
    in ``(0, 1]`` by convention but allowed above 1 for a deliberately
    overloaded asset, because that is a question plants ask.
    """

    name: str
    segments: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a scenario needs at least one segment")
        for dur, load in self.segments:
            if dur <= 0:
                raise ValueError(f"segment duration must be > 0 (got {dur})")
            if load <= 0:
                raise ValueError(
                    f"segment load must be > 0 (got {load}); a stopped asset does not "
                    "have a remaining life in days, it has one in running hours -- "
                    "model a shutdown by shortening the horizon, not by load 0"
                )

    @classmethod
    def constant(cls, load_frac: float, *, name: Optional[str] = None) -> "LoadScenario":
        return cls(name=name or f"constant_{load_frac:.2f}",
                   segments=((1e6, float(load_frac)),))

    @property
    def is_constant(self) -> bool:
        return len(self.segments) == 1

    @property
    def mean_load(self) -> float:
        """Duration-weighted mean load over the finite segments."""
        if self.is_constant:
            return float(self.segments[0][1])
        total = sum(d for d, _ in self.segments[:-1])
        if total <= 0:  # pragma: no cover - guarded by __post_init__
            return float(self.segments[-1][1])
        return float(sum(d * u for d, u in self.segments[:-1]) / total)

    @property
    def peak_load(self) -> float:
        return float(max(u for _, u in self.segments))

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "segments": [[round(float(d), 3), round(float(u), 4)] for d, u in self.segments],
            "mean_load": round(self.mean_load, 4),
            "peak_load": round(self.peak_load, 4),
        }


@dataclass
class ScenarioRULEstimate:
    """A scenario-conditional RUL posterior, or an explicit refusal.

    ``status`` mirrors :class:`~pdmcore.health.rul.RULEstimate` and adds two
    refusals specific to the conditional question:

    ``load_history_unknown``
        The observed loads were not recorded, so there is no reference load to
        condition against.
    ``scenario_not_identifiable``
        The requested load is too far from the observed load for the load-life
        exponent to be anything but an extrapolation.
    """

    status: str
    scenario: Optional[LoadScenario] = None
    median_days: Optional[float] = None
    p10_days: Optional[float] = None
    p90_days: Optional[float] = None
    mean_days: Optional[float] = None
    reference_load: Optional[float] = None
    load_history_cv: Optional[float] = None
    rate_multiplier_median: Optional[float] = None
    #: Fraction of the p10-p90 width that comes from uncertainty in the
    #: load-life exponent rather than from the degradation posterior.
    load_model_width_share: Optional[float] = None
    unconditional: Optional[RULEstimate] = None
    n_observations: int = 0
    horizon_censored_share: Optional[float] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def interval_width_days(self) -> Optional[float]:
        if self.p10_days is None or self.p90_days is None:
            return None
        return float(self.p90_days - self.p10_days)

    def as_dict(self) -> Dict[str, object]:
        def r(v, n=3):
            return None if v is None else round(float(v), n)

        return {
            "status": self.status,
            "scenario": None if self.scenario is None else self.scenario.as_dict(),
            "median_days": r(self.median_days),
            "p10_days": r(self.p10_days),
            "p90_days": r(self.p90_days),
            "mean_days": r(self.mean_days),
            "interval_width_days": r(self.interval_width_days),
            "reference_load": r(self.reference_load, 4),
            "load_history_cv": r(self.load_history_cv, 4),
            "rate_multiplier_median": r(self.rate_multiplier_median, 4),
            "load_model_width_share": r(self.load_model_width_share, 4),
            "horizon_censored_share": r(self.horizon_censored_share, 4),
            "n_observations": int(self.n_observations),
            "unconditional_status": None if self.unconditional is None else self.unconditional.status,
            "unconditional_median_days": (
                None if self.unconditional is None else r(self.unconditional.median_days)
            ),
            "note": self.note,
        }


@dataclass
class ScenarioRUL:
    """Wraps a :class:`ParticleFilterRUL` and answers conditional questions.

    The wrapped filter is the single source of degradation evidence; this class
    adds no parameters to it and never re-fits it. Feed observations through
    :meth:`update` so that the load history is recorded alongside the health
    series.
    """

    rul: ParticleFilterRUL
    #: Load-life exponent prior. ``p ~ N(mean, sd)`` truncated to ``[lo, hi]``.
    load_exponent: float = BALL_BEARING_EXPONENT
    load_exponent_sd: float = 0.6
    load_exponent_bounds: Tuple[float, float] = (1.0, 5.0)
    #: Extra observations required beyond the unconditional filter's own gate.
    extra_observations: int = 6
    #: Largest ``max(u/u_ref, u_ref/u)`` for which the exponent is treated as
    #: interpolation. 1.6 at ``p = 3`` is a 4x rate change, which is already a
    #: lot to accept from an assumed exponent.
    max_load_ratio: float = 1.6
    #: Observations used to define the reference load, matching the filter's own
    #: trend window: the reference load must be the load that produced the
    #: *observed* trend, not the load the asset saw a year ago.
    load_window: Optional[int] = None
    seed: int = 7

    _loads: List[Optional[float]] = field(default_factory=list, init=False)

    # -- plumbing ------------------------------------------------------------
    def _exponents(self, n: int) -> np.ndarray:
        """``n`` draws from the load-life exponent prior.

        Re-seeded from :attr:`seed` on every call rather than drawn from a
        running generator, so that :meth:`estimate` is idempotent. A decision
        service that returns a slightly different RUL each time it is asked the
        same question is unauditable, and the drift is small enough that nobody
        would notice until a disputed work order.
        """
        rng = np.random.default_rng(self.seed)
        lo, hi = self.load_exponent_bounds
        return np.clip(rng.normal(self.load_exponent, self.load_exponent_sd, n), lo, hi)

    def update(self, t_days: float, health: float, *, load_frac: Optional[float] = None) -> None:
        """Add one health observation and the load it was taken under.

        ``load_frac=None`` is accepted and recorded as unknown rather than
        defaulted, so that a partially instrumented asset refuses conditional
        questions instead of answering them against an invented reference.
        """
        self.rul.update(t_days, health)
        if load_frac is not None and load_frac <= 0:
            raise ValueError("load_frac must be > 0 or None")
        self._loads.append(None if load_frac is None else float(load_frac))

    # -- the reference load --------------------------------------------------
    @property
    def _window(self) -> int:
        if self.load_window is not None:
            return max(1, int(self.load_window))
        return max(int(self.rul.min_observations), int(self.rul.trend_window))

    def reference_load(self) -> Optional[float]:
        """Mean load over the trend window, or ``None`` if any is unknown.

        A mean, not a median, because damage accumulates additively: the load
        that produced the observed damage rate is the one whose rate multiplier
        averages to the observed one. That is strictly ``mean(u^p)^(1/p)``
        rather than ``mean(u)``; for the load spreads real assets show
        (CV < 0.3) the two differ by under 4 % at ``p = 3``, and the difference
        is dwarfed by the exponent prior -- but it is biased low, so the
        exact form is used.
        """
        window = self._loads[-self._window:]
        if not window or any(u is None for u in window):
            return None
        u = np.asarray(window, dtype=np.float64)
        p = float(self.load_exponent)
        return float(np.mean(u**p) ** (1.0 / p))

    def load_history_cv(self) -> Optional[float]:
        window = self._loads[-self._window:]
        if not window or any(u is None for u in window):
            return None
        u = np.asarray(window, dtype=np.float64)
        m = float(np.mean(u))
        return float(np.std(u) / m) if m > 0 else None

    # -- the damage clock ----------------------------------------------------
    def _solve_calendar_time(
        self, remaining_damage_days: np.ndarray, scenario: LoadScenario, u_ref: float,
        *, fixed_exponent: bool = False,
    ) -> np.ndarray:
        """Invert the damage clock: calendar days until ``tau`` is exhausted.

        The exponent is drawn **once** per particle and held across every
        segment of the scenario: a bearing has one load-life exponent, and
        re-drawing it per segment would average the load model's uncertainty
        away over a duty profile, making a multi-segment scenario look more
        certain than a constant one.
        """
        n = remaining_damage_days.size
        p = (np.full(n, float(self.load_exponent)) if fixed_exponent else self._exponents(n))
        remaining = np.array(remaining_damage_days, dtype=np.float64, copy=True)
        elapsed = np.zeros(n, dtype=np.float64)
        done = remaining <= 0.0
        elapsed[done] = 0.0
        for i, (dur, load) in enumerate(scenario.segments):
            lam = np.maximum((float(load) / float(u_ref)) ** p, 1e-12)
            last = i == len(scenario.segments) - 1
            # Damage time available in this segment (the final segment is open).
            avail = np.full(n, float("inf")) if last else lam * float(dur)
            crosses = (~done) & (remaining <= avail)
            elapsed[crosses] += remaining[crosses] / lam[crosses]
            done |= crosses
            if np.all(done):
                break
            carry = ~done
            elapsed[carry] += float(dur)
            remaining[carry] -= lam[carry] * float(dur)
        return elapsed

    def _remaining_damage_days(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(remaining_damage_days, weights)`` from the filter's particles.

        The crossing-time arithmetic deliberately mirrors
        :meth:`ParticleFilterRUL.estimate` rather than reaching for a private
        helper: if that method's model changes, this one has to be changed
        knowingly, and the equality test at ``u == u_ref`` fails if it is not.
        """
        particles = getattr(self.rul, "_particles", None)
        weights = getattr(self.rul, "_weights", None)
        if particles is None or weights is None:
            return None
        t_now = self.rul._t[-1]
        h0, a, b = particles[:, 0], particles[:, 1], particles[:, 2]
        target = (self.rul.failure_threshold - h0) / np.maximum(a, 1e-12) + 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            t_fail = np.where(
                target > 1.0,
                np.log(np.maximum(target, 1.0 + 1e-12)) / np.maximum(b, 1e-12),
                np.inf,
            )
        remaining = t_fail - t_now
        remaining = np.where(np.isfinite(remaining), remaining, self.rul.horizon_days)
        remaining = np.clip(remaining, 0.0, self.rul.horizon_days)
        w = np.asarray(weights, dtype=np.float64)
        w = w / float(np.sum(w))
        return remaining, w

    # -- public --------------------------------------------------------------
    def _refusal(self, status: str, note: str, scenario: Optional[LoadScenario],
                 unconditional: Optional[RULEstimate]) -> ScenarioRULEstimate:
        return ScenarioRULEstimate(
            status=status,
            scenario=scenario,
            reference_load=self.reference_load(),
            load_history_cv=self.load_history_cv(),
            unconditional=unconditional,
            n_observations=len(self.rul._t),
            note=note,
        )

    def _gate(self, scenario: LoadScenario) -> Tuple[Optional[ScenarioRULEstimate], RULEstimate, float]:
        """All four abstention gates. Returns ``(refusal_or_None, uncond, u_ref)``."""
        uncond = self.rul.estimate()
        if uncond.status != "ok":
            return (
                self._refusal(uncond.status, uncond.note or
                              "unconditional RUL is not available, so no conditional one is",
                              scenario, uncond),
                uncond, float("nan"),
            )
        need = int(self.rul.min_observations) + int(self.extra_observations)
        n_obs = len(self.rul._t)
        if n_obs < need:
            return (
                self._refusal(
                    "insufficient_evidence",
                    f"a conditional RUL needs >= {need} observations (have {n_obs}); the "
                    f"unconditional estimate is available at {self.rul.min_observations} but "
                    "conditioning extrapolates a rate into a regime that has not been "
                    "observed, which needs more evidence, not the same",
                    scenario, uncond),
                uncond, float("nan"),
            )
        u_ref = self.reference_load()
        if u_ref is None:
            return (
                self._refusal(
                    "load_history_unknown",
                    f"load was not recorded for all of the last {self._window} observations, "
                    "so there is no reference duty to condition against; "
                    f"'{scenario.name}' names nothing without one",
                    scenario, uncond),
                uncond, float("nan"),
            )
        ratio = max(scenario.peak_load / u_ref, u_ref / scenario.peak_load)
        if ratio > self.max_load_ratio:
            rate_change = ratio ** self.load_exponent
            return (
                self._refusal(
                    "scenario_not_identifiable",
                    f"scenario peak load {scenario.peak_load:.2f} is {ratio:.2f}x the observed "
                    f"reference load {u_ref:.2f}, beyond the {self.max_load_ratio:.2f}x limit; "
                    f"at exponent p={self.load_exponent:.2f} that is a {rate_change:.1f}x damage-"
                    "rate change supported by no observation in this regime",
                    scenario, uncond),
                uncond, u_ref,
            )
        return None, uncond, u_ref

    def estimate(self, scenario: LoadScenario) -> ScenarioRULEstimate:
        """Conditional RUL under ``scenario``, or an explicit refusal."""
        refusal, uncond, u_ref = self._gate(scenario)
        if refusal is not None:
            return refusal
        state = self._remaining_damage_days()
        if state is None:  # pragma: no cover - unreachable once uncond is ok
            return self._refusal("insufficient_evidence", "filter has no particles",
                                 scenario, uncond)
        remaining, w = state
        horizon = float(self.rul.horizon_days)

        days = np.clip(self._solve_calendar_time(remaining, scenario, u_ref), 0.0, horizon)
        # Same solve with the exponent pinned at its prior mean: the difference
        # in interval width is the contribution of the load model.
        days_fixed = np.clip(
            self._solve_calendar_time(remaining, scenario, u_ref, fixed_exponent=True),
            0.0, horizon,
        )

        def wq(values: np.ndarray, q: float) -> float:
            order = np.argsort(values)
            cw = np.cumsum(w[order])
            return float(values[order][int(np.searchsorted(cw, q))])

        p10, median, p90 = wq(days, 0.1), wq(days, 0.5), wq(days, 0.9)
        width = p90 - p10
        width_fixed = wq(days_fixed, 0.9) - wq(days_fixed, 0.1)
        share = 0.0 if width <= 0 else max(0.0, (width - width_fixed) / width)
        censored = float(np.sum(w[days >= horizon * 0.999]))

        lam_median = (scenario.mean_load / u_ref) ** float(self.load_exponent)
        notes: List[str] = []
        if censored > 0.2:
            notes.append(
                f"{censored * 100:.0f}% of the posterior exceeds the {horizon:.0f}-day horizon; "
                "the upper bound is censored, not estimated"
            )
        cv = self.load_history_cv()
        if cv is not None and cv > 0.25:
            notes.append(
                f"observed load varied with CV {cv:.2f}; the reference duty is an average over a "
                "genuinely variable history, so the conditional answer inherits that averaging"
            )
        if share > 0.5:
            notes.append(
                f"{share * 100:.0f}% of the interval width comes from the assumed load-life "
                "exponent, not from the degradation evidence -- this is a statement about "
                "ISO 281, not about this bearing"
            )
        return ScenarioRULEstimate(
            status="ok",
            scenario=scenario,
            median_days=median,
            p10_days=p10,
            p90_days=p90,
            mean_days=float(np.sum(w * days)),
            reference_load=u_ref,
            load_history_cv=cv,
            rate_multiplier_median=lam_median,
            load_model_width_share=share,
            unconditional=uncond,
            n_observations=len(self.rul._t),
            horizon_censored_share=censored,
            note="; ".join(notes),
        )

    def failure_day_samples(self, scenario: LoadScenario) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(days_to_failure, weights)`` under ``scenario``, or ``None`` if refused.

        This is what the decision layer consumes: a weighted sample of failure
        times, not a summary. Summarising first and deciding on the summary is
        how an asymmetric cost function gets evaluated at the median, which is
        the wrong point when the cost of being late is 20x the cost of being
        early.
        """
        refusal, _uncond, u_ref = self._gate(scenario)
        if refusal is not None:
            return None
        state = self._remaining_damage_days()
        if state is None:  # pragma: no cover
            return None
        remaining, w = state
        days = np.clip(
            self._solve_calendar_time(remaining, scenario, u_ref),
            0.0, float(self.rul.horizon_days),
        )
        return days, w

    def probability_of_failure_before(
        self, scenario: LoadScenario, day: float
    ) -> Optional[float]:
        """P(failure within ``day`` days) under ``scenario``, or ``None`` if refused.

        The planner's actual question -- "will it reach the shutdown?" -- and the
        cheapest one to answer, because it needs posterior *mass* rather than a
        quantile. An estimator far too coarse to quote a median can be precise
        enough for this, which is the whole point of the value-of-information
        result on this branch.
        """
        samples = self.failure_day_samples(scenario)
        if samples is None:
            return None
        days, w = samples
        return float(np.sum(w[days <= float(day)]))

    def compare(
        self, scenarios: Sequence[LoadScenario]
    ) -> Dict[str, ScenarioRULEstimate]:
        """Estimate under several scenarios, keyed by scenario name."""
        return {s.name: self.estimate(s) for s in scenarios}

    # -- convenience ---------------------------------------------------------
    @classmethod
    def from_threshold(
        cls, failure_threshold: float, **kwargs
    ) -> "ScenarioRUL":
        """Build with a fresh :class:`ParticleFilterRUL` at ``failure_threshold``."""
        rul_kwargs = {}
        for key in ("n_particles", "min_observations", "min_b", "min_r2", "min_health",
                    "trend_window", "obs_noise", "horizon_days", "seed"):
            if key in kwargs:
                rul_kwargs[key] = kwargs.pop(key)
        return cls(rul=ParticleFilterRUL(failure_threshold=failure_threshold, **rul_kwargs),
                   **kwargs)


def doubling_time_at_load(doubling_time_days: float, load_ratio: float,
                          exponent: float = BALL_BEARING_EXPONENT) -> float:
    """Rescale an observed doubling time to a different load.

    Exposed because it is the one-line version of the whole module and is
    useful in a report: a 30-day doubling time at 70 % load becomes
    ``30 / (1/0.7)^3 = 10.3`` days at 100 %. If that sentence is all a reader
    takes from this module, it is the right sentence.
    """
    if load_ratio <= 0:
        raise ValueError("load_ratio must be > 0")
    return float(doubling_time_days) / (float(load_ratio) ** float(exponent))
