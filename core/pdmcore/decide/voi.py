"""How much is a better RUL estimate actually worth?

The question
------------
A prognostics programme is sold on accuracy. The honest question is whether
accuracy changes the action, because an estimate that does not change the action
is worth exactly nothing however good it is. If the decision is "do it at the
next shutdown" whether the remaining life is six weeks or ten, then improving
the estimate from +-4 weeks to +-2 weeks buys nothing at all -- and the
engineering effort that bought it was spent on a number nobody acted on
differently.

The framing
-----------
Standard decision-theoretic value of information, with the decision set from
:mod:`pdmcore.decide.policy` (shutdown windows, an unscheduled stop, run to
failure) and the cost model from :mod:`pdmcore.decide.costs`.

For each Monte-Carlo draw:

1. A true remaining life ``T`` is drawn.
2. An estimator of log-precision ``s`` observes ``T`` with multiplicative noise
   and reports the implied posterior. The posterior is *calibrated* by
   construction (``T | m ~ LogNormal(log m, s)``), which is generous to the
   estimator: a real one is biased, and ``bias_log`` exists to measure how much
   that costs.
3. The planner chooses the minimum-expected-cost opportunity under that
   posterior.
4. The cost actually incurred is evaluated against the true ``T``.

Averaging over draws gives ``C(s)``. Two reference points bracket it:

``C_perfect``   the cost of choosing with ``T`` known -- ``s = 0``.
``C_baseline``  the better of the two policies that need no estimate at all:
                run to failure, and intervene at the next shutdown regardless.

``EVPI = C_baseline - C_perfect`` is the entire value available to any
prognostics system, ever, at this cost model and this shutdown calendar. It is
the ceiling the sibling twin branches are competing for, and it is usually much
smaller than the headline "cost of unplanned downtime" that justifies the
programme.

The decision-relevant precision
-------------------------------
:func:`decision_relevant_precision` finds the widest credible interval that
still captures a stated fraction (default 95 %) of ``EVPI``. Below that width,
measurement has stopped changing the action: more accuracy is free to produce
and worth nothing to have. The expectation -- confirmed by the results on this
branch -- is that it scales with the **shutdown interval**, not with anything
about the sensor, because the action can only change at window granularity.

A deliberate simplification, named
----------------------------------
This is a *single-shot* decision: choose once, now. A real programme re-decides
every week as new data arrives, and waiting is itself an option with value. The
sequential version is strictly better than the single-shot one and needs *less*
precision now, because precision deferred is precision you may never need to
buy. So the numbers here are an **upper bound** on the value of RUL precision --
generous to the twin -- and they are still small. The end-to-end simulation on
this branch runs the sequential policy and can be compared against them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .costs import MaintenanceCostModel
from .policy import RUN_TO_FAILURE, Opportunity, evaluate_opportunities, shutdown_calendar

__all__ = [
    "PrecisionPoint",
    "VoIStudy",
    "interval_width_days",
    "log_sd_for_width",
    "realised_cost",
    "voi_curve",
    "decision_relevant_precision",
    "break_even_precision",
    "precision_table",
    "precision_step_value",
]

#: Two-sided 80 % normal quantile: the p10/p90 pair the RUL estimator reports.
_Z80 = 1.2815515655446004


def interval_width_days(median_days: float, log_sd: float) -> float:
    """p10-p90 width of a lognormal with the given median and log-sd."""
    return float(median_days) * (math.exp(_Z80 * log_sd) - math.exp(-_Z80 * log_sd))


def log_sd_for_width(median_days: float, width_days: float) -> float:
    """Inverse of :func:`interval_width_days` -- the log-sd giving that width.

    Expressing precision as a p10-p90 width in days rather than as a log-sd is
    not cosmetic: it is the only form a maintenance planner can check against
    experience, and a figure axis in log-sd units cannot be argued with.
    """
    if median_days <= 0:
        raise ValueError("median_days must be > 0")
    r = float(width_days) / float(median_days)
    if r <= 0:
        return 0.0
    # Solve e^{zs} - e^{-zs} = r, i.e. 2 sinh(zs) = r.
    return float(math.asinh(r / 2.0) / _Z80)


def realised_cost(
    opportunity: Opportunity, true_life_days: float, cost: MaintenanceCostModel
) -> float:
    """What the plant actually pays, with hindsight.

    The counterfactual the whole analysis rests on: the planner decided under a
    posterior, and this is the bill. Note that run-to-failure has no good luck
    available to it -- the asset always fails -- while a planned intervention can
    be wasted but can never be catastrophic. That asymmetry, not the RUL's
    accuracy, is most of what condition monitoring sells.
    """
    if opportunity.kind == "run_to_failure" or true_life_days <= opportunity.day:
        return cost.unplanned_event_eur()
    return (opportunity.intervention_eur(cost)
            + cost.residual_life_eur(true_life_days - opportunity.day))


@dataclass
class PrecisionPoint:
    """Expected cost and behaviour of the policy at one estimator precision."""

    log_sd: float
    mean_interval_days: float
    expected_cost_eur: float
    value_vs_baseline_eur: float
    information_efficiency: float
    p_action_matches_perfect: float
    p_failed_in_service: float
    action_mix: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "log_sd": round(self.log_sd, 4),
            "interval_width_days": round(self.mean_interval_days, 2),
            "interval_width_weeks": round(self.mean_interval_days / 7.0, 2),
            "expected_cost_eur": round(self.expected_cost_eur, 2),
            "value_vs_baseline_eur": round(self.value_vs_baseline_eur, 2),
            "information_efficiency": round(self.information_efficiency, 4),
            "p_action_matches_perfect": round(self.p_action_matches_perfect, 4),
            "p_failed_in_service": round(self.p_failed_in_service, 4),
            "action_mix": {k: round(v, 4) for k, v in sorted(self.action_mix.items())},
        }


@dataclass
class VoIStudy:
    """A value-of-information experiment over one cost model and calendar.

    Parameters
    ----------
    cost
        The plant's cost model.
    shutdown_interval_days
        Spacing of planned shutdown windows.
    horizon_days
        How far ahead windows are offered. Must comfortably exceed the true-life
        distribution or the decision set is artificially truncated and
        run-to-failure wins by construction.
    true_life_median_days, true_life_log_sd
        Distribution of true remaining life *at the decision epoch*, i.e. after
        detection. Defaults are placeholders; the branch's prototype injects the
        values measured by running the real pipeline on
        :func:`pdmcore.sim.degradation_series`.
    allow_unscheduled_stop
        Whether stopping the asset outside a shutdown is permitted. Turning this
        off is the common real constraint (a continuous process cannot be
        stopped on a Tuesday) and it *raises* the value of precision, because
        the only lever left is which window.
    """

    cost: MaintenanceCostModel
    shutdown_interval_days: float = 42.0
    horizon_days: float = 400.0
    true_life_median_days: float = 90.0
    true_life_log_sd: float = 0.55
    allow_unscheduled_stop: bool = True
    first_shutdown_day: Optional[float] = None
    n_draws: int = 4000
    n_posterior: int = 600
    bias_log: float = 0.0
    seed: int = 11

    def opportunities(self) -> List[Opportunity]:
        opps = shutdown_calendar(
            interval_days=self.shutdown_interval_days,
            horizon_days=self.horizon_days,
            first_day=self.first_shutdown_day,
        )
        if self.allow_unscheduled_stop:
            opps = [Opportunity(name="stop_now", day=0.0, kind="unscheduled_stop")] + opps
        return opps + [RUN_TO_FAILURE]

    # -- draws ---------------------------------------------------------------
    def true_lives(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return self.true_life_median_days * np.exp(
            rng.normal(0.0, self.true_life_log_sd, self.n_draws)
        )

    # -- reference points ----------------------------------------------------
    def baselines(self) -> Dict[str, float]:
        """The two policies that need no estimate, plus perfect information."""
        T = self.true_lives()
        opps = self.opportunities()
        shutdowns = [o for o in opps if o.kind == "planned_shutdown"]
        next_shutdown = min(shutdowns, key=lambda o: o.day)
        rtf = float(np.mean([realised_cost(RUN_TO_FAILURE, t, self.cost) for t in T]))
        blanket = float(np.mean([realised_cost(next_shutdown, t, self.cost) for t in T]))
        perfect = float(np.mean([
            min(realised_cost(o, t, self.cost) for o in opps) for t in T
        ]))
        best_baseline = min(rtf, blanket)
        return {
            "run_to_failure_eur": rtf,
            "blanket_next_shutdown_eur": blanket,
            "best_baseline_eur": best_baseline,
            "best_baseline_name": "run_to_failure" if rtf <= blanket else "blanket_next_shutdown",
            "perfect_information_eur": perfect,
            "evpi_eur": best_baseline - perfect,
        }

    # -- the curve -----------------------------------------------------------
    def _plan(self, opps: Sequence[Opportunity]) -> Dict[str, np.ndarray]:
        """Pre-compute the per-opportunity constants the fast path needs."""
        return {
            "days": np.array([o.day for o in opps], dtype=np.float64),
            "intervention": np.array(
                [0.0 if o.kind == "run_to_failure" else o.intervention_eur(self.cost)
                 for o in opps], dtype=np.float64),
            "is_rtf": np.array([o.kind == "run_to_failure" for o in opps], dtype=bool),
        }

    def expected_costs_fast(
        self, posterior_days: np.ndarray, plan: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Vectorised twin of :func:`pdmcore.decide.policy.evaluate_opportunities`.

        The VoI sweep evaluates every opportunity against every posterior for
        every precision -- millions of evaluations -- so the per-opportunity
        Python loop in ``policy`` is too slow. This is the same arithmetic done
        with one sort and two cumulative sums.

        It is a second implementation of a formula, which this repository treats
        as a liability. ``core/tests/test_decide_voi.py`` asserts the two agree
        to 1e-9 on randomised inputs, so a change to one that is not made to the
        other fails a test rather than silently shifting every VoI number.
        """
        x = np.sort(np.asarray(posterior_days, dtype=np.float64))
        n = x.size
        c_fail = self.cost.unplanned_event_eur()
        idx = np.searchsorted(x, plan["days"], side="right")
        p_fail = idx / n
        p_surv = 1.0 - p_fail
        # Mean of (T - day) over the surviving tail.
        tail_sum = np.r_[np.cumsum(x[::-1])[::-1], 0.0]
        sum_above = tail_sum[np.minimum(idx, n)]
        with np.errstate(invalid="ignore", divide="ignore"):
            residual_days = np.where(
                p_surv > 0, sum_above / n / np.maximum(p_surv, 1e-15) - plan["days"], 0.0)
        residual_days = np.where(np.isfinite(residual_days), residual_days, 0.0)
        residual_days = np.maximum(residual_days, 0.0)
        residual_eur = self.cost.spare_part_eur * np.clip(
            residual_days, 0.0, self.cost.design_life_days) / self.cost.design_life_days
        out = p_fail * c_fail + p_surv * (plan["intervention"] + residual_eur)
        return np.where(plan["is_rtf"], c_fail, out)

    def _best_action_under_posterior(
        self, posterior_days: np.ndarray, opps: Sequence[Opportunity],
        plan: Optional[Dict[str, np.ndarray]] = None,
    ) -> Opportunity:
        if plan is None:
            evals = evaluate_opportunities(opps, posterior_days, None, self.cost)
            return min(evals, key=lambda e: e.expected_eur).opportunity
        return opps[int(np.argmin(self.expected_costs_fast(posterior_days, plan)))]

    def _perfect_actions(self, opps: Sequence[Opportunity], T: np.ndarray) -> List[str]:
        out = []
        for t in T:
            best = min(opps, key=lambda o: realised_cost(o, t, self.cost))
            out.append(best.name)
        return out

    def point(self, log_sd: float, *, _cache: Optional[Dict] = None) -> PrecisionPoint:
        """Evaluate the policy at one estimator precision."""
        opps = self.opportunities()
        T = self.true_lives()
        base = (_cache or {}).get("baselines") or self.baselines()
        perfect_actions = (_cache or {}).get("perfect_actions") or self._perfect_actions(opps, T)
        rng = np.random.default_rng(self.seed + 1000)
        plan = self._plan(opps)
        s = float(log_sd)
        costs = np.empty(T.size)
        widths = np.empty(T.size)
        matches = 0
        failures = 0
        mix: Dict[str, int] = {}
        for i, t in enumerate(T):
            if s <= 1e-9:
                posterior = np.full(8, t)
                median = t
            else:
                m = t * math.exp(rng.normal(self.bias_log, s))
                posterior = m * np.exp(rng.normal(0.0, s, self.n_posterior))
                median = m
            widths[i] = interval_width_days(median, s)
            chosen = self._best_action_under_posterior(posterior, opps, plan)
            costs[i] = realised_cost(chosen, t, self.cost)
            mix[chosen.name] = mix.get(chosen.name, 0) + 1
            if chosen.name == perfect_actions[i]:
                matches += 1
            if chosen.kind == "run_to_failure" or t <= chosen.day:
                failures += 1
        expected = float(np.mean(costs))
        value = base["best_baseline_eur"] - expected
        evpi = base["evpi_eur"]
        return PrecisionPoint(
            log_sd=s,
            mean_interval_days=float(np.mean(widths)),
            expected_cost_eur=expected,
            value_vs_baseline_eur=value,
            information_efficiency=(float("nan") if evpi <= 0 else value / evpi),
            p_action_matches_perfect=matches / T.size,
            p_failed_in_service=failures / T.size,
            action_mix={k: v / T.size for k, v in mix.items()},
        )

    def curve(self, log_sds: Sequence[float]) -> Dict[str, object]:
        """Evaluate a sweep of precisions, with the reference points attached."""
        opps = self.opportunities()
        T = self.true_lives()
        cache = {"baselines": self.baselines(),
                 "perfect_actions": self._perfect_actions(opps, T)}
        points = [self.point(s, _cache=cache) for s in log_sds]
        return {
            "config": self.as_dict(),
            "baselines": {k: (round(v, 2) if isinstance(v, float) else v)
                          for k, v in cache["baselines"].items()},
            "points": [p.as_dict() for p in points],
            "_points": points,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "shutdown_interval_days": self.shutdown_interval_days,
            "horizon_days": self.horizon_days,
            "true_life_median_days": self.true_life_median_days,
            "true_life_log_sd": self.true_life_log_sd,
            "allow_unscheduled_stop": self.allow_unscheduled_stop,
            "n_draws": self.n_draws,
            "bias_log": self.bias_log,
            "seed": self.seed,
            "cost_fingerprint": self.cost.fingerprint(),
        }


def voi_curve(
    study: VoIStudy, *, widths_weeks: Sequence[float] = (0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24)
) -> Dict[str, object]:
    """Convenience: sweep by p10-p90 width in weeks rather than by log-sd.

    The widths are converted at the true-life median, so the axis label
    ("+-N weeks on a 90-day life") is the thing a planner recognises.
    """
    sds = [log_sd_for_width(study.true_life_median_days, w * 7.0) for w in widths_weeks]
    out = study.curve([0.0] + list(sds))
    out["widths_weeks"] = [0.0] + list(widths_weeks)
    return out


def decision_relevant_precision(
    curve: Dict[str, object], *, capture: float = 0.95
) -> Dict[str, object]:
    """The widest credible interval that still captures ``capture`` of EVPI.

    Returns the width in days and weeks, and as a fraction of the shutdown
    interval -- the last of which is the result worth reading, because it says
    the answer is a property of the *maintenance calendar*, not of the sensor.

    ``None`` when no swept precision reaches the capture threshold, which is
    itself a finding: it means precision is the binding constraint at this cost
    model, and the sibling twin branches' accuracy work has somewhere to go.
    """
    points: List[PrecisionPoint] = list(curve["_points"])
    base = curve["baselines"]
    evpi = float(base["evpi_eur"])
    interval = float(curve["config"]["shutdown_interval_days"])
    qualifying = [p for p in points
                  if p.mean_interval_days > 0 and p.information_efficiency >= capture]
    if not qualifying:
        return {
            "capture": capture,
            "evpi_eur": round(evpi, 2),
            "width_days": None,
            "width_weeks": None,
            "width_over_shutdown_interval": None,
            "note": (
                "no swept precision captured the target share of EVPI; at this cost model "
                "RUL precision is the binding constraint rather than the calendar"
            ),
        }
    best = max(qualifying, key=lambda p: p.mean_interval_days)
    return {
        "capture": capture,
        "evpi_eur": round(evpi, 2),
        "width_days": round(best.mean_interval_days, 2),
        "width_weeks": round(best.mean_interval_days / 7.0, 2),
        "width_over_shutdown_interval": round(best.mean_interval_days / interval, 3),
        "at_log_sd": round(best.log_sd, 4),
        "expected_cost_eur": round(best.expected_cost_eur, 2),
        "p_action_matches_perfect": round(best.p_action_matches_perfect, 4),
        "note": (
            "credible intervals narrower than this change the action in fewer cases than "
            f"{(1 - capture) * 100:.0f} % of EVPI is worth; measurement effort spent getting "
            "below it is spent on a number the planner does not act on differently"
        ),
    }


def break_even_precision(curve: Dict[str, object]) -> Dict[str, object]:
    """The interval width at which acting on the RUL stops beating the calendar.

    ``information_efficiency`` is the share of EVPI the policy captures. It goes
    *negative* at wide intervals: a policy that acts on a very imprecise
    estimate does worse than the estimate-free policy of intervening at the
    next shutdown, because it talks itself into stopping assets early and into
    deferring ones it should not. The crossing point is the quantitative
    version of :mod:`pdmcore.health.rul`'s design position -- a bad RUL is worse
    than no RUL -- and it is the number that justifies that module's
    abstention behaviour in money rather than in principle.

    Returns ``None`` for the width when the swept range never crosses zero.
    """
    points: List[PrecisionPoint] = list(curve["_points"])
    usable = [p for p in points if p.mean_interval_days > 0
              and not math.isnan(p.information_efficiency)]
    usable.sort(key=lambda p: p.mean_interval_days)
    crossing = None
    for a, b in zip(usable, usable[1:]):
        if a.information_efficiency >= 0.0 > b.information_efficiency:
            span = a.information_efficiency - b.information_efficiency
            frac = a.information_efficiency / span if span > 0 else 0.0
            crossing = a.mean_interval_days + frac * (b.mean_interval_days - a.mean_interval_days)
            break
    interval = float(curve["config"]["shutdown_interval_days"])
    return {
        "width_days": None if crossing is None else round(crossing, 2),
        "width_weeks": None if crossing is None else round(crossing / 7.0, 2),
        "width_over_shutdown_interval": (
            None if crossing is None else round(crossing / interval, 3)),
        "note": (
            "beyond this credible-interval width, acting on the RUL costs more than "
            "intervening at the next shutdown regardless -- the estimate is not merely "
            "uninformative, it is harmful, which is why abstention is the correct output"
            if crossing is not None else
            "the swept range never made the policy worse than the calendar baseline"
        ),
    }


def precision_table(
    curve: Dict[str, object], *, captures: Sequence[float] = (0.5, 0.8, 0.9, 0.95)
) -> List[Dict[str, object]]:
    """:func:`decision_relevant_precision` at several capture thresholds.

    One capture threshold is a knob, and a single number derived from a knob
    invites the reader to assume it was the only choice available. The table
    makes the sensitivity to that choice visible, which is cheaper than
    defending 0.95.
    """
    return [decision_relevant_precision(curve, capture=c) | {"capture": c} for c in captures]


def precision_step_value(
    study: VoIStudy, width_from_weeks: float, width_to_weeks: float
) -> Dict[str, object]:
    """Euro value of narrowing the interval from one width to another.

    The question an accuracy-improvement project has to answer. Also reports how
    often the two precisions choose *different* actions, which is the number to
    put in front of an engineer: if the action is identical in 97 % of cases,
    the remaining 3 % has to carry the whole business case.
    """
    s_from = log_sd_for_width(study.true_life_median_days, width_from_weeks * 7.0)
    s_to = log_sd_for_width(study.true_life_median_days, width_to_weeks * 7.0)
    opps = study.opportunities()
    T = study.true_lives()
    cache = {"baselines": study.baselines(),
             "perfect_actions": study._perfect_actions(opps, T)}
    p_from = study.point(s_from, _cache=cache)
    p_to = study.point(s_to, _cache=cache)

    # Action agreement between the two precisions, on the same draws.
    rng_a = np.random.default_rng(study.seed + 1000)
    rng_b = np.random.default_rng(study.seed + 1000)
    plan = study._plan(opps)
    same = 0
    for t in T:
        m_a = t * math.exp(rng_a.normal(study.bias_log, s_from))
        post_a = m_a * np.exp(rng_a.normal(0.0, s_from, study.n_posterior))
        m_b = t * math.exp(rng_b.normal(study.bias_log, s_to))
        post_b = m_b * np.exp(rng_b.normal(0.0, s_to, study.n_posterior))
        a = study._best_action_under_posterior(post_a, opps, plan)
        b = study._best_action_under_posterior(post_b, opps, plan)
        same += int(a.name == b.name)
    # Rounded once, at the end: deriving the difference from already-rounded
    # components puts a cent of arithmetic noise into a reported value.
    value = p_from.expected_cost_eur - p_to.expected_cost_eur
    evpi = cache["baselines"]["evpi_eur"]
    return {
        "width_from_weeks": width_from_weeks,
        "width_to_weeks": width_to_weeks,
        "expected_cost_from_eur": round(p_from.expected_cost_eur, 2),
        "expected_cost_to_eur": round(p_to.expected_cost_eur, 2),
        "value_eur_per_decision": round(value, 2),
        "evpi_eur": round(evpi, 2),
        "share_of_evpi": None if evpi <= 0 else round(value / evpi, 4),
        "p_same_action": round(same / T.size, 4),
    }
