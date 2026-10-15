"""The maintenance decision: which opportunity, and is the choice even close.

The decision problem
--------------------
At a decision epoch an asset has a weighted sample of failure times -- from
:meth:`pdmcore.health.scenario_rul.ScenarioRUL.failure_day_samples` -- and a
finite set of **opportunities**:

* ``run_to_failure``   -- do nothing; pay for the failure when it comes.
* ``unscheduled_stop`` -- stop the asset now (or on a chosen day) outside a
  shutdown. Production is lost at the unplanned rate, but the repair is orderly.
* ``planned_shutdown`` -- wait for a shutdown window the plant is taking anyway.

Each opportunity has an expected cost under the posterior. The decision is the
argmin. Three things about that are worth stating because they are what make the
answer different from "intervene when the RUL median drops below the lead time":

**The cost function is asymmetric, so the median is the wrong statistic.** Being
a week late costs the unplanned-event premium; being a week early costs a week
of a part's life. At this repository's default cost model those differ by a
factor of 30, so the decision is driven by the *left tail* of the RUL posterior,
not its centre. That is why the whole posterior is passed in and summarised
afterwards, rather than summarised and then decided on.

**The opportunity set is discrete.** Shutdowns are every N weeks. An RUL that
moves from 8 weeks to 6 weeks changes nothing if both land before the same
shutdown. :attr:`Decision.margin_eur` reports how far the chosen action is from
the runner-up, and it is the quantity the value-of-information analysis
integrates over: when the margin is large, precision is worthless.

**The horizon is one failure, not a renewal process.** The expected cost of
intervening at day *d* includes the residual life discarded
(:meth:`pdmcore.decide.costs.MaintenanceCostModel.residual_life_eur`), which is
the first-order correction for "intervening early brings the next failure
forward". It is not a full renewal-theoretic treatment, and on an asset whose
replacement interval is short relative to the planning horizon that difference
matters; stated here rather than buried.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .costs import MaintenanceCostModel
from .counts import CountedRate

__all__ = [
    "Opportunity",
    "ActionEvaluation",
    "Decision",
    "RUN_TO_FAILURE",
    "shutdown_calendar",
    "evaluate_opportunities",
    "decide",
    "programme_balance",
]


@dataclass(frozen=True)
class Opportunity:
    """A moment at which the asset could be worked on.

    ``day`` is days from the decision epoch; ``math.inf`` means "never", i.e.
    run to failure. ``downtime_h`` overrides the cost model's MTTR when a
    specific job is longer or shorter than the asset class default.
    ``line`` groups opportunities that share one production stoppage, which is
    what :mod:`pdmcore.decide.schedule` exploits.
    """

    name: str
    day: float
    kind: str = "planned_shutdown"
    downtime_h: Optional[float] = None
    line: Optional[str] = None
    #: Crew-hours the window can absorb. ``None`` means unconstrained, which is
    #: only true for a single-asset decision.
    capacity_h: Optional[float] = None

    _KINDS = ("planned_shutdown", "unscheduled_stop", "run_to_failure")

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS}, got {self.kind!r}")
        if self.day < 0:
            raise ValueError("day must be >= 0")
        if self.kind == "run_to_failure" and math.isfinite(self.day):
            raise ValueError("run_to_failure must have day = inf")

    def intervention_eur(self, cost: MaintenanceCostModel) -> float:
        if self.kind == "planned_shutdown":
            return cost.planned_event_eur(self.downtime_h)
        if self.kind == "unscheduled_stop":
            return cost.unscheduled_stop_eur(self.downtime_h)
        return float("inf")

    def job_hours(self, cost: MaintenanceCostModel) -> float:
        return cost.mttr_planned_h if self.downtime_h is None else float(self.downtime_h)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "day": None if math.isinf(self.day) else round(float(self.day), 2),
            "kind": self.kind,
            "downtime_h": self.downtime_h,
            "line": self.line,
        }


#: The do-nothing action, always in the opportunity set. A decision model that
#: cannot choose "run it to failure" will recommend an intervention on every
#: asset it is shown, which is the behaviour that gets a programme switched off.
RUN_TO_FAILURE = Opportunity(name="run_to_failure", day=math.inf, kind="run_to_failure")


def shutdown_calendar(
    *,
    interval_days: float,
    horizon_days: float,
    first_day: Optional[float] = None,
    line: Optional[str] = None,
    capacity_h: Optional[float] = None,
    downtime_h: Optional[float] = None,
) -> List[Opportunity]:
    """Planned shutdowns every ``interval_days`` out to ``horizon_days``.

    ``first_day`` defaults to ``interval_days``, i.e. the decision epoch is
    immediately after a shutdown -- the pessimistic phase, and the one worth
    defaulting to, because the optimistic phase (a shutdown next week) makes
    every policy look good.
    """
    if interval_days <= 0:
        raise ValueError("interval_days must be > 0")
    start = interval_days if first_day is None else float(first_day)
    out: List[Opportunity] = []
    d = start
    k = 1
    while d <= horizon_days + 1e-9:
        out.append(Opportunity(name=f"shutdown_{k}", day=float(d), kind="planned_shutdown",
                               line=line, capacity_h=capacity_h, downtime_h=downtime_h))
        d += interval_days
        k += 1
    return out


@dataclass
class ActionEvaluation:
    """Expected cost of one opportunity under the failure-time posterior."""

    opportunity: Opportunity
    expected_eur: float
    p_failure_first: float
    expected_residual_days: float
    failure_term_eur: float
    intervention_term_eur: float
    residual_term_eur: float

    def as_dict(self) -> Dict[str, object]:
        return {
            **self.opportunity.as_dict(),
            "expected_eur": round(float(self.expected_eur), 2),
            "p_failure_first": round(float(self.p_failure_first), 4),
            "expected_residual_days": round(float(self.expected_residual_days), 2),
            "failure_term_eur": round(float(self.failure_term_eur), 2),
            "intervention_term_eur": round(float(self.intervention_term_eur), 2),
            "residual_term_eur": round(float(self.residual_term_eur), 2),
        }


@dataclass
class Decision:
    """The chosen action, the alternatives, and how close the call was."""

    status: str                      # "ok" | "abstained"
    chosen: Optional[ActionEvaluation] = None
    runner_up: Optional[ActionEvaluation] = None
    evaluations: List[ActionEvaluation] = field(default_factory=list)
    cost_fingerprint: str = ""
    note: str = ""

    @property
    def margin_eur(self) -> Optional[float]:
        """Expected-cost gap to the runner-up.

        The decision-relevant quantity. A margin of 50 EUR on a 7 000 EUR
        decision means the action is a coin toss and no amount of extra
        measurement precision will make it a better one; a margin of 9 000 EUR
        means the action is robust and no amount of extra precision is needed
        either. Precision only pays in between.
        """
        if self.chosen is None or self.runner_up is None:
            return None
        return float(self.runner_up.expected_eur - self.chosen.expected_eur)

    @property
    def margin_frac(self) -> Optional[float]:
        m = self.margin_eur
        if m is None or self.chosen is None or self.chosen.expected_eur <= 0:
            return None
        return float(m / self.chosen.expected_eur)

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "chosen": None if self.chosen is None else self.chosen.as_dict(),
            "runner_up": None if self.runner_up is None else self.runner_up.as_dict(),
            "margin_eur": None if self.margin_eur is None else round(self.margin_eur, 2),
            "margin_frac": None if self.margin_frac is None else round(self.margin_frac, 4),
            "evaluations": [e.as_dict() for e in self.evaluations],
            "cost_fingerprint": self.cost_fingerprint,
            "note": self.note,
        }


def _normalise(days: Sequence[float], weights: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(days, dtype=np.float64).ravel()
    if d.size == 0:
        raise ValueError("need at least one failure-time sample")
    if weights is None:
        w = np.full(d.size, 1.0 / d.size)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != d.size:
            raise ValueError("days and weights must have the same length")
        total = float(np.sum(w))
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        w = w / total
    return d, w


def evaluate_opportunities(
    opportunities: Sequence[Opportunity],
    days: Sequence[float],
    weights: Optional[Sequence[float]],
    cost: MaintenanceCostModel,
    *,
    horizon_days: Optional[float] = None,
) -> List[ActionEvaluation]:
    """Expected cost of every opportunity under one failure-time posterior.

    ``horizon_days`` bounds the planning cycle. Without it, run-to-failure costs
    a full unplanned event however far away the failure is -- correct for an
    asset that has been detected as degrading and will fail within the planning
    period, and wrong for one with fourteen years left, which it would have the
    model replace today. With it, run-to-failure costs
    ``P(T <= H) * unplanned + P(T > H) * planned``: the same horizon correction
    :meth:`pdmcore.decide.schedule.FleetScheduler.defer_cost` applies, and for
    the same reason -- a cost pushed past the horizon must not become free.

    This is a single-cycle model either way: no discounting, one failure. On an
    asset whose replacement interval is short relative to the planning horizon
    the renewal structure matters and this does not capture it.
    """
    d, w = _normalise(days, weights)
    c_fail = cost.unplanned_event_eur()
    out: List[ActionEvaluation] = []
    for opp in opportunities:
        if opp.kind == "run_to_failure":
            if horizon_days is None:
                expected, p_fail = c_fail, 1.0
            else:
                p_fail = float(np.sum(w[d <= float(horizon_days)]))
                expected = p_fail * c_fail + (1.0 - p_fail) * cost.planned_event_eur()
            out.append(ActionEvaluation(
                opportunity=opp, expected_eur=expected, p_failure_first=p_fail,
                expected_residual_days=0.0, failure_term_eur=p_fail * c_fail,
                intervention_term_eur=expected - p_fail * c_fail, residual_term_eur=0.0,
            ))
            continue
        before = d <= opp.day
        p_fail = float(np.sum(w[before]))
        p_survive = 1.0 - p_fail
        if p_survive <= 1e-12:
            residual_days = 0.0
        else:
            residual_days = float(np.sum(w[~before] * (d[~before] - opp.day)) / p_survive)
        c_int = opp.intervention_eur(cost)
        residual_eur = cost.residual_life_eur(residual_days)
        failure_term = p_fail * c_fail
        intervention_term = p_survive * c_int
        residual_term = p_survive * residual_eur
        out.append(ActionEvaluation(
            opportunity=opp,
            expected_eur=failure_term + intervention_term + residual_term,
            p_failure_first=p_fail,
            expected_residual_days=residual_days,
            failure_term_eur=failure_term,
            intervention_term_eur=intervention_term,
            residual_term_eur=residual_term,
        ))
    return out


def decide(
    opportunities: Sequence[Opportunity],
    days: Optional[Sequence[float]],
    weights: Optional[Sequence[float]],
    cost: MaintenanceCostModel,
    *,
    horizon_days: Optional[float] = None,
    require_configured: bool = True,
    abstain_note: str = "",
) -> Decision:
    """Choose the minimum-expected-cost opportunity, or abstain.

    Abstains -- rather than defaulting to any action -- when ``days`` is
    ``None``, which is what
    :meth:`ScenarioRUL.failure_day_samples <pdmcore.health.scenario_rul.ScenarioRUL.failure_day_samples>`
    returns whenever the RUL estimator refuses. The abstention is propagated
    rather than converted into "run to failure": those are different statements,
    and a planner shown "run to failure" believes a decision was made.
    """
    if require_configured:
        cost.require_configured()
    fp = cost.fingerprint()
    if days is None:
        return Decision(
            status="abstained", cost_fingerprint=fp,
            note=abstain_note or "no failure-time posterior available; RUL abstained",
        )
    evals = evaluate_opportunities(opportunities, days, weights, cost,
                                   horizon_days=horizon_days)
    if not evals:
        return Decision(status="abstained", cost_fingerprint=fp,
                        note="empty opportunity set")
    order = sorted(evals, key=lambda e: e.expected_eur)
    return Decision(
        status="ok",
        chosen=order[0],
        runner_up=order[1] if len(order) > 1 else None,
        evaluations=evals,
        cost_fingerprint=fp,
    )


# ---------------------------------------------------------------------------
# does the programme pay for itself
# ---------------------------------------------------------------------------
def programme_balance(
    cost: MaintenanceCostModel,
    *,
    failures_prevented: CountedRate,
    failures_remaining: CountedRate,
    false_alarms: CountedRate,
    early_interventions_days_lost: float = 0.0,
) -> Dict[str, object]:
    """Annual euro balance of running the monitoring programme, per asset.

    Everything is driven by :class:`~pdmcore.decide.counts.CountedRate` rather
    than by bare rates, so the output carries the event counts the conclusion
    rests on. That is not decoration: at a realistic 0.2 failures per asset-year
    a one-site, one-year trial sees a handful of events, and the Poisson
    interval on the benefit spans a factor of three. A payback figure quoted
    without its count is a number whose error bar was dropped on the way to the
    slide.

    ``early_interventions_days_lost`` is the total residual life discarded per
    asset-year by intervening before failure -- the cost of the programme
    *working*, which a naive balance omits entirely.
    """
    avoidable = cost.avoidable_eur()
    fa_eur = cost.false_alarm_eur()
    prog = cost.programme_cost()

    benefit, benefit_lo, benefit_hi = failures_prevented.scaled(avoidable)
    fa_cost, fa_lo, fa_hi = false_alarms.scaled(fa_eur)
    residual = cost.residual_life_eur(early_interventions_days_lost) if \
        early_interventions_days_lost else 0.0
    # residual_life_eur clamps at design life; for a per-year total that can
    # exceed one part's life, price it linearly instead.
    if early_interventions_days_lost > cost.design_life_days:
        residual = cost.spare_part_eur * early_interventions_days_lost / cost.design_life_days

    net = benefit - fa_cost - prog.total_eur - residual
    net_lo = benefit_lo - fa_hi - prog.total_eur - residual
    net_hi = benefit_hi - fa_lo - prog.total_eur - residual
    total_cost = fa_cost + prog.total_eur + residual
    return {
        "avoidable_eur_per_failure": round(avoidable, 2),
        "false_alarm_eur_each": round(fa_eur, 2),
        "programme_cost": prog.as_dict(),
        "failures_prevented": failures_prevented.as_dict(),
        "failures_remaining": failures_remaining.as_dict(),
        "false_alarms": false_alarms.as_dict(),
        "benefit_eur_per_asset_year": round(benefit, 2),
        "benefit_ci_eur": [round(benefit_lo, 2), round(benefit_hi, 2)],
        "false_alarm_cost_eur_per_asset_year": round(fa_cost, 2),
        "false_alarm_cost_ci_eur": [round(fa_lo, 2), round(fa_hi, 2)],
        "residual_life_cost_eur_per_asset_year": round(residual, 2),
        "total_cost_eur_per_asset_year": round(total_cost, 2),
        "net_eur_per_asset_year": round(net, 2),
        "net_ci_eur": [round(net_lo, 2), round(net_hi, 2)],
        "benefit_cost_ratio": (None if total_cost <= 0 else round(benefit / total_cost, 3)),
        "pays_for_itself": bool(net > 0.0),
        # The honest verdict: positive point estimate is not the same as a
        # demonstrated positive, and at these counts the difference is usual.
        "pays_for_itself_demonstrated": bool(net_lo > 0.0),
        "verdict_note": (
            "net_lo > 0: the benefit exceeds the cost across the whole Poisson interval "
            "on the event counts"
            if net_lo > 0.0 else
            "the point estimate and the lower Poisson bound disagree; the event count is "
            "too small to demonstrate the sign of the balance, whatever the point estimate says"
        ),
    }
