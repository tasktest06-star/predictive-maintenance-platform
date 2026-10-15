"""The per-asset decision: the action set, the asymmetry, and the abstention."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdmcore.decide.costs import MaintenanceCostModel
from pdmcore.decide.policy import (
    RUN_TO_FAILURE,
    Opportunity,
    decide,
    evaluate_opportunities,
    shutdown_calendar,
)


def configured(**kw) -> MaintenanceCostModel:
    base = MaintenanceCostModel(site="test_site", **kw)
    return MaintenanceCostModel(site="test_site",
                                acknowledged_defaults=base.unverified_defaults(), **kw)


def opportunities(interval=42.0, horizon=400.0, stop_now=True):
    opps = shutdown_calendar(interval_days=interval, horizon_days=horizon)
    if stop_now:
        opps = [Opportunity("stop_now", 0.0, "unscheduled_stop")] + opps
    return opps + [RUN_TO_FAILURE]


def point_mass(day: float, n: int = 400) -> np.ndarray:
    return np.full(n, float(day))


# --- the action set ---------------------------------------------------------
def test_opportunity_validation():
    with pytest.raises(ValueError, match="kind"):
        Opportunity("x", 10.0, "whenever")
    with pytest.raises(ValueError, match="day"):
        Opportunity("x", -1.0)
    with pytest.raises(ValueError, match="run_to_failure"):
        Opportunity("x", 10.0, "run_to_failure")
    assert RUN_TO_FAILURE.day == math.inf


def test_shutdown_calendar_spacing_and_pessimistic_phase():
    opps = shutdown_calendar(interval_days=42.0, horizon_days=180.0)
    assert [o.day for o in opps] == [42.0, 84.0, 126.0, 168.0]
    assert shutdown_calendar(interval_days=42.0, horizon_days=180.0,
                             first_day=7.0)[0].day == 7.0
    with pytest.raises(ValueError):
        shutdown_calendar(interval_days=0.0, horizon_days=10.0)


def test_run_to_failure_costs_exactly_the_unplanned_event():
    c = configured()
    (ev,) = evaluate_opportunities([RUN_TO_FAILURE], point_mass(50.0), None, c)
    assert ev.expected_eur == pytest.approx(c.unplanned_event_eur())
    assert ev.p_failure_first == 1.0


# --- the decision itself ----------------------------------------------------
def test_an_imminent_failure_is_stopped_now():
    c = configured()
    d = decide(opportunities(), point_mass(3.0), None, c)
    assert d.status == "ok"
    assert d.chosen.opportunity.name == "stop_now"
    assert d.margin_eur > 0


def test_a_distant_failure_waits_for_the_last_shutdown_before_it():
    c = configured()
    d = decide(opportunities(), point_mass(200.0), None, c)
    assert d.chosen.opportunity.kind == "planned_shutdown"
    # 42-day windows: the last one before day 200 is day 168.
    assert d.chosen.opportunity.day == 168.0
    assert d.chosen.p_failure_first == 0.0


def test_the_choice_is_driven_by_the_left_tail_not_the_median():
    """A later median with a heavier left tail must bring the action forward.

    If the decision were taken on the median, the skewed posterior -- whose
    median is *later* -- would choose a later window. It chooses an earlier one,
    because a quarter of its mass fails before the window the tight posterior
    picks and that quarter costs 5x what a wasted month of bearing life does.
    """
    c = configured()
    tight = np.full(600, 90.0)
    skewed = np.r_[np.full(150, 60.0), np.full(450, 95.0)]
    assert float(np.median(skewed)) > float(np.median(tight))
    a = decide(opportunities(stop_now=False), tight, None, c)
    b = decide(opportunities(stop_now=False), skewed, None, c)
    assert a.chosen.opportunity.day == 84.0
    assert b.chosen.opportunity.day == 42.0


def test_without_an_unscheduled_stop_option_the_answer_changes():
    """The common real constraint: a continuous process cannot stop on a Tuesday."""
    c = configured()
    free = decide(opportunities(stop_now=True), point_mass(10.0), None, c)
    constrained = decide(opportunities(stop_now=False), point_mass(10.0), None, c)
    assert free.chosen.opportunity.name == "stop_now"
    assert constrained.chosen.opportunity.kind in ("planned_shutdown", "run_to_failure")
    assert constrained.chosen.expected_eur > free.chosen.expected_eur


def test_residual_life_makes_early_intervention_cost_something():
    c = configured()
    evals = {e.opportunity.name: e for e in evaluate_opportunities(
        opportunities(), point_mass(300.0), None, c)}
    early = evals["shutdown_1"]
    late = evals["shutdown_7"]
    assert early.residual_term_eur > late.residual_term_eur
    assert early.expected_eur > late.expected_eur


def test_margin_is_the_gap_to_the_runner_up():
    c = configured()
    d = decide(opportunities(), point_mass(90.0), None, c)
    costs = sorted(e.expected_eur for e in d.evaluations)
    assert d.margin_eur == pytest.approx(costs[1] - costs[0])
    assert d.margin_frac == pytest.approx(d.margin_eur / costs[0])


# --- abstention and refusal -------------------------------------------------
def test_abstains_rather_than_defaulting_to_run_to_failure():
    c = configured()
    d = decide(opportunities(), None, None, c)
    assert d.status == "abstained"
    assert d.chosen is None
    assert d.margin_eur is None
    assert "abstained" in d.note
    assert d.cost_fingerprint == c.fingerprint()


def test_refuses_against_an_unconfigured_cost_model():
    from pdmcore.decide.costs import CostModelNotConfigured

    with pytest.raises(CostModelNotConfigured):
        decide(opportunities(), point_mass(10.0), None, MaintenanceCostModel())


def test_weights_are_honoured_and_validated():
    c = configured()
    days = np.array([5.0, 500.0])
    near = decide(opportunities(), days, np.array([0.99, 0.01]), c)
    far = decide(opportunities(), days, np.array([0.01, 0.99]), c)
    assert near.chosen.opportunity.name == "stop_now"
    assert far.chosen.opportunity.kind == "planned_shutdown"
    with pytest.raises(ValueError):
        evaluate_opportunities(opportunities(), days, np.array([1.0]), c)
    with pytest.raises(ValueError):
        evaluate_opportunities(opportunities(), days, np.array([0.0, 0.0]), c)
    with pytest.raises(ValueError):
        evaluate_opportunities(opportunities(), np.array([]), None, c)


def test_a_long_lived_asset_is_left_alone_once_the_horizon_is_bounded():
    """The decision model must be able to say 'do nothing', or it is a salesman.

    Also the test for the horizon correction: with no horizon, a failure
    fourteen years away costs a full unplanned event *today*, and the model
    replaces a healthy bearing. Bounding the planning cycle fixes that, and the
    difference between the two answers is the size of the modelling choice.
    """
    from pdmcore.decide.costs import SITE_PROFILES

    base = MaintenanceCostModel(site="spared", **SITE_PROFILES["spared_duty_standby"])
    c = MaintenanceCostModel(site="spared", acknowledged_defaults=base.unverified_defaults(),
                             **SITE_PROFILES["spared_duty_standby"])
    far = point_mass(5000.0)
    unbounded = decide(opportunities(), far, None, c)
    bounded = decide(opportunities(), far, None, c, horizon_days=400.0)
    assert unbounded.chosen.opportunity.kind == "planned_shutdown"
    assert bounded.chosen.opportunity.name == "run_to_failure"
    assert bounded.chosen.p_failure_first == 0.0
    # Deferring past the horizon is not free: it still costs a planned job.
    assert bounded.chosen.expected_eur == pytest.approx(c.planned_event_eur())
