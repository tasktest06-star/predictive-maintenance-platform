"""Value of information: the cross-check, the invariants, and the break-even.

The first test is the important one. :meth:`VoIStudy.expected_costs_fast` is a
second implementation of the expected-cost arithmetic in
:mod:`pdmcore.decide.policy`, written because the sweep evaluates it millions of
times. Two implementations of one formula is exactly the defect class this
repository keeps finding, so the agreement is asserted on randomised inputs
rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdmcore.decide.costs import MaintenanceCostModel
from pdmcore.decide.policy import RUN_TO_FAILURE, Opportunity, evaluate_opportunities
from pdmcore.decide.voi import (
    VoIStudy,
    break_even_precision,
    decision_relevant_precision,
    interval_width_days,
    log_sd_for_width,
    precision_step_value,
    precision_table,
    realised_cost,
    voi_curve,
)


def configured(**kw) -> MaintenanceCostModel:
    base = MaintenanceCostModel(site="test_site", **kw)
    return MaintenanceCostModel(site="test_site",
                                acknowledged_defaults=base.unverified_defaults(), **kw)


def study(**kw) -> VoIStudy:
    args = dict(cost=configured(), n_draws=500, n_posterior=200)
    args.update(kw)
    return VoIStudy(**args)


# --- the cross-check --------------------------------------------------------
def test_fast_path_agrees_with_the_policy_implementation():
    st = study()
    opps = st.opportunities()
    plan = st._plan(opps)
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(5, 400))
        posterior = rng.lognormal(math.log(rng.uniform(5, 300)), rng.uniform(0.05, 1.2), n)
        fast = st.expected_costs_fast(posterior, plan)
        slow = np.array([e.expected_eur for e in
                         evaluate_opportunities(opps, posterior, None, st.cost)])
        assert fast == pytest.approx(slow, rel=1e-9, abs=1e-9)


def test_fast_path_handles_the_degenerate_posteriors():
    st = study()
    opps = st.opportunities()
    plan = st._plan(opps)
    for posterior in (np.full(5, 1e-6), np.full(5, 1e6), np.array([10.0])):
        fast = st.expected_costs_fast(posterior, plan)
        slow = np.array([e.expected_eur for e in
                         evaluate_opportunities(opps, posterior, None, st.cost)])
        assert fast == pytest.approx(slow, rel=1e-9, abs=1e-9)
        assert np.all(np.isfinite(fast))


# --- precision bookkeeping --------------------------------------------------
def test_width_and_log_sd_round_trip():
    for median in (30.0, 90.0, 400.0):
        for width in (1.0, 14.0, 90.0):
            s = log_sd_for_width(median, width)
            assert interval_width_days(median, s) == pytest.approx(width, rel=1e-9)
    assert log_sd_for_width(90.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        log_sd_for_width(0.0, 10.0)


def test_realised_cost_is_the_hindsight_bill():
    c = configured()
    opp = Opportunity("w", 40.0, "planned_shutdown")
    assert realised_cost(opp, 39.0, c) == pytest.approx(c.unplanned_event_eur())
    assert realised_cost(opp, 40.0, c) == pytest.approx(c.unplanned_event_eur())
    assert realised_cost(opp, 40.0001, c) == pytest.approx(c.planned_event_eur(), rel=1e-4)
    assert realised_cost(RUN_TO_FAILURE, 1e9, c) == pytest.approx(c.unplanned_event_eur())
    late = realised_cost(opp, 400.0, c)
    assert late > realised_cost(opp, 50.0, c)


# --- the reference points ---------------------------------------------------
def test_evpi_is_non_negative_and_bounded_by_the_unplanned_event():
    b = study().baselines()
    assert b["evpi_eur"] >= 0.0
    assert b["perfect_information_eur"] <= b["best_baseline_eur"]
    assert b["evpi_eur"] < b["run_to_failure_eur"]


def test_perfect_information_captures_all_of_evpi_and_matches_every_action():
    p = study().point(0.0)
    assert p.information_efficiency == pytest.approx(1.0, abs=1e-9)
    assert p.p_action_matches_perfect == pytest.approx(1.0)
    assert p.mean_interval_days == 0.0


def test_detection_is_worth_far_more_than_precision():
    """The headline. Conditional on detection, perfect RUL buys little."""
    b = study().baselines()
    value_of_detection = b["run_to_failure_eur"] - b["blanket_next_shutdown_eur"]
    assert value_of_detection > 5.0 * b["evpi_eur"]


# --- the curve --------------------------------------------------------------
def test_expected_cost_rises_with_imprecision():
    st = study()
    fine = st.point(log_sd_for_width(st.true_life_median_days, 7.0))
    coarse = st.point(log_sd_for_width(st.true_life_median_days, 120.0))
    assert coarse.expected_cost_eur > fine.expected_cost_eur
    assert coarse.p_action_matches_perfect < fine.p_action_matches_perfect


def test_a_bad_rul_is_worse_than_no_rul():
    """The break-even width: past it, acting on the estimate loses money."""
    st = study()
    cur = voi_curve(st, widths_weeks=(0.5, 1, 2, 4, 8, 16, 32, 64))
    be = break_even_precision(cur)
    assert be["width_days"] is not None
    assert be["width_over_shutdown_interval"] > 0.0
    # And the efficiency really does go negative beyond it.
    worst = max(cur["_points"], key=lambda p: p.mean_interval_days)
    assert worst.information_efficiency < 0.0


def test_decision_relevant_precision_is_reported_with_its_capture_level():
    st = study()
    cur = voi_curve(st, widths_weeks=(0.5, 1, 2, 4, 8, 16))
    table = precision_table(cur, captures=(0.5, 0.8, 0.95))
    assert [row["capture"] for row in table] == [0.5, 0.8, 0.95]
    widths = [row["width_days"] for row in table if row["width_days"] is not None]
    # Tighter capture cannot permit a wider interval.
    assert widths == sorted(widths, reverse=True)
    loose = decision_relevant_precision(cur, capture=0.5)
    if loose["width_days"] is not None:
        assert loose["width_over_shutdown_interval"] > 0


def test_decision_relevant_precision_scales_with_the_shutdown_interval():
    """The result worth reading: the answer is a property of the calendar."""
    widths = {}
    for interval in (14.0, 42.0, 84.0):
        st = study(shutdown_interval_days=interval, horizon_days=600.0)
        cur = voi_curve(st, widths_weeks=(0.5, 1, 2, 4, 8, 16, 32))
        row = decision_relevant_precision(cur, capture=0.5)
        widths[interval] = row["width_days"]
    present = [v for v in widths.values() if v is not None]
    assert len(present) >= 2
    ordered = [widths[k] for k in sorted(widths) if widths[k] is not None]
    assert ordered == sorted(ordered)


def test_almost_all_of_evpi_is_the_option_to_act_outside_the_calendar():
    """A result that contradicted the expectation when it was first measured.

    The guess was that removing the ability to stop an asset between shutdowns
    would *raise* the value of precision, because the only remaining lever is
    which window. It is the other way round, by a factor of tens: without that
    option the clairvoyant can only shave residual life off a job that was
    happening anyway, which is worth a few hundred euros of bearing, while with
    it the clairvoyant can rescue the cases that fail before the first window.

    So the value of prognostic precision lives almost entirely in the freedom to
    act off-calendar -- a property of the plant's operating constraints, not of
    the measurement. On a continuous process that cannot stop between
    shutdowns, perfect RUL knowledge is worth close to nothing.
    """
    free = study(allow_unscheduled_stop=True).baselines()["evpi_eur"]
    constrained = study(allow_unscheduled_stop=False).baselines()["evpi_eur"]
    assert constrained < free / 5.0
    # And what is left is bounded by the part cost, because residual life is all
    # the clairvoyant can still save.
    assert constrained < configured().spare_part_eur


def test_precision_step_value_reports_how_often_the_action_changes():
    st = study(n_draws=300)
    step = precision_step_value(st, 8.0, 4.0)
    assert 0.0 <= step["p_same_action"] <= 1.0
    # Tolerance of two cents: the reported difference is rounded from the
    # unrounded costs, so it need not equal the difference of the rounded ones.
    assert step["value_eur_per_decision"] == pytest.approx(
        step["expected_cost_from_eur"] - step["expected_cost_to_eur"], abs=0.02)
    assert step["expected_cost_from_eur"] > step["expected_cost_to_eur"]


def test_a_biased_estimator_costs_more_than_an_unbiased_one():
    unbiased = study(bias_log=0.0).point(0.3)
    optimistic = study(bias_log=0.45).point(0.3)
    assert optimistic.expected_cost_eur > unbiased.expected_cost_eur
    assert optimistic.p_failed_in_service > unbiased.p_failed_in_service


def test_curve_carries_its_cost_fingerprint():
    st = study()
    cur = voi_curve(st, widths_weeks=(1, 4))
    assert cur["config"]["cost_fingerprint"] == st.cost.fingerprint()
    assert len(cur["points"]) == 3  # perfect information plus the two widths
