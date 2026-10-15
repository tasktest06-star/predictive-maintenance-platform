"""The cost model, its refusals, and the small-count discipline.

Two of these tests exist to stop a drift rather than to check a formula:

``test_unplanned_event_matches_the_repo_cost_model`` pins the decomposition to
``pdmcore.eval.metrics.expected_cost``'s own default, so the plant model and the
detector threshold cannot come to disagree silently about what a failure costs.

``test_poisson_bounds_match_published_values`` checks the Garwood interval
against hand-computable chi-square quantiles, because the whole small-count
argument on this branch rests on those bounds being right.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from pdmcore.decide import counts as C
from pdmcore.decide.costs import (
    SITE_PROFILES,
    CostModelNotConfigured,
    MaintenanceCostModel,
)
from pdmcore.decide.counts import CountedRate, SmallCountError
from pdmcore.decide.policy import programme_balance
from pdmcore.eval.metrics import expected_cost, expected_cost_from_model


def configured(**overrides) -> MaintenanceCostModel:
    """A model a site has taken responsibility for, defaults acknowledged."""
    base = MaintenanceCostModel(site="test_site", **overrides)
    return MaintenanceCostModel(site="test_site",
                                acknowledged_defaults=base.unverified_defaults(),
                                **overrides)


# --- the anchor to the existing cost model ---------------------------------
def test_unplanned_event_matches_the_repo_cost_model():
    """The decomposition must reproduce metrics.expected_cost's cost_fn."""
    default_fn = inspect.signature(expected_cost).parameters["cost_fn"].default
    assert default_fn == 25000.0
    derived = MaintenanceCostModel().unplanned_event_eur()
    assert derived == pytest.approx(default_fn, rel=0.10)


def test_inspection_cost_is_the_repo_false_positive_cost():
    default_fp = inspect.signature(expected_cost).parameters["cost_fp"].default
    assert MaintenanceCostModel().inspection_eur == default_fp


def test_event_costs_are_ordered():
    c = MaintenanceCostModel()
    assert c.planned_event_eur() < c.unscheduled_stop_eur() < c.unplanned_event_eur()
    assert 0 < c.avoidable_eur() < c.unplanned_event_eur()


def test_false_alarm_costs_more_than_an_inspection_because_machines_get_opened():
    """The CN0549 finding, in money: a confident wrong answer opens a machine."""
    c = MaintenanceCostModel()
    assert c.false_alarm_eur() > c.inspection_eur
    assert c.false_alarm_eur(p_opens=0.0) == pytest.approx(c.inspection_eur)
    assert c.false_alarm_eur(p_opens=1.0) > c.false_alarm_eur(p_opens=0.3)


def test_residual_life_is_linear_and_clamped():
    c = MaintenanceCostModel()
    assert c.residual_life_eur(0.0) == 0.0
    assert c.residual_life_eur(c.design_life_days) == pytest.approx(c.spare_part_eur)
    assert c.residual_life_eur(10 * c.design_life_days) == pytest.approx(c.spare_part_eur)
    assert c.residual_life_eur(-5.0) == 0.0
    half = c.residual_life_eur(c.design_life_days / 2)
    assert half == pytest.approx(c.spare_part_eur / 2)


def test_programme_cost_excludes_false_alarms_on_purpose():
    c = MaintenanceCostModel()
    p = c.programme_cost()
    assert p.total_eur == pytest.approx(
        p.hardware_eur + p.install_eur + p.gateway_eur + p.cloud_eur + p.analyst_eur)
    # The analyst, not the hardware, is the biggest line item -- which is the
    # finding this breakdown exists to expose.
    assert p.analyst_eur > p.hardware_eur


def test_rejects_a_model_where_unplanned_downtime_is_cheaper_than_planned():
    with pytest.raises(ValueError, match="redundant"):
        MaintenanceCostModel(unplanned_downtime_eur_per_h=100.0,
                             planned_downtime_eur_per_h=900.0)


def test_site_profiles_are_constructible_and_ordered_by_avoidable_cost():
    models = {k: MaintenanceCostModel(**v) for k, v in SITE_PROFILES.items()}
    assert (models["spared_duty_standby"].avoidable_eur()
            < models["mid_size_line"].avoidable_eur()
            < models["critical_single_train"].avoidable_eur())


# --- the configuration gate -------------------------------------------------
def test_refuses_without_a_site():
    with pytest.raises(CostModelNotConfigured, match="no site"):
        MaintenanceCostModel().require_configured()


def test_refuses_when_defaults_were_never_looked_at():
    c = MaintenanceCostModel(site="plant_a")
    with pytest.raises(CostModelNotConfigured, match="library\n?\\s*defaults|library defaults"):
        c.require_configured()
    assert len(c.unverified_defaults()) > 10


def test_site_quoted_parameters_do_not_need_acknowledgement():
    c = MaintenanceCostModel().for_site("plant_a", unplanned_downtime_eur_per_h=9000.0)
    assert "unplanned_downtime_eur_per_h" not in c.unverified_defaults()
    assert c.provenance["unplanned_downtime_eur_per_h"] == "site_quoted"


def test_acknowledging_every_default_satisfies_the_gate():
    configured().require_configured()


def test_a_site_quoted_value_equal_to_the_default_is_still_quoted():
    """Equality with the default must not undo an explicit quote."""
    c = MaintenanceCostModel().for_site(
        "plant_a", spare_part_eur=MaintenanceCostModel().spare_part_eur)
    assert "spare_part_eur" not in c.unverified_defaults()


def test_fingerprint_is_stable_and_sensitive():
    a = configured()
    assert a.fingerprint() == configured().fingerprint()
    assert a.fingerprint() != configured(spare_part_eur=801.0).fingerprint()
    other = MaintenanceCostModel(site="plant_b")
    assert a.fingerprint() != MaintenanceCostModel(
        site="plant_b", acknowledged_defaults=other.unverified_defaults()).fingerprint()


def test_audit_record_names_the_unverified_defaults():
    rec = MaintenanceCostModel(site="plant_a").audit_record()
    assert rec["site"] == "plant_a"
    assert rec["unverified_defaults"]
    assert set(rec["derived"]) >= {"unplanned_event_eur", "avoidable_eur",
                                   "programme_eur_per_asset_year"}


# --- the bridge to the detector threshold ----------------------------------
def test_detector_costs_use_the_avoidable_not_the_event_cost():
    c = configured()
    fp, fn = c.detector_costs()
    assert fp == c.false_alarm_eur()
    assert fn == c.avoidable_eur()
    assert fn < c.unplanned_event_eur()


def test_expected_cost_from_model_reports_both_asymmetries():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400, dtype=int), np.ones(40, dtype=int)]
    s = np.r_[rng.normal(0.0, 1.0, 400), rng.normal(2.2, 1.0, 40)]
    c = configured()
    out = expected_cost_from_model(y, s, c, prevalence=0.02)
    assert out["asymmetry_decision_relevant"] < out["asymmetry_naive"]
    # And both are far below the 62x that the library defaults imply, because
    # the false-alarm cost is not one inspection.
    assert out["asymmetry_naive"] < 25000.0 / 400.0
    assert out["cost_model_fingerprint"] == c.fingerprint()
    assert out["cost_model_site"] == "test_site"


def test_the_corrected_costs_move_the_threshold_towards_fewer_alarms():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(2000, dtype=int), np.ones(80, dtype=int)]
    s = np.r_[rng.normal(0.0, 1.0, 2000), rng.normal(2.0, 1.0, 80)]
    naive = expected_cost(y, s, prevalence=0.02)
    corrected = expected_cost_from_model(y, s, configured(), prevalence=0.02)
    assert corrected["threshold"] >= naive["threshold"]
    assert corrected["fpr"] <= naive["fpr"]


# --- counts -----------------------------------------------------------------
def test_poisson_bounds_match_published_values():
    """Garwood bounds: lo = chi2(a/2, 2k)/2, hi = chi2(1-a/2, 2k+2)/2."""
    lo, hi = C.poisson_rate_ci(0, 1.0)
    assert lo == 0.0
    assert hi == pytest.approx(3.6889, rel=1e-3)
    lo, hi = C.poisson_rate_ci(4, 1.0)
    assert lo == pytest.approx(1.0899, rel=1e-3)
    assert hi == pytest.approx(10.2416, rel=1e-3)
    lo, hi = C.poisson_rate_ci(10, 1.0)
    assert lo == pytest.approx(4.7954, rel=1e-3)
    assert hi == pytest.approx(18.3904, rel=1e-3)


def test_zero_failures_does_not_prove_a_zero_rate():
    r = CountedRate(count=0, exposure=40.0, unit="asset-year")
    assert r.rate == 0.0
    assert r.ci[1] == pytest.approx(3.6889 / 40.0, rel=1e-3)
    assert math.isinf(r.relative_width)
    assert r.as_dict()["relative_width"] is None


def test_counted_rate_requires_an_exposure():
    with pytest.raises(ValueError):
        CountedRate(count=3, exposure=0.0, unit="asset-year")
    with pytest.raises(ValueError):
        CountedRate(count=-1, exposure=1.0, unit="asset-year")


def test_poisson_count_band_admits_the_noise_the_ratio_gate_rejected():
    """The Jetson branch's bug: at lambda = 1.8 a count of 6 is ordinary."""
    lo, hi = C.poisson_count_band(1.8)
    assert lo == 0.0
    assert hi > 6.0
    # And the band still rejects a genuine excursion.
    assert 96 > hi


def test_poisson_count_band_pad_is_what_makes_small_lambda_safe():
    _lo, hi_padded = C.poisson_count_band(1.8, pad=3.0)
    _lo, hi_bare = C.poisson_count_band(1.8, pad=0.0)
    assert hi_bare < 6.0 < hi_padded


def test_require_counts_refuses_and_says_what_the_count_supports():
    C.require_counts(12, minimum=10, what="prevented failures")
    with pytest.raises(SmallCountError, match="58 %"):
        C.require_counts(3, minimum=10, what="prevented failures")
    with pytest.raises(SmallCountError, match="undefined"):
        C.require_counts(0, minimum=10, what="prevented failures")


def test_counts_distinguishable():
    assert not C.counts_distinguishable(2, 10.0, 3, 10.0)
    assert C.counts_distinguishable(1, 10.0, 50, 10.0)


def test_scaled_carries_the_interval_through_the_multiplication():
    r = CountedRate(count=6, exposure=40.0, unit="asset-year")
    point, lo, hi = r.scaled(19855.0)
    assert lo < point < hi
    assert point == pytest.approx(r.rate * 19855.0)


# --- the programme balance --------------------------------------------------
def _balance(prevented: int, alarms: int, exposure: float, **kw):
    return programme_balance(
        configured(**kw),
        failures_prevented=CountedRate(prevented, exposure, "asset-year"),
        failures_remaining=CountedRate(2, exposure, "asset-year"),
        false_alarms=CountedRate(alarms, exposure, "asset-year"),
    )


def test_payback_is_not_demonstrated_at_realistic_trial_sizes():
    """A positive point estimate on six events is not a demonstrated payback."""
    b = _balance(prevented=6, alarms=10, exposure=40.0)
    assert b["pays_for_itself"] is True
    assert b["pays_for_itself_demonstrated"] is False
    assert b["net_ci_eur"][0] < 0 < b["net_eur_per_asset_year"]
    assert "too small" in b["verdict_note"]


def test_payback_becomes_demonstrable_at_a_large_enough_exposure():
    b = _balance(prevented=150, alarms=250, exposure=1000.0)
    assert b["pays_for_itself_demonstrated"] is True
    assert b["net_ci_eur"][0] > 0


def test_benefit_is_priced_on_avoidable_not_on_the_whole_event():
    c = configured()
    b = _balance(prevented=10, alarms=0, exposure=100.0)
    assert b["avoidable_eur_per_failure"] == pytest.approx(c.avoidable_eur())
    assert b["benefit_eur_per_asset_year"] == pytest.approx(0.1 * c.avoidable_eur())


def test_false_alarms_can_outweigh_the_whole_hardware_bill():
    """The result that decides programme economics, asserted rather than claimed."""
    c = configured()
    b = _balance(prevented=4, alarms=10, exposure=40.0)
    assert b["false_alarm_cost_eur_per_asset_year"] > b["programme_cost"]["total_eur"]


def test_a_spared_asset_does_not_pay_for_itself():
    c = MaintenanceCostModel(site="spared", **SITE_PROFILES["spared_duty_standby"])
    c = MaintenanceCostModel(site="spared", acknowledged_defaults=c.unverified_defaults(),
                             **SITE_PROFILES["spared_duty_standby"])
    b = programme_balance(
        c,
        failures_prevented=CountedRate(12, 100.0, "asset-year"),
        failures_remaining=CountedRate(8, 100.0, "asset-year"),
        false_alarms=CountedRate(25, 100.0, "asset-year"),
    )
    assert b["net_eur_per_asset_year"] < 0
    assert b["pays_for_itself"] is False
