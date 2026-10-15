"""Scenario-conditional RUL: the invariants, and every abstention path.

The invariant that matters most is the first one: conditioning on the *observed*
duty must reproduce the unconditional answer exactly. A scenario model that
quietly shifts the baseline number would make every conditional answer wrong in
a way no reviewer could see, because the shift would be inside the plausible
range of the thing it is predicting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdmcore.health import ParticleFilterRUL
from pdmcore.health.scenario_rul import (
    BALL_BEARING_EXPONENT,
    LoadScenario,
    ScenarioRUL,
    doubling_time_at_load,
)


def _degrading(
    n: int = 45, b: float = 0.075, load: float = 0.85, noise: float = 0.05,
    *, seed: int = 0, exponent_sd: float = 0.6, **kwargs
) -> ScenarioRUL:
    """A filter fed a clean exponential trend that clears every gate."""
    sr = ScenarioRUL(
        rul=ParticleFilterRUL(failure_threshold=4.0, min_health=1.0, seed=3),
        load_exponent_sd=exponent_sd,
        **kwargs,
    )
    rng = np.random.default_rng(seed)
    for i in range(n):
        t = float(i)
        h = 0.4 + 0.05 * (math.exp(b * t) - 1.0) + rng.normal(0.0, noise)
        sr.update(t, h, load_frac=load)
    return sr


# --- the central invariant --------------------------------------------------
def test_conditioning_on_the_observed_duty_reproduces_the_unconditional_answer():
    sr = _degrading(load=0.85)
    uncond = sr.rul.estimate()
    assert uncond.status == "ok"
    cond = sr.estimate(LoadScenario.constant(0.85))
    assert cond.status == "ok"
    for a, b in ((cond.median_days, uncond.median_days),
                 (cond.p10_days, uncond.p10_days),
                 (cond.p90_days, uncond.p90_days)):
        assert a == pytest.approx(b, rel=1e-9)
    # And none of the interval width is attributable to the load model, because
    # the load model is the identity here.
    assert cond.load_model_width_share == pytest.approx(0.0, abs=1e-12)


def test_load_model_contributes_width_only_away_from_the_observed_duty():
    sr = _degrading(load=0.85)
    at = sr.estimate(LoadScenario.constant(0.85))
    away = sr.estimate(LoadScenario.constant(1.0))
    assert at.load_model_width_share == pytest.approx(0.0, abs=1e-12)
    assert away.load_model_width_share > 0.05


def test_higher_load_shortens_and_lower_load_lengthens_the_life():
    sr = _degrading(load=0.85)
    hi = sr.estimate(LoadScenario.constant(1.0)).median_days
    mid = sr.estimate(LoadScenario.constant(0.85)).median_days
    lo = sr.estimate(LoadScenario.constant(0.70)).median_days
    assert hi < mid < lo


def test_rate_multiplier_follows_the_load_life_exponent():
    """With the exponent pinned, the answer scales exactly as (u_ref/u)^p."""
    sr = _degrading(load=0.85, exponent_sd=0.0)
    mid = sr.estimate(LoadScenario.constant(0.85)).median_days
    hi = sr.estimate(LoadScenario.constant(1.0)).median_days
    expected = mid / (1.0 / 0.85) ** BALL_BEARING_EXPONENT
    assert hi == pytest.approx(expected, rel=0.02)


def test_duty_profile_lands_between_its_two_constant_scenarios():
    sr = _degrading(load=0.85, exponent_sd=0.0)
    hi = sr.estimate(LoadScenario.constant(1.0))
    lo = sr.estimate(LoadScenario.constant(0.70))
    # Back off for less than the backed-off life, then return to full load, so
    # the profile genuinely spans both segments.
    first = 0.5 * hi.median_days
    profile = LoadScenario(name="ramp", segments=((first, 0.70), (1e6, 1.0)))
    mixed = sr.estimate(profile)
    assert mixed.median_days > first  # the second segment is reached
    assert hi.median_days < mixed.median_days < lo.median_days


def test_estimate_is_idempotent():
    """A decision service that answers differently twice is unauditable."""
    sr = _degrading()
    s = LoadScenario.constant(1.0)
    a, b = sr.estimate(s), sr.estimate(s)
    assert a.median_days == b.median_days
    assert a.p10_days == b.p10_days and a.p90_days == b.p90_days


# --- abstention -------------------------------------------------------------
def test_abstains_when_the_unconditional_estimate_abstains_and_says_why():
    sr = ScenarioRUL(rul=ParticleFilterRUL(failure_threshold=4.0, min_health=1.0))
    for i in range(5):
        sr.update(float(i), 0.3, load_frac=1.0)
    est = sr.estimate(LoadScenario.constant(1.0))
    assert est.status == "insufficient_evidence"
    assert est.median_days is None
    assert est.note == sr.rul.estimate().note


def test_abstains_on_a_flat_trend_even_with_plenty_of_observations():
    sr = ScenarioRUL(rul=ParticleFilterRUL(failure_threshold=4.0, min_health=1.0))
    rng = np.random.default_rng(1)
    for i in range(60):
        sr.update(float(i), 1.4 + rng.normal(0.0, 0.08), load_frac=1.0)
    est = sr.estimate(LoadScenario.constant(1.0))
    assert est.status == "insufficient_evidence"
    assert "trend" in est.note


def test_conditional_needs_more_observations_than_unconditional():
    """The extra-evidence gate, and that it can actually fire."""
    sr = ScenarioRUL(
        rul=ParticleFilterRUL(failure_threshold=4.0, min_health=1.0, min_observations=12, seed=3),
        extra_observations=6,
    )
    rng = np.random.default_rng(0)
    for i in range(14):
        sr.update(float(i), 0.9 + 0.06 * (math.exp(0.18 * i) - 1.0) + rng.normal(0, 0.03),
                  load_frac=1.0)
    assert sr.rul.estimate().status == "ok"
    est = sr.estimate(LoadScenario.constant(1.0))
    assert est.status == "insufficient_evidence"
    assert ">= 18 observations" in est.note


def test_refuses_when_the_load_history_was_not_recorded():
    sr = _degrading(load=0.85)
    sr.update(45.0, 2.4, load_frac=None)
    est = sr.estimate(LoadScenario.constant(0.85))
    assert est.status == "load_history_unknown"
    assert est.median_days is None
    assert est.reference_load is None
    # The unconditional answer is still available: this gate is specific to the
    # conditional question, not a regression in the base estimator.
    assert sr.rul.estimate().status == "ok"


def test_refuses_a_scenario_too_far_from_the_observed_duty():
    sr = _degrading(load=0.50)
    est = sr.estimate(LoadScenario.constant(1.0))
    assert est.status == "scenario_not_identifiable"
    assert "2.00x" in est.note
    # ...and quantifies the damage-rate change it declined to extrapolate.
    assert "8.0x damage-rate change" in est.note


def test_extrapolation_gate_uses_the_peak_not_the_mean_load():
    """A duty profile whose mean is safe but whose peak is not must refuse."""
    sr = _degrading(load=0.85)
    profile = LoadScenario(name="spiky", segments=((5.0, 1.6), (1e6, 0.6)))
    assert profile.mean_load == pytest.approx(1.6)  # only finite segments count
    assert sr.estimate(profile).status == "scenario_not_identifiable"


# --- the planner's actual question -----------------------------------------
def test_probability_of_failure_before_is_monotone_and_abstains_cleanly():
    sr = _degrading(load=0.85)
    s = LoadScenario.constant(0.85)
    probs = [sr.probability_of_failure_before(s, d) for d in (1, 3, 5, 10, 30)]
    assert all(b >= a for a, b in zip(probs, probs[1:]))
    assert probs[-1] == pytest.approx(1.0, abs=1e-6)

    empty = ScenarioRUL(rul=ParticleFilterRUL(failure_threshold=4.0))
    for i in range(4):
        empty.update(float(i), 0.2, load_frac=1.0)
    assert empty.probability_of_failure_before(s, 42.0) is None
    assert empty.failure_day_samples(s) is None


def test_failure_day_samples_weights_normalise_and_match_the_summary():
    sr = _degrading(load=0.85)
    s = LoadScenario.constant(1.0)
    days, w = sr.failure_day_samples(s)
    assert np.sum(w) == pytest.approx(1.0)
    est = sr.estimate(s)
    # Weighted median of the samples is the reported median.
    order = np.argsort(days)
    cw = np.cumsum(w[order])
    median = float(days[order][int(np.searchsorted(cw, 0.5))])
    assert median == pytest.approx(est.median_days, rel=1e-12)


# --- scenario plumbing ------------------------------------------------------
def test_scenario_validation():
    with pytest.raises(ValueError):
        LoadScenario(name="bad", segments=())
    with pytest.raises(ValueError, match="duration"):
        LoadScenario(name="bad", segments=((0.0, 1.0),))
    with pytest.raises(ValueError, match="stopped asset"):
        LoadScenario(name="bad", segments=((10.0, 0.0),))


def test_update_rejects_a_nonpositive_load():
    sr = ScenarioRUL(rul=ParticleFilterRUL())
    with pytest.raises(ValueError):
        sr.update(0.0, 1.0, load_frac=-0.2)


def test_doubling_time_at_load_matches_hand_arithmetic():
    # 30 days at 70 % becomes 30 / (1/0.7)^3 = 10.29 days at 100 %.
    assert doubling_time_at_load(30.0, 1.0 / 0.7) == pytest.approx(10.29, rel=1e-3)
    assert doubling_time_at_load(30.0, 1.0) == pytest.approx(30.0)
    with pytest.raises(ValueError):
        doubling_time_at_load(30.0, 0.0)


def test_reference_load_uses_the_damage_weighted_mean():
    """mean(u^p)^(1/p), which is above the arithmetic mean for a varying duty."""
    sr = ScenarioRUL(rul=ParticleFilterRUL(min_observations=4, trend_window=4))
    for i, u in enumerate((0.5, 1.0, 0.5, 1.0)):
        sr.update(float(i), 1.0, load_frac=u)
    arithmetic = 0.75
    expected = float(np.mean(np.array([0.5, 1.0, 0.5, 1.0]) ** 3.0) ** (1 / 3.0))
    assert sr.reference_load() == pytest.approx(expected)
    assert sr.reference_load() > arithmetic
    assert sr.load_history_cv() == pytest.approx(0.25 / 0.75, rel=1e-6)


def test_high_load_variability_is_reported_in_the_note():
    sr = ScenarioRUL(
        rul=ParticleFilterRUL(failure_threshold=4.0, min_health=1.0, seed=3),
        extra_observations=0,
    )
    rng = np.random.default_rng(2)
    for i in range(45):
        h = 0.4 + 0.05 * (math.exp(0.075 * i) - 1.0) + rng.normal(0.0, 0.05)
        sr.update(float(i), h, load_frac=0.55 if i % 2 else 1.0)
    est = sr.estimate(LoadScenario.constant(0.85))
    assert est.status == "ok"
    assert est.load_history_cv > 0.25
    assert "varied with CV" in est.note


def test_from_threshold_splits_constructor_arguments():
    sr = ScenarioRUL.from_threshold(12.0, min_observations=20, load_exponent_sd=0.1)
    assert sr.rul.failure_threshold == 12.0
    assert sr.rul.min_observations == 20
    assert sr.load_exponent_sd == 0.1
