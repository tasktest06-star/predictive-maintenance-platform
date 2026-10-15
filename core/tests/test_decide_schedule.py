"""Fleet scheduling: the refusal, the capacities, and the grouping incentive."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdmcore.decide.costs import MaintenanceCostModel
from pdmcore.decide.schedule import AssetJob, FleetScheduler, ShutdownWindow


def configured(**kw) -> MaintenanceCostModel:
    base = MaintenanceCostModel(site="test_site", **kw)
    return MaintenanceCostModel(site="test_site",
                                acknowledged_defaults=base.unverified_defaults(), **kw)


def job(aid: str, line: str, median_days: float, *, hours: float = 4.0,
        crew: int = 2, spread: float = 0.35, n: int = 300) -> AssetJob:
    rng = np.random.default_rng(abs(hash(aid)) % 2**31)
    days = np.clip(rng.lognormal(math.log(median_days), spread, n), 1.0, 2000.0)
    return AssetJob(asset_id=aid, line=line, job_hours=hours, crew_required=crew,
                    failure_days=days)


def windows(n: int = 4, interval: float = 42.0, crew_hours: float = 200.0, line=None):
    return [ShutdownWindow(f"W{k}", day=interval * (k + 1), line=line,
                           crew_hours=crew_hours) for k in range(n)]


# --- the refusal ------------------------------------------------------------
def test_an_abstaining_asset_is_never_scheduled():
    jobs = [job("A", "L1", 60.0),
            AssetJob("B", "L1", job_hours=4.0,
                     abstain_reason="RUL status=insufficient_evidence")]
    out = FleetScheduler(configured(), windows()).solve(jobs)
    chosen = out["chosen"]
    assert chosen["assignments"]["B"] is None
    assert chosen["no_recommendation"]["B"].startswith("RUL status")
    assert "A" not in chosen["no_recommendation"]


def test_a_fleet_that_entirely_abstains_produces_no_plan_rather_than_an_empty_one():
    jobs = [AssetJob(f"A{i}", "L1", job_hours=4.0, abstain_reason="no trend")
            for i in range(3)]
    out = FleetScheduler(configured(), windows()).solve(jobs)
    assert set(out["chosen"]["assignments"].values()) == {None}
    assert len(out["chosen"]["no_recommendation"]) == 3


def test_refuses_an_unconfigured_cost_model():
    from pdmcore.decide.costs import CostModelNotConfigured

    with pytest.raises(CostModelNotConfigured):
        FleetScheduler(MaintenanceCostModel(), windows())


def test_rejects_duplicate_asset_ids():
    s = FleetScheduler(configured(), windows())
    with pytest.raises(ValueError, match="unique"):
        s.solve([job("A", "L1", 60.0), job("A", "L2", 60.0)])


def test_needs_at_least_one_window():
    with pytest.raises(ValueError):
        FleetScheduler(configured(), [])


# --- capacities -------------------------------------------------------------
def test_crew_capacity_is_respected_by_both_solvers():
    jobs = [job(f"A{i:02d}", "L1", 40.0 + 3 * i, hours=6.0) for i in range(10)]
    s = FleetScheduler(configured(), windows(n=2, crew_hours=24.0), crew_size=4)
    out = s.solve(jobs)
    for key in ("greedy", "milp"):
        res = out[key]
        if res is None:
            continue
        for _name, w in res["per_window"].items():
            assert not w["over_capacity"]
            assert w["crew_hours_used"] <= 24.0 + 1e-6


def test_parts_budget_is_respected():
    c = configured()
    jobs = [job(f"A{i:02d}", "L1", 30.0) for i in range(8)]
    budget = 3.5 * c.spare_part_eur
    out = FleetScheduler(c, windows(n=3), parts_budget_eur=budget).solve(jobs)
    for key in ("greedy", "milp"):
        res = out[key]
        if res is None:
            continue
        n = sum(1 for v in res["assignments"].values() if v is not None)
        assert n * c.spare_part_eur <= budget + 1e-6


def test_a_line_restricted_window_only_takes_its_own_line():
    wins = [ShutdownWindow("L1_only", day=30.0, line="L1", crew_hours=200.0)]
    jobs = [job("A", "L1", 35.0), job("B", "L2", 35.0)]
    out = FleetScheduler(configured(), wins, horizon_days=400.0).solve(jobs)
    assert out["chosen"]["assignments"]["B"] is None


# --- the grouping incentive -------------------------------------------------
def test_sharing_a_stoppage_costs_one_makespan_not_two():
    s = FleetScheduler(configured(), windows(crew_hours=200.0), crew_size=4)
    a, b = job("A", "L1", 50.0), job("B", "L1", 50.0)
    assert s._makespan([a]) == pytest.approx(4.0)
    assert s._makespan([a, b]) == pytest.approx(4.0)  # two 2-crew jobs on 4 fitters
    # A third job no longer fits in parallel and extends the stoppage.
    c3 = job("C", "L1", 50.0)
    assert s._makespan([a, b, c3]) > 4.0


def test_grouping_beats_spreading_when_the_lives_allow_it():
    """Two assets on one line, both due about the same time: one stoppage."""
    c = configured()
    s = FleetScheduler(c, windows(n=3, crew_hours=200.0), crew_size=4, horizon_days=300.0)
    jobs = [job("A", "L1", 50.0, spread=0.12), job("B", "L1", 52.0, spread=0.12)]
    out = s.solve(jobs)
    grouped = {"A": "W0", "B": "W0"}
    spread_plan = {"A": "W0", "B": "W1"}
    g_obj, _gc, _gp = s.evaluate_plan(jobs, grouped)
    sp_obj, _sc, _sp = s.evaluate_plan(jobs, spread_plan)
    assert g_obj < sp_obj
    assert out["chosen"]["assignments"] == grouped


# --- solvers ----------------------------------------------------------------
def test_milp_is_at_least_as_good_as_the_heuristic():
    jobs = [job(f"A{i:02d}", f"L{i % 3}", 30.0 + 7 * i) for i in range(14)]
    out = FleetScheduler(configured(), windows(n=4, crew_hours=40.0),
                         crew_size=4, parts_budget_eur=7000.0).solve(jobs)
    assert out["milp"] is not None
    assert out["milp"]["objective_eur"] <= out["greedy"]["objective_eur"] + 1e-6
    assert out["greedy_gap_vs_milp"] >= -1e-9
    assert out["chosen"]["solver"] == "milp"


def test_the_heuristic_is_close_enough_to_defend_running_without_a_solver():
    """Measured, not assumed: the gap on an instance of the size we target."""
    jobs = [job(f"A{i:02d}", f"L{i % 4}", 25.0 + 5 * i) for i in range(24)]
    out = FleetScheduler(configured(), windows(n=5, crew_hours=48.0),
                         crew_size=4, parts_budget_eur=12000.0).solve(jobs)
    assert out["greedy_gap_vs_milp"] < 0.05


# --- baselines --------------------------------------------------------------
def test_baselines_are_present_and_the_plan_beats_the_executable_ones():
    jobs = [job(f"A{i:02d}", f"L{i % 2}", 45.0 + 6 * i) for i in range(10)]
    out = FleetScheduler(configured(), windows(n=4, crew_hours=200.0),
                         crew_size=4, horizon_days=260.0).solve(jobs)
    chosen = out["chosen"]
    b = chosen["baselines"]
    assert set(b) == {"run_to_failure", "blanket_next_window", "fix_everything_always"}
    assert not b["blanket_next_window"]["windows_over_capacity"]
    for name in b:
        assert chosen["objective_eur"] <= b[name]["objective_eur"] + 1e-6
        assert chosen["savings_vs_baselines_eur"][name] >= -1e-6


def test_blanket_next_window_is_flagged_when_it_cannot_actually_be_executed():
    """A baseline that violates crew capacity is not a competitor, and says so."""
    jobs = [job(f"A{i:02d}", "L1", 40.0, hours=8.0) for i in range(10)]
    out = FleetScheduler(configured(), windows(n=3, crew_hours=32.0), crew_size=4).solve(jobs)
    b = out["chosen"]["baselines"]["blanket_next_window"]
    assert b["windows_over_capacity"]
    assert "not executable" in b["note"]


def test_fix_everything_always_is_the_most_expensive_policy():
    jobs = [job(f"A{i:02d}", "L1", 300.0) for i in range(6)]
    out = FleetScheduler(configured(), windows(n=4, crew_hours=200.0),
                         crew_size=4, horizon_days=400.0).solve(jobs)
    b = out["chosen"]["baselines"]
    assert (b["fix_everything_always"]["objective_eur"]
            > b["blanket_next_window"]["objective_eur"])


# --- the horizon correction -------------------------------------------------
def test_deferring_past_the_horizon_still_costs_a_planned_job():
    c = configured()
    s = FleetScheduler(c, windows(n=2), horizon_days=100.0)
    safe = AssetJob("A", "L1", job_hours=4.0, failure_days=np.full(200, 5000.0))
    assert s.defer_cost(safe) == pytest.approx(c.planned_event_eur())
    doomed = AssetJob("B", "L1", job_hours=4.0, failure_days=np.full(200, 5.0))
    assert s.defer_cost(doomed) == pytest.approx(c.unplanned_event_eur())


def test_a_long_lived_asset_is_deferred():
    jobs = [job("A", "L1", 4000.0, spread=0.1)]
    out = FleetScheduler(configured(), windows(n=4), horizon_days=300.0).solve(jobs)
    assert out["chosen"]["assignments"]["A"] is None


def test_assignment_cost_excludes_line_downtime_so_it_is_not_double_charged():
    c = configured()
    s = FleetScheduler(c, windows(n=1, crew_hours=200.0), crew_size=4)
    j = AssetJob("A", "L1", job_hours=4.0, failure_days=np.full(200, 5000.0))
    w = s.windows[0]
    direct = s.assignment_cost(j, w)
    total, comp, _pw = s.evaluate_plan([j], {"A": w.name})
    assert comp["asset_expected_eur"] == pytest.approx(direct)
    assert comp["line_downtime_eur"] == pytest.approx(
        c.planned_downtime_eur_per_h * 4.0)
    assert total == pytest.approx(comp["asset_expected_eur"] + comp["line_downtime_eur"])
    # The per-asset planned cost would have charged downtime too; the scheduler
    # must not, or grouping would be invisible.
    assert direct < c.planned_event_eur()
