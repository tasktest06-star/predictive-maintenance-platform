#!/usr/bin/env python3
"""The fleet scheduler against its baselines, on posteriors from the real filter.

Each asset in the instance is a life from the pool, stopped at a random point in
its degradation stage, with its failure-time posterior taken from the
scenario-conditional estimator fed the real pipeline's indicator series. Assets
whose estimator abstains stay in the instance and are **not scheduled** -- which
is the point of including them.

Reported: the objective against the three baselines, the measured gap between
the heuristic and the MILP optimum across instance sizes, and sweeps over the
two constraints a plant actually argues about (crew-hours per window and annual
parts budget).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core")))

from pdmcore.decide.costs import SITE_PROFILES, MaintenanceCostModel  # noqa: E402
from pdmcore.decide.schedule import AssetJob, FleetScheduler, ShutdownWindow  # noqa: E402
from pdmcore.health.rul import ParticleFilterRUL  # noqa: E402
from pdmcore.health.scenario_rul import LoadScenario, ScenarioRUL  # noqa: E402
from pipeline import RESULTS, RUL_FAILURE_THRESHOLD, RUL_MIN_HEALTH, load_pool  # noqa: E402

ATTACH_STATES = ("watch", "alarm", "critical")


def site_model(name: str = "mid_size_line") -> MaintenanceCostModel:
    overrides = SITE_PROFILES[name]
    probe = MaintenanceCostModel(site=name, **overrides)
    return MaintenanceCostModel(site=name, acknowledged_defaults=probe.unverified_defaults(),
                                note=f"SITE_PROFILES[{name!r}]", **overrides)


def job_from_life(life, asset_id: str, line: str, *, stop_at: float,
                  job_hours: float, crew: int) -> AssetJob:
    """Feed one life up to ``stop_at`` and turn its posterior into a job."""
    sr = ScenarioRUL(rul=ParticleFilterRUL(failure_threshold=RUL_FAILURE_THRESHOLD,
                                           min_health=RUL_MIN_HEALTH, seed=3))
    attached = False
    for t, ind, state in zip(life.t_days, life.indicator, life.alarm_state):
        if t > stop_at:
            break
        if not attached:
            if state not in ATTACH_STATES:
                continue
            attached = True
        sr.update(float(t), float(ind), load_frac=life.duty)
    scenario = LoadScenario.constant(life.duty, name="as_operated")
    samples = sr.failure_day_samples(scenario) if attached else None
    if samples is None:
        est = sr.estimate(scenario) if attached else None
        return AssetJob(asset_id=asset_id, line=line, job_hours=job_hours,
                        crew_required=crew,
                        abstain_reason=(f"RUL status={est.status}: {est.note[:110]}"
                                        if est is not None else
                                        "asset never left the normal band"))
    days, weights = samples
    return AssetJob(asset_id=asset_id, line=line, job_hours=job_hours, crew_required=crew,
                    failure_days=days, weights=weights)


def build_instance(lives, n: int, *, n_lines: int = 4, seed: int = 0,
                   job_hours: float = 4.0, crew: int = 2) -> List[AssetJob]:
    rng = np.random.default_rng(seed)
    jobs: List[AssetJob] = []
    for i in range(n):
        life = lives[int(rng.integers(0, len(lives)))]
        # A random moment in the degradation stage: an instance where every
        # asset is at the same point in its life is not an instance a planner
        # ever sees.
        stop_at = float(rng.uniform(0.25, 0.95)) * float(life.degradation_days)
        jobs.append(job_from_life(life, f"A{i:03d}", f"L{i % n_lines}",
                                  stop_at=stop_at, job_hours=job_hours, crew=crew))
    return jobs


def summarise(out: Dict[str, object]) -> Dict[str, object]:
    chosen = out["chosen"]
    b = chosen["baselines"]
    return {
        "solver": chosen["solver"],
        "objective_eur": chosen["objective_eur"],
        "n_scheduled": chosen["n_scheduled"],
        "n_deferred": chosen["n_deferred"],
        "n_no_recommendation": len(chosen["no_recommendation"]),
        "baselines_eur": {k: v["objective_eur"] for k, v in b.items()},
        "savings_eur": chosen["savings_vs_baselines_eur"],
        "blanket_over_capacity_windows": b["blanket_next_window"].get(
            "windows_over_capacity", []),
        "greedy_gap_vs_milp": out["greedy_gap_vs_milp"],
        "components_eur": chosen["components"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--assets", type=int, default=40)
    args = ap.parse_args(argv)

    _meta, lives = load_pool(args.pool)
    cost = site_model()
    t0 = time.time()

    def windows(crew_hours: float, n: int = 6, interval: float = 42.0):
        return [ShutdownWindow(f"W{k}", day=interval * (k + 1), crew_hours=crew_hours)
                for k in range(n)]

    reference_jobs = build_instance(lives, args.assets, seed=0)
    n_abstain = sum(1 for j in reference_jobs if j.abstained)
    ref = FleetScheduler(cost, windows(48.0), crew_size=4, parts_budget_eur=40000.0,
                         horizon_days=300.0).solve(reference_jobs)
    print(f"reference instance: {args.assets} assets, {n_abstain} abstaining")
    print(f"  objective {ref['chosen']['objective_eur']:.0f} EUR  "
          f"({ref['chosen']['solver']}), greedy gap {ref['greedy_gap_vs_milp']}")
    for k, v in ref["chosen"]["baselines"].items():
        print(f"  baseline {k:24s} {v['objective_eur']:12.0f} EUR  "
              f"saving {ref['chosen']['savings_vs_baselines_eur'][k]:12.0f}")

    size_study = []
    for n in (8, 16, 24, 40, 60):
        jobs = build_instance(lives, n, seed=7)
        t1 = time.time()
        res = FleetScheduler(cost, windows(48.0), crew_size=4,
                             parts_budget_eur=40000.0, horizon_days=300.0).solve(jobs)
        size_study.append({
            "n_assets": n,
            "n_abstaining": sum(1 for j in jobs if j.abstained),
            "greedy_gap_vs_milp": res["greedy_gap_vs_milp"],
            "milp_objective_eur": None if res["milp"] is None else res["milp"]["objective_eur"],
            "greedy_objective_eur": res["greedy"]["objective_eur"],
            "solve_wall_clock_s": round(time.time() - t1, 3),
        })
        print(f"  n={n:3d}  gap {res['greedy_gap_vs_milp']}  "
              f"{size_study[-1]['solve_wall_clock_s']:.2f} s")

    crew_study = []
    for crew_hours in (8.0, 16.0, 32.0, 48.0, 96.0, 400.0):
        res = FleetScheduler(cost, windows(crew_hours), crew_size=4,
                             parts_budget_eur=40000.0,
                             horizon_days=300.0).solve(reference_jobs)
        crew_study.append({"crew_hours_per_window": crew_hours, **summarise(res)})
        print(f"  crew {crew_hours:6.0f} h  objective {crew_study[-1]['objective_eur']:12.0f}  "
              f"scheduled {crew_study[-1]['n_scheduled']:3d}  "
              f"deferred {crew_study[-1]['n_deferred']:3d}")

    budget_study = []
    for budget in (4000.0, 8000.0, 16000.0, 40000.0, None):
        res = FleetScheduler(cost, windows(400.0), crew_size=4, parts_budget_eur=budget,
                             horizon_days=300.0).solve(reference_jobs)
        budget_study.append({"parts_budget_eur": budget, **summarise(res)})
        print(f"  budget {str(budget):>8s}  objective {budget_study[-1]['objective_eur']:12.0f}  "
              f"scheduled {budget_study[-1]['n_scheduled']:3d}")

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t0, 1),
        "cost_model": cost.audit_record(),
        "reference": {
            "n_assets": args.assets,
            "n_abstaining": n_abstain,
            "crew_size": 4,
            "crew_hours_per_window": 48.0,
            "parts_budget_eur": 40000.0,
            "windows": ref["windows"],
            **summarise(ref),
            "no_recommendation": ref["chosen"]["no_recommendation"],
            "per_window": ref["chosen"]["per_window"],
        },
        "instance_size_study": size_study,
        "crew_capacity_study": crew_study,
        "parts_budget_study": budget_study,
        "notes": [
            "Posteriors come from the scenario-conditional estimator fed the real "
            "pipeline's indicator series, so the abstentions in this instance are the "
            "estimator's own, not injected.",
            "blanket_next_window is costed even when it violates crew capacity, and "
            "flagged when it does: a baseline that cannot be executed is not a "
            "competitor, and quoting a saving against it would be flattering.",
            "The heuristic-versus-MILP gap is measured on each instance size rather "
            "than asserted. scipy's milp is already a core dependency, so the solver "
            "costs nothing to use; the heuristic exists for a gateway without scipy and "
            "its measured gap is what justifies keeping it as the fallback.",
        ],
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "scheduler.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
