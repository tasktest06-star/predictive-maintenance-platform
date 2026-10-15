#!/usr/bin/env python3
"""Does the monitoring programme pay for itself, and when does it not?

Runs the multi-year fleet simulation under run-to-failure, the best
fixed-interval preventive policy, and condition-based maintenance with the
twin -- the last of which pays for its own hardware, installation, cloud,
analyst time and false alarms.

Then sweeps the assumptions. The sweep is the headline, not an appendix: every
cost parameter in :mod:`pdmcore.decide.costs` is an estimate, so a single
"saves X EUR per asset-year" figure is a statement about the estimates. What
the sweep produces instead is the *boundary*: the value of each assumption at
which the conclusion flips, which a plant can check against its own numbers
without re-running anything.

Three replicate seeds are run for the reference case, because the fleet sees a
small number of failures and a one-seed difference between policies can be
counting noise. The simulated counts are reported as
:class:`~pdmcore.decide.counts.CountedRate`, so the Poisson interval travels
with every rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core")))

from pdmcore.decide.costs import SITE_PROFILES, MaintenanceCostModel  # noqa: E402
from fleetsim import FleetConfig, compare  # noqa: E402
from pipeline import RESULTS, load_pool  # noqa: E402


def site_model(name: str, **overrides) -> MaintenanceCostModel:
    merged = dict(SITE_PROFILES[name])
    merged.update(overrides)
    probe = MaintenanceCostModel(site=name, **merged)
    return MaintenanceCostModel(
        site=name, acknowledged_defaults=probe.unverified_defaults(),
        note=f"SITE_PROFILES[{name!r}]; unquoted parameters are estimates",
        **merged,
    )


#: Each entry: (label, what it changes, the three levels, which is the reference).
#: Levels are chosen to span what a real plant could plausibly report, not to
#: bracket the default symmetrically -- a symmetric sweep around a guess tells
#: you about the sweep, not about the plant.
SENSITIVITY = [
    ("unplanned_downtime_eur_per_h", "cost", [600.0, 3400.0, 11000.0]),
    ("nuisance_rate_per_asset_year", "config", [0.05, 0.25, 1.0]),
    ("p_opens_machine_on_false_alarm", "cost", [0.05, 0.30, 0.70]),
    ("shutdown_interval_days", "config", [14.0, 42.0, 120.0]),
    ("crew_hours_per_window", "config", [16.0, 48.0, 200.0]),
    ("spare_part_eur", "cost", [250.0, 800.0, 4000.0]),
    ("analyst_h_per_asset_year", "cost", [0.5, 1.5, 6.0]),
    ("secondary_damage_eur", "cost", [1000.0, 6500.0, 20000.0]),
    ("allow_unscheduled_stop", "config", [False, True]),
]


def run_one(cfg: FleetConfig, cost: MaintenanceCostModel, lives) -> Dict[str, object]:
    return compare(cfg, cost, lives)


def headline_row(res: Dict[str, object]) -> Dict[str, object]:
    cbm = res["condition_based_twin"]
    rtf = res["run_to_failure"]
    pm = res["fixed_interval_best"]
    bal = res["programme_balance"]
    return {
        "run_to_failure_eur_per_asset_year": rtf["eur_per_asset_year"],
        "best_fixed_interval": pm["policy"],
        "best_fixed_interval_eur_per_asset_year": pm["eur_per_asset_year"],
        "condition_based_eur_per_asset_year": cbm["eur_per_asset_year"],
        "saving_vs_run_to_failure": res["saving_vs_run_to_failure_eur_per_asset_year"],
        "saving_vs_best_fixed_interval": res["saving_vs_best_fixed_interval_eur_per_asset_year"],
        "pays_for_itself": res["pays_for_itself"],
        "programme_cost_eur_per_asset_year": bal["programme_cost"]["total_eur"],
        "false_alarm_cost_eur_per_asset_year": bal["false_alarm_cost_eur_per_asset_year"],
        "failures_rtf": rtf["counts"]["unplanned_failures"],
        "failures_cbm": cbm["counts"]["unplanned_failures"],
        "false_alarms_cbm": cbm["counts"]["false_alarms"],
        "abstained_decision_epochs": cbm["counts"]["abstained_decisions"],
        "deferred_by_the_scheduler": cbm["counts"]["deferred_by_the_scheduler"],
        "net_ci_eur": bal["net_ci_eur"],
        "payback_demonstrated": bal["pays_for_itself_demonstrated"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--measurements", default=os.path.join(RESULTS, "pipeline_measurements.json"))
    ap.add_argument("--assets", type=int, default=40)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--site", default="mid_size_line")
    ap.add_argument("--quick", action="store_true",
                    help="reference case only, one seed, no sensitivity sweep")
    args = ap.parse_args(argv)

    _meta, lives = load_pool(args.pool)
    nuisance = 0.25
    if os.path.exists(args.measurements):
        with open(args.measurements) as fh:
            m = json.load(fh)
        nuisance = float(m["nuisance"]["rate"]["rate"])
        print(f"measured nuisance alarm rate: {nuisance:.3f}/asset-year")

    base_cfg = FleetConfig(n_assets=args.assets, horizon_years=args.years,
                           nuisance_rate_per_asset_year=nuisance, seed=0)
    base_cost = site_model(args.site)

    t0 = time.time()
    replicates: List[Dict[str, object]] = []
    n_seeds = 1 if args.quick else args.seeds
    for s in range(n_seeds):
        res = run_one(replace(base_cfg, seed=s), base_cost, lives)
        replicates.append(res)
        row = headline_row(res)
        print(f"  seed {s}: RTF {row['run_to_failure_eur_per_asset_year']:9.0f}  "
              f"PM {row['best_fixed_interval_eur_per_asset_year']:9.0f}  "
              f"CBM {row['condition_based_eur_per_asset_year']:9.0f}  "
              f"pays={row['pays_for_itself']}", flush=True)

    def spread(key: str) -> Dict[str, float]:
        vals = [headline_row(r)[key] for r in replicates]
        return {"mean": round(float(np.mean(vals)), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2)}

    reference = {
        "seeds": n_seeds,
        "per_seed": [headline_row(r) for r in replicates],
        "across_seeds": {
            k: spread(k) for k in (
                "run_to_failure_eur_per_asset_year",
                "best_fixed_interval_eur_per_asset_year",
                "condition_based_eur_per_asset_year",
                "saving_vs_run_to_failure",
                "saving_vs_best_fixed_interval",
            )
        },
        "detail_seed0": replicates[0],
    }

    sensitivity: List[Dict[str, object]] = []
    if not args.quick:
        for name, target, levels in SENSITIVITY:
            rows = []
            for level in levels:
                cfg = base_cfg
                cost = base_cost
                if target == "config":
                    cfg = replace(base_cfg, **{name: level})
                else:
                    cost = site_model(args.site, **{name: level})
                res = run_one(cfg, cost, lives)
                row = headline_row(res)
                row["level"] = level
                rows.append(row)
                print(f"  {name:34s} = {str(level):>8s}  "
                      f"CBM {row['condition_based_eur_per_asset_year']:9.0f}  "
                      f"best alt {min(row['run_to_failure_eur_per_asset_year'], row['best_fixed_interval_eur_per_asset_year']):9.0f}  "
                      f"pays={row['pays_for_itself']}", flush=True)
            flips = [r["level"] for r in rows if not r["pays_for_itself"]]
            sensitivity.append({
                "parameter": name,
                "applies_to": target,
                "levels": [r["level"] for r in rows],
                "rows": rows,
                "flips_the_conclusion_at": flips,
                "swing_eur_per_asset_year": round(
                    max(r["saving_vs_best_fixed_interval"] for r in rows)
                    - min(r["saving_vs_best_fixed_interval"] for r in rows), 2),
            })

    # Which assumption matters most: the one whose plausible range moves the
    # saving furthest, reported next to whether it actually flips the sign.
    ranked = sorted(sensitivity, key=lambda s: -s["swing_eur_per_asset_year"])
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t0, 1),
        "site": args.site,
        "nuisance_rate_measured": nuisance,
        "base_config": base_cfg.as_dict(),
        "cost_model": base_cost.audit_record(),
        "reference": reference,
        "sensitivity": sensitivity,
        "sensitivity_ranked_by_swing": [
            {"parameter": s["parameter"],
             "swing_eur_per_asset_year": s["swing_eur_per_asset_year"],
             "flips_at": s["flips_the_conclusion_at"]}
            for s in ranked
        ],
        "site_archetypes": {},
        "caveats": [
            "Health trajectories, alarm states and prognostic indicators come from the "
            "real DSP pipeline (pipeline.py). The healthy stage of each life is a drawn "
            "duration, and the false-alarm stream is a Poisson process at the measured "
            "rate rather than a simulated detector output over years.",
            "The life pool is reused across assets and across policies, so the three "
            "policies are compared on common random numbers -- which is what makes the "
            "differences readable -- but the fleet is not 40 independent machine "
            "histories.",
            "Every cost parameter is an estimate. The sensitivity table, not the "
            "reference case, is the result.",
        ],
    }

    # The three site archetypes, because the answer differs by asset class and a
    # single fleet-wide verdict is the most misleading form of this result.
    for name in SITE_PROFILES:
        res = run_one(base_cfg, site_model(name), lives)
        out["site_archetypes"][name] = headline_row(res)
        print(f"  site {name:24s} CBM {out['site_archetypes'][name]['condition_based_eur_per_asset_year']:9.0f}  "
              f"pays={out['site_archetypes'][name]['pays_for_itself']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "fleet_sim.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path} ({time.time() - t0:.0f} s)")
    if ranked:
        print(f"largest swing: {ranked[0]['parameter']} "
              f"({ranked[0]['swing_eur_per_asset_year']:.0f} EUR/asset-year), "
              f"flips at {ranked[0]['flips_the_conclusion_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
