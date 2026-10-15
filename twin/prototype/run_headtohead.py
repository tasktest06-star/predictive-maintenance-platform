#!/usr/bin/env python3
"""The experiment that justifies the branch, or does not.

    PYTHONPATH=core:twin/prototype python3 twin/prototype/run_headtohead.py

Seven arms, one dataset, one matching rule: every threshold is bisected until
the *measured* average run length to false alarm on the calibration window hits
the target, then verified on a third window against a Poisson band on the alarm
count. Detection delay is then read off the degradation trajectories at that
matched operating point.

Protocol: ``twin/research/01_protocol.md``, committed before this ran.
Output:   ``twin/prototype/results/headtohead.json``
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, _HERE)

from pdmcore.eval.metrics import partial_auc  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY  # noqa: E402
from pdmtwin.baseline import BaselineArm  # noqa: E402
from pdmtwin.data import (Dataset, ObsSpec, degradation_runs, extractor_for,  # noqa: E402
                          healthy_pool, time_ordered_windows)
from pdmtwin.detect import first_alarm, naive_alarms_per_day  # noqa: E402
from pdmtwin.residual import ResidualInputs  # noqa: E402
from pdmtwin.twin import HybridTwin, TwinConfig  # noqa: E402

MACHINE = "motor_pump_37kw"
CADENCE_H = 8.0
#: ARL0 targets in frames. 500 frames at an 8 h cadence is 167 days -- about one
#: nuisance alarm per asset per five months, which is the headline.
ARL0_TARGETS = (125.0, 250.0, 500.0)


def host_info() -> Dict[str, object]:
    """What was actually measured, so a number cannot be quoted without it."""
    info: Dict[str, object] = {
        "machine": platform.machine(), "python": platform.python_version(),
        "numpy": np.__version__, "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="ascii") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "bogomips")):
                    k, _, v = line.partition(":")
                    info[k.strip().lower()] = v.strip()
                    break
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------
def build_arms(spec, ex, obs, obs_all) -> Dict[str, object]:
    """The seven arms of the protocol, constructed but not yet fitted."""
    return {
        "A_baseline63": BaselineArm(label="A_baseline63", sequential=False),
        "B_baseline63_cusum": BaselineArm(label="B_baseline63_cusum", sequential=True),
        "C_baseline9_cusum": BaselineArm(label="C_baseline9_cusum", sequential=True,
                                         feature_subset=obs),
        "D_twin_physics": HybridTwin(spec, ex, obs, TwinConfig(
            n_state=2, residual_kind="none", filter_mode="ukf")),
        "E_twin_hybrid": HybridTwin(spec, ex, obs, TwinConfig(
            n_state=2, residual_kind="kernel_ridge", residual_alpha=1.0,
            residual_inputs=ResidualInputs.EXOGENOUS, filter_mode="ukf")),
        "F_twin_point": HybridTwin(spec, ex, obs, TwinConfig(
            n_state=2, residual_kind="kernel_ridge", residual_alpha=1.0,
            residual_inputs=ResidualInputs.EXOGENOUS, filter_mode="point")),
        "G_twin_pf": HybridTwin(spec, ex, obs, TwinConfig(
            n_state=2, residual_kind="kernel_ridge", residual_alpha=1.0,
            residual_inputs=ResidualInputs.EXOGENOUS, filter_mode="pf")),
    }


def q_streams(arm, ds: Dataset) -> List[np.ndarray]:
    """Per-trajectory deviation streams, for any arm."""
    if isinstance(arm, BaselineArm):
        return arm._by_trajectory(ds)  # noqa: SLF001 -- same package, same harness
    return arm._by_trajectory(ds)  # noqa: SLF001


def traj_views(dg: Dataset) -> List[Dataset]:
    """Each trajectory as its own dataset, in time order."""
    out = []
    for t in np.unique(dg.traj):
        idx = np.flatnonzero(dg.traj == t)
        out.append(dg.take(idx[np.argsort(dg.t_days[idx], kind="mergesort")]))
    return out


def delays_from_q(arm, subs: Sequence[Dataset], qs: Sequence[np.ndarray]) -> List[Dict]:
    """First confirmed alarm per trajectory, from precomputed streams."""
    out = []
    for sub, q in zip(subs, qs):
        i = first_alarm(q, arm.test)
        out.append({
            "traj": int(sub.traj[0]), "mode": sub.mode[0],
            "alarm_index": None if i is None else int(i),
            "delay_h": None if i is None else float(sub.t_days[i] * 24.0),
            "delay_days": None if i is None else float(sub.t_days[i]),
            "severity_at_alarm": None if i is None else float(sub.severity[i]),
            "n_frames": int(sub.n),
        })
    return out


def delays_for(arm, dg: Dataset) -> List[Dict]:
    """First confirmed alarm per degradation trajectory (recomputes streams)."""
    out = []
    for sub in traj_views(dg):
        q = q_streams(arm, sub)[0]
        i = first_alarm(q, arm.test)
        out.append({
            "traj": int(t), "mode": sub.mode[0],
            "alarm_index": None if i is None else int(i),
            "delay_h": None if i is None else float(sub.t_days[i] * 24.0),
            "delay_days": None if i is None else float(sub.t_days[i]),
            "severity_at_alarm": None if i is None else float(sub.severity[i]),
            "n_frames": int(sub.n),
        })
    return out


def summarise(delays: Sequence[Dict], modes: Sequence[str]) -> Dict:
    """Median and mean delay per mode, with misses counted not dropped."""
    out: Dict[str, Dict] = {}
    for m in list(modes) + ["all"]:
        rows = [d for d in delays if m == "all" or d["mode"] == m]
        hit = [d for d in rows if d["alarm_index"] is not None]
        out[m] = {
            "n_traj": len(rows), "n_detected": len(hit),
            "median_delay_days": float(np.median([d["delay_days"] for d in hit])) if hit else None,
            "mean_delay_days": float(np.mean([d["delay_days"] for d in hit])) if hit else None,
            "median_severity_at_alarm": float(np.median([d["severity_at_alarm"] for d in hit])) if hit else None,
            "mean_severity_at_alarm": float(np.mean([d["severity_at_alarm"] for d in hit])) if hit else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "headtohead.json"))
    ap.add_argument("--n-units", type=int, default=10)
    ap.add_argument("--n-per-unit", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-points", type=int, default=120)
    ap.add_argument("--rho", type=float, default=0.0,
                    help="AR(1) nuisance autocorrelation; 0.0 is the simulator as shipped")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_start = time.time()
    spec = MACHINE_LIBRARY[MACHINE]
    ex = extractor_for(spec)

    print("building datasets ...", flush=True)
    hp = healthy_pool(n_units=args.n_units, n_per_unit=args.n_per_unit,
                      interval_h=CADENCE_H, nuisance_rho=args.rho)
    dg = degradation_runs(n_seeds=args.n_seeds, n_points=args.n_points, days=40.0,
                          nuisance_rho=args.rho)
    print(f"  healthy {hp.n} frames, degradation {dg.n} frames "
          f"({time.time() - t_start:.0f} s)", flush=True)

    shaft_range = (float(min(hp.shaft_hz.min(), dg.shaft_hz.min())) * 0.99,
                   float(max(hp.shaft_hz.max(), dg.shaft_hz.max())) * 1.01)
    obs = ObsSpec.gated(hp.feature_names, ex.config, shaft_range, healthy=hp.X)
    obs_all = ObsSpec.from_feature_names(hp.feature_names)
    print(f"  observation channels kept {len(obs)}: {obs.names}")
    for name, reason in obs.dropped:
        print(f"  DROPPED {name}: {reason}")

    wf, wc, wv = time_ordered_windows(hp)
    fit, cal, ver = hp.take(wf), hp.take(wc), hp.take(wv)
    modes = sorted(set(dg.mode))

    results: Dict[str, object] = {
        "host": host_info(),
        "config": {
            "machine": MACHINE, "cadence_h": CADENCE_H, "nuisance_rho": args.rho,
            "n_healthy": hp.n, "n_fit": fit.n, "n_cal": cal.n, "n_ver": ver.n,
            "n_degradation": dg.n, "modes": modes,
            "n_traj_per_mode": args.n_seeds, "n_points_per_traj": args.n_points,
            "arl0_targets_frames": list(ARL0_TARGETS),
            "obs_channels": list(obs.names),
            "obs_dropped": [{"channel": c, "reason": r} for c, r in obs.dropped],
        },
        "arms": {},
    }

    # The observation-model probe is per asset, not per arm: sharing it between
    # the twin arms is what the deployment would do, and it keeps the arms
    # comparable down to the calibration of the measurement model.
    shared_model = None
    for key, arm in build_arms(spec, ex, obs, obs_all).items():
        t0 = time.time()
        print(f"\n== {key}", flush=True)
        per_target: Dict[str, object] = {}
        if isinstance(arm, HybridTwin):
            arm.identify(fit, model=shared_model).identify_process(fit).fit_residual(fit)
            shared_model = arm.core.model
            arm.calibrate(fit, cal, target_arl0_frames=ARL0_TARGETS[0])
        else:
            arm.fit(fit, cal, target_arl0_frames=ARL0_TARGETS[0])
        # The deviation streams are properties of the windows, not of the
        # operating point: compute each once and let the targets move only the
        # threshold. Three filter passes saved per arm, and the three operating
        # points are guaranteed to be the same model.
        subs = traj_views(dg)
        q_ver_list = q_streams(arm, ver)
        q_dg_list = [q_streams(arm, s)[0] for s in subs]
        q_ver = np.concatenate(q_ver_list)
        q_dg = np.concatenate(q_dg_list)
        sev = np.concatenate([s.severity for s in subs])
        faulty = sev >= 0.05
        y = np.r_[np.zeros(q_ver.size), np.ones(int(np.count_nonzero(faulty)))]
        s_scores = np.r_[q_ver, q_dg[faulty]]
        for target in ARL0_TARGETS:
            arm.set_target(target)
            v = arm.verify(q_ver=q_ver_list)
            d = delays_from_q(arm, subs, q_dg_list)
            naive = naive_alarms_per_day(q_ver_list, arm.test,
                                         acquisitions_per_day=24.0 / CADENCE_H)
            per_target[f"arl0_{int(target)}"] = {
                "threshold": float(arm.test.threshold),
                "calibration": arm.reports["calibration"],
                "verification": v,
                "chance_alarm_prob_per_traj": float(
                    1.0 - np.exp(-args.n_points / max(1e-9, v["arl0_frames"]))),
                "alarms_per_day": naive,
                "pauc_10_sev_ge_0p05": float(partial_auc(y, s_scores, 0.1)),
                "delays": d,
                "summary": summarise(d, modes),
            }
            print(f"  ARL0 target {target:.0f}: thr {arm.test.threshold:.3f} "
                  f"cal {arm.reports['calibration']['arl0_frames']:.0f} "
                  f"ver {v['arl0_frames']:.0f} ({v['n_alarms']:.0f} alarms, "
                  f"allowed {v['allowed_alarms_3sigma_poisson']:.1f}, pass {v['passed']}) "
                  f"median delay {per_target[f'arl0_{int(target)}']['summary']['all']['median_delay_days']} d "
                  f"sev {per_target[f'arl0_{int(target)}']['summary']['all']['median_severity_at_alarm']}",
                  flush=True)
        results["arms"][key] = {
            "reports": getattr(arm, "reports", {}),
            "config": arm.config.as_dict() if isinstance(arm, HybridTwin) else {
                "kind": arm.kind, "sequential": arm.sequential,
                "n_features": int(len(arm.feature_subset) if arm.feature_subset else len(obs_all)),
            },
            "by_arl0": per_target,
            "wall_s": float(time.time() - t0),
        }

    results["wall_s_total"] = float(time.time() - t_start)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = args.out if not args.tag else args.out.replace(".json", f"_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=False)
    print(f"\nwrote {out} ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
