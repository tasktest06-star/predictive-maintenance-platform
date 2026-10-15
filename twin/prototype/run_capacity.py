#!/usr/bin/env python3
"""The capacity trap, measured on one axis.

    PYTHONPATH=core:twin/prototype python3 twin/prototype/run_capacity.py

The claim under test: **a residual model with enough capacity absorbs the fault
signature itself, which is exactly the thing you are trying to detect.** Four
sweeps, all scored at a matched false-alarm rate so that "absorbed the fault"
cannot hide as "raised the threshold":

1. ``alpha`` over six decades at fixed inputs -> capacity against a finite
   healthy window. Reported against **measured** effective degrees of freedom,
   ``edf = tr(K(K+alpha I)^-1)``, not against alpha.
2. residual inputs: exogenous / state / measured -> endogenous-input absorption.
3. online refitting on a trailing window -> contaminated-training absorption.
4. state dimension 1/2/3 -> the state-space form of the same trap, including
   the damage state that can explain the fault away.

Plus one control that is not a sweep: ``R`` estimated on the **fit** window
instead of the calibration window, which is what a careless implementation
does and which inflates the realised false-alarm rate.

Output: ``twin/prototype/results/capacity.json``
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
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, _HERE)

from pdmcore.sim import MACHINE_LIBRARY  # noqa: E402
from pdmtwin.data import (Dataset, ObsSpec, degradation_runs, extractor_for,  # noqa: E402
                          healthy_pool, time_ordered_windows)
from pdmtwin.detect import first_alarm  # noqa: E402
from pdmtwin.residual import ResidualInputs  # noqa: E402
from pdmtwin.twin import HybridTwin, TwinConfig  # noqa: E402

from run_headtohead import (CADENCE_H, MACHINE, delays_for, host_info,  # noqa: E402
                            summarise)

ARL0_TARGET = 500.0
ALPHAS = (1e3, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-5)


def one_config(
    spec, ex, obs, fit: Dataset, cal: Dataset, ver: Dataset, dg: Dataset,
    cfg: TwinConfig, *, shared_model=None, R_from_fit: bool = False,
    online: int = 0, label: str = "",
) -> Dict:
    """Fit, calibrate, verify and score one twin configuration."""
    t0 = time.time()
    tw = HybridTwin(spec, ex, obs, cfg)
    tw.identify(fit, model=shared_model).identify_process(fit).fit_residual(fit)
    tw.calibrate(fit, fit if R_from_fit else cal, target_arl0_frames=ARL0_TARGET) \
        if R_from_fit else tw.calibrate(fit, cal, target_arl0_frames=ARL0_TARGET)
    v = tw.verify(ver)
    if online > 0:
        delays = []
        for t in np.unique(dg.traj):
            idx = np.flatnonzero(dg.traj == t)
            idx = idx[np.argsort(dg.t_days[idx], kind="mergesort")]
            sub = dg.take(idx)
            q = tw.run_trajectory_online(sub.obs(obs), sub.shaft_hz, window=online)["q"]
            i = first_alarm(q, tw.test)
            delays.append({
                "traj": int(t), "mode": sub.mode[0],
                "alarm_index": None if i is None else int(i),
                "delay_h": None if i is None else float(sub.t_days[i] * 24.0),
                "delay_days": None if i is None else float(sub.t_days[i]),
                "severity_at_alarm": None if i is None else float(sub.severity[i]),
                "n_frames": int(sub.n),
            })
    else:
        delays = delays_for(tw, dg)
    modes = sorted(set(dg.mode))
    return {
        "label": label,
        "config": cfg.as_dict(),
        "online_window": online,
        "R_from_fit_window": R_from_fit,
        "residual": tw.reports.get("residual", {}),
        "process": tw.reports.get("process", {}),
        "physics_rms_log_residual": tw.reports["physics"]["rms_log_residual"],
        "threshold": float(tw.test.threshold),
        "calibration": tw.reports["calibration"],
        "verification": v,
        "summary": summarise(delays, modes),
        "delays": delays,
        "wall_s": float(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "capacity.json"))
    ap.add_argument("--n-units", type=int, default=10)
    ap.add_argument("--n-per-unit", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-points", type=int, default=120)
    args = ap.parse_args()

    t_start = time.time()
    spec = MACHINE_LIBRARY[MACHINE]
    ex = extractor_for(spec)
    hp = healthy_pool(n_units=args.n_units, n_per_unit=args.n_per_unit, interval_h=CADENCE_H)
    dg = degradation_runs(n_seeds=args.n_seeds, n_points=args.n_points, days=40.0)
    shaft_range = (float(min(hp.shaft_hz.min(), dg.shaft_hz.min())) * 0.99,
                   float(max(hp.shaft_hz.max(), dg.shaft_hz.max())) * 1.01)
    obs = ObsSpec.gated(hp.feature_names, ex.config, shaft_range, healthy=hp.X)
    wf, wc, wv = time_ordered_windows(hp)
    fit, cal, ver = hp.take(wf), hp.take(wc), hp.take(wv)
    print(f"healthy {hp.n} degr {dg.n} channels {len(obs)} "
          f"({time.time() - t_start:.0f} s)", flush=True)

    runs: List[Dict] = []
    shared = None

    def go(label, cfg, **kw):
        nonlocal shared
        r = one_config(spec, ex, obs, fit, cal, ver, dg, cfg, shared_model=shared,
                       label=label, **kw)
        if shared is None:
            shared = HybridTwin(spec, ex, obs, cfg).identify(fit).core.model
        runs.append(r)
        s = r["summary"]["all"]
        res = r["residual"]
        print(f"{label:38s} edf {res.get('edf', 0.0):8.2f} "
              f"edf/frame {res.get('edf_per_frame', 0.0):7.4f} "
              f"thr {r['threshold']:7.3f} verARL {r['verification']['arl0_frames']:8.1f} "
              f"det {s['n_detected']}/{s['n_traj']} "
              f"delay {s['median_delay_days']} d sev {s['median_severity_at_alarm']}",
              flush=True)

    # 0. The control: physics only, no learned residual.
    go("control_physics_only", TwinConfig(n_state=2, residual_kind="none"))

    # 1. Capacity against a finite healthy window, honest (exogenous) inputs.
    for a in ALPHAS:
        go(f"krr_exogenous_alpha{a:g}", TwinConfig(
            n_state=2, residual_kind="kernel_ridge", residual_alpha=a,
            residual_inputs=ResidualInputs.EXOGENOUS))

    # 2. Endogenous inputs, at three capacities each.
    for mode in (ResidualInputs.STATE, ResidualInputs.MEASURED):
        for a in (1e1, 1.0, 1e-2, 1e-5):
            go(f"krr_{mode.value}_alpha{a:g}", TwinConfig(
                n_state=2, residual_kind="kernel_ridge", residual_alpha=a,
                residual_inputs=mode))

    # 3. MLP residual, for the "parameter count is not capacity" point.
    for hidden in ((4,), (16,), (64, 64)):
        go(f"mlp_exogenous_{'x'.join(map(str, hidden))}", TwinConfig(
            n_state=2, residual_kind="mlp", residual_hidden=hidden, residual_alpha=1e-4,
            residual_inputs=ResidualInputs.EXOGENOUS))

    # 4. Contaminated training: refit online on a trailing window.
    for w in (20, 60):
        go(f"online_refit_w{w}", TwinConfig(
            n_state=2, residual_kind="kernel_ridge", residual_alpha=1.0,
            residual_inputs=ResidualInputs.EXOGENOUS, online_refit=w), online=w)

    # 5. State dimension: the same trap in state-space form.
    for ns in (1, 2, 3):
        go(f"state_dim_{ns}", TwinConfig(n_state=ns, residual_kind="none"))

    # 6. The careless control: R from the window the residual was fitted on.
    go("R_from_fit_window", TwinConfig(
        n_state=2, residual_kind="kernel_ridge", residual_alpha=1e-3,
        residual_inputs=ResidualInputs.EXOGENOUS), R_from_fit=True)

    out = {
        "host": host_info(),
        "config": {
            "machine": MACHINE, "cadence_h": CADENCE_H, "arl0_target_frames": ARL0_TARGET,
            "n_healthy": hp.n, "n_fit": fit.n, "n_cal": cal.n, "n_ver": ver.n,
            "n_degradation": dg.n, "obs_channels": list(obs.names),
            "alphas": list(ALPHAS),
        },
        "runs": runs,
        "wall_s_total": float(time.time() - t_start),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out} ({time.time() - t_start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
