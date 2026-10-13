#!/usr/bin/env python3
"""Train, quantise and export an edge detector to a C header.

    python3 tools/export_model.py --machine motor_pump_37kw \
        --out hardware/stwinbx1/firmware/model/pdm_detector.h

Closes the gap between "a detector is trained" and "the microcontroller can
score with it". Writes the header plus a JSON sidecar for the model registry,
and prints the verification numbers that decide whether the export is usable.

In production the training data comes from the asset's own commissioning window
(`--features <npz>`); the simulated path exists so the whole chain can be
exercised, and so each hardware branch has a header to build against before any
hardware exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "core")))

from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.models.anomaly import DeepSVDDDetector  # noqa: E402
from pdmcore.models.export import emit_c_header, export_deep_svdd  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, simulate  # noqa: E402


def _simulated_features(machine: str, n_healthy: int, demod: Sequence[float], seed: int):
    spec = MACHINE_LIBRARY[machine]
    fx = FeatureExtractor(FeatureConfig(
        fs_hz=spec.fs_hz, bearing=spec.bearing,
        demod_band_hz=(float(demod[0]), float(demod[1])),
        velocity_band_hz=(2.0, 1000.0) if spec.shaft_hz < 10 else (10.0, 1000.0)))
    healthy = [simulate(spec, None, duration_s=0.4, seed=seed + i) for i in range(n_healthy)]
    Xh = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in healthy])
    # Faults are needed for *activation-scale calibration*, not for training.
    # Calibrating the quantisation on healthy data alone sets the scales too
    # small and saturates on exactly the signals that matter.
    modes = ["bearing_outer", "bearing_inner", "bearing_ball", "unbalance",
             "misalignment", "looseness", "cavitation"]
    faulty = [
        simulate(spec, {m: s}, duration_s=0.4, seed=seed + 5000 + i)
        for i, (m, s) in enumerate([(m, s) for m in modes for s in (0.25, 0.5, 0.85)])
    ]
    Xf = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in faulty])
    return fx, Xh, Xf, spec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output C header path")
    ap.add_argument("--name", default="pdm_detector", help="C symbol prefix")
    ap.add_argument("--features", default=None,
                    help="npz with 'healthy' and optional 'faulty' feature matrices")
    ap.add_argument("--machine", default="motor_pump_37kw",
                    help="simulate from this preset when --features is absent")
    ap.add_argument("--demod", nargs=2, type=float, default=(2500.0, 6000.0),
                    metavar=("LO", "HI"))
    ap.add_argument("--hidden", type=int, nargs="+", default=(24, 12))
    ap.add_argument("--n-healthy", type=int, default=120)
    ap.add_argument("--train-frames", type=int, default=60,
                    help="healthy frames used for fitting; the rest calibrate the threshold")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--fpr", type=float, default=0.01, help="target false-alarm rate")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if args.features:
        data = np.load(args.features, allow_pickle=False)
        Xh = np.asarray(data["healthy"], dtype=np.float64)
        Xf = np.asarray(data["faulty"], dtype=np.float64) if "faulty" in data else np.empty((0, Xh.shape[1]))
        names = [str(n) for n in data["names"]] if "names" in data else None
        fx = None
        print(f"features   {args.features}: {Xh.shape[0]} healthy, {Xf.shape[0]} faulty")
        if Xf.shape[0] == 0:
            print("WARNING    no faulty frames supplied. Activation scales will be "
                  "calibrated on healthy data only and will saturate on faults -- "
                  "measured at 36 % mean score error when this was the default.",
                  file=sys.stderr)
    else:
        fx, Xh, Xf, spec = _simulated_features(args.machine, args.n_healthy, args.demod, args.seed)
        names = list(fx.names)
        print(f"features   SIMULATED from {spec.name} ({spec.rpm:.0f} rpm): "
              f"{Xh.shape[0]} healthy, {Xf.shape[0]} faulty")
        print("           not the real asset -- re-export from commissioning data before "
              "this model scores a work order")

    if Xh.shape[0] < args.train_frames + 10:
        raise SystemExit(f"need >= {args.train_frames + 10} healthy frames, got {Xh.shape[0]}")

    train, holdout = Xh[: args.train_frames], Xh[args.train_frames :]
    det = DeepSVDDDetector(feature_names=names, hidden=tuple(args.hidden),
                           epochs=args.epochs, seed=args.seed)
    det.fit(train)
    print(f"trained    {len(names or [])}-{'-'.join(str(h) for h in args.hidden)} on "
          f"{train.shape[0]} frames, final loss {det.loss_history_[-1]:.5f}")

    X_cal = np.vstack([holdout, Xf]) if Xf.size else holdout
    qm = export_deep_svdd(det, X_cal, X_healthy_holdout=holdout, fpr=args.fpr)
    v = qm.verification

    print(f"threshold  {v['threshold']:.4f} at {args.fpr * 100:g} % FPR "
          f"(from {holdout.shape[0]} held-out healthy frames)")
    print(f"           train-score threshold would have been "
          f"{det.threshold_at_fpr(args.fpr):.4f} -- "
          f"{v['threshold'] / max(det.threshold_at_fpr(args.fpr), 1e-9):.1f}x lower, "
          f"which would alarm on healthy data")
    print(f"flash      {qm.bytes_int8} B int8 vs {qm.bytes_float32} B float32 "
          f"({qm.bytes_float32 / max(1, qm.bytes_int8):.1f}x), {qm.macs} MACs")
    print(f"quant err  mean {v['mean_rel_error'] * 100:.2f} %  "
          f"p95 {v['p95_rel_error'] * 100:.2f} %  max {v['max_rel_error'] * 100:.2f} %")
    print(f"flips      {v['decision_flips']}/{v['n_calibration']} "
          f"({v['decision_flip_rate'] * 100:.1f} %): "
          f"{v['flips_to_alarm']} -> alarm, {v['flips_to_normal']} -> normal; "
          f"worst margin {v['max_flip_margin'] * 100:.1f} % of threshold")

    if v["decision_flip_rate"] > 0.05:
        print("REJECT     flip rate above 5 %: do not deploy this export", file=sys.stderr)
        return 1
    if v["flips_to_normal"] and v["max_flip_margin"] > 0.25:
        print("REJECT     a non-borderline sample flipped to normal: that is a missed "
              "detection introduced by quantisation", file=sys.stderr)
        return 1

    emit_c_header(qm, name=args.name, path=args.out)
    print(f"wrote      {args.out} ({os.path.getsize(args.out)} B)")
    sidecar = os.path.splitext(args.out)[0] + ".json"
    print(f"wrote      {sidecar}  sha256 {qm.sha256()[:16]}...")
    with open(sidecar, encoding="utf-8") as fh:
        meta = json.load(fh)
    print(f"registry   kind={meta['kind']} params={meta['n_parameters']} "
          f"compression={meta['compression']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
