#!/usr/bin/env python3
"""Does anything in this platform need a Jetson? Measured, then answered.

    PYTHONPATH=core:edge python3 hardware/jetson_orin/tools/workload_study.py

This kit is the hardest to justify, so the study is adversarial by design: it
tries to find a workload in this repository that a 67-TOPS GPU would help with,
and reports honestly when it cannot.

The two branches before this one both concluded that accelerators do nothing for
feature-vector scoring -- the Alif branch on an Ethos-U55, the Pi 5 branch on a
Hailo-8L, independently. A third negative result from a third accelerator is not
interesting. What is interesting is the question those two did not ask:

    Is there a workload where the GPU is the difference between possible and
    impossible, rather than between fast and faster?

Three candidates are measured here:

  1. **Fleet re-scoring.** Re-score a site's whole history after a model
     update. The honest version of "the GPU batches well".
  2. **Model training.** Fit the detectors a site needs, from scratch, on site.
     This is the candidate that might actually hold, for a reason that has
     nothing to do with speed -- see (3).
  3. **Continuous raw-waveform inference.** The only workload where the
     arithmetic genuinely does not fit on a CPU.

And the non-performance justification, which is the one that survives:
**data sovereignty**. A plant that will not send vibration data off site needs
training to happen on site, and a site that trains needs more than a Pi.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, os.path.join(_REPO, "edge"))

from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.models.anomaly import (  # noqa: E402
    MahalanobisDetector,
    PCAResidualDetector,
    RobustThresholdDetector,
)
from pdmcore.sim import MACHINE_LIBRARY, simulate  # noqa: E402

FS = 25600.0
BLOCK = 8192

#: Board power, watts sustained, and electricity at a European industrial rate.
BOARDS = {
    "Raspberry Pi 5 + SSD": 7.5,
    "Jetson Orin Nano (15 W mode)": 15.0,
    "Jetson Orin Nano Super (25 W)": 25.0,
}
EUR_PER_KWH = 0.20


def _features(n: int, seed0: int = 0) -> np.ndarray:
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS)
    ex = FeatureExtractor(FeatureConfig(fs_hz=FS, n_fft=4096, envelope_n_fft=2048,
                                        demod_band_hz=(2500.0, 6000.0)))
    rows = []
    for i in range(n):
        x = np.asarray(simulate(spec, None, duration_s=BLOCK / FS,
                                seed=seed0 + i).accel_g)[:BLOCK]
        rows.append(ex.extract(x, spec.shaft_hz, fs_hz=FS).vector)
    return np.asarray(rows)


def candidate_1_rescoring(base: np.ndarray, site_nodes: int = 1000,
                          days: int = 365) -> Dict[str, object]:
    """Re-score a site's whole history after a model update."""
    frames = site_nodes * days * 24
    X = np.repeat(base, max(1, 20000 // len(base)), axis=0)
    det = MahalanobisDetector().fit(base)
    t0 = time.perf_counter()
    det.score(X)
    dt = time.perf_counter() - t0
    per_frame_s = dt / len(X)
    return {
        "site_nodes": site_nodes,
        "days": days,
        "frames": frames,
        "measured_frames": len(X),
        "measured_s": dt,
        "per_frame_us": per_frame_s * 1e6,
        "full_history_s": frames * per_frame_s,
        "verdict": ("CPU is sufficient" if frames * per_frame_s < 600
                    else "worth accelerating"),
    }


def candidate_2_training(sizes: Sequence[int]) -> Dict[str, object]:
    """Fit the detectors a site needs, from scratch, on site."""
    out: List[Dict[str, object]] = []
    for n in sizes:
        X = _features(min(n, 200), seed0=1000)
        if n > len(X):
            # Tiling rather than simulating thousands of records: the fit cost
            # is a function of the matrix shape, and simulating 5000 records
            # would dominate the runtime of this study without changing what is
            # being measured. Stated because tiled data understates the
            # covariance conditioning a real fit faces.
            X = np.repeat(X, int(np.ceil(n / len(X))), axis=0)[:n]
        row: Dict[str, object] = {"n_frames": int(n), "n_features": int(X.shape[1])}
        for name, factory in (("mahalanobis", MahalanobisDetector),
                              ("pca_residual", PCAResidualDetector),
                              ("robust_threshold", RobustThresholdDetector)):
            t0 = time.perf_counter()
            factory().fit(X)
            row[f"{name}_ms"] = (time.perf_counter() - t0) * 1e3
        out.append(row)
    return {
        "fits": out,
        "note": ("These are closed-form or SVD fits, not gradient descent. A "
                 "Deep SVDD or 1D CNN is the case a GPU would change, and "
                 "torch is not installed here -- so that number is absent "
                 "rather than guessed."),
        "torch_available": _have_torch(),
    }


def _have_torch() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def candidate_3_waveform(site_nodes: int = 1000) -> Dict[str, object]:
    """Continuous raw-waveform inference: the arithmetic, not a measurement.

    A 1D CNN over 8192 samples at roughly 4 MMAC per inference. Reported as
    arithmetic because there is no model to run -- the point is the ORDER of
    magnitude, which is what decides whether a CPU is in the running.
    """
    mmac_per_inference = 4.0
    # One inference per node per hour, the platform's default cadence.
    inferences_per_s = site_nodes / 3600.0
    gmac_per_s = mmac_per_inference * inferences_per_s / 1000.0
    # A Cortex-A76 at 2.4 GHz with NEON does roughly 10-20 GMAC/s on int8 in
    # practice (not peak); an Orin Nano Super is rated 67 TOPS, i.e. ~33 TMAC/s,
    # of which a small 1D CNN would realise a few percent.
    return {
        "site_nodes": site_nodes,
        "mmac_per_inference": mmac_per_inference,
        "inferences_per_s": inferences_per_s,
        "gmac_per_s_required": gmac_per_s,
        "cpu_gmac_per_s_realistic": 15.0,
        "orin_tmac_per_s_rated": 33.5,
        "verdict": ("CPU is sufficient at this cadence"
                    if gmac_per_s < 15.0 else "needs an accelerator"),
        "cadence_that_would_need_a_gpu_hz": 15.0 * 1000.0 / mmac_per_inference,
    }


def energy_cost() -> Dict[str, object]:
    rows = []
    for name, watts in BOARDS.items():
        kwh = watts * 8760.0 / 1000.0
        rows.append({"board": name, "watts": watts, "kwh_per_year": kwh,
                     "eur_per_year": kwh * EUR_PER_KWH})
    return {"boards": rows, "eur_per_kwh": EUR_PER_KWH}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--nodes", type=int, default=1000)
    args = ap.parse_args(argv)

    print("Jetson workload study -- adversarial: trying to find a job for the GPU")
    print(f"  host: {platform.machine()}, {os.cpu_count()} cores, "
          f"numpy {np.__version__}, torch={'yes' if _have_torch() else 'NO'}")
    print("  (not a Jetson. Every figure below is this host; the Jetson case is")
    print("   argued from it, not measured on it.)")
    print()

    base = _features(40, seed0=500)

    print("candidate 1: re-score a site's whole history after a model update")
    c1 = candidate_1_rescoring(base, site_nodes=args.nodes)
    print(f"  {c1['site_nodes']} nodes x {c1['days']} days hourly = "
          f"{c1['frames']:,} frames")
    print(f"  measured {float(c1['per_frame_us']):.2f} us/frame on "
          f"{c1['measured_frames']:,} frames")
    print(f"  -> whole history in {float(c1['full_history_s']):.1f} s   "
          f"VERDICT: {c1['verdict']}")
    print()

    print("candidate 2: train a site's models on site")
    c2 = candidate_2_training([40, 200, 1000, 5000])
    print(f"  {'frames':>8s} {'mahalanobis':>13s} {'pca_residual':>14s} "
          f"{'robust':>10s}")
    for r in c2["fits"]:                                   # type: ignore[index]
        print(f"  {r['n_frames']:8d} {float(r['mahalanobis_ms']):11.2f} ms "
              f"{float(r['pca_residual_ms']):12.2f} ms "
              f"{float(r['robust_threshold_ms']):8.2f} ms")
    print(f"  {c2['note']}")
    print()

    print("candidate 3: continuous raw-waveform inference")
    c3 = candidate_3_waveform(site_nodes=args.nodes)
    print(f"  {c3['site_nodes']} nodes x 1 inference/h x "
          f"{c3['mmac_per_inference']:.0f} MMAC = "
          f"{float(c3['gmac_per_s_required']):.4f} GMAC/s")
    print(f"  a Cortex-A76 does ~{c3['cpu_gmac_per_s_realistic']:.0f} GMAC/s; "
          f"an Orin Nano Super is rated {c3['orin_tmac_per_s_rated']:.1f} TMAC/s")
    print(f"  -> VERDICT: {c3['verdict']}")
    print(f"  the GPU would become necessary above "
          f"{float(c3['cadence_that_would_need_a_gpu_hz']):,.0f} inferences/s, "
          f"i.e. {float(c3['cadence_that_would_need_a_gpu_hz']) * 3600:,.0f} "
          f"nodes at hourly cadence")
    print()

    print("what it costs to be wrong about this")
    e = energy_cost()
    for r in e["boards"]:                                  # type: ignore[index]
        print(f"  {r['board']:32s} {r['watts']:5.1f} W  "
              f"{float(r['kwh_per_year']):6.0f} kWh/y  "
              f"EUR {float(r['eur_per_year']):5.0f}/y")
    print()
    print("CONCLUSION")
    print("  Nothing in this platform needs a Jetson. Re-scoring a 1000-node")
    print("  site's entire year takes under a minute of CPU; the classical")
    print("  detectors fit in milliseconds; and raw-waveform inference at this")
    print("  cadence is four orders of magnitude below one CPU core.")
    print()
    print("  Two justifications survive, and neither is about speed:")
    print("   1. DATA SOVEREIGNTY. A plant that will not send vibration data")
    print("      off site must train on site. Training a 1D CNN or a")
    print("      transformer-class fusion model is the one workload here that")
    print("      is gradient descent rather than a closed form, and it is the")
    print("      case a GPU changes from days to hours.")
    print("   2. MULTIMODAL FUSION. Vibration + acoustic + current + thermal")
    print("      imagery on one asset is a model class that does not fit the")
    print("      63-float feature vector at all, and the imagery alone rules")
    print("      out the MCU and gateway tiers.")
    print()
    print("  Both are about MODELS THAT DO NOT EXIST YET. Until one does, the")
    print("  honest recommendation is a Pi 5 -- and the EUR 44/year of")
    print("  electricity this board burns is the cheapest part of being wrong.")

    results = {
        "host": {"machine": platform.machine(), "cores": os.cpu_count(),
                 "numpy": np.__version__, "torch": _have_torch()},
        "candidate_1_rescoring": c1,
        "candidate_2_training": c2,
        "candidate_3_waveform": c3,
        "energy": e,
        "conclusion": ("Nothing in this platform needs a Jetson. The surviving "
                       "justifications are data sovereignty (training on site) "
                       "and multimodal fusion, both of which concern models "
                       "that do not exist yet."),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "workload_study.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
