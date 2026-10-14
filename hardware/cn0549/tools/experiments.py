#!/usr/bin/env python3
"""Two experiments that decide whether the CN0549 is worth its cost.

    python3 hardware/cn0549/tools/experiments.py --out hardware/cn0549/results

This kit is the expensive one: ~EUR 600 of ADI hardware plus a EUR 300
calibrated piezo, mains-powered, against a EUR 90 battery MEMS node. The
justification has to be a measurement, not a datasheet comparison, so here are
the two measurements.

EXPERIMENT 1 -- WHAT DOES 24-BIT BUY? (ANSWER: NOTHING. THE NOISE FLOOR DOES.)

The first version of this experiment quantised the same record to 16, 20 and 24
bits and compared the BPFO harmonic-energy ratio. All three agreed to three
decimal places at every severity from 0.002 to 0.6. That is not a bug; it is
the answer:

    16-bit at +-16 g -> 488 ug LSB -> quantisation noise 141 ug RMS
                                   -> 1.1 ug/sqrt(Hz) over 16 kHz

and the IIS3DWB's own SENSOR noise is 75 ug/sqrt(Hz), 68x higher. The converter
is irrelevant on every part in this lineup. Adding bits to a sensor-limited
measurement buys dynamic range, not resolution.

So the experiment was rewritten to sweep what actually differs:

    IIS3DWB  16-bit, +-16 g,  75 ug/sqrt(Hz)
    ADXL355  20-bit, +-8 g,   25 ug/sqrt(Hz)
    TE 8911  24-bit, +-41 g,   7 ug/sqrt(Hz)   <- the CN0549 chain

Method: simulate the same machine at a sweep of outer-race severities, add each
part's noise density, quantise to its grid, extract the real feature vector, and
track the BPFO harmonic-energy ratio -- the feature the bearing diagnosis
actually rests on. The lowest severity each part can separate from healthy at
3 sigma is the answer.

Keep in mind what this is measured on: a simulator whose noise model was written
in this repository. It can show that noise density dominates bit depth, which is
an argument about physics. It cannot tell you the true detection threshold of a
real piezo on a real bearing.

EXPERIMENT 2 -- WHAT A DISCONNECTED SENSOR LOOKS LIKE

The failure this kit's IEPE bias check exists for.

THE HYPOTHESIS WAS WRONG, and the measurement is more useful than the guess. The
prediction was that an AC-coupled input with no sensor -- converter noise, the
smoothest signal the system will ever see -- would score as *less* anomalous
than a healthy machine, making the worst possible condition invisible.

It does not. The measurement:

    threshold (1 % FPR)            9.7
    healthy machine               5.8   ->  0/20 alarms
    disconnected sensor          52.9   -> 20/20 alarms
    outer-race fault at 0.35     84.8   -> 20/20 alarms

A Mahalanobis detector flags a dead channel loudly, because converter noise is
not near the healthy centre -- it is far outside it in a different direction.
Featureless is as anomalous as faulty.

The real failure is one level up: the detector alarms, and the RULE BASE then
diagnoses the noise. A maintenance planner receives a bearing work order for a
machine that is fine, while the actual fault is a cut cable. The score cannot
distinguish "bearing fault" from "no sensor" -- only an out-of-band check can,
which is what the bias voltage read is for. Experiment 2 reports what the rule
base actually says about converter noise, because that is the artefact that
reaches a human.

Both experiments write JSON and print a table. The figures for the deck are
generated from the same JSON, so a slide cannot disagree with the measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.health.rules import baseline_from_vectors, diagnose  # noqa: E402
from pdmcore.models.anomaly import MahalanobisDetector  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, simulate  # noqa: E402

FS = 32000.0
BLOCK = 16384
#: (bits, full_scale_g, noise_density_g_per_rt_hz) per sensing chain.
#: Noise densities are datasheet figures: IIS3DWB DS12749, ADXL355 Rev C,
#: TE 8911 datasheet (broadband resolution converted to a density over its
#: stated band). The bit depth is carried along to show that it does not matter.
CHAINS = {
    "IIS3DWB 16-bit": (16, 16.0, 75e-6),
    "ADXL355 20-bit": (20, 8.0, 25e-6),
    "CN0549 24-bit": (24, 40.96, 7e-6),
}


def quantise(x: np.ndarray, bits: int, full_scale_g: float) -> np.ndarray:
    """Round to a converter's grid, and clip at its full scale.

    Mid-tread rounding with a symmetric clip, which is what a real bipolar
    converter does. Clipping is included on purpose: the 24-bit path has 2.5x
    the range of the 16-bit one here, and ignoring that would flatter the
    coarse quantiser on the loudest records.
    """
    lsb = 2.0 * full_scale_g / (1 << bits)
    q = np.round(x / lsb) * lsb
    return np.clip(q, -full_scale_g, full_scale_g - lsb)


def add_sensor_noise(x: np.ndarray, density_g: float, fs_hz: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Add white noise at a stated density.

    A density in g/sqrt(Hz) becomes an RMS of density * sqrt(fs/2) over the
    Nyquist band, which is the figure that competes with the signal. This is
    the term the simulator does not already model: its own broadband noise
    stands for the MACHINE, not for the sensor.
    """
    rms = density_g * np.sqrt(fs_hz / 2.0)
    return x + rng.normal(0.0, rms, x.size)


def _extractor() -> FeatureExtractor:
    return FeatureExtractor(FeatureConfig(
        fs_hz=FS, n_fft=4096, envelope_n_fft=2048,
        demod_band_hz=(2500.0, 6000.0),
    ))


def experiment_resolution(severities: Sequence[float], seeds: int = 5) -> Dict[str, object]:
    """BPFO harmonic-energy ratio vs severity, per quantiser."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS)
    ex = _extractor()
    key = "bearing.outer_ratio"
    if key not in ex.names:
        candidates = [n for n in ex.names if "outer" in n]
        if not candidates:
            raise SystemExit(f"no outer-race feature in {ex.names}")
        key = candidates[0]

    out: Dict[str, List[float]] = {q: [] for q in CHAINS}
    spread: Dict[str, List[float]] = {q: [] for q in CHAINS}
    for sev in severities:
        per_q: Dict[str, List[float]] = {q: [] for q in CHAINS}
        for s in range(seeds):
            fault = None if sev <= 0.0 else {"bearing_outer": float(sev)}
            res = simulate(spec, fault, duration_s=BLOCK / FS, seed=1000 + s)
            x = np.asarray(res.accel_g, dtype=np.float64)[:BLOCK]
            for qname, (bits, fsg, density) in CHAINS.items():
                # One RNG per (severity, seed, chain) so the three chains see
                # different noise realisations but the same machine record --
                # the comparison is between sensing chains, not between
                # simulations.
                rng = np.random.default_rng(hash((qname, s, round(sev, 6))) % (2**32))
                noisy = add_sensor_noise(x, density, FS, rng)
                frame = ex.extract(quantise(noisy, bits, fsg), spec.shaft_hz, fs_hz=FS)
                per_q[qname].append(frame.get(key))
        for qname, vals in per_q.items():
            out[qname].append(float(np.median(vals)))
            spread[qname].append(float(np.std(vals)))

    # Where does each quantiser stop separating fault from healthy? Defined as
    # the lowest severity whose median ratio exceeds the healthy median by more
    # than 3 standard deviations of the healthy case -- the same 3-sigma
    # convention the health index uses, so the numbers are comparable.
    thresholds: Dict[str, Optional[float]] = {}
    margins: Dict[str, Optional[float]] = {}
    for qname in CHAINS:
        healthy = out[qname][0]
        noise = max(spread[qname][0], 1e-9)
        limit = healthy + 3.0 * noise
        found = None
        margin = None
        for sev, value in zip(severities, out[qname]):
            if sev > 0.0 and value > limit:
                found = float(sev)
                # How far past the limit it landed, in units of the limit. A
                # margin near 1.0 means the chain only just crossed, so the
                # reported severity is one grid step away from flipping -- and
                # a single number that flips between runs is worse than a
                # number with its margin attached. The first version of this
                # function reported only the severity, and the committed JSON
                # then disagreed with the README that quoted an earlier run.
                margin = float(value / limit)
                break
        thresholds[qname] = found
        margins[qname] = margin

    return {
        "feature": key,
        "severities": [float(s) for s in severities],
        "median": out,
        "std": spread,
        "detection_severity": thresholds,
        "detection_margin": margins,
        "seeds": seeds,
        "block_samples": BLOCK,
        "fs_hz": FS,
        "chains": {k: {"bits": v[0], "full_scale_g": v[1], "noise_density_g": v[2],
                       "adc_noise_density_g": (2.0 * v[1] / (1 << v[0])) / np.sqrt(12.0)
                                              / np.sqrt(FS / 2.0)}
                   for k, v in CHAINS.items()},
    }


def experiment_disconnected(n_healthy: int = 60, seeds_eval: int = 20) -> Dict[str, object]:
    """Score a healthy machine, a faulty machine, and a disconnected sensor."""
    spec = replace(MACHINE_LIBRARY["motor_pump_37kw"], fs_hz=FS)
    ex = _extractor()
    rng = np.random.default_rng(7)

    def frames(fault, n, seed0):
        rows = []
        for i in range(n):
            res = simulate(spec, fault, duration_s=BLOCK / FS, seed=seed0 + i)
            x = np.asarray(res.accel_g, dtype=np.float64)[:BLOCK]
            x = add_sensor_noise(x, 7e-6, FS, rng)
            rows.append(ex.extract(quantise(x, 24, 40.96), spec.shaft_hz, fs_hz=FS).vector)
        return np.asarray(rows)

    # A disconnected AC-coupled input: converter noise only. 4.9 ug LSB with
    # ~1 LSB RMS of broadband noise is generous to the failure case -- a real
    # open input often picks up mains hum, which would look *more* anomalous,
    # so this is the hard version of the test.
    # A disconnected AC-coupled input: the converter's and front end's own
    # noise, nothing else. Generous to the failure case -- a real open input
    # often picks up mains hum, which would look MORE anomalous, so this is the
    # hard version of the test.
    lsb_g = 2.0 * 40.96 / (1 << 24)
    def noise_frames(n):
        rows = []
        for i in range(n):
            x = rng.normal(0.0, lsb_g, BLOCK)
            x = add_sensor_noise(x, 7e-6, FS, rng)
            rows.append(ex.extract(quantise(x, 24, 40.96), spec.shaft_hz, fs_hz=FS).vector)
        return np.asarray(rows)

    train = frames(None, n_healthy, 2000)
    holdout = frames(None, seeds_eval, 5000)
    faulty = frames({"bearing_outer": 0.35}, seeds_eval, 6000)
    dead = noise_frames(seeds_eval)

    # Mahalanobis with Ledoit-Wolf shrinkage: the cheapest detector that
    # accounts for feature covariance, and the one the cloud uses as its
    # baseline. The finding below is not an artefact of a fancy model.
    det = MahalanobisDetector().fit(train)
    threshold = det.calibrate(holdout, fpr=0.01)
    s_healthy = det.score(holdout)
    s_faulty = det.score(faulty)
    s_dead = det.score(dead)

    # What reaches a human: the rule base's reading of converter noise. The
    # score says "anomalous"; this says what the work order will claim.
    baseline = baseline_from_vectors(train, ex.names)
    dead_frame = ex.extract(
        rng.normal(0.0, lsb_g, BLOCK), spec.shaft_hz, fs_hz=FS)
    dead_dx = diagnose(dead_frame, baseline=baseline, min_confidence=0.0, max_results=3)
    faulty_res = simulate(spec, {"bearing_outer": 0.35}, duration_s=BLOCK / FS, seed=9999)
    faulty_frame = ex.extract(
        add_sensor_noise(np.asarray(faulty_res.accel_g)[:BLOCK], 7e-6, FS, rng),
        spec.shaft_hz, fs_hz=FS)
    faulty_dx = diagnose(faulty_frame, baseline=baseline, min_confidence=0.0, max_results=3)

    return {
        "threshold": float(threshold),
        "diagnosis_disconnected": [
            {"fault": d.fault, "confidence": round(float(d.confidence), 3),
             "severity": str(d.severity), "action": d.action}
            for d in dead_dx
        ],
        "diagnosis_faulty": [
            {"fault": d.fault, "confidence": round(float(d.confidence), 3),
             "severity": str(d.severity), "action": d.action}
            for d in faulty_dx
        ],
        "healthy": {"median": float(np.median(s_healthy)), "p95": float(np.percentile(s_healthy, 95))},
        "faulty": {"median": float(np.median(s_faulty)), "p95": float(np.percentile(s_faulty, 95))},
        "disconnected": {"median": float(np.median(s_dead)), "p95": float(np.percentile(s_dead, 95))},
        "disconnected_alarms": int(np.sum(s_dead > threshold)),
        "faulty_alarms": int(np.sum(s_faulty > threshold)),
        "healthy_alarms": int(np.sum(s_healthy > threshold)),
        "n_eval": seeds_eval,
        "scores": {
            "healthy": [float(v) for v in s_healthy],
            "faulty": [float(v) for v in s_faulty],
            "disconnected": [float(v) for v in s_dead],
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="fewer severities and seeds")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    # Deliberately weighted towards VERY low severities. The first version of
    # this sweep started at 0.05 and every quantiser gave identical answers,
    # because at 0.05 the simulated impulse train is already far above a 488 ug
    # LSB. The interesting region is an order of magnitude below that.
    sevs = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.30]
    if args.quick:
        sevs = [0.0, 0.005, 0.02, 0.10]

    print("experiment 1: what does 24-bit buy?")
    r1 = experiment_resolution(sevs, seeds=2 if args.quick else args.seeds)
    print(f"  feature: {r1['feature']}, {r1['seeds']} seeds, "
          f"{r1['block_samples']} samples at {r1['fs_hz']:.0f} Hz")
    print("  sensing chain        bit depth   sensor noise   ADC noise    ratio")
    for name, c in r1["chains"].items():
        print(f"  {name:20s} {c['bits']:9d}   "
              f"{c['noise_density_g'] * 1e6:8.1f} ug/rtHz "
              f"{c['adc_noise_density_g'] * 1e6:8.2f} ug/rtHz "
              f"{c['noise_density_g'] / c['adc_noise_density_g']:7.0f}x")
    print("  (the right-hand column is why bit depth does not matter: every"
          " chain is sensor-limited)")
    head = "  severity  " + "".join(f"{q:>20s}" for q in CHAINS)
    print(head)
    for i, sev in enumerate(sevs):
        row = f"  {sev:8.3f}  "
        for q in CHAINS:
            row += f"{r1['median'][q][i]:>20.3f}"
        print(row)
    print("  first severity detected at 3 sigma above healthy, with the margin")
    print("  by which it crossed (near 1.0 = one grid step from flipping):")
    for q, sev in r1["detection_severity"].items():
        m = r1["detection_margin"][q]
        note = "" if m is None or m > 1.5 else "   <-- borderline"
        print(f"    {q:22s} {sev if sev is not None else 'never in this sweep'}"
              f"   margin {m:.2f}{note}" if m is not None else
              f"    {q:22s} never in this sweep")

    print("\nexperiment 2: does a disconnected sensor look healthy?")
    r2 = experiment_disconnected(n_healthy=40 if args.quick else 60,
                                 seeds_eval=20)
    print(f"  threshold (1 % FPR on held-out healthy): {r2['threshold']:.3f}")
    for name in ("healthy", "faulty", "disconnected"):
        d = r2[name]
        print(f"  {name:14s} median score {d['median']:8.3f}   p95 {d['p95']:8.3f}")
    print(f"  alarms: healthy {r2['healthy_alarms']}/{r2['n_eval']}, "
          f"faulty {r2['faulty_alarms']}/{r2['n_eval']}, "
          f"disconnected {r2['disconnected_alarms']}/{r2['n_eval']}")
    if r2["disconnected"]["median"] < r2["healthy"]["median"]:
        print("  DISCONNECTED SCORES BELOW HEALTHY -- invisible in the data")
    else:
        print("  a disconnected sensor DOES alarm: featureless is as anomalous "
              "as faulty.")
        print("  but the score cannot say which, and the rule base then "
              "diagnoses the noise:")
        for d in r2["diagnosis_disconnected"]:
            print(f"    disconnected -> {d['fault']:22s} confidence "
                  f"{d['confidence']:.2f}  severity {d['severity']}")
        for d in r2["diagnosis_faulty"]:
            print(f"    real fault   -> {d['fault']:22s} confidence "
                  f"{d['confidence']:.2f}  severity {d['severity']}")
        print("  A work order is raised either way. Only the IEPE bias read "
              "says which one is real.")

    path = os.path.join(args.out, "experiments.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"resolution": r1, "disconnected": r2}, fh, indent=1)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
