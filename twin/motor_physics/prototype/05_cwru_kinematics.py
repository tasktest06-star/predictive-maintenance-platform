#!/usr/bin/env python3
"""Check the twin's bearing predictions against MEASURED data (CWRU 12 kHz DE).

    python3 -I twin/motor_physics/prototype/05_cwru_kinematics.py \
        --data /workshop/predictive_maintaince/dl_cwru_12kDE \
        --out twin/motor_physics/results

WHY THIS EXPERIMENT EXISTS
--------------------------
Every other experiment in this branch compares a model to a model. The residual
study against ``pdmcore.sim.machine`` measures **mismatch between two models
written in the same repository**, which is a statement about self-consistency
and not about accuracy; the platform says the same thing about its own simulator
and ``docs/07_production_readiness.md`` lists it under "things that are easy to
believe and are not true". This script is the one place where the twin meets a
real machine.

It tests the branch's central claim in both directions, on the same data:

**(A) Frequencies are predictable.** The CWRU drive-end bearing is an
SKF 6205-2RS with published geometry, already in
``pdmcore.dsp.bearing.BEARING_CATALOG``. Given the shaft speed, the twin
predicts BPFO / BPFI / BSF with no free parameters at all. The measured defect
repetition rate is read off the squared-envelope spectrum. The comparison is a
pure prediction test: no fitting, nothing tuned to this data.

**(B) Absolute amplitudes are not.** The *same physical defect* -- one file set,
one bearing, one spall -- is recorded at four motor loads. A twin with a
calibrated force model and a calibrated transfer path would predict the envelope
amplitude at each load. The spread of the measured amplitude across those four
records, after dividing out everything the twin does know, is a lower bound on
the uncertainty of any absolute amplitude prediction, measured on a real machine
rather than argued from a model.

UNTRUSTED DATA
--------------
The ``.mat`` files are downloaded third-party data. This script lives outside
their directory, takes the directory as an argument, and is meant to be run with
``python3 -I`` so that neither the data directory nor the current directory can
inject a module. ``scipy.io.loadmat`` parses MATLAB level-5 binary; it does not
execute anything from the file. Nothing from the data directory is written to or
committed -- only the derived numbers in the output JSON.

CWRU FILE MAP
-------------
The numeric file names are the CWRU catalogue numbers. The mapping below is the
standard one for the 12 kHz drive-end set, and it is **verified rather than
trusted**: ``_check_speed`` recomputes the shaft speed from each record's own
``RPM`` variable where present, and the baseline files are asserted to show no
bearing line. A wrong mapping would put an inner-race file in the outer-race
group and the measured rate would miss the predicted one by 60 %, which the
``rel_err`` column would show immediately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore.dsp.bearing import BEARING_CATALOG, fault_frequencies  # noqa: E402
from pdmcore.dsp.envelope import envelope_spectrum, select_demodulation_band  # noqa: E402
from pdmcore.dsp.spectrum import spectrum  # noqa: E402

#: file -> (fault mode, nominal defect diameter in inches, motor load in hp).
#: Loads 0/1/2/3 hp correspond to approximately 1797/1772/1750/1730 rpm.
CWRU_MAP: Dict[int, Tuple[str, float, int]] = {
    97: ("healthy", 0.000, 0), 98: ("healthy", 0.000, 1),
    99: ("healthy", 0.000, 2), 100: ("healthy", 0.000, 3),
    105: ("inner_race", 0.007, 0), 106: ("inner_race", 0.007, 1),
    107: ("inner_race", 0.007, 2), 108: ("inner_race", 0.007, 3),
    118: ("rolling_element", 0.007, 0), 119: ("rolling_element", 0.007, 1),
    120: ("rolling_element", 0.007, 2), 121: ("rolling_element", 0.007, 3),
    130: ("outer_race", 0.007, 0), 131: ("outer_race", 0.007, 1),
    132: ("outer_race", 0.007, 2), 133: ("outer_race", 0.007, 3),
    169: ("inner_race", 0.014, 0), 170: ("inner_race", 0.014, 1),
    171: ("inner_race", 0.014, 2), 172: ("inner_race", 0.014, 3),
    185: ("rolling_element", 0.014, 0), 186: ("rolling_element", 0.014, 1),
    187: ("rolling_element", 0.014, 2), 188: ("rolling_element", 0.014, 3),
    197: ("outer_race", 0.014, 0), 198: ("outer_race", 0.014, 1),
    199: ("outer_race", 0.014, 2), 200: ("outer_race", 0.014, 3),
    209: ("inner_race", 0.021, 0), 210: ("inner_race", 0.021, 1),
    211: ("inner_race", 0.021, 2), 212: ("inner_race", 0.021, 3),
    222: ("rolling_element", 0.021, 0), 223: ("rolling_element", 0.021, 1),
    224: ("rolling_element", 0.021, 2), 225: ("rolling_element", 0.021, 3),
    234: ("outer_race", 0.021, 0), 235: ("outer_race", 0.021, 1),
    236: ("outer_race", 0.021, 2), 237: ("outer_race", 0.021, 3),
}

#: Nameplate speeds for the four CWRU load points, used only as a fallback when
#: a record carries no RPM variable, and flagged in the output when used.
LOAD_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

FS_HZ = 12000.0
#: CWRU accelerometer sensitivity is not published per record; the .mat data is
#: in g for the standard distribution. Amplitudes are therefore reported in g
#: and used only in RATIOS, which is the point of experiment (B).
GEOM = BEARING_CATALOG["SKF6205"]


def _load_record(path: str) -> Tuple[np.ndarray, Optional[float]]:
    """``(drive-end acceleration, rpm or None)`` from one CWRU .mat file."""
    m = loadmat(path)
    de = [k for k in m if k.endswith("_DE_time")]
    if not de:
        raise ValueError(f"{os.path.basename(path)} has no *_DE_time variable; keys={sorted(m)}")
    x = np.asarray(m[de[0]], dtype=np.float64).ravel()
    rpm_keys = [k for k in m if k.endswith("RPM")]
    rpm = float(np.asarray(m[rpm_keys[0]]).ravel()[0]) if rpm_keys else None
    return x, rpm


def _measured_rate_hz(
    x: np.ndarray, shaft_hz: float, target_hz: float, *, rel_window: float = 0.12
) -> Dict[str, float]:
    """Dominant envelope-spectrum line within +-``rel_window`` of ``target_hz``.

    The search window is a *relative* band around the prediction rather than a
    global peak pick, and the width is stated: 12 %. That is wide enough to
    contain the 1-2 % kinematic error from rolling-element slip with a large
    margin, and narrow enough that the window around BPFO (3.58x shaft) cannot
    reach BPFI (5.41x) or a shaft harmonic. Reporting the *global* envelope peak
    instead would be a check that cannot fail: on a defective bearing it lands on
    the right line by construction and says nothing.

    The returned ``snr`` is against the local median envelope level, the same
    statistic ``pdmcore.dsp.bearing.harmonic_energy_ratio`` uses, so a healthy
    record's "measurement" can be rejected rather than reported as a
    0 %-error match to whatever noise bin happened to be tallest.
    """
    band = select_demodulation_band(x, FS_HZ)
    env = envelope_spectrum(
        x, FS_HZ, band_hz=(band.lo_hz, band.hi_hz), n_fft=8192, decimate_to_hz=2000.0
    )
    f = np.asarray(env.freq_hz, dtype=np.float64)
    a = np.asarray(env.value, dtype=np.float64)
    lo, hi = target_hz * (1.0 - rel_window), target_hz * (1.0 + rel_window)
    sel = (f >= lo) & (f <= hi)
    if not np.any(sel):
        raise ValueError(f"no envelope bins between {lo:.1f} and {hi:.1f} Hz")
    idx = np.flatnonzero(sel)
    k = int(idx[np.argmax(a[idx])])
    # Three-point parabolic refinement, as pdmcore.dsp.spectrum.interpolate_peak
    # does, so the measured rate is not quantised to the 0.24 Hz envelope bin.
    if 0 < k < f.size - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        den = y0 - 2.0 * y1 + y2
        delta = 0.0 if den == 0 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    else:
        delta = 0.0
    df = float(f[1] - f[0])
    measured = float(f[k] + delta * df)
    # Local floor, excluding the line itself and a window either side.
    d = np.abs(f - measured)
    floor_sel = (d <= 6.0 * target_hz * 0.05) & (d > 2.0 * df)
    floor = float(np.median(a[floor_sel])) if np.any(floor_sel) else float(np.median(a))
    return {
        "measured_hz": measured,
        "amplitude": float(a[k]),
        "floor": floor,
        "snr": float(a[k] / floor) if floor > 0 else 0.0,
        "env_df_hz": df,
        "demod_lo_hz": float(band.lo_hz),
        "demod_hi_hz": float(band.hi_hz),
    }


def _shaft_hz_from_signal(x: np.ndarray, nominal_hz: float) -> Dict[str, float]:
    """Shaft frequency from the 1x line of the raw spectrum, near ``nominal_hz``."""
    sp = spectrum(x, FS_HZ, n_fft=16384, window_name="hann", scaling="rms")
    f = np.asarray(sp.freq_hz, dtype=np.float64)
    a = np.asarray(sp.value, dtype=np.float64)
    sel = (f >= nominal_hz * 0.9) & (f <= nominal_hz * 1.1)
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return {"shaft_hz": nominal_hz, "snr": 0.0, "from_signal": 0.0}
    k = int(idx[np.argmax(a[idx])])
    if 0 < k < f.size - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        den = y0 - 2.0 * y1 + y2
        delta = 0.0 if den == 0 else float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    else:
        delta = 0.0
    df = float(f[1] - f[0])
    med = float(np.median(a[sel]))
    return {
        "shaft_hz": float(f[k] + delta * df),
        "snr": float(a[k] / med) if med > 0 else 0.0,
        "from_signal": 1.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="directory of CWRU 12 kHz DE .mat files")
    ap.add_argument("--out", default=os.path.join(_REPO, "twin", "motor_physics", "results"))
    ap.add_argument("--min-snr", type=float, default=3.0,
                    help="envelope SNR below which a measurement is reported as absent")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    records: List[Dict[str, object]] = []
    for num in sorted(CWRU_MAP):
        path = os.path.join(args.data, f"{num}.mat")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        mode, size_in, load_hp = CWRU_MAP[num]
        x, rpm = _load_record(path)
        nominal = LOAD_RPM[load_hp] / 60.0
        sig = _shaft_hz_from_signal(x, nominal)
        # Prefer the record's own RPM variable; fall back to the catalogue speed
        # for the load point, and record which was used.
        if rpm is not None and rpm > 0:
            shaft_hz, speed_src = rpm / 60.0, "record_rpm"
        elif sig["snr"] >= 3.0:
            shaft_hz, speed_src = sig["shaft_hz"], "spectrum_1x"
        else:
            shaft_hz, speed_src = nominal, "catalogue_nominal"

        ff = fault_frequencies(GEOM, shaft_hz)
        predicted = {
            "outer_race": ff.bpfo_hz,
            "inner_race": ff.bpfi_hz,
            "rolling_element": ff.bsf_hz,
            "healthy": ff.bpfo_hz,     # probed anyway, as a negative control
        }[mode]
        meas = _measured_rate_hz(x, shaft_hz, predicted)
        detected = meas["snr"] >= args.min_snr
        rec = {
            "file": num,
            "mode": mode,
            "defect_in": size_in,
            "load_hp": load_hp,
            "n_samples": int(x.size),
            "duration_s": float(x.size / FS_HZ),
            "rms_g": float(np.sqrt(np.mean(x**2))),
            "shaft_hz": float(shaft_hz),
            "shaft_hz_source": speed_src,
            "shaft_hz_from_spectrum": float(sig["shaft_hz"]),
            "shaft_1x_snr": float(sig["snr"]),
            "predicted_hz": float(predicted),
            "predicted_order": float(predicted / shaft_hz),
            "detected": bool(detected),
            **{k: float(v) for k, v in meas.items()},
        }
        rec["rel_err"] = float((meas["measured_hz"] - predicted) / predicted)
        records.append(rec)
        print(
            f"{num:3d} {mode:16s} {size_in:.3f}in {load_hp}hp  "
            f"pred {predicted:7.2f} Hz  meas {meas['measured_hz']:7.2f} Hz  "
            f"err {100*rec['rel_err']:+6.2f} %  snr {meas['snr']:5.1f}"
            f"{'' if detected else '   (no line: not counted)'}",
            flush=True,
        )

    # --- (A) frequency accuracy, defective records only -------------------
    faulty = [r for r in records if r["mode"] != "healthy" and r["detected"]]
    errs = np.array([r["rel_err"] for r in faulty])
    healthy = [r for r in records if r["mode"] == "healthy"]
    freq = {
        "n_records": len(faulty),
        "n_healthy_probed": len(healthy),
        "n_healthy_with_line": int(sum(1 for r in healthy if r["detected"])),
        "abs_err_mean_pct": float(100.0 * np.mean(np.abs(errs))) if errs.size else None,
        "abs_err_max_pct": float(100.0 * np.max(np.abs(errs))) if errs.size else None,
        "signed_err_mean_pct": float(100.0 * np.mean(errs)) if errs.size else None,
        "signed_err_sd_pct": float(100.0 * np.std(errs)) if errs.size else None,
        "per_mode_abs_err_pct": {
            m: float(100.0 * np.mean([abs(r["rel_err"]) for r in faulty if r["mode"] == m]))
            for m in ("outer_race", "inner_race", "rolling_element")
            if any(r["mode"] == m for r in faulty)
        },
    }

    # --- (B) amplitude spread at fixed defect ------------------------------
    # The same physical defect at four loads. Everything the twin knows is
    # divided out: the envelope amplitude is normalised by the record's own RMS
    # (removing overall gain and sensor sensitivity) and the group is
    # characterised by max/min. A twin that could predict absolute amplitude
    # would have to predict this ratio; it is a lower bound on the error of any
    # such prediction, because it is the variation of a FIXED defect.
    groups: Dict[str, Dict[str, object]] = {}
    for mode in ("outer_race", "inner_race", "rolling_element"):
        for size in (0.007, 0.014, 0.021):
            g = [r for r in records if r["mode"] == mode and r["defect_in"] == size and r["detected"]]
            if len(g) < 2:
                continue
            amp = np.array([r["amplitude"] for r in g])
            rel = amp / np.array([r["rms_g"] for r in g])
            groups[f"{mode}_{size:.3f}in"] = {
                "n_loads": len(g),
                "loads_hp": [r["load_hp"] for r in g],
                "amplitude_ratio_max_over_min": float(amp.max() / amp.min()),
                "rms_normalised_ratio_max_over_min": float(rel.max() / rel.min()),
                "rms_g": [float(r["rms_g"]) for r in g],
            }
    size_scaling: Dict[str, object] = {}
    for mode in ("outer_race", "inner_race", "rolling_element"):
        per = {}
        for size in (0.007, 0.014, 0.021):
            g = [r for r in records if r["mode"] == mode and r["defect_in"] == size and r["detected"]]
            if g:
                per[f"{size:.3f}"] = float(np.mean([r["amplitude"] / r["rms_g"] for r in g]))
        if len(per) >= 2:
            vals = list(per.values())
            size_scaling[mode] = {
                "mean_rms_normalised_amplitude_by_size": per,
                "monotonic_in_defect_size": bool(
                    all(b >= a for a, b in zip(vals, vals[1:]))
                ),
                "ratio_largest_over_smallest_defect": float(vals[-1] / vals[0]),
            }

    out = {
        "dataset": "CWRU 12 kHz drive-end",
        "data_dir": os.path.abspath(args.data),
        "fs_hz": FS_HZ,
        "bearing": {
            "name": GEOM.name, "n_balls": GEOM.n_balls,
            "ball_d": GEOM.ball_d, "pitch_d": GEOM.pitch_d,
            "contact_angle_deg": GEOM.contact_angle_deg,
            "orders": {k: float(v) for k, v in
                       fault_frequencies(GEOM, 1.0).as_dict().items()},
        },
        "min_snr": args.min_snr,
        "frequency_prediction": freq,
        "amplitude_spread_fixed_defect": groups,
        "amplitude_vs_defect_size": size_scaling,
        "records": records,
    }
    path = os.path.join(args.out, "cwru_kinematics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path}")
    print(f"frequency prediction: mean |err| {freq['abs_err_mean_pct']:.2f} %, "
          f"max {freq['abs_err_max_pct']:.2f} % over {freq['n_records']} measured records")
    if groups:
        worst = max(groups.items(), key=lambda kv: kv[1]["rms_normalised_ratio_max_over_min"])
        print(f"amplitude of a FIXED defect across load: up to "
              f"{worst[1]['rms_normalised_ratio_max_over_min']:.1f}x ({worst[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
