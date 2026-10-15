#!/usr/bin/env python3
"""Every number in the method note, computed from the committed code.

    python3 mcsa/prototype/method_tables.py --out mcsa/results/method_tables.json

Four tables, none of them quoted from a textbook:

1. **Window leakage** at the sideband offsets that MCSA actually works at, from
   each window's own DTFT. This is the table that decides whether a claimed
   sideband level is a measurement or an artefact.
2. **Resolution and record length versus load**, from
   ``min_record_duration_s``, plus the slip-induced position error at each
   load. The two cross over at low load, which is where broken-bar detection
   stops being possible rather than merely getting harder.
3. **The coupling budget**: for each fault mode, the sideband level the torque
   path and the air-gap path predict, from the parameters in
   ``pdmcore.sim.electrical.CouplingModel``. This is the table the adversarial
   bearing claim rests on.
4. **The negative-sequence equivalence**: how much supply unbalance, how much
   shorted turn and how much CT channel mismatch each produce the same
   negative-sequence current.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO, "core"), os.path.join(_REPO, "edge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pdmcore.dsp.bearing import BEARING_CATALOG, fault_frequencies     # noqa: E402
from pdmcore.dsp.windows import WINDOW_NAMES, window                   # noqa: E402
from pdmcore.mcsa import mcsa_frequencies, window_leakage_db           # noqa: E402
from pdmcore.mcsa.frequencies import (                                 # noqa: E402
    as_json_dict,
    min_record_duration_s,
    sideband_position_error_hz,
    slip_relative_error,
)
from pdmcore.sim import MACHINE_LIBRARY, CouplingModel                 # noqa: E402
from pdmcore.sim.electrical import CurrentSensorSpec, mech_transfer    # noqa: E402

LINE_HZ = 60.0
POLE_PAIRS = 2
SLIP_RATED = 1.0 - (1780.0 / 60.0) * POLE_PAIRS / LINE_HZ


def leakage_table() -> Dict[str, object]:
    """Window leakage at the offsets a 1 s and a 10 s record put a sideband at."""
    offsets = {
        "1s_record_2sf1": 2.0 * SLIP_RATED * LINE_HZ / 1.0,     # Hz -> bins at df=1 Hz
        "10s_record_2sf1": 2.0 * SLIP_RATED * LINE_HZ / 0.1,
        "10s_record_2sf1_quarter_load": 2.0 * (SLIP_RATED / 4) * LINE_HZ / 0.1,
        "10s_record_fr": 29.67 / 0.1,
    }
    rows: List[Dict[str, object]] = []
    for name in ("rect", "hann", "hamming", "blackman", "blackman_harris", "flattop"):
        w = window(name, 4096)
        row: Dict[str, object] = {
            "window": name,
            "enbw_bins": float(w.enbw_bins),
            "scallop_loss_db": float(w.scallop_loss_db),
        }
        for tag, bins in offsets.items():
            row[f"leak_db@{tag}"] = round(window_leakage_db(name, bins), 1)
        rows.append(row)
    return {
        "offsets_in_bins": {k: round(v, 2) for k, v in offsets.items()},
        "note": "df = 1 Hz for a 1 s record and 0.1 Hz for a 10 s one; the "
                "offset in bins is 2 s f1 / df",
        "rows": rows,
    }


def resolution_table() -> Dict[str, object]:
    rows = []
    for load in (1.0, 0.75, 0.5, 0.25, 0.1, 0.05):
        s = SLIP_RATED * load
        off = 2.0 * s * LINE_HZ
        rows.append({
            "load_frac": load,
            "slip": s,
            "offset_2sf1_hz": off,
            "min_duration_s_bh": min_record_duration_s(LINE_HZ, s,
                                                       window_name="blackman_harris"),
            "min_duration_s_hann": min_record_duration_s(LINE_HZ, s, window_name="hann"),
            "position_error_hz_at_0p5pct_speed": sideband_position_error_hz(
                LINE_HZ, s, 0.005),
            "slip_rel_error_at_0p5pct_speed": slip_relative_error(s, 0.005),
            "offset_over_position_error": off / sideband_position_error_hz(LINE_HZ, s, 0.005),
        })
    return {
        "note": "the sideband is only usable where offset_over_position_error > 1: "
                "below that the predicted sideband position is less certain than "
                "its distance from the fundamental",
        "rows": rows,
    }


def coupling_table() -> Dict[str, object]:
    """Predicted sideband level per fault mode, per path, at the reference point."""
    spec = MACHINE_LIBRARY["motor_pump_37kw"]
    c = CouplingModel()
    fr = spec.shaft_hz
    ff = fault_frequencies(BEARING_CATALOG["SKF6205"], fr)
    i_rated = spec.power_kw * 1000.0 / (400.0 * np.sqrt(3) * 0.88)
    i_mag = c.mag_current_frac * i_rated
    i_tq = float(np.sqrt(i_rated ** 2 - i_mag ** 2))
    i1 = float(np.hypot(i_mag, i_tq))
    w_sync = 2.0 * np.pi * LINE_HZ / POLE_PAIRS
    t_rated = spec.power_kw * 1000.0 / (w_sync * (1.0 - SLIP_RATED))
    slope = t_rated / (SLIP_RATED * w_sync)
    fc = slope / (2.0 * np.pi * c.inertia_kg_m2)

    freqs = {
        "unbalance": fr, "misalignment": 2 * fr, "looseness": fr,
        "bearing_outer": ff.bpfo_hz, "bearing_inner": ff.bpfi_hz,
        "bearing_ball": ff.bsf_hz, "bearing_cage": ff.ftf_hz,
        "gear_tooth": fr, "gear_wear": fr, "belt": fr,
    }
    rows = []
    for mode, fm in freqs.items():
        h = mech_transfer(fm, torque_slope=slope, inertia_kg_m2=c.inertia_kg_m2)
        tq = c.torque_frac.get(mode, 0.0)
        sb_tq = 0.5 * tq * h * (i_tq / i1) if tq else 0.0
        d_um = c.disp_um.get(mode, 0.0)
        eps = d_um * 1e-3 / c.airgap_mm
        sb_ag = 0.5 * eps * (i_mag / i1) * c.k_winding if d_um else 0.0
        total = float(np.hypot(sb_tq, sb_ag))
        rows.append({
            "mode": mode,
            "modulation_hz": fm,
            "h_mech": h,
            "torque_frac_at_sev1": tq,
            "disp_um_at_sev1": d_um,
            "ecc_frac": eps,
            "sideband_torque_db": _db(sb_tq),
            "sideband_airgap_db": _db(sb_ag),
            "sideband_total_db": _db(total),
            "dominant_path": "airgap" if sb_ag > sb_tq else "torque",
        })
    # Rotor bar, by the closed form, at one and at four broken bars.
    bars = []
    for n_broken in (1, 2, 4):
        lsb = (n_broken / (2.0 * c.rotor_bars * POLE_PAIRS)) * (i_tq / i1)
        bars.append({"n_broken": n_broken, "of_bars": c.rotor_bars,
                     "lsb_db_full_load": _db(lsb),
                     "lsb_db_half_load": _db(lsb * _load_scale(0.5, c, i_rated)),
                     "lsb_db_quarter_load": _db(lsb * _load_scale(0.25, c, i_rated))})
    return {
        "operating_point": {
            "shaft_hz": fr, "slip": SLIP_RATED, "i_rated_a": i_rated,
            "i_mag_a": i_mag, "i_torque_a": i_tq, "i1_a": i1,
            "t_rated_nm": t_rated, "torque_slope_nm_s_per_rad": slope,
            "mech_corner_hz": fc,
            "bearing_hz": ff.as_dict(),
        },
        "coupling_model": c.as_dict(),
        "modes": rows,
        "rotor_bar_closed_form": bars,
        "note": "sideband levels are PER SIDEBAND, dB relative to the fundamental, "
                "at severity 1.0 and full load",
    }


def _load_scale(load: float, c: CouplingModel, i_rated: float) -> float:
    """How the rotor-bar sideband-to-fundamental ratio scales with load."""
    i_mag = c.mag_current_frac * i_rated
    i_tq_full = float(np.sqrt(i_rated ** 2 - i_mag ** 2))
    i_tq = i_tq_full * load
    i1 = float(np.hypot(i_mag, i_tq))
    i1_full = float(np.hypot(i_mag, i_tq_full))
    return (i_tq / i1) / (i_tq_full / i1_full)


def _db(x: float) -> float:
    return round(float(20.0 * np.log10(x)), 1) if x > 0 else -200.0


def negative_sequence_table() -> Dict[str, object]:
    """What produces the same negative-sequence current as what."""
    c = CouplingModel()
    z = 0.17
    ct = CurrentSensorSpec()
    rows = []
    for unb in (0.5, 1.0, 2.0, 3.0):
        rows.append({"cause": "supply voltage unbalance", "level_pct": unb,
                     "i_neg_pct": unb / z})
    for turns in (0.5, 1.0, 2.0):
        rows.append({"cause": "shorted turns, fraction of one phase",
                     "level_pct": turns, "i_neg_pct": turns / z})
    for g in (0.5, 1.0, 2.0):
        # A gain error g on one of three phases maps almost entirely into the
        # negative sequence: |I-|/|I+| ~ g/3 for a single-channel error.
        rows.append({"cause": "CT channel gain mismatch (one phase)",
                     "level_pct": g, "i_neg_pct": g / 3.0})
    return {
        "neg_seq_impedance_pu": z,
        "ct_class": ct.name,
        "rows": rows,
        "note": "a 1 % supply unbalance and a 1 % shorted-turn fraction produce "
                "IDENTICAL negative-sequence current, which is why the "
                "indicator needs the voltage channel and the Z- ratio",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="mcsa/results/method_tables.json")
    args = ap.parse_args(argv)

    spec = MACHINE_LIBRARY["motor_pump_37kw"]
    ff = fault_frequencies(BEARING_CATALOG["SKF6205"], spec.shaft_hz)
    bearing_hz = {k: v for k, v in ff.as_dict().items() if k != "shaft_hz"}
    freqs = mcsa_frequencies(LINE_HZ, POLE_PAIRS, shaft_hz=spec.shaft_hz,
                             bearing_hz=bearing_hz, rotor_slots=28,
                             load_osc_hz=2.0 * SLIP_RATED * LINE_HZ)
    out = {
        "machine": "motor_pump_37kw",
        "line_hz": LINE_HZ,
        "pole_pairs": POLE_PAIRS,
        "slip_rated": SLIP_RATED,
        "frequencies": as_json_dict(freqs),
        "leakage": leakage_table(),
        "resolution": resolution_table(),
        "coupling": coupling_table(),
        "negative_sequence": negative_sequence_table(),
        "windows_available": list(WINDOW_NAMES),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote", args.out)
    print("\nleakage at the 2sf1 offset:")
    for r in out["leakage"]["rows"]:
        print("  %-18s 1 s: %7.1f dB   10 s: %7.1f dB" % (
            r["window"], r["leak_db@1s_record_2sf1"], r["leak_db@10s_record_2sf1"]))
    print("\ncoupling, severity 1.0, per sideband:")
    for r in out["coupling"]["modes"]:
        print("  %-16s %7.1f Hz  H=%.3f  torque %7.1f dB  airgap %7.1f dB  -> %7.1f dB (%s)"
              % (r["mode"], r["modulation_hz"], r["h_mech"], r["sideband_torque_db"],
                 r["sideband_airgap_db"], r["sideband_total_db"], r["dominant_path"]))
    print("\nrotor bar closed form:")
    for r in out["coupling"]["rotor_bar_closed_form"]:
        print("  %d of %d bars: %.1f dB full load, %.1f dB half, %.1f dB quarter" % (
            r["n_broken"], r["of_bars"], r["lsb_db_full_load"],
            r["lsb_db_half_load"], r["lsb_db_quarter_load"]))
    print("\nmech corner %.1f Hz" % out["coupling"]["operating_point"]["mech_corner_hz"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
