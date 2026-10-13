#!/usr/bin/env python3
"""Cross-check this kit's app_config.h against the Python reference.

    python3 hardware/<kit>/tools/check_config.py [--define PDM_SENSOR_IIS3DWB]

Run from the build before every firmware build. A firmware whose n_fft, sample
rate, bearing geometry or demodulation band disagrees with the model it is
running produces confident, wrong diagnoses, and nothing in the output reveals
it -- the numbers are all plausible. This is the cheapest place to catch that.

It **compiles** the header rather than text-parsing it. Text parsing looked
adequate until a kit shipped two `#ifdef` sensor profiles defining the same
macro names: the parser silently reported whichever profile appeared last in the
file, which on the ESP32-S3 branch meant validating the configuration the
firmware was not built with. Preprocessing also proves the header compiles,
which a regex cannot.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.dsp.bearing import BEARING_CATALOG, fault_frequencies  # noqa: E402

#: Numeric macros the checks below need. Printed by a generated C program so the
#: values are exactly what the compiler sees, expressions and all.
WANTED = [
    ("APP_FEATURE_SPEC", "d"),
    ("APP_FS_HZ", "g"),
    ("APP_N_FFT", "d"),
    ("APP_ACQ_SAMPLES", "d"),
    ("APP_ENVELOPE_N_FFT", "d"),
    ("APP_ENVELOPE_DECIMATE", "d"),
    ("APP_ENV_SAMPLES", "d"),
    ("APP_ENVELOPE_LP_HZ", "g"),
    ("APP_DEMOD_LO_HZ", "g"),
    ("APP_DEMOD_HI_HZ", "g"),
    ("APP_VELOCITY_LO_HZ", "g"),
    ("APP_VELOCITY_HI_HZ", "g"),
    ("APP_BEARING_N_BALLS", "g"),
    ("APP_BEARING_BALL_D", "g"),
    ("APP_BEARING_PITCH_D", "g"),
    ("APP_BEARING_HARMONICS", "d"),
    ("APP_CLIP_MARGIN", "g"),
    ("APP_BATT_LOW_MV", "d"),
    ("APP_BATT_CRITICAL_MV", "d"),
    ("APP_DEFAULT_INTERVAL_S", "d"),
    ("APP_QUEUE_DEPTH", "d"),
    ("APP_SENSOR_SETTLE_MS", "d"),
    ("APP_SENSOR_KEEPALIVE_THRESHOLD_S", "d"),
    ("APP_FULL_SCALE_G", "g"),
]


def read_config(cfg_path: str, defines) -> dict:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise SystemExit("no C compiler found; cannot evaluate app_config.h")
    lines = ['#include "app_config.h"', "#include <stdio.h>", "int main(void){"]
    for name, fmt in WANTED:
        lines.append(f'    printf("{name}\\t%{fmt}\\n", ({"double" if fmt == "g" else "long"})'
                     f'({name}));')
    lines.append("    return 0;\n}")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "probe.c")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        exe = os.path.join(tmp, "probe")
        cmd = [cc, "-std=c99", "-I" + os.path.dirname(cfg_path),
               "-I" + os.path.join(_REPO, "cdsp", "include")]
        for d in defines or []:
            cmd.append("-D" + d)
        cmd += [src, "-o", exe]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"app_config.h does not compile:\n{proc.stderr}")
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit(f"probe failed: {run.stderr}")
    out = {}
    for line in run.stdout.splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            out[k] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--define", action="append", default=[],
                    help="preprocessor define, e.g. PDM_SENSOR_IIS3DWB")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(_HERE, "..", "firmware", "app_config.h")
    cfg_path = os.path.abspath(cfg_path)
    d = read_config(cfg_path, args.define)
    problems, notes = [], []

    if int(d["APP_FEATURE_SPEC"]) != FEATURE_SPEC_VERSION:
        problems.append(f"APP_FEATURE_SPEC is {int(d['APP_FEATURE_SPEC'])} but pdmcore is "
                        f"at {FEATURE_SPEC_VERSION}: the cloud will reject every frame")

    n_fft = int(d["APP_N_FFT"])
    if n_fft & (n_fft - 1):
        problems.append(f"APP_N_FFT {n_fft} is not a power of two")

    acq, env_n = int(d["APP_ACQ_SAMPLES"]), int(d["APP_ENVELOPE_N_FFT"])
    dec = int(d["APP_ENVELOPE_DECIMATE"])
    if int(d["APP_ENV_SAMPLES"]) != env_n * dec:
        problems.append("APP_ENV_SAMPLES != APP_ENVELOPE_N_FFT * APP_ENVELOPE_DECIMATE")
    if env_n * dec > acq:
        problems.append(f"APP_ACQ_SAMPLES {acq} is too short for a {env_n}-point envelope "
                        f"FFT decimated by {dec} (needs {env_n * dec}): the envelope stage "
                        f"will silently never run and the bearing block will read as "
                        f"'no fault'")
    if acq < n_fft:
        problems.append(f"APP_ACQ_SAMPLES {acq} < APP_N_FFT {n_fft}")

    fs = d["APP_FS_HZ"]
    fs_env = fs / dec
    if d["APP_DEMOD_HI_HZ"] >= fs / 2:
        problems.append(f"demodulation band tops out at {d['APP_DEMOD_HI_HZ']:.0f} Hz, at or "
                        f"above Nyquist ({fs / 2:.0f} Hz)")
    if d["APP_ENVELOPE_LP_HZ"] >= fs_env / 2:
        problems.append(f"APP_ENVELOPE_LP_HZ {d['APP_ENVELOPE_LP_HZ']:.0f} is at or above "
                        f"the decimated Nyquist ({fs_env / 2:.0f} Hz): the envelope aliases")

    geom = BEARING_CATALOG["SKF6205"]
    for name, key in (("n_balls", "APP_BEARING_N_BALLS"), ("ball_d", "APP_BEARING_BALL_D"),
                      ("pitch_d", "APP_BEARING_PITCH_D")):
        if abs(d[key] - float(getattr(geom, name))) > 1e-4:
            problems.append(f"{key} is {d[key]} but SKF6205 {name} is {getattr(geom, name)}")

    shaft = 29.6667
    ff = fault_frequencies(geom, shaft)
    df_env = fs_env / env_n
    harmonics = int(d["APP_BEARING_HARMONICS"])
    floor = 1.5 * df_env / ff.bpfo_hz
    if floor > 0.06:
        problems.append(f"envelope resolution {df_env:.2f} Hz gives a tolerance floor of "
                        f"+-{floor * 100:.1f} % at BPFO {ff.bpfo_hz:.1f} Hz -- wide enough to "
                        f"match a neighbouring family. Lengthen APP_ACQ_SAMPLES.")
    elif floor > 0.03:
        notes.append(f"envelope tolerance floor is +-{floor * 100:.1f} % at BPFO; acceptable "
                     f"but the +-2 % slip term is no longer dominant")
    if ff.bpfo_hz * harmonics > fs_env / 2:
        problems.append(f"{harmonics} BPFO harmonics reach {ff.bpfo_hz * harmonics:.0f} Hz, "
                        f"above the decimated Nyquist {fs_env / 2:.0f} Hz: the top harmonics "
                        f"will alias")
    # Does the demodulation band plausibly contain a structural resonance?
    if d["APP_DEMOD_HI_HZ"] < 2000.0:
        notes.append(f"the demodulation band ({d['APP_DEMOD_LO_HZ']:.0f}-"
                     f"{d['APP_DEMOD_HI_HZ']:.0f} Hz) is below the 2-20 kHz range where "
                     f"bearing impacts excite structural resonances. Bearing evidence from "
                     f"this configuration is weak; treat a negative result as 'not measured'")

    if shaft < d["APP_VELOCITY_LO_HZ"]:
        problems.append(f"the 1x line at {shaft:.1f} Hz is below APP_VELOCITY_LO_HZ "
                        f"{d['APP_VELOCITY_LO_HZ']:.0f} Hz: the ISO velocity reading misses it")
    if d["APP_BATT_CRITICAL_MV"] >= d["APP_BATT_LOW_MV"]:
        problems.append("APP_BATT_CRITICAL_MV must be below APP_BATT_LOW_MV")
    queue_days = d["APP_QUEUE_DEPTH"] * d["APP_DEFAULT_INTERVAL_S"] / 86400.0
    if queue_days < 7.0:
        problems.append(f"the queue covers only {queue_days:.1f} days at the default interval")

    print(f"{os.path.relpath(cfg_path, _REPO)}"
          + (f"  [{' '.join(args.define)}]" if args.define else ""))
    print(f"  fs {fs:.0f} Hz, n_fft {n_fft} ({fs / n_fft:.2f} Hz), "
          f"acq {acq} ({acq / fs * 1000:.0f} ms)")
    print(f"  envelope: decimate {dec} -> {fs_env:.0f} Hz, {env_n} pts ({df_env:.2f} Hz), "
          f"band {d['APP_DEMOD_LO_HZ']:.0f}-{d['APP_DEMOD_HI_HZ']:.0f} Hz")
    print(f"  bearing SKF6205 at {shaft:.2f} Hz: BPFO {ff.bpfo_hz:.1f}, BPFI {ff.bpfi_hz:.1f}, "
          f"BSF {ff.bsf_hz:.1f}, FTF {ff.ftf_hz:.2f} Hz")
    print(f"  queue {int(d['APP_QUEUE_DEPTH'])} frames at {int(d['APP_DEFAULT_INTERVAL_S'])} s "
          f"= {queue_days:.0f} days")
    for n in notes:
        print(f"  NOTE  {n}")
    for p in problems:
        print(f"  FAIL  {p}")
    if problems:
        print(f"\n{len(problems)} problem(s). Fix before building.")
        return 1
    print("\nconfig is consistent with the Python reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
