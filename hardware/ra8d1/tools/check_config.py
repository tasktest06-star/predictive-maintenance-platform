#!/usr/bin/env python3
"""Cross-check this kit's app_config.h against the Python reference.

    python3 hardware/stwinbx1/tools/check_config.py

Run from the Makefile before every firmware build. A firmware whose n_fft,
sample rate, bearing geometry or demodulation band disagrees with the model it
is running produces confident, wrong diagnoses, and nothing in the output
reveals it -- the numbers are all plausible. This is the cheapest possible place
to catch that.
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore import FEATURE_SPEC_VERSION  # noqa: E402
from pdmcore.dsp.bearing import BEARING_CATALOG, fault_frequencies  # noqa: E402


def parse_defines(path: str) -> dict:
    out = {}
    pattern = re.compile(r"^\s*#define\s+(APP_\w+)\s+(.+?)\s*(?:/\*.*)?$")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def as_float(raw: str) -> float:
    return float(raw.rstrip("fu").rstrip("f").rstrip("u"))


def as_int(raw: str) -> int:
    return int(raw.rstrip("u"), 0)


def main() -> int:
    cfg_path = os.path.join(_HERE, "..", "firmware", "app_config.h")
    d = parse_defines(cfg_path)
    problems, notes = [], []

    if as_int(d["APP_FEATURE_SPEC"]) != FEATURE_SPEC_VERSION:
        problems.append(
            f"APP_FEATURE_SPEC is {d['APP_FEATURE_SPEC']} but pdmcore is at "
            f"{FEATURE_SPEC_VERSION}: the cloud will reject every frame")

    n_fft = as_int(d["APP_N_FFT"])
    if n_fft & (n_fft - 1):
        problems.append(f"APP_N_FFT {n_fft} is not a power of two")

    acq = as_int(d["APP_ACQ_SAMPLES"])
    env_n = as_int(d["APP_ENVELOPE_N_FFT"])
    dec = as_int(d["APP_ENVELOPE_DECIMATE"])
    if env_n * dec > acq:
        problems.append(
            f"APP_ACQ_SAMPLES {acq} is too short for a {env_n}-point envelope FFT "
            f"decimated by {dec} (needs {env_n * dec}): the envelope stage will "
            f"silently never run and the bearing block will read as 'no fault'")
    if acq < n_fft:
        problems.append(f"APP_ACQ_SAMPLES {acq} < APP_N_FFT {n_fft}")

    fs = as_float(d["APP_FS_HZ"])
    demod_lo = as_float(d["APP_DEMOD_LO_HZ"])
    demod_hi = as_float(d["APP_DEMOD_HI_HZ"])
    if demod_hi >= fs / 2.0:
        problems.append(f"demodulation band tops out at {demod_hi} Hz, at or above "
                        f"Nyquist ({fs / 2:.0f} Hz)")
    env_lp = as_float(d["APP_ENVELOPE_LP_HZ"])
    fs_env = fs / dec
    if env_lp >= fs_env / 2.0:
        problems.append(f"APP_ENVELOPE_LP_HZ {env_lp} is at or above the decimated "
                        f"Nyquist ({fs_env / 2:.0f} Hz): the envelope will alias")

    geom = BEARING_CATALOG["SKF6205"]
    for name, key in (("n_balls", "APP_BEARING_N_BALLS"),
                      ("ball_d", "APP_BEARING_BALL_D"),
                      ("pitch_d", "APP_BEARING_PITCH_D")):
        want = float(getattr(geom, name))
        got = as_float(d[key])
        if abs(got - want) > 1e-4:
            problems.append(f"{key} is {got} but SKF6205 {name} is {want}")

    # The envelope resolution must actually resolve the fault frequencies.
    shaft = 29.6667
    ff = fault_frequencies(geom, shaft)
    df_env = fs_env / env_n
    harmonics = as_int(d["APP_BEARING_HARMONICS"])
    rel_tol_floor = 1.5 * df_env / ff.bpfo_hz
    if rel_tol_floor > 0.06:
        problems.append(
            f"envelope resolution {df_env:.2f} Hz gives a tolerance floor of "
            f"+-{rel_tol_floor * 100:.1f} % at BPFO {ff.bpfo_hz:.1f} Hz -- wide enough "
            f"to match a neighbouring family. Lengthen APP_ACQ_SAMPLES.")
    elif rel_tol_floor > 0.03:
        notes.append(f"envelope tolerance floor is +-{rel_tol_floor * 100:.1f} % at BPFO; "
                     f"acceptable but the +-2 % slip term is no longer dominant")
    if ff.bpfo_hz * harmonics > fs_env / 2.0:
        problems.append(
            f"{harmonics} BPFO harmonics reach {ff.bpfo_hz * harmonics:.0f} Hz, above the "
            f"decimated Nyquist {fs_env / 2:.0f} Hz: the top harmonics will alias")

    vel_lo = as_float(d["APP_VELOCITY_LO_HZ"])
    if shaft < vel_lo:
        problems.append(
            f"the 1x line at {shaft:.1f} Hz is below APP_VELOCITY_LO_HZ {vel_lo:.0f} Hz: "
            f"the ISO velocity reading will miss it entirely and read low")

    print(f"app_config.h: {len(d)} definitions")
    print(f"  fs {fs:.0f} Hz, n_fft {n_fft} ({fs / n_fft:.2f} Hz), "
          f"acq {acq} ({acq / fs * 1000:.0f} ms)")
    print(f"  envelope: decimate {dec} -> {fs_env:.0f} Hz, {env_n} pts "
          f"({df_env:.2f} Hz), band {demod_lo:.0f}-{demod_hi:.0f} Hz")
    print(f"  bearing SKF6205 at {shaft:.2f} Hz: BPFO {ff.bpfo_hz:.1f}, "
          f"BPFI {ff.bpfi_hz:.1f}, BSF {ff.bsf_hz:.1f}, FTF {ff.ftf_hz:.2f} Hz")
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
