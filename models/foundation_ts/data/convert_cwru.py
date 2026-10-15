#!/usr/bin/env python3
"""Convert downloaded CWRU 12k drive-end .mat files into one plain .npz + JSON manifest.

Usage: python -I convert_cwru.py <raw_dir> <out_dir>

No resampling / filtering / normalisation / truncation. Raw stored values, flattened
to 1-D, cast to float32.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Verified metadata.
#
# File-number -> condition mapping was scraped from the official CWRU tables:
#   https://engineering.case.edu/bearingdatacenter/normal-baseline-data
#   https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
# Nominal motor speeds (1797/1772/1750/1730 rpm at 0/1/2/3 hp) are printed in the
# "Approx. Motor Speed (rpm)" column of both tables.
# fs: apparatus page states "Digital data was collected at 12,000 samples per second,
# and data was also collected at 48,000 samples per second for drive end bearing
# faults."  All fault files below come from the *12k* Drive End table.
# ---------------------------------------------------------------------------
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# num: (family, label, fault_size_in, load_hp)
FILES = {
    97:  ("normal",    "normal",      None,  0),
    98:  ("normal",    "normal",      None,  1),
    99:  ("normal",    "normal",      None,  2),
    100: ("normal",    "normal",      None,  3),

    105: ("IR007",     "inner_race",  0.007, 0),
    106: ("IR007",     "inner_race",  0.007, 1),
    107: ("IR007",     "inner_race",  0.007, 2),
    108: ("IR007",     "inner_race",  0.007, 3),

    118: ("B007",      "ball",        0.007, 0),
    119: ("B007",      "ball",        0.007, 1),
    120: ("B007",      "ball",        0.007, 2),
    121: ("B007",      "ball",        0.007, 3),

    130: ("OR007at6",  "outer_race",  0.007, 0),
    131: ("OR007at6",  "outer_race",  0.007, 1),
    132: ("OR007at6",  "outer_race",  0.007, 2),
    133: ("OR007at6",  "outer_race",  0.007, 3),

    169: ("IR014",     "inner_race",  0.014, 0),
    170: ("IR014",     "inner_race",  0.014, 1),
    171: ("IR014",     "inner_race",  0.014, 2),
    172: ("IR014",     "inner_race",  0.014, 3),

    185: ("B014",      "ball",        0.014, 0),
    186: ("B014",      "ball",        0.014, 1),
    187: ("B014",      "ball",        0.014, 2),
    188: ("B014",      "ball",        0.014, 3),

    197: ("OR014at6",  "outer_race",  0.014, 0),
    198: ("OR014at6",  "outer_race",  0.014, 1),
    199: ("OR014at6",  "outer_race",  0.014, 2),
    200: ("OR014at6",  "outer_race",  0.014, 3),
}

BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Verified from https://engineering.case.edu/bearingdatacenter/bearing-information
# n_balls / contact_angle are not printed on that page, but are confirmed exactly by
# the published defect-frequency multipliers:
#   6205 BPFO 3.5848 == (9/2)*(1 - 0.3126/1.537)   -> 9 balls, contact angle 0
#   6203 BPFO 3.0530 == (8/2)*(1 - 0.2656/1.122)   -> 8 balls, contact angle 0
GEOM_DE = {
    "bearing": "SKF 6205-2RS JEM (deep groove ball bearing)",
    "n_balls": 9,
    "ball_dia_in": 0.3126,
    "pitch_dia_in": 1.537,
    "contact_angle_deg": 0,
    "inside_dia_in": 0.9843,
    "outside_dia_in": 2.0472,
    "thickness_in": 0.5906,
    "defect_freq_orders": {"inner_ring": 5.4152, "outer_ring": 3.5848,
                           "cage_train": 0.39828, "rolling_element": 4.7135},
}
GEOM_FE = {
    "bearing": "SKF 6203-2RS JEM (deep groove ball bearing)",
    "n_balls": 8,
    "ball_dia_in": 0.2656,
    "pitch_dia_in": 1.122,
    "contact_angle_deg": 0,
    "inside_dia_in": 0.6693,
    "outside_dia_in": 1.5748,
    "thickness_in": 0.4724,
    "defect_freq_orders": {"inner_ring": 4.9469, "outer_ring": 3.0530,
                           "cage_train": 0.3817, "rolling_element": 3.9874},
}

CHANNELS = [("DE", "drive_end_accelerometer", GEOM_DE),
            ("FE", "fan_end_accelerometer", GEOM_FE),
            ("BA", "base_accelerometer", None)]

NORMAL_FS_NOTE = (
    "fs_hz=12000 is taken from the CWRU Apparatus/Procedures page ('Digital data was "
    "collected at 12,000 samples per second'). The Normal Baseline Data page itself "
    "does not state a rate for files 97-100, and parts of the literature assume 48 kHz "
    "for the normal baseline. Treat fs for the NORMAL files as documented-but-not-"
    "explicitly-stated-per-file."
)
OR_POS_NOTE = (
    "Outer-race fault position is '@6:00' per the 12k Drive End data table, which labels "
    "@6:00 'Centered' with 'Load Zone Centered at 6:00'. The Apparatus/Procedures page "
    "instead calls 3 o'clock 'directly in the load zone' and 6 o'clock 'orthogonal to the "
    "load zone'. This inconsistency is in the CWRU source documentation, not introduced here."
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    arrays = {}
    recordings = {}
    source_urls = {}
    sha = {}
    file_bytes = {}
    skipped = []

    for num in sorted(FILES):
        fn = f"{num}.mat"
        path = os.path.join(raw_dir, fn)
        if not os.path.exists(path):
            skipped.append({"file": fn, "reason": "not downloaded"})
            continue
        family, label, size_in, load = FILES[num]
        source_urls[fn] = BASE_URL + fn
        sha[fn] = sha256_file(path)
        file_bytes[fn] = os.path.getsize(path)

        mat = scipy.io.loadmat(path)
        present = sorted(k for k in mat if not k.startswith("__"))

        for ch, ch_desc, geom in CHANNELS:
            var = f"X{num:03d}_{ch}_time"
            if var not in mat:
                continue
            sig = np.asarray(mat[var]).reshape(-1).astype(np.float32)
            key = f"cwru_{ch}_{family}_{load}hp"
            if key in arrays:
                raise SystemExit(f"duplicate npz key {key}")
            arrays[key] = sig

            notes = []
            if label == "normal":
                notes.append(NORMAL_FS_NOTE)
            if label == "outer_race":
                notes.append(OR_POS_NOTE)
            if ch == "BA":
                notes.append("Base accelerometer mounted on the motor housing base "
                             "plate; not attached to a specific bearing, so "
                             "bearing_geometry is null.")

            # Per-file measured shaft speed, when the .mat carries an X###RPM variable.
            rpm_var = f"X{num:03d}RPM"
            rpm_from_mat = int(np.asarray(mat[rpm_var]).ravel()[0]) if rpm_var in mat else None
            if rpm_from_mat is None:
                notes.append(f"No {rpm_var} variable in {fn}; rpm is the nominal table "
                             f"value only, no per-file measured speed available.")

            extra = [v for v in present
                     if v.endswith("_time") and not v.startswith(f"X{num:03d}")]
            if extra:
                notes.append(f"{fn} also contains unrelated variables {extra} (a packaging "
                             f"quirk of the CWRU archive); only {var} was used for this key.")

            recordings[key] = {
                "file": fn,
                "mat_var": var,
                "fs_hz": 12000,
                "n_samples": int(sig.size),
                "label": label,
                "fault_size_in": size_in,
                "load_hp": load,
                "rpm": NOMINAL_RPM[load],
                "bearing_geometry": geom,
                # --- extra provenance, all verified ---
                "channel": ch,
                "channel_desc": ch_desc,
                "rpm_source": "nominal, 'Approx. Motor Speed (rpm)' column of the CWRU data tables",
                "rpm_from_mat": rpm_from_mat,
                "fault_size_unit": "inch (diameter of seeded EDM fault)",
                "outer_race_position": "@6:00" if label == "outer_race" else None,
                "mat_vars_present": present,
                "notes": " ".join(notes) if notes else None,
            }

    manifest = {
        "dataset": "CWRU Bearing Data Center - 12k Drive End Bearing Fault Data + Normal Baseline",
        "dataset_home": "https://engineering.case.edu/bearingdatacenter",
        "doc_pages_used": [
            "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
            "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
            "https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures",
            "https://engineering.case.edu/bearingdatacenter/bearing-information",
        ],
        "source_base_url": BASE_URL,
        "source_urls": source_urls,
        "sha256": sha,
        "file_bytes": file_bytes,
        "total_downloaded_bytes": sum(file_bytes.values()),
        "download_utc": DOWNLOAD_UTC,
        "converted_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "npz_file": "cwru_12kDE.npz",
        "n_recordings": len(recordings),
        "dtype": "float32",
        "units": "raw stored accelerometer values as distributed by CWRU (g); no "
                 "resampling, filtering, normalisation or truncation applied",
        "subset_rationale": "Full 4-load x 3-fault-type factorial at 0.007\" and 0.014\" "
                            "fault diameters, plus all four normal baselines. 0.021\"/0.028\" "
                            "and the @3:00/@12:00 outer-race positions were not downloaded to "
                            "stay inside the size budget.",
        "skipped_files": skipped,
        "notes": OR_POS_NOTE + " " + NORMAL_FS_NOTE,
        "recordings": recordings,
    }

    npz_path = os.path.join(out_dir, "cwru_12kDE.npz")
    np.savez(npz_path, **arrays)
    man_path = os.path.join(out_dir, "cwru_12kDE.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    print(f"wrote {npz_path} ({os.path.getsize(npz_path)} bytes, {len(arrays)} keys)")
    print(f"wrote {man_path}")
    print(f"downloaded bytes: {manifest['total_downloaded_bytes']}")
    if skipped:
        print("skipped:", skipped)


DOWNLOAD_UTC = None
if __name__ == "__main__":
    # download timestamp comes from the raw dir's download log
    rd = sys.argv[1]
    logp = os.path.join(rd, "_download_log.json")
    try:
        DOWNLOAD_UTC = datetime.datetime.fromtimestamp(
            os.path.getmtime(logp), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        DOWNLOAD_UTC = None
    main()
