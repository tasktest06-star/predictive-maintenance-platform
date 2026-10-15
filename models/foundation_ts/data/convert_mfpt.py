#!/usr/bin/env python3
"""Convert extracted MFPT .mat files into one plain .npz + JSON manifest.

Usage: python -I convert_mfpt.py <extract_dir> <zip_path> <out_dir>

No resampling / filtering / normalisation / truncation. Raw stored values, flattened
to 1-D, cast to float32.
"""
import datetime
import hashlib
import json
import os
import re
import sys

import numpy as np
import scipy.io

ZIP_URL = ("https://codeload.github.com/mathworks/"
           "RollingElementBearingFaultDiagnosis-Data/zip/refs/heads/master")
REPO_URL = "https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data"

# MFPT bearing (NICE bearing used on the MFPT test rig).
# n_balls and ball/pitch diameter are confirmed against the defect frequencies stored
# inside every .mat (BPFO 81.125 Hz, BPFI 118.875 Hz at a 25 Hz shaft rate):
#   BPFI/rate + BPFO/rate = n            -> 4.755 + 3.245 = 8.000  (n_balls = 8)
#   (BPFI - BPFO)/(rate*n) = d/D         -> 1.510/8    = 0.188750
#   0.235 / 1.245                        =              0.188755   (agrees to 5 dp)
#   BSF check: (D/2d)(1-(d/D)^2)*rate    -> 63.86 Hz vs stored 63.91 Hz
GEOM = {
    "bearing": "NICE bearing (MFPT test rig)",
    "n_balls": 8,
    "ball_dia_in": 0.235,
    "pitch_dia_in": 1.245,
    "contact_angle_deg": 0,
    "verification": (
        "n_balls=8 and ball_dia/pitch_dia ratio 0.188755 are reproduced exactly by the "
        "BPFO/BPFI/BSF values stored inside each .mat at the stored 25 Hz shaft rate; "
        "see script comments for the arithmetic."
    ),
}

FTF_NOTE = (
    "The FTF value stored in the .mat files (14.8375 Hz) is NOT consistent with the "
    "standard cage formula for this geometry, which gives (1/2)(1-d/D)*25 Hz = 10.14 Hz. "
    "BPFO, BPFI and BSF are all consistent. The stored FTF is reproduced verbatim in "
    "defect_freq_hz without correction; treat it with suspicion."
)


def classify(base):
    if base.startswith("baseline"):
        return "normal", "baseline"
    if base.startswith("OuterRaceFault_vload"):
        return "outer_race", "outer_vload"
    if base.startswith("OuterRaceFault"):
        return "outer_race", "outer"
    if base.startswith("InnerRaceFault_vload"):
        return "inner_race", "inner_vload"
    raise SystemExit("unclassified file: " + base)


def keyname(base):
    label, fam = classify(base)
    idx = re.search(r"_(\d+)\.mat$", base).group(1)
    if fam == "baseline":
        return f"mfpt_baseline_{idx}"
    if fam == "outer":
        return f"mfpt_outer_270lb_{idx}"
    if fam == "outer_vload":
        return f"mfpt_outer_vload_{idx}"
    return f"mfpt_inner_vload_{idx}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(x):
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.item())
    return a.ravel()[0]


def main():
    ext_dir, zip_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    arrays = {}
    recordings = {}

    mats = sorted(f for f in os.listdir(ext_dir) if f.endswith(".mat"))
    for flat in mats:
        # flat name is "<train_data|test_data>__<original>.mat"
        split, base = flat.split("__", 1)
        label, fam = classify(base)
        key = keyname(base)
        if key in arrays:
            raise SystemExit("duplicate key " + key)

        m = scipy.io.loadmat(os.path.join(ext_dir, flat))
        b = m["bearing"]
        gs = np.asarray(b["gs"][0, 0]).reshape(-1).astype(np.float32)
        fs = int(scalar(b["sr"]))
        rate_hz = float(scalar(b["rate"]))
        load_raw = scalar(b["load"])
        load_lbs = int(str(load_raw)) if isinstance(load_raw, (str, np.str_)) else int(load_raw)

        arrays[key] = gs
        recordings[key] = {
            "file": f"{split}/{base}",
            "mat_var": "bearing.gs",
            "fs_hz": fs,
            "n_samples": int(gs.size),
            "label": label,
            "fault_size_in": None,
            "load_hp": None,
            "rpm": int(round(rate_hz * 60.0)),
            "bearing_geometry": GEOM,
            # --- extra provenance, all read out of the .mat itself ---
            "shaft_rate_hz": rate_hz,
            "radial_load_lbs": load_lbs,
            "mathworks_split": split.replace("_data", ""),
            "fs_source": "bearing.sr field inside the .mat",
            "rpm_source": "bearing.rate field inside the .mat (shaft rate in Hz) x 60",
            "load_source": "bearing.load field inside the .mat",
            "defect_freq_hz": {
                "FTF": float(scalar(m["FTF"])),
                "BPFO": float(scalar(m["BPFO"])),
                "BPFI": float(scalar(m["BPFI"])),
                "BSF": float(scalar(m["BSF"])),
            },
            "notes": (
                "load_hp is null: MFPT documents radial load in POUNDS (see radial_load_lbs), "
                "not motor horsepower. fault_size_in is null: the MFPT distribution does not "
                "state a seeded fault diameter for these records. " + FTF_NOTE
            ),
        }

    by_label = {}
    for r in recordings.values():
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    manifest = {
        "dataset": "MFPT (Machinery Failure Prevention Technology) bearing fault dataset, "
                   "folders 1-4 (three baseline, three outer-race, seven outer-race varying "
                   "load, seven inner-race varying load)",
        "obtained_via": "MathWorks' official redistribution of MFPT folders 1-4: " + REPO_URL,
        "source_urls": {"RollingElementBearingFaultDiagnosis-Data-master.zip": ZIP_URL},
        "sha256": {os.path.basename(zip_path): sha256_file(zip_path)},
        "file_bytes": {os.path.basename(zip_path): os.path.getsize(zip_path)},
        "total_downloaded_bytes": os.path.getsize(zip_path),
        "download_utc": DOWNLOAD_UTC,
        "converted_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "npz_file": "mfpt.npz",
        "n_recordings": len(recordings),
        "recordings_per_label": by_label,
        "dtype": "float32",
        "units": "raw stored acceleration values (g) as distributed; no resampling, "
                 "filtering, normalisation or truncation applied",
        "original_data_source": "http://data-acoustics.com/measurements/bearing-faults/bearing-2/",
        "data_owner": "Eric Bechhoefer",
        "citation": "Bechhoefer, Eric. 'Condition Based Maintenance Fault Database for "
                    "Testing Diagnostics and Prognostic Algorithms.' 2013.",
        "license": "CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/) "
                   "per license.txt and README.md in the source zip",
        "provenance_notes": (
            "mfpt.org now redirects into the ASNT site and its fault-data-sets page offers "
            "only an email contact (datamanagement.ai@mfpt.org), so no direct download from "
            "the originating site was available. The MathWorks repo README states it contains "
            "folders 1-4 of the original dataset, 're-organized and split into train_folder "
            "and test_folder' - i.e. files were regrouped into directories but the .mat "
            "contents are the originals. MFPT folders 5 (Analyses) and 6 (Real World "
            "Examples) are NOT present in this redistribution."
        ),
        "notes": FTF_NOTE,
        "recordings": recordings,
    }

    npz_path = os.path.join(out_dir, "mfpt.npz")
    np.savez(npz_path, **arrays)
    man_path = os.path.join(out_dir, "mfpt.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    print(f"wrote {npz_path} ({os.path.getsize(npz_path)} bytes, {len(arrays)} keys)")
    print(f"wrote {man_path}")
    print("per-label:", by_label)


DOWNLOAD_UTC = None
if __name__ == "__main__":
    try:
        DOWNLOAD_UTC = datetime.datetime.fromtimestamp(
            os.path.getmtime(sys.argv[2]), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        DOWNLOAD_UTC = None
    main()
