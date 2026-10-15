#!/usr/bin/env python3
"""Verify a converted .npz against its .json manifest.

Usage: python -I verify_npz.py <npz_path> <json_path>
Prints key -> (n_samples, dtype, min, max, std) and cross-checks the manifest.
"""
import json
import sys

import numpy as np

npz_path, man_path = sys.argv[1], sys.argv[2]
z = np.load(npz_path)
man = json.load(open(man_path))
recs = man["recordings"]

keys = list(z.files)
print(f"npz: {npz_path}")
print(f"npz keys: {len(keys)}   manifest recordings: {len(recs)}   "
      f"manifest n_recordings field: {man.get('n_recordings')}")

problems = []
if set(keys) != set(recs):
    problems.append(f"KEY SET MISMATCH: only-in-npz={sorted(set(keys)-set(recs))} "
                    f"only-in-manifest={sorted(set(recs)-set(keys))}")

print()
hdr = f"{'key':34s} {'n_samples':>10s} {'dtype':>8s} {'min':>12s} {'max':>12s} {'std':>12s}  fs  label"
print(hdr)
print("-" * len(hdr))
label_counts = {}
fs_counts = {}
total = 0
for k in sorted(keys):
    a = z[k]
    r = recs.get(k, {})
    total += a.size
    label_counts[r.get("label")] = label_counts.get(r.get("label"), 0) + 1
    fs_counts[r.get("fs_hz")] = fs_counts.get(r.get("fs_hz"), 0) + 1
    print(f"{k:34s} {a.size:10d} {str(a.dtype):>8s} {a.min():12.5f} {a.max():12.5f} "
          f"{a.std():12.5f}  {r.get('fs_hz')}  {r.get('label')}")
    if a.ndim != 1:
        problems.append(f"{k}: ndim={a.ndim}, expected 1-D")
    if a.dtype != np.float32:
        problems.append(f"{k}: dtype={a.dtype}, expected float32")
    if r.get("n_samples") != int(a.size):
        problems.append(f"{k}: manifest n_samples={r.get('n_samples')} but array has {a.size}")
    if not np.all(np.isfinite(a)):
        problems.append(f"{k}: contains non-finite values")
    if a.size == 0:
        problems.append(f"{k}: empty array")

print()
print("recordings per label:", dict(sorted(label_counts.items(), key=lambda x: str(x[0]))))
print("recordings per fs_hz:", dict(sorted(fs_counts.items(), key=lambda x: str(x[0]))))
print("total samples:", total)
print("npz bytes on disk:", __import__("os").path.getsize(npz_path))

# manifest field completeness
required = ["file", "mat_var", "fs_hz", "n_samples", "label", "fault_size_in",
            "load_hp", "rpm", "bearing_geometry"]
for k, r in recs.items():
    missing = [f for f in required if f not in r]
    if missing:
        problems.append(f"{k}: manifest missing fields {missing}")
    if r.get("label") not in ("normal", "inner_race", "ball", "outer_race"):
        problems.append(f"{k}: unexpected label {r.get('label')!r}")
for f in ["source_urls", "sha256", "download_utc", "recordings"]:
    if f not in man:
        problems.append(f"manifest missing top-level field {f}")

print()
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("OK: all keys 1-D float32, finite, counts match manifest, required fields present.")
