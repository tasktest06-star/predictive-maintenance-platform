#!/usr/bin/env python3
"""Inspect CWRU .mat files: list variables and shapes. Usage: python -I inspect_cwru.py <dir>"""
import json
import os
import sys

import scipy.io

d = sys.argv[1]
log = os.path.join(d, "_download_log.json")
sizes = {}
if os.path.exists(log):
    with open(log) as f:
        for line in f:
            for rec in json.loads(line):
                if "bytes" in rec:
                    sizes[rec["num"]] = (rec["bytes"], rec.get("content_length"))

print("== size check (bytes vs Content-Length) ==")
bad = 0
for k in sorted(sizes, key=int):
    b, cl = sizes[k]
    ok = (cl is None) or (int(cl) == b)
    if not ok:
        bad += 1
        print(f"  MISMATCH {k}: got {b} expected {cl}")
print(f"  files logged: {len(sizes)}, mismatches: {bad}")

print("== mat variables ==")
for fn in sorted(os.listdir(d), key=lambda x: (not x[0].isdigit(), x)):
    if not fn.endswith(".mat"):
        continue
    m = scipy.io.loadmat(os.path.join(d, fn))
    items = []
    for k, v in m.items():
        if k.startswith("__"):
            continue
        try:
            items.append(f"{k}{tuple(v.shape)}:{v.dtype}")
        except Exception:
            items.append(f"{k}:{type(v).__name__}")
    print(f"{fn}\t{' '.join(sorted(items))}")
