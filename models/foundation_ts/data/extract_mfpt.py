#!/usr/bin/env python3
"""Safely extract only .mat/.md/.txt members of the MFPT zip, then report struct fields.

Usage: python -I extract_mfpt.py <zip_path> <extract_dir>
"""
import os
import posixpath
import sys
import zipfile

import scipy.io

ALLOW = (".mat", ".md", ".txt")


def safe_extract(zip_path, dest):
    os.makedirs(dest, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or not name.lower().endswith(ALLOW):
                continue
            # flatten to <lastdir>__<basename>, refusing anything weird
            norm = posixpath.normpath(name)
            if norm.startswith(("/", "..")) or ".." in norm.split("/"):
                print("REFUSED unsafe path:", name)
                continue
            parts = norm.split("/")
            base = parts[-1]
            sub = parts[-2] if len(parts) >= 2 and parts[-2] not in (
                "RollingElementBearingFaultDiagnosis-Data-master",) else ""
            flat = f"{sub}__{base}" if sub else base
            target = os.path.join(dest, flat)
            with z.open(info) as src, open(target, "wb") as dst:
                while True:
                    c = src.read(1 << 20)
                    if not c:
                        break
                    dst.write(c)
            os.chmod(target, 0o644)
            out.append((flat, info.file_size, norm))
    return out


def describe(v, depth=0):
    import numpy as np
    if isinstance(v, np.ndarray) and v.dtype.names:
        bits = []
        for n in v.dtype.names:
            try:
                sub = v[n]
                while isinstance(sub, np.ndarray) and sub.dtype == object and sub.size == 1:
                    sub = sub.item()
                if isinstance(sub, np.ndarray) and sub.dtype.names:
                    bits.append(f"{n}={{{describe(sub, depth+1)}}}")
                elif isinstance(sub, np.ndarray) and sub.size == 1:
                    bits.append(f"{n}={sub.ravel()[0]!r}")
                elif isinstance(sub, np.ndarray):
                    bits.append(f"{n}=array{tuple(sub.shape)}:{sub.dtype}")
                else:
                    bits.append(f"{n}={sub!r}")
            except Exception as e:
                bits.append(f"{n}=<{e}>")
        return ", ".join(bits)
    return repr(v)


def main():
    zip_path, dest = sys.argv[1], sys.argv[2]
    got = safe_extract(zip_path, dest)
    print(f"extracted {len(got)} members")
    for flat, size, orig in got:
        print(f"  {size}\t{flat}\t<- {orig}")
    print()
    for flat, _, _ in sorted(got):
        if not flat.endswith(".mat"):
            continue
        m = scipy.io.loadmat(os.path.join(dest, flat))
        for k, v in m.items():
            if k.startswith("__"):
                continue
            print(f"{flat}\t{k}\t{describe(v)}")


if __name__ == "__main__":
    main()
