#!/usr/bin/env python3
"""Verify the downloaded datasets before any model touches them.

    python -I prototype/supervised_cnn/verify_data.py --cwru <dir> [--mfpt <dir>]

Three checks, all of which can fail:

1. **Acquisition rate per record**, measured from the shaft-order family rather
   than assumed. This is how the 48 kHz normal baseline was found.
2. **Fault location vs bearing kinematics**, using ``pdmcore.dsp.bearing``. A
   transposed row in the record table fails here.
3. **sha256 per file**, so every result JSON can be tied to the exact bytes.

Writes ``results/data_verification.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

import cwru  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwru", required=True, help="directory holding the CWRU 12k drive-end .mat files")
    ap.add_argument("--mfpt", default=None, help="directory holding the MFPT .mat files")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "data_verification.json"))
    args = ap.parse_args(argv)

    out = {
        "host": {"machine": platform.machine(), "python": platform.python_version(),
                 "numpy": np.__version__},
        "cwru_root": os.path.abspath(args.cwru),
    }

    print("1. measuring the acquisition rate of each record ...", flush=True)
    rates = cwru.verify_sample_rates(args.cwru)
    agree = sum(1 for v in rates.values() if v["measured_fs_hz"] == v["declared_fs_hz"])
    out["sample_rate"] = rates
    out["sample_rate_summary"] = {
        "n_records": len(rates),
        "n_agree_with_declared": agree,
        "normal_records": {k: rates[k]["measured_fs_hz"] for k in ("97", "98", "99", "100")},
    }
    for stem in ("97", "98", "99", "100", "105", "130", "222"):
        r = rates[stem]
        print(f"   {stem:>4}  rpm={r['rpm']:7.1f}  order-family score: "
              f"12k={r['score_12k']:8.2f}  48k={r['score_48k']:8.2f}  "
              f"-> {r['measured_fs_hz']:.0f} Hz (declared {r['declared_fs_hz']:.0f})")
    print(f"   {agree}/{len(rates)} records agree with the declared rate")

    print("2. checking the record table against bearing kinematics ...", flush=True)
    labels = cwru.verify_fault_labels(args.cwru)
    out["fault_labels"] = labels
    print(f"   per-record agreement: {labels['n_agree']}/{labels['n_records']}")
    for loc, frac in sorted(labels["agreement_by_location"].items()):
        print(f"     {loc:<6} {frac * 100:5.1f} %")
    print("   permutation test on the table (gate): "
          + "  ".join(f"{k}={v:.4f}" for k, v in labels["permutation_scores"].items())
          + f"  -> best={labels['permutation_best']}  ok={labels['ok']}")
    if not labels["ok"]:
        raise SystemExit(
            "the declared location assignment does not beat a cyclic relabelling: "
            "the record table in prototype/supervised_cnn/cwru.py is wrong"
        )

    print("3. digests ...", flush=True)
    out["sha256"] = cwru.file_digests(args.cwru)

    if args.mfpt:
        import mfpt  # noqa: E402
        print("4. MFPT ...", flush=True)
        out["mfpt"] = mfpt.verify(args.mfpt)
        print(json.dumps(out["mfpt"]["summary"], indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
