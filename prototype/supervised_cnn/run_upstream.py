#!/usr/bin/env python3
"""Run ``impl/ml-supervised-1dcnn``'s own PyTorch model under this protocol.

    /workshop/predictive_maintaince/.venv-torch/bin/python -I \
        prototype/supervised_cnn/run_upstream.py --cwru DIR --src DIR

This is the point of the branch that matters most: the number below is not from a
reimplementation, it is from ``BearingFaultCNN1D`` as committed on
``impl/ml-supervised-1dcnn``, with their ``segment_and_fft`` preprocessing, fed
by their ``CWRUBearingDataset``.

What is theirs and what is this branch's, exactly
-------------------------------------------------
Theirs, imported and executed unmodified:
  * ``model.BearingFaultCNN1D``        -- the architecture, 1.35 M parameters
  * ``dataset.segment_and_fft``        -- Hann window, rfft magnitude, zero-pad
                                          to 2048, per-window standardise
  * ``dataset.CWRUBearingDataset``     -- windowing and batching, via its
                                          ``records=`` constructor
  * ``dataset.CWRUBearingDataset.split_by_bearing`` -- run as one of the arms, so
                                          their own split is in the comparison

This branch's:
  * The **record table**: labels and the 12 kHz rate come from
    ``prototype/supervised_cnn/cwru.py``, which is verified against bearing
    kinematics. Their ``_scan_directory`` labels every file "Normal" against CWRU
    as published (see ``audit_upstream.py``), so feeding it a correct record list
    is the only way to evaluate their *model* rather than their directory parser.
  * The **splits**: ``pdmcore.eval.splits``, audited.
  * The **training loop**: a fixed epoch budget, Adam, cosine decay. Their
    ``train.py`` is not used because it selects its checkpoint by the best macro
    F1 on the test loader, which would make every number here a best-of-N over
    epochs on the fold being reported.
  * The **metrics**: ``pdmcore.eval.metrics``.

Cost, and the honest consequence
--------------------------------
Measured on this host: see ``samples_per_second`` in the output. Their model is
1.35 M parameters over a 2048-point input, and their published schedule is 100
epochs with early stopping at patience 15. At the throughput this 2-core
container delivers, that schedule over the full CWRU window set is tens of CPU
hours, so this runs a **reduced, stated** configuration:
``--max-windows-per-record`` windows per record and ``--epochs`` epochs. The
reduction is reported in the JSON next to every number it produced, and the
under-training it causes suppresses the leaky arm more than the clean one --
i.e. it understates this branch's own headline effect rather than inflating it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

import cwru  # noqa: E402
import protocol  # noqa: E402
from pdmcore.eval.splits import Split, leakage_report  # noqa: E402

#: Their four reachable classes, in their index order.
CLASS_NAMES = ("Normal", "BPFO", "BPFI", "BSF")
#: ``cwru.CwruRecord.location`` -> their label index.
LOCATION_TO_INDEX = {"normal": 0, "outer": 1, "inner": 2, "ball": 3}


def host() -> Dict[str, object]:
    info: Dict[str, object] = {
        "machine": platform.machine(), "processor": platform.processor() or "unknown",
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
    }
    try:
        info["loadavg_1_5_15"] = [round(v, 2) for v in os.getloadavg()]
    except OSError:  # pragma: no cover
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "cpu part")):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover
        pass
    return info


def build_their_dataset(up_dataset, cwru_root: str):
    """Their ``CWRUBearingDataset``, fed this branch's verified record table.

    Returns ``(ds, per_window_columns)``. Their windowing produces windows in
    record order, so the per-window metadata columns are rebuilt by replaying the
    same order rather than assumed.
    """
    import pathlib
    recs = []
    for i, rec in enumerate(cwru.CWRU_RECORDS):
        recs.append((pathlib.Path(os.path.join(cwru_root, f"{rec.stem}.mat")),
                     LOCATION_TO_INDEX[rec.location], f"cwru_{i:04d}_{rec.stem}"))
    ds = up_dataset.CWRUBearingDataset(records=recs)
    # ds._samples is [(windows, label, bearing_id)] in the order given; ds._flat
    # is [(sample_idx, window_idx)] in dataset order.
    by_bid = {f"cwru_{i:04d}_{rec.stem}": rec for i, rec in enumerate(cwru.CWRU_RECORDS)}
    cols: Dict[str, List] = {k: [] for k in ("unit", "recording", "segment", "window_id",
                                             "load_hp", "label", "location")}
    n_seg = 6
    per_sample_counts = [w.shape[0] for w, _, _ in ds._samples]
    for s_idx, w_idx in ds._flat:
        _, label, bid = ds._samples[s_idx]
        rec = by_bid[bid]
        n = per_sample_counts[s_idx]
        seg = min(n_seg - 1, (w_idx * n_seg) // max(1, n))
        cols["unit"].append(rec.unit)
        cols["recording"].append(rec.stem)
        cols["segment"].append(f"{rec.stem}#{seg}")
        cols["window_id"].append(f"{rec.stem}@{w_idx}")
        cols["load_hp"].append(rec.load_hp)
        cols["label"].append(int(label))
        cols["location"].append(rec.location)
    return ds, {k: np.asarray(v) for k, v in cols.items()}


def thin(cols: Dict[str, np.ndarray], max_per_record: Optional[int]) -> np.ndarray:
    """Uniformly thin to ``max_per_record`` windows per recording, keeping segments."""
    keep: List[int] = []
    for rec in np.unique(cols["recording"]):
        idx = np.flatnonzero(cols["recording"] == rec)
        if max_per_record is None or idx.size <= max_per_record:
            keep.extend(idx.tolist())
            continue
        segs = cols["segment"][idx]
        per = max(1, max_per_record // np.unique(segs).size)
        for s in np.unique(segs):
            in_s = idx[segs == s]
            take = np.linspace(0, in_s.size - 1, num=min(per, in_s.size)).round().astype(int)
            keep.extend(in_s[np.unique(take)].tolist())
    return np.asarray(sorted(set(keep)), dtype=int)


def stack_windows(ds, idx: np.ndarray) -> np.ndarray:
    """Materialise their preprocessed windows for ``idx`` as ``(n, 1, 2048)``."""
    out = np.empty((idx.size, 1, ds.window_size), dtype=np.float32)
    for j, i in enumerate(idx):
        x, _ = ds[int(i)]
        out[j] = x.numpy()
    return out


def train_eval(torch, nn, model, Xtr, ytr, Xte, yte, *, epochs: int, lr: float,
               batch: int, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-6)
    counts = np.bincount(ytr, minlength=int(ytr.max()) + 1).astype(np.float64)
    w = counts.sum() / (np.maximum(counts, 1) * np.count_nonzero(counts))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.int64))
    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    hist: List[float] = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot = 0.0
        for b0 in range(0, n, batch):
            sel = torch.from_numpy(order[b0 : b0 + batch].astype(np.int64))
            if sel.numel() < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss) * sel.numel()
        sched.step()
        hist.append(tot / n)
        print(f"      epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    fit_s = time.time() - t0
    model.eval()
    probs = []
    with torch.no_grad():
        for b0 in range(0, Xte.shape[0], 128):
            logits = model(torch.from_numpy(Xte[b0 : b0 + 128]))
            probs.append(torch.softmax(logits, dim=1).numpy())
    P = np.vstack(probs)
    pred = np.argmax(P, axis=1)
    return {
        "fit_seconds": round(fit_s, 1),
        "samples_per_second": round(epochs * n / fit_s, 2),
        "loss_history": [round(v, 5) for v in hist],
        "pred": pred,
        "score_not_healthy": 1.0 - P[:, LOCATION_TO_INDEX["normal"]],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwru", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-windows-per-record", type=int, default=24)
    ap.add_argument("--upstream-sha", default=os.environ.get("UPSTREAM_SHA", "unknown"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "upstream_model_runs.json"))
    args = ap.parse_args(argv)

    src = os.path.abspath(args.src)
    if src not in sys.path:
        sys.path.insert(0, src)
    import torch
    import torch.nn as nn
    import dataset as up_dataset  # type: ignore
    import model as up_model  # type: ignore

    print("building their dataset from this branch's verified record table ...", flush=True)
    ds, cols = build_their_dataset(up_dataset, args.cwru)
    keep = thin(cols, args.max_windows_per_record)
    sub = {k: v[keep] for k, v in cols.items()}
    print(f"  {len(ds)} windows -> {keep.size} after thinning to "
          f"{args.max_windows_per_record}/record", flush=True)

    X = stack_windows(ds, keep)
    y = sub["label"].astype(int)
    win_s = ds.window_size / cwru.FS_FAULT_HZ
    # Window start time on the thinned set, for the overlap accounting.
    t_start = np.asarray([int(w.split("@")[1]) for w in sub["window_id"]]) * \
        (ds.window_size * (1.0 - ds.overlap)) / cwru.FS_FAULT_HZ

    out: Dict[str, object] = {
        "host": host(), "torch": torch.__version__,
        "upstream_branch": "impl/ml-supervised-1dcnn", "upstream_sha": args.upstream_sha,
        "model": "BearingFaultCNN1D (imported, unmodified)",
        "preprocessing": "dataset.segment_and_fft (imported, unmodified): Hann, "
                         "rfft magnitude, zero-pad to 2048, per-window z-score",
        "config": {k: v for k, v in vars(args).items()},
        "n_windows_total": len(ds),
        "n_windows_used": int(keep.size),
        "class_names": list(CLASS_NAMES),
        "class_counts": {CLASS_NAMES[c]: int(np.count_nonzero(y == c)) for c in np.unique(y)},
        "cells": [],
    }

    def audit_cols(sp: Split) -> Dict[str, object]:
        rep = leakage_report(sp, groups=sub["recording"], labels=y)
        rep_unit = leakage_report(sp, groups=sub["unit"], labels=y)
        same = sub["recording"][sp.train][:, None] == sub["recording"][sp.test][None, :]
        dt = np.abs(t_start[sp.train][:, None] - t_start[sp.test][None, :])
        frac = float(np.mean(np.any(same & (dt < win_s), axis=0))) if sp.test.size else 0.0
        return {
            "violations_vs_recording": rep["violations"],
            "violations_vs_physical_unit": rep_unit["violations"],
            "ok_vs_recording": rep["ok"],
            "ok_vs_physical_unit": rep_unit["ok"],
            "frac_test_windows_sharing_samples_with_train": round(frac, 4),
            "test_class_counts": rep["test_class_counts"],
        }

    # ---- the three arms -------------------------------------------------
    arms: Dict[str, Split] = {}
    from pdmcore.eval.splits import group_split, stratified_group_split
    arms["random_window"] = group_split(sub["window_id"], test_fraction=0.3, seed=0,
                                        name="random_window")
    arms["recording_grouped"] = stratified_group_split(sub["recording"], y,
                                                       test_fraction=0.3, seed=0)
    # Their own split, reproduced by calling their method and intersecting with
    # the thinned index set.
    tr_sub, te_sub = ds.split_by_bearing(test_fraction=0.2, seed=42)
    pos = {int(g): i for i, g in enumerate(keep)}
    tr_i = np.asarray(sorted(pos[i] for i in tr_sub.indices if i in pos), dtype=int)
    te_i = np.asarray(sorted(pos[i] for i in te_sub.indices if i in pos), dtype=int)
    arms["their_split_by_bearing"] = Split(train=tr_i, test=te_i, name="split_by_bearing(seed=42)")

    for name, sp in arms.items():
        a = audit_cols(sp)
        print(f"  arm {name}: sizes={sp.sizes} unit-clean={a['ok_vs_physical_unit']} "
              f"overlap={a['frac_test_windows_sharing_samples_with_train']}", flush=True)
        net = up_model.BearingFaultCNN1D(num_classes=up_model.NUM_CLASSES, input_length=2048)
        n_params = int(sum(p.numel() for p in net.parameters()))
        res = train_eval(torch, nn, net, X[sp.train], y[sp.train], X[sp.test], y[sp.test],
                         epochs=args.epochs, lr=args.lr, batch=args.batch_size, seed=0)
        pred = res.pop("pred")
        scores = res.pop("score_not_healthy")
        names_true = np.asarray([CLASS_NAMES[c] for c in y[sp.test]])
        names_pred = np.asarray([CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class{c}"
                                 for c in pred])
        metrics = protocol.evaluate(names_true, names_pred, scores, healthy="Normal")
        cell = {
            "arm": name, "split_name": sp.name, "sizes": list(sp.sizes),
            "n_params": n_params, "audit": a, **res, **metrics,
            "n_predicted_classes_outside_the_reachable_four": int(
                np.count_nonzero(pred >= len(CLASS_NAMES))),
        }
        print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}", flush=True)
        out["cells"].append(cell)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
