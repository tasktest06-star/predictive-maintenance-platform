#!/usr/bin/env python3
"""Audit the two upstream supervised branches by RUNNING their code.

    /workshop/predictive_maintaince/.venv-torch/bin/python -I \
        prototype/supervised_cnn/audit_upstream.py --cwru DIR --src DIR

``--src`` is a checkout of ``impl/ml-supervised-1dcnn``'s
``impl/ml_supervised_1dcnn/`` directory. ``tools/fetch_upstream.sh`` materialises
it; nothing from it is committed here.

Why this exists
---------------
Two branches on this repository's remote already implement supervised bearing
CNNs -- ``impl/ml-supervised-1dcnn`` (~2 000 lines, PyTorch) and
``impl/ml-transformer-faultformer`` (~3 200 lines, including a 529-line
``cross_domain.py`` aimed at exactly this branch's question). Neither has been
merged into ``platform/core`` and neither carries evidence of having been run on
measured data. Writing a fourth implementation and ignoring them would be the
least useful thing available, so this module checks what they do when handed the
real CWRU files.

Both branches' dataset layers are *trying* to do the right thing: both cite the
data-leakage literature, both call their split "bearing-wise", and both warn in
their docstrings that a random window split inflates accuracy. The findings below
are not about carelessness. They are about the gap between a split that is named
bearing-wise and a split that is bearing-wise.

What each check establishes
---------------------------
1. ``check_flat_directory_labels`` -- their ``_scan_directory`` derives the label
   from *directory names*. CWRU is published as flat numbered ``.mat`` files, so
   against the data as distributed every record falls through to
   ``("normal", "0")`` and the 10-class problem silently becomes a one-class
   problem on which any model reports 100 %.
2. ``check_99_mat`` -- ``_load_cwru_mat`` returns the first variable whose name
   ends ``_DE_time``. ``99.mat`` as published contains ``X098_DE_time`` *and*
   ``X099_DE_time``, so the 2 hp baseline record silently returns the 1 hp one.
3. ``check_bearing_id_is_a_file`` -- their ``bearing_id`` is one per file. CWRU
   has 40 files and 10 physical bearings, so their "bearing-wise" split puts the
   same physical bearing on both sides of the partition. Audited with
   ``pdmcore.eval.splits.leakage_report`` against the unit key.
4. ``check_split_can_drop_classes`` -- their ``split_by_bearing`` shuffles ids and
   takes a fraction; nothing keeps every class on both sides.
   ``pdmcore.eval.splits.stratified_group_split`` exists for this and raises
   instead.
5. ``check_model_selection_on_test`` -- static: ``train.py`` has one loader for
   both validation and test and keeps the checkpoint with the best
   ``val_macro_f1`` on it.
6. ``check_synthetic_fallback`` -- static quotation plus a transcription:
   ``impl/ml-transformer-faultformer/cross_domain.py`` falls back to
   ``_make_synthetic_labeled_dataset`` when the dataset directory is absent, then
   runs and logs the cross-domain transfer experiment anyway. The synthetic
   signals are ``sin(2*pi*(50 + 30*label)*t) + 0.5*noise`` -- separable by a
   single FFT bin.

Results go to ``results/upstream_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

import cwru  # noqa: E402
from pdmcore.eval.splits import Split, leakage_report, stratified_group_split  # noqa: E402

#: Commit audited. Recorded so a later change upstream cannot silently
#: invalidate these findings without the sha disagreeing.
UPSTREAM_BRANCH = "impl/ml-supervised-1dcnn"
FAULTFORMER_BRANCH = "impl/ml-transformer-faultformer"


def _attr(obj, name):
    """Their ``labels`` / ``bearing_ids`` are properties on one class and methods
    on another. Tolerate both rather than hard-coding one and reporting a
    TypeError as a finding."""
    v = getattr(obj, name)
    return v() if callable(v) else v


def _import_upstream(src: str):
    """Import their ``model`` and ``dataset`` modules. Needs torch."""
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    import dataset as up_dataset  # type: ignore  # noqa: E402
    import model as up_model  # type: ignore  # noqa: E402
    return up_model, up_dataset


# ---------------------------------------------------------------------------
# 1. Labels against the data as published
# ---------------------------------------------------------------------------
def check_flat_directory_labels(up_dataset, cwru_root: str) -> Dict[str, object]:
    ds = up_dataset.CWRUBearingDataset(data_root=cwru_root)
    labels = np.asarray(_attr(ds, 'labels'))
    uniq, counts = np.unique(labels, return_counts=True)
    names = [up_dataset.FAULT_CLASSES[i] if hasattr(up_dataset, "FAULT_CLASSES") else str(i)
             for i in uniq]
    return {
        "finding": "label is derived from directory names; CWRU is published as "
                   "flat numbered .mat files, so every record falls through to "
                   "('normal','0')",
        "n_windows": int(labels.size),
        "n_distinct_labels": int(uniq.size),
        "label_counts": {str(n): int(c) for n, c in zip(names, counts)},
        "n_bearing_ids": int(np.unique(_attr(ds, "bearing_ids")).size),
        "accuracy_of_always_predicting_the_majority": float(counts.max() / labels.size),
        "ok": bool(uniq.size > 1),
    }


# ---------------------------------------------------------------------------
# 2. 99.mat
# ---------------------------------------------------------------------------
def check_99_mat(up_dataset, cwru_root: str) -> Dict[str, object]:
    from scipy.io import loadmat
    path = os.path.join(cwru_root, "99.mat")
    mat = loadmat(path)
    present = sorted(k for k in mat if not k.startswith("__"))
    got = up_dataset._load_cwru_mat(__import__("pathlib").Path(path), channel="DE")
    x98 = np.asarray(mat["X098_DE_time"]).ravel()
    x99 = np.asarray(mat["X099_DE_time"]).ravel()
    is98 = got is not None and got.size == x98.size and bool(np.allclose(got[:1000], x98[:1000]))
    is99 = got is not None and got.size == x99.size and bool(np.allclose(got[:1000], x99[:1000]))
    return {
        "finding": "_load_cwru_mat returns the first variable ending '_DE_time'; "
                   "99.mat as published holds X098_DE_time and X099_DE_time",
        "variables_in_99_mat": present,
        "returned_n_samples": int(got.size) if got is not None else None,
        "n_samples_X098": int(x98.size),
        "n_samples_X099": int(x99.size),
        "returned_is_X098": is98,
        "returned_is_X099": is99,
        "ok": is99,
    }


# ---------------------------------------------------------------------------
# 3. bearing_id granularity
# ---------------------------------------------------------------------------
def _records_for_upstream(cwru_root: str):
    """Build their ``records`` argument from *our* verified record table.

    This hands their loader correct labels, so checks 3 and 4 measure their
    *split*, not their directory scanner. Their ``bearing_id`` convention
    (one per file) is reproduced exactly: ``f"cwru_{counter:04d}_{stem}"``.
    """
    import pathlib
    label_of = {"normal": 0, "outer": 1, "inner": 2, "ball": 3}
    recs, unit_of = [], {}
    for i, rec in enumerate(cwru.CWRU_RECORDS):
        bid = f"cwru_{i:04d}_{rec.stem}"
        recs.append((pathlib.Path(os.path.join(cwru_root, f"{rec.stem}.mat")),
                     label_of[rec.location], bid))
        unit_of[bid] = rec.unit
    return recs, unit_of


def _per_window_bearing_ids(ds) -> np.ndarray:
    col = np.empty(len(ds), dtype=object)
    for bid, idxs in ds._bearing_index.items():
        for i in idxs:
            col[i] = bid
    return np.asarray([str(v) for v in col])


def check_bearing_id_is_a_file(up_dataset, cwru_root: str) -> Dict[str, object]:
    recs, unit_of = _records_for_upstream(cwru_root)
    ds = up_dataset.CWRUBearingDataset(records=recs)
    # ``bearing_ids`` is the list of distinct ids (40); the per-window column has
    # to be rebuilt from their ``_bearing_index`` map of id -> window indices.
    bids = _per_window_bearing_ids(ds)
    train, test = ds.split_by_bearing(test_fraction=0.2, seed=42)
    sp = Split(train=np.asarray(train.indices), test=np.asarray(test.indices))
    units = np.asarray([unit_of[b] for b in bids])
    rep_unit = leakage_report(sp, groups=units, labels=np.asarray(_attr(ds, 'labels')))
    rep_file = leakage_report(sp, groups=bids)
    shared = sorted(set(units[sp.train]) & set(units[sp.test]))
    return {
        "finding": "bearing_id is one per .mat file. CWRU has 40 files and 10 "
                   "physical bearings (the four files per bearing are the same "
                   "bearing at 0/1/2/3 hp), so a 'bearing-wise' split puts the "
                   "same physical bearing on both sides",
        "n_bearing_ids_upstream": int(np.unique(bids).size),
        "n_physical_units": int(np.unique(units).size),
        "audit_against_their_own_key": {"ok": rep_file["ok"], "violations": rep_file["violations"]},
        "audit_against_physical_unit": {"ok": rep_unit["ok"], "violations": rep_unit["violations"]},
        "physical_units_on_both_sides": shared,
        "ok": bool(rep_unit["ok"]),
    }


# ---------------------------------------------------------------------------
# 4. class coverage of their split
# ---------------------------------------------------------------------------
def check_split_can_drop_classes(up_dataset, cwru_root: str,
                                 seeds: Sequence[int] = tuple(range(40))) -> Dict[str, object]:
    recs, _ = _records_for_upstream(cwru_root)
    ds = up_dataset.CWRUBearingDataset(records=recs)
    y = np.asarray(_attr(ds, 'labels'))
    bids_win = _per_window_bearing_ids(ds)
    n_classes = int(np.unique(y).size)
    bad_test, bad_train, examples = 0, 0, []
    for s in seeds:
        train, test = ds.split_by_bearing(test_fraction=0.2, seed=s)
        ytr, yte = y[np.asarray(train.indices)], y[np.asarray(test.indices)]
        miss_te = sorted(set(np.unique(y)) - set(np.unique(yte)))
        miss_tr = sorted(set(np.unique(y)) - set(np.unique(ytr)))
        if miss_te:
            bad_test += 1
            if len(examples) < 5:
                examples.append({"seed": int(s), "classes_absent_from_test": [int(c) for c in miss_te]})
        if miss_tr:
            bad_train += 1
    # What pdmcore does instead, on the same groups and labels.
    try:
        sp = stratified_group_split(bids_win, y, test_fraction=0.2, seed=0)
        pdm = {"raised": None, "sizes": list(sp.sizes),
               "test_class_counts": leakage_report(sp, groups=bids_win, labels=y)["test_class_counts"]}
    except ValueError as exc:
        pdm = {"raised": str(exc)}
    return {
        "finding": "split_by_bearing shuffles ids and takes a fraction; nothing "
                   "keeps every class on both sides, and a class absent from the "
                   "test fold is simply not evaluated",
        "n_seeds_tried": len(seeds),
        "n_classes": n_classes,
        "n_seeds_with_a_class_missing_from_test": bad_test,
        "n_seeds_with_a_class_missing_from_train": bad_train,
        "examples": examples,
        "pdmcore_stratified_group_split_on_the_same_input": pdm,
        "ok": bad_test == 0 and bad_train == 0,
    }


# ---------------------------------------------------------------------------
# 5 and 6. static findings, quoted
# ---------------------------------------------------------------------------
def check_model_selection_on_test(src: str) -> Dict[str, object]:
    """``train.py`` selects the checkpoint by the test fold's own macro F1."""
    path = os.path.join(src, "train.py")
    quoted: List[str] = []
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if ("evaluate(model, test_loader" in s or "best_val_f1" in s
                or ("val_loader" in s and "=" in s) or "patience" in s.lower()):
            quoted.append(f"{i}: {s}")
    one_loader = not any("val_loader" in ln for ln in lines)
    return {
        "finding": "the loader passed to evaluate() each epoch is test_loader, and "
                   "the checkpoint kept is the one with the best macro F1 on it. "
                   "Early stopping and model selection therefore read the test "
                   "fold, so the reported figure is a best-of-N over epochs",
        "no_separate_validation_loader": one_loader,
        "quoted_lines": quoted[:12],
        "note": "a separate validation fold is not actually available in CWRU: "
                "every fold that is not the test fold shares a physical bearing "
                "with it, which is the same structural limit this branch reports. "
                "The honest options are a fixed epoch budget or nested "
                "cross-validation over units, not selection on the test fold.",
        "ok": not one_loader,
    }


#: Transcribed from ``impl/ml_transformer_faultformer/finetune.py`` at
#: ``_make_synthetic_labeled_dataset``. A transcription and not an import,
#: because that module imports ``mlflow``, which is not installed here -- so this
#: is explicitly a reproduction of their generator and not their code running.
def _transcribed_synthetic(n_samples: int = 500, window_length: int = 1024,
                           num_classes: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n_samples):
        label = i % num_classes
        t = np.linspace(0, 1, window_length)
        freq = 50 + label * 30
        sig = rng.standard_normal(window_length) * 0.5 + np.sin(2 * np.pi * freq * t)
        X.append(sig)
        y.append(label)
    return np.asarray(X), np.asarray(y)


def check_synthetic_fallback(ff_src: Optional[str]) -> Dict[str, object]:
    """Cost the fallback: how good does a trivial model look on it?"""
    X, y = _transcribed_synthetic()
    # One rfft bin per class is enough; use the single most discriminative bin.
    mag = np.abs(np.fft.rfft(X, axis=1))
    freqs = np.fft.rfftfreq(X.shape[1], 1.0 / X.shape[1])
    pick = [int(np.argmin(np.abs(freqs - (50 + 30 * c)))) for c in range(4)]
    pred = np.argmax(mag[:, pick], axis=1)
    acc = float(np.mean(pred == y))
    quoted: List[str] = []
    if ff_src:
        path = os.path.join(ff_src, "cross_domain.py")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if "synthetic" in line.lower() or "not found" in line.lower():
                        quoted.append(f"{i}: {line.strip()}")
    return {
        "finding": "cross_domain.py's MFPT and MIMII loaders fall back to "
                   "_make_synthetic_labeled_dataset when the data directory is "
                   "absent, print one line saying so, and then run and log the "
                   "cross-domain transfer experiment as if it were a measurement",
        "synthetic_signal": "sin(2*pi*(50 + 30*label)*t) + 0.5*N(0,1)",
        "accuracy_of_a_4_bin_argmax_on_the_fallback_data": acc,
        "quoted_lines": quoted[:10],
        "note": "a transcription of their generator, not their code running: "
                "finetune.py imports mlflow, which is not installed here. The "
                "control flow was read from the committed source and is quoted "
                "above. 500 windows, 1024 samples, 4 classes, 20 bearing ids, "
                "and label = i %% num_classes with 20 ids means each id carries "
                "exactly one class, so the bearing-wise split applied to it is "
                "also class-degenerate",
        "ok": acc < 0.9,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cwru", required=True)
    ap.add_argument("--src", required=True, help=f"checkout of {UPSTREAM_BRANCH}'s impl dir")
    ap.add_argument("--faultformer-src", default=None, help=f"checkout of {FAULTFORMER_BRANCH}'s impl dir")
    ap.add_argument("--upstream-sha", default=os.environ.get("UPSTREAM_SHA", "unknown"))
    ap.add_argument("--faultformer-sha", default=os.environ.get("FAULTFORMER_SHA", "unknown"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "upstream_audit.json"))
    args = ap.parse_args(argv)

    up_model, up_dataset = _import_upstream(args.src)
    import torch
    out: Dict[str, object] = {
        "branches": {UPSTREAM_BRANCH: args.upstream_sha, FAULTFORMER_BRANCH: args.faultformer_sha},
        "torch": torch.__version__,
        "upstream_num_classes": int(up_model.NUM_CLASSES),
        "upstream_fault_classes": list(up_model.FAULT_CLASSES),
        "upstream_classes_reachable_from_cwru_and_mfpt": sorted(
            set(up_dataset.CWRU_LABEL_MAP.values()) | set(up_dataset.MFPT_LABEL_MAP.values())),
        "checks": {},
    }
    out["upstream_unreachable_class_count"] = int(
        up_model.NUM_CLASSES - len(out["upstream_classes_reachable_from_cwru_and_mfpt"]))

    checks = out["checks"]
    print("1. labels against the data as published ...", flush=True)
    checks["flat_directory_labels"] = check_flat_directory_labels(up_dataset, args.cwru)
    print("2. 99.mat ...", flush=True)
    checks["99_mat"] = check_99_mat(up_dataset, args.cwru)
    print("3. bearing_id granularity ...", flush=True)
    checks["bearing_id_is_a_file"] = check_bearing_id_is_a_file(up_dataset, args.cwru)
    print("4. class coverage of their split ...", flush=True)
    checks["split_can_drop_classes"] = check_split_can_drop_classes(up_dataset, args.cwru)
    print("5. model selection ...", flush=True)
    checks["model_selection_on_test"] = check_model_selection_on_test(args.src)
    print("6. synthetic fallback ...", flush=True)
    checks["synthetic_fallback"] = check_synthetic_fallback(args.faultformer_src)

    out["n_checks"] = len(checks)
    out["n_checks_the_upstream_code_passes"] = sum(int(bool(c.get("ok"))) for c in checks.values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    for name, c in checks.items():
        print(f"  {'PASS' if c.get('ok') else 'FAIL'}  {name}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
