#!/usr/bin/env python3
"""Run the evaluation matrix and write results as JSON.

    python -I prototype/supervised_cnn/run_matrix.py --arm describe --cwru DIR --mfpt DIR
    python -I prototype/supervised_cnn/run_matrix.py --arm flaw1    --cwru DIR
    python -I prototype/supervised_cnn/run_matrix.py --arm flaw2    --cwru DIR
    python -I prototype/supervised_cnn/run_matrix.py --arm flaw3    --cwru DIR --mfpt DIR
    python -I prototype/supervised_cnn/run_matrix.py --arm confound --cwru DIR

Arms, each answering one question and writing one JSON:

``describe``  dataset inventories, the unit-degeneracy analysis, and every split
              with its leakage audit. No model: this is the part of the result
              that is a property of the benchmarks rather than of any classifier,
              and it is the part that does not depend on a CPU budget.
``flaw1``     random-window vs segment-grouped vs recording-grouped, 10 classes.
``flaw2``     cross-load: train on three motor loads, test on the fourth.
``flaw3``     cross-dataset: CWRU <-> MFPT on the shared 3-class label space,
              under both resampling conventions, plus the geometry ablation for
              the engineered-feature path.
``confound``  the synthetic sample-rate confound, to cost a mistake the files on
              disk do not actually have.

Every cell records the host, the wall-clock fit time, the parameter count, the
leakage audit, and the full metric set -- so a number in the deck can be traced
to the cell that produced it and to the machine it ran on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS_DIR, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

import baselines  # noqa: E402
import cwru  # noqa: E402
import mfpt  # noqa: E402
import protocol  # noqa: E402
from pdmcore.dsp.bearing import BEARING_CATALOG  # noqa: E402
from wdcnn import CnnClassifier, CnnConfig  # noqa: E402

MODELS = ("wdcnn", "ticnn", "upstream_1dcnn", "feat_logreg", "feat_mlp")


def host() -> Dict[str, object]:
    """What the timings were taken on. Quoted with every duration in the JSON."""
    info: Dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
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


# ---------------------------------------------------------------------------
# Feature cache: features do not depend on the split, so extract once.
# ---------------------------------------------------------------------------
class Features:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, str], Tuple[np.ndarray, Tuple[str, ...]]] = {}
        self.seconds = 0.0

    def get(self, ws, *, bearing_name: str = "SKF6205"):
        key = (id(ws), bearing_name)
        if key not in self._cache:
            t0 = time.time()
            self._cache[key] = baselines.extract(ws, geometry=BEARING_CATALOG[bearing_name])
            self.seconds += time.time() - t0
        return self._cache[key]


FEATURES = Features()


def fit_and_eval(
    model: str,
    ws_train,
    ws_test,
    *,
    n_classes: int,
    epochs: int,
    seed: int,
    bearing_train: str = "SKF6205",
    bearing_test: Optional[str] = None,
) -> Dict[str, object]:
    """One cell. Returns metrics plus everything needed to defend them."""
    ytr = ws_train.labels(n_classes)
    yte = ws_test.labels(n_classes)
    t0 = time.time()
    if model.startswith("feat_"):
        head = model.split("_", 1)[1]
        Ftr, names = FEATURES.get(ws_train, bearing_name=bearing_train)
        Fte, _ = FEATURES.get(ws_test, bearing_name=bearing_test or bearing_train)
        clf = baselines.FeatureClassifier(head=head, seed=seed, feature_names=names)
        clf.fit(Ftr, ytr)
        fit_s = time.time() - t0
        pred = clf.predict(Fte)
        scores = clf.score(Fte) if "healthy" in set(ytr) else None
        n_params = clf.n_params
        extra = {"bearing_train": bearing_train, "bearing_test": bearing_test or bearing_train}
    else:
        # ticnn shares WDCNN's layer stack; the delta this file implements is
        # input dropout plus train-time amplitude and noise augmentation, which
        # is the part of the TICNN paper its varying-working-conditions claim
        # rests on. See wdcnn.py for what is not implemented.
        cfg = CnnConfig(
            arch=model,
            n_classes=int(np.unique(ytr).size), epochs=epochs, seed=seed,
            input_dropout=0.1 if model == "ticnn" else 0.0,
            augment_amplitude=0.15 if model == "ticnn" else 0.0,
            augment_noise_snr_db=20.0 if model == "ticnn" else None,
        )
        clf = CnnClassifier(cfg).fit(ws_train.X, ytr)
        fit_s = clf.fit_seconds_
        pred = clf.predict(ws_test.X)
        scores = clf.score(ws_test.X) if "healthy" in set(ytr) else None
        n_params = clf.n_params
        extra = {"final_train_loss": float(clf.history_[-1]), "epochs": epochs,
                 "loss_history": [round(v, 5) for v in clf.history_]}
    metrics = protocol.evaluate(yte, pred, scores)
    metrics["gate"] = protocol.passes_gate(metrics)
    metrics["model"] = model
    metrics["n_params"] = int(n_params)
    metrics["fit_seconds"] = round(float(fit_s), 2)
    metrics["eval_seconds"] = round(time.time() - t0 - float(fit_s), 2)
    metrics["n_train"] = int(len(ws_train))
    metrics["seed"] = seed
    metrics.update(extra)
    return metrics


def _label_count(ws, n_classes: int) -> int:
    return int(np.unique(ws.labels(n_classes)).size)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def arm_describe(args) -> Dict[str, object]:
    out: Dict[str, object] = {"host": host(), "arm": "describe"}
    cw = cwru.load_records(args.cwru, window=args.window, overlap=args.overlap,
                           n_segments=args.n_segments,
                           max_windows_per_record=args.max_windows_per_record)
    out["cwru"] = {
        "describe": protocol.describe(cw, 10),
        "unit_degeneracy": {str(k): protocol.unit_groups_are_degenerate(cw, k) for k in (10, 4, 2)},
        "sha256_count": len(cwru.CWRU_RECORDS),
    }
    splits: Dict[str, object] = {}
    for kind in protocol.SPLIT_KINDS:
        try:
            sp = protocol.build_split(cw, kind, n_classes=10, seed=0)
        except ValueError as exc:
            splits[kind] = {"raised": str(exc)}
            continue
        rep = protocol.audit(cw, sp, n_classes=10, group_key="recording")
        splits[kind] = {
            "name": sp.name, "sizes": list(sp.sizes),
            "leakage_report": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                               for k, v in rep.items()},
        }
    out["cwru"]["splits"] = splits

    if args.mfpt:
        for align in ("rate", "order"):
            mw = mfpt.load_records(args.mfpt, window=args.window, overlap=args.overlap,
                                   n_segments=args.n_segments, align=align,
                                   max_windows_per_record=args.max_windows_per_record)
            out[f"mfpt_{align}"] = {
                "describe": protocol.describe(mw, 4),
                "unit_degeneracy": {str(k): protocol.unit_groups_are_degenerate(mw, k) for k in (4, 2)},
                "resample_ratios": {k: round(v, 6) for k, v in mw.resample_ratios.items()},
            }
    return out


def arm_flaw1(args) -> Dict[str, object]:
    """Random vs grouped. The literature's split against the honest ones."""
    cw = cwru.load_records(args.cwru, window=args.window, overlap=args.overlap,
                           n_segments=args.n_segments,
                           max_windows_per_record=args.max_windows_per_record)
    out: Dict[str, object] = {"host": host(), "arm": "flaw1",
                              "data": protocol.describe(cw, 10), "cells": []}
    for kind in ("random_window", "segment_grouped", "recording_grouped"):
        for seed in range(args.seeds):
            sp = protocol.build_split(cw, kind, n_classes=10, seed=seed)
            rep = protocol.audit(cw, sp, n_classes=10, group_key="recording")
            tr, te = cw.take(sp.train), cw.take(sp.test)
            for model in args.models:
                print(f"  [{kind} seed={seed}] {model} ...", flush=True)
                cell = fit_and_eval(model, tr, te, n_classes=10, epochs=args.epochs, seed=seed)
                cell.update({"split": kind, "split_name": sp.name,
                             "sizes": list(sp.sizes),
                             "leak_frac": rep["frac_test_windows_sharing_samples_with_train"],
                             "leak_violations": rep["violations"]})
                print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}  "
                      f"{cell['fit_seconds']:.0f}s", flush=True)
                out["cells"].append(cell)
    return out


def arm_flaw2(args) -> Dict[str, object]:
    """Cross-load. Train on three motor loads, test on the fourth."""
    cw = cwru.load_records(args.cwru, window=args.window, overlap=args.overlap,
                           n_segments=args.n_segments,
                           max_windows_per_record=args.max_windows_per_record)
    out: Dict[str, object] = {"host": host(), "arm": "flaw2",
                              "data": protocol.describe(cw, 10), "cells": []}
    for held in (0, 1, 2, 3):
        kind = f"cross_load:{held}"
        sp = protocol.build_split(cw, kind, n_classes=10)
        rep = protocol.audit(cw, sp, n_classes=10, group_key="recording")
        tr, te = cw.take(sp.train), cw.take(sp.test)
        for model in args.models:
            print(f"  [{kind}] {model} ...", flush=True)
            cell = fit_and_eval(model, tr, te, n_classes=10, epochs=args.epochs, seed=0)
            cell.update({"split": kind, "split_name": sp.name, "held_out_load_hp": held,
                         "sizes": list(sp.sizes),
                         "leak_frac": rep["frac_test_windows_sharing_samples_with_train"],
                         "leak_violations": rep["violations"],
                         "train_shaft_hz": [float(tr.shaft_hz.min()), float(tr.shaft_hz.max())],
                         "test_shaft_hz": [float(te.shaft_hz.min()), float(te.shaft_hz.max())]})
            print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}  "
                  f"{cell['fit_seconds']:.0f}s", flush=True)
            out["cells"].append(cell)
    return out


#: The only label space CWRU and MFPT share. MFPT has no ball-defect records, so
#: CWRU's three ball classes are dropped from both sides of this arm rather than
#: being mapped onto something.
SHARED_CLASSES = ("normal", "inner", "outer")


def arm_flaw3(args) -> Dict[str, object]:
    """Cross-dataset, both directions, both resampling conventions."""
    if not args.mfpt:
        raise SystemExit("--mfpt is required for arm flaw3")
    cw_full = cwru.load_records(args.cwru, window=args.window, overlap=args.overlap,
                                n_segments=args.n_segments,
                                max_windows_per_record=args.max_windows_per_record)
    cw = protocol.subset_classes(cw_full, SHARED_CLASSES, n_classes=4)
    out: Dict[str, object] = {"host": host(), "arm": "flaw3",
                              "shared_classes": list(SHARED_CLASSES),
                              "cwru": protocol.describe(cw, 4), "cells": []}

    # The within-dataset reference these transfers are measured against: the same
    # 3-class problem under CWRU's strictest usable split.
    for kind in ("recording_grouped",):
        sp = protocol.build_split(cw, kind, n_classes=4, seed=0)
        tr, te = cw.take(sp.train), cw.take(sp.test)
        for model in args.models:
            print(f"  [reference {kind} 3-class] {model} ...", flush=True)
            cell = fit_and_eval(model, tr, te, n_classes=4, epochs=args.epochs, seed=0)
            cell.update({"split": f"within_cwru_{kind}", "direction": "cwru->cwru",
                         "align": "n/a", "sizes": list(sp.sizes)})
            print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}", flush=True)
            out["cells"].append(cell)

    for align in ("rate", "order"):
        mw = protocol.subset_classes(
            mfpt.load_records(args.mfpt, window=args.window, overlap=args.overlap,
                              n_segments=args.n_segments, align=align,
                              max_windows_per_record=args.max_windows_per_record),
            SHARED_CLASSES, n_classes=4)
        out[f"mfpt_{align}"] = protocol.describe(mw, 4)
        for direction, (src, tgt, b_src, b_tgt) in {
            "cwru->mfpt": (cw, mw, "SKF6205", "NICE_MFPT"),
            "mfpt->cwru": (mw, cw, "NICE_MFPT", "SKF6205"),
        }.items():
            both, sp = protocol.cross_dataset_split(src, tgt)
            rep = protocol.audit(both, sp, n_classes=4, group_key="recording")
            for model in args.models:
                # For the feature path, two variants: the target's own bearing
                # geometry (what a deployment knows) and the source's (the
                # ablation that separates feature transfer from kinematic
                # transfer). The CNN has no geometry input, so one variant only.
                variants = [(b_tgt, "target_geometry")] if not model.startswith("feat_") \
                    else [(b_tgt, "target_geometry"), (b_src, "source_geometry")]
                for b_test, variant in variants:
                    print(f"  [{direction} align={align} {variant}] {model} ...", flush=True)
                    cell = fit_and_eval(model, src, tgt, n_classes=4, epochs=args.epochs,
                                        seed=0, bearing_train=b_src, bearing_test=b_test)
                    cell.update({"split": "cross_dataset", "direction": direction,
                                 "align": align, "geometry_variant": variant,
                                 "sizes": list(sp.sizes),
                                 "leak_frac": rep["frac_test_windows_sharing_samples_with_train"],
                                 "leak_violations": rep["violations"]})
                    print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}",
                          flush=True)
                    out["cells"].append(cell)
    return out


def arm_confound(args) -> Dict[str, object]:
    """Cost the sample-rate confound. Binary healthy-vs-faulty, grouped split."""
    out: Dict[str, object] = {"host": host(), "arm": "confound", "cells": []}
    for confound in (False, True):
        cw = cwru.load_records(args.cwru, window=args.window, overlap=args.overlap,
                               n_segments=args.n_segments, rate_confound=confound,
                               max_windows_per_record=args.max_windows_per_record)
        sp = protocol.build_split(cw, "recording_grouped", n_classes=2, seed=0)
        tr, te = cw.take(sp.train), cw.take(sp.test)
        for model in args.models:
            print(f"  [rate_confound={confound}] {model} ...", flush=True)
            cell = fit_and_eval(model, tr, te, n_classes=2, epochs=args.epochs, seed=0)
            cell.update({"split": "recording_grouped", "rate_confound": confound,
                         "sizes": list(sp.sizes),
                         "note": "rate_confound=True is SYNTHETIC: the healthy class is "
                                 "resampled 4x to emulate a 48 kHz record read as 12 kHz. "
                                 "The CWRU files on disk do not have this problem."})
            print(f"      acc {cell['accuracy']:.4f}  macroF1 {cell['macro_f1']:.4f}", flush=True)
            out["cells"].append(cell)
    return out


ARMS = {"describe": arm_describe, "flaw1": arm_flaw1, "flaw2": arm_flaw2,
        "flaw3": arm_flaw3, "confound": arm_confound}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--cwru", required=True)
    ap.add_argument("--mfpt", default=None)
    ap.add_argument("--models", default="wdcnn,feat_logreg,feat_mlp")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--n-segments", type=int, default=6)
    ap.add_argument("--max-windows-per-record", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in args.models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}; expected from {MODELS}")

    t0 = time.time()
    print(f"arm {args.arm}: models={args.models} epochs={args.epochs} seeds={args.seeds}",
          flush=True)
    result = ARMS[args.arm](args)
    result["config"] = {k: v for k, v in vars(args).items()}
    result["feature_extraction_seconds"] = round(FEATURES.seconds, 2)
    result["total_seconds"] = round(time.time() - t0, 2)
    result["host_end"] = host()
    out = args.out or os.path.join(RESULTS_DIR, f"matrix_{args.arm}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
    print(f"wrote {out}  ({result['total_seconds']:.0f} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
