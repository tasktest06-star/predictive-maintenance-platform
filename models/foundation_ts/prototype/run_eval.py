#!/usr/bin/env python3
"""Run the comparison and commit the numbers.

    /workshop/predictive_maintaince/.venv/bin/python -I \
        models/foundation_ts/prototype/run_eval.py \
        --cache-dir /workshop/predictive_maintaince/fm_work \
        --out models/foundation_ts/results

Produces one JSON per dataset plus ``summary.json``. Every number quoted in
``docs/09_foundation_models.md``, in ``slides/foundation_models.tex`` and in the
commit message comes from these files, and the figures are generated from them
rather than from a second run -- the pattern
``slides/figures/make_core_figures.py`` established, for the reason recorded in
``docs/08_branch_catalogue.md``: prose in this repository has disagreed with its
own measurement more than once.

The foundation-model arm is included only if ``fm_<dataset>.npz`` exists next
to the cache. Without it the script still runs and still produces the baseline
numbers on measured data -- which is the more valuable half, because
``docs/07_production_readiness.md`` records that no model in this repository had
been validated on measured data at all.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_REPO, "core"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pdmcore.eval.metrics import auc                                    # noqa: E402

from evaluate import (ArmResult, GATE_LATENCY_MULTIPLE, MIN_TRAIN_WINDOWS,  # noqa: E402
                      TARGET_FPR, aggregate, build_folds, decide,
                      delay_on_series, leaky_fold, paired_bootstrap_ci,
                      score_arm, score_native)

#: Fold scheme per dataset, and whether it is measured or simulated. The
#: ``measured`` flag is carried into the results JSON so no figure can quote a
#: simulator number as if it were a measurement.
DATASETS: Dict[str, Dict[str, object]] = {
    "cwru_de":   {"scheme": "cwru_load",   "measured": True},
    "cwru_fe":   {"scheme": "cwru_load",   "measured": True},
    "mfpt_97k":  {"scheme": "mfpt_record", "measured": True},
    "sim_fleet": {"scheme": "sim_unit",    "measured": False},
}

BASELINE_ARM = "features63"
BASELINE_HEAD = "mahalanobis"
EMB_HEADS = ("pca63_mahalanobis", "mahalanobis")


def saturation_diagnostics(X: np.ndarray, levels: np.ndarray, y: np.ndarray,
                           feature_names: Sequence[str]) -> Dict[str, object]:
    """How hard is this dataset, independent of any model?

    Three numbers, and the first two are the ones that decide whether a
    detector comparison on this dataset means anything:

    ``auc_level_alone``
        AUC of 20*log10(RMS) of the window. One number, no fitting, no
        features. If this is 1.0 the dataset cannot rank detectors, because
        every detector that looks at amplitude at all already wins.
    ``best_single_feature``
        The best of the 63 features by AUC. Same purpose, one step less naive.
    ``trivial``
        True when ``auc_level_alone >= 0.999``. A dataset flagged trivial is
        reported, and its *ranking* results are explicitly not used to choose
        between arms -- see ``research/02_protocol.md``.

    These use the labels over the whole dataset, so they are a property of the
    dataset and not a model result; they are kept in their own section of the
    JSON for exactly that reason.
    """
    a_level = float(auc(y, levels))
    per_feature = [(float(auc(y, X[:, i])), str(feature_names[i]))
                   for i in range(X.shape[1])]
    per_feature.sort(reverse=True)
    return {
        "auc_level_alone": a_level,
        "auc_level_alone_note": "20*log10(window RMS), no model, no fitting",
        "best_single_features": [{"auc": v, "feature": n} for v, n in per_feature[:5]],
        "trivial": bool(a_level >= 0.999),
        "n_features_above_auc_099": int(sum(1 for v, _ in per_feature if v >= 0.99)),
    }


def host_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="ascii") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "bogomips")):
                    k, _, v = line.partition(":")
                    info[k.strip().lower()] = v.strip()
                    break
    except OSError:
        pass
    return info


def load_cache(cache_dir: str, name: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    npz_path = os.path.join(cache_dir, f"cache_{name}.npz")
    json_path = os.path.join(cache_dir, f"cache_{name}.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"no cache for {name} at {npz_path}. Build it first:\n"
            f"  python -I models/foundation_ts/prototype/build_cache.py "
            f"--dataset <d> --out {cache_dir}")
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {k: np.asarray(npz[k]) for k in npz.files}
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return arrays, meta


def load_fm(cache_dir: str, name: str) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    npz_path = os.path.join(cache_dir, f"fm_{name}.npz")
    if not os.path.exists(npz_path):
        return {}, None
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {k: np.asarray(npz[k]) for k in npz.files}
    with open(os.path.join(cache_dir, f"fm_{name}.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return arrays, meta


def fm_ms_per_window(fm_meta: Dict, arm: str) -> float:
    """Per-window inference cost for one arm, from the embedding run's own
    timings. Measured on this host, which is named in the JSON."""
    for key, model in fm_meta.get("models", {}).items():
        if not arm.startswith(f"emb_{key}_") and not arm.startswith(f"score_{key}_"):
            continue
        pres = arm.split(f"{key}_", 1)[1].replace("fcast_", "").replace("recon_", "")
        pres = pres.replace("_plus_level", "")
        t = model.get("timings_ms_per_window", {}).get(pres, {})
        if arm.startswith("emb_"):
            return float(t.get("embed_ms_per_window", float("nan")))
        return float(t.get("forecast_ms_per_window",
                           t.get("reconstruct_ms_per_window", float("nan"))))
    return float("nan")


def run_dataset(name: str, cache_dir: str, *, with_fm: bool) -> Dict[str, object]:
    arrays, cmeta = load_cache(cache_dir, name)
    cfg = DATASETS[name]
    meta_cols = {k: arrays[k] for k in ("y", "label", "group", "cond", "t_index")}
    y = arrays["y"].astype(int)
    X63 = arrays["features"].astype(np.float64)
    levels = arrays["levels_db"].astype(np.float64)[:, None]
    feature_names = cmeta["feature_names"]

    folds = build_folds(meta_cols, str(cfg["scheme"]))
    if not folds:
        raise RuntimeError(f"{name}: no usable folds (need >= {MIN_TRAIN_WINDOWS} "
                           f"healthy training windows per fold)")
    audits = [{"fold": f.name, "sizes": f.sizes(),
               "group_audit": f.group_audit, "time_audit": f.time_audit,
               "oneclass_audit": f.oneclass_audit}
              for f in folds]
    if not all(f.ok and f.time_audit["ok"] for f in folds):
        raise RuntimeError(f"{name}: leakage audit failed; see audits in the "
                           f"results JSON. Refusing to report metrics from a "
                           f"split that leaks.")

    results: List[ArmResult] = []
    t0 = time.time()
    results += score_arm(BASELINE_ARM, X63, y, folds,
                         heads=("mahalanobis", "pca_residual", "robust_threshold"),
                         feature_names=feature_names)
    baseline_s = time.time() - t0

    # The leaky control, baseline arm only: one number, to size the inflation.
    lf = leaky_fold(meta_cols)
    leaky = score_arm(f"{BASELINE_ARM}_LEAKY", X63, y, [lf],
                      heads=("mahalanobis",), feature_names=feature_names)
    if lf.group_audit.get("ok", False):
        raise AssertionError(
            "the deliberately leaky control passed the leakage audit; the "
            "audit is not working and no number in this run can be trusted")

    fm_arrays, fm_meta = load_fm(cache_dir, name) if with_fm else ({}, None)
    fm_timing: Dict[str, float] = {}
    for key in sorted(fm_arrays):
        if key.startswith("emb_"):
            arm = key
            Z = fm_arrays[key].astype(np.float64)
            results += score_arm(arm, Z, y, folds, heads=EMB_HEADS)
            fm_timing[arm] = fm_ms_per_window(fm_meta or {}, arm)
            # Generous variant: restore the overall level that every one of
            # these models normalises away before its encoder sees the window.
            arm_l = f"{key}_plus_level"
            results += score_arm(arm_l, np.hstack([Z, levels]), y, folds,
                                 heads=("pca63_mahalanobis",))
            fm_timing[arm_l] = fm_timing[arm]
        elif key.startswith("score_"):
            arm = key
            results += score_native(arm, fm_arrays[key].astype(np.float64), y, folds)
            fm_timing[arm] = fm_ms_per_window(fm_meta or {}, arm)

    agg = aggregate(results)
    agg_leaky = aggregate(leaky)
    return {
        "dataset": name,
        "measured_data": bool(cfg["measured"]),
        "fold_scheme": cfg["scheme"],
        "n_folds": len(folds),
        "target_fpr": TARGET_FPR,
        "cache_meta": {k: cmeta[k] for k in
                       ("summary", "provenance", "geometries", "per_window_ms",
                        "feature_configs", "feature_spec_version",
                        "demod_fraction_of_nyquist", "seq_len")
                       if k in cmeta},
        "fold_audits": audits,
        "aggregate": agg,
        "saturation": saturation_diagnostics(X63, levels[:, 0], y, feature_names),
        "leaky_control": agg_leaky,
        "leaky_control_note": (
            "A random window-level split, ignoring recordings. For a ONE-CLASS "
            "detector the familiar random-split inflation does not show up in "
            "AUC -- the model never sees a fault either way -- it shows up in "
            "the realised false-alarm rate, because the test healthy windows "
            "come from the same recordings and the same operating points the "
            "threshold was calibrated on. Compare "
            "leaky_control[..].fpr_at_threshold with "
            "aggregate[features63|mahalanobis].fpr_at_threshold."),
        "per_fold": [r.as_dict() for r in results],
        "fm_meta": fm_meta,
        "fm_ms_per_window": fm_timing,
        "baseline_eval_seconds": round(baseline_s, 2),
    }


def run_degradation(cache_dir: str, *, with_fm: bool) -> Dict[str, object]:
    """Detection delay on the simulated run-to-failure series.

    Simulated, and labelled so everywhere: CWRU and MFPT are seeded-fault
    snapshots, so neither can produce a detection delay. The arms are compared
    on the same series with the same calibration windows.
    """
    arrays, cmeta = load_cache(cache_dir, "sim_degradation")
    sev = arrays["severity"]
    days = float(cmeta["provenance"]["days"])
    onset = int(np.argmax(sev > 0.05))
    # Pre-onset points only, split in time: first 60 % fits, next 30 % calibrates.
    n_fit = int(0.6 * onset)
    n_cal = int(0.3 * onset)
    if n_fit < MIN_TRAIN_WINDOWS // 2 or n_cal < 10:
        raise RuntimeError(f"degradation series: only {onset} pre-onset points, "
                           f"not enough to fit and calibrate separately")
    tr = np.arange(n_fit)
    ca = np.arange(onset - n_cal, onset)

    from pdmcore.models.anomaly import MahalanobisDetector

    out: Dict[str, object] = {"dataset": "sim_degradation", "measured_data": False,
                              "onset_index": onset, "days": days,
                              "n_fit": int(tr.size), "n_calib": int(ca.size),
                              "arms": {}}

    def one(arm: str, Z: np.ndarray) -> None:
        det = MahalanobisDetector()
        det.fit(Z[tr])
        s_all = det.score(Z)
        out["arms"][arm] = delay_on_series(s_all, det.score(Z[ca]), sev, days=days)

    one(BASELINE_ARM, arrays["features"].astype(np.float64))
    if with_fm:
        fm_arrays, _ = load_fm(cache_dir, "sim_degradation")
        for key in sorted(fm_arrays):
            if key.startswith("emb_"):
                Z = fm_arrays[key].astype(np.float64)
                k = int(min(63, tr.size - 1, Z.shape[1]))
                mu = Z[tr].mean(axis=0)
                _u, _s, vt = np.linalg.svd(Z[tr] - mu, full_matrices=False)
                one(key, (Z - mu) @ vt[:k].T)
    return out


def summarise(per_dataset: Dict[str, Dict[str, object]],
              degradation: Optional[Dict[str, object]]) -> Dict[str, object]:
    """The headline: best challenger vs the baseline, with the four gates."""
    summary: Dict[str, object] = {"host": host_info(), "datasets": {},
                                  "gate_latency_multiple": GATE_LATENCY_MULTIPLE}
    delays = ((degradation or {}).get("arms") or {})
    base_delay = (delays.get(BASELINE_ARM) or {}).get("delay_h")

    for name, res in per_dataset.items():
        agg = res["aggregate"]
        base_key = f"{BASELINE_ARM}|{BASELINE_HEAD}"
        if base_key not in agg:
            continue
        base = agg[base_key]
        base_ms = float(res["cache_meta"]["per_window_ms"]["feature_extraction"])
        challengers = {k: v for k, v in agg.items()
                       if not str(v["arm"]).startswith(BASELINE_ARM)}
        best_key, best = None, None
        for k, v in challengers.items():
            m = v["pauc_10"]["mean"]
            if not np.isfinite(m):
                continue
            if best is None or m > best["pauc_10"]["mean"]:
                best_key, best = k, v
        entry: Dict[str, object] = {
            "measured_data": res["measured_data"],
            "fold_scheme": res["fold_scheme"],
            "n_folds": res["n_folds"],
            "baseline": {"key": base_key,
                         "pauc_10": base["pauc_10"], "auc": base["auc"],
                         "tpr_at_threshold": base["tpr_at_threshold"],
                         "fpr_at_threshold": base["fpr_at_threshold"],
                         "alarms_per_day_per_100": base["alarms_per_day_per_100"],
                         "ms_per_window": base_ms},
            "baseline_other_heads": {
                k: {"pauc_10": v["pauc_10"]["mean"], "auc": v["auc"]["mean"]}
                for k, v in agg.items()
                if str(v["arm"]) == BASELINE_ARM and k != base_key},
            "saturation": res["saturation"],
            "leaky_control": {
                k: {"auc": v["auc"]["mean"], "pauc_10": v["pauc_10"]["mean"],
                    "fpr_at_threshold": v["fpr_at_threshold"]["mean"],
                    "alarms_per_day_per_100": v["alarms_per_day_per_100"]["mean"]}
                for k, v in res["leaky_control"].items()},
            "n_challengers": len(challengers),
        }
        if best is not None and best_key is not None:
            ch_ms = float(res["fm_ms_per_window"].get(str(best["arm"]), float("nan")))
            ch_delay = (delays.get(str(best["arm"])) or {}).get("delay_h")
            entry["best_challenger"] = {
                "key": best_key, "pauc_10": best["pauc_10"], "auc": best["auc"],
                "tpr_at_threshold": best["tpr_at_threshold"],
                "fpr_at_threshold": best["fpr_at_threshold"],
                "alarms_per_day_per_100": best["alarms_per_day_per_100"],
                "ms_per_window": ch_ms,
            }
            entry["decision"] = decide(
                best, base,
                challenger_ms_per_window=ch_ms, baseline_ms_per_window=base_ms,
                challenger_delay_h=ch_delay, baseline_delay_h=base_delay)
            entry["challenger_ranking"] = sorted(
                ({"key": k, "pauc_10": v["pauc_10"]["mean"],
                  "auc": v["auc"]["mean"],
                  "tpr_at_threshold": v["tpr_at_threshold"]["mean"]}
                 for k, v in challengers.items()
                 if np.isfinite(v["pauc_10"]["mean"])),
                key=lambda d: -d["pauc_10"])
        summary["datasets"][name] = entry

    if degradation is not None:
        summary["degradation"] = degradation
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/workshop/predictive_maintaince/fm_work")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--datasets", default="cwru_de,mfpt_97k,sim_fleet")
    ap.add_argument("--no-fm", action="store_true",
                    help="baseline only, even if fm_*.npz exists")
    ap.add_argument("--no-degradation", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with_fm = not args.no_fm

    per_dataset: Dict[str, Dict[str, object]] = {}
    for name in [d for d in args.datasets.split(",") if d]:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name!r}; have {sorted(DATASETS)}")
        print(f"== {name} ==", flush=True)
        res = run_dataset(name, args.cache_dir, with_fm=with_fm)
        per_dataset[name] = res
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        base = res["aggregate"].get(f"{BASELINE_ARM}|{BASELINE_HEAD}")
        if base:
            print(f"   baseline pAUC@10% {base['pauc_10']['mean']:.3f}"
                  f"+-{base['pauc_10']['std']:.3f}  AUC {base['auc']['mean']:.3f}"
                  f"  TPR@thr {base['tpr_at_threshold']['mean']:.3f}"
                  f"  realised FPR {base['fpr_at_threshold']['mean']:.4f}")
        print(f"   {res['aggregate'].__len__()} (arm,head) combinations scored",
              flush=True)

    degradation = None
    if not args.no_degradation:
        print("== sim_degradation (detection delay) ==", flush=True)
        degradation = run_degradation(args.cache_dir, with_fm=with_fm)
        with open(os.path.join(out_dir, "sim_degradation.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(degradation, fh, indent=1, sort_keys=True)
        for arm, d in sorted(degradation["arms"].items()):
            print(f"   {arm:44s} delay {d['delay_days']} d")

    summary = summarise(per_dataset, degradation)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(f"\nwrote {out_dir}/summary.json")
    for name, e in summary["datasets"].items():
        b = e["baseline"]["pauc_10"]["mean"]
        if "best_challenger" in e:
            c = e["best_challenger"]
            print(f"  {name:10s} baseline pAUC {b:.3f}  best challenger "
                  f"{c['key']} {c['pauc_10']['mean']:.3f}  "
                  f"adopt={e['decision']['adopt']}")
        else:
            print(f"  {name:10s} baseline pAUC {b:.3f}  (no challenger arm run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
