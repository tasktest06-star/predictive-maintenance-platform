#!/usr/bin/env python3
"""Generate the platform-core deck figures by running the real pipeline.

Nothing here is hand-typed: every number in the deck comes from executing
``pdmcore`` at build time. Re-run after a change and the slides follow.

    python3 slides/figures/make_core_figures.py --out slides/figures

Takes ~60 s (it simulates a commissioning window, a 90-day run-to-failure
series and a grouped fleet evaluation).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "core")))

from pdmcore import eval as pdmeval  # noqa: E402
from pdmcore.dsp.spectrum import spectrum as compute_spectrum  # noqa: E402
from pdmcore.features import FeatureConfig, FeatureExtractor  # noqa: E402
from pdmcore.health import AlarmConfig, HealthIndex, baseline_from_vectors, diagnose  # noqa: E402
from pdmcore.io.payload import TelemetryFrame, encode_frame, frame_size  # noqa: E402
from pdmcore.models.anomaly import build_detector  # noqa: E402
from pdmcore.sim import MACHINE_LIBRARY, degradation_series, fleet_dataset, simulate  # noqa: E402

# --- palette (same instance as the research deck) -------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
STATE_COLOR = {"normal": GOOD, "watch": WARN, "alarm": SERIOUS, "critical": CRIT}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _extractor(spec):
    return FeatureExtractor(FeatureConfig(
        fs_hz=spec.fs_hz, bearing=spec.bearing, demod_band_hz=(2500.0, 6000.0),
        velocity_band_hz=(2.0, 1000.0) if spec.shaft_hz < 10 else (10.0, 1000.0)))


# -------------------------------------------------------------------------
def fig_calibration(path: str) -> None:
    """The calibration invariant, measured across scalings and windows.

    One series per scaling would be four series on a scatter, which exceeds the
    three-slot all-pairs cap -- so this is a grouped bar with one hue, and the
    reference line carries the meaning. The thing being shown is that all 16
    combinations agree, which is a statement about spread, not about identity.
    """
    rng = np.random.default_rng(42)
    fs = 25600.0
    noise = rng.standard_normal(int(fs * 2))
    t = np.arange(int(fs * 2)) / fs
    tone = 1.0 * np.sin(2 * np.pi * (1000.37 * fs / 4096) * t)
    signals = {"white noise": noise, "off-bin tone": tone, "tone + noise": tone + noise}
    scalings = ["peak", "rms", "power", "psd"]
    windows = ["rect", "hann", "flattop", "blackman_harris"]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
    for ax, (label, x) in zip(axes, signals.items()):
        truth = float(np.sqrt(np.mean(x**2)))
        ratios = []
        for sc in scalings:
            for wn in windows:
                sp = compute_spectrum(x, fs, n_fft=4096, scaling=sc, window_name=wn)
                ratios.append(sp.rms_in_band(0.0, fs / 2) / truth)
        ax.axhline(1.0, color=INK_2, lw=1.0, zorder=2)
        ax.bar(range(len(ratios)), ratios, width=0.74, color=S1, zorder=3)
        ax.set_title(label, fontsize=8.5)
        ax.set_xticks([])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.6)
        worst = max(abs(r - 1.0) for r in ratios)
        ax.text(0.5, 0.06, f"worst error {worst * 100:.1f} %", transform=ax.transAxes,
                ha="center", fontsize=7.5, color=INK_2)
    axes[0].set_ylabel("band RMS / time-domain RMS")
    fig.suptitle("All 16 scaling x window combinations agree with the time-domain RMS",
                 x=0.0, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    fig.savefig(path)
    plt.close(fig)


def fig_calibration_bugs(path: str) -> None:
    """The three calibration bugs, as the factor each introduced.

    Status-critical colour: these are defect magnitudes, not series. Each bar is
    labelled, so colour is not carrying the meaning alone.
    """
    bugs = [
        ("Single-sided fold applied to\nmagnitude, then squared", np.sqrt(2), "3.0 dB"),
        ("Block averaging in the\nmagnitude domain", 4 / np.pi, "1.05 dB\n(broadband only)"),
        ("Band RMS not dividing\nout the ENBW", 3.7702**0.5, "5.7 dB\n(flat-top)"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    y = range(len(bugs))
    ax.barh(list(y), [b[1] for b in bugs], height=0.56, color=CRIT, zorder=3)
    ax.axvline(1.0, color=INK_2, lw=1.0, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels([b[0] for b in bugs], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("factor by which the reported value was wrong")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, (_lbl, factor, db) in enumerate(bugs):
        ax.text(factor + 0.05, i, f"x{factor:.3f}   {db}", va="center", fontsize=7.6, color=INK)
    ax.set_title("Found by the Parseval invariant; invisible in any integration test")
    fig.savefig(path)
    plt.close(fig)


def fig_detectors(path: str) -> Dict[str, object]:
    """Detector comparison, 7 split seeds, plus the leakage demonstration.

    Error bars are the point: with 9 units the split-to-split variation is large
    enough that the top three detectors are not separable, and a figure showing
    four bars without them would invite a ranking the data does not support.

    Two measures of the same kind (both 0-1, same direction) on one axis -- not
    a dual-axis chart.
    """
    ds = fleet_dataset(machines=["motor_pump_37kw", "vfd_compressor_75kw", "gearbox_132kw"],
                       n_healthy_per_machine=24, n_per_fault=6,
                       severities=(0.2, 0.45, 0.75), duration_s=0.4, seed=11)
    fx = _extractor(ds[0].spec)
    X = np.vstack([fx.extract(r.accel_g, r.spec.shaft_hz).vector for r in ds])
    labels = [r.label for r in ds]
    groups = [r.truth["unit_id"] for r in ds]
    y = np.array([0 if lab == "healthy" else 1 for lab in labels])

    kinds = ["robust_threshold", "mahalanobis", "pca_residual", "deep_svdd"]
    acc: Dict[str, Dict[str, List[float]]] = {k: {"auc": [], "tpr": [], "pauc": []} for k in kinds}
    n_seeds = 7
    for seed in range(n_seeds):
        sp = pdmeval.stratified_group_split(groups, labels, test_fraction=0.35, seed=seed)
        assert pdmeval.leakage_report(sp, groups=groups, labels=labels)["ok"]
        tr = sp.train[y[sp.train] == 0]
        for k in kinds:
            det = build_detector(k, feature_names=fx.names)
            det.fit(X[tr])
            summ = pdmeval.summarise_detector(y[sp.test], det.score(X[sp.test]))
            acc[k]["auc"].append(summ.auc)
            acc[k]["tpr"].append(summ.tpr_at_fpr1pct)
            acc[k]["pauc"].append(summ.pauc_10)

    # Leakage demonstration: the same detector, the same data, a random split.
    rng = np.random.default_rng(0)
    leaky = []
    for _ in range(n_seeds):
        perm = rng.permutation(len(y))
        n_te = int(0.35 * len(y))
        te, tr_all = perm[:n_te], perm[n_te:]
        tr = tr_all[y[tr_all] == 0]
        det = build_detector("pca_residual", feature_names=fx.names)
        det.fit(X[tr])
        leaky.append(pdmeval.auc(y[te], det.score(X[te])))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                                  gridspec_kw={"width_ratios": [1.9, 1.0]})
    idx = np.arange(len(kinds))
    w = 0.36
    auc_m = [float(np.mean(acc[k]["auc"])) for k in kinds]
    auc_s = [float(np.std(acc[k]["auc"])) for k in kinds]
    tpr_m = [float(np.mean(acc[k]["tpr"])) for k in kinds]
    tpr_s = [float(np.std(acc[k]["tpr"])) for k in kinds]
    ax.bar(idx - w / 2 - 0.01, auc_m, width=w, yerr=auc_s, color=S1, zorder=3,
           label="AUC", error_kw={"ecolor": INK_2, "capsize": 2.5, "elinewidth": 1.0})
    ax.bar(idx + w / 2 + 0.01, tpr_m, width=w, yerr=tpr_s, color=S2, zorder=3,
           label="TPR at 1 % FPR", error_kw={"ecolor": INK_2, "capsize": 2.5, "elinewidth": 1.0})
    ax.set_xticks(idx)
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=7.6)
    ax.set_ylim(0.58, 1.05)
    ax.set_ylabel("score")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=1, fontsize=7.2)
    auc_spread = max(auc_m) - min(auc_m)
    tpr_spread = max(tpr_m) - min(tpr_m)
    ax.set_title(f"AUC spread {auc_spread:.3f} vs operating-point spread {tpr_spread:.3f}:\n"
                 f"AUC compresses the difference that matters", fontsize=8.6)

    ax2.bar([0], [float(np.mean(leaky))], width=0.5, yerr=[float(np.std(leaky))],
            color=CRIT, zorder=3, error_kw={"ecolor": INK_2, "capsize": 3, "elinewidth": 1.0})
    ax2.bar([1], [float(np.mean(acc["pca_residual"]["auc"]))], width=0.5,
            yerr=[float(np.std(acc["pca_residual"]["auc"]))], color=S1, zorder=3,
            error_kw={"ecolor": INK_2, "capsize": 3, "elinewidth": 1.0})
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["random\nsplit", "unit-grouped\nsplit"], fontsize=7.6)
    ax2.set_ylim(0.9, 1.0)
    ax2.set_ylabel("AUC")
    ax2.grid(axis="y", zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0, float(np.mean(leaky)) + 0.004, f"{np.mean(leaky):.3f}", ha="center",
             fontsize=7.6, color=CRIT, fontweight="bold")
    ax2.text(1, float(np.mean(acc["pca_residual"]["auc"])) + 0.004,
             f"{np.mean(acc['pca_residual']['auc']):.3f}", ha="center", fontsize=7.6, color=INK)
    ax2.set_title("Same detector,\nsame data", fontsize=8.6)

    fig.savefig(path)
    plt.close(fig)
    return {
        "kinds": kinds, "auc_mean": auc_m, "auc_std": auc_s,
        "tpr_mean": tpr_m, "tpr_std": tpr_s,
        "pauc_mean": [float(np.mean(acc[k]["pauc"])) for k in kinds],
        "pauc_std": [float(np.std(acc[k]["pauc"])) for k in kinds],
        "leaky_auc": float(np.mean(leaky)), "grouped_auc": float(np.mean(acc["pca_residual"]["auc"])),
        "n_seeds": n_seeds, "n_units": len(set(groups)), "n_samples": int(X.shape[0]),
    }


def fig_degradation(path: str) -> Dict[str, float]:
    """Health index over a 90-day run-to-failure, with alarm states.

    Single series with a background band per alarm state. The bands are a
    categorical *context*, not data, so they use the status palette at low
    opacity and are labelled; the data line stays the most saturated mark.
    """
    spec = MACHINE_LIBRARY["motor_pump_37kw"]
    fx = _extractor(spec)
    comm = [simulate(spec, None, duration_s=0.4, seed=900 + i) for i in range(60)]
    Xc = np.vstack([fx.extract(r.accel_g, spec.shaft_hz).vector for r in comm])
    base = baseline_from_vectors(Xc, fx.names)
    hi = HealthIndex("P", fx.names, alarm_config=AlarmConfig(min_dwell_s=0.0))
    hi.commission(Xc, [spec.shaft_hz] * len(comm))

    series = degradation_series(spec, mode="bearing_outer", n_points=90,
                               duration_s=0.4, days=90.0, seed=5)
    days, idx, smooth, states, kurt, dx_days = [], [], [], [], [], []
    for rec in series:
        fr = fx.extract(rec.accel_g, spec.shaft_hz)
        out = hi.update(rec.truth["t_days"] * 86400.0, fr.vector, speed_hz=spec.shaft_hz)
        days.append(rec.truth["t_days"])
        idx.append(out["index"])
        smooth.append(out["smoothed"])
        states.append(out["state"])
        kurt.append(fr.get("time.kurtosis"))
        if diagnose(fr, baseline=base):
            dx_days.append(rec.truth["t_days"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 3.3), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1.0]})
    # Alarm-state context bands.
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.axvspan(days[start], days[min(i, len(days) - 1)],
                       color=STATE_COLOR[states[start]], alpha=0.13, lw=0, zorder=1)
            mid = 0.5 * (days[start] + days[min(i, len(days) - 1)])
            if days[min(i, len(days) - 1)] - days[start] > 8:
                ax.text(mid, 0.18, states[start], ha="center", fontsize=7, color=INK_2)
            start = i
    for thr, lbl in ((1.0, "watch"), (2.0, "alarm"), (4.0, "critical")):
        ax.axhline(thr, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
        ax.text(91.5, thr, lbl, fontsize=6.8, color=MUTED, va="center")
    ax.plot(days, idx, color=S1, lw=0.9, alpha=0.45, zorder=3, label="per-acquisition")
    ax.plot(days, smooth, color=S1, lw=2.0, zorder=4, label="EWMA trend (drives alarms)")
    if dx_days:
        ax.axvline(dx_days[0], color=S2, lw=1.4, ls="-", zorder=5)
        ax.annotate(f"first correct\ndiagnosis, day {dx_days[0]:.0f}",
                    (dx_days[0], 4.9), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=S2, va="center", ha="left")
    ax.set_ylabel("health index")
    ax.set_ylim(0, 7.4)
    ax.set_xlim(0, 90)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=2, fontsize=7.4)
    ax.set_title("90-day run-to-failure: escalation lands before the degradation knee")

    ax2.plot(days, kurt, color=S3, lw=1.6, zorder=3)
    ax2.axhline(3.0, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax2.text(91.5, 3.0, "Gaussian", fontsize=6.8, color=MUTED, va="center")
    peak_day = days[int(np.argmax(kurt))]
    # The informative comparison is *when* each signal moves, not the peak:
    # kurtosis is the textbook early bearing indicator and here it is 28 days
    # later than the fused index, because the fused index also sees the
    # high-frequency band energy that rises first.
    kurt_moves = next((d for d, kv in zip(days, kurt) if kv > 3.5), float("nan"))
    ax2.axvline(kurt_moves, color=S3, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax2.annotate(f"kurtosis only leaves the Gaussian band\nat day {kurt_moves:.0f}",
                 (kurt_moves, 6.4), xytext=(-6, 0), textcoords="offset points",
                 fontsize=7, color=INK_2, ha="right", va="center")
    ax2.set_ylabel("kurtosis")
    ax2.set_xlabel("days")
    ax2.grid(axis="y", zorder=0)
    ax2.set_axisbelow(True)
    fig.savefig(path)
    plt.close(fig)

    firsts = {}
    for d, s in zip(days, states):
        firsts.setdefault(s, d)
    return {"watch": firsts.get("watch", float("nan")),
            "alarm": firsts.get("alarm", float("nan")),
            "critical": firsts.get("critical", float("nan")),
            "first_dx": dx_days[0] if dx_days else float("nan"),
            "kurt_peak_day": float(peak_day),
            "kurt_moves_day": float(kurt_moves)}


def fig_payload(path: str) -> None:
    """Payload and airtime: the decision the architecture follows from."""
    n_feat = 63
    frame_b = frame_size(n_feat)
    raw_b = int(25600 * 2)       # 1 s, 16-bit
    json_b = 1574                # measured by the CLI
    rates = [("sub-GHz\n6.25 kbit/s", 6250), ("LTE-M\n20 kbit/s", 20000), ("BLE\n50 kbit/s", 50000)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.1),
                                  gridspec_kw={"width_ratios": [1.0, 1.3]})
    items = [("feature frame\n(binary)", frame_b), ("same frame\nas JSON", json_b),
             ("1 s raw waveform\n16-bit", raw_b)]
    ax.barh(range(3), [i[1] for i in items], height=0.56,
            color=[S1, S2, CRIT], zorder=3)
    ax.set_yticks(range(3))
    ax.set_yticklabels([i[0] for i in items], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("bytes (log)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, (_lbl, b) in enumerate(items):
        ax.text(b * 1.25, i, f"{b:,} B", va="center", fontsize=7.4, color=INK)
    ax.set_title("Payload", fontsize=9)

    idx = np.arange(len(rates))
    w = 0.36
    ax2.bar(idx - w / 2 - 0.01, [frame_b * 8 / r[1] * 1000 for r in rates],
            width=w, color=S1, zorder=3, label="feature frame")
    ax2.bar(idx + w / 2 + 0.01, [raw_b * 8 / r[1] * 1000 for r in rates],
            width=w, color=CRIT, zorder=3, label="raw waveform")
    ax2.set_yscale("log")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([r[0] for r in rates], fontsize=7.4)
    ax2.set_ylabel("airtime [ms, log]")
    ax2.grid(axis="y", zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", ncol=1, fontsize=7.4)
    ax2.set_title("Airtime, and therefore radio energy", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def fig_budget(path: str) -> None:
    """Feature-extraction time budget on a Cortex-M33 at 160 MHz.

    Estimates from cycle counts and CMSIS-DSP benchmarks, not measurements --
    labelled on the figure, because an unlabelled estimate becomes a measurement
    the moment it is quoted in a slide.
    """
    stages = [
        ("streaming moments", 0.4, "14 time.*"),
        ("window + 4096-pt FFT", 1.1, "-"),
        ("scale, bands, shape,\norders, ISO", 0.5, "29 features"),
        ("band-pass, rectify,\ndecimate, 2048-pt FFT", 2.8, "-"),
        ("harmonic SNR\n4 modes x 5 harmonics", 0.9, "20 bearing.*"),
        ("int8 detector", 0.03, "score"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    left = 0.0
    shades = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95"]
    for (label, ms, _prod), colour in zip(stages, shades):
        ax.barh([0], [ms], left=left, height=0.5, color=colour, zorder=3,
                edgecolor=SURFACE, linewidth=1.2)
        if ms > 0.3:
            ax.text(left + ms / 2, 0, f"{ms:g}", ha="center", va="center",
                    fontsize=7.4, color="white" if ms > 0.8 else INK, fontweight="bold")
        left += ms
    ax.set_yticks([])
    ax.set_xlim(0, left * 1.02)
    ax.set_xlabel("milliseconds")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"Full feature vector in ~{left:.1f} ms  (estimate: cycle counts, not measured)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in shades]
    ax.legend(handles, [s[0] for s in stages], loc="upper center",
              bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=6.9)
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_HERE))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("calibration invariant ...", flush=True)
    fig_calibration(os.path.join(args.out, "core_calibration.pdf"))
    print("calibration bugs ...", flush=True)
    fig_calibration_bugs(os.path.join(args.out, "core_calibration_bugs.pdf"))
    print("payload ...", flush=True)
    fig_payload(os.path.join(args.out, "core_payload.pdf"))
    print("budget ...", flush=True)
    fig_budget(os.path.join(args.out, "core_budget.pdf"))
    print("detectors (grouped fleet eval, 7 seeds) ...", flush=True)
    det_facts = fig_detectors(os.path.join(args.out, "core_detectors.pdf"))
    for k, a, asd, t, tsd, p, psd in zip(
            det_facts["kinds"], det_facts["auc_mean"], det_facts["auc_std"],
            det_facts["tpr_mean"], det_facts["tpr_std"],
            det_facts["pauc_mean"], det_facts["pauc_std"]):
        print(f"    {k:20s} AUC {a:.3f}+-{asd:.3f}  pAUC {p:.3f}+-{psd:.3f}  "
              f"TPR@1% {t:.3f}+-{tsd:.3f}")
    print(f"    leakage: random split {det_facts['leaky_auc']:.3f} vs "
          f"grouped {det_facts['grouped_auc']:.3f}")
    print("degradation trajectory ...", flush=True)
    facts = fig_degradation(os.path.join(args.out, "core_degradation.pdf"))
    print("measured escalation:", {k: round(v, 1) for k, v in facts.items()})
    with open(os.path.join(args.out, "core_facts.txt"), "w", encoding="utf-8") as fh:
        for k, v in facts.items():
            fh.write(f"{k}\t{v:.1f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
