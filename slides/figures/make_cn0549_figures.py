#!/usr/bin/env python3
"""CN0549 deck figures, read from the experiment JSON.

    python3 hardware/cn0549/tools/experiments.py --out hardware/cn0549/results
    python3 slides/figures/make_cn0549_figures.py --out slides/figures

Reads `hardware/cn0549/results/experiments.json` rather than recomputing, so a
slide cannot disagree with the measurement it illustrates. If the JSON is
missing the script says so and exits non-zero instead of drawing something
plausible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, WARN, GOOD = "#d03b3b", "#fab219", "#0ca30c"
ORDINAL = ["#86b6ef", "#3987e5", "#104281"]

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


def fig_noise_floor(data: dict, path: str) -> None:
    """Sensor noise against converter noise, per sensing chain.

    The figure that kills the bit-depth argument: every chain is sensor-limited
    by two to three orders of magnitude, so adding converter bits changes
    nothing about what can be detected.
    """
    chains = data["resolution"]["chains"]
    names = list(chains)
    sensor = np.array([chains[n]["noise_density_g"] * 1e6 for n in names])
    adc = np.array([chains[n]["adc_noise_density_g"] * 1e6 for n in names])

    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    y = np.arange(len(names))
    ax.barh(y - 0.19, sensor, height=0.34, color=S1, zorder=3, label="sensor noise")
    ax.barh(y + 0.19, adc, height=0.34, color=MUTED, zorder=3, label="converter noise")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{chains[n]['bits']}-bit, $\\pm${chains[n]['full_scale_g']:.0f} g"
                        for n in names], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("noise density [$\\mu$g/$\\sqrt{\\mathrm{Hz}}$, log]")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, n in enumerate(names):
        ratio = sensor[i] / adc[i]
        ax.text(sensor[i] * 1.25, i - 0.19, f"{sensor[i]:.0f}", va="center",
                fontsize=7.6, color=INK)
        ax.text(adc[i] * 1.25, i + 0.19, f"{adc[i]:.2f}   $\\rightarrow$ "
                f"sensor-limited {ratio:.0f}$\\times$", va="center",
                fontsize=7.6, color=CRIT if ratio > 50 else INK_2)
    ax.set_xlim(adc.min() * 0.4, sensor.max() * 40)
    ax.legend(loc="lower right", fontsize=7.4)
    ax.set_title("Every chain is sensor-limited: bit depth buys range, not resolution")
    fig.savefig(path)
    plt.close(fig)


def fig_detection(data: dict, path: str) -> None:
    """BPFO harmonic ratio vs severity, per chain."""
    r = data["resolution"]
    sevs = np.array(r["severities"])
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for colour, name in zip(ORDINAL, list(r["median"])):
        med = np.array(r["median"][name])
        sd = np.array(r["std"][name])
        ax.plot(sevs, med, color=colour, zorder=4, marker="o", markersize=3.5, label=name)
        ax.fill_between(sevs, med - sd, med + sd, color=colour, alpha=0.14, zorder=2)
    healthy = np.mean([r["median"][n][0] for n in r["median"]])
    ax.axhline(healthy, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.text(sevs[-1], healthy * 1.04, "healthy", fontsize=7.2, color=MUTED, ha="right")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("simulated outer-race severity (log)")
    ax.set_ylabel(f"{r['feature']}")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.4)
    det = r["detection_severity"]
    same = len(set(v for v in det.values() if v is not None)) <= 1
    ax.set_title("A 10$\\times$ quieter sensor does not lower the detection threshold"
                 if same else "Detection threshold by sensing chain")
    fig.savefig(path)
    plt.close(fig)


def fig_disconnected(data: dict, path: str) -> None:
    """Score distributions: healthy, disconnected, faulty -- and the diagnosis.

    Status colours are used here and only here, because these really are
    states: good / serious / critical. They are never reused as series colours.
    """
    d = data["disconnected"]
    scores = d["scores"]
    thr = d["threshold"]

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    groups = [("healthy machine", scores["healthy"], GOOD),
              ("disconnected sensor", scores["disconnected"], WARN),
              ("outer-race fault 0.35", scores["faulty"], CRIT)]
    for i, (label, vals, colour) in enumerate(groups):
        v = np.array(vals)
        ax.scatter(v, np.full_like(v, i) + np.random.default_rng(i).normal(0, 0.055, v.size),
                   s=22, color=colour, alpha=0.75, zorder=4, edgecolor=SURFACE, linewidth=0.6)
        ax.plot([np.median(v)], [i], marker="|", markersize=20, color=INK, zorder=5)
        ax.text(np.median(v), i + 0.30, f"median {np.median(v):.1f}", fontsize=7.4,
                color=INK, ha="center")
    ax.axvline(thr, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=3)
    ax.text(thr * 1.06, 2.42, f"alarm threshold {thr:.1f}\n(1 % FPR)", fontsize=7.4, color=INK)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("anomaly score (Mahalanobis, log)")
    ax.set_xlim(min(scores["healthy"]) * 0.5, max(scores["faulty"]) * 2.2)
    ax.set_ylim(-0.6, 2.75)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    dx = d.get("diagnosis_disconnected") or []
    claim = (f"diagnosed as {dx[0]['fault'].replace('_', ' ')} "
             f"at {dx[0]['confidence']:.0%} confidence" if dx else "")
    ax.set_title("A cut cable alarms like a fault, and is " + claim)
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--results", default=os.path.join(
        _REPO, "hardware", "cn0549", "results", "experiments.json"))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.results):
        print(f"no results at {args.results}. Run:\n"
              f"  python3 hardware/cn0549/tools/experiments.py", file=sys.stderr)
        return 1
    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    fig_noise_floor(data, os.path.join(args.out, "cn0549_noise_floor.pdf"))
    fig_detection(data, os.path.join(args.out, "cn0549_detection.pdf"))
    fig_disconnected(data, os.path.join(args.out, "cn0549_disconnected.pdf"))
    print("cn0549: wrote 3 figures from the measured JSON", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
