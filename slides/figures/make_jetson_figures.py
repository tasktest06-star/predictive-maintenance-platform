#!/usr/bin/env python3
"""Jetson figures, read from the workload study's JSON.

    PYTHONPATH=core:edge python3 hardware/jetson_orin/tools/workload_study.py
    python3 slides/figures/make_jetson_figures.py --out slides/figures

Two figures:

    jetson_headroom.pdf   what each candidate workload needs against what one
                          CPU core supplies -- the figure that argues against
                          the board it is in a deck for
    jetson_gate.pdf       the three-window training split, and the two bugs the
                          two-window version hid
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
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

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


def fig_headroom(data: dict, path: str) -> None:
    """Required versus supplied, in the natural units of each candidate.

    A grouped log bar chart rather than a ratio, because the ratios span nine
    orders of magnitude and a reader needs to see the actual quantities to
    believe them.
    """
    c1 = data["candidate_1_rescoring"]
    c2 = max(data["candidate_2_training"]["fits"], key=lambda r: r["n_frames"])
    c3 = data["candidate_3_waveform"]

    items = [
        ("re-score a site-year\n(1000 nodes x 365 d)",
         float(c1["full_history_s"]), 600.0, "seconds"),
        (f"fit the detectors\n({c2['n_frames']} frames)",
         float(c2["mahalanobis_ms"]) / 1000.0, 600.0, "seconds"),
        ("raw-waveform inference\n(1000 nodes hourly)",
         float(c3["gmac_per_s_required"]), float(c3["cpu_gmac_per_s_realistic"]),
         "GMAC/s"),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    y = np.arange(len(items))
    need = np.array([i[1] for i in items])
    have = np.array([i[2] for i in items])
    ax.barh(y - 0.19, need, height=0.34, color=S1, zorder=3, label="what the workload needs")
    ax.barh(y + 0.19, have, height=0.34, color=MUTED, zorder=3,
            label="what one CPU core supplies")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds, or GMAC/s (log) --- note the units differ per row")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, (label, n, h, unit) in enumerate(items):
        ax.text(n * 1.4, i - 0.19, f"{n:,.4g} {unit}", va="center", fontsize=7.4, color=INK)
        margin = h / n if n > 0 else float("inf")
        ax.text(h * 1.4, i + 0.19,
                f"{h:,.4g} {unit}   $\\rightarrow$ {margin:,.0f}$\\times$ headroom",
                va="center", fontsize=7.4, color=GOOD, fontweight="bold")
    ax.set_xlim(need.min() * 0.25, have.max() * 300)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              fontsize=7.4)
    ax.set_title("Nothing in this platform needs a GPU: three candidates, three "
                 "negative results")
    fig.savefig(path)
    plt.close(fig)


def fig_gate(path: str) -> None:
    """The three-window split, and why two windows hid two bugs.

    Hand-drawn schematic rather than data: the point is the *structure* of the
    split, and a bar chart of it would be a chart of nothing.
    """
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    # Two-window version, on top, with the circular check marked.
    ax.add_patch(plt.Rectangle((0, 2.1), 75, 0.52, color=ORDINAL[1], zorder=3))
    ax.add_patch(plt.Rectangle((75, 2.1), 25, 0.52, color=WARN, zorder=3))
    ax.text(37, 2.36, "fit", ha="center", va="center", color="white",
            fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(87.5, 2.36, "calibrate\n+ check", ha="center", va="center", color=INK,
            fontsize=7.2, fontweight="bold", zorder=4)
    ax.text(0, 2.78, "The two-window version, which was wrong:", fontsize=8.5,
            fontweight="bold", color=CRIT)
    ax.annotate("", xy=(99, 2.06), xytext=(76, 2.06),
                arrowprops=dict(arrowstyle="<->", color=CRIT, lw=1.3))
    ax.text(100, 1.92, "calibrated AND verified on the same window: the rate\n"
            "equals the target by construction", ha="right", va="top",
            fontsize=7.0, color=CRIT)

    # Three-window version.
    ax.add_patch(plt.Rectangle((0, 0.70), 50, 0.52, color=ORDINAL[1], zorder=3))
    ax.add_patch(plt.Rectangle((50, 0.70), 25, 0.52, color=ORDINAL[0], zorder=3))
    ax.add_patch(plt.Rectangle((75, 0.70), 25, 0.52, color=GOOD, zorder=3))
    ax.text(25, 0.96, "fit", ha="center", va="center", color="white",
            fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(62.5, 0.96, "calibrate", ha="center", va="center", color=INK,
            fontsize=7.6, fontweight="bold", zorder=4)
    ax.text(87.5, 0.96, "verify", ha="center", va="center", color="white",
            fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(0, 1.40, "The three-window version, split by TIME not at random:",
            fontsize=8.5, fontweight="bold", color=GOOD)
    ax.text(87.5, 0.55, "a later window the threshold\nhas never seen",
            ha="center", va="top", fontsize=7.0, color=INK_2)
    ax.text(0, 0.20, "and the gate is a 3-sigma POISSON band on the alarm COUNT, not a "
            "multiple of the rate:\n"
            "on 175 frames at a 1 % target, 1.8 alarms are expected and noise alone "
            "spans 0-6.", fontsize=7.2, color=INK_2, va="top")
    ax.annotate("time", xy=(100, 0.36), xytext=(0, 0.36), fontsize=7.4, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9), va="center")
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--results", default=os.path.join(
        _REPO, "hardware", "jetson_orin", "results", "workload_study.json"))
    args = ap.parse_args(argv)

    if not os.path.isfile(args.results):
        print(f"missing {args.results}; run tools/workload_study.py first",
              file=sys.stderr)
        return 1
    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    fig_headroom(data, os.path.join(args.out, "jetson_headroom.pdf"))
    fig_gate(os.path.join(args.out, "jetson_gate.pdf"))
    print("jetson: wrote 2 figures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
