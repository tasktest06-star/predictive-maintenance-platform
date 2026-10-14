#!/usr/bin/env python3
"""Pi 5 gateway figures, read from the measured benchmark JSON.

    PYTHONPATH=core:edge python3 hardware/rpi5_hailo/tools/benchmark.py
    python3 slides/figures/make_rpi5_figures.py --out slides/figures

These are the only figures in the repository drawn from a stopwatch rather than
a cycle count, because the gateway tier runs on a general-purpose aarch64 Linux
box and so did the measurement. The host is named in the figure, because a
throughput number without its machine is unquotable.
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


def fig_latency(data: dict, path: str) -> None:
    """Measured latency per stage, log scale, against the acquisition it serves.

    The point of the log axis here: the stages span five orders of magnitude,
    and the figure's argument is precisely that the two the AI HAT could touch
    are at the bottom of it.
    """
    feats = {f"features {f['block_samples'] // 1024}k": float(f["median_ms"])
             for f in data["features"]}
    stages = {
        "frame decode": float(data["wire"]["decode"]["median_ms"]),
        "frame encode": float(data["wire"]["encode"]["median_ms"]),
        "score 1 frame": float(data["scoring"]["single"]["median_ms"]),
        "queue push (fsync)": float(data["queue"]["durable"]["ms_per_frame"]),
    }
    stages.update(feats)
    names = list(stages)
    vals = np.array([stages[n] for n in names])

    fig, ax = plt.subplots(figsize=(5.9, 2.6))
    colours = [CRIT if "score" in n else (S1 if "features" in n else MUTED) for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.6, color=colours, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("milliseconds, measured (log)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        label = f"{v * 1e3:.0f} $\\mu$s" if v < 1.0 else f"{v:.1f} ms"
        ax.text(v * 1.35, i, label, va="center", fontsize=7.6,
                color=CRIT if "score" in names[i] else INK)
    ax.set_xlim(vals.min() * 0.35, vals.max() * 12)
    host = data["host"].get("machine", "?")
    ax.set_title(f"Scoring a frame takes "
                 f"{float(data['scoring']['single']['median_ms']) * 1e3:.0f} $\\mu$s "
                 f"on one {host} core: the AI HAT has no job here")
    fig.savefig(path)
    plt.close(fig)


def fig_fanout(data: dict, path: str) -> None:
    """Nodes per core vs reporting interval, with the real constraint named."""
    fans = data["fan_out"]
    intervals = np.array([f["interval_s"] for f in fans]) / 60.0
    nodes = np.array([f["nodes"] for f in fans], dtype=float)

    fig, ax = plt.subplots(figsize=(5.9, 2.4))
    ax.plot(intervals, nodes, color=S1, marker="o", markersize=5, zorder=4,
            label="one core, 75 % offered")
    ax.plot(intervals, nodes * 4, color=ORDINAL[0], marker="s", markersize=4,
            zorder=3, ls=(0, (4, 2)), label="four cores (a Pi 5 has four)")
    for ref, label in ((50, "a typical plant pilot: 50 nodes"),
                       (1000, "a large site: 1000 nodes")):
        ax.axhline(ref, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
        ax.text(intervals[0] * 1.05, ref * 1.25, label, fontsize=7.2, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("node reporting interval [min, log]")
    ax.set_ylabel("nodes the CPU can keep up with [log]")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.4)
    hourly = [f for f in fans if f["interval_s"] == 3600.0]
    n = hourly[0]["nodes"] if hourly else 0
    ax.set_title(f"Compute is not the constraint: {n:,} hourly nodes on one core")
    fig.savefig(path)
    plt.close(fig)


def fig_holdover(data: dict, path: str) -> None:
    """Backlog flush time per uplink, and what actually sizes the UPS."""
    rows = data["uplinks"]
    names = [r["uplink"] for r in rows]
    secs = np.array([float(r["flush_s"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.9, 2.3))
    y = np.arange(len(names))
    colours = [CRIT if s > 600 else (WARN if s > 60 else GOOD) for s in secs]
    ax.barh(y, secs, height=0.58, color=colours, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds to flush the backlog (log)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, s in enumerate(secs):
        label = f"{s:.1f} s" if s < 90 else f"{s / 60:.0f} min"
        ax.text(s * 1.35, i, label, va="center", fontsize=7.6, color=INK)
    rec = float(data["recommended_holdover_s"])
    ax.axvline(rec, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=5)
    ax.annotate(f"recommended holdover {rec:.0f} s\n"
                f"({float(data['recommended_wh']):.2f} Wh, a supercapacitor)",
                xy=(rec, len(names) - 0.55), xytext=(rec * 1.4, len(names) - 0.15),
                fontsize=7.4, color=INK, va="center", ha="left")
    ax.set_xlim(secs.min() * 0.3, secs.max() * 18)
    ax.set_ylim(len(names) + 0.1, -0.7)
    ax.set_title(f"{data['nodes']} nodes x {data['outage_days']:.0f}-day outage = "
                 f"{data['backlog_mb']:.0f} MB; the flush is not what needs battery")
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--results", default=os.path.join(
        _REPO, "hardware", "rpi5_hailo", "results"))
    args = ap.parse_args(argv)

    bench_path = os.path.join(args.results, "benchmark.json")
    hold_path = os.path.join(args.results, "holdover.json")
    for p in (bench_path, hold_path):
        if not os.path.isfile(p):
            print(f"missing {p}. Run the tools in hardware/rpi5_hailo/tools/ first.",
                  file=sys.stderr)
            return 1
    with open(bench_path, encoding="utf-8") as fh:
        bench = json.load(fh)
    with open(hold_path, encoding="utf-8") as fh:
        hold = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    fig_latency(bench, os.path.join(args.out, "rpi5_latency.pdf"))
    fig_fanout(bench, os.path.join(args.out, "rpi5_fanout.pdf"))
    fig_holdover(hold, os.path.join(args.out, "rpi5_holdover.pdf"))
    print("rpi5: wrote 3 figures from the measured JSON", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
