#!/usr/bin/env python3
"""Per-kit deck figures, computed from the shared energy model.

    python3 slides/figures/make_kit_figures.py --kit stwinbx1 --out slides/figures

One generator for all eight kits so the figures cannot disagree with
`tools/power_budget.py`, and so a corrected datasheet figure propagates into
every deck. Emits:

    kit_<key>_energy.pdf     per-cycle energy breakdown, which term dominates
    kit_<key>_battery.pdf    battery life vs interval, with the self-discharge
                             floor drawn explicitly
    kit_<key>_timing.pdf     compute budget against the acquisition window

The battery figure draws the self-discharge floor because on several kits that
floor, not the circuit, sets the replacement interval -- and a battery-life
curve without it invites an optimisation that cannot work.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "tools")))

import power_budget  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, WARN = "#d03b3b", "#fab219"
#: Validated ordinal blue ramp; the light end clears 2:1 against the surface.
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

#: Per-kit compute budget, milliseconds. Estimates from cycle counts and vendor
#: benchmarks -- labelled as such on every figure that uses them, because an
#: unlabelled estimate becomes a measurement the moment it appears on a slide.
TIMING_MS: Dict[str, List] = {
    "stwinbx1":  [("moments", 0.8), ("3x FFT 4096", 3.3), ("bands/shape/orders", 0.5),
                  ("envelope chain", 2.8), ("bearing SNR", 0.9), ("int8 model", 0.03)],
    "esp32s3":   [("moments", 0.6), ("3x FFT 4096", 2.1), ("bands/shape/orders", 0.4),
                  ("envelope chain", 1.9), ("bearing SNR", 0.7), ("int8 model", 0.02)],
    "nrf9160":   [("moments", 2.0), ("3x FFT 4096", 8.2), ("bands/shape/orders", 1.2),
                  ("envelope chain", 7.0), ("bearing SNR", 2.2), ("int8 model", 0.08)],
    "alif_e7":   [("moments", 0.4), ("3x FFT 4096 (Helium)", 1.1), ("bands/shape/orders", 0.3),
                  ("envelope chain", 1.0), ("bearing SNR", 0.4), ("int8 model (NPU)", 0.002)],
    # Three axes, so the whole pipeline runs three times out of one burst. The
    # per-axis figures are the Helium ones; the entries below are x3, plus the
    # angle-domain resampling pass that no other kit runs.
    "ra8d1":     [("moments x3", 0.6), ("9x FFT 4096 (Helium)", 1.8), ("bands/shape/orders x3", 0.45),
                  ("envelope chain x3", 1.5), ("bearing SNR x3", 0.6), ("int8 model x3", 0.03),
                  ("angular resample", 0.9)],
}


def fig_energy(kit: power_budget.Kit, path: str, interval_s: float) -> None:
    """Per-cycle energy by term. Ordinal ramp: the terms are a sequence in time."""
    if kit.mains_powered:
        return
    terms = kit.energy_per_cycle_mas()
    order = ["settle", "acquire", "compute", "radio"]
    values = [terms[k] for k in order]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    y = range(len(values))
    dominant = max(terms, key=lambda k: terms[k])
    colors = [CRIT if k == dominant else ORDINAL[i] for i, k in enumerate(order)]
    ax.barh(list(y), values, height=0.58, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mA$\\cdot$s per acquisition cycle (log)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(values):
        ax.text(v * 1.3, i, f"{v:.3g}   {100 * v / total:.1f} %", va="center",
                fontsize=7.8, color=CRIT if order[i] == dominant else INK,
                fontweight="bold" if order[i] == dominant else "normal")
    ax.set_xlim(min(v for v in values if v > 0) * 0.4, total * 6)
    ax.set_title(f"{dominant} dominates: {100 * terms[dominant] / total:.0f} % of "
                 f"{total:.2f} mA$\\cdot$s per cycle")
    fig.savefig(path)
    plt.close(fig)


def fig_battery(kit: power_budget.Kit, path: str) -> None:
    """Battery life vs interval, with the self-discharge floor drawn.

    The floor is the point of the figure on several kits: where it sits above
    the load curve, no firmware change can improve battery life and the only
    lever is the cell.
    """
    if kit.mains_powered:
        return
    intervals = np.geomspace(30.0, 86400.0, 60)
    load_ua = np.array([kit.average_current_ua(float(iv)) for iv in intervals])
    life = np.array([kit.battery_life_years(float(iv)) for iv in intervals])
    sd_ua = kit.battery_mah * (kit.self_discharge_pct_month / 100.0) / (30.0 * 24.0) * 1000.0
    life_no_sd = kit.battery_mah / (load_ua / 1000.0) / 24.0 / 365.25

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    ax.plot(intervals / 60.0, load_ua, color=S1, zorder=4, label="circuit load")
    ax.axhline(sd_ua, color=CRIT, lw=1.6, ls=(0, (4, 2)), zorder=3,
               label=f"cell self-discharge ({sd_ua:.0f} $\\mu$A)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("acquisition interval [min, log]")
    ax.set_ylabel("current [$\\mu$A, log]")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7.2)
    ax.set_title("Load vs the floor", fontsize=9)

    ax2.plot(intervals / 60.0, life, color=S1, zorder=4, label="with self-discharge")
    ax2.plot(intervals / 60.0, life_no_sd, color=MUTED, lw=1.2, ls=(0, (3, 2)),
             zorder=3, label="load only")
    ax2.set_xscale("log")
    ax2.set_xlabel("acquisition interval [min, log]")
    ax2.set_ylabel("battery life [years]")
    ax2.grid(zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", fontsize=7.2)
    cap = float(life[-1])
    ax2.axhline(cap, color=CRIT, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax2.text(1.2, cap * 1.04, f"saturates at {cap:.1f} y", fontsize=7.2, color=CRIT)
    ax2.set_title(f"{kit.battery_name}", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def fig_timing(key: str, kit: power_budget.Kit, path: str) -> None:
    """Compute budget against the acquisition window."""
    stages = TIMING_MS.get(key)
    if not stages:
        return
    compute = sum(s[1] for s in stages)
    acquire = kit.acquire_ms
    settle = kit.settle_ms

    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    left = 0.0
    shades = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95"]
    for (label, ms), colour in zip(stages, shades):
        ax.barh([1], [ms], left=left, height=0.46, color=colour, zorder=3,
                edgecolor=SURFACE, linewidth=1.0)
        left += ms
    ax.barh([0], [settle], height=0.46, color=WARN, zorder=3, edgecolor=SURFACE, linewidth=1.0)
    ax.barh([0], [acquire], left=settle, height=0.46, color=CRIT, zorder=3,
            edgecolor=SURFACE, linewidth=1.0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"sense\n{settle + acquire:.0f} ms", f"compute\n{compute:.1f} ms"],
                       fontsize=8)
    ax.set_xlabel("milliseconds")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.text(settle + acquire + 8, 0, f"settle {settle:.0f} + acquire {acquire:.0f} ms",
            va="center", fontsize=7.4, color=INK)
    ax.text(compute + 8, 1, " + ".join(f"{s[0]} {s[1]:g}" for s in stages[:3]) + " ...",
            va="center", fontsize=6.8, color=INK_2)
    ax.set_xlim(0, (settle + acquire) * 1.55)
    ax.set_title(f"Compute is {100 * compute / (settle + acquire):.1f} % of the "
                 f"sensing window  (estimate, not measured)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in shades[:len(stages)]]
    ax.legend(handles, [s[0] for s in stages], loc="upper center",
              bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=6.8)
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", required=True, help=f"one of {', '.join(power_budget.KITS)}")
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--interval-s", type=float, default=3600.0)
    args = ap.parse_args(argv)

    if args.kit not in power_budget.KITS:
        raise SystemExit(f"unknown kit {args.kit!r}; choose from {', '.join(power_budget.KITS)}")
    kit = power_budget.KITS[args.kit]
    os.makedirs(args.out, exist_ok=True)

    made = []
    for name, fn in (
        (f"kit_{args.kit}_energy.pdf", lambda p: fig_energy(kit, p, args.interval_s)),
        (f"kit_{args.kit}_battery.pdf", lambda p: fig_battery(kit, p)),
        (f"kit_{args.kit}_timing.pdf", lambda p: fig_timing(args.kit, kit, p)),
    ):
        out = os.path.join(args.out, name)
        fn(out)
        if os.path.exists(out):
            made.append(name)
    print(f"{args.kit}: wrote {', '.join(made) if made else 'nothing (mains-powered kit)'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
