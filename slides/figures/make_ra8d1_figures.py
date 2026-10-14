#!/usr/bin/env python3
"""RA8D1 deck figures: what angular resampling actually buys, measured.

    python3 slides/figures/make_ra8d1_figures.py --out slides/figures

Two figures, both computed rather than drawn:

    ra8d1_order_gain.pdf     peak-amplitude recovery vs order, at three drift
                             levels, with the two points the C test measures
                             marked on top
    ra8d1_order_spectrum.pdf the same 40x line in the frequency and angle
                             domains -- the visual version of the argument

The numpy code here uses the same method as hardware/ra8d1/tests/test_order.c
(swept-phase synthesis, Hann window, length-normalised single-sided magnitude),
and the script runs that C binary when it is present so the markers are
MEASURED values rather than the same numpy restated. If the two disagree the
figure says so in its own title, because a figure that silently diverges from
the test it illustrates is worse than no figure.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CRIT, WARN = "#d03b3b", "#fab219"
#: Validated ordinal blue ramp; drift level is an ordered quantity.
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

FS = 26667.0
N_IN = 8192
SPR = 256
F0 = 29.0
#: Hann coherent gain: a unit sine on an exact bin peaks here after the 2/N
#: normalisation. Both domains are measured against this same ideal, which is
#: what makes two different FFT lengths comparable.
IDEAL = 0.5


def _swept(order: float, drift_pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """A line at `order` on a shaft whose rate sweeps linearly by drift_pct.

    Returns the samples and the revolution-mark times. The phase is the
    integral of the rate, so the marks are exact rather than detected -- the
    point here is the resampler, not a tacho model.
    """
    t_rec = (N_IN - 1) / FS
    rate = F0 * (drift_pct / 100.0) / t_rec
    t = np.arange(N_IN) / FS
    theta = F0 * t + 0.5 * rate * t * t          # revolutions
    x = np.sin(2.0 * np.pi * order * theta)
    k = np.arange(0, 200.0)
    with np.errstate(invalid="ignore"):
        tk = ((-F0 + np.sqrt(F0 * F0 + 2.0 * rate * k)) / rate) if rate != 0 else k / F0
    tk = tk[np.isfinite(tk)]
    return x, tk[tk <= t_rec]


def _spectrum(x: np.ndarray) -> np.ndarray:
    n = len(x)
    w = np.hanning(n + 1)[:n]
    return np.abs(np.fft.rfft(x * w)) * 2.0 / n


def _resample(x: np.ndarray, marks: np.ndarray, spr: int, cap: int) -> np.ndarray:
    """Catmull-Rom angular resampling: the numpy twin of pdm_order.c."""
    out: List[float] = []
    for k in range(len(marks) - 1):
        t0, dt = marks[k], marks[k + 1] - marks[k]
        for j in range(spr):
            if len(out) >= cap:
                return np.asarray(out)
            t = t0 + dt * (j / spr)
            idx = t * FS
            i1 = int(idx)
            frac = idx - i1
            i0, i2, i3 = max(i1 - 1, 0), min(i1 + 1, len(x) - 1), min(i1 + 2, len(x) - 1)
            p0, p1, p2, p3 = x[i0], x[i1], x[i2], x[i3]
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * frac
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * frac ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * frac ** 3))
    return np.asarray(out)


def _peak(mag: np.ndarray, bins_per_order: float, order: float) -> float:
    lo = max(1, int((order - 3.0) * bins_per_order))
    hi = min(len(mag) - 1, int((order + 3.0) * bins_per_order))
    return float(mag[lo:hi + 1].max()) if hi > lo else 0.0


def _gain(order: float, drift_pct: float) -> Tuple[float, float]:
    """(time-domain peak, angle-domain peak), both normalised to IDEAL."""
    x, marks = _swept(order, drift_pct)
    mean_hz = (len(marks) - 1) / (marks[-1] - marks[0]) if len(marks) > 1 else F0
    pk_t = _peak(_spectrum(x), N_IN / FS * mean_hz, order)

    n_cap = 2048
    res = _resample(x, marks, SPR, n_cap)
    n_a = 1 << int(np.floor(np.log2(len(res)))) if len(res) >= 1024 else 1024
    pk_a = _peak(_spectrum(res[:n_a]), n_a / SPR, order)
    return pk_t / IDEAL, pk_a / IDEAL


def _c_measured() -> Dict[float, float]:
    """Run the C test and parse its measured gains. {} if it is not built."""
    binary = os.path.join(_REPO, "hardware", "ra8d1", "tests", "test_order")
    if not os.path.isfile(binary):
        return {}
    try:
        out = subprocess.run([binary], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    # Only the "low" and "high" cases: those are the 4 % sweep. The "flat" line
    # is the 0 % control and matching it too silently overwrote the order-6
    # value with the steady-shaft one -- 1.14x instead of 1.19x, close enough
    # to look right on the figure.
    found: Dict[float, float] = {}
    for m in re.finditer(r"^\s+(?:low|high)\s+order\s+(\d+):.*gain ([0-9.]+)x",
                         out, re.MULTILINE):
        found[float(m.group(1))] = float(m.group(2))
    return found


def fig_gain(path: str) -> None:
    orders = np.arange(2.0, 61.0, 2.0)
    drifts = [1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))

    for colour, drift in zip(ORDINAL, drifts):
        gains = [(_gain(float(o), drift)) for o in orders]
        ax.plot(orders, [a / t for t, a in gains], color=colour, zorder=4,
                label=f"{drift:g} % drift")

    measured = _c_measured()
    if measured:
        xs = sorted(measured)
        ax.scatter(xs, [measured[x] for x in xs], s=46, color=CRIT, zorder=6,
                   marker="D", edgecolor=SURFACE, linewidth=1.2,
                   label="measured by tests/test_order.c (4 %)")
        for x in xs:
            ax.annotate(f"{measured[x]:.2f}$\\times$", (x, measured[x]),
                        textcoords="offset points", xytext=(7, -2),
                        fontsize=7.6, color=CRIT, fontweight="bold")

    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.text(52, 1.015, "no gain", fontsize=7.2, color=MUTED, ha="left", va="bottom")
    ax.axvspan(2, 12, color=WARN, alpha=0.10, zorder=1)
    ax.annotate("bearing orders\n(BPFO 3.6x, BPFI 5.4x):\nlittle to gain",
                xy=(9.0, 1.16), xytext=(13.0, 1.72), fontsize=7.4, color=INK_2,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("gear mesh, blade pass,\nelectrical lines", xy=(48, 2.10),
                xytext=(40.0, 1.52), fontsize=7.4, color=INK_2, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xlabel("order (multiples of shaft rate)")
    ax.set_ylabel("peak amplitude recovered\n(angle / frequency domain)")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 62)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=7.4)
    ax.set_title("Order tracking pays at high orders, not at bearing orders"
                 "  (307 ms record, 256 samples/rev)")
    fig.savefig(path)
    plt.close(fig)


def fig_spectrum(path: str) -> None:
    order = 40.0
    drift = 4.0
    x, marks = _swept(order, drift)
    mean_hz = (len(marks) - 1) / (marks[-1] - marks[0])
    mag_t = _spectrum(x)
    orders_t = np.arange(len(mag_t)) * FS / N_IN / mean_hz

    res = _resample(x, marks, SPR, 2048)
    n_a = 1 << int(np.floor(np.log2(len(res))))
    mag_a = _spectrum(res[:n_a])
    orders_a = np.arange(len(mag_a)) * SPR / n_a

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    m = (orders_t > order - 4) & (orders_t < order + 4)
    ax.plot(orders_t[m], mag_t[m] / IDEAL, color=MUTED, lw=1.6, zorder=3,
            label=f"frequency domain  ({mag_t[m].max() / IDEAL:.2f} of ideal)")
    m2 = (orders_a > order - 4) & (orders_a < order + 4)
    ax.plot(orders_a[m2], mag_a[m2] / IDEAL, color=S1, lw=1.8, zorder=4,
            label=f"angle domain  ({mag_a[m2].max() / IDEAL:.2f} of ideal)")
    ax.axvline(order, color=CRIT, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.text(order + 0.12, 1.02, "true order", fontsize=7.2, color=CRIT)
    ax.set_xlabel("order")
    ax.set_ylabel("amplitude / ideal")
    ax.set_ylim(0, 1.15)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.4)
    ax.set_title(f"A {order:.0f}x line at {drift:g} % speed drift: "
                 f"smeared into its own skirts, then put back")
    fig.savefig(path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_HERE)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    fig_gain(os.path.join(args.out, "ra8d1_order_gain.pdf"))
    fig_spectrum(os.path.join(args.out, "ra8d1_order_spectrum.pdf"))
    measured = _c_measured()
    print(f"ra8d1: wrote 2 figures; C-measured gains: "
          f"{ {k: v for k, v in sorted(measured.items())} or 'binary not built'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
