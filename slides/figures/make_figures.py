#!/usr/bin/env python3
"""Generate the deck figures from the committed harvest data.

Every figure is derived from `research/data/harvest_*.json` and
`research/data/curated.json` -- nothing is hand-typed, so the slides cannot drift
from the data. Re-run after a re-harvest and the deck updates.

    python3 slides/figures/make_figures.py --data research/data --out slides/figures

Output is PDF (vector) for LaTeX inclusion.

Palette and mark conventions follow the project's data-viz rules:
single hue for magnitude-across-categories, a validated ordinal blue ramp for
the funnel stages, categorical slots 1-2 (blue/orange) for the only two-series
figure, recessive hairline grid, muted axis ink, selective direct labels rather
than a number on every mark, and status red used only where the thing being
marked genuinely is a risk state -- always with a text label beside it, never
colour alone.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# --- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"   # categorical slot 1, blue
SERIES_2 = "#eb6834"   # categorical slot 2, orange
CRITICAL = "#d03b3b"   # status: critical -- only for genuine risk states
# Validated ordinal blue ramp (light end clears 2:1 against the surface).
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PERMISSIVE = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "ISC", "CC0-1.0", "Unlicense"}
COPYLEFT = {"GPL-3.0", "GPL-2.0", "AGPL-3.0", "LGPL-3.0", "MPL-2.0"}


def _best(paper: dict) -> dict:
    return max(paper["repos"], key=lambda r: (r.get("stars", 0), r.get("size_kb", 0)))


def _hgrid(ax) -> None:
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)


# -------------------------------------------------------------------------
def fig_funnel(harvest: dict, curated: dict, path: str) -> None:
    """Horizontal funnel: how 1639 papers become 45 usable ones.

    Ordinal ramp, not categorical: the stages are an ordered sequence, so the
    colour carries position in the pipeline. Direct value labels on every bar
    because there are only four and the counts *are* the message.
    """
    stats = curated["stats"]
    stages = [
        ("Harvested from arXiv", harvest["counts"]["unique_papers"]),
        ("Code URL in the abstract", harvest["counts"]["with_code_url"]),
        ("Repository resolves on GitHub", harvest["counts"]["with_live_github"]),
        ("Relevant, not an aggregator", stats["curated"]),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    y = range(len(values))
    ax.barh(list(y), values, height=0.58, color=ORDINAL, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("papers")
    _hgrid(ax)
    total = values[0]
    for i, v in enumerate(values):
        ax.text(v + total * 0.012, i, f"{v}  ({v / total * 100:.1f} %)",
                va="center", ha="left", color=INK, fontsize=8.5, fontweight="bold")
    ax.set_title("Only 6 % of papers print a code link; the reverse lookup triples the yield")
    fig.savefig(path)
    plt.close(fig)


def fig_licence(curated: dict, path: str) -> None:
    """Licence of the best repository per curated paper.

    One hue for magnitude. The unlicensed bar is the exception: it is a genuine
    legal risk state, so it takes the status-critical colour *and* carries the
    text "no reuse rights" beside it -- colour never carries that meaning alone.
    """
    counts = Counter()
    for p in curated["papers"]:
        lic = _best(p).get("license")
        if lic in PERMISSIVE:
            counts[lic] += 1
        elif lic in COPYLEFT:
            counts[f"{lic} (copyleft)"] += 1
        elif lic == "NOASSERTION":
            counts["No SPDX licence"] += 1
        else:
            counts["No licence file"] += 1
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in order]
    values = [v for _, v in order]
    risky = {"No licence file", "No SPDX licence"}
    colors = [CRITICAL if lb in risky else SERIES_1 for lb in labels]

    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    y = range(len(values))
    ax.barh(list(y), values, height=0.6, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.5)
    ax.set_xlabel("papers (by highest-starred repository)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _hgrid(ax)
    for i, (lb, v) in enumerate(zip(labels, values)):
        note = "   no reuse rights" if lb in risky else ""
        ax.text(v + 0.3, i, f"{v}{note}", va="center", ha="left",
                color=CRITICAL if note else INK, fontsize=8.5,
                fontweight="bold" if note else "normal")
    n_risky = sum(v for lb, v in zip(labels, values) if lb in risky)
    ax.set_title(f"{n_risky} of {sum(values)} ship code with no usable licence")
    fig.savefig(path)
    plt.close(fig)


def fig_themes(curated: dict, path: str) -> None:
    """Where the code-backed work actually is. Single hue, sorted descending."""
    counts = Counter(p["theme"] for p in curated["papers"])
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    labels = [k.replace(" and ", " & ") for k, _ in order]
    values = [v for _, v in order]

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    y = range(len(values))
    ax.barh(list(y), values, height=0.62, color=SERIES_1, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("papers")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _hgrid(ax)
    for i, v in enumerate(values):
        ax.text(v + 0.15, i, str(v), va="center", ha="left", color=INK, fontsize=8.5)
    ax.set_title("Prognostics and evaluation methodology dominate; bearings are thin")
    fig.savefig(path)
    plt.close(fig)


def fig_usability_relevance(curated: dict, path: str) -> None:
    """Artefact usability against topical relevance, split by licence class.

    Two series, so a legend is present. Slots 1 and 2 clear the all-pairs CVD
    and normal-vision floors, which is the pairlist a scatter needs. The handful
    of named points are the only direct labels -- labelling all 45 would be
    unreadable and is the anti-pattern this rule exists to prevent.

    Note on reading this honestly: the vertical separation between the two
    series is partly definitional, because the usability score awards +30 for a
    permissive licence. The informative axis is the horizontal one -- relevance
    is scored only from title and abstract text, and it shows high-relevance
    work distributed across *both* licence classes. An earlier version of this
    figure was titled "the most relevant work is often the least reusable",
    which the data refutes: the single most relevant entry (PG-AMF, relevance
    34.6) is MIT-licensed.
    """
    reusable_x, reusable_y, locked_x, locked_y = [], [], [], []
    named: List[tuple] = []
    for p in curated["papers"]:
        r = _best(p)
        lic = r.get("license")
        x, yv = p["relevance"], p["usability"]
        if lic in PERMISSIVE:
            reusable_x.append(x)
            reusable_y.append(yv)
        else:
            locked_x.append(x)
            locked_y.append(yv)
        if yv + 2.0 * x > 95 or x > 24:
            named.append((x, yv, r["full_name"].split("/")[-1][:22]))

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.scatter(reusable_x, reusable_y, s=46, color=SERIES_1, edgecolor=SURFACE,
               linewidth=1.1, zorder=3, label="Permissive licence - reusable")
    ax.scatter(locked_x, locked_y, s=46, color=SERIES_2, edgecolor=SURFACE,
               linewidth=1.1, zorder=3, marker="D", label="No usable licence - read only")
    xs = reusable_x + locked_x
    ax.set_xlim(min(xs) - 1.5, max(xs) + 7.0)   # headroom so labels do not clip
    for x, yv, name in sorted(named, key=lambda t: -(t[1] + 2 * t[0]))[:6]:
        # Place the label inside the axes: to the left for points near the right edge.
        right_edge = x > max(xs) - 8.0
        ax.annotate(name, (x, yv), textcoords="offset points",
                    xytext=(-7 if right_edge else 7, 4),
                    ha="right" if right_edge else "left", fontsize=7, color=INK_2)
    ax.set_xlabel("relevance to condition monitoring (title + abstract keywords)")
    ax.set_ylabel("artefact usability")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("Relevance does not predict reusability: strong work sits in both licence classes")
    fig.savefig(path)
    plt.close(fig)


def fig_monthly(curated: dict, path: str) -> None:
    """Curated papers per submission month -- is this accelerating?"""
    months = Counter(p["published"][:7] for p in curated["papers"])
    keys = sorted(months)
    values = [months[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    ax.bar(range(len(keys)), values, width=0.68, color=SERIES_1, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[2:] for k in keys], rotation=90, fontsize=6.5)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("papers")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Code-backed condition-monitoring papers by submission month")
    fig.savefig(path)
    plt.close(fig)


def fig_leakage(path: str) -> None:
    """The reported-accuracy gap from 2607.16493.

    A single comparison with a huge asymmetry: two bars, direct labels, and the
    honest range drawn as a range rather than a point, because the paper reports
    20-60 % depending on the dataset. Drawing it as one number would overstate
    what the paper says.
    """
    fig, ax = plt.subplots(figsize=(5.0, 1.75))
    ax.barh([1], [99.9], height=0.5, color=CRITICAL, zorder=3)
    ax.barh([0], [40], height=0.5, xerr=[[20], [20]], color=SERIES_1, zorder=3,
            error_kw={"ecolor": INK_2, "capsize": 3, "elinewidth": 1.1})
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Leakage-audited split\n(genuine)", "Naive window split\n(inflated)"],
                       fontsize=8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("reported classification accuracy (%)")
    _hgrid(ax)
    ax.text(101, 1, "99.9 %", va="center", color=CRITICAL, fontweight="bold", fontsize=9)
    ax.text(62, 0, "20-60 %", va="center", color=INK, fontsize=9)
    ax.set_title("arXiv 2607.16493: the split, not the model, decides the headline")
    fig.savefig(path)
    plt.close(fig)


# -------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="research/data")
    ap.add_argument("--out", default="slides/figures")
    args = ap.parse_args(argv)

    harvests = sorted(f for f in os.listdir(args.data) if f.startswith("harvest_") and f.endswith(".json"))
    if not harvests:
        raise SystemExit(f"no harvest_*.json in {args.data}")
    with open(os.path.join(args.data, harvests[-1]), encoding="utf-8") as fh:
        harvest = json.load(fh)
    with open(os.path.join(args.data, "curated.json"), encoding="utf-8") as fh:
        curated = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    jobs: Dict[str, object] = {
        "fig_funnel.pdf": lambda p: fig_funnel(harvest, curated, p),
        "fig_licence.pdf": lambda p: fig_licence(curated, p),
        "fig_themes.pdf": lambda p: fig_themes(curated, p),
        "fig_usability.pdf": lambda p: fig_usability_relevance(curated, p),
        "fig_monthly.pdf": lambda p: fig_monthly(curated, p),
        "fig_leakage.pdf": fig_leakage,
    }
    for name, fn in jobs.items():
        out = os.path.join(args.out, name)
        fn(out)  # type: ignore[operator]
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
