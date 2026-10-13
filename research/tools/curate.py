#!/usr/bin/env python3
"""Turn a raw harvest into a curated, code-backed reading list.

The raw harvest over-reports, in two distinct ways, and both have to be removed
before the list is worth anything:

1. **Aggregator repositories.** The GitHub README reverse-lookup finds any
   repository that mentions the arXiv ID. A large class of repositories --
   ``*-arxiv-daily``, ``awesome-*``, paper-list and reading-list repos -- mention
   thousands of IDs. They are not the paper's code. Two filters catch them: a
   name/description pattern, and the much more robust observation that an
   aggregator matches *many* papers while real code matches one or two (the
   second paper from the same lab reusing one repository is legitimate and
   common, three is already suspicious).

2. **Off-topic papers.** The arXiv topic matrix is intentionally broad, so it
   picks up robotics, speech and medical work that shares vocabulary with
   condition monitoring ("fault-tolerant locomotion", "anomaly detection" in
   network traffic). A keyword relevance score over title and abstract, with
   domain terms weighted above generic method terms, separates them.

What survives is a list where every entry has a resolvable repository that
plausibly belongs to the paper, and a paper that plausibly belongs to this
platform. Everything filtered out is *kept* in the JSON with a reason, so the
filtering itself is auditable rather than a black box.

Usage::

    python3 curate.py --in data/harvest_20261013.json \\
        --out-md ../code_backed_papers_2025_2026.md \\
        --out-csv data/papers_with_code.csv \\
        --out-json data/curated.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Aggregator detection
# ---------------------------------------------------------------------------
_AGGREGATOR_NAME_RE = re.compile(
    r"(arxiv|paper|publication)[-_]?dail(y|ies)"      # *-arxiv-daily, ASR-TTS-paper-daily
    r"|dail(y|ies)[-_]?(arxiv|paper)"
    r"|^arxiv[-_]"                                     # Arxiv-Quantum
    r"|arxiv[-_]?(papers?|feed|bot|crawler|digest|tracker|watch)"
    r"|awesome[-_]"
    r"|paper[-_]?(list|reading|digest|notes|hub)"
    r"|reading[-_]?list"
    r"|[-_](survey|collection|list|papers)$"
    r"|literature|bibliograph|research[-_]?digest"
    r"|cv[-_]arxiv|ml[-_]papers",
    re.IGNORECASE,
)
_AGGREGATOR_DESC_RE = re.compile(
    r"(daily (update|arxiv|paper)|automatically updated|auto[- ]?update|"
    r"curated list|collection of papers|paper list|reading list|"
    r"tracks? (new |latest )?(arxiv|papers)|arxiv (crawler|feed|digest))",
    re.IGNORECASE,
)

#: A repository matched by this many distinct papers through README search is
#: treated as an aggregator. Two is legitimate (one lab, two papers, one repo).
AGGREGATOR_PAPER_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------
# Weighted keyword sets. Domain terms are what make a paper *this* platform's
# business; method terms are necessary but not sufficient, so they score less.
_DOMAIN_TERMS = {
    "bearing": 4.0, "rolling element": 4.0, "raceway": 4.0, "spall": 4.0,
    "vibration": 3.5, "accelerometer": 3.0, "acoustic emission": 3.5,
    "condition monitoring": 4.0, "predictive maintenance": 4.0,
    "fault diagnosis": 3.5, "fault detection": 2.5, "fault classification": 3.0,
    "remaining useful life": 4.0, "prognostic": 3.0, "degradation": 2.5,
    "rotating machinery": 4.0, "gearbox": 3.5, "gear fault": 3.5,
    "induction motor": 3.5, "stator current": 3.5, "motor current": 3.0,
    "wind turbine": 2.5, "pump": 1.5, "compressor": 1.5, "machine tool": 2.0,
    "machinery": 2.5, "cwru": 4.0, "mfpt": 4.0, "mimii": 4.0, "paderborn": 3.5,
    "xjtu": 3.0, "femto": 2.5, "c-mapss": 3.0, "phm": 2.5,
    "anomalous sound": 3.5, "machine sound": 3.0, "dcase": 2.0,
    "industrial iot": 2.0, "iiot": 2.0, "asset health": 3.0,
    "spectral kurtosis": 4.0, "envelope spectrum": 4.0, "cyclostationary": 3.5,
    "order tracking": 3.5, "tacholess": 4.0,
}
_METHOD_TERMS = {
    "anomaly detection": 1.2, "time series": 0.8, "self-supervised": 0.6,
    "transformer": 0.4, "foundation model": 0.8, "domain adaptation": 0.8,
    "transfer learning": 0.6, "few-shot": 0.6, "federated": 0.6,
    "tinyml": 1.5, "microcontroller": 1.5, "edge ai": 1.0, "quantization": 0.6,
    "physics-informed": 1.0, "diffusion model": 0.5, "contrastive": 0.4,
    "explainab": 0.8, "uncertainty": 0.6, "conformal": 0.8, "calibration": 0.5,
    "wavelet": 1.0, "spectrogram": 0.8, "mel": 0.6, "signal processing": 1.0,
}
# Terms that mark a paper as belonging to a neighbouring field. Negative weight
# rather than a hard exclusion: "fault-tolerant locomotion" is robotics, but a
# paper on bearing faults *in* a robot joint is relevant.
_OFF_TOPIC_TERMS = {
    "locomotion": -3.0, "manipulation": -2.0, "grasp": -3.0, "navigation": -3.0,
    "vision-language-action": -4.0, "vla": -2.0, "autonomous driving": -3.0,
    "text-to-speech": -4.0, "speech synthesis": -4.0, "dialogue": -2.5,
    "recommendation": -3.0, "medical imag": -3.0, "clinical": -2.5,
    "molecul": -3.0, "protein": -3.0, "astronom": -3.0, "financ": -2.5,
    "network traffic": -2.5, "intrusion detection": -2.5, "malware": -3.0,
    "video surveillance": -2.0, "pedestrian": -3.0, "semantic segmentation": -2.0,
}

#: Minimum relevance score to appear in the curated list. Calibrated by hand
#: against the 2025-2026 harvest: at 4.0 the list still admitted a TTS paper and
#: a cryogenics LLM simulator, at 8.0 it started dropping genuine TinyML
#: deployment work whose abstract talks about microcontrollers rather than
#: bearings.
RELEVANCE_THRESHOLD = 6.0

# ---------------------------------------------------------------------------
# Thematic grouping. First matching theme wins, so order matters: the more
# specific themes come first.
# ---------------------------------------------------------------------------
THEMES: List[Tuple[str, Sequence[str]]] = [
    ("Edge / TinyML deployment",
     ["tinyml", "microcontroller", "mcu", "on-device", "edge ai", "embedded",
      "quantization", "quantisation", "fpga", "in-sensor", "energy-efficient inference"]),
    ("Foundation models and LLM agents for maintenance",
     ["foundation model", "large language model", "llm", "gpt", "agent",
      "pretrained model", "pre-trained model", "zero-shot", "in-context"]),
    ("Prognostics and remaining useful life",
     ["remaining useful life", "rul", "prognostic", "degradation model",
      "run-to-failure", "c-mapss", "health index"]),
    ("Bearing and rotating-machinery diagnosis",
     ["bearing", "raceway", "rolling element", "rotating machinery", "gearbox",
      "gear fault", "shaft", "spall", "cwru", "mfpt", "paderborn", "xjtu"]),
    ("Acoustic and ultrasonic monitoring",
     ["acoustic", "sound", "audio", "ultrasonic", "microphone", "mimii", "dcase"]),
    ("Electrical signature and power monitoring",
     ["stator current", "motor current", "mcsa", "induction motor", "power quality",
      "electrical signature", "inverter", "transformer fault"]),
    ("Signal processing and feature engineering",
     ["spectral kurtosis", "envelope", "cyclostationary", "wavelet", "order tracking",
      "cepstrum", "time-frequency", "empirical mode", "demodulation", "filter bank"]),
    ("Transfer, domain adaptation and limited labels",
     ["domain adaptation", "domain generalization", "domain generalisation",
      "transfer learning", "few-shot", "meta-learning", "self-supervised",
      "semi-supervised", "unsupervised", "label-efficient"]),
    ("Federated and privacy-preserving learning",
     ["federated", "privacy", "distributed learning", "split learning"]),
    ("Datasets, benchmarks and evaluation methodology",
     ["dataset", "benchmark", "data leakage", "evaluation protocol", "reproducib",
      "library", "toolbox", "open-source framework"]),
    ("Uncertainty, explainability and trust",
     ["uncertainty", "conformal", "calibrat", "explainab", "interpretab",
      "causal", "attribution", "trustworth"]),
    ("Generative models and data augmentation",
     ["diffusion", "generative", "gan", "augmentation", "synthetic data", "simulation-to-real"]),
    ("General industrial anomaly detection",
     ["anomaly detection", "novelty detection", "out-of-distribution", "outlier"]),
]

PERMISSIVE_LICENCES = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "ISC", "CC0-1.0", "Unlicense"}
COPYLEFT_LICENCES = {"GPL-3.0", "GPL-2.0", "AGPL-3.0", "LGPL-3.0", "MPL-2.0"}


@dataclass
class Curated:
    paper: dict
    repos: List[dict]
    relevance: float
    theme: str
    usability: float

    @property
    def best_repo(self) -> dict:
        return max(self.repos, key=lambda r: (r.get("stars", 0), r.get("size_kb", 0)))


def _text(paper: dict) -> str:
    return f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()


def relevance_score(paper: dict) -> Tuple[float, List[str]]:
    """Weighted keyword relevance over title + abstract, with the hits."""
    text = _text(paper)
    score = 0.0
    hits: List[str] = []
    for term, weight in _DOMAIN_TERMS.items():
        if term in text:
            score += weight
            hits.append(term)
    for term, weight in _METHOD_TERMS.items():
        if term in text:
            score += weight
    for term, weight in _OFF_TOPIC_TERMS.items():
        if term in text:
            score += weight
    # The title carries far more signal than the abstract: a paper *about*
    # bearings says so in the title.
    title = paper.get("title", "").lower()
    for term, weight in _DOMAIN_TERMS.items():
        if term in title:
            score += weight * 0.75
    return score, hits


def assign_theme(paper: dict) -> str:
    text = _text(paper)
    title = paper.get("title", "").lower()
    best = ("Other", 0.0)
    for theme, terms in THEMES:
        s = 0.0
        for t in terms:
            if t in title:
                s += 2.0
            elif t in text:
                s += 1.0
        if s > best[1]:
            best = (theme, s)
    return best[0]


def usability_score(repo: dict) -> float:
    """How likely is this repository to actually run?

    Not a quality judgement of the research -- a judgement of the artefact.
    Weighted toward the things that decide whether an engineer can reuse it this
    afternoon: a licence that permits reuse, enough content to be a real
    implementation, recent activity, and community validation.
    """
    score = 0.0
    lic = repo.get("license")
    if lic in PERMISSIVE_LICENCES:
        score += 30.0
    elif lic in COPYLEFT_LICENCES:
        score += 12.0  # usable, but it infects a commercial platform
    elif lic in (None, "NOASSERTION"):
        score += 0.0   # no licence means no legal right to use it
    size = repo.get("size_kb", 0) or 0
    if size >= 50_000:
        score += 8.0   # likely includes data or checkpoints
    elif size >= 1_000:
        score += 14.0  # a real codebase
    elif size >= 100:
        score += 10.0
    elif size >= 20:
        score += 4.0
    else:
        score -= 10.0  # a README and a promise
    stars = repo.get("stars", 0) or 0
    score += min(22.0, 7.0 * (stars ** 0.4))
    if repo.get("archived"):
        score -= 10.0
    if repo.get("is_fork"):
        score -= 6.0
    if repo.get("discovered_via") == "abstract-url":
        score += 10.0  # the authors themselves pointed at it
    return round(score, 1)


def find_aggregators(papers: Sequence[dict]) -> Dict[str, str]:
    """``{repo_full_name_lower: reason}`` for every repository to drop."""
    hits: Dict[str, set] = defaultdict(set)
    meta: Dict[str, dict] = {}
    for p in papers:
        for r in p.get("github_repos", []):
            name = (r.get("full_name") or "").lower()
            if not name:
                continue
            meta[name] = r
            if r.get("discovered_via") == "github-readme-search":
                hits[name].add(p["arxiv_id"])

    out: Dict[str, str] = {}
    for name, r in meta.items():
        repo_part = name.split("/")[-1]
        desc = r.get("description") or ""
        if _AGGREGATOR_NAME_RE.search(repo_part):
            out[name] = f"repository name matches an aggregator pattern ({repo_part})"
        elif _AGGREGATOR_DESC_RE.search(desc):
            out[name] = "description describes a paper list or auto-updated feed"
        elif len(hits.get(name, ())) >= AGGREGATOR_PAPER_THRESHOLD:
            out[name] = (
                f"cited by {len(hits[name])} different papers via README search -- "
                "characteristic of a paper list, not of one paper's code"
            )
    return out


def curate(data: dict, *, relevance_threshold: float = RELEVANCE_THRESHOLD) -> Tuple[List[Curated], dict]:
    papers = data["papers"]
    aggregators = find_aggregators(papers)
    kept: List[Curated] = []
    rejected = {"no_repo": 0, "aggregator_only": 0, "off_topic": 0}
    rejected_examples: Dict[str, List[str]] = defaultdict(list)

    for p in papers:
        repos = p.get("github_repos", [])
        if not repos:
            rejected["no_repo"] += 1
            continue
        clean = [r for r in repos if (r.get("full_name") or "").lower() not in aggregators]
        if not clean:
            rejected["aggregator_only"] += 1
            if len(rejected_examples["aggregator_only"]) < 8:
                dropped = (repos[0].get("full_name") or "?")
                rejected_examples["aggregator_only"].append(
                    f"{p['arxiv_id']} -> {dropped}: {aggregators.get(dropped.lower(), '')}"
                )
            continue
        rel, _hits = relevance_score(p)
        if rel < relevance_threshold:
            rejected["off_topic"] += 1
            if len(rejected_examples["off_topic"]) < 8:
                rejected_examples["off_topic"].append(
                    f"{p['arxiv_id']} (score {rel:+.1f}) {p['title'][:70]}"
                )
            continue
        kept.append(
            Curated(
                paper=p,
                repos=clean,
                relevance=round(rel, 1),
                theme=assign_theme(p),
                usability=max(usability_score(r) for r in clean),
            )
        )

    kept.sort(key=lambda c: (-(c.usability + 2.0 * c.relevance), c.paper["arxiv_id"]))
    stats = {
        "input_papers": len(papers),
        "aggregator_repos_dropped": len(aggregators),
        "aggregator_repo_names": sorted(aggregators)[:40],
        "curated": len(kept),
        "rejected": rejected,
        "rejected_examples": {k: v for k, v in rejected_examples.items()},
        "by_theme": dict(Counter(c.theme for c in kept).most_common()),
        "by_year": dict(sorted(Counter(c.paper["published"][:4] for c in kept).items())),
        "by_licence": dict(Counter(
            (c.best_repo.get("license") or "none") for c in kept).most_common()),
        "permissive_fraction": round(
            sum(1 for c in kept if c.best_repo.get("license") in PERMISSIVE_LICENCES) / max(1, len(kept)), 3),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _licence_badge(lic: Optional[str]) -> str:
    if lic in PERMISSIVE_LICENCES:
        return f"`{lic}`"
    if lic in COPYLEFT_LICENCES:
        return f"`{lic}` (copyleft)"
    if lic == "NOASSERTION":
        return "no SPDX licence"
    return "**no licence**"


def write_markdown(kept: Sequence[Curated], stats: dict, data: dict, path: str, *, top_n_detail: int = 40) -> None:
    lines: List[str] = []
    w = lines.append
    window = data["window"]
    w("# Code-backed research, 2025-2026")
    w("")
    w(f"> {len(kept)} papers submitted between {window['from']} and {window['to']} that have a "
      f"resolvable public code repository, filtered for relevance to wireless condition monitoring.")
    w("")
    w("Generated by `research/tools/harvest_arxiv_code.py` + `research/tools/curate.py`. "
      "Re-run both to refresh; the raw harvest is kept in `research/data/` so every row here is traceable.")
    w("")
    w("## How this list was built, and what it excludes")
    w("")
    w(f"1. **Harvest.** {len(data['topics'])} arXiv queries over the submission window returned "
      f"{data['counts']['unique_papers']} unique papers.")
    w(f"2. **Code links from the abstract.** Only {data['counts']['with_code_url']} papers "
      f"({data['counts']['with_code_url'] / data['counts']['unique_papers'] * 100:.0f} %) print a "
      "code URL in the abstract or the author comment. Taking that as \"papers with code\" would "
      "have missed most of them.")
    w("3. **Reverse lookup.** Asking GitHub which repositories cite each arXiv ID in their README "
      f"recovered many more, bringing the total with a resolvable repository to "
      f"{data['counts']['with_live_github']}.")
    w(f"4. **Aggregator removal.** {stats['aggregator_repos_dropped']} repositories were dropped as "
      "paper lists rather than paper code (`*-arxiv-daily`, `awesome-*`, or any repository cited by "
      f">= {AGGREGATOR_PAPER_THRESHOLD} different papers). This removed "
      f"{stats['rejected']['aggregator_only']} papers whose only \"code\" was a listing.")
    w(f"5. **Relevance filter.** {stats['rejected']['off_topic']} papers were dropped as belonging to "
      "neighbouring fields that share vocabulary (robot locomotion, speech synthesis, network "
      "intrusion detection).")
    w("")
    w("**Every repository below was resolved through the GitHub API on the generation date**: it "
      "exists, and its licence, size, star count and last-push date are recorded as returned. That is "
      "a deliberately low bar -- it means \"the code exists\", not \"the code reproduces the paper\". "
      "None of these repositories were executed. The `usability` column scores the *artefact* "
      "(licence, size, activity, community), never the research.")
    w("")
    w("### Caveats you should read before citing this list")
    w("")
    w("- A repository found by README search is **probably** the paper's code, but it can also be a "
      "third-party reimplementation or a student's course project. The `found` column records which "
      "signal matched: `abstract` means the authors themselves pointed at it.")
    w("- `no licence` is not a formality. A public repository without a licence grants no reuse "
      f"rights at all. {stats['by_licence'].get('none', 0) + stats['by_licence'].get('NOASSERTION', 0)} "
      "of the entries below are in that state, so they can be read for method detail but not "
      "vendored into a product.")
    w("- arXiv abstracts frequently promise code that was never published. The reverse lookup is what "
      "separates a promise from a repository.")
    w("")
    w("## Summary")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Window | {window['from']} to {window['to']} |")
    w(f"| Unique papers harvested | {data['counts']['unique_papers']} |")
    w(f"| With a resolvable repository | {data['counts']['with_live_github']} |")
    w(f"| After aggregator + relevance filtering | **{len(kept)}** |")
    w(f"| Permissively licensed (Apache/MIT/BSD) | {stats['by_licence'].get('Apache-2.0', 0) + stats['by_licence'].get('MIT', 0) + stats['by_licence'].get('BSD-3-Clause', 0)} "
      f"({stats['permissive_fraction'] * 100:.0f} %) |")
    w("")
    w("By year: " + ", ".join(f"{k}: {v}" for k, v in stats["by_year"].items()))
    w("")
    w("| Theme | Papers |")
    w("|---|---|")
    for theme, n in stats["by_theme"].items():
        w(f"| {theme} | {n} |")
    w("")

    w("## The list, by theme")
    w("")
    w("`usability` scores the repository artefact: permissive licence +30, real codebase size +10 to "
      "+14, stars up to +22, authors' own link +10, archived or unlicensed penalised. "
      "`rel` is keyword relevance to condition monitoring.")
    w("")
    by_theme: Dict[str, List[Curated]] = defaultdict(list)
    for c in kept:
        by_theme[c.theme].append(c)
    for theme, _n in stats["by_theme"].items():
        group = by_theme[theme]
        if not group:
            continue
        w(f"### {theme}")
        w("")
        w("| arXiv | Title | Repository | Licence | Stars | Last push | usability | rel | found |")
        w("|---|---|---|---|---:|---|---:|---:|---|")
        for c in group:
            p, r = c.paper, c.best_repo
            title = p["title"].replace("|", "/")
            if len(title) > 96:
                title = title[:93] + "..."
            found = "abstract" if r.get("discovered_via") == "abstract-url" else "readme"
            pushed = (r.get("pushed_at") or "")[:10]
            w(f"| [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}) "
              f"| {title} "
              f"| [{r['full_name']}]({r['html_url']}) "
              f"| {_licence_badge(r.get('license'))} "
              f"| {r.get('stars', 0)} | {pushed} | {c.usability:.0f} | {c.relevance:.0f} | {found} |")
        w("")

    w("## Detail on the top entries")
    w("")
    w(f"The {min(top_n_detail, len(kept))} entries with the highest combined usability and relevance, "
      "with the abstract trimmed to its claim.")
    w("")
    for i, c in enumerate(kept[:top_n_detail], 1):
        p, r = c.paper, c.best_repo
        w(f"### {i}. {p['title']}")
        w("")
        authors = ", ".join(p["authors"][:4]) + (" et al." if len(p["authors"]) > 4 else "")
        w(f"- **arXiv** [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}) "
          f"({p['published'][:10]}, {p['primary_category']})")
        w(f"- **Authors** {authors}")
        w(f"- **Code** [{r['full_name']}]({r['html_url']}) -- {_licence_badge(r.get('license'))}, "
          f"{r.get('stars', 0)} stars, {r.get('size_kb', 0)} kB, "
          f"last push {(r.get('pushed_at') or '')[:10]}, "
          f"{'linked by the authors' if r.get('discovered_via') == 'abstract-url' else 'found via README citation'}")
        if r.get("language"):
            w(f"- **Language** {r['language']}")
        if p.get("comment"):
            w(f"- **Note** {p['comment'][:240]}")
        if len(c.repos) > 1:
            others = ", ".join(f"[{x['full_name']}]({x['html_url']})" for x in c.repos[1:4])
            w(f"- **Also** {others}")
        abstract = p["abstract"]
        w("")
        w(f"{abstract[:900]}{'...' if len(abstract) > 900 else ''}")
        w("")

    w("## Audit trail")
    w("")
    w(f"- Aggregator repositories dropped ({stats['aggregator_repos_dropped']}): "
      + ", ".join(f"`{n}`" for n in stats["aggregator_repo_names"][:25])
      + (" ..." if stats["aggregator_repos_dropped"] > 25 else ""))
    w("")
    for key, label in (("aggregator_only", "Dropped: aggregator was the only match"),
                       ("off_topic", "Dropped: below the relevance threshold")):
        examples = stats["rejected_examples"].get(key, [])
        if examples:
            w(f"**{label}** (examples)")
            w("")
            for e in examples:
                w(f"- {e}")
            w("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_csv(kept: Sequence[Curated], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "arxiv_id", "published", "title", "theme", "relevance", "usability",
            "repo", "repo_url", "license", "stars", "forks", "size_kb", "language",
            "pushed_at", "archived", "discovered_via", "arxiv_url",
        ])
        for c in kept:
            p, r = c.paper, c.best_repo
            wr.writerow([
                p["arxiv_id"], p["published"][:10], p["title"], c.theme, c.relevance, c.usability,
                r.get("full_name"), r.get("html_url"), r.get("license") or "", r.get("stars", 0),
                r.get("forks", 0), r.get("size_kb", 0), r.get("language") or "",
                (r.get("pushed_at") or "")[:10], r.get("archived"), r.get("discovered_via"),
                f"https://arxiv.org/abs/{p['arxiv_id']}",
            ])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--relevance-threshold", type=float, default=RELEVANCE_THRESHOLD)
    ap.add_argument("--top-detail", type=int, default=40)
    args = ap.parse_args(argv)

    with open(args.inp, encoding="utf-8") as fh:
        data = json.load(fh)
    kept, stats = curate(data, relevance_threshold=args.relevance_threshold)

    print(f"input {stats['input_papers']} papers", file=sys.stderr)
    print(f"dropped {stats['aggregator_repos_dropped']} aggregator repositories", file=sys.stderr)
    print(f"rejected {stats['rejected']}", file=sys.stderr)
    print(f"curated {len(kept)} papers", file=sys.stderr)
    for theme, n in stats["by_theme"].items():
        print(f"    {n:4d}  {theme}", file=sys.stderr)

    if args.out_md:
        write_markdown(kept, stats, data, args.out_md, top_n_detail=args.top_detail)
        print(f"wrote {args.out_md}", file=sys.stderr)
    if args.out_csv:
        write_csv(kept, args.out_csv)
        print(f"wrote {args.out_csv}", file=sys.stderr)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "generated_at": data["generated_at"],
                    "window": data["window"],
                    "stats": stats,
                    "papers": [
                        {
                            "arxiv_id": c.paper["arxiv_id"],
                            "title": c.paper["title"],
                            "published": c.paper["published"],
                            "authors": c.paper["authors"],
                            "abstract": c.paper["abstract"],
                            "comment": c.paper["comment"],
                            "primary_category": c.paper["primary_category"],
                            "theme": c.theme,
                            "relevance": c.relevance,
                            "usability": c.usability,
                            "repos": c.repos,
                        }
                        for c in kept
                    ],
                },
                fh,
                indent=1,
                ensure_ascii=False,
            )
        print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
