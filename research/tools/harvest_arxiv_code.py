#!/usr/bin/env python3
"""Harvest recent arXiv papers on machine condition monitoring that ship code.

Only the standard library is used so this runs on any Python >= 3.8 without a
virtualenv (useful in CI and on gateway hardware).

The harvester has two stages:

1. ``arxiv``  -- query the arXiv Atom API across a fixed topic matrix, inside a
   submission-date window, and extract every source-code URL that appears in
   the abstract or the author comment field.
2. ``verify`` -- resolve every GitHub URL through the GitHub REST API so that
   "has code" means "a repository that actually exists and has commits", not
   "the authors promised code in the abstract".

Usage
-----
    python3 harvest_arxiv_code.py --out data/harvest.json
    python3 harvest_arxiv_code.py --out data/harvest.json --from 2025-01-01
    GITHUB_TOKEN=... python3 harvest_arxiv_code.py --out data/harvest.json

Without ``GITHUB_TOKEN`` the GitHub stage still runs but is limited to 60
requests/hour, so pass a token (any classic or fine-grained token with public
read scope) when harvesting the full matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com"
USER_AGENT = "pdm-platform-research-harvester/1.0 (+https://github.com/tasktest06-star/predictive-maintenance-platform)"

# arXiv asks for >= 3 s between programmatic requests.
ARXIV_DELAY_S = 3.2
GITHUB_DELAY_S = 0.1

# ---------------------------------------------------------------------------
# Topic matrix. Each entry maps a review section to an arXiv search expression.
# Keep the expressions narrow: a broad `all:` query returns mostly unrelated
# control-theory papers and burns the rate limit.
# ---------------------------------------------------------------------------
#
# NOTE ON QUERY STYLE: arXiv's abstract search matches literal phrases, so a
# long phrase such as abs:"bearing fault diagnosis" returns only ~47 hits of
# all time while abs:bearing AND abs:fault returns ~80 in a 21-month window.
# Prefer short phrases ANDed with single terms; the counts in the comments were
# measured for the 2025-01-01..2026-10-13 window.
TOPICS: list[tuple[str, str]] = [
    ("fault-diagnosis", 'abs:"fault diagnosis"'),                                   # ~240
    ("bearing", 'abs:bearing AND abs:fault'),                                       # ~80
    ("predictive-maintenance", 'abs:"predictive maintenance"'),                     # ~231
    ("rul-prognostics", 'abs:"remaining useful life"'),                             # ~102
    ("prognostics", 'abs:prognostics AND (abs:machine OR abs:equipment OR abs:industrial)'),
    ("condition-monitoring", 'abs:"condition monitoring"'),                         # ~107
    ("acoustic-anomaly", 'abs:"anomalous sound detection" OR abs:"machine condition monitoring"'),
    ("acoustic-emission", 'abs:"acoustic emission"'),                               # ~36
    ("vibration-anomaly", 'abs:vibration AND abs:"anomaly detection"'),             # ~10
    ("vibration-learning", 'abs:vibration AND (abs:"deep learning" OR abs:transformer OR abs:"neural network")'),
    ("edge-tinyml", 'abs:TinyML OR (abs:microcontroller AND abs:anomaly)'),         # ~122
    ("rotating-machinery", 'abs:"rotating machinery"'),                             # ~39
    ("motor-current", 'abs:"induction motor" OR abs:"motor current"'),              # ~45
    ("gearbox-wind", 'abs:"wind turbine" AND abs:fault'),                           # ~17
    ("domain-adaptation", 'abs:"domain adaptation" AND abs:fault'),
    ("foundation-model", '(abs:"foundation model" OR abs:"large language model") AND (abs:fault OR abs:maintenance OR abs:industrial)'),
    ("self-supervised", 'abs:"self-supervised" AND (abs:fault OR abs:vibration OR abs:machinery)'),
    ("physics-informed", 'abs:"physics-informed" AND (abs:fault OR abs:degradation OR abs:prognostics)'),
    ("federated", 'abs:"federated learning" AND (abs:industrial OR abs:fault OR abs:maintenance)'),
    ("generative-augmentation", '(abs:"diffusion model" OR abs:"data augmentation") AND abs:fault'),
    ("time-series-anomaly", 'abs:"time series" AND abs:"anomaly detection" AND (abs:industrial OR abs:sensor)'),
    ("datasets-benchmarks", 'abs:dataset AND (abs:fault OR abs:"condition monitoring" OR abs:machinery)'),
]

# ---------------------------------------------------------------------------
# Code-link extraction
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(github\.com|gitlab\.com|bitbucket\.org|gitee\.com|huggingface\.co|zenodo\.org|codeocean\.com|"
    r"figshare\.com|osf\.io|data\.mendeley\.com|sourceforge\.net)"
    r"/[^\s,;)\]\}>\"']+",
    re.IGNORECASE,
)

# Trailing punctuation that commonly rides along in abstract text.
_TRAILING = ".,;:)]}>\"'`"

_GITHUB_REPO_RE = re.compile(
    r"github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)/(?P<repo>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)

# Phrases that signal a code release even when no URL is in the abstract.
_PROMISE_RE = re.compile(
    r"(code|source code|implementation|dataset|models?|weights|toolbox|framework)"
    r"[^.]{0,80}?"
    r"(is|are|will be|has been|have been|publicly|freely)"
    r"[^.]{0,40}?"
    r"(available|released|open[- ]sourced?|accessible|published)",
    re.IGNORECASE,
)

_HOST_KIND = {
    "github.com": "git",
    "gitlab.com": "git",
    "bitbucket.org": "git",
    "gitee.com": "git",
    "huggingface.co": "model-hub",
    "zenodo.org": "archive",
    "codeocean.com": "capsule",
    "figshare.com": "archive",
    "osf.io": "archive",
    "data.mendeley.com": "archive",
    "sourceforge.net": "git",
}


@dataclass
class CodeLink:
    url: str
    host: str
    kind: str
    source: str  # "abstract" | "comment"


@dataclass
class Paper:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    abstract: str
    comment: str
    doi: str
    journal_ref: str
    topics: list[str] = field(default_factory=list)
    code_links: list[dict] = field(default_factory=list)
    code_promised: bool = False
    github_repos: list[dict] = field(default_factory=list)

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _retrying_get(url: str, headers: dict | None = None, attempts: int = 4, timeout: int = 60) -> bytes | None:
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return _http_get(url, headers=headers, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) and attempt < attempts:
                reset = exc.headers.get("x-ratelimit-reset")
                wait = delay
                if reset:
                    try:
                        wait = max(delay, float(reset) - time.time() + 2)
                    except ValueError:
                        pass
                wait = min(wait, 120.0)
                print(f"  rate-limited ({exc.code}); sleeping {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                delay *= 2
                continue
            if exc.code == 404:
                return None
            if attempt >= attempts:
                print(f"  HTTP {exc.code} for {url}", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= attempts:
                print(f"  network error for {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# Stage 1: arXiv
# ---------------------------------------------------------------------------
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_code_links(abstract: str, comment: str) -> tuple[list[CodeLink], bool]:
    links: list[CodeLink] = []
    seen: set[str] = set()
    for source, text in (("abstract", abstract), ("comment", comment)):
        for match in _URL_RE.finditer(text or ""):
            url = match.group(0).rstrip(_TRAILING)
            # Abstracts frequently wrap URLs; drop obvious truncation artifacts.
            url = url.split("\\")[0]
            host = match.group(1).lower()
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            links.append(CodeLink(url=url, host=host, kind=_HOST_KIND.get(host, "other"), source=source))
    promised = bool(_PROMISE_RE.search(abstract or "")) or bool(_PROMISE_RE.search(comment or ""))
    return links, promised


def parse_entry(entry: ET.Element) -> Paper:
    raw_id = _clean(entry.findtext(f"{ATOM}id"))
    # e.g. http://arxiv.org/abs/2504.11581v2
    tail = raw_id.rsplit("/", 1)[-1]
    if "v" in tail and re.search(r"v\d+$", tail):
        arxiv_id, version = tail.rsplit("v", 1)
    else:
        arxiv_id, version = tail, ""
    abstract = _clean(entry.findtext(f"{ATOM}summary"))
    comment = _clean(entry.findtext(f"{ARX}comment"))
    links, promised = extract_code_links(abstract, comment)
    categories = [
        el.get("term", "")
        for el in entry.findall(f"{ATOM}category")
        if el.get("term")
    ]
    primary_el = entry.find(f"{ARX}primary_category")
    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=_clean(entry.findtext(f"{ATOM}title")),
        authors=[_clean(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
        published=_clean(entry.findtext(f"{ATOM}published")),
        updated=_clean(entry.findtext(f"{ATOM}updated")),
        primary_category=(primary_el.get("term", "") if primary_el is not None else ""),
        categories=categories,
        abstract=abstract,
        comment=comment,
        doi=_clean(entry.findtext(f"{ARX}doi")),
        journal_ref=_clean(entry.findtext(f"{ARX}journal_ref")),
        code_links=[asdict(link) for link in links],
        code_promised=promised,
    )


def query_arxiv(
    topic: str,
    expression: str,
    date_from: str,
    date_to: str,
    max_results: int,
    page_size: int = 100,
) -> list[Paper]:
    window = f"submittedDate:[{date_from}0000 TO {date_to}2359]"
    # arXiv wants '+' as the token separator; everything else is percent-encoded.
    search = f"({expression}) AND {window}".replace(" ", "+")
    papers: list[Paper] = []
    for start in range(0, max_results, page_size):
        encoded = urllib.parse.quote(search, safe="+:")
        params = (
            f"search_query={encoded}"
            f"&start={start}&max_results={min(page_size, max_results - start)}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        url = f"{ARXIV_ENDPOINT}?{params}"
        raw = _retrying_get(url)
        time.sleep(ARXIV_DELAY_S)
        if raw is None:
            break
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            print(f"  XML parse error on {topic} start={start}: {exc}", file=sys.stderr)
            break
        entries = root.findall(f"{ATOM}entry")
        if not entries:
            break
        for entry in entries:
            paper = parse_entry(entry)
            paper.topics = [topic]
            papers.append(paper)
        if len(entries) < page_size:
            break
    return papers


def merge_papers(batches: Iterable[Sequence[Paper]]) -> dict[str, Paper]:
    merged: dict[str, Paper] = {}
    for batch in batches:
        for paper in batch:
            existing = merged.get(paper.arxiv_id)
            if existing is None:
                merged[paper.arxiv_id] = paper
            else:
                for topic in paper.topics:
                    if topic not in existing.topics:
                        existing.topics.append(topic)
    return merged


# ---------------------------------------------------------------------------
# Stage 2: GitHub verification
# ---------------------------------------------------------------------------
_BAD_REPO_SUFFIX = re.compile(r"\.(git|md|pdf|zip|html?)$", re.IGNORECASE)
# Paths that are org-level or site-level, never a repository.
_NON_REPO_OWNERS = {"orgs", "sponsors", "features", "about", "topics", "collections"}


def github_candidates(paper: Paper) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for link in paper.code_links:
        if link["host"] != "github.com":
            continue
        match = _GITHUB_REPO_RE.search(link["url"])
        if not match:
            continue
        owner = match.group("owner")
        repo = _BAD_REPO_SUFFIX.sub("", match.group("repo"))
        if owner.lower() in _NON_REPO_OWNERS or not repo or repo in {".", ".."}:
            continue
        key = (owner.lower(), repo.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((owner, repo))
    return out


def _repo_record(data: dict) -> dict:
    licence = data.get("license") or {}
    return {
        "full_name": data.get("full_name"),
        "html_url": data.get("html_url"),
        "description": (data.get("description") or "")[:400],
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "language": data.get("language"),
        "license": licence.get("spdx_id") or licence.get("key"),
        "created_at": data.get("created_at"),
        "pushed_at": data.get("pushed_at"),
        "size_kb": data.get("size", 0),
        "archived": bool(data.get("archived")),
        "is_fork": bool(data.get("fork")),
        "topics": data.get("topics", []),
        "default_branch": data.get("default_branch"),
    }


def search_github_by_arxiv_id(arxiv_id: str, token: str | None, limit: int = 3) -> list[dict]:
    """Find repositories whose README cites this arXiv ID.

    Many papers put the code URL only in the PDF, so the abstract scan misses
    them. The reverse lookup -- ask GitHub which repositories cite the paper --
    recovers those and additionally surfaces third-party reimplementations,
    which are often more usable than the authors' dump.

    Repository search is capped at 30 requests/minute even when authenticated,
    so callers must budget these calls.
    """
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query = urllib.parse.quote(f'"{arxiv_id}" in:readme', safe="")
    raw = _retrying_get(
        f"{GITHUB_API}/search/repositories?q={query}&sort=stars&order=desc&per_page={limit}",
        headers=headers,
        timeout=45,
    )
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for item in data.get("items", [])[:limit]:
        record = _repo_record(item)
        record["discovered_via"] = "github-readme-search"
        out.append(record)
    return out


def verify_github(owner: str, repo: str, token: str | None, cache: dict) -> dict | None:
    key = f"{owner}/{repo}".lower()
    if key in cache:
        return cache[key]
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    raw = _retrying_get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers, timeout=45)
    time.sleep(GITHUB_DELAY_S)
    if raw is None:
        cache[key] = None
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cache[key] = None
        return None
    record = _repo_record(data)
    record["discovered_via"] = "abstract-url"
    cache[key] = record
    return record


# ---------------------------------------------------------------------------
# Scoring: rank the code-backed papers for review depth
# ---------------------------------------------------------------------------
def score_paper(paper: Paper, now: datetime) -> float:
    """Heuristic usefulness score for a code-backed paper.

    Rewards: a live repository, permissive licence, recent pushes, community
    traction, and venue acceptance. Penalises archived or empty repositories.
    """
    if not paper.github_repos:
        return 0.0
    best = 0.0
    for repo in paper.github_repos:
        score = 10.0
        stars = repo.get("stars", 0) or 0
        score += min(25.0, 6.0 * (stars ** 0.5))
        if repo.get("license") in {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause"}:
            score += 12.0
        elif repo.get("license") in {"GPL-3.0", "AGPL-3.0", "LGPL-3.0", "MPL-2.0"}:
            score += 5.0
        pushed = repo.get("pushed_at")
        if pushed:
            try:
                age_days = (now - datetime.fromisoformat(pushed.replace("Z", "+00:00"))).days
                score += max(0.0, 14.0 - age_days / 60.0)
            except ValueError:
                pass
        size = repo.get("size_kb", 0) or 0
        if size < 20:
            score -= 8.0  # effectively an empty placeholder repo
        elif size > 500:
            score += 3.0
        if repo.get("archived"):
            score -= 6.0
        if repo.get("is_fork"):
            score -= 4.0
        best = max(best, score)
    if paper.journal_ref or paper.doi:
        best += 6.0
    if re.search(r"accepted|camera[- ]ready|to appear|ICASSP|NeurIPS|ICML|CVPR|IJCAI|AAAI|KDD|DCASE|PHM",
                 paper.comment, re.IGNORECASE):
        best += 5.0
    return round(best, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--from", dest="date_from", default="2025-01-01", help="submission window start (YYYY-MM-DD)")
    ap.add_argument("--to", dest="date_to", default=None, help="submission window end (YYYY-MM-DD), default today")
    ap.add_argument("--max-per-topic", type=int, default=120, help="max arXiv hits per topic")
    ap.add_argument("--topics", default=None, help="comma-separated subset of topic keys")
    ap.add_argument("--skip-github", action="store_true", help="skip GitHub verification stage")
    ap.add_argument(
        "--readme-search-budget",
        type=int,
        default=0,
        help="number of GitHub README reverse-lookups to spend (30/min cap; 0 disables)",
    )
    args = ap.parse_args(argv)

    date_from = args.date_from.replace("-", "")
    date_to = (args.date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")).replace("-", "")
    wanted = set(args.topics.split(",")) if args.topics else None
    topics = [(k, v) for k, v in TOPICS if wanted is None or k in wanted]

    print(f"harvesting {len(topics)} topics, window {date_from}..{date_to}", file=sys.stderr)
    batches = []
    for idx, (topic, expression) in enumerate(topics, 1):
        print(f"[{idx}/{len(topics)}] {topic}", file=sys.stderr)
        batch = query_arxiv(topic, expression, date_from, date_to, args.max_per_topic)
        print(f"    {len(batch)} hits", file=sys.stderr)
        batches.append(batch)

    merged = merge_papers(batches)
    print(f"{len(merged)} unique papers", file=sys.stderr)

    with_links = [p for p in merged.values() if p.code_links]
    print(f"{len(with_links)} papers advertise a code/data URL", file=sys.stderr)

    token = os.environ.get("GITHUB_TOKEN") or None
    if not args.skip_github:
        cache: dict = {}
        total = len(with_links)
        for idx, paper in enumerate(with_links, 1):
            cands = github_candidates(paper)
            if not cands:
                continue
            if idx % 25 == 0:
                print(f"    verifying {idx}/{total}", file=sys.stderr)
            for owner, repo in cands:
                record = verify_github(owner, repo, token, cache)
                if record:
                    paper.github_repos.append(record)
        live = sum(1 for p in with_links if p.github_repos)
        print(f"{live} papers have a resolvable GitHub repository", file=sys.stderr)

    # Stage 3: reverse lookup. Priority order -- papers that promise code but
    # printed no URL first, then everything else, newest first.
    if args.readme_search_budget > 0:
        pending = [p for p in merged.values() if not p.github_repos]
        def _recency_key(paper: Paper) -> tuple:
            try:  # modern IDs are YYMM.NNNNN; pre-2007 IDs are archive/YYMMNNN
                numeric = int(paper.arxiv_id.replace(".", "").split("/")[-1])
            except ValueError:
                numeric = 0
            return (0 if paper.code_promised else 1, -numeric)

        pending.sort(key=_recency_key)
        budget = min(args.readme_search_budget, len(pending))
        print(f"reverse-looking-up {budget} papers on GitHub (of {len(pending)} without repos)", file=sys.stderr)
        found = 0
        for idx, paper in enumerate(pending[:budget], 1):
            hits = search_github_by_arxiv_id(paper.arxiv_id, token)
            if hits:
                paper.github_repos.extend(hits)
                found += 1
            if idx % 50 == 0:
                print(f"    {idx}/{budget} searched, {found} matched", file=sys.stderr)
            time.sleep(2.2)  # 30 searches/minute ceiling
        print(f"reverse lookup matched {found} papers", file=sys.stderr)

    now = datetime.now(timezone.utc)
    payload = {
        "generated_at": now.isoformat(),
        "window": {"from": args.date_from, "to": args.date_to or now.strftime("%Y-%m-%d")},
        "topics": [t for t, _ in topics],
        "counts": {
            "unique_papers": len(merged),
            "with_code_url": len(with_links),
            "with_live_github": sum(1 for p in merged.values() if p.github_repos),
        },
        "papers": [
            {**asdict(p), "abs_url": p.abs_url, "score": score_paper(p, now)}
            for p in sorted(merged.values(), key=lambda p: (-score_paper(p, now), p.arxiv_id))
        ],
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
