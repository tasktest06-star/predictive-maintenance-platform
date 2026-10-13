# Method: how the code-backed list was produced

> Reproduce with `make research` (or the two commands at the bottom). Everything
> below is derived from `research/data/harvest_20261013.json`, which is committed
> so that every row in the reading list is traceable to the API response that
> produced it.

## The question this answers

The upstream `resources/academic_papers.md` lists 32 papers chosen for relevance.
Relevance is the right first filter, but it is not the filter an engineering team
needs when deciding what to build on. The question here is narrower and more
useful:

> Of the recent literature on machine condition monitoring, which papers come
> with code we can actually read, run and — licence permitting — reuse?

That is a question about artefacts, not ideas, and it has a verifiable answer.

## Pipeline

```
arXiv Atom API                GitHub REST API                 filters
──────────────                ───────────────                 ───────
22 topic queries      ─┐
2025-01-01..2026-10-13 ├─► 1639 unique papers
sorted by submission  ─┘        │
                                ├─► 103 print a code URL in the abstract
                                │   or author comment        (6.3 %)
                                │        │
                                │        └─► GET /repos/{owner}/{repo}
                                │            → 92 resolve
                                │
                                └─► reverse lookup: 900 README searches
                                    "<arxiv_id>" in:readme
                                    → 239 further papers matched
                                             │
                                    331 papers with a resolvable repository
                                             │
                                             ├─► −140  aggregator repos only
                                             ├─► −146  below relevance threshold
                                             │
                                             └─►  45  curated
```

### Stage 1 — arXiv harvest

22 queries (`research/tools/harvest_arxiv_code.py:TOPICS`) over a 21-month
submission window, paged at 100 results, 3.2 s between requests per arXiv's
request.

One practical note that cost an hour: arXiv's abstract search matches **literal
phrases**. `abs:"bearing fault diagnosis"` returns 47 hits *of all time*, while
`abs:bearing AND abs:fault` returns 80 in this window alone. Long phrases look
precise and are actually just narrow. The topic matrix therefore uses short
phrases ANDed with single terms, and the measured hit count for each query is
recorded in a comment next to it.

### Stage 2 — code links from the paper itself

Regex over abstract and author-comment fields for GitHub, GitLab, Bitbucket,
Gitee, Hugging Face, Zenodo, Code Ocean, Figshare, OSF and Mendeley Data URLs,
plus a separate "code is available" phrase detector so a *promise* of code can be
distinguished from a *link* to code.

**Only 6.3 % of papers print a code URL in the abstract.** Treating that as
"papers with code" — which is the obvious implementation — would have missed
roughly three quarters of the papers that do have code. The link usually lives in
the PDF, in a footnote, or in a GitHub README that cites the paper.

### Stage 3 — reverse lookup

For the 1547 papers with no abstract URL, ask GitHub the inverse question:

```
GET /search/repositories?q="2607.08273"+in:readme&sort=stars
```

239 of 900 searched (27 %) matched at least one repository. This also surfaces
third-party reimplementations, which are frequently more usable than the
authors' own upload.

Repository search is capped at 30 requests/minute even when authenticated, so
this stage is budgeted (`--readme-search-budget`) and prioritised: papers that
*promise* code without printing a URL are searched first.

### Stage 4 — filtering (`research/tools/curate.py`)

The reverse lookup over-reports in a specific, systematic way, and the first
version of this list was visibly wrong because of it.

**Aggregator repositories.** Repositories like `*-arxiv-daily`, `awesome-*` and
paper-list repos cite thousands of arXiv IDs in their READMEs. The first run
ranked `BaiShuanghao/my_arXiv_daily` (217 stars) as the "code" for eight
unrelated papers, and `halsay/ASR-TTS-paper-daily` (513 stars) as the code for a
paper on analog resonant networks. Two filters remove them:

1. a name/description pattern (`arxiv-daily`, `paper-daily`, `awesome-`, `^arxiv-`,
   `*-list`, `literature`, …);
2. more robustly, **any repository cited by ≥ 3 different papers via README
   search**. Real code is cited by one paper, occasionally two when a lab reuses
   a repository across a pair of papers. Three is already an aggregator.

54 repositories were dropped this way, eliminating 140 papers whose only "code"
was a listing.

**Relevance.** The topic matrix is deliberately broad, so it catches neighbouring
fields that share vocabulary: fault-tolerant robot locomotion, speech synthesis,
network intrusion detection. A weighted keyword score over title and abstract
(domain terms like *bearing*, *envelope spectrum*, *CWRU* weighted 3–4; method
terms like *transformer*, *contrastive* weighted 0.4–1.5; off-topic markers like
*vision-language-action*, *text-to-speech*, *grasp* weighted negative) with the
title weighted 1.75× removed a further 146.

The threshold (6.0) was calibrated by hand against this harvest: at 4.0 the list
still admitted a cryogenics LLM simulator, at 8.0 it began dropping genuine
TinyML deployment papers whose abstracts talk about microcontrollers rather than
bearings. Both failure directions are recorded in the audit trail at the end of
`research/code_backed_papers_2025_2026.md`, with examples, so the threshold can
be re-argued rather than taken on trust.

## What "has code" does and does not mean here

**It means:** on 2026-10-13 the GitHub API returned a 200 for that repository,
and its licence, size, star count, language and last-push date are recorded as
returned.

**It does not mean:**

- that the code reproduces the paper's numbers — nothing here was executed;
- that the code is the authors' (the `found` column distinguishes `abstract`, the
  authors' own link, from `readme`, a citation-based match);
- that the code is legally reusable — see below.

The `usability` score rates the *artefact*: licence, repository size, star count,
recency, archived/fork status, and whether the authors linked it themselves. It
is explicitly not a judgement of the research. A 0-star repository attached to a
careful leakage study is more valuable to this platform than a 2000-star
repository attached to something off-topic, which is why relevance is reported as
a separate column rather than folded into one number.

## The licence finding

Of the 45 curated entries, by best repository:

Counted per paper on its best repository (highest stars, then largest size --
the same tiebreak `slides/figures/make_figures.py` uses, so the table and the
deck agree):

| Licence | Count | What it means for a commercial platform |
|---|---:|---|
| MIT | 16 | Reusable with attribution |
| Apache-2.0 | 2 | Reusable with attribution; explicit patent grant |
| **No licence file** | **23** | **No reuse rights at all** |
| NOASSERTION | 4 | GitHub could not map it to an SPDX identifier; read the file |
| GPL / AGPL / LGPL / MPL | 0 | None in this window |

**27 of 45 code-backed papers in this window -- 60 % -- ship code with no usable
licence.** A public GitHub repository without a licence is "all rights
reserved": it can be read and learned from, but not copied into a product, not
vendored, and not redistributed. This is the single most actionable finding of
the whole exercise, and it is why `research/deep_dive/03_adoption_shortlist.md`
separates "read for method" from "vendor into the platform".

## Reproducing

```bash
# ~45 min: 22 arXiv queries, 103 repo lookups, 900 README searches
GITHUB_TOKEN=<token-with-public-read> python3 research/tools/harvest_arxiv_code.py \
    --out research/data/harvest_$(date +%Y%m%d).json \
    --from 2025-01-01 --max-per-topic 150 --readme-search-budget 900

# instant
python3 research/tools/curate.py --in research/data/harvest_$(date +%Y%m%d).json \
    --out-md research/code_backed_papers_2025_2026.md \
    --out-csv research/data/papers_with_code.csv \
    --out-json research/data/curated.json
```

Both tools are standard-library only, so they run in CI and on a gateway without
a virtualenv. Without a token the GitHub stages still work but are limited to 60
requests/hour, which is not enough for the full matrix.

## Known limitations of this method

- **Window.** 2025-01-01 onward. Foundational work (CWRU evaluation practice,
  Antoni's kurtogram, the MIMII dataset) predates it and is covered by the
  upstream `resources/academic_papers.md`, not here.
- **arXiv only.** `Mechanical Systems and Signal Processing`, `IEEE TIM` and the
  PHM Society conferences carry much of the serious condition-monitoring
  literature and are not on arXiv. This list is biased toward the ML-facing half
  of the field, and anyone using it should know that.
- **English-language abstracts.**
- **README-search recall is bounded by GitHub's indexing**, which lags and does
  not cover every repository.
- **A single snapshot.** Star counts and push dates age immediately; the
  `generated_at` field in the JSON is the timestamp they refer to.
