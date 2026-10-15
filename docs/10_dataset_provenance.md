# Dataset provenance, licensing, and what may be shipped

This repository's long-standing admission has been that **no model in it had
been validated on measured data**, with CWRU, MFPT and MIMII named as the gate.
That is now changing, and two things become true the moment a result depends on
700 MB of bearing data held outside the repository.

**A result is only reproducible if the bytes are pinned.** "Evaluated on CWRU"
is not a provenance statement. Several CWRU redistributions are in circulation
with different file subsets, different channel naming, and in some mirrors
resampled or truncated records. A conclusion drawn from one is not
automatically a conclusion about another.

**The licences are not the same, and one of them restricts what can be
shipped.** That is the part most likely to be discovered late, by a lawyer,
after a model has been trained.

So: the data is **not committed** — it is large, it is not ours to redistribute,
and in one case redistributing a derivative would impose a share-alike
obligation. A checksum manifest *is* committed:

```bash
python3 tools/dataset_manifest.py --write     # scan and record
python3 tools/dataset_manifest.py --verify    # has anything changed?
```

Manifest: [`docs/datasets/manifest.json`](datasets/manifest.json). **Any result
computed from one of these datasets should quote the relevant root's `digest`.**

---

## What is on disk, and where it came from

Verified by reading the shipped licence files and by hashing every file — not
from memory.

| Root | Files | Size | Source |
|---|---:|---:|---|
| `data_raw/cwru` | 29 | 153.9 MB | **Official** — `engineering.case.edu/sites/default/files/` |
| `data_raw/mfpt` | 47 | 159.2 MB | `github.com/mathworks/RollingElementBearingFaultDiagnosis-Data` |
| `dl_cwru_12kDE` | 40 | 140.3 MB | Official CWRU, 12 kHz drive-end set |
| `dl_mfpt` | 49 | 298.5 MB | Two further third-party mirrors — see [below](#the-two-weaker-mirrors) |

Paths are relative to the parent of this repository. They are outside the repo
deliberately: downloaded archives are untrusted data, and the convention here is
that each download gets its own directory, scripts live elsewhere, and any
Python that reads them runs with `-I`.

### The two CWRU copies agree

Checked rather than assumed, because a silent disagreement between two copies of
"CWRU" would corrupt every result built on them:

- `data_raw/cwru` holds 28 `.mat` files; `dl_cwru_12kDE` holds 40.
- All 28 overlapping filenames are **byte-identical** (SHA-256 match).
- `dl_cwru_12kDE` is a strict superset — the 12 extra files are `209`–`212`,
  `222`–`225` and others from the 12 kHz drive-end set.

So the two can be used interchangeably for the 28 shared records, and
`dl_cwru_12kDE` is the fuller one. **Prefer `dl_cwru_12kDE`** and say which was
used.

---

## Licences, and the one that restricts shipping

### CWRU — attribution expected, licence unpublished

| | |
|---|---|
| Owner | Case Western Reserve University |
| Canonical | <https://engineering.case.edu/bearingdatacenter> |
| Licence | **No formal licence is published.** The Data Center asks that use be acknowledged |
| Citation | Case Western Reserve University Bearing Data Center |
| Commercial use | **Unclear.** No licence grant exists to rely on |

Treat this as permission-to-use-with-attribution, *not* as public domain. The
absence of a licence is not the same as a permissive one: a commercial
deployment trained on CWRU needs its own legal review, and that review should
happen before the model is trained, not after.

### MFPT via MathWorks — CC BY-NC-SA 4.0, and the NonCommercial clause bites

Taken verbatim from the `license.txt` and `README.md` shipped inside the source
repository, which are present at `data_raw/mfpt/extracted/`:

> Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

> The data is provided under a Creative Commons
> Attribution-NonCommercial-ShareAlike 4.0 International License. Mathworks Inc.
> has obtained permission from the data owner Eric Bechhoefer to use the data
> for commercial purpose. Please contact the owner for any other commercial
> uses.

| | |
|---|---|
| Data owner | Eric Bechhoefer |
| Original publication | Society for Machinery Failure Prevention Technology; data at <http://data-acoustics.com/measurements/bearing-faults/bearing-2/> |
| Licence | CC BY-NC-SA 4.0 |
| Citation | Bechhoefer, Eric. *Condition Based Maintenance Fault Database for Testing Diagnostics and Prognostic Algorithms.* 2013 |

**Three consequences, and they are not theoretical:**

1. **NonCommercial.** A model trained on this data **may not be shipped
   commercially** without permission from the data owner. MathWorks obtained
   that permission for its own use; **that grant does not transfer to anyone
   else**, and the README says so explicitly. Any `production/` variant in this
   repository whose weights were fitted on MFPT is, as it stands, not shippable.
2. **ShareAlike.** A derivative of the dataset — a converted `.npz` cache, a
   resampled subset, a feature table — carries the same licence. That is a
   second reason not to commit a converted cache into a repository whose own
   licence differs: it would either violate ShareAlike or silently relicense the
   repository. Commit the conversion *script* and the checksum manifest instead,
   so a reader reproduces the cache rather than receiving it.
3. **Attribution.** The citation above must appear wherever a result derived
   from it is reported — including slides.

**Validation is the safe use.** Measuring a model's accuracy on MFPT and
reporting the number is ordinary scholarly use under BY-NC-SA. *Fitting
deployable weights* on it is the thing that creates the problem. The practical
rule for this repository: **use MFPT to test, use CWRU or site data to train,
and record which was which in the model card** (`docs/03_model_cards.md`).

### MIMII — named in the gate, not yet fetched

The third dataset named in `docs/07_production_readiness.md` is not on disk. It
is acoustic rather than vibration, distributed under CC BY-SA 4.0 — share-alike
but **not** non-commercial, so it is less restrictive than MFPT for deployment
purposes.

### The two weaker mirrors

`dl_mfpt` additionally contains:

- `mfpt_full/MFPT_Fault_Data_Sets-main` — 40 files, **no licence file present**
- `wuyanan/CWRU-and-MFPT-data-main` — 24 files, **no licence file present**

Both are third-party GitHub mirrors. Neither ships a licence, neither documents
what was modified, and the second mixes CWRU and MFPT records under a renaming
scheme (`0_0.mat`, `14_1.mat`, …) that discards the original filenames — which
are the only thing connecting a record to its documented fault size, load and
bearing position.

**Do not use these for a reported result.** They are kept because they were
fetched, and deleting evidence of what was downloaded is worse than labelling
it. The licensed, documented copies are `data_raw/mfpt` (MathWorks, licence
present) and `dl_cwru_12kDE` (official CWRU).

---

## The important subtlety: `data_raw/mfpt` is not "MFPT"

The MathWorks README states plainly that it is a **modification** of the
original data, listing which of the six original folders it kept and how it
reorganised them.

So a result reported as "evaluated on MFPT" is really "evaluated on the
MathWorks subset of MFPT". Those are different populations and the difference
matters for exactly the question this programme is asking — cross-dataset
generalisation. A model that transfers to a curated six-condition subset has not
been shown to transfer to the full set.

State which one was used. The manifest digest makes that checkable after the
fact.

---

## Checklist before any result is published

- [ ] `python3 tools/dataset_manifest.py --verify` passes
- [ ] The result quotes the dataset root's `digest`
- [ ] The result names *which* redistribution was used, not just "CWRU" or "MFPT"
- [ ] Splits are grouped by bearing/unit and the leakage audit was run
      (`pdmcore.eval.splits`) — consecutive windows from one recording share a
      bearing, a load and a noise realisation
- [ ] If MFPT was used for **fitting** rather than testing, the NonCommercial
      restriction is recorded in the model card and the deployment path is
      flagged
- [ ] Citations present in the document and on the slides
- [ ] No dataset bytes added to git — `git diff --stat` shows no `.mat`,
      `.npz` or archive

---

## What this does not settle

The manifest pins bytes and records licences. It does not establish that any of
these datasets is *representative*. CWRU is a 2 hp motor on a test rig with
seeded, machined defects at a handful of discrete sizes; MFPT likewise. Neither
contains a naturally propagating fault, a real mounting, a VFD, or a plant's
electrical and mechanical noise. A model that generalises across both has
cleared a real bar — the first real bar this repository has cleared — and is
still not a model that has seen a factory.
