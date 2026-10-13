# Adoption shortlist: what to vendor, what to read, what to ignore

> Licence triage of the 45 curated entries, with a concrete action for each
> recommended item. Licences are as returned by the GitHub API on 2026-10-13 —
> **re-check before any code is copied.** An SPDX identifier from the API is a
> convenience, not legal advice, and a repository's licence can change.

## The headline constraint

| Category | Count | What you may do |
|---|---:|---|
| Permissive (MIT 16 / Apache-2.0 2) | **18** | Copy into the product with attribution |
| Copyleft (GPL / AGPL / LGPL / MPL) | 0 | — none in this window |
| **No licence (23) or unmappable (4)** | **27** | **Read only.** No copying, no vendoring, no redistribution |

**27 of 45 code-backed papers in this window ship code you have no legal right to
reuse.** A public GitHub repository with no `LICENSE` file is "all rights
reserved" by default. You may read it, learn the method, and write your own
implementation from the paper. You may not copy a function out of it.

That asymmetry drives this document: the most *useful* repositories in this list
are frequently not the most *usable* ones. `BdLab405/SDALR` (41 stars, the
best-regarded on-topic bearing repo in the window) is unlicensed. `EPFL-IMOS/picid`
— the evaluation infrastructure we would most like to adopt — is `NOASSERTION`.

A practical mitigation, and the right one for the two cases above: **open an
issue asking the authors to add a licence.** Most academic authors intend reuse
and simply did not add the file. It costs an issue and a week.

---

## Tier 1 — vendor or integrate now

Permissive licence, relevant, and the artefact is substantial enough to be worth
the integration cost.

### 1. `therealdeadbeef/aerospace-fl-quantization` — MIT, 12 MB, Python
**Paper:** [2604.08474](https://arxiv.org/abs/2604.08474) — INT4 quantisation
indistinguishable from FP32 (p = 0.34 / 0.26, 10 seeds) on a 9 697-parameter 1-D CNN.

**Action:** lift the **symmetric uniform quantisation harness and the multi-seed
significance protocol**, not the model. Our `pdmcore.models.quantize` needs
exactly this test structure: quantise at 32/8/4/2 bits, run N seeds, report a
p-value against FP32 rather than a single-run delta. Target: the Deep SVDD
detector on all eight `hw/*` branches.

**Caveat:** the result is for C-MAPSS RUL regression, not vibration anomaly
scoring. The *protocol* transfers; the conclusion must be re-measured.

### 2. `TakuyaFujimura/dcase-asd-toolkit` — MIT, Python
**Paper:** [2607.16678](https://arxiv.org/abs/2607.16678) — pseudo-label
distillation for discriminative anomalous sound detection.

**Action:** use as the **reference implementation for the acoustic channel**. It
is a toolkit rather than a one-paper dump, MIT-licensed, and sits in the DCASE
lineage that the upstream list already builds on (MIMII, 1909.09347). Pairs with
the DCASE 2026 Task 2 baseline below for a complete acoustic baseline.

### 3. `yucongzh/ECHO` — MIT, 1.2 MB, Python (highest usability score in the list)
**Paper:** [2508.14689](https://arxiv.org/abs/2508.14689) — frequency-aware
hierarchical encoding for **variable-length** signals.

**Action:** evaluate for the variable-acquisition-length problem. Our pipeline
fixes acquisition length at 1 s because the feature vector must be
fixed-dimensional; a device that shortens its acquisition to save energy
currently produces a non-comparable frame. Frequency-aware hierarchical encoding
is a direct answer. Medium priority, genuinely useful if it works.

### 4. `ZhixiangWang-CN/gossip-lstm-cmapss` — MIT, 15 MB, Python
**Paper:** [2609.35792](https://arxiv.org/abs/2609.35792) — matched-protocol
comparison of ring gossip vs FedAvg vs local vs centralised.

**Action:** take the **matched-protocol comparison harness** for
`plans/05_cloud_platform.md`. Its value is methodological: one implementation,
one architecture, one initialisation, one split, one training budget across all
four learning modes. That is the only way a federation-vs-alternatives decision
can be made honestly, and building it ourselves would cost a week.

### 5. `nbbllxx0/Empirical-Calibration-and-Conditional-Reliability-Diagnostics` — MIT, 61 MB, Python
**Paper:** [2607.08273](https://arxiv.org/abs/2607.08273) — bearing RUL under
operating-regime shift; nMAE 0.1477 with empirical 90 % coverage.

**Action:** the **regime-based held-out split and the empirical residual
calibration** are both directly applicable. This is the closest paper in the
window to our actual problem (bearings, time-varying load and speed, calibrated
intervals, 10 documented units) and it is MIT-licensed with the authors' own
link. Highest-priority read of the entire list.

### 6. `rajeevfee/PG_AMF-Bearing-Fault-Analysis-` — MIT, 1.3 MB, Python
**Paper:** [2606.26317](https://arxiv.org/abs/2606.26317) — parametric
generalised adaptive moment features; learned rather than hand-specified
descriptors.

**Action:** benchmark against our 14 hand-specified time-domain indicators
(`pdmcore.dsp.features`). If learned moment parameters beat fixed ones at
comparable cost, that is a drop-in upgrade to the edge feature extractor — the
parameters are constants in flash either way, so there is no runtime penalty.
Highest relevance score in the list (34.6).

### 7. `vishal-1344/sci` — MIT
**Paper:** [2511.12240](https://arxiv.org/abs/2511.12240) — closed-loop
metacognitive control; entropy-gated stop/continue/escalate.

**Action:** adopt the **control structure** for firmware tiering (see
`02_gap_analysis_vs_plans.md` → plan 02). The repository is mostly TeX, so this
is a design adoption rather than a code adoption.

### 8. `IBM/AssetOpsBench` — Apache-2.0, 112 MB, 2350 stars
**Papers:** [2506.03828](https://arxiv.org/abs/2506.03828),
[2604.23446](https://arxiv.org/abs/2604.23446) — benchmarking AI agents for
industrial asset operations.

**Action:** use as the **evaluation harness if and when an LLM-agent layer is
built** over the CMMS (the `implementation_llm-augmented` branch upstream). By
far the best-maintained repository in the list — Apache-2.0, pushed the day of
the harvest, 2350 stars, IBM-backed. Not needed for the sensing stack; the right
starting point for the agent layer.

### 9. `ISAO9/cephalon-v2` — MIT, Zenodo-archived
**Paper:** [2606.29339](https://arxiv.org/abs/2606.29339) — disentangling fault
tolerance from low-SNR robustness.

**Action:** adopt the **two-axis robustness test design**: channel dropout and
SNR degradation as separate, separately-reported test conditions. Cheap to
implement against `pdmcore.sim` (drop an axis; scale the noise term) and it
closes a real gap — we currently have no sensor-failure test at all.

---

## Tier 2 — read, reimplement if needed

Unlicensed or unmappable, but the method is worth having. Read the paper, read
the code for clarification, write our own.

| Repo | Paper | Why it matters | Licence risk |
|---|---|---|---|
| `BdLab405/SDALR` (41★) | [2503.08749](https://arxiv.org/abs/2503.08749) | Source-free domain adaptation for bearings — the right cross-tenant transfer mechanism (plan 05). Best-regarded on-topic repo in the window. | **none** — ask the authors |
| `EPFL-IMOS/picid` | [2605.28345](https://arxiv.org/abs/2605.28345) | Leakage-safe PHM evaluation infrastructure; closest external thing to `pdmcore.eval`. | NOASSERTION — read the file |
| `nttcslab/dcase2023_task2_baseline_ae` (105★) | [2606.01578](https://arxiv.org/abs/2606.01578) | The DCASE 2026 Task 2 baseline. NTT CS Lab; the reference point for acoustic first-shot ASD. | NOASSERTION |
| `fuen1590/PEFT-MuTS` | [2601.22631](https://arxiv.org/abs/2601.22631) | Few-shot RUL via PEFT on a cross-domain pretrained TS model; claims transfer across *dissimilar* devices. | none |
| `keyplay/TACDA` | [2512.02610](https://arxiv.org/abs/2512.02610) | Cross-domain RUL preserving target-specific and degradation-stage information. | none |
| `JYREN-Source/ISGFAN` | [2510.16033](https://arxiv.org/abs/2510.16033) | Transfer diagnosis where noise *and* domain shift coexist — our actual field condition. | none |
| `sarahmish/M2AD` | [2504.15225](https://arxiv.org/abs/2504.15225) | Multi-sensor multi-system anomaly detection — the fleet-scale shape. | none |
| `dhirajneupane/AIRL-MFD-DN` | [2607.22987](https://arxiv.org/abs/2607.22987) | Label-free fault detection via adversarial IRL; recovers an intrinsic health reward instead of using a static reconstruction margin. Interesting alternative framing for the health index. | none |
| `deshpandetanmay/chronos2-rul-replication` | [2606.11990](https://arxiv.org/abs/2606.11990) | Frozen Chronos-2 + small head for RUL (MIT, but a replication repo — check provenance). | MIT |
| `ibm-granite/granite-tsfm` (910★) | [2604.20122](https://arxiv.org/abs/2604.20122) | Apache-2.0 TSFM stack + adaptive conformal anomaly detection. **Tier 1 by licence**, listed here because the integration is Phase 2. | Apache-2.0 |

---

## Tier 3 — datasets to acquire

Worth more than any architecture in this list, because the simulator in
`core/pdmcore/sim` validates the *pipeline* and can never validate the *physics*.

| Dataset | Paper | Why | Status |
|---|---|---|---|
| Marine Engine Fault Dataset | [2607.19444](https://arxiv.org/abs/2607.19444) | Real engine, documented 30–90 % load sweep, 5 anomaly classes introduced **after** stabilised fault-free operation. The before/after structure is exactly what a commissioning-baseline design needs to be tested against. | Under review; check for the data link |
| Scania Component X | [2606.12486](https://arxiv.org/abs/2606.12486) | Real heavy-duty fleet, monotonic-wear formulation. Fleet-scale rather than testbed. | Public |
| DCASE 2026 Task 2 | [2606.01578](https://arxiv.org/abs/2606.01578) | Current acoustic first-shot ASD task + baseline; the live successor to MIMII. | Public |
| Synthetic fibre rope RUL | [2605.04262](https://arxiv.org/abs/2605.04262) | ~34 700 images, 11 rope samples, 7 load levels, full degradation lifecycle. Imagery not vibration, but a genuine run-to-failure lifecycle with ground truth. | MIT, public |

Together with CWRU, MFPT, MIMII and XJTU-SY from the upstream list, that is a
defensible validation suite. The gap that remains, and it is the important one:
**no public dataset covers a wireless MEMS sensor on a real industrial machine
over a multi-month degradation**, which is precisely our deployment. That has to
come from our own pilot, and `plans/00_overview.md` Phase 2 ("customer pilot with
20–50 sensors") is therefore on the critical path for the ML work, not parallel
to it.

---

## Explicitly not adopting

- **Anything evaluated only on C-MAPSS.** Nine entries. Simulated turbofan, no
  vibration channel, saturated as a benchmark. Using it makes results comparable
  to the literature and uninformative about this product.
- **Battery / Li-ion RUL work** ([2503.13558](https://arxiv.org/abs/2503.13558),
  [2603.22323](https://arxiv.org/abs/2603.22323),
  [2605.15351](https://arxiv.org/abs/2605.15351)). Methodologically adjacent and
  sometimes elegant — the Lie Generator Network result, extracting EIS-grade
  diagnostics from 60 s of pulse relaxation, is a nice piece of work — but a
  different asset class with different physics. Note 2503.13558 **has been
  withdrawn by arXiv for unverifiable authorship**; it stays in the generated
  list because the harvester records what the API returned, and that withdrawal
  is itself a useful signal about citing from automated harvests.
- **Quantum-annealing RUL** ([2606.18503](https://arxiv.org/abs/2606.18503)). The
  matched repository is a user profile page, not an implementation — a good
  example of why the reverse lookup needs the aggregator filter *and* human
  review.
- **Multimodal domain-generalisation benchmarks**
  ([2605.06643](https://arxiv.org/abs/2605.06643),
  [2512.24679](https://arxiv.org/abs/2512.24679)). Action-recognition-centric
  despite the framing; the modalities are not ours.

---

## Suggested sequencing

| When | Item | Cost | Payoff |
|---|---|---|---|
| Now | Two-axis robustness tests (`cephalon-v2` design) | 1 day | Closes a real gap: no sensor-failure test exists today |
| Now | Multi-seed quantisation protocol (`aerospace-fl-quantization`) | 2 days | Every per-kit INT8/INT4 claim becomes defensible |
| Now | Regime-based splits + residual calibration (2607.08273) | 3 days | The closest paper to our problem; directly improves `pdmcore.eval` |
| Now | Open licence issues on SDALR and picid | 1 hour | Converts the two best Tier-2 items into Tier 1 |
| Phase 1 | PG-AMF vs our fixed time-domain indicators | 3 days | Possible free upgrade to the edge feature set |
| Phase 1 | Adaptive conformal thresholds (`granite-tsfm`) | 1 week | Replaces parametric tail extrapolation; drift-robust |
| Phase 2 | Frozen TSFM + small head | 2 weeks | Few-shot classification once failures exist |
| Phase 2 | Source-free adaptation (SDALR reimplementation) | 3 weeks | Cross-tenant transfer without data movement |
| Phase 3 | AssetOpsBench for the agent layer | — | Only if an LLM layer is actually built |
