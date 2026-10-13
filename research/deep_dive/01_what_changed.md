# What changed in 2025–2026

> Synthesis of the 45 code-backed papers in
> `research/code_backed_papers_2025_2026.md`, read against the upstream
> `resources/academic_papers.md` (32 papers, mostly 2019–2024).
>
> Claims below are attributed to the papers that make them. Nothing here was
> independently reproduced — no repository in the list was executed.

The upstream list answers "which architecture should we use?". The 2025–2026
literature has largely stopped asking that question. Five shifts matter for this
platform, roughly in descending order of how much they should change the plans.

---

## 1. Evaluation methodology stopped being a footnote and became the finding

The single most consistent theme in this window is that **previously reported
numbers were wrong, and the mechanism was the train/test split.** This is no
longer one corrective paper; it is a cluster, and two entries are infrastructure
rather than argument.

**[2607.16493](https://arxiv.org/abs/2607.16493)** — *Leakage-Robust Evaluation
and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning*
([CALIBRA](https://github.com/iamsandeshdhital/CALIBRA), MIT) studies a
multi-task fault-classification + RUL model on C-MAPSS, NASA IMS and the UCI
Hydraulic System dataset. Its headline claim: naive sliding-window splitting
inflates classification accuracy **from a genuine 20–60 % to 99.9 %** — or
collapses it to 0 % through degenerate class representation, depending on where
the cut lands. It proposes a chunk-based, leakage-audited protocol.

Read that range again. The gap between the honest number and the leaked number
is not a few points; the leaked number is in a different category of claim. Any
model comparison in this field that does not state its split protocol is
uninterpretable.

**[2605.28345](https://arxiv.org/abs/2605.28345)** — *Picid: A Modular Evaluation
Infrastructure for Reproducible PHM*
([EPFL-IMOS/picid](https://github.com/EPFL-IMOS/picid), 47 MB of Python)
formalises the PHM evaluation pipeline as an executable protocol — data splits,
preprocessing, label alignment, temporal windowing, metrics — and *enforces*
deterministic, leakage-safe dataset construction across detection, diagnosis and
prognosis. The fact that a lab thought this was worth building is itself the
finding: ad-hoc protocol choices were costing the field comparability.

**[2607.08273](https://arxiv.org/abs/2607.08273)** — *Empirical Calibration and
Conditional-Reliability Diagnostics for Bearing RUL under Operating-Regime Shift*
(MIT, 60 MB, authors' own link) goes further: it defines held-out units by
**load–speed regime**, not by recording, and reports normalised MAE 0.1477 *with*
empirical 90 % interval coverage under strict train/validation/calibration/test
separation. It states plainly that "convenient mixed splits can hide that
failure".

**[2606.29339](https://arxiv.org/abs/2606.29339)** — *Two kinds of robustness are
not the same* (MIT, archived at Zenodo) disentangles two things routinely
conflated: tolerance to **sensor failure** and robustness to **low SNR**. It
builds a unified 8-channel/256-sample benchmark from three physically distinct
real sources — Hi-net seismic, Utah FORGE borehole DAS, and MAFAULDA industrial
vibration — and argues that architectural complexity is often credited with
robustness it does not have. For a fleet product these are genuinely different
requirements: a dead accelerometer axis and a machine running quietly under light
load need different responses.

**What this means for us.** `core/pdmcore/eval/splits.py` already refuses random
splits and ships a `leakage_report` audit. This cluster says that is table
stakes, and adds two requirements we did not have:

- **Regime-based held-out units**, not just unit-based. Our
  `OperatingBaseline` bins by speed and load for *baselining*; evaluation should
  hold out whole load–speed regimes as well, per 2607.08273.
- **Separate the two robustness axes in test**: inject channel dropout and
  inject noise as distinct test conditions, and report them separately.

---

## 2. Time-series foundation models became a frozen feature extractor you can afford

In the upstream list, "foundation model" means RmGPT (2409.17604) — a
domain-specific model trained for fault diagnosis. In this window the pattern
inverted: take a **general** pretrained time-series model, freeze it, and train a
small head.

**[2606.11990](https://arxiv.org/abs/2606.11990)** — *Time-Series Foundation Model
Embeddings for RUL* (accepted EUSIPCO 2026;
[replication repo](https://github.com/deshpandetanmay/chronos2-rul-replication),
MIT) uses **Chronos-2 frozen** as a backbone and trains a small regression head,
reporting consistent improvement over recurrent baselines on real industrial
sensor data from two device types — without feature engineering or large labelled
sets.

**[2601.22631](https://arxiv.org/abs/2601.22631)** — *PEFT-MuTS* makes the
sharper claim: parameter-efficient fine-tuning on a cross-domain pretrained
time-series representation model enables **few-shot** RUL, and — explicitly
"contrary to the widely held view that knowledge transfer in RUL prediction can
only occur within similar devices" — that the transfer does **not** need to come
from similar equipment.

**[2604.20122](https://arxiv.org/abs/2604.20122)** — *Adaptive Conformal Anomaly
Detection with Time Series Foundation Models* sits on
[ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)
(910 stars, Apache-2.0), i.e. a maintained, permissively licensed TSFM stack
rather than a research dump.

**What this means for us.** This is a genuinely new option for the cold-start
problem, and it is cheap to test: a frozen backbone plus a 10 k-parameter head is
a gateway-class workload, not a training cluster. It does **not** replace the
unsupervised path — a frozen TSFM still needs labels for its head, and a new site
has none — but it is a strong candidate for the *second* phase, where a handful
of confirmed failures exist and a per-site classifier must be trained on them.
Added to `plans/04_ml_ai_engine.md` as a Phase-2 track, behind the
unsupervised-first commitment.

The honest caveat: both papers evaluate on C-MAPSS-class multivariate sensor
streams at low sample rates. Neither shows a TSFM ingesting 25.6 kHz vibration.
The plausible architecture is TSFM-over-features (our 63-float vector at one
frame per acquisition), not TSFM-over-waveform, and that remains untested.

---

## 3. Calibrated uncertainty moved from "nice to have" to standard practice

Four of the 45 entries are centrally about whether a prediction's confidence can
be trusted, which is a visible change in emphasis.

- **[2607.08273](https://arxiv.org/abs/2607.08273)** forms RUL intervals by
  empirical residual calibration and reports *coverage*, not just MAE.
- **[2604.20122](https://arxiv.org/abs/2604.20122)** applies **adaptive conformal**
  prediction to anomaly detection, giving distribution-free coverage guarantees
  that survive drift.
- **[2511.12240](https://arxiv.org/abs/2511.12240)** — *SCI* wraps an existing
  model in a closed-loop "metacognitive" controller that monitors a normalised
  entropy confidence signal and decides whether to stop, continue sampling, or
  escalate — spending more computation on ambiguous cases. For an edge device
  with a hard energy budget, "spend more only when uncertain" is exactly the
  right control structure.
- **[2606.12486](https://arxiv.org/abs/2606.12486)** (Scania Component X) builds
  its whole method on the assumption that component wear is **monotonically
  non-decreasing**, and uses that as a constraint — a physics prior that doubles
  as a calibration check.

**What this means for us.** `pdmcore.eval.metrics` already reports Brier score
and expected calibration error, and `pdmcore.health.rul` returns a distribution
with an explicit `insufficient_evidence` state. The addition this cluster
suggests is **conformal intervals** rather than particle-filter quantiles for the
anomaly score, because conformal coverage is distribution-free and therefore
survives the covariate shift that a plant's seasons guarantee. Worth a spike;
it would replace the Gumbel tail extrapolation in
`AnomalyDetector.threshold_at_fpr`.

Also worth adopting directly: the monotonicity constraint from 2606.12486. A
health index that is permitted to go down hides a developing fault behind a
favourable load change.

---

## 4. INT4 is now a defensible edge target, not just INT8

**[2604.08474](https://arxiv.org/abs/2604.08474)** — *Quantization Impact on the
Accuracy and Communication Efficiency Trade-off in Federated Learning for
Aerospace Predictive Maintenance* (MIT, authors' own link) takes a deliberately
tiny 1-D CNN — **AeroConv1D, 9 697 parameters** — trained by federated learning
on C-MAPSS under a non-IID client partition, and evaluates symmetric uniform
quantisation at 32/8/4/2 bits over 10 seeds. Its claim: **INT4 is statistically
indistinguishable from FP32** (p = 0.341 on FD001, p = 0.264 MAE on FD002) while
giving 8× reduction.

A multi-seed, p-value-reporting quantisation study is more rigorous than the norm
in this literature, and the model size is the same order as ours (our Deep SVDD
60-24-12 is ~1.7 k parameters).

**[2606.24173](https://arxiv.org/abs/2606.24173)** benchmarks lightweight
transformers (DistilBERT, TinyBERT-4L/6L, MobileBERT) against classical ML
(RF, XGBoost, SVM, logistic regression) for on-device fault detection across
C-MAPSS, SECOM and UCI AI4I, measuring F1, AUC, model size and **CPU latency**,
with INT8 dynamic quantisation and a two-stage adaptive cascade. The useful part
is the comparison discipline, not the transformer: classical baselines are
included and measured on the same axes.

**What this means for us.** Two concrete changes to the hardware branches:

1. Our quantisation target should be **INT8 with an INT4 evaluation path**, not
   INT8 only. On the Alif Ensemble E7 the Ethos-U55 NPU handles INT8 natively;
   INT4 weight packing halves flash pressure and is worth measuring.
2. Keep classical baselines in every benchmark table. 2606.24173's structure —
   accuracy *and* size *and* latency, classical *and* deep — is the right shape
   for the per-kit comparison tables, and the per-kit READMEs follow it.

---

## 5. Domain adaptation went source-free, and federation went serverless

Both shifts are driven by the same constraint: **you cannot move the data.**

**[2503.08749](https://arxiv.org/abs/2503.08749)** — *Source-free domain
adaptation based on label reliability* ([SDALR](https://github.com/BdLab405/SDALR),
41 stars — the most-starred genuinely on-topic bearing repository in the window)
adapts a bearing-fault model to a new domain **without access to the source
data**, using a data-augmentation-based label-voting strategy that exploits both
reliable *and* unreliable pseudo-labels. Source-free matters commercially: it
lets a model trained on customer A's machines be adapted to customer B without
either customer's data leaving their site.

**[2510.16033](https://arxiv.org/abs/2510.16033)** — *ISGFAN* targets the case
where severe noise **and** domain shift coexist, which prior transfer methods
assume away, using information separation with an orthogonality loss.

**[2512.02610](https://arxiv.org/abs/2512.02610)** — *TACDA* argues that
domain-invariant features alone are insufficient for cross-domain RUL because
they discard target-specific information and degradation-stage consistency.

**[2609.35792](https://arxiv.org/abs/2609.35792)** — a **matched-protocol**
comparison of synchronous ring gossip against FedAvg, isolated local training and
a centralised reference for a stacked LSTM on C-MAPSS — same implementation,
architecture, initialisation, optimiser, split and training budget, with one
terminal window per test engine to avoid statistical inflation. Gossip removes
the aggregation server entirely.

**What this means for us.** Source-free adaptation is the technically correct
answer to the multi-tenant transfer problem in `plans/05_cloud_platform.md`, and
it is a better answer than federated learning for the specific case of
*onboarding a new customer with a model learned from existing ones*: no shared
training round, no server, no data movement — just the weights. Federated
learning remains the right tool for continuous multi-site improvement within one
customer.

---

## New open data in this window

New datasets are worth more to us than new architectures, because the simulator
in `core/pdmcore/sim` can only validate the pipeline, never the physics.

| Dataset | arXiv | What it adds |
|---|---|---|
| Marine Engine Fault Dataset | [2607.19444](https://arxiv.org/abs/2607.19444) | Turbocharged 3-cylinder marine diesel on a testbed, 30–90 % load sweep plus 5 anomaly classes introduced **after** stabilised fault-free operation. Documented operating conditions and subsystem-level interventions — rare and valuable. |
| Scania Component X | [2606.12486](https://arxiv.org/abs/2606.12486) | Heavy-duty truck fleet PdM with a monotonic-wear formulation; real fleet data rather than a testbed. |
| Synthetic fibre rope RUL | [2605.04262](https://arxiv.org/abs/2605.04262) | ~34 700 images, 11 Dyneema SK75/78 rope samples, cyclic fatigue at 7 axial load levels, full degradation lifecycle. Imagery, not vibration, but a genuine run-to-failure lifecycle. |
| DCASE 2026 Task 2 | [2606.01578](https://arxiv.org/abs/2606.01578) | The current first-shot anomalous-sound-detection task definition and baseline. The successor to the MIMII line in the upstream list. |
| MAFAULDA (re-used) | [2606.29339](https://arxiv.org/abs/2606.29339) | Mapped into a cross-source robustness benchmark alongside seismic and DAS data. |

---

## What did *not* change

Worth stating, because it validates keeping the existing design rather than
chasing the literature:

- **Envelope analysis and spectral kurtosis remain the technique of record** for
  bearing faults. Nothing in this window displaces them;
  [2606.26317](https://arxiv.org/abs/2606.26317) (PG-AMF) generalises adaptive
  moment features but is still operating in exactly that framework, and still
  compares against statistical descriptors.
- **Unsupervised / one-class detection is still the only day-one deployable
  option.** No paper in this window offers a way around the absence of labelled
  failures at a new site.
- **Nobody solved cross-machine generalisation.** Source-free adaptation, PEFT
  and TSFM embeddings all *reduce* the data needed on the target; none removes
  the need for target data entirely.
- **C-MAPSS is still over-used.** Nine of the 45 entries evaluate on it. It is a
  simulated turbofan dataset with no vibration channel, and a decade of
  architecture comparisons on it has not transferred to rotating-machinery
  condition monitoring. Our own evaluation should not add to that pile.
