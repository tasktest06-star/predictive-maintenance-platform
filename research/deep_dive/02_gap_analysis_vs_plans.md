# Gap analysis: what the 2025–2026 evidence changes in the seven product plans

> Read against `plans/00_overview.md` … `plans/07_electrical_monitoring.md`.
> Each row is an actionable change, not a reading recommendation. "No change"
> entries are included on purpose: knowing which bets the new evidence *supports*
> is as useful as knowing which it undermines.

---

## Summary table

| Plan | Change | Severity | Driver |
|---|---|---|---|
| 00 Overview | Add an evaluation-protocol gate to every phase exit | **High** | [2607.16493](https://arxiv.org/abs/2607.16493), [2605.28345](https://arxiv.org/abs/2605.28345) |
| 01 Sensor hardware | Keep the AE differentiator; drop the implied 400 kHz requirement for v1 | Medium | No code-backed AE work in window |
| 02 Edge firmware | Add INT4 evaluation path; add uncertainty-gated compute | Medium | [2604.08474](https://arxiv.org/abs/2604.08474), [2511.12240](https://arxiv.org/abs/2511.12240) |
| 03 IoT pipeline | No architectural change | None | — |
| 04 ML/AI engine | Add frozen-TSFM track as Phase 2; add conformal intervals | **High** | [2606.11990](https://arxiv.org/abs/2606.11990), [2601.22631](https://arxiv.org/abs/2601.22631), [2604.20122](https://arxiv.org/abs/2604.20122) |
| 05 Cloud platform | Prefer source-free adaptation over federated for customer onboarding | Medium | [2503.08749](https://arxiv.org/abs/2503.08749), [2609.35792](https://arxiv.org/abs/2609.35792) |
| 06 CMMS dashboard | Surface calibrated confidence and abstention as first-class UI states | Medium | [2607.08273](https://arxiv.org/abs/2607.08273), [2511.12240](https://arxiv.org/abs/2511.12240) |
| 07 Electrical monitoring | Thin evidence base; keep as planned, lower confidence | Low | Only 1 code-backed MCSA paper in window |

---

## 00 — Portfolio overview

**Change: make the evaluation protocol a phase-exit gate, not a work item.**

`plans/00_overview.md` puts "Begin CWRU/MFPT/MIMII dataset training for baseline
models" in Phase 0 and "Full fault classification" in Phase 2, with no statement
of how either is validated. Given that [2607.16493](https://arxiv.org/abs/2607.16493)
measures the leaked-vs-honest gap at **99.9 % vs 20–60 %**, an unvalidated
accuracy number is not weak evidence — it is no evidence, and it will be believed
by whoever reads the phase report.

Concretely, add to each phase exit:

- Every reported metric names its split protocol, and the split passes
  `pdmcore.eval.splits.leakage_report` with zero violations.
- Held-out units are **whole physical units and whole load–speed regimes**
  (2607.08273), not recordings.
- Sensor-failure robustness and low-SNR robustness are reported **separately**
  (2606.29339).
- Detection metrics are quoted at a stated alarm budget (partial AUC at the
  agreed FPR, alarms/day/100 assets), never as bare accuracy.

This is cheap — the machinery exists in `core/pdmcore/eval` — and it is the
difference between a pilot that converts and a pilot that gets quietly dropped
after the third false work order.

---

## 01 — Smart sensor hardware

The plan's stated differentiator is "true AE sensing (100–400 kHz)".

**Finding: the window contains no code-backed acoustic-emission work at all.**
The upstream list has two AE papers (2405.20887, 2502.11786); the 2025–2026
code-backed harvest has none. The acoustic work that *does* appear is all
audible-band machine sound (DCASE lineage: [2606.01578](https://arxiv.org/abs/2606.01578),
[2607.16678](https://arxiv.org/abs/2607.16678), [2508.14689](https://arxiv.org/abs/2508.14689)).

Two readings, and they point the same way:

1. AE is an instrumentation field, published in `Mechanical Systems and Signal
   Processing` and NDT venues that are not on arXiv and do not ship code. The
   absence is a sampling artefact of the method (see
   `00_method.md` → limitations), not evidence that AE does not work.
2. Either way, **there is no reusable code and no public AE benchmark dataset** to
   develop against. An AE channel in v1 means building the algorithms from
   scratch with no reference implementation and no dataset to validate on.

**Change:** keep AE as the long-term hardware differentiator and keep the analog
front-end provisioned on the PCB, but move the AE *algorithm* work out of v1 and
ship the 6 kHz-bandwidth MEMS path first — which is where all the reusable
evidence is. The `hw/adi-cn0549-piezo-ae` branch exists precisely to carry the
high-fidelity path as a parallel, lower-certainty track.

Also note [2606.29339](https://arxiv.org/abs/2606.29339) uses **distributed
acoustic sensing** (fibre) alongside vibration. Not a v1 option, but the first
code-backed appearance of DAS next to industrial condition monitoring, and worth
tracking.

---

## 02 — Edge firmware

**Change 1: add an INT4 evaluation path.** The plan commits to "TensorFlow Lite
Micro + CMSIS-NN" and INT8. [2604.08474](https://arxiv.org/abs/2604.08474)
reports INT4 statistically indistinguishable from FP32 (p = 0.34 / 0.26 across 10
seeds) on a 9 697-parameter 1-D CNN. Our Deep SVDD detector is ~1.7 k parameters
— the same regime. INT4 halves flash pressure and, on targets without an INT4
NPU path, can still be stored packed and unpacked to INT8 at load time.

This is an *evaluation* path, not a commitment: the claim is about C-MAPSS
regression, not vibration anomaly scoring, so it has to be re-measured on our
own task before it ships.

**Change 2: uncertainty-gated computation.** The plan's "power-aware sampling"
is schedule-driven. [2511.12240](https://arxiv.org/abs/2511.12240) (SCI)
proposes a closed loop: monitor a normalised-entropy confidence signal and decide
whether to stop, continue sampling, or escalate. Mapped onto our firmware this is
a genuinely better policy than a fixed schedule:

```
cheap pass   : time-domain indicators only          (~0.2 ms, every cycle)
if uncertain : + spectrum and band RMS              (~2 ms)
if uncertain : + envelope analysis and bearing evidence (~5 ms)
if uncertain : + raw waveform snapshot to cloud     (~300 ms radio)
```

The energy saving comes from the common case terminating at step 1, and the
diagnostic quality comes from the rare case reaching step 4. This matches the
three-tier hierarchy in the upstream list's paper 23 (2411.07168), but driven by
*measured uncertainty* rather than a fixed tier assignment — and it is directly
implementable against `AnomalyDetector.score` plus a threshold band.

**Change 3 (minor, already applied in code):** the kurtogram's `n_fft` is tied to
the expected fault rate, not a free parameter — SK goes blind when the fault rate
approaches the STFT frame rate. Pinned by
`core/tests/test_dsp_spectrum.py::test_spectral_kurtosis_needs_impulses_sparse_relative_to_frame_rate`.

---

## 03 — IoT data pipeline

**No architectural change.** Kafka + Flink + VictoriaMetrics is unaffected by
anything in this window; no paper here argues about transport or storage.

One operational addition worth making, and it comes from the firmware side rather
than the literature: **sequence-gap accounting per device per day**. A sensor
dropping 30 % of frames is indistinguishable from a healthy one on a trend chart.
`pdmcore.io.payload.sequence_gaps` implements it; the pipeline needs to aggregate
and alarm on it.

---

## 04 — ML/AI diagnostic engine

This plan changes the most.

**Change 1: add a frozen-TSFM track as a Phase 2 capability.** Keep the
unsupervised-first commitment for Phase 1 — nothing in this window offers a
day-one alternative — but add, for sites with a handful of confirmed failures:

```
63-float feature frames  ->  frozen time-series foundation model
                         ->  small supervised head (~10 k params)
```

Evidence: [2606.11990](https://arxiv.org/abs/2606.11990) (Chronos-2 frozen +
small head beats recurrent baselines on real industrial sensor data, EUSIPCO
2026) and [2601.22631](https://arxiv.org/abs/2601.22631) (PEFT on a cross-domain
pretrained TS model enables few-shot RUL, and transfer does *not* require similar
devices). Permissively licensed, maintained TSFM infrastructure exists
([granite-tsfm](https://github.com/ibm-granite/granite-tsfm), Apache-2.0, 910
stars).

Stated assumption, because it is load-bearing and untested: the backbone consumes
our **feature frames at one frame per acquisition**, not 25.6 kHz waveforms. No
paper in this window demonstrates a TSFM on raw high-rate vibration, and claiming
otherwise would be inventing evidence.

**Change 2: replace the tail-extrapolation threshold with conformal prediction.**
`AnomalyDetector.threshold_at_fpr` currently extrapolates the healthy upper tail
with a Gumbel fit when the commissioning set is smaller than `1/fpr`. That is
honest but parametric. [2604.20122](https://arxiv.org/abs/2604.20122) applies
**adaptive conformal** prediction to exactly this problem, giving
distribution-free coverage that adapts under drift — which a plant's seasonal
cycle guarantees. This is the highest-value single swap available in the ML
layer.

**Change 3: enforce monotonicity on the health index.**
[2606.12486](https://arxiv.org/abs/2606.12486) builds its Scania fleet method on
wear being monotonically non-decreasing. Our EWMA trend can and does go down when
load drops. A monotone envelope (running maximum of the smoothed index, with an
explicit, logged reset at maintenance events) prevents a favourable load change
from masking a developing fault. Cheap, and it closes a real hole.

**Change 4: prefer source-free adaptation for cross-customer transfer** — see
plan 05.

**Not changed:** the VAE / 1D-CNN / transformer branch structure from the
upstream `impl/ml-*` branches stays. Nothing here invalidates them; the
2025–2026 contribution is about *how they are evaluated and adapted*, not which
one wins.

---

## 05 — Cloud platform & API

**Change: source-free domain adaptation becomes the primary cross-customer
transfer mechanism; federated learning is scoped to within-customer.**

The plan reaches for federated learning (upstream papers 24, 2211.09406) for
multi-site learning. Two 2025–2026 results sharpen this:

- [2503.08749](https://arxiv.org/abs/2503.08749) (SDALR) adapts a bearing-fault
  model to a new domain **with no access to source data**. For onboarding
  customer B from a model trained on customer A, this is strictly better than
  federation: no shared training round, no aggregation server, no data movement,
  and no contractual question about whose data trained what. Only weights move.
- [2609.35792](https://arxiv.org/abs/2609.35792) shows **serverless gossip**
  training is a viable alternative to FedAvg for recurrent failure detectors
  under a matched protocol, removing the central aggregator.

So: **source-free adaptation across tenants, gossip or FedAvg within a tenant.**
This also simplifies the multi-tenant isolation story, which matters for the
FedRAMP ambition in the plan — "customer data never participates in another
customer's training round" is a far easier claim to defend than any form of
federated aggregation across tenants.

[2608.06695](https://arxiv.org/abs/2608.06695) is adjacent and relevant: a
common-to-target probabilistic model for **rare** failures across equipment
families with *heterogeneous sensor configurations*, producing calibrated failure
probabilities. That is the real multi-tenant shape — same family, different
sensor fit-out. Evaluated only on synthetic refrigerator data, so treat as a
design pattern rather than a result.

---

## 06 — CMMS & monitoring dashboard

**Change: abstention and calibrated confidence become first-class UI states.**

The plan lists "real-time spectrum display, mobile-first, offline-capable" — all
about *displaying data*. The uncertainty cluster says the harder UI problem is
displaying *how much to trust* the inference:

- `pdmcore.health.rul` returns `insufficient_evidence` far more often than a
  number, by design. The dashboard needs a real state for "we are watching this
  and cannot yet forecast", not a blank field or an optimistic default.
- RUL must render as a **distribution** — median with a 10–90 band — because
  2607.08273 reports coverage alongside MAE, and a point RUL gets acted on as
  fact.
- Diagnosis confidence should be visibly calibrated (reliability diagram in the
  model card, ECE reported per asset class), per
  `pdmcore.eval.metrics.expected_calibration_error`.
- SCI's escalation ladder (2511.12240) has a UI analogue: show *why* the system
  spent more effort on this asset, because that is the explanation an operator
  actually wants.

Also keep the explainability contract from `AnomalyDetector.explain`: alarms
present as "band 10 RMS and outer-race SNR carry 85 % of the deviation", never as
a bare score. That is the form a millwright can take to the machine.

---

## 07 — Electrical monitoring

**Finding: only one code-backed electrical-signature paper in the window**, and
the MCSA literature is largely absent from arXiv.

The plan's non-invasive multi-phase current + power-quality approach is sound
engineering and is not contradicted — but it is proceeding on thinner published
evidence than the vibration path, with no reusable code and no public MCSA
benchmark comparable to CWRU. The simulator in `core/pdmcore/sim` models the
classic `(1 ± 2ks)f_line` broken-rotor-bar sidebands and 1x load-torque
modulation, so the pipeline can be built and tested, but the physics cannot be
validated against public data the way the vibration path can.

**Change:** keep the plan, lower the stated confidence, and budget for **own
measurement campaign** rather than literature transfer. Treat the electrical
product's model development as requiring a test rig from the start, not as a
dataset-and-transfer exercise.

---

## Three things to not do, based on this window

1. **Do not benchmark on C-MAPSS.** Nine of 45 entries use it. It is a simulated
   turbofan with no vibration channel, and a decade of architecture comparisons
   on it has not transferred to rotating machinery. Using it would make our
   results *comparable to the literature* and *uninformative about our product*.
2. **Do not vendor unlicensed code.** 28 of 45 curated entries have no licence
   file. Read them; do not copy them. See `03_adoption_shortlist.md`.
3. **Do not chase architecture.** The measured gap between a leaked and an honest
   split (99.9 % vs 20–60 %) is larger than the gap between any two architectures
   in this window. Protocol discipline buys more than model selection.
