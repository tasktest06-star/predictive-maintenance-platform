# Programme: applied models, and digital twins

Two programmes of six branches, built in parallel, each asking one question that
the surrounding literature usually leaves unasked.

**Status: in flight.** The six branches were commissioned together and are being
built concurrently. This document records *why each exists and what it was
briefed to answer* — written before the results, deliberately, so the questions
cannot be retrofitted to the findings. The findings table at the end is filled
in as branches land, and a branch that answers its question in the negative is
reported in exactly the same place and tone as one that does not.

---

## Why these two programmes

The eight `hw/*` branches answer "can this silicon measure a machine?" and
`platform/core` answers "what does the measurement become?" Neither touches the
two questions a buyer actually asks:

1. **Are the published models worth adopting?** The predictive-maintenance
   literature reports extraordinary accuracy — 99 %+ on bearing fault
   classification is routine — and almost none of it survives contact with a
   plant. Which parts of that literature transfer, and which are artefacts of
   how they were evaluated?
2. **Is a digital twin worth building?** "Digital twin" is the most oversold
   term in industrial IoT. A twin that is never calibrated against the specific
   machine it claims to twin, or whose output nobody acts on, is an expense.
   Under what conditions does one earn its cost?

Both programmes are structured the same way, and each branch carries all three
maturity levels rather than splitting them across branches:

| Directory | What it holds |
|---|---|
| `research/` | The method, the literature position, and **the evaluation protocol written before any result**, including a preregistered expected outcome |
| `prototype/` | The experiment, actually run, with results committed as JSON |
| `production/` | The variant that would be deployed, with its gates implemented in code rather than described in prose |

---

## The shared evaluation standard

Every branch in both programmes was briefed against the same standard. It exists
because this repository has been bitten by each of these failures at least once.

**Splits are grouped, never random.** Consecutive windows from one continuous
recording share a bearing, a load, a mounting and a noise realisation. A random
split puts near-duplicate windows in train and test and inflates every metric.
`pdmcore.eval.splits` provides unit-grouped, stratified-group and time splits
**with a leakage audit**, and the audit is not optional.

**Thresholds are calibrated on data the fit has not seen, and verified on data
the calibration has not seen.** Two windows are not enough: calibrating a
threshold to a false-alarm rate and then measuring the false-alarm rate on the
same window returns the target *by construction*, so the check cannot fail. This
was a live defect in this repository; see the table in
[`08_branch_catalogue.md`](08_branch_catalogue.md).

**Gates are expressed on counts, not on ratios of rates.** At a 1 % target on a
175-frame window the expected count is 1.8, and Poisson noise alone spans 0–6 at
3 σ — so a "4× the target rate" gate refuses stationary data a third of the
time. Maintenance decisions are full of small counts; so are false-alarm
budgets.

**Comparisons are at matched false-alarm rate.** A detector that alarms earlier
because it alarms more often has achieved nothing. Detection delay is only
meaningful with the false-alarm rate pinned.

**The baseline must be strong.** Every branch's comparator is
`pdmcore.models.anomaly` — a properly calibrated one-class Mahalanobis or PCA
detector over 63 calibrated features. It is cheap, it is measured at ~17 µs per
frame on this host, and beating a strawman version of it proves nothing.

**Nothing is stubbed.** If a model cannot be run here, the code raises with a
message saying precisely what is missing, and no number is produced. The
precedent is `hardware/rpi5_hailo/gateway/hailo_backend.py`, which refuses to
load a model that does not exist rather than returning a plausible probability.

**Every number in prose comes from committed code via a committed results
JSON**, and figures are generated from that JSON. Prose that silently disagrees
with its own measurement is the defect class this repository tracks most
closely.

---

## Programme A — applied models

| Branch | The question it must answer |
|---|---|
| `models/foundation-ts-zeroshot` | Do time-series foundation models (Chronos, MOMENT, TimesFM, Lag-Llama) earn their place against a calibrated 63-feature one-class detector? Vibration is unlike their pretraining corpora — the diagnostic content is in the spectrum and in modulation, not in the level — and that predicts something about zero-shot transfer |
| `models/supervised-cnn-crossdataset` | Do the published supervised bearing CNNs (WDCNN, TICNN, MA1DCNN) generalise, or is their accuracy an artefact of evaluation? Three specific flaws to price: random splits that leak, cross-load transfer, cross-dataset transfer |
| `models/mcsa-current-signature` | Motor current needs only a clamp in the switchgear panel — no sensor on the machine, no battery, no radio. Which fault modes can it see, and which is it oversold for? This is the most commercially consequential question in the repository, because it changes the economics of a programme rather than its accuracy |

### Relationship to the upstream `impl/*` branches

The upstream repository already carries substantial model implementations, and
they are not ignored:

| Upstream branch | Size | Relationship |
|---|---:|---|
| `impl/ml-supervised-1dcnn` | ~2 000 lines, torch/TF | Directly overlaps `models/supervised-cnn-crossdataset` |
| `impl/ml-transformer-faultformer` | ~3 200 lines, torch, incl. `pretrain.py` and `cross_domain.py` | Directly overlaps `models/foundation-ts-zeroshot` |
| `impl/ml-unsupervised-vae` | ~1 050 lines, torch | Same job as `pdmcore.models.anomaly` |

None has been merged into `platform/core`, and none carries evidence of
validation on measured data or of evaluation under a leakage-audited grouped
split. So the two affected branches were re-briefed mid-flight: **writing a
fourth from-scratch implementation is the least valuable thing available.**
The valuable thing is to evaluate what already exists under the standard above —
and where that is not possible here (no torch in this environment), to say so
explicitly rather than let a reader infer that someone else's code was measured.

The transformer case gained a useful three-way framing from this: engineered
features, versus in-domain pretraining (which `impl/ml-transformer-faultformer`
already is), versus external-corpus zero-shot (which is what "foundation model"
actually means). The field routinely muddles the second and the third.

---

## Programme B — digital twins

The three branches split along the only boundary that matters — can the twin be
*calibrated*, does it *detect earlier*, and does it *change the decision*:

| Branch | The question it must answer |
|---|---|
| `twin/motor-physics-core` | A twin is only as good as its calibration. How much of an induction-motor twin can be identified from what a plant actually has — nameplate alone, nameplate plus one commissioning record, nameplate plus a coast-down — and which predictions are identifiable at each tier? A twin whose structural transfer path is unidentifiable cannot predict absolute amplitude and therefore cannot drive ISO-zone alarming, even if it predicts frequencies perfectly |
| `twin/hybrid-residual-ukf` | Does a twin detect a developing fault **earlier at matched false-alarm rate** than the one-class detector already shipping — or does it only cost more? Includes the trap that makes this hard: a learned residual with enough capacity absorbs the fault signature it exists to expose, and the branch must quantify how much capacity is too much |
| `twin/decisions-rul-scheduling` | Does a twin change the maintenance decision, and what is that worth? Including the value-of-information question that is uncomfortable for the whole category: if the action is "intervene at the next shutdown" either way, improving RUL from ±4 weeks to ±2 weeks is worth exactly nothing |

The third branch's value-of-information result is what tells the first two
whether their accuracy improvements matter, which is why it was commissioned
alongside them rather than after.

### Boundaries between the three

Each branch owns one layer and consumes the others' interfaces rather than
reimplementing them: physics and identification; hybrid structure and state
estimation; decisions and scheduling. Each was told to state in its own docs
where its boundary lies and what a better upstream twin would and would not
change about its conclusions. None waits on another — they were built
concurrently, which means some duplication of forward modelling is expected and
should be reconciled afterwards rather than designed away in advance.

---

## Findings

Completed as branches land. Each row gets the finding **and its number**, in the
same place whether it supports or undermines the branch's premise.

| Branch | Headline finding | Validated on measured data? |
|---|---|---|
| `models/foundation-ts-zeroshot` | *pending* | *pending* |
| `models/supervised-cnn-crossdataset` | *pending* | *pending* |
| `models/mcsa-current-signature` | *pending* | *pending* |
| `twin/motor-physics-core` | *pending* | *pending* |
| `twin/hybrid-residual-ukf` | *pending* | *pending* |
| `twin/decisions-rul-scheduling` | *pending* | *pending* |

The last column is the one that matters most. This repository's standing
admission is that **no model in it has been validated on measured data** — CWRU,
MFPT and MIMII are the named gate. Network access is available in this
environment and all six branches were pointed at that gate explicitly. Any
branch that closes it, even partially, is worth more than any model result in
either programme.
