# Supervised bearing-fault CNNs: the lineage, and what each claim rests on

The question this branch exists to answer: *do the published supervised
bearing-fault CNNs actually generalise, or is their reported accuracy an artefact
of how they are evaluated?*

The literature is large and reports 99 %+ accuracy routinely. That is not, by
itself, evidence of anything wrong — some of these papers are careful, and
several of them say explicitly what their numbers do and do not cover. What
follows is an attempt to be specific about **which evaluation choice each
headline claim depends on**, because that is the thing a reader can check and
the thing this branch measures.

## The architecture lineage

### WDCNN — wide first-layer deep CNN (2017)

Zhang, Peng, Li, Chen, Zhang, *A New Deep Learning Model for Fault Diagnosis with
Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals*, Sensors
17(2):425, 2017.

The founding paper of the family. 2048 raw time samples in; a first convolution
with **64 taps at stride 16** — unusually wide, and that is the whole idea: at
12 kHz a 64-tap kernel spans 5.3 ms and has roughly 190 Hz of frequency
resolution, so the layer behaves as a learned band-pass filter bank rather than
as an edge detector. Five conv stages with batch norm, ReLU and max-pooling, then
FC-100 and softmax. 54 k parameters.

**Claims made:** ~100 % on CWRU under a 10-class protocol; robustness to additive
noise down to low SNR; and "domain adaptation ability", demonstrated as transfer
between CWRU motor loads.

**What the claims depend on:**
- The headline accuracy depends on a **random split of overlapping windows** cut
  from the same recordings. This is the load-bearing choice. There is exactly one
  physical bearing per fault class in CWRU, so the test fold cannot contain a
  bearing the model has not trained on.
- The noise robustness result is a genuine, well-posed experiment and does not
  depend on the split: noise is added to the test windows, and the degradation
  curve is informative regardless.
- The load-transfer result is the strongest part of the paper and is a real
  generalisation test — of the three flaws this branch measures, WDCNN's own
  paper already addresses the second one. It is also where its reported numbers
  are lowest.

### TICNN — training interference CNN (2018)

Zhang, Li, Peng, Chen, Zhang, *Deep Convolutional Neural Networks with Wide
First-layer Kernels for Fault Diagnosis under Varying Working Conditions*,
Mechanical Systems and Signal Processing 100:439–453, 2018.

Same wide-first-kernel skeleton, plus three regularisers aimed squarely at
working-condition shift: **dropout applied to the raw input**, **very small
training batches**, and an **ensemble** of the resulting models.

**Claims made:** better than WDCNN under load shift and under noise; accuracy
maintained across working conditions.

**What the claims depend on:** this paper's central claim is *about* the
cross-condition split, so it does not depend on a random split — it is one of
the careful ones. What it does depend on is the ensemble and the small-batch
schedule, neither of which is fully specified, and on CWRU's load axis being a
meaningful proxy for deployment variability. A motor load change on a test rig
moves the shaft speed by about 4 % (1797 → 1725 rpm) and the load-zone angle;
it does not change the bearing, the mount, the resonance map or the sensor.

### MA1DCNN — multi-attention 1-D CNN (2020)

Wang, Liu, Zhu, Shen, *Understanding and Learning Discriminant Features based on
Multiattention 1DCNN for Wheelset Bearing Fault Diagnosis*, IEEE Transactions on
Industrial Informatics 16(9):5735, 2020.

Adds channel attention (squeeze-and-excitation-like) and a "efficient channel
attention" block at every stage, with the stated aim of making the learned
features interpretable — the attention maps are presented as showing the network
attending to the impulse train.

**Claims made:** improved accuracy, and interpretability of what the network uses.

**What the claims depend on:** the accuracy claim is on wheelset bearing data and
CWRU under conventional protocols. The interpretability claim depends on the
attention map being a faithful account of the decision, which is a separate and
contested question — attention weights are not an explanation in the sense
``pdmcore.models.anomaly.AnomalyDetector.explain`` means it, where the
contribution decomposes the score exactly.

**Not reproduced on this branch.** The attention blocks change *which features
are used*; this branch's question is about *how the result is measured*, and the
measurement is invariant to the attention mechanism. Stating this rather than
implementing a half-faithful version: see the "what was not done" section of
`docs/09_supervised_cnn_crossdataset.md`.

### The long tail

Several hundred variants: CNN-LSTM hybrids, 2-D CNNs on time–frequency images
(scalograms, STFT, Markov transition fields, Gramian angular fields),
capsule networks, graph networks over sensor arrays, and since about 2022 a
steady stream of transformer models. The 2-D-image branch deserves one specific
remark because it interacts with flaw 1: converting each window to an image and
*then* splitting randomly adds no leakage of its own, but it makes the leakage
harder to see, because the near-duplicate structure of overlapping windows is no
longer visible in the input representation.

### The counter-literature

The problem is known and has been written about from inside the field. The
honest summary of the position this branch takes is that it is not novel — it is
a *measurement* of something several authors have already argued:

- Leakage from random splits of overlapping windows is documented, with
  quantified effect sizes, in the data-leakage literature that both upstream
  branches on this repository cite (`arXiv 2509.22267`, cited in
  `impl/ml-supervised-1dcnn/dataset.py` and
  `impl/ml-transformer-faultformer/finetune.py`).
- CWRU's saturation as a benchmark — the observation that it is "solved" and that
  the remaining variance is between evaluation protocols rather than between
  models — has been made repeatedly, including by authors proposing replacement
  benchmarks (Paderborn/KAT, XJTU-SY, Ottawa, PU).
- `pdmcore/eval/splits.py` in this repository states the mechanism and the
  expected effect size (99 % → 70–85 %) in its own module docstring, which
  predates this branch.

So the contribution here is not the observation. It is three things:

1. **Numbers from this repository's own harness on measured data**, which closes
   part of the gate in `docs/07_production_readiness.md` that says no model in
   this repository has been validated on anything but a simulator.
2. **A structural result about the benchmarks** rather than about the models:
   neither CWRU nor MFPT contains more than one physical unit per class in any
   label space, so neither can support a bearing-grouped evaluation at all. The
   drop from a random split to a *grouped* split is therefore a lower bound on
   the drop to a genuinely unseen bearing, and no experiment on these two
   datasets can establish the upper bound.
3. **An audit of the two implementations already on this repository's remote**,
   by running them (`prototype/supervised_cnn/audit_upstream.py`).

## What a "bearing-wise split" has to mean

The crux, stated once, because every finding on this branch follows from it.

CWRU's 12 kHz drive-end set, as used by essentially the whole literature, is 40
recordings:

- 4 normal, at motor loads 0/1/2/3 hp
- 3 fault locations (inner race, outer race @ 6 o'clock, ball)
  x 3 seeded defect diameters (0.007", 0.014", 0.021")
  x 4 motor loads

The four recordings in each (location, diameter) cell are **the same physical
bearing, re-run at four loads**. There is one healthy bearing. So:

| grouping key | number of groups | what holding one out tests |
|---|---|---|
| window | ~5 900 | nothing — a random split |
| segment (contiguous time block) | 240 | that the model is not memorising adjacent windows |
| recording = (bearing, load) | 40 | an operating point it has not seen |
| **physical bearing** | **10** | **a bearing it has not seen — the deployment question** |

A split keyed on the *file* is a 40-group split, and because each bearing owns
four files, holding out 20 % of files leaves every held-out bearing's other three
files in the training set. That is what both upstream implementations do, and
their own leakage audit passes because it is run against the same key that
produced the split.

A split keyed on the physical bearing is impossible here: the bearing key and
the 10-class label are the same string, so holding out a bearing holds out a
class. `pdmcore.eval.splits.stratified_group_split` raises, with the message
"collect more units per class rather than falling back to a random split", which
is the right answer and is the answer this branch reports.

MFPT is in the same position: 3 baseline recordings, 3 outer-race at 270 lb, 7
outer-race at varying load, 7 inner-race at varying load — operating conditions,
not independent bearings. One unit per class.

## Bearing geometry, for reference

Derived with `pdmcore.dsp.bearing.fault_frequencies`; the catalogue entries
`SKF6205` and `NICE_MFPT` predate this branch.

| | CWRU drive end (SKF 6205-2RS JEM) | MFPT (NICE) |
|---|---|---|
| rollers | 9 | 8 |
| ball / pitch diameter | 0.3126 / 1.537 in | 0.311 / 1.175 in |
| d/D | 0.2034 | 0.2647 |
| BPFO (orders of shaft) | 3.5848x | 2.9413x |
| BPFI | 5.4152x | 5.0587x |
| BSF | 2.3567x | 1.7567x |
| FTF | 0.3983x | 0.3677x |
| shaft rate in the data | 28.75–29.95 Hz | 25.0 Hz |
| BPFO in Hz at that speed | 103.1–107.4 | 73.5 |

BSF is in `pdmcore`'s per-ball convention; several papers quote 2x these values.

BPFO differs by **18 % in shaft orders** and by **32 % in hertz** between the two
rigs. So a model that has learned "energy at 3.58 shaft orders in the envelope"
has learned something that is wrong on the other dataset, and a model that has
learned "energy at 107 Hz" has learned something that is wrong at any other
speed. Those are two different failures, and separating them is why the
cross-dataset arm is run under two resampling conventions: match the sample rate
(hertz align, orders do not) or match samples per revolution (orders align as far
as the shaft is concerned, bearing orders still differ by 18 %).
