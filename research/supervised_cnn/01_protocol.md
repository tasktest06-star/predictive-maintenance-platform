# Evaluation protocol, and a preregistration

Written before the experiment matrix was run. The code that implements it is
`prototype/supervised_cnn/protocol.py`, committed in
`ca9933b` — before any result in `results/matrix_flaw*.json` existed. The
numerical pass marks live in one place, `protocol.GENERALISATION_GATE`, and are
read by the matrix, by the docs and by
`production/supervised_cnn/gates.py`, so those three cannot quote different
thresholds.

## What is held constant

Every model in every cell sees **identical windows**: 2048 raw samples at
12 kHz, 50 % overlap, taken from the same `WindowSet`. The only thing that
differs between a CNN cell and a feature cell is what happens to the window
after it is cut. A comparison in which one model sees more signal than the other
is not a comparison.

- **Window length 2048** because that is the published WDCNN/TICNN input.
- **50 % overlap** because that is the literature's default, and because it is
  the *mechanism* of the leak being measured. Measured: a random partition of
  these windows puts 97.1 % of test windows in sample-level contact with the
  training set.
- **12 kHz for everything.** MFPT is resampled onto this axis; the achieved
  rational ratios (1000/4069 and 500/4069) are recorded per record in the result
  JSON, so a silent frequency-axis error cannot hide inside a transfer number.
- **No early stopping, no checkpoint selection.** A fixed epoch budget. There is
  no fold available that is both disjoint from the test fold and not a sibling
  recording of it, so any validation-based selection reads the test distribution.
  The upstream branch selects its checkpoint on the test loader; this does not.

## The split ladder

| name | group key | n groups (CWRU) | isolates |
|---|---|---|---|
| `random_window` | window | 5 887 | — the literature's default |
| `segment_grouped` | contiguous time block | 240 | overlap leakage |
| `recording_grouped` | (bearing, load) | 40 | operating point |
| `cross_load:<hp>` | load, held out whole | 4 | operating point, adversarially |
| `unit_grouped` | physical bearing | 10 | **the deployment question** |

Grouped splits are **purged**: test windows sharing samples with any training
window are dropped. Measured: a contiguous-segment split leaves 8.4 % of test
windows in sample-level contact across the segment boundaries, and purging takes
that to 0. `random_window` is deliberately *not* purged — its leak is the thing
being measured.

Every split is audited with `pdmcore.eval.splits.leakage_report` **against the
recording key, not against the key the split was built on**. Auditing a grouped
split against its own grouping cannot fail; that is the defect class
`docs/08_branch_catalogue.md` tabulates six instances of, and the upstream
branch's audit is a seventh.

## Metrics

From `pdmcore.eval.metrics`, with no reimplementation:

- **Macro F1** as the headline, not accuracy. CWRU's healthy class carries 28 %
  of the windows (1 653 of 5 887) because the normal records are 2–4x longer, and
  accuracy rewards predicting it.
- **TPR at 1 % FPR** and **pAUC at max_fpr = 0.1** on the binary
  healthy-vs-faulty view, because those are the operating points
  `docs/03_model_cards.md` already reports every detector at.
- **Alarms per day per 100 assets** from `alarms_per_day`, to put the operating
  point in the unit a customer uses.
- Per-class F1 and support, always, because a macro average over a class with no
  test support is not a number.

## What "generalises" means, numerically

Preregistered. A model generalises if, **under the strictest split the dataset
admits**, it reaches all three of:

```
macro_f1        >= 0.70
tpr_at_fpr1pct  >= 0.90
pauc_10         >= 0.90
```

0.70 macro F1 is the bottom of the range `pdmcore/eval/splits.py`'s own docstring
predicts for a grouped split ("70–85 %"), so the gate is set at the pessimistic
end of this repository's own prior. The two detector criteria are the
`docs/03_model_cards.md` operating points.

## Preregistered expectations

Recorded so that surprise is on the record. What was known at the time of
writing, and therefore is *not* a prediction:

- The data verification had run: all 40 CWRU records measure as 12 kHz, the
  record table passes the kinematic permutation test, and CWRU's
  rolling-element records agree with their declared location only 25 % of the
  time under envelope-spectrum evidence.
- The split audits had run: 97.1 % overlap on the random split, 0 after purging.
- The unit-degeneracy analysis had run: `unit_grouped` is impossible on both
  datasets in every label space.
- One throw-away 2-epoch WDCNN fit on `segment_grouped` had been run to time the
  implementation, and scored 0.974 accuracy. That is the single piece of
  outcome information available before preregistration, and it is why
  prediction 1 below is hedged the way it is.

The predictions:

1. **Flaw 1 (random → grouped) will be the smallest of the three.** Expected
   10-class macro F1: random ~0.99, segment-grouped ~0.96, recording-grouped
   ~0.93. So a drop of **3–6 points**, not the 20–30 the leakage literature
   reports. Reasoning: CWRU's seeded defects are large (0.007–0.021" machined
   holes) and the resulting impulse trains are strong and stationary within a
   recording, so a model does not *need* to memorise the noise realisation to
   score well — it can learn the signature. The leakage literature's large effect
   sizes come mostly from run-to-failure datasets where the within-recording
   variation is the signal.
2. **Flaw 2 (cross-load) will cost 5–15 points of macro F1**, with the held-out
   0 hp fold worst, because 0 hp is the extreme of the load range and the
   load-zone geometry there is least like the others.
3. **Flaw 3 (cross-dataset) will be catastrophic: macro F1 at or barely above
   chance (0.33 on three classes).** Both directions. Expected 0.30–0.45.
4. **Engineered features will beat the CNN cross-dataset and lose to it
   within-dataset.** Specifically: WDCNN ahead by 2–5 points on
   `recording_grouped`, features ahead by 15+ points on cross-dataset, because
   the feature path is given the bearing geometry and the shaft rate and
   therefore does its kinematic normalisation analytically instead of learning
   one rig's resonance map.
5. **Order alignment will help cross-dataset transfer but not rescue it** —
   maybe 5–10 points over sample-rate alignment, still well under the gate.
6. **Nothing will pass the gate except within-dataset arms.** Expected
   conclusion: the supervised result is real within one rig and does not survive
   leaving it.
7. **TICNN's regularisers will help on cross-load and not elsewhere**, because
   that is what they were designed for, by 2–5 points.
8. The **synthetic sample-rate confound** will push binary healthy-vs-faulty
   accuracy to ~1.00 under a grouped split, demonstrating that a class-correlated
   acquisition difference is worth more than any amount of bearing physics.

## What this protocol cannot do

Stated here rather than in a limitations section at the end, because it bounds
every number the branch reports.

- **It cannot measure generalisation to an unseen bearing.** No split of CWRU or
  MFPT can. The cross-dataset arm is the closest substitute and it conflates
  bearing, rig, mount, sensor, speed and load into one step. So the three drops
  reported are **lower bounds** on the drop a deployment would see.
- **It cannot distinguish "the CNN is a bad model" from "the CNN is
  under-trained"** at the epoch budget this host affords. The budget is reported
  per cell. The direction of the bias is stated: under-training suppresses the
  leaky arm more than the clean one, so it understates flaw 1.
- **It says nothing about a real plant.** Two seeded-fault test rigs are not a
  plant, the defects are machined rather than grown, and
  `docs/07_production_readiness.md`'s pilot stage remains at ❌.
