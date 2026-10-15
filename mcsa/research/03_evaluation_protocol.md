# Evaluation protocol

Written before the experiments were run, and committed before the results JSON,
so that the decision rule cannot have been chosen to suit the numbers. The
commit order is the evidence: `git log --oneline mcsa/` shows this file landing
in a commit that contains no `mcsa/results/*.json`.

Everything here is implemented in `mcsa/prototype/common.py` and
`mcsa/prototype/head_to_head.py`. Where this document and the code disagree,
the code is wrong and must be changed; where the *results* disagree with this
document's expectations, the results win and the expectation is reported as
having been wrong.

---

## 1. What "detection" means

A record is **detected** when its anomaly score exceeds a threshold that was
calibrated, on *held-out healthy data from different physical units*, to a 1 %
false-positive rate. Nothing else counts. In particular:

- not "the sideband is visible in a plot";
- not "the feature is higher than for the healthy record";
- not "AUC is 0.97" — an AUC is not an operating point.

A **fault mode is detected at severity `s`** when at least **80 %** of the test
records at that severity alarm. The hit rate is reported for every severity, not
just the crossing, because a single crossing severity hides a non-monotonic
detector and a detector that fires at 0.2 and goes quiet at 0.55 is worse than
one that fires at 0.55 and stays.

**Detection severity** is the lowest severity meeting that bar, averaged over
the three folds, with the per-fold values reported so that a mean of a `None`
and a `0.2` cannot masquerade as a `0.2`.

---

## 2. Metrics

From `pdmcore.eval.metrics`, unchanged:

| Metric | Why this one |
|---|---|
| `detection_severity` (defined above) | The number a purchase decision turns on |
| `partial_auc(max_fpr=0.1)` | The full AUC averages over operating points no plant accepts; two detectors with equal AUC can differ 10x in nuisance alarms |
| `auc` | Reported only for comparability with the rest of the repository |
| `tpr_at_1pct` | Recall at the *shared* operating point — the only honest way to compare two modalities |
| `fpr_on_test_healthy` | The calibration's own check: it must land near 1 % on *unseen units*, and if it does not, the comparison is invalid regardless of the TPR |
| `detection_delay` (2 consecutive exceedances) | Decides whether a work order can be scheduled or the machine fails first |

No accuracy, no F1 as a headline. At one failure per asset per three years the
prevalence is ~1 in 13 000 hourly acquisitions, where accuracy is meaningless
and F1 recommends the wrong threshold (`cost_fn/cost_fp` is 60x here).

---

## 3. Splits

From `pdmcore.eval.splits`, unchanged:

- **`group_kfold(3)` by physical unit.** Three units of the same machine class
  with 3 % rpm jitter each. No unit appears on both sides of a fold. Audited by
  `leakage_report`, and the audit result is written into the results JSON — a
  fold whose audit fails invalidates that fold.
- **Fit / calibrate are themselves disjoint.** The healthy records of the
  training units are split 65/35 into fit and calibration; the threshold comes
  from the calibration part only. Calibrating on fitted scores is 15.4x too low
  for a detector with capacity and is one of the six "checks that cannot fail"
  in `docs/08_branch_catalogue.md`.
- **For detection delay**, a chronological split on a single unit's own
  history: fit on the first 20 % of the series, calibrate on the next 10 %,
  score the whole series. That is what a commissioning window is. A grouped
  split is wrong here because the question is about *this* machine's trend.

---

## 4. The speed input: what each device actually gets

This is the decision most capable of rigging the comparison, so it is fixed
here.

| Pipeline | Speed input | Why |
|---|---|---|
| Vibration | nameplate shaft speed with a 0.5 % relative error, drawn per record | Most installed vibration nodes have no tachometer. `order_bands(rel_width=0.06)` is already sized for this. Bearing frequencies scale *linearly* with the error |
| Current | **its own estimate** from the rotation sidebands `f1 ± fr`, via `estimate_slip_from_sidebands` | A panel device has no speed input at all. Estimating slip from the current is a real capability of the modality, and the rotation family is independent of the rotor-bar family being tested |

Giving both the simulator's true slip would flatter current, because current's
fault frequencies depend on slip and vibration's do not. Giving current the
nameplate slip would cripple it: a 0.5 % speed error is a 45 % slip error at
rated slip. Each gets what it can really obtain, and the realised slip error is
reported in the results.

**The circularity rule.** Slip estimated from family X may not be used to
diagnose family X. `SlipEstimate.independent_of` records which families an
estimate is admissible for, and `mcsa_admissible` enforces it. Consequence,
accepted in advance: **the eccentricity verdict from a self-estimated slip is
inadmissible**, and is reported as such rather than quietly included.

---

## 5. Acquisition length: two variants, both reported

| Variant | Vibration | Current | What it answers |
|---|---|---|---|
| `equal_cost` | 1 s @ 25.6 kHz | 10 s @ 10 kHz | The deployment question. Both devices cost about the same; the battery node cannot afford 10 s because the acquisition window is 63–85 % of its energy budget, and the mains-powered panel device has no such limit |
| `equal_time` | 1 s @ 25.6 kHz | 1 s @ 10 kHz | The modality question, with the acquisition held constant |

Reporting only `equal_cost` would credit current with an advantage that is
really about mains power; reporting only `equal_time` would deny it an
advantage it genuinely has in deployment. The difference between the two is
itself a result.

---

## 6. The decision rule for "current is sufficient for this fault mode"

Fixed in advance, implemented in `head_to_head.py::_verdict`:

```
current_sufficient    detection_severity(cur) <= 1.5 * detection_severity(vib)
                      AND pauc_10(cur) >= 0.90
current_complementary detects at some severity, but later than that
current_blind         never reaches an 80 % hit rate, even at severity 1.0
current_only          current detects and vibration does not
```

The `1.5x` is a judgement and is stated as one: a fault caught 50 % later in
severity is still caught weeks before failure on a bearing degradation curve,
and the saving on a mounted, battery-powered, radio-equipped sensor is worth
that. A factor of 3 would not be — by then the warning window is shorter than a
maintenance planning cycle.

`pauc_10 >= 0.90` is the nuisance-alarm gate. Without it a mode could qualify
on detection severity while generating an alarm rate no plant would tolerate.

---

## 7. What would invalidate the result

Stated in advance so that it cannot be rationalised afterwards:

1. **A realised false-positive rate on unseen units far from 1 %.** If the
   calibration does not transfer between units, the shared operating point is
   fictional and every TPR comparison with it is meaningless.
2. **A healthy class with no variance.** The pre-existing current model had a
   healthy current-RMS coefficient of variation of 0.004 % against
   acceleration's 10.4 %, which makes any detector look perfect. The audit in
   `mcsa/prototype/audit_sim_current.py` runs first and its numbers are
   published alongside; if the fixed generator still shows an order-of-magnitude
   variance asymmetry between the channels, the comparison is void.
3. **Sensitivity to the air-gap coupling coefficient.** `k_winding` is a single
   uncertain constant that scales every air-gap-path signature.
   `coupling_sweep.py` sweeps it over two decades; any verdict that flips inside
   that range is reported as *undetermined by this work*, not as a finding.
4. **Simulator-only evidence for a positive claim.** A "current is sufficient"
   verdict that cannot be corroborated on measured data stays provisional. The
   repository's standing gate (`docs/07_production_readiness.md`) is that no
   model has been validated on measured data; this branch does not get an
   exemption.

---

## 8. Confounder tests (run separately, same protocol)

Each injected into otherwise healthy records, with the detector and threshold
from the healthy baseline, reporting **the apparent severity and the apparent
fault mode** the pipeline reports:

| Confounder | Level |
|---|---|
| Supply voltage unbalance | 0, 0.5, 1, 2, 3 % |
| Load torque oscillation at `2 s f1` | 0.25, 0.5, 1, 2 % of rated torque |
| Load torque oscillation at `fr` | same levels |
| Supply frequency wander | 0, 20, 50, 100 mHz p-p |
| VFD supply | carrier 2, 4, 8 kHz |
| Slip error | 0, 10, 25, 50, 100 % relative |
| CT channel gain mismatch | 0, 0.5, 1, 2 % |

A confounder **defeats** an indicator when it produces, on a healthy machine,
an indicator value exceeding the value that indicator takes at the detection
severity found in §6. That is the comparison that matters: not "the confounder
is visible", but "the confounder is as large as the fault".
