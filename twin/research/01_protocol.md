# Evaluation protocol — written before the results

Committed before any detection-delay number was computed. The git history is the
evidence: this file lands in the same commit as the design note and before
`twin/prototype/results/*.json` exists. Deviations from it are recorded at the
bottom, with reasons, rather than quietly applied.

The author had seen, at the time of writing: the physics identification residual
(3.5 % RMS in log space), the filter consistency diagnostics, and the
observation-channel gate output. The author had **not** seen any detection delay,
any pAUC, or any arm-vs-arm comparison, except on a 4-unit smoke dataset whose
thresholds were not matched and whose output is not used anywhere.

## 1. Data

All from `pdmcore.sim`, `nuisance=True` throughout — without it the healthy
cloud is unrealistically tight, every threshold becomes hypersensitive, and the
comparison is run on a signal no plant produces.

| dataset | content |
|---|---|
| healthy pool | `motor_pump_37kw`, 10 units (3 % speed jitter), 1 000 frames each at an 8 h cadence, 0.5 s records = **10 000 frames** |
| degradation | 3 modes × 6 seeds × 120 frames at 8 h = **2 160 frames**, two-stage severity profile with the knee at 55 % |
| autocorrelated nuisance | the same two, with an AR(1) load path (ρ = 0.9, σ = 0.12 in log) modulating `base_1x_g`/`base_2x_g` |

Modes: `bearing_outer` (high-frequency energy the twin models only as a lumped
floor), `bearing_inner` (adds 1x modulation the twin does not model at all),
`unbalance` (moves a line the twin models explicitly). Chosen to span what a
physics core can and cannot see, not to flatter it.

## 2. Windows — three, in time order, per unit

| window | share | used for |
|---|---|---|
| fit | 45 % (4 500) | physics coefficients, process model, residual model, baseline detector fit |
| calibrate | 25 % (2 500) | innovation covariance R, deviation standardisation, threshold |
| verify | 30 % (3 000) | realised false-alarm count, **and nothing else** |

Split within each unit so drift between windows shows up rather than being
confounded with unit-to-unit variation.

`docs/08_branch_catalogue.md` records two defects of the form *calibrated and
verified on the same window, so the check could never fail*. The verification
window is therefore touched exactly once per arm, by `verify()`, and its result
is a pass/fail against a **Poisson band on the alarm count**,
`allowed = λ + 3√λ + 3` — not a ratio of rates, for the reason that branch
recorded: at small counts the ratio is the wrong parameterisation.

## 3. Arms

Every arm consumes the same frames, produces a standardised per-frame deviation
`q`, and is driven through the identical confirmation logic (2 consecutive
exceedances). The only differences are the ones named.

| # | arm | information | test | isolates |
|---|---|---|---|---|
| A | `baseline63` | 63 features | per-frame threshold | what ships today |
| B | `baseline63_cusum` | 63 features | CUSUM | the value of sequential testing alone |
| C | `baseline9_cusum` | the 9 gated channels | CUSUM | the twin's information handicap |
| D | `twin_physics` | 9 channels + physics | CUSUM on innovation | the physics core alone |
| E | `twin_hybrid` | + kernel-ridge residual | CUSUM | what the learned residual adds |
| F | `twin_point` | as E, no filter | CUSUM | what the UKF adds over per-frame estimation |
| G | `twin_pf` | as E, particle filter | CUSUM | whether the UKF's Gaussian posterior is the problem |

Arm C is the one that makes the headline honest. Without it, "the twin lost" and
"the twin was shown 9 numbers instead of 63" are the same measurement.

## 4. Matching the false-alarm rate

The only fair comparison. A twin that alarms earlier because it alarms more
often has achieved nothing.

1. For each arm, bisect its threshold on the **calibration** window until the
   measured average run length to false alarm (ARL0) hits the target. ARL0 is
   measured by running the test over healthy trajectories with reset-on-alarm
   and **counting alarms** — not by evaluating a rate formula.
2. Verify on the **verification** window: realised count vs the Poisson band.
3. Report detection delay at three targets: ARL0 = 125, 250 and 500 frames
   (42, 83 and 167 days at an 8 h cadence). 500 frames — roughly one nuisance
   alarm per asset per five months — is the headline.

**What this protocol cannot do, stated now:** a plant-grade budget of one
nuisance alarm per asset per three years is 3 285 frames of ARL0, and verifying
it needs ~15 000 healthy frames per asset. That is a 10-asset, 3-year pilot, or
tail extrapolation of the kind `AnomalyDetector._extrapolate_tail` does. Neither
is in scope here. The reported operating points are the ones the data supports,
and the gap is stated rather than extrapolated past.

A 120-frame trajectory at ARL0 = 500 carries a 21 % chance of a false alarm
before any fault is detectable. That risk is **identical for every arm by
construction of the matching**, and the measured per-trajectory chance-alarm
probability is reported next to the delays so the reader can discount them.

## 5. Metrics

- **Detection delay**: frames and hours from trajectory start to the first
  confirmed alarm, via `pdmcore.eval.metrics.detection_delay` semantics
  (onset index 0 — the defect exists from the first frame by construction of
  `degradation_series`).
- **Severity at detection**: the ground-truth severity when the alarm fires.
  Reported because it is invariant to the series' time parameterisation and is
  the physically meaningful quantity: "the twin alarms at severity 0.07, the
  detector at 0.05".
- **Partial AUC at FPR ≤ 0.1** (`pdmcore.eval.metrics.partial_auc`,
  McClish-standardised) on pooled healthy-verification vs faulty frames,
  threshold-free.
- **Alarms/day** measured, with the `p**N` estimate reported beside it to
  quantify how wrong the independence assumption is on a cumulative statistic.
- **Cost**: µs/frame per arm on this host, named; plus the twin's per-asset
  identification cost, which is the one a fleet feels.

## 6. The capacity sweep

The claim under test: *a residual model with enough capacity absorbs the fault
signature itself.* Three mechanisms, swept on one axis — effective degrees of
freedom, `edf = tr(K(K+αI)⁻¹)`, measured, not nominal.

| mechanism | how it is produced |
|---|---|
| contaminated training | `online_refit > 0`: the residual is refitted on a sliding window of recent frames, including faulty ones |
| endogenous inputs | `residual_inputs ∈ {EXOGENOUS, STATE, MEASURED}` |
| capacity vs a finite healthy window | α swept over 6 decades at fixed inputs |

Reported per configuration: edf, edf per training frame, healthy residual
spread before/after, detection delay and severity at detection at matched ARL0,
and — separately — what happens when R is estimated on the fit window instead
of the calibration window (the optimistic variant a careless implementation
produces).

Also swept: **state dimension** (1, 2, 3 states). The damage state is the
state-space form of the same trap, and it belongs on the same axis.

"How much capacity is too much" will be answered as the edf-per-training-frame
at which the matched-ARL0 detection delay becomes worse than the `NullResidual`
(physics-only) control.

## 7. Virtual sensing

Estimate **load torque** — a quantity no node transmits — from the features one
does. Truth: `truth["load_factor"] × rated torque`, never shown to any
estimator. Scored on held-out frames against three references:

1. predict the mean (the trivial control);
2. ridge regression on the twin's 9 channels (the "is the physics earning
   anything" control);
3. ridge regression on all 63 features (the strongest cheap alternative).

And against an **information ceiling computed from the generating process**:
since only `load_factor × resonance_gain` is identifiable, and both are
log-normal with σ = 0.12 and 0.10, no estimator can exceed
`R² = 1 − (0.10/0.12)² = 0.306`. If the twin gets less than a ridge regression,
that is the finding.

## 8. Preregistered expectations

Recorded so that agreement counts for something and disagreement cannot be
reinterpreted afterwards.

1. **The twin will not beat the baseline on detection delay at matched ARL0 for
   the bearing modes.** Confidence: high. The 63-feature detector sees envelope
   kurtosis, bearing-frequency SNR and matched-harmonic counts — the features
   built specifically for incipient bearing faults — and the twin's physics
   cannot predict any of them, so they are not in its observation vector.
   Predicted: arm A or B detects at a lower severity than arm E on
   `bearing_outer` and `bearing_inner`, by at least a factor of 1.5 in severity.
2. **The twin may win on `unbalance`**, where the physics predicts the exact
   channel the fault moves and the nuisance is the dominant confound.
   Confidence: low-moderate.
3. **Arm C (baseline on the same 9 channels) will be close to arm E.** This is
   the prediction that matters, and the reason is structural: a whitened
   one-class model already discounts the nuisance subspace for free, because the
   nuisance is the leading principal direction of the healthy covariance. The
   twin re-derives that discount by hand from physics. Predicted: the two differ
   by less than 2 frames of delay.
4. **The UKF will add nothing over the per-frame point estimate (arm F ≈ arm E)**
   on the as-shipped datasets, because `pdmcore.sim` redraws the nuisance
   independently every frame, so the state has no temporal correlation to
   exploit. Predicted lag-1 autocorrelation of the estimated state: |ρ| < 0.1.
   On the AR(1) datasets (ρ = 0.9) the filter should pull ahead.
5. **The capacity trap will be real and sharp.** Predicted: detection degrades
   monotonically with edf once edf per training frame exceeds ~0.05, and
   collapses entirely with `MEASURED` inputs at any capacity. Online refitting
   with a window shorter than the degradation timescale will hide the fault
   completely.
6. **Cost: the twin will be 30–100× the baseline per frame** and still
   irrelevant to gateway sizing, because the feature extractor (2.8–6.9 ms) is
   three orders of magnitude above both.
7. **Virtual sensing will land near the ceiling and not beat ridge regression**
   on the same inputs, because the mapping from line levels to load is almost
   linear in log space, which is exactly what ridge fits.

If (1) and (3) both hold, the branch's headline is that the twin does not pay
for itself on this platform, and the title says so.

## 9. Stopping rules

- No arm's hyperparameters are tuned on the degradation data. The CUSUM drift is
  fixed at 0.5 σ *a priori*; the residual α is swept and reported as a curve
  rather than picked.
- The number of trajectories is fixed at 6 per mode in advance. No extension
  after seeing the result.
- If an arm fails its Poisson verification band, its delay numbers are reported
  as invalid rather than as a win.

## 10. Deviations from this protocol

*(Filled in after the runs. Empty at preregistration.)*

- **Added after preregistration:** a run on **measured CWRU bearing data**
  (`run_cwru.py`). Not in the original protocol because the data was not
  available when it was written; added because a twin tested only against the
  simulator that trained it measures mismatch between two models, and the
  repository's own production-readiness document names measured-data validation
  as the gate nothing here has passed. It is reported as a separate experiment
  with its own, much weaker, claims — CWRU has no run-to-failure trajectory, so
  it can answer "does the residual separate healthy from faulty" but **not**
  "does the twin detect earlier".
