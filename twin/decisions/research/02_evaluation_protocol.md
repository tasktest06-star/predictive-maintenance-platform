# Evaluation protocol

*Fixed before any result on this branch was produced. If a number below turns
out to be inconvenient, the number is reported and this document is not edited
to accommodate it. Where a prediction made here was wrong, it is marked
**WRONG** in `04_findings.md` with what actually happened.*

## What runs the DSP, and what does not

One stage executes the signal chain: `prototype/pipeline.py`. It generates a
pool of run-to-failure lives through
`sim.degradation_series → FeatureExtractor → MahalanobisDetector → HealthIndex →
ScenarioRUL`, and commits the resulting health trajectories to
`results/life_pool.json`. Everything downstream consumes that file, so the
value-of-information study, the scheduler study and the multi-year fleet
economics can be swept over hundreds of parameter combinations without running
25 kHz DSP tens of thousands of times.

Two acknowledged shortcuts, both recorded in the output file:

1. **The healthy stage of a life is a drawn duration, not a simulated signal.**
   `pdmcore.sim`'s healthy output is stationary, so simulating three hundred
   flat days costs three hundred DSP runs and adds no information. Each life is
   a 60-frame commissioning window plus the degradation stage from onset to
   failure, at one acquisition per day.
2. **The false-alarm stream in the fleet simulation is a Poisson process** at
   the rate measured on the commissioning frames, not a detector run over years
   of healthy signal. This is the weakest measurement on the branch and the
   sensitivity analysis treats it as the pivotal assumption rather than as data.

## Detector protocol

Fitted on healthy frames of each machine preset, **calibrated on a held-out
third**. Both the held-out threshold and the in-sample threshold are reported in
`life_pool.json`, because calibrating on the data that was fitted reads 15.4×
too low on this platform (`docs/08_branch_catalogue.md`) and the gap should stay
visible rather than becoming a remembered fact. Operating point: 1 % FPR, which
is the platform's standard.

The health index consumes the detector score with
`external_reference = threshold`, so index 1.0 is the 1 % operating point and
the `AlarmConfig` thresholds (watch 1.0, alarm 2.0, critical 4.0) carry their
usual meaning. The prognostic filter consumes `band.10_rms` normalised to its
healthy median, at `failure_threshold = 20.0` and `min_health = 2.0` — the same
indicator and the same two numbers `cloud/ingest/app.py` already uses, so the
decision layer is scored against the shipped configuration rather than a tuned
one.

## Pre-registered measurements

### Pipeline (`results/pipeline_measurements.json`)

| Measurement | Why | Prediction made in advance |
|---|---|---|
| Detection lead time: true remaining life at the first confirmed ALARM | bounds everything downstream | median 20–60 d |
| Abstention share of observations | `rul.py` abstains by design; this prices it | 40–70 % |
| Lead time lost to abstention | the cost of the abstention policy | 10–30 d |
| **p10–p90 coverage against ground truth** | the VoI analysis assumes calibration | **below 0.80 — over-confident** |
| Interval width at the first answer | where the shipped estimator sits on the VoI curve | 20–60 d |
| Confirmed nuisance alarms per asset-year | the programme's dominant cost | 0.1–1.0 |

### Value of information (`results/voi.json`)

| Measurement | Prediction |
|---|---|
| Value of detection per event | 10 000 – 20 000 EUR at `mid_size_line` |
| EVPI per event | 500 – 3 000 EUR, i.e. under 20 % of the value of detection |
| Decision-relevant width (80 % capture) | **of order the shutdown interval** |
| Scaling of that width with the shutdown interval | monotone increasing |
| Break-even width (acting becomes worse than the calendar) | exists, and is 1–3× the shutdown interval |
| Value of 4 weeks → 2 weeks | **under 300 EUR per decision** |
| EVPI with no option to stop between shutdowns | *predicted higher*, because the only lever left is which window |

Sweeps: three site archetypes × {14, 28, 42, 84} day shutdown intervals,
plus `mid_size_line` × the same intervals with the unscheduled stop removed.

### Scheduler (`results/scheduler.json`)

Posteriors from the real filter, assets stopped at random points in their
degradation stage. Reported against three baselines:
`run_to_failure`, `blanket_next_window`, `fix_everything_always`. A baseline
that violates crew capacity is costed **and flagged**, because quoting a saving
against an unexecutable plan is flattering.

Also: the measured gap between the greedy+local heuristic and the MILP optimum
at instance sizes 8/16/24/40/60, and sweeps over crew-hours per window and
annual parts budget. Prediction: the heuristic is within 2 % of the optimum and
scipy's `milp` solves a 60-asset instance in under a second.

### Fleet simulation (`results/fleet_sim.json`)

40 assets, 4 lines, 5 years, 42-day shutdowns, 4 fitters, 48 crew-hours per
window, 40 000 EUR annual parts budget. Three policies on **common random
numbers** — asset *a*'s *k*-th life is the same realisation under every policy,
because the cost differences are a few hundred euros per asset-year against
event costs of tens of thousands.

The fixed-interval policy is **swept** over {180, 270, 365, 480, 600} day
intervals and the *best* one is used as the comparator. A calendar policy tuned
on the same life data the twin sees is the honest baseline and a much stronger
one than a strawman.

Three replicate seeds for the reference case. Nine single-parameter sensitivity
sweeps at three levels each, spanning what a plant could plausibly report rather
than bracketing the default symmetrically.

Prediction: condition-based maintenance beats both baselines at
`mid_size_line` and `critical_single_train`, does **not** at
`spared_duty_standby`, and the assumption that flips the conclusion is
**the nuisance-alarm rate**.

## Rules fixed in advance

1. **Every rate carries its count.** `pdmcore.decide.counts.CountedRate` cannot
   be constructed without the numerator count and cannot be serialised without
   its exact (Garwood) Poisson interval. No ratio-of-rates gate anywhere.
2. **Both verdicts are reported.** `pays_for_itself` (point estimate) and
   `pays_for_itself_demonstrated` (lower Poisson bound also positive). Where
   they disagree, the disagreement is the finding.
3. **Abstention propagates and is never filled in.** A RUL that refuses
   produces a decision that refuses; an asset that refuses is excluded from the
   schedule with its reason recorded, not scheduled on an invented posterior.
4. **Figures are generated from the committed JSON**, never from a terminal run
   — `slides/figures/make_decision_figures.py` reads the JSON and nothing else.
   A slide cannot disagree with its measurement.
5. **The sensitivity table is the headline**, the reference case is context.
6. **A result that contradicts the branch's premise is promoted, not buried.**
   The premise being tested is that scenario-conditional RUL changes maintenance
   decisions enough to pay for a twin. If it does not, that is the result.

## What this protocol cannot establish

- **Anything about a real plant.** Every cost is an estimate and every
  trajectory is simulated. The output is a boundary a plant can check, not a
  measurement of one.
- **The load-life exponent.** `pdmcore.sim` has no load-dependent damage model.
  The scenario-conditional arithmetic and all four of its abstention gates are
  exercised; `lambda(u)` is validated by nothing here and by nothing elsewhere
  in this repository.
- **Detector performance in the field.** The detector is fitted and calibrated
  on simulated healthy data from the same generator that produced the faults.
  The platform-wide caveat in `docs/07_production_readiness.md` applies
  unchanged: no model in this repository has been validated on measured data.
- **The false-alarm rate.** Extrapolated from 60 commissioning frames per life
  to multi-year exposure. Treated as an assumption throughout.
