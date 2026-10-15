# The decision layer: framing

*Written before the results on this branch were run. The evaluation protocol in
`02_evaluation_protocol.md` was fixed first, and the numbers in
`../prototype/results/*.json` came afterwards.*

## The question

Does a digital twin change the maintenance decision, and what is that worth in
money?

Every other branch in this repository answers a measurement question: can this
sensor see a bearing defect, does this accelerator help, does the C twin agree
with the Python reference. Those questions have the pleasant property that the
answer is checkable. This branch answers the question that decides whether any
of them mattered, and it has the unpleasant property that the answer depends on
numbers nobody in this repository has measured — a plant's downtime cost, its
shutdown calendar, its crew.

So the deliverable is not a number. It is a **boundary**: the conditions under
which the programme pays for itself, stated so that a plant can check its own
side of the arithmetic.

## Why "RUL accuracy" is the wrong objective

A prognostics programme is sold on accuracy. The decision-theoretic position is
that accuracy only has value through the actions it changes:

> An estimate that does not change the action is worth nothing, however good it
> is.

Three structural features of maintenance make that bite harder than it does in
most decision problems.

**The action set is discrete and sparse.** Interventions happen in shutdown
windows. A plant with six-weekly shutdowns has an action every 42 days. An RUL
that moves from 8 weeks to 6 weeks changes nothing if both land before the same
window. The decision-relevant resolution is therefore set by the *maintenance
calendar*, not by the sensor — and no amount of DSP changes the calendar.

**The cost function is steeply asymmetric.** Being late costs the unplanned
premium; being early costs a fraction of a bearing. At the cost model in
`pdmcore.decide.costs` those differ by more than an order of magnitude, so the
decision is driven by the **left tail** of the RUL posterior rather than by its
centre. This is why the decision layer consumes weighted posterior *samples*
from `ScenarioRUL.failure_day_samples` and summarises afterwards, rather than
consuming a median and an interval.

**Most of the value is in detection, not prognosis.** Converting an unplanned
failure into a planned job captures the downtime difference, the expedite
premium and the secondary damage. Knowing *exactly when* it would have failed
captures only the residual life, which is bounded by the price of the part.
These two quantities differ by a large factor, and they are routinely conflated
in the same sentence.

## The decision model

Four things can happen to a monitored asset, and a cost model that omits any one
of them gives the wrong answer:

| | |
|---|---|
| It fails in service | downtime at the unplanned rate, expedited part, longer MTTR, secondary damage with probability *p* |
| It is repaired in a shutdown | marginal shutdown hours, part, labour, **plus the residual life discarded** |
| It is stopped outside a shutdown | downtime at the *unplanned* rate for a *planned* duration — the action a twin makes available and a calendar does not |
| It was healthy and we looked | an inspection, **plus** the machine actually being opened with probability *q* |

The fourth is the one usually left out, and it is the one `hw/cn0549-piezo-ae`
measured the hard way: a cut IEPE cable was diagnosed as mechanical looseness at
**0.97 confidence**. At that confidence the machine does not get a glance, it
gets opened. So the false-alarm cost is not one inspection; it is an inspection
plus `q` × (a planned job and a thrown-away good part). The parameter `q` is how
"a confident wrong answer costs more than an unconfident one" enters the money.

The third is the one usually left in but mispriced. Stopping an asset on a
Tuesday does not cost planned-downtime rates — the line was not stopping. It
costs production, for a shorter and calmer repair.

## What a twin adds, precisely

Three things, in decreasing order of what we expect them to be worth:

1. **Detection.** The asset is degrading. This needs a detector, a baseline and
   an alarm policy — all of which `platform/core` already has.
2. **A date.** Roughly when, with an interval. This needs a trend and a
   prognostic model, i.e. `pdmcore.health.rul`.
3. **Conditional dates.** How long at 100 % load, at 70 %, if we defer to the
   shutdown in six weeks. This is the capability that distinguishes a twin from
   a trend, and it is implemented here as a **damage clock**: a time change
   `tau(t) = integral lambda(u) dt` with `lambda(u) = (u/u_ref)^p` and `p` the
   ISO 281 load-life exponent, sampled rather than fixed.

(3) is the branch's distinctive contribution and also its weakest link, because
nothing in this repository measures `p`. The honest framing is that the
conditional machinery is *arithmetic plus an assumption*, and the assumption is
labelled in every output.

## Where this branch's boundary is

The two sibling twin branches — `twin/motor-physics-core` (physics and
identification) and `twin/hybrid-residual-ukf` (hybrid model and state
estimation) — improve the **upstream** estimate. This branch consumes
`pdmcore.health.rul` as given and owns everything downstream.

What a better upstream twin would change:

- **A narrower RUL interval.** Quantified here as a value, and the value is
  bounded by EVPI. If EVPI is small, this is a small prize however much the
  interval narrows.
- **A better-calibrated interval.** This is *not* the same thing and may be
  worth more: the value-of-information analysis assumes a calibrated posterior,
  and an over-confident one makes the decision layer act on a certainty it does
  not have. `../prototype/results/pipeline_measurements.json` reports the
  shipped estimator's measured p10–p90 coverage against ground truth.
- **A physically derived `lambda(u)`.** This would turn the conditional answers
  from arithmetic-plus-assumption into a prediction. It is the single upstream
  improvement that would change a *conclusion* on this branch rather than a
  number.
- **Earlier detection.** Lead time bounds everything: an asset detected with
  less remaining life than the time to the next shutdown cannot have its failure
  converted into a planned job at all, whatever the RUL says.

What a better upstream twin would **not** change: the shutdown calendar, the
crew, the budget, the false-alarm cost, or the fact that the decision-relevant
precision is set by the calendar.

## The trap this branch had to avoid

`docs/08_branch_catalogue.md` tabulates six checks that could not fail, and
names one as a general lesson:

> A gate expressed as a ratio of rates is really a gate on a count, and at small
> counts the ratio is the wrong parameterisation.

Maintenance is made of small counts. A 40-asset fleet at one failure per asset
per five years sees eight failures a year. "We prevented 60 % of failures" is
then a statement about five events, and five events have a 95 % Poisson interval
of 1.6–11.7. The structural defence adopted here is
`pdmcore.decide.counts.CountedRate`: a rate that cannot be constructed without
its count and cannot be serialised without its exact Poisson interval. Every
rate in every results JSON on this branch is one.

The consequence is visible in the headline. The programme balance reports both
`pays_for_itself` (the point estimate) and `pays_for_itself_demonstrated` (the
lower Poisson bound is also positive), and at realistic trial sizes those
disagree. That disagreement is the finding, not a defect in the reporting.
