# The cost model, parameter by parameter

**Nothing in this table is measured.** Every figure is an estimate for a
mid-size European manufacturing plant, written down so it can be argued with.
Two are pinned to values already in this repository so that the decision layer
and the detector threshold cannot come to disagree silently; the rest are
guesses with a stated plausible range.

The code is `core/pdmcore/decide/costs.py`. Each field carries a `provenance`
tag (`estimate` / `site_quoted` / `measured` / `repo_default:...`), and
`MaintenanceCostModel.require_configured()` **refuses** to make a recommendation
unless a site has either quoted a figure or explicitly acknowledged the default.
Silence is not acceptance.

## The parameters

| Parameter | Default | Unit | Provenance | Plausible range | What it would take to measure |
|---|---:|---|---|---|---|
| `unplanned_downtime_eur_per_h` | 3 400 | EUR/h | estimate, back-solved | 200 – 30 000 | Contribution margin per production hour × the fraction of output this asset gates. The plant's controller has this. |
| `planned_downtime_eur_per_h` | 900 | EUR/h | estimate | 0 – 5 000 | The *marginal* cost of an extra job inside a shutdown: either window extension or the job it displaces. Usually nobody has it. |
| `mttr_unplanned_h` | 6 | h | estimate | 3 – 48 | CMMS history. The one parameter here a plant usually *can* produce. |
| `mttr_planned_h` | 4 | h | estimate | 2 – 16 | Same. |
| `spare_part_eur` | 800 | EUR | estimate | 100 – 20 000 | Purchasing. Trivial to get, wide across asset classes. |
| `expedite_premium_frac` | 0.60 | — | estimate | 0 – 3 | Compare expedited to stock purchase-order prices. |
| `labour_eur_per_h` | 75 | EUR/h | estimate | 35 – 150 | Fully loaded internal rate, or the contractor rate. |
| `crew_size` | 2 | — | estimate | 1 – 4 | Work instruction. |
| `secondary_damage_eur` | 6 500 | EUR | estimate | 0 – 50 000 | Strip-down reports on assets that ran to failure. Rarely recorded in a usable form. |
| `p_secondary_damage` | 0.35 | — | estimate | 0.05 – 0.9 | Same source. The weakest-evidenced parameter with a large effect. |
| `inspection_eur` | 400 | EUR | **pinned** to `metrics.expected_cost`'s `cost_fp` | 150 – 1 200 | Half a day plus travel. |
| `p_opens_machine_on_false_alarm` | 0.30 | — | estimate | 0.02 – 0.9 | Work-order outcomes: `work_order.outcome = 'not_confirmed'` with `parts_replaced` non-empty. The CMMS schema in `cloud/sql/001_schema.sql` already records exactly this. |
| `design_life_days` | 1 825 | d | estimate | 365 – 7 300 | L10 from the bearing catalogue at the actual load. Computable, not usually computed. |
| `sensor_capex_eur` | 180 | EUR | estimate | 60 – 900 | The eight `hw/*` branches' bills of materials. |
| `install_eur` | 250 | EUR | estimate | 80 – 1 500 | Two hours of an electrician plus commissioning. Dominates capex at scale and is routinely omitted from vendor TCO. |
| `hardware_life_years` | 5 | y | estimate | 2 – 10 | Battery life bounds it. `tools/power_budget.py` says 1.6 y on the shipped LiPo and 15.2 y on a LiSOCl₂ AA — a factor of ten, on this parameter, from a cell choice. |
| `gateway_eur` / `assets_per_gateway` | 450 / 25 | EUR / — | estimate | — | `hw/rpi5-hailo-gateway` measured 781 331 hourly nodes of compute headroom, so this ratio is set by radio and site layout, not by the gateway. |
| `cloud_eur_per_asset_year` | 36 | EUR/y | estimate | 10 – 300 | Vendor list price. The one number a vendor will state. |
| `analyst_h_per_asset_year` | 1.5 | h/y | estimate | 0.2 – 12 | **The largest line item in the programme cost at the default.** Time spent reviewing alarms, closing work orders and re-baselining. Measurable only from a running programme. |

## The two pins, and why

`inspection_eur = 400` is `metrics.expected_cost`'s `cost_fp` default, unchanged.

`unplanned_downtime_eur_per_h` and `mttr_unplanned_h` were chosen so that
`unplanned_event_eur()` lands on `metrics.expected_cost`'s `cost_fn` default of
25 000, which `core/tests/test_decide_costs.py` asserts to within 10 %. Without
that pin the repository would carry two cost models disagreeing by a factor of
two, and the detector threshold in one would not match the maintenance decision
in the other.

## Two corrections the decomposition forced

**`cost_fn` is not the cost of a failure.** `metrics.expected_cost` documents
its `cost_fn` as "an unplanned stop on a mid-size production line". That is the
cost of the *event*. A missed detection does not cost a plant the event — the
repair, the part and the labour are paid whether or not the detector saw it
coming. The decision-relevant figure is `avoidable_eur()`: the difference
between the unplanned event and the planned one. Using the event cost overstates
the false-negative penalty and pushes the cost-optimal threshold to a *higher*
alarm rate than the plant's numbers justify — an error in the flattering
direction for anyone selling detection.

**`cost_fp` is not the cost of a false alarm.** It is the cost of an
*inspection*. The false alarm costs an inspection plus `q` × (a planned job plus
a discarded good part). At the defaults that is 2 140 EUR rather than 400.

Together these take the asymmetry that the library defaults imply — 25 000 / 400
= 62× — down to roughly 9×. `expected_cost_from_model` reports both so the size
of the correction is visible in every report, and
`core/tests/test_decide_costs.py` asserts that the corrected costs move the
threshold towards fewer alarms.

## The site archetypes

A single default is the most misleading form of this model, because the answer
differs by asset class. Three archetypes are defined in `SITE_PROFILES`:

| | unplanned EUR/h | planned EUR/h | secondary damage | What it is |
|---|---:|---:|---:|---|
| `critical_single_train` | 11 000 | 1 800 | 14 000 | One process train. Stopping the asset stops the plant. |
| `mid_size_line` | 3 400 | 900 | 6 500 | The default. A line with some buffering. |
| `spared_duty_standby` | 350 | 300 | 3 000 | A duty/standby pump pair. The "unplanned" cost is running unspared until the repair, not lost production. |

`MaintenanceCostModel.__post_init__` **rejects** a model whose unplanned
downtime rate is below its planned rate, with the message that such an asset is
redundant and condition monitoring has nothing to sell it. That is not a
validation nicety: it is the boundary of the product.

## The audit trail

Every recommendation carries `cost_model.fingerprint()` — the SHA-256 of the
canonical parameter set including the site — plus the list of parameters still
at unverified defaults. When a work order is disputed eighteen months later, the
assumption set is recoverable rather than reconstructed. The full record is in
`MaintenanceCostModel.audit_record()` and is written into
`work_order.evidence` by the production service.
