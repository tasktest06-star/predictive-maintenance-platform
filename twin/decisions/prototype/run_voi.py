#!/usr/bin/env python3
"""The value-of-information study: how much is a better RUL estimate worth?

Driven by what the real pipeline measured (``pipeline_measurements.json``):
the distribution of true remaining life at the moment the alarm fires, and the
credible-interval width the shipped estimator actually achieves. Swept over
three site cost archetypes, four shutdown spacings, and the presence or absence
of the option to stop an asset between shutdowns.

Four numbers come out, in increasing order of how uncomfortable they are:

1. **The value of detection** -- run-to-failure cost minus the cost of
   intervening at the next shutdown. This is what the monitoring programme
   sells.
2. **EVPI** -- the entire value of *perfect* prognostics on top of detection.
   The ceiling on what any twin, any sensor and any model can add.
3. **The decision-relevant precision** -- the widest credible interval that
   still captures almost all of EVPI. Below it, measurement has stopped
   changing the action.
4. **The break-even precision** -- the width beyond which acting on the RUL
   costs *more* than ignoring it, which is the quantitative case for
   ``pdmcore.health.rul`` abstaining rather than reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core")))

from pdmcore.decide.costs import SITE_PROFILES, MaintenanceCostModel  # noqa: E402
from pdmcore.decide.voi import (  # noqa: E402
    VoIStudy,
    break_even_precision,
    log_sd_for_width,
    precision_step_value,
    precision_table,
    voi_curve,
)
from pipeline import RESULTS  # noqa: E402

WIDTHS_WEEKS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0)
INTERVALS_DAYS = (14.0, 28.0, 42.0, 84.0)


def site_model(name: str) -> MaintenanceCostModel:
    """A cost model a site has signed for, with every default acknowledged.

    Acknowledging rather than quoting is honest here: these are archetypes, not
    a plant's numbers. The audit record says so, and the gate in
    :meth:`MaintenanceCostModel.require_configured` is satisfied by an explicit
    act rather than by silence.
    """
    overrides = SITE_PROFILES[name]
    probe = MaintenanceCostModel(site=name, **overrides)
    return MaintenanceCostModel(
        site=name, acknowledged_defaults=probe.unverified_defaults(),
        note=f"SITE_PROFILES[{name!r}] archetype; every unquoted parameter is an estimate",
        **overrides,
    )


def one_study(
    cost: MaintenanceCostModel, *, interval: float, allow_stop: bool,
    life_median: float, life_log_sd: float, measured_width_days: Optional[float],
    n_draws: int, n_posterior: int,
) -> Dict[str, object]:
    st = VoIStudy(
        cost=cost,
        shutdown_interval_days=interval,
        horizon_days=max(600.0, 8 * interval),
        true_life_median_days=life_median,
        true_life_log_sd=life_log_sd,
        allow_unscheduled_stop=allow_stop,
        n_draws=n_draws,
        n_posterior=n_posterior,
    )
    cur = voi_curve(st, widths_weeks=WIDTHS_WEEKS)
    base = cur["baselines"]
    out = {
        "site": cost.site,
        "shutdown_interval_days": interval,
        "allow_unscheduled_stop": allow_stop,
        "baselines": base,
        "value_of_detection_eur": round(
            base["run_to_failure_eur"] - base["blanket_next_shutdown_eur"], 2),
        "evpi_eur": base["evpi_eur"],
        "evpi_share_of_detection_value": (
            None if base["run_to_failure_eur"] <= base["blanket_next_shutdown_eur"]
            else round(base["evpi_eur"]
                       / (base["run_to_failure_eur"] - base["blanket_next_shutdown_eur"]), 4)),
        "points": cur["points"],
        "precision_table": precision_table(cur, captures=(0.5, 0.8, 0.9, 0.95)),
        "break_even": break_even_precision(cur),
        "step_8w_to_4w": precision_step_value(st, 8.0, 4.0),
        "step_4w_to_2w": precision_step_value(st, 4.0, 2.0),
    }
    if measured_width_days:
        s = log_sd_for_width(life_median, measured_width_days)
        p = st.point(s)
        out["at_measured_precision"] = {
            "interval_width_days": round(measured_width_days, 2),
            **p.as_dict(),
            "value_left_on_the_table_eur": round(
                p.expected_cost_eur - base["perfect_information_eur"], 2),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements", default=os.path.join(RESULTS, "pipeline_measurements.json"))
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--posterior", type=int, default=300)
    args = ap.parse_args(argv)

    with open(args.measurements) as fh:
        meas = json.load(fh)
    vi = meas["voi_inputs"]
    life_median = float(vi["true_life_median_days"])
    life_log_sd = float(vi["true_life_log_sd"])
    measured_width = vi.get("measured_interval_width_days")

    studies: List[Dict[str, object]] = []
    for name in ("critical_single_train", "mid_size_line", "spared_duty_standby"):
        cost = site_model(name)
        for interval in INTERVALS_DAYS:
            studies.append(one_study(
                cost, interval=interval, allow_stop=True, life_median=life_median,
                life_log_sd=life_log_sd, measured_width_days=measured_width,
                n_draws=args.draws, n_posterior=args.posterior))
            print(f"  {name:22s} {interval:5.0f} d  stop_now  "
                  f"detection {studies[-1]['value_of_detection_eur']:10.0f}  "
                  f"EVPI {studies[-1]['evpi_eur']:9.0f}", flush=True)

    # The operating constraint that turned out to dominate: can the asset be
    # stopped between shutdowns at all?
    constrained: List[Dict[str, object]] = []
    cost = site_model("mid_size_line")
    for interval in INTERVALS_DAYS:
        constrained.append(one_study(
            cost, interval=interval, allow_stop=False, life_median=life_median,
            life_log_sd=life_log_sd, measured_width_days=measured_width,
            n_draws=args.draws, n_posterior=args.posterior))
        print(f"  {'mid_size_line':22s} {interval:5.0f} d  no-stop   "
              f"detection {constrained[-1]['value_of_detection_eur']:10.0f}  "
              f"EVPI {constrained[-1]['evpi_eur']:9.0f}", flush=True)

    ref = next(s for s in studies
               if s["site"] == "mid_size_line" and s["shutdown_interval_days"] == 42.0)
    ref_nostop = next(s for s in constrained if s["shutdown_interval_days"] == 42.0)
    headline = {
        "reference_case": "mid_size_line, 42-day shutdown interval, stop-now available",
        "value_of_detection_eur_per_event": ref["value_of_detection_eur"],
        "evpi_eur_per_event": ref["evpi_eur"],
        "evpi_share_of_detection_value": ref["evpi_share_of_detection_value"],
        "decision_relevant_width_days_capture80": next(
            (r["width_days"] for r in ref["precision_table"] if r["capture"] == 0.8), None),
        "decision_relevant_width_over_shutdown_interval_capture80": next(
            (r["width_over_shutdown_interval"] for r in ref["precision_table"]
             if r["capture"] == 0.8), None),
        "break_even_width_days": ref["break_even"]["width_days"],
        "value_of_halving_the_interval_4w_to_2w_eur": ref["step_4w_to_2w"][
            "value_eur_per_decision"],
        "p_action_unchanged_when_halving_4w_to_2w": ref["step_4w_to_2w"]["p_same_action"],
        "evpi_without_the_option_to_stop_between_shutdowns_eur": ref_nostop["evpi_eur"],
        "findings": [
            "Conditional on detection having happened, the entire value of PERFECT "
            "prognostics is EVPI. Compare it with the value of detection itself: the "
            "programme is sold on the first number and funded on the second.",
            "Almost all of EVPI is the option to act between shutdowns. Removing that "
            "option -- a continuous process that cannot stop on a Tuesday -- collapses "
            "it, because the clairvoyant is then only shaving residual life off a job "
            "that was happening anyway, and that is bounded by the price of the part.",
            "Beyond the break-even width, acting on the RUL costs more than ignoring "
            "it. That is the case for abstention in money rather than in principle.",
            "The decision-relevant width scales with the shutdown interval, not with "
            "anything about the sensor: the action can only change at window "
            "granularity.",
        ],
        "limits": [
            "Single-shot decision: choose once, now. A real programme re-decides weekly "
            "and waiting has option value, so these figures are an UPPER bound on the "
            "value of precision -- generous to the twin.",
            "The estimator is modelled as calibrated (posterior centred on the truth). "
            "pipeline_measurements.json reports the real estimator's p10-p90 coverage; "
            "where that is below 0.80 the real value is lower than computed here.",
            "Every cost parameter is an estimate. See research/01_cost_model_assumptions.md "
            "and the sensitivity analysis in fleet_sim.json.",
        ],
    }
    out = {
        "voi_inputs": vi,
        "widths_weeks": list(WIDTHS_WEEKS),
        "intervals_days": list(INTERVALS_DAYS),
        "n_draws": args.draws,
        "n_posterior": args.posterior,
        "headline": headline,
        "studies": studies,
        "studies_without_unscheduled_stop": constrained,
        "cost_models": {name: site_model(name).audit_record()
                        for name in SITE_PROFILES},
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "voi.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print()
    for k, v in headline.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k:62s} {v}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
