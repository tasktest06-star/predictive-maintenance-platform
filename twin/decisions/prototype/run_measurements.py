#!/usr/bin/env python3
"""What the real pipeline actually delivers to the decision layer.

Reads the life pool produced by ``pipeline.py`` (which ran the DSP) and
measures the four things the decision layer needs and the one thing it assumes:

**Detection lead time.** Days of true remaining life at the first confirmed
ALARM. This is what decides whether an unplanned failure can be converted into
a planned job at all, and it bounds everything the decision layer can achieve.

**Abstention behaviour.** How much of the degradation stage the RUL estimator
declines to answer for, and how much true life is left when it first speaks.
``pdmcore.health.rul`` abstains by design; this puts a number on what that
costs in lead time.

**RUL interval calibration.** Does the reported p10-p90 interval contain the
true remaining life 80 % of the time? The value-of-information analysis assumes
a calibrated estimator, which is generous. If the real one is over-confident,
every VoI figure is an over-estimate, so the coverage is measured rather than
assumed -- and it is reported whether or not it is flattering.

**Nuisance alarm rate.** Confirmed ALARM transitions raised on healthy frames,
per asset-year. The dominant *cost* of the monitoring programme turns out to be
this number rather than the hardware, so it is measured on the same frames the
detector was calibrated against.

**The assumption:** the load-life exponent. ``pdmcore.sim`` has no
load-dependent damage model, so the scenario-conditional answers here exercise
the arithmetic and the gates, and validate nothing about ISO 281. Said again in
the output, because this is the one place a reader could mistake an exercised
code path for a measured effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core")))

from pdmcore.decide.counts import CountedRate  # noqa: E402
from pdmcore.health.scenario_rul import LoadScenario  # noqa: E402
from pipeline import N_COMMISSION, RESULTS, build_scenario_rul, load_pool, replay_rul  # noqa: E402

ALARM_STATES = ("alarm", "critical")


def _first_alarm_index(life) -> Optional[int]:
    for i, s in enumerate(life.alarm_state):
        if s in ALARM_STATES:
            return i
    return None


def _pct(x: Sequence[float], q: float) -> Optional[float]:
    a = np.asarray([v for v in x if v is not None], dtype=np.float64)
    return None if a.size == 0 else float(np.percentile(a, q))


def measure(pool_meta: Dict[str, object], lives: List) -> Dict[str, object]:
    detection: List[Dict[str, object]] = []
    first_ok: List[Dict[str, object]] = []
    coverage_hits = 0
    coverage_n = 0
    widths_at_decision: List[float] = []
    log_ratio: List[float] = []
    rel_error: List[float] = []
    abstain_days = 0
    total_days = 0
    nuisance_alarms = 0
    healthy_asset_years = 0.0
    healthy_exceed = 0
    healthy_scored = 0
    never_detected = 0
    scenario_examples: List[Dict[str, object]] = []

    for life in lives:
        total_days += len(life.t_days)
        nuisance_alarms += life.commissioning_alarms
        healthy_exceed += life.healthy_scores_above_threshold
        healthy_scored += life.n_healthy_scored
        # Exposure: the whole healthy stage of the life is time at risk of a
        # nuisance alarm, not just the commissioning frames that were simulated.
        healthy_asset_years += life.healthy_days / 365.25

        ai = _first_alarm_index(life)
        if ai is None:
            never_detected += 1
        else:
            detection.append({
                "life_id": life.life_id, "preset": life.preset, "mode": life.mode,
                "lead_time_days": float(life.true_rul_days[ai]),
                "degradation_days": life.degradation_days,
                "lead_fraction": float(life.true_rul_days[ai] / life.degradation_days),
            })

        records = replay_rul(life)
        said_something = False
        for rec in records:
            if rec["status"] != "ok":
                abstain_days += 1
                continue
            true_rul = float(rec["true_rul_days"])
            p10, p90 = float(rec["p10_days"]), float(rec["p90_days"])
            coverage_n += 1
            if p10 <= true_rul <= p90:
                coverage_hits += 1
            if true_rul > 0.5 and rec["median_days"]:
                log_ratio.append(math.log(float(rec["median_days"]) / true_rul))
                rel_error.append((float(rec["median_days"]) - true_rul) / true_rul)
            if not said_something:
                said_something = True
                first_ok.append({
                    "life_id": life.life_id,
                    "true_rul_days": true_rul,
                    "median_days": rec["median_days"],
                    "interval_width_days": rec["interval_width_days"],
                })
                widths_at_decision.append(float(rec["interval_width_days"]))

    # Scenario-conditional examples, from the real filter on real lives.
    for life in lives[:6]:
        sr, attach = build_scenario_rul(life)
        if attach < 0:
            continue
        at = sr.estimate(LoadScenario.constant(life.duty, name="as_operated"))
        if at.status != "ok":
            continue
        row = {"life_id": life.life_id, "preset": life.preset, "mode": life.mode,
               "observed_duty": round(life.duty, 3),
               "unconditional_median_days": at.unconditional.median_days,
               "scenarios": {}}
        for target in (0.70, 0.85, 1.0):
            est = sr.estimate(LoadScenario.constant(target, name=f"duty_{target:.2f}"))
            row["scenarios"][f"{target:.2f}"] = {
                "status": est.status,
                "median_days": est.median_days,
                "p10_days": est.p10_days,
                "p90_days": est.p90_days,
                "load_model_width_share": est.load_model_width_share,
                "note": est.note[:160],
            }
        # The planner's actual question, at the observed duty.
        row["p_failure_before_42d"] = sr.probability_of_failure_before(
            LoadScenario.constant(life.duty), 42.0)
        scenario_examples.append(row)

    lead = [d["lead_time_days"] for d in detection]
    first_ok_rul = [r["true_rul_days"] for r in first_ok]
    nuisance = CountedRate(nuisance_alarms, max(healthy_asset_years, 1e-6),
                           "asset-year", label="confirmed nuisance alarms")

    # The true-life distribution at the decision epoch, as a lognormal, which is
    # what the VoI study needs as input. Fitted on the remaining life at the
    # first ALARM, because that is when the decision is actually taken.
    lead_arr = np.asarray([v for v in lead if v > 0], dtype=np.float64)
    if lead_arr.size >= 5:
        life_median = float(np.exp(np.median(np.log(lead_arr))))
        life_log_sd = float(np.std(np.log(lead_arr)))
    else:  # pragma: no cover - only on a degenerate pool
        life_median, life_log_sd = 60.0, 0.5

    return {
        "n_lives": len(lives),
        "never_detected": never_detected,
        "detection": {
            "n": len(lead),
            "lead_time_days_p10": _pct(lead, 10),
            "lead_time_days_median": _pct(lead, 50),
            "lead_time_days_p90": _pct(lead, 90),
            "lead_fraction_median": _pct([d["lead_fraction"] for d in detection], 50),
            "per_mode_median_lead_days": {
                mode: _pct([d["lead_time_days"] for d in detection if d["mode"] == mode], 50)
                for mode in sorted({d["mode"] for d in detection})
            },
        },
        "abstention": {
            "abstained_observation_share": round(abstain_days / max(total_days, 1), 4),
            "n_lives_with_an_answer": len(first_ok),
            "true_rul_at_first_answer_days_median": _pct(first_ok_rul, 50),
            "true_rul_at_first_answer_days_p10": _pct(first_ok_rul, 10),
            "lead_time_lost_to_abstention_days_median": (
                None if not lead or not first_ok_rul
                else round(_pct(lead, 50) - _pct(first_ok_rul, 50), 2)),
        },
        "rul_quality": {
            "n_scored": coverage_n,
            "p10_p90_coverage": (None if coverage_n == 0
                                 else round(coverage_hits / coverage_n, 4)),
            "p10_p90_coverage_target": 0.80,
            "median_log_ratio": None if not log_ratio else round(float(np.median(log_ratio)), 4),
            "median_relative_error": (None if not rel_error
                                      else round(float(np.median(rel_error)), 4)),
            "interval_width_at_first_answer_days_median": _pct(widths_at_decision, 50),
            "interval_width_at_first_answer_days_p90": _pct(widths_at_decision, 90),
        },
        "nuisance": {
            "rate": nuisance.as_dict(),
            "healthy_frame_exceedance": {
                "n_above_threshold": healthy_exceed,
                "n_scored": healthy_scored,
                "share": round(healthy_exceed / max(healthy_scored, 1), 5),
                "target_fpr": pool_meta.get("detector_fpr_target"),
                "note": (
                    "per-frame exceedance against the held-out 1 % threshold; the "
                    "confirmed-alarm rate above is much lower because the n-of-m rule "
                    "and the dwell time in pdmcore.health.index reject most single-frame "
                    "excursions. The gap between the two is what debouncing buys."
                ),
            },
            "commissioning_frames_per_life": N_COMMISSION,
            "exposure_note": (
                "Nuisance alarms are counted on the simulated commissioning frames but "
                "the exposure is the whole drawn healthy stage of each life, so this "
                "rate EXTRAPOLATES from 60 frames per life to hundreds of healthy days. "
                "It is the weakest measurement on this branch and the sensitivity "
                "analysis treats it as the pivotal assumption rather than as data."
            ),
        },
        "voi_inputs": {
            "true_life_median_days": round(life_median, 2),
            "true_life_log_sd": round(life_log_sd, 4),
            "measured_interval_width_days": _pct(widths_at_decision, 50),
            "note": ("the VoI study is driven by these, so its numbers are about the "
                     "lives this pipeline produced, not about a plant"),
        },
        "scenario_conditional": {
            "examples": scenario_examples,
            "caveat": (
                "pdmcore.sim has NO load-dependent damage model. These conditional "
                "answers exercise the damage-clock arithmetic and the abstention gates; "
                "they do not validate the ISO 281 load-life exponent, and no measurement "
                "in this repository does."
            ),
        },
        "pool_caveats": pool_meta.get("caveats", []),
        "detectors": pool_meta.get("detectors", {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args(argv)

    meta, lives = load_pool(args.pool)
    out = measure(meta, lives)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "pipeline_measurements.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)

    d, a, q, n = out["detection"], out["abstention"], out["rul_quality"], out["nuisance"]
    print(f"lives {out['n_lives']}, never detected {out['never_detected']}")
    print(f"detection lead time   p10 {d['lead_time_days_p10']:.1f}  "
          f"median {d['lead_time_days_median']:.1f}  p90 {d['lead_time_days_p90']:.1f} days")
    print(f"RUL abstains on       {a['abstained_observation_share'] * 100:.0f} % of observations")
    print(f"true RUL when it      median {a['true_rul_at_first_answer_days_median']:.1f} days "
          f"(lead lost to abstention {a['lead_time_lost_to_abstention_days_median']:.1f} d)")
    print(f"p10-p90 coverage      {q['p10_p90_coverage']:.3f} (target 0.80) on "
          f"{q['n_scored']} estimates")
    print(f"interval width        median {q['interval_width_at_first_answer_days_median']:.1f} days "
          f"at the first answer")
    print(f"nuisance alarms       {n['rate']['rate']:.3f}/asset-year "
          f"[{n['rate']['ci_lo']:.3f}, {n['rate']['ci_hi']:.3f}] on n={n['rate']['count']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
