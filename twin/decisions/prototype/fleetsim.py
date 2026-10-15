"""Multi-year fleet simulation under three maintenance policies.

Policies
--------
``run_to_failure``        No monitoring, no calendar. Assets fail; they are
                          repaired unplanned. The policy most plants actually
                          run on their non-critical assets.
``fixed_interval``        Preventive replacement every ``pm_interval_days``,
                          executed at the first shutdown window on or after the
                          due date. No monitoring cost, and no knowledge: good
                          parts are thrown away and bad ones still fail early.
``condition_based_twin``  Detection from the committed health pipeline, RUL from
                          the scenario-conditional estimator, per-asset action
                          from :func:`pdmcore.decide.policy.decide`, and fleet
                          booking from :class:`pdmcore.decide.schedule.FleetScheduler`
                          on a rolling weekly horizon. Pays the programme's
                          hardware, installation, cloud, analyst time **and its
                          own false alarms**.

Common random numbers
---------------------
The three policies are compared on the *same* degradation realisations: asset
``a``'s ``k``-th life is the same pool entry under every policy. That is
essential -- the cost differences here are a few hundred euros per asset-year
against event costs of tens of thousands, so independent sampling would bury
the signal in Monte-Carlo noise. Where the policies diverge is in *when* a life
ends, so the number of lives consumed differs; drawing from a per-asset ordered
sequence keeps the realisations aligned for as long as the histories agree.

What is simulated and what is drawn
-----------------------------------
The health trajectories, alarm states and prognostic indicators come from
``pipeline.py``, which ran the real DSP. The healthy stage of each life is a
drawn duration rather than a simulated signal, and the false-alarm stream is a
Poisson process at the rate ``run_measurements.py`` measured. Both are stated
in the output, and the false-alarm rate is the assumption the sensitivity
analysis pivots on.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_CORE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "core"))
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from pdmcore.decide.costs import MaintenanceCostModel  # noqa: E402
from pdmcore.decide.counts import CountedRate  # noqa: E402
from pdmcore.decide.policy import (  # noqa: E402
    RUN_TO_FAILURE,
    Opportunity,
    decide,
    programme_balance,
    shutdown_calendar,
)
from pdmcore.decide.schedule import AssetJob, FleetScheduler, ShutdownWindow  # noqa: E402
from pdmcore.health.rul import ParticleFilterRUL  # noqa: E402
from pdmcore.health.scenario_rul import LoadScenario, ScenarioRUL  # noqa: E402
from pipeline import RUL_FAILURE_THRESHOLD, RUL_MIN_HEALTH  # noqa: E402

DAYS_PER_YEAR = 365.25
ATTACH_STATES = ("watch", "alarm", "critical")


@dataclass
class FleetConfig:
    n_assets: int = 40
    n_lines: int = 4
    horizon_years: float = 5.0
    shutdown_interval_days: float = 42.0
    crew_size: int = 4
    crew_hours_per_window: float = 48.0
    parts_budget_eur_per_year: Optional[float] = 40000.0
    job_hours: float = 4.0
    crew_per_job: int = 2
    #: Decision cadence of the condition-based policy.
    decision_interval_days: float = 7.0
    #: Windows offered to the scheduler at each decision epoch.
    rolling_windows: int = 4
    #: Preventive-replacement interval for the fixed-interval policy.
    pm_interval_days: float = 365.0
    #: Confirmed nuisance alarms per asset-year. Measured, weakly -- see
    #: ``run_measurements.py``'s exposure note.
    nuisance_rate_per_asset_year: float = 0.25
    #: Whether an asset may be stopped between shutdowns.
    allow_unscheduled_stop: bool = True
    seed: int = 0

    @property
    def horizon_days(self) -> float:
        return self.horizon_years * DAYS_PER_YEAR

    @property
    def asset_years(self) -> float:
        return self.n_assets * self.horizon_years

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Event:
    day: float
    asset: str
    kind: str          # failure | planned | unscheduled_stop | false_alarm
    eur: float
    detail: Dict[str, object] = field(default_factory=dict)


@dataclass
class Outcome:
    policy: str
    events: List[Event] = field(default_factory=list)
    programme_eur: float = 0.0
    residual_days_lost: float = 0.0
    n_abstained_decisions: int = 0
    n_decision_epochs: int = 0
    n_deferred_by_capacity: int = 0

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)

    def eur(self, kind: Optional[str] = None) -> float:
        return sum(e.eur for e in self.events if kind is None or e.kind == kind)

    @property
    def total_eur(self) -> float:
        return self.eur() + self.programme_eur


class FleetSim:
    """One fleet, one cost model, three policies."""

    def __init__(self, cfg: FleetConfig, cost: MaintenanceCostModel, lives: Sequence):
        if not lives:
            raise ValueError("need a non-empty life pool")
        self.cfg = cfg
        self.cost = cost
        self.pool = list(lives)
        self.assets = [f"A{i:03d}" for i in range(cfg.n_assets)]
        self.line_of = {a: f"L{i % cfg.n_lines}" for i, a in enumerate(self.assets)}
        self.windows = shutdown_calendar(
            interval_days=cfg.shutdown_interval_days,
            horizon_days=cfg.horizon_days + 4 * cfg.shutdown_interval_days,
        )

    # -- life sequences (common random numbers) ------------------------------
    def life_sequence(self, asset: str, n: int = 24) -> List:
        """A deterministic, per-asset ordered draw from the pool.

        Same asset, same index, same realisation under every policy -- the
        paired comparison this whole study depends on.
        """
        rng = np.random.default_rng(
            [self.cfg.seed, int(asset[1:]), 0xA55E7])
        return [self.pool[int(i)] for i in rng.integers(0, len(self.pool), n)]

    # -- cost helpers --------------------------------------------------------
    def _window_day_on_or_after(self, day: float) -> Optional[float]:
        for w in self.windows:
            if w.day >= day - 1e-9:
                return w.day
        return None  # pragma: no cover - the calendar runs past the horizon

    def _charge_grouped_windows(self, outcome: Outcome,
                                bookings: Dict[Tuple[float, str], List[Tuple[str, float]]]) -> None:
        """Charge one stoppage per (window, line), not one per asset.

        ``bookings`` maps ``(window_day, line)`` to ``(asset, residual_days)``
        pairs. Grouping is the only reason a scheduler exists, so the cost
        accounting has to reflect it for every policy, not only for the one
        that optimises it.
        """
        for (day, _line), jobs in sorted(bookings.items()):
            hours = max(
                self.cfg.job_hours,
                len(jobs) * self.cfg.job_hours * self.cfg.crew_per_job / max(1, self.cfg.crew_size),
            )
            outcome.events.append(Event(
                day=day, asset="(line)", kind="planned_downtime",
                eur=self.cost.planned_downtime_eur_per_h * hours,
                detail={"n_jobs": len(jobs), "makespan_h": round(hours, 2)},
            ))
            for asset, residual in jobs:
                outcome.residual_days_lost += max(0.0, residual)
                outcome.events.append(Event(
                    day=day, asset=asset, kind="planned",
                    eur=(self.cost.spare_part_eur
                         + self.cost.labour_eur(self.cfg.job_hours)
                         + self.cost.residual_life_eur(residual)),
                    detail={"residual_days": round(residual, 1)},
                ))

    def _add_false_alarms(self, outcome: Outcome) -> None:
        """A Poisson stream of nuisance work orders over the monitored exposure.

        Drawn rather than simulated: the measured rate comes from the
        commissioning frames in the life pool, and extrapolating it to a
        multi-year exposure is an assumption, not a measurement. Keeping it a
        single explicit parameter is what makes it sweepable.
        """
        rng = np.random.default_rng([self.cfg.seed, 0xFA15E])
        lam = self.cfg.nuisance_rate_per_asset_year * self.cfg.asset_years
        n = int(rng.poisson(lam))
        for k in range(n):
            outcome.events.append(Event(
                day=float(rng.uniform(0.0, self.cfg.horizon_days)),
                asset=self.assets[int(rng.integers(0, len(self.assets)))],
                kind="false_alarm", eur=self.cost.false_alarm_eur(),
                detail={"index": k},
            ))

    # -- policies ------------------------------------------------------------
    def run_to_failure(self) -> Outcome:
        out = Outcome(policy="run_to_failure")
        for asset in self.assets:
            day = 0.0
            for life in self.life_sequence(asset):
                if day >= self.cfg.horizon_days:
                    break
                fail = day + life.total_days
                if fail > self.cfg.horizon_days:
                    break
                out.events.append(Event(day=fail, asset=asset, kind="failure",
                                        eur=self.cost.unplanned_event_eur(),
                                        detail={"life_id": life.life_id}))
                day = fail + self.cost.mttr_unplanned_h / 24.0
        return out

    def fixed_interval(self, pm_interval_days: Optional[float] = None) -> Outcome:
        pm = self.cfg.pm_interval_days if pm_interval_days is None else float(pm_interval_days)
        out = Outcome(policy=f"fixed_interval_{pm:.0f}d")
        bookings: Dict[Tuple[float, str], List[Tuple[str, float]]] = {}
        for asset in self.assets:
            day = 0.0
            for life in self.life_sequence(asset):
                if day >= self.cfg.horizon_days:
                    break
                fail = day + life.total_days
                due = self._window_day_on_or_after(day + pm)
                if due is None or due > self.cfg.horizon_days:
                    due = math.inf
                if fail <= due:
                    if fail > self.cfg.horizon_days:
                        break
                    out.events.append(Event(day=fail, asset=asset, kind="failure",
                                            eur=self.cost.unplanned_event_eur(),
                                            detail={"life_id": life.life_id}))
                    day = fail + self.cost.mttr_unplanned_h / 24.0
                else:
                    bookings.setdefault((due, self.line_of[asset]), []).append(
                        (asset, fail - due))
                    day = due + self.cfg.job_hours / 24.0
        self._charge_grouped_windows(out, bookings)
        return out

    def condition_based(self) -> Outcome:
        cfg = self.cfg
        out = Outcome(policy="condition_based_twin")
        # Per-asset state machine.
        state: Dict[str, Dict[str, object]] = {}
        for asset in self.assets:
            seq = self.life_sequence(asset)
            state[asset] = {
                "seq": seq, "k": 0, "life_start": 0.0, "filter": None,
                "attached": False, "booked_day": None, "fed": 0,
            }
        bookings: Dict[Tuple[float, str], List[Tuple[str, float]]] = {}
        booked_by_window: Dict[float, Dict[str, float]] = {}

        def life_of(asset: str):
            st = state[asset]
            seq = st["seq"]
            return seq[int(st["k"])] if int(st["k"]) < len(seq) else None

        def renew(asset: str, day: float) -> None:
            st = state[asset]
            # Release any crew hours this asset had reserved: an asset that
            # failed before its booking does not also consume the window it was
            # booked into. Leaving the reservation in place would make the
            # scheduler progressively more capacity-constrained as failures
            # accumulated -- a slow leak that would look like a capacity finding.
            booked = st["booked_day"]
            if booked is not None:
                booked_by_window.get(float(booked), {}).pop(asset, None)
            st["k"] = int(st["k"]) + 1
            st["life_start"] = day
            st["filter"] = None
            st["attached"] = False
            st["booked_day"] = None
            st["fed"] = 0

        day = 0.0
        while day <= cfg.horizon_days:
            # 1. Failures and bookings that fall due before the next epoch.
            horizon = min(day + cfg.decision_interval_days, cfg.horizon_days + 1e-9)
            for asset in self.assets:
                st = state[asset]
                life = life_of(asset)
                if life is None:
                    continue
                fail_day = float(st["life_start"]) + life.total_days
                booked = st["booked_day"]
                if booked is not None and booked <= horizon and booked <= fail_day:
                    residual = fail_day - booked
                    bookings.setdefault((booked, self.line_of[asset]), []).append(
                        (asset, residual))
                    renew(asset, booked + cfg.job_hours / 24.0)
                    continue
                if fail_day <= horizon:
                    out.events.append(Event(day=fail_day, asset=asset, kind="failure",
                                            eur=self.cost.unplanned_event_eur(),
                                            detail={"life_id": life.life_id,
                                                    "was_booked_for": booked}))
                    renew(asset, fail_day + self.cost.mttr_unplanned_h / 24.0)

            # 2. Feed the prognostic filters with everything up to today.
            for asset in self.assets:
                st = state[asset]
                life = life_of(asset)
                if life is None:
                    continue
                onset = float(st["life_start"]) + life.healthy_days
                if day < onset:
                    continue
                elapsed = day - onset
                while int(st["fed"]) < len(life.t_days) and life.t_days[int(st["fed"])] <= elapsed:
                    i = int(st["fed"])
                    if not st["attached"]:
                        if life.alarm_state[i] in ATTACH_STATES:
                            st["attached"] = True
                            st["filter"] = ScenarioRUL(rul=ParticleFilterRUL(
                                failure_threshold=RUL_FAILURE_THRESHOLD,
                                min_health=RUL_MIN_HEALTH, seed=3))
                        else:
                            st["fed"] = i + 1
                            continue
                    st["filter"].update(float(life.t_days[i]), float(life.indicator[i]),
                                        load_frac=life.duty)
                    st["fed"] = i + 1

            # 3. Decide, per asset, then book as a fleet.
            out.n_decision_epochs += 1
            jobs: List[AssetJob] = []
            opportunities = [w for w in self.windows
                             if day < w.day <= day + cfg.rolling_windows
                             * cfg.shutdown_interval_days]
            if not opportunities:
                day += cfg.decision_interval_days
                continue
            rel_windows = [
                ShutdownWindow(name=w.name, day=w.day - day, line=None,
                               crew_hours=max(0.0, cfg.crew_hours_per_window
                                              - sum(booked_by_window.get(w.day, {}).values())))
                for w in opportunities
            ]
            for asset in self.assets:
                st = state[asset]
                life = life_of(asset)
                if life is None or not st["attached"] or st["booked_day"] is not None:
                    continue
                sr: ScenarioRUL = st["filter"]  # type: ignore[assignment]
                scenario = LoadScenario.constant(life.duty, name="as_operated")
                samples = sr.failure_day_samples(scenario)
                if samples is None:
                    out.n_abstained_decisions += 1
                    continue
                days_arr, weights = samples
                # An asset that may need stopping between shutdowns is decided
                # on its own, because that action is not the scheduler's to give.
                if cfg.allow_unscheduled_stop:
                    single = decide(
                        [Opportunity("stop_now", 0.0, "unscheduled_stop")]
                        + [Opportunity(w.name, w.day, "planned_shutdown") for w in rel_windows]
                        + [RUN_TO_FAILURE],
                        days_arr, weights, self.cost,
                        horizon_days=cfg.rolling_windows * cfg.shutdown_interval_days,
                    )
                    if single.status == "ok" and single.chosen.opportunity.name == "stop_now":
                        fail_day = float(st["life_start"]) + life.total_days
                        out.events.append(Event(
                            day=day, asset=asset, kind="unscheduled_stop",
                            eur=(self.cost.unscheduled_stop_eur()
                                 + self.cost.residual_life_eur(fail_day - day)),
                            detail={"life_id": life.life_id,
                                    "residual_days": round(fail_day - day, 1)}))
                        out.residual_days_lost += max(0.0, fail_day - day)
                        renew(asset, day + cfg.job_hours / 24.0)
                        continue
                jobs.append(AssetJob(
                    asset_id=asset, line=self.line_of[asset], job_hours=cfg.job_hours,
                    crew_required=cfg.crew_per_job, failure_days=days_arr, weights=weights,
                ))
            if jobs:
                # Parts budget pro-rated over the rolling horizon, which is the
                # share of the annual budget this plan is allowed to commit.
                rolling_days = cfg.rolling_windows * cfg.shutdown_interval_days
                budget = (None if cfg.parts_budget_eur_per_year is None else
                          cfg.parts_budget_eur_per_year * rolling_days / DAYS_PER_YEAR)
                sched = FleetScheduler(
                    self.cost, rel_windows, crew_size=cfg.crew_size,
                    parts_budget_eur=budget, horizon_days=rolling_days,
                )
                res = sched.solve_milp(jobs) or sched.solve_greedy(jobs)
                by_name = {w.name: w for w in opportunities}
                for asset, wname in res.assignments.items():
                    if wname is None:
                        out.n_deferred_by_capacity += 1
                        continue
                    absolute = by_name[wname].day
                    state[asset]["booked_day"] = absolute
                    slot = booked_by_window.setdefault(absolute, {})
                    slot[asset] = cfg.job_hours * cfg.crew_per_job
            day += cfg.decision_interval_days

        self._charge_grouped_windows(out, bookings)
        self._add_false_alarms(out)
        out.programme_eur = self.cost.programme_cost().total_eur * cfg.asset_years
        return out

    # -- reporting -----------------------------------------------------------
    def summarise(self, out: Outcome) -> Dict[str, object]:
        ay = self.cfg.asset_years
        failures = out.count("failure")
        planned = out.count("planned")
        stops = out.count("unscheduled_stop")
        alarms = out.count("false_alarm")
        return {
            "policy": out.policy,
            "asset_years": round(ay, 2),
            "total_eur": round(out.total_eur, 2),
            "eur_per_asset_year": round(out.total_eur / ay, 2),
            "breakdown_eur": {
                "unplanned_failures": round(out.eur("failure"), 2),
                "planned_jobs": round(out.eur("planned"), 2),
                "planned_downtime": round(out.eur("planned_downtime"), 2),
                "unscheduled_stops": round(out.eur("unscheduled_stop"), 2),
                "false_alarms": round(out.eur("false_alarm"), 2),
                "monitoring_programme": round(out.programme_eur, 2),
            },
            "counts": {
                "unplanned_failures": failures,
                "planned_jobs": planned,
                "unscheduled_stops": stops,
                "false_alarms": alarms,
                "line_stoppages": out.count("planned_downtime"),
                "abstained_decisions": out.n_abstained_decisions,
                "decision_epochs": out.n_decision_epochs,
                "deferred_by_the_scheduler": out.n_deferred_by_capacity,
            },
            "rates": {
                "failures": CountedRate(failures, ay, "asset-year",
                                        label="unplanned failures").as_dict(),
                "planned_jobs": CountedRate(planned, ay, "asset-year",
                                            label="planned jobs").as_dict(),
                "false_alarms": CountedRate(alarms, ay, "asset-year",
                                            label="false alarms").as_dict(),
            },
            "residual_life_days_lost_per_asset_year": round(out.residual_days_lost / ay, 2),
        }


def compare(
    cfg: FleetConfig, cost: MaintenanceCostModel, lives: Sequence,
    *, pm_sweep: Sequence[float] = (180.0, 270.0, 365.0, 480.0, 600.0),
) -> Dict[str, object]:
    """Run all three policies and the preventive-interval sweep.

    The fixed-interval policy is swept and the *best* interval is used as the
    comparator, not a strawman: a calendar policy tuned on the same life data
    the twin sees is the honest baseline, and it is a much stronger one.
    """
    sim = FleetSim(cfg, cost, lives)
    rtf = sim.summarise(sim.run_to_failure())
    pm_results = []
    for pm in pm_sweep:
        pm_results.append(sim.summarise(sim.fixed_interval(pm)))
    best_pm = min(pm_results, key=lambda r: r["eur_per_asset_year"])
    cbm_out = sim.condition_based()
    cbm = sim.summarise(cbm_out)

    # Programme balance, priced from the counts the simulation produced rather
    # than from an assumed prevention rate.
    ay = cfg.asset_years
    prevented = max(0, rtf["counts"]["unplanned_failures"] - cbm["counts"]["unplanned_failures"])
    balance = programme_balance(
        cost,
        failures_prevented=CountedRate(prevented, ay, "asset-year", label="failures prevented"),
        failures_remaining=CountedRate(cbm["counts"]["unplanned_failures"], ay, "asset-year",
                                       label="failures remaining"),
        false_alarms=CountedRate(cbm["counts"]["false_alarms"], ay, "asset-year",
                                 label="false alarms"),
        early_interventions_days_lost=cbm_out.residual_days_lost / ay,
    )
    return {
        "config": cfg.as_dict(),
        "cost_model": cost.audit_record(),
        "run_to_failure": rtf,
        "fixed_interval_sweep": pm_results,
        "fixed_interval_best": best_pm,
        "condition_based_twin": cbm,
        "saving_vs_run_to_failure_eur_per_asset_year": round(
            rtf["eur_per_asset_year"] - cbm["eur_per_asset_year"], 2),
        "saving_vs_best_fixed_interval_eur_per_asset_year": round(
            best_pm["eur_per_asset_year"] - cbm["eur_per_asset_year"], 2),
        "programme_balance": balance,
        "pays_for_itself": bool(cbm["eur_per_asset_year"]
                                < min(rtf["eur_per_asset_year"],
                                      best_pm["eur_per_asset_year"])),
    }
