"""Fleet scheduling: many assets, one crew, a few shutdown windows, a budget.

The problem
-----------
A single-asset decision (:mod:`pdmcore.decide.policy`) ignores the three things
that actually bind a maintenance plan:

* **Shared downtime.** Two jobs on the same line in the same window cost one
  stoppage, not two. This is the only genuinely combinatorial part of the
  problem and the reason a per-asset decision rule leaves money on the table.
* **Crew capacity.** A shutdown window is a fixed number of crew-hours. A plan
  that assigns 300 hours into a 120-hour window is not a plan.
* **Budget.** Parts spend is annual and capped, so some correct interventions
  must still be deferred -- and *which* ones is the question.

Formulation
-----------
Binary assignment ``x[i][w]`` of asset ``i`` to window ``w``, plus ``defer[i]``.
Continuous ``m[l][w]`` is the stoppage length of line ``l`` in window ``w``,
bounded below by the longest job assigned to it and by the crew-limited
makespan of all jobs assigned to it. Line downtime is charged on ``m``, once,
which is what creates the incentive to group.

Objective: expected cost, with each asset's assignment cost taken from the same
expected-cost arithmetic as the single-asset decision
(:func:`pdmcore.decide.policy.evaluate_opportunities`) minus the downtime term,
which has moved into ``m``.

Solved with :func:`scipy.optimize.milp` (HiGHS) when scipy is available --
scipy is already a core dependency, so this adds nothing to the install -- and
with a greedy-plus-local-search heuristic otherwise. **Both are always run**
and both objectives are reported. That is deliberate: a scheduler whose
heuristic is within a fraction of a per cent of the optimum does not need a
solver in production, and the only way to know that is to measure it on the
instances that actually arise. At fleet sizes this repository targets (tens to a
few hundred assets, under ten windows) HiGHS solves to proven optimality in
milliseconds, so the gap is a statement about the heuristic, not about the
solver.

Baselines, because a scheduler that cannot beat them is not worth operating
--------------------------------------------------------------------------
``run_to_failure``        defer everything.
``blanket_next_window``   intervene on every asset at the earliest eligible
                          window. The "just fix them all" plan.
``fix_everything_always`` preventive replacement of every asset at *every*
                          window -- the reductio, included because it is what
                          "we do it on a schedule" degenerates into when the
                          schedule is set by caution rather than by life data.

Assets whose RUL abstained are **not scheduled**. They are returned in
:attr:`ScheduleResult.no_recommendation` with the reason. Filling in a plausible
posterior so the optimiser has something to chew on would produce a plan that
looks complete and is partly fiction; the cost of refusing is a planner who has
to think about three assets by hand.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .costs import MaintenanceCostModel

__all__ = [
    "AssetJob",
    "ShutdownWindow",
    "ScheduleResult",
    "FleetScheduler",
]


@dataclass
class AssetJob:
    """One asset's candidate intervention."""

    asset_id: str
    line: str
    job_hours: float
    crew_required: int = 2
    #: Weighted posterior samples of days-to-failure from the decision epoch,
    #: or ``None`` when the RUL estimator abstained.
    failure_days: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    part_eur: Optional[float] = None
    criticality: int = 3
    abstain_reason: str = ""

    @property
    def abstained(self) -> bool:
        return self.failure_days is None

    def _norm(self) -> Tuple[np.ndarray, np.ndarray]:
        d = np.asarray(self.failure_days, dtype=np.float64).ravel()
        if self.weights is None:
            w = np.full(d.size, 1.0 / d.size)
        else:
            w = np.asarray(self.weights, dtype=np.float64).ravel()
            w = w / float(np.sum(w))
        return d, w

    def p_failure_before(self, day: float) -> float:
        d, w = self._norm()
        return float(np.sum(w[d <= day]))

    def expected_residual_days(self, day: float) -> float:
        d, w = self._norm()
        after = d > day
        p = float(np.sum(w[after]))
        if p <= 1e-12:
            return 0.0
        return float(np.sum(w[after] * (d[after] - day)) / p)


@dataclass(frozen=True)
class ShutdownWindow:
    """A planned stoppage the plant is taking anyway."""

    name: str
    day: float
    #: ``None`` means plant-wide: every line can be worked in it.
    line: Optional[str] = None
    #: Crew-hours available in the window.
    crew_hours: float = 120.0
    #: Marginal cost per hour of extending this line's stoppage. ``None`` takes
    #: the cost model's planned rate.
    downtime_eur_per_h: Optional[float] = None

    def accepts(self, job: AssetJob) -> bool:
        return self.line is None or self.line == job.line


@dataclass
class ScheduleResult:
    """A plan, its cost, and what it was compared against."""

    solver: str
    assignments: Dict[str, Optional[str]] = field(default_factory=dict)
    objective_eur: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    per_window: Dict[str, Dict[str, object]] = field(default_factory=dict)
    no_recommendation: Dict[str, str] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, object]] = field(default_factory=dict)
    infeasible: bool = False
    note: str = ""

    @property
    def n_deferred(self) -> int:
        return sum(1 for v in self.assignments.values() if v is None)

    def saving_vs(self, baseline: str) -> Optional[float]:
        b = self.baselines.get(baseline)
        if b is None:
            return None
        return float(b["objective_eur"]) - self.objective_eur

    def as_dict(self) -> Dict[str, object]:
        return {
            "solver": self.solver,
            "objective_eur": round(self.objective_eur, 2),
            "components": {k: round(v, 2) for k, v in self.components.items()},
            "assignments": dict(self.assignments),
            "n_scheduled": sum(1 for v in self.assignments.values() if v is not None),
            "n_deferred": self.n_deferred,
            "per_window": self.per_window,
            "no_recommendation": dict(self.no_recommendation),
            "baselines": self.baselines,
            "savings_vs_baselines_eur": {
                k: (None if self.saving_vs(k) is None else round(self.saving_vs(k), 2))
                for k in self.baselines
            },
            "infeasible": self.infeasible,
            "note": self.note,
        }


class FleetScheduler:
    """Builds and solves the assignment problem for one planning horizon."""

    def __init__(
        self,
        cost: MaintenanceCostModel,
        windows: Sequence[ShutdownWindow],
        *,
        crew_size: Optional[int] = None,
        parts_budget_eur: Optional[float] = None,
        horizon_days: Optional[float] = None,
        require_configured: bool = True,
    ):
        if require_configured:
            cost.require_configured()
        if not windows:
            raise ValueError("need at least one shutdown window")
        self.cost = cost
        self.windows = list(sorted(windows, key=lambda w: w.day))
        self.crew_size = int(crew_size if crew_size is not None else cost.crew_size)
        self.parts_budget_eur = parts_budget_eur
        self.horizon_days = (horizon_days if horizon_days is not None
                             else max(w.day for w in self.windows) * 1.5)

    # -- cost pieces ---------------------------------------------------------
    def _part_eur(self, job: AssetJob) -> float:
        return self.cost.spare_part_eur if job.part_eur is None else float(job.part_eur)

    def assignment_cost(self, job: AssetJob, window: ShutdownWindow) -> float:
        """Expected cost of doing ``job`` in ``window``, **excluding** line downtime.

        Downtime is excluded here and charged once per (line, window) on the
        makespan variable, which is the whole point of the formulation. Keeping
        the two apart is also why this cannot just call
        ``policy.evaluate_opportunities``; the per-asset function necessarily
        charges downtime per asset.
        """
        p_fail = job.p_failure_before(window.day)
        residual = self.cost.residual_life_eur(job.expected_residual_days(window.day))
        return (
            p_fail * self.cost.unplanned_event_eur()
            + (1.0 - p_fail) * (self._part_eur(job)
                                + self.cost.labour_eur(job.job_hours)
                                + residual)
        )

    def defer_cost(self, job: AssetJob) -> float:
        """Expected cost of not scheduling ``job`` inside the horizon.

        A failure before the horizon costs the unplanned event; survival to the
        horizon is charged a planned event rather than nothing, because
        otherwise a finite horizon makes deferral free and every scheduler
        becomes a procrastinator. The bias is real and this is the correction
        for it, stated rather than absorbed.
        """
        p_fail = job.p_failure_before(self.horizon_days)
        return (p_fail * self.cost.unplanned_event_eur()
                + (1.0 - p_fail) * self.cost.planned_event_eur())

    def _downtime_rate(self, window: ShutdownWindow) -> float:
        return (self.cost.planned_downtime_eur_per_h
                if window.downtime_eur_per_h is None else float(window.downtime_eur_per_h))

    def _makespan(self, jobs: Sequence[AssetJob]) -> float:
        """Stoppage hours for a set of jobs sharing one line and window."""
        if not jobs:
            return 0.0
        longest = max(j.job_hours for j in jobs)
        crew_limited = sum(j.job_hours * j.crew_required for j in jobs) / max(1, self.crew_size)
        return max(longest, crew_limited)

    # -- objective evaluation of an explicit plan ----------------------------
    def evaluate_plan(
        self, jobs: Sequence[AssetJob], assignment: Dict[str, Optional[str]]
    ) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, object]]]:
        by_name = {w.name: w for w in self.windows}
        jobs_by_id = {j.asset_id: j for j in jobs}
        asset_term = 0.0
        downtime_term = 0.0
        parts = 0.0
        groups: Dict[Tuple[str, str], List[AssetJob]] = {}
        for aid, wname in assignment.items():
            job = jobs_by_id[aid]
            if wname is None:
                asset_term += self.defer_cost(job)
                continue
            w = by_name[wname]
            asset_term += self.assignment_cost(job, w)
            parts += self._part_eur(job)
            groups.setdefault((job.line, wname), []).append(job)
        per_window: Dict[str, Dict[str, object]] = {}
        for w in self.windows:
            lines: Dict[str, object] = {}
            used = 0.0
            for (line, wname), group in groups.items():
                if wname != w.name:
                    continue
                span = self._makespan(group)
                downtime_term += self._downtime_rate(w) * span
                used += sum(j.job_hours * j.crew_required for j in group)
                lines[line] = {
                    "assets": [j.asset_id for j in group],
                    "makespan_h": round(span, 2),
                    "crew_hours": round(sum(j.job_hours * j.crew_required for j in group), 2),
                }
            per_window[w.name] = {
                "day": w.day,
                "lines": lines,
                "crew_hours_used": round(used, 2),
                "crew_hours_available": w.crew_hours,
                "over_capacity": bool(used > w.crew_hours + 1e-9),
            }
        components = {
            "asset_expected_eur": asset_term,
            "line_downtime_eur": downtime_term,
            "parts_committed_eur": parts,
        }
        return asset_term + downtime_term, components, per_window

    # -- baselines -----------------------------------------------------------
    def _baselines(self, jobs: Sequence[AssetJob]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}

        rtf = {j.asset_id: None for j in jobs}
        obj, comp, pw = self.evaluate_plan(jobs, rtf)
        out["run_to_failure"] = {
            "objective_eur": round(obj, 2),
            "components": {k: round(v, 2) for k, v in comp.items()},
            "note": "defer everything; the policy the plant had before the programme",
        }

        blanket: Dict[str, Optional[str]] = {}
        for j in jobs:
            eligible = [w for w in self.windows if w.accepts(j)]
            blanket[j.asset_id] = eligible[0].name if eligible else None
        obj, comp, pw = self.evaluate_plan(jobs, blanket)
        over = [name for name, d in pw.items() if d["over_capacity"]]
        out["blanket_next_window"] = {
            "objective_eur": round(obj, 2),
            "components": {k: round(v, 2) for k, v in comp.items()},
            "windows_over_capacity": over,
            "note": ("intervene on everything at the earliest eligible window; "
                     + (f"violates crew capacity in {len(over)} window(s), so it is not "
                        "merely expensive, it is not executable"
                        if over else "within crew capacity")),
        }

        # Preventive replacement of everything at every window. Not an
        # assignment (each asset appears once in the assignment problem), so
        # costed directly.
        total = 0.0
        for w in self.windows:
            by_line: Dict[str, List[AssetJob]] = {}
            for j in jobs:
                if w.accepts(j):
                    by_line.setdefault(j.line, []).append(j)
            for line, group in by_line.items():
                total += self._downtime_rate(w) * self._makespan(group)
                for j in group:
                    # A part changed at every window throws away nearly all its life.
                    total += (self._part_eur(j) + self.cost.labour_eur(j.job_hours)
                              + self.cost.residual_life_eur(self.cost.design_life_days))
        out["fix_everything_always"] = {
            "objective_eur": round(total, 2),
            "note": (f"preventive replacement of every asset in every one of the "
                     f"{len(self.windows)} windows, residual life written off each time"),
        }
        return out

    # -- greedy --------------------------------------------------------------
    def solve_greedy(self, jobs: Sequence[AssetJob], *, local_passes: int = 4) -> ScheduleResult:
        """Greedy by expected saving, then local moves.

        The greedy rule is "whichever asset gains most from being scheduled at
        all, placed where it gains most given what is already placed". The
        marginal makespan cost is recomputed at every placement, so grouping
        emerges rather than being imposed -- a second job in a window that is
        already stopping that line is close to free, and the rule sees that.
        """
        active = [j for j in jobs if not j.abstained]
        assignment: Dict[str, Optional[str]] = {j.asset_id: None for j in jobs}
        budget_used = 0.0
        crew_used = {w.name: 0.0 for w in self.windows}
        placed: Dict[Tuple[str, str], List[AssetJob]] = {}

        def marginal(job: AssetJob, w: ShutdownWindow) -> Optional[float]:
            if not w.accepts(job):
                return None
            need = job.job_hours * job.crew_required
            if crew_used[w.name] + need > w.crew_hours + 1e-9:
                return None
            if (self.parts_budget_eur is not None
                    and budget_used + self._part_eur(job) > self.parts_budget_eur + 1e-9):
                return None
            group = placed.get((job.line, w.name), [])
            before = self._makespan(group)
            after = self._makespan(list(group) + [job])
            return (self.assignment_cost(job, w)
                    + self._downtime_rate(w) * (after - before)
                    - self.defer_cost(job))

        remaining = list(active)
        while remaining:
            best: Optional[Tuple[float, AssetJob, ShutdownWindow]] = None
            for job in remaining:
                for w in self.windows:
                    delta = marginal(job, w)
                    if delta is None or delta >= 0.0:
                        continue
                    if best is None or delta < best[0]:
                        best = (delta, job, w)
            if best is None:
                break
            _delta, job, w = best
            assignment[job.asset_id] = w.name
            crew_used[w.name] += job.job_hours * job.crew_required
            budget_used += self._part_eur(job)
            placed.setdefault((job.line, w.name), []).append(job)
            remaining.remove(job)

        # Local search: single-asset moves, then pairwise swaps. The swaps
        # matter because a parts budget or a full window makes the move
        # neighbourhood disconnected -- asset X cannot move into the window it
        # belongs in until asset Y moves out, and no sequence of single moves
        # that each improve the objective gets there.
        obj, comp, pw = self.evaluate_plan(jobs, assignment)
        for _ in range(local_passes):
            improved = False
            for job in active:
                current = assignment[job.asset_id]
                options: List[Optional[str]] = [None] + [
                    w.name for w in self.windows if w.accepts(job)]
                for opt in options:
                    if opt == current:
                        continue
                    trial = dict(assignment)
                    trial[job.asset_id] = opt
                    if not self._feasible(jobs, trial):
                        continue
                    t_obj, t_comp, t_pw = self.evaluate_plan(jobs, trial)
                    if t_obj < obj - 1e-6:
                        assignment, obj, comp, pw = trial, t_obj, t_comp, t_pw
                        improved = True
                        break
            for a in active:
                for b in active:
                    if a.asset_id >= b.asset_id:
                        continue
                    wa, wb = assignment[a.asset_id], assignment[b.asset_id]
                    if wa == wb:
                        continue
                    if wa is not None and not any(
                            w.name == wa and w.accepts(b) for w in self.windows):
                        continue
                    if wb is not None and not any(
                            w.name == wb and w.accepts(a) for w in self.windows):
                        continue
                    trial = dict(assignment)
                    trial[a.asset_id], trial[b.asset_id] = wb, wa
                    if not self._feasible(jobs, trial):
                        continue
                    t_obj, t_comp, t_pw = self.evaluate_plan(jobs, trial)
                    if t_obj < obj - 1e-6:
                        assignment, obj, comp, pw = trial, t_obj, t_comp, t_pw
                        improved = True
            if not improved:
                break
        return ScheduleResult(
            solver="greedy+local",
            assignments=assignment,
            objective_eur=obj,
            components=comp,
            per_window=pw,
            no_recommendation={j.asset_id: (j.abstain_reason or "RUL abstained")
                               for j in jobs if j.abstained},
        )

    def _feasible(self, jobs: Sequence[AssetJob], assignment: Dict[str, Optional[str]]) -> bool:
        jobs_by_id = {j.asset_id: j for j in jobs}
        crew = {w.name: 0.0 for w in self.windows}
        parts = 0.0
        for aid, wname in assignment.items():
            if wname is None:
                continue
            j = jobs_by_id[aid]
            crew[wname] += j.job_hours * j.crew_required
            parts += self._part_eur(j)
        for w in self.windows:
            if crew[w.name] > w.crew_hours + 1e-9:
                return False
        if self.parts_budget_eur is not None and parts > self.parts_budget_eur + 1e-9:
            return False
        return True

    # -- MILP ----------------------------------------------------------------
    def solve_milp(self, jobs: Sequence[AssetJob]) -> Optional[ScheduleResult]:
        """Exact assignment by :func:`scipy.optimize.milp`, or ``None``.

        Returns ``None`` -- rather than a fabricated plan -- when scipy's MILP
        interface is not importable, so the caller can fall back to the
        heuristic knowingly.
        """
        try:
            from scipy.optimize import LinearConstraint, milp  # noqa: PLC0415
            from scipy.optimize import Bounds  # noqa: PLC0415
        except Exception:  # pragma: no cover - scipy is a core dependency here
            return None

        active = [j for j in jobs if not j.abstained]
        if not active:
            return ScheduleResult(
                solver="milp",
                assignments={j.asset_id: None for j in jobs},
                objective_eur=self.evaluate_plan(jobs, {j.asset_id: None for j in jobs})[0],
                no_recommendation={j.asset_id: (j.abstain_reason or "RUL abstained")
                                   for j in jobs if j.abstained},
                note="every asset abstained; nothing to optimise",
            )
        W = self.windows
        lines = sorted({j.line for j in active})
        n_x = len(active) * len(W)
        n_d = len(active)
        n_m = len(lines) * len(W)
        n = n_x + n_d + n_m

        def xi(i: int, w: int) -> int:
            return i * len(W) + w

        def di(i: int) -> int:
            return n_x + i

        def mi(l: int, w: int) -> int:
            return n_x + n_d + l * len(W) + w

        c = np.zeros(n)
        for i, job in enumerate(active):
            for w, win in enumerate(W):
                c[xi(i, w)] = (self.assignment_cost(job, win) if win.accepts(job) else 0.0)
            c[di(i)] = self.defer_cost(job)
        for l, _line in enumerate(lines):
            for w, win in enumerate(W):
                c[mi(l, w)] = self._downtime_rate(win)

        A: List[np.ndarray] = []
        lb: List[float] = []
        ub: List[float] = []

        # Exactly one choice per asset; ineligible windows forced to zero.
        for i, job in enumerate(active):
            row = np.zeros(n)
            for w, win in enumerate(W):
                row[xi(i, w)] = 1.0
            row[di(i)] = 1.0
            A.append(row)
            lb.append(1.0)
            ub.append(1.0)
            for w, win in enumerate(W):
                if not win.accepts(job):
                    r = np.zeros(n)
                    r[xi(i, w)] = 1.0
                    A.append(r)
                    lb.append(0.0)
                    ub.append(0.0)

        # Crew capacity per window.
        for w, win in enumerate(W):
            row = np.zeros(n)
            for i, job in enumerate(active):
                row[xi(i, w)] = job.job_hours * job.crew_required
            A.append(row)
            lb.append(-np.inf)
            ub.append(float(win.crew_hours))

        # Makespan lower bounds.
        for l, line in enumerate(lines):
            members = [i for i, j in enumerate(active) if j.line == line]
            for w, _win in enumerate(W):
                for i in members:
                    row = np.zeros(n)
                    row[mi(l, w)] = 1.0
                    row[xi(i, w)] = -active[i].job_hours
                    A.append(row)
                    lb.append(0.0)
                    ub.append(np.inf)
                row = np.zeros(n)
                row[mi(l, w)] = 1.0
                for i in members:
                    row[xi(i, w)] = -active[i].job_hours * active[i].crew_required / max(
                        1, self.crew_size)
                A.append(row)
                lb.append(0.0)
                ub.append(np.inf)

        # Parts budget.
        if self.parts_budget_eur is not None:
            row = np.zeros(n)
            for i, job in enumerate(active):
                for w in range(len(W)):
                    row[xi(i, w)] = self._part_eur(job)
            A.append(row)
            lb.append(-np.inf)
            ub.append(float(self.parts_budget_eur))

        integrality = np.zeros(n)
        integrality[: n_x + n_d] = 1
        bounds = Bounds(lb=np.zeros(n), ub=np.r_[np.ones(n_x + n_d), np.full(n_m, np.inf)])
        res = milp(c=c, constraints=LinearConstraint(np.vstack(A), lb, ub),
                   integrality=integrality, bounds=bounds)
        if not res.success or res.x is None:
            return ScheduleResult(solver="milp", infeasible=True,
                                  note=f"MILP did not solve: {res.message}")
        x = res.x
        assignment: Dict[str, Optional[str]] = {j.asset_id: None for j in jobs}
        for i, job in enumerate(active):
            for w, win in enumerate(W):
                if x[xi(i, w)] > 0.5:
                    assignment[job.asset_id] = win.name
        obj, comp, pw = self.evaluate_plan(jobs, assignment)
        return ScheduleResult(
            solver="milp",
            assignments=assignment,
            objective_eur=obj,
            components=comp,
            per_window=pw,
            no_recommendation={j.asset_id: (j.abstain_reason or "RUL abstained")
                               for j in jobs if j.abstained},
            note=f"HiGHS status: {res.message}",
        )

    # -- the entry point -----------------------------------------------------
    def solve(self, jobs: Sequence[AssetJob]) -> Dict[str, object]:
        """Solve with both methods, attach the baselines, report the gap."""
        ids = [j.asset_id for j in jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("asset ids must be unique")
        baselines = self._baselines(jobs)
        greedy = self.solve_greedy(jobs)
        greedy.baselines = baselines
        milp = self.solve_milp(jobs)
        best = greedy
        gap = None
        if milp is not None and not milp.infeasible:
            milp.baselines = baselines
            if milp.objective_eur <= greedy.objective_eur + 1e-9:
                best = milp
            denom = abs(milp.objective_eur) if milp.objective_eur else 1.0
            gap = (greedy.objective_eur - milp.objective_eur) / denom
        return {
            "chosen": best.as_dict(),
            "greedy": greedy.as_dict(),
            "milp": None if milp is None else milp.as_dict(),
            "greedy_gap_vs_milp": None if gap is None else round(gap, 6),
            "crew_size": self.crew_size,
            "parts_budget_eur": self.parts_budget_eur,
            "horizon_days": self.horizon_days,
            "windows": [
                {"name": w.name, "day": w.day, "line": w.line, "crew_hours": w.crew_hours}
                for w in self.windows
            ],
            "cost_fingerprint": self.cost.fingerprint(),
        }
