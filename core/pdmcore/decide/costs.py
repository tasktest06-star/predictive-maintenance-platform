"""The plant's cost model, and the audit trail that has to travel with it.

Relationship to :mod:`pdmcore.eval.metrics`
-------------------------------------------
``metrics.expected_cost`` already puts a currency figure on the
false-positive / false-negative trade-off, with two concrete defaults:
``cost_fp = 400`` (one technician inspecting a healthy machine) and
``cost_fn = 25000`` (an unplanned stop on a mid-size production line). This
module does **not** replace that. It decomposes those two numbers into the
quantities a plant actually negotiates -- downtime rates, MTTR, parts, labour,
secondary damage -- and then feeds them *back* into ``expected_cost`` via
:meth:`MaintenanceCostModel.detector_costs`, so the detector threshold and the
maintenance decision are priced by one model rather than two.

That decomposition immediately corrects the asymmetry. ``cost_fn = 25000`` is
the cost of an unplanned stop, but a missed detection does not cost a plant an
unplanned stop; it costs the *difference* between the unplanned stop it got and
the planned intervention it could have had. The repair, the part and the labour
were going to be paid either way. See :meth:`avoidable_eur` -- the number is
materially smaller than 25 000, which moves the cost-optimal detector threshold
in the direction of fewer alarms.

Why the provenance machinery
----------------------------
Every number below is an estimate unless a site has replaced it. A
recommendation derived from library defaults and a recommendation derived from
a site's own quoted downtime rate look identical in a work-order table, and
only one of them is a decision. So the model carries:

* a ``site`` it was configured for (``None`` means nobody has),
* a ``provenance`` tag per field (``estimate`` / ``site_quoted`` / ``measured``),
* an explicit acknowledgement list for defaults the site has chosen to accept,
* a :meth:`fingerprint` -- the SHA-256 of the canonical parameter set.

:meth:`require_configured` refuses when the site has not been set or when a
default is in use that nobody acknowledged. The fingerprint goes into every
recommendation so that when a work order is disputed eighteen months later the
exact assumption set can be recovered rather than reconstructed from memory.

Sources
-------
No figure here is measured. The defaults are chosen to be *consistent with the
rest of this repository* rather than sourced from a survey: the inspection cost
is ``metrics.expected_cost``'s own ``cost_fp``, and the unplanned event cost
derived from the downtime decomposition reproduces its ``cost_fn`` to within
10 % -- which is asserted as a test so the two cannot silently diverge.
``research/`` on this branch states each assumption, its plausible range and
what it would take to measure it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, Mapping, Optional, Tuple

__all__ = [
    "CostModelNotConfigured",
    "MaintenanceCostModel",
    "ProgrammeCost",
    "DEFAULT_PROVENANCE",
    "SITE_PROFILES",
]


class CostModelNotConfigured(RuntimeError):
    """Raised when a decision is requested against unconfigured cost parameters."""


#: Provenance of every default. ``estimate`` means exactly that: a plausible
#: figure for a mid-size European manufacturing plant, written down so it can be
#: argued with, not a measurement. Nothing here is tagged ``measured``, and
#: nothing should be until a site has produced the number.
DEFAULT_PROVENANCE: Dict[str, str] = {
    "unplanned_downtime_eur_per_h": "estimate",
    "planned_downtime_eur_per_h": "estimate",
    "mttr_unplanned_h": "estimate",
    "mttr_planned_h": "estimate",
    "spare_part_eur": "estimate",
    "expedite_premium_frac": "estimate",
    "labour_eur_per_h": "estimate",
    "crew_size": "estimate",
    "secondary_damage_eur": "estimate",
    "p_secondary_damage": "estimate",
    "inspection_eur": "repo_default:metrics.expected_cost.cost_fp",
    "p_opens_machine_on_false_alarm": "estimate",
    "design_life_days": "estimate",
    "sensor_capex_eur": "estimate",
    "install_eur": "estimate",
    "hardware_life_years": "estimate",
    "gateway_eur": "estimate",
    "assets_per_gateway": "estimate",
    "cloud_eur_per_asset_year": "estimate",
    "analyst_h_per_asset_year": "estimate",
}


@dataclass(frozen=True)
class ProgrammeCost:
    """What the monitoring programme itself costs, per asset per year."""

    hardware_eur: float
    install_eur: float
    gateway_eur: float
    cloud_eur: float
    analyst_eur: float

    @property
    def total_eur(self) -> float:
        return (self.hardware_eur + self.install_eur + self.gateway_eur
                + self.cloud_eur + self.analyst_eur)

    def as_dict(self) -> Dict[str, float]:
        d = {k: round(v, 2) for k, v in asdict(self).items()}
        d["total_eur"] = round(self.total_eur, 2)
        return d


@dataclass(frozen=True)
class MaintenanceCostModel:
    """Costs of the four things that can happen to a monitored asset.

    1. It fails in service            -> :meth:`unplanned_event_eur`
    2. It is repaired in a shutdown   -> :meth:`planned_event_eur`
    3. It is stopped outside one      -> :meth:`unscheduled_stop_eur`
    4. It was healthy and we looked   -> :meth:`false_alarm_eur`

    All four are needed. A cost model with only (1) and (2) makes
    condition-based maintenance look free, because the only way to lose money is
    to be late. (4) is what the ADI CN0549 branch measured the hard way: a cut
    IEPE cable was diagnosed as mechanical looseness at 0.97 confidence, which
    does not merely waste an inspection -- at that confidence the machine gets
    opened. ``p_opens_machine_on_false_alarm`` is the parameter that encodes
    "a confident wrong answer costs more than an unconfident one".
    """

    # --- downtime -----------------------------------------------------------
    #: Lost contribution margin per hour when the asset stops unexpectedly.
    #: Chosen, with ``mttr_unplanned_h``, so that
    #: :meth:`unplanned_event_eur` lands on ``metrics.expected_cost``'s
    #: ``cost_fn`` default of 25 000 -- the two are cross-checked by a test.
    unplanned_downtime_eur_per_h: float = 3400.0
    #: Marginal cost per hour of work done inside a shutdown the plant was
    #: taking anyway. Much smaller than the unplanned rate, and *not* zero:
    #: shutdown hours are contended, and an extra job either extends the window
    #: or displaces another one.
    planned_downtime_eur_per_h: float = 900.0
    #: Unplanned repairs take longer: diagnosis under pressure, parts chase,
    #: collateral damage found on strip-down.
    mttr_unplanned_h: float = 6.0
    mttr_planned_h: float = 4.0

    # --- parts and labour ---------------------------------------------------
    spare_part_eur: float = 800.0
    #: Premium paid on an expedited part: courier, out-of-hours, spot price.
    expedite_premium_frac: float = 0.60
    labour_eur_per_h: float = 75.0
    crew_size: int = 2

    # --- consequence of running a failed element ----------------------------
    #: Shaft, housing and seal damage from running a bearing past failure.
    secondary_damage_eur: float = 6500.0
    p_secondary_damage: float = 0.35

    # --- the cost of being wrong in the other direction ---------------------
    #: A fitter dispatched to a healthy machine: half a day plus travel. This is
    #: ``metrics.expected_cost``'s ``cost_fp`` and is kept numerically identical
    #: on purpose.
    inspection_eur: float = 400.0
    #: Probability that a false alarm results in the machine actually being
    #: opened rather than only inspected.
    p_opens_machine_on_false_alarm: float = 0.30

    # --- residual life thrown away by intervening early ---------------------
    #: Service life of the replaced element. Used to price the life discarded
    #: when a part with time left on it is changed.
    design_life_days: float = 1825.0

    # --- the programme itself -----------------------------------------------
    sensor_capex_eur: float = 180.0
    install_eur: float = 250.0
    hardware_life_years: float = 5.0
    gateway_eur: float = 450.0
    assets_per_gateway: int = 25
    cloud_eur_per_asset_year: float = 36.0
    analyst_h_per_asset_year: float = 1.5

    # --- audit --------------------------------------------------------------
    #: The site these numbers were quoted for. ``None`` means nobody configured
    #: them, and :meth:`require_configured` refuses.
    site: Optional[str] = None
    provenance: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_PROVENANCE))
    #: Fields the site has explicitly decided to accept at the library default.
    #: An empty tuple with ``site`` set is still a refusal: accepting a default
    #: has to be a recorded act, not a silence.
    acknowledged_defaults: Tuple[str, ...] = ()
    #: Free-text note carried into the audit record.
    note: str = ""

    # -- validation ----------------------------------------------------------
    def __post_init__(self) -> None:
        if self.unplanned_downtime_eur_per_h < self.planned_downtime_eur_per_h:
            raise ValueError(
                "unplanned downtime must cost at least as much per hour as planned "
                "downtime; if it does not, the asset is redundant and condition "
                "monitoring has nothing to sell -- model it as a spared asset instead"
            )
        if not 0.0 <= self.p_secondary_damage <= 1.0:
            raise ValueError("p_secondary_damage must be a probability")
        if not 0.0 <= self.p_opens_machine_on_false_alarm <= 1.0:
            raise ValueError("p_opens_machine_on_false_alarm must be a probability")
        if self.design_life_days <= 0:
            raise ValueError("design_life_days must be > 0")
        if self.assets_per_gateway < 1:
            raise ValueError("assets_per_gateway must be >= 1")
        if self.hardware_life_years <= 0:
            raise ValueError("hardware_life_years must be > 0")

    # -- the four event costs ------------------------------------------------
    def labour_eur(self, hours: float) -> float:
        return self.labour_eur_per_h * self.crew_size * float(hours)

    def unplanned_event_eur(self) -> float:
        """The asset failed in service."""
        return (
            self.unplanned_downtime_eur_per_h * self.mttr_unplanned_h
            + self.spare_part_eur * (1.0 + self.expedite_premium_frac)
            + self.labour_eur(self.mttr_unplanned_h)
            + self.p_secondary_damage * self.secondary_damage_eur
        )

    def planned_event_eur(self, downtime_h: Optional[float] = None) -> float:
        """The asset was repaired inside a shutdown window."""
        h = self.mttr_planned_h if downtime_h is None else float(downtime_h)
        return (
            self.planned_downtime_eur_per_h * h
            + self.spare_part_eur
            + self.labour_eur(h)
        )

    def unscheduled_stop_eur(self, downtime_h: Optional[float] = None) -> float:
        """The asset was stopped deliberately, but not in a shutdown window.

        Production is lost at the unplanned rate -- the line was not going to
        stop -- but the repair itself is orderly: planned MTTR, no expedite
        premium, no secondary damage. This is the action a twin makes available
        and a calendar does not, and it is the reason the decision set is not
        just {planned, failed}.
        """
        h = self.mttr_planned_h if downtime_h is None else float(downtime_h)
        return (
            self.unplanned_downtime_eur_per_h * h
            + self.spare_part_eur
            + self.labour_eur(h)
        )

    def residual_life_eur(self, days_remaining: float) -> float:
        """Value of the service life discarded by changing a part early.

        Linear in remaining life against :attr:`design_life_days`, priced on the
        part only -- the labour is spent whenever the job is done, so it is not
        life thrown away. Linear is the conservative choice: a hazard-weighted
        version would value the last days of life at less, which makes early
        intervention look cheaper and condition monitoring look better.
        """
        d = max(0.0, min(float(days_remaining), self.design_life_days))
        return self.spare_part_eur * d / self.design_life_days

    def false_alarm_eur(self, *, p_opens: Optional[float] = None) -> float:
        """A recommendation raised against a healthy asset.

        ``inspection`` always; with probability ``p_opens`` the machine is also
        opened, which costs a planned intervention *plus* the whole remaining
        life of a part that had all of it left.
        """
        p = self.p_opens_machine_on_false_alarm if p_opens is None else float(p_opens)
        return self.inspection_eur + p * (
            self.planned_event_eur() + self.residual_life_eur(self.design_life_days)
        )

    def avoidable_eur(self) -> float:
        """What converting one failure into one planned job is worth.

        The number a condition-monitoring programme actually sells, and it is
        **not** ``unplanned_event_eur``. The repair, the part and the labour are
        paid in both worlds; only the downtime difference, the expedite premium
        and the secondary damage are avoided.
        """
        return self.unplanned_event_eur() - self.planned_event_eur()

    # -- the programme -------------------------------------------------------
    def programme_cost(self) -> ProgrammeCost:
        """Annualised cost of monitoring one asset, excluding false alarms.

        False alarms are deliberately *not* folded in here: their number is a
        property of the detector and the fleet, not of the price list, and
        hiding them inside a fixed per-asset figure is how a programme's real
        cost disappears. :func:`pdmcore.decide.policy.programme_balance` adds
        them from a :class:`~pdmcore.decide.counts.CountedRate`.
        """
        return ProgrammeCost(
            hardware_eur=self.sensor_capex_eur / self.hardware_life_years,
            install_eur=self.install_eur / self.hardware_life_years,
            gateway_eur=self.gateway_eur / self.assets_per_gateway / self.hardware_life_years,
            cloud_eur=self.cloud_eur_per_asset_year,
            analyst_eur=self.analyst_h_per_asset_year * self.labour_eur_per_h,
        )

    # -- the bridge to the detector threshold --------------------------------
    def detector_costs(self) -> Tuple[float, float]:
        """``(cost_fp, cost_fn)`` for :func:`pdmcore.eval.metrics.expected_cost`.

        ``cost_fp`` is :meth:`false_alarm_eur`; ``cost_fn`` is
        :meth:`avoidable_eur`, not :meth:`unplanned_event_eur`. Passing the full
        unplanned cost as ``cost_fn`` over-states the asymmetry and pushes the
        cost-optimal threshold to a higher alarm rate than the plant's own
        numbers justify -- the error is in the generous direction for a
        prognostics vendor, which is exactly why it is worth naming.
        """
        return self.false_alarm_eur(), self.avoidable_eur()

    # -- audit ---------------------------------------------------------------
    def _numeric_dict(self) -> Dict[str, float]:
        skip = {"site", "provenance", "acknowledged_defaults", "note"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}

    def fingerprint(self) -> str:
        """SHA-256 over the canonical parameter set, including the site.

        Two recommendations with the same fingerprint were made under the same
        assumptions; two with different fingerprints were not, whatever their
        text says.
        """
        payload = {
            "site": self.site,
            "params": self._numeric_dict(),
            "provenance": dict(sorted(self.provenance.items())),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def unverified_defaults(self) -> Tuple[str, ...]:
        """Fields still at the library default, not site-quoted, not acknowledged."""
        library = MaintenanceCostModel()
        out = []
        for name, value in self._numeric_dict().items():
            if name in self.acknowledged_defaults:
                continue
            tag = self.provenance.get(name, "estimate")
            if tag in ("site_quoted", "measured"):
                continue
            if value == getattr(library, name):
                out.append(name)
        return tuple(sorted(out))

    def require_configured(self) -> None:
        """Refuse to be used for a recommendation unless a site owns these numbers.

        Two separate refusals, because they are different failures:

        * ``site is None`` -- nobody has taken responsibility for the numbers.
        * unverified defaults -- somebody has, but has not looked at all of them.

        A plant whose real downtime cost is 200 EUR/h and whose model says
        5 000 EUR/h will be told to intervene on assets where running to failure
        is correct, and the recommendation will look exactly as authoritative as
        a right one.
        """
        if not self.site:
            raise CostModelNotConfigured(
                "cost model has no site: refusing to make a recommendation. "
                "Set site= and quote or acknowledge each parameter; see "
                "twin/decisions/production/config/costs.example.json"
            )
        missing = self.unverified_defaults()
        if missing:
            raise CostModelNotConfigured(
                f"site {self.site!r}: {len(missing)} cost parameter(s) are library "
                f"defaults that have been neither quoted nor acknowledged: "
                f"{', '.join(missing)}. Add each to acknowledged_defaults with a "
                f"note, or quote the site's own figure."
            )

    def audit_record(self) -> Dict[str, object]:
        """Everything needed to re-derive a recommendation years later."""
        return {
            "fingerprint": self.fingerprint(),
            "site": self.site,
            "note": self.note,
            "params": {k: round(float(v), 6) for k, v in self._numeric_dict().items()},
            "provenance": dict(sorted(self.provenance.items())),
            "acknowledged_defaults": list(self.acknowledged_defaults),
            "unverified_defaults": list(self.unverified_defaults()),
            "derived": {
                "unplanned_event_eur": round(self.unplanned_event_eur(), 2),
                "planned_event_eur": round(self.planned_event_eur(), 2),
                "unscheduled_stop_eur": round(self.unscheduled_stop_eur(), 2),
                "false_alarm_eur": round(self.false_alarm_eur(), 2),
                "avoidable_eur": round(self.avoidable_eur(), 2),
                "programme_eur_per_asset_year": round(self.programme_cost().total_eur, 2),
            },
        }

    def for_site(self, site: str, **overrides) -> "MaintenanceCostModel":
        """Return a copy configured for ``site``, marking overrides site-quoted."""
        prov = dict(self.provenance)
        for k in overrides:
            if k in prov or k in self._numeric_dict():
                prov[k] = "site_quoted"
        return replace(self, site=site, provenance=prov, **overrides)

    def as_dict(self) -> Dict[str, object]:
        return self.audit_record()


#: Three site archetypes, used by the sensitivity analysis and by the tests.
#: They exist because the answer to "does the programme pay for itself" is
#: different for each, and a single default would hide that.
SITE_PROFILES: Dict[str, Dict[str, float]] = {
    # A single-train process line. Stopping the asset stops the plant.
    "critical_single_train": {
        "unplanned_downtime_eur_per_h": 11000.0,
        "planned_downtime_eur_per_h": 1800.0,
        "secondary_damage_eur": 14000.0,
    },
    # The default archetype: a mid-size line with some buffering.
    "mid_size_line": {},
    # A spared asset: a duty/standby pump pair. The unplanned rate is the cost
    # of running unspared until the repair, not the cost of lost production.
    "spared_duty_standby": {
        "unplanned_downtime_eur_per_h": 350.0,
        "planned_downtime_eur_per_h": 300.0,
        "secondary_damage_eur": 3000.0,
        "p_secondary_damage": 0.25,
    },
}
