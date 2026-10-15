"""Validity domain and the gates that enforce it.

**A twin that silently extrapolates is the central failure mode of this entire
product category.** It is not a bug that shows up in a test: the model happily
produces a number outside everything it was identified on, the number looks
like the numbers inside the domain, and nobody finds out until a work order is
disputed. So the refusal is structural here: :func:`check` is the only way to
obtain a prediction through :class:`GatedTwin`, and it returns a *verdict*, not
a clamped value.

Four independent gates, in the order they fire
----------------------------------------------
1. **Unknown prediction.** The key is not in
   :data:`pdmcore.twin.forward.PREDICTIONS`. Refused rather than returning
   ``None``, because ``None`` propagates.
2. **Never identifiable.** The quantity is one factor of an unobservable
   product. No tier and no amount of data unlocks it. Refused permanently.
3. **Tier.** The quantity needs more information than this twin was identified
   from, or needs an impact test the twin does not have.
4. **Operating point.** The point is outside the box the twin was identified
   over, or in a region the identification data never covered.

Why the operating-point gate is not just a box check
----------------------------------------------------
The naive implementation is ``lo <= x <= hi`` with ``lo`` and ``hi`` taken from
the identification data. That check **cannot fail on the identification data** --
by construction every identification point is inside its own min/max -- and
``docs/08_branch_catalogue.md`` tabulates six defects of exactly this shape. The
box is still necessary, so it is kept, but two things are added:

* The domain is declared with explicit margins (:attr:`ValidityDomain.speed_hz`
  etc.) that are set from the *physics* of where the model is valid, not from
  the data extent. For the speed range that is "between 1.0 and 1.6 times the
  rated slip region where the torque-slip curve is linear"; outside it the
  circuit is still solvable and still wrong.
* :attr:`ValidityDomain.covered_cells` records which (speed, load) cells the
  identification data actually visited, mirroring
  :class:`pdmcore.health.index.OperatingBaseline`. A point inside the box but in
  an unvisited cell gets :data:`Verdict.NOT_IDENTIFIED_HERE`, which is a
  different answer from "out of range" and needs a different response:
  out-of-range means never ask again, uncovered means collect a commissioning
  record at that operating point.

The negative controls that make these gates non-vacuous are in
``core/tests/test_twin_gates.py``: each gate has a test that it *fires* on a
point chosen to be outside, and a test that it *passes* on a held-out point that
was never in the identification set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .forward import PREDICTIONS, TIERS, MotorTwin, TwinPrediction
from .params import OperatingPoint, TwinSpec

__all__ = [
    "Verdict",
    "ValidityDomain",
    "ValidityVerdict",
    "OutsideValidityDomain",
    "UnidentifiablePrediction",
    "OperatingPointNotIdentified",
    "GatedTwin",
    "domain_of",
    "check",
]


class Verdict(str, Enum):
    """Outcome of a validity check."""

    OK = "ok"
    UNKNOWN_PREDICTION = "unknown_prediction"
    NEVER_IDENTIFIABLE = "never_identifiable"
    TIER_TOO_LOW = "tier_too_low"
    NEEDS_IMPACT_TEST = "needs_impact_test"
    OUT_OF_RANGE = "out_of_range"
    NOT_IDENTIFIED_HERE = "not_identified_here"
    VERSION_MISMATCH = "version_mismatch"


class OutsideValidityDomain(ValueError):
    """The operating point is outside the domain the twin was identified over."""


class UnidentifiablePrediction(ValueError):
    """The prediction is not identified at this twin's information tier."""


class OperatingPointNotIdentified(ValueError):
    """Inside the box, but the identification data never visited this cell."""


@dataclass(frozen=True)
class ValidityDomain:
    """Where the twin is allowed to speak, declared before it was identified.

    ``speed_hz``, ``load_fraction``, ``ambient_c``, ``supply_hz``
        Hard bounds. Outside them the forward model is still solvable, which is
        exactly the danger.

    ``covered_cells``
        ``(speed_bin, load_bin)`` pairs the identification data visited, on the
        grid defined by ``speed_bin_hz`` and ``load_bin``. Empty means "coverage
        not recorded", and in that case the coverage gate is **disabled and says
        so** rather than passing silently -- a coverage check against an empty
        set would pass everything, which is the vacuous-check failure again.

    ``iso_class``
        ISO 20816 machine class, if it was reviewed at commissioning. Absent
        means the twin refuses to express anything as an ISO zone.
    """

    speed_hz: Tuple[float, float]
    load_fraction: Tuple[float, float] = (0.25, 1.15)
    ambient_c: Tuple[float, float] = (-10.0, 50.0)
    supply_hz: Tuple[float, float] = (45.0, 65.0)
    speed_bin_hz: float = 1.0
    load_bin: float = 0.25
    covered_cells: Tuple[Tuple[int, int], ...] = ()
    iso_class: Optional[str] = None
    rationale: str = ""

    def __post_init__(self) -> None:
        for name in ("speed_hz", "load_fraction", "ambient_c", "supply_hz"):
            lo, hi = getattr(self, name)
            if not lo < hi:
                raise ValueError(f"{name} bounds must satisfy lo < hi, got ({lo}, {hi})")
        if self.speed_bin_hz <= 0 or self.load_bin <= 0:
            raise ValueError("bin widths must be positive")

    def cell(self, op: OperatingPoint) -> Tuple[int, int]:
        return (
            int(np.floor(op.shaft_hz / self.speed_bin_hz)),
            int(np.floor(op.load_fraction / self.load_bin)),
        )

    @property
    def coverage_recorded(self) -> bool:
        return len(self.covered_cells) > 0

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["covered_cells"] = [list(c) for c in self.covered_cells]
        for k in ("speed_hz", "load_fraction", "ambient_c", "supply_hz"):
            d[k] = list(getattr(self, k))
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ValidityDomain":
        kw = dict(d)
        for k in ("speed_hz", "load_fraction", "ambient_c", "supply_hz"):
            if k in kw:
                kw[k] = tuple(kw[k])  # type: ignore[assignment,arg-type]
        if "covered_cells" in kw:
            kw["covered_cells"] = tuple(tuple(int(x) for x in c) for c in kw["covered_cells"])  # type: ignore
        return cls(**kw)  # type: ignore[arg-type]


def domain_of(spec: TwinSpec) -> ValidityDomain:
    """Rehydrate the validity domain from a :class:`TwinSpec`.

    Raises if the spec has none. There is no default domain: a twin whose author
    did not state where it is valid is a twin that will be used everywhere.
    """
    if not spec.validity:
        raise ValueError(
            f"twin {spec.name!r} (version {spec.version}) carries no validity domain. "
            "Refusing to synthesise one -- a default domain is an unbounded domain with "
            "extra steps. Identify the twin with pdmcore.twin.identify, which writes the "
            "domain it was identified over"
        )
    return ValidityDomain.from_dict(dict(spec.validity))


@dataclass
class ValidityVerdict:
    """Why a prediction was or was not allowed, with enough detail to act on."""

    key: str
    verdict: Verdict
    allowed: bool
    reason: str
    twin_version: str
    tier: str
    detail: Dict[str, object] = field(default_factory=dict)

    def raise_for_verdict(self) -> None:
        """Raise the exception type that matches the verdict."""
        if self.allowed:
            return
        if self.verdict in (Verdict.NEVER_IDENTIFIABLE, Verdict.TIER_TOO_LOW,
                            Verdict.NEEDS_IMPACT_TEST, Verdict.UNKNOWN_PREDICTION):
            raise UnidentifiablePrediction(self.reason)
        if self.verdict is Verdict.NOT_IDENTIFIED_HERE:
            raise OperatingPointNotIdentified(self.reason)
        raise OutsideValidityDomain(self.reason)

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "verdict": self.verdict.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "twin_version": self.twin_version,
            "tier": self.tier,
            "detail": dict(self.detail),
        }


def check(spec: TwinSpec, op: OperatingPoint, key: str) -> ValidityVerdict:
    """Decide whether ``spec`` may predict ``key`` at ``op``.

    Pure: no side effects, no prediction computed. Call it before predicting, or
    use :class:`GatedTwin` which does both in the right order.
    """
    tier = str(spec.provenance.get("tier", "nameplate"))
    ver = spec.version

    def verdict(v: Verdict, reason: str, **detail) -> ValidityVerdict:
        return ValidityVerdict(
            key=key, verdict=v, allowed=(v is Verdict.OK), reason=reason,
            twin_version=ver, tier=tier, detail=detail,
        )

    # 1. unknown key
    sp = PREDICTIONS.get(key)
    if sp is None:
        near = sorted(k for k in PREDICTIONS if k.split(".")[0] == key.split(".")[0])
        return verdict(
            Verdict.UNKNOWN_PREDICTION,
            f"{key!r} is not a prediction this twin makes. "
            + (f"Predictions in the {key.split('.')[0]!r} group: {near}" if near
               else f"Known groups: {sorted({k.split('.')[0] for k in PREDICTIONS})}"),
            candidates=near,
        )

    # 2. permanently unidentifiable
    if sp.never_identifiable:
        return verdict(
            Verdict.NEVER_IDENTIFIABLE,
            f"{key!r} is not identifiable at any information tier: {sp.note}. "
            "This is a property of the physics, not of the data volume -- collecting "
            "more records does not change it.",
            note=sp.note,
        )

    # 3. tier and the impact test
    if sp.needs_impact_test and not spec.path.impact_tested:
        return verdict(
            Verdict.NEEDS_IMPACT_TEST,
            f"{key!r} is linear in the force-to-casing receptance, which is not "
            "identifiable from passive vibration data. It needs an instrumented impact "
            "test on the bearing housing with the accelerometer in place (a 2-channel "
            "measurement, ~1 h per machine). Until then the twin can report RATIOS of "
            "this quantity between operating points, which are receptance-free: see "
            "'vibration.accel_1x_ratio'.",
            substitute="vibration.accel_1x_ratio",
        )
    if TIERS.index(sp.min_tier) > TIERS.index(tier):
        return verdict(
            Verdict.TIER_TOO_LOW,
            f"{key!r} needs information tier {sp.min_tier!r}; this twin was identified "
            f"at tier {tier!r}. {sp.note}",
            required_tier=sp.min_tier, have_tier=tier,
        )

    # 4. operating point
    dom = domain_of(spec)
    for name, value, bounds in (
        ("shaft_hz", op.shaft_hz, dom.speed_hz),
        ("load_fraction", op.load_fraction, dom.load_fraction),
        ("ambient_c", op.ambient_c, dom.ambient_c),
        ("supply_hz", op.supply_hz if op.supply_hz is not None else spec.nameplate.line_hz,
         dom.supply_hz),
    ):
        lo, hi = bounds
        if not lo <= value <= hi:
            return verdict(
                Verdict.OUT_OF_RANGE,
                f"{name} = {value:g} is outside the twin's validity domain "
                f"[{lo:g}, {hi:g}]. The forward model would still return a number; it "
                "would be an extrapolation with no identified basis. "
                f"Domain rationale: {dom.rationale or '(none recorded)'}",
                field=name, value=float(value), bounds=[float(lo), float(hi)],
            )

    if dom.coverage_recorded:
        cell = dom.cell(op)
        if cell not in set(dom.covered_cells):
            return verdict(
                Verdict.NOT_IDENTIFIED_HERE,
                f"the operating point is inside the validity box but the identification "
                f"data never visited cell {cell} "
                f"(speed bin {dom.speed_bin_hz:g} Hz, load bin {dom.load_bin:g}). "
                "This is recoverable: collect a commissioning record at this operating "
                "point and re-identify. It is reported separately from 'out of range' "
                "because the actions differ.",
                cell=list(cell), n_covered=len(dom.covered_cells),
            )
    else:
        # Not an error, but it must be visible: an empty coverage set would make
        # the coverage gate pass everything.
        return verdict(
            Verdict.OK,
            "allowed; NOTE: the twin records no operating-point coverage, so the "
            "coverage gate is disabled for it. Only the hard bounds were checked.",
            coverage_gate="disabled",
        )

    return verdict(Verdict.OK, "allowed")


class GatedTwin:
    """A :class:`MotorTwin` that cannot be made to extrapolate.

    The only public way to get a number out is :meth:`predict`, which runs
    :func:`check` first and raises on refusal. There is no ``force=True``, no
    ``clamp=True`` and no ``default=``: each of those is a documented way that a
    twin ends up extrapolating in production with a plausible-looking number.

    :meth:`explain` gives the verdict without raising, for a dashboard or a
    sibling branch that wants to decide for itself -- but it returns the verdict
    object, never a value.
    """

    def __init__(self, spec: TwinSpec):
        self.spec = spec
        self.twin = MotorTwin(spec)

    @property
    def version(self) -> str:
        return self.twin.version

    def explain(self, op: OperatingPoint, key: str) -> ValidityVerdict:
        return check(self.spec, op, key)

    def predict(self, op: OperatingPoint, key: str) -> float:
        """One prediction, or an exception. Never a clamped or default value."""
        v = check(self.spec, op, key)
        v.raise_for_verdict()
        pred = self.twin.predict(op, include_gated=True)
        if key not in pred.values:
            # The gate said yes and the forward model produced nothing: that is an
            # internal inconsistency between the registry and the model, not a
            # user error, and it must not degrade into a None.
            raise AssertionError(
                f"validity gate allowed {key!r} but the forward model did not produce "
                f"it. Gate and registry have drifted. Note from the model: "
                f"{pred.notes.get(key, '(none)')}"
            )
        return float(pred.values[key])

    def predict_all(self, op: OperatingPoint) -> Tuple[TwinPrediction, List[ValidityVerdict]]:
        """Every allowed prediction, plus the verdicts for every refused one.

        Returns both halves deliberately. A caller that logs only the values has
        no record of what was refused; a caller that logs both can show an
        operator "the twin predicted these 14 quantities and refused these 5, for
        these reasons", which is the output this product category needs and
        almost never has.
        """
        pred = self.twin.predict(op, include_gated=True)
        refused: List[ValidityVerdict] = []
        allowed: Dict[str, float] = {}
        for key in PREDICTIONS:
            v = check(self.spec, op, key)
            if v.allowed and key in pred.values:
                allowed[key] = float(pred.values[key])
            elif v.allowed:
                refused.append(
                    ValidityVerdict(
                        key=key, verdict=Verdict.OK, allowed=False,
                        reason="gate allowed it but the forward model produced no value: "
                               + pred.notes.get(key, "(no note)"),
                        twin_version=self.version, tier=self.twin.tier,
                    )
                )
            else:
                refused.append(v)
        # Derived keys the registry does not enumerate (band edges, bounds) are
        # passed through: they are annotations on allowed predictions, not
        # predictions in their own right.
        for k, val in pred.values.items():
            if k not in PREDICTIONS:
                allowed[k] = float(val)
        out = TwinPrediction(
            values=allowed,
            twin_version=self.version,
            twin_name=self.spec.name,
            tier=self.twin.tier,
            operating_point=op.as_dict(),
            steady_state=pred.steady_state,
            notes=pred.notes,
        )
        return out, refused
