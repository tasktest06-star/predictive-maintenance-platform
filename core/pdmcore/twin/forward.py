"""The composed forward model: twin spec + operating point -> predictions.

This is the interface the two sibling branches consume. It is deliberately a
pure function of ``(TwinSpec, OperatingPoint)`` with no state, no filtering and
no decision logic:

* ``twin/hybrid-residual-ukf`` wraps :meth:`MotorTwin.predict` as the
  measurement model of a state estimator and adds a learned residual term. The
  boundary is that **this module never holds state across calls** -- no
  previous-frame memory, no EWMA, no particle set. If a prediction depends on
  history, it belongs on that branch.
* ``twin/decisions-rul-scheduling`` consumes the residuals from
  :mod:`pdmcore.twin.residual` and the validity verdict from
  :mod:`pdmcore.twin.validity`, and decides what to do. The boundary is that
  **this module never returns a recommendation, a severity, a cost or a
  deadline.** It returns physical quantities and an honest statement about which
  of them it is allowed to produce.

The prediction registry
-----------------------
:data:`PREDICTIONS` is the catalogue of everything the twin can be asked for,
with -- for each one -- the minimum information tier that identifies it and
whether it needs the one measurement nobody makes (an impact test). It exists so
that "which predictions are identifiable at that tier" is a data structure that
code can gate on, rather than a paragraph in a design note that the code does
not read. :mod:`pdmcore.twin.validity` is the enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..dsp.bearing import fault_frequencies
from ..dsp.iso import iso_zone
from .electrical import (
    SteadyState,
    breakdown_torque,
    locked_rotor_current,
    slip_for_torque,
    steady_state,
)
from .params import OperatingPoint, TwinSpec
from .rotor import casing_accel_1x_g, response_shape
from .thermal import steady_temperature_c

__all__ = [
    "PredictionSpec",
    "PREDICTIONS",
    "TIERS",
    "TwinPrediction",
    "MotorTwin",
]

#: The three information tiers of the identification study, ordered.
TIERS: Tuple[str, ...] = ("nameplate", "commissioning", "sweep")


@dataclass(frozen=True)
class PredictionSpec:
    """Metadata for one thing the twin can predict.

    ``min_tier``
        The least informative tier at which this quantity is identified to a
        useful width. "Useful" is defined per quantity in
        ``twin/motor_physics/research/02_validity_domain.md`` and was written
        before the identification was run.

    ``needs_impact_test``
        ``True`` for every absolute vibration amplitude. The force-to-casing
        receptance is not identifiable from passive measurement at any tier, so
        these are gated on an explicit flag rather than on a tier.

    ``never_identifiable``
        ``True`` for quantities that no amount of field data fixes, because they
        are one factor of a product whose other factor is also unknown. These
        are in the registry *because* they are the ones a customer asks for.
    """

    key: str
    unit: str
    min_tier: str
    needs_impact_test: bool = False
    never_identifiable: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.min_tier not in TIERS:
            raise ValueError(f"min_tier must be one of {TIERS}, got {self.min_tier!r}")


def _p(*a, **kw) -> Tuple[str, PredictionSpec]:
    s = PredictionSpec(*a, **kw)
    return s.key, s


#: Everything the twin can be asked for. Read this table before reading anything
#: else in the branch: it is the answer to the question the branch was set.
PREDICTIONS: Dict[str, PredictionSpec] = dict(
    [
        # --- kinematics: exact, from geometry and speed ---------------------
        _p("bearing.bpfo_hz", "Hz", "nameplate",
           note="pure kinematics; the only error is the shaft-speed error"),
        _p("bearing.bpfi_hz", "Hz", "nameplate"),
        _p("bearing.bsf_hz", "Hz", "nameplate"),
        _p("bearing.ftf_hz", "Hz", "nameplate"),
        _p("electrical.line_2x_hz", "Hz", "nameplate",
           note="twice line frequency; needs no parameters at all"),
        _p("electrical.rotor_bar_sideband_hz", "Hz", "commissioning",
           note="(1 +- 2ks) f_line: needs the OPERATING slip, not the rated slip, "
                "and the operating slip needs a speed measurement good to ~0.01 Hz"),
        # --- electromechanical ---------------------------------------------
        _p("motor.slip", "-", "nameplate"),
        _p("motor.shaft_hz", "Hz", "nameplate"),
        _p("motor.stator_current_a", "A", "nameplate",
           note="constrained by the plate's own rated current at the rated point; "
                "extrapolates poorly to light load where the magnetising branch dominates"),
        _p("motor.shaft_torque_nm", "N.m", "nameplate"),
        _p("motor.efficiency", "-", "nameplate"),
        _p("motor.loss_w", "W", "nameplate"),
        _p("motor.breakdown_torque_nm", "N.m", "sweep",
           note="depends on the leakage SUM, which the plate sees, and on the "
                "leakage SPLIT and rotor skin effect, which it does not"),
        _p("motor.starting_current_a", "A", "sweep",
           note="s = 1 is where deep-bar skin effect is largest and where no "
                "nameplate quantity constrains the rotor resistance"),
        # --- thermal --------------------------------------------------------
        _p("thermal.rise_k", "K", "commissioning",
           note="needs a commissioning temperature to anchor R_th; the prediction "
                "is then a baseline, and its value is as a residual reference"),
        _p("thermal.steady_temperature_c", "degC", "commissioning"),
        # --- rotor ----------------------------------------------------------
        _p("rotor.critical_speed_hz", "Hz", "sweep",
           note="NOT identifiable at a fixed speed: see pdmcore.twin.rotor"),
        _p("rotor.resonance_proximity", "-", "sweep",
           note="running speed / critical speed. The number that decides whether a "
                "1x trend means unbalance or a softening mount"),
        _p("rotor.amplification", "-", "sweep"),
        # --- vibration amplitude: the gated block ---------------------------
        _p("vibration.accel_1x_g", "g", "sweep", needs_impact_test=True,
           note="linear in the path receptance, which needs an instrumented impact test"),
        _p("vibration.velocity_1x_mm_s", "mm/s", "sweep", needs_impact_test=True),
        _p("vibration.iso_zone", "-", "sweep", needs_impact_test=True,
           note="an ISO zone is a function of absolute velocity, so it inherits the "
                "receptance. A twin-predicted ISO zone without an impact test is the "
                "single most dangerous output this twin could produce"),
        _p("vibration.accel_1x_ratio", "-", "nameplate",
           note="ratio between two operating points. The receptance cancels, so this "
                "is available at every tier and is what trending actually needs"),
        _p("bearing.demod_band_hz", "Hz", "commissioning",
           note="the resonance FREQUENCIES are identifiable from a healthy record; "
                "only their gains are not"),
        # --- the things a customer asks for and cannot have -----------------
        _p("bearing.defect_force_n", "N", "sweep", never_identifiable=True,
           note="one factor of force x receptance; the product is the only observable"),
        _p("bearing.defect_size_mm", "mm", "sweep", never_identifiable=True,
           note="would need the force, which needs the receptance, plus contact "
                "mechanics this twin deliberately does not model"),
        _p("vibration.absolute_alarm_threshold_g", "g", "sweep", never_identifiable=True,
           note="an absolute threshold from a twin is an absolute amplitude by "
                "another name. Use the commissioning baseline in "
                "pdmcore.health.index, which is measured on the actual machine"),
    ]
)


@dataclass
class TwinPrediction:
    """What the forward model produced, and which twin produced it.

    ``twin_version`` is the content hash of the spec. It is on every prediction
    rather than looked up later, because the failure mode being designed against
    is a prediction that outlives its parameter set -- stored in a time series,
    read six weeks later, attributed to whatever the twin happens to be now.
    """

    values: Dict[str, float]
    twin_version: str
    twin_name: str
    tier: str
    operating_point: Dict[str, object]
    steady_state: Optional[SteadyState] = None
    notes: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        if key not in self.values:
            if key in PREDICTIONS:
                raise KeyError(
                    f"{key!r} is a known prediction but was not produced for this "
                    f"operating point. Check the validity verdict: "
                    f"pdmcore.twin.validity.check(spec, op, {key!r})"
                )
            raise KeyError(f"{key!r} is not a prediction this twin makes; see PREDICTIONS")
        return self.values[key]

    def as_dict(self) -> Dict[str, object]:
        return {
            "twin_name": self.twin_name,
            "twin_version": self.twin_version,
            "tier": self.tier,
            "operating_point": dict(self.operating_point),
            "values": {k: float(v) for k, v in self.values.items()},
            "notes": dict(self.notes),
        }


class MotorTwin:
    """A forward model bound to one identified parameter set.

    Stateless by design (see the module docstring). Construction is cheap;
    :meth:`predict` costs about 0.1 ms, dominated by one root find on the
    torque-slip curve.
    """

    def __init__(self, spec: TwinSpec):
        self.spec = spec
        self._version = spec.version       # cached: hashing is not free
        self._tier = str(spec.provenance.get("tier", "nameplate"))
        if self._tier not in TIERS:
            raise ValueError(
                f"twin spec provenance declares tier {self._tier!r}, which is not one "
                f"of {TIERS}. A twin with no declared tier cannot be gated, and an "
                "ungated twin is the failure mode this branch exists to prevent"
            )

    # -- identity ----------------------------------------------------------
    @property
    def version(self) -> str:
        return self._version

    @property
    def tier(self) -> str:
        return self._tier

    # -- the forward model -------------------------------------------------
    def operating_slip(self, op: OperatingPoint) -> float:
        """Slip at an operating point, from load if given, else from speed.

        Prefers the load fraction, because inverting the torque-slip curve is
        well conditioned in that direction. Going the other way -- slip from a
        measured speed -- is the badly conditioned direction and is why
        :func:`pdmcore.twin.identify.load_from_speed` exists as a separate,
        loudly documented function rather than being buried here.
        """
        plate = self.spec.nameplate
        torque = op.load_fraction * plate.rated_torque_nm
        if torque <= 0:
            # No load: slip is set by the drag torque alone. Use the smallest
            # slip the circuit supports rather than zero, which is singular.
            return 1e-5
        return slip_for_torque(
            self.spec.electrical,
            plate,
            torque,
            supply_hz=op.supply_hz,
            supply_voltage_v=op.supply_voltage_v,
        )

    def predict(self, op: OperatingPoint, *, include_gated: bool = False) -> TwinPrediction:
        """Every prediction the forward model can compute at ``op``.

        ``include_gated=False`` (the default) omits the predictions that need an
        impact test the spec does not have, and the ones that are never
        identifiable. **The omission is the point**: a caller that iterates over
        the returned values cannot accidentally consume an unidentifiable
        quantity, because it is not there. ``notes`` records why each one was
        left out, so the omission is visible rather than silent.

        Use :func:`pdmcore.twin.validity.check` for a single prediction with a
        reasoned verdict; this method is the bulk path.
        """
        spec = self.spec
        plate = spec.nameplate
        slip = self.operating_slip(op)
        ss = steady_state(
            spec.electrical, plate, slip,
            supply_hz=op.supply_hz, supply_voltage_v=op.supply_voltage_v,
        )
        shaft_hz = ss.shaft_hz
        f_line = op.supply_hz if op.supply_hz is not None else plate.line_hz
        ff = fault_frequencies(spec.bearing, shaft_hz)

        v: Dict[str, float] = {
            "bearing.bpfo_hz": ff.bpfo_hz,
            "bearing.bpfi_hz": ff.bpfi_hz,
            "bearing.bsf_hz": ff.bsf_hz,
            "bearing.ftf_hz": ff.ftf_hz,
            "electrical.line_2x_hz": 2.0 * f_line,
            "electrical.rotor_bar_sideband_hz": f_line * (1.0 - 2.0 * slip),
            "motor.slip": slip,
            "motor.shaft_hz": shaft_hz,
            "motor.stator_current_a": ss.stator_current_a,
            "motor.shaft_torque_nm": ss.shaft_torque_nm,
            "motor.efficiency": ss.efficiency,
            "motor.loss_w": ss.loss_w,
            "thermal.rise_k": spec.thermal.r_th_k_per_w * ss.loss_w,
            "thermal.steady_temperature_c": steady_temperature_c(
                spec.thermal, ss.loss_w, ambient_c=op.ambient_c
            ),
            "rotor.critical_speed_hz": spec.rotor.critical_speed_hz,
            "rotor.resonance_proximity": shaft_hz / spec.rotor.critical_speed_hz,
            "rotor.amplification": spec.rotor.amplification_at_resonance,
            # Gain-free: the ratio of the 1x response here to the response at the
            # rated speed. The receptance and the unbalance both cancel.
            "vibration.accel_1x_ratio": float(
                response_shape(shaft_hz / spec.rotor.critical_speed_hz, spec.rotor.damping_ratio)
                / response_shape(
                    plate.rated_shaft_hz / spec.rotor.critical_speed_hz, spec.rotor.damping_ratio
                )
            ),
        }
        if spec.path.modes:
            lo = min(spec.path.mode_freqs_hz)
            hi = max(spec.path.mode_freqs_hz)
            # The demodulation band the twin recommends is the strongest mode
            # plus/minus half its 3 dB width -- the band that collects the
            # impact energy without collecting shaft-order content.
            best = max(spec.path.modes, key=lambda m: m.gain_g_per_n)
            v["bearing.demod_band_hz"] = best.freq_hz
            v["bearing.demod_lo_hz"] = best.freq_hz * (1.0 - 1.0 / best.q)
            v["bearing.demod_hi_hz"] = best.freq_hz * (1.0 + 1.0 / best.q)
            v["bearing.path_mode_lo_hz"] = lo
            v["bearing.path_mode_hi_hz"] = hi

        notes: Dict[str, str] = {}

        def _tier_ok(key: str) -> bool:
            return TIERS.index(PREDICTIONS[key].min_tier) <= TIERS.index(self._tier)

        # --- tier-gated electrical extrapolations ---------------------------
        # These are real computations, not placeholders: the circuit does predict
        # breakdown torque and locked-rotor current. They are gated because at
        # the nameplate tier they depend on parameter combinations the nameplate
        # cannot see (the leakage split, and rotor skin effect at s = 1), so the
        # VALUE would be produced and would be wrong by an unknown factor.
        for key, fn in (
            ("motor.breakdown_torque_nm", lambda: float(
                breakdown_torque(spec.electrical, plate,
                                 supply_hz=op.supply_hz,
                                 supply_voltage_v=op.supply_voltage_v)[1])),
            ("motor.starting_current_a", lambda: float(
                locked_rotor_current(spec.electrical, plate))),
        ):
            if _tier_ok(key):
                v[key] = fn()
            else:
                notes[key] = (
                    f"omitted: needs tier {PREDICTIONS[key].min_tier!r}, twin is tier "
                    f"{self._tier!r}. {PREDICTIONS[key].note}"
                )

        # --- amplitude predictions: gated on the impact test ----------------
        for key, fn in (
            ("vibration.accel_1x_g", lambda: float(
                casing_accel_1x_g(spec.rotor, spec.path, shaft_hz))),
            # Velocity from acceleration at a single frequency: a = (2 pi f)^2 x
            # and v = 2 pi f x, so v = a / (2 pi f). 9.80665 converts g to m/s^2,
            # 1e3 to mm/s. This is an AMPLITUDE, not an RMS.
            ("vibration.velocity_1x_mm_s", lambda: float(
                casing_accel_1x_g(spec.rotor, spec.path, shaft_hz)
                * 9.80665 / (2.0 * np.pi * shaft_hz) * 1e3)),
        ):
            if not spec.path.impact_tested:
                notes[key] = (
                    "omitted: linear in the force-to-casing receptance, which needs an "
                    "instrumented impact test. path.impact_tested is False"
                )
            elif not _tier_ok(key):
                notes[key] = (
                    f"omitted: needs tier {PREDICTIONS[key].min_tier!r} to pin the "
                    f"critical speed, twin is tier {self._tier!r}"
                )
            elif not include_gated:
                notes[key] = "omitted: gated prediction, pass include_gated=True to receive it"
            else:
                v[key] = fn()

        # --- ISO zone: refused, with a named lower bound instead -------------
        iso_class = spec.validity.get("iso_class")
        if "vibration.velocity_1x_mm_s" not in v:
            notes["vibration.iso_zone"] = (
                "not produced: the 1x velocity amplitude it would be computed from was "
                "itself not produced"
            )
        elif not iso_class:
            notes["vibration.iso_zone"] = (
                "not produced: no ISO 20816 machine class in the twin's validity domain. "
                "The class depends on the support condition (rigid vs flexible), which is "
                "asset metadata reviewed at commissioning and is NOT inferable from the "
                "nameplate -- guessing it wrong shifts every zone boundary by ~1.6x "
                "(see pdmcore.dsp.iso.classify_machine)"
            )
        else:
            # The 1x line is one component of a broadband RMS, so the zone it
            # implies is a LOWER BOUND on the real zone. Named accordingly: a key
            # called "iso_zone" would be consumed as the zone.
            v["vibration.iso_zone_lower_bound_from_1x"] = float(
                "ABCD".index(iso_zone(v["vibration.velocity_1x_mm_s"] / np.sqrt(2.0), str(iso_class)))
            )
            notes["vibration.iso_zone"] = (
                "not produced. An ISO 20816 zone is a function of broadband 10-1000 Hz "
                "velocity RMS, which includes content this twin does not model: "
                "misalignment, looseness, blade pass, belt orders, structural response. "
                "vibration.iso_zone_lower_bound_from_1x is the zone implied by the 1x "
                "line alone and is a lower bound"
            )

        for key in ("bearing.defect_force_n", "bearing.defect_size_mm",
                    "vibration.absolute_alarm_threshold_g"):
            notes[key] = f"never produced: {PREDICTIONS[key].note}"

        return TwinPrediction(
            values=v,
            twin_version=self._version,
            twin_name=spec.name,
            tier=self._tier,
            operating_point=op.as_dict(),
            steady_state=ss,
            notes=notes,
        )
