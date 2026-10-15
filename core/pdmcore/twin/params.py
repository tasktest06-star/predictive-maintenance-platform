"""Parameter objects for the motor twin, and the information a plant has.

Everything in this module is a frozen dataclass with SI units in the field
names, because a digital twin's parameter set is the artefact that outlives the
code: it is serialised, versioned, shipped to an edge node and used months later
to justify a work order. If the units are implicit, the twin is unfalsifiable.

Three groups of objects, in order of how much a plant actually knows:

:class:`Nameplate`
    What is stamped on the motor. Available for free on every motor in every
    plant, and the *only* thing available on most of them. Note the
    ``speed_rounding_rpm`` field: a nameplate speed is a rounded number, and
    because slip is a small difference of two large numbers, that rounding
    dominates the uncertainty of the rotor resistance. See
    :meth:`Nameplate.slip_bounds`.

:class:`ElectricalParams`, :class:`RotorParams`, :class:`PathParams`,
:class:`ThermalParams`, :class:`DefectParams`
    The twin's actual parameters. These are what identification estimates. They
    are deliberately split by subsystem because the identifiability differs by an
    order of magnitude between them: the electrical parameters are constrained by
    nameplate arithmetic, the rotor parameters need a speed sweep, and the
    transfer path needs an instrumented impact test that no condition-monitoring
    deployment performs.

:class:`TwinSpec`
    All of the above plus the bearing geometry, the identification provenance,
    the uncertainty and the validity domain, with a content-addressed
    :attr:`TwinSpec.version`. A prediction can always be traced back to the
    parameter set that produced it; see :mod:`pdmcore.twin.validity`.

Sign and reference conventions
------------------------------
* Impedances are **per phase, referred to the stator, star (Y) equivalent**, at
  the rated supply frequency. A delta-connected machine is converted on entry
  (:meth:`Nameplate.phase_voltage_v`), because mixing the two is the classic way
  to be wrong by exactly three.
* Reactances are stored at ``Nameplate.line_hz`` and scaled linearly with the
  actual supply frequency by :mod:`pdmcore.twin.electrical`. Inductances would
  avoid the scaling but nobody publishes inductances for induction motors.
* Slip is positive for motoring: ``s = (n_sync - n) / n_sync``.
* Vibration amplitudes are in **g**, matching ``pdmcore.features`` and the wire
  format. Forces are in newtons; the transfer path carries the g/N.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite as _isfinite
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..dsp.bearing import BEARING_CATALOG, BearingGeometry

__all__ = [
    "Nameplate",
    "ElectricalParams",
    "RotorParams",
    "PathMode",
    "PathParams",
    "ThermalParams",
    "DefectParams",
    "OperatingPoint",
    "TwinSpec",
    "ParameterInterval",
    "IE_EFFICIENCY_4POLE_50HZ",
    "ie_efficiency_floor",
]


# ---------------------------------------------------------------------------
# Nameplate
# ---------------------------------------------------------------------------
#: IEC 60034-30-1 minimum efficiencies, 4-pole 50 Hz, as percentages, for the
#: rated-power points that bracket this platform's asset classes.
#:
#: PROVENANCE: transcribed from the IEC 60034-30-1:2014 Table 2 values in common
#: circulation. **Not verified against the printed standard in this
#: environment.** Two things make that tolerable:
#:
#: 1. The standard defines these as *minima*, not measurements. A compliant IE3
#:    motor has efficiency >= the table value, typically 0-1.5 points above. So
#:    the twin uses them as a **one-sided bound**, never as an observation
#:    (:func:`ie_efficiency_floor`), and a transcription error of a few tenths of
#:    a point is small against the width of the bound itself.
#: 2. ``twin/motor_physics/prototype/01_catalogue_check.py`` asserts that every
#:    reference motor's published equivalent circuit produces an efficiency
#:    inside the plausible band for its power rating. A table error large enough
#:    to matter would show up there.
IE_EFFICIENCY_4POLE_50HZ: Dict[float, Dict[str, float]] = {
    2.2: {"IE1": 81.0, "IE2": 84.3, "IE3": 86.7, "IE4": 89.1},
    7.5: {"IE1": 86.0, "IE2": 88.7, "IE3": 90.4, "IE4": 92.0},
    18.5: {"IE1": 89.4, "IE2": 91.2, "IE3": 92.6, "IE4": 94.0},
    37.0: {"IE1": 91.0, "IE2": 92.7, "IE3": 93.9, "IE4": 95.0},
    75.0: {"IE1": 92.7, "IE2": 94.0, "IE3": 95.0, "IE4": 95.9},
    132.0: {"IE1": 93.7, "IE2": 94.8, "IE3": 95.6, "IE4": 96.4},
    375.0: {"IE1": 94.6, "IE2": 95.6, "IE3": 96.3, "IE4": 96.9},
}


def ie_efficiency_floor(power_kw: float, ie_class: str = "IE3") -> float:
    """Minimum efficiency (fraction, not percent) for a power and IE class.

    Log-interpolated between the tabulated power points, because the IEC curve is
    close to linear in ``log P``. Returns a **lower bound**: a motor that carries
    an IE3 label is guaranteed to be at least this efficient and is usually
    better, which is why the identification treats it as an inequality
    constraint rather than a measurement. Treating a class floor as a measured
    efficiency biases every loss-derived parameter in the same direction.
    """
    if power_kw <= 0:
        raise ValueError("power_kw must be positive")
    key = ie_class.upper()
    powers = np.array(sorted(IE_EFFICIENCY_4POLE_50HZ))
    try:
        vals = np.array([IE_EFFICIENCY_4POLE_50HZ[float(p)][key] for p in powers])
    except KeyError as exc:
        classes = sorted(next(iter(IE_EFFICIENCY_4POLE_50HZ.values())))
        raise ValueError(f"unknown IE class {ie_class!r}; expected one of {classes}") from exc
    eta_pct = float(np.interp(np.log(power_kw), np.log(powers), vals))
    return eta_pct / 100.0


@dataclass(frozen=True)
class Nameplate:
    """What is stamped on the motor, with the precision it is stamped at.

    This is tier 1 of the identification study: the information a plant has for
    free on every asset. Three fields carry the *precision* of the plate rather
    than a value, and they turn out to matter more than anything else:

    ``speed_rounding_rpm``
        Nameplate speed is rounded. 1780 rpm on a 4-pole 60 Hz motor means
        ``s = 0.0111``, but if the plate rounds to the nearest 5 rpm the true
        slip is anywhere in ``[0.0097, 0.0125]`` -- +-12.5 % -- and rotor
        resistance is very nearly proportional to rated slip. Rounding to the
        nearest 10 rpm doubles that. :meth:`slip_bounds` returns the interval.

    ``current_rounding_pct``
        Rated current is printed to 3 significant figures at best, often 2.

    ``efficiency_is_class_floor``
        ``True`` (the default) when the only efficiency information is an IE
        class label, so efficiency is known as a one-sided bound.
    """

    name: str
    rated_power_kw: float
    rated_voltage_v: float          # line-to-line RMS
    rated_current_a: float          # line current RMS
    rated_speed_rpm: float
    line_hz: float
    pole_pairs: int
    connection: str = "Y"           # "Y" or "D"
    ie_class: str = "IE3"
    rated_power_factor: Optional[float] = None
    rated_efficiency: Optional[float] = None       # fraction, if actually measured
    efficiency_is_class_floor: bool = True
    service_factor: float = 1.15
    insulation_class: str = "F"
    cooling: str = "TEFC"
    locked_rotor_current_pu: Optional[float] = None   # NEMA code letter, if present
    speed_rounding_rpm: float = 5.0
    current_rounding_pct: float = 2.5

    def __post_init__(self) -> None:
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")
        if self.rated_voltage_v <= 0 or self.rated_current_a <= 0:
            raise ValueError("rated voltage and current must be positive")
        if self.pole_pairs < 1:
            raise ValueError("pole_pairs must be >= 1")
        if self.line_hz <= 0:
            raise ValueError("line_hz must be positive")
        if self.connection not in ("Y", "D"):
            raise ValueError("connection must be 'Y' or 'D'")
        if not 0 < self.rated_speed_rpm < self.sync_rpm:
            raise ValueError(
                f"rated_speed_rpm {self.rated_speed_rpm} must be in (0, {self.sync_rpm}); "
                "a nameplate speed at or above synchronous means the pole count is wrong"
            )

    # -- geometry of the supply -------------------------------------------
    @property
    def sync_rpm(self) -> float:
        return 60.0 * self.line_hz / self.pole_pairs

    @property
    def sync_hz(self) -> float:
        """Synchronous *mechanical* frequency in Hz."""
        return self.line_hz / self.pole_pairs

    @property
    def phase_voltage_v(self) -> float:
        """Per-phase RMS voltage of the star equivalent circuit."""
        return self.rated_voltage_v / np.sqrt(3.0) if self.connection == "Y" else self.rated_voltage_v

    @property
    def phase_current_a(self) -> float:
        """Per-phase RMS current of the star equivalent circuit.

        For a delta machine the published (line) current is ``sqrt(3)`` times the
        winding current; the star equivalent carries the line current, so this is
        the line current in both cases. The factor lives in
        :attr:`phase_voltage_v` instead. Stated explicitly because getting it
        wrong scales every resistance by three.
        """
        return self.rated_current_a

    @property
    def rated_slip(self) -> float:
        return float((self.sync_rpm - self.rated_speed_rpm) / self.sync_rpm)

    @property
    def rated_shaft_hz(self) -> float:
        return self.rated_speed_rpm / 60.0

    @property
    def rated_torque_nm(self) -> float:
        return float(self.rated_power_kw * 1e3 / (2.0 * np.pi * self.rated_shaft_hz))

    @property
    def rated_apparent_va(self) -> float:
        return float(np.sqrt(3.0) * self.rated_voltage_v * self.rated_current_a)

    def slip_bounds(self) -> Tuple[float, float]:
        """``(s_lo, s_hi)`` implied by nameplate speed rounding.

        The single most under-appreciated number in motor-twin identification.
        Rated slip is ``(n_sync - n) / n_sync``; the numerator is a difference of
        two numbers that are ~100x larger than it is, so the plate's rounding is
        amplified by ``n_sync / (n_sync - n)``, which is 90 for a typical 4-pole
        motor. At the default 5 rpm rounding this is +-12.5 % on slip.
        """
        half = 0.5 * self.speed_rounding_rpm
        hi = (self.sync_rpm - (self.rated_speed_rpm - half)) / self.sync_rpm
        lo = (self.sync_rpm - (self.rated_speed_rpm + half)) / self.sync_rpm
        return float(max(1e-6, lo)), float(hi)

    def current_bounds(self) -> Tuple[float, float]:
        f = self.current_rounding_pct / 100.0
        return float(self.rated_current_a * (1.0 - f)), float(self.rated_current_a * (1.0 + f))

    def efficiency_bounds(self) -> Tuple[float, float]:
        """``(eta_lo, eta_hi)`` from whatever efficiency information exists.

        If a measured efficiency is on the plate it is taken as +-0.5 point. If
        only an IE class is known, the floor is a hard lower bound and the upper
        bound is the floor plus 1.5 points, which is where compliant motors
        actually sit -- a manufacturer that exceeded the next class by much would
        label it as the next class.
        """
        if self.rated_efficiency is not None and not self.efficiency_is_class_floor:
            e = float(self.rated_efficiency)
            return max(0.0, e - 0.005), min(0.999, e + 0.005)
        floor = ie_efficiency_floor(self.rated_power_kw, self.ie_class)
        return float(floor), float(min(0.995, floor + 0.015))

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Subsystem parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ElectricalParams:
    """Induction-motor equivalent circuit, per phase, star, at rated frequency.

    The exact T-circuit, not the Thevenin approximation: the approximation moves
    the magnetising branch ahead of the stator impedance, which is accurate for
    torque near rated slip but wrong by several percent in *current* -- and
    current is one of the few nameplate observations there is, so the
    approximation would corrupt exactly the quantity being matched.

    ``r_core_ohm`` is the shunt core-loss resistance. It is included because
    without it the only loss mechanisms are copper losses, and the identification
    then has to put the iron loss somewhere -- usually into ``rs``, biasing it
    high by 20-40 % with no warning.
    """

    rs_ohm: float
    rr_ohm: float
    xls_ohm: float
    xlr_ohm: float
    xm_ohm: float
    r_core_ohm: float = float("inf")
    friction_loss_w: float = 0.0          # constant component
    windage_coeff_w_per_hz2: float = 0.0  # quadratic in shaft Hz

    def __post_init__(self) -> None:
        for name in ("rs_ohm", "rr_ohm", "xls_ohm", "xlr_ohm", "xm_ohm"):
            v = getattr(self, name)
            if not _isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive, got {v}")
        if self.r_core_ohm <= 0:
            raise ValueError("r_core_ohm must be positive (use inf to disable core loss)")
        if self.friction_loss_w < 0 or self.windage_coeff_w_per_hz2 < 0:
            raise ValueError("loss coefficients must be non-negative")

    @property
    def x_leakage_sum_ohm(self) -> float:
        """``xls + xlr``. The combination that nameplate data can actually see.

        The split between stator and rotor leakage is not identifiable from any
        terminal measurement at a single frequency -- see
        :mod:`pdmcore.twin.identify`. The universal convention ``xls = xlr`` is
        an assumption, not a result, and this property exists so that code can
        depend on the identifiable quantity instead.
        """
        return float(self.xls_ohm + self.xlr_ohm)

    @property
    def leakage_split(self) -> float:
        """``xls / (xls + xlr)``; 0.5 is the conventional assumption."""
        return float(self.xls_ohm / self.x_leakage_sum_ohm)

    def mechanical_loss_w(self, shaft_hz: float) -> float:
        return float(self.friction_loss_w + self.windage_coeff_w_per_hz2 * shaft_hz**2)

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        if not np.isfinite(d["r_core_ohm"]):
            d["r_core_ohm"] = None  # type: ignore[assignment]
        return d


@dataclass(frozen=True)
class RotorParams:
    """Lumped rotor-bearing system: one Jeffcott mode per radial direction.

    The rotor is modelled as a single mass on an isotropic flexible support with
    viscous damping, excited by residual unbalance. That is two degrees of
    freedom collapsed into one complex response, which is the least model that
    can represent the thing that actually changes interpretation: **the first
    critical speed, and how close the running speed is to it.**

    Why that matters more than the amplitude. Near a resonance the 1x response
    is amplified by ``1 / (2 zeta)`` -- 10 to 25x for a machine structure -- and
    is extremely sensitive to small stiffness changes. A machine running at
    0.8x its critical speed will show a 1x rise of tens of percent from a mount
    bolt relaxing, with no change in unbalance at all. A machine at 0.3x will
    not. The same 1x trend therefore means different things on the two machines,
    and the twin cannot tell them apart unless the critical speed is identified.

    ``unbalance_kgm`` is ``m_u * e``, the unbalance moment. Only the **product**
    with the path gain is observable; see :class:`PathParams`.
    """

    inertia_kgm2: float
    rotor_mass_kg: float
    support_stiffness_n_per_m: float
    damping_ratio: float
    unbalance_kgm: float = 0.0

    def __post_init__(self) -> None:
        if self.inertia_kgm2 <= 0 or self.rotor_mass_kg <= 0:
            raise ValueError("inertia and rotor mass must be positive")
        if self.support_stiffness_n_per_m <= 0:
            raise ValueError("support stiffness must be positive")
        if not 0 < self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be in (0, 1); a rotor support is underdamped")
        if self.unbalance_kgm < 0:
            raise ValueError("unbalance_kgm must be non-negative")

    @property
    def critical_speed_hz(self) -> float:
        """First critical speed (undamped natural frequency) in Hz."""
        return float(np.sqrt(self.support_stiffness_n_per_m / self.rotor_mass_kg) / (2.0 * np.pi))

    @property
    def critical_speed_rpm(self) -> float:
        return 60.0 * self.critical_speed_hz

    @property
    def amplification_at_resonance(self) -> float:
        """Q of the rotor mode, ``1 / (2 zeta)``."""
        return float(0.5 / self.damping_ratio)

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["critical_speed_hz"] = self.critical_speed_hz
        return d


@dataclass(frozen=True)
class PathMode:
    """One structural mode of the force-to-casing-acceleration path.

    ``gain_g_per_n`` is a **receptance**: the peak casing acceleration in g per
    newton of broadband force injected at the bearing. This is the parameter that
    the whole twin's credibility about absolute amplitude rests on, and it is the
    one that nothing in a condition-monitoring deployment measures.
    """

    freq_hz: float
    q: float
    gain_g_per_n: float

    def __post_init__(self) -> None:
        if self.freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        if self.q < 1.0:
            raise ValueError("q must be >= 1 for a structural mode")
        if self.gain_g_per_n <= 0:
            raise ValueError("gain_g_per_n must be positive")


@dataclass(frozen=True)
class PathParams:
    """Force -> measured acceleration, including the sensor mount.

    **This is the least identifiable part of the entire twin, and the twin is
    built so that saying so is unavoidable.**

    Two separate gains, because they are excited by different things and are
    identifiable to different degrees:

    ``low_freq_gain_g_per_n``
        Quasi-static transmissibility from a bearing force at 1x/2x shaft order
        (tens of hertz, far below every structural mode) to casing acceleration.
        Appears in every shaft-order prediction, always multiplied by the
        unbalance moment, and never separable from it without a calibrated
        trial-mass balancing run.

    ``modes``
        The resonances that bearing impacts ring. Their *frequencies* are
        identifiable from a healthy record -- the broadband background excites
        them, and a kurtogram finds them (``pdmcore.dsp.envelope.kurtogram``).
        Their *gains* are not identifiable from any passive measurement, because
        the only observable is the product ``gain x force``, and the force is
        what the twin is trying to infer.

    Identifying ``gain_g_per_n`` needs an instrumented impact test: an
    instrumented hammer on the bearing housing with the accelerometer in place,
    which gives the receptance directly. That is a one-hour job for a vibration
    engineer with a 2-channel analyser and it is not something a wireless-sensor
    deployment does. ``pdmcore.twin.validity`` therefore refuses absolute
    amplitude predictions unless ``impact_tested`` is set.
    """

    low_freq_gain_g_per_n: float
    modes: Tuple[PathMode, ...] = ()
    mount_corner_hz: float = 8000.0
    impact_tested: bool = False

    def __post_init__(self) -> None:
        if self.low_freq_gain_g_per_n <= 0:
            raise ValueError("low_freq_gain_g_per_n must be positive")
        if self.mount_corner_hz <= 0:
            raise ValueError("mount_corner_hz must be positive")

    @property
    def mode_freqs_hz(self) -> Tuple[float, ...]:
        return tuple(m.freq_hz for m in self.modes)

    def as_dict(self) -> Dict[str, object]:
        return {
            "low_freq_gain_g_per_n": self.low_freq_gain_g_per_n,
            "mount_corner_hz": self.mount_corner_hz,
            "impact_tested": self.impact_tested,
            "modes": [asdict(m) for m in self.modes],
        }


@dataclass(frozen=True)
class ThermalParams:
    """First-order lumped thermal model of the winding / bearing station.

    ``C dT/dt = P_loss - (T - T_ambient) / R_th``

    One node, not two. A two-node model (winding and frame) is the textbook
    choice and fits a measured heat run much better, but the platform reports
    **one** temperature scalar per telemetry frame
    (``pdmcore.io.payload.TelemetryFrame.temperature_c``) with no load label and
    no sub-frame time resolution. A two-node model identified from a single
    scalar is four parameters against one observation; the second node would be
    pure decoration. :mod:`pdmcore.twin.thermal` records the error that the
    one-node reduction costs.

    ``r_th_k_per_w`` is prior-dominated rather than measured: the insulation
    class bounds the permitted rise, and a motor that reached its class limit at
    rated load would not have been sold. :func:`thermal_prior_bounds` turns that
    into an interval.
    """

    r_th_k_per_w: float
    c_th_j_per_k: float
    ambient_c: float = 25.0

    def __post_init__(self) -> None:
        if self.r_th_k_per_w <= 0 or self.c_th_j_per_k <= 0:
            raise ValueError("thermal resistance and capacitance must be positive")

    @property
    def tau_s(self) -> float:
        return float(self.r_th_k_per_w * self.c_th_j_per_k)

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["tau_s"] = self.tau_s
        return d


@dataclass(frozen=True)
class DefectParams:
    """A localised bearing defect, as a forcing function.

    ``peak_force_n`` is the peak of the force pulse generated when a rolling
    element enters the spall, and ``pulse_duration_s`` its width. Together they
    set the impulse ``~ 0.64 * F * tau`` (half-sine) and therefore the broadband
    level the path rings with.

    Both are **physically meaningful and jointly unobservable with the path
    gain**: every passive vibration measurement sees only
    ``gain_g_per_n * peak_force_n``. The twin keeps them separate anyway, because
    the point of a twin is to state what it would take to break the degeneracy,
    not to hide it in a single fitted constant.
    """

    mode: str = "outer_race"      # outer_race | inner_race | rolling_element | cage
    peak_force_n: float = 0.0
    pulse_duration_s: float = 1.0e-4
    slip_pct: float = 1.5
    load_zone_depth: float = 0.0  # amplitude-modulation depth, 0-1

    _MODES = ("outer_race", "inner_race", "rolling_element", "cage")

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"unknown defect mode {self.mode!r}; expected one of {self._MODES}")
        if self.peak_force_n < 0:
            raise ValueError("peak_force_n must be non-negative")
        if self.pulse_duration_s <= 0:
            raise ValueError("pulse_duration_s must be positive")
        if not 0.0 <= self.load_zone_depth <= 1.0:
            raise ValueError("load_zone_depth must be in [0, 1]")

    @property
    def impulse_ns(self) -> float:
        """Momentum delivered per rolling-element pass, in N.s (half-sine pulse)."""
        return float(2.0 / np.pi * self.peak_force_n * self.pulse_duration_s)


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatingPoint:
    """Where the machine is running. The twin's independent variable.

    ``load_fraction`` is the shaft torque as a fraction of rated. **It is not
    measured by anything this platform transmits** -- the wire format has
    ``shaft_hz`` and ``temperature_c`` and 63 features, and no load, current or
    power channel. So load either comes from a plant historian over a separate
    integration, or it is inferred from shaft speed via the torque-slip
    relation, and ``pdmcore.twin.identify.load_from_speed`` reports how badly
    that inference is conditioned.
    """

    shaft_hz: float
    load_fraction: float = 1.0
    supply_hz: Optional[float] = None    # None => nameplate line_hz
    supply_voltage_v: Optional[float] = None
    ambient_c: float = 25.0

    def __post_init__(self) -> None:
        if self.shaft_hz <= 0:
            raise ValueError("shaft_hz must be positive")
        if self.load_fraction < 0:
            raise ValueError("load_fraction must be non-negative")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterInterval:
    """A parameter's identified value and the width that goes with it.

    ``lo``/``hi`` are a 95 % **profile-likelihood** interval, not a standard
    error: the whole point of this branch is that several of these intervals are
    unbounded, and a standard error from a Fisher matrix with a near-zero
    eigenvalue reports a finite number for an infinite interval. ``identifiable``
    is ``False`` when the profile is flat -- when the data cannot distinguish
    the parameter from one a factor of ``flat_ratio`` away.
    """

    name: str
    value: float
    lo: float
    hi: float
    identifiable: bool = True
    unit: str = ""
    note: str = ""

    @property
    def rel_width(self) -> float:
        """Half-width as a fraction of the point estimate; ``inf`` if unbounded."""
        if self.value == 0 or not np.isfinite(self.lo) or not np.isfinite(self.hi):
            return float("inf")
        return float(0.5 * (self.hi - self.lo) / abs(self.value))

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # Non-finite bounds are the *point* of this class, and ``Infinity`` is
        # not valid JSON (Python emits it; most other parsers reject it). An
        # unbounded bound serialises as the string "inf"/"-inf", which survives
        # a round trip through any parser and cannot be mistaken for a number.
        for k in ("lo", "hi", "value"):
            v = float(d[k])
            if not np.isfinite(v):
                d[k] = "inf" if v > 0 else ("-inf" if v < 0 else None)
        d["rel_width"] = self.rel_width if np.isfinite(self.rel_width) else None
        return d


# ---------------------------------------------------------------------------
# The twin spec
# ---------------------------------------------------------------------------
#: Bumped whenever the forward model changes in a way that alters a prediction
#: from the same parameters. Part of :attr:`TwinSpec.version`, so an old
#: prediction can never be silently re-attributed to a new model.
TWIN_MODEL_VERSION = 1


@dataclass(frozen=True)
class TwinSpec:
    """A serialisable, versioned, uncertainty-carrying twin.

    The content-addressed :attr:`version` is the mechanism that makes a
    prediction traceable. It hashes the parameters, the validity domain, the
    identification provenance *and* :data:`TWIN_MODEL_VERSION`, so:

    * two twins with the same parameters but identified from different data have
      different versions (provenance is in the hash);
    * the same twin run under a changed forward model has a different version;
    * a prediction record carrying ``twin_version`` can be matched back to
      exactly one parameter set, or to none, and "none" is a detectable fault
      rather than a silent mismatch.

    ``validity`` is a plain dict here rather than a
    :class:`pdmcore.twin.validity.ValidityDomain` to keep this module free of
    import cycles; :func:`pdmcore.twin.validity.domain_of` rehydrates it.
    """

    name: str
    nameplate: Nameplate
    electrical: ElectricalParams
    rotor: RotorParams
    path: PathParams
    thermal: ThermalParams
    bearing: BearingGeometry = field(default_factory=lambda: BEARING_CATALOG["SKF6205"])
    defect: Optional[DefectParams] = None
    intervals: Tuple[ParameterInterval, ...] = ()
    validity: Dict[str, object] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)
    model_version: int = TWIN_MODEL_VERSION

    # -- serialisation -----------------------------------------------------
    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "model_version": self.model_version,
            "nameplate": self.nameplate.as_dict(),
            "electrical": self.electrical.as_dict(),
            "rotor": self.rotor.as_dict(),
            "path": self.path.as_dict(),
            "thermal": self.thermal.as_dict(),
            "bearing": {
                "name": self.bearing.name,
                "n_balls": self.bearing.n_balls,
                "ball_d": self.bearing.ball_d,
                "pitch_d": self.bearing.pitch_d,
                "contact_angle_deg": self.bearing.contact_angle_deg,
            },
            "defect": asdict(self.defect) if self.defect is not None else None,
            "intervals": [i.as_dict() for i in self.intervals],
            "validity": dict(self.validity),
            "provenance": dict(self.provenance),
        }

    def to_json(self, *, indent: int = 2) -> str:
        d = self.as_dict()
        d["version"] = self.version
        return json.dumps(d, indent=indent, sort_keys=True, default=_json_default)

    @property
    def version(self) -> str:
        """16-hex-character content hash of everything that affects a prediction.

        ``intervals`` are included: a twin re-identified from more data has the
        same point estimates only by coincidence, but even if it did, its
        uncertainty is different and therefore so is what it is allowed to
        predict. Excluding them would let a wider twin masquerade as a narrower
        one.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_json(cls, text: str) -> "TwinSpec":
        d = json.loads(text)
        claimed = d.pop("version", None)
        spec = cls.from_dict(d)
        if claimed is not None and claimed != spec.version:
            raise ValueError(
                f"twin spec version mismatch: file says {claimed}, content hashes to "
                f"{spec.version}. The file has been edited since it was identified, or it "
                "was written by a different forward-model version. Refusing to load it -- a "
                "prediction from an untraceable parameter set is worse than no prediction."
            )
        return spec

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TwinSpec":
        el = dict(d["electrical"])  # type: ignore[arg-type]
        if el.get("r_core_ohm") is None:
            el["r_core_ohm"] = float("inf")
        ro = dict(d["rotor"])  # type: ignore[arg-type]
        ro.pop("critical_speed_hz", None)
        th = dict(d["thermal"])  # type: ignore[arg-type]
        th.pop("tau_s", None)
        pa = dict(d["path"])  # type: ignore[arg-type]
        modes = tuple(PathMode(**m) for m in pa.pop("modes", []))
        be = dict(d["bearing"])  # type: ignore[arg-type]
        iv = tuple(
            ParameterInterval(
                **{
                    k: (float(v) if k in ("lo", "hi", "value") else v)
                    for k, v in i.items()
                    if k != "rel_width"
                }
            )
            for i in d.get("intervals", [])  # type: ignore[union-attr]
        )
        defect = d.get("defect")
        return cls(
            name=str(d["name"]),
            nameplate=Nameplate(**d["nameplate"]),  # type: ignore[arg-type]
            electrical=ElectricalParams(**el),
            rotor=RotorParams(**ro),
            path=PathParams(modes=modes, **pa),
            thermal=ThermalParams(**th),
            bearing=BearingGeometry(**be),
            defect=DefectParams(**defect) if defect else None,  # type: ignore[arg-type]
            intervals=iv,
            validity=dict(d.get("validity", {})),  # type: ignore[arg-type]
            provenance=dict(d.get("provenance", {})),  # type: ignore[arg-type]
            model_version=int(d.get("model_version", TWIN_MODEL_VERSION)),  # type: ignore[arg-type]
        )

    def interval(self, name: str) -> Optional[ParameterInterval]:
        for i in self.intervals:
            if i.name == name:
                return i
        return None

    def with_electrical(self, **kw) -> "TwinSpec":
        return replace(self, electrical=replace(self.electrical, **kw))


def thermal_prior_bounds(nameplate: Nameplate, loss_w: float) -> Tuple[float, float]:
    """``(r_th_lo, r_th_hi)`` K/W from the insulation class and a loss estimate.

    The permitted temperature rise for the insulation class is an upper bound on
    ``R_th * P_loss`` at rated load, because a motor that reached its limit at
    rated load in a 40 C ambient would not be certifiable. The lower bound is
    set by the observation that motors are not built with 3x more cooling than
    they need: real rises at rated load are 50-80 K for class F / B-rise TEFC
    machines.

    So this returns a prior spanning roughly a factor of 1.6, and that factor is
    the floor on what a single temperature reading can say about electrical
    losses. See ``twin/motor_physics/results/thermal_identifiability.json``.
    """
    if loss_w <= 0:
        raise ValueError("loss_w must be positive")
    limit_k = {"A": 60.0, "E": 75.0, "B": 80.0, "F": 105.0, "H": 125.0}
    permitted = limit_k.get(nameplate.insulation_class.upper())
    if permitted is None:
        raise ValueError(f"unknown insulation class {nameplate.insulation_class!r}")
    # Class F insulation is almost always used at "B rise" (80 K) for margin.
    typical_hi = 0.80 * permitted
    typical_lo = 0.45 * permitted
    return float(typical_lo / loss_w), float(typical_hi / loss_w)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


__all__.append("thermal_prior_bounds")
__all__.append("TWIN_MODEL_VERSION")
