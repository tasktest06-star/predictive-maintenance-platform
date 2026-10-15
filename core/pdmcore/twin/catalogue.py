"""Reference motors with *published* equivalent circuits, and why they are here.

The identification study in this branch has to answer "how close does nameplate
data get you to the truth?", and that question is meaningless unless there is a
truth that was not produced by this code. Identifying a model from data generated
by the same model measures only self-consistency; it is the circularity the
platform already warns about for its own simulator
(``core/pdmcore/sim/machine.py``, "Honesty about what this is").

So the catalogue holds induction machines whose equivalent-circuit parameters
were published independently of this repository. Tier-1 identification is then a
genuine inverse problem: the nameplate is derived from the published circuit
(so the nameplate itself is exact), identification sees only the nameplate, and
the recovered parameters are scored against the published ones.

PROVENANCE AND ITS LIMITS
-------------------------
These parameter sets are **transcribed from standard textbook machine data from
memory, and have not been checked against the printed sources in this
environment** (no network-fetched copy of either book is in this repository).
That is a real weakness and it is stated rather than hidden. Three things
contain it:

1. :func:`check_catalogue` asserts, for every entry, that the published circuit
   produces a rated efficiency, power factor, breakdown-torque ratio and
   locked-rotor current inside the band that machines of that size actually
   occupy. A transcription error large enough to change a conclusion would move
   at least one of those out of band. The check is run as a test
   (``core/tests/test_twin_catalogue.py``) and by
   ``twin/motor_physics/prototype/01_catalogue_check.py``, which commits the
   numbers.
2. The *conclusions* of the identification study are about **interval widths and
   rank deficiency** -- which parameter combinations the data can and cannot
   see. Those are properties of the model structure and the observation set, not
   of the particular numbers; a 10 % error in a published ``rr`` moves the
   recovered value by 10 % and leaves the width of its interval, and the flatness
   of the leakage-split profile, exactly where they were.
3. Where an absolute accuracy claim *is* made (the tier-1 recovery errors in
   ``twin/motor_physics/results/tier1_recovery.json``), it is stated as "error
   against the transcribed reference", with this caveat attached.

If anyone has the books: ``krause_3hp`` and ``krause_2250hp`` are the two
machines used throughout Krause, Wasynczuk & Sudhoff, *Analysis of Electric
Machinery and Drive Systems*; ``chapman_25hp`` is the 460 V 25 hp motor of
Chapman, *Electric Machinery Fundamentals*, chapter 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .electrical import breakdown_torque, locked_rotor_current, slip_for_torque, steady_state
from .params import ElectricalParams, Nameplate, ie_efficiency_floor

__all__ = [
    "ReferenceMotor",
    "REFERENCE_MOTORS",
    "PU_BANDS",
    "check_catalogue",
    "nameplate_from_circuit",
]


@dataclass(frozen=True)
class ReferenceMotor:
    """A published equivalent circuit plus the ratings its source states."""

    key: str
    source: str
    rated_power_kw: float
    rated_voltage_v: float
    line_hz: float
    pole_pairs: int
    connection: str
    electrical: ElectricalParams
    inertia_kgm2: Optional[float] = None
    note: str = ""


#: ``xls``/``xlr``/``xm`` are at the stated line frequency, per phase, star
#: equivalent, referred to the stator. Krause's machines are given with
#: ``w_b = 377 rad/s`` reactances, which is exactly this convention at 60 Hz.
REFERENCE_MOTORS: Dict[str, ReferenceMotor] = {
    "krause_3hp": ReferenceMotor(
        key="krause_3hp",
        source="Krause, Wasynczuk & Sudhoff, Analysis of Electric Machinery (3 hp machine)",
        rated_power_kw=2.238,            # 3 hp
        rated_voltage_v=220.0,
        line_hz=60.0,
        pole_pairs=2,
        connection="Y",
        electrical=ElectricalParams(
            rs_ohm=0.435, rr_ohm=0.816, xls_ohm=0.754, xlr_ohm=0.754, xm_ohm=26.13,
            # Neither source publishes a core-loss resistance or a friction
            # figure. Rather than invent one, core loss is disabled and the
            # mechanical loss is set to zero: the catalogue holds what was
            # published. The consequence -- that this reference cannot test the
            # identifiability of core loss -- is a finding, not a gap to paper
            # over.
            r_core_ohm=float("inf"), friction_loss_w=0.0,
        ),
        inertia_kgm2=0.089,
        note="small industrial machine; high per-unit resistance, poor efficiency",
    ),
    "krause_2250hp": ReferenceMotor(
        key="krause_2250hp",
        source="Krause, Wasynczuk & Sudhoff, Analysis of Electric Machinery (2250 hp machine)",
        rated_power_kw=1678.0,           # 2250 hp
        rated_voltage_v=2300.0,
        line_hz=60.0,
        pole_pairs=2,
        connection="Y",
        electrical=ElectricalParams(
            rs_ohm=0.029, rr_ohm=0.022, xls_ohm=0.226, xlr_ohm=0.226, xm_ohm=13.04,
            r_core_ohm=float("inf"), friction_loss_w=0.0,
        ),
        inertia_kgm2=63.87,
        note="large machine; very low per-unit resistance, slip under 1 %",
    ),
    "chapman_25hp": ReferenceMotor(
        key="chapman_25hp",
        source="Chapman, Electric Machinery Fundamentals, ch. 7 (460 V 25 hp motor)",
        rated_power_kw=18.65,            # 25 hp
        rated_voltage_v=460.0,
        line_hz=60.0,
        pole_pairs=2,
        connection="Y",
        electrical=ElectricalParams(
            rs_ohm=0.641, rr_ohm=0.332, xls_ohm=1.106, xlr_ohm=0.464, xm_ohm=26.3,
            r_core_ohm=float("inf"),
            # The source states 1100 W of rotational loss, assumed constant, with
            # core loss lumped into it. Modelled as constant friction rather than
            # as a core-loss resistance, exactly as published.
            friction_loss_w=1100.0,
        ),
        note=(
            "the one catalogue entry with an ASYMMETRIC leakage split "
            "(xls/xlr = 2.38), which is what makes it the useful test of whether "
            "the conventional xls = xlr assumption costs anything"
        ),
    ),
}


def nameplate_from_circuit(
    ref: ReferenceMotor,
    *,
    ie_class: str = "IE3",
    speed_rounding_rpm: float = 5.0,
    current_rounding_pct: float = 2.5,
    include_power_factor: bool = True,
    include_efficiency: bool = False,
    include_code_letter: bool = False,
) -> Nameplate:
    """Build the nameplate a manufacturer would stamp on ``ref``.

    The rated point is found by solving the published circuit for the slip at
    which shaft power equals the rating, then **rounding the speed and current
    the way a plate does**. That rounding is not a detail: it is the dominant
    term in the tier-1 uncertainty budget, and the whole point of generating the
    plate rather than guessing it is that the rounding is then the *only*
    difference between the plate and the truth.

    ``include_power_factor``, ``include_efficiency`` and ``include_code_letter``
    select the tier-1 sub-levels (1a to 1d): how much nameplate information the
    study is allowed to use. They are separate flags rather than one "detail"
    enum because the whole point is to price each field. Note that
    ``include_efficiency=False`` is not "no efficiency information" -- the IE
    class is always present, and :meth:`Nameplate.efficiency_bounds` turns it
    into a one-sided bound.
    """
    # Bootstrap plate: speed is needed to compute rated torque, and rated torque
    # is needed to find the speed. Two passes converge immediately because the
    # slip correction to torque is ~1 %.
    sync_rpm = 60.0 * ref.line_hz / ref.pole_pairs
    plate0 = Nameplate(
        name=ref.key,
        rated_power_kw=ref.rated_power_kw,
        rated_voltage_v=ref.rated_voltage_v,
        rated_current_a=1.0,
        rated_speed_rpm=sync_rpm * 0.97,
        line_hz=ref.line_hz,
        pole_pairs=ref.pole_pairs,
        connection=ref.connection,
    )
    target_w = ref.rated_power_kw * 1e3
    s = 0.02
    for _ in range(40):
        ss = steady_state(ref.electrical, plate0, s)
        if abs(ss.shaft_power_w - target_w) < 1e-6 * target_w:
            break
        # Shaft power is very nearly linear in slip on the stable branch.
        s = float(np.clip(s * target_w / max(ss.shaft_power_w, 1e-9), 1e-6, 0.5))
    ss = steady_state(ref.electrical, plate0, s)

    true_rpm = ss.shaft_hz * 60.0
    rounded_rpm = round(true_rpm / speed_rounding_rpm) * speed_rounding_rpm
    if rounded_rpm >= sync_rpm:
        rounded_rpm = sync_rpm - speed_rounding_rpm
    # Nameplate current is printed to 3 significant figures.
    i = ss.stator_current_a
    digits = max(0, 2 - int(np.floor(np.log10(i))))
    rounded_a = round(i, digits)

    lr_pu: Optional[float] = None
    if include_code_letter:
        lr_pu = locked_rotor_current(ref.electrical, plate0) / i

    return Nameplate(
        name=ref.key,
        rated_power_kw=ref.rated_power_kw,
        rated_voltage_v=ref.rated_voltage_v,
        rated_current_a=float(rounded_a),
        rated_speed_rpm=float(rounded_rpm),
        line_hz=ref.line_hz,
        pole_pairs=ref.pole_pairs,
        connection=ref.connection,
        ie_class=ie_class,
        rated_power_factor=round(float(ss.power_factor), 2) if include_power_factor else None,
        # A European plate carries a measured efficiency to 0.1 of a point. When
        # that field is used the study treats it as +-0.5 point, which covers the
        # printing resolution plus the IEC 60034-2-1 measurement tolerance.
        rated_efficiency=round(float(ss.efficiency), 3) if include_efficiency else None,
        efficiency_is_class_floor=not include_efficiency,
        locked_rotor_current_pu=round(float(lr_pu), 1) if lr_pu is not None else None,
        speed_rounding_rpm=speed_rounding_rpm,
        current_rounding_pct=current_rounding_pct,
    )


#: Per-unit impedance bands for squirrel-cage induction machines, and the
#: terminal quantities they imply. This is the transcription check, and it is
#: expressed in per-unit deliberately: ohmic values span four orders of magnitude
#: across the catalogue (``rr`` from 0.022 to 0.816 ohm) so an ohmic band would
#: have to be per-motor and would then be a check that cannot fail. In per-unit
#: the same band applies to a 2 kW motor and a 1.7 MW motor, and a decimal-point
#: slip, a swapped ``rs``/``rr`` column or an ``xm`` read off the wrong row moves
#: a value clean out of it.
#:
#: Bands are wide on purpose -- the job is to catch a transcription error, not to
#: assert what a motor's resistance is. The one that would be most informative,
#: ``xm_pu``, is also the tightest in reality: the magnetising reactance of an
#: induction machine is between about 1.5 and 5 per unit for anything with an
#: air gap, because it is set by the gap geometry and nothing else.
PU_BANDS: Dict[str, Tuple[float, float]] = {
    "rs_pu": (0.003, 0.10),
    "rr_pu": (0.003, 0.10),
    "x_leakage_pu": (0.06, 0.40),
    "xm_pu": (1.2, 6.0),
    # Terminal consequences. The efficiency band is for the **copper-loss-only**
    # efficiency, which is what these published circuits produce: none of the
    # three sources publishes a core-loss resistance, and two publish no
    # friction either. A real machine of the same rating is 4-8 points worse.
    # ``core_and_friction_omitted`` in the output records that gap explicitly so
    # that nobody reads 98.2 % off this table and believes it.
    "efficiency_copper_only": (0.80, 0.99),
    "power_factor": (0.70, 0.96),
    "breakdown_torque_pu": (1.5, 5.5),
    "locked_rotor_current_pu": (3.0, 10.0),
}


def check_catalogue(keys: Optional[List[str]] = None) -> List[Dict[str, object]]:
    """Verify every published circuit against :data:`PU_BANDS`.

    Returns one record per motor with the computed quantities and a per-check
    pass flag. Raises nothing: the caller decides whether a failure is fatal.
    ``core/tests/test_twin_catalogue.py`` treats it as fatal, and
    ``twin/motor_physics/prototype/01_catalogue_check.py`` commits the numbers.

    The record also carries ``typical_efficiency_for_rating`` and
    ``core_and_friction_omitted_points``: the difference between the circuit's
    copper-only efficiency and what a machine of that rating actually achieves.
    That difference is a real limitation of the reference data -- it means the
    catalogue **cannot** be used to test the identifiability of core loss, only
    of the copper and leakage parameters -- and it is reported as a number
    rather than as a sentence in a docstring.
    """
    out: List[Dict[str, object]] = []
    for key in keys or sorted(REFERENCE_MOTORS):
        ref = REFERENCE_MOTORS[key]
        plate = nameplate_from_circuit(ref, include_code_letter=True, include_efficiency=True)
        s_rated = slip_for_torque(ref.electrical, plate, plate.rated_torque_nm)
        ss = steady_state(ref.electrical, plate, s_rated)
        s_bk, t_bk = breakdown_torque(ref.electrical, plate)
        i_lr = locked_rotor_current(ref.electrical, plate)

        z_base = plate.phase_voltage_v / ss.stator_current_a
        values = {
            "rs_pu": ref.electrical.rs_ohm / z_base,
            "rr_pu": ref.electrical.rr_ohm / z_base,
            "x_leakage_pu": ref.electrical.x_leakage_sum_ohm / z_base,
            "xm_pu": ref.electrical.xm_ohm / z_base,
            "efficiency_copper_only": ss.efficiency,
            "power_factor": ss.power_factor,
            "breakdown_torque_pu": t_bk / plate.rated_torque_nm,
            "locked_rotor_current_pu": i_lr / plate.rated_current_a,
        }
        checks = {k: bool(PU_BANDS[k][0] <= v <= PU_BANDS[k][1]) for k, v in values.items()}

        typical_eta = ie_efficiency_floor(ref.rated_power_kw, "IE2")
        out.append(
            {
                "key": key,
                "source": ref.source,
                "rated_power_kw": ref.rated_power_kw,
                "rated_slip": float(s_rated),
                "rated_speed_rpm": float(ss.shaft_hz * 60.0),
                "nameplate_speed_rpm": plate.rated_speed_rpm,
                "nameplate_current_a": plate.rated_current_a,
                "stator_current_a": float(ss.stator_current_a),
                "z_base_ohm": float(z_base),
                "breakdown_slip": float(s_bk),
                "leakage_split": float(ref.electrical.leakage_split),
                "values": {k: float(v) for k, v in values.items()},
                "bands": {k: list(PU_BANDS[k]) for k in values},
                "checks": checks,
                "all_pass": all(checks.values()),
                "typical_efficiency_for_rating": float(typical_eta),
                "core_and_friction_omitted_points": float(100.0 * (ss.efficiency - typical_eta)),
            }
        )
    return out
