r"""Induction-machine electromechanics: steady state, quasi-steady, and d-q.

Three models, each with a stated job. The reason there are three rather than one
is that identification and simulation want different things, and the usual
digital-twin failure is to build the impressive one and identify the cheap one.

1. :func:`steady_state` -- the **exact T equivalent circuit**, solved
   algebraically for one slip. This is the model every identification in
   :mod:`pdmcore.twin.identify` uses, because every nameplate quantity and every
   commissioning observable is a steady-state quantity. It is also the only one
   fast enough to sit inside a profile-likelihood inner loop (~15 us per call).

2. :func:`run_up_quasi_steady` -- torque from (1) at the instantaneous slip, fed
   to ``J dw/dt = T_em - T_load - T_loss``. One state. This is what a run-up or
   coast-down identification fits.

3. :func:`run_up_dq` -- the full five-state d-q model in the synchronous
   reference frame. **Not used for identification.** Its only job is to measure
   how wrong (2) is, so that the claim "quasi-steady is adequate for run-up
   identification" is a measurement rather than an assertion.
   ``twin/motor_physics/prototype/02_runup_model_error.py`` runs that comparison
   and writes the answer to JSON.

Equivalent circuit
------------------
Per phase, star equivalent, referred to the stator::

      Rs    jXls                      jXlr
    o--[]---UUU----+-------------+-----UUU---[]--o
                   |             |            Rr/s
    V1          [Rc]          jXm
                   |             |
    o--------------+-------------+-------------- o

with the magnetising branch (core-loss resistance ``Rc`` in parallel with
``jXm``) between the stator and rotor leakage impedances. Air-gap power is
:math:`3 |I_2|^2 R_r/s`, electromagnetic torque is that divided by the
synchronous mechanical speed, and shaft power subtracts friction and windage.

The Thevenin ("approximate") circuit, which moves the magnetising branch ahead of
``Rs + jXls``, is deliberately **not** used: it is good for torque near rated
slip but wrong by 2-5 % in stator current, and stator current is one of only
four or five nameplate observations there are. Approximating the quantity you
are fitting to is how a 5 % circuit error becomes a 30 % parameter error.

Reactance scaling
-----------------
All reactances are stored at :attr:`Nameplate.line_hz` and scaled linearly with
the actual supply frequency. ``Rr`` is treated as frequency-independent, which
ignores rotor deep-bar skin effect. That matters at high slip -- during a
direct-on-line start the rotor sees 50/60 Hz and a double-cage or deep-bar rotor
can show 2-3x its running-slip resistance -- so :func:`run_up_quasi_steady`
takes an optional ``skin_ratio`` and the validity domain refuses starting-torque
predictions when it has not been identified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite as _isfinite
from math import sqrt as _sqrt
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .params import ElectricalParams, Nameplate

_SQRT3 = _sqrt(3.0)
_TWO_PI = 2.0 * 3.141592653589793

__all__ = [
    "SteadyState",
    "steady_state",
    "torque_slip_curve",
    "slip_for_torque",
    "breakdown_torque",
    "locked_rotor_current",
    "run_up_quasi_steady",
    "run_up_dq",
    "dq_steady_state_check",
]


@dataclass(frozen=True)
class SteadyState:
    """Everything the equivalent circuit says at one slip.

    ``loss_w`` is the total loss and is what feeds the thermal model: stator
    copper + rotor copper + core + friction/windage. It is reported as a
    breakdown because the thermal node sees the sum but the *identifiability*
    differs per term -- rotor copper is pinned by slip and torque, core loss is
    not pinned by anything on a nameplate.
    """

    slip: float
    supply_hz: float
    shaft_hz: float
    stator_current_a: float       # per-phase RMS = line RMS for the Y equivalent
    rotor_current_a: float
    power_factor: float
    input_power_w: float
    airgap_power_w: float
    torque_em_nm: float
    shaft_power_w: float
    shaft_torque_nm: float
    efficiency: float
    loss_w: float
    loss_breakdown_w: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k != "loss_breakdown_w"}
        d["loss_breakdown_w"] = dict(self.loss_breakdown_w)
        return d


def _scaled(p: ElectricalParams, f_ratio: float) -> Tuple[float, float, float]:
    """Reactances at a supply frequency ``f_ratio`` times rated."""
    return p.xls_ohm * f_ratio, p.xlr_ohm * f_ratio, p.xm_ohm * f_ratio


def steady_state(
    p: ElectricalParams,
    plate: Nameplate,
    slip: float,
    *,
    supply_hz: Optional[float] = None,
    supply_voltage_v: Optional[float] = None,
    rr_scale: float = 1.0,
) -> SteadyState:
    """Solve the exact T circuit at one slip.

    Parameters
    ----------
    slip
        Positive for motoring. Must be non-zero; at exactly zero the rotor branch
        is an open circuit and torque is identically zero, which is a legitimate
        answer but a degenerate one for every optimiser, so a slip below 1e-9 is
        rejected rather than silently clamped.
    supply_hz, supply_voltage_v
        Default to the nameplate values. A VFD asset runs at many frequencies;
        pass both and the twin keeps V/f constant only if you do.
    rr_scale
        Multiplier on rotor resistance, for deep-bar skin effect at high slip.
    """
    slip = float(slip)
    if not _isfinite(slip) or abs(slip) < 1e-9:
        raise ValueError(
            "slip must be finite and non-zero; at s = 0 the rotor branch is open and "
            "torque is identically zero -- the caller almost certainly wants a small "
            "positive slip instead"
        )
    f = float(supply_hz if supply_hz is not None else plate.line_hz)
    if f <= 0:
        raise ValueError("supply_hz must be positive")
    v_ll = float(supply_voltage_v if supply_voltage_v is not None else plate.rated_voltage_v)
    v1 = v_ll / _SQRT3 if plate.connection == "Y" else v_ll

    ratio = f / plate.line_hz
    xls, xlr, xm = _scaled(p, ratio)

    z_mag = 1j * xm if not _isfinite(p.r_core_ohm) else (p.r_core_ohm * 1j * xm) / (p.r_core_ohm + 1j * xm)
    z_rotor = p.rr_ohm * rr_scale / slip + 1j * xlr
    z_par = z_mag * z_rotor / (z_mag + z_rotor)
    z_tot = p.rs_ohm + 1j * xls + z_par

    i1 = v1 / z_tot
    e_gap = i1 * z_par
    i2 = e_gap / z_rotor

    i1_mag = abs(i1)
    i2_mag = abs(i2)
    p_in = 3.0 * (v1 * i1.conjugate()).real
    pf = p_in / (3.0 * v1 * i1_mag) if i1_mag > 0 else 0.0

    p_airgap = 3.0 * i2_mag**2 * (p.rr_ohm * rr_scale / slip)
    omega_sync_mech = _TWO_PI * f / plate.pole_pairs
    t_em = p_airgap / omega_sync_mech
    shaft_hz = (f / plate.pole_pairs) * (1.0 - slip)
    p_mech_gross = p_airgap * (1.0 - slip)
    p_fw = p.mechanical_loss_w(abs(shaft_hz))
    p_shaft = p_mech_gross - p_fw

    stator_cu = 3.0 * i1_mag**2 * p.rs_ohm
    rotor_cu = 3.0 * i2_mag**2 * p.rr_ohm * rr_scale
    core = 3.0 * abs(e_gap) ** 2 / p.r_core_ohm if _isfinite(p.r_core_ohm) else 0.0
    loss = stator_cu + rotor_cu + core + p_fw

    return SteadyState(
        slip=slip,
        supply_hz=f,
        shaft_hz=float(shaft_hz),
        stator_current_a=i1_mag,
        rotor_current_a=i2_mag,
        power_factor=float(pf),
        input_power_w=float(p_in),
        airgap_power_w=float(p_airgap),
        torque_em_nm=float(t_em),
        shaft_power_w=float(p_shaft),
        shaft_torque_nm=float(p_shaft / (_TWO_PI * shaft_hz)) if shaft_hz > 0 else 0.0,
        efficiency=float(p_shaft / p_in) if p_in > 0 else 0.0,
        loss_w=float(loss),
        loss_breakdown_w={
            "stator_copper": float(stator_cu),
            "rotor_copper": float(rotor_cu),
            "core": float(core),
            "friction_windage": float(p_fw),
        },
    )


def torque_slip_curve(
    p: ElectricalParams,
    plate: Nameplate,
    *,
    n: int = 400,
    s_min: float = 1e-4,
    s_max: float = 1.0,
    supply_hz: Optional[float] = None,
    log_spaced: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(slip, torque_em_nm)``. Log-spaced by default.

    Log spacing is not cosmetic: the useful operating range of an induction motor
    is ``s < 0.03`` while the curve is defined to ``s = 1``, so a linear grid of
    400 points puts 12 of them where the machine actually runs.
    """
    s = np.geomspace(s_min, s_max, n) if log_spaced else np.linspace(s_min, s_max, n)
    t = np.array([steady_state(p, plate, float(si), supply_hz=supply_hz).torque_em_nm for si in s])
    return s, t


def slip_for_torque(
    p: ElectricalParams,
    plate: Nameplate,
    torque_nm: float,
    *,
    supply_hz: Optional[float] = None,
    supply_voltage_v: Optional[float] = None,
    shaft_torque: bool = True,
) -> float:
    """Invert the torque-slip relation on the **stable** branch.

    "Stable" means slip below breakdown. The torque-slip curve is non-monotonic,
    so a naive root find can return the unstable high-slip solution, which is a
    real root of the equation and a machine state that lasts milliseconds. A twin
    that reports it as the operating point will predict a current 4x too high and
    a temperature to match.

    Raises ``ValueError`` when the requested torque exceeds breakdown -- that is
    a stall, not an operating point, and clamping to breakdown would turn an
    impossible load into a plausible-looking prediction.
    """
    s_break, t_break = breakdown_torque(p, plate, supply_hz=supply_hz, supply_voltage_v=supply_voltage_v)

    def f(s: float) -> float:
        ss = steady_state(p, plate, s, supply_hz=supply_hz, supply_voltage_v=supply_voltage_v)
        t = ss.shaft_torque_nm if shaft_torque else ss.torque_em_nm
        return t - torque_nm

    if f(s_break) < 0.0:
        raise ValueError(
            f"requested torque {torque_nm:.1f} N.m exceeds breakdown "
            f"({t_break:.1f} N.m at s = {s_break:.4f}); the machine stalls rather than "
            "reaching this operating point"
        )
    lo = 1e-7
    if f(lo) > 0.0:
        # Shaft torque at near-zero slip is negative (friction), so this means the
        # requested torque is below the no-load drag: the machine is being driven.
        raise ValueError(
            f"requested torque {torque_nm:.3f} N.m is below the no-load drag torque; "
            "the shaft is driving the motor, which is outside this model's motoring branch"
        )
    return float(brentq(f, lo, s_break, xtol=1e-12, rtol=1e-12))


def breakdown_torque(
    p: ElectricalParams,
    plate: Nameplate,
    *,
    supply_hz: Optional[float] = None,
    supply_voltage_v: Optional[float] = None,
) -> Tuple[float, float]:
    """``(slip_at_breakdown, torque_nm)`` by golden-section search on a log grid.

    Found numerically rather than from the Thevenin closed form, because the
    closed form belongs to the approximate circuit and is 2-4 % off for the exact
    one. 2-4 % on breakdown torque is not important for a prediction; it is
    important when breakdown is used as the bracket for a root find, because a
    bracket on the wrong side of the peak silently selects the unstable branch.
    """
    s = np.geomspace(1e-4, 1.0, 120)
    t = np.array(
        [
            steady_state(p, plate, float(si), supply_hz=supply_hz, supply_voltage_v=supply_voltage_v).torque_em_nm
            for si in s
        ]
    )
    i = int(np.argmax(t))
    lo = s[max(0, i - 1)]
    hi = s[min(s.size - 1, i + 1)]
    # Local refinement: 40 points over a factor of ~1.04 either side is plenty,
    # the peak is quadratic there.
    s2 = np.geomspace(lo, hi, 41)
    t2 = np.array(
        [
            steady_state(p, plate, float(si), supply_hz=supply_hz, supply_voltage_v=supply_voltage_v).torque_em_nm
            for si in s2
        ]
    )
    j = int(np.argmax(t2))
    return float(s2[j]), float(t2[j])


def locked_rotor_current(
    p: ElectricalParams,
    plate: Nameplate,
    *,
    rr_scale: float = 1.0,
) -> float:
    """Stator current at ``s = 1``, per phase RMS.

    Reported as an absolute current; divide by :attr:`Nameplate.rated_current_a`
    for the per-unit value that a NEMA code letter encodes. When a code letter is
    on the plate this is the one extra nameplate observation that constrains the
    leakage reactances, and :mod:`pdmcore.twin.identify` quantifies how much it
    buys (tier 1b).
    """
    return steady_state(p, plate, 1.0, rr_scale=rr_scale).stator_current_a


# ---------------------------------------------------------------------------
# Transients
# ---------------------------------------------------------------------------
def run_up_quasi_steady(
    p: ElectricalParams,
    plate: Nameplate,
    *,
    inertia_kgm2: float,
    load_torque: Optional[Callable[[float], float]] = None,
    t_end_s: float = 10.0,
    shaft_hz0: float = 0.0,
    supply_hz: Optional[float] = None,
    supply_voltage_v: Optional[float] = None,
    skin_ratio: float = 1.0,
    coast_down: bool = False,
    n_eval: int = 400,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """One-state run-up or coast-down with quasi-steady electromagnetic torque.

    ``J dw_m/dt = T_em(s) - T_load(w_m) - T_fw(w_m)``

    ``skin_ratio`` interpolates rotor resistance linearly in slip from
    ``rr`` at ``s = 0`` to ``skin_ratio * rr`` at ``s = 1``, which is the
    cheapest representation of deep-bar skin effect that is not simply wrong at
    standstill. It is an unidentified parameter at every information tier in this
    branch; it is here so that a starting-torque prediction can be *refused* for
    a stated reason rather than produced from ``skin_ratio = 1``.

    ``coast_down=True`` removes the supply: ``J dw/dt = -T_load - T_fw``. The
    resulting trace is the cleanest mechanical measurement a plant can make and
    it identifies **only the ratio** of drag torque to inertia, never either
    alone. :mod:`pdmcore.twin.identify` demonstrates the flat likelihood ridge.

    Returns ``{"t_s", "shaft_hz", "slip", "torque_em_nm", "stator_current_a"}``.
    """
    if inertia_kgm2 <= 0:
        raise ValueError("inertia_kgm2 must be positive")
    f = float(supply_hz if supply_hz is not None else plate.line_hz)
    sync_hz = f / plate.pole_pairs
    load = load_torque or (lambda shaft_hz: 0.0)

    def accel(_t: float, y: np.ndarray) -> np.ndarray:
        shaft_hz = float(y[0])
        omega_m = 2.0 * np.pi * max(shaft_hz, 0.0)
        t_load = float(load(shaft_hz))
        if omega_m > 1e-6:
            # Friction and windage are specified as a power, so they become a
            # torque only above a speed where that division is meaningful.
            t_load += p.mechanical_loss_w(abs(shaft_hz)) / omega_m
        if coast_down:
            t_em = 0.0
        else:
            s = (sync_hz - shaft_hz) / sync_hz
            s = float(np.clip(s, 1e-7, 2.0))
            rr_scale = 1.0 + (skin_ratio - 1.0) * s
            t_em = steady_state(
                p, plate, s, supply_hz=f, supply_voltage_v=supply_voltage_v, rr_scale=rr_scale
            ).torque_em_nm
        dw_m = (t_em - t_load) / inertia_kgm2        # rad/s^2, mechanical
        return np.array([dw_m / (2.0 * np.pi)])      # d(shaft_hz)/dt

    def stopped(_t: float, y: np.ndarray) -> float:
        return float(y[0]) - 1e-3

    stopped.terminal = coast_down  # type: ignore[attr-defined]
    stopped.direction = -1.0       # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, t_end_s, n_eval)
    sol = solve_ivp(
        accel,
        (0.0, t_end_s),
        np.array([float(shaft_hz0)]),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=stopped if coast_down else None,
        dense_output=False,
    )
    if not sol.success and sol.status != 1:
        raise RuntimeError(f"run-up integration failed: {sol.message}")

    hz = np.asarray(sol.y[0], dtype=np.float64)
    slip = (sync_hz - hz) / sync_hz
    if coast_down:
        t_em = np.zeros_like(hz)
        cur = np.zeros_like(hz)
    else:
        s_c = np.clip(slip, 1e-7, 2.0)
        rr_s = 1.0 + (skin_ratio - 1.0) * s_c
        ss = [
            steady_state(p, plate, float(si), supply_hz=f, supply_voltage_v=supply_voltage_v, rr_scale=float(ri))
            for si, ri in zip(s_c, rr_s)
        ]
        t_em = np.array([x.torque_em_nm for x in ss])
        cur = np.array([x.stator_current_a for x in ss])
    return {
        "t_s": np.asarray(sol.t, dtype=np.float64),
        "shaft_hz": hz,
        "slip": slip,
        "torque_em_nm": t_em,
        "stator_current_a": cur,
    }


def run_up_dq(
    p: ElectricalParams,
    plate: Nameplate,
    *,
    inertia_kgm2: float,
    load_torque: Optional[Callable[[float], float]] = None,
    t_end_s: float = 2.0,
    supply_hz: Optional[float] = None,
    supply_voltage_v: Optional[float] = None,
    n_eval: int = 2000,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Dict[str, np.ndarray]:
    r"""Five-state d-q model in the synchronous reference frame.

    States are flux linkages and mechanical speed,
    :math:`[\lambda_{qs}, \lambda_{ds}, \lambda_{qr}, \lambda_{dr}, \omega_m]`.
    The synchronous frame is chosen deliberately: in it the steady-state
    solution is **constant**, so the integrator does not have to resolve 50/60 Hz
    carrier oscillations for the whole run-up and the whole thing costs a few
    tens of milliseconds instead of seconds. The transient torque pulsation
    still appears -- it is physical -- but as a decaying oscillation of the dq
    states rather than as a carrier.

    Core loss is omitted here (``Rc`` has no place in the standard two-axis
    model without a third winding), which is a known difference from
    :func:`steady_state`. :func:`dq_steady_state_check` measures the resulting
    disagreement so that the difference is a number, not a caveat.

    Amplitude-invariant Park transform, so ``v_qs`` is the **peak** phase voltage
    and the torque constant carries the 3/2.
    """
    if inertia_kgm2 <= 0:
        raise ValueError("inertia_kgm2 must be positive")
    f = float(supply_hz if supply_hz is not None else plate.line_hz)
    v_ll = float(supply_voltage_v if supply_voltage_v is not None else plate.rated_voltage_v)
    v_phase = v_ll / np.sqrt(3.0) if plate.connection == "Y" else v_ll
    w_rated = 2.0 * np.pi * plate.line_hz
    w_e = 2.0 * np.pi * f
    pp = plate.pole_pairs

    lls = p.xls_ohm / w_rated
    llr = p.xlr_ohm / w_rated
    lm = p.xm_ohm / w_rated
    ls, lr = lls + lm, llr + lm
    det = ls * lr - lm * lm
    if det <= 0:
        raise ValueError("degenerate inductance matrix: Ls*Lr <= Lm^2")

    vqs, vds = np.sqrt(2.0) * v_phase, 0.0
    load = load_torque or (lambda shaft_hz: 0.0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        lqs, lds, lqr, ldr, w_m = y
        w_r = pp * w_m
        iqs = (lr * lqs - lm * lqr) / det
        ids = (lr * lds - lm * ldr) / det
        iqr = (ls * lqr - lm * lqs) / det
        idr = (ls * ldr - lm * lds) / det
        d_lqs = vqs - p.rs_ohm * iqs - w_e * lds
        d_lds = vds - p.rs_ohm * ids + w_e * lqs
        d_lqr = -p.rr_ohm * iqr - (w_e - w_r) * ldr
        d_ldr = -p.rr_ohm * idr + (w_e - w_r) * lqr
        t_em = 1.5 * pp * lm * (iqs * idr - ids * iqr)
        shaft_hz = w_m / (2.0 * np.pi)
        t_drag = float(load(shaft_hz))
        if w_m > 1e-3:
            t_drag += p.mechanical_loss_w(abs(shaft_hz)) / w_m
        return np.array([d_lqs, d_lds, d_lqr, d_ldr, (t_em - t_drag) / inertia_kgm2])

    t_eval = np.linspace(0.0, t_end_s, n_eval)
    sol = solve_ivp(
        rhs, (0.0, t_end_s), np.zeros(5), t_eval=t_eval, rtol=rtol, atol=atol, method="LSODA"
    )
    if not sol.success:
        raise RuntimeError(f"d-q integration failed: {sol.message}")

    lqs, lds, lqr, ldr, w_m = sol.y
    iqs = (lr * lqs - lm * lqr) / det
    ids = (lr * lds - lm * ldr) / det
    iqr = (ls * lqr - lm * lqs) / det
    idr = (ls * ldr - lm * lds) / det
    t_em = 1.5 * pp * lm * (iqs * idr - ids * iqr)
    # Amplitude-invariant transform: |i_dq| is the peak phase current.
    i_rms = np.sqrt(iqs**2 + ids**2) / np.sqrt(2.0)
    return {
        "t_s": np.asarray(sol.t, dtype=np.float64),
        "shaft_hz": w_m / (2.0 * np.pi),
        "slip": (f / pp - w_m / (2.0 * np.pi)) / (f / pp),
        "torque_em_nm": t_em,
        "stator_current_a": i_rms,
    }


def dq_steady_state_check(
    p: ElectricalParams,
    plate: Nameplate,
    slip: float,
    *,
    t_settle_s: float = 6.0,
) -> Dict[str, float]:
    """Agreement between :func:`run_up_dq` at a held speed and :func:`steady_state`.

    Holds the rotor at the requested slip by integrating the d-q model with an
    enormous inertia and a load torque that exactly balances -- in practice, by
    integrating with the speed state frozen. Returns the relative differences in
    torque and current.

    This is the cross-check that makes both implementations trustworthy: they are
    derived from different formulations (phasor circuit vs. two-axis state space)
    and agreement to a fraction of a percent is strong evidence that neither has
    a factor-of-3, factor-of-2 or per-phase/per-machine error in it. Those are
    *the* errors that machine models have, and they are invisible in a single
    implementation because every number still looks plausible.

    The residual disagreement is not zero: the d-q model has no core-loss branch.
    With ``r_core_ohm = inf`` the two should agree to integrator tolerance.
    """
    f = plate.line_hz
    sync_hz = f / plate.pole_pairs
    target_hz = sync_hz * (1.0 - slip)
    w_rated = 2.0 * np.pi * plate.line_hz
    lls, llr, lm = p.xls_ohm / w_rated, p.xlr_ohm / w_rated, p.xm_ohm / w_rated
    ls, lr = lls + lm, llr + lm
    det = ls * lr - lm * lm
    v_phase = plate.rated_voltage_v / np.sqrt(3.0) if plate.connection == "Y" else plate.rated_voltage_v
    vqs = np.sqrt(2.0) * v_phase
    w_e = 2.0 * np.pi * f
    w_r = plate.pole_pairs * 2.0 * np.pi * target_hz

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        lqs, lds, lqr, ldr = y
        iqs = (lr * lqs - lm * lqr) / det
        ids = (lr * lds - lm * ldr) / det
        iqr = (ls * lqr - lm * lqs) / det
        idr = (ls * ldr - lm * lds) / det
        return np.array(
            [
                vqs - p.rs_ohm * iqs - w_e * lds,
                0.0 - p.rs_ohm * ids + w_e * lqs,
                -p.rr_ohm * iqr - (w_e - w_r) * ldr,
                -p.rr_ohm * idr + (w_e - w_r) * lqr,
            ]
        )

    sol = solve_ivp(rhs, (0.0, t_settle_s), np.zeros(4), rtol=1e-10, atol=1e-12, method="LSODA")
    if not sol.success:
        raise RuntimeError(f"d-q settle failed: {sol.message}")
    lqs, lds, lqr, ldr = sol.y[:, -1]
    iqs = (lr * lqs - lm * lqr) / det
    ids = (lr * lds - lm * ldr) / det
    iqr = (ls * lqr - lm * lqs) / det
    idr = (ls * ldr - lm * lds) / det
    t_dq = 1.5 * plate.pole_pairs * lm * (iqs * idr - ids * iqr)
    i_dq = float(np.hypot(iqs, ids) / np.sqrt(2.0))

    ss = steady_state(p, plate, slip)
    return {
        "slip": float(slip),
        "torque_dq_nm": float(t_dq),
        "torque_circuit_nm": float(ss.torque_em_nm),
        "torque_rel_err": float(abs(t_dq - ss.torque_em_nm) / abs(ss.torque_em_nm)),
        "current_dq_a": i_dq,
        "current_circuit_a": float(ss.stator_current_a),
        "current_rel_err": float(abs(i_dq - ss.stator_current_a) / ss.stator_current_a),
    }
