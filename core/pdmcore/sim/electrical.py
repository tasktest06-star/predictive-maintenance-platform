"""Three-phase stator current and voltage, derived from two coupling paths.

What this module replaces, and why
----------------------------------
The first version of the current channel (``_simulate_current`` in
``machine.py`` before this branch) wrote the sidebands down directly: a
broken-bar sideband was ``0.02 * severity`` of the fundamental, full stop. Three
properties of that model made it unusable for the question this branch asks:

* **The healthy class had no variance.** Measured over 40 acquisitions with
  nuisance on, the current RMS coefficient of variation was 0.004 % against
  10.4 % for acceleration, because the load factor was applied to the mechanical
  signal and not to the current. A detector evaluated on that data reports an
  AUC of 1.0 for any signal at all, and the comparison with vibration is decided
  by the generator rather than by the physics.
* **No mechanical fault reached the current except through hard-coded terms.**
  Bearing coupling was structurally absent -- so "current cannot detect bearing
  faults" would have been a statement about the simulator, not about motors.
* **Sideband amplitude did not depend on load**, while sideband *position* was
  driven by a slip that the speed nuisance could drive to exactly zero (a
  synchronous induction motor).

The replacement derives every current signature from one of two paths, each
with parameters that are physical quantities a reader can disagree with:

The torque path
---------------
Anything that modulates the shaft torque at ``fm`` must be supplied by the
motor. The motor is a first-order system in speed:

    J d(dw)/dt = dT - k dw,      k = dT/dw = T_rated / (s_rated w_sync)

so the speed ripple is ``dw = dT / (k + j 2pi fm J)`` and the corner is at
``fc = k / (2 pi J)``. Below ``fc`` the torque-speed curve is stiff, the speed
barely moves, and the motor supplies the ripple as *current*: the relative
current modulation equals the relative torque modulation. Above ``fc`` the
inertia absorbs it and the response rolls off as ``1/f``.

    dI/I (both sidebands) = (dT / T_load) * 1/sqrt(1 + (fm/fc)^2)

For the reference 37 kW 4-pole machine ``fc`` is 50 Hz, which is the single most
consequential number in this file: **a load oscillation at 1 Hz couples into the
current essentially perfectly, and a bearing impact train at 107 Hz is already
past the corner.**

This model is deliberately *optimistic* about high-frequency coupling: it
ignores the rotor-flux lag (rotor time constant 0.1-0.3 s, i.e. a second corner
near 1 Hz) which attenuates everything above a few hertz much harder. The
optimism is the point -- the branch's conclusion is that current cannot see
bearing faults, and a model that over-states the coupling makes that conclusion
harder to reach, not easier.

The air-gap path
----------------
A radial rotor displacement ``d`` against an air gap ``g`` modulates the
permeance by ``eps = d/g``. To first order the flux acquires components at
``f1 +- fm`` with relative amplitude ``eps/2`` -- but the flux is held by the
applied voltage, so what actually changes is the *magnetising* current, and the
perturbing field has ``p +- 1`` pole pairs rather than ``p``, for which the
stator winding's distribution factor is roughly an order of magnitude smaller:

    dI/I (both sidebands) = (d/g) * (I_mag/I_1) * k_winding,   k_winding ~ 0.1

``k_winding`` is the weakest number in this file. It is a single coefficient
standing in for the winding's harmonic coupling, and it moves every air-gap
result by however much it is wrong. ``mcsa/prototype/coupling_sweep.py`` sweeps
it over two decades and reports the value at which each verdict flips, because
that is the only honest way to use a number nobody here has measured.

Rotor asymmetry
---------------
Broken bars are not a modulation of a healthy machine, so they get the standard
closed form rather than either path: for ``n`` broken bars out of ``N`` with
``p`` pole pairs,

    I_lsb / I_1 = n / (2 N p) * (I_torque / I_1)

which gives -44 dB for one bar of 40 on a 4-pole machine and -32 dB for four.
Those are the textbook field figures (one bar at -45 to -50 dB, several bars
above -35 dB), which is the check that the coefficient is right. The upper
sideband follows from the ``2 s f1`` torque pulsation through the torque path,
so its size relative to the lower one is ``H_mech(2 s f1)`` -- essentially 1 at
rated slip, i.e. near-symmetric sidebands. That is a *prediction*, and
``mcsa/prototype/real_data.py`` checks it against measured data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CurrentSensorSpec",
    "SupplySpec",
    "CouplingModel",
    "ElectricalResult",
    "simulate_electrical",
    "mech_transfer",
]


@dataclass(frozen=True)
class CurrentSensorSpec:
    """Split-core CT into a burden resistor into an ADC.

    The chain is the current-domain twin of the piezo chain in
    ``edge/pdmedge/calibration.py``: ``primary A -> secondary A (turns ratio)
    -> volts (burden) -> ADC codes``. Defaults describe a 100:5 class-1
    split-core CT with a 2.5 ohm burden into a 3.3 V single-supply ADC, which is
    what a panel device actually contains.

    ``channel_gain_error_pct`` is the number that matters most and is easiest to
    overlook: a *common* gain error cancels in every dB-relative measurement
    this branch makes, but a gain error that *differs between phases* creates a
    negative-sequence current out of nothing -- i.e. a stator fault that is not
    there. Class 1 CTs are +-1 % each, so 1 % of differential error is normal.
    """

    name: str = "100:5 split-core class 1 + 0.22 ohm burden"
    primary_rating_a: float = 100.0
    turns_ratio: float = 20.0          # 100:5
    #: Sized so that full scale is ~150 A primary on a 1.65 V bipolar input:
    #: 0.22 ohm / 20 turns = 11 mV per primary ampere. A 2.5 ohm "metering"
    #: burden would put 12.5 V across the input and clip everything, which is
    #: the most common way a CT front end is got wrong.
    burden_ohm: float = 0.22
    adc_bits: int = 16
    adc_full_scale_v: float = 1.65     # bipolar about a 1.65 V reference
    ratio_error_pct: float = 1.0       # common-mode: cancels in dB-relative work
    channel_gain_error_pct: float = 1.0
    #: Input-referred noise of the front end, in volts per root hertz.
    noise_density_v_rthz: float = 30e-9
    bandwidth_hz: float = 5000.0

    @property
    def volts_per_amp(self) -> float:
        """Primary amperes to volts at the ADC."""
        return self.burden_ohm / self.turns_ratio

    @property
    def full_scale_a(self) -> float:
        return self.adc_full_scale_v / self.volts_per_amp

    @property
    def lsb_a(self) -> float:
        return 2.0 * self.full_scale_a / (2 ** self.adc_bits)

    def noise_rms_a(self, fs_hz: float) -> float:
        bw = min(self.bandwidth_hz, fs_hz / 2.0)
        return self.noise_density_v_rthz * float(np.sqrt(bw)) / self.volts_per_amp


@dataclass(frozen=True)
class SupplySpec:
    """The supply, and the four things about it that defeat MCSA."""

    #: Negative-sequence voltage as a percentage of positive sequence. A plant
    #: bus is typically 0.3-1.5 %; above 1 % IEC 60034-26 starts derating.
    unbalance_pct: float = 0.4
    #: Voltage harmonics as (order, amplitude relative to the fundamental).
    harmonics: Tuple[Tuple[int, float], ...] = ((3, 0.005), (5, 0.025), (7, 0.015), (11, 0.006))
    #: Peak-to-peak supply frequency wander, Hz. ENTSO-E / NERC grids hold
    #: 50/60 Hz to about +-20 mHz on a quiet day and +-100 mHz under stress.
    #: This is what puts a floor on useful frequency resolution: wander of
    #: 20 mHz smears every line by 20 mHz no matter how long the record is.
    freq_wander_hz: float = 0.02
    freq_wander_rate_hz: float = 0.03
    #: Ratio of negative- to positive-sequence impedance. 0.17 is typical for a
    #: cage motor (locked-rotor impedance), and it is why a 1 % voltage
    #: unbalance produces ~6 % negative-sequence current.
    neg_seq_impedance_pu: float = 0.17
    vfd: bool = False
    carrier_hz: float = 4000.0
    #: Carrier sideband amplitude relative to the fundamental.
    carrier_rel: float = 0.04
    #: Load torque oscillation: amplitude as a fraction of the operating torque.
    load_osc_hz: float = 0.0
    load_osc_frac: float = 0.0
    #: Broadband load turbulence, as a fraction of operating torque, 1/f shaped.
    load_turbulence_frac: float = 0.004
    #: Sampling rate for the current channels. ``None`` means "the same rate as
    #: the accelerometer", which is what the pre-existing single-phase model
    #: did; a panel device would use 5-10 kHz.
    current_fs_hz: Optional[float] = None
    sensor: CurrentSensorSpec = field(default_factory=CurrentSensorSpec)


@dataclass(frozen=True)
class CouplingModel:
    """How a mechanical fault reaches the stator current. The uncertain part."""

    #: Nominal radial air gap, mm. 0.5-1.0 mm for a 20-150 kW 4-pole machine.
    airgap_mm: float = 0.6
    #: Rotor + load inertia, kg m^2.
    inertia_kg_m2: float = 0.30
    #: Stator-winding coupling factor for a (p +- 1)-pole-pair perturbing
    #: field. THE uncertain coefficient; see the module docstring.
    k_winding: float = 0.10
    #: Magnetising current as a fraction of rated current.
    mag_current_frac: float = 0.30
    #: Residual eccentricity of a healthy machine, as a fraction of the gap.
    residual_ecc_frac: float = 0.06
    #: Radial displacement at severity 1.0, micrometres, per mechanical mode.
    disp_um: Dict[str, float] = field(default_factory=lambda: {
        "bearing_outer": 25.0,
        "bearing_inner": 25.0,
        "bearing_ball": 15.0,
        "bearing_cage": 10.0,
        "unbalance": 120.0,
        "misalignment": 60.0,
        "looseness": 80.0,
    })
    #: Torque ripple at severity 1.0, as a fraction of operating torque.
    torque_frac: Dict[str, float] = field(default_factory=lambda: {
        "bearing_outer": 0.005,
        "bearing_inner": 0.005,
        "bearing_ball": 0.003,
        "bearing_cage": 0.002,
        "unbalance": 0.010,
        "misalignment": 0.020,
        "looseness": 0.015,
        "gear_tooth": 0.030,
        "gear_wear": 0.010,
        "cavitation": 0.020,
        "belt": 0.020,
        "rub": 0.030,
    })
    #: Broken bars at severity 1.0, and the rotor's bar count.
    broken_bars_at_full: float = 4.0
    rotor_bars: int = 40
    #: Shorted-turn fraction of one phase at severity 1.0.
    shorted_turn_frac_at_full: float = 0.02

    def as_dict(self) -> Dict[str, object]:
        return {
            "airgap_mm": self.airgap_mm,
            "inertia_kg_m2": self.inertia_kg_m2,
            "k_winding": self.k_winding,
            "mag_current_frac": self.mag_current_frac,
            "residual_ecc_frac": self.residual_ecc_frac,
            "disp_um": dict(self.disp_um),
            "torque_frac": dict(self.torque_frac),
            "broken_bars_at_full": self.broken_bars_at_full,
            "rotor_bars": self.rotor_bars,
            "shorted_turn_frac_at_full": self.shorted_turn_frac_at_full,
        }


def mech_transfer(f_hz: float, *, torque_slope: float, inertia_kg_m2: float) -> float:
    """``1/sqrt(1 + (f/fc)^2)`` with ``fc = k/(2 pi J)``.

    The whole torque path in one function, so that the corner frequency is
    computed in one place and can be printed next to any result that depends on
    it.
    """
    fc = torque_slope / (2.0 * np.pi * max(inertia_kg_m2, 1e-9))
    return float(1.0 / np.sqrt(1.0 + (float(f_hz) / max(fc, 1e-9)) ** 2))


@dataclass
class ElectricalResult:
    """Three-phase current and voltage with the operating point attached."""

    current_abc_a: np.ndarray            # (3, n), amperes at the primary
    voltage_abc_v: np.ndarray            # (3, n), volts phase-neutral
    fs_hz: float
    truth: Dict[str, float]

    @property
    def phase_a(self) -> np.ndarray:
        return self.current_abc_a[0]


def _wander_phase(n: int, fs: float, f1: float, supply: SupplySpec,
                  rng: np.random.Generator) -> np.ndarray:
    """Supply phase with slow frequency wander integrated into it."""
    t = np.arange(n, dtype=np.float64) / fs
    if supply.freq_wander_hz > 0:
        dev = 0.5 * supply.freq_wander_hz * np.sin(
            2.0 * np.pi * supply.freq_wander_rate_hz * t + rng.uniform(0, 2 * np.pi))
    else:
        dev = np.zeros(n)
    return 2.0 * np.pi * np.cumsum(f1 + dev) / fs


def _am_term(t: np.ndarray, fm: float, rel_sideband: float,
             rng: np.random.Generator) -> np.ndarray:
    """AM term whose **each** sideband is ``rel_sideband`` of the carrier.

    Amplitude modulation with index ``m`` puts ``m/2`` in each sideband, so the
    index applied here is ``2 * rel_sideband``. Everything in this module is
    quoted per sideband, because that is what the MCSA literature quotes
    thresholds in (-45 dB for one broken bar is *one* sideband) and mixing the
    two conventions is a 6 dB error in the only number that matters.
    """
    if rel_sideband <= 0 or fm <= 0:
        return np.zeros_like(t)
    return 2.0 * rel_sideband * np.sin(2.0 * np.pi * fm * t + rng.uniform(0, 2 * np.pi))


def simulate_electrical(
    *,
    n: int,
    fs_hz: float,
    line_hz: float,
    pole_pairs: int,
    power_kw: float,
    shaft_hz: float,
    load_frac: float,
    faults: Dict[str, float],
    bearing_hz: Dict[str, float],
    supply: SupplySpec,
    coupling: CouplingModel,
    rng: np.random.Generator,
    apply_sensor: bool = True,
) -> ElectricalResult:
    """Generate three-phase current and voltage for one operating point.

    ``load_frac`` is the shaft torque as a fraction of rated. It sets the slip,
    the torque current and therefore the amplitude of every rotor-side
    signature, which is the dependence the previous model did not have.
    """
    fs = float(fs_hz)
    t = np.arange(n, dtype=np.float64) / fs
    f1 = float(line_hz)
    p = int(pole_pairs)
    w_sync = 2.0 * np.pi * f1 / p
    slip = max(1.0 - p * float(shaft_hz) / f1, 1e-6)

    # --- operating point ---------------------------------------------------
    v_phase = 400.0 / np.sqrt(3.0)
    i_rated = power_kw * 1000.0 / (400.0 * np.sqrt(3.0) * 0.88)
    i_mag = coupling.mag_current_frac * i_rated
    i_tq = float(np.sqrt(max(i_rated ** 2 - i_mag ** 2, 0.0))) * float(load_frac)
    i1 = float(np.hypot(i_mag, i_tq))
    t_rated = power_kw * 1000.0 / (w_sync * (1.0 - slip))
    t_load = t_rated * float(load_frac)
    # Torque-speed slope at the operating point, N.m per rad/s.
    torque_slope = t_rated / max(slip * w_sync, 1e-9)
    fc_hz = torque_slope / (2.0 * np.pi * coupling.inertia_kg_m2)

    def h_mech(f: float) -> float:
        return mech_transfer(f, torque_slope=torque_slope,
                             inertia_kg_m2=coupling.inertia_kg_m2)

    # --- amplitude modulation of the fundamental ---------------------------
    # Every path below contributes an AM term; they are summed before being
    # applied so that two faults at the same frequency interfere, which is what
    # physically happens and is the reason a confounder can mask a fault.
    am = np.zeros(n, dtype=np.float64)
    truth: Dict[str, float] = {
        "slip": slip,
        "load_frac": float(load_frac),
        "i_fundamental_a": i1,
        "i_mag_a": i_mag,
        "i_torque_a": i_tq,
        "t_load_nm": t_load,
        "torque_slope_nm_s_rad": torque_slope,
        "mech_corner_hz": fc_hz,
        "line_hz": f1,
        "shaft_hz": float(shaft_hz),
    }

    def add_torque_path(name: str, fm: float, sev: float) -> float:
        """Relative amplitude of ONE sideband at ``f1 +- fm``."""
        frac = coupling.torque_frac.get(name, 0.0) * sev
        if frac <= 0 or fm <= 0 or load_frac <= 0:
            return 0.0
        rel = 0.5 * frac * h_mech(fm) * (i_tq / i1)
        am[:] += _am_term(t, fm, rel, rng)
        truth[f"sb_torque_{name}"] = rel
        return rel

    def add_airgap_path(name: str, fm: float, sev: float) -> float:
        """Relative amplitude of ONE sideband at ``f1 +- fm``."""
        d_um = coupling.disp_um.get(name, 0.0) * sev
        if d_um <= 0 or fm <= 0:
            return 0.0
        eps = d_um * 1e-3 / coupling.airgap_mm
        rel = 0.5 * eps * (i_mag / i1) * coupling.k_winding
        am[:] += _am_term(t, fm, rel, rng)
        truth[f"sb_airgap_{name}"] = rel
        return rel

    fr = float(shaft_hz)

    # Residual eccentricity: present on every healthy machine, which is why
    # f1 +- fr is never actually absent and a detector keyed on its presence
    # alone fires on everything.
    res_eps = coupling.residual_ecc_frac * float(np.exp(rng.normal(0.0, 0.25)))
    sb_res = 0.5 * res_eps * (i_mag / i1) * coupling.k_winding
    am += _am_term(t, fr, sb_res, rng)
    truth["sb_residual_ecc"] = sb_res
    truth["residual_ecc_frac"] = res_eps

    # Mechanical faults, through both paths.
    for name, fm in (("unbalance", fr), ("misalignment", 2.0 * fr),
                     ("looseness", fr), ("rub", 0.5 * fr)):
        sev = faults.get(name, 0.0)
        if sev > 0:
            add_torque_path(name, fm, sev)
            add_airgap_path(name, fm, sev)
    for name, key in (("bearing_outer", "bpfo_hz"), ("bearing_inner", "bpfi_hz"),
                      ("bearing_ball", "bsf_hz"), ("bearing_cage", "ftf_hz")):
        sev = faults.get(name, 0.0)
        fm = float(bearing_hz.get(key, 0.0))
        if sev > 0 and fm > 0:
            add_torque_path(name, fm, sev)
            add_airgap_path(name, fm, sev)
    # Gear and belt faults modulate the torque at the shaft rate; cavitation is
    # broadband and produces no discrete current line at all, which is listed
    # here with a zero rather than omitted so that the absence is deliberate.
    for name, fm in (("gear_tooth", fr), ("gear_wear", fr),
                     ("belt", fr), ("cavitation", 0.0)):
        sev = faults.get(name, 0.0)
        if sev > 0 and fm > 0:
            add_torque_path(name, fm, sev)

    # Eccentricity as its own fault mode: dynamic and mixed produce f1 +- k fr;
    # static alone does not (its gap perturbation is stationary in the stator
    # frame), so it is modelled through the 2 f1 force only -- which is on the
    # vibration channel, in machine.py.
    for name in ("eccentricity_dynamic", "eccentricity_mixed"):
        sev = faults.get(name, 0.0)
        if sev <= 0:
            continue
        eps = 0.35 * sev            # severity 1.0 == 35 % of the air gap
        rel = 0.5 * eps * (i_mag / i1) * coupling.k_winding
        am += _am_term(t, fr, rel, rng)
        truth[f"sb_{name}"] = rel
        truth[f"ecc_frac_{name}"] = eps
        if name == "eccentricity_mixed":
            am += _am_term(t, 2.0 * fr, 0.4 * rel, rng)

    # Load oscillation: the confounder, through the same path as a real fault.
    if supply.load_osc_frac > 0 and supply.load_osc_hz > 0:
        rel = 0.5 * supply.load_osc_frac * h_mech(supply.load_osc_hz) * (i_tq / i1)
        am += _am_term(t, supply.load_osc_hz, rel, rng)
        truth["sb_load_osc"] = rel
    # Broadband load turbulence, 1/f: the real near-fundamental noise floor.
    if supply.load_turbulence_frac > 0:
        white = rng.standard_normal(n)
        sp = np.fft.rfft(white)
        f = np.arange(sp.size, dtype=np.float64)
        f[0] = 1.0
        sp /= np.sqrt(f)
        turb = np.fft.irfft(sp, n=n)
        sd = float(np.std(turb))
        if sd > 0:
            turb = turb / sd
        am += supply.load_turbulence_frac * (i_tq / i1) * turb
        truth["load_turbulence_frac"] = supply.load_turbulence_frac

    # --- rotor asymmetry: the closed form, not a modulation ----------------
    sev_bar = faults.get("rotor_bar", 0.0)
    lsb_rel = usb_rel = 0.0
    if sev_bar > 0:
        n_broken = coupling.broken_bars_at_full * sev_bar
        lsb_rel = (n_broken / (2.0 * coupling.rotor_bars * p)) * (i_tq / i1)
        usb_rel = lsb_rel * h_mech(2.0 * slip * f1)
        truth["rotor_bar_n_broken"] = n_broken
        truth["rotor_bar_lsb_rel"] = lsb_rel
        truth["rotor_bar_usb_rel"] = usb_rel

    # --- synthesis ---------------------------------------------------------
    theta = _wander_phase(n, fs, f1, supply, rng)
    pf_angle = np.arccos(min(0.99, 0.55 + 0.35 * float(load_frac)))
    env = i1 * (1.0 + am)
    cur = np.empty((3, n), dtype=np.float64)
    vol = np.empty((3, n), dtype=np.float64)
    # Negative-sequence current from the supply unbalance and from any stator
    # asymmetry, through the same impedance -- which is exactly why the two are
    # not separable from current alone.
    v_neg = supply.unbalance_pct / 100.0
    stator_sev = faults.get("stator_asymmetry", 0.0)
    turn_frac = coupling.shorted_turn_frac_at_full * stator_sev
    i_neg_rel = (v_neg + turn_frac) / supply.neg_seq_impedance_pu
    truth["v_unbalance_pct"] = supply.unbalance_pct
    truth["shorted_turn_frac"] = turn_frac
    truth["i_neg_rel_predicted"] = i_neg_rel

    for ph in range(3):
        shift = -2.0 * np.pi * ph / 3.0
        vol[ph] = v_phase * np.sqrt(2.0) * np.sin(theta + shift)
        # Negative sequence rotates the other way: +shift instead of -shift.
        vol[ph] += v_phase * np.sqrt(2.0) * v_neg * np.sin(theta - shift + 0.7)
        for order, rel in supply.harmonics:
            vol[ph] += v_phase * np.sqrt(2.0) * rel * np.sin(order * (theta + shift) + 0.3 * order)
        cur[ph] = env * np.sqrt(2.0) * np.sin(theta + shift - pf_angle)
        cur[ph] += i1 * np.sqrt(2.0) * i_neg_rel * np.sin(theta - shift + 0.7 - pf_angle)
        for order, rel in supply.harmonics:
            # Harmonic current through the (much smaller) harmonic impedance.
            cur[ph] += i1 * np.sqrt(2.0) * rel * 1.8 * np.sin(
                order * (theta + shift) + 0.3 * order - pf_angle)
        if sev_bar > 0:
            cur[ph] += i1 * np.sqrt(2.0) * lsb_rel * np.sin(
                (1.0 - 2.0 * slip) * theta + shift + 0.2)
            cur[ph] += i1 * np.sqrt(2.0) * usb_rel * np.sin(
                (1.0 + 2.0 * slip) * theta + shift - 0.4)
            # k = 2 members, an order of magnitude down.
            cur[ph] += i1 * np.sqrt(2.0) * 0.25 * lsb_rel * np.sin(
                (1.0 - 4.0 * slip) * theta + shift + 1.1)
            cur[ph] += i1 * np.sqrt(2.0) * 0.25 * usb_rel * np.sin(
                (1.0 + 4.0 * slip) * theta + shift - 1.3)
        if supply.vfd:
            # Carrier plus its first pair of sidebands at the output frequency.
            for fsw, rel in ((supply.carrier_hz, supply.carrier_rel),
                             (supply.carrier_hz - f1, 0.4 * supply.carrier_rel),
                             (supply.carrier_hz + f1, 0.4 * supply.carrier_rel),
                             (2.0 * supply.carrier_hz, 0.3 * supply.carrier_rel)):
                if 0 < fsw < 0.48 * fs:
                    cur[ph] += i1 * np.sqrt(2.0) * rel * np.sin(
                        2.0 * np.pi * fsw * t + rng.uniform(0, 2 * np.pi))

    truth["mech_corner_hz"] = fc_hz
    if apply_sensor:
        cur = _apply_ct(cur, supply.sensor, fs, rng, truth)
    return ElectricalResult(current_abc_a=cur, voltage_abc_v=vol, fs_hz=fs, truth=truth)


def _apply_ct(cur: np.ndarray, sensor: CurrentSensorSpec, fs: float,
              rng: np.random.Generator, truth: Dict[str, float]) -> np.ndarray:
    """CT ratio error, per-channel gain mismatch, front-end noise, quantisation.

    The per-channel gain error is applied *per channel* and recorded, because
    the false negative-sequence current it produces is the floor on the
    stator-fault indicator and has nothing to do with the motor.
    """
    out = np.empty_like(cur)
    common = 1.0 + sensor.ratio_error_pct / 100.0 * rng.standard_normal()
    gains = []
    for ph in range(cur.shape[0]):
        g = common * (1.0 + sensor.channel_gain_error_pct / 100.0 * rng.standard_normal())
        gains.append(g)
        x = cur[ph] * g
        x = x + sensor.noise_rms_a(fs) * rng.standard_normal(x.size)
        x = np.clip(x, -sensor.full_scale_a, sensor.full_scale_a)
        out[ph] = np.round(x / sensor.lsb_a) * sensor.lsb_a
    truth["ct_common_gain"] = float(common)
    truth["ct_channel_gain_spread_pct"] = float(
        100.0 * (max(gains) - min(gains)) / float(np.mean(gains)))
    truth["ct_lsb_a"] = float(sensor.lsb_a)
    truth["ct_noise_rms_a"] = float(sensor.noise_rms_a(fs))
    return out
