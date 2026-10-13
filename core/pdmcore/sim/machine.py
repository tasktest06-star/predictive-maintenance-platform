"""Physics-flavoured rotating-machine signal simulator.

Purpose
-------
Every per-kit prototype in this repository has to be demonstrable and
regression-testable *without* the hardware on the bench. This module generates
acceleration, current and temperature records whose spectral structure matches
what real sensors see, with exact ground truth attached. That makes it possible
to:

* assert in CI that the diagnosis engine finds a 0.3-severity outer-race defect
  and does not find one in a healthy record;
* size an edge feature budget against signals with realistic crest factors;
* produce the figures in the LaTeX decks reproducibly;
* give a reviewer a one-command demo on a laptop.

Honesty about what this is
--------------------------
This is a **signal model**, not a multibody dynamics model. It reproduces the
spectral *signatures* that diagnosis relies on -- impulse trains at kinematic
frequencies, resonance ringing, load-zone modulation, slip jitter, shaft-order
families, sensor noise and quantisation. It does not derive those from contact
mechanics. Models tuned only on simulated data must therefore be validated on
measured data (CWRU/MFPT/MIMII or the customer's own) before they go anywhere
near a work order. ``docs/03_model_cards.md`` records that gate.

What is modelled, and why each one matters diagnostically
---------------------------------------------------------
=====================  ===========================================================
fault                  signature produced here
=====================  ===========================================================
``unbalance``          1x radial, stable phase, rises with speed^2
``misalignment``       2x (and 3x) with strong axial content
``looseness``          0.5x/1.5x subharmonics plus a raised noise floor
``bearing_outer``      BPFO impulse train, little modulation
``bearing_inner``      BPFI impulse train, amplitude-modulated at 1x
``bearing_ball``       BSF impulse train, modulated at the cage rate
``bearing_cage``       low-frequency FTF train, broadband rise
``gear_tooth``         once-per-rev impulse + gear-mesh sidebands at 1x
``gear_wear``          elevated gear-mesh harmonics without sidebands
``cavitation``         broadband 2-10 kHz hiss, no discrete lines
``rotor_bar``          2x line-frequency vibration + current sidebands at 2sf
``belt``               belt-rate harmonics, non-integer order
``rub``                fractional subharmonics (1/2, 1/3) and clipping
=====================  ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsp.bearing import BEARING_CATALOG, BearingGeometry, fault_frequencies

__all__ = [
    "Resonance",
    "SensorSpec",
    "MachineSpec",
    "SimResult",
    "FAULT_MODES",
    "MACHINE_LIBRARY",
    "simulate",
    "degradation_series",
    "fleet_dataset",
]

FAULT_MODES = (
    "healthy",
    "unbalance",
    "misalignment",
    "looseness",
    "bearing_outer",
    "bearing_inner",
    "bearing_ball",
    "bearing_cage",
    "gear_tooth",
    "gear_wear",
    "cavitation",
    "rotor_bar",
    "belt",
    "rub",
)


@dataclass(frozen=True)
class Resonance:
    """A structural resonance that bearing impacts ring.

    ``q`` is the quality factor; machine structures typically land between 10
    (heavily damped cast iron with a soft mount) and 60 (a thin bracket). The
    resonance map is the main reason the *same* bearing defect looks completely
    different on two installations, and the reason a fixed demodulation band
    does not transfer between assets.
    """

    freq_hz: float
    q: float = 25.0
    gain: float = 1.0


@dataclass(frozen=True)
class SensorSpec:
    """Accelerometer front-end, defaulting to a wideband MEMS device.

    Defaults track the IIS3DWB used on the ST industrial node: +-16 g,
    26.7 kHz ODR, ~75 ug/sqrt(Hz) noise density, 16-bit output.
    """

    name: str = "IIS3DWB-like"
    full_scale_g: float = 16.0
    bits: int = 16
    noise_density_ug_rthz: float = 75.0
    bandwidth_hz: float = 6300.0
    sensitivity_error_pct: float = 2.0
    offset_mg: float = 5.0
    # Mounting transfer function: a magnet mount rolls off hard above ~2 kHz,
    # a stud mount reaches past 10 kHz. Modelled as a single-pole low-pass.
    mount_corner_hz: float = 8000.0
    # Fraction of the measured axis that leaks in from the orthogonal axes.
    cross_axis_pct: float = 2.0

    @property
    def lsb_g(self) -> float:
        return 2.0 * self.full_scale_g / (2**self.bits)

    def noise_rms_g(self, fs_hz: float) -> float:
        bw = min(self.bandwidth_hz, fs_hz / 2.0)
        return self.noise_density_ug_rthz * 1e-6 * float(np.sqrt(bw))


@dataclass
class MachineSpec:
    """Kinematics and structure of the asset under observation."""

    name: str = "generic_motor_pump"
    rpm: float = 1780.0
    fs_hz: float = 25600.0
    power_kw: float = 37.0
    line_hz: float = 60.0
    pole_pairs: int = 2
    bearing: BearingGeometry = field(default_factory=lambda: BEARING_CATALOG["SKF6205"])
    resonances: Tuple[Resonance, ...] = (
        Resonance(3200.0, q=22.0, gain=1.0),
        Resonance(7400.0, q=35.0, gain=0.55),
        Resonance(12800.0, q=45.0, gain=0.25),
    )
    sensor: SensorSpec = field(default_factory=SensorSpec)
    gear_teeth: Optional[int] = None
    n_blades: Optional[int] = None
    belt_ratio: Optional[float] = None
    # Baseline (healthy) amplitudes in g for equipment of this class. Chosen so
    # that a healthy unit sits in ISO 20816 zone A: 0.030 g peak at 29.7 Hz is
    # 1.1 mm/s RMS of velocity, comfortably inside the 1.4 mm/s A/B boundary for
    # a rigid-mounted group 2 machine. If these baselines put a healthy machine
    # in zone B, every alarm threshold downstream inherits the error.
    base_1x_g: float = 0.022
    base_2x_g: float = 0.010
    base_noise_g: float = 0.012
    # Speed stability. A DOL motor drifts ~0.1 %; a VFD under varying load 1-3 %.
    speed_drift_pct: float = 0.4
    speed_drift_hz: float = 0.17
    ambient_c: float = 32.0

    @property
    def shaft_hz(self) -> float:
        return self.rpm / 60.0

    @property
    def slip(self) -> float:
        """Induction-motor slip from nameplate speed vs synchronous speed."""
        sync_rpm = 120.0 * self.line_hz / (2 * self.pole_pairs)
        return float(max(0.0, (sync_rpm - self.rpm) / sync_rpm))


#: Preset machines spanning the asset classes the platform targets.
#:
#: Healthy baselines are set so each preset lands in ISO 20816 zone A when
#: healthy -- note that this requires *smaller* acceleration on slower machines,
#: because velocity (what ISO limits) is acceleration divided by frequency. The
#: slow agitator is included precisely because it breaks the default 10-1000 Hz
#: measurement band: its 1x line is at 6.6 Hz, so it must be measured over
#: 2-1000 Hz or it reads as perfectly smooth no matter how unbalanced it is.
MACHINE_LIBRARY: Dict[str, MachineSpec] = {
    "motor_pump_37kw": MachineSpec(),
    "fan_7kw_belt": MachineSpec(
        name="fan_7kw_belt",
        rpm=880.0,
        power_kw=7.5,
        bearing=BEARING_CATALOG["6204"],
        resonances=(Resonance(1900.0, q=15.0), Resonance(4600.0, q=28.0, gain=0.5)),
        n_blades=8,
        belt_ratio=2.37,
        base_1x_g=0.018,
        base_2x_g=0.007,
        base_noise_g=0.015,
        sensor=SensorSpec(name="ADXL355-like", full_scale_g=8.0, bits=20,
                          noise_density_ug_rthz=25.0, bandwidth_hz=1500.0, mount_corner_hz=2500.0),
    ),
    "gearbox_132kw": MachineSpec(
        name="gearbox_132kw",
        rpm=1480.0,
        power_kw=132.0,
        line_hz=50.0,
        bearing=BEARING_CATALOG["6308"],
        resonances=(Resonance(2600.0, q=18.0), Resonance(6100.0, q=30.0, gain=0.7),
                    Resonance(11200.0, q=40.0, gain=0.3)),
        gear_teeth=37,
        base_1x_g=0.025,
        base_2x_g=0.010,
        base_noise_g=0.025,
        speed_drift_pct=1.2,
    ),
    "slow_agitator_400rpm": MachineSpec(
        name="slow_agitator_400rpm",
        rpm=395.0,
        power_kw=18.5,
        bearing=BEARING_CATALOG["22220"],
        resonances=(Resonance(1200.0, q=12.0), Resonance(3400.0, q=25.0, gain=0.6)),
        base_1x_g=0.0062,
        base_2x_g=0.0022,
        base_noise_g=0.006,
        speed_drift_pct=0.2,
    ),
    "vfd_compressor_75kw": MachineSpec(
        name="vfd_compressor_75kw",
        rpm=2950.0,
        power_kw=75.0,
        line_hz=50.0,
        pole_pairs=1,
        bearing=BEARING_CATALOG["6206"],
        resonances=(Resonance(4100.0, q=26.0), Resonance(9300.0, q=38.0, gain=0.45)),
        base_1x_g=0.042,
        base_2x_g=0.013,
        base_noise_g=0.030,
        speed_drift_pct=2.5,
        speed_drift_hz=0.4,
    ),
}


@dataclass
class SimResult:
    """A simulated acquisition with ground truth."""

    accel_g: np.ndarray
    fs_hz: float
    spec: MachineSpec
    faults: Dict[str, float]
    truth: Dict[str, float]
    current_a: Optional[np.ndarray] = None
    temperature_c: Optional[float] = None
    speed_hz_profile: Optional[np.ndarray] = None
    speed_time_s: Optional[np.ndarray] = None

    @property
    def duration_s(self) -> float:
        return self.accel_g.size / self.fs_hz

    @property
    def label(self) -> str:
        """Dominant fault label, or ``"healthy"``."""
        active = {k: v for k, v in self.faults.items() if v > 0.0}
        return max(active, key=lambda k: active[k]) if active else "healthy"

    @property
    def severity(self) -> float:
        return float(max(self.faults.values())) if self.faults else 0.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _resonance_ir(res: Resonance, fs_hz: float, decay_to: float = 1e-3) -> np.ndarray:
    """Impulse response of a single resonance: decaying sinusoid."""
    zeta = 1.0 / (2.0 * max(1.0, res.q))
    wn = 2.0 * np.pi * res.freq_hz
    tau = 1.0 / (zeta * wn)
    n = int(min(fs_hz * tau * np.log(1.0 / decay_to), fs_hz * 0.25))
    n = max(8, n)
    t = np.arange(n, dtype=np.float64) / fs_hz
    wd = wn * np.sqrt(max(1e-9, 1.0 - zeta**2))
    return res.gain * np.exp(-zeta * wn * t) * np.sin(wd * t)


def _impulse_train(
    n: int,
    fs_hz: float,
    rate_hz: float,
    rng: np.random.Generator,
    *,
    slip_pct: float = 1.5,
    amp_jitter_pct: float = 20.0,
    modulation_hz: float = 0.0,
    modulation_depth: float = 0.0,
    modulation_phase: float = 0.0,
) -> np.ndarray:
    """Sparse impulse train with fractional-sample placement.

    * ``slip_pct`` randomises each inter-arrival interval. Real rolling elements
      slip, which is why bearing lines are slightly broadened and never exactly
      an integer order. Without this the simulated SES is unrealistically clean
      and models trained on it over-fit to needle-sharp lines.
    * ``modulation_*`` applies load-zone amplitude modulation: at 1x for an
      inner-race defect, at the cage rate for a rolling-element defect.
    """
    out = np.zeros(n, dtype=np.float64)
    if rate_hz <= 0:
        return out
    period = fs_hz / rate_hz
    pos = rng.uniform(0.0, period)
    while pos < n - 1:
        amp = 1.0 + amp_jitter_pct / 100.0 * rng.standard_normal()
        if modulation_depth > 0 and modulation_hz > 0:
            amp *= 1.0 + modulation_depth * np.sin(
                2.0 * np.pi * modulation_hz * pos / fs_hz + modulation_phase
            )
        amp = max(0.0, amp)
        i = int(np.floor(pos))
        frac = pos - i
        out[i] += amp * (1.0 - frac)
        out[i + 1] += amp * frac
        pos += period * (1.0 + slip_pct / 100.0 * rng.standard_normal())
    return out


def _ring_impulses(train: np.ndarray, spec: MachineSpec, fs_hz: float) -> np.ndarray:
    """Convolve an impulse train with the machine's resonance map."""
    out = np.zeros_like(train)
    for res in spec.resonances:
        if res.freq_hz >= fs_hz / 2.0:
            continue
        ir = _resonance_ir(res, fs_hz)
        out += np.convolve(train, ir, mode="same")
    return out


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise: structural background and sensor flicker."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n=n)
    std = np.std(out)
    return out / std if std > 0 else out


def _band_noise(n: int, fs_hz: float, lo_hz: float, hi_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited white noise, used for cavitation and lubrication hiss."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    spec[(f < lo_hz) | (f > hi_hz)] = 0.0
    out = np.fft.irfft(spec, n=n)
    std = np.std(out)
    return out / std if std > 0 else out


def _speed_profile(
    n: int, fs_hz: float, spec: MachineSpec, rng: np.random.Generator
) -> np.ndarray:
    """Instantaneous shaft frequency per sample."""
    t = np.arange(n, dtype=np.float64) / fs_hz
    f0 = spec.shaft_hz
    amp = f0 * spec.speed_drift_pct / 100.0
    wander = amp * np.sin(2.0 * np.pi * spec.speed_drift_hz * t + rng.uniform(0, 2 * np.pi))
    # Slow random walk on top, low-pass filtered so it does not add broadband noise.
    steps = rng.standard_normal(n) * amp * 0.02
    walk = np.cumsum(steps)
    if n > 64:
        k = min(n // 4, 4096)
        kernel = np.ones(k) / k
        walk = np.convolve(walk, kernel, mode="same")
    walk -= np.mean(walk)
    return f0 + wander + walk


def _phase_from_speed(speed_hz: np.ndarray, fs_hz: float) -> np.ndarray:
    """Integrate instantaneous frequency to phase in radians."""
    return 2.0 * np.pi * np.cumsum(speed_hz) / fs_hz


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def simulate(
    spec: Optional[MachineSpec] = None,
    faults: Optional[Dict[str, float]] = None,
    *,
    duration_s: float = 1.0,
    seed: Optional[int] = 0,
    with_current: bool = False,
    apply_sensor: bool = True,
    nuisance: bool = True,
) -> SimResult:
    """Generate one acquisition.

    Parameters
    ----------
    spec
        Machine description; defaults to ``MACHINE_LIBRARY["motor_pump_37kw"]``.
    faults
        ``{mode: severity}`` with severity in ``[0, 1]``. Multiple simultaneous
        faults are allowed and are what real machines actually present --
        a misaligned coupling destroys the bearing it is driving, so
        ``{"misalignment": 0.5, "bearing_inner": 0.3}`` is a realistic pairing
        and a good test of whether the diagnosis engine reports both.
    duration_s
        Record length. 1 s at 25.6 kHz is the platform's standard acquisition.
    apply_sensor
        Apply mount roll-off, noise, offset, gain error, clipping and ADC
        quantisation. Turn this off to inspect the underlying mechanical signal.
    nuisance
        Apply per-acquisition nuisance variation: load factor, resonance-gain
        drift and small speed offset. **Keep this on.** Without it, successive
        "healthy" acquisitions differ only by additive noise, so the
        commissioning MAD of a high-frequency band is unrealistically tiny and
        every robust z-score becomes hypersensitive -- a health index tuned
        against such data will fire on the first warm afternoon in the real
        plant. Real healthy variation comes from load, temperature, grease
        redistribution and mount relaxation, and it is 10-20 %, not 0.5 %.

    Returns
    -------
    SimResult
    """
    spec = spec or MACHINE_LIBRARY["motor_pump_37kw"]
    faults = {k: float(v) for k, v in (faults or {}).items() if v and v > 0}
    for mode in faults:
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {mode!r}; expected one of {FAULT_MODES}")
    rng = np.random.default_rng(seed)
    fs = float(spec.fs_hz)
    n = int(round(duration_s * fs))
    if n < 256:
        raise ValueError("duration too short; need >= 256 samples")

    # Per-acquisition nuisance variation. These three factors are what make the
    # healthy distribution realistically wide; see the `nuisance` docstring.
    if nuisance:
        load_factor = float(np.exp(rng.normal(0.0, 0.12)))        # +-12 % load
        resonance_gain = float(np.exp(rng.normal(0.0, 0.10)))     # mount/temperature
        speed_offset = 1.0 + 0.005 * rng.standard_normal()        # +-0.5 % speed
    else:
        load_factor = resonance_gain = speed_offset = 1.0
    spec = replace(spec, rpm=spec.rpm * speed_offset) if nuisance else spec

    speed = _speed_profile(n, fs, spec, rng)
    phase = _phase_from_speed(speed, fs)
    f0 = spec.shaft_hz
    ff = fault_frequencies(spec.bearing, f0)

    x = np.zeros(n, dtype=np.float64)

    # --- baseline healthy content -------------------------------------------
    sev_unb = faults.get("unbalance", 0.0)
    sev_mis = faults.get("misalignment", 0.0)
    sev_loose = faults.get("looseness", 0.0)

    # Unbalance force grows with speed^2; amplitude here grows ~linearly in
    # severity on top of the residual baseline unbalance every machine has.
    a1 = spec.base_1x_g * (1.0 + 6.0 * sev_unb)
    x += a1 * np.sin(phase + rng.uniform(0, 2 * np.pi))
    a2 = spec.base_2x_g * (1.0 + 9.0 * sev_mis)
    x += a2 * np.sin(2.0 * phase + rng.uniform(0, 2 * np.pi))
    if sev_mis > 0:
        x += 0.35 * a2 * sev_mis * np.sin(3.0 * phase + rng.uniform(0, 2 * np.pi))
    x += 0.25 * spec.base_2x_g * np.sin(3.0 * phase + rng.uniform(0, 2 * np.pi))

    # Electrical 2x line frequency, always present on an induction motor.
    t = np.arange(n, dtype=np.float64) / fs
    x += 0.4 * spec.base_2x_g * np.sin(2.0 * np.pi * 2.0 * spec.line_hz * t + rng.uniform(0, 2 * np.pi))

    # --- looseness: subharmonics + raised floor ------------------------------
    if sev_loose > 0:
        for sub in (0.5, 1.5, 2.5):
            x += sev_loose * spec.base_1x_g * 0.8 * np.sin(sub * phase + rng.uniform(0, 2 * np.pi))
        x += sev_loose * spec.base_noise_g * 3.0 * _pink_noise(n, rng)

    # --- bearing faults ------------------------------------------------------
    bearing_map = {
        "bearing_outer": (ff.bpfo_hz, 0.0, 0.0),
        "bearing_inner": (ff.bpfi_hz, f0, 0.65),
        "bearing_ball": (ff.bsf_hz, ff.ftf_hz, 0.5),
        "bearing_cage": (ff.ftf_hz, 0.0, 0.0),
    }
    for mode, (rate, mod_hz, mod_depth) in bearing_map.items():
        sev = faults.get(mode, 0.0)
        if sev <= 0 or rate <= 0:
            continue
        train = _impulse_train(
            n, fs, rate, rng,
            slip_pct=1.5 if mode != "bearing_cage" else 3.0,
            amp_jitter_pct=25.0,
            modulation_hz=mod_hz,
            modulation_depth=mod_depth,
            modulation_phase=rng.uniform(0, 2 * np.pi),
        )
        rung = _ring_impulses(train, spec, fs)
        peak = np.max(np.abs(rung))
        if peak > 0:
            # Must vanish as severity -> 0: a constant offset here injects a
            # detectable impulse train into a "healthy" record, which then
            # teaches every downstream threshold that BPFO is normal.
            # sev 0.1 -> ~0.14 g (incipient spall, just above the floor);
            # sev 1.0 -> ~4 g (a bearing about to seize).
            target_peak = 0.12 * np.sqrt(sev) + 4.0 * sev**1.4
            x += rung / peak * target_peak * spec.base_1x_g / 0.030
        # A spreading defect also raises the broadband floor.
        x += sev * spec.base_noise_g * 1.5 * _band_noise(n, fs, 1000.0, min(10000.0, fs / 2 * 0.9), rng)

    # --- gear ---------------------------------------------------------------
    if spec.gear_teeth:
        gmf_phase = spec.gear_teeth * phase
        gear_base = spec.base_1x_g * 0.6
        sev_wear = faults.get("gear_wear", 0.0)
        sev_tooth = faults.get("gear_tooth", 0.0)
        x += gear_base * (1.0 + 4.0 * sev_wear) * np.sin(gmf_phase + rng.uniform(0, 2 * np.pi))
        x += gear_base * 0.4 * (1.0 + 6.0 * sev_wear) * np.sin(2.0 * gmf_phase + rng.uniform(0, 2 * np.pi))
        if sev_tooth > 0:
            # Sidebands at +-1x around the mesh line.
            depth = 0.8 * sev_tooth
            x += gear_base * depth * np.sin(gmf_phase) * np.sin(phase)
            # Plus a once-per-revolution impact from the damaged tooth.
            train = _impulse_train(n, fs, f0, rng, slip_pct=0.2, amp_jitter_pct=10.0)
            rung = _ring_impulses(train, spec, fs)
            peak = np.max(np.abs(rung))
            if peak > 0:
                x += rung / peak * (0.1 + 3.0 * sev_tooth) * spec.base_1x_g / 0.030

    # --- blade pass / belt ---------------------------------------------------
    if spec.n_blades:
        x += spec.base_1x_g * 0.3 * np.sin(spec.n_blades * phase + rng.uniform(0, 2 * np.pi))
    sev_belt = faults.get("belt", 0.0)
    if sev_belt > 0 and spec.belt_ratio:
        belt_hz = f0 / spec.belt_ratio
        for h in (1, 2, 3, 4):
            x += sev_belt * spec.base_1x_g * 0.9 / h * np.sin(
                2.0 * np.pi * belt_hz * h * t + rng.uniform(0, 2 * np.pi)
            )

    # --- cavitation ----------------------------------------------------------
    sev_cav = faults.get("cavitation", 0.0)
    if sev_cav > 0:
        hi = min(10000.0, fs / 2.0 * 0.9)
        x += sev_cav * spec.base_noise_g * 12.0 * _band_noise(n, fs, 2000.0, hi, rng)

    # --- rub: fractional subharmonics and clipping ---------------------------
    sev_rub = faults.get("rub", 0.0)
    if sev_rub > 0:
        for sub in (1 / 2, 1 / 3, 2 / 3, 1 / 4):
            x += sev_rub * spec.base_1x_g * 0.7 * np.sin(sub * phase + rng.uniform(0, 2 * np.pi))
        clip = (1.0 - 0.45 * sev_rub) * np.max(np.abs(x))
        if clip > 0:
            x = np.clip(x, -clip, clip)

    # --- broken rotor bar ----------------------------------------------------
    sev_bar = faults.get("rotor_bar", 0.0)
    if sev_bar > 0:
        x += sev_bar * spec.base_2x_g * 4.0 * np.sin(
            2.0 * np.pi * 2.0 * spec.line_hz * t + rng.uniform(0, 2 * np.pi)
        )

    # --- nuisance scaling ----------------------------------------------------
    # Load scales the whole mechanical signal; resonance gain scales only the
    # impact-driven high-frequency content, which is how a temperature change
    # or a relaxing mount actually shows up.
    x *= load_factor * resonance_gain

    # --- background ----------------------------------------------------------
    x += spec.base_noise_g * _pink_noise(n, rng)

    truth = {
        "load_factor": load_factor,
        "resonance_gain": resonance_gain,
        "shaft_hz": f0,
        "line_hz": spec.line_hz,
        "slip": spec.slip,
        **{k: v for k, v in ff.as_dict().items() if k != "shaft_hz"},
    }
    if spec.gear_teeth:
        truth["gmf_hz"] = spec.gear_teeth * f0
    if spec.belt_ratio:
        truth["belt_hz"] = f0 / spec.belt_ratio

    current = _simulate_current(spec, faults, n, fs, rng) if with_current else None

    if apply_sensor:
        x = _apply_sensor(x, spec, fs, rng)

    # Bearing and winding temperature rise with fault severity and load.
    severity_total = float(sum(faults.values()))
    temperature = spec.ambient_c + 18.0 + 22.0 * min(1.0, severity_total) + rng.normal(0.0, 0.4)

    step = max(1, n // 256)
    return SimResult(
        accel_g=x,
        fs_hz=fs,
        spec=spec,
        faults=faults,
        truth=truth,
        current_a=current,
        temperature_c=float(temperature),
        speed_hz_profile=speed[::step].copy(),
        speed_time_s=t[::step].copy(),
    )


def _simulate_current(
    spec: MachineSpec, faults: Dict[str, float], n: int, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """Single-phase stator current with MCSA fault signatures.

    Broken rotor bars and rotor eccentricity appear as sidebands around the line
    frequency at ``(1 +- 2ks) f_line``. Current sensing is attractive because a
    split-core CT needs no machine contact and no mounting -- the electrical
    monitoring product in ``plans/07_electrical_monitoring.md`` is built on this.
    """
    t = np.arange(n, dtype=np.float64) / fs
    f_line = spec.line_hz
    rated_a = spec.power_kw * 1000.0 / (400.0 * np.sqrt(3) * 0.88)
    i = rated_a * np.sqrt(2.0) * np.sin(2.0 * np.pi * f_line * t)
    # Normal harmonic content of an induction motor on a stiff supply.
    for h, rel in ((3, 0.02), (5, 0.035), (7, 0.018), (11, 0.008)):
        i += rated_a * np.sqrt(2.0) * rel * np.sin(2.0 * np.pi * f_line * h * t + rng.uniform(0, 2 * np.pi))
    s = max(spec.slip, 1e-3)
    sev_bar = faults.get("rotor_bar", 0.0)
    if sev_bar > 0:
        for k in (1, 2, 3):
            amp = rated_a * np.sqrt(2.0) * 0.02 * sev_bar / k
            for sign in (-1.0, 1.0):
                f_sb = f_line * (1.0 + sign * 2.0 * k * s)
                i += amp * np.sin(2.0 * np.pi * f_sb * t + rng.uniform(0, 2 * np.pi))
    # Mechanical unbalance and misalignment modulate the load torque at 1x.
    mech = faults.get("unbalance", 0.0) + faults.get("misalignment", 0.0)
    if mech > 0:
        for sign in (-1.0, 1.0):
            f_sb = f_line + sign * spec.shaft_hz
            i += rated_a * np.sqrt(2.0) * 0.012 * mech * np.sin(
                2.0 * np.pi * f_sb * t + rng.uniform(0, 2 * np.pi)
            )
    i += rated_a * 0.004 * rng.standard_normal(n)
    return i


def _apply_sensor(x: np.ndarray, spec: MachineSpec, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Mount roll-off, noise, gain/offset error, clipping, quantisation."""
    s = spec.sensor
    # Mount transfer function: single-pole low-pass, applied in the frequency domain.
    if s.mount_corner_hz < fs / 2.0:
        spec_f = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.size, d=1.0 / fs)
        spec_f *= 1.0 / (1.0 + 1j * f / s.mount_corner_hz)
        x = np.fft.irfft(spec_f, n=x.size)
    # Anti-alias / sensor bandwidth limit.
    if s.bandwidth_hz < fs / 2.0:
        spec_f = np.fft.rfft(x)
        f = np.fft.rfftfreq(x.size, d=1.0 / fs)
        spec_f *= 1.0 / np.sqrt(1.0 + (f / s.bandwidth_hz) ** 4)  # 2-pole magnitude
        x = np.fft.irfft(spec_f, n=x.size)
    # Cross-axis leakage: a scaled, decorrelated copy of the signal.
    if s.cross_axis_pct > 0:
        leak = np.roll(x, int(fs * 1e-3)) * (s.cross_axis_pct / 100.0)
        x = x + leak
    # Electronic noise.
    x = x + s.noise_rms_g(fs) * rng.standard_normal(x.size)
    # Gain and offset error, constant per unit -- the reason two sensors on the
    # same machine never read identically and why thresholds must be per-asset.
    gain = 1.0 + s.sensitivity_error_pct / 100.0 * rng.standard_normal()
    x = x * gain + s.offset_mg * 1e-3
    # Full-scale clipping, then ADC quantisation.
    x = np.clip(x, -s.full_scale_g, s.full_scale_g)
    lsb = s.lsb_g
    return np.round(x / lsb) * lsb


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def degradation_series(
    spec: Optional[MachineSpec] = None,
    *,
    mode: str = "bearing_outer",
    n_points: int = 120,
    duration_s: float = 0.5,
    days: float = 90.0,
    knee_fraction: float = 0.55,
    final_severity: float = 0.95,
    seed: Optional[int] = 0,
) -> List[SimResult]:
    """A run-to-failure series for trending, alarm and RUL testing.

    Severity follows the two-stage profile that bearing life actually shows: a
    long flat "healthy" stage, then an exponential knee. ``knee_fraction`` sets
    where the knee starts as a fraction of the series. Each result carries
    ``truth["t_days"]``, ``truth["severity"]`` and ``truth["rul_days"]`` so a
    prognostics model can be scored against the real remaining life.

    This profile is what makes the non-monotonic behaviour of kurtosis visible:
    kurtosis peaks in the middle of the knee and then *falls* as the defect
    spreads, while RMS keeps climbing. Any health index that trusts kurtosis
    alone will show the machine recovering just before it fails.
    """
    spec = spec or MACHINE_LIBRARY["motor_pump_37kw"]
    if mode not in FAULT_MODES:
        raise ValueError(f"unknown fault mode {mode!r}")
    out: List[SimResult] = []
    for i in range(n_points):
        frac = i / max(1, n_points - 1)
        if frac < knee_fraction:
            sev = 0.02 * (frac / max(1e-9, knee_fraction))
        else:
            u = (frac - knee_fraction) / max(1e-9, 1.0 - knee_fraction)
            sev = 0.02 + (final_severity - 0.02) * (np.exp(3.2 * u) - 1.0) / (np.exp(3.2) - 1.0)
        t_days = frac * days
        res = simulate(
            spec,
            {mode: float(sev)} if sev > 0 else None,
            duration_s=duration_s,
            seed=None if seed is None else seed + i,
        )
        res.truth["t_days"] = float(t_days)
        res.truth["severity"] = float(sev)
        res.truth["rul_days"] = float(days - t_days)
        out.append(res)
    return out


def fleet_dataset(
    *,
    machines: Optional[Sequence[str]] = None,
    modes: Optional[Sequence[str]] = None,
    severities: Sequence[float] = (0.15, 0.35, 0.6, 0.9),
    n_healthy_per_machine: int = 24,
    n_per_fault: int = 8,
    duration_s: float = 0.5,
    seed: int = 0,
    rpm_jitter_pct: float = 3.0,
) -> List[SimResult]:
    """A labelled multi-asset dataset for classifier training and evaluation.

    ``rpm_jitter_pct`` varies the nameplate speed per unit so that models cannot
    cheat by memorising an exact shaft frequency -- the simulated analogue of
    the data-leakage problem that inflates reported accuracy on CWRU when the
    same bearing appears in train and test. Each result carries
    ``truth["unit_id"]`` so :mod:`pdmcore.eval.splits` can group by unit.
    """
    machine_keys = list(machines) if machines else list(MACHINE_LIBRARY)
    fault_modes = list(modes) if modes else [
        "unbalance", "misalignment", "looseness",
        "bearing_outer", "bearing_inner", "bearing_ball", "cavitation",
    ]
    rng = np.random.default_rng(seed)
    out: List[SimResult] = []
    counter = 0
    for mk in machine_keys:
        base = MACHINE_LIBRARY[mk]
        n_units = 3
        for unit in range(n_units):
            jitter = 1.0 + rpm_jitter_pct / 100.0 * rng.standard_normal()
            unit_spec = replace(base, rpm=float(base.rpm * jitter), name=f"{mk}_u{unit}")
            unit_id = f"{mk}_u{unit}"
            for _ in range(n_healthy_per_machine):
                counter += 1
                r = simulate(unit_spec, None, duration_s=duration_s, seed=seed + counter)
                r.truth["unit_id"] = unit_id
                r.truth["machine_class"] = mk
                out.append(r)
            for mode in fault_modes:
                if mode == "gear_tooth" and not unit_spec.gear_teeth:
                    continue
                if mode == "belt" and not unit_spec.belt_ratio:
                    continue
                for sev in severities:
                    for _ in range(max(1, n_per_fault // len(severities))):
                        counter += 1
                        r = simulate(
                            unit_spec, {mode: float(sev)}, duration_s=duration_s, seed=seed + counter
                        )
                        r.truth["unit_id"] = unit_id
                        r.truth["machine_class"] = mk
                        out.append(r)
    return out
