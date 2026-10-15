r"""Bearing defect excitation and the structural transfer path.

``pdmcore.dsp.bearing`` already owns the *kinematics* -- BPFO, BPFI, BSF, FTF
from the geometry and the shaft speed. Nothing here duplicates that; it is
imported. What this module adds is the step from a defect to a **measurable
surface vibration**, which is the step a digital twin has to make and the step
where the honesty lives.

The chain, and what is known about each link
--------------------------------------------
==========================================  ==================================
link                                        identifiable from field data?
==========================================  ==================================
defect geometry -> contact force pulse      no (needs the spall dimensions)
pulse train rate                            **yes, exactly** -- kinematics
pulse train jitter                           yes, 1-2 %, from line broadening
load-zone modulation depth                   yes, from sideband ratios
force -> casing acceleration (receptance)   **no. Needs an impact test.**
mount and sensor response                    partly, from a calibration sheet
==========================================  ==================================

Two links in that table are unknown and they are **multiplied together** into
the one number a sensor reports. So:

.. math::

    a_{\text{measured}}(t) = \underbrace{G(f)}_{\text{unknown}} *
                             \underbrace{\sum_k F_k\,p(t - t_k)}_{\text{unknown scale,
                             known rate}}

Only the product is observable. The rate is exact, the jitter and the modulation
are observable, the **scale is not, at any information tier.** That single fact
decides what a bearing twin can be used for:

* **Can** predict which frequencies to look at, how the sidebands should be
  arranged, which demodulation band carries the energy, and how the envelope
  spectrum should change as a defect spreads from a point spall to a raceway.
* **Cannot** predict an amplitude in g or a velocity in mm/s, and therefore
  cannot predict an ISO 20816 zone, and therefore must not be used to set an
  absolute alarm threshold.
* **Can** still predict ratios and trends, because the unknown gain cancels in
  a ratio taken at the same operating point with the same sensor.

What this deliberately does not model
-------------------------------------
* Hertzian contact mechanics. The force pulse is parameterised (peak, duration),
  not derived from spall depth and raceway curvature. Deriving it would add two
  more unknowns and no observability, and would make the twin look more
  physical while being exactly as uncertain.
* Non-stationary slip. Modelled as per-interval jitter, which reproduces the
  observed 1-2 % line broadening but not the cage-slip correlation structure.
* The second resonance path through the shaft and coupling. One path.
* Lubrication film and its temperature dependence, which is a real and large
  effect on impact transmission and would need a temperature-dependent
  receptance -- i.e. a product of two unidentifiable things.

Relationship to ``pdmcore.sim.machine``
---------------------------------------
The simulator rings impact trains through a resonance map in exactly this shape
(``_ring_impulses``), but it then **normalises the result to a target peak in g**
chosen from a severity scalar:

    ``target_peak = 0.12*sqrt(sev) + 4.0*sev**1.4``

That is the right design for a signal simulator -- it guarantees a stated crest
factor for CI -- and it is precisely what a twin may not do, because it hides the
unidentifiable receptance inside a tuning constant with no units. Here the
receptance is an explicit parameter in g/N (:class:`PathMode.gain_g_per_n`) and
the twin refuses absolute predictions until it has been measured.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..dsp.bearing import BearingGeometry, fault_frequencies
from .params import DefectParams, PathParams

__all__ = [
    "force_pulse",
    "defect_force_train",
    "path_impulse_response",
    "casing_acceleration",
    "most_loaded_contact_force_n",
    "defect_force_prior_n",
    "amplitude_gain_degeneracy",
]


def force_pulse(duration_s: float, fs_hz: float, peak_n: float = 1.0) -> np.ndarray:
    """A single half-sine contact-force pulse, sampled.

    A half-sine rather than a rectangle or a delta. The pulse width sets the
    *spectral envelope* of the excitation -- a pulse of duration ``tau`` has its
    first null at ``1/tau`` -- and that envelope is what decides which structural
    resonance receives the energy. A delta would excite every mode equally and
    would make the twin's demodulation-band prediction meaningless; a rectangle
    has far too much high-frequency content. 50-200 us is the range measured for
    rolling-element impacts, which puts the first null at 5-20 kHz, comfortably
    above the 2-8 kHz resonances that bearing diagnosis uses -- so in practice
    the band choice is set by the *structure*, not by the defect, and the twin
    says so.
    """
    if duration_s <= 0 or fs_hz <= 0:
        raise ValueError("duration_s and fs_hz must be positive")
    n = max(2, int(round(duration_s * fs_hz)))
    t = np.arange(n, dtype=np.float64) / fs_hz
    return peak_n * np.sin(np.pi * np.clip(t / duration_s, 0.0, 1.0))


def defect_force_train(
    defect: DefectParams,
    bearing: BearingGeometry,
    shaft_hz: float,
    *,
    n: int,
    fs_hz: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Force history in newtons from a localised defect.

    The pulse rate is the kinematic frequency for ``defect.mode``, taken from
    :func:`pdmcore.dsp.bearing.fault_frequencies` -- not recomputed here. The
    load-zone modulation frequency is the one that the diagnosis side already
    assumes (shaft rate for an inner-race defect, cage rate for a
    rolling-element defect), which is deliberate: the twin and the diagnosis
    engine must agree about the modulation structure or the twin cannot be used
    to validate the engine.
    """
    if n < 2 or fs_hz <= 0:
        raise ValueError("n must be >= 2 and fs_hz positive")
    rng = rng or np.random.default_rng(0)
    ff = fault_frequencies(bearing, shaft_hz)
    rate_hz, mod_hz = {
        "outer_race": (ff.bpfo_hz, 0.0),
        "inner_race": (ff.bpfi_hz, ff.shaft_hz),
        "rolling_element": (ff.bsf_hz, ff.ftf_hz),
        "cage": (ff.ftf_hz, 0.0),
    }[defect.mode]
    out = np.zeros(n, dtype=np.float64)
    if defect.peak_force_n <= 0 or rate_hz <= 0:
        return out
    pulse = force_pulse(defect.pulse_duration_s, fs_hz, defect.peak_force_n)
    period = fs_hz / rate_hz
    if period < pulse.size:
        raise ValueError(
            f"pulse duration {defect.pulse_duration_s*1e6:.0f} us is longer than the "
            f"{1e6/rate_hz:.0f} us between rolling-element passes at {rate_hz:.1f} Hz; "
            "the defect model's pulses would overlap, which is a distributed defect, "
            "not a localised one"
        )
    pos = rng.uniform(0.0, period)
    phase0 = rng.uniform(0.0, 2.0 * np.pi)
    while pos < n - pulse.size:
        amp = 1.0
        if defect.load_zone_depth > 0 and mod_hz > 0:
            amp *= 1.0 + defect.load_zone_depth * np.sin(
                2.0 * np.pi * mod_hz * pos / fs_hz + phase0
            )
        i = int(pos)
        out[i : i + pulse.size] += max(0.0, amp) * pulse
        pos += period * (1.0 + defect.slip_pct / 100.0 * rng.standard_normal())
    return out


def path_impulse_response(
    path: PathParams, fs_hz: float, *, decay_to: float = 1e-3, max_s: float = 0.25
) -> np.ndarray:
    r"""Impulse response of the force-to-casing-acceleration path, in g per N.s.

    Sum of decaying sinusoids, one per :class:`PathMode`, normalised so that the
    **steady-state magnitude response at each mode's resonance** equals that
    mode's ``gain_g_per_n``. That normalisation is the entire reason this
    function exists rather than reusing ``pdmcore.sim.machine._resonance_ir``:
    the simulator's version has a unit-free ``gain`` and is later rescaled to a
    target peak, so its impulse response has no calibration. Without the
    normalisation, ``gain_g_per_n`` would not be a receptance and the twin's one
    honest statement about absolute amplitude -- that it cannot make one --
    would be unfalsifiable rather than true.

    For a single-degree-of-freedom oscillator excited by a force and observed in
    acceleration, the peak receptance at resonance is :math:`Q/k_{\text{eff}}`
    in displacement terms; rather than carry an effective stiffness the mode is
    parameterised directly by its peak acceleration receptance and the impulse
    response scaled to match, which is checked numerically in
    ``core/tests/test_twin_path.py``.
    """
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    if not path.modes:
        raise ValueError(
            "the path has no structural modes, so it cannot ring an impact. "
            "An impact-driven prediction from a mode-free path would be silently zero"
        )
    n_total = 8
    irs = []
    for m in path.modes:
        if m.freq_hz >= fs_hz / 2.0:
            continue
        zeta = 1.0 / (2.0 * m.q)
        wn = 2.0 * np.pi * m.freq_hz
        tau = 1.0 / (zeta * wn)
        n = int(min(fs_hz * tau * np.log(1.0 / decay_to), fs_hz * max_s))
        n = max(8, n)
        t = np.arange(n, dtype=np.float64) / fs_hz
        wd = wn * np.sqrt(1.0 - zeta**2)
        h = np.exp(-zeta * wn * t) * np.sin(wd * t)
        # Scale so that the DISCRETE-TIME frequency response of the convolution
        # kernel has magnitude exactly ``gain_g_per_n`` at the resonance:
        #   H(f) = sum_j h[j] exp(-2*pi*i*f*j/fs),  |H(f_res)| := gain.
        # Evaluated numerically rather than from the 1/(2*zeta*fs) asymptotic, so
        # that a coarse sample rate or a short truncation cannot silently change
        # the calibration.
        hf = float(np.abs(np.sum(h * np.exp(-2j * np.pi * m.freq_hz * t))))
        if hf <= 0:
            raise AssertionError("degenerate path mode normalisation")
        irs.append(h * (m.gain_g_per_n / hf))
        n_total = max(n_total, n)
    if not irs:
        raise ValueError(
            f"every path mode is at or above Nyquist ({fs_hz/2:.0f} Hz). "
            "The sensor cannot see this structure; a prediction would be zero for the "
            "wrong reason"
        )
    out = np.zeros(n_total, dtype=np.float64)
    for h in irs:
        out[: h.size] += h
    return out


def casing_acceleration(
    defect: DefectParams,
    bearing: BearingGeometry,
    path: PathParams,
    shaft_hz: float,
    *,
    n: int,
    fs_hz: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Casing acceleration in g from a bearing defect: force train through path.

    Linear in ``defect.peak_force_n`` and linear in every mode's
    ``gain_g_per_n``. That is not an implementation detail, it is the statement
    of the degeneracy: scaling the force up by 10 and every gain down by 10
    produces a bit-identical output. :func:`amplitude_gain_degeneracy` verifies
    it numerically rather than asserting it, because "the model is linear in
    both" is the kind of claim that is true of the equations and false of the
    code.
    """
    force = defect_force_train(defect, bearing, shaft_hz, n=n, fs_hz=fs_hz, rng=rng)
    h = path_impulse_response(path, fs_hz)
    y = np.convolve(force, h)[:n]
    if path.mount_corner_hz < fs_hz / 2.0:
        # Single-pole mount roll-off, same form as pdmcore.sim.machine's sensor
        # model so that a residual against the simulator is not dominated by a
        # different mount assumption.
        spec = np.fft.rfft(y)
        fr = np.fft.rfftfreq(y.size, d=1.0 / fs_hz)
        spec = spec / (1.0 + 1j * fr / path.mount_corner_hz)
        y = np.fft.irfft(spec, n=y.size)
    return y


def most_loaded_contact_force_n(radial_load_n: float, n_balls: int) -> float:
    """Stribeck's most-loaded rolling-element force for a radially loaded bearing.

    ``F_max ~ 4.37 * F_r / Z`` for a deep-groove ball bearing with nominal
    clearance (the classic Stribeck factor; it is ``4.06/Z`` for zero clearance
    and ``4.37/Z`` for normal clearance). The spall sees this force when it
    passes through the load zone, so it is the physical scale for
    ``DefectParams.peak_force_n`` -- and it is computable from the machine's
    weight and belt or coupling pull, which a plant does know.

    This is the one genuine lever on the force/gain degeneracy: the force is
    *bounded* by contact mechanics even though it is not observable. See
    :func:`defect_force_prior_n`.
    """
    if radial_load_n <= 0:
        raise ValueError("radial_load_n must be positive")
    if n_balls < 3:
        raise ValueError("n_balls must be >= 3")
    return float(4.37 * radial_load_n / n_balls)


def defect_force_prior_n(
    radial_load_n: float, n_balls: int, *, severity: float
) -> Tuple[float, float]:
    """``(lo, hi)`` newtons for the defect pulse peak, from contact mechanics.

    The impulsive force when a rolling element drops into a spall is a *fraction*
    of the static contact force -- the element unloads as it enters and reloads
    as it leaves, and the transient is driven by the rate of that change. Field
    and rig literature puts the impulsive peak between roughly 2 % of the
    contact load for an incipient pit and 100 % for a spall wide enough that the
    element loses contact entirely.

    So the prior is a factor of ~50 wide at fixed severity, and it is still the
    tightest constraint available on the force. Combined with the path gain's own
    factor-of-30 ignorance it leaves the *product* -- the only observable --
    uncertain by more than three orders of magnitude, which is why absolute
    amplitude prediction is refused rather than reported with error bars.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    f_max = most_loaded_contact_force_n(radial_load_n, n_balls)
    lo = 0.02 * f_max * max(severity, 1e-3)
    hi = 1.00 * f_max * max(severity, 1e-3)
    return float(lo), float(hi)


def amplitude_gain_degeneracy(
    defect: DefectParams,
    bearing: BearingGeometry,
    path: PathParams,
    shaft_hz: float,
    *,
    n: int = 8192,
    fs_hz: float = 25600.0,
    factor: float = 10.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Numerically verify that force and path gain enter only as a product.

    Generates the same realisation twice -- once with ``(F, G)`` and once with
    ``(factor*F, G/factor)`` -- and reports the maximum relative difference.
    A non-zero answer means the implementation has acquired a nonlinearity (a
    clip, a normalisation, a saturating sensor model) and that the twin's
    central honesty claim no longer follows from the code.

    This is written as a measurement rather than a test assertion because the
    number belongs in the results JSON: "the degeneracy is exact to 2e-16" is
    the evidence, and a passing test is only the absence of counter-evidence.
    """
    from dataclasses import replace

    a = casing_acceleration(
        defect, bearing, path, shaft_hz, n=n, fs_hz=fs_hz, rng=np.random.default_rng(seed)
    )
    d2 = replace(defect, peak_force_n=defect.peak_force_n * factor)
    p2 = replace(
        path,
        low_freq_gain_g_per_n=path.low_freq_gain_g_per_n / factor,
        modes=tuple(replace(m, gain_g_per_n=m.gain_g_per_n / factor) for m in path.modes),
    )
    b = casing_acceleration(
        d2, bearing, p2, shaft_hz, n=n, fs_hz=fs_hz, rng=np.random.default_rng(seed)
    )
    scale = float(np.max(np.abs(a)))
    if scale <= 0:
        raise ValueError("degenerate test: the reference signal is identically zero")
    return {
        "factor": float(factor),
        "max_abs_rel_diff": float(np.max(np.abs(a - b)) / scale),
        "rms_a_g": float(np.sqrt(np.mean(a**2))),
        "rms_b_g": float(np.sqrt(np.mean(b**2))),
        "rms_ratio": float(np.sqrt(np.mean(b**2) / np.mean(a**2))),
    }
