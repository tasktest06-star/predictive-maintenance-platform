"""Bearing and gear characteristic frequencies, and evidence scoring.

Kinematic frequencies (shaft rotating, outer race stationary), with ``n`` balls,
ball diameter ``d``, pitch diameter ``D``, contact angle ``alpha`` and shaft
frequency ``fr``:

.. math::

    \\mathrm{FTF}  &= \\frac{f_r}{2}\\left(1 - \\frac{d}{D}\\cos\\alpha\\right) \\\\
    \\mathrm{BPFO} &= \\frac{n f_r}{2}\\left(1 - \\frac{d}{D}\\cos\\alpha\\right) = n\\,\\mathrm{FTF} \\\\
    \\mathrm{BPFI} &= \\frac{n f_r}{2}\\left(1 + \\frac{d}{D}\\cos\\alpha\\right) \\\\
    \\mathrm{BSF}  &= \\frac{D f_r}{2d}\\left(1 - \\left(\\frac{d}{D}\\cos\\alpha\\right)^2\\right)

Two practical points that the formulas hide:

1. **These frequencies are not exact.** Rolling elements slip, so the observed
   rate wanders by 1-2 % and is *not* an exact multiple of shaft speed. Any
   matcher must use a relative tolerance (``rel_tol``), and a fault line that
   falls exactly on a shaft harmonic should be treated with suspicion.
2. **Sidebands carry the diagnosis.** An outer-race defect sits in a fixed load
   zone so BPFO appears with harmonics but little modulation. An inner-race
   defect passes in and out of the load zone once per revolution, so BPFI
   appears with ``+-fr`` sidebands. A rolling-element defect is modulated by the
   cage, so BSF appears with ``+-FTF`` sidebands. :func:`sideband_ratio` scores
   this, and it is what separates "a bearing is faulty" from "which part of the
   bearing is faulty".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .spectrum import Spectrum

__all__ = [
    "BearingGeometry",
    "FaultFrequencies",
    "BEARING_CATALOG",
    "fault_frequencies",
    "gear_mesh_frequency",
    "harmonic_energy_ratio",
    "sideband_ratio",
    "match_tones",
    "bearing_evidence",
]


@dataclass(frozen=True)
class BearingGeometry:
    """Rolling-element bearing geometry.

    ``ball_d`` and ``pitch_d`` must share units (any units -- only the ratio is
    used), which is why catalogue entries can stay in inches as published.
    """

    name: str
    n_balls: int
    ball_d: float
    pitch_d: float
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_balls < 3:
            raise ValueError("n_balls must be >= 3")
        if not 0 < self.ball_d < self.pitch_d:
            raise ValueError("require 0 < ball_d < pitch_d")
        if self.ratio >= 1.0:
            raise ValueError("ball_d/pitch_d must be < 1")

    @property
    def ratio(self) -> float:
        """``(d/D) cos(alpha)`` -- the only geometric quantity that matters."""
        return (self.ball_d / self.pitch_d) * float(np.cos(np.radians(self.contact_angle_deg)))


# Geometries published for the bearings used by the two most-cited open
# benchmark datasets, so a model trained on them can be re-derived exactly.
BEARING_CATALOG: Dict[str, BearingGeometry] = {
    # CWRU drive-end bearing (SKF 6205-2RS JEM). Dimensions in inches.
    "SKF6205": BearingGeometry("SKF6205", n_balls=9, ball_d=0.3126, pitch_d=1.537),
    # CWRU fan-end bearing (SKF 6203-2RS JEM).
    "SKF6203": BearingGeometry("SKF6203", n_balls=9, ball_d=0.2656, pitch_d=1.122),
    # MFPT test rig bearing (NICE, 8 rollers, 0.311 in roller, 1.175 in pitch).
    "NICE_MFPT": BearingGeometry("NICE_MFPT", n_balls=8, ball_d=0.311, pitch_d=1.175),
    # Generic deep-groove ball bearings commonly found on 1-50 kW motors.
    "6204": BearingGeometry("6204", n_balls=8, ball_d=0.3150, pitch_d=1.2205),
    "6206": BearingGeometry("6206", n_balls=9, ball_d=0.3543, pitch_d=1.7717),
    "6308": BearingGeometry("6308", n_balls=8, ball_d=0.5906, pitch_d=2.4409),
    # Spherical roller bearing with a non-zero contact angle, to exercise alpha.
    "22220": BearingGeometry("22220", n_balls=14, ball_d=0.9449, pitch_d=5.5118, contact_angle_deg=12.0),
}


@dataclass(frozen=True)
class FaultFrequencies:
    """Characteristic frequencies in Hz for a given shaft speed."""

    shaft_hz: float
    ftf_hz: float
    bpfo_hz: float
    bpfi_hz: float
    bsf_hz: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "shaft_hz": self.shaft_hz,
            "ftf_hz": self.ftf_hz,
            "bpfo_hz": self.bpfo_hz,
            "bpfi_hz": self.bpfi_hz,
            "bsf_hz": self.bsf_hz,
        }

    def as_orders(self) -> Dict[str, float]:
        """Same frequencies expressed in shaft orders (speed-independent)."""
        if self.shaft_hz <= 0:
            raise ValueError("shaft_hz must be positive to express orders")
        return {k: v / self.shaft_hz for k, v in self.as_dict().items()}


def fault_frequencies(geom: BearingGeometry, shaft_hz: float) -> FaultFrequencies:
    """Kinematic fault frequencies for ``geom`` at ``shaft_hz``."""
    if shaft_hz < 0:
        raise ValueError("shaft_hz must be non-negative")
    r = geom.ratio
    n = geom.n_balls
    ftf = shaft_hz / 2.0 * (1.0 - r)
    return FaultFrequencies(
        shaft_hz=float(shaft_hz),
        ftf_hz=float(ftf),
        bpfo_hz=float(n * ftf),
        bpfi_hz=float(n * shaft_hz / 2.0 * (1.0 + r)),
        bsf_hz=float(shaft_hz * geom.pitch_d / (2.0 * geom.ball_d) * (1.0 - r * r)),
    )


def gear_mesh_frequency(shaft_hz: float, n_teeth: int) -> float:
    """Gear mesh frequency. Sidebands appear at ``+- shaft_hz`` for a local
    tooth defect and at ``+- 2*shaft_hz`` for eccentricity."""
    if n_teeth < 1:
        raise ValueError("n_teeth must be >= 1")
    return float(shaft_hz * n_teeth)


def _amp(spec: Spectrum) -> np.ndarray:
    """Amplitude-like view of a spectrum regardless of its scaling."""
    v = np.asarray(spec.value, dtype=np.float64)
    return np.sqrt(np.maximum(v, 0.0)) if spec.scaling in ("psd", "power") else v


def _peak_near(spec: Spectrum, amp: np.ndarray, f_hz: float, tol_hz: float) -> float:
    if f_hz <= 0 or f_hz > spec.freq_hz[-1]:
        return 0.0
    m = np.abs(spec.freq_hz - f_hz) <= max(tol_hz, spec.df_hz)
    return float(np.max(amp[m])) if np.any(m) else 0.0


def _local_floor(spec: Spectrum, amp: np.ndarray, f_hz: float, half_width_hz: float, exclude_hz: float) -> float:
    """Median amplitude around ``f_hz``, excluding the peak region itself.

    Excluding ``+-exclude_hz`` matters: if the line being tested is strong it
    drags its own "floor" estimate up and the SNR collapses, which would make
    the detector *less* sensitive the worse the fault gets.
    """
    d = np.abs(spec.freq_hz - f_hz)
    m = (d <= half_width_hz) & (d > exclude_hz)
    if not np.any(m):
        m = d <= half_width_hz
    if not np.any(m):
        return float(np.median(amp)) if amp.size else 0.0
    return float(np.median(amp[m]))


def harmonic_energy_ratio(
    spec: Spectrum,
    f0_hz: float,
    *,
    n_harmonics: int = 6,
    rel_tol: float = 0.02,
    min_tol_hz: Optional[float] = None,
    floor_width_hz: float = 120.0,
    match_snr: float = 4.0,
) -> Dict[str, float]:
    """Score a harmonic family at ``f0_hz``.

    The statistic is the **mean per-harmonic SNR** against a local median floor,
    not the summed amplitude. Two reasons, both learned the hard way:

    * Summed amplitude divided by one floor estimate scales with
      ``n_harmonics``, so the number means nothing on its own and every
      threshold has to be re-tuned when the harmonic count changes.
    * Taking the maximum bin inside a tolerance window biases noise upward
      (max of *k* samples, not one). With the tolerance windows used here a
      pure-noise family scores ``ratio`` around 2-3, so a threshold of 4-6 is a
      real detection and the number is interpretable as "harmonics stand 5x
      above the local floor".

    Returns
    -------
    dict
        ``f0_hz``, ``amplitude`` (summed matched amplitude, for trending),
        ``floor`` (median local amplitude), ``ratio`` (mean per-harmonic SNR --
        the detection statistic), ``ratio_db``, ``peak_snr``, ``n_matched``
        (harmonics with SNR >= ``match_snr``), ``n_tested``, ``harmonics`` and
        ``snr`` per harmonic.
    """
    amp = _amp(spec)
    harmonics: list = []
    snrs: list = []
    matched = 0
    total = 0.0
    floors: list = []
    for h in range(1, n_harmonics + 1):
        fh = f0_hz * h
        if fh <= 0 or fh > spec.freq_hz[-1]:
            harmonics.append(0.0)
            snrs.append(0.0)
            continue
        tol = max(rel_tol * fh, min_tol_hz if min_tol_hz is not None else 1.5 * spec.df_hz)
        a = _peak_near(spec, amp, fh, tol)
        fl = _local_floor(spec, amp, fh, floor_width_hz, tol)
        harmonics.append(a)
        floors.append(fl)
        total += a
        snr = a / fl if fl > 0 else 0.0
        snrs.append(snr)
        if snr >= match_snr:
            matched += 1
    tested = [s for s, hv in zip(snrs, harmonics) if hv > 0.0]
    ratio = float(np.mean(tested)) if tested else 0.0
    floor = float(np.median(floors)) if floors else 0.0
    return {
        "f0_hz": float(f0_hz),
        "amplitude": float(total),
        "floor": floor,
        "ratio": ratio,
        "ratio_db": float(20.0 * np.log10(ratio)) if ratio > 0 else -120.0,
        "peak_snr": float(max(snrs)) if snrs else 0.0,
        "n_matched": float(matched),
        "n_tested": float(len(tested)),
        "harmonics": [float(v) for v in harmonics],
        "snr": [float(v) for v in snrs],
    }


def sideband_ratio(
    spec: Spectrum,
    carrier_hz: float,
    modulation_hz: float,
    *,
    n_sidebands: int = 2,
    rel_tol: float = 0.02,
    floor_width_hz: float = 120.0,
    min_carrier_snr: float = 3.0,
) -> Dict[str, float]:
    """Energy in ``carrier +- k*modulation`` relative to the carrier.

    High for inner-race defects (modulation = shaft rate) and rolling-element
    defects (modulation = cage rate); low for outer-race defects.

    ``min_carrier_snr`` guards the division: with no carrier present the ratio
    of two noise samples is unbounded and happily reports 6.0, which reads as
    "strongly modulated" on a perfectly healthy machine. If the carrier does
    not stand clear of its local floor the result is reported as ``0.0`` with
    ``valid=0.0`` so downstream rules can tell "not modulated" apart from "no
    carrier to modulate".
    """
    amp = _amp(spec)
    tol_c = max(rel_tol * carrier_hz, 1.5 * spec.df_hz)
    carrier = _peak_near(spec, amp, carrier_hz, tol_c)
    floor = _local_floor(spec, amp, carrier_hz, floor_width_hz, tol_c)
    carrier_snr = carrier / floor if floor > 0 else 0.0
    if carrier_snr < min_carrier_snr:
        return {
            "carrier_hz": float(carrier_hz),
            "modulation_hz": float(modulation_hz),
            "carrier_amp": float(carrier),
            "carrier_snr": float(carrier_snr),
            "sideband_amp": 0.0,
            "ratio": 0.0,
            "n_sidebands_found": 0.0,
            "valid": 0.0,
        }
    side = 0.0
    found = 0
    for k in range(1, n_sidebands + 1):
        for sign in (-1.0, 1.0):
            f = carrier_hz + sign * k * modulation_hz
            if f <= 0 or f > spec.freq_hz[-1]:
                continue
            tol = max(rel_tol * f, 1.5 * spec.df_hz)
            a = _peak_near(spec, amp, f, tol)
            fl = _local_floor(spec, amp, f, floor_width_hz, tol)
            # Only count a sideband that is itself above the noise.
            if fl > 0 and a >= 2.5 * fl:
                side += a
                found += 1
    return {
        "carrier_hz": float(carrier_hz),
        "modulation_hz": float(modulation_hz),
        "carrier_amp": float(carrier),
        "carrier_snr": float(carrier_snr),
        "sideband_amp": float(side),
        "ratio": float(side / carrier) if carrier > 0 else 0.0,
        "n_sidebands_found": float(found),
        "valid": 1.0,
    }


def match_tones(
    tones: Sequence[dict],
    targets: Dict[str, float],
    *,
    rel_tol: float = 0.02,
    min_tol_hz: float = 0.5,
) -> Dict[str, Optional[dict]]:
    """Assign detected tones to named target frequencies.

    Each target takes the most prominent tone inside its tolerance. A tone can
    legitimately match two targets (BPFO harmonics frequently collide with gear
    sidebands); the ambiguity is preserved rather than silently resolved,
    because resolving it needs the sideband evidence from
    :func:`sideband_ratio`.
    """
    out: Dict[str, Optional[dict]] = {}
    for name, f0 in targets.items():
        if f0 <= 0:
            out[name] = None
            continue
        tol = max(rel_tol * f0, min_tol_hz)
        cands = [t for t in tones if abs(t["freq_hz"] - f0) <= tol]
        out[name] = max(cands, key=lambda t: t["prominence_db"]) if cands else None
    return out


def bearing_evidence(
    env_spec: Spectrum,
    geom: BearingGeometry,
    shaft_hz: float,
    *,
    n_harmonics: int = 5,
    rel_tol: float = 0.02,
) -> Dict[str, Dict[str, float]]:
    """Full evidence bundle for one bearing from an envelope spectrum.

    Parameters
    ----------
    env_spec
        Output of :func:`pdmcore.dsp.envelope.envelope_spectrum`. Passing a raw
        acceleration spectrum here also works but is much less sensitive -- the
        whole point of the envelope stage is to move fault energy out from under
        the shaft-order content.

    Returns
    -------
    dict
        One entry per fault mode (``outer_race``, ``inner_race``,
        ``rolling_element``, ``cage``), each with the
        :func:`harmonic_energy_ratio` fields plus ``sideband_ratio``.
    """
    ff = fault_frequencies(geom, shaft_hz)
    modes = {
        "outer_race": (ff.bpfo_hz, 0.0),
        "inner_race": (ff.bpfi_hz, ff.shaft_hz),
        "rolling_element": (ff.bsf_hz, ff.ftf_hz),
        "cage": (ff.ftf_hz, 0.0),
    }
    out: Dict[str, Dict[str, float]] = {}
    for mode, (f0, fmod) in modes.items():
        ev = harmonic_energy_ratio(env_spec, f0, n_harmonics=n_harmonics, rel_tol=rel_tol)
        if fmod > 0:
            sb = sideband_ratio(env_spec, f0, fmod, rel_tol=rel_tol)
            ev["sideband_ratio"] = sb["ratio"]
            ev["n_sidebands_found"] = sb["n_sidebands_found"]
            ev["sideband_valid"] = sb["valid"]
        else:
            ev["sideband_ratio"] = 0.0
            ev["n_sidebands_found"] = 0.0
            ev["sideband_valid"] = 0.0
        out[mode] = ev
    return out
