"""Acceleration -> velocity -> displacement, done the way ISO expects.

Why not ``numpy.cumsum``
------------------------
Time-domain integration of an accelerometer signal amplifies every low-frequency
error without bound: a 1 mg DC offset integrates into a velocity ramp, and
sensor 1/f noise integrates into a random walk that swamps the real signal. The
result is the classic "ski slope" at the bottom of a velocity spectrum, and a
velocity RMS that depends on record length.

The correct approach is frequency-domain ("omega arithmetic") integration with
an explicit high-pass: divide each bin by ``j*2*pi*f`` and zero the bins below
the high-pass corner. ISO 2954 and ISO 20816 both specify measurement in a band
(commonly 10-1000 Hz, or 2-1000 Hz for slow machines), so discarding content
below the corner is not an approximation -- it is the definition of the
measurement.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .spectrum import Spectrum, spectrum

__all__ = [
    "accel_g_to_velocity_mm_s",
    "accel_g_to_displacement_um",
    "velocity_rms_mm_s",
    "velocity_spectrum",
    "G_TO_MM_S2",
    "integrate_spectrum",
]

#: Standard gravity in mm/s^2 (ISO 80000-3).
G_TO_MM_S2 = 9806.65


def _omega_integrate(
    x: np.ndarray,
    fs_hz: float,
    *,
    order: int,
    hp_hz: float,
    lp_hz: Optional[float],
    taper_octaves: float = 0.5,
) -> np.ndarray:
    n = x.size
    spec = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    out = np.zeros_like(spec)
    nz = f > 0
    jw = 2j * np.pi * f[nz]
    out[nz] = spec[nz] / (jw**order)

    # Smooth roll-off instead of a brick wall: a hard cut rings in the time
    # domain and corrupts crest factor and peak-to-peak readings.
    gain = np.ones_like(f)
    if hp_hz > 0:
        f_start = hp_hz / (2.0**taper_octaves)
        band = (f > f_start) & (f < hp_hz)
        gain[f <= f_start] = 0.0
        if np.any(band):
            t = (np.log2(f[band]) - np.log2(f_start)) / taper_octaves
            gain[band] = 0.5 * (1.0 - np.cos(np.pi * t))
    if lp_hz is not None and lp_hz < fs_hz / 2.0:
        f_end = lp_hz * (2.0**taper_octaves)
        band = (f > lp_hz) & (f < f_end)
        gain[f >= f_end] = 0.0
        if np.any(band):
            t = (np.log2(f[band]) - np.log2(lp_hz)) / taper_octaves
            gain[band] = 0.5 * (1.0 + np.cos(np.pi * t))
    out *= gain
    return np.fft.irfft(out, n=n)


def accel_g_to_velocity_mm_s(
    accel_g: Sequence[float],
    fs_hz: float,
    *,
    hp_hz: float = 10.0,
    lp_hz: Optional[float] = 1000.0,
) -> np.ndarray:
    """Integrate acceleration in g to velocity in mm/s.

    Defaults implement the ISO 20816 10-1000 Hz band used for machines above
    600 rpm. For machines below 600 rpm use ``hp_hz=2.0`` (ISO 20816-1 allows
    2-1000 Hz), otherwise the 1x line itself falls in the stop band and the
    reading reads low -- a mistake that has passed many genuinely unbalanced
    slow fans.
    """
    x = np.asarray(accel_g, dtype=np.float64).ravel() * G_TO_MM_S2
    return _omega_integrate(x, fs_hz, order=1, hp_hz=hp_hz, lp_hz=lp_hz)


def accel_g_to_displacement_um(
    accel_g: Sequence[float],
    fs_hz: float,
    *,
    hp_hz: float = 10.0,
    lp_hz: Optional[float] = 1000.0,
) -> np.ndarray:
    """Double-integrate acceleration in g to displacement in micrometres."""
    x = np.asarray(accel_g, dtype=np.float64).ravel() * G_TO_MM_S2
    return _omega_integrate(x, fs_hz, order=2, hp_hz=hp_hz, lp_hz=lp_hz) * 1000.0


def velocity_spectrum(
    accel_g: Sequence[float],
    fs_hz: float,
    *,
    n_fft: int = 4096,
    hp_hz: float = 10.0,
    lp_hz: Optional[float] = 1000.0,
    window_name: str = "hann",
) -> Spectrum:
    """Velocity spectrum in mm/s RMS -- the ISO-comparable spectrum."""
    v = accel_g_to_velocity_mm_s(accel_g, fs_hz, hp_hz=hp_hz, lp_hz=lp_hz)
    return spectrum(v, fs_hz, n_fft=n_fft, window_name=window_name, scaling="rms")


def velocity_rms_mm_s(
    accel_g: Sequence[float],
    fs_hz: float,
    *,
    band_hz: Tuple[float, float] = (10.0, 1000.0),
    n_fft: int = 4096,
) -> float:
    """Broadband velocity RMS in mm/s over ``band_hz``.

    This single number is what ISO 20816 / ISO 10816 severity charts are written
    against, and it is the number a maintenance engineer will compare against a
    handheld meter. Computing it in the frequency domain (rather than as the
    time-domain RMS of an integrated waveform) makes the band limits exact and
    the result independent of record length.
    """
    spec = velocity_spectrum(accel_g, fs_hz, n_fft=n_fft, hp_hz=band_hz[0], lp_hz=band_hz[1])
    return spec.rms_in_band(band_hz[0], band_hz[1])


def integrate_spectrum(spec: Spectrum, *, order: int = 1, hp_hz: float = 10.0) -> Spectrum:
    """Integrate an existing acceleration spectrum in the frequency domain.

    Useful on the edge: the MCU has already paid for one FFT, so velocity can be
    derived from the same spectrum for the cost of one divide per bin instead of
    a second acquisition and transform. Input must be an acceleration spectrum
    in g; output is mm/s (``order=1``) or um (``order=2``) in the same scaling.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    f = spec.freq_hz
    v = np.asarray(spec.value, dtype=np.float64).copy()
    scale = np.zeros_like(f)
    nz = f >= hp_hz
    omega = 2.0 * np.pi * f[nz]
    scale[nz] = 1.0 / omega**order
    factor = G_TO_MM_S2 * (1000.0 if order == 2 else 1.0)
    if spec.scaling in ("psd", "power"):
        v = v * (scale * factor) ** 2
    else:
        v = v * scale * factor
    return Spectrum(
        freq_hz=f,
        value=v,
        scaling=spec.scaling,
        fs_hz=spec.fs_hz,
        n_fft=spec.n_fft,
        n_averages=spec.n_averages,
        window_name=spec.window_name,
        enbw_hz=spec.enbw_hz,
    )
