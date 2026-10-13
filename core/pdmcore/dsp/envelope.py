"""Envelope analysis -- the single highest-value technique for bearing faults.

Why envelope analysis and not the raw spectrum
----------------------------------------------
A spalled bearing raceway produces a train of tiny impacts at a few hundred Hz
(BPFO/BPFI). Each impact rings a structural resonance at 2-20 kHz. In the raw
spectrum the fault energy is therefore *not* at BPFO -- it is spread around the
resonance as a modulation sideband family, typically 30-50 dB below the 1x shaft
line that dominates the low-frequency end. Band-pass around the resonance,
rectify (take the analytic envelope), and transform again: the modulation rate
appears as a clean line at BPFO with harmonics. That is the "squared envelope
spectrum" (SES).

Choosing the band is the hard part, and in a fleet of thousands of assets it
cannot be done by hand. Spectral kurtosis picks the most impulsive band
automatically (Antoni's kurtogram). :func:`select_demodulation_band` implements
a coarse kurtogram and is what the cloud pipeline uses; firmware receives the
resulting band edges as configuration so the MCU only runs a fixed filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from .spectrum import Spectrum, spectrum, stft

__all__ = [
    "DemodBand",
    "bandpass",
    "analytic_envelope",
    "envelope_spectrum",
    "select_demodulation_band",
    "kurtogram",
    "cepstrum",
    "cepstrum_prewhiten",
]


@dataclass(frozen=True)
class DemodBand:
    """A chosen demodulation band plus the evidence for choosing it."""

    lo_hz: float
    hi_hz: float
    kurtosis: float
    level: int
    energy_frac: float = 0.0

    @property
    def centre_hz(self) -> float:
        return 0.5 * (self.lo_hz + self.hi_hz)

    @property
    def width_hz(self) -> float:
        return self.hi_hz - self.lo_hz


def bandpass(
    x: Sequence[float],
    fs_hz: float,
    lo_hz: float,
    hi_hz: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass (``sosfiltfilt``).

    Zero-phase matters: a causal filter delays the impacts, and after
    rectification an uneven group delay smears the envelope and reduces the SES
    peak. ``filtfilt`` doubles the effective order, hence ``order=4`` here is an
    8th-order response.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = fs_hz / 2.0
    lo = max(1e-6, lo_hz / nyq)
    hi = min(0.999999, hi_hz / nyq)
    if not lo < hi:
        raise ValueError(f"invalid band ({lo_hz}, {hi_hz}) for fs={fs_hz}")
    if hi >= 0.999999 and lo <= 1e-6:
        return x
    if hi >= 0.999999:
        sos = sps.butter(order, lo, btype="highpass", output="sos")
    elif lo <= 1e-6:
        sos = sps.butter(order, hi, btype="lowpass", output="sos")
    else:
        sos = sps.butter(order, [lo, hi], btype="bandpass", output="sos")
    # filtfilt needs padlen < len(x); fall back to a causal pass on tiny blocks.
    padlen = 3 * (2 * order + 1)
    if x.size <= padlen:
        return sps.sosfilt(sos, x)
    return sps.sosfiltfilt(sos, x)


def analytic_envelope(x: Sequence[float]) -> np.ndarray:
    """Magnitude of the analytic signal (Hilbert envelope)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.abs(sps.hilbert(x))


def envelope_spectrum(
    x: Sequence[float],
    fs_hz: float,
    *,
    band_hz: Optional[Tuple[float, float]] = None,
    n_fft: int = 4096,
    squared: bool = True,
    remove_dc: bool = True,
    decimate_to_hz: Optional[float] = 2500.0,
    filter_order: int = 4,
) -> Spectrum:
    """Squared envelope spectrum (SES).

    Parameters
    ----------
    band_hz
        Demodulation band. ``None`` runs :func:`select_demodulation_band`.
    squared
        Square the envelope before the FFT. The squared envelope spectrum has
        better statistical properties than the plain envelope (Randall & Antoni
        2011) and is the recommended default.
    decimate_to_hz
        Low-pass + decimate the envelope to this rate before the FFT. The
        envelope is a slowly varying signal: bearing modulation rates are
        < 1 kHz, so keeping 25.6 kHz of envelope bandwidth wastes 10x the FFT
        resolution where it matters. Set to ``None`` to skip.

    Returns
    -------
    Spectrum
        ``scaling="peak"``; the amplitude of a line at BPFO is directly
        comparable between acquisitions on the same asset.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if band_hz is None:
        band = select_demodulation_band(x, fs_hz)
        band_hz = (band.lo_hz, band.hi_hz)
    filtered = bandpass(x, fs_hz, band_hz[0], band_hz[1], order=filter_order)
    env = analytic_envelope(filtered)
    if squared:
        env = env**2
    fs_env = float(fs_hz)
    if decimate_to_hz is not None and fs_hz > 2.5 * decimate_to_hz:
        factor = int(np.floor(fs_hz / decimate_to_hz))
        factor = max(2, min(factor, 100))
        env = sps.decimate(env, factor, ftype="fir", zero_phase=True)
        fs_env = fs_hz / factor
    if remove_dc:
        env = env - np.mean(env)
    n_fft = int(min(n_fft, 1 << int(np.floor(np.log2(max(2, env.size))))))
    return spectrum(env, fs_env, n_fft=n_fft, window_name="hann", scaling="peak", detrend="mean")


def kurtogram(
    x: Sequence[float],
    fs_hz: float,
    *,
    max_level: int = 5,
    n_fft: int = 256,
) -> Tuple[np.ndarray, list]:
    """Coarse fast-kurtogram.

    Returns ``(grid, bands)`` where ``grid[level, k]`` is the spectral kurtosis
    of band ``k`` at dyadic ``level`` (``2**(level+1)`` bands), and ``bands`` is
    a list of ``DemodBand`` sorted by descending kurtosis.

    This is the 1/3-binary-tree variant simplified to pure dyadic splits: for
    each level the STFT frames are grouped into equal-width bands, the complex
    envelope of each band is formed by summing its bins, and the kurtosis of
    that envelope magnitude is taken over time. Good enough to pick a band; the
    full Antoni tree buys a few percent and costs 5x the compute.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    freq, _t, frames = stft(x, fs_hz, n_fft=n_fft, hop=n_fft // 4, window_name="hann")
    if frames.shape[0] < 8:
        raise ValueError("record too short for a kurtogram; need >= 8 STFT frames")
    n_bins = frames.shape[1]
    power = np.abs(frames) ** 2
    total_power = float(np.sum(power))
    grid = np.full((max_level, 2 ** (max_level + 1)), np.nan)
    bands: list = []
    for level in range(max_level):
        n_bands = 2 ** (level + 1)
        edges = np.linspace(0, n_bins, n_bands + 1).astype(int)
        for k in range(n_bands):
            lo_i, hi_i = edges[k], edges[k + 1]
            if hi_i - lo_i < 2:
                continue
            # Complex envelope of the sub-band: coherent sum of its bins.
            sub = np.sum(frames[:, lo_i:hi_i], axis=1)
            env = np.abs(sub)
            env = env - np.mean(env)
            var = np.mean(env**2)
            if var <= 0:
                kurt = 0.0
            else:
                kurt = float(np.mean(env**4) / (var * var) - 2.0)
            frac = float(np.sum(power[:, lo_i:hi_i]) / total_power) if total_power > 0 else 0.0
            grid[level, k] = kurt
            bands.append(
                DemodBand(lo_hz=float(freq[lo_i]), hi_hz=float(freq[min(hi_i, n_bins - 1)]),
                          kurtosis=kurt, level=level + 1, energy_frac=frac)
            )
    bands.sort(key=lambda b: -b.kurtosis)
    return grid, bands


def select_demodulation_band(
    x: Sequence[float],
    fs_hz: float,
    *,
    min_width_hz: float = 500.0,
    min_centre_hz: float = 500.0,
    min_energy_frac: float = 0.01,
    max_level: int = 5,
    n_fft: int = 256,
) -> DemodBand:
    """Pick the most impulsive band suitable for envelope demodulation.

    Constraints applied on top of raw kurtosis:

    * ``min_centre_hz`` -- reject bands dominated by shaft-order content; the
      impulsiveness there is usually looseness or a bad mount, not a bearing.
    * ``min_width_hz`` -- the band must be at least a few times wider than the
      highest modulation rate of interest or the sidebands get clipped and the
      SES amplitude becomes speed-dependent.
    * ``min_energy_frac`` -- **the constraint that makes this usable.** Spectral
      kurtosis is scale-free, so the most "impulsive" band is often one holding
      0.01 % of the signal energy, above the sensor's own bandwidth, where a few
      quantisation-level wiggles look extremely peaky. Demodulating there
      returns noise. Requiring the band to carry at least ~1 % of total power
      keeps the choice on a real structural resonance.

    Falls back to the top half of the band if nothing qualifies, which is the
    conventional default for an unknown machine.
    """
    fallback = DemodBand(lo_hz=fs_hz / 4.0, hi_hz=fs_hz / 2.0 * 0.95, kurtosis=0.0, level=1)
    try:
        _grid, bands = kurtogram(x, fs_hz, max_level=max_level, n_fft=n_fft)
    except ValueError:
        return fallback
    for b in bands:
        if (
            b.width_hz >= min_width_hz
            and b.centre_hz >= min_centre_hz
            and b.energy_frac >= min_energy_frac
            and np.isfinite(b.kurtosis)
        ):
            return b
    return fallback


def cepstrum(x: Sequence[float], fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Real cepstrum -> ``(quefrency_s, amplitude)``.

    Periodic families of sidebands (gear mesh modulation, bearing harmonics)
    collapse to a single cepstral peak at the family spacing, which makes the
    cepstrum a compact way to detect "is there a modulation family at all"
    before paying for a full envelope analysis.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    log_spec = np.log(np.maximum(spec, 1e-20))
    ceps = np.fft.irfft(log_spec)
    q = np.arange(ceps.size, dtype=np.float64) / fs_hz
    half = ceps.size // 2
    return q[:half], ceps[:half]


def cepstrum_prewhiten(x: Sequence[float], fs_hz: float, *, notch_quefrency_s: Sequence[float] = ()) -> np.ndarray:
    """Cepstrum pre-whitening: flatten the spectrum, keep the modulation.

    Division by the spectrum magnitude (equivalently, zeroing the real cepstrum
    except for the terms of interest) removes the structural transfer function
    and the deterministic shaft-order content, leaving the stochastic impact
    content that bearing diagnosis depends on. This is the cheapest way to make
    one threshold work across machines with very different resonance maps --
    directly useful when scaling to a fleet where per-asset tuning is
    impossible.

    Parameters
    ----------
    notch_quefrency_s
        Optional quefrencies to additionally zero (e.g. ``1/shaft_hz``) to
        suppress a specific periodic family.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    whitened = np.where(mag > 1e-20, spec / np.maximum(mag, 1e-20), 0.0)
    out = np.fft.irfft(whitened, n=x.size)
    if len(notch_quefrency_s):
        q, ceps = cepstrum(out, fs_hz)
        for qs in notch_quefrency_s:
            idx = int(round(qs * fs_hz))
            for h in range(1, 6):
                j = idx * h
                if 0 < j < ceps.size:
                    ceps[max(0, j - 1) : j + 2] = 0.0
        full = np.concatenate([ceps, ceps[::-1][: x.size - ceps.size]])
        out = np.fft.irfft(np.exp(np.fft.rfft(full)), n=x.size)
    return out
