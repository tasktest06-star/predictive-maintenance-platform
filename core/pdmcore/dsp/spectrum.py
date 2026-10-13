"""Calibrated spectra, PSDs, band energies and spectral kurtosis.

Every spectrum-returning function here takes a ``scaling`` argument and states
its units, because the single most common integration bug between firmware,
cloud and a handheld reference analyser is a factor of sqrt(2), N or ENBW.

Scaling modes
-------------
``"peak"``      amplitude of a sinusoid, same units as the input (g).
``"rms"``       ``peak/sqrt(2)`` for a tone; this is what ISO limits use.
``"psd"``       units^2/Hz -- the only mode whose value is independent of the
                block length, so the only correct mode for broadband content.
``"power"``     units^2 per bin (``rms**2``).

Single-sided conventions: bins ``1..N/2-1`` are doubled; DC and Nyquist are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .windows import WindowSpec, window

__all__ = [
    "Spectrum",
    "spectrum",
    "welch_psd",
    "band_rms",
    "third_octave_bands",
    "mel_filterbank",
    "log_mel_spectrogram",
    "spectral_kurtosis",
    "spectral_shape_features",
    "find_tones",
    "interpolate_peak",
    "stft",
]

_SCALINGS = ("peak", "rms", "psd", "power")


@dataclass(frozen=True)
class Spectrum:
    """A single-sided spectrum with its provenance attached."""

    freq_hz: np.ndarray
    value: np.ndarray
    scaling: str
    fs_hz: float
    n_fft: int
    n_averages: int
    window_name: str
    enbw_hz: float

    @property
    def df_hz(self) -> float:
        return self.fs_hz / self.n_fft

    @property
    def enbw_bins(self) -> float:
        return self.enbw_hz / self.df_hz

    def band(self, lo_hz: float, hi_hz: float) -> np.ndarray:
        """Boolean mask for ``lo_hz <= f <= hi_hz``."""
        return (self.freq_hz >= lo_hz) & (self.freq_hz <= hi_hz)

    def rms_in_band(self, lo_hz: float, hi_hz: float) -> float:
        """RMS of the content between two frequencies, in input units.

        The ENBW division is the part that is easy to get wrong. A windowed
        spectrum spreads one tone over ``enbw_bins`` bins, each of which is
        calibrated to report *that tone's* amplitude; summing them therefore
        over-counts the energy by exactly ``enbw_bins``. Dividing it back out
        makes the band RMS agree with the time-domain RMS for broadband content
        **and** with the tone amplitude for a single line, which is the property
        that lets one function serve both uses. ``psd`` needs no correction
        because the ENBW is already inside its definition.
        """
        m = self.band(lo_hz, hi_hz)
        if not np.any(m):
            return 0.0
        if self.scaling == "psd":
            return float(np.sqrt(np.sum(self.value[m]) * self.df_hz))
        enbw = max(1e-12, self.enbw_bins)
        if self.scaling == "power":
            return float(np.sqrt(np.sum(self.value[m]) / enbw))
        if self.scaling == "rms":
            return float(np.sqrt(np.sum(self.value[m] ** 2) / enbw))
        # peak
        return float(np.sqrt(np.sum(self.value[m] ** 2) / 2.0 / enbw))

    def at(self, f_hz: float, tol_hz: Optional[float] = None) -> float:
        """Largest value within ``tol_hz`` of ``f_hz`` (default: +-1.5 bins)."""
        tol = 1.5 * self.df_hz if tol_hz is None else tol_hz
        m = np.abs(self.freq_hz - f_hz) <= tol
        return float(np.max(self.value[m])) if np.any(m) else 0.0


def _detrend(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "mean":
        return x - np.mean(x)
    if mode == "linear":
        n = x.size
        t = np.arange(n, dtype=np.float64)
        # Closed form least squares; avoids pulling in numpy.polyfit warnings.
        tm, xm = t.mean(), x.mean()
        denom = np.sum((t - tm) ** 2)
        slope = 0.0 if denom == 0 else float(np.sum((t - tm) * (x - xm)) / denom)
        return x - (slope * (t - tm) + xm)
    raise ValueError(f"unknown detrend mode {mode!r}")


def _block_power(mag: np.ndarray, n_fft: int) -> np.ndarray:
    """Single-sided power of one windowed block, ``|X|**2`` with the fold applied.

    The fold is applied to *power*, not magnitude: doubling the magnitude and
    then squaring over-counts by 2x and makes a white-noise PSD integrate to
    ``sqrt(2)*sigma`` instead of ``sigma`` -- the kind of error that only
    surfaces when someone compares the dashboard against a calibrated reference
    meter. Pinned by ``test_spectrum_parseval``.
    """
    p = mag.astype(np.float64) ** 2
    if p.size > 1:
        last_is_nyquist = n_fft % 2 == 0
        hi = p.size - 1 if last_is_nyquist else p.size
        p[1:hi] *= 2.0
    return p


def _scale_power(p: np.ndarray, win: WindowSpec, fs: float, scaling: str) -> np.ndarray:
    """Convert accumulated single-sided power into the requested scaling."""
    s1 = float(np.sum(win.coeff))
    s2 = float(np.sum(win.coeff**2))
    if scaling == "psd":
        # Power per bin divided by the noise bandwidth of one bin: units^2/Hz.
        return p / (fs * s2)
    if scaling == "power":
        # Mean-square contribution of each bin: units^2 (== rms^2).
        return p / (s1 * s1)
    amp_rms = np.sqrt(p) / s1  # RMS amplitude of a tone in this bin
    if scaling == "rms":
        return amp_rms
    if scaling == "peak":
        return amp_rms * np.sqrt(2.0)
    raise ValueError(f"unknown scaling {scaling!r}; expected one of {_SCALINGS}")


def spectrum(
    x: Sequence[float],
    fs_hz: float,
    *,
    n_fft: Optional[int] = None,
    window_name: str = "hann",
    scaling: str = "rms",
    detrend: str = "mean",
    overlap: float = 0.5,
    average: str = "linear",
) -> Spectrum:
    """Averaged single-sided spectrum.

    Parameters
    ----------
    x
        Real input, e.g. acceleration in g.
    fs_hz
        Sample rate.
    n_fft
        Block size. Defaults to the largest power of two that fits, capped at
        16384 -- matching what the MCU firmware can hold in RAM.
    window_name
        See :func:`pdmcore.dsp.windows.window`.
    scaling
        ``"peak" | "rms" | "psd" | "power"``.
    overlap
        Fractional block overlap for averaging. 0.5 with a Hann window gives
        near-optimal variance reduction per sample (Welch).
    average
        ``"linear"`` (stationary content), ``"peak"`` (hold the maximum --
        catches intermittent impacts that linear averaging buries), or
        ``"exponential"`` (weight recent blocks, for live trending).

    Returns
    -------
    Spectrum

    Notes
    -----
    Averaging happens in the **power** domain regardless of the requested
    scaling. Averaging magnitudes instead ("linear magnitude averaging" on a
    bench analyser) biases broadband content low by ``pi/4``, i.e. -1.05 dB,
    because the mean of a Rayleigh magnitude is not the root of its mean square.
    A 1 dB bias is small enough to pass review and large enough to make the
    noise floor of a trended spectrum disagree with the time-domain RMS of the
    same record forever.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty input")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be positive")
    if n_fft is None:
        n_fft = min(16384, 1 << int(np.floor(np.log2(max(2, x.size)))))
    n_fft = int(n_fft)
    if n_fft < 2:
        raise ValueError("n_fft must be >= 2")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    win = window(window_name, n_fft)
    step = max(1, int(round(n_fft * (1.0 - overlap))))
    if x.size < n_fft:  # zero-pad a short record rather than failing
        x = np.concatenate([x, np.zeros(n_fft - x.size)])
    starts = range(0, x.size - n_fft + 1, step)

    if average not in ("linear", "peak", "exponential"):
        raise ValueError(f"unknown average mode {average!r}")

    acc: Optional[np.ndarray] = None
    count = 0
    alpha = 0.2  # exponential forgetting factor
    for start in starts:
        block = _detrend(x[start : start + n_fft], detrend) * win.coeff
        p = _block_power(np.abs(np.fft.rfft(block)), n_fft)
        if acc is None:
            acc = p
        elif average == "linear":
            acc = acc + p
        elif average == "peak":
            acc = np.maximum(acc, p)
        else:  # exponential
            acc = (1.0 - alpha) * acc + alpha * p
        count += 1
    assert acc is not None
    if average == "linear" and count > 0:
        acc = acc / count

    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    return Spectrum(
        freq_hz=freq,
        value=_scale_power(acc, win, fs_hz, scaling),
        scaling=scaling,
        fs_hz=float(fs_hz),
        n_fft=n_fft,
        n_averages=count,
        window_name=win.name,
        enbw_hz=win.enbw_bins * fs_hz / n_fft,
    )


def welch_psd(
    x: Sequence[float],
    fs_hz: float,
    *,
    n_fft: int = 4096,
    overlap: float = 0.5,
    window_name: str = "hann",
) -> Spectrum:
    """Welch PSD in units^2/Hz. Thin alias for the ``psd`` scaling path."""
    return spectrum(
        x, fs_hz, n_fft=n_fft, window_name=window_name, scaling="psd", overlap=overlap, average="linear"
    )


def band_rms(spec: Spectrum, bands_hz: Sequence[Tuple[float, float]]) -> np.ndarray:
    """RMS inside each ``(lo, hi)`` band.

    Band RMS vectors are the workhorse edge feature: 12-20 numbers per axis
    carry most of the diagnostic information of a 4096-point spectrum at ~1/300
    of the radio payload.
    """
    return np.array([spec.rms_in_band(lo, hi) for lo, hi in bands_hz], dtype=np.float64)


def third_octave_bands(f_lo_hz: float = 12.5, f_hi_hz: float = 20000.0) -> np.ndarray:
    """Base-10 one-third-octave band edges per IEC 61260, as ``(n, 2)`` array."""
    if f_lo_hz <= 0 or f_hi_hz <= f_lo_hz:
        raise ValueError("require 0 < f_lo_hz < f_hi_hz")
    g = 10.0 ** (3.0 / 10.0)  # base-10 octave ratio
    centres = []
    # Band index 0 is 1 kHz by definition.
    for idx in range(-40, 40):
        fc = 1000.0 * g ** (idx / 3.0)
        if f_lo_hz <= fc <= f_hi_hz:
            centres.append(fc)
    edges = [(fc / g ** (1.0 / 6.0), fc * g ** (1.0 / 6.0)) for fc in centres]
    return np.array(edges, dtype=np.float64)


def mel_filterbank(n_mels: int, n_fft: int, fs_hz: float, f_lo_hz: float = 50.0, f_hi_hz: Optional[float] = None) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape ``(n_mels, n_fft//2+1)``.

    Used by the acoustic branch: log-mel + autoencoder is the dominant
    anomalous-sound-detection recipe, and a 64-band log-mel frame is small
    enough to run on a Cortex-M33.
    """
    f_hi = fs_hz / 2.0 if f_hi_hz is None else f_hi_hz
    if not 0 < f_lo_hz < f_hi <= fs_hz / 2.0 + 1e-9:
        raise ValueError("require 0 < f_lo_hz < f_hi_hz <= fs/2")

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(np.array(f_lo_hz)), hz_to_mel(np.array(f_hi)), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    fb = np.zeros((n_mels, bins.size), dtype=np.float64)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        if hi <= lo:
            continue
        rise = (bins >= lo) & (bins <= ctr)
        fall = (bins > ctr) & (bins <= hi)
        if ctr > lo:
            fb[m, rise] = (bins[rise] - lo) / (ctr - lo)
        if hi > ctr:
            fb[m, fall] = (hi - bins[fall]) / (hi - ctr)
        # Slaney normalisation: unit area, so a flat PSD gives a flat mel frame.
        norm = 2.0 / (hi - lo)
        fb[m] *= norm
    return fb


def stft(
    x: Sequence[float],
    fs_hz: float,
    *,
    n_fft: int = 1024,
    hop: Optional[int] = None,
    window_name: str = "hann",
    detrend: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform.

    Returns ``(freq_hz, time_s, complex_frames)`` with ``complex_frames`` of
    shape ``(n_frames, n_fft//2+1)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    hop = n_fft // 2 if hop is None else int(hop)
    if hop <= 0:
        raise ValueError("hop must be positive")
    win = window(window_name, n_fft)
    if x.size < n_fft:
        x = np.concatenate([x, np.zeros(n_fft - x.size)])
    starts = np.arange(0, x.size - n_fft + 1, hop)
    frames = np.empty((starts.size, n_fft // 2 + 1), dtype=np.complex128)
    for i, s in enumerate(starts):
        frames[i] = np.fft.rfft(_detrend(x[s : s + n_fft], detrend) * win.coeff)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    time = (starts + n_fft / 2.0) / fs_hz
    return freq, time, frames


def log_mel_spectrogram(
    x: Sequence[float],
    fs_hz: float,
    *,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop: Optional[int] = None,
    floor_db: float = -120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(time_s, log_mel_db)`` with ``log_mel_db`` shaped ``(n_frames, n_mels)``."""
    freq, time, frames = stft(x, fs_hz, n_fft=n_fft, hop=hop)
    power = np.abs(frames) ** 2
    fb = mel_filterbank(n_mels, n_fft, fs_hz)
    mel = power @ fb.T
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(mel, 10.0 ** (floor_db / 10.0)))
    return time, db


def spectral_kurtosis(
    x: Sequence[float],
    fs_hz: float,
    *,
    n_fft: int = 512,
    hop: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral kurtosis per frequency bin, ``(freq_hz, sk)``.

    SK answers "which frequency band behaves most impulsively?" -- Gaussian
    background gives SK ~ 0, a repetitive impact train gives SK > 0. This is how
    the demodulation band for envelope analysis is chosen automatically instead
    of being hand-tuned per machine, which does not scale past a few hundred
    assets.

    Definition used (Antoni 2006, normalised so Gaussian noise -> 0):
    ``SK(f) = <|X|^4> / <|X|^2>^2 - 2``.

    .. warning::
       ``n_fft`` is **not** a free parameter. SK measures impulsiveness *across
       frames*, so it only sees impacts that are sparse relative to the frame
       rate. At ``fs = 25.6 kHz`` and ``n_fft = 256`` the frame period is 10 ms,
       so a 100 Hz impact train puts one impact in every frame, the band
       envelope becomes constant, and SK collapses to zero for a signal that is
       nothing but impacts. Choose ``n_fft`` so that the frame period is at
       least ~5x the expected fault period: for BPFO near 100 Hz at 25.6 kHz
       that means ``n_fft <= 64``, or decimate first. Pinned by
       ``test_spectral_kurtosis_needs_impulses_sparse_relative_to_frame_rate``.
    """
    freq, _t, frames = stft(x, fs_hz, n_fft=n_fft, hop=hop, window_name="hann")
    if frames.shape[0] < 4:
        return freq, np.zeros_like(freq)
    p2 = np.mean(np.abs(frames) ** 2, axis=0)
    p4 = np.mean(np.abs(frames) ** 4, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(p2 > 0, p4 / (p2**2) - 2.0, 0.0)
    return freq, np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)


def spectral_shape_features(spec: Spectrum, f_lo_hz: float = 0.0, f_hi_hz: Optional[float] = None) -> dict:
    """Centroid / spread / skew / kurtosis / rolloff / flatness / entropy.

    These summarise *how* broadband energy is distributed, which separates
    cavitation and lubrication starvation (energy climbs at high frequency)
    from looseness (energy spreads at low frequency) without needing a tone
    model.
    """
    f_hi = spec.freq_hz[-1] if f_hi_hz is None else f_hi_hz
    m = (spec.freq_hz >= f_lo_hz) & (spec.freq_hz <= f_hi)
    f = spec.freq_hz[m]
    v = np.asarray(spec.value[m], dtype=np.float64)
    p = v if spec.scaling in ("psd", "power") else v**2
    total = float(np.sum(p))
    if total <= 0 or f.size < 2:
        return {k: 0.0 for k in
                ("centroid_hz", "spread_hz", "skew", "kurtosis", "rolloff95_hz", "flatness", "entropy")}
    w = p / total
    centroid = float(np.sum(f * w))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * w)))
    if spread > 0:
        skew = float(np.sum(((f - centroid) ** 3) * w) / spread**3)
        kurt = float(np.sum(((f - centroid) ** 4) * w) / spread**4)
    else:
        skew = kurt = 0.0
    cdf = np.cumsum(w)
    rolloff = float(f[int(np.searchsorted(cdf, 0.95))]) if cdf[-1] >= 0.95 else float(f[-1])
    pos = p[p > 0]
    flatness = float(np.exp(np.mean(np.log(pos))) / np.mean(pos)) if pos.size else 0.0
    entropy = float(-np.sum(w[w > 0] * np.log(w[w > 0])) / np.log(w.size))
    return {
        "centroid_hz": centroid,
        "spread_hz": spread,
        "skew": skew,
        "kurtosis": kurt,
        "rolloff95_hz": rolloff,
        "flatness": flatness,
        "entropy": entropy,
    }


def interpolate_peak(spec: Spectrum, idx: int) -> Tuple[float, float]:
    """Parabolic interpolation around bin ``idx`` -> ``(freq_hz, value)``.

    Recovers most of the scallop loss without a flat-top window, which matters
    on an MCU where the window is baked into flash.
    """
    v = np.asarray(spec.value, dtype=np.float64)
    if idx <= 0 or idx >= v.size - 1:
        return float(spec.freq_hz[idx]), float(v[idx])
    y0, y1, y2 = v[idx - 1], v[idx], v[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(spec.freq_hz[idx] + delta * spec.df_hz), float(y1 - 0.25 * (y0 - y2) * delta)


def find_tones(
    spec: Spectrum,
    *,
    min_prominence_db: float = 6.0,
    f_lo_hz: float = 2.0,
    f_hi_hz: Optional[float] = None,
    max_tones: int = 32,
    noise_median_bins: int = 65,
) -> list:
    """Discrete tones ranked by prominence over the local noise floor.

    The floor is a running median (robust to the tones themselves), so a machine
    with a high broadband floor does not produce 200 false "peaks".

    Returns a list of dicts: ``freq_hz``, ``value``, ``prominence_db``, ``bin``.
    """
    f_hi = spec.freq_hz[-1] if f_hi_hz is None else f_hi_hz
    v = np.asarray(spec.value, dtype=np.float64)
    amp = np.sqrt(v) if spec.scaling in ("psd", "power") else v
    k = max(3, int(noise_median_bins) | 1)
    pad = k // 2
    padded = np.pad(amp, pad, mode="edge")
    floor = np.array([np.median(padded[i : i + k]) for i in range(amp.size)])
    floor = np.maximum(floor, 1e-20)
    prom_db = 20.0 * np.log10(np.maximum(amp, 1e-20) / floor)

    out = []
    for i in range(1, amp.size - 1):
        f = spec.freq_hz[i]
        if f < f_lo_hz or f > f_hi:
            continue
        if amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1] and prom_db[i] >= min_prominence_db:
            fi, vi = interpolate_peak(spec, i)
            out.append({"freq_hz": fi, "value": vi, "prominence_db": float(prom_db[i]), "bin": i})
    out.sort(key=lambda d: -d["prominence_db"])
    return out[:max_tones]
