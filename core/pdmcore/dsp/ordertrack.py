"""Order tracking: analyse in the angle domain, not the time domain.

Under variable speed, every shaft-related line in a time-domain spectrum smears
across many bins. A 2 % speed drift over a 1-second record turns a crisp 6x line
at 300 Hz into a 6 Hz-wide blur, dropping its peak amplitude by over 10 dB and
hiding anything underneath. Resampling the signal at constant *angular*
increments collapses each order back into one bin.

Two paths are supported, because most installed machines have no tachometer:

``tacho``     Use a pulse train (1 or more pulses per revolution). Exact, and
              the only way to resolve non-integer orders reliably.
``tacholess`` Track the instantaneous frequency of a chosen reference order
              through an STFT ridge, then integrate it to phase. Accurate to a
              few tenths of a percent on machines with a dominant 1x or gear
              mesh line, which covers motors, pumps and fans; it degrades on
              machines whose dominant line vanishes at some loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .spectrum import Spectrum, spectrum, stft

__all__ = [
    "SpeedProfile",
    "tacho_to_phase",
    "estimate_speed_tacholess",
    "angular_resample",
    "order_spectrum",
    "time_synchronous_average",
]


@dataclass(frozen=True)
class SpeedProfile:
    """Shaft speed as a function of time."""

    time_s: np.ndarray
    speed_hz: np.ndarray

    @property
    def mean_hz(self) -> float:
        return float(np.mean(self.speed_hz))

    @property
    def rpm(self) -> np.ndarray:
        return self.speed_hz * 60.0

    @property
    def drift_pct(self) -> float:
        """Peak-to-peak speed variation as a percentage of the mean.

        Above ~0.5 % a plain FFT starts losing amplitude accuracy on high
        orders; above ~2 % order tracking is mandatory for bearing work.
        """
        m = self.mean_hz
        if m <= 0:
            return 0.0
        return float((np.max(self.speed_hz) - np.min(self.speed_hz)) / m * 100.0)

    def phase_rev(self, t: np.ndarray) -> np.ndarray:
        """Cumulative shaft revolutions at times ``t`` (trapezoidal)."""
        order = np.argsort(self.time_s)
        ts, fs = self.time_s[order], self.speed_hz[order]
        cum = np.concatenate([[0.0], np.cumsum(np.diff(ts) * 0.5 * (fs[1:] + fs[:-1]))])
        return np.interp(t, ts, cum)


def tacho_to_phase(
    pulse_times_s: Sequence[float],
    *,
    pulses_per_rev: int = 1,
    smooth_revs: float = 0.0,
) -> SpeedProfile:
    """Speed profile from tachometer pulse timestamps.

    Parameters
    ----------
    pulse_times_s
        Rising-edge timestamps. On an MCU these come from a timer input-capture
        channel; the firmware ships the raw capture values so the cloud can
        recompute speed with whatever smoothing it wants.
    pulses_per_rev
        Pulses per revolution of the tacho target (keyphasor = 1, a 60-tooth
        gear = 60).
    smooth_revs
        Moving-average length in revolutions. Mild smoothing (0.5-2 rev)
        suppresses jitter from an uneven target without hiding real torsional
        variation.
    """
    t = np.asarray(pulse_times_s, dtype=np.float64).ravel()
    if t.size < 3:
        raise ValueError("need at least 3 tacho pulses")
    if np.any(np.diff(t) <= 0):
        raise ValueError("tacho pulse times must be strictly increasing")
    dt = np.diff(t)
    speed = 1.0 / (dt * pulses_per_rev)
    tc = 0.5 * (t[1:] + t[:-1])
    if smooth_revs > 0:
        k = max(1, int(round(smooth_revs * pulses_per_rev)))
        if k > 1:
            kernel = np.ones(k) / k
            speed = np.convolve(speed, kernel, mode="same")
    return SpeedProfile(time_s=tc, speed_hz=speed)


def estimate_speed_tacholess(
    x: Sequence[float],
    fs_hz: float,
    *,
    f_min_hz: float,
    f_max_hz: float,
    reference_order: float = 1.0,
    n_fft: int = 4096,
    hop: Optional[int] = None,
    max_jump_hz: Optional[float] = None,
) -> SpeedProfile:
    """Track the dominant line in ``[f_min_hz, f_max_hz]`` and convert to speed.

    Implemented as a Viterbi-style ridge track: for each STFT frame the best bin
    is chosen subject to a penalty on jumping away from the previous frame, so a
    momentary dropout of the reference line does not throw the track onto a
    harmonic. ``reference_order`` divides the tracked frequency to get shaft
    speed (use ``n_teeth`` to track a gear mesh line, which is usually the
    strongest and cleanest line on a gearbox).
    """
    freq, time, frames = stft(x, fs_hz, n_fft=n_fft, hop=hop, window_name="hann")
    mag = np.abs(frames)
    band = (freq >= f_min_hz) & (freq <= f_max_hz)
    if not np.any(band):
        raise ValueError("search band contains no FFT bins; raise n_fft or widen the band")
    fb = freq[band]
    mb = mag[:, band]
    if mb.shape[0] == 0:
        raise ValueError("record too short for the requested n_fft")

    df = float(fb[1] - fb[0]) if fb.size > 1 else 1.0
    max_jump = max_jump_hz if max_jump_hz is not None else max(5.0 * df, 0.05 * np.mean(fb))
    jump_bins = max(1, int(round(max_jump / df)))

    n_frames, n_bins = mb.shape
    eps = 1e-20
    log_obs = np.log(mb + eps)
    score = np.full((n_frames, n_bins), -np.inf)
    back = np.zeros((n_frames, n_bins), dtype=np.int32)
    score[0] = log_obs[0]
    for i in range(1, n_frames):
        for j in range(n_bins):
            lo = max(0, j - jump_bins)
            hi = min(n_bins, j + jump_bins + 1)
            prev = score[i - 1, lo:hi]
            k = int(np.argmax(prev))
            back[i, j] = lo + k
            score[i, j] = prev[k] + log_obs[i, j]
    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for i in range(n_frames - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    # Parabolic refinement of each chosen bin for sub-bin speed resolution.
    tracked = np.empty(n_frames)
    for i, j in enumerate(path):
        if 0 < j < n_bins - 1:
            y0, y1, y2 = mb[i, j - 1], mb[i, j], mb[i, j + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.0 if denom == 0 else float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        else:
            delta = 0.0
        tracked[i] = fb[j] + delta * df
    return SpeedProfile(time_s=time, speed_hz=tracked / float(reference_order))


def angular_resample(
    x: Sequence[float],
    fs_hz: float,
    profile: SpeedProfile,
    *,
    samples_per_rev: int = 256,
) -> Tuple[np.ndarray, float]:
    """Resample ``x`` onto a uniform shaft-angle grid.

    Returns ``(resampled, samples_per_rev)``. The second value lets callers feed
    the result straight into :func:`order_spectrum`, where the "sample rate" is
    ``samples_per_rev`` and the frequency axis becomes shaft orders.

    ``samples_per_rev`` sets the maximum resolvable order to
    ``samples_per_rev/2``: 256 reaches order 128, which comfortably covers gear
    mesh on a single-stage box. Interpolation is cubic -- linear interpolation
    introduces an amplitude-dependent error that looks exactly like a 2x
    sideband family and has caused more than one false misalignment call.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.arange(x.size, dtype=np.float64) / fs_hz
    rev = profile.phase_rev(t)
    total_rev = float(rev[-1] - rev[0])
    if total_rev <= 1.0:
        raise ValueError(f"record covers only {total_rev:.2f} revolutions; need > 1")
    n_out = int(np.floor(total_rev * samples_per_rev))
    rev_grid = rev[0] + np.arange(n_out, dtype=np.float64) / samples_per_rev
    # rev is monotonically non-decreasing for non-negative speed, so np.interp
    # is valid; use a cubic spline where scipy is available for accuracy.
    try:
        from scipy.interpolate import CubicSpline

        order = np.argsort(rev)
        spline = CubicSpline(rev[order], x[order], extrapolate=False)
        out = spline(rev_grid)
        out = np.nan_to_num(out, nan=0.0)
    except Exception:  # pragma: no cover - scipy is a hard dep, this is belt+braces
        out = np.interp(rev_grid, rev, x)
    return out, float(samples_per_rev)


def order_spectrum(
    x: Sequence[float],
    fs_hz: float,
    profile: SpeedProfile,
    *,
    samples_per_rev: int = 256,
    n_fft: Optional[int] = None,
    window_name: str = "hann",
    scaling: str = "rms",
) -> Spectrum:
    """Order spectrum: ``freq_hz`` carries **shaft orders**, not Hz.

    An order spectrum is directly comparable between acquisitions taken at
    different speeds, which is what makes fleet-wide thresholds possible on
    VFD-driven equipment.
    """
    resampled, sprev = angular_resample(x, fs_hz, profile, samples_per_rev=samples_per_rev)
    if n_fft is None:
        n_fft = 1 << int(np.floor(np.log2(max(2, resampled.size))))
        n_fft = int(min(n_fft, 8192))
    return spectrum(resampled, sprev, n_fft=n_fft, window_name=window_name, scaling=scaling)


def time_synchronous_average(
    x: Sequence[float],
    fs_hz: float,
    profile: SpeedProfile,
    *,
    samples_per_rev: int = 256,
    n_revs: int = 1,
) -> np.ndarray:
    """Average over shaft revolutions to isolate what is locked to the shaft.

    Everything synchronous with the shaft (a cracked tooth, a bent shaft, a rub)
    survives; everything else (bearing faults, which slip, and broadband noise)
    averages toward zero as ``1/sqrt(n_averages)``. Running the residual
    (original minus TSA) through bearing analysis is the standard way to look at
    a bearing inside a gearbox, where the gear mesh is 30 dB louder.
    """
    resampled, _ = angular_resample(x, fs_hz, profile, samples_per_rev=samples_per_rev)
    period = samples_per_rev * n_revs
    n_blocks = resampled.size // period
    if n_blocks < 2:
        raise ValueError(f"need >= 2 full {n_revs}-rev blocks, got {n_blocks}")
    return resampled[: n_blocks * period].reshape(n_blocks, period).mean(axis=0)
