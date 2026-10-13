"""Time-domain condition indicators.

These are the cheap features: every one is computable in a single pass over a
block with no FFT, which is why the firmware computes them every acquisition
while a full spectrum is sent only on demand or on alarm.

Diagnostic character of each indicator (field-tested ordering):

================  =========================================================
indicator         what it reacts to first
================  =========================================================
``rms``           overall energy; tracks unbalance, looseness, late-stage wear
``peak``          single worst impact in the block
``crest``         early-stage localised defects (peak rises before RMS does)
``kurtosis``      impulsiveness; the classic early bearing-defect indicator
``clearance``     very early pitting -- most sensitive, also most noise-prone
``shape``         waveform "peakiness" independent of amplitude scaling
``skewness``      asymmetric rubs, soft foot, directional looseness
``zcr``           dominant-frequency proxy; jumps when a resonance is excited
================  =========================================================

Caveat that matters in production: kurtosis is *not* monotonic with damage. It
rises as a defect localises, then falls back toward 3 once the defect spreads
over the raceway and the signal becomes quasi-Gaussian again. A rule that only
watches kurtosis will declare a severely damaged bearing healthy. This is why
:mod:`pdmcore.health.index` fuses kurtosis with RMS trend rather than
thresholding it alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np

__all__ = ["TimeFeatures", "time_features", "time_feature_vector", "TIME_FEATURE_NAMES", "running_stats"]

TIME_FEATURE_NAMES = (
    "mean",
    "rms",
    "std",
    "peak",
    "peak_to_peak",
    "crest",
    "kurtosis",
    "skewness",
    "shape",
    "impulse",
    "clearance",
    "margin",
    "zcr",
    "entropy",
)


@dataclass(frozen=True)
class TimeFeatures:
    """One feature set for one channel of one acquisition block."""

    mean: float
    rms: float
    std: float
    peak: float
    peak_to_peak: float
    crest: float
    kurtosis: float
    skewness: float
    shape: float
    impulse: float
    clearance: float
    margin: float
    zcr: float
    entropy: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in TIME_FEATURE_NAMES], dtype=np.float64)


def time_features(x: Sequence[float], *, remove_dc: bool = True, hist_bins: int = 32) -> TimeFeatures:
    """Compute the standard indicator set.

    Parameters
    ----------
    x
        Block of samples, e.g. acceleration in g.
    remove_dc
        Subtract the mean before computing shape statistics. Keep this on for
        accelerometers: the 1 g static component on a vertically mounted sensor
        otherwise dominates RMS and makes crest factor meaningless. ``mean`` is
        still reported from the raw block so the DC level remains observable
        (it is a useful mounting-orientation check).
    hist_bins
        Bin count for the amplitude-distribution entropy.

    Notes
    -----
    ``kurtosis`` is the *non-excess* (Pearson) definition: 3.0 for Gaussian
    noise. Firmware reports the same convention; mixing excess and non-excess
    kurtosis between edge and cloud shifts every threshold by exactly 3, which
    is a subtle and very annoying bug.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 samples")
    raw_mean = float(np.mean(x))
    xc = x - raw_mean if remove_dc else x

    std = float(np.std(xc))
    rms = float(np.sqrt(np.mean(xc**2)))
    abs_x = np.abs(xc)
    peak = float(np.max(abs_x))
    p2p = float(np.max(xc) - np.min(xc))
    mean_abs = float(np.mean(abs_x))
    mean_sqrt_abs = float(np.mean(np.sqrt(abs_x)))

    eps = 1e-20
    crest = peak / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = peak / (mean_abs + eps)
    clearance = peak / (mean_sqrt_abs**2 + eps)
    margin = clearance / (shape + eps)

    if std > 0:
        z = xc / std
        kurt = float(np.mean(z**4))
        skew = float(np.mean(z**3))
    else:  # perfectly flat block: a stuck ADC or a sensor in reset
        kurt, skew = 0.0, 0.0

    signs = np.signbit(xc)
    zcr = float(np.count_nonzero(signs[1:] != signs[:-1]) / (xc.size - 1))

    if peak > 0:
        hist, _ = np.histogram(xc, bins=hist_bins, range=(-peak, peak))
        p = hist.astype(np.float64)
        total = p.sum()
        if total > 0:
            p /= total
            nz = p[p > 0]
            entropy = float(-np.sum(nz * np.log(nz)) / np.log(hist_bins))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return TimeFeatures(
        mean=raw_mean,
        rms=rms,
        std=std,
        peak=peak,
        peak_to_peak=p2p,
        crest=crest,
        kurtosis=kurt,
        skewness=skew,
        shape=shape,
        impulse=impulse,
        clearance=clearance,
        margin=margin,
        zcr=zcr,
        entropy=entropy,
    )


def time_feature_vector(x: Sequence[float], **kwargs) -> np.ndarray:
    """Feature vector ordered exactly as :data:`TIME_FEATURE_NAMES`."""
    return time_features(x, **kwargs).as_vector()


def running_stats(x: Sequence[float]) -> Dict[str, float]:
    """Single-pass streaming moments, mirroring the firmware implementation.

    The MCU cannot hold a block and then iterate it several times for long
    records, so it accumulates raw power sums (Welford for mean/variance plus
    third and fourth central moments) and derives the indicators at the end.
    This function exists so the C implementation can be diffed against a Python
    reference that uses the identical algorithm -- not merely the identical
    definition.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = 0
    mean = m2 = m3 = m4 = 0.0
    peak = 0.0
    sum_abs = 0.0
    sum_sqrt_abs = 0.0
    for v in x:
        n += 1
        delta = v - mean
        dn = delta / n
        dn2 = dn * dn
        term = delta * dn * (n - 1)
        mean += dn
        m4 += term * dn2 * (n * n - 3 * n + 3) + 6 * dn2 * m2 - 4 * dn * m3
        m3 += term * dn * (n - 2) - 3 * dn * m2
        m2 += term
        a = abs(v)
        peak = a if a > peak else peak
        sum_abs += a
        sum_sqrt_abs += np.sqrt(a)
    if n < 2:
        raise ValueError("need at least 2 samples")
    var = m2 / n
    std = float(np.sqrt(var))
    eps = 1e-20
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "rms": float(np.sqrt(var + mean * mean)),
        "peak": peak,
        "kurtosis": float(n * m4 / (m2 * m2)) if m2 > 0 else 0.0,
        "skewness": float(np.sqrt(n) * m3 / (m2**1.5)) if m2 > 0 else 0.0,
        "crest": float(peak / (std + eps)),
        "mean_abs": float(sum_abs / n),
        "clearance": float(peak / ((sum_sqrt_abs / n) ** 2 + eps)),
    }
