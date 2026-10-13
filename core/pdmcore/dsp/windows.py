"""Window functions with explicit correction factors.

Why this module exists instead of calling ``scipy.signal.get_window``:

* The microcontroller firmware (``cdsp/src/pdm_window.c``) must generate the
  *same* coefficients to the last bit, so the definitions are written out
  explicitly here and pinned by golden vectors rather than inherited from a
  library whose conventions may change.
* Condition monitoring quotes amplitudes in engineering units (g, mm/s). A
  windowed FFT under-reports amplitude unless the coherent gain is divided out,
  and under-reports *noise* unless the equivalent noise bandwidth is divided
  out. Getting these two mixed up is the classic reason a dashboard shows
  "1.8 mm/s" where the handheld analyser shows "2.6 mm/s". Both factors are
  returned alongside the coefficients so callers cannot forget them.

Conventions
-----------
All windows are **periodic** (``sym=False`` in scipy terms): coefficient *k*
uses ``2*pi*k/N``, not ``2*pi*k/(N-1)``. Periodic is correct for spectral
analysis; symmetric is correct for FIR design. Mixing them produces a small but
real amplitude bias that is maddening to track down in the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

__all__ = [
    "WindowSpec",
    "window",
    "coherent_gain",
    "enbw_bins",
    "amplitude_correction",
    "energy_correction",
    "scallop_loss_db",
    "WINDOW_NAMES",
]

WINDOW_NAMES = ("rect", "hann", "hamming", "blackman", "blackman_harris", "flattop", "kaiser", "tukey")

# Cosine-sum coefficients. w[k] = sum_i (-1)^i a_i cos(2 pi i k / N) with the
# sign folded into the tabulated values below.
_COSINE_SUM: Dict[str, tuple] = {
    # Hann: 0.5 - 0.5 cos. -31 dB sidelobes, -1.42 dB scallop loss.
    "hann": (0.5, -0.5),
    # Hamming: -43 dB sidelobes, slightly wider main lobe than Hann.
    "hamming": (0.54, -0.46),
    # Blackman (exact): -58 dB.
    "blackman": (0.42, -0.5, 0.08),
    # 4-term Blackman-Harris: -92 dB sidelobes. Use when a strong 1x line would
    # otherwise mask a small bearing tone nearby.
    "blackman_harris": (0.35875, -0.48829, 0.14128, -0.01168),
    # 5-term flat-top: 0.0098 dB max scallop loss. Use whenever an *amplitude*
    # is being reported for a tone that is not bin-centred -- which, with real
    # shaft speed drift, is always.
    "flattop": (0.21557895, -0.41663158, 0.277263158, -0.083578947, 0.006947368),
}


@dataclass(frozen=True)
class WindowSpec:
    """A window plus the scale factors needed to report calibrated numbers.

    Attributes
    ----------
    name, size
        Identity of the window.
    coeff
        The coefficients, length ``size``.
    coherent_gain
        ``mean(coeff)``. Divide a windowed FFT magnitude by this to recover the
        true amplitude of a bin-centred sinusoid.
    enbw_bins
        Equivalent noise bandwidth in bins, ``N*sum(w^2)/sum(w)^2``. Divide a
        windowed power spectrum by ``enbw_bins * df`` to get a PSD in
        units^2/Hz.
    scallop_loss_db
        Worst-case amplitude error for a tone exactly between two bins. Quote
        this as the amplitude uncertainty of any spectral measurement made
        without interpolation.
    """

    name: str
    size: int
    coeff: np.ndarray
    coherent_gain: float
    enbw_bins: float
    scallop_loss_db: float

    @property
    def amplitude_correction(self) -> float:
        """Multiply ``|FFT|*2/N`` by this to get peak amplitude in input units."""
        return 1.0 / self.coherent_gain

    @property
    def energy_correction(self) -> float:
        """Multiply an RMS computed from the windowed signal by this."""
        return float(np.sqrt(self.size / np.sum(self.coeff**2)))


def _cosine_sum(name: str, n: int) -> np.ndarray:
    a = _COSINE_SUM[name]
    k = np.arange(n, dtype=np.float64)
    out = np.full(n, a[0], dtype=np.float64)
    for i, ai in enumerate(a[1:], start=1):
        out += ai * np.cos(2.0 * np.pi * i * k / n)
    return out


def _kaiser(n: int, beta: float) -> np.ndarray:
    # Periodic Kaiser: evaluate the symmetric form on n+1 points, drop the last.
    m = n  # so that alpha = m/2 and the sequence is periodic in n
    k = np.arange(n, dtype=np.float64)
    r = 2.0 * k / m - 1.0
    return np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - r * r))) / np.i0(beta)


def _tukey(n: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1:
        return _cosine_sum("hann", n)
    k = np.arange(n, dtype=np.float64)
    x = k / n
    w = np.ones(n, dtype=np.float64)
    taper = alpha / 2.0
    lo = x < taper
    hi = x > 1.0 - taper
    w[lo] = 0.5 * (1.0 + np.cos(np.pi * (x[lo] / taper - 1.0)))
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * ((x[hi] - 1.0) / taper + 1.0)))
    return w


def window(name: str = "hann", size: int = 4096, *, beta: float = 8.6, alpha: float = 0.25) -> WindowSpec:
    """Build a :class:`WindowSpec`.

    Parameters
    ----------
    name
        One of :data:`WINDOW_NAMES`.
    size
        Number of coefficients. Must be positive.
    beta
        Kaiser shape parameter (ignored otherwise). ``beta=8.6`` gives roughly
        -90 dB sidelobes.
    alpha
        Tukey taper fraction (ignored otherwise).

    Notes
    -----
    Selection guidance used across this platform:

    ======================  ==========================================
    measurement             window
    ======================  ==========================================
    tone amplitude (g, mm/s)  ``flattop``  (lowest amplitude error)
    broadband / PSD / noise   ``hann``     (best ENBW/leakage trade)
    small tone near a big one ``blackman_harris`` (deepest sidelobes)
    transient / impact capture ``rect``    (do not smear the impulse)
    ======================  ==========================================
    """
    if size <= 0:
        raise ValueError(f"window size must be positive, got {size}")
    key = name.lower()
    if key == "rect":
        coeff = np.ones(size, dtype=np.float64)
    elif key in _COSINE_SUM:
        coeff = _cosine_sum(key, size)
    elif key == "kaiser":
        coeff = _kaiser(size, beta)
    elif key == "tukey":
        coeff = _tukey(size, alpha)
    else:
        raise ValueError(f"unknown window {name!r}; expected one of {WINDOW_NAMES}")

    cg = float(np.mean(coeff))
    enbw = float(size * np.sum(coeff**2) / np.sum(coeff) ** 2)
    return WindowSpec(
        name=key,
        size=size,
        coeff=coeff,
        coherent_gain=cg,
        enbw_bins=enbw,
        scallop_loss_db=_scallop_loss_db(coeff),
    )


def _scallop_loss_db(coeff: np.ndarray) -> float:
    """Amplitude loss for a tone at the half-bin frequency, in dB (positive)."""
    n = coeff.size
    k = np.arange(n, dtype=np.float64)
    half_bin = np.exp(-2j * np.pi * 0.5 * k / n)
    peak = np.abs(np.sum(coeff))
    mid = np.abs(np.sum(coeff * half_bin))
    if peak <= 0 or mid <= 0:
        return float("inf")
    return float(-20.0 * np.log10(mid / peak))


# Thin functional wrappers -- handy in tests and in the C cross-check harness.
def coherent_gain(coeff: np.ndarray) -> float:
    return float(np.mean(coeff))


def enbw_bins(coeff: np.ndarray) -> float:
    n = coeff.size
    return float(n * np.sum(coeff**2) / np.sum(coeff) ** 2)


def amplitude_correction(coeff: np.ndarray) -> float:
    return 1.0 / coherent_gain(coeff)


def energy_correction(coeff: np.ndarray) -> float:
    return float(np.sqrt(coeff.size / np.sum(coeff**2)))


def scallop_loss_db(coeff: np.ndarray) -> float:
    return _scallop_loss_db(np.asarray(coeff, dtype=np.float64))
