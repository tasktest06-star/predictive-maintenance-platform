"""Three phases together: Clarke, Park, EPVA, instantaneous power, sequences.

These are the four methods that cannot be expressed on a single phase, which is
why they are the methods a three-phase panel device buys over a single clamp.

Clarke and Park
---------------
The Clarke (alpha-beta) transform of a balanced three-phase set is a circle in
the complex plane. The *power-invariant* form is used here,

    i_alpha = sqrt(2/3) (ia - ib/2 - ic/2)
    i_beta  = sqrt(2/3) (sqrt(3)/2)(ib - ic)

so that ``|i|^2`` is the instantaneous three-phase power for unit voltage and
the modulus of a balanced set with phase amplitude ``I`` is ``sqrt(3/2) I``.
The alternative "amplitude-invariant" 2/3 scaling gives modulus ``I`` and
differs by sqrt(3/2) = 1.2247, i.e. 1.76 dB. Both appear in the literature and
the factor is almost never stated; pinned here by
``test_clarke_is_power_invariant``.

The Park's vector approach (PVA) looks at the alpha-beta locus directly: a
healthy machine traces a circle, a faulty or unbalanced one an ellipse. It is a
*picture*, not a number, and the ellipse it draws for a broken bar looks like
the ellipse it draws for a 1 % supply unbalance.

EPVA
----
The extended Park's vector approach takes the spectrum of the *modulus* of the
space vector. The point is a coordinate change: for a fault that modulates the
current at ``fm``, the two sidebands at ``f1 +- fm`` become a single line at
``fm`` in the modulus spectrum, and the fundamental becomes DC.

That buys two things and costs one:

* The 40-60 dB dynamic-range problem disappears. The line being measured is no
  longer 1.3 Hz from a component 50 dB larger; it is 1.3 Hz from DC, and DC is
  removed by detrending. **This is the single strongest technical argument for
  three-phase acquisition.**
* Amplitude and frequency modulation are separated: the modulus carries AM and
  the angle carries FM.
* It costs the sideband *asymmetry*, which is the information that distinguishes
  a rotor asymmetry (lower sideband dominant) from a load oscillation
  (symmetric). EPVA is therefore more sensitive and less specific than the
  single-phase sideband measurement, so this module returns both and
  :mod:`pdmcore.mcsa.features` uses both.

Sequence components
-------------------
The symmetrical-component decomposition at the fundamental. The negative
sequence is the classical stator-turn-fault indicator -- and the classical trap,
because a supply voltage unbalance produces negative-sequence *current* through
the motor's (small) negative-sequence impedance, typically 5-8 times larger per
unit than the positive-sequence current per unit of voltage. A 1 % voltage
unbalance therefore produces 5-8 % negative-sequence current, which swamps the
signature of a shorted turn. :func:`negative_sequence_impedance` is the standard
answer: divide the negative-sequence current by the negative-sequence voltage,
so the supply's contribution cancels. It needs voltage, which is the one
measurement a current clamp alone cannot make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..dsp.spectrum import Spectrum, spectrum

__all__ = [
    "ParkVector",
    "clarke_transform",
    "park_vector",
    "epva_spectrum",
    "instantaneous_power",
    "sequence_components",
    "negative_sequence_impedance",
    "unbalance_pct",
]

#: Power-invariant Clarke scaling.
_CLARKE_K = float(np.sqrt(2.0 / 3.0))


@dataclass
class ParkVector:
    """The current space vector and its scalar summaries."""

    alpha: np.ndarray
    beta: np.ndarray
    #: Zero-sequence component; non-zero only on a 4-wire system or with a
    #: channel gain error. Carried because it is the cheapest instrument check
    #: available: on a 3-wire delta or star without neutral, ia+ib+ic == 0 by
    #: Kirchhoff, so a non-zero zero-sequence is a *measurement* fault.
    zero: np.ndarray
    fs_hz: float

    @property
    def modulus(self) -> np.ndarray:
        return np.sqrt(self.alpha ** 2 + self.beta ** 2)

    @property
    def complex_vector(self) -> np.ndarray:
        return self.alpha + 1j * self.beta

    def zero_sequence_ratio(self) -> float:
        """RMS zero sequence over RMS modulus. Should be ~0 on a 3-wire system."""
        m = float(np.sqrt(np.mean(self.modulus ** 2)))
        z = float(np.sqrt(np.mean(self.zero ** 2)))
        return float(z / m) if m > 0 else 0.0

    def ellipticity(self) -> float:
        """1 - minor/major axis of the alpha-beta locus, from its covariance.

        0 for a perfect circle, rising with any unbalance. This is the Park's
        vector picture reduced to one number -- and it is reported *with* the
        supply unbalance, never instead of it, because the two are not
        separable from current alone.
        """
        c = np.cov(np.vstack([self.alpha, self.beta]))
        w = np.linalg.eigvalsh(np.atleast_2d(c))
        w = np.sqrt(np.maximum(w, 0.0))
        if w[-1] <= 0:
            return 0.0
        return float(1.0 - w[0] / w[-1])


def clarke_transform(ia: Sequence[float], ib: Sequence[float], ic: Sequence[float],
                     fs_hz: float) -> ParkVector:
    """Power-invariant Clarke transform of three phase currents."""
    a = np.asarray(ia, dtype=np.float64).ravel()
    b = np.asarray(ib, dtype=np.float64).ravel()
    c = np.asarray(ic, dtype=np.float64).ravel()
    if not (a.size == b.size == c.size):
        raise ValueError(f"phase lengths differ: {a.size}, {b.size}, {c.size}")
    if a.size == 0:
        raise ValueError("empty input")
    alpha = _CLARKE_K * (a - 0.5 * b - 0.5 * c)
    beta = _CLARKE_K * (np.sqrt(3.0) / 2.0) * (b - c)
    zero = (a + b + c) / np.sqrt(3.0)
    return ParkVector(alpha=alpha, beta=beta, zero=zero, fs_hz=float(fs_hz))


def park_vector(ia: Sequence[float], ib: Sequence[float], ic: Sequence[float],
                fs_hz: float) -> ParkVector:
    """Alias for :func:`clarke_transform`.

    Park's transform proper rotates into the synchronous frame, which needs the
    rotor angle; the "Park's vector" of the MCSA literature is the stationary
    Clarke vector. The alias exists so that code can use the name the
    literature uses without the module pretending they are the same operation.
    """
    return clarke_transform(ia, ib, ic, fs_hz)


def epva_spectrum(
    pv: ParkVector,
    *,
    n_fft: Optional[int] = None,
    window_name: str = "blackman_harris",
    scaling: str = "rms",
    detrend: str = "mean",
) -> Spectrum:
    """Spectrum of the space-vector modulus: the EPVA signature.

    The DC term is removed by ``detrend="mean"``; what remains at ``fm`` is the
    amplitude-modulation depth of the space vector, in input units. Divide by
    the mean modulus to express it as a modulation index -- which is what
    :func:`pdmcore.mcsa.features.CurrentFeatureExtractor` stores, because a
    modulation index transfers between machines and an absolute ampere does not.
    """
    m = pv.modulus
    if n_fft is None:
        n_fft = 1 << int(np.floor(np.log2(max(2, m.size))))
    return spectrum(m, pv.fs_hz, n_fft=int(n_fft), window_name=window_name,
                    scaling=scaling, detrend=detrend)


def instantaneous_power(
    ia: Sequence[float], ib: Sequence[float], ic: Sequence[float],
    va: Sequence[float], vb: Sequence[float], vc: Sequence[float],
) -> np.ndarray:
    """Instantaneous three-phase power ``va ia + vb ib + vc ic``.

    The instantaneous-power spectrum carries the same fault sidebands as the
    current, plus a term at ``2 f1``, and its fault lines appear at ``2 s f1``
    and ``fm`` directly rather than as sidebands -- the same coordinate
    advantage EPVA has, obtained differently. It needs voltage, so it is a
    panel-device method and not a clamp method.
    """
    arrs = [np.asarray(x, dtype=np.float64).ravel() for x in (ia, ib, ic, va, vb, vc)]
    n = arrs[0].size
    if any(a.size != n for a in arrs):
        raise ValueError("all six channels must have the same length")
    a, b, c, u, v, w = arrs
    return a * u + b * v + c * w


def _fundamental_phasor(x: np.ndarray, fs_hz: float, f1_hz: float) -> complex:
    """Complex amplitude at ``f1`` by direct projection (one-bin DFT).

    A projection onto exp(-j2pi f1 t) over an integer-ish number of cycles,
    Hann-weighted so that a non-integer record length does not leak the
    neighbouring content into the estimate. Returns the *peak* amplitude.
    """
    n = x.size
    t = np.arange(n, dtype=np.float64) / fs_hz
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    ref = np.exp(-2j * np.pi * f1_hz * t)
    return complex(2.0 * np.sum(x * w * ref) / np.sum(w))


def sequence_components(
    ia: Sequence[float], ib: Sequence[float], ic: Sequence[float],
    fs_hz: float, f1_hz: float,
) -> Dict[str, complex]:
    """Positive, negative and zero sequence phasors at the fundamental.

    ``I+ = (Ia + a Ib + a^2 Ic)/3`` with ``a = exp(j2pi/3)``;
    ``I- = (Ia + a^2 Ib + a Ic)/3``; ``I0 = (Ia + Ib + Ic)/3``.

    The phase order is the convention a-b-c; a reversed clamp or a swapped pair
    exchanges ``I+`` and ``I-`` entirely, which turns a healthy motor into a
    100 % unbalanced one. :func:`unbalance_pct` is therefore reported with the
    *ratio*, and a ratio above ~0.5 should be read as "the instrument is wired
    wrong" rather than "the motor is destroyed".
    """
    pa = _fundamental_phasor(np.asarray(ia, dtype=np.float64).ravel(), fs_hz, f1_hz)
    pb = _fundamental_phasor(np.asarray(ib, dtype=np.float64).ravel(), fs_hz, f1_hz)
    pc = _fundamental_phasor(np.asarray(ic, dtype=np.float64).ravel(), fs_hz, f1_hz)
    a = np.exp(2j * np.pi / 3.0)
    return {
        "positive": (pa + a * pb + a * a * pc) / 3.0,
        "negative": (pa + a * a * pb + a * pc) / 3.0,
        "zero": (pa + pb + pc) / 3.0,
    }


def unbalance_pct(seq: Dict[str, complex]) -> float:
    """``100 |I-| / |I+|`` -- the IEC 61000 definition of unbalance factor."""
    p = abs(seq["positive"])
    return float(100.0 * abs(seq["negative"]) / p) if p > 0 else float("inf")


def negative_sequence_impedance(
    i_seq: Dict[str, complex], v_seq: Dict[str, complex]
) -> Dict[str, float]:
    """``Z- = V- / I-`` and the quantities needed to judge it.

    The point of the ratio: a supply unbalance raises ``V-`` and ``I-``
    together, leaving ``Z-`` unchanged, while a shorted turn raises ``I-``
    alone and lowers ``Z-``. So ``Z-`` separates the two and ``I-`` alone
    cannot.

    The catch, stated because it is why this indicator is hard: ``Z-`` of a
    healthy machine is 5-15 % of its positive-sequence impedance, so ``I-`` is
    a *large* multiple of the voltage unbalance and ``Z-`` is a ratio of two
    small numbers. Below ``v_unbalance_pct ~ 0.3`` both numerator and
    denominator are at the instrument's accuracy and the ratio is noise;
    ``valid`` is 0 there. The commonly cited sensitivity (a few percent change
    in ``Z-`` for one shorted turn) therefore requires a deliberately
    unbalanced supply or a very good instrument -- which is a real finding
    about this method, not a limitation of this implementation.
    """
    i_neg = abs(i_seq["negative"])
    v_neg = abs(v_seq["negative"])
    i_pos = abs(i_seq["positive"])
    v_pos = abs(v_seq["positive"])
    v_unb = 100.0 * v_neg / v_pos if v_pos > 0 else float("inf")
    valid = 1.0 if (v_unb >= 0.3 and i_neg > 0) else 0.0
    return {
        "z_neg_ohm": float(v_neg / i_neg) if i_neg > 0 else float("nan"),
        "z_pos_ohm": float(v_pos / i_pos) if i_pos > 0 else float("nan"),
        "z_ratio": float((v_neg / i_neg) / (v_pos / i_pos)) if i_neg > 0 and i_pos > 0 else float("nan"),
        "i_unbalance_pct": float(100.0 * i_neg / i_pos) if i_pos > 0 else float("inf"),
        "v_unbalance_pct": float(v_unb),
        "valid": valid,
    }
