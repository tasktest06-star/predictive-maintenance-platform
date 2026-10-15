"""How a 25.6 kHz (or 12 kHz) vibration window is presented to a model that was
pretrained on daily electricity demand.

This module is where the central difficulty of the branch lives, so it is worth
stating it plainly before the code.

A time-series foundation model takes **one univariate sequence of a few hundred
to a few thousand points** and was pretrained on sequences whose information is
in the *level* and its *trend/seasonality* at scales of hours to years. A
vibration acquisition is 2048-32768 points at 12-25.6 kHz whose diagnostic
information is almost entirely in (a) the distribution of energy across
frequency, and (b) the *modulation* of a high-frequency carrier at a kinematic
rate. Neither is a pattern in the raw sample sequence at the scale a 512-token
context can hold:

* at 12 kHz, 512 samples is **42.7 ms** -- about 1.2 shaft revolutions at
  1750 rpm, and about 4 outer-race impacts. There is no "seasonality" to find;
* the impacts ring the structure at 2-5 kHz, i.e. at 0.17-0.42 cycles/sample.
  A patch embedding with a 16- or 64-sample patch integrates straight over
  that, so the carrier is gone before the first attention layer;
* a bearing fault changes a 2-5 kHz band by 10-30 dB while moving the raw RMS
  by a factor of 2-10. Both Chronos and MOMENT **normalise each instance**
  before the encoder (Chronos by mean absolute value, MOMENT by instance
  norm). That normalisation is correct for forecasting and it deletes the
  single most informative vibration feature.

So "feed the waveform to Chronos" is not one experiment, it is a family of
them, and the choice of presentation decides the answer. Five presentations are
implemented and all five are run, because reporting only the one that loses
would be as dishonest as reporting only the one that wins:

``raw_crop``
    The centre 512 raw samples. Full bandwidth, one quarter of the record.
    This is the naive thing, and the thing most papers' "zero-shot on raw
    sensor data" means.
``raw_decim``
    The whole 2048-sample record, anti-alias filtered and decimated 4x to 512
    points. Keeps the whole record and throws away everything above fs/8 --
    i.e. exactly the band the bearing impacts excite. The pair
    ``raw_crop``/``raw_decim`` is a direct measurement of the time-span vs
    bandwidth trade a fixed 512-token context forces.
``envelope``
    Band-pass (a fixed fraction of Nyquist, never chosen from labels), Hilbert
    envelope, squared, low-pass and decimate to 512 points spanning the whole
    record. This is the *generous* presentation: it demodulates the carrier so
    that the impact-repetition rate becomes a low-frequency periodicity a
    512-point sequence can actually represent -- in other words it hands the
    foundation model the one piece of signal processing that 60 years of
    condition monitoring says you must do.
``logspec``
    512-point log magnitude spectrum, frequency axis fed in as if it were time.
    Common in the literature and structurally dishonest about what the model
    was pretrained on; included because it is common.
``envelope_spec``
    512-point log squared-envelope spectrum. The closest thing to "the answer
    written down", and the strongest presentation we can construct without
    using labels.

Everything here is built from ``pdmcore.dsp``; no filter, window or FFT is
reimplemented.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if os.path.join(_REPO, "core") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "core"))

from pdmcore.dsp.envelope import analytic_envelope, bandpass  # noqa: E402
from pdmcore.dsp.spectrum import spectrum as compute_spectrum  # noqa: E402

__all__ = ["PRESENTATIONS", "SEQ_LEN", "DEMOD_FRACTION", "present", "present_all",
           "window_level_db"]

#: Context length handed to every foundation model. 512 because MOMENT-1 has a
#: fixed 512-point input and Chronos-Bolt's patch grid is defined on 512; using
#: the same number for both keeps the comparison about the model and not about
#: the context budget.
SEQ_LEN = 512

#: Demodulation band as a fraction of Nyquist, FIXED and label-free.
#: (0.25, 0.90) x Nyquist is 1.5-5.4 kHz at 12 kHz and 12-44 kHz at 97.656 kHz.
#: Chosen a priori rather than by a kurtogram: ``select_demodulation_band`` is
#: data-adaptive, and letting the band move per window makes the presentation a
#: function of the signal under test, which would quietly smuggle a detector
#: into the feature definition.
DEMOD_FRACTION: Tuple[float, float] = (0.25, 0.90)

PRESENTATIONS: Tuple[str, ...] = (
    "raw_crop", "raw_decim", "envelope", "logspec", "envelope_spec",
)


def _resample_to(y: np.ndarray, n: int) -> np.ndarray:
    """Linear resample to exactly ``n`` points.

    Used only on signals that have already been low-passed below the new
    Nyquist (the envelope), or on a spectrum, where the axis is not time and
    interpolation is the honest operation.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == n:
        return y
    src = np.linspace(0.0, 1.0, y.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, y)


def _decimate_antialias(x: np.ndarray, fs_hz: float, n_out: int) -> np.ndarray:
    """Low-pass to the new Nyquist, then take every k-th sample.

    The low-pass is `pdmcore.dsp.envelope.bandpass` in low-pass mode, so the
    filter is the same zero-phase Butterworth the rest of the platform uses and
    the decimation cannot alias a 4 kHz resonance down onto a 400 Hz shaft
    order -- which is the single most common way a "downsampled vibration"
    pipeline invents a fault.
    """
    k = max(1, int(round(x.size / n_out)))
    if k == 1:
        return _resample_to(x, n_out)
    f_new_nyq = fs_hz / (2.0 * k)
    lp = bandpass(x, fs_hz, 0.0, 0.95 * f_new_nyq, order=4)
    return _resample_to(lp[::k], n_out)


def window_level_db(x: np.ndarray) -> float:
    """20*log10(RMS) of the window, in dB re 1 g.

    Kept separate from every presentation because all the foundation models
    normalise it away internally. It is handed back so that the comparison can
    be run both ways: embedding alone (what the model actually sees) and
    embedding + level (what the model would see if someone restored the one
    feature instance normalisation deletes). The second is strictly generous to
    the foundation-model arm.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    rms = float(np.sqrt(np.mean((x - np.mean(x)) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-12))


def present(x: np.ndarray, fs_hz: float, kind: str, *, seq_len: int = SEQ_LEN) -> np.ndarray:
    """One window -> one ``(seq_len,)`` float64 sequence."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    nyq = fs_hz / 2.0
    lo, hi = DEMOD_FRACTION[0] * nyq, DEMOD_FRACTION[1] * nyq

    if kind == "raw_crop":
        if x.size < seq_len:
            return _resample_to(x, seq_len)
        s = (x.size - seq_len) // 2
        return x[s:s + seq_len].copy()

    if kind == "raw_decim":
        return _decimate_antialias(x, fs_hz, seq_len)

    if kind == "envelope":
        env = analytic_envelope(bandpass(x, fs_hz, lo, hi, order=4)) ** 2
        # The envelope of a band of width (hi-lo) is band-limited to that
        # width, so decimating it to seq_len points over the whole record needs
        # the same anti-alias care as the raw path.
        return _decimate_antialias(env - np.mean(env), fs_hz, seq_len)

    if kind == "logspec":
        n_fft = 2 * seq_len
        spec = compute_spectrum(x, fs_hz, n_fft=n_fft, window_name="hann",
                                scaling="rms", detrend="mean")
        v = np.asarray(spec.value, dtype=np.float64)[1:]        # drop DC
        return 20.0 * np.log10(np.maximum(_resample_to(v, seq_len), 1e-12))

    if kind == "envelope_spec":
        env = analytic_envelope(bandpass(x, fs_hz, lo, hi, order=4)) ** 2
        env = env - np.mean(env)
        # Decimate the envelope to ~8x the highest kinematic rate of interest
        # before the FFT, exactly as pdmcore.dsp.envelope.envelope_spectrum
        # does, so the resolution goes where the information is.
        dec = _decimate_antialias(env, fs_hz, 2 * seq_len)
        fs_env = fs_hz * (2 * seq_len) / x.size
        spec = compute_spectrum(dec, fs_env, n_fft=2 * seq_len, window_name="hann",
                                scaling="peak", detrend="mean")
        v = np.asarray(spec.value, dtype=np.float64)[1:]
        return 20.0 * np.log10(np.maximum(_resample_to(v, seq_len), 1e-12))

    raise ValueError(f"unknown presentation {kind!r}; expected one of {PRESENTATIONS}")


def present_all(
    x: np.ndarray,
    fs_hz: np.ndarray,
    kinds: Sequence[str] = PRESENTATIONS,
    *,
    seq_len: int = SEQ_LEN,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``(n, L)`` windows -> ``{kind: (n, seq_len) float32}`` plus the levels.

    float32 because that is what crosses into the torch process; the sequences
    are only ever consumed by a model that casts to float32 anyway, and the
    cache is 4 MB per presentation instead of 8.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    fs = np.asarray(fs_hz, dtype=np.float64).ravel()
    if fs.size != x.shape[0]:
        raise ValueError("need one sample rate per window")
    out: Dict[str, List[np.ndarray]] = {k: [] for k in kinds}
    levels = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        levels[i] = window_level_db(x[i])
        for k in kinds:
            out[k].append(present(x[i], float(fs[i]), k, seq_len=seq_len))
    return ({k: np.asarray(np.vstack(v), dtype=np.float32) for k, v in out.items()},
            levels)
