"""Calibration tests for the spectral layer.

These are the tests that matter most in the whole repository. Every number a
customer sees -- mm/s on a dashboard, a g-level in a work order, an ISO zone --
flows through these functions, and a scaling error here is invisible in
integration tests but wrong by a constant factor forever.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdmcore.dsp import spectrum as sp_mod
from pdmcore.dsp import windows as win_mod
from pdmcore.dsp.spectrum import (
    find_tones,
    interpolate_peak,
    mel_filterbank,
    spectral_kurtosis,
    spectral_shape_features,
    spectrum,
    stft,
    third_octave_bands,
    welch_psd,
)

FS = 25600.0


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected_cg,expected_enbw",
    [
        ("rect", 1.0, 1.0),
        ("hann", 0.5, 1.5),
        ("hamming", 0.54, 1.3628),
        ("blackman", 0.42, 1.7269),
        ("blackman_harris", 0.35875, 2.0044),
        ("flattop", 0.21557895, 3.7702),
    ],
)
def test_window_gain_and_enbw(name, expected_cg, expected_enbw):
    """Coherent gain and ENBW must match the published values for each window."""
    w = win_mod.window(name, 4096)
    assert w.coherent_gain == pytest.approx(expected_cg, rel=1e-4)
    assert w.enbw_bins == pytest.approx(expected_enbw, rel=1e-3)


def test_window_scallop_loss_ordering():
    """Flat-top must have the lowest scallop loss; rect the highest."""
    losses = {n: win_mod.window(n, 2048).scallop_loss_db for n in
              ("rect", "hann", "hamming", "blackman_harris", "flattop")}
    assert losses["flattop"] < 0.05
    assert losses["rect"] > 3.0
    assert losses["flattop"] < losses["blackman_harris"] < losses["hann"] < losses["rect"]


def test_window_is_periodic_not_symmetric():
    """Periodic convention: w[0] == w[N] would be the symmetric form."""
    w = win_mod.window("hann", 8).coeff
    assert w[0] == pytest.approx(0.0)
    # For the symmetric form the last sample would also be 0; periodic is not.
    assert w[-1] > 0.1


def test_unknown_window_rejected():
    with pytest.raises(ValueError, match="unknown window"):
        win_mod.window("kaiser_bessel_derived", 256)


# ---------------------------------------------------------------------------
# Scaling -- the core invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scaling", ["rms", "peak", "power", "psd"])
@pytest.mark.parametrize("window_name", ["rect", "hann", "flattop", "blackman_harris"])
def test_band_rms_matches_time_domain_for_noise(scaling, window_name):
    """Parseval: band RMS over the full band == time-domain RMS, for every
    scaling and every window. This is the invariant that was violated by a
    factor of sqrt(2) before the single-sided fold was moved to the power
    domain, and by sqrt(ENBW) before rms_in_band divided it out."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(int(FS * 2))
    spec = spectrum(x, FS, n_fft=4096, scaling=scaling, window_name=window_name)
    assert spec.rms_in_band(0.0, FS / 2) == pytest.approx(np.sqrt(np.mean(x**2)), rel=0.02)


@pytest.mark.parametrize("scaling", ["rms", "peak", "power", "psd"])
def test_band_rms_matches_time_domain_for_tone(scaling):
    """Same invariant for a pure tone, including a deliberately off-bin one."""
    n_fft = 4096
    t = np.arange(int(FS * 2)) / FS
    f = 1000.37 * (FS / n_fft)  # 37 % of a bin away from bin centre
    x = 1.0 * np.sin(2 * np.pi * f * t)
    spec = spectrum(x, FS, n_fft=n_fft, scaling=scaling, window_name="hann")
    assert spec.rms_in_band(0.0, FS / 2) == pytest.approx(1.0 / np.sqrt(2), rel=0.02)


def test_peak_scaling_reads_tone_amplitude_with_flattop():
    """A flat-top window must read a tone's amplitude to <0.05 dB regardless of
    where it falls between bins -- that is the entire reason it exists."""
    n_fft = 4096
    t = np.arange(int(FS * 2)) / FS
    for frac in (0.0, 0.25, 0.5):
        f = (1000 + frac) * (FS / n_fft)
        x = 2.5 * np.sin(2 * np.pi * f * t)
        spec = spectrum(x, FS, n_fft=n_fft, scaling="peak", window_name="flattop")
        assert float(np.max(spec.value)) == pytest.approx(2.5, rel=0.01)


def test_hann_half_bin_tone_is_underread_without_interpolation():
    """Documents the error the flat-top window avoids: a Hann-windowed tone at
    the half-bin frequency reads ~1.42 dB low, and parabolic interpolation
    recovers most of it."""
    n_fft = 4096
    t = np.arange(int(FS * 2)) / FS
    f = 1000.5 * (FS / n_fft)
    x = np.sin(2 * np.pi * f * t)
    spec = spectrum(x, FS, n_fft=n_fft, scaling="peak", window_name="hann")
    raw = float(np.max(spec.value))
    assert raw == pytest.approx(10 ** (-1.42 / 20), rel=0.02)
    # Parabolic interpolation on the amplitude spectrum recovers -1.42 dB to
    # -0.60 dB; it does not fully recover it, because a Hann main lobe is not a
    # parabola. Quoting the residual honestly is the point: this is the
    # amplitude uncertainty of any Hann-windowed tone measurement.
    _f_i, v_i = interpolate_peak(spec, int(np.argmax(spec.value)))
    assert v_i > raw
    assert v_i == pytest.approx(10 ** (-0.60 / 20), rel=0.02)


def test_psd_is_independent_of_block_length():
    """A PSD must not change when n_fft changes; an amplitude spectrum must."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(int(FS * 4))
    psds = [welch_psd(x, FS, n_fft=n).value.mean() for n in (1024, 4096, 16384)]
    assert psds[0] == pytest.approx(psds[1], rel=0.05)
    assert psds[1] == pytest.approx(psds[2], rel=0.05)


def test_averaging_happens_in_power_domain():
    """Linear averaging of magnitudes under-reports noise by pi/4 (-1.05 dB).
    Averaging in the power domain is what keeps the Parseval test above
    passing with many averages."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(FS * 8))
    spec = spectrum(x, FS, n_fft=2048, scaling="rms", overlap=0.5)
    assert spec.n_averages > 50
    assert spec.rms_in_band(0.0, FS / 2) == pytest.approx(1.0, rel=0.02)


def test_peak_hold_averaging_exceeds_linear():
    """Peak-hold must be >= linear averaging everywhere, and strictly greater
    somewhere -- otherwise intermittent impacts are being averaged away."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(int(FS))
    x[12345:12400] += 20.0  # one transient
    lin = spectrum(x, FS, n_fft=1024, scaling="rms", average="linear")
    pk = spectrum(x, FS, n_fft=1024, scaling="rms", average="peak")
    assert np.all(pk.value >= lin.value - 1e-12)
    assert np.any(pk.value > lin.value * 1.5)


def test_dc_and_nyquist_are_not_doubled():
    """Single-sided folding must skip DC and Nyquist. A constant signal's RMS
    must come out equal to its level, not sqrt(2) times it."""
    x = np.full(4096, 3.0)
    spec = spectrum(x, FS, n_fft=4096, scaling="rms", window_name="rect", detrend="none")
    assert spec.value[0] == pytest.approx(3.0, rel=1e-6)


def test_invalid_scaling_and_average_rejected():
    x = np.random.default_rng(0).standard_normal(2048)
    with pytest.raises(ValueError, match="unknown scaling"):
        spectrum(x, FS, n_fft=1024, scaling="dBfs")
    with pytest.raises(ValueError, match="unknown average"):
        spectrum(x, FS, n_fft=1024, average="median")


def test_spectrum_rejects_bad_inputs():
    with pytest.raises(ValueError, match="empty input"):
        spectrum([], FS)
    with pytest.raises(ValueError, match="fs_hz must be positive"):
        spectrum([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="overlap"):
        spectrum(np.zeros(4096), FS, n_fft=1024, overlap=1.0)


def test_short_record_is_zero_padded_not_rejected():
    """A truncated acquisition must still produce a spectrum: dropping the
    record loses the only data from a sensor that may be about to die."""
    spec = spectrum(np.random.default_rng(0).standard_normal(500), FS, n_fft=1024)
    assert spec.n_fft == 1024
    assert spec.n_averages == 1


# ---------------------------------------------------------------------------
# Derived spectral quantities
# ---------------------------------------------------------------------------
def _impulse_signal(rate_hz, rng):
    n = int(FS)
    x = rng.standard_normal(n) * 0.05
    t_ir = np.arange(200) / FS
    ir = np.exp(-2 * np.pi * 0.02 * 5000 * t_ir) * np.sin(2 * np.pi * 5000 * t_ir)
    for start in range(0, n - 200, int(FS / rate_hz)):
        x[start : start + 200] += ir
    return x


def test_spectral_kurtosis_separates_impulsive_from_gaussian():
    """SK ~ 0 for Gaussian noise, clearly positive for a sparse impact train."""
    rng = np.random.default_rng(5)
    f, sk_noise = spectral_kurtosis(rng.standard_normal(int(FS)) * 0.05, FS, n_fft=256)
    _f, sk_imp = spectral_kurtosis(_impulse_signal(20, np.random.default_rng(5)), FS, n_fft=256)
    band = (f > 4000) & (f < 6000)
    assert abs(float(np.mean(sk_noise[band]))) < 0.3
    assert float(np.mean(sk_imp[band])) > 3.0


def test_spectral_kurtosis_needs_impulses_sparse_relative_to_frame_rate():
    """A real and easily-missed limitation: when the fault rate approaches the
    STFT frame rate, every frame contains an impact, the band envelope becomes
    constant, and SK collapses to ~0 -- so the kurtogram reports "nothing
    impulsive here" for a signal that is nothing but impacts.

    At n_fft=256 and fs=25.6 kHz the frame period is 10 ms (100 Hz). The test
    pins the failure so that n_fft selection in the kurtogram stays tied to the
    expected fault frequency rather than being treated as a free parameter."""
    band_mean = {}
    for rate in (20, 50, 100, 200):
        f, sk = spectral_kurtosis(_impulse_signal(rate, np.random.default_rng(5)), FS, n_fft=256)
        band = (f > 4000) & (f < 6000)
        band_mean[rate] = float(np.mean(sk[band]))
    assert band_mean[20] > 3.0
    assert band_mean[50] > 1.0
    assert abs(band_mean[100]) < 0.5      # fault rate == frame rate: SK blind
    assert band_mean[200] < 0.0           # faster than frame rate: SK negative


def test_third_octave_band_edges_follow_iec_61260():
    bands = third_octave_bands(20.0, 20000.0)
    centres = np.sqrt(bands[:, 0] * bands[:, 1])
    assert np.any(np.isclose(centres, 1000.0, rtol=1e-6))
    ratios = centres[1:] / centres[:-1]
    assert np.allclose(ratios, 10 ** (1 / 10), rtol=1e-6)


def test_mel_filterbank_is_normalised_and_contiguous():
    fb = mel_filterbank(40, 1024, FS)
    assert fb.shape == (40, 513)
    assert np.all(fb >= 0)
    # Every filter must contain at least one non-zero bin, else n_mels is too
    # high for this n_fft and frames would silently contain dead features.
    assert np.all(fb.sum(axis=1) > 0)


def test_mel_filterbank_rejects_bad_band():
    with pytest.raises(ValueError):
        mel_filterbank(32, 1024, FS, f_lo_hz=100.0, f_hi_hz=FS)


def test_spectral_shape_on_known_signals():
    """Centroid of a single tone sits at the tone; white noise sits near fs/4."""
    t = np.arange(int(FS)) / FS
    tone = spectrum(np.sin(2 * np.pi * 3000 * t), FS, n_fft=4096, scaling="psd")
    assert spectral_shape_features(tone)["centroid_hz"] == pytest.approx(3000.0, rel=0.02)
    noise = spectrum(np.random.default_rng(1).standard_normal(int(FS)), FS, n_fft=4096, scaling="psd")
    shape = spectral_shape_features(noise)
    assert shape["centroid_hz"] == pytest.approx(FS / 4, rel=0.15)
    assert shape["flatness"] > 0.4
    assert shape["entropy"] > 0.9


def test_find_tones_uses_local_floor():
    """Tones must be found over a sloped noise floor, and a pure noise record
    must not produce a long list of spurious peaks."""
    t = np.arange(int(FS)) / FS
    rng = np.random.default_rng(2)
    pink = np.cumsum(rng.standard_normal(t.size))
    pink /= np.std(pink)
    x = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 2400 * t) + 0.2 * pink
    spec = spectrum(x, FS, n_fft=8192, scaling="rms")
    tones = find_tones(spec, min_prominence_db=10.0)
    found = [tn["freq_hz"] for tn in tones]
    assert any(abs(f - 120.0) < 3.0 for f in found)
    assert any(abs(f - 2400.0) < 3.0 for f in found)
    noise_spec = spectrum(rng.standard_normal(int(FS)), FS, n_fft=8192, scaling="rms")
    assert len(find_tones(noise_spec, min_prominence_db=12.0)) <= 3


def test_stft_shape_and_time_axis():
    f, t, frames = stft(np.zeros(8192), FS, n_fft=1024, hop=256)
    assert frames.shape == (t.size, 513)
    assert f.size == 513
    assert t[0] == pytest.approx(512 / FS)


def test_spectrum_at_and_band_helpers():
    tt = np.arange(int(FS)) / FS
    spec = spectrum(np.sin(2 * np.pi * 500 * tt), FS, n_fft=4096, scaling="peak")
    assert spec.at(500.0) == pytest.approx(1.0, rel=0.05)
    assert spec.at(5000.0) < 0.01
    assert spec.rms_in_band(0.0, 10.0) < 0.01
    assert spec.enbw_bins == pytest.approx(1.5, rel=1e-3)
