from __future__ import annotations

import numpy as np
from scipy import stats

from src.common.models import SensorReading


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _crest_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    return float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0


def _shape_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    mean_abs = float(np.mean(np.abs(x)))
    return rms / mean_abs if mean_abs > 0 else 0.0


def _impulse_factor(x: np.ndarray) -> float:
    mean_abs = float(np.mean(np.abs(x)))
    return float(np.max(np.abs(x)) / mean_abs) if mean_abs > 0 else 0.0


def _time_domain_features(x: np.ndarray) -> tuple[list[float], list[str]]:
    feats = [
        _rms(x),
        float(np.max(np.abs(x))),
        _crest_factor(x),
        float(stats.kurtosis(x)),
        float(stats.skew(x)),
        float(np.max(x) - np.min(x)),
        float(np.var(x)),
        _shape_factor(x),
        _impulse_factor(x),
    ]
    names = ["rms", "peak", "crest_factor", "kurtosis", "skewness",
             "peak_to_peak", "variance", "shape_factor", "impulse_factor"]
    return feats, names


def _freq_domain_features(
    x: np.ndarray, sample_rate: int, rpm: float
) -> tuple[list[float], list[str]]:
    n = len(x)
    window = np.hanning(n)
    x_w = x * window
    fft_vals = np.abs(np.fft.rfft(x_w)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    total_power = np.sum(fft_vals ** 2) + 1e-12
    norm_power = fft_vals ** 2 / total_power

    # Dominant frequency
    dom_freq = float(freqs[np.argmax(fft_vals[1:]) + 1])  # skip DC

    # Spectral centroid
    spectral_centroid = float(np.sum(freqs * norm_power))

    # Spectral bandwidth
    spectral_bw = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * norm_power)))

    # Spectral entropy
    p = norm_power + 1e-12
    spectral_entropy = float(-np.sum(p * np.log2(p)))

    # Power in RPM harmonic bands (±5 Hz around 1x, 2x, 3x, 4x shaft frequency)
    shaft_freq = rpm / 60.0
    harmonic_powers: list[float] = []
    harmonic_names: list[str] = []
    for h in range(1, 5):
        center = shaft_freq * h
        band_mask = (freqs >= center - 5) & (freqs <= center + 5)
        harmonic_powers.append(float(np.sum(fft_vals[band_mask] ** 2)))
        harmonic_names.append(f"harmonic_{h}x_power")

    feats = [dom_freq, spectral_centroid, spectral_bw, spectral_entropy] + harmonic_powers
    names = ["dom_freq", "spectral_centroid", "spectral_bw", "spectral_entropy"] + harmonic_names
    return feats, names


def _cross_axis_features(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[list[float], list[str]]:
    feats = [
        float(np.corrcoef(x, y)[0, 1]),
        float(np.corrcoef(x, z)[0, 1]),
        float(np.corrcoef(y, z)[0, 1]),
    ]
    names = ["corr_xy", "corr_xz", "corr_yz"]
    return feats, names


class FeatureExtractor:
    def __init__(self) -> None:
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            # Build names via a dummy extract call shape reference
            dummy = self._build_names()
            self._feature_names = dummy
        return self._feature_names

    def _build_names(self) -> list[str]:
        names: list[str] = []
        for axis in ("x", "y", "z"):
            _, axis_names = _time_domain_features(np.zeros(128))
            names += [f"{axis}_{n}" for n in axis_names]
        for axis in ("x", "y", "z"):
            _, freq_names = _freq_domain_features(np.zeros(128), 12800, 1500.0)
            names += [f"{axis}_{n}" for n in freq_names]
        _, cross_names = _cross_axis_features(
            np.zeros(128), np.zeros(128), np.zeros(128)
        )
        names += cross_names
        names += ["temperature", "rpm"]
        return names

    def extract(self, reading: SensorReading) -> np.ndarray:
        feats: list[float] = []

        for vib in (reading.vibration_x, reading.vibration_y, reading.vibration_z):
            f, _ = _time_domain_features(vib)
            feats.extend(f)

        for vib in (reading.vibration_x, reading.vibration_y, reading.vibration_z):
            f, _ = _freq_domain_features(vib, reading.sample_rate, reading.rpm)
            feats.extend(f)

        f, _ = _cross_axis_features(
            reading.vibration_x, reading.vibration_y, reading.vibration_z
        )
        feats.extend(f)

        feats.append(reading.temperature)
        feats.append(reading.rpm)

        vec = np.array(feats, dtype=np.float64)
        # Replace any NaN/Inf that can arise from degenerate signals
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
