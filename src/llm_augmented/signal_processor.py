from __future__ import annotations

import numpy as np
from scipy import signal, stats

from src.common.models import SensorReading


class SignalProcessor:
    def _harmonic_power(self, freqs: np.ndarray, psd: np.ndarray, center_hz: float, bw_hz: float = 5.0) -> float:
        mask = (freqs >= center_hz - bw_hz) & (freqs <= center_hz + bw_hz)
        return float(np.sum(psd[mask])) if mask.any() else 0.0

    def extract_features(self, reading: SensorReading) -> np.ndarray:
        desc = self.describe(reading)
        h = desc["harmonic_powers"]
        return np.array([
            desc["rms_g"],
            desc["peak_g"],
            desc["kurtosis"],
            desc["spectral_entropy"],
            desc["dominant_freq_hz"],
            desc["temperature_c"],
            desc["rpm"],
            h["1x"], h["2x"], h["3x"], h["4x"],
        ], dtype=np.float32)

    def describe(self, reading: SensorReading) -> dict:
        vib = reading.vibration_x
        n = len(vib)
        sr = reading.sample_rate

        rms = float(np.sqrt(np.mean(vib ** 2)))
        peak = float(np.max(np.abs(vib)))
        kurt = float(stats.kurtosis(vib))

        freqs, psd = signal.welch(vib, fs=sr, nperseg=min(n, 1024))

        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
        else:
            spectral_entropy = 0.0

        dominant_freq = float(freqs[np.argmax(psd)])
        # Guard against zero RPM to avoid DC-component aliasing in harmonic bins
        base_freq = max(reading.rpm, 1.0) / 60.0

        harmonics = {
            "1x": self._harmonic_power(freqs, psd, base_freq),
            "2x": self._harmonic_power(freqs, psd, 2 * base_freq),
            "3x": self._harmonic_power(freqs, psd, 3 * base_freq),
            "4x": self._harmonic_power(freqs, psd, 4 * base_freq),
        }

        return {
            "rms_g": rms,
            "peak_g": peak,
            "dominant_freq_hz": dominant_freq,
            "temperature_c": reading.temperature,
            "rpm": reading.rpm,
            "harmonic_powers": harmonics,
            "spectral_entropy": spectral_entropy,
            "kurtosis": kurt,
        }
