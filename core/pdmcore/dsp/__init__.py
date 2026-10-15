"""Deterministic signal processing.

Pure functions, float64, no I/O, no global state.

The hot paths here (windowing, real FFT, time-domain indicators, envelope
demodulation) have a portable C99 counterpart in ``cdsp/`` on each ``hw/*``
branch, cross-checked against this implementation by generated golden vectors so
that the number a microcontroller computes equals the number the cloud
computes.
"""

from __future__ import annotations

from .bearing import (
    BEARING_CATALOG,
    BearingGeometry,
    FaultFrequencies,
    bearing_evidence,
    fault_frequencies,
    gear_mesh_frequency,
    harmonic_energy_ratio,
    line_snr,
    match_tones,
    sideband_ratio,
)
from .envelope import (
    DemodBand,
    analytic_envelope,
    bandpass,
    cepstrum,
    cepstrum_prewhiten,
    envelope_spectrum,
    kurtogram,
    select_demodulation_band,
)
from .features import TIME_FEATURE_NAMES, TimeFeatures, running_stats, time_feature_vector, time_features
from .integrate import (
    G_TO_MM_S2,
    accel_g_to_displacement_um,
    accel_g_to_velocity_mm_s,
    integrate_spectrum,
    velocity_rms_mm_s,
    velocity_spectrum,
)
from .iso import ISO_20816_3_LIMITS, IsoClass, classify_machine, iso_zone, iso_zone_margin
from .ordertrack import (
    SpeedProfile,
    angular_resample,
    estimate_speed_tacholess,
    order_spectrum,
    tacho_to_phase,
    time_synchronous_average,
)
from .spectrum import (
    Spectrum,
    band_rms,
    find_tones,
    interpolate_peak,
    log_mel_spectrogram,
    mel_filterbank,
    spectral_kurtosis,
    spectral_shape_features,
    spectrum,
    stft,
    third_octave_bands,
    welch_psd,
)
from .windows import WINDOW_NAMES, WindowSpec, window

__all__ = [
    # windows
    "WindowSpec", "window", "WINDOW_NAMES",
    # spectrum
    "Spectrum", "spectrum", "welch_psd", "stft", "band_rms", "third_octave_bands",
    "mel_filterbank", "log_mel_spectrogram", "spectral_kurtosis", "spectral_shape_features",
    "find_tones", "interpolate_peak",
    # features
    "TimeFeatures", "time_features", "time_feature_vector", "TIME_FEATURE_NAMES", "running_stats",
    # envelope
    "DemodBand", "bandpass", "analytic_envelope", "envelope_spectrum",
    "select_demodulation_band", "kurtogram", "cepstrum", "cepstrum_prewhiten",
    # bearing
    "BearingGeometry", "FaultFrequencies", "BEARING_CATALOG", "fault_frequencies",
    "gear_mesh_frequency", "harmonic_energy_ratio", "line_snr", "sideband_ratio", "match_tones",
    "bearing_evidence",
    # integrate
    "accel_g_to_velocity_mm_s", "accel_g_to_displacement_um", "velocity_rms_mm_s",
    "velocity_spectrum", "integrate_spectrum", "G_TO_MM_S2",
    # iso
    "IsoClass", "ISO_20816_3_LIMITS", "iso_zone", "iso_zone_margin", "classify_machine",
    # order tracking
    "SpeedProfile", "tacho_to_phase", "estimate_speed_tacholess", "angular_resample",
    "order_spectrum", "time_synchronous_average",
]
