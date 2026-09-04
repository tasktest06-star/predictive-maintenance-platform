/*
 * MEMS Low-Power Sensor — DSP Feature Extraction Header
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DSP_FEATURES_H
#define DSP_FEATURES_H

#include <arm_math.h>
#include <stdbool.h>
#include <stdint.h>

/* ──────────────────────────────────────────────────────────────────────────
 * Configuration constants
 * ──────────────────────────────────────────────────────────────────────── */

#define DSP_FFT_LEN         2048U   /* must be power of 2 */
#define BEARING_MAX_HARMONICS 4     /* store 2×, 3×, 4×, 5× harmonics */

/* ──────────────────────────────────────────────────────────────────────────
 * Bearing geometry — pre-computed frequency multiplier factors
 *
 * Usage:
 *   BPFO_Hz = BPFO_factor × (RPM / 60)
 *   BPFI_Hz = BPFI_factor × (RPM / 60)
 *   BSF_Hz  = BSF_factor  × (RPM / 60)
 *   FTF_Hz  = FTF_factor  × (RPM / 60)
 *
 * Example (6205-2RS deep groove ball bearing, 9 balls, contact angle 0°):
 *   BPFO_factor = 3.047   (N/2 × (1 - d/D))
 *   BPFI_factor = 4.953   (N/2 × (1 + d/D))
 *   BSF_factor  = 1.994   (D/2d × (1 - (d/D)²))
 *   FTF_factor  = 0.381   (1/2 × (1 - d/D))
 * ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    float32_t BPFO_factor;  /**< Ball pass frequency outer race multiplier */
    float32_t BPFI_factor;  /**< Ball pass frequency inner race multiplier */
    float32_t BSF_factor;   /**< Ball spin frequency multiplier */
    float32_t FTF_factor;   /**< Fundamental train frequency multiplier */
} bearing_geometry_t;

/* ──────────────────────────────────────────────────────────────────────────
 * Bearing fault feature output
 * ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    /* Fundamental fault frequency magnitudes */
    float32_t bpfo_mag;  /**< BPFO (1×) spectral magnitude */
    float32_t bpfi_mag;  /**< BPFI (1×) spectral magnitude */
    float32_t bsf_mag;   /**< BSF  (1×) spectral magnitude */
    float32_t ftf_mag;   /**< FTF  (1×) spectral magnitude */

    /* Harmonic magnitudes: index 0 = 2×, index 1 = 3×, ... index 3 = 5× */
    float32_t bpfo_harmonics[BEARING_MAX_HARMONICS];
    float32_t bpfi_harmonics[BEARING_MAX_HARMONICS];
    float32_t bsf_harmonics[BEARING_MAX_HARMONICS];
    float32_t ftf_harmonics[BEARING_MAX_HARMONICS];
} bearing_features_t;

/* ──────────────────────────────────────────────────────────────────────────
 * Public function declarations
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Apply a Hann window to data[] in-place.
 */
void apply_hann_window(float32_t *data, uint32_t len);

/**
 * @brief Compute RMS of a float signal.
 * @return RMS value (same units as input).
 */
float32_t compute_rms(float32_t *data, uint32_t len);

/**
 * @brief Compute kurtosis (4th standardised moment).
 * @return Kurtosis value; ~3 for Gaussian, >6 for impulsive faults.
 */
float32_t compute_kurtosis(float32_t *data, uint32_t len);

/**
 * @brief Compute crest factor = peak / RMS.
 * @return Crest factor; rising from ~3 indicates early bearing damage.
 */
float32_t compute_crest_factor(float32_t *data, uint32_t len);

/**
 * @brief Extract bearing fault frequency magnitudes from an FFT magnitude spectrum.
 *
 * Populates bpfo_mag, bpfi_mag, bsf_mag, ftf_mag and their 1×-5× harmonics.
 *
 * @param fft_mag     Magnitude spectrum (fft_len/2 bins, half-spectrum).
 * @param fft_len     Full FFT length (number of real input samples).
 * @param sample_rate Sampling rate in Hz.
 * @param rpm         Shaft speed in RPM.
 * @param bearing     Pre-computed bearing geometry factors.
 * @param out         Output feature structure.
 */
void compute_bearing_frequencies(float32_t *fft_mag,
                                 uint32_t fft_len,
                                 float32_t sample_rate,
                                 float32_t rpm,
                                 bearing_geometry_t *bearing,
                                 bearing_features_t *out);

/**
 * @brief Compute Squared Envelope Spectrum for advanced bearing diagnostics.
 *
 * Applies: bandpass filter → rectify → lowpass filter → FFT.
 * Output is suitable for input to compute_bearing_frequencies().
 *
 * @param data        Raw acceleration signal (DSP_FFT_LEN samples).
 * @param len         Number of samples (must equal DSP_FFT_LEN).
 * @param fft_mag_out Output magnitude spectrum (len/2 bins).
 */
void compute_envelope_spectrum(float32_t *data, uint32_t len,
                               float32_t *fft_mag_out);

/**
 * @brief Sum FFT energy in a given frequency band.
 *
 * @param fft_mag     Magnitude spectrum (fft_len/2 bins).
 * @param fft_len     Full FFT length.
 * @param sample_rate Sampling rate in Hz.
 * @param freq_lo     Lower band edge in Hz.
 * @param freq_hi     Upper band edge in Hz.
 * @return Sum of squared magnitudes in the band (Parseval energy estimate).
 */
float32_t compute_band_energy(const float32_t *fft_mag,
                              uint32_t fft_len,
                              float32_t sample_rate,
                              float32_t freq_lo,
                              float32_t freq_hi);

#endif /* DSP_FEATURES_H */
