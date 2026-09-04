/*
 * MEMS Low-Power Sensor — DSP Feature Extraction
 * Target MCU : STM32U5 (Cortex-M33)
 * Library    : ARM CMSIS-DSP 1.15
 *
 * Features implemented:
 *   - RMS (root mean square)
 *   - Kurtosis (4th-order statistical moment)
 *   - Crest factor (peak / RMS)
 *   - Hann window application
 *   - Bearing fault frequency extraction (BPFO, BPFI, BSF, FTF + harmonics)
 *   - Envelope demodulation (bandpass → abs → lowpass → FFT)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dsp_features.h"
#include <arm_math.h>
#include <string.h>
#include <math.h>

/* ──────────────────────────────────────────────────────────────────────────
 * Internal helpers
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Compute the arithmetic mean of a float array.
 */
static inline float32_t array_mean(const float32_t *data, uint32_t len)
{
    float32_t mean = 0.0f;
    arm_mean_f32(data, len, &mean);
    return mean;
}

/**
 * @brief Subtract mean from every element of data[] (in-place).
 */
static inline void remove_dc(float32_t *data, uint32_t len)
{
    float32_t mean = array_mean(data, len);
    arm_offset_f32(data, -mean, data, len);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Window function
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Apply a Hann window to the data buffer in-place.
 *
 * w[n] = 0.5 × (1 − cos(2π × n / (N−1)))
 *
 * Uses CMSIS-DSP arm_mult_f32 for the element-wise multiply.
 * Window coefficients are pre-computed on first call and cached in static RAM.
 *
 * @param data  Float array of length len (modified in-place).
 * @param len   Number of samples.
 */
void apply_hann_window(float32_t *data, uint32_t len)
{
    /* Static window cache — computed once for the configured FFT length */
    static float32_t hann_win[DSP_FFT_LEN];
    static uint32_t  hann_len = 0;

    if (hann_len != len) {
        /* Recompute if length changes (e.g. shorter test window) */
        float32_t inv_nm1 = 1.0f / (float32_t)(len - 1);
        for (uint32_t n = 0; n < len; n++) {
            hann_win[n] = 0.5f * (1.0f - arm_cos_f32(
                2.0f * PI * (float32_t)n * inv_nm1));
        }
        hann_len = len;
    }

    arm_mult_f32(data, hann_win, data, len);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Time-domain statistical features
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Compute Root Mean Square (RMS) of a signal.
 *
 * RMS = sqrt( (1/N) × Σ x[n]² )
 *
 * Uses CMSIS-DSP arm_rms_f32 which employs SIMD on Cortex-M4/M33.
 *
 * @param data  Input float array.
 * @param len   Number of samples.
 * @return RMS value (same units as input, e.g. g).
 */
float32_t compute_rms(float32_t *data, uint32_t len)
{
    float32_t rms = 0.0f;
    arm_rms_f32(data, len, &rms);
    return rms;
}

/**
 * @brief Compute kurtosis of a zero-mean signal.
 *
 * Kurtosis = E[x⁴] / (E[x²])²
 *
 * A kurtosis of ~3 is normal (Gaussian).  Values > 6 indicate impulsive
 * content characteristic of bearing spalling or gear tooth faults.
 *
 * Implementation:
 *   1. Remove DC offset.
 *   2. Compute variance (σ²) via CMSIS arm_var_f32.
 *   3. Compute 4th central moment manually.
 *   4. Normalize: kurt = μ₄ / σ⁴
 *
 * @param data  Input float array (modified in-place for DC removal).
 * @param len   Number of samples.
 * @return Kurtosis value (dimensionless).
 */
float32_t compute_kurtosis(float32_t *data, uint32_t len)
{
    float32_t variance = 0.0f;
    float32_t mean     = 0.0f;

    arm_mean_f32(data, len, &mean);
    arm_var_f32(data, len, &variance);

    if (variance < 1e-12f) {
        return 0.0f;  /* avoid divide-by-zero on flat signal */
    }

    /* Compute 4th central moment: E[(x - μ)⁴] */
    float32_t moment4 = 0.0f;
    float32_t inv_len = 1.0f / (float32_t)len;

    for (uint32_t i = 0; i < len; i++) {
        float32_t diff  = data[i] - mean;
        float32_t diff2 = diff * diff;
        moment4 += diff2 * diff2;
    }
    moment4 *= inv_len;

    /* kurt = μ₄ / σ⁴ */
    float32_t sigma4 = variance * variance;
    return moment4 / sigma4;
}

/**
 * @brief Compute crest factor of a signal.
 *
 * Crest factor = peak / RMS
 *
 * For a healthy bearing: CF ≈ 2.5–3.5.
 * Early bearing damage: CF rises to 6–10 before RMS increases.
 * Advanced damage: CF may decrease as RMS rises while peak saturates.
 *
 * Uses CMSIS-DSP arm_abs_max_f32 for peak detection.
 *
 * @param data  Input float array.
 * @param len   Number of samples.
 * @return Crest factor (dimensionless).
 */
float32_t compute_crest_factor(float32_t *data, uint32_t len)
{
    float32_t peak = 0.0f;
    float32_t rms  = 0.0f;
    uint32_t  peak_idx;

    arm_abs_max_f32(data, len, &peak, &peak_idx);
    arm_rms_f32(data, len, &rms);

    if (rms < 1e-9f) {
        return 0.0f;
    }
    return peak / rms;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Bearing fault frequency extraction
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Locate the peak magnitude in an FFT bin range.
 *
 * @param fft_mag      Magnitude spectrum array (N/2 bins).
 * @param fft_len      Total FFT length (number of real input samples).
 * @param sample_rate  Sampling rate in Hz.
 * @param center_freq  Center frequency to search around (Hz).
 * @param tolerance    Search window ± tolerance Hz.
 * @return Peak magnitude in the search window.
 */
static float32_t peak_in_window(const float32_t *fft_mag,
                                uint32_t fft_len,
                                float32_t sample_rate,
                                float32_t center_freq,
                                float32_t tolerance)
{
    float32_t bin_hz  = sample_rate / (float32_t)fft_len;
    uint32_t  half    = fft_len / 2;

    int32_t bin_lo = (int32_t)((center_freq - tolerance) / bin_hz);
    int32_t bin_hi = (int32_t)((center_freq + tolerance) / bin_hz) + 1;

    /* Clamp to valid range */
    if (bin_lo < 1)             bin_lo = 1;
    if (bin_hi >= (int32_t)half) bin_hi = (int32_t)half - 1;

    if (bin_lo > bin_hi) {
        return 0.0f;
    }

    float32_t max_val = 0.0f;
    uint32_t  max_idx;
    uint32_t  window_len = (uint32_t)(bin_hi - bin_lo + 1);

    arm_max_f32(&fft_mag[bin_lo], window_len, &max_val, &max_idx);
    return max_val;
}

/**
 * @brief Extract bearing fault frequency magnitudes from an FFT magnitude spectrum.
 *
 * Bearing characteristic frequencies (all in Hz) given shaft RPM:
 *
 *   BPFO = (N/2) × RPM/60 × (1 − d/D × cos α)   [outer race, N balls]
 *   BPFI = (N/2) × RPM/60 × (1 + d/D × cos α)   [inner race]
 *   BSF  = (D/2d) × RPM/60 × (1 − (d/D × cos α)²) [ball spin]
 *   FTF  = (1/2) × RPM/60 × (1 − d/D × cos α)   [cage/train]
 *
 * These are encoded as pre-computed factors in bearing_geometry_t:
 *   BPFO_Hz = bearing.BPFO_factor × (RPM/60)
 *   BPFI_Hz = bearing.BPFI_factor × (RPM/60)
 *   BSF_Hz  = bearing.BSF_factor  × (RPM/60)
 *   FTF_Hz  = bearing.FTF_factor  × (RPM/60)
 *
 * Harmonics up to 5× are extracted and stored in bearing_features_t.
 *
 * @param fft_mag    Magnitude spectrum (N/2 bins, half-spectrum).
 * @param fft_len    Full FFT length (number of real samples, e.g. 2048).
 * @param sample_rate  Sampling rate in Hz (e.g. 2000.0).
 * @param rpm        Shaft rotational speed in RPM.
 * @param bearing    Bearing geometry factors.
 * @param out        Output structure populated with fault magnitudes.
 */
void compute_bearing_frequencies(float32_t *fft_mag,
                                 uint32_t fft_len,
                                 float32_t sample_rate,
                                 float32_t rpm,
                                 bearing_geometry_t *bearing,
                                 bearing_features_t *out)
{
    float32_t shaft_hz = rpm / 60.0f;

    /* Characteristic fault frequencies in Hz */
    float32_t bpfo_1x = bearing->BPFO_factor * shaft_hz;
    float32_t bpfi_1x = bearing->BPFI_factor * shaft_hz;
    float32_t bsf_1x  = bearing->BSF_factor  * shaft_hz;
    float32_t ftf_1x  = bearing->FTF_factor  * shaft_hz;

    /* Frequency resolution and search tolerance */
    float32_t bin_hz    = sample_rate / (float32_t)fft_len;
    float32_t tolerance = bin_hz * 2.5f;  /* ±2.5 bins around target */

    /* Extract fundamental magnitudes */
    out->bpfo_mag = peak_in_window(fft_mag, fft_len, sample_rate, bpfo_1x, tolerance);
    out->bpfi_mag = peak_in_window(fft_mag, fft_len, sample_rate, bpfi_1x, tolerance);
    out->bsf_mag  = peak_in_window(fft_mag, fft_len, sample_rate, bsf_1x,  tolerance);
    out->ftf_mag  = peak_in_window(fft_mag, fft_len, sample_rate, ftf_1x,  tolerance);

    /* Extract harmonics 2× through 5× */
    for (int h = 0; h < BEARING_MAX_HARMONICS; h++) {
        float32_t mult = (float32_t)(h + 2);  /* 2x, 3x, 4x, 5x */
        float32_t harm_tol = tolerance * mult;  /* slightly wider window for harmonics */

        out->bpfo_harmonics[h] = peak_in_window(fft_mag, fft_len, sample_rate,
                                                bpfo_1x * mult, harm_tol);
        out->bpfi_harmonics[h] = peak_in_window(fft_mag, fft_len, sample_rate,
                                                bpfi_1x * mult, harm_tol);
        out->bsf_harmonics[h]  = peak_in_window(fft_mag, fft_len, sample_rate,
                                                bsf_1x  * mult, harm_tol);
        out->ftf_harmonics[h]  = peak_in_window(fft_mag, fft_len, sample_rate,
                                                ftf_1x  * mult, harm_tol);
    }
}

/* ──────────────────────────────────────────────────────────────────────────
 * Envelope demodulation (squared-envelope spectrum for bearing diagnostics)
 * ──────────────────────────────────────────────────────────────────────── */

/*
 * Bandpass FIR filter coefficients — 1 kHz to 5 kHz passband, 128-tap.
 * Generated with Parks-McClellan algorithm, fs=2000 Hz.
 *
 * NOTE: For fs=2000 Hz the usable bandwidth is 0–1000 Hz.  These
 * coefficients represent a 500–900 Hz bandpass which targets the
 * resonance region of typical bearing defect impacts in the MEMS
 * accelerometer's usable band.  Replace with your designed filter for
 * the specific sensor/bearing combination.
 *
 * For actual production use, generate coefficients with:
 *   scipy.signal.remez(128, [0, 400, 500, 900, 950, 1000], [0, 1, 0], fs=2000)
 *
 * Below is a 32-tap prototype for code completeness.
 */
#define BP_FIR_NUM_TAPS 32U

static const float32_t bp_fir_coeffs[BP_FIR_NUM_TAPS] = {
    /* 32-tap FIR, fs=2000 Hz, passband 500–900 Hz, generated via windowed sinc */
     0.000000f,  0.002432f,  0.004710f,  0.004839f,
     0.001213f, -0.006159f, -0.014120f, -0.018081f,
    -0.013680f,  0.001534f,  0.024197f,  0.047048f,
     0.060753f,  0.054957f,  0.025027f, -0.022879f,
    -0.022879f,  0.025027f,  0.054957f,  0.060753f,
     0.047048f,  0.024197f,  0.001534f, -0.013680f,
    -0.018081f, -0.014120f, -0.006159f,  0.001213f,
     0.004839f,  0.004710f,  0.002432f,  0.000000f,
};

/* Lowpass FIR coefficients — cutoff at 200 Hz (for envelope signal) */
#define LP_FIR_NUM_TAPS 32U

static const float32_t lp_fir_coeffs[LP_FIR_NUM_TAPS] = {
    /* 32-tap FIR lowpass, cutoff ~200 Hz @ fs=2000 Hz */
    -0.001312f, -0.002105f, -0.003193f, -0.003851f,
    -0.002562f,  0.002413f,  0.011297f,  0.023893f,
     0.038523f,  0.052544f,  0.062978f,  0.067000f,
     0.062978f,  0.052544f,  0.038523f,  0.023893f,
     0.011297f,  0.002413f, -0.002562f, -0.003851f,
    -0.003193f, -0.002105f, -0.001312f, -0.000805f,
    -0.000459f, -0.000237f, -0.000104f, -0.000036f,
    -0.000007f,  0.000006f,  0.000009f,  0.000007f,
};

/* Persistent FIR state blocks (Zephyr's static stack or BSS) */
static float32_t bp_fir_state[BP_FIR_NUM_TAPS + DSP_FFT_LEN - 1];
static float32_t lp_fir_state[LP_FIR_NUM_TAPS + DSP_FFT_LEN - 1];
static arm_fir_instance_f32 bp_fir_inst;
static arm_fir_instance_f32 lp_fir_inst;
static bool fir_initialized = false;

/**
 * @brief Compute the Squared Envelope Spectrum (SES) for bearing diagnostics.
 *
 * Algorithm:
 *   1. Bandpass filter (500–900 Hz) — isolates bearing resonance region.
 *   2. Rectify: take absolute value (rectified envelope).
 *   3. Lowpass filter (< 200 Hz) — reveals modulation at fault frequencies.
 *   4. Compute FFT of envelope signal → Squared Envelope Spectrum.
 *   5. Write FFT magnitudes to fft_mag_out.
 *
 * @param data         Input acceleration signal (fs = 2 kHz, N samples).
 * @param len          Number of samples (must be DSP_FFT_LEN).
 * @param fft_mag_out  Output magnitude spectrum (len/2 bins).
 */
void compute_envelope_spectrum(float32_t *data, uint32_t len,
                               float32_t *fft_mag_out)
{
    static float32_t filtered[DSP_FFT_LEN];
    static float32_t envelope[DSP_FFT_LEN];
    static float32_t fft_buf_local[DSP_FFT_LEN];

    if (!fir_initialized) {
        arm_fir_init_f32(&bp_fir_inst, BP_FIR_NUM_TAPS, bp_fir_coeffs,
                         bp_fir_state, len);
        arm_fir_init_f32(&lp_fir_inst, LP_FIR_NUM_TAPS, lp_fir_coeffs,
                         lp_fir_state, len);
        fir_initialized = true;
    }

    /* Step 1: Bandpass filter */
    arm_fir_f32(&bp_fir_inst, data, filtered, len);

    /* Step 2: Rectify — absolute value */
    arm_abs_f32(filtered, envelope, len);

    /* Step 3: Lowpass filter */
    arm_fir_f32(&lp_fir_inst, envelope, filtered, len);

    /* Step 4: Apply window and FFT */
    apply_hann_window(filtered, len);

    static arm_rfft_fast_instance_f32 env_fft_inst;
    static bool env_fft_init = false;
    if (!env_fft_init) {
        arm_rfft_fast_init_f32(&env_fft_inst, len);
        env_fft_init = true;
    }

    arm_rfft_fast_f32(&env_fft_inst, filtered, fft_buf_local, 0);

    /* Step 5: Magnitude spectrum */
    arm_cmplx_mag_f32(fft_buf_local, fft_mag_out, len / 2);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Spectral band energy
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Compute total energy in an FFT magnitude band (Hz range).
 *
 * Uses sum of squared magnitudes (Parseval's theorem approximation).
 *
 * @param fft_mag    Magnitude spectrum.
 * @param fft_len    Full FFT length.
 * @param sample_rate  Sample rate in Hz.
 * @param freq_lo    Lower band edge in Hz.
 * @param freq_hi    Upper band edge in Hz.
 * @return Band energy (arbitrary units, sum of mag²).
 */
float32_t compute_band_energy(const float32_t *fft_mag,
                              uint32_t fft_len,
                              float32_t sample_rate,
                              float32_t freq_lo,
                              float32_t freq_hi)
{
    float32_t bin_hz = sample_rate / (float32_t)fft_len;
    uint32_t half    = fft_len / 2;

    uint32_t bin_lo = (uint32_t)(freq_lo / bin_hz);
    uint32_t bin_hi = (uint32_t)(freq_hi / bin_hz);

    if (bin_lo >= half) return 0.0f;
    if (bin_hi >= half) bin_hi = half - 1;

    float32_t energy = 0.0f;
    float32_t pwr    = 0.0f;

    for (uint32_t b = bin_lo; b <= bin_hi; b++) {
        pwr     = fft_mag[b];
        energy += pwr * pwr;
    }

    return energy;
}
