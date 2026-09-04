/**
 * vibration_hf.h — High-frequency vibration processing function prototypes
 */

#pragma once

#include <stdint.h>
#include <zephyr/drivers/spi.h>

/**
 * vib_features_t — extracted vibration and envelope spectrum features
 */
typedef struct {
    float bpfo_amplitude;    /* Bearing outer race defect amplitude (envelope spectrum) */
    float bpfi_amplitude;    /* Bearing inner race defect amplitude (envelope spectrum) */
    float bsf_amplitude;     /* Ball spin defect amplitude (envelope spectrum) */
    float overall_rms;       /* Overall vibration RMS (0–80 kHz) */
} vib_features_t;

/** Initialize SPI link to accelerometer and CMSIS-DSP FFT instance */
void vibration_hf_init(const struct device *spi_dev, uint32_t sample_rate);

/**
 * vibration_hf_acquire — acquire 'samples' int16 samples from accelerometer via SPI.
 *
 * @param buffer      output buffer, must hold 'samples' int16_t values
 * @param samples     number of samples to acquire (e.g. 65536)
 * @param sample_rate configured sample rate in Hz (for timing reference)
 */
void vibration_hf_acquire(int16_t *buffer, uint32_t samples, uint32_t sample_rate);

/**
 * vibration_hf_fft — 64k-point real FFT of int16 raw samples.
 *
 * Applies Hann window, computes magnitude spectrum using CMSIS-DSP.
 *
 * @param raw      int16 input samples (length N)
 * @param fft_mag  float output magnitude spectrum (length N, bins 0..N/2)
 * @param N        FFT size (must be 65536 for this build)
 */
void vibration_hf_fft(int16_t *raw, float *fft_mag, uint32_t N);

/**
 * envelope_demod — compute envelope spectrum in a specified HF band.
 *
 * Isolates the resonance band [f_low_kHz, f_high_kHz], computes the
 * amplitude envelope, and returns its frequency spectrum for bearing
 * fault frequency identification.
 *
 * @param fft_mag          input: FFT magnitude spectrum from vibration_hf_fft
 * @param envelope_spectrum output: envelope spectrum (same size array)
 * @param f_low_kHz        lower band edge (e.g. 20.0)
 * @param f_high_kHz       upper band edge (e.g. 40.0)
 * @param sample_rate      sample rate in Hz
 */
void envelope_demod(float *fft_mag, float *envelope_spectrum,
                    float f_low_kHz, float f_high_kHz, float sample_rate);

/**
 * vibration_extract_features — extract BPFO/BPFI/BSF amplitudes from spectra.
 *
 * Evaluates bearing fault frequencies in the envelope spectrum and returns
 * peak amplitudes for outer race, inner race, and ball spin defects.
 */
vib_features_t vibration_extract_features(float *fft_mag,
                                           float *envelope_spectrum,
                                           uint32_t N,
                                           uint32_t sample_rate_hz);
