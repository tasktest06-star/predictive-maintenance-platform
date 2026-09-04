/**
 * ae_features.c — Acoustic Emission feature extraction
 *
 * Processes ADC burst buffers from the AE transducer (Physical Acoustics R15α)
 * to extract standard AE waveform descriptors per ASTM E1316 / ISO 22096.
 *
 * All features are computed on the raw 12-bit ADC buffer (1 Msps).
 * Voltage conversion: V = (adc_count / 4096) * 3.3V
 * ADC input is the output of the INA128 preamp (40 dB gain from transducer).
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "ae_features.h"

/* ADC reference voltage and resolution */
#define ADC_VREF_MV         3300.0f
#define ADC_BITS            12
#define ADC_COUNTS          4096.0f   /* 2^12 */
#define ADC_MV_PER_COUNT    (ADC_VREF_MV / ADC_COUNTS)

/* AE amplitude reference: 1 µV at transducer face (standard dBAE reference) */
#define AE_REF_UV           1.0f
#define AE_PREAMP_GAIN_DB   40.0f    /* INA128 @ 40 dB */

/* Sample period at 1 Msps */
#define AE_SAMPLE_PERIOD_US 1.0f    /* 1 µs per sample at 1 Msps */

/* -------------------------------------------------------------------------
 * ae_adc_to_mv — convert raw 12-bit ADC count to millivolts
 * ------------------------------------------------------------------------- */
static inline float ae_adc_to_mv(uint16_t adc_count)
{
    return (float)adc_count * ADC_MV_PER_COUNT;
}

/* -------------------------------------------------------------------------
 * ae_mv_to_dbae — convert millivolt amplitude to dBAE
 *
 * dBAE = 20 * log10(V_peak_uV / 1 uV) - preamp_gain_dB
 * Here V is measured after the 40 dB preamp, so we subtract 40 dB.
 * ------------------------------------------------------------------------- */
static inline float ae_mv_to_dbae(float peak_mv)
{
    if (peak_mv <= 0.0f) return 0.0f;
    float peak_uv = peak_mv * 1000.0f;
    float dBae = 20.0f * log10f(peak_uv / AE_REF_UV) - AE_PREAMP_GAIN_DB;
    return dBae;
}

/* -------------------------------------------------------------------------
 * ae_extract_features — main AE feature extraction function
 *
 * Parameters:
 *   adc_buffer   : raw 12-bit ADC samples, length len
 *   len          : number of samples in buffer (typically AE_BURST_SAMPLES=4096)
 *   threshold_mv : threshold voltage for counts and duration (millivolts)
 *
 * Returns:
 *   ae_event_t with all standard AE waveform descriptors
 * ------------------------------------------------------------------------- */
ae_event_t ae_extract_features(uint16_t *adc_buffer, uint32_t len,
                                float threshold_mv)
{
    ae_event_t event = {0};

    if (!adc_buffer || len == 0) {
        return event;
    }

    float peak_mv = 0.0f;
    uint32_t peak_idx = 0;
    uint32_t first_crossing_idx = UINT32_MAX;
    uint32_t last_crossing_idx = 0;
    uint32_t counts = 0;
    float energy_acc = 0.0f;
    bool above_threshold = false;

    for (uint32_t i = 0; i < len; i++) {
        float v_mv = ae_adc_to_mv(adc_buffer[i]);

        /* Track peak amplitude */
        if (v_mv > peak_mv) {
            peak_mv = v_mv;
            peak_idx = i;
        }

        /* Accumulate energy: integral of v^2 dt (mV^2 * µs) */
        energy_acc += v_mv * v_mv * AE_SAMPLE_PERIOD_US;

        /* Threshold crossing detection (simple hysteresis: count rising edges) */
        if (v_mv >= threshold_mv) {
            if (!above_threshold) {
                /* Rising edge crossing */
                counts++;
                above_threshold = true;

                if (first_crossing_idx == UINT32_MAX) {
                    first_crossing_idx = i;
                }
                last_crossing_idx = i;
            }
        } else {
            if (above_threshold) {
                /* Falling edge — update last crossing to falling side */
                last_crossing_idx = i;
                above_threshold = false;
            }
        }
    }

    /* Amplitude in dBAE */
    event.amplitude_dBae = ae_mv_to_dbae(peak_mv);

    /* Counts */
    event.counts = counts;

    /* Duration: from first to last threshold crossing (µs) */
    if (first_crossing_idx != UINT32_MAX && last_crossing_idx > first_crossing_idx) {
        event.duration_us = (float)(last_crossing_idx - first_crossing_idx)
                            * AE_SAMPLE_PERIOD_US;
    } else {
        event.duration_us = 0.0f;
    }

    /* Energy */
    event.energy = energy_acc;

    /* Rise time: from first threshold crossing to peak amplitude (µs) */
    if (first_crossing_idx != UINT32_MAX && peak_idx >= first_crossing_idx) {
        event.rise_time_us = (float)(peak_idx - first_crossing_idx)
                             * AE_SAMPLE_PERIOD_US;
    } else {
        event.rise_time_us = 0.0f;
    }

    /* rms_100ms is set by the caller (ae_rms_update) */
    event.rms_100ms = 0.0f;

    return event;
}

/* -------------------------------------------------------------------------
 * ae_rms_update — rolling RMS over a circular ring buffer
 *
 * Maintains a running sum-of-squares to avoid O(N) recomputation.
 * Each call inserts new_sample, evicts the oldest, returns updated RMS.
 *
 * Parameters:
 *   ring_buffer  : float array of length window_size (caller owns)
 *   new_sample   : new sample value to insert (mV)
 *   window_size  : number of samples in window (e.g. 100000 for 100 ms @ 1 Msps)
 *
 * Returns:
 *   current RMS value across the window (mV)
 * ------------------------------------------------------------------------- */

/* Static running sum for efficiency — assumes single-instance use */
static float rms_running_sum_sq = 0.0f;
static uint32_t rms_write_idx = 0;
static bool rms_initialized = false;

float ae_rms_update(float *ring_buffer, float new_sample, uint32_t window_size)
{
    if (!ring_buffer || window_size == 0) {
        return 0.0f;
    }

    if (!rms_initialized) {
        memset(ring_buffer, 0, window_size * sizeof(float));
        rms_running_sum_sq = 0.0f;
        rms_write_idx = 0;
        rms_initialized = true;
    }

    /* Evict oldest sample from running sum */
    float old_sample = ring_buffer[rms_write_idx];
    rms_running_sum_sq -= old_sample * old_sample;

    /* Insert new sample */
    ring_buffer[rms_write_idx] = new_sample;
    rms_running_sum_sq += new_sample * new_sample;

    /* Advance write index (circular) */
    rms_write_idx = (rms_write_idx + 1) % window_size;

    /* Guard against floating point drift below zero */
    if (rms_running_sum_sq < 0.0f) {
        rms_running_sum_sq = 0.0f;
    }

    return sqrtf(rms_running_sum_sq / (float)window_size);
}

/* -------------------------------------------------------------------------
 * ae_is_bearing_distress — heuristic bearing surface distress detection
 *
 * Signature: moderate-to-high amplitude (35–70 dBAE) with high count rate.
 * Physical mechanism: asperity contact and surface fatigue micro-cracks emit
 * short, repetitive AE bursts as rolling elements traverse damaged areas.
 *
 * Reference threshold (arXiv 2405.20887, Table 3):
 *   counts > 500 in one 4ms burst window ≈ 125,000 counts/sec (very active)
 *   Practical warning onset: counts > 50 per 4ms window @ 35-70 dBAE
 * ------------------------------------------------------------------------- */
bool ae_is_bearing_distress(ae_event_t *event)
{
    if (!event) return false;

    /* Amplitude in bearing distress range: 35–70 dBAE */
    bool amp_in_range = (event->amplitude_dBae >= 35.0f &&
                         event->amplitude_dBae <= 70.0f);

    /* High count rate: >50 threshold crossings in the 4 ms burst window
     * (equivalent to ~12,500 counts/sec at this burst length) */
    bool high_counts = (event->counts > 50);

    /* Moderate duration: typical bearing distress bursts are 50–2000 µs */
    bool duration_ok = (event->duration_us >= 50.0f &&
                        event->duration_us <= 2000.0f);

    return (amp_in_range && high_counts && duration_ok);
}

/* -------------------------------------------------------------------------
 * ae_is_cavitation — heuristic cavitation detection in pumps
 *
 * Signature: very high amplitude (>70 dBAE), very high count rate,
 * and near-continuous activity (long duration or persistent RMS elevation).
 *
 * Physical mechanism: bubble collapse in low-pressure zones produces intense,
 * broadband AE. Unlike bearing distress (discrete bursts), cavitation is
 * semi-continuous with many high-amplitude bursts per second.
 * ------------------------------------------------------------------------- */
bool ae_is_cavitation(ae_event_t *event)
{
    if (!event) return false;

    /* High amplitude (>70 dBAE) — cavitation produces very energetic AE */
    bool high_amp = (event->amplitude_dBae > 70.0f);

    /* Very high count rate: >200 crossings in 4ms burst window */
    bool very_high_counts = (event->counts > 200);

    /* Elevated background RMS: cavitation raises the continuous RMS level */
    bool elevated_rms = (event->rms_100ms > 50.0f);  /* >50 mV RMS background */

    /* High energy: cavitation bursts are energy-rich */
    bool high_energy = (event->energy > 1e6f);  /* mV^2 * µs threshold */

    /* Cavitation requires all intensity indicators */
    return (high_amp && very_high_counts && (elevated_rms || high_energy));
}
