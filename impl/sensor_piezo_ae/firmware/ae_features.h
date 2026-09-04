/**
 * ae_features.h — Acoustic Emission feature types and function prototypes
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

/**
 * ae_event_t — standard AE waveform descriptors (ASTM E1316 / ISO 22096)
 */
typedef struct {
    float    amplitude_dBae;   /* Peak amplitude in dBAE (ref: 1 µV at transducer) */
    uint32_t counts;           /* Number of threshold crossings in this event */
    float    duration_us;      /* Time from first to last threshold crossing (µs) */
    float    energy;           /* Integral of v^2 dt over event (mV^2 * µs) */
    float    rise_time_us;     /* Time from first crossing to peak amplitude (µs) */
    float    rms_100ms;        /* Rolling 100 ms RMS of AE channel (mV) */
} ae_event_t;

/**
 * ae_extract_features — extract AE waveform descriptors from ADC burst buffer.
 *
 * @param adc_buffer   12-bit ADC samples at 1 Msps (length: len)
 * @param len          number of samples in buffer
 * @param threshold_mv threshold voltage for counts/duration in millivolts
 * @return             ae_event_t with all features populated (rms_100ms = 0, set by caller)
 */
ae_event_t ae_extract_features(uint16_t *adc_buffer, uint32_t len,
                                float threshold_mv);

/**
 * ae_rms_update — update the rolling 100 ms RMS estimate.
 *
 * Maintains a circular ring buffer internally for O(1) update per sample.
 *
 * @param ring_buffer  caller-owned float array of length window_size
 * @param new_sample   new signal value (mV)
 * @param window_size  ring buffer length (= sample_rate * 0.1 for 100 ms)
 * @return             current RMS over the window (mV)
 */
float ae_rms_update(float *ring_buffer, float new_sample, uint32_t window_size);

/**
 * ae_is_bearing_distress — heuristic: bearing surface fatigue signature.
 *
 * Rule: amplitude 35–70 dBAE AND counts > 50 per 4 ms burst AND duration 50–2000 µs.
 * Maps to ISO 13373 Stage 0–1 bearing surface distress, detectable 4–6 weeks
 * before vibration-based methods (arXiv 2405.20887).
 */
bool ae_is_bearing_distress(ae_event_t *event);

/**
 * ae_is_cavitation — heuristic: pump cavitation signature.
 *
 * Rule: amplitude > 70 dBAE AND counts > 200 per burst AND elevated background RMS.
 * Cavitation produces semi-continuous, high-energy, broadband AE bursts.
 */
bool ae_is_cavitation(ae_event_t *event);
