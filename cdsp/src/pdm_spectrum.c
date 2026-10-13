/* Calibrated spectra, band RMS, peak interpolation, velocity integration. */

#include "pdm_dsp.h"

#include <math.h>
#include <string.h>

pdm_status pdm_spectrum_scale(const float *mag, uint16_t n_bins, uint16_t n_fft,
                              float fs_hz, const pdm_window_info *win,
                              pdm_scaling scaling, float *out)
{
    if (mag == NULL || out == NULL || win == NULL || n_bins == 0u || fs_hz <= 0.0f) {
        return PDM_ERR_ARG;
    }
    const float s1 = win->sum;
    const float s2 = win->sum_sq;
    if (s1 == 0.0f || s2 == 0.0f) {
        return PDM_ERR_NUMERIC;
    }

    /* The last bin is Nyquist when n_fft is even, which it always is here. */
    const uint16_t nyq_index = (uint16_t)(n_fft / 2u);

    for (uint16_t k = 0; k < n_bins; ++k) {
        float p = mag[k] * mag[k];
        /* Single-sided fold applied to POWER. Doubling magnitude then squaring
         * over-counts by 2x -- see the note in pdm_dsp.h. DC and Nyquist are
         * not doubled. */
        if (k != 0u && k != nyq_index) {
            p *= 2.0f;
        }
        switch (scaling) {
        case PDM_SCALE_PSD:
            out[k] = p / (fs_hz * s2);
            break;
        case PDM_SCALE_POWER:
            out[k] = p / (s1 * s1);
            break;
        case PDM_SCALE_RMS:
            out[k] = sqrtf(p) / s1;
            break;
        case PDM_SCALE_PEAK:
            out[k] = sqrtf(p) / s1 * 1.41421356237f;
            break;
        default:
            return PDM_ERR_UNSUPPORTED;
        }
    }
    return PDM_OK;
}

float pdm_band_rms(const float *spec, uint16_t n_bins, float df_hz,
                   pdm_scaling scaling, float enbw_bins,
                   float lo_hz, float hi_hz)
{
    if (spec == NULL || n_bins == 0u || df_hz <= 0.0f) {
        return 0.0f;
    }
    if (enbw_bins <= 0.0f) {
        enbw_bins = 1.0f;
    }

    /* Accumulate in double: a 4096-bin sum of float32 power values spans many
     * orders of magnitude and loses the small bins, which are exactly the ones
     * a band RMS of a quiet band consists of. */
    double acc = 0.0;
    for (uint16_t k = 0; k < n_bins; ++k) {
        const float f = (float)k * df_hz;
        if (f < lo_hz || f > hi_hz) {
            continue;
        }
        const double v = (double)spec[k];
        switch (scaling) {
        case PDM_SCALE_PSD:
            acc += v;                /* * df applied once below */
            break;
        case PDM_SCALE_POWER:
            acc += v;
            break;
        case PDM_SCALE_RMS:
            acc += v * v;
            break;
        case PDM_SCALE_PEAK:
            acc += v * v * 0.5;
            break;
        default:
            return 0.0f;
        }
    }
    if (acc <= 0.0) {
        return 0.0f;
    }
    if (scaling == PDM_SCALE_PSD) {
        /* The ENBW is already inside the PSD definition. */
        return (float)sqrt(acc * (double)df_hz);
    }
    /* Every other scaling calibrates each bin for a tone at that bin, so a
     * windowed tone spread over enbw_bins bins is counted enbw_bins times. */
    return (float)sqrt(acc / (double)enbw_bins);
}

pdm_status pdm_band_rms_vector(const float *spec, uint16_t n_bins, float df_hz,
                               pdm_scaling scaling, float enbw_bins,
                               const float *edges, uint16_t n_bands, float *out)
{
    if (spec == NULL || edges == NULL || out == NULL || n_bands == 0u) {
        return PDM_ERR_ARG;
    }
    for (uint16_t b = 0; b < n_bands; ++b) {
        out[b] = pdm_band_rms(spec, n_bins, df_hz, scaling, enbw_bins,
                              edges[2u * b], edges[2u * b + 1u]);
    }
    return PDM_OK;
}

pdm_status pdm_peak_interp(const float *spec, uint16_t n_bins, float df_hz,
                           uint16_t lo_bin, uint16_t hi_bin,
                           float *freq_hz, float *value)
{
    if (spec == NULL || n_bins < 3u || df_hz <= 0.0f) {
        return PDM_ERR_ARG;
    }
    if (hi_bin >= n_bins) {
        hi_bin = (uint16_t)(n_bins - 1u);
    }
    if (lo_bin >= hi_bin) {
        return PDM_ERR_ARG;
    }

    uint16_t best = lo_bin;
    for (uint16_t k = lo_bin; k <= hi_bin; ++k) {
        if (spec[k] > spec[best]) {
            best = k;
        }
    }
    float delta = 0.0f;
    float v = spec[best];
    if (best > 0u && best + 1u < n_bins) {
        const float y0 = spec[best - 1u];
        const float y1 = spec[best];
        const float y2 = spec[best + 1u];
        const float denom = y0 - 2.0f * y1 + y2;
        if (denom != 0.0f) {
            delta = 0.5f * (y0 - y2) / denom;
            if (delta > 0.5f) {
                delta = 0.5f;
            } else if (delta < -0.5f) {
                delta = -0.5f;
            }
            v = y1 - 0.25f * (y0 - y2) * delta;
        }
    }
    if (freq_hz != NULL) {
        *freq_hz = ((float)best + delta) * df_hz;
    }
    if (value != NULL) {
        *value = v;
    }
    return PDM_OK;
}

float pdm_velocity_rms_mm_s(const float *accel_spec_g, uint16_t n_bins,
                            float df_hz, pdm_scaling scaling, float enbw_bins,
                            float lo_hz, float hi_hz)
{
    if (accel_spec_g == NULL || n_bins == 0u || df_hz <= 0.0f || lo_hz <= 0.0f) {
        return 0.0f;
    }
    if (enbw_bins <= 0.0f) {
        enbw_bins = 1.0f;
    }
    /* Standard gravity in mm/s^2 (ISO 80000-3). */
    const float g_to_mm_s2 = 9806.65f;
    const double two_pi = 6.283185307179586;

    double acc = 0.0;
    for (uint16_t k = 1u; k < n_bins; ++k) {   /* skip DC: 1/omega is infinite */
        const float f = (float)k * df_hz;
        if (f < lo_hz || f > hi_hz) {
            continue;
        }
        const double omega = two_pi * (double)f;
        const double scale = (double)g_to_mm_s2 / omega;
        const double v = (double)accel_spec_g[k];
        switch (scaling) {
        case PDM_SCALE_PSD:
            acc += v * scale * scale;
            break;
        case PDM_SCALE_POWER:
            acc += v * scale * scale;
            break;
        case PDM_SCALE_RMS:
            acc += (v * scale) * (v * scale);
            break;
        case PDM_SCALE_PEAK:
            acc += (v * scale) * (v * scale) * 0.5;
            break;
        default:
            return 0.0f;
        }
    }
    if (acc <= 0.0) {
        return 0.0f;
    }
    if (scaling == PDM_SCALE_PSD) {
        return (float)sqrt(acc * (double)df_hz);
    }
    return (float)sqrt(acc / (double)enbw_bins);
}

uint8_t pdm_iso_zone(float velocity_rms_mm_s, pdm_iso_class cls)
{
    /* ISO 20816-3 / ISO 10816-3 zone boundaries, velocity RMS mm/s. */
    static const float limits[4][3] = {
        {1.4f, 2.8f, 4.5f},    /* group 2, rigid    */
        {2.3f, 4.5f, 7.1f},    /* group 2, flexible */
        {2.3f, 4.5f, 7.1f},    /* group 1, rigid    */
        {3.5f, 7.1f, 11.0f}    /* group 1, flexible */
    };
    const int idx = ((int)cls >= 0 && (int)cls < 4) ? (int)cls : 0;
    const float *l = limits[idx];
    if (velocity_rms_mm_s <= l[0]) {
        return 0u;
    }
    if (velocity_rms_mm_s <= l[1]) {
        return 1u;
    }
    if (velocity_rms_mm_s <= l[2]) {
        return 2u;
    }
    return 3u;
}

/* Median by partial selection sort. n is small (a floor window, <= 128 bins),
 * so an O(n^2/2) selection beats pulling in qsort -- which would also mean a
 * function-pointer call per comparison and a dependency on stdlib. */
float pdm_median_f32(float *scratch, uint16_t n)
{
    if (scratch == NULL || n == 0u) {
        return 0.0f;
    }
    const uint16_t mid = (uint16_t)(n / 2u);
    for (uint16_t i = 0; i <= mid; ++i) {
        uint16_t min_i = i;
        for (uint16_t j = (uint16_t)(i + 1u); j < n; ++j) {
            if (scratch[j] < scratch[min_i]) {
                min_i = j;
            }
        }
        if (min_i != i) {
            const float t = scratch[i];
            scratch[i] = scratch[min_i];
            scratch[min_i] = t;
        }
    }
    if ((n & 1u) != 0u) {
        return scratch[mid];
    }
    return 0.5f * (scratch[mid - 1u] + scratch[mid]);
}
