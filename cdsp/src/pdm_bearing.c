/* Bearing kinematics and harmonic-family scoring. */

#include "pdm_dsp.h"

#include <math.h>

void pdm_fault_frequencies(const pdm_bearing_geometry *g, float shaft_hz,
                           pdm_fault_freqs *out)
{
    if (g == NULL || out == NULL) {
        return;
    }
    const float ratio = (g->ball_d / g->pitch_d)
                      * cosf(g->contact_angle_deg * 0.01745329252f);
    const float ftf = shaft_hz * 0.5f * (1.0f - ratio);

    out->shaft_hz = shaft_hz;
    out->ftf_hz = ftf;
    out->bpfo_hz = g->n_balls * ftf;
    out->bpfi_hz = g->n_balls * shaft_hz * 0.5f * (1.0f + ratio);
    out->bsf_hz = shaft_hz * g->pitch_d / (2.0f * g->ball_d) * (1.0f - ratio * ratio);
}

/* Largest value within tol_hz of f_hz. */
static float peak_near(const float *spec, uint16_t n_bins, float df_hz,
                       float f_hz, float tol_hz)
{
    if (f_hz <= 0.0f) {
        return 0.0f;
    }
    int lo = (int)floorf((f_hz - tol_hz) / df_hz);
    int hi = (int)ceilf((f_hz + tol_hz) / df_hz);
    if (lo < 0) {
        lo = 0;
    }
    if (hi >= (int)n_bins) {
        hi = (int)n_bins - 1;
    }
    if (lo > hi) {
        return 0.0f;
    }
    float best = 0.0f;
    for (int k = lo; k <= hi; ++k) {
        if (spec[k] > best) {
            best = spec[k];
        }
    }
    return best;
}

/* Median amplitude in a window around f_hz, EXCLUDING +-exclude_hz.
 *
 * The exclusion matters: if the line being tested is strong it drags its own
 * floor estimate up, so the SNR falls as the fault gets worse -- the detector
 * becomes less sensitive the more there is to detect. Found by inspection in
 * the Python implementation and fixed in both.
 *
 * Capped at 96 samples so the scratch array is a known stack cost. A floor
 * estimate from 96 bins spread over the window is as good as one from 400. */
#define PDM_FLOOR_MAX 96

static float local_floor(const float *spec, uint16_t n_bins, float df_hz,
                         float f_hz, float half_width_hz, float exclude_hz)
{
    float scratch[PDM_FLOOR_MAX];
    uint16_t m = 0;

    int lo = (int)floorf((f_hz - half_width_hz) / df_hz);
    int hi = (int)ceilf((f_hz + half_width_hz) / df_hz);
    if (lo < 0) {
        lo = 0;
    }
    if (hi >= (int)n_bins) {
        hi = (int)n_bins - 1;
    }
    const int excl_lo = (int)floorf((f_hz - exclude_hz) / df_hz);
    const int excl_hi = (int)ceilf((f_hz + exclude_hz) / df_hz);

    /* Stride so that a wide window still fits the scratch array. */
    int span = hi - lo + 1;
    int stride = (span > PDM_FLOOR_MAX) ? (span / PDM_FLOOR_MAX + 1) : 1;
    for (int k = lo; k <= hi && m < PDM_FLOOR_MAX; k += stride) {
        if (k >= excl_lo && k <= excl_hi) {
            continue;
        }
        scratch[m++] = spec[k];
    }
    if (m == 0u) {
        /* Window was entirely inside the exclusion zone -- fall back to it. */
        for (int k = lo; k <= hi && m < PDM_FLOOR_MAX; k += stride) {
            scratch[m++] = spec[k];
        }
    }
    if (m == 0u) {
        return 0.0f;
    }
    return pdm_median_f32(scratch, m);
}

float pdm_harmonic_snr(const float *spec, uint16_t n_bins, float df_hz,
                       float f0_hz, uint8_t n_harmonics, float rel_tol,
                       float floor_width_hz, uint8_t *n_matched)
{
    if (n_matched != NULL) {
        *n_matched = 0u;
    }
    if (spec == NULL || n_bins < 4u || df_hz <= 0.0f || f0_hz <= 0.0f || n_harmonics == 0u) {
        return 0.0f;
    }
    const float f_max = (float)(n_bins - 1u) * df_hz;
    const float min_tol = 1.5f * df_hz;

    double snr_sum = 0.0;
    uint8_t tested = 0u;
    uint8_t matched = 0u;

    for (uint8_t h = 1u; h <= n_harmonics; ++h) {
        const float fh = f0_hz * (float)h;
        if (fh > f_max) {
            break;
        }
        float tol = rel_tol * fh;
        if (tol < min_tol) {
            tol = min_tol;
        }
        const float a = peak_near(spec, n_bins, df_hz, fh, tol);
        if (a <= 0.0f) {
            continue;
        }
        const float fl = local_floor(spec, n_bins, df_hz, fh, floor_width_hz, tol);
        if (fl <= 0.0f) {
            continue;
        }
        const float snr = a / fl;
        snr_sum += (double)snr;
        ++tested;
        if (snr >= 4.0f) {
            ++matched;
        }
    }
    if (n_matched != NULL) {
        *n_matched = matched;
    }
    if (tested == 0u) {
        return 0.0f;
    }
    /* MEAN per-harmonic SNR, not the sum: the sum scales with n_harmonics and
     * is therefore uninterpretable as a threshold. ~2-3 on noise, >= 4-6 is a
     * real detection. Same statistic as the Python reference. */
    return (float)(snr_sum / (double)tested);
}
