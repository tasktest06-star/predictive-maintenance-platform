/* pdm_order.c -- angular resampling. See pdm_order.h for the rationale. */

#include "pdm_order.h"

/* Catmull-Rom at fractional position t in [0,1) between p1 and p2.
 *
 * Chosen over linear interpolation because linear is badly lossy at the orders
 * that matter: at a quarter of the output Nyquist it attenuates ~2.4 dB against
 * Catmull-Rom's ~0.3 dB, and an amplitude error that grows with order is
 * indistinguishable from the smearing this function exists to remove.
 *
 * Chosen over a windowed-sinc resampler because sinc needs 16-32 taps for a
 * worthwhile improvement, which is 8x the arithmetic for a correction below the
 * sensor's own 75 ug/sqrt(Hz) noise floor at these orders.
 */
static float catmull(float p0, float p1, float p2, float p3, float t)
{
    const float t2 = t * t;
    const float t3 = t2 * t;
    return 0.5f * ((2.0f * p1)
                   + (-p0 + p2) * t
                   + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2
                   + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

static float sample_at(const float *x, uint32_t n, float idx)
{
    /* Clamp rather than wrap. Wrapping would splice the end of the record onto
     * its beginning, which introduces a step discontinuity -- broadband energy
     * that looks exactly like an impact, at the two places an analyst is least
     * likely to look. */
    int32_t i1 = (int32_t)idx;
    if (i1 < 0) {
        i1 = 0;
    }
    if ((uint32_t)i1 >= n) {
        i1 = (int32_t)n - 1;
    }
    const float t = idx - (float)i1;
    const int32_t i0 = (i1 > 0) ? (i1 - 1) : 0;
    const int32_t i2 = ((uint32_t)(i1 + 1) < n) ? (i1 + 1) : (int32_t)n - 1;
    const int32_t i3 = ((uint32_t)(i1 + 2) < n) ? (i1 + 2) : (int32_t)n - 1;
    return catmull(x[i0], x[i1], x[i2], x[i3], t);
}

float pdm_order_mean_hz(const float *rev_s, uint16_t n_marks)
{
    if (rev_s == NULL || n_marks < 2u) {
        return -1.0f;
    }
    const float span = rev_s[n_marks - 1u] - rev_s[0];
    if (!(span > 0.0f)) {
        return -1.0f;
    }
    return (float)(n_marks - 1u) / span;
}

float pdm_order_speed_drift_pct(const float *rev_s, uint16_t n_marks)
{
    if (rev_s == NULL || n_marks < 3u) {
        return -1.0f;
    }
    float lo = 1e30f, hi = 0.0f, sum = 0.0f;
    for (uint16_t k = 1u; k < n_marks; ++k) {
        const float dt = rev_s[k] - rev_s[k - 1u];
        if (!(dt > 0.0f)) {
            return -1.0f;
        }
        if (dt < lo) {
            lo = dt;
        }
        if (dt > hi) {
            hi = dt;
        }
        sum += dt;
    }
    const float mean = sum / (float)(n_marks - 1u);
    /* Expressed in speed, not period: an operator reads "the shaft moved 2 %",
     * and period variation and speed variation differ at the second order. */
    const float f_hi = 1.0f / lo;
    const float f_lo = 1.0f / hi;
    const float f_mean = 1.0f / mean;
    return (f_hi - f_lo) / f_mean * 100.0f;
}

pdm_status pdm_order_resample(const float *x, uint32_t n, float fs_hz,
                              const float *rev_s, uint16_t n_marks,
                              uint16_t samples_per_rev,
                              float *out, uint32_t out_cap, uint32_t *out_n)
{
    if (x == NULL || rev_s == NULL || out == NULL || out_n == NULL) {
        return PDM_ERR_ARG;
    }
    if (n < 4u || n_marks < 2u || samples_per_rev < 8u || (samples_per_rev & 1u) != 0u) {
        return PDM_ERR_ARG;
    }
    if (!(fs_hz > 0.0f)) {
        return PDM_ERR_ARG;
    }
    *out_n = 0u;

    /* A non-ascending mark list means a missed or doubled tacho pulse. One
     * dropped revolution shifts every subsequent angle by a full turn, and the
     * resulting spectrum is not noisy, it is confidently wrong -- lines appear
     * at the right orders with the wrong amplitudes. Refuse. */
    for (uint16_t k = 1u; k < n_marks; ++k) {
        if (!(rev_s[k] > rev_s[k - 1u])) {
            return PDM_ERR_ARG;
        }
    }

    const float t_end = (float)(n - 1u) / fs_hz;
    uint32_t w = 0u;
    for (uint16_t k = 0u; k + 1u < n_marks; ++k) {
        const float t0 = rev_s[k];
        const float dt = rev_s[k + 1u] - t0;
        for (uint16_t j = 0u; j < samples_per_rev; ++j) {
            if (w >= out_cap) {
                /* Truncation is reported, not silent: an order spectrum from a
                 * short record has coarser order resolution than the caller
                 * configured, and the caller must be able to notice. */
                *out_n = w;
                return PDM_ERR_SIZE;
            }
            /* Angle advances linearly between marks. A quadratic fit through
             * three marks is marginally better under constant acceleration but
             * is unstable when a mark is jittery, and tacho jitter is the
             * common case while constant acceleration is not. */
            const float t = t0 + dt * ((float)j / (float)samples_per_rev);
            if (t < 0.0f || t > t_end) {
                /* The marks extend beyond the sample record: the tacho and the
                 * accelerometer were not captured over the same window. Stop
                 * rather than extrapolate. */
                *out_n = w;
                return (w >= samples_per_rev) ? PDM_OK : PDM_ERR_ARG;
            }
            out[w++] = sample_at(x, n, t * fs_hz);
        }
    }
    *out_n = w;
    return PDM_OK;
}
