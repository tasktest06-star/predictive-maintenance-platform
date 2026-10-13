/* Biquad filtering, envelope demodulation, decimation.
 *
 * Butterworth sections are designed on the device with the bilinear transform
 * rather than shipped as coefficient tables. The reason is operational: the
 * demodulation band is chosen per asset by the cloud's kurtogram and pushed
 * down as two numbers in a config message. Shipping tables would mean a
 * firmware update to retune a bearing band, which is exactly the kind of
 * friction that makes a fleet deployment stop being retuned at all.
 *
 * Design cost: ~40 transcendental calls, ~60 us on a 160 MHz M33. Paid once per
 * configuration change, not per acquisition.
 */

#include "pdm_dsp.h"

#include <math.h>
#include <string.h>

#define PDM_PI 3.14159265358979323846f

void pdm_biquad_reset(pdm_biquad_cascade *c)
{
    if (c == NULL) {
        return;
    }
    for (uint8_t i = 0; i < PDM_BIQUAD_MAX_SECTIONS; ++i) {
        c->s[i].x1 = 0.0f;
        c->s[i].x2 = 0.0f;
        c->s[i].y1 = 0.0f;
        c->s[i].y2 = 0.0f;
    }
}

/* One Butterworth band-pass section at pole angle theta.
 *
 * Analog prototype pole pair for section k of an order-2m Butterworth, then the
 * bilinear transform with frequency pre-warping. Written out rather than taken
 * from a cookbook because the cookbook band-pass is a *resonator* (constant
 * skirt, Q-parameterised), not a Butterworth band-pass, and the two differ
 * materially in a 2 kHz-wide band at 25.6 kHz: the resonator's skirts let
 * shaft-order energy through into the envelope, which is precisely what the
 * band-pass is there to remove. */
static void bp_section(pdm_biquad *b, float fs, float lo, float hi, float theta)
{
    const float w1 = 2.0f * PDM_PI * lo / fs;
    const float w2 = 2.0f * PDM_PI * hi / fs;
    /* Pre-warp to digital frequencies. */
    const float t1 = tanf(w1 * 0.5f);
    const float t2 = tanf(w2 * 0.5f);
    const float bw = t2 - t1;
    const float w0sq = t1 * t2;

    /* Analog pole: s = -sin(theta) + j cos(theta) scaled by the bandwidth. */
    const float alpha = bw * sinf(theta);

    /* Band-pass biquad from the normalised low-pass prototype. */
    const float denom = 1.0f + alpha + w0sq;
    b->b0 = bw / denom;
    b->b1 = 0.0f;
    b->b2 = -bw / denom;
    b->a1 = 2.0f * (w0sq - 1.0f) / denom;
    b->a2 = (1.0f - alpha + w0sq) / denom;

    /* Normalise to unity gain at the geometric centre frequency.
     *
     * Without this the cascade has real gain -- a 2-section design measured
     * +8.7 dB -- which silently scales every amplitude derived from the
     * envelope. The envelope statistics used for diagnosis are ratios, so the
     * decision would still have been right, but any absolute number taken from
     * the envelope (a trended envelope RMS, an alarm threshold in g) would have
     * been wrong by a factor that changes with the section count. Evaluating
     * |H| once at design time is far cheaper than discovering that in the
     * field.
     */
    const float w0 = 2.0f * PDM_PI * sqrtf(lo * hi) / fs;
    const float cw = cosf(w0), sw = sinf(w0);
    const float c2w = cosf(2.0f * w0), s2w = sinf(2.0f * w0);
    /* H(e^{jw}) = (b0 + b1 e^-jw + b2 e^-2jw) / (1 + a1 e^-jw + a2 e^-2jw) */
    const float nr = b->b0 + b->b1 * cw + b->b2 * c2w;
    const float ni = -(b->b1 * sw + b->b2 * s2w);
    const float dr = 1.0f + b->a1 * cw + b->a2 * c2w;
    const float di = -(b->a1 * sw + b->a2 * s2w);
    const float num_mag = sqrtf(nr * nr + ni * ni);
    const float den_mag = sqrtf(dr * dr + di * di);
    if (num_mag > 1e-20f && den_mag > 1e-20f) {
        const float g = num_mag / den_mag;
        b->b0 /= g;
        b->b1 /= g;
        b->b2 /= g;
    }
    b->x1 = b->x2 = b->y1 = b->y2 = 0.0f;
}

static void lp_section(pdm_biquad *b, float fs, float fc, float theta)
{
    const float w = 2.0f * PDM_PI * fc / fs;
    const float k = tanf(w * 0.5f);
    const float kk = k * k;
    const float alpha = 2.0f * k * sinf(theta);
    const float denom = 1.0f + alpha + kk;
    b->b0 = kk / denom;
    b->b1 = 2.0f * kk / denom;
    b->b2 = kk / denom;
    b->a1 = 2.0f * (kk - 1.0f) / denom;
    b->a2 = (1.0f - alpha + kk) / denom;
    b->x1 = b->x2 = b->y1 = b->y2 = 0.0f;
}

pdm_status pdm_biquad_bandpass(pdm_biquad_cascade *c, float fs_hz,
                               float lo_hz, float hi_hz, uint8_t n_sections)
{
    if (c == NULL || n_sections == 0u || n_sections > PDM_BIQUAD_MAX_SECTIONS) {
        return PDM_ERR_ARG;
    }
    if (!(lo_hz > 0.0f) || !(hi_hz > lo_hz) || !(hi_hz < 0.5f * fs_hz)) {
        return PDM_ERR_ARG;
    }
    c->n_sections = n_sections;
    for (uint8_t k = 0; k < n_sections; ++k) {
        /* Butterworth pole angles, evenly spaced on the unit semicircle. */
        const float theta = PDM_PI * (2.0f * (float)k + 1.0f) / (4.0f * (float)n_sections);
        bp_section(&c->s[k], fs_hz, lo_hz, hi_hz, theta);
    }
    return PDM_OK;
}

pdm_status pdm_biquad_lowpass(pdm_biquad_cascade *c, float fs_hz, float fc_hz,
                              uint8_t n_sections)
{
    if (c == NULL || n_sections == 0u || n_sections > PDM_BIQUAD_MAX_SECTIONS) {
        return PDM_ERR_ARG;
    }
    if (!(fc_hz > 0.0f) || !(fc_hz < 0.5f * fs_hz)) {
        return PDM_ERR_ARG;
    }
    c->n_sections = n_sections;
    for (uint8_t k = 0; k < n_sections; ++k) {
        const float theta = PDM_PI * (2.0f * (float)k + 1.0f) / (4.0f * (float)n_sections);
        lp_section(&c->s[k], fs_hz, fc_hz, theta);
    }
    return PDM_OK;
}

void pdm_biquad_run(pdm_biquad_cascade *c, const float *in, float *out, size_t n)
{
    if (c == NULL || in == NULL || out == NULL) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        float v = in[i];
        for (uint8_t s = 0; s < c->n_sections; ++s) {
            pdm_biquad *b = &c->s[s];
            /* Direct Form I. */
            const float y = b->b0 * v + b->b1 * b->x1 + b->b2 * b->x2
                          - b->a1 * b->y1 - b->a2 * b->y2;
            b->x2 = b->x1;
            b->x1 = v;
            b->y2 = b->y1;
            b->y1 = y;
            v = y;
        }
        out[i] = v;
    }
}

pdm_status pdm_envelope_rectify(const float *x, size_t n, float fs_hz,
                                float lp_hz, float *out, int squared)
{
    if (x == NULL || out == NULL || n == 0u) {
        return PDM_ERR_ARG;
    }
    pdm_biquad_cascade lp;
    const pdm_status st = pdm_biquad_lowpass(&lp, fs_hz, lp_hz, 2u);
    if (st != PDM_OK) {
        return st;
    }
    /* Rectify in place into out, then smooth. */
    for (size_t i = 0; i < n; ++i) {
        const float a = fabsf(x[i]);
        out[i] = squared ? (a * a) : a;
    }
    pdm_biquad_run(&lp, out, out, n);

    /* Remove the DC term: full-wave rectification puts the mean of |x| at DC,
     * which is 30-40 dB above the modulation lines and would dominate the
     * envelope spectrum's first bins. */
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean += (double)out[i];
    }
    mean /= (double)n;
    for (size_t i = 0; i < n; ++i) {
        out[i] -= (float)mean;
    }
    return PDM_OK;
}

pdm_status pdm_decimate(const float *x, size_t n, uint8_t factor,
                        float fs_hz, float *out, size_t *n_out)
{
    if (x == NULL || out == NULL || n_out == NULL || factor == 0u) {
        return PDM_ERR_ARG;
    }
    if (factor == 1u) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = x[i];
        }
        *n_out = n;
        return PDM_OK;
    }
    /* Anti-alias at 0.8 * the new Nyquist. 0.8 rather than 1.0 leaves room for
     * the filter transition: a 4th-order Butterworth is only ~12 dB down one
     * octave past its corner, so cutting at exactly the new Nyquist folds real
     * energy back into the band. */
    const float new_nyq = fs_hz / (2.0f * (float)factor);
    pdm_biquad_cascade lp;
    const pdm_status st = pdm_biquad_lowpass(&lp, fs_hz, 0.8f * new_nyq, 2u);
    if (st != PDM_OK) {
        return st;
    }
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        for (uint8_t s = 0; s < lp.n_sections; ++s) {
            pdm_biquad *b = &lp.s[s];
            const float y = b->b0 * v + b->b1 * b->x1 + b->b2 * b->x2
                          - b->a1 * b->y1 - b->a2 * b->y2;
            b->x2 = b->x1;
            b->x1 = v;
            b->y2 = b->y1;
            b->y1 = y;
            v = y;
        }
        if ((i % (size_t)factor) == 0u) {
            out[j++] = v;
        }
    }
    *n_out = j;
    return PDM_OK;
}
