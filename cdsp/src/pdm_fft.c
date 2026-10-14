/* Real FFT via the N/2-point complex FFT trick.
 *
 * Structure deliberately mirrors CMSIS-DSP's arm_rfft_fast_f32 so that on a
 * Cortex-M the whole file can be replaced by a call into CMSIS-DSP without any
 * surrounding code changing, and the golden-vector tests still apply.
 *
 * Algorithm:
 *   1. Treat the n real samples as n/2 complex samples z[k] = x[2k] + i x[2k+1].
 *   2. Complex radix-2 DIT FFT of length n/2 (decimation in time, bit-reversed
 *      input, natural-order output).
 *   3. Split/unpack to recover the n/2+1 real-FFT bins.
 *
 * Cost: one n/2 complex FFT instead of one n complex FFT -- just over half the
 * work and half the RAM, which is the whole reason real FFTs are done this way
 * on an MCU.
 */

#include "pdm_dsp.h"

#include <math.h>

/* Twiddle layout, PDM_RFFT_TWIDDLE_LEN(n) = n+2 floats total:
 *   [0 .. n/2)     : n/4 interleaved (cos, sin) pairs, W_(n/2)^k, k = 0..n/4-1
 *                    for the complex FFT butterflies
 *   [n/2 .. n+2)   : n/4+1 interleaved (cos, sin) pairs, W_n^k, k = 0..n/4
 *                    for the split stage, which indexes k up to half/2 inclusive
 * Angles use the -2*pi*k/N convention (forward transform).
 */

pdm_status pdm_rfft_init(pdm_rfft_ctx *ctx, uint16_t n, float *twiddle, uint16_t *bitrev)
{
    if (ctx == NULL || twiddle == NULL || bitrev == NULL) {
        return PDM_ERR_ARG;
    }
    if (!pdm_is_power_of_two((uint32_t)n) || n < 64u || n > PDM_FFT_MAX) {
        return PDM_ERR_SIZE;
    }

    const uint16_t half = (uint16_t)(n / 2u);   /* complex FFT length */
    const double two_pi = 6.283185307179586476925286766559;

    /* Complex-FFT twiddles: W_half^k for k = 0 .. half/2-1 */
    for (uint16_t k = 0; k < half / 2u; ++k) {
        const double ang = -two_pi * (double)k / (double)half;
        twiddle[2u * k]      = (float)cos(ang);
        twiddle[2u * k + 1u] = (float)sin(ang);
    }
    /* Split-stage twiddles: W_n^k for k = 0 .. half/2 INCLUSIVE -- the split
     * loop indexes k up to half/2, so the table needs one extra pair. */
    float *split = twiddle + half;
    for (uint16_t k = 0; k <= half / 2u; ++k) {
        const double ang = -two_pi * (double)k / (double)n;
        split[2u * k]      = (float)cos(ang);
        split[2u * k + 1u] = (float)sin(ang);
    }

    /* Bit-reversal permutation for the half-length complex FFT. */
    uint16_t bits = 0;
    while ((1u << bits) < half) {
        ++bits;
    }
    for (uint16_t i = 0; i < half; ++i) {
        uint16_t r = 0;
        for (uint16_t b = 0; b < bits; ++b) {
            if (i & (uint16_t)(1u << b)) {
                r |= (uint16_t)(1u << (bits - 1u - b));
            }
        }
        bitrev[i] = r;
    }

    ctx->n = n;
    ctx->twiddle = twiddle;
    ctx->bitrev = bitrev;
    return PDM_OK;
}

pdm_status pdm_rfft_attach(pdm_rfft_ctx *ctx, uint16_t n,
                           const float *twiddle, const uint16_t *bitrev)
{
    if (ctx == NULL || twiddle == NULL || bitrev == NULL) {
        return PDM_ERR_ARG;
    }
    if (!pdm_is_power_of_two((uint32_t)n) || n < 64u || n > PDM_FFT_MAX) {
        return PDM_ERR_SIZE;
    }
    ctx->n = n;
    ctx->twiddle = twiddle;
    ctx->bitrev = bitrev;
    return PDM_OK;
}

/* In-place complex radix-2 DIT FFT of `len` complex values in `z`
 * (interleaved re, im), using a precomputed twiddle table of len/2 pairs. */
static void complex_fft_inplace(float *z, uint16_t len, const float *tw,
                                const uint16_t *bitrev)
{
    /* Bit-reversal shuffle. */
    for (uint16_t i = 0; i < len; ++i) {
        const uint16_t j = bitrev[i];
        if (j > i) {
            float tr = z[2u * i];
            float ti = z[2u * i + 1u];
            z[2u * i]       = z[2u * j];
            z[2u * i + 1u]  = z[2u * j + 1u];
            z[2u * j]       = tr;
            z[2u * j + 1u]  = ti;
        }
    }

    /* Butterflies. */
    for (uint16_t stage_len = 2u; stage_len <= len; stage_len = (uint16_t)(stage_len * 2u)) {
        const uint16_t half_stage = (uint16_t)(stage_len / 2u);
        const uint16_t step = (uint16_t)(len / stage_len);   /* twiddle stride */
        for (uint16_t base = 0; base < len; base = (uint16_t)(base + stage_len)) {
            for (uint16_t k = 0; k < half_stage; ++k) {
                const uint16_t ti = (uint16_t)(k * step);
                const float wr = tw[2u * ti];
                const float wi = tw[2u * ti + 1u];
                const uint16_t i0 = (uint16_t)(base + k);
                const uint16_t i1 = (uint16_t)(i0 + half_stage);

                const float ar = z[2u * i1];
                const float ai = z[2u * i1 + 1u];
                /* t = w * a */
                const float tr = wr * ar - wi * ai;
                const float tii = wr * ai + wi * ar;

                const float br = z[2u * i0];
                const float bi = z[2u * i0 + 1u];

                z[2u * i0]      = br + tr;
                z[2u * i0 + 1u] = bi + tii;
                z[2u * i1]      = br - tr;
                z[2u * i1 + 1u] = bi - tii;
            }
        }
    }
}

pdm_status pdm_rfft(const pdm_rfft_ctx *ctx, float *data)
{
    if (ctx == NULL || data == NULL || ctx->twiddle == NULL || ctx->bitrev == NULL) {
        return PDM_ERR_ARG;
    }
    const uint16_t n = ctx->n;
    const uint16_t half = (uint16_t)(n / 2u);

    /* Step 1+2: the real samples are already in the interleaved layout the
     * complex FFT wants -- x[2k] is the real part and x[2k+1] the imaginary
     * part of z[k]. No copy needed. */
    complex_fft_inplace(data, half, ctx->twiddle, ctx->bitrev);

    /* Step 3: split. Recover X[k] for k = 0 .. half from Z[k].
     *
     *   X[k]      = 0.5*(Z[k] + conj(Z[half-k])) - 0.5i*W_n^k*(Z[k] - conj(Z[half-k]))
     *   X[0]      = Re(Z[0]) + Im(Z[0])
     *   X[half]   = Re(Z[0]) - Im(Z[0])      (both purely real)
     *
     * Walking k from 1 to half/2 and handling k and half-k together keeps this
     * in place: each pair of output bins depends only on the same pair of input
     * bins. */
    const float *split = ctx->twiddle + half;

    const float z0r = data[0];
    const float z0i = data[1];

    for (uint16_t k = 1u; k <= half / 2u; ++k) {
        const uint16_t m = (uint16_t)(half - k);

        const float zkr = data[2u * k];
        const float zki = data[2u * k + 1u];
        const float zmr = data[2u * m];
        const float zmi = data[2u * m + 1u];

        /* even/odd split components */
        const float er = 0.5f * (zkr + zmr);
        const float ei = 0.5f * (zki - zmi);
        const float orr = 0.5f * (zki + zmi);
        const float oi = -0.5f * (zkr - zmr);

        const float wr = split[2u * k];
        const float wi = split[2u * k + 1u];

        /* t = W_n^k * o */
        const float tr = wr * orr - wi * oi;
        const float ti = wr * oi + wi * orr;

        const float xkr = er + tr;
        const float xki = ei + ti;
        /* X[half-k] = conj(E[k] - T[k]) with E,T evaluated at k */
        const float xmr = er - tr;
        const float xmi = -(ei - ti);

        data[2u * k]      = xkr;
        data[2u * k + 1u] = xki;
        if (m != k) {
            data[2u * m]      = xmr;
            data[2u * m + 1u] = xmi;
        }
    }

    /* DC and Nyquist, packed into slots 0 and 1 (CMSIS-DSP convention). This
     * packing is what makes the transform strictly in place. */
    data[0] = z0r + z0i;
    data[1] = z0r - z0i;

    return PDM_OK;
}

pdm_status pdm_rfft_magnitude(const pdm_rfft_ctx *ctx, const float *packed, float *mag)
{
    if (ctx == NULL || packed == NULL || mag == NULL) {
        return PDM_ERR_ARG;
    }
    const uint16_t n = ctx->n;
    const uint16_t half = (uint16_t)(n / 2u);

    mag[0] = fabsf(packed[0]);          /* DC      */
    mag[half] = fabsf(packed[1]);       /* Nyquist */
    for (uint16_t k = 1u; k < half; ++k) {
        const float re = packed[2u * k];
        const float im = packed[2u * k + 1u];
        mag[k] = sqrtf(re * re + im * im);
    }
    return PDM_OK;
}
