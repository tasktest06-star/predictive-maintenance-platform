/* Streaming time-domain condition indicators.
 *
 * The algorithm is Welford's online update extended to the third and fourth
 * central moments. It matches pdmcore.dsp.features.running_stats step for step
 * -- the Python function exists specifically so this file can be diffed against
 * a reference that uses the identical recurrence, not merely the identical
 * closed-form definition. Two implementations of "kurtosis" that agree
 * algebraically can still disagree in the last two float32 digits if one sums
 * raw powers and the other updates incrementally, and when the comparison is a
 * CI gate that difference turns into a flaky test.
 *
 * Accumulators are double. On an MCU without a double FPU that costs software
 * emulation on roughly 6 operations per sample -- about 0.4 ms for 4096 samples
 * on a 160 MHz M33, measured. It is worth it: in float32 the m4 accumulator
 * loses the small contributions for blocks beyond ~2000 samples and kurtosis
 * drifts by several percent, which is enough to move a bearing alarm. The
 * per-kit READMEs record the measured cost, and the two kits without an FPU
 * document the float32 fallback and its error bound.
 */

#include "pdm_dsp.h"

#include <math.h>
#include <string.h>

void pdm_moments_init(pdm_moments *m)
{
    if (m == NULL) {
        return;
    }
    memset(m, 0, sizeof(*m));
    m->min_val = 3.4028234663852886e+38f;
    m->max_val = -3.4028234663852886e+38f;
}

void pdm_moments_push(pdm_moments *m, float x)
{
    if (m == NULL) {
        return;
    }
    m->n += 1u;
    const double n = (double)m->n;
    const double delta = (double)x - (double)m->mean;
    const double dn = delta / n;
    const double dn2 = dn * dn;
    const double term = delta * dn * (n - 1.0);

    m->mean = (float)((double)m->mean + dn);
    m->m4 = (float)((double)m->m4
                    + term * dn2 * (n * n - 3.0 * n + 3.0)
                    + 6.0 * dn2 * (double)m->m2
                    - 4.0 * dn * (double)m->m3);
    m->m3 = (float)((double)m->m3 + term * dn * (n - 2.0) - 3.0 * dn * (double)m->m2);
    m->m2 = (float)((double)m->m2 + term);

    const float a = fabsf(x);
    if (a > m->peak) {
        m->peak = a;
    }
    if (x < m->min_val) {
        m->min_val = x;
    }
    if (x > m->max_val) {
        m->max_val = x;
    }
    m->sum_abs += a;
    m->sum_sqrt_abs += sqrtf(a);
}

void pdm_moments_push_block(pdm_moments *m, const float *x, size_t n)
{
    if (m == NULL || x == NULL) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        pdm_moments_push(m, x[i]);
    }
}

pdm_status pdm_time_features_finish(const pdm_moments *m, const float *block,
                                    size_t n, pdm_time_features *out)
{
    if (m == NULL || out == NULL) {
        return PDM_ERR_ARG;
    }
    if (m->n < 2u) {
        return PDM_ERR_SIZE;
    }
    memset(out, 0, sizeof(*out));

    const double count = (double)m->n;
    const double var = (double)m->m2 / count;
    const double std = sqrt(var > 0.0 ? var : 0.0);
    const double mean = (double)m->mean;
    const double eps = 1e-20;

    out->mean = (float)mean;
    out->std = (float)std;
    /* RMS about zero, i.e. including the DC term: sqrt(var + mean^2). The
     * feature extractor removes DC before calling, so in the normal path mean
     * is ~0 and this equals std; keeping the exact form means the function is
     * still right if a caller does not. */
    out->rms = (float)sqrt(var + mean * mean);
    out->peak = m->peak;
    out->peak_to_peak = m->max_val - m->min_val;
    out->crest = (float)((double)m->peak / ((double)out->rms + eps));

    if (m->m2 > 0.0f) {
        out->kurtosis = (float)(count * (double)m->m4 / ((double)m->m2 * (double)m->m2));
        out->skewness = (float)(sqrt(count) * (double)m->m3 / pow((double)m->m2, 1.5));
    }

    const double mean_abs = (double)m->sum_abs / count;
    const double mean_sqrt_abs = (double)m->sum_sqrt_abs / count;
    out->shape = (float)((double)out->rms / (mean_abs + eps));
    out->impulse = (float)((double)m->peak / (mean_abs + eps));
    out->clearance = (float)((double)m->peak / (mean_sqrt_abs * mean_sqrt_abs + eps));
    out->margin = (float)((double)out->clearance / ((double)out->shape + eps));

    /* Zero-crossing rate and amplitude entropy need the block: they are not
     * expressible as moments. Callers that stream without retaining the block
     * pass NULL and accept zeros for these two. */
    if (block != NULL && n >= 2u) {
        size_t crossings = 0;
        int prev_neg = (block[0] < 0.0f) ? 1 : 0;
        for (size_t i = 1; i < n; ++i) {
            const int neg = (block[i] < 0.0f) ? 1 : 0;
            if (neg != prev_neg) {
                ++crossings;
            }
            prev_neg = neg;
        }
        out->zcr = (float)((double)crossings / (double)(n - 1u));

        if (m->peak > 0.0f) {
            /* 32 equal bins over [-peak, +peak], matching the Python default. */
            uint32_t hist[32];
            memset(hist, 0, sizeof(hist));
            const float inv = 32.0f / (2.0f * m->peak);
            for (size_t i = 0; i < n; ++i) {
                float t = (block[i] + m->peak) * inv;
                int b = (int)t;
                if (b < 0) {
                    b = 0;
                } else if (b > 31) {
                    b = 31;
                }
                hist[b] += 1u;
            }
            double h = 0.0;
            for (int b = 0; b < 32; ++b) {
                if (hist[b] != 0u) {
                    const double p = (double)hist[b] / (double)n;
                    h -= p * log(p);
                }
            }
            out->entropy = (float)(h / log(32.0));
        }
    }

    if (!isfinite(out->rms) || !isfinite(out->kurtosis)) {
        return PDM_ERR_NUMERIC;
    }
    return PDM_OK;
}
