/* Window generation and correction factors. See pdm_dsp.h for conventions. */

#include "pdm_dsp.h"

#include <math.h>

/* Cosine-sum coefficients, signs folded in: w[k] = sum_i a_i cos(2 pi i k / n).
 * Same tables as pdmcore.dsp.windows._COSINE_SUM, written out rather than
 * inherited from a library so the two cannot drift. */
static const float k_cos_sum[PDM_WIN__COUNT][5] = {
    /* RECT            */ {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    /* HANN            */ {0.5f, -0.5f, 0.0f, 0.0f, 0.0f},
    /* HAMMING         */ {0.54f, -0.46f, 0.0f, 0.0f, 0.0f},
    /* BLACKMAN        */ {0.42f, -0.5f, 0.08f, 0.0f, 0.0f},
    /* BLACKMAN_HARRIS */ {0.35875f, -0.48829f, 0.14128f, -0.01168f, 0.0f},
    /* FLATTOP         */ {0.21557895f, -0.41663158f, 0.277263158f,
                           -0.083578947f, 0.006947368f}
};

int pdm_is_power_of_two(uint32_t v)
{
    return v != 0u && (v & (v - 1u)) == 0u;
}

pdm_status pdm_window_make(pdm_window_kind kind, float *w, uint16_t n,
                           pdm_window_info *info)
{
    if (w == NULL || n == 0u) {
        return PDM_ERR_ARG;
    }
    if ((int)kind < 0 || kind >= PDM_WIN__COUNT) {
        return PDM_ERR_UNSUPPORTED;
    }

    const float *a = k_cos_sum[kind];
    const float two_pi_over_n = 6.2831853071795864769f / (float)n;

    double sum = 0.0;      /* accumulate the two sums in double: they are the */
    double sum_sq = 0.0;   /* calibration constants, and n can reach 8192 so  */
                           /* float32 accumulation loses ~3 significant bits. */
    for (uint16_t k = 0; k < n; ++k) {
        const float theta = two_pi_over_n * (float)k;
        float v = a[0];
        for (int i = 1; i < 5; ++i) {
            if (a[i] != 0.0f) {
                v += a[i] * cosf((float)i * theta);
            }
        }
        w[k] = v;
        sum += (double)v;
        sum_sq += (double)v * (double)v;
    }

    if (info != NULL) {
        info->kind = kind;
        info->n = n;
        info->sum = (float)sum;
        info->sum_sq = (float)sum_sq;
        info->coherent_gain = (float)(sum / (double)n);
        info->enbw_bins = (sum != 0.0)
            ? (float)((double)n * sum_sq / (sum * sum))
            : 1.0f;
    }
    return PDM_OK;
}
