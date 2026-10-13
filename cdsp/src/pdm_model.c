/* int8 quantised MLP inference for the one-class (Deep SVDD) detector.
 *
 * Two properties make this cheap enough to run every acquisition cycle on a
 * Cortex-M33 with no NPU:
 *
 *   - **No bias terms.** Not a size optimisation: Deep SVDD with biases can
 *     collapse to the constant centre and reach zero training loss having
 *     learned nothing. The trained model omits them, so the firmware must too.
 *   - **Per-tensor symmetric quantisation.** w_float = w_int8 * w_scale, zero
 *     point fixed at 0. Asymmetric quantisation would need a zero-point
 *     correction term per output, which is an extra multiply-accumulate per
 *     weight and buys nothing for weights that are already roughly centred.
 *
 * The accumulator is int32 and the de-quantisation happens once per layer
 * output, so the inner loop is a pure int8 x int8 -> int32 MAC. That is the
 * operation CMSIS-NN's arm_fully_connected_s8 and the Ethos-U55 both
 * accelerate, so this reference implementation can be swapped for either
 * without the surrounding code moving.
 *
 * Measured on the reference 63-24-12 model: 1 800 MACs, ~23 us at 160 MHz with
 * this plain C loop, ~8 us with CMSIS-NN on an M33, ~2 us on an Ethos-U55.
 * Compared against a 1.1 ms FFT, model inference is not the bottleneck -- which
 * is why there is no urgency to reach for the NPU on most of the kits.
 */

#include "pdm_dsp.h"

#include <math.h>

pdm_status pdm_model_score(const pdm_model_i8 *m, const float *features,
                           float *scratch, float *score_out)
{
    if (m == NULL || features == NULL || scratch == NULL || score_out == NULL) {
        return PDM_ERR_ARG;
    }
    if (m->n_layers == 0u || m->n_layers > PDM_MODEL_MAX_LAYERS) {
        return PDM_ERR_ARG;
    }
    if (m->n_features == 0u || m->n_features > PDM_MODEL_MAX_WIDTH) {
        return PDM_ERR_SIZE;
    }

    float *cur = scratch;
    float *next = scratch + PDM_MODEL_MAX_WIDTH;

    /* 1. Robust standardisation, mirroring pdmcore.models.anomaly.StandardScaler.
     *    Heavy-tailed positive features are log-transformed first: their
     *    distribution is roughly log-normal, and a z-score on the raw value
     *    calls every busy hour an anomaly. Which features those are is decided
     *    at training time and shipped as the in_log bitmap, so the device does
     *    not have to decide. */
    for (uint16_t i = 0; i < m->n_features; ++i) {
        float v = features[i];
        if (m->in_log != NULL && m->in_log[i] != 0u) {
            v = logf(v > 1e-12f ? v : 1e-12f);
        }
        const float c = (m->in_centre != NULL) ? m->in_centre[i] : 0.0f;
        const float s = (m->in_scale != NULL && m->in_scale[i] != 0.0f) ? m->in_scale[i] : 1.0f;
        cur[i] = (v - c) / s;
    }
    uint16_t cur_n = m->n_features;

    /* 2. Layers. LeakyReLU on every layer except the last, which is linear --
     *    the output is a point in the embedding space, not a probability. */
    for (uint8_t l = 0; l < m->n_layers; ++l) {
        const pdm_layer_i8 *layer = &m->layers[l];
        if (layer->n_in != cur_n || layer->n_out > PDM_MODEL_MAX_WIDTH || layer->w == NULL) {
            return PDM_ERR_SIZE;
        }
        const int last = (l + 1u == m->n_layers);

        for (uint16_t o = 0; o < layer->n_out; ++o) {
            int32_t acc = 0;
            /* Row-major [n_in][n_out] so the inner stride is n_out. Column-
             * major would give unit stride here, but then the weight export
             * would not match the row-major convention every training framework
             * uses, and a silent transpose is a much worse bug than a cache
             * miss on a part with no cache. */
            const float inv_in_scale = (layer->in_scale > 0.0f) ? (1.0f / layer->in_scale) : 1.0f;
            for (uint16_t i = 0; i < layer->n_in; ++i) {
                /* Inputs are float at this point; quantise on the fly to int8
                 * using this layer's calibrated scale. Round half away from
                 * zero -- the exporter's bit-exact simulation does the same, and
                 * numpy's default round-half-even would disagree on exactly the
                 * values that sit on a quantisation boundary. */
                float xi = cur[i] * inv_in_scale;
                if (xi > 127.0f) {
                    xi = 127.0f;
                } else if (xi < -127.0f) {
                    xi = -127.0f;
                }
                acc += (int32_t)(int8_t)(xi >= 0.0f ? (xi + 0.5f) : (xi - 0.5f))
                     * (int32_t)layer->w[(size_t)i * layer->n_out + o];
            }
            float y = (float)acc * layer->w_scale * layer->in_scale;
            if (!last && y < 0.0f) {
                y *= m->leak;
            }
            next[o] = y;
        }
        /* Swap the ping-pong buffers. */
        float *tmp = cur;
        cur = next;
        next = tmp;
        cur_n = layer->n_out;
    }

    /* 3. Distance from the hypersphere centre. */
    double acc = 0.0;
    for (uint16_t i = 0; i < cur_n; ++i) {
        const float c = (m->centre != NULL) ? m->centre[i] : 0.0f;
        const double d = (double)cur[i] - (double)c;
        acc += d * d;
    }
    const float score = (float)sqrt(acc);
    if (!isfinite(score)) {
        return PDM_ERR_NUMERIC;
    }
    *score_out = score;
    return PDM_OK;
}
