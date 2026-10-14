/* pdm_pipeline.h -- the tiered feature pipeline.
 *
 * Vendor-independent: this and pdm_pipeline.c contain no SDK calls, so the same
 * pipeline compiles for the STM32U5, nRF9160, ESP32-S3, Alif E7 and RA8D1
 * branches -- and for the build host, where it is tested.
 *
 * Each kit supplies two headers on the include path:
 *   app_config.h      sample rate, FFT size, demodulation band, thresholds
 *   pdm_detector.h    the exported int8 model (tools/export_model.py)
 *
 * And implements nothing else. Board bring-up, the sensor driver and the radio
 * live in the kit's own firmware/ directory behind its own interfaces; the
 * pipeline only ever sees a block of float g values and a shaft frequency.
 */

#ifndef PDM_PIPELINE_H
#define PDM_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "pdm_dsp.h"
#include "pdm_features_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Frame flags. Must match pdmcore.io.payload.FrameFlags -- the cloud reads
 * these to decide whether a frame may be scored at all. */
#define PDM_FRAME_CLIPPED            (1u << 0)
#define PDM_FRAME_FIFO_OVERRUN       (1u << 1)
#define PDM_FRAME_SELFTEST_FAIL      (1u << 2)
#define PDM_FRAME_LOW_BATTERY        (1u << 3)
#define PDM_FRAME_FALLBACK_CONFIG    (1u << 4)
#define PDM_FRAME_SPEED_ESTIMATED    (1u << 5)
#define PDM_FRAME_TEMP_OUT_OF_RANGE  (1u << 6)
#define PDM_FRAME_STORED_FORWARDED   (1u << 7)
/* Channel index, bits 8-9. Zero on every single-axis kit, so a frame from one
 * is unchanged and an older decoder reads it identically. A multi-axis node
 * (Renesas RA8D1) ships one frame per axis per acquisition, distinguished only
 * by these bits -- a decoder that ignores them sees three frames with the same
 * timestamp and different features, which looks like a sensor fault. */
#define PDM_FRAME_CHANNEL_SHIFT      8u
#define PDM_FRAME_CHANNEL_MASK       (3u << 8)
#define PDM_FRAME_CHANNEL(ch)        (((uint16_t)(ch) & 3u) << PDM_FRAME_CHANNEL_SHIFT)
/* The acquisition was resampled into the angle domain before analysis, so the
 * spectral features are per-revolution orders rather than hertz. The cloud MUST
 * know: comparing an order-tracked frame against a frequency-domain baseline is
 * a guaranteed false alarm on any machine that is not at exactly nominal speed.
 */
#define PDM_FRAME_ORDER_TRACKED      (1u << 10)
/* The shaft speed moved materially during the record, measured by a real
 * tachometer. The features are still frequency-domain, so the cloud must widen
 * its harmonic-match tolerance and discount the high-order content: smearing
 * grows with order, so a gear-mesh line at 40x loses ~6 dB at 4 % drift while
 * the 1x line loses ~1 dB. Without this flag that looks like a developing
 * gear-mesh fault that mysteriously heals when production steadies. */
#define PDM_FRAME_SPEED_UNSTEADY     (1u << 11)

/* Number of log-spaced bands. Comes from the generated layout so it cannot
 * disagree with the Python extractor. */
#define PDM_N_BANDS PDM_N_BAND

typedef struct {
    float    features[PDM_N_FEATURES];
    /* Highest tier reached. 0 = time domain only (or clipped), 1 = spectral,
     * 2 = envelope + bearing, 3 = snapshot requested. Shipped in the frame so
     * the cloud knows whether a zero in the bearing block means "no fault" or
     * "not computed" -- a distinction that is invisible in the numbers and
     * changes the diagnosis completely. */
    uint8_t  tier;
    uint8_t  request_snapshot;
    uint8_t  iso_zone;
    uint16_t flags;
    float    score;
    /* The calibrated operating threshold, supplied by the caller before the
     * run (pushed down from the cloud as configuration). The pipeline needs it
     * to decide whether to escalate; it does not invent one. */
    float    threshold_hint;
    float    velocity_rms_mm_s;
    /* Number of overlapped blocks averaged into the spectrum. Shipped so the
     * cloud can tell a 1-block frame (a short acquisition) from a 3-block one:
     * the noise-floor variance differs by sqrt(3) and a threshold tuned on one
     * does not transfer to the other. */
    uint16_t n_averages;
} pdm_pipeline_result;

/* Build windows, twiddle tables, filters and the model. Call once. */
pdm_status pdm_pipeline_init(void);

/* Run the tiered pipeline over one acquisition block.
 *
 * `accel_g`      at least APP_N_FFT samples, in g.
 * `shaft_hz`     must be > 0: order and bearing features are meaningless
 *                without it, and a vector computed against a wrong speed is
 *                worse than no vector.
 * `out`          caller-provided; set `out->threshold_hint` before calling.
 */
pdm_status pdm_pipeline_run(const float *accel_g, size_t n_samples, float shaft_hz,
                            pdm_pipeline_result *out);

uint16_t     pdm_pipeline_n_features(void);
const float *pdm_pipeline_band_edges(void);

#ifdef __cplusplus
}
#endif
#endif /* PDM_PIPELINE_H */
