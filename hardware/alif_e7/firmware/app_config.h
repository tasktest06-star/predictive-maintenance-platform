/* app_config.h -- Alif Ensemble E7 node.
 *
 * E7 = 2x Cortex-M55 (HE @ 160 MHz, HP @ 400 MHz) + 2x Ethos-U55 NPU
 *      (128 and 256 MAC/cycle) + 2x Cortex-A32. 5.5 MB on-chip SRAM.
 *
 * THE HONEST POSITION ON THE NPU, stated here because it determines the whole
 * configuration:
 *
 * The baseline pipeline does not need it. Compute is 0.02 mA.s of a 4.7 mA.s
 * cycle -- 0.5 % -- and the detector inference is 1 800 MACs, about 23 us in
 * plain C. An Ethos-U55 runs it in 2 us. Saving 21 microseconds once an hour is
 * not a reason to choose a part.
 *
 * What the NPU buys is a DIFFERENT MODEL, not a faster one. At 128 MAC/cycle and
 * 160 MHz the small NPU sustains ~20 GMAC/s, which puts a 1D CNN over the raw
 * 8192-sample waveform -- 2-5 MMAC per inference -- in the 0.25 ms range. That
 * is the architecture the literature actually uses (arXiv 2602.09699, 2608.09174
 * in research/code_backed_papers_2025_2026.md), and it is out of reach on every
 * other kit here.
 *
 * So this kit's configuration adds a SECOND inference path: the shared 63-float
 * feature detector (which every kit runs, for comparability) plus an optional
 * waveform model on the NPU. The feature path is what the platform depends on;
 * the waveform path is the experiment this kit exists to run.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

#define APP_KIT_NAME            "alif-ensemble-e7"
#define APP_FIRMWARE_VERSION    "0.3.0"
#define APP_FEATURE_SPEC        2

/* Identical to the STWIN.box and nRF9160 so the three kits' frames are directly
 * comparable. A kit that changes the configuration to suit its silicon makes
 * cross-kit comparison impossible, which is the one thing a multi-kit
 * evaluation must preserve. */
#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096
#define APP_ACQ_SAMPLES         8192u
#define APP_WINDOW              PDM_WIN_HANN
#define APP_WINDOW_ID           1
#define APP_SENSOR_SETTLE_MS    100u
#define APP_SENSOR_KEEPALIVE_THRESHOLD_S  1u

#define APP_DEMOD_LO_HZ         2500.0f
#define APP_DEMOD_HI_HZ         6000.0f
#define APP_ENVELOPE_LP_HZ      1200.0f
#define APP_ENVELOPE_DECIMATE   8u
#define APP_ENVELOPE_N_FFT      1024
#define APP_ENV_SAMPLES         (APP_ENVELOPE_N_FFT * APP_ENVELOPE_DECIMATE)

#define APP_VELOCITY_LO_HZ      10.0f
#define APP_VELOCITY_HI_HZ      1000.0f
#define APP_ISO_CLASS           PDM_ISO_GROUP2_RIGID

#define APP_BEARING_N_BALLS     9.0f
#define APP_BEARING_BALL_D      0.3126f
#define APP_BEARING_PITCH_D     1.537f
#define APP_BEARING_ANGLE_DEG   0.0f
#define APP_BEARING_HARMONICS   5u

#define APP_TIER1_ESCALATE_FRAC 0.55f
#define APP_TIER2_ESCALATE_FRAC 1.00f
#define APP_SNAPSHOT_MIN_INTERVAL_S  3600u
#define APP_FULL_SCALE_G        16.0f
#define APP_CLIP_MARGIN         0.97f
#define APP_BATT_LOW_MV         3500u
#define APP_BATT_CRITICAL_MV    3300u
#define APP_DEFAULT_INTERVAL_S  3600u
#define APP_QUEUE_DEPTH         512u
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

/* ------------------------------------------------------- the NPU path ----- */
/* Enable the waveform model. Off by default: it is an experiment, and a kit
 * whose experimental path is on by default produces data nobody can compare
 * against the other seven. */
#ifndef APP_WAVEFORM_MODEL
#define APP_WAVEFORM_MODEL 0
#endif

#if APP_WAVEFORM_MODEL
/* Input to the waveform model: the acquisition decimated to 6667 Hz and
 * truncated to 2048 samples (0.307 s). Why not the raw 8192 at 26.667 kHz:
 * the published 1D CNN architectures operate on 1-2 k samples, the NPU's SRAM
 * working set is the binding constraint, and a 4x decimation keeps the 2-3 kHz
 * resonance band that carries the bearing information while cutting the input
 * by 4. The band above 3.3 kHz is lost, which is a real cost and the reason the
 * FEATURE path remains the primary one. */
#define APP_WAVEFORM_DECIMATE   4u
#define APP_WAVEFORM_SAMPLES    2048u
/* INT8 input, as the Vela-compiled model expects. The quantisation scale is
 * baked into the model and emitted alongside it. */
#define APP_WAVEFORM_SCALE      0.00390625f   /* 1/256 g per count, +-0.5 g */
#endif

/* Tables in flash. Not forced by RAM here -- the E7 has 5.5 MB -- but it
 * removes ~20 ms of boot transcendentals from the HE core, and this kit's whole
 * value proposition is what it does with spare cycles. */
#define PDM_TABLES_IN_FLASH     1

#endif /* APP_CONFIG_H */
