/* app_config.h -- Renesas EK-RA8D1 node.
 *
 * RA8D1: Cortex-M85 @ 480 MHz with Helium MVE, 1 MB SRAM, 2 MB flash,
 *        TrustZone. The fastest DSP part in the lineup by a wide margin.
 *
 * THE HONEST POSITION ON SPEED, because it decides the configuration:
 *
 * A faster core does not save energy. Energy is work, and the work is fixed:
 * the M85 at 480 MHz does the pipeline in 1.3 ms at 62 mA = 0.08 mA.s; the
 * STM32U5 at 160 MHz does it in 8.6 ms at 21 mA = 0.18 mA.s. Two-fold, on a
 * term that is 2 % of the cycle. The 307 ms acquisition window is 85 % on both.
 *
 * What the headroom buys is a BIGGER WORKLOAD. This kit therefore does three
 * things no other kit here does, and they are the reason it exists:
 *
 *   1. THREE AXES. The pipeline runs once per axis: 3 x 1.3 ms = 3.9 ms, still
 *      1 % of the cycle. Misalignment and looseness are far clearer with an
 *      axial channel; bearing faults rarely need it. Three frames per
 *      acquisition instead of one.
 *   2. A REAL TACHOMETER. The only kit with a counter input wired to a
 *      connector, so shaft speed is MEASURED, not configured or estimated.
 *      Every other kit's frames carry PDM_FRAME_SPEED_ESTIMATED.
 *   3. ORDER TRACKING. With a tacho the acquisition can be resampled into the
 *      angle domain, which is the only way to analyse a VFD-driven machine
 *      whose speed moves during the record. At 2 % speed drift a 6x line smears
 *      across 6 Hz and loses 10 dB of peak amplitude; angular resampling puts
 *      it back in one bin.
 *
 * Point 3 is the one that matters commercially: an increasing fraction of the
 * installed base is VFD-driven, and on those machines fixed-frequency analysis
 * is progressively wrong as the duty varies.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

#define APP_KIT_NAME            "renesas-ek-ra8d1"
#define APP_FIRMWARE_VERSION    "0.3.0"
#define APP_FEATURE_SPEC        2

/* Same DSP configuration as the other IIS3DWB kits, so the frames are
 * comparable. The differences on this kit are in WHAT is processed (three axes,
 * angle domain), not in HOW. */
#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096
#define APP_ACQ_SAMPLES         8192u
#define APP_WINDOW              PDM_WIN_HANN
#define APP_WINDOW_ID           1
#define APP_SPEC_OVERLAP_NUM    1u            /* 50 % overlap: step = n_fft/2 */
#define APP_SPEC_OVERLAP_DEN    2u
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
#define APP_BATT_LOW_MV         3400u
#define APP_BATT_CRITICAL_MV    3200u
#define APP_DEFAULT_INTERVAL_S  3600u
/* Three frames per acquisition, so the queue holds a third as many
 * acquisitions for the same depth. 1536 frames = 512 acquisitions = 21 days. */
#define APP_QUEUE_DEPTH         1536u
#define APP_BLE_DEVICE_NAME     "PDM-RA8"
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

/* ----------------------------------------------------- what makes this kit - */
/* Three axes. Axis 0 is radial (bearing, unbalance), 1 is tangential, 2 is
 * axial (misalignment, thrust). Which physical direction each is depends on
 * mounting and is recorded at commissioning. */
#define APP_N_AXES              3u

/* Tachometer on GPT channel 0, input capture. 1 pulse per revolution
 * (keyphasor) by default; a 60-tooth gear target would be 60. */
#define APP_TACHO_ENABLED       1
#define APP_TACHO_PULSES_PER_REV 1u
/* Reject a speed reading whose pulse-to-pulse jitter exceeds this: a flaky
 * optical pickup produces plausible but wrong speeds, and a wrong speed is
 * worse than no speed because the order and bearing features are computed
 * against it. */
#define APP_TACHO_MAX_JITTER_PCT 5.0f

/* Order tracking. Enabled only when the tacho reports speed drift above this
 * threshold: resampling a steady machine costs cycles and gains nothing, and at
 * low drift the interpolation error is comparable to the smearing it removes. */
#define APP_ORDER_TRACK_DRIFT_PCT 0.5f
#define APP_ORDER_SAMPLES_PER_REV 256u

/* 1 MB SRAM: no need for flash tables, and three axes plus an angle-domain
 * buffer would not fit on any other kit here. */
#define PDM_TABLES_IN_FLASH     0

#endif /* APP_CONFIG_H */
