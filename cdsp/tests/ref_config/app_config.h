/* Reference app_config.h for host testing of the shared pipeline.
 *
 * Mirrors the STWIN.box configuration, which is also what
 * cdsp/tools/gen_golden.py uses to produce the reference feature vector. A real
 * kit supplies its own copy of this file; this one exists so the pipeline can
 * be compiled and tested without any kit.
 */
#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

#define APP_KIT_NAME            "host-reference"
#define APP_FIRMWARE_VERSION    "0.3.0-host"
#define APP_FEATURE_SPEC        2

#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096
#define APP_WINDOW              PDM_WIN_HANN
/* Acquisition length is set by the ENVELOPE stage, not the spectral one.
 *
 * Envelope resolution is a function of record *duration*, not of decimation:
 * a 4096-sample block at 26.667 kHz is 154 ms, so no amount of decimation gets
 * below 6.5 Hz. BPFO on this bearing is ~106 Hz and the harmonic matcher uses a
 * +-2 % tolerance with a 1.5-bin floor, so at 6.5 Hz the floor alone is +-9 % --
 * wide enough to match a neighbouring family. 8192 samples (307 ms) decimated
 * by 8 gives 1024 envelope bins at 3.26 Hz, where the 1.5-bin floor is +-4.6 %
 * and the tolerance is set by the +-2 % slip term as intended.
 *
 * The first version of this config asked for a 2048-point envelope FFT from a
 * 4096-sample block -- 512 samples into a 2048-point transform. The pipeline
 * correctly refused and silently stayed at tier 1, so bearing features were
 * never computed and every frame looked clean. Caught by test_pipeline.c.
 *
 * The spectral stage averages 50 %-overlapped 4096-point blocks across the same
 * record (3 blocks here), matching pdmcore.dsp.spectrum's default. Averaging is
 * not optional for agreement: a single block and a 3-block average differ by
 * several percent on every broadband feature.
 *
 * Kits with more RAM can double this (16384 samples, 2048-point envelope FFT,
 * 1.63 Hz) -- see each kit README for what it uses and why. */
#define APP_ACQ_SAMPLES         8192u         /* 307 ms */
#define APP_SPEC_OVERLAP_NUM    1u            /* 50 % overlap: step = n_fft/2 */
#define APP_SPEC_OVERLAP_DEN    2u
#define APP_SENSOR_SETTLE_MS    100u
#define APP_SENSOR_KEEPALIVE_THRESHOLD_S  90u

#define APP_DEMOD_LO_HZ         2500.0f
#define APP_DEMOD_HI_HZ         6000.0f
#define APP_ENVELOPE_LP_HZ      1200.0f
#define APP_ENVELOPE_DECIMATE   8u
#define APP_ENVELOPE_N_FFT      1024          /* 3.26 Hz envelope resolution */
/* Samples the envelope stage consumes. Asserted against APP_ACQ_SAMPLES at
 * compile time in pdm_pipeline.c -- the mismatch above is not repeatable. */
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
#define APP_SNAPSHOT_MIN_INTERVAL_S 3600u

#define APP_FULL_SCALE_G        16.0f
#define APP_CLIP_MARGIN         0.97f
#define APP_BATT_LOW_MV         3500u
#define APP_BATT_CRITICAL_MV    3300u
#define APP_DEFAULT_INTERVAL_S  3600u
#define APP_QUEUE_DEPTH         512u
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

#endif /* APP_CONFIG_H */
