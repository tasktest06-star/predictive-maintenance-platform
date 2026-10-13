/* app_config.h -- build-time configuration for the STWIN.box node.
 *
 * Everything that affects the numbers lives here, and everything here is
 * cross-checked against the Python reference at build time by
 * tools/check_config.py (run from the Makefile). A firmware whose n_fft or
 * demodulation band disagrees with the model it is running produces confident,
 * wrong diagnoses, and nothing in the output reveals it.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

/* ---------------------------------------------------------------- identity */
#define APP_KIT_NAME            "steval-stwinbx1"
#define APP_FIRMWARE_VERSION    "0.3.0"
/* Must equal pdmcore.FEATURE_SPEC_VERSION. Checked by tools/check_config.py. */
#define APP_FEATURE_SPEC        2

/* ------------------------------------------------------------- acquisition */
/* IIS3DWB at 26.667 kHz ODR, its only fixed rate. We decimate by 1 (none):
 * the whole point of this part is the 6.3 kHz flat bandwidth, and throwing it
 * away in the first stage would make the IIS3DWB an expensive ISM330. */
#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096          /* 6.51 Hz resolution */
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

/* The IIS3DWB needs 100 ms to reach full accuracy after power-up (DS12749).
 * Sampling before that gives a settling transient that looks exactly like an
 * impact train -- the most expensive 100 ms in the budget, and the one most
 * often skipped. */
#define APP_SENSOR_SETTLE_MS    100u

/* Sensor power policy: ALWAYS power down between acquisitions.
 *
 * The crossover is not where intuition puts it. Settling costs
 * 3.0 mA x 100 ms = 0.30 mA.s once per cycle. Keeping the IIS3DWB powered costs
 * its 1.1 mA operating current continuously, so staying alive is cheaper only
 * for intervals below 0.30 / 1.1 = 0.27 SECONDS. At any realistic acquisition
 * interval, powering down wins by two to three orders of magnitude.
 *
 * An earlier version of this file claimed a ~90 s crossover and set the
 * threshold accordingly, which would have burned 99 mA.s per cycle instead of
 * 0.30 -- a 20x increase in total node energy, and a battery life of weeks
 * instead of years. Caught by
 * hardware/*/tests/test_kit.py::test_sensor_keepalive_crossover_is_consistent_with_the_energy_model,
 * which recomputes the crossover from tools/power_budget.py rather than
 * trusting the constant.
 *
 * Kept as a parameter rather than deleted because a kit whose sensor has a much
 * longer settling time or a much lower standing current could legitimately
 * change the answer -- the ADXL355 on the ESP32-S3 branch draws 200 uA, which
 * moves the crossover to 1.5 s. Still not 90. */
#define APP_SENSOR_KEEPALIVE_THRESHOLD_S  1u

/* ---------------------------------------------------------------- analysis */
/* Demodulation band. Pushed down by the cloud as configuration after a
 * kurtogram; this is the commissioning default for a 1780 rpm motor-pump with
 * the structural resonance map in pdmcore.sim. The device never chooses its own
 * band -- that needs the full signal and a band search. */
#define APP_DEMOD_LO_HZ         2500.0f
#define APP_DEMOD_HI_HZ         6000.0f
#define APP_ENVELOPE_LP_HZ      1200.0f
#define APP_ENVELOPE_DECIMATE   8u            /* 26.667 kHz -> 3.33 kHz */
#define APP_ENVELOPE_N_FFT      1024          /* 3.26 Hz envelope resolution */
/* Samples the envelope stage consumes. Asserted against APP_ACQ_SAMPLES at
 * compile time in pdm_pipeline.c -- the mismatch above is not repeatable. */
#define APP_ENV_SAMPLES         (APP_ENVELOPE_N_FFT * APP_ENVELOPE_DECIMATE)

/* ISO 20816-3 velocity band. 10-1000 Hz is correct above 600 rpm; below that
 * the 1x line falls in the stop band and the reading is meaningless. The
 * commissioning tool sets this from the nameplate speed. */
#define APP_VELOCITY_LO_HZ      10.0f
#define APP_VELOCITY_HI_HZ      1000.0f
#define APP_ISO_CLASS           PDM_ISO_GROUP2_RIGID

/* Bearing geometry: SKF 6205-2RS, the CWRU drive-end bearing. Per-asset
 * configuration; this is a build-time default so the firmware is testable
 * before commissioning. */
#define APP_BEARING_N_BALLS     9.0f
#define APP_BEARING_BALL_D      0.3126f
#define APP_BEARING_PITCH_D     1.537f
#define APP_BEARING_ANGLE_DEG   0.0f
#define APP_BEARING_HARMONICS   5u

/* ------------------------------------------------------------------ tiering */
/* Escalation is driven by measured uncertainty, not a fixed schedule: the
 * common case stops at tier 1 and the energy saving comes from that. Thresholds
 * are in anomaly-score units relative to the calibrated operating threshold.
 */
#define APP_TIER1_ESCALATE_FRAC 0.55f   /* > 55 % of threshold -> run tier 2 */
#define APP_TIER2_ESCALATE_FRAC 1.00f   /* >= threshold        -> snapshot   */
#define APP_SNAPSHOT_MIN_INTERVAL_S  3600u   /* never more than hourly       */

/* ------------------------------------------------------------------- limits */
/* IIS3DWB full scale. +-16 g covers a severely damaged bearing on a 37 kW
 * machine with margin; +-8 g would clip, and a clipped acquisition is refused
 * by the cloud so the data is simply lost. */
#define APP_FULL_SCALE_G        16.0f
#define APP_CLIP_MARGIN         0.97f   /* flag CLIPPED above 97 % of FS */

/* Battery cut-offs. Below the critical level the node stops acquiring and only
 * beacons, so the last thing it does is tell someone rather than going silent. */
#define APP_BATT_LOW_MV         3500u
#define APP_BATT_CRITICAL_MV    3300u

/* ------------------------------------------------------------------- radio */
#define APP_DEFAULT_INTERVAL_S  3600u
#define APP_BLE_DEVICE_NAME     "PDM-STWIN"
/* Frames held in flash when the link is down. 512 x 300 B = 150 kB in the
 * 2 MB flash, i.e. three weeks at hourly intervals. Sized to survive a long
 * holiday shutdown, which is exactly when nobody notices a dead gateway. */
#define APP_QUEUE_DEPTH         512u

/* -------------------------------------------------------------- buffers/RAM */
/* Static allocation only -- see cdsp/README.md. Total below is asserted
 * against the linker's .bss in tools/check_config.py. */
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

#endif /* APP_CONFIG_H */
