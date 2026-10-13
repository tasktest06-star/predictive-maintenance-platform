/* app_config.h -- ESP32-S3 retrofit node.
 *
 * TWO SENSOR PROFILES, and the choice changes what this node can diagnose.
 * Select with -DPDM_SENSOR_IIS3DWB or -DPDM_SENSOR_ADXL355 (default).
 *
 * This is not a convenience switch. The ADXL355 and the IIS3DWB are not
 * substitutes:
 *
 *   ADXL355   1 kHz usable bandwidth, 25 ug/sqrt(Hz), 20-bit, 200 uA
 *   IIS3DWB   6.3 kHz flat, 75 ug/sqrt(Hz), 16-bit, 1.1 mA
 *
 * Bearing impacts excite structural resonances at 2-20 kHz. The ADXL355 cannot
 * see them -- not because it is a bad part, but because its anti-alias filter
 * is at 1 kHz by design. On the ADXL355 profile this node is a very good
 * LOW-FREQUENCY instrument: unbalance, misalignment, looseness, ISO 20816
 * velocity, with a noise floor three times better than the IIS3DWB's and four
 * extra bits. It is not a bearing-diagnosis node, and the envelope stage is
 * configured to say so rather than to produce confident numbers from a band
 * that contains no resonance.
 *
 * hardware/esp32s3/README.md states which profile to pick for which asset.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

#if !defined(PDM_SENSOR_IIS3DWB) && !defined(PDM_SENSOR_ADXL355)
#define PDM_SENSOR_ADXL355 1
#endif

#define APP_KIT_NAME            "esp32s3-devkitc"
#define APP_FIRMWARE_VERSION    "0.3.0"
#define APP_FEATURE_SPEC        2

#ifdef PDM_SENSOR_ADXL355
/* ---- ADXL355 profile: low-frequency, low-noise ---------------------- */
/* ODR 4000 Hz with the internal 1 kHz low-pass. Sampling at 4 kHz rather than
 * at 1 kHz gives 2x oversampling of the filtered band, which costs nothing (the
 * part runs at one current regardless) and keeps the anti-alias margin honest. */
#define APP_FS_HZ               4000.0f
#define APP_N_FFT               4096          /* 0.98 Hz resolution */
#define APP_ACQ_SAMPLES         4096u         /* 1.024 s */
#define APP_FULL_SCALE_G        8.0f
#define APP_SENSOR_SETTLE_MS    10u           /* ADXL355 Rev C: ~10 ms */
/* Demodulation inside the part's usable band. The honest position: there is no
 * structural resonance down here on most machines, so bearing evidence from
 * this profile is weak and the README says to treat a negative result as "not
 * measured" rather than "no fault". */
#define APP_DEMOD_LO_HZ         500.0f
#define APP_DEMOD_HI_HZ         1000.0f
#define APP_ENVELOPE_LP_HZ      700.0f
#define APP_ENVELOPE_DECIMATE   2u            /* 4000 -> 2000 Hz */
#define APP_ENVELOPE_N_FFT      1024          /* 1.95 Hz envelope resolution */
/* ADXL355 standing current is 200 uA, so the keepalive crossover is
 * 0.12 mA.s / 0.2 mA = 0.6 s. Still far below any realistic interval. */
#define APP_SENSOR_KEEPALIVE_THRESHOLD_S  1u

#else
/* ---- IIS3DWB profile: wideband, for bearing work -------------------- */
#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096
#define APP_ACQ_SAMPLES         8192u         /* 307 ms */
#define APP_FULL_SCALE_G        16.0f
#define APP_SENSOR_SETTLE_MS    100u
#define APP_DEMOD_LO_HZ         2500.0f
#define APP_DEMOD_HI_HZ         6000.0f
#define APP_ENVELOPE_LP_HZ      1200.0f
#define APP_ENVELOPE_DECIMATE   8u
#define APP_ENVELOPE_N_FFT      1024
#define APP_SENSOR_KEEPALIVE_THRESHOLD_S  1u
#endif

#define APP_WINDOW              PDM_WIN_HANN
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
#define APP_CLIP_MARGIN         0.97f

/* Li-ion, not LiFePO4 or LiSOCl2: Wi-Fi's 350 mA peaks need a chemistry with a
 * low internal resistance, which rules out the long-shelf-life cells the BLE
 * kits use. See the README -- this is the kit where the radio choice dictates
 * the battery choice, and then the battery choice dictates the duty cycle. */
#define APP_BATT_LOW_MV         3400u
#define APP_BATT_CRITICAL_MV    3200u

/* 6 h, not 1 h. Wi-Fi association costs 240 mA.s against 0.17 mA.s for sense
 * and compute combined -- three orders of magnitude. Sampling hourly on this
 * kit is a deliberate decision to spend the battery, not a default. */
#define APP_DEFAULT_INTERVAL_S  21600u
/* 8 MB flash: a 4096-frame queue is 1.2 MB and covers 11 months at the default
 * interval. Sized generously because flash is the one resource this kit has in
 * abundance. */
#define APP_QUEUE_DEPTH         4096u
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

#endif /* APP_CONFIG_H */
