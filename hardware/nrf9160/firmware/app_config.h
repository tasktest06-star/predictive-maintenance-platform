/* app_config.h -- Nordic nRF9160 cellular node.
 *
 * Two constraints shape this kit, and neither is the CPU:
 *
 * 1. **RAM.** 256 kB total, of which Zephyr takes ~25 kB and the Nordic modem
 *    library ~40 kB, leaving ~190 kB. The naive pipeline layout -- the one the
 *    STWIN.box uses with 786 kB to spare -- needs ~168 kB and would leave 20 kB
 *    for everything else. Two changes fix it and both are improvements every
 *    kit should adopt:
 *      - window and twiddle tables as `const` in flash, not RAM (-36 kB);
 *      - convert the sensor FIFO to float per chunk instead of staging the raw
 *        int16 block first (-48 kB).
 *    Result: ~101 kB, comfortable. See hardware/nrf9160/README.md.
 *
 * 2. **The radio costs 95 mA.s per connection** -- 260x the sensing and compute
 *    combined. Everything about the interval, the queue depth and the batching
 *    follows from that.
 *
 * The configuration itself is deliberately IDENTICAL to the STWIN.box's, so the
 * two kits' frames are directly comparable: same rate, same FFT, same bands,
 * same bearing geometry. A cellular node and a BLE node on the same machine
 * should produce the same numbers, and making the configuration identical is the
 * cheapest way to be sure they do.
 */

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "pdm_dsp.h"

#define APP_KIT_NAME            "nrf9160-dk"
#define APP_FIRMWARE_VERSION    "0.3.0"
#define APP_FEATURE_SPEC        2

#define APP_FS_HZ               26667.0f
#define APP_N_FFT               4096
#define APP_ACQ_SAMPLES         8192u         /* 307 ms */
#define APP_WINDOW              PDM_WIN_HANN
/* Integer twin of APP_WINDOW, required by the PDM_TABLES_IN_FLASH guard: the
 * preprocessor cannot compare C enum names. rect 0, hann 1, hamming 2,
 * blackman 3, blackman_harris 4, flattop 5. */
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
#define APP_SNAPSHOT_MIN_INTERVAL_S  86400u   /* once a day at most: a 51 kB
                                               * snapshot is 540 s of LTE-M
                                               * airtime and 2 600 mA.s --
                                               * 27 acquisitions' worth of
                                               * energy for one waveform. */
#define APP_FULL_SCALE_G        16.0f
#define APP_CLIP_MARGIN         0.97f

/* LiSOCl2 D-cell. Its 0.08 %/month self-discharge is 4 uA against the circuit's
 * 40 uA, so unlike the LiPo and Li-ion kits the cell is NOT the binding
 * constraint here. LTE-M's 200-250 mA pulses are within a D-cell's capability
 * where they are not within an AA's -- the cell size is driven by pulse current,
 * not by capacity. A 3.0 V nominal chemistry also sits comfortably above the
 * nRF9160's 3.0 V minimum for the whole discharge curve. */
/* The nRF9160's absolute minimum VDD is 3.0 V (PS v2.1, Table 'Recommended
 * operating conditions'). The first version of this file set the critical
 * threshold to 2900 mV, reasoning from the LiSOCl2 cell's discharge curve and
 * forgetting the part's own limit -- the node would have kept acquiring and
 * transmitting below the voltage at which the modem's behaviour is undefined.
 * Caught by hardware/<kit>/tests/test_kit.py::test_battery_thresholds_are_ordered,
 * which asserts a 3000 mV floor for exactly this reason.
 *
 * 3200/3050 leaves 50 mV of margin above the part minimum and still uses most
 * of the cell: a LiSOCl2 D-cell holds ~3.3 V under a 250 mA pulse for the great
 * majority of its discharge and only falls through 3.2 V near the end. */
#define APP_BATT_LOW_MV         3200u
#define APP_BATT_CRITICAL_MV    3050u

#define APP_DEFAULT_INTERVAL_S  3600u
/* 1024 frames at hourly = 42 days. Deeper than the BLE kits because a cellular
 * node is deployed where nobody visits: a remote pump station whose coverage
 * drops for six weeks must not lose the six weeks. */
#define APP_QUEUE_DEPTH         1024u
#define APP_SCRATCH_FLOATS      (2u * PDM_MODEL_MAX_WIDTH)

/* Put the window and twiddle tables in flash. 36 kB of 1 MB flash against 36 kB
 * of 256 kB RAM: on this part the trade is obvious, and the only cost is that
 * flash reads are ~2 cycles slower than RAM on the M33's bus -- about 3 % on the
 * FFT, which at 14 ms is 0.4 ms. */
#define PDM_TABLES_IN_FLASH     1

#endif /* APP_CONFIG_H */
