/* board_host.c -- pdm_board.h implemented on the build host.
 *
 * Turns the real firmware into a "virtual device": the same pdm_app.c,
 * pdm_pipeline.c and pdm_payload.c that a microcontroller runs, driven from a
 * file of samples instead of an accelerometer, emitting real 300-byte wire
 * frames on stdout.
 *
 * This is how every per-kit POC runs in CI, and it is a much stronger statement
 * than a unit test: the frames it produces are decoded by
 * pdmcore.io.payload.decode_frame and scored by the actual ingest service. If
 * the firmware's feature extraction, its wire encoding, its sequence handling
 * or its store-and-forward logic is wrong, the cloud side notices.
 *
 * What it does not test: the sensor driver, DMA, clock setup, radio stack,
 * sleep modes and power. Those are the per-kit board_*.c files and they need
 * hardware. The gap is stated in docs/07_production_readiness.md rather than
 * papered over.
 *
 * Deliberate behaviours, because a stub that always succeeds tests nothing:
 *   - the transmit path fails on a configurable schedule, exercising the queue;
 *   - the queue is finite and reports overflow;
 *   - the battery decays, so the low/critical paths are reachable;
 *   - FIFO overruns can be injected.
 */

#include "pdm_board.h"
#include "board_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app_config.h"
#include "pdm_pipeline.h"

/* ------------------------------------------------------- injected faults */
typedef struct {
    FILE    *samples;          /* text file of float g values */
    uint32_t tx_fail_every;    /* 0 = never fail; fails unless call %% N == 0 */
    /* Outage then recovery: the first N transmit attempts fail, then every
     * attempt succeeds. This is the model store-and-forward exists for -- a
     * gateway down for a fortnight, then back. A periodic failure pattern never
     * lets a drain complete, because the drain's own transmit is the next
     * attempt and fails too; that made the first version of the
     * store-and-forward test unsatisfiable by construction. */
    uint32_t tx_fail_until;
    uint32_t overrun_every;    /* 0 = never overrun */
    uint16_t battery_start_mv;
    uint16_t battery_decay_mv; /* per cycle */
    float    shaft_hz;
    int      shaft_estimated;
    int      selftest_fail;
} host_cfg;

static host_cfg s_cfg = {NULL, 0u, 0u, 0u, 3700u, 0u, 29.6667f, 0, 0};
static uint32_t s_tx_calls = 0;
static uint32_t s_acq_calls = 0;
static uint16_t s_batt = 3700u;
static int      s_accel_on = 0;
static uint64_t s_sleep_total_ms = 0;

/* Store-and-forward queue. In firmware this is a flash ring buffer; here it is
 * RAM with the same depth, so the drain logic and the overflow path behave
 * identically. */
#define HOST_FRAME_MAX 512u
static uint8_t  s_queue[APP_QUEUE_DEPTH][HOST_FRAME_MAX];
static size_t   s_queue_len[APP_QUEUE_DEPTH];
static uint32_t s_q_head = 0, s_q_tail = 0, s_q_count = 0;

void pdm_board_host_configure(FILE *samples, uint32_t tx_fail_every,
                              uint32_t tx_fail_until, uint32_t overrun_every,
                              uint16_t batt_mv, uint16_t batt_decay_mv,
                              float shaft_hz, int shaft_estimated,
                              int selftest_fail)
{
    s_cfg.samples = samples;
    s_cfg.tx_fail_every = tx_fail_every;
    s_cfg.tx_fail_until = tx_fail_until;
    s_cfg.overrun_every = overrun_every;
    s_cfg.battery_start_mv = batt_mv;
    s_cfg.battery_decay_mv = batt_decay_mv;
    s_cfg.shaft_hz = shaft_hz;
    s_cfg.shaft_estimated = shaft_estimated;
    s_cfg.selftest_fail = selftest_fail;
    s_batt = batt_mv;
    s_tx_calls = 0;
    s_acq_calls = 0;
    s_q_head = s_q_tail = s_q_count = 0;
}

/* ------------------------------------------------------------- interface */
pdm_board_status pdm_board_init(pdm_board_info *info)
{
    if (info != NULL) {
        info->kit = APP_KIT_NAME;
        info->accel_part = "host-file";
        info->device_uid = 0x00A1B2C3D4E5F600ULL;
        info->fs_hz = APP_FS_HZ;
        info->full_scale_g = APP_FULL_SCALE_G;
        info->adc_bits = 16u;
        info->flash_bytes = 2u * 1024u * 1024u;
        info->ram_bytes = 786u * 1024u;
    }
    return s_cfg.selftest_fail ? PDM_BOARD_ERR_SELFTEST : PDM_BOARD_OK;
}

pdm_board_status pdm_board_accel_power(int on)
{
    s_accel_on = on;
    return PDM_BOARD_OK;
}

uint32_t pdm_board_accel_settle_ms(void)
{
    return APP_SENSOR_SETTLE_MS;
}

#if defined(APP_N_AXES) && APP_N_AXES > 1u
/* Multi-channel emulation, for the RA8D1's three-axis path.
 *
 * Channel 0 is the sample file. Channels 1 and 2 are attenuated copies with a
 * one-sample shift -- NOT a physical model of a triaxial mount, just enough
 * decorrelation that the three frames differ, so a test can prove the channel
 * tag matches the channel's data rather than that three identical frames went
 * out. The real cross-axis relationship needs a rotor rig.
 */
static float   s_block[APP_ACQ_SAMPLES];
static uint8_t s_block_valid;
static uint8_t s_channel;
static uint32_t s_releases;

void pdm_board_select_channel(uint8_t channel)
{
    s_channel = (uint8_t)(channel % APP_N_AXES);
}

void pdm_board_release_block(void)
{
    s_block_valid = 0u;
    ++s_releases;
}

uint16_t pdm_board_extra_flags(void)
{
    /* The RA8D1 sets PDM_FRAME_ORDER_TRACKED here when it has resampled. The
     * host build never resamples, so the flag must stay clear -- a frame
     * wrongly claiming the angle domain would be compared against the wrong
     * baseline in the cloud. */
    return 0u;
}

uint32_t pdm_board_host_release_count(void)
{
    return s_releases;
}
#endif

pdm_board_status pdm_board_acquire(float *out, size_t n, float *fs_hz_actual)
{
    if (out == NULL || s_cfg.samples == NULL) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    if (fs_hz_actual != NULL) {
        *fs_hz_actual = APP_FS_HZ;
    }
#if defined(APP_N_AXES) && APP_N_AXES > 1u
    if (s_channel > 0u) {
        /* Deliberately BEFORE the power gate. The shared app loop powers the
         * sensor down the moment channel 0's burst is in RAM -- the extra axes
         * are a memory copy and must not require a powered sensor. The first
         * version checked power first, so axes 1 and 2 returned NOT_READY and
         * the three-axis kit silently shipped one frame per acquisition
         * instead of three. Found by test_multi_axis_emits_one_tagged_frame.
         */
        /* A RAM copy, exactly as on the real board: no bus traffic, no new
         * acquisition, and crucially no second ++s_acq_calls -- the injected
         * overrun and tx-failure cadences count ACQUISITIONS, and letting the
         * extra axes advance them would make a 3-axis run fail every injected
         * fault test for reasons that have nothing to do with the axes. */
        if (!s_block_valid) {
            return PDM_BOARD_ERR_NOT_READY;
        }
        const float att = (s_channel == 1u) ? 0.60f : 0.35f;
        for (size_t i = 0; i < n; ++i) {
            const size_t j = (i + s_channel) % n;
            out[i] = s_block[j] * att;
        }
        return PDM_BOARD_OK;
    }
#endif
    if (!s_accel_on) {
        /* Acquiring from a powered-down sensor returns zeros on real hardware,
         * which looks like a perfectly smooth machine. Refusing is the only
         * safe behaviour, and this stub enforces it so the application's power
         * sequencing is actually tested. */
        return PDM_BOARD_ERR_NOT_READY;
    }
    ++s_acq_calls;
    for (size_t i = 0; i < n; ++i) {
        double v = 0.0;
        if (fscanf(s_cfg.samples, "%lf", &v) != 1) {
            /* Rewind and continue: the sample file is a loop, so a long run can
             * be driven from a short record. */
            rewind(s_cfg.samples);
            if (fscanf(s_cfg.samples, "%lf", &v) != 1) {
                return PDM_BOARD_ERR_TIMEOUT;
            }
        }
        out[i] = (float)v;
    }
#if defined(APP_N_AXES) && APP_N_AXES > 1u
    for (size_t i = 0; i < n && i < APP_ACQ_SAMPLES; ++i) {
        s_block[i] = out[i];
    }
    s_block_valid = 1u;
#endif
    if (s_cfg.overrun_every && (s_acq_calls % s_cfg.overrun_every) == 0u) {
        return PDM_BOARD_ERR_OVERRUN;
    }
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_shaft_hz(float *shaft_hz, int *estimated)
{
    if (shaft_hz == NULL) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    if (!(s_cfg.shaft_hz > 0.0f)) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    *shaft_hz = s_cfg.shaft_hz;
    if (estimated != NULL) {
        *estimated = s_cfg.shaft_estimated;
    }
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_temperature_c(float *temp_c)
{
    if (temp_c != NULL) {
        *temp_c = 48.5f;
    }
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_battery_mv(uint16_t *mv)
{
    if (mv != NULL) {
        *mv = s_batt;
    }
    if (s_cfg.battery_decay_mv && s_batt > s_cfg.battery_decay_mv) {
        s_batt = (uint16_t)(s_batt - s_cfg.battery_decay_mv);
    }
    return PDM_BOARD_OK;
}

int16_t pdm_board_rssi_dbm(void)
{
    return -95;
}

pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len)
{
    ++s_tx_calls;
    if (s_cfg.tx_fail_until && s_tx_calls <= s_cfg.tx_fail_until) {
        return PDM_BOARD_ERR_BUS;
    }
    if (s_cfg.tx_fail_every && (s_tx_calls % s_cfg.tx_fail_every) != 0u) {
        return PDM_BOARD_ERR_BUS;
    }
    /* Length-prefixed so the consumer can frame the stream without parsing the
     * payload -- the real transports are message-oriented and do not need this.
     */
    const uint8_t hdr[2] = {(uint8_t)(len & 0xFFu), (uint8_t)((len >> 8) & 0xFFu)};
    if (fwrite(hdr, 1u, 2u, stdout) != 2u) {
        return PDM_BOARD_ERR_BUS;
    }
    if (fwrite(frame, 1u, len, stdout) != len) {
        return PDM_BOARD_ERR_BUS;
    }
    (void)fflush(stdout);
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_queue_push(const uint8_t *frame, size_t len)
{
    if (frame == NULL || len > HOST_FRAME_MAX) {
        return PDM_BOARD_ERR_BUS;
    }
    if (s_q_count >= APP_QUEUE_DEPTH) {
        /* Full. Drop the OLDEST, not the newest: on a machine that is
         * degrading, the newest frame is the most informative one, and a queue
         * that rejects new data while holding three-week-old data has its
         * priorities backwards. */
        s_q_head = (s_q_head + 1u) % APP_QUEUE_DEPTH;
        --s_q_count;
    }
    memcpy(s_queue[s_q_tail], frame, len);
    s_queue_len[s_q_tail] = len;
    s_q_tail = (s_q_tail + 1u) % APP_QUEUE_DEPTH;
    ++s_q_count;
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_queue_pop(uint8_t *frame, size_t cap, size_t *len)
{
    if (frame == NULL || len == NULL || s_q_count == 0u) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    const size_t l = s_queue_len[s_q_head];
    if (l > cap) {
        return PDM_BOARD_ERR_BUS;
    }
    memcpy(frame, s_queue[s_q_head], l);
    *len = l;
    s_q_head = (s_q_head + 1u) % APP_QUEUE_DEPTH;
    --s_q_count;
    return PDM_BOARD_OK;
}

uint32_t pdm_board_queue_depth(void)
{
    return s_q_count;
}

uint32_t pdm_board_sleep_ms(uint32_t ms)
{
    /* Accounted, not actually slept: a test that waits an hour per cycle is a
     * test nobody runs. The accumulated total is reported so the sleep
     * behaviour (back-off when queuing, speed-up when alarming) is still
     * observable. */
    s_sleep_total_ms += ms;
    return ms;
}

uint64_t pdm_board_host_sleep_total_ms(void)
{
    return s_sleep_total_ms;
}

void pdm_board_identify(uint32_t seconds)
{
    (void)seconds;
}

void pdm_board_watchdog_feed(void)
{
}
