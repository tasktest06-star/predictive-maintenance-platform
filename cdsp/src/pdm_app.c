/* pdm_app.c -- the application state machine, shared by every MCU kit.
 *
 * A kit's main() is three lines: init the board, init the app, loop. Everything
 * here is vendor-independent; everything vendor-specific is behind pdm_board.h.
 * That is why five different silicon vendors' branches contain a board driver
 * and nothing else.
 *
 * The loop, and why each step is where it is:
 *
 *   1. feed the watchdog
 *   2. check the battery FIRST, before spending any energy
 *   3. power the sensor, wait out the settling time
 *   4. acquire
 *   5. run the tiered pipeline
 *   6. encode a frame
 *   7. try to transmit; queue on failure
 *   8. drain the queue opportunistically while the link is up
 *   9. power the sensor down (or not -- interval dependent)
 *  10. sleep
 *
 * Step 2 before step 3 is deliberate. A node at 3.25 V can still transmit a
 * beacon saying "my battery is flat"; it cannot reliably complete a 307 ms
 * acquisition and a radio transmission, and attempting both is how a node dies
 * mid-frame and leaves a corrupt entry at the head of its flash queue. The
 * last useful thing a dying sensor does should be to tell someone.
 *
 * Step 8 after step 7 is also deliberate: the link is demonstrably up at that
 * moment, which is the cheapest possible time to drain the backlog. Draining on
 * a schedule instead means paying a connection cost to discover the link is
 * still down.
 */

#include "pdm_app.h"

#include <string.h>

#include "app_config.h"
#include "pdm_board.h"
#include "pdm_payload.h"
#include "pdm_pipeline.h"

/* Acquisition buffer. The single largest allocation in the firmware, and the
 * number that decides which parts are in scope. */
static float    s_acq[APP_ACQ_SAMPLES];
static uint8_t  s_frame[PDM_PAYLOAD_HEADER_SIZE + 4u * PDM_N_FEATURES + 4u];

static pdm_board_info s_board;
static pdm_app_state  s_state;

/* ------------------------------------------------------------------- init */
pdm_status pdm_app_init(float operating_threshold)
{
    memset(&s_state, 0, sizeof(s_state));
    s_state.interval_s = APP_DEFAULT_INTERVAL_S;
    s_state.threshold = operating_threshold;

    if (pdm_board_init(&s_board) != PDM_BOARD_OK) {
        /* Self-test failure included. A node that cannot verify its own sensor
         * must say so at boot rather than stream plausible noise for a year --
         * the pending flag makes the first frame carry SELFTEST_FAIL, which the
         * cloud refuses to score. */
        s_state.pending_flags |= PDM_FRAME_SELFTEST_FAIL;
    }
    const pdm_status st = pdm_pipeline_init();
    if (st != PDM_OK) {
        return st;
    }
    /* Feature-count agreement between the pipeline, the exported model and the
     * wire format. Checked once at boot because the alternative is a fleet
     * emitting vectors the cloud will reject -- or worse, accept. */
    if (pdm_pipeline_n_features() != PDM_N_FEATURES) {
        return PDM_ERR_SIZE;
    }
    s_state.initialised = 1u;
    return PDM_OK;
}

const pdm_board_info *pdm_app_board(void)
{
    return &s_board;
}

const pdm_app_state *pdm_app_state_get(void)
{
    return &s_state;
}

void pdm_app_restore_sequence(uint16_t sequence, uint32_t cycles)
{
    /* For kits whose sleep mode loses RAM. The ESP32-S3 uses deep sleep
     * (7 uA against light sleep's 240 uA, which at a 6-hour interval is 40x the
     * entire acquisition budget), and deep sleep restarts from app_main with
     * SRAM gone. Only RTC slow memory survives.
     *
     * The sequence number MUST be restored. Without it every wake restarts at
     * zero, and the ingest service -- which deduplicates on
     * (device_id, sequence) precisely so that store-and-forward replays are
     * idempotent -- silently discards every frame after the first wake as a
     * duplicate. The device would look healthy and transmit nothing.
     *
     * `cycles` is restored so the first-cycle sensor-power path and the
     * diagnostics stay coherent across the sleep. */
    s_state.sequence = sequence;
    s_state.cycles = cycles;
}

void pdm_app_set_config(uint32_t interval_s, float threshold)
{
    /* Clamped rather than trusted. A downlink that sets the interval to zero
     * would busy-loop the acquisition and flatten the battery in hours, and a
     * malformed config message is a far more likely cause of that than malice.
     */
    if (interval_s < 10u) {
        interval_s = 10u;
        s_state.pending_flags |= PDM_FRAME_FALLBACK_CONFIG;
    }
    if (interval_s > 86400u) {
        interval_s = 86400u;
        s_state.pending_flags |= PDM_FRAME_FALLBACK_CONFIG;
    }
    s_state.interval_s = interval_s;
    if (threshold > 0.0f) {
        s_state.threshold = threshold;
    }
}

/* ------------------------------------------------------------- one cycle */
pdm_app_result pdm_app_step(uint64_t now_ms)
{
    pdm_app_result r;
    memset(&r, 0, sizeof(r));
    if (!s_state.initialised) {
        r.status = PDM_ERR_ARG;
        return r;
    }
    pdm_board_watchdog_feed();
    ++s_state.cycles;

    /* --- 2. battery before anything expensive -------------------------- */
    uint16_t mv = 0;
    if (pdm_board_battery_mv(&mv) == PDM_BOARD_OK) {
        s_state.battery_mv = mv;
        if (mv < APP_BATT_CRITICAL_MV) {
            /* Stop acquiring. Beacon only, so the last thing this node does is
             * tell someone it is dying rather than going silent -- which on a
             * dashboard is indistinguishable from "healthy machine". */
            s_state.pending_flags |= PDM_FRAME_LOW_BATTERY;
            r.status = PDM_OK;
            r.action = PDM_APP_BATTERY_CRITICAL;
            return r;
        }
        if (mv < APP_BATT_LOW_MV) {
            s_state.pending_flags |= PDM_FRAME_LOW_BATTERY;
        }
    }

    /* --- 3. sensor power and settling ---------------------------------- */
    const int keep_alive = (s_state.interval_s < APP_SENSOR_KEEPALIVE_THRESHOLD_S);
    if (!keep_alive || s_state.cycles == 1u) {
        if (pdm_board_accel_power(1) != PDM_BOARD_OK) {
            r.status = PDM_ERR_UNSUPPORTED;
            r.action = PDM_APP_SENSOR_FAIL;
            return r;
        }
        /* Sampling before the sensor has settled gives a transient that looks
         * exactly like an impact train -- the most expensive 100 ms in the
         * budget and the one most often skipped. */
        (void)pdm_board_sleep_ms(pdm_board_accel_settle_ms());
    }

    /* --- shaft speed, before acquiring ------------------------------- */
    float shaft_hz = 0.0f;
    int estimated = 0;
    if (pdm_board_shaft_hz(&shaft_hz, &estimated) != PDM_BOARD_OK || !(shaft_hz > 0.0f)) {
        /* No speed, no frame. Order and bearing features computed against a
         * guessed speed are worse than no features: they look plausible and
         * point at the wrong bearing. */
        if (!keep_alive) {
            (void)pdm_board_accel_power(0);
        }
        r.status = PDM_ERR_ARG;
        r.action = PDM_APP_NO_SPEED;
        return r;
    }
    if (estimated) {
        s_state.pending_flags |= PDM_FRAME_SPEED_ESTIMATED;
    }

    /* --- 4. acquire ---------------------------------------------------- */
    float fs_actual = APP_FS_HZ;
    const pdm_board_status acq = pdm_board_acquire(s_acq, APP_ACQ_SAMPLES, &fs_actual);
    if (!keep_alive) {
        (void)pdm_board_accel_power(0);
    }
    if (acq == PDM_BOARD_ERR_OVERRUN) {
        /* The block is a concatenation of two different times: its spectrum has
         * a step discontinuity that looks exactly like an impact. So it is not
         * scored -- but a *diagnostic* frame is still emitted, with zeroed
         * features and the overrun flag set.
         *
         * Two reasons for emitting rather than staying silent:
         *   - a failed acquisition is otherwise invisible to the cloud except
         *     as a missing hour, indistinguishable from a dead sensor;
         *   - the flag must travel on THIS cycle's frame. The first version set
         *     it as a sticky pending flag, which carried it onto the *next*
         *     acquisition -- a perfectly good one -- and made the cloud refuse
         *     to score it. Caught by test_virtual_device.py.
         */
        ++s_state.overruns;
        float temp_ov = 0.0f;
        (void)pdm_board_temperature_c(&temp_ov);
        pdm_payload_meta om;
        memset(&om, 0, sizeof(om));
        om.device_id = s_board.device_uid;
        om.timestamp_ms = now_ms;
        om.uptime_s = (uint32_t)(now_ms / 1000u);
        om.battery_mv = s_state.battery_mv;
        om.rssi_dbm = pdm_board_rssi_dbm();
        om.shaft_hz = shaft_hz;
        om.temperature_c = temp_ov;
        om.flags = (uint16_t)(PDM_FRAME_FIFO_OVERRUN | s_state.pending_flags);
        om.sequence = s_state.sequence;
        static float zeros[PDM_N_FEATURES];
        size_t olen = 0;
        if (pdm_payload_encode(&om, zeros, PDM_N_FEATURES, s_frame,
                               sizeof(s_frame), &olen) == PDM_OK) {
            s_state.sequence = (uint16_t)((s_state.sequence + 1u) & 0xFFFFu);
            s_state.pending_flags = 0u;
            if (pdm_board_transmit(s_frame, olen) == PDM_BOARD_OK) {
                ++s_state.frames_sent;
                r.transmitted = 1u;
            } else if (pdm_board_queue_push(s_frame, olen) == PDM_BOARD_OK) {
                ++s_state.frames_queued;
                r.queued = 1u;
            } else {
                ++s_state.frames_dropped;
            }
        }
        r.status = PDM_OK;
        r.action = PDM_APP_ACQUIRE_FAIL;
        return r;
    }
    if (acq != PDM_BOARD_OK) {
        r.status = PDM_ERR_UNSUPPORTED;
        r.action = PDM_APP_ACQUIRE_FAIL;
        return r;
    }

    /* --- 5. pipeline --------------------------------------------------- */
    pdm_pipeline_result pr;
    memset(&pr, 0, sizeof(pr));
    pr.threshold_hint = s_state.threshold;
    const pdm_status pst = pdm_pipeline_run(s_acq, APP_ACQ_SAMPLES, shaft_hz, &pr);
    if (pst != PDM_OK) {
        r.status = pst;
        r.action = PDM_APP_PIPELINE_FAIL;
        return r;
    }
    s_state.last_score = pr.score;
    s_state.last_tier = pr.tier;

    /* --- 6. encode ----------------------------------------------------- */
    float temp_c = 0.0f;
    (void)pdm_board_temperature_c(&temp_c);

    pdm_payload_meta meta;
    memset(&meta, 0, sizeof(meta));
    meta.device_id = s_board.device_uid;
    meta.timestamp_ms = now_ms;
    meta.uptime_s = (uint32_t)(now_ms / 1000u);
    meta.battery_mv = s_state.battery_mv;
    meta.rssi_dbm = pdm_board_rssi_dbm();
    /* The rate the hardware actually used, not the nominal one. A 3 % MEMS
     * oscillator error moves a 106 Hz BPFO line by 3.2 Hz, which the cloud needs
     * in order to re-derive the kinematic frequencies correctly. */
    meta.shaft_hz = shaft_hz;
    meta.temperature_c = temp_c;
    meta.flags = (uint16_t)(pr.flags | s_state.pending_flags);
    meta.sequence = s_state.sequence;

    size_t len = 0;
    const pdm_status est = pdm_payload_encode(&meta, pr.features, PDM_N_FEATURES,
                                              s_frame, sizeof(s_frame), &len);
    if (est != PDM_OK) {
        r.status = est;
        r.action = PDM_APP_ENCODE_FAIL;
        return r;
    }
    /* Sequence advances only once a frame has been built, so a failed
     * acquisition does not create a gap the cloud would report as lost data. */
    s_state.sequence = (uint16_t)((s_state.sequence + 1u) & 0xFFFFu);
    s_state.pending_flags = 0u;

    /* --- 7. transmit, or queue ----------------------------------------- */
    if (pdm_board_transmit(s_frame, len) == PDM_BOARD_OK) {
        ++s_state.frames_sent;
        r.transmitted = 1u;
        /* --- 8. the link is demonstrably up: drain the backlog --------- */
        uint32_t drained = 0;
        while (pdm_board_queue_depth() > 0u && drained < PDM_APP_MAX_DRAIN) {
            size_t qlen = 0;
            if (pdm_board_queue_pop(s_frame, sizeof(s_frame), &qlen) != PDM_BOARD_OK) {
                break;
            }
            /* Mark the replay so the cloud can distinguish a backlog flush from
             * live data; it deduplicates on (device, sequence) either way, but
             * a sudden burst of in-order frames with old timestamps is
             * otherwise indistinguishable from a clock fault. */
            s_frame[40] = (uint8_t)(s_frame[40] | PDM_FRAME_STORED_FORWARDED);
            /* The flags byte changed, so the CRC must be recomputed. Forgetting
             * this turns every replayed frame into a CRC rejection, i.e. the
             * store-and-forward feature silently throws away everything it
             * saved. */
            {
                const uint32_t crc = pdm_crc32(s_frame, qlen - 4u);
                for (int i = 0; i < 4; ++i) {
                    s_frame[qlen - 4u + (size_t)i] = (uint8_t)((crc >> (8 * i)) & 0xFFu);
                }
            }
            if (pdm_board_transmit(s_frame, qlen) != PDM_BOARD_OK) {
                /* Push it back and stop: the link went down mid-drain. */
                (void)pdm_board_queue_push(s_frame, qlen);
                break;
            }
            ++drained;
            ++s_state.frames_sent;
            pdm_board_watchdog_feed();
        }
        r.drained = drained;
    } else {
        if (pdm_board_queue_push(s_frame, len) == PDM_BOARD_OK) {
            ++s_state.frames_queued;
            r.queued = 1u;
        } else {
            ++s_state.frames_dropped;
        }
    }

    r.status = PDM_OK;
    r.action = (pr.request_snapshot) ? PDM_APP_SNAPSHOT_REQUESTED : PDM_APP_FRAME_READY;
    r.tier = pr.tier;
    r.score = pr.score;
    r.iso_zone = pr.iso_zone;
    r.velocity_rms_mm_s = pr.velocity_rms_mm_s;
    return r;
}

uint32_t pdm_app_next_sleep_ms(void)
{
    uint32_t interval = s_state.interval_s;
    /* A node that cannot reach its gateway and is filling its queue should not
     * also be burning its battery at the normal rate -- it backs off. Capped at
     * 4x so a transient outage does not blind the asset for a day. */
    const uint32_t depth = pdm_board_queue_depth();
    if (depth > APP_QUEUE_DEPTH / 2u) {
        interval *= 4u;
    } else if (depth > APP_QUEUE_DEPTH / 8u) {
        interval *= 2u;
    }
    /* An asset in alarm is sampled faster: the detection delay that matters is
     * the one after the first indication, not the one before it. */
    if (s_state.threshold > 0.0f && s_state.last_score > s_state.threshold) {
        interval /= 4u;
        if (interval < 60u) {
            interval = 60u;
        }
    }
    return interval * 1000u;
}
