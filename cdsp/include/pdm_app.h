/* pdm_app.h -- the vendor-independent application state machine.
 *
 * A kit's main() is three lines:
 *
 *     pdm_app_init(threshold);
 *     for (;;) {
 *         pdm_app_step(now_ms());
 *         pdm_board_sleep_ms(pdm_app_next_sleep_ms());
 *     }
 *
 * Everything else is in pdm_app.c (shared) or behind pdm_board.h (per kit).
 */

#ifndef PDM_APP_H
#define PDM_APP_H

#include <stdint.h>

#include "pdm_board.h"
#include "pdm_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Frames drained per successful transmission. Bounded so a large backlog
 * cannot hold the radio on long enough to brown out the supply -- on the
 * cellular kit a 64-frame drain is 6 s of continuous TX. */
#define PDM_APP_MAX_DRAIN 16u

typedef enum {
    PDM_APP_FRAME_READY = 0,
    PDM_APP_SNAPSHOT_REQUESTED,
    PDM_APP_NO_SPEED,
    PDM_APP_ACQUIRE_FAIL,
    PDM_APP_SENSOR_FAIL,
    PDM_APP_PIPELINE_FAIL,
    PDM_APP_ENCODE_FAIL,
    PDM_APP_BATTERY_CRITICAL
} pdm_app_action;

typedef struct {
    pdm_status      status;
    pdm_app_action  action;
    uint8_t         tier;
    uint8_t         transmitted;
    uint8_t         queued;
    uint8_t         iso_zone;
    uint32_t        drained;
    float           score;
    float           velocity_rms_mm_s;
} pdm_app_result;

typedef struct {
    uint8_t  initialised;
    uint16_t sequence;
    uint16_t battery_mv;
    /* STICKY conditions only: SELFTEST_FAIL, LOW_BATTERY, FALLBACK_CONFIG --
     * things that persist until resolved and should mark every frame until they
     * are. Per-acquisition conditions (CLIPPED, FIFO_OVERRUN, SPEED_ESTIMATED)
     * belong on that cycle's frame alone; putting FIFO_OVERRUN here made the
     * *next*, perfectly good, acquisition unscoreable. */
    uint16_t pending_flags;
    uint32_t interval_s;
    uint32_t cycles;
    uint32_t frames_sent;
    uint32_t frames_queued;
    uint32_t frames_dropped;
    uint32_t overruns;
    uint8_t  last_tier;
    float    last_score;
    float    threshold;
} pdm_app_state;

/* `operating_threshold` is the calibrated anomaly-score threshold, pushed down
 * from the cloud. Pass 0 to disable on-device escalation entirely (the device
 * then always computes tier 2, which costs ~3 ms more per cycle). */
pdm_status pdm_app_init(float operating_threshold);

/* One acquisition cycle. `now_ms` is Unix milliseconds UTC. */
pdm_app_result pdm_app_step(uint64_t now_ms);

/* Milliseconds to sleep before the next cycle. Backs off when the queue is
 * filling and speeds up when the asset is in alarm. */
uint32_t pdm_app_next_sleep_ms(void);

void pdm_app_set_config(uint32_t interval_s, float threshold);
const pdm_board_info *pdm_app_board(void);
const pdm_app_state  *pdm_app_state_get(void);

#ifdef __cplusplus
}
#endif
#endif /* PDM_APP_H */
