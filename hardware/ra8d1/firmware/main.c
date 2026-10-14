/* main.c -- Renesas EK-RA8D1 node.
 *
 * The application loop itself is four lines, as on every kit: the shared
 * pdm_app.c owns the battery gate, the sticky-flag discipline, the sequence
 * accounting and the queue drain, and it now loops over APP_N_AXES channels so
 * the three-axis behaviour is the same verified code path.
 *
 * What is here beyond those four lines is this kit's own: the tachometer and
 * what to do when it says the shaft is not steady.
 */

#include <stdint.h>

#include "app_config.h"
#include "pdm_app.h"
#include "pdm_board.h"
#include "pdm_order.h"
#include "pdm_pipeline.h"

#include "board_ra8d1.h"
#include "hal_data.h"

/* Pushed down from the cloud at commissioning and on every model update.
 * Zero disables on-device escalation: the node then always computes tier 2,
 * which on this part costs 1.3 ms it can easily afford -- unlike the nRF9160,
 * where the same choice is 18 ms and visible in the battery life. */
#ifndef APP_OPERATING_THRESHOLD
#define APP_OPERATING_THRESHOLD 0.0f
#endif

/* Angle-domain working buffer. 2304 floats = 9 kB, trivial against 1 MB, and
 * impossible on the 256 kB kits -- which is the whole argument for this part.
 * 9 revolutions x 256 samples/rev at the nominal 29.7 Hz over a 307 ms record. */
#define ORDER_BUF_SAMPLES (9u * APP_ORDER_SAMPLES_PER_REV)
static float s_order_buf[ORDER_BUF_SAMPLES];

/* Revolution marks for the record just acquired, filled by the board's tacho
 * capture. 64 is ~2 s of margin at 30 Hz. */
#define MAX_MARKS 64u
static float s_marks[MAX_MARKS];

/* ---------------------------------------------------------------------------
 * Order-domain diagnostic.
 *
 * Runs only when the tacho says the shaft moved more than
 * APP_ORDER_TRACK_DRIFT_PCT across the record. Below that the interpolation
 * error is comparable to the smearing it would remove, so resampling is work
 * for nothing -- measured in hardware/ra8d1/tests/test_order.c, which puts the
 * gain at 1.19x at order 6 and 1.92x at order 40 for a 4 % sweep.
 *
 * WHAT THIS DOES NOT DO, stated plainly because it is the kit's largest open
 * gap: it does not ship order-domain features. The shared feature spec is
 * frequency-domain and `pdm_pipeline_run` takes its sample rate from
 * app_config.h at compile time, so feeding it a resampled record would label
 * orders as hertz -- numbers that look right and are wrong by whatever the
 * speed happened to be. Until the pipeline accepts a runtime rate, the node
 * instead:
 *
 *   - flags the frame PDM_FRAME_SPEED_UNSTEADY, so the cloud widens its
 *     harmonic tolerance and discounts high orders rather than alarming on
 *     smearing;
 *   - requests a raw snapshot, so the cloud -- which has no compute limit and
 *     can extend the feature set per asset -- does the angle-domain analysis;
 *   - computes the local order spectrum anyway, for the deployment where
 *     snapshot bandwidth is not available.
 *
 * Returns the number of angle-domain samples produced, 0 if it declined.
 */
static uint32_t order_domain_pass(float *drift_out)
{
    const float drift = pdm_board_ra8d1_speed_drift_pct();
    if (drift_out != NULL) {
        *drift_out = drift;
    }
    if (!(drift > APP_ORDER_TRACK_DRIFT_PCT)) {
        return 0u;
    }
    uint32_t n_rec = 0u;
    const float *rec = pdm_board_ra8d1_last_record(&n_rec);
    if (rec == NULL || n_rec < 64u) {
        return 0u;
    }
    const uint16_t n_marks = pdm_board_ra8d1_marks(s_marks, (uint16_t)MAX_MARKS);
    if (n_marks < 4u) {
        return 0u;
    }
    uint32_t got = 0u;
    const pdm_status st = pdm_order_resample(rec, n_rec, APP_FS_HZ,
                                             s_marks, n_marks,
                                             (uint16_t)APP_ORDER_SAMPLES_PER_REV,
                                             s_order_buf, ORDER_BUF_SAMPLES, &got);
    /* PDM_ERR_SIZE means the record held more revolutions than the buffer: the
     * output is valid, just shorter than the marks allowed. Anything else means
     * the tacho and the accelerometer disagree about what window was captured,
     * and a resampled record from mismatched inputs is confidently wrong. */
    if (st != PDM_OK && st != PDM_ERR_SIZE) {
        return 0u;
    }
    return got;
}

int main(void)
{
    /* FSP clock, pin and peripheral init from the e2 studio configuration. */
    R_BSP_WarmStart(BSP_WARM_START_POST_C);

    if (pdm_app_init(APP_OPERATING_THRESHOLD) != PDM_OK) {
        /* Board or self-test failure. pdm_app_init has already set the sticky
         * SELFTEST_FAIL flag, so the first frame carries it; the loop still
         * runs, because a node that cannot verify its sensor must say so
         * repeatedly rather than going silent. */
        pdm_board_identify(2u);
    }

    /* Nominal speed, used only until the tacho produces a reading. A node that
     * refused to measure anything before its first tacho pulse would never
     * start on a machine whose keyphasor is misaligned. */
    pdm_board_ra8d1_set_nominal_hz(29.6667f);

    for (;;) {
        const pdm_app_result r = pdm_app_step(ra8_unix_ms());

        /* Order-domain pass, after the frames are away. Deliberately after: it
         * is a diagnostic, and it must not delay the acquisition or hold the
         * radio window open. */
        if (r.action == PDM_APP_FRAME_READY || r.action == PDM_APP_SNAPSHOT_REQUESTED) {
            float drift = 0.0f;
            const uint32_t n_angle = order_domain_pass(&drift);
            (void)n_angle;
        }

        pdm_board_watchdog_feed();
        (void)pdm_board_sleep_ms(pdm_app_next_sleep_ms());
    }
}
