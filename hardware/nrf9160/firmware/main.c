/* main.c -- nRF9160 cellular node.
 *
 * Zephyr's main thread. Unlike the ESP32-S3 this kit keeps SRAM across sleep,
 * so the loop is a real loop.
 *
 * The LTE link is brought up once at boot and then left REGISTERED in PSM. It
 * is never torn down: re-attaching costs roughly 1000 mA.s against an RRC
 * connection's 95, so a node that detaches to "save power" spends ten
 * transmissions' worth of energy to save a microamp.
 */

#include "pdm_app.h"
#include "pdm_board.h"

#include "app_config.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <modem/lte_lc.h>
#include <modem/nrf_modem_lib.h>
#include <date_time.h>

LOG_MODULE_REGISTER(pdm_main, LOG_LEVEL_INF);

void pdm_board_nrf9160_set_shaft_hz(float hz, int measured);
int  lte_transport_init(void);

#define BOOT_SHAFT_HZ 29.6667f

static uint64_t unix_ms(void)
{
    int64_t ms = 0;
    if (date_time_now(&ms) == 0) {
        return (uint64_t)ms;
    }
    /* No network time yet. Return uptime so the cloud can order the frames but
     * cannot align them with other assets -- and the ingest service can see
     * from the implausible timestamp that this is a pre-sync frame rather than
     * trusting a fabricated date. */
    return (uint64_t)k_uptime_get();
}

int main(void)
{
    LOG_INF("%s %s, feature spec v%d", APP_KIT_NAME, APP_FIRMWARE_VERSION, APP_FEATURE_SPEC);

    if (nrf_modem_lib_init() != 0) {
        LOG_ERR("modem library init failed");
        /* Keep going. A node that cannot reach the network can still acquire,
         * score and queue -- and the queue is 42 days deep. Refusing to boot
         * because the radio is unavailable would turn a coverage problem into a
         * data loss. */
    }
    if (lte_transport_init() != 0) {
        LOG_WRN("LTE not up at boot; frames will queue");
    }

    if (pdm_app_init(0.0f) != PDM_OK) {
        LOG_ERR("pdm_app_init failed -- pipeline/model inconsistency");
        for (;;) {
            pdm_board_identify(10u);
            pdm_board_watchdog_feed();
        }
    }
    pdm_app_set_config(APP_DEFAULT_INTERVAL_S, 0.0f);
    pdm_board_nrf9160_set_shaft_hz(BOOT_SHAFT_HZ, 0);

    const pdm_board_info *bi = pdm_app_board();
    LOG_INF("accel %s, fs %d Hz, %u features, RAM %u kB",
            bi->accel_part, (int)bi->fs_hz, (unsigned)pdm_pipeline_n_features(),
            (unsigned)(bi->ram_bytes / 1024u));

    for (;;) {
        const pdm_app_result r = pdm_app_step(unix_ms());
        const pdm_app_state *st = pdm_app_state_get();
        LOG_INF("cycle %u tier %u score %d.%02d %s queue %u batt %u mV rsrp %d",
                (unsigned)st->cycles, (unsigned)r.tier,
                (int)r.score, (int)((r.score - (float)(int)r.score) * 100.0f),
                r.transmitted ? "tx" : (r.queued ? "queued" : "-"),
                (unsigned)pdm_board_queue_depth(), (unsigned)st->battery_mv,
                (int)pdm_board_rssi_dbm());
        (void)pdm_board_sleep_ms(pdm_app_next_sleep_ms());
    }
    return 0;
}
