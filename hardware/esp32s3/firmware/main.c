/* main.c -- ESP32-S3 retrofit node.
 *
 * Structurally different from the other MCU kits, and the difference is deep
 * sleep. `esp_deep_sleep_start()` does not return: the chip resets and
 * re-enters app_main. So the "loop" is one iteration per wake, and any state
 * that must survive lives in RTC slow memory.
 *
 * Light sleep would let the loop actually loop, at 240 uA. At the 6-hour
 * default interval that is 5.2 mA.h per cycle -- roughly 40x the entire
 * acquisition budget. Deep sleep is not optional here, so the structure follows
 * from the power budget rather than from taste.
 */

#include "pdm_app.h"
#include "pdm_board.h"

#include "app_config.h"

#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "pdm";

/* Survives deep sleep: 8 kB of RTC slow memory is retained while the rest of
 * SRAM is not. Everything the application needs to resume coherently goes here
 * -- lose the sequence number and the cloud sees duplicate frames; lose the
 * threshold and the node reverts to computing tier 2 every cycle. */
RTC_DATA_ATTR static uint16_t rtc_sequence;
RTC_DATA_ATTR static uint32_t rtc_cycles;
RTC_DATA_ATTR static uint32_t rtc_interval_s;
RTC_DATA_ATTR static float    rtc_threshold;
RTC_DATA_ATTR static uint64_t rtc_epoch_ms;     /* wall clock at the last sync */
RTC_DATA_ATTR static uint64_t rtc_uptime_ms;    /* accumulated across sleeps */
RTC_DATA_ATTR static uint8_t  rtc_valid;

void pdm_board_esp32s3_set_shaft_hz(float hz, int measured);
int  mqtt_transport_sync_time(uint64_t *epoch_ms);

#define BOOT_SHAFT_HZ 29.6667f

void app_main(void)
{
    const esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();
    const int cold_boot = (cause != ESP_SLEEP_WAKEUP_TIMER) || !rtc_valid;

    if (cold_boot) {
        rtc_sequence = 0u;
        rtc_cycles = 0u;
        rtc_interval_s = APP_DEFAULT_INTERVAL_S;
        rtc_threshold = 0.0f;   /* compute tier 2 until the cloud says otherwise */
        rtc_epoch_ms = 0u;
        rtc_uptime_ms = 0u;
        rtc_valid = 1u;
        ESP_LOGI(TAG, "cold boot");
    }

    if (pdm_app_init(rtc_threshold) != PDM_OK) {
        /* Blink and sleep rather than reboot-looping: a node that reboots every
         * second is indistinguishable from a flat battery, and it also drains
         * one. A long sleep makes the fault visible and survivable. */
        ESP_LOGE(TAG, "pdm_app_init failed");
        pdm_board_identify(10u);
        (void)pdm_board_sleep_ms(3600u * 1000u);
    }
    pdm_app_set_config(rtc_interval_s, rtc_threshold);
    pdm_board_esp32s3_set_shaft_hz(BOOT_SHAFT_HZ, 0);

    /* Restore the sequence number so the cloud's deduplication and gap counting
     * keep working across deep sleep. Without this every wake restarts at zero
     * and the ingest service deduplicates away real data. */
    extern void pdm_app_restore_sequence(uint16_t seq, uint32_t cycles);
    pdm_app_restore_sequence(rtc_sequence, rtc_cycles);

    const uint64_t now_ms = (rtc_epoch_ms > 0u)
        ? rtc_epoch_ms + rtc_uptime_ms
        : (uint64_t)(esp_timer_get_time() / 1000);

    const pdm_app_result r = pdm_app_step(now_ms);
    ESP_LOGI(TAG, "cycle %lu tier %u score %.3f %s queue %lu",
             (unsigned long)rtc_cycles, (unsigned)r.tier, (double)r.score,
             r.transmitted ? "tx" : (r.queued ? "queued" : "-"), 
             (unsigned long)pdm_board_queue_depth());

    /* Re-sync the wall clock whenever the link was up. A node whose timestamps
     * drift cannot have its trend aligned with the rest of the fleet, and the
     * ESP32's RTC drifts by minutes per day. */
    if (r.transmitted) {
        uint64_t synced = 0;
        if (mqtt_transport_sync_time(&synced) == 0 && synced > 0u) {
            rtc_epoch_ms = synced;
            rtc_uptime_ms = 0u;
        }
    }

    const pdm_app_state *st = pdm_app_state_get();
    rtc_sequence = st->sequence;
    rtc_cycles = st->cycles;
    rtc_interval_s = st->interval_s;
    rtc_threshold = st->threshold;

    const uint32_t sleep_ms = pdm_app_next_sleep_ms();
    rtc_uptime_ms += sleep_ms + (uint64_t)(esp_timer_get_time() / 1000);
    (void)pdm_board_sleep_ms(sleep_ms);   /* does not return */
}
