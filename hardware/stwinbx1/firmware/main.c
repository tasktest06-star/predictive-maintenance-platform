/* main.c -- STEVAL-STWINBX1 node.
 *
 * Three lines of application logic. Everything else is either shared
 * (cdsp/src/pdm_app.c) or board-specific (board_stwinbx1.c) -- which is the
 * point of the split, and the reason five silicon vendors' branches differ only
 * in a driver.
 */

#include "pdm_app.h"
#include "pdm_board.h"

#include "app_config.h"
#include "stm32u5xx_hal.h"

/* CubeMX-generated, in board_init.c. */
void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_SPI2_Init(void);
void MX_I2C2_Init(void);
void MX_ADC1_Init(void);
void MX_GPDMA1_Init(void);
void MX_IWDG_Init(void);
void MX_LPTIM1_Init(void);

/* board_stwinbx1.c */
void pdm_board_stwinbx1_set_shaft_hz(float hz, int measured);

/* ble_transport.c: receives the retained `cfg` message on connect and calls
 * pdm_app_set_config / pdm_board_stwinbx1_set_shaft_hz. A device wakes from
 * Stop 2 with no idea whether it was reconfigured while asleep, so the
 * configuration has to arrive on connection rather than being polled. */
int  ble_transport_init(const char *name);
void ble_transport_poll(void);

/* The operating threshold and the shaft speed both come from commissioning.
 * These are the fallbacks used before the first successful connection, chosen
 * so the node does something sensible rather than nothing:
 *
 *   - threshold 0 disables on-device escalation, so every cycle computes the
 *     full tier-2 vector. Costs ~3 ms more per acquisition and guarantees the
 *     cloud gets bearing evidence from the very first frame, which is what the
 *     commissioning window needs.
 *   - a shaft speed of 0 would make the node refuse to acquire at all, so the
 *     nameplate default is compiled in and the frame is flagged as estimated
 *     until a measured value arrives.
 */
#define BOOT_THRESHOLD   0.0f
#define BOOT_SHAFT_HZ    29.6667f   /* 1780 rpm, the commissioning default */

static uint64_t unix_ms(void)
{
    /* RTC, set from the gateway on each connection. Before the first
     * connection this is time-since-boot, which the cloud can still order but
     * cannot align with other assets -- so the first frames carry a timestamp
     * the ingest service treats with suspicion rather than a fabricated one. */
    RTC_TimeTypeDef t;
    RTC_DateTypeDef d;
    extern RTC_HandleTypeDef hrtc;
    if (HAL_RTC_GetTime(&hrtc, &t, RTC_FORMAT_BIN) != HAL_OK
        || HAL_RTC_GetDate(&hrtc, &d, RTC_FORMAT_BIN) != HAL_OK) {
        return (uint64_t)HAL_GetTick();
    }
    /* Days since the Unix epoch for a 20xx date, via the civil-from-days
     * algorithm. Deliberately not mktime(): newlib's mktime pulls in the
     * timezone machinery and 6 kB of flash for a UTC-only device. */
    const int32_t y = 2000 + (int32_t)d.Year;
    const int32_t m = (int32_t)d.Month;
    const int32_t day = (int32_t)d.Date;
    const int32_t a = (m <= 2) ? 1 : 0;
    const int32_t yy = y - a;
    const int32_t mm = m + 12 * a - 3;
    const int32_t era = (yy >= 0 ? yy : yy - 399) / 400;
    const int32_t yoe = yy - era * 400;
    const int32_t doy = (153 * mm + 2) / 5 + day - 1;
    const int32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = (int64_t)era * 146097 + doe - 719468 - 60;
    return (uint64_t)(days * 86400 + t.Hours * 3600 + t.Minutes * 60 + t.Seconds) * 1000u;
}

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_GPDMA1_Init();
    MX_SPI2_Init();
    MX_I2C2_Init();
    MX_ADC1_Init();
    MX_LPTIM1_Init();
    MX_IWDG_Init();

    (void)ble_transport_init(APP_BLE_DEVICE_NAME);
    pdm_board_stwinbx1_set_shaft_hz(BOOT_SHAFT_HZ, 0);

    if (pdm_app_init(BOOT_THRESHOLD) != PDM_OK) {
        /* Cannot run the pipeline: window generation, twiddle tables or the
         * model failed a consistency check. Blink continuously and keep the
         * watchdog fed so the node is visibly broken rather than silently
         * rebooting in a loop, which looks identical to a flat battery. */
        for (;;) {
            pdm_board_identify(10u);
            pdm_board_watchdog_feed();
        }
    }

    for (;;) {
        ble_transport_poll();                 /* apply any pending cfg/cmd */
        (void)pdm_app_step(unix_ms());
        (void)pdm_board_sleep_ms(pdm_app_next_sleep_ms());
    }
}
