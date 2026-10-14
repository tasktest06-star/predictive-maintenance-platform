/* fsp_stub.c -- definitions for the stub FSP symbols, so the syntax check can
 * LINK as well as compile. Linking matters: it catches a driver that calls a
 * function nobody provides, which on the real build is a late and confusing
 * error from the vendor makefile.
 *
 * Every function here does nothing and returns success. That is deliberate --
 * this is not a simulator. A test that exercised behaviour through these stubs
 * would be testing the stubs.
 */

#include <stddef.h>
#include <stdint.h>

#include "hal_data.h"

ioport_instance_ctrl_t    g_ioport_ctrl;
sci_b_spi_instance_ctrl_t g_spi0_ctrl;
const spi_cfg_t           g_spi0_cfg = {1000000u};
gpt_instance_ctrl_t       g_timer_tacho_ctrl;
const timer_cfg_t         g_timer_tacho_cfg = {0u};
adc_instance_ctrl_t       g_adc0_ctrl;
wdt_instance_ctrl_t       g_wdt0_ctrl;
rtc_instance_ctrl_t       g_rtc_ctrl;
volatile spi_event_t      g_spi0_event = SPI_EVENT_TRANSFER_COMPLETE;

void R_BSP_SoftwareDelay(uint32_t delay, bsp_delay_units_t units)
{
    (void)delay;
    (void)units;
}

void R_BSP_WarmStart(bsp_warm_start_event_t event)
{
    (void)event;
}

fsp_err_t R_IOPORT_PinWrite(ioport_instance_ctrl_t *p, bsp_io_port_pin_t pin,
                            bsp_io_level_t level)
{
    (void)p;
    (void)pin;
    (void)level;
    return FSP_SUCCESS;
}

fsp_err_t R_SCI_B_SPI_Open(sci_b_spi_instance_ctrl_t *p, const spi_cfg_t *c)
{
    (void)p;
    (void)c;
    return FSP_SUCCESS;
}

fsp_err_t R_SCI_B_SPI_WriteRead(sci_b_spi_instance_ctrl_t *p, const void *s, void *d,
                                uint32_t len, spi_bit_width_t w)
{
    (void)p;
    (void)s;
    (void)d;
    (void)len;
    (void)w;
    return FSP_SUCCESS;
}

fsp_err_t R_GPT_Open(gpt_instance_ctrl_t *p, const timer_cfg_t *c)
{
    (void)p;
    (void)c;
    return FSP_SUCCESS;
}

fsp_err_t R_GPT_Start(gpt_instance_ctrl_t *p)
{
    (void)p;
    return FSP_SUCCESS;
}

fsp_err_t R_ADC_ScanStart(adc_instance_ctrl_t *p)
{
    (void)p;
    return FSP_SUCCESS;
}

fsp_err_t R_ADC_StatusGet(adc_instance_ctrl_t *p, adc_status_t *s)
{
    (void)p;
    if (s != NULL) {
        s->state = ADC_STATE_IDLE;
    }
    return FSP_SUCCESS;
}

fsp_err_t R_ADC_Read(adc_instance_ctrl_t *p, adc_channel_t ch, uint16_t *d)
{
    (void)p;
    (void)ch;
    if (d != NULL) {
        *d = 2048u;
    }
    return FSP_SUCCESS;
}

fsp_err_t R_WDT_Refresh(wdt_instance_ctrl_t *p)
{
    (void)p;
    return FSP_SUCCESS;
}

fsp_err_t R_RTC_CalendarTimeGet(rtc_instance_ctrl_t *p, rtc_time_t *t)
{
    (void)p;
    if (t != NULL) {
        t->tm_year = 125;    /* 2025 */
        t->tm_mon = 9;
        t->tm_mday = 13;
        t->tm_hour = t->tm_min = t->tm_sec = 0;
    }
    return FSP_SUCCESS;
}

/* Platform shims the driver declares and the real port must provide. */
uint32_t ra8_gpt_source_hz(void);
uint32_t ra8_gpt_source_hz(void)
{
    return 120000000u;      /* PCLKD at 120 MHz in the reference clock tree */
}

void ra8_enter_standby_ms(uint32_t ms);
void ra8_enter_standby_ms(uint32_t ms)
{
    (void)ms;
}

uint64_t ra8_timegm(const rtc_time_t *tm);
uint64_t ra8_timegm(const rtc_time_t *tm)
{
    /* Days from the civil epoch, Howard Hinnant's algorithm: no libc, no
     * 32-bit time_t, correct past 2038. A node whose clock is wrong ships
     * frames the ingest service discards as stale. */
    int32_t y = tm->tm_year + 1900;
    const uint32_t m = (uint32_t)tm->tm_mon + 1u;
    const uint32_t d = (uint32_t)tm->tm_mday;
    y -= (m <= 2u) ? 1 : 0;
    const int32_t era = (y >= 0 ? y : y - 399) / 400;
    const uint32_t yoe = (uint32_t)(y - era * 400);
    const uint32_t doy = (153u * (m + (m > 2u ? -3u : 9u)) + 2u) / 5u + d - 1u;
    const uint32_t doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    const int64_t days = (int64_t)era * 146097 + (int64_t)doe - 719468;
    return (uint64_t)(days * 86400 + tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec);
}

int ble_transport_send(const uint8_t *data, uint32_t len, int16_t *rssi);
int ble_transport_send(const uint8_t *data, uint32_t len, int16_t *rssi)
{
    (void)data;
    (void)len;
    (void)rssi;
    return 0;
}
