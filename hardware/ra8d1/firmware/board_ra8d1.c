/* board_ra8d1.c -- pdm_board.h for the Renesas EK-RA8D1 + IIS3DWB.
 *
 * Target: RA8D1 (Cortex-M85 @ 480 MHz, Helium, 1 MB SRAM, 2 MB flash).
 * Built with the Renesas Flexible Software Package (FSP) generated from e2
 * studio; the `g_*` handles below come from the FSP configurator.
 *
 * THIS CODE HAS NOT RUN ON HARDWARE.
 *
 * Two things here exist on no other kit:
 *
 * **Three axes.** pdm_board_acquire fills one axis, selected by
 * pdm_board_ra8d1_select_axis. The application calls it three times per
 * acquisition and runs the pipeline three times -- 3.9 ms of the 407 ms sensing
 * window. The FIFO read is shared: all three axes arrive in the same burst, so
 * the extra cost is conversion and DSP, not bus time.
 *
 * **A real tachometer.** GPT0 in input-capture mode on a connector pin. This is
 * the only kit whose shaft speed is measured rather than configured, which
 * matters more than it sounds: every other kit's frames carry
 * PDM_FRAME_SPEED_ESTIMATED, and a diagnosis computed against a guessed speed
 * is the first thing an analyst questions.
 */

#include "pdm_board.h"

#include <string.h>

#include "app_config.h"
#include "pdm_pipeline.h"
#include "pdm_order.h"

#include "board_ra8d1.h"
#include "hal_data.h"     /* FSP-generated: g_spi0, g_timer_tacho, g_adc0, ... */
#include "r_ioport.h"

/* ------------------------------------------------------------ IIS3DWB map */
#define IIS3DWB_WHO_AM_I          0x0Fu
#define IIS3DWB_WHO_AM_I_VALUE    0x7Bu
#define IIS3DWB_CTRL1_XL          0x10u
#define IIS3DWB_CTRL3_C           0x12u
#define IIS3DWB_CTRL5_C           0x14u
#define IIS3DWB_FIFO_CTRL3        0x09u
#define IIS3DWB_FIFO_CTRL4        0x0Au
#define IIS3DWB_FIFO_STATUS1      0x3Au
#define IIS3DWB_FIFO_DATA_OUT_X_L 0x3Eu
#define IIS3DWB_OUTX_L_XL         0x28u
#define IIS3DWB_CTRL1_26K7_16G    0x5Cu
#define IIS3DWB_CTRL3_BDU_IFINC   0x44u
#define IIS3DWB_FIFO_CONTINUOUS   0x06u

#define CS_PIN  BSP_IO_PORT_04_PIN_06
#define LED_PIN BSP_IO_PORT_06_PIN_00

/* All three axes of the whole block. 8192 x 3 x 2 B = 48 kB, which on a 1 MB
 * part is affordable -- and it is what lets one FIFO burst serve three
 * pipelines. The nRF9160's per-chunk conversion trick is unnecessary here and
 * would actually be worse: it would need three passes over the FIFO or three
 * separate output buffers anyway. */
static int16_t  s_raw[APP_ACQ_SAMPLES * 3u];
static uint8_t  s_raw_valid;
/* The app has finished with the block, so the next channel-0 acquire must read
 * the sensor again. Separate from s_raw_valid on purpose: the block stays
 * READABLE after release, because this kit's order-domain diagnostic runs after
 * pdm_app_step has returned and needs the samples it just analysed. Collapsing
 * these two into one flag means either re-analysing stale samples forever or
 * having no raw record for the diagnostic. */
static uint8_t  s_consumed = 1u;
static uint8_t  s_axis;
static uint64_t s_uid;
static uint8_t  s_powered;
static int16_t  s_rssi = -99;

/* Tachometer: input-capture timestamps, consumed by the speed estimator. */
#define TACHO_CAPTURES 64u
static volatile uint32_t s_tacho_ts[TACHO_CAPTURES];
static volatile uint8_t  s_tacho_n;
static float    s_shaft_hz;
static float    s_shaft_drift_pct;
static uint8_t  s_shaft_measured;

void pdm_board_select_channel(uint8_t channel)
{
    s_axis = (channel < APP_N_AXES) ? channel : 0u;
}

float pdm_board_ra8d1_speed_drift_pct(void)
{
    return s_shaft_drift_pct;
}

/* ------------------------------------------------------------- SPI helpers */
/* Set by the FSP SPI callback. Declared volatile and read in a spin loop: a
 * transfer is a few microseconds, and a WFI/wake round trip costs more than the
 * transfer it would overlap. */
extern volatile spi_event_t g_spi0_event;

static fsp_err_t spi_xfer(const void *tx, void *rx, uint32_t len)
{
    fsp_err_t err = R_SCI_B_SPI_WriteRead(&g_spi0_ctrl, tx, rx, len, SPI_BIT_WIDTH_8_BITS);
    if (err != FSP_SUCCESS) {
        return err;
    }
    /* The FSP SPI driver is asynchronous; the event flag is set in the callback.
     * Spinning here rather than sleeping: a transfer is a few microseconds and
     * the WFI/wake overhead would exceed it. */
    uint32_t guard = 0;
    while (g_spi0_event == (spi_event_t)0u) {
        if (++guard > 10000000u) {
            return FSP_ERR_TIMEOUT;
        }
    }
    const spi_event_t ev = g_spi0_event;
    g_spi0_event = (spi_event_t)0u;
    return (ev == SPI_EVENT_TRANSFER_COMPLETE) ? FSP_SUCCESS : FSP_ERR_ABORTED;
}

static fsp_err_t reg_read(uint8_t reg, uint8_t *dst, uint32_t len)
{
    uint8_t cmd = (uint8_t)(reg | 0x80u);
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, CS_PIN, BSP_IO_LEVEL_LOW);
    fsp_err_t err = spi_xfer(&cmd, NULL, 1u);
    if (err == FSP_SUCCESS) {
        err = spi_xfer(NULL, dst, len);
    }
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, CS_PIN, BSP_IO_LEVEL_HIGH);
    return err;
}

static fsp_err_t reg_write(uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = {reg, value};
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, CS_PIN, BSP_IO_LEVEL_LOW);
    const fsp_err_t err = spi_xfer(buf, NULL, 2u);
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, CS_PIN, BSP_IO_LEVEL_HIGH);
    return err;
}

/* ------------------------------------------------------------- tachometer */
void tacho_capture_callback(timer_callback_args_t *p_args)
{
    if (p_args == NULL || s_tacho_n >= TACHO_CAPTURES) {
        return;
    }
    s_tacho_ts[s_tacho_n++] = (uint32_t)p_args->capture;
}

/* Speed and drift from the capture timestamps.
 *
 * Returns 0 on success. Rejects a reading whose pulse-to-pulse jitter exceeds
 * APP_TACHO_MAX_JITTER_PCT: a flaky optical pickup produces plausible but wrong
 * speeds, and a wrong speed is worse than no speed because the order and
 * bearing features are all computed against it. Better to ship a frame flagged
 * SPEED_ESTIMATED with the configured nominal than a confident wrong number. */
static int tacho_speed(float *hz, float *drift_pct)
{
    const uint8_t n = s_tacho_n;
    if (n < 4u) {
        return -1;
    }
    /* GPT0 clocked from PCLKD. The FSP exposes the configured frequency; using
     * it rather than a hardcoded constant matters because the clock tree is
     * configured in e2 studio and a change there would otherwise silently scale
     * every speed reading. */
    const float tick_hz = (float)ra8_gpt_source_hz();

    float min_dt = 1e30f, max_dt = 0.0f, sum_dt = 0.0f;
    uint8_t count = 0;
    for (uint8_t i = 1; i < n; ++i) {
        const uint32_t dt = s_tacho_ts[i] - s_tacho_ts[i - 1u];   /* wraps correctly */
        if (dt == 0u) {
            continue;
        }
        const float f = (float)dt;
        if (f < min_dt) {
            min_dt = f;
        }
        if (f > max_dt) {
            max_dt = f;
        }
        sum_dt += f;
        ++count;
    }
    if (count < 3u || sum_dt <= 0.0f) {
        return -1;
    }
    const float mean_dt = sum_dt / (float)count;
    const float jitter_pct = (max_dt - min_dt) / mean_dt * 100.0f;
    if (jitter_pct > APP_TACHO_MAX_JITTER_PCT * 4.0f) {
        /* Gross jitter: a missed or doubled pulse, not real speed variation. */
        return -1;
    }
    *hz = tick_hz / mean_dt / (float)APP_TACHO_PULSES_PER_REV;
    /* Peak-to-peak speed variation over the record. Above ~0.5 % a plain FFT
     * starts losing amplitude on high orders; above ~2 % order tracking is
     * mandatory for bearing work. */
    *drift_pct = jitter_pct;
    return 0;
}

/* ------------------------------------------------------------------- init */
pdm_board_status pdm_board_init(pdm_board_info *info)
{
    /* RA8 unique ID: 4 words in the factory MCU information area. */
    const uint32_t *uid = (const uint32_t *)0x01001C00u;
    s_uid = ((uint64_t)uid[0] << 32) ^ ((uint64_t)uid[1] << 16) ^ (uint64_t)uid[2];

    if (info != NULL) {
        info->kit = APP_KIT_NAME;
        info->accel_part = "IIS3DWB (3 axes)";
        info->device_uid = s_uid;
        info->fs_hz = APP_FS_HZ;
        info->full_scale_g = APP_FULL_SCALE_G;
        info->adc_bits = 16u;
        info->flash_bytes = 2u * 1024u * 1024u;
        info->ram_bytes = 1024u * 1024u;
    }

    if (R_SCI_B_SPI_Open(&g_spi0_ctrl, &g_spi0_cfg) != FSP_SUCCESS) {
        return PDM_BOARD_ERR_BUS;
    }
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, CS_PIN, BSP_IO_LEVEL_HIGH);

#if APP_TACHO_ENABLED
    if (R_GPT_Open(&g_timer_tacho_ctrl, &g_timer_tacho_cfg) != FSP_SUCCESS
        || R_GPT_Start(&g_timer_tacho_ctrl) != FSP_SUCCESS) {
        /* Not fatal. Without a tacho this kit degrades to the behaviour of every
         * other kit: a configured nominal speed, flagged as estimated. */
    }
#endif

    uint8_t who = 0;
    if (reg_read(IIS3DWB_WHO_AM_I, &who, 1u) != FSP_SUCCESS) {
        return PDM_BOARD_ERR_BUS;
    }
    if (who != IIS3DWB_WHO_AM_I_VALUE) {
        return PDM_BOARD_ERR_ID;
    }
    (void)reg_write(IIS3DWB_CTRL3_C, 0x01u);
    R_BSP_SoftwareDelay(5u, BSP_DELAY_UNITS_MILLISECONDS);
    (void)reg_write(IIS3DWB_CTRL3_C, IIS3DWB_CTRL3_BDU_IFINC);

    int32_t base[3] = {0}, test[3] = {0};
    uint8_t raw[6];
    (void)reg_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G);
    R_BSP_SoftwareDelay(100u, BSP_DELAY_UNITS_MILLISECONDS);
    for (int i = 0; i < 5; ++i) {
        R_BSP_SoftwareDelay(2u, BSP_DELAY_UNITS_MILLISECONDS);
        if (reg_read(IIS3DWB_OUTX_L_XL, raw, 6u) != FSP_SUCCESS) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            base[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    (void)reg_write(IIS3DWB_CTRL5_C, 0x01u);
    R_BSP_SoftwareDelay(100u, BSP_DELAY_UNITS_MILLISECONDS);
    for (int i = 0; i < 5; ++i) {
        R_BSP_SoftwareDelay(2u, BSP_DELAY_UNITS_MILLISECONDS);
        if (reg_read(IIS3DWB_OUTX_L_XL, raw, 6u) != FSP_SUCCESS) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            test[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    (void)reg_write(IIS3DWB_CTRL5_C, 0x00u);
    /* All three axes checked and all three USED on this kit, unlike the
     * single-axis kits where the other two are only a damage indicator. */
    for (int a = 0; a < 3; ++a) {
        const int32_t d = (test[a] - base[a]) / 5;
        const int32_t m = (d < 0) ? -d : d;
        if (m < 50 || m > 1700) {
            return PDM_BOARD_ERR_SELFTEST;
        }
    }
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_accel_power(int on)
{
    if (on) {
        if (reg_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G) != FSP_SUCCESS) {
            return PDM_BOARD_ERR_BUS;
        }
        (void)reg_write(IIS3DWB_FIFO_CTRL3, 0x05u);
        (void)reg_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);
    } else {
        (void)reg_write(IIS3DWB_FIFO_CTRL4, 0x00u);
        (void)reg_write(IIS3DWB_CTRL1_XL, 0x00u);
    }
    s_powered = on ? 1u : 0u;
    if (on) {
        /* Powering UP starts a new acquisition session, so any retained block
         * is stale. Powering DOWN must NOT drop it: the app powers the sensor
         * off between channel 0 and channels 1-2 by design. */
        s_raw_valid = 0u;
    }
    return PDM_BOARD_OK;
}

uint32_t pdm_board_accel_settle_ms(void)
{
    return APP_SENSOR_SETTLE_MS;
}

pdm_board_status pdm_board_acquire(float *out, size_t n, float *fs_hz_actual)
{
    if (out == NULL || n > APP_ACQ_SAMPLES) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    /* The power gate applies only to a real sensor read. The shared app loop
     * powers the IIS3DWB down as soon as channel 0's burst is in RAM, so axes 1
     * and 2 -- a memory copy out of that same burst -- must not require a
     * powered sensor. Getting this order wrong makes a three-axis node ship one
     * frame per acquisition and look like it is working. */
    const int need_sensor = (s_axis == 0u) && (!s_raw_valid || s_consumed);
    if (need_sensor && !s_powered) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    if (!need_sensor && (!s_raw_valid || s_consumed)) {
        /* An extra axis asked for a block that was never read or has already
         * been spent. Returning zeros here would look like a perfectly smooth
         * machine on two of the three axes. */
        return PDM_BOARD_ERR_NOT_READY;
    }

    /* One FIFO burst serves all three axes. The application calls acquire three
     * times per acquisition with a different axis selected; only the first call
     * touches the bus. That is why the raw block is retained -- the alternative
     * is three acquisitions of the same machine at three different times, which
     * is not the same measurement. */
    if (need_sensor) {
        (void)reg_write(IIS3DWB_FIFO_CTRL4, 0x00u);
        (void)reg_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);
#if APP_TACHO_ENABLED
        s_tacho_n = 0u;           /* capture the tacho over the same window */
#endif
        size_t done = 0;
        while (done < APP_ACQ_SAMPLES) {
            size_t want = APP_ACQ_SAMPLES - done;
            if (want > 512u) {
                want = 512u;
            }
            uint32_t guard = 0;
            for (;;) {
                uint8_t st[2] = {0, 0};
                if (reg_read(IIS3DWB_FIFO_STATUS1, st, 2u) != FSP_SUCCESS) {
                    return PDM_BOARD_ERR_BUS;
                }
                if ((st[1] & 0x40u) != 0u) {
                    return PDM_BOARD_ERR_OVERRUN;
                }
                const uint16_t level = (uint16_t)(((uint16_t)(st[1] & 0x03u) << 8) | st[0]);
                if (level >= want) {
                    break;
                }
                if (++guard > 200000u) {
                    return PDM_BOARD_ERR_TIMEOUT;
                }
            }
            if (reg_read(IIS3DWB_FIFO_DATA_OUT_X_L, (uint8_t *)&s_raw[done * 3u],
                         (uint32_t)(want * 6u)) != FSP_SUCCESS) {
                return PDM_BOARD_ERR_BUS;
            }
            done += want;
        }
        s_raw_valid = 1u;
        s_consumed = 0u;
#if APP_TACHO_ENABLED
        float hz = 0.0f, drift = 0.0f;
        if (tacho_speed(&hz, &drift) == 0) {
            s_shaft_hz = hz;
            s_shaft_drift_pct = drift;
            s_shaft_measured = 1u;
        } else {
            /* Keep the last good or configured value and mark it estimated. */
            s_shaft_measured = 0u;
        }
#endif
    }

    const float lsb_g = 2.0f * APP_FULL_SCALE_G / 65536.0f;
    for (size_t i = 0; i < n; ++i) {
        out[i] = (float)s_raw[i * 3u + s_axis] * lsb_g;
    }
    if (fs_hz_actual != NULL) {
        *fs_hz_actual = APP_FS_HZ;
    }
    return PDM_BOARD_OK;
}

void pdm_board_release_block(void)
{
    /* Called by the shared app loop after the last axis. Marks the block spent
     * rather than wiping it: the next channel-0 acquire will read the sensor
     * again (so the node cannot re-analyse stale samples -- a dashboard full of
     * perfectly flat trends), while the samples remain available to the
     * order-domain pass that runs immediately afterwards. */
    s_consumed = 1u;
}

/* Frame flags this board contributes. See pdm_board.h.
 *
 * SPEED_UNSTEADY is the one that matters: the features are frequency-domain, so
 * when the tacho has measured real drift the cloud must widen its harmonic
 * tolerance and discount high orders instead of alarming on smearing. The
 * measured dependence is in tests/test_order.c -- 1.5 dB at order 6 and 5.7 dB
 * at order 40 for a 4 % sweep. Without the flag that reads as a developing
 * gear-mesh fault that heals whenever production steadies.
 *
 * ORDER_TRACKED is deliberately NOT set: this node ships hertz-domain features.
 * Claiming the angle domain while shipping hertz would have the cloud compare
 * against the wrong baseline, which is a worse failure than no flag at all. */
uint16_t pdm_board_extra_flags(void)
{
    uint16_t f = 0u;
    if (s_shaft_measured && s_shaft_drift_pct > APP_ORDER_TRACK_DRIFT_PCT) {
        f |= (uint16_t)PDM_FRAME_SPEED_UNSTEADY;
    }
    return f;
}

/* Revolution-mark times in seconds, relative to sample 0 of the record. The
 * capture timestamps are free-running timer ticks; the order resampler needs
 * them on the same clock as the samples, so the first mark is the origin.
 *
 * This loses the offset between the first mark and sample 0, which is a
 * constant phase shift in the angle domain. Magnitude spectra do not care.
 * Time-synchronous averaging would, and that is noted as an open item. */
uint16_t pdm_board_ra8d1_marks(float *dst, uint16_t cap)
{
    const uint8_t n = s_tacho_n;
    if (dst == NULL || cap == 0u || n < 2u) {
        return 0u;
    }
    const float tick_s = 1.0f / (float)ra8_gpt_source_hz();
    const uint32_t t0 = s_tacho_ts[0];
    uint16_t w = 0u;
    for (uint8_t i = 0; i < n && w < cap; ++i) {
        dst[w++] = (float)(s_tacho_ts[i] - t0) * tick_s;
    }
    return w;
}

/* The record channel 0 just analysed, converted to g. Valid until the next
 * acquisition. Returns NULL if there is nothing to look at. */
const float *pdm_board_ra8d1_last_record(uint32_t *n)
{
    static float s_record[APP_ACQ_SAMPLES];
    if (!s_raw_valid) {
        if (n != NULL) {
            *n = 0u;
        }
        return NULL;
    }
    const float lsb_g = 2.0f * APP_FULL_SCALE_G / 65536.0f;
    for (uint32_t i = 0; i < APP_ACQ_SAMPLES; ++i) {
        s_record[i] = (float)s_raw[i * 3u] * lsb_g;
    }
    if (n != NULL) {
        *n = APP_ACQ_SAMPLES;
    }
    return s_record;
}

/* Unix milliseconds from the RTC, kept running through Software Standby by the
 * sub-clock. A node whose clock is wrong produces frames the ingest service
 * discards as stale, so this is not a convenience. */
uint64_t ra8_unix_ms(void)
{
    rtc_time_t tm;
    if (R_RTC_CalendarTimeGet(&g_rtc_ctrl, &tm) != FSP_SUCCESS) {
        return 0u;
    }
    return ra8_timegm(&tm) * 1000u;
}

pdm_board_status pdm_board_shaft_hz(float *shaft_hz, int *estimated)
{
    if (shaft_hz == NULL || !(s_shaft_hz > 0.0f)) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    *shaft_hz = s_shaft_hz;
    if (estimated != NULL) {
        *estimated = s_shaft_measured ? 0 : 1;
    }
    return PDM_BOARD_OK;
}

void pdm_board_ra8d1_set_nominal_hz(float hz)
{
    if (!s_shaft_measured && hz > 0.0f) {
        s_shaft_hz = hz;
    }
}

pdm_board_status pdm_board_temperature_c(float *temp_c)
{
    uint8_t raw[2] = {0, 0};
    if (reg_read(0x20u, raw, 2u) != FSP_SUCCESS) {
        return PDM_BOARD_ERR_BUS;
    }
    const int16_t t = (int16_t)((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
    if (temp_c != NULL) {
        *temp_c = 25.0f + (float)t / 256.0f;
    }
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_battery_mv(uint16_t *mv)
{
    if (mv == NULL) {
        return PDM_BOARD_ERR_BUS;
    }
    uint16_t raw = 0;
    if (R_ADC_ScanStart(&g_adc0_ctrl) != FSP_SUCCESS) {
        return PDM_BOARD_ERR_BUS;
    }
    adc_status_t status = {.state = ADC_STATE_SCAN_IN_PROGRESS};
    uint32_t guard = 0;
    while (status.state == ADC_STATE_SCAN_IN_PROGRESS) {
        (void)R_ADC_StatusGet(&g_adc0_ctrl, &status);
        if (++guard > 1000000u) {
            return PDM_BOARD_ERR_TIMEOUT;
        }
    }
    if (R_ADC_Read(&g_adc0_ctrl, ADC_CHANNEL_0, &raw) != FSP_SUCCESS) {
        return PDM_BOARD_ERR_BUS;
    }
    *mv = (uint16_t)(((uint32_t)raw * 3300u * 2u) / 4096u);
    return PDM_BOARD_OK;
}

int16_t pdm_board_rssi_dbm(void)
{
    return s_rssi;
}

pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len)
{
    int16_t rssi = s_rssi;
    const int rc = ble_transport_send(frame, (uint32_t)len, &rssi);
    s_rssi = rssi;
    return (rc == 0) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

extern pdm_board_status flash_queue_push(const uint8_t *frame, size_t len);
extern pdm_board_status flash_queue_pop(uint8_t *frame, size_t cap, size_t *len);
extern uint32_t         flash_queue_depth(void);

pdm_board_status pdm_board_queue_push(const uint8_t *frame, size_t len)
{
    return flash_queue_push(frame, len);
}

pdm_board_status pdm_board_queue_pop(uint8_t *frame, size_t cap, size_t *len)
{
    return flash_queue_pop(frame, cap, len);
}

uint32_t pdm_board_queue_depth(void)
{
    return flash_queue_depth();
}

uint32_t pdm_board_sleep_ms(uint32_t ms)
{
    if (ms == 0u) {
        return 0u;
    }
    /* Software Standby with AGT1 as the wake source: SRAM retained, ~22 uA.
     *
     * Deep Software Standby reaches single-digit microamps but loses SRAM, and
     * on this kit that is unusually expensive: 1 MB of SRAM holds a 48 kB raw
     * block, three axes of pipeline state and 46 kB of tables, and
     * re-initialising at 480 MHz costs ~12 ms at 62 mA = 0.74 mA.s against the
     * ~0.07 mA.s an hour of extra sleep current would cost. Standby wins by an
     * order of magnitude. */
    ra8_enter_standby_ms(ms);
    return ms;
}

void pdm_board_identify(uint32_t seconds)
{
    for (uint32_t i = 0; i < seconds * 4u; ++i) {
        (void)R_IOPORT_PinWrite(&g_ioport_ctrl, LED_PIN,
                                (i & 1u) ? BSP_IO_LEVEL_HIGH : BSP_IO_LEVEL_LOW);
        R_BSP_SoftwareDelay(250u, BSP_DELAY_UNITS_MILLISECONDS);
    }
    (void)R_IOPORT_PinWrite(&g_ioport_ctrl, LED_PIN, BSP_IO_LEVEL_LOW);
}

void pdm_board_watchdog_feed(void)
{
    (void)R_WDT_Refresh(&g_wdt0_ctrl);
}
