/* board_alif_e7.c -- pdm_board.h for the Alif Ensemble E7 DevKit + IIS3DWB.
 *
 * THIS CODE HAS NOT RUN ON HARDWARE. Written against the Alif Ensemble SDK
 * (CMSIS-driver style) and the IIS3DWB datasheet.
 *
 * What is distinctive here is the core split, not the drivers.
 *
 * The E7 has two Cortex-M55s with very different characters:
 *
 *   HE core, 160 MHz, ~4.5 mA, Helium MVE, 128-MAC Ethos-U55
 *   HP core, 400 MHz, ~22 mA,  Helium MVE, 256-MAC Ethos-U55
 *
 * Everything in the normal cycle runs on the HE core: acquisition, the whole
 * DSP pipeline, the feature detector. Helium halves the FFT time against a
 * scalar M33, so the pipeline is ~2.4 ms at 9 mA -- 0.02 mA.s, half a percent
 * of the cycle.
 *
 * The HP core and the large NPU stay powered down and are woken ONLY when the
 * waveform model is enabled and the feature path has escalated. That is the
 * right division: a 22 mA core that is awake 0.3 % of the time costs less than
 * a 4.5 mA core that is awake 100 % of it, and the expensive silicon earns its
 * keep by doing something the cheap silicon cannot rather than by doing the
 * same thing faster.
 */

#include "pdm_board.h"

#include <string.h>

#include "app_config.h"
#include "pdm_pipeline.h"

/* Alif Ensemble SDK. */
#include "Driver_GPIO.h"
#include "Driver_SPI.h"
#include "Driver_ADC.h"
#include "RTE_Components.h"
#include "board.h"
#include "pm.h"              /* power management: subsystem on/off */
#include "se_services_port.h" /* secure enclave: clocks, power, unique ID */

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

extern ARM_DRIVER_SPI  Driver_SPI1;
extern ARM_DRIVER_GPIO Driver_GPIO7;
extern ARM_DRIVER_ADC  Driver_ADC12;
static ARM_DRIVER_SPI  *s_spi = &Driver_SPI1;
static ARM_DRIVER_GPIO *s_gpio = &Driver_GPIO7;

/* Per-chunk conversion, as on the nRF9160: no raw staging block. Adopted here
 * not because RAM is scarce (5.5 MB) but because it is simply better -- the
 * conversion happens either way, and the intermediate buffer only exists to be
 * read once. */
#define CHUNK_SAMPLES 512u
static int16_t  s_chunk[CHUNK_SAMPLES * 3u];
static uint64_t s_uid;
static uint8_t  s_powered;
static float    s_shaft_hz;
static uint8_t  s_shaft_measured;
static int16_t  s_rssi = -99;

void pdm_board_alif_set_shaft_hz(float hz, int measured)
{
    s_shaft_hz = hz;
    s_shaft_measured = measured ? 1u : 0u;
}

/* -------------------------------------------------------------- SPI I/O */
#define CS_PIN 7u

static int32_t cs(uint32_t level)
{
    return s_gpio->SetValue(CS_PIN, level);
}

static int32_t reg_read(uint8_t reg, uint8_t *dst, uint32_t len)
{
    uint8_t cmd = (uint8_t)(reg | 0x80u);
    (void)cs(GPIO_PIN_OUTPUT_STATE_LOW);
    int32_t st = s_spi->Send(&cmd, 1u);
    if (st == ARM_DRIVER_OK) {
        st = s_spi->Receive(dst, len);
    }
    (void)cs(GPIO_PIN_OUTPUT_STATE_HIGH);
    return st;
}

static int32_t reg_write(uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = {reg, value};
    (void)cs(GPIO_PIN_OUTPUT_STATE_LOW);
    const int32_t st = s_spi->Send(buf, 2u);
    (void)cs(GPIO_PIN_OUTPUT_STATE_HIGH);
    return st;
}

static void delay_ms(uint32_t ms)
{
    /* SysTick-based. The Alif SDK's PMU cycle counter would be more accurate
     * but is reset by the power-management transitions this driver makes. */
    extern void sys_busy_loop_us(uint32_t us);
    while (ms--) {
        sys_busy_loop_us(1000u);
    }
}

/* ------------------------------------------------------------------- init */
pdm_board_status pdm_board_init(pdm_board_info *info)
{
    /* The secure enclave owns the unique ID and the clock/power services. An
     * application that pokes clock registers directly works until the enclave
     * decides otherwise, which on this part is a real class of intermittent
     * bug. */
    uint8_t uid[8] = {0};
    (void)se_services_get_unique_id(uid, sizeof(uid));
    s_uid = 0;
    for (int i = 0; i < 8; ++i) {
        s_uid = (s_uid << 8) | uid[i];
    }

    if (info != NULL) {
        info->kit = APP_KIT_NAME;
        info->accel_part = "IIS3DWB";
        info->device_uid = s_uid;
        info->fs_hz = APP_FS_HZ;
        info->full_scale_g = APP_FULL_SCALE_G;
        info->adc_bits = 16u;
        info->flash_bytes = 5u * 1024u * 1024u;   /* MRAM */
        info->ram_bytes = 5u * 1024u * 1024u + 512u * 1024u;
    }

    if (s_gpio->Initialize(CS_PIN, NULL) != ARM_DRIVER_OK
        || s_gpio->PowerControl(CS_PIN, ARM_POWER_FULL) != ARM_DRIVER_OK
        || s_gpio->SetDirection(CS_PIN, GPIO_PIN_DIRECTION_OUTPUT) != ARM_DRIVER_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    (void)cs(GPIO_PIN_OUTPUT_STATE_HIGH);

    if (s_spi->Initialize(NULL) != ARM_DRIVER_OK
        || s_spi->PowerControl(ARM_POWER_FULL) != ARM_DRIVER_OK
        || s_spi->Control(ARM_SPI_MODE_MASTER | ARM_SPI_CPOL0_CPHA0
                          | ARM_SPI_DATA_BITS(8), 8000000u) != ARM_DRIVER_OK) {
        return PDM_BOARD_ERR_BUS;
    }

    uint8_t who = 0;
    if (reg_read(IIS3DWB_WHO_AM_I, &who, 1u) != ARM_DRIVER_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    if (who != IIS3DWB_WHO_AM_I_VALUE) {
        return PDM_BOARD_ERR_ID;
    }
    (void)reg_write(IIS3DWB_CTRL3_C, 0x01u);
    delay_ms(5);
    (void)reg_write(IIS3DWB_CTRL3_C, IIS3DWB_CTRL3_BDU_IFINC);

    /* Self-test, same window as the other IIS3DWB kits. */
    int32_t base[3] = {0}, test[3] = {0};
    uint8_t raw[6];
    (void)reg_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G);
    delay_ms(100);
    for (int i = 0; i < 5; ++i) {
        delay_ms(2);
        if (reg_read(IIS3DWB_OUTX_L_XL, raw, 6u) != ARM_DRIVER_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            base[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    (void)reg_write(IIS3DWB_CTRL5_C, 0x01u);
    delay_ms(100);
    for (int i = 0; i < 5; ++i) {
        delay_ms(2);
        if (reg_read(IIS3DWB_OUTX_L_XL, raw, 6u) != ARM_DRIVER_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            test[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    (void)reg_write(IIS3DWB_CTRL5_C, 0x00u);
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
        if (reg_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G) != ARM_DRIVER_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        (void)reg_write(IIS3DWB_FIFO_CTRL3, 0x05u);
        (void)reg_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);
    } else {
        (void)reg_write(IIS3DWB_FIFO_CTRL4, 0x00u);
        (void)reg_write(IIS3DWB_CTRL1_XL, 0x00u);
    }
    s_powered = on ? 1u : 0u;
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
    if (!s_powered) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    (void)reg_write(IIS3DWB_FIFO_CTRL4, 0x00u);
    (void)reg_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);

    const float lsb_g = 2.0f * APP_FULL_SCALE_G / 65536.0f;
    size_t done = 0;
    while (done < n) {
        size_t want = n - done;
        if (want > CHUNK_SAMPLES) {
            want = CHUNK_SAMPLES;
        }
        uint32_t guard = 0;
        for (;;) {
            uint8_t st[2] = {0, 0};
            if (reg_read(IIS3DWB_FIFO_STATUS1, st, 2u) != ARM_DRIVER_OK) {
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
        if (reg_read(IIS3DWB_FIFO_DATA_OUT_X_L, (uint8_t *)s_chunk,
                     (uint32_t)(want * 6u)) != ARM_DRIVER_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (size_t i = 0; i < want; ++i) {
            out[done + i] = (float)s_chunk[i * 3u + 2u] * lsb_g;
        }
        done += want;
    }
    if (fs_hz_actual != NULL) {
        *fs_hz_actual = APP_FS_HZ;
    }
    return PDM_BOARD_OK;
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

pdm_board_status pdm_board_temperature_c(float *temp_c)
{
    uint8_t raw[2] = {0, 0};
    if (reg_read(0x20u, raw, 2u) != ARM_DRIVER_OK) {
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
    /* ADC12 channel 0 through a 2:1 divider, 1.8 V reference. */
    uint32_t raw = 0;
    extern int32_t alif_adc_read_blocking(uint8_t channel, uint32_t *value);
    if (alif_adc_read_blocking(0u, &raw) != 0) {
        return PDM_BOARD_ERR_BUS;
    }
    *mv = (uint16_t)((raw * 1800u * 2u) / 4096u);
    return PDM_BOARD_OK;
}

int16_t pdm_board_rssi_dbm(void)
{
    return s_rssi;
}

extern int ble_transport_send(const uint8_t *data, size_t len, int16_t *rssi_out);

pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len)
{
    int16_t rssi = s_rssi;
    const int rc = ble_transport_send(frame, len, &rssi);
    s_rssi = rssi;
    return (rc == 0) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

extern pdm_board_status mram_queue_push(const uint8_t *frame, size_t len);
extern pdm_board_status mram_queue_pop(uint8_t *frame, size_t cap, size_t *len);
extern uint32_t         mram_queue_depth(void);

pdm_board_status pdm_board_queue_push(const uint8_t *frame, size_t len)
{
    return mram_queue_push(frame, len);
}

pdm_board_status pdm_board_queue_pop(uint8_t *frame, size_t cap, size_t *len)
{
    return mram_queue_pop(frame, cap, len);
}

uint32_t pdm_board_queue_depth(void)
{
    return mram_queue_depth();
}

uint32_t pdm_board_sleep_ms(uint32_t ms)
{
    if (ms == 0u) {
        return 0u;
    }
    /* STOP mode with the HE core retained and the HP core + large NPU already
     * powered down (they are only brought up for the waveform model).
     *
     * The E7's power domains are the reason this part is interesting: the HP
     * core at 400 MHz draws ~22 mA, and leaving it powered between acquisitions
     * would cost 79 mA.s per hour -- 17x the entire rest of the cycle. The
     * expensive silicon has to be genuinely off, not idle, and on this part it
     * can be. */
    extern void pm_core_enter_deep_sleep_request_subsys_off(uint32_t ms);
    pm_core_enter_deep_sleep_request_subsys_off(ms);
    return ms;
}

void pdm_board_identify(uint32_t seconds)
{
    for (uint32_t i = 0; i < seconds * 4u; ++i) {
        (void)s_gpio->SetValue(BOARD_LED1_PIN_NO, (i & 1u)
            ? GPIO_PIN_OUTPUT_STATE_HIGH : GPIO_PIN_OUTPUT_STATE_LOW);
        delay_ms(250);
    }
}

void pdm_board_watchdog_feed(void)
{
    extern void alif_wdt_feed(void);
    alif_wdt_feed();
}
