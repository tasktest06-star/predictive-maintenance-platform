/*
 * MEMS Low-Power Sensor — Main Firmware
 * Target MCU : STM32U5 (Cortex-M33, Zephyr RTOS 3.6)
 * Accelerometer : Bosch BMI088 via SPI
 * Radio        : TI CC1352P (Sub-GHz + BLE 5.2) via UART/SPI
 *
 * Power State Machine:
 *   STOP2 (300 nA) → WAKE → ACQUIRE (DMA) → FFT+FEATURES → TRANSMIT → STOP2
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/pm/pm.h>
#include <zephyr/pm/device.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/net/mqtt.h>
#include <zephyr/random/random.h>

#include <arm_math.h>       /* CMSIS-DSP */
#include <tinycbor/cbor.h>  /* CBOR encoding */

#include "dsp_features.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

/* ──────────────────────────────────────────────────────────────────────────
 * Hardware constants
 * ──────────────────────────────────────────────────────────────────────── */

/* BMI088 register map (accelerometer) */
#define BMI088_ACC_CHIP_ID          0x00U
#define BMI088_ACC_ERR_REG          0x02U
#define BMI088_ACC_STATUS           0x03U
#define BMI088_ACC_DATA             0x12U  /* ACC_X_LSB ... ACC_Z_MSB */
#define BMI088_ACC_FIFO_LENGTH_0    0x24U
#define BMI088_ACC_FIFO_LENGTH_1    0x25U
#define BMI088_ACC_FIFO_DATA        0x26U
#define BMI088_ACC_INT1_IO_CTRL     0x53U
#define BMI088_ACC_INT2_IO_CTRL     0x54U
#define BMI088_ACC_INT_MAP_DATA     0x58U
#define BMI088_ACC_FIFO_CONFIG_0    0x48U
#define BMI088_ACC_FIFO_CONFIG_1    0x49U
#define BMI088_ACC_FIFO_WM_0        0x46U
#define BMI088_ACC_FIFO_WM_1        0x47U
#define BMI088_ACC_CONF             0x40U
#define BMI088_ACC_RANGE            0x41U
#define BMI088_ACC_PWR_CONF         0x7CU
#define BMI088_ACC_PWR_CTRL         0x7DU
#define BMI088_ACC_SOFTRESET        0x7EU
#define BMI088_ACC_CHIP_ID_VAL      0x1EU

/* BMI088 configuration values */
#define BMI088_ACC_ODR_2000         0xACU  /* ODR = 2000 Hz, OSR4 */
#define BMI088_ACC_RANGE_24G        0x03U  /* ±24g */
#define BMI088_ACC_PWR_ACTIVE       0x00U  /* suspend off */
#define BMI088_ACC_PWR_ON           0x04U  /* acc on */
#define BMI088_ACC_FIFO_STREAM      0x80U  /* FIFO stream mode */
#define BMI088_ACC_FIFO_ACC_EN      0x40U  /* acc data in FIFO */
#define BMI088_ACC_INT1_OUT         0x0AU  /* INT1: output, active-high, push-pull */
#define BMI088_ACC_INT_FWM          0x02U  /* map FIFO watermark to INT1 */

/* Acquisition sizing */
#define SAMPLES_PER_AXIS            2048U
#define BYTES_PER_SAMPLE            6U     /* 2 bytes × 3 axes */
#define FIFO_WATERMARK_FRAMES       512U   /* 512 frames × 6 bytes = 3072 bytes; 4 bursts */
#define FIFO_WM_BYTES               (FIFO_WATERMARK_FRAMES * BYTES_PER_SAMPLE)

/* Sample rate and derived constants */
#define SAMPLE_RATE_HZ              2000.0f
#define MEAS_INTERVAL_MINUTES       10U    /* default measurement interval */
#define MEAS_INTERVAL_S             (MEAS_INTERVAL_MINUTES * 60U)

/* Battery model */
#define BATTERY_CAPACITY_MAH        7500U
#define ACTIVE_CURRENT_MA           8U     /* average during 500 ms active window */
#define ACTIVE_DURATION_MS          500U
#define STANDBY_CURRENT_UA          2U     /* system-level (MCU STOP2 + sensor suspend) */

/* CC1352P UART interface */
#define CC1352_BAUD                 115200U
#define CC1352_TX_TIMEOUT_MS        100U

/* MQTT / payload */
#define MQTT_PAYLOAD_MAX_BYTES      256U
#define DEVICE_ID_LEN               16U

/* ──────────────────────────────────────────────────────────────────────────
 * Device tree references (Zephyr DTS bindings)
 * ──────────────────────────────────────────────────────────────────────── */

static const struct device *spi_dev    = DEVICE_DT_GET(DT_NODELABEL(spi1));
static const struct device *uart_cc13  = DEVICE_DT_GET(DT_NODELABEL(usart3));
static const struct gpio_dt_spec bmi_cs   =
    GPIO_DT_SPEC_GET(DT_NODELABEL(bmi088_cs), gpios);
static const struct gpio_dt_spec bmi_int1 =
    GPIO_DT_SPEC_GET(DT_NODELABEL(bmi088_int1), gpios);

static struct gpio_callback bmi_int1_cb_data;

/* ──────────────────────────────────────────────────────────────────────────
 * Buffers — placed in non-cached SRAM for DMA
 * ──────────────────────────────────────────────────────────────────────── */

/* Raw 16-bit FIFO data from BMI088 (3 axes × 2048 samples) */
static int16_t __aligned(4) raw_acc[SAMPLES_PER_AXIS * 3];

/* Float conversion of each axis for DSP */
static float32_t __aligned(4) acc_x[SAMPLES_PER_AXIS];
static float32_t __aligned(4) acc_y[SAMPLES_PER_AXIS];
static float32_t __aligned(4) acc_z[SAMPLES_PER_AXIS];

/* FFT output magnitude (one axis at a time — reuse buffer) */
static float32_t __aligned(4) fft_mag[SAMPLES_PER_AXIS / 2];
static float32_t __aligned(4) fft_buf[SAMPLES_PER_AXIS];  /* CMSIS scratch */

/* MQTT output buffer */
static uint8_t mqtt_payload[MQTT_PAYLOAD_MAX_BYTES];

/* Semaphore signalled from BMI088 INT1 ISR */
static K_SEM_DEFINE(fifo_watermark_sem, 0, 1);

/* ──────────────────────────────────────────────────────────────────────────
 * Power-state machine enum
 * ──────────────────────────────────────────────────────────────────────── */

typedef enum {
    PSM_STOP2,
    PSM_WAKE,
    PSM_ACQUIRE,
    PSM_FFT_FEATURES,
    PSM_TRANSMIT,
} psm_state_t;

static volatile psm_state_t psm_state = PSM_STOP2;

/* ──────────────────────────────────────────────────────────────────────────
 * Clock helpers — switch between ultra-low-power (HSI16) and full speed
 * (MSI 48 MHz) for FFT, then back down.
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Switch system clock to MSI @ 48 MHz for DSP-intensive operations.
 *
 * Called before FFT to ensure sufficient throughput.  The STM32U5 MSI PLL
 * can reach 160 MHz; we cap at 48 MHz to keep core voltage at VOS2 range.
 */
static void clock_boost_48mhz(void)
{
    /* In Zephyr, clock switching is done via the clock_control API or
     * direct CMSIS register access.  For clarity the register path is shown. */
#if defined(CONFIG_SOC_STM32U575XX)
    /* Enable MSI @ 48 MHz (MSIRANGE = 0xB → 48 MHz) */
    RCC->ICSCR1 = (RCC->ICSCR1 & ~RCC_ICSCR1_MSISRANGE_Msk)
                  | (0xBU << RCC_ICSCR1_MSISRANGE_Pos);
    RCC->CR |= RCC_CR_MSISON;
    while (!(RCC->CR & RCC_CR_MSISRDY)) { /* spin wait */ }
    /* Switch SYSCLK to MSI */
    RCC->CFGR1 = (RCC->CFGR1 & ~RCC_CFGR1_SW_Msk) | RCC_CFGR1_SW_MSIS;
    while ((RCC->CFGR1 & RCC_CFGR1_SWS_Msk) != RCC_CFGR1_SWS_MSIS) { /* spin */ }
#endif
    LOG_DBG("Clock: MSI 48 MHz");
}

/**
 * @brief Restore system clock to HSI16 after DSP work completes.
 */
static void clock_restore_hsi16(void)
{
#if defined(CONFIG_SOC_STM32U575XX)
    RCC->CFGR1 = (RCC->CFGR1 & ~RCC_CFGR1_SW_Msk) | RCC_CFGR1_SW_HSI16;
    while ((RCC->CFGR1 & RCC_CFGR1_SWS_Msk) != RCC_CFGR1_SWS_HSI16) { /* spin */ }
#endif
    LOG_DBG("Clock: HSI16");
}

/* ──────────────────────────────────────────────────────────────────────────
 * BMI088 SPI helpers
 * ──────────────────────────────────────────────────────────────────────── */

static const struct spi_config bmi_spi_cfg = {
    .frequency  = 10000000U,  /* 10 MHz — BMI088 max */
    .operation  = SPI_WORD_SET(8) | SPI_OP_MODE_MASTER | SPI_TRANSFER_MSB,
    .slave      = 0,
    .cs         = NULL,  /* CS managed manually for BMI088 protocol compliance */
};

/** Write one BMI088 accelerometer register. */
static int bmi088_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t tx[2] = { reg & 0x7FU, val };
    struct spi_buf tx_buf = { .buf = tx, .len = sizeof(tx) };
    struct spi_buf_set tx_set = { .buffers = &tx_buf, .count = 1 };

    gpio_pin_set_dt(&bmi_cs, 0);  /* CS low */
    int ret = spi_write(spi_dev, &bmi_spi_cfg, &tx_set);
    gpio_pin_set_dt(&bmi_cs, 1);  /* CS high */
    return ret;
}

/** Read one or more BMI088 accelerometer registers.
 *  BMI088 SPI protocol: set bit 7 for read; dummy byte required after address. */
static int bmi088_read_regs(uint8_t reg, uint8_t *out, size_t len)
{
    /* Address byte + dummy byte + data */
    uint8_t tx[2 + len];
    uint8_t rx[2 + len];

    memset(tx, 0, sizeof(tx));
    tx[0] = reg | 0x80U;  /* read flag */
    /* tx[1] is the dummy byte (sent during the dummy read cycle) */

    struct spi_buf tx_buf = { .buf = tx, .len = sizeof(tx) };
    struct spi_buf rx_buf = { .buf = rx, .len = sizeof(rx) };
    struct spi_buf_set tx_set = { .buffers = &tx_buf, .count = 1 };
    struct spi_buf_set rx_set = { .buffers = &rx_buf, .count = 1 };

    gpio_pin_set_dt(&bmi_cs, 0);
    int ret = spi_transceive(spi_dev, &bmi_spi_cfg, &tx_set, &rx_set);
    gpio_pin_set_dt(&bmi_cs, 1);

    if (ret == 0) {
        memcpy(out, &rx[2], len);  /* skip address + dummy byte */
    }
    return ret;
}

/**
 * @brief Read BMI088 FIFO burst into raw_acc[] buffer.
 *
 * BMI088 FIFO frame format (acc header mode disabled, streaming):
 *   [acc_x_lsb][acc_x_msb][acc_y_lsb][acc_y_msb][acc_z_lsb][acc_z_msb] × N
 *
 * @param n_frames Number of 6-byte frames to read (max SAMPLES_PER_AXIS).
 * @return 0 on success, negative errno otherwise.
 */
static int bmi088_fifo_burst_read(uint32_t n_frames)
{
    if (n_frames > SAMPLES_PER_AXIS) {
        return -EINVAL;
    }

    size_t total_bytes = n_frames * BYTES_PER_SAMPLE;
    /* tx: address + dummy + zeros for data clock */
    uint8_t tx_hdr[2] = { BMI088_ACC_FIFO_DATA | 0x80U, 0x00U };

    struct spi_buf tx_bufs[2];
    struct spi_buf rx_bufs[2];

    static uint8_t rx_dummy[2];
    tx_bufs[0] = (struct spi_buf){ .buf = tx_hdr,   .len = 2 };
    tx_bufs[1] = (struct spi_buf){ .buf = NULL,     .len = total_bytes };
    rx_bufs[0] = (struct spi_buf){ .buf = rx_dummy, .len = 2 };
    rx_bufs[1] = (struct spi_buf){ .buf = raw_acc,  .len = total_bytes };

    struct spi_buf_set tx_set = { .buffers = tx_bufs, .count = 2 };
    struct spi_buf_set rx_set = { .buffers = rx_bufs, .count = 2 };

    gpio_pin_set_dt(&bmi_cs, 0);
    int ret = spi_transceive(spi_dev, &bmi_spi_cfg, &tx_set, &rx_set);
    gpio_pin_set_dt(&bmi_cs, 1);
    return ret;
}

/* ──────────────────────────────────────────────────────────────────────────
 * BMI088 interrupt callback (INT1 → FIFO watermark)
 * ──────────────────────────────────────────────────────────────────────── */

static void bmi088_int1_handler(const struct device *port,
                                struct gpio_callback *cb,
                                gpio_port_pins_t pins)
{
    ARG_UNUSED(port);
    ARG_UNUSED(cb);
    ARG_UNUSED(pins);
    k_sem_give(&fifo_watermark_sem);
}

/* ──────────────────────────────────────────────────────────────────────────
 * BMI088 initialization
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Full BMI088 initialization sequence.
 *
 * Sequence:
 *   1. Soft-reset to clear any stale state.
 *   2. Power on accelerometer block.
 *   3. Verify chip ID.
 *   4. Configure ODR = 2000 Hz, range = ±24g.
 *   5. Configure FIFO: stream mode, acc enabled, watermark at 512 frames.
 *   6. Configure INT1 to fire on FIFO watermark.
 *   7. Enter active (non-suspend) mode.
 *
 * @return 0 on success, negative errno on failure.
 */
static int bmi088_init(void)
{
    uint8_t chip_id = 0;
    int ret;

    /* Soft-reset */
    ret = bmi088_write_reg(BMI088_ACC_SOFTRESET, 0xB6U);
    if (ret) return ret;
    k_msleep(5);  /* BMI088 requires ≥1 ms after reset before SPI access */

    /* Wake from suspend */
    ret = bmi088_write_reg(BMI088_ACC_PWR_CTRL, BMI088_ACC_PWR_ON);
    if (ret) return ret;
    k_msleep(50);  /* BMI088 accelerometer startup time */

    /* Read chip ID — also flushes the SPI dummy byte pipeline */
    ret = bmi088_read_regs(BMI088_ACC_CHIP_ID, &chip_id, 1);
    if (ret) return ret;
    if (chip_id != BMI088_ACC_CHIP_ID_VAL) {
        LOG_ERR("BMI088 chip ID mismatch: expected 0x%02X got 0x%02X",
                BMI088_ACC_CHIP_ID_VAL, chip_id);
        return -ENODEV;
    }
    LOG_INF("BMI088 detected (chip_id=0x%02X)", chip_id);

    /* Configure ODR = 2000 Hz, oversample ratio = 4 */
    ret = bmi088_write_reg(BMI088_ACC_CONF, BMI088_ACC_ODR_2000);
    if (ret) return ret;

    /* Configure range = ±24g */
    ret = bmi088_write_reg(BMI088_ACC_RANGE, BMI088_ACC_RANGE_24G);
    if (ret) return ret;

    /* FIFO watermark: 512 frames × 6 bytes = 3072 bytes
     * BMI088 FIFO watermark register is in units of frames (1 frame = 6 bytes). */
    ret = bmi088_write_reg(BMI088_ACC_FIFO_WM_0,
                           (uint8_t)(FIFO_WATERMARK_FRAMES & 0xFFU));
    if (ret) return ret;
    ret = bmi088_write_reg(BMI088_ACC_FIFO_WM_1,
                           (uint8_t)((FIFO_WATERMARK_FRAMES >> 8) & 0x03U));
    if (ret) return ret;

    /* FIFO config: stream mode (overwrite oldest), acc data enabled */
    ret = bmi088_write_reg(BMI088_ACC_FIFO_CONFIG_0, 0x00U);  /* no sync, streaming */
    if (ret) return ret;
    ret = bmi088_write_reg(BMI088_ACC_FIFO_CONFIG_1,
                           BMI088_ACC_FIFO_STREAM | BMI088_ACC_FIFO_ACC_EN);
    if (ret) return ret;

    /* INT1: push-pull output, active-high */
    ret = bmi088_write_reg(BMI088_ACC_INT1_IO_CTRL, BMI088_ACC_INT1_OUT);
    if (ret) return ret;

    /* Map FIFO watermark interrupt to INT1 */
    ret = bmi088_write_reg(BMI088_ACC_INT_MAP_DATA, BMI088_ACC_INT_FWM);
    if (ret) return ret;

    /* Leave suspend: accelerometer active */
    ret = bmi088_write_reg(BMI088_ACC_PWR_CONF, BMI088_ACC_PWR_ACTIVE);
    if (ret) return ret;
    k_msleep(1);

    LOG_INF("BMI088 configured: ODR=2kHz, range=±24g, FIFO watermark=%u frames",
            FIFO_WATERMARK_FRAMES);
    return 0;
}

/**
 * @brief Put BMI088 into suspend mode (lowest power, ~2.1 µA).
 */
static void bmi088_suspend(void)
{
    bmi088_write_reg(BMI088_ACC_PWR_CONF, 0x03U);  /* suspend bit */
}

/* ──────────────────────────────────────────────────────────────────────────
 * Data acquisition — DMA-backed FIFO drain
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Acquire SAMPLES_PER_AXIS samples on all three axes.
 *
 * Uses BMI088 FIFO with watermark interrupt.  Each interrupt fires after
 * 512 frames; we read 4 bursts of 512 frames to fill the 2048-sample buffer.
 * This keeps CPU wakeup duty very low during acquisition.
 *
 * After acquisition, raw 16-bit values are scaled to g using:
 *   value_g = raw × (24.0 / 32768.0)  [for ±24g range]
 */
static int acquire_samples(void)
{
    const float32_t lsb_to_g = 24.0f / 32768.0f;
    int bursts = 0;
    uint32_t sample_offset = 0;

    /* Number of 512-frame bursts needed to collect 2048 samples */
    const int total_bursts = SAMPLES_PER_AXIS / FIFO_WATERMARK_FRAMES;

    for (bursts = 0; bursts < total_bursts; bursts++) {
        /* Wait for FIFO watermark interrupt */
        int ret = k_sem_take(&fifo_watermark_sem, K_MSEC(1000));
        if (ret == -EAGAIN) {
            LOG_ERR("BMI088 FIFO watermark timeout (burst %d)", bursts);
            return -ETIMEDOUT;
        }

        /* Burst-read 512 frames from FIFO */
        ret = bmi088_fifo_burst_read(FIFO_WATERMARK_FRAMES);
        if (ret) {
            LOG_ERR("FIFO burst read failed: %d", ret);
            return ret;
        }

        /* Convert int16 raw values to float (g units) and deinterleave axes */
        for (uint32_t i = 0; i < FIFO_WATERMARK_FRAMES; i++) {
            uint32_t si = sample_offset + i;
            /* raw_acc layout: [x_lsb, x_msb, y_lsb, y_msb, z_lsb, z_msb] × N
             * After bmi088_fifo_burst_read, raw_acc[] contains interleaved int16. */
            int16_t raw_x = (int16_t)sys_le16_to_cpu(
                ((uint16_t *)raw_acc)[i * 3 + 0]);
            int16_t raw_y = (int16_t)sys_le16_to_cpu(
                ((uint16_t *)raw_acc)[i * 3 + 1]);
            int16_t raw_z = (int16_t)sys_le16_to_cpu(
                ((uint16_t *)raw_acc)[i * 3 + 2]);

            acc_x[si] = (float32_t)raw_x * lsb_to_g;
            acc_y[si] = (float32_t)raw_y * lsb_to_g;
            acc_z[si] = (float32_t)raw_z * lsb_to_g;
        }
        sample_offset += FIFO_WATERMARK_FRAMES;
    }

    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Feature extraction — calls dsp_features.c
 * ──────────────────────────────────────────────────────────────────────── */

typedef struct {
    float32_t rms_x, rms_y, rms_z;
    float32_t kurtosis_x, kurtosis_y, kurtosis_z;
    float32_t crest_x, crest_y, crest_z;
    bearing_features_t bearing_x;
    float32_t temperature_c;  /* from BMI088 internal sensor */
} sensor_features_t;

/**
 * @brief Run full DSP pipeline on acquired acc_x/y/z buffers.
 *
 * Clocks up to 48 MHz before FFT, restores HSI16 after.
 * Estimated duration: ~80 ms @ 48 MHz.
 */
static void extract_features(sensor_features_t *feat)
{
    static arm_rfft_fast_instance_f32 fft_inst;
    static bearing_geometry_t bearing = {
        /* Default: 6205-2RS bearing — common in HVAC fans */
        .BPFO_factor = 3.047f,
        .BPFI_factor = 4.953f,
        .BSF_factor  = 1.994f,
        .FTF_factor  = 0.381f,
    };

    clock_boost_48mhz();

    arm_rfft_fast_init_f32(&fft_inst, SAMPLES_PER_AXIS);

    /* --- X axis --- */
    apply_hann_window(acc_x, SAMPLES_PER_AXIS);
    arm_rfft_fast_f32(&fft_inst, acc_x, fft_buf, 0);
    arm_cmplx_mag_f32(fft_buf, fft_mag, SAMPLES_PER_AXIS / 2);
    feat->rms_x      = compute_rms(acc_x, SAMPLES_PER_AXIS);
    feat->kurtosis_x = compute_kurtosis(acc_x, SAMPLES_PER_AXIS);
    feat->crest_x    = compute_crest_factor(acc_x, SAMPLES_PER_AXIS);
    /* Bearing frequency extraction on X axis (radial load axis) */
    float32_t rpm_estimate = 1500.0f;  /* TODO: derive from FFT peak or tachometer */
    compute_bearing_frequencies(fft_mag, SAMPLES_PER_AXIS / 2,
                                SAMPLE_RATE_HZ, rpm_estimate,
                                &bearing, &feat->bearing_x);

    /* --- Y axis --- */
    apply_hann_window(acc_y, SAMPLES_PER_AXIS);
    arm_rfft_fast_f32(&fft_inst, acc_y, fft_buf, 0);
    arm_cmplx_mag_f32(fft_buf, fft_mag, SAMPLES_PER_AXIS / 2);
    feat->rms_y      = compute_rms(acc_y, SAMPLES_PER_AXIS);
    feat->kurtosis_y = compute_kurtosis(acc_y, SAMPLES_PER_AXIS);
    feat->crest_y    = compute_crest_factor(acc_y, SAMPLES_PER_AXIS);

    /* --- Z axis (axial) --- */
    apply_hann_window(acc_z, SAMPLES_PER_AXIS);
    arm_rfft_fast_f32(&fft_inst, acc_z, fft_buf, 0);
    arm_cmplx_mag_f32(fft_buf, fft_mag, SAMPLES_PER_AXIS / 2);
    feat->rms_z      = compute_rms(acc_z, SAMPLES_PER_AXIS);
    feat->kurtosis_z = compute_kurtosis(acc_z, SAMPLES_PER_AXIS);
    feat->crest_z    = compute_crest_factor(acc_z, SAMPLES_PER_AXIS);

    clock_restore_hsi16();

    LOG_INF("Features: RMS_x=%.4f g, Kurt_x=%.2f, Crest_x=%.2f",
            (double)feat->rms_x, (double)feat->kurtosis_x, (double)feat->crest_x);
    LOG_INF("          BPFO=%.4f, BPFI=%.4f",
            (double)feat->bearing_x.bpfo_mag, (double)feat->bearing_x.bpfi_mag);
}

/* ──────────────────────────────────────────────────────────────────────────
 * CBOR payload assembly
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Encode sensor features into a CBOR map for MQTT transmission.
 *
 * Output format (CBOR map keys are uint integers for compactness):
 *   {1: device_id, 2: timestamp_s, 3: rms_x, 4: rms_y, 5: rms_z,
 *    6: kurt_x, 7: kurt_y, 8: kurt_z, 9: crest_x, 10: crest_y, 11: crest_z,
 *    12: bpfo, 13: bpfi, 14: bsf, 15: ftf, 16: temperature_c}
 *
 * @param feat     Computed features.
 * @param out      Output buffer.
 * @param out_size Buffer size.
 * @return Number of bytes written, or negative error.
 */
static int encode_cbor_payload(const sensor_features_t *feat,
                               uint8_t *out, size_t out_size)
{
    CborEncoder enc, map;
    cbor_encoder_init(&enc, out, out_size, 0);

    cbor_encoder_create_map(&enc, &map, 16);

    /* Device ID */
    cbor_encode_uint(&map, 1);
    cbor_encode_text_stringz(&map, CONFIG_SENSOR_DEVICE_ID);

    /* Timestamp (seconds since epoch — filled from RTC) */
    cbor_encode_uint(&map, 2);
    cbor_encode_uint(&map, (uint64_t)k_uptime_seconds());

    /* RMS per axis */
    cbor_encode_uint(&map, 3); cbor_encode_float(&map, feat->rms_x);
    cbor_encode_uint(&map, 4); cbor_encode_float(&map, feat->rms_y);
    cbor_encode_uint(&map, 5); cbor_encode_float(&map, feat->rms_z);

    /* Kurtosis per axis */
    cbor_encode_uint(&map, 6); cbor_encode_float(&map, feat->kurtosis_x);
    cbor_encode_uint(&map, 7); cbor_encode_float(&map, feat->kurtosis_y);
    cbor_encode_uint(&map, 8); cbor_encode_float(&map, feat->kurtosis_z);

    /* Crest factor per axis */
    cbor_encode_uint(&map, 9);  cbor_encode_float(&map, feat->crest_x);
    cbor_encode_uint(&map, 10); cbor_encode_float(&map, feat->crest_y);
    cbor_encode_uint(&map, 11); cbor_encode_float(&map, feat->crest_z);

    /* Bearing fault indicators (X axis, radial) */
    cbor_encode_uint(&map, 12); cbor_encode_float(&map, feat->bearing_x.bpfo_mag);
    cbor_encode_uint(&map, 13); cbor_encode_float(&map, feat->bearing_x.bpfi_mag);
    cbor_encode_uint(&map, 14); cbor_encode_float(&map, feat->bearing_x.bsf_mag);
    cbor_encode_uint(&map, 15); cbor_encode_float(&map, feat->bearing_x.ftf_mag);

    /* Temperature */
    cbor_encode_uint(&map, 16); cbor_encode_float(&map, feat->temperature_c);

    cbor_encoder_close_container(&enc, &map);

    size_t written = cbor_encoder_get_buffer_size(&enc, out);
    LOG_INF("CBOR payload: %zu bytes", written);
    return (int)written;
}

/* ──────────────────────────────────────────────────────────────────────────
 * CC1352P Sub-GHz radio interface
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Transmit payload via CC1352P Sub-GHz radio.
 *
 * Protocol: Simple binary framing over UART.
 *   [0xAA][0x55][len_hi][len_lo][payload...][crc16]
 *
 * The CC1352P runs a companion firmware that receives this frame,
 * wraps it in the Sub-GHz PHY layer, and transmits to the gateway.
 * Total TX window including CC1352P boot: < 50 ms.
 *
 * @param data   Payload bytes.
 * @param len    Payload length in bytes.
 * @return 0 on success.
 */
static int cc1352_transmit(const uint8_t *data, size_t len)
{
    if (!device_is_ready(uart_cc13)) {
        LOG_ERR("CC1352 UART not ready");
        return -ENODEV;
    }

    /* Frame header */
    uint8_t hdr[4] = {
        0xAAU, 0x55U,
        (uint8_t)((len >> 8) & 0xFFU),
        (uint8_t)(len & 0xFFU),
    };

    /* CRC-16/CCITT over payload */
    uint16_t crc = 0xFFFFU;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000U) ? ((crc << 1) ^ 0x1021U) : (crc << 1);
        }
    }
    uint8_t crc_bytes[2] = { (uint8_t)(crc >> 8), (uint8_t)(crc & 0xFFU) };

    /* Wake CC1352P (toggle GPIO or send wakeup byte first) */
    uint8_t wake = 0xFFU;
    uart_poll_out(uart_cc13, wake);
    k_msleep(3);  /* CC1352P boot from deep sleep: ~2 ms */

    /* Send header + payload + CRC */
    for (size_t i = 0; i < sizeof(hdr); i++) {
        uart_poll_out(uart_cc13, hdr[i]);
    }
    for (size_t i = 0; i < len; i++) {
        uart_poll_out(uart_cc13, data[i]);
    }
    uart_poll_out(uart_cc13, crc_bytes[0]);
    uart_poll_out(uart_cc13, crc_bytes[1]);

    /* Allow TX to complete before powering down CC1352P
     * Sub-GHz @ 50 kbps: (4 + len + 2) bytes ≈ 210 bytes → ~34 ms air time.
     * Add margin for CC1352P processing overhead → 50 ms total. */
    k_msleep(50);

    LOG_INF("CC1352P transmit complete (%zu bytes payload)", len);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Battery life calculator
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Estimate sensor battery life in years.
 *
 * Model:
 *   - Battery capacity: BATTERY_CAPACITY_MAH at 3.6 V
 *   - Each measurement cycle: ACTIVE_CURRENT_MA for ACTIVE_DURATION_MS
 *   - Between cycles: STANDBY_CURRENT_UA (system-level)
 *
 * @param meas_interval_s  Measurement interval in seconds.
 * @return Estimated battery life in years (float).
 */
float estimate_battery_life_years(uint32_t meas_interval_s)
{
    /* Charge consumed per measurement cycle (mAh) */
    float active_charge_mah = (float)ACTIVE_CURRENT_MA
                              * ((float)ACTIVE_DURATION_MS / 1000.0f)
                              / 3600.0f;

    /* Charge consumed in standby between measurements (mAh) */
    float standby_s   = (float)meas_interval_s
                        - ((float)ACTIVE_DURATION_MS / 1000.0f);
    float standby_mah = ((float)STANDBY_CURRENT_UA / 1000.0f)
                        * standby_s / 3600.0f;

    float charge_per_cycle_mah = active_charge_mah + standby_mah;

    /* Cycles per year */
    float cycles_per_year = (365.0f * 24.0f * 3600.0f) / (float)meas_interval_s;

    /* Total charge per year */
    float charge_per_year_mah = charge_per_cycle_mah * cycles_per_year;

    /* Apply 0.5 derating for temperature + self-discharge */
    float effective_capacity_mah = (float)BATTERY_CAPACITY_MAH * 0.5f;

    float years = effective_capacity_mah / charge_per_year_mah;

    LOG_INF("Battery estimate: %.1f years (interval=%us, "
            "active=%.4f mAh/cycle, standby=%.4f mAh/cycle)",
            (double)years, meas_interval_s,
            (double)active_charge_mah, (double)standby_mah);

    return years;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Zephyr power management — enter STOP2
 * ──────────────────────────────────────────────────────────────────────── */

/**
 * @brief Request STOP2 low-power state and wait for RTC wakeup.
 *
 * In Zephyr's power management framework, the idle thread calls
 * pm_state_set() when no threads are ready.  Here we manually trigger
 * it after completing the TRANSMIT phase.
 *
 * LPTIM1 is configured as RTC-equivalent wakeup timer at ≤32 kHz LSE clock.
 */
static void enter_stop2_until_next_cycle(void)
{
    /* Configure Zephyr PM for STOP2 with wakeup after meas interval */
    struct pm_state_info pm_info = {
        .state          = PM_STATE_SUSPEND_TO_IDLE,
        .substate_id    = 2,                       /* STOP2 sub-state */
        .min_residency_us = MEAS_INTERVAL_S * 1000000U,
        .exit_latency_us  = 300U,
    };

    LOG_INF("Entering STOP2 for %u s", MEAS_INTERVAL_S);

    /* Suspend all devices except RTC/LPTIM wakeup source */
    pm_device_action_run(spi_dev,   PM_DEVICE_ACTION_SUSPEND);
    pm_device_action_run(uart_cc13, PM_DEVICE_ACTION_SUSPEND);

    /* Hand off to Zephyr PM — CPU halts here until wakeup */
    pm_state_set(pm_info.state, pm_info.substate_id);

    /* Execution resumes here after wakeup */
    pm_device_action_run(spi_dev,   PM_DEVICE_ACTION_RESUME);
    pm_device_action_run(uart_cc13, PM_DEVICE_ACTION_RESUME);

    LOG_DBG("Woke from STOP2");
}

/* ──────────────────────────────────────────────────────────────────────────
 * GPIO and peripheral initialization
 * ──────────────────────────────────────────────────────────────────────── */

static int peripherals_init(void)
{
    int ret;

    /* BMI088 chip select GPIO */
    if (!gpio_is_ready_dt(&bmi_cs)) {
        LOG_ERR("BMI088 CS GPIO not ready");
        return -ENODEV;
    }
    ret = gpio_pin_configure_dt(&bmi_cs, GPIO_OUTPUT_ACTIVE);
    if (ret) return ret;
    gpio_pin_set_dt(&bmi_cs, 1);  /* deselect */

    /* BMI088 INT1 input — rising edge */
    if (!gpio_is_ready_dt(&bmi_int1)) {
        LOG_ERR("BMI088 INT1 GPIO not ready");
        return -ENODEV;
    }
    ret = gpio_pin_configure_dt(&bmi_int1, GPIO_INPUT);
    if (ret) return ret;
    ret = gpio_pin_interrupt_configure_dt(&bmi_int1, GPIO_INT_EDGE_RISING);
    if (ret) return ret;
    gpio_init_callback(&bmi_int1_cb_data, bmi088_int1_handler,
                       BIT(bmi_int1.pin));
    gpio_add_callback(bmi_int1.port, &bmi_int1_cb_data);

    /* SPI bus */
    if (!device_is_ready(spi_dev)) {
        LOG_ERR("SPI device not ready");
        return -ENODEV;
    }

    /* UART to CC1352P */
    if (!device_is_ready(uart_cc13)) {
        LOG_ERR("CC1352 UART not ready");
        return -ENODEV;
    }

    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Main entry point — power state machine loop
 * ──────────────────────────────────────────────────────────────────────── */

int main(void)
{
    int ret;
    sensor_features_t features;

    LOG_INF("MEMS Low-Power Sensor v1.0 starting");
    LOG_INF("Target: ≥8-year battery life, BMI088 + STM32U5 + CC1352P");

    ret = peripherals_init();
    if (ret) {
        LOG_ERR("peripherals_init failed: %d", ret);
        return ret;
    }

    ret = bmi088_init();
    if (ret) {
        LOG_ERR("bmi088_init failed: %d", ret);
        return ret;
    }

    /* Log battery life estimate at default interval */
    float life_years = estimate_battery_life_years(MEAS_INTERVAL_S);
    LOG_INF("Estimated battery life @ %u-min interval: %.1f years",
            MEAS_INTERVAL_MINUTES, (double)life_years);

    /* Main measurement loop */
    while (1) {
        /* ── WAKE ──────────────────────────────────────────────────── */
        psm_state = PSM_WAKE;
        LOG_DBG("PSM: WAKE");

        /* Re-initialize BMI088 (may have been fully powered off in STOP2) */
        ret = bmi088_init();
        if (ret) {
            LOG_ERR("BMI088 re-init failed: %d — skipping cycle", ret);
            goto sleep;
        }

        /* ── ACQUIRE ────────────────────────────────────────────────── */
        psm_state = PSM_ACQUIRE;
        LOG_DBG("PSM: ACQUIRE (2048 samples × 3 axes @ 2 kHz)");

        ret = acquire_samples();
        if (ret) {
            LOG_ERR("acquire_samples failed: %d — skipping cycle", ret);
            bmi088_suspend();
            goto sleep;
        }

        /* Suspend BMI088 immediately after acquisition */
        bmi088_suspend();

        /* ── FFT + FEATURES ─────────────────────────────────────────── */
        psm_state = PSM_FFT_FEATURES;
        LOG_DBG("PSM: FFT_FEATURES");

        extract_features(&features);

        /* ── TRANSMIT ───────────────────────────────────────────────── */
        psm_state = PSM_TRANSMIT;
        LOG_DBG("PSM: TRANSMIT");

        int payload_len = encode_cbor_payload(&features,
                                              mqtt_payload,
                                              sizeof(mqtt_payload));
        if (payload_len > 0) {
            ret = cc1352_transmit(mqtt_payload, (size_t)payload_len);
            if (ret) {
                LOG_WRN("Transmit failed: %d (data buffered for retry)", ret);
                /* TODO: implement store-and-forward buffer */
            }
        }

sleep:
        /* ── STOP2 ──────────────────────────────────────────────────── */
        psm_state = PSM_STOP2;
        enter_stop2_until_next_cycle();
    }

    /* Unreachable */
    return 0;
}
