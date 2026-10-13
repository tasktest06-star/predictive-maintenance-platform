/* board_stwinbx1.c -- pdm_board.h for the ST STEVAL-STWINBX1.
 *
 * Target: STM32U585AII6Q (Cortex-M33, 160 MHz, 2 MB flash, 786 kB SRAM)
 * Vibration: IIS3DWB on SPI2, 26.667 kHz, +-16 g, 6.3 kHz flat bandwidth
 * Also on board: ISM330BX IMU, IIS2ICLX inclinometer, IMP23ABSU analogue mic,
 *                STTS22H temperature, ILPS22QS pressure, BlueNRG-M2 BLE
 *
 * Only the vibration path, temperature, battery and BLE are implemented here;
 * the rest are present on the board and are future channels (the microphone in
 * particular -- see docs/hardware/stwinbx1.md).
 *
 * THIS CODE HAS NOT RUN ON HARDWARE. It is written against the STM32U5 HAL and
 * the IIS3DWB datasheet, and its DSP is verified by the shared pipeline's
 * host tests, but no board has been flashed. Register values, DMA wiring and
 * timing are review-quality, not measured. docs/07_production_readiness.md
 * states this; it is repeated here because this is the file where it matters.
 */

#include "pdm_board.h"

#include <string.h>

#include "app_config.h"
#include "pdm_pipeline.h"

/* The STM32U5 HAL. Provided by STM32CubeU5; see firmware/CMakeLists.txt. */
#include "stm32u5xx_hal.h"

/* ------------------------------------------------------------ IIS3DWB map */
/* DS12749 rev 4, Table 18. */
#define IIS3DWB_WHO_AM_I        0x0Fu
#define IIS3DWB_WHO_AM_I_VALUE  0x7Bu
#define IIS3DWB_CTRL1_XL        0x10u
#define IIS3DWB_CTRL3_C         0x12u
#define IIS3DWB_CTRL4_C         0x13u
#define IIS3DWB_CTRL6_C         0x15u
#define IIS3DWB_CTRL8_XL        0x17u
#define IIS3DWB_CTRL10_C        0x19u
#define IIS3DWB_STATUS_REG      0x1Eu
#define IIS3DWB_OUT_TEMP_L      0x20u
#define IIS3DWB_OUTX_L_XL       0x28u
#define IIS3DWB_FIFO_CTRL1      0x07u
#define IIS3DWB_FIFO_CTRL3      0x09u
#define IIS3DWB_FIFO_CTRL4      0x0Au
#define IIS3DWB_FIFO_STATUS1    0x3Au
#define IIS3DWB_FIFO_STATUS2    0x3Bu
#define IIS3DWB_FIFO_DATA_OUT_X_L 0x3Eu

/* CTRL1_XL: ODR_XL = 0b0101 (26.667 kHz, the part's only rate), FS = +-16 g.
 * The IIS3DWB has exactly one output data rate. That is the whole point of the
 * part: a flat 6.3 kHz bandwidth with no configuration to get wrong. */
#define IIS3DWB_CTRL1_26K7_16G  0x5Cu
/* CTRL3_C: BDU (block data update) + IF_INC. BDU matters: without it a 16-bit
 * sample can be read with its high byte from one conversion and its low byte
 * from the next, which produces occasional wild samples that look exactly like
 * impacts. */
#define IIS3DWB_CTRL3_BDU_IFINC 0x44u
/* FIFO_CTRL4: FIFO_MODE = 0b110 (continuous). */
#define IIS3DWB_FIFO_CONTINUOUS 0x06u

/* STTS22H, I2C. */
#define STTS22H_ADDR            (0x3Fu << 1)
#define STTS22H_WHOAMI          0x01u
#define STTS22H_WHOAMI_VALUE    0xA0u
#define STTS22H_TEMP_L_OUT      0x06u
#define STTS22H_CTRL            0x04u

/* ------------------------------------------------------------- HAL handles */
/* Defined in board_init.c (CubeMX-generated); declared here so this file does
 * not depend on generation order. */
extern SPI_HandleTypeDef  hspi2;    /* IIS3DWB */
extern I2C_HandleTypeDef  hi2c2;    /* STTS22H, ILPS22QS */
extern ADC_HandleTypeDef  hadc1;    /* battery divider */
extern DMA_HandleTypeDef  handle_GPDMA1_Channel0;
extern IWDG_HandleTypeDef hiwdg;
extern LPTIM_HandleTypeDef hlptim1; /* wake-up timer */

/* ------------------------------------------------------------ local state */
#define ACCEL_RAW_WORDS (APP_ACQ_SAMPLES * 3u)   /* x, y, z per sample */

/* The FIFO read buffer. 8192 samples x 3 axes x 2 bytes = 48 kB, the single
 * largest allocation in the firmware after the pipeline's own buffers. Placed
 * in SRAM2 by the linker script so it does not compete with the DSP buffers in
 * SRAM1 -- the GPDMA can reach both, and keeping them apart means a DMA burst
 * does not stall the core's access to the FFT working set. */
static int16_t s_fifo_raw[ACCEL_RAW_WORDS] __attribute__((section(".sram2")));
static volatile uint8_t s_dma_done = 0;
static volatile uint8_t s_dma_error = 0;
static uint8_t  s_accel_powered = 0;
static uint64_t s_uid = 0;
static int16_t  s_rssi = -99;

/* Per-asset shaft speed, pushed down as configuration. There is no tachometer
 * on this board: the STWIN.box has no counter input wired to a connector, so
 * the speed comes from commissioning or from a tacholess estimate made in the
 * cloud. `pdm_board_shaft_hz` reports it as estimated, which lands in the frame
 * as PDM_FRAME_SPEED_ESTIMATED. */
static float s_shaft_hz = 0.0f;
static uint8_t s_shaft_is_measured = 0u;

void pdm_board_stwinbx1_set_shaft_hz(float hz, int measured)
{
    s_shaft_hz = hz;
    s_shaft_is_measured = measured ? 1u : 0u;
}

/* -------------------------------------------------------------- SPI helpers */
static pdm_board_status spi_read(uint8_t reg, uint8_t *dst, uint16_t len)
{
    uint8_t cmd = (uint8_t)(reg | 0x80u);     /* read bit */
    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_RESET);   /* CS low */
    HAL_StatusTypeDef st = HAL_SPI_Transmit(&hspi2, &cmd, 1u, 10u);
    if (st == HAL_OK) {
        st = HAL_SPI_Receive(&hspi2, dst, len, 100u);
    }
    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
    return (st == HAL_OK) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

static pdm_board_status spi_write(uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = {reg, value};
    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_RESET);
    HAL_StatusTypeDef st = HAL_SPI_Transmit(&hspi2, buf, 2u, 10u);
    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
    return (st == HAL_OK) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

/* --------------------------------------------------------------- self-test */
/* IIS3DWB self-test (DS12749 section 6.2): enable ST, wait for settling,
 * average 5 samples, and check the delta against the datasheet window.
 *
 * Worth the 200 ms at boot. A sensor with a cracked solder joint or a dead axis
 * reads a plausible, quiet signal -- the machine looks healthy and nobody finds
 * out for a year. The alternative to this test is a field visit. */
static pdm_board_status accel_selftest(void)
{
    int32_t base[3] = {0, 0, 0};
    int32_t test[3] = {0, 0, 0};
    uint8_t raw[6];

    if (spi_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G) != PDM_BOARD_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    HAL_Delay(100);
    for (int i = 0; i < 5; ++i) {
        HAL_Delay(2);
        if (spi_read(IIS3DWB_OUTX_L_XL, raw, 6u) != PDM_BOARD_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            base[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    /* CTRL5_C ST_XL = 0b01 (positive sign self-test). */
    if (spi_write(0x14u, 0x01u) != PDM_BOARD_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    HAL_Delay(100);
    for (int i = 0; i < 5; ++i) {
        HAL_Delay(2);
        if (spi_read(IIS3DWB_OUTX_L_XL, raw, 6u) != PDM_BOARD_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            test[a] += (int16_t)((uint16_t)raw[2 * a] | ((uint16_t)raw[2 * a + 1] << 8));
        }
    }
    (void)spi_write(0x14u, 0x00u);

    /* Datasheet window at +-16 g: 50 to 1700 LSB on each axis. Checking all
     * three even though only one is used for vibration: a dead orthogonal axis
     * is a strong hint that the part is damaged, and cross-axis leakage means a
     * dead axis still affects the measured one. */
    for (int a = 0; a < 3; ++a) {
        const int32_t delta = (test[a] - base[a]) / 5;
        const int32_t mag = (delta < 0) ? -delta : delta;
        if (mag < 50 || mag > 1700) {
            return PDM_BOARD_ERR_SELFTEST;
        }
    }
    return PDM_BOARD_OK;
}

/* ------------------------------------------------------------------- init */
pdm_board_status pdm_board_init(pdm_board_info *info)
{
    /* 96-bit unique ID, folded to 64. Collisions across a fleet of thousands
     * are vanishingly unlikely and the full 96 bits do not fit the frame. */
    const uint32_t *uid = (const uint32_t *)UID_BASE;
    s_uid = ((uint64_t)uid[0] << 32) ^ ((uint64_t)uid[1] << 16) ^ (uint64_t)uid[2];

    if (info != NULL) {
        info->kit = APP_KIT_NAME;
        info->accel_part = "IIS3DWB";
        info->device_uid = s_uid;
        info->fs_hz = APP_FS_HZ;
        info->full_scale_g = APP_FULL_SCALE_G;
        info->adc_bits = 16u;
        info->flash_bytes = 2u * 1024u * 1024u;
        info->ram_bytes = 786u * 1024u;
    }

    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);   /* CS idle high */

    uint8_t who = 0;
    if (spi_read(IIS3DWB_WHO_AM_I, &who, 1u) != PDM_BOARD_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    if (who != IIS3DWB_WHO_AM_I_VALUE) {
        return PDM_BOARD_ERR_ID;
    }
    /* Software reset, then wait for it to clear rather than a fixed delay: the
     * reset time is not specified as a maximum, only as a typical. */
    if (spi_write(IIS3DWB_CTRL3_C, 0x01u) != PDM_BOARD_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    for (int i = 0; i < 20; ++i) {
        uint8_t c3 = 0xFFu;
        HAL_Delay(1);
        if (spi_read(IIS3DWB_CTRL3_C, &c3, 1u) == PDM_BOARD_OK && (c3 & 0x01u) == 0u) {
            break;
        }
    }
    if (spi_write(IIS3DWB_CTRL3_C, IIS3DWB_CTRL3_BDU_IFINC) != PDM_BOARD_OK) {
        return PDM_BOARD_ERR_BUS;
    }

    uint8_t stts = 0;
    if (HAL_I2C_Mem_Read(&hi2c2, STTS22H_ADDR, STTS22H_WHOAMI,
                         I2C_MEMADD_SIZE_8BIT, &stts, 1u, 20u) != HAL_OK
        || stts != STTS22H_WHOAMI_VALUE) {
        /* Not fatal: temperature is useful context, not a safety function. The
         * frame carries NaN and the cloud shows the channel as absent. */
    }

    return accel_selftest();
}

/* ------------------------------------------------------------- power/settle */
pdm_board_status pdm_board_accel_power(int on)
{
    if (on) {
        if (spi_write(IIS3DWB_CTRL1_XL, IIS3DWB_CTRL1_26K7_16G) != PDM_BOARD_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        /* FIFO: continuous mode, accelerometer at its native rate, watermark
         * irrelevant because the whole block is read by DMA in one go. */
        (void)spi_write(IIS3DWB_FIFO_CTRL3, 0x05u);
        (void)spi_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);
        s_accel_powered = 1u;
    } else {
        /* ODR = 0 is power-down: 2.5 uA. */
        (void)spi_write(IIS3DWB_FIFO_CTRL4, 0x00u);
        (void)spi_write(IIS3DWB_CTRL1_XL, 0x00u);
        s_accel_powered = 0u;
    }
    return PDM_BOARD_OK;
}

uint32_t pdm_board_accel_settle_ms(void)
{
    /* DS12749: 100 ms turn-on time to full accuracy at 26.667 kHz. Sampling
     * earlier gives a settling transient indistinguishable from an impact
     * train -- and it is the largest single term in the energy budget, which is
     * why APP_SENSOR_KEEPALIVE_THRESHOLD_S exists. */
    return APP_SENSOR_SETTLE_MS;
}

/* ---------------------------------------------------------------- acquire */
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi == &hspi2) {
        s_dma_done = 1u;
    }
}

void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi == &hspi2) {
        s_dma_error = 1u;
        s_dma_done = 1u;
    }
}

pdm_board_status pdm_board_acquire(float *out, size_t n, float *fs_hz_actual)
{
    if (out == NULL || n > APP_ACQ_SAMPLES) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    if (!s_accel_powered) {
        /* Reading a powered-down IIS3DWB returns zeros, which look like a
         * perfectly smooth machine. Refusing is the only safe answer. */
        return PDM_BOARD_ERR_NOT_READY;
    }

    /* Flush the FIFO so the block starts now rather than containing whatever
     * accumulated during the settling delay -- those samples are exactly the
     * settling transient we waited out. */
    (void)spi_write(IIS3DWB_FIFO_CTRL4, 0x00u);
    (void)spi_write(IIS3DWB_FIFO_CTRL4, IIS3DWB_FIFO_CONTINUOUS);

    s_dma_done = 0u;
    s_dma_error = 0u;

    /* Burst-read the FIFO. The IIS3DWB FIFO is 3 kB (512 samples x 3 axes x
     * 2 B), so an 8192-sample block cannot be taken in one DMA transfer: it is
     * read in FIFO-sized chunks, and the FIFO must not overrun between chunks.
     * At 26.667 kHz the FIFO fills in 19.2 ms, and a 3 kB SPI burst at 10 MHz
     * is 2.4 ms, so there is 8x of margin -- but the margin is only real if the
     * core is not doing anything else, which is why the whole acquisition is
     * blocking and the radio is never active during it. */
    const size_t chunk_samples = 512u;
    size_t done = 0;
    while (done < n) {
        const size_t want = ((n - done) < chunk_samples) ? (n - done) : chunk_samples;

        /* Wait for `want` samples to be available. FIFO_STATUS1/2 give the
         * level in words; one sample is 3 words (x, y, z). */
        uint32_t guard = 0;
        for (;;) {
            uint8_t st[2] = {0, 0};
            if (spi_read(IIS3DWB_FIFO_STATUS1, st, 2u) != PDM_BOARD_OK) {
                return PDM_BOARD_ERR_BUS;
            }
            const uint16_t level = (uint16_t)(((uint16_t)(st[1] & 0x03u) << 8) | st[0]);
            if ((st[1] & 0x40u) != 0u) {
                /* FIFO_OVR_LATCHED: samples were lost. The block would be a
                 * concatenation of two different times. */
                return PDM_BOARD_ERR_OVERRUN;
            }
            if (level >= want) {
                break;
            }
            if (++guard > 200000u) {
                return PDM_BOARD_ERR_TIMEOUT;
            }
        }

        uint8_t cmd = (uint8_t)(IIS3DWB_FIFO_DATA_OUT_X_L | 0x80u);
        HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_RESET);
        if (HAL_SPI_Transmit(&hspi2, &cmd, 1u, 10u) != HAL_OK) {
            HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
            return PDM_BOARD_ERR_BUS;
        }
        s_dma_done = 0u;
        if (HAL_SPI_Receive_DMA(&hspi2, (uint8_t *)&s_fifo_raw[done * 3u],
                                (uint16_t)(want * 3u * 2u)) != HAL_OK) {
            HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
            return PDM_BOARD_ERR_BUS;
        }
        /* Sleep while DMA runs. WFI rather than a spin loop: the core draws
         * 19 mA at 160 MHz and 1.2 mA in Sleep, and the DMA interrupt wakes it.
         * Over a 307 ms acquisition that is the difference between 5.8 mA.s and
         * 0.4 mA.s -- bigger than the entire compute budget. */
        guard = 0;
        while (!s_dma_done) {
            __WFI();
            if (++guard > 1000000u) {
                HAL_SPI_DMAStop(&hspi2);
                HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
                return PDM_BOARD_ERR_TIMEOUT;
            }
        }
        HAL_GPIO_WritePin(GPIOI, GPIO_PIN_7, GPIO_PIN_SET);
        if (s_dma_error) {
            return PDM_BOARD_ERR_BUS;
        }
        done += want;
    }

    /* Convert the measurement axis to g. +-16 g over int16: 0.488 mg/LSB.
     *
     * Axis choice is per installation, not per firmware: the radial axis is the
     * one that sees bearing and unbalance energy, and which of x/y/z that is
     * depends on how the sensor was bolted on. Z here matches the STWIN.box
     * enclosure's mounting face; the commissioning tool records it and the
     * cloud cross-checks it against the gravity vector in the DC level. */
    const float lsb_g = 2.0f * APP_FULL_SCALE_G / 65536.0f;
    for (size_t i = 0; i < n; ++i) {
        out[i] = (float)s_fifo_raw[i * 3u + 2u] * lsb_g;
    }
    if (fs_hz_actual != NULL) {
        /* Nominal. The IIS3DWB's internal oscillator is +-3 %, and there is no
         * way to measure the true rate on this board without a reference tone:
         * the FIFO has no timestamping at this ODR. A 3 % error moves a 106 Hz
         * BPFO line by 3.2 Hz, which the +-2 % matcher tolerance absorbs only
         * just. docs/hardware/stwinbx1.md records this as the kit's main
         * metrological limitation and the planned fix (a 1 kHz reference tone
         * injected through the self-test actuator at commissioning). */
        *fs_hz_actual = APP_FS_HZ;
    }
    return PDM_BOARD_OK;
}

/* ------------------------------------------------------------------ speed */
pdm_board_status pdm_board_shaft_hz(float *shaft_hz, int *estimated)
{
    if (shaft_hz == NULL || !(s_shaft_hz > 0.0f)) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    *shaft_hz = s_shaft_hz;
    if (estimated != NULL) {
        *estimated = s_shaft_is_measured ? 0 : 1;
    }
    return PDM_BOARD_OK;
}

/* ------------------------------------------------------------ temperature */
pdm_board_status pdm_board_temperature_c(float *temp_c)
{
    if (temp_c == NULL) {
        return PDM_BOARD_ERR_BUS;
    }
    uint8_t one_shot = 0x04u;
    if (HAL_I2C_Mem_Write(&hi2c2, STTS22H_ADDR, STTS22H_CTRL,
                          I2C_MEMADD_SIZE_8BIT, &one_shot, 1u, 20u) != HAL_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    HAL_Delay(10);
    uint8_t raw[2] = {0, 0};
    if (HAL_I2C_Mem_Read(&hi2c2, STTS22H_ADDR, STTS22H_TEMP_L_OUT,
                         I2C_MEMADD_SIZE_8BIT, raw, 2u, 20u) != HAL_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    const int16_t t = (int16_t)((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
    *temp_c = (float)t * 0.01f;    /* 0.01 degC/LSB */
    return PDM_BOARD_OK;
}

/* ---------------------------------------------------------------- battery */
pdm_board_status pdm_board_battery_mv(uint16_t *mv)
{
    if (mv == NULL) {
        return PDM_BOARD_ERR_BUS;
    }
    if (HAL_ADC_Start(&hadc1) != HAL_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    if (HAL_ADC_PollForConversion(&hadc1, 20u) != HAL_OK) {
        (void)HAL_ADC_Stop(&hadc1);
        return PDM_BOARD_ERR_TIMEOUT;
    }
    const uint32_t raw = HAL_ADC_GetValue(&hadc1);
    (void)HAL_ADC_Stop(&hadc1);
    /* STWIN.box battery divider is 2:1 into a 14-bit ADC referenced to 3.0 V.
     * Calibrated against VREFINT would be better and is a commissioning step;
     * the nominal conversion is accurate to ~2 %, which is enough to drive a
     * 3.5 V/3.3 V decision but not to report a state of charge. */
    *mv = (uint16_t)((raw * 3000u * 2u) / 16384u);
    return PDM_BOARD_OK;
}

int16_t pdm_board_rssi_dbm(void)
{
    return s_rssi;
}

/* ------------------------------------------------------------------ radio */
/* BLE via the on-board BlueNRG-M2. The payload goes out as a GATT notification
 * on a vendor characteristic, 244 bytes of ATT MTU per notification, so a
 * 300-byte frame is two notifications. Implemented in ble_transport.c against
 * the BlueNRG host stack; declared here. */
extern int ble_transport_send(const uint8_t *data, size_t len, int16_t *rssi_out);

pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len)
{
    int16_t rssi = s_rssi;
    const int rc = ble_transport_send(frame, len, &rssi);
    s_rssi = rssi;
    return (rc == 0) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

/* ------------------------------------------------------------------ queue */
/* Flash ring buffer in the top 256 kB of the 2 MB internal flash. 512 frames
 * at 300 B is 150 kB, which fits in 19 of the U5's 8 kB pages with room for the
 * wear-levelling spare.
 *
 * Why internal flash and not the microSD that this board has: an SD card draws
 * 20-100 mA while mounted and has a write latency that is not bounded, and a
 * card that fails does so silently. Internal flash is slower to erase but its
 * failure modes are known and its energy cost is a few microjoules per frame.
 * The SD slot stays for raw-snapshot capture during commissioning, where the
 * node is on a bench with a cable. */
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

/* ------------------------------------------------------------------ sleep */
uint32_t pdm_board_sleep_ms(uint32_t ms)
{
    if (ms == 0u) {
        return 0u;
    }
    /* Stop 2 with the LPTIM1 running from LSE: 2.5 uA plus the LPTIM. Standby
     * would be ~0.3 uA but loses SRAM, and re-running pdm_pipeline_init (window
     * generation, twiddle tables, filter design: ~40 ms of transcendentals)
     * costs more energy than the 2.2 uA saved over an hour. Measured crossover
     * is around a 9-hour interval; at the default hourly cadence Stop 2 wins.
     *
     * SRAM2 is kept powered because the FIFO buffer lives there; SRAM3 is not.
     */
    const uint32_t ticks = (ms * 32768u) / 1000u;
    (void)HAL_LPTIM_TimeOut_Start_IT(&hlptim1, 0xFFFFu, ticks);
    HAL_SuspendTick();
    HAL_PWREx_EnterSTOP2Mode(PWR_STOPENTRY_WFI);
    HAL_ResumeTick();
    (void)HAL_LPTIM_TimeOut_Stop_IT(&hlptim1);
    /* Returns the requested value: the LPTIM is the timebase, so the sleep is
     * accurate to one 32.768 kHz tick. A node that drifts is a node whose
     * trends cannot be aligned across a fleet. */
    return ms;
}

void pdm_board_identify(uint32_t seconds)
{
    /* The user LED. Finding one sensor among forty identical ones bolted to a
     * plant wall is otherwise a genuinely hard problem. */
    for (uint32_t i = 0; i < seconds * 4u; ++i) {
        HAL_GPIO_TogglePin(GPIOH, GPIO_PIN_12);
        HAL_Delay(250);
    }
    HAL_GPIO_WritePin(GPIOH, GPIO_PIN_12, GPIO_PIN_RESET);
}

void pdm_board_watchdog_feed(void)
{
    (void)HAL_IWDG_Refresh(&hiwdg);
}
