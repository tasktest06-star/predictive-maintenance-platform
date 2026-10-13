/* board_esp32s3.c -- pdm_board.h for an ESP32-S3-DevKitC-1 retrofit node.
 *
 * Target: ESP32-S3 (dual Xtensa LX7 @ 240 MHz, 512 kB SRAM, 8 MB PSRAM,
 *         8 MB flash), Wi-Fi 802.11n + BLE 5, ESP-IDF v5.x
 * Sensor: ADXL355 (default) or IIS3DWB, both over SPI2 -- see app_config.h for
 *         why the choice changes what this node can diagnose
 *
 * THIS CODE HAS NOT RUN ON HARDWARE. Written against ESP-IDF v5.2 and the part
 * datasheets; the DSP above it is verified by the shared pipeline's host tests.
 *
 * What is different about this kit, and it is not the MCU:
 *
 * **Wi-Fi association dominates the energy budget by three orders of
 * magnitude.** 240 mA.s per transmission against 0.17 mA.s for sensing and
 * compute combined. Every design decision here follows from that: a 6-hour
 * default interval, a deep queue, and `pdm_board_transmit` that BATCHES. A
 * conventional one-frame-per-connection design would flatten a 2600 mAh cell in
 * under two months.
 *
 * **It is the only kit here with enough RAM pressure to need PSMA.** 512 kB of
 * internal SRAM minus ~180 kB for the Wi-Fi/LWIP stacks leaves the pipeline
 * tight at 4096-sample acquisitions; the sample buffer therefore lives in PSRAM,
 * which is slower but DMA-reachable on the S3. The FFT working set stays
 * internal, because a 4096-point transform out of PSRAM is roughly 3x slower.
 */

#include "pdm_board.h"

#include <string.h>

#include "app_config.h"
#include "pdm_pipeline.h"

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_sleep.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "heap_caps.h"
#include "nvs_flash.h"

static const char *TAG = "pdm_board";

/* ---------------------------------------------------------------- pin map */
#define PIN_SCLK    GPIO_NUM_12
#define PIN_MOSI    GPIO_NUM_11
#define PIN_MISO    GPIO_NUM_13
#define PIN_CS      GPIO_NUM_10
#define PIN_LED     GPIO_NUM_38   /* on-board WS2812 on DevKitC-1; see identify */
#define PIN_BATT    ADC_CHANNEL_3 /* GPIO4, 2:1 divider */

/* --------------------------------------------------------------- ADXL355 */
#define ADXL355_DEVID_AD        0x00u
#define ADXL355_DEVID_AD_VALUE  0xADu
#define ADXL355_PARTID          0x02u
#define ADXL355_PARTID_VALUE    0xEDu
#define ADXL355_STATUS          0x04u
#define ADXL355_FIFO_ENTRIES    0x05u
#define ADXL355_TEMP2           0x06u
#define ADXL355_FIFO_DATA       0x11u
#define ADXL355_FILTER          0x28u
#define ADXL355_RANGE          0x2Cu
#define ADXL355_POWER_CTL       0x2Du
#define ADXL355_SELF_TEST       0x2Eu
#define ADXL355_RESET           0x2Fu

/* FILTER: ODR_LPF = 0b0001 (2000 Hz ODR, 500 Hz LPF) .. 0b0000 is 4000/1000.
 * 4000 Hz ODR with the 1 kHz filter: the part's maximum, and the reason this
 * profile exists. */
#define ADXL355_FILTER_4000HZ   0x00u
#define ADXL355_RANGE_8G        0x02u

/* ---------------------------------------------------------------- state */
static spi_device_handle_t   s_spi;
static adc_oneshot_unit_handle_t s_adc;
static uint64_t  s_uid;
static uint8_t   s_powered;
static int16_t   s_rssi = -99;
static float     s_shaft_hz;
static uint8_t   s_shaft_measured;
static uint8_t   s_wifi_up;

/* Acquisition staging in PSRAM: 4096 samples x 3 axes x 3 bytes (the ADXL355 is
 * 20-bit, packed into 3 bytes per axis) = 36 kB, or 48 kB for the IIS3DWB
 * profile. Internal SRAM is better spent on the FFT working set. */
static uint8_t *s_raw;

void pdm_board_esp32s3_set_shaft_hz(float hz, int measured)
{
    s_shaft_hz = hz;
    s_shaft_measured = measured ? 1u : 0u;
}

/* ------------------------------------------------------------ SPI helpers */
static esp_err_t reg_write(uint8_t reg, uint8_t val)
{
    uint8_t tx[2] = {(uint8_t)(reg << 1), val};   /* ADXL355: 7-bit reg, R/W in bit 0 */
    spi_transaction_t t = {.length = 16, .tx_buffer = tx};
    return spi_device_polling_transmit(s_spi, &t);
}

static esp_err_t reg_read(uint8_t reg, uint8_t *dst, size_t len)
{
    uint8_t tx = (uint8_t)((reg << 1) | 1u);
    spi_transaction_t t = {
        .length = 8, .rxlength = 8u * len, .tx_buffer = &tx, .rx_buffer = NULL,
    };
    /* Two-phase: command then burst. The IDF's half-duplex mode would do this
     * in one transaction, but the explicit split keeps the ADXL355 and IIS3DWB
     * paths identical and the register maps are the only difference. */
    esp_err_t err = spi_device_polling_transmit(s_spi, &t);
    if (err != ESP_OK) {
        return err;
    }
    spi_transaction_t r = {.length = 8u * len, .rxlength = 8u * len, .rx_buffer = dst};
    return spi_device_polling_transmit(s_spi, &r);
}

/* ------------------------------------------------------------------- init */
pdm_board_status pdm_board_init(pdm_board_info *info)
{
    if (nvs_flash_init() == ESP_ERR_NVS_NO_FREE_PAGES) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    uint8_t mac[6] = {0};
    (void)esp_efuse_mac_get_default(mac);
    s_uid = 0;
    for (int i = 0; i < 6; ++i) {
        s_uid = (s_uid << 8) | mac[i];
    }

    s_raw = heap_caps_malloc(APP_ACQ_SAMPLES * 3u * 3u, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA);
    if (s_raw == NULL) {
        /* The one malloc in the firmware, at init, never freed -- the shared
         * no-dynamic-allocation rule exists to avoid fragmentation over years of
         * runtime, and a single permanent allocation at boot does not create
         * that risk. Failing here is a hard failure: there is no degraded mode
         * that produces trustworthy data. */
        ESP_LOGE(TAG, "no PSRAM for the acquisition buffer");
        return PDM_BOARD_ERR_BUS;
    }

    spi_bus_config_t bus = {
        .mosi_io_num = PIN_MOSI, .miso_io_num = PIN_MISO, .sclk_io_num = PIN_SCLK,
        .quadwp_io_num = -1, .quadhd_io_num = -1,
        .max_transfer_sz = (int)(APP_ACQ_SAMPLES * 9u),
    };
    if (spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    spi_device_interface_config_t dev = {
        /* 8 MHz. The ADXL355 tolerates 10 MHz and the IIS3DWB 10 MHz; 8 leaves
         * margin for the DevKitC's long unshielded jumpers, which is the real
         * constraint on a breadboarded node and a common cause of intermittent
         * WHO_AM_I failures that get blamed on the sensor. */
        .clock_speed_hz = 8 * 1000 * 1000,
        .mode = 0, .spics_io_num = PIN_CS, .queue_size = 4,
    };
    if (spi_bus_add_device(SPI2_HOST, &dev, &s_spi) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }

    adc_oneshot_unit_init_cfg_t adc_cfg = {.unit_id = ADC_UNIT_1};
    (void)adc_oneshot_new_unit(&adc_cfg, &s_adc);
    adc_oneshot_chan_cfg_t chan = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12};
    (void)adc_oneshot_config_channel(s_adc, PIN_BATT, &chan);

    if (info != NULL) {
        info->kit = APP_KIT_NAME;
#ifdef PDM_SENSOR_ADXL355
        info->accel_part = "ADXL355";
        info->adc_bits = 20u;
#else
        info->accel_part = "IIS3DWB";
        info->adc_bits = 16u;
#endif
        info->device_uid = s_uid;
        info->fs_hz = APP_FS_HZ;
        info->full_scale_g = APP_FULL_SCALE_G;
        info->flash_bytes = 8u * 1024u * 1024u;
        info->ram_bytes = 512u * 1024u;
    }

#ifdef PDM_SENSOR_ADXL355
    uint8_t id[2] = {0, 0};
    if (reg_read(ADXL355_DEVID_AD, id, 1u) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    if (id[0] != ADXL355_DEVID_AD_VALUE) {
        ESP_LOGE(TAG, "DEVID_AD 0x%02x, expected 0x%02x", id[0], ADXL355_DEVID_AD_VALUE);
        return PDM_BOARD_ERR_ID;
    }
    (void)reg_write(ADXL355_RESET, 0x52u);
    vTaskDelay(pdMS_TO_TICKS(10));
    (void)reg_write(ADXL355_RANGE, ADXL355_RANGE_8G);
    (void)reg_write(ADXL355_FILTER, ADXL355_FILTER_4000HZ);

    /* ADXL355 self-test (Rev C, "SELF TEST"): assert ST1+ST2 and check the
     * X/Y/Z deltas against the datasheet windows. Worth the 100 ms: a sensor
     * with a cracked joint reads a plausible, quiet signal and the machine looks
     * healthy for a year. */
    int32_t base[3] = {0}, test[3] = {0};
    (void)reg_write(ADXL355_POWER_CTL, 0x00u);   /* measurement mode */
    vTaskDelay(pdMS_TO_TICKS(APP_SENSOR_SETTLE_MS * 3u));
    for (int i = 0; i < 8; ++i) {
        uint8_t raw[9];
        vTaskDelay(pdMS_TO_TICKS(1));
        if (reg_read(0x08u, raw, 9u) != ESP_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            const int32_t v = ((int32_t)raw[a * 3] << 12) | ((int32_t)raw[a * 3 + 1] << 4)
                            | ((int32_t)raw[a * 3 + 2] >> 4);
            base[a] += (v & 0x80000) ? (v - 0x100000) : v;
        }
    }
    (void)reg_write(ADXL355_SELF_TEST, 0x03u);
    vTaskDelay(pdMS_TO_TICKS(50));
    for (int i = 0; i < 8; ++i) {
        uint8_t raw[9];
        vTaskDelay(pdMS_TO_TICKS(1));
        if (reg_read(0x08u, raw, 9u) != ESP_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (int a = 0; a < 3; ++a) {
            const int32_t v = ((int32_t)raw[a * 3] << 12) | ((int32_t)raw[a * 3 + 1] << 4)
                            | ((int32_t)raw[a * 3 + 2] >> 4);
            test[a] += (v & 0x80000) ? (v - 0x100000) : v;
        }
    }
    (void)reg_write(ADXL355_SELF_TEST, 0x00u);
    /* Datasheet: X/Y 0.2-3.0 g, Z 0.5-6.5 g at +-8 g, i.e. 1 LSB = 15.6 ug.
     * Converted to counts: X/Y 12800-192000, Z 32000-416000. */
    for (int a = 0; a < 3; ++a) {
        const int32_t delta = (test[a] - base[a]) / 8;
        const int32_t mag = (delta < 0) ? -delta : delta;
        const int32_t lo = (a == 2) ? 32000 : 12800;
        const int32_t hi = (a == 2) ? 416000 : 192000;
        if (mag < lo || mag > hi) {
            ESP_LOGE(TAG, "self-test axis %d delta %ld outside [%ld, %ld]",
                     a, (long)mag, (long)lo, (long)hi);
            return PDM_BOARD_ERR_SELFTEST;
        }
    }
    (void)reg_write(ADXL355_POWER_CTL, 0x01u);   /* standby until needed */
#endif
    return PDM_BOARD_OK;
}

pdm_board_status pdm_board_accel_power(int on)
{
#ifdef PDM_SENSOR_ADXL355
    /* POWER_CTL bit 0: 1 = standby (21 uA), 0 = measurement (200 uA). */
    if (reg_write(ADXL355_POWER_CTL, on ? 0x00u : 0x01u) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
#else
    if (reg_write(0x10u, on ? 0x5Cu : 0x00u) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
#endif
    s_powered = on ? 1u : 0u;
    return PDM_BOARD_OK;
}

uint32_t pdm_board_accel_settle_ms(void)
{
    return APP_SENSOR_SETTLE_MS;
}

pdm_board_status pdm_board_acquire(float *out, size_t n, float *fs_hz_actual)
{
    if (out == NULL || n > APP_ACQ_SAMPLES || s_raw == NULL) {
        return PDM_BOARD_ERR_NOT_READY;
    }
    if (!s_powered) {
        return PDM_BOARD_ERR_NOT_READY;
    }
#ifdef PDM_SENSOR_ADXL355
    /* The ADXL355 FIFO is 96 samples deep; at 4 kHz it fills in 24 ms, so a
     * 4096-sample block is read in 43 chunks. Each chunk must complete before
     * the next 24 ms elapses, which an 8 MHz SPI burst does in 0.9 ms. The
     * margin is large, and it is only real because the Wi-Fi radio is off
     * during acquisition -- the S3's Wi-Fi task runs at a higher priority than
     * this one and will happily starve it for tens of milliseconds. The
     * application serialises them; see pdm_app.c. */
    size_t done = 0;
    const float lsb_g = APP_FULL_SCALE_G / 524288.0f;   /* 20-bit, +-8 g */
    while (done < n) {
        uint8_t entries = 0;
        uint32_t guard = 0;
        for (;;) {
            if (reg_read(ADXL355_FIFO_ENTRIES, &entries, 1u) != ESP_OK) {
                return PDM_BOARD_ERR_BUS;
            }
            if (entries >= 3u) {
                break;
            }
            if (++guard > 100000u) {
                return PDM_BOARD_ERR_TIMEOUT;
            }
            taskYIELD();
        }
        uint8_t status = 0;
        if (reg_read(ADXL355_STATUS, &status, 1u) != ESP_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        if (status & 0x02u) {        /* FIFO_OVR */
            return PDM_BOARD_ERR_OVERRUN;
        }
        /* FIFO entries are counted in 3-byte axis words, 3 per sample. */
        size_t samples = entries / 3u;
        if (samples > (n - done)) {
            samples = n - done;
        }
        if (reg_read(ADXL355_FIFO_DATA, &s_raw[done * 9u], samples * 9u) != ESP_OK) {
            return PDM_BOARD_ERR_BUS;
        }
        for (size_t i = 0; i < samples; ++i) {
            /* Z axis, 20-bit left-justified in 3 bytes, two's complement. */
            const uint8_t *p = &s_raw[(done + i) * 9u + 6u];
            int32_t v = ((int32_t)p[0] << 12) | ((int32_t)p[1] << 4) | ((int32_t)p[2] >> 4);
            if (v & 0x80000) {
                v -= 0x100000;
            }
            out[done + i] = (float)v * lsb_g;
        }
        done += samples;
    }
#else
    /* IIS3DWB profile: identical structure to board_stwinbx1.c's acquire. */
    return PDM_BOARD_ERR_NOT_READY;   /* implemented in the IIS3DWB variant */
#endif
    if (fs_hz_actual != NULL) {
        /* The ADXL355's ODR derives from an internal oscillator with no
         * published tolerance better than +-5 % over temperature. Unlike the
         * STWIN.box, this kit CAN measure it: the ESP32-S3's 40 MHz crystal is
         * +-10 ppm, and esp_timer gives the true elapsed time of the
         * acquisition. Reporting the measured rate is the single cheapest
         * accuracy win available on this kit. */
        *fs_hz_actual = APP_FS_HZ;   /* replaced by the measured value below */
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
#ifdef PDM_SENSOR_ADXL355
    uint8_t raw[2] = {0, 0};
    if (reg_read(ADXL355_TEMP2, raw, 2u) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    const uint16_t t = (uint16_t)(((uint16_t)(raw[0] & 0x0Fu) << 8) | raw[1]);
    /* Rev C: 1852 LSB at 25 C, -9.05 LSB/degC. */
    *temp_c = 25.0f + ((float)t - 1852.0f) / -9.05f;
    return PDM_BOARD_OK;
#else
    (void)temp_c;
    return PDM_BOARD_ERR_NOT_READY;
#endif
}

pdm_board_status pdm_board_battery_mv(uint16_t *mv)
{
    if (mv == NULL) {
        return PDM_BOARD_ERR_BUS;
    }
    int raw = 0;
    if (adc_oneshot_read(s_adc, PIN_BATT, &raw) != ESP_OK) {
        return PDM_BOARD_ERR_BUS;
    }
    /* 12 dB attenuation gives roughly 0-3.1 V full scale on the S3; the 2:1
     * divider doubles it. The S3's ADC is notoriously non-linear and needs the
     * eFuse calibration curve (esp_adc_cal) for anything better than +-5 %.
     * +-5 % is enough for a 3.4 V/3.2 V decision and nowhere near enough for a
     * state-of-charge display, which is why the frame carries millivolts and
     * the dashboard shows a trend rather than a percentage. */
    *mv = (uint16_t)((raw * 3100 * 2) / 4095);
    return PDM_BOARD_OK;
}

int16_t pdm_board_rssi_dbm(void)
{
    wifi_ap_record_t ap;
    if (s_wifi_up && esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
        s_rssi = (int16_t)ap.rssi;
    }
    return s_rssi;
}

/* ------------------------------------------------------------------ radio */
/* mqtt_transport.c: brings Wi-Fi up, connects to the broker over TLS, publishes,
 * and tears Wi-Fi down again. The bring-up/tear-down is the expensive part and
 * is why pdm_board_transmit batches. */
extern int mqtt_transport_publish_batch(const uint8_t *const *frames, const size_t *lens,
                                        size_t count, int16_t *rssi_out);

pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len)
{
    /* Single-frame transmit still pays the full 240 mA.s association cost, so
     * the application is expected to queue and let the drain batch. This path
     * exists for the commissioning case and for an alarm that must not wait. */
    const uint8_t *frames[1] = {frame};
    const size_t lens[1] = {len};
    int16_t rssi = s_rssi;
    const int rc = mqtt_transport_publish_batch(frames, lens, 1u, &rssi);
    s_rssi = rssi;
    s_wifi_up = (rc == 0) ? 1u : 0u;
    return (rc == 0) ? PDM_BOARD_OK : PDM_BOARD_ERR_BUS;
}

/* ------------------------------------------------------------------ queue */
/* NVS-backed ring buffer in the 8 MB flash. 4096 frames x 300 B = 1.2 MB.
 * NVS rather than a raw partition: it brings its own wear levelling and CRC,
 * and at one write per 6 hours the overhead is irrelevant. */
extern pdm_board_status nvs_queue_push(const uint8_t *frame, size_t len);
extern pdm_board_status nvs_queue_pop(uint8_t *frame, size_t cap, size_t *len);
extern uint32_t         nvs_queue_depth(void);

pdm_board_status pdm_board_queue_push(const uint8_t *frame, size_t len)
{
    return nvs_queue_push(frame, len);
}

pdm_board_status pdm_board_queue_pop(uint8_t *frame, size_t cap, size_t *len)
{
    return nvs_queue_pop(frame, cap, len);
}

uint32_t pdm_board_queue_depth(void)
{
    return nvs_queue_depth();
}

/* ------------------------------------------------------------------ sleep */
uint32_t pdm_board_sleep_ms(uint32_t ms)
{
    if (ms == 0u) {
        return 0u;
    }
    /* Deep sleep, 7 uA. It loses all RAM, so the application state that must
     * survive -- sequence number, queue head, the operating threshold -- lives
     * in RTC_DATA_ATTR (8 kB of RTC slow memory, retained) rather than in the
     * pdm_app_state struct.
     *
     * Deep sleep also restarts from app_main, so this function does not return:
     * the "loop" in main.c is really one iteration per wake. That is a real
     * structural difference from the other kits and it is why the shared
     * pdm_app.c keeps all its state in one struct that can be snapshotted.
     *
     * Light sleep (240 uA) would let the loop actually loop, and at a 6-hour
     * interval 240 uA is 5.2 mA.h per cycle -- 40x the entire acquisition
     * budget. Deep sleep is not optional on this kit. */
    (void)esp_sleep_enable_timer_wakeup((uint64_t)ms * 1000ull);
    esp_deep_sleep_start();
    return ms;   /* unreachable */
}

void pdm_board_identify(uint32_t seconds)
{
    gpio_set_direction(PIN_LED, GPIO_MODE_OUTPUT);
    for (uint32_t i = 0; i < seconds * 4u; ++i) {
        gpio_set_level(PIN_LED, (int)(i & 1u));
        vTaskDelay(pdMS_TO_TICKS(250));
    }
    gpio_set_level(PIN_LED, 0);
}

void pdm_board_watchdog_feed(void)
{
    /* The IDF task watchdog is fed by yielding; esp_task_wdt_reset() is the
     * explicit form and is used here because the acquisition loop deliberately
     * does not yield for long stretches. */
    extern void esp_task_wdt_reset(void);
    esp_task_wdt_reset();
}
