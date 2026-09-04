/**
 * main.c — STM32H743 + Zephyr RTOS
 * Piezoelectric + Acoustic Emission (AE) Dual Sensing Pipeline
 *
 * Two concurrent acquisition channels:
 *   1. Vibration: SPI from piezo accelerometer @ 160 kSps → 64k FFT → features
 *   2. AE: ADC @ 1 Msps → threshold detect (COMP) → ring-down capture → features
 *
 * Power strategy:
 *   - AE comparator threshold monitor: always-on (~0.5 mW)
 *   - AE ADC burst capture: on threshold event only
 *   - Vibration channel: wake on RPM present OR scheduled interval
 *   - AE alert: immediate wakeup + full-stack transmission
 */

#include <zephyr/kernel.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/comparator.h>
#include <zephyr/pm/pm.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include <math.h>

#include "ae_features.h"
#include "vibration_hf.h"
#include "radio.h"

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

/* -------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */

/* Vibration pipeline */
#define VIB_SAMPLE_RATE_HZ      160000U   /* 160 kSps — 2x Nyquist for 80 kHz */
#define VIB_FFT_SIZE            65536U    /* 64k-point FFT */
#define VIB_SPI_FREQ_HZ         5000000U  /* 5 MHz SPI clock to accelerometer */

/* AE pipeline */
#define AE_SAMPLE_RATE_HZ       1000000U  /* 1 Msps */
#define AE_BURST_SAMPLES        4096U     /* ~4 ms capture window */
#define AE_THRESHOLD_MV         50.0f     /* threshold for event detection (mV) */
#define AE_RMS_WINDOW_MS        100U      /* rolling RMS window */
#define AE_RMS_SAMPLES          (AE_SAMPLE_RATE_HZ / 1000 * AE_RMS_WINDOW_MS)

/* Scheduling */
#define VIB_INTERVAL_SEC        60        /* vibration snapshot every 60 s */
#define RADIO_TX_INTERVAL_SEC   300       /* routine telemetry every 5 min */

/* -------------------------------------------------------------------------
 * Buffers (placed in AXI SRAM — 512 KB on STM32H743)
 * ------------------------------------------------------------------------- */

/* Vibration: 64k int16 samples = 128 KB */
static int16_t  vib_raw[VIB_FFT_SIZE]   __attribute__((section(".AXI_SRAM")));
/* Vibration: 64k float FFT magnitude = 256 KB */
static float    vib_fft[VIB_FFT_SIZE]   __attribute__((section(".AXI_SRAM")));
/* Vibration: envelope spectrum (same size as FFT output) */
static float    vib_envelope[VIB_FFT_SIZE] __attribute__((section(".AXI_SRAM")));

/* AE: burst capture buffer */
static uint16_t ae_burst[AE_BURST_SAMPLES] __attribute__((section(".AXI_SRAM")));

/* AE RMS ring buffer */
static float    ae_rms_ring[AE_RMS_SAMPLES] __attribute__((section(".AXI_SRAM")));
static uint32_t ae_rms_idx = 0;
static float    ae_rms_current = 0.0f;

/* -------------------------------------------------------------------------
 * Thread stacks and synchronization
 * ------------------------------------------------------------------------- */

#define VIB_STACK_SIZE  4096
#define AE_STACK_SIZE   2048
#define RADIO_STACK_SIZE 2048

K_THREAD_STACK_DEFINE(vib_stack, VIB_STACK_SIZE);
K_THREAD_STACK_DEFINE(ae_stack, AE_STACK_SIZE);
K_THREAD_STACK_DEFINE(radio_stack, RADIO_STACK_SIZE);

static struct k_thread vib_thread_data;
static struct k_thread ae_thread_data;
static struct k_thread radio_thread_data;

/* Semaphore: AE burst ready for feature extraction */
K_SEM_DEFINE(ae_burst_sem, 0, 1);

/* Semaphore: AE alert (priority wakeup for transmission) */
K_SEM_DEFINE(ae_alert_sem, 0, 1);

/* Message queue for outgoing feature packets */
K_MSGQ_DEFINE(radio_tx_queue, sizeof(sensor_packet_t), 8, 4);

/* -------------------------------------------------------------------------
 * Device handles
 * ------------------------------------------------------------------------- */

static const struct device *adc_dev;
static const struct device *spi_dev;
static const struct device *comp_dev;

/* -------------------------------------------------------------------------
 * AE comparator ISR — fires on threshold crossing
 * Very low overhead: just triggers DMA and signals thread
 * ------------------------------------------------------------------------- */

static volatile bool ae_capture_active = false;

static void ae_comparator_isr(const struct device *dev, void *user_data)
{
    ARG_UNUSED(dev);
    ARG_UNUSED(user_data);

    if (ae_capture_active) {
        return;  /* Already capturing a burst — ignore retriggering */
    }

    ae_capture_active = true;

    /*
     * Trigger ADC DMA burst capture: AE_BURST_SAMPLES at 1 Msps.
     * The DMA-complete callback will give ae_burst_sem.
     * This ISR must return as fast as possible (<1 µs).
     */
    adc_start_dma_burst(adc_dev, ae_burst, AE_BURST_SAMPLES);
}

/* ADC DMA complete callback — called from DMA ISR context */
static void ae_dma_complete_cb(const struct device *dev, int result, void *user_data)
{
    ARG_UNUSED(dev);
    ARG_UNUSED(user_data);

    if (result == 0) {
        k_sem_give(&ae_burst_sem);
    }
    ae_capture_active = false;
}

/* -------------------------------------------------------------------------
 * AE Thread — feature extraction and anomaly detection
 * Priority 2 (higher than vibration, lower than radio alert)
 * ------------------------------------------------------------------------- */

static void ae_thread_fn(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    ae_event_t event;
    sensor_packet_t pkt;

    LOG_INF("AE thread started, threshold=%.1f mV", (double)AE_THRESHOLD_MV);

    while (1) {
        /* Wait for DMA burst capture to complete */
        k_sem_take(&ae_burst_sem, K_FOREVER);

        /* Extract AE features from burst buffer */
        event = ae_extract_features(ae_burst, AE_BURST_SAMPLES, AE_THRESHOLD_MV);

        /* Update rolling RMS with mean amplitude of this burst */
        float burst_rms = 0.0f;
        for (uint32_t i = 0; i < AE_BURST_SAMPLES; i++) {
            float v = (float)ae_burst[i] * (3300.0f / 4096.0f);  /* 12-bit, 3.3V ref */
            burst_rms += v * v;
        }
        burst_rms = sqrtf(burst_rms / AE_BURST_SAMPLES);
        ae_rms_current = ae_rms_update(ae_rms_ring, burst_rms, AE_RMS_SAMPLES);
        event.rms_100ms = ae_rms_current;

        /* Log event */
        LOG_DBG("AE event: amp=%.1f dBAE counts=%u dur=%.1f us energy=%.3f",
                (double)event.amplitude_dBae, event.counts,
                (double)event.duration_us, (double)event.energy);

        /* Build outgoing packet */
        memset(&pkt, 0, sizeof(pkt));
        pkt.type = PKT_TYPE_AE_EVENT;
        pkt.timestamp_ms = k_uptime_get_32();
        pkt.ae = event;

        /* Check for actionable fault signatures */
        if (ae_is_bearing_distress(&event)) {
            LOG_WRN("BEARING DISTRESS detected: amp=%.1f dBAE, counts=%u/s",
                    (double)event.amplitude_dBae, event.counts);
            pkt.alert_flags |= ALERT_BEARING_DISTRESS;
            k_sem_give(&ae_alert_sem);  /* Priority wakeup for immediate TX */
        }

        if (ae_is_cavitation(&event)) {
            LOG_WRN("CAVITATION detected: amp=%.1f dBAE",
                    (double)event.amplitude_dBae);
            pkt.alert_flags |= ALERT_CAVITATION;
            k_sem_give(&ae_alert_sem);
        }

        /* Enqueue for radio TX (non-blocking; drops oldest if full) */
        k_msgq_put(&radio_tx_queue, &pkt, K_NO_WAIT);
    }
}

/* -------------------------------------------------------------------------
 * Vibration Thread — 64k FFT + envelope demodulation
 * Priority 3 (lower than AE)
 * ------------------------------------------------------------------------- */

static void vib_thread_fn(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    sensor_packet_t pkt;
    vib_features_t  vib_feat;

    LOG_INF("Vibration thread started, interval=%d s", VIB_INTERVAL_SEC);

    while (1) {
        /* Wait for scheduled interval or AE-triggered wakeup */
        k_sleep(K_SECONDS(VIB_INTERVAL_SEC));

        /* Acquire VIB_FFT_SIZE samples from accelerometer via SPI */
        vibration_hf_acquire(vib_raw, VIB_FFT_SIZE, VIB_SAMPLE_RATE_HZ);

        /* 64k-point FFT */
        vibration_hf_fft(vib_raw, vib_fft, VIB_FFT_SIZE);

        /* Envelope demodulation in high-frequency band (20–40 kHz) */
        envelope_demod(vib_fft, vib_envelope,
                       20.0f,   /* f_low_kHz  */
                       40.0f,   /* f_high_kHz */
                       (float)VIB_SAMPLE_RATE_HZ);

        /* Extract bearing fault frequencies from envelope spectrum */
        vib_feat = vibration_extract_features(vib_fft, vib_envelope,
                                              VIB_FFT_SIZE, VIB_SAMPLE_RATE_HZ);

        /* Build packet */
        memset(&pkt, 0, sizeof(pkt));
        pkt.type = PKT_TYPE_VIB_SNAPSHOT;
        pkt.timestamp_ms = k_uptime_get_32();
        pkt.vib = vib_feat;

        if (vib_feat.bpfo_amplitude > VIB_BPFO_WARN_THRESHOLD ||
            vib_feat.bpfi_amplitude > VIB_BPFI_WARN_THRESHOLD) {
            pkt.alert_flags |= ALERT_VIBRATION_BEARING;
            LOG_WRN("Vibration bearing fault: BPFO=%.2f BPFI=%.2f",
                    (double)vib_feat.bpfo_amplitude,
                    (double)vib_feat.bpfi_amplitude);
        }

        k_msgq_put(&radio_tx_queue, &pkt, K_NO_WAIT);
    }
}

/* -------------------------------------------------------------------------
 * Radio Thread — transmit queued packets
 * AE alerts trigger immediate transmission; routine telemetry batched
 * ------------------------------------------------------------------------- */

static void radio_thread_fn(void *p1, void *p2, void *p3)
{
    ARG_UNUSED(p1); ARG_UNUSED(p2); ARG_UNUSED(p3);

    sensor_packet_t pkt;
    int64_t last_routine_tx = 0;

    LOG_INF("Radio thread started");

    while (1) {
        /*
         * Wait for either:
         *   - AE alert semaphore (immediate priority TX), or
         *   - Scheduled routine telemetry interval
         */
        int ret = k_sem_take(&ae_alert_sem,
                             K_SECONDS(RADIO_TX_INTERVAL_SEC));

        if (ret == 0) {
            LOG_INF("AE alert: immediate radio wakeup");
        } else {
            LOG_DBG("Routine telemetry window");
        }

        /* Drain the TX queue */
        while (k_msgq_get(&radio_tx_queue, &pkt, K_NO_WAIT) == 0) {
            radio_transmit(&pkt, sizeof(pkt));
        }

        last_routine_tx = k_uptime_get();
    }
}

/* -------------------------------------------------------------------------
 * Power Management
 * ------------------------------------------------------------------------- */

/**
 * ae_always_on_init — configure AE comparator for threshold monitoring.
 *
 * The COMP peripheral consumes ~0.1 mW. It triggers ae_comparator_isr on
 * threshold crossing without waking the MCU from Stop mode. This is the key
 * to low idle power: only the comparator watches the AE channel.
 */
static int ae_always_on_init(void)
{
    comp_dev = DEVICE_DT_GET(DT_NODELABEL(comp1));
    if (!device_is_ready(comp_dev)) {
        LOG_ERR("AE comparator not ready");
        return -ENODEV;
    }

    /* Set threshold to AE_THRESHOLD_MV using internal DAC reference */
    comparator_set_threshold_mv(comp_dev, (uint32_t)AE_THRESHOLD_MV);
    comparator_set_callback(comp_dev, ae_comparator_isr, NULL);
    comparator_enable(comp_dev);

    LOG_INF("AE always-on comparator enabled at %.0f mV", (double)AE_THRESHOLD_MV);
    return 0;
}

/* -------------------------------------------------------------------------
 * main — hardware init, thread creation, idle loop
 * ------------------------------------------------------------------------- */

int main(void)
{
    int ret;

    LOG_INF("Piezo+AE sensor starting — STM32H743 @ 480 MHz");

    /* Initialize AE comparator (always-on, very low power) */
    ret = ae_always_on_init();
    if (ret < 0) {
        LOG_ERR("AE init failed: %d", ret);
        return ret;
    }

    /* Initialize high-speed ADC for AE burst capture */
    adc_dev = DEVICE_DT_GET(DT_NODELABEL(adc1));
    if (!device_is_ready(adc_dev)) {
        LOG_ERR("ADC not ready");
        return -ENODEV;
    }
    adc_configure_burst(adc_dev, AE_SAMPLE_RATE_HZ, 12, ae_dma_complete_cb);

    /* Initialize SPI for piezoelectric accelerometer */
    spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi1));
    if (!device_is_ready(spi_dev)) {
        LOG_ERR("SPI not ready");
        return -ENODEV;
    }
    vibration_hf_init(spi_dev, VIB_SAMPLE_RATE_HZ);

    /* Initialize radio (TI CC1352P) */
    ret = radio_init();
    if (ret < 0) {
        LOG_ERR("Radio init failed: %d", ret);
        return ret;
    }

    /* Create threads */
    k_thread_create(&ae_thread_data, ae_stack, AE_STACK_SIZE,
                    ae_thread_fn, NULL, NULL, NULL,
                    2, 0, K_NO_WAIT);
    k_thread_name_set(&ae_thread_data, "ae_features");

    k_thread_create(&vib_thread_data, vib_stack, VIB_STACK_SIZE,
                    vib_thread_fn, NULL, NULL, NULL,
                    3, 0, K_NO_WAIT);
    k_thread_name_set(&vib_thread_data, "vib_pipeline");

    k_thread_create(&radio_thread_data, radio_stack, RADIO_STACK_SIZE,
                    radio_thread_fn, NULL, NULL, NULL,
                    1, 0, K_NO_WAIT);
    k_thread_name_set(&radio_thread_data, "radio_tx");

    LOG_INF("All threads started. AE monitoring active.");

    /*
     * Main thread enters idle / power management loop.
     * The AE comparator remains active in Stop mode; all other
     * peripherals can be clock-gated until needed.
     */
    while (1) {
        pm_state_force(0U, &(struct pm_state_info){PM_STATE_SUSPEND_TO_IDLE, 0, 0});
        k_sleep(K_FOREVER);
    }

    return 0;
}
