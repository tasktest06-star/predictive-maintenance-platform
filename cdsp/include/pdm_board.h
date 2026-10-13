/* pdm_board.h -- the interface every hardware kit implements.
 *
 * The contract that keeps the pipeline portable. `pdm_pipeline.c` never calls a
 * vendor SDK; the kit's `board_*.c` never computes a feature. Everything that
 * differs between an STM32U5, an nRF9160, an ESP32-S3, an Alif E7 and an RA8D1
 * lives behind these fifteen functions.
 *
 * Design rules that the five implementations all follow:
 *
 * **Blocking acquisition, DMA underneath.** `pdm_board_acquire` does not
 * return until the block is captured or has failed. The MCU sleeps inside it
 * while DMA fills the buffer -- a callback-based API would be more flexible and
 * would also let the application start a radio transmission mid-acquisition,
 * which on every one of these parts couples supply noise straight into the
 * accelerometer's analogue front end. Serialising it in the API makes that
 * mistake impossible rather than merely discouraged.
 *
 * **Failures are reported, never hidden.** A FIFO overrun, a failed self-test
 * or a clipped sample sets a flag the cloud can see. Silently returning
 * plausible samples is the one behaviour that is genuinely worse than returning
 * nothing.
 *
 * **No implicit power management.** The application decides when the sensor is
 * powered, because that decision depends on the acquisition interval (see
 * APP_SENSOR_KEEPALIVE_THRESHOLD_S) and the application is the only layer that
 * knows it.
 */

#ifndef PDM_BOARD_H
#define PDM_BOARD_H

#include <stddef.h>
#include <stdint.h>

#include "pdm_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PDM_BOARD_OK = 0,
    PDM_BOARD_ERR_BUS = -1,        /* SPI/I2C transaction failed */
    PDM_BOARD_ERR_ID = -2,         /* WHO_AM_I mismatch: wrong or dead part */
    PDM_BOARD_ERR_SELFTEST = -3,
    PDM_BOARD_ERR_OVERRUN = -4,    /* FIFO overran: samples were lost */
    PDM_BOARD_ERR_TIMEOUT = -5,
    PDM_BOARD_ERR_NOT_READY = -6
} pdm_board_status;

typedef struct {
    const char *kit;               /* APP_KIT_NAME */
    const char *accel_part;
    uint64_t    device_uid;        /* MCU unique ID, truncated to 64 bits */
    float       fs_hz;             /* actual, measured rate -- see below */
    float       full_scale_g;
    uint8_t     adc_bits;
    uint32_t    flash_bytes;
    uint32_t    ram_bytes;
} pdm_board_info;

/* Bring up clocks, buses, sensors. Runs the accelerometer self-test and returns
 * PDM_BOARD_ERR_SELFTEST if it fails -- a node that cannot verify its own
 * sensor must say so at boot rather than stream plausible noise for a year. */
pdm_board_status pdm_board_init(pdm_board_info *info);

/* Power the accelerometer up or down. Settling is the caller's problem (it owns
 * the interval policy); `pdm_board_accel_settle_ms` reports how long. */
pdm_board_status pdm_board_accel_power(int on);
uint32_t         pdm_board_accel_settle_ms(void);

/* Capture `n` samples of the measurement axis into `out`, in g.
 *
 * `fs_hz_actual` receives the rate the hardware really used. This matters: the
 * IIS3DWB's 26.667 kHz is nominal and its internal oscillator is +-3 %, which
 * at 3 % puts a 106 Hz BPFO line 3.2 Hz off where the matcher expects it --
 * inside the +-2 % tolerance only by luck. Kits that can measure their true
 * rate against a crystal do so and report it; the rest report the nominal value
 * and their README says which.
 *
 * Returns PDM_BOARD_ERR_OVERRUN if the FIFO overran at any point. The block is
 * then NOT valid: the samples are a concatenation of two different times, and a
 * spectrum computed from it contains a step discontinuity that looks exactly
 * like an impact. */
pdm_board_status pdm_board_acquire(float *out, size_t n, float *fs_hz_actual);

/* Shaft speed. Returns PDM_BOARD_ERR_NOT_READY when neither a tacho nor a
 * configured nominal speed is available, because a feature vector computed
 * against a guessed speed is worse than no feature vector. `estimated` is set
 * when the value came from a tacholess estimate rather than a measurement --
 * the frame carries that as PDM_FRAME_SPEED_ESTIMATED. */
pdm_board_status pdm_board_shaft_hz(float *shaft_hz, int *estimated);

pdm_board_status pdm_board_temperature_c(float *temp_c);
pdm_board_status pdm_board_battery_mv(uint16_t *mv);
int16_t          pdm_board_rssi_dbm(void);

/* Transmit one frame. Returns PDM_BOARD_OK only when the frame is known to have
 * left the device; anything else means the caller must queue it. */
pdm_board_status pdm_board_transmit(const uint8_t *frame, size_t len);

/* Store-and-forward queue in non-volatile storage. A node whose gateway is down
 * for a fortnight must not lose the fortnight -- that is frequently the most
 * interesting fortnight, because whatever took the gateway down may have taken
 * the machine with it. */
pdm_board_status pdm_board_queue_push(const uint8_t *frame, size_t len);
pdm_board_status pdm_board_queue_pop(uint8_t *frame, size_t cap, size_t *len);
uint32_t         pdm_board_queue_depth(void);

/* Sleep until the next acquisition. The implementation picks the deepest mode
 * that retains what it needs; `pdm_board_sleep_ms` may return early if woken by
 * a downlink or an interrupt, and returns the milliseconds actually slept so
 * the caller can keep its own schedule. */
uint32_t pdm_board_sleep_ms(uint32_t ms);

/* Blink/beep for commissioning. Finding one sensor among forty identical ones
 * bolted to a plant wall is otherwise a genuinely hard problem, and it is a
 * commissioning cost nobody budgets for. */
void pdm_board_identify(uint32_t seconds);

/* Watchdog. Fed once per main-loop iteration; the timeout is set long enough to
 * cover the longest acquisition plus the longest radio attempt. */
void pdm_board_watchdog_feed(void);

#ifdef __cplusplus
}
#endif
#endif /* PDM_BOARD_H */
