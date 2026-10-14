/* board_ra8d1.h -- this kit's extensions beyond the pdm_board.h contract.
 *
 * Everything here exists because the RA8D1 node has a tachometer and three
 * live axes, and no other kit in the lineup has either. The shared headers
 * stay clean; the kit-specific surface is declared here and nowhere else.
 */

#ifndef BOARD_RA8D1_H
#define BOARD_RA8D1_H

#include <stdint.h>

#include "hal_data.h"
#include "pdm_board.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Peak-to-peak shaft-speed variation across the last record, percent of mean.
 * Negative if the tacho produced too few marks to say. */
float pdm_board_ra8d1_speed_drift_pct(void);

/* Revolution-mark times in seconds, relative to the first mark of the record.
 * Returns the number written. Fewer than 4 means no order analysis. */
uint16_t pdm_board_ra8d1_marks(float *dst, uint16_t cap);

/* The channel-0 record the pipeline just analysed, in g. Valid until the next
 * acquisition; NULL if there is nothing. Used by the order-domain pass, which
 * deliberately runs after the frames are away. */
const float *pdm_board_ra8d1_last_record(uint32_t *n);

/* Fallback shaft rate, used only until the tacho produces a reading. A node
 * that refused to measure anything before its first keyphasor pulse would
 * never start on a machine whose pickup is misaligned -- and a misaligned
 * pickup is a commissioning problem, not a reason to ship no data. */
void pdm_board_ra8d1_set_nominal_hz(float hz);

/* GPT input-capture ISR, registered through the FSP configurator. */
void tacho_capture_callback(timer_callback_args_t *p_args);

/* Unix milliseconds from the RTC, which the sub-clock keeps running through
 * Software Standby. */
uint64_t ra8_unix_ms(void);

/* Provided by the FSP-generated port; declared here so the driver does not
 * carry local externs.
 *
 * ra8_gpt_source_hz is read rather than hardcoded because the clock tree is
 * configured in e2 studio: a change there would otherwise silently rescale
 * every speed reading this node has ever produced. */
uint32_t ra8_gpt_source_hz(void);
void     ra8_enter_standby_ms(uint32_t ms);
uint64_t ra8_timegm(const rtc_time_t *tm);
int      ble_transport_send(const uint8_t *data, uint32_t len, int16_t *rssi);

#ifdef __cplusplus
}
#endif
#endif /* BOARD_RA8D1_H */
