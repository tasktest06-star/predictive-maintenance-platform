/* board_host.h -- test hooks for the host board implementation.
 *
 * Separate from pdm_board.h because these are not part of the board contract:
 * a real kit has no "inject a FIFO overrun every N cycles" entry point.
 */
#ifndef BOARD_HOST_H
#define BOARD_HOST_H

#include <stdint.h>
#include <stdio.h>

void     pdm_board_host_configure(FILE *samples, uint32_t tx_fail_every,
                                  uint32_t tx_fail_until, uint32_t overrun_every,
                                  uint16_t batt_mv, uint16_t batt_decay_mv,
                                  float shaft_hz, int shaft_estimated,
                                  int selftest_fail);
uint64_t pdm_board_host_sleep_total_ms(void);

#endif /* BOARD_HOST_H */
