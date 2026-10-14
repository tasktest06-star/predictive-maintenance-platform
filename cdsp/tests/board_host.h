/* board_host.h -- test hooks for the host board implementation.
 *
 * Separate from pdm_board.h because these are not part of the board contract:
 * a real kit has no "inject a FIFO overrun every N cycles" entry point.
 */
#ifndef BOARD_HOST_H
#define BOARD_HOST_H

#include <stdint.h>

#include "pdm_board.h"
#include <stdio.h>

void     pdm_board_host_configure(FILE *samples, uint32_t tx_fail_every,
                                  uint32_t tx_fail_until, uint32_t overrun_every,
                                  uint16_t batt_mv, uint16_t batt_decay_mv,
                                  float shaft_hz, int shaft_estimated,
                                  int selftest_fail);
uint64_t pdm_board_host_sleep_total_ms(void);
#if defined(APP_N_AXES) && APP_N_AXES > 1u
/* Proves the multi-axis path released the raw block once per acquisition. A
 * node that forgets to release it re-analyses the same samples forever, which
 * on a dashboard is a perfectly flat, perfectly healthy trend. */
uint32_t pdm_board_host_release_count(void);
#endif

#endif /* BOARD_HOST_H */
