/* pdm_order.h -- angular resampling (order tracking) for the RA8D1 node.
 *
 * Kit-local, not part of the shared core, for one reason: it needs a
 * tachometer, and this is the only kit in the lineup that has one.
 *
 * THE PROBLEM IT SOLVES
 *
 * A plain FFT assumes the shaft speed is constant across the record. On a
 * VFD-driven machine it is not: 1-3 % drift over a 300 ms record is normal, and
 * more under load changes. The effect on a spectrum is that a line at order k
 * smears over k times the speed variation, so the error grows with order --
 * which is exactly backwards for condition monitoring, because the diagnostic
 * content (bearing orders at 3-11x, gear mesh at 20-100x) lives at high orders
 * while the 1x line, the only one that stays sharp, is the least informative.
 *
 * MEASURED, in tests/test_order.c, on this kit's 307 ms record at 4 % drift:
 *
 *     order  6 (bearing)     -1.50 dB,  2 bins wide,  resampling gains 1.19x
 *     order 40 (gear mesh)   -5.67 dB,  6 bins wide,  resampling gains 1.92x
 *
 * The first version of this comment claimed 8-10 dB at order 6. It was wrong by
 * a lot, and in the flattering direction. At a 307 ms record the FFT bins are
 * 3.26 Hz and a 6x line at 4 % drift smears over ~7 Hz -- barely wider than a
 * Hann main lobe. Order tracking is nearly pointless for bearing orders here;
 * it pays at gear mesh, blade pass and electrical lines, and it pays more the
 * longer the record.
 *
 * Where it does apply, it matters for a reason beyond amplitude: a trend built
 * on a smeared peak tracks the VFD's duty cycle rather than the machine's
 * condition -- a false alarm every time production changes speed, which is one
 * of the most common reasons a vibration programme gets switched off.
 *
 * A side effect worth having: resampling also removes scallop loss. The
 * steady-shaft control measures -1.16 dB in the frequency domain, where the
 * 174 Hz line falls between bins, against -0.00 dB in the angle domain where it
 * lands exactly on an order bin.
 *
 * THE METHOD
 *
 * Resample the record at constant angular increments instead of constant time
 * increments. Between two revolution marks the angle is taken to advance
 * linearly (standard keyphasor practice); the signal is interpolated at the
 * resulting instants with a 4-point Catmull-Rom kernel. The result is a record
 * in the angle domain whose FFT has axes in orders, where a 6x line is at
 * exactly 6.000 regardless of what the speed did.
 *
 * WHAT IT COSTS
 *
 * ~0.9 ms for 2304 output samples on the M85 at 480 MHz, which is why this kit
 * and not another: the same work on the nRF9160's M33 at 64 MHz would be ~11 ms
 * on top of an 18 ms budget, and on the ESP32-S3 it competes with the Wi-Fi
 * stack for the same cores.
 *
 * LIMITS, stated because they decide whether a result is usable:
 *   - Order resolution is samples_per_rev / n_output_samples. At 256 and 2048
 *     that is 0.125 orders; a BPFO at 3.58 orders needs ~0.05 to separate from
 *     3.5x, so 4096 output samples (16 revolutions) is the real minimum for
 *     bearing work. At 29.7 Hz that is a 0.54 s record, not 0.31 s.
 *   - Maximum order is samples_per_rev / 2 = 128. Gear mesh above that aliases.
 *   - Interpolation is not free of error: Catmull-Rom attenuates by ~0.3 dB at
 *     0.25 of the output Nyquist and ~2 dB at 0.4. Keep the orders of interest
 *     below ~half the maximum.
 *   - Below ~0.5 % drift the interpolation error is comparable to the smearing
 *     it removes, so the caller should not resample. APP_ORDER_TRACK_DRIFT_PCT.
 */

#ifndef PDM_ORDER_H
#define PDM_ORDER_H

#include <stddef.h>
#include <stdint.h>

#include "pdm_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resample `x` (n samples at fs_hz) into the angle domain.
 *
 * `rev_s` holds the times of n_marks revolution marks in seconds on the same
 * clock as sample 0 of `x`, strictly ascending. `samples_per_rev` must be even.
 *
 * Writes (n_marks - 1) * samples_per_rev samples, capped at out_cap, and sets
 * *out_n. Returns PDM_ERR_ARG on a non-ascending or too-short mark list, which
 * is what a missed tacho pulse looks like -- refusing is correct, because a
 * dropped revolution shifts every subsequent angle by a full turn and the
 * resulting spectrum is confidently wrong.
 */
pdm_status pdm_order_resample(const float *x, uint32_t n, float fs_hz,
                              const float *rev_s, uint16_t n_marks,
                              uint16_t samples_per_rev,
                              float *out, uint32_t out_cap, uint32_t *out_n);

/* Peak-to-peak speed variation over the marks, in percent of the mean. The
 * caller compares this against APP_ORDER_TRACK_DRIFT_PCT to decide whether
 * resampling is worth doing at all. Returns a negative value if n_marks < 3. */
float pdm_order_speed_drift_pct(const float *rev_s, uint16_t n_marks);

/* Mean shaft rate over the marks, Hz. Negative if n_marks < 2. */
float pdm_order_mean_hz(const float *rev_s, uint16_t n_marks);

#ifdef __cplusplus
}
#endif
#endif /* PDM_ORDER_H */
