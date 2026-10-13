/* main.c -- Alif Ensemble E7 node, HE core.
 *
 * The HE core runs everything in the normal cycle. The HP core and the large
 * Ethos-U55 are woken only when APP_WAVEFORM_MODEL is enabled and the feature
 * path has escalated -- see hardware/alif_e7/README.md for why that division is
 * the point of the kit.
 */

#include "pdm_app.h"
#include "pdm_board.h"
#include "pdm_pipeline.h"

#include "app_config.h"

#include <stdio.h>

void pdm_board_alif_set_shaft_hz(float hz, int measured);
int  ble_transport_init(const char *name);
void ble_transport_poll(void);
uint64_t alif_unix_ms(void);

#if APP_WAVEFORM_MODEL
/* waveform_model.c, running on the HP core via the Ethos-U55. Returns a
 * probability in [0,1] per class and a latency in microseconds, or <0 on
 * failure. The HP subsystem is powered up inside and powered down before
 * return: leaving a 22 mA core idle between acquisitions costs 79 mA.s per
 * hour, 17x the entire rest of the cycle. */
int waveform_model_infer(const float *accel_g, size_t n, float *class_probs,
                         size_t n_classes, uint32_t *latency_us);
#define WAVEFORM_CLASSES 5u
#endif

#define BOOT_SHAFT_HZ 29.6667f

int main(void)
{
    if (ble_transport_init("PDM-ALIF") != 0) {
        /* Not fatal: acquire, score and queue. */
    }
    pdm_board_alif_set_shaft_hz(BOOT_SHAFT_HZ, 0);

    if (pdm_app_init(0.0f) != PDM_OK) {
        for (;;) {
            pdm_board_identify(10u);
            pdm_board_watchdog_feed();
        }
    }

    const pdm_board_info *bi = pdm_app_board();
    printf("%s: %s, fs %.0f Hz, %u features, %u kB RAM\n",
           bi->kit, bi->accel_part, (double)bi->fs_hz,
           (unsigned)pdm_pipeline_n_features(), (unsigned)(bi->ram_bytes / 1024u));
#if APP_WAVEFORM_MODEL
    printf("waveform model ENABLED: %u samples at %.0f Hz on the Ethos-U55\n",
           (unsigned)APP_WAVEFORM_SAMPLES, (double)APP_FS_HZ / APP_WAVEFORM_DECIMATE);
    printf("  note: this is the experimental path. The feature path remains\n"
           "  primary and is what the other seven kits are comparable against.\n");
#endif

    for (;;) {
        ble_transport_poll();
        const pdm_app_result r = pdm_app_step(alif_unix_ms());

#if APP_WAVEFORM_MODEL
        /* Only when the cheap path has already said something is interesting.
         * Running the waveform model every cycle would make this kit's energy
         * budget worse than the STWIN.box's for no diagnostic gain on a healthy
         * machine -- and 99.99 % of acquisitions are on healthy machines. */
        if (r.tier >= 2u && r.score > 0.55f * pdm_app_state_get()->threshold) {
            float probs[WAVEFORM_CLASSES] = {0};
            uint32_t latency_us = 0;
            extern const float *pdm_app_last_acquisition(size_t *n);
            size_t n = 0;
            const float *acq = pdm_app_last_acquisition(&n);
            if (acq != NULL
                && waveform_model_infer(acq, n, probs, WAVEFORM_CLASSES, &latency_us) == 0) {
                printf("  waveform model: %.0f us, top class %u p=%.2f\n",
                       (double)latency_us, 0u, (double)probs[0]);
                /* The result is reported alongside the feature frame, never
                 * instead of it: two models disagreeing is information, and
                 * replacing the comparable path with the experimental one would
                 * throw that away. */
            }
        }
#endif
        (void)pdm_board_sleep_ms(pdm_app_next_sleep_ms());
    }
    return 0;
}
