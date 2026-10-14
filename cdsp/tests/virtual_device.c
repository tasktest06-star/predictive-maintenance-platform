/* virtual_device.c -- the real firmware, driven from a file.
 *
 *   virtual_device --samples run.txt --cycles 90 [options] > frames.bin
 *
 * Runs pdm_app_step() -- the same application loop a microcontroller runs, over
 * the same pipeline and the same wire encoder -- and writes length-prefixed
 * 300-byte frames to stdout. A Python test decodes them with
 * pdmcore.io.payload and feeds them to the ingest service, which closes the
 * loop from C firmware to cloud diagnosis without any hardware.
 *
 * Progress and faults go to stderr so stdout stays a clean binary stream.
 */

#include "pdm_app.h"
#include "pdm_payload.h"
#include "pdm_pipeline.h"
#include "board_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app_config.h"

static const char *action_name(pdm_app_action a)
{
    switch (a) {
    case PDM_APP_FRAME_READY:        return "frame";
    case PDM_APP_SNAPSHOT_REQUESTED: return "snapshot";
    case PDM_APP_NO_SPEED:           return "no-speed";
    case PDM_APP_ACQUIRE_FAIL:       return "acquire-fail";
    case PDM_APP_SENSOR_FAIL:        return "sensor-fail";
    case PDM_APP_PIPELINE_FAIL:      return "pipeline-fail";
    case PDM_APP_ENCODE_FAIL:        return "encode-fail";
    case PDM_APP_BATTERY_CRITICAL:   return "battery-critical";
    default:                         return "?";
    }
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    uint32_t cycles = 10u, tx_fail_every = 0u, tx_fail_until = 0u;
    uint32_t overrun_every = 0u, interval_s = 3600u;
    uint16_t batt = 3700u, decay = 0u;
    float shaft = 29.6667f, threshold = 4.0f;
    int estimated = 0, selftest_fail = 0, quiet = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--samples") && i + 1 < argc) {
            path = argv[++i];
        } else if (!strcmp(argv[i], "--cycles") && i + 1 < argc) {
            cycles = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--tx-fail-every") && i + 1 < argc) {
            tx_fail_every = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--tx-fail-until") && i + 1 < argc) {
            tx_fail_until = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--overrun-every") && i + 1 < argc) {
            overrun_every = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--battery-mv") && i + 1 < argc) {
            batt = (uint16_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--battery-decay-mv") && i + 1 < argc) {
            decay = (uint16_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--shaft-hz") && i + 1 < argc) {
            shaft = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) {
            threshold = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--interval-s") && i + 1 < argc) {
            interval_s = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--speed-estimated")) {
            estimated = 1;
        } else if (!strcmp(argv[i], "--selftest-fail")) {
            selftest_fail = 1;
        } else if (!strcmp(argv[i], "--quiet")) {
            quiet = 1;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: virtual_device --samples FILE [--cycles N] "
                        "[--tx-fail-every N] [--tx-fail-until N] [--overrun-every N] "
                        "[--battery-mv MV] [--battery-decay-mv MV] "
                        "[--shaft-hz HZ] [--threshold T] [--interval-s S] "
                        "[--speed-estimated] [--selftest-fail] [--quiet]\n");
        return 2;
    }
    FILE *fs = fopen(path, "r");
    if (fs == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }

    pdm_board_host_configure(fs, tx_fail_every, tx_fail_until, overrun_every,
                             batt, decay, shaft, estimated, selftest_fail);
    if (pdm_app_init(threshold) != PDM_OK) {
        fprintf(stderr, "pdm_app_init failed\n");
        return 1;
    }
    pdm_app_set_config(interval_s, threshold);

    const pdm_board_info *bi = pdm_app_board();
    if (!quiet) {
        fprintf(stderr, "kit %s, accel %s, fs %.1f Hz, %u features, %zu B frames\n",
                bi->kit, bi->accel_part, (double)bi->fs_hz,
                (unsigned)pdm_pipeline_n_features(),
                PDM_PAYLOAD_HEADER_SIZE + 4u * (size_t)PDM_N_FEATURES + 4u);
    }

    uint64_t now_ms = 1760000000000ULL;
    for (uint32_t c = 0; c < cycles; ++c) {
        const pdm_app_result r = pdm_app_step(now_ms);
        if (!quiet) {
            fprintf(stderr,
                    "cycle %4u  %-16s tier %u  score %8.3f  %s%s  queue %u  drained %u\n",
                    c, action_name(r.action), (unsigned)r.tier, (double)r.score,
                    r.transmitted ? "tx" : "  ", r.queued ? " queued" : "",
                    (unsigned)pdm_board_queue_depth(), (unsigned)r.drained);
        }
        now_ms += (uint64_t)pdm_app_next_sleep_ms();
        (void)pdm_board_sleep_ms(pdm_app_next_sleep_ms());
    }

    const pdm_app_state *st = pdm_app_state_get();
    if (!quiet) {
        fprintf(stderr,
                "\n%u cycles: %u sent, %u queued, %u dropped, %u overruns, "
                "battery %u mV, simulated sleep %.1f h\n",
                (unsigned)st->cycles, (unsigned)st->frames_sent,
                (unsigned)st->frames_queued, (unsigned)st->frames_dropped,
                (unsigned)st->overruns, (unsigned)st->battery_mv,
                (double)pdm_board_host_sleep_total_ms() / 3600000.0);
#if defined(APP_N_AXES) && APP_N_AXES > 1u
        fprintf(stderr, "%u axes, raw block released %u times\n",
                (unsigned)APP_N_AXES, (unsigned)pdm_board_host_release_count());
#endif
    }
    fclose(fs);
    return 0;
}
