/* The edge/cloud feature-agreement test.
 *
 * Runs the shared tiered pipeline on one acquisition and compares its 63-float
 * feature vector against the vector the Python extractor produced from the same
 * samples. This is the test that makes "the MCU and the cloud compute the same
 * features" a verified statement rather than an intention.
 *
 * Scope, stated precisely:
 *
 *   indices 0..42  (time, band, shape, iso, order)  compared BY VALUE
 *   indices 43..62 (bearing)                        checked BY PROPERTY
 *
 * The bearing block is excluded from the numeric comparison on purpose: the
 * device uses rectify-and-low-pass for the envelope where the cloud uses the
 * analytic signal, because a 90-degree all-pass costs either RAM or a second
 * FFT on an MCU. Modulation spectra agree up to a DC term and a 2/pi factor,
 * and all the bearing statistics are ratios against a local floor, so the
 * *decision* is unaffected while the absolute numbers are not comparable. See
 * cdsp/README.md. What is checked instead is the property the diagnosis
 * actually depends on: the planted BPFO family must score far above the other
 * three modes.
 */

/* PDM_WANT_FEATURE_NAMES must be defined before the layout header is pulled in
 * by pdm_pipeline.h, not after -- the names are inside an #ifdef. */
#define PDM_WANT_FEATURE_NAMES

#include "app_config.h"
#include "pdm_pipeline.h"
#include "golden.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_checks = 0;

static void check_rel(const char *what, double got, double want, double tol)
{
    ++g_checks;
    const double scale = (fabs(want) > 1e-9) ? fabs(want) : 1.0;
    const double err = fabs(got - want) / scale;
    if (err > tol || !isfinite(got)) {
        printf("  FAIL %-34s got %+.6e want %+.6e (rel %.2e, tol %.1e)\n",
               what, got, want, err, tol);
        ++g_fail;
    }
}

static void expect(const char *what, int condition, const char *detail)
{
    ++g_checks;
    if (!condition) {
        printf("  FAIL %-34s %s\n", what, detail);
        ++g_fail;
    }
}

int main(void)
{
    printf("pipeline: edge/cloud feature agreement\n");
    printf("======================================\n");

    if (pdm_pipeline_init() != PDM_OK) {
        printf("  FAIL pdm_pipeline_init\n");
        return 1;
    }
    expect("feature count", pdm_pipeline_n_features() == GOLD_KIT_NFEAT,
           "pipeline and golden vector disagree on the feature count");

    pdm_pipeline_result res;
    memset(&res, 0, sizeof(res));
    /* Force escalation to tier 2 so the bearing block is computed: this test is
     * about agreement, not about the tiering policy, which is exercised
     * separately below. */
    res.threshold_hint = 0.0f;

    const pdm_status st = pdm_pipeline_run(gold_kit_accel, GOLD_KIT_NACQ,
                                           GOLD_KIT_SHAFT, &res);
    if (st != PDM_OK) {
        printf("  FAIL pdm_pipeline_run returned %d\n", (int)st);
        return 1;
    }
    printf("  tier reached %u, flags 0x%04x, score %.4f, ISO zone %c\n",
           (unsigned)res.tier, (unsigned)res.flags, (double)res.score,
           (char)('A' + res.iso_zone));
    expect("reached tier 2", res.tier >= 2u, "envelope stage did not run");
    expect("not clipped", (res.flags & PDM_FRAME_CLIPPED) == 0u,
           "the reference acquisition should not clip at +-16 g");

    /* ---- blocks compared by value ----------------------------------- */
    /* Tolerances per block, reflecting what each one accumulates.
     *
     * time:  float32 moments over 4096 samples. kurtosis and skewness are
     *        fourth and third moments and carry the most error.
     * band:  a sum over up to ~1800 bins in double, then a sqrt. Tight.
     * shape: ratios of sums over all bins; the flatness term is a product of
     *        4096 logs, so it is the loosest here.
     * iso:   1/omega weighting concentrates the error at low frequency where
     *        the bins are largest, so it is tighter than band.
     * order: narrow bands, few bins each, so quantisation of the band edges
     *        onto the bin grid dominates.
     */
    struct { const char *name; int lo, n; double tol; } blocks[] = {
        {"time",  PDM_IDX_TIME,  PDM_N_TIME,  5e-4},
        {"band",  PDM_IDX_BAND,  PDM_N_BAND,  2e-3},
        {"shape", PDM_IDX_SHAPE, PDM_N_SHAPE, 5e-3},
        {"iso",   PDM_IDX_ISO,   PDM_N_ISO,   2e-3},
        {"order", PDM_IDX_ORDER, PDM_N_ORDER, 5e-3},
    };
    char label[64];
    for (size_t b = 0; b < sizeof(blocks) / sizeof(blocks[0]); ++b) {
        for (int i = 0; i < blocks[b].n; ++i) {
            const int idx = blocks[b].lo + i;
            snprintf(label, sizeof(label), "%s", pdm_feature_names[idx]);
            check_rel(label, res.features[idx], gold_kit_features[idx], blocks[b].tol);
        }
    }

    /* ---- bearing block: property, not value ------------------------- */
    /* The planted family is BPFO. The statistic that matters is that it wins
     * decisively against the three modes that are not present -- that is the
     * whole basis of the diagnosis, and it must hold on the device's envelope
     * even though the absolute numbers differ from the cloud's. */
    const float outer = res.features[PDM_IDX_BEARING + 0];
    const float inner = res.features[PDM_IDX_BEARING + 4];
    const float ball  = res.features[PDM_IDX_BEARING + 8];
    const float cage  = res.features[PDM_IDX_BEARING + 12];
    const float matched = res.features[PDM_IDX_BEARING + 2];
    printf("  bearing SNR: outer %.2f (matched %.0f), inner %.2f, ball %.2f, cage %.2f\n",
           (double)outer, (double)matched, (double)inner, (double)ball, (double)cage);

    expect("BPFO family detected", outer >= 4.0f,
           "the planted outer-race family did not reach the detection threshold");
    expect("BPFO harmonics matched", matched >= 3.0f,
           "fewer than 3 BPFO harmonics stood above the local floor");
    expect("BPFO beats inner race", outer > 1.5f * inner, "inner race scored too close");
    expect("BPFO beats rolling element", outer > 1.5f * ball, "BSF scored too close");
    expect("BPFO beats cage", outer > 1.5f * cage, "FTF scored too close");
    expect("demod band reported",
           fabsf(res.features[PDM_IDX_BEARING + 18] - APP_DEMOD_LO_HZ) < 1.0f
           && fabsf(res.features[PDM_IDX_BEARING + 19] - APP_DEMOD_HI_HZ) < 1.0f,
           "the frame must carry the band that produced the evidence");

    /* ---- tiering policy --------------------------------------------- */
    /* With a high threshold the pipeline must stop at tier 1 -- that is where
     * the energy saving comes from, so a regression that always escalates is a
     * battery-life bug that no functional test would catch. */
    memset(&res, 0, sizeof(res));
    res.threshold_hint = 1.0e9f;
    if (pdm_pipeline_run(gold_kit_accel, GOLD_KIT_NACQ, GOLD_KIT_SHAFT, &res) != PDM_OK) {
        printf("  FAIL second pdm_pipeline_run\n");
        ++g_fail;
    } else {
        expect("stops at tier 1 when quiet", res.tier == 1u,
               "escalated despite a threshold it cannot reach");
        /* And the bearing block must be left at zero, with the tier telling the
         * cloud that zero means "not computed" rather than "no fault". */
        int bearing_zero = 1;
        for (int i = 0; i < 18; ++i) {
            if (res.features[PDM_IDX_BEARING + i] != 0.0f) {
                bearing_zero = 0;
            }
        }
        expect("bearing block left unset", bearing_zero,
               "tier 1 wrote bearing features it did not compute");
    }

    /* ---- input validation ------------------------------------------- */
    memset(&res, 0, sizeof(res));
    expect("rejects zero shaft speed",
           pdm_pipeline_run(gold_kit_accel, GOLD_KIT_NACQ, 0.0f, &res) == PDM_ERR_ARG,
           "order and bearing features are meaningless without a speed");
    expect("rejects a short block",
           pdm_pipeline_run(gold_kit_accel, 64, GOLD_KIT_SHAFT, &res) == PDM_ERR_ARG,
           "accepted fewer samples than the FFT needs");
    expect("rejects NULL",
           pdm_pipeline_run(NULL, GOLD_KIT_NACQ, GOLD_KIT_SHAFT, &res) == PDM_ERR_ARG,
           "accepted a NULL sample buffer");

    printf("======================================\n");
    if (g_fail == 0) {
        printf("PASS  %d checks\n", g_checks);
        return 0;
    }
    printf("FAIL  %d of %d checks\n", g_fail, g_checks);
    return 1;
}
