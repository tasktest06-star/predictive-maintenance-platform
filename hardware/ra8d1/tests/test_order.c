/* test_order.c -- does angular resampling actually recover a smeared order?
 *
 * The claim is quantitative, so it is measured here rather than asserted in a
 * README. Method: synthesise a signal whose phase is the integral of a linearly
 * swept shaft rate, so the revolution-mark times are known in closed form.
 * Compare the FFT peak at a given order before and after resampling, with both
 * spectra normalised by their own length so the two FFT sizes are comparable.
 *
 * WHAT IT FOUND, and what the deck and README now say instead of what they
 * said first:
 *
 * Smearing is proportional to (order x drift x record length), and at this
 * kit's 307 ms record the first factor is the only one that is large. A 6x line
 * at 4 % drift loses only ~1.5 dB, because a 8192-point FFT at 26.667 kHz has
 * 3.26 Hz bins and the smear is ~7 Hz -- barely two bins, which a Hann main
 * lobe is anyway. Order tracking is nearly pointless there.
 *
 * At order 40 -- gear mesh, blade pass, the orders an analyst actually fights
 * with -- the same drift smears the line over ~46 Hz, 14 bins, and the peak
 * loses most of its amplitude. That is where resampling pays, and the test
 * measures both so the dependence is on the record rather than in a sentence.
 *
 *     cc -std=c99 -O2 -Iinclude -I<kit>/firmware test_order.c pdm_order.c \
 *        libpdmdsp.a -lm
 */

#include <math.h>
#include <stdio.h>

#include "pdm_dsp.h"
#include "pdm_order.h"

#define FS        26667.0f
#define N_IN      8192u
#define SPR       256u
#define N_OUT     2048u
#define TWO_PI    6.28318530718f

/* A unit-amplitude sine on an exact bin, windowed by Hann and normalised by
 * 2/N, peaks at the window's coherent gain: 0.5. Both domains are compared
 * against that same ideal, which is what makes the two FFT sizes comparable.
 * (The first version of this file used 0.25 -- amplitude rather than
 * amplitude x coherent gain -- and every "ideal" ratio came out at exactly
 * 2.000, which is how the error announced itself.) */
#define IDEAL_PEAK 0.5f

static int s_fail;
static int s_checks;

static void check(int ok, const char *what, double got, double want)
{
    ++s_checks;
    if (!ok) {
        ++s_fail;
        printf("  FAIL  %-56s got %.4f want %.4f\n", what, got, want);
    } else {
        printf("  ok    %-56s %.4f\n", what, got);
    }
}

/* Shaft rate f(t) = f0 + rate*t, so phase in revolutions is
 * theta(t) = f0*t + rate*t^2/2. Inverting for theta = k gives the mark times. */
static float mark_time(float f0, float rate, float k)
{
    if (fabsf(rate) < 1e-9f) {
        return k / f0;
    }
    return (-f0 + sqrtf(f0 * f0 + 2.0f * rate * k)) / rate;
}

static void spectrum(const float *x, uint32_t n, float *mag)
{
    static float work[8192];
    static float tw[8192 + 2];
    static uint16_t br[4096];
    pdm_rfft_ctx ctx;
    if (pdm_rfft_init(&ctx, (uint16_t)n, tw, br) != PDM_OK) {
        printf("  FAIL  rfft init n=%u\n", (unsigned)n);
        ++s_fail;
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const float w = 0.5f - 0.5f * cosf(TWO_PI * (float)i / (float)n);
        work[i] = x[i] * w;
    }
    (void)pdm_rfft(&ctx, work);
    (void)pdm_rfft_magnitude(&ctx, work, mag);
    /* Length-normalised, single-sided: *2/n for a real sine. */
    for (uint32_t i = 0; i <= n / 2u; ++i) {
        mag[i] *= 2.0f / (float)n;
    }
}

/* Peak and -3 dB width within +-3 orders of `order`. The window is specified in
 * orders, not bins, so it means the same thing in both domains. */
static void peak_and_width(const float *mag, uint32_t nbins, float bins_per_order,
                           float order, float *peak, int *width)
{
    int lo = (int)((order - 3.0f) * bins_per_order);
    int hi = (int)((order + 3.0f) * bins_per_order);
    if (lo < 1) {
        lo = 1;
    }
    if ((uint32_t)hi >= nbins) {
        hi = (int)nbins - 1;
    }
    float pk = 0.0f;
    for (int i = lo; i <= hi; ++i) {
        if (mag[i] > pk) {
            pk = mag[i];
        }
    }
    int w = 0;
    for (int i = lo; i <= hi; ++i) {
        if (mag[i] >= pk * 0.7071f) {
            ++w;
        }
    }
    *peak = pk;
    *width = w;
}

typedef struct {
    float peak_time;
    float peak_angle;
    int   width_time;
    int   width_angle;
    float drift_pct;
    float mean_hz;
    uint32_t n_angle;
} order_case;

/* One (order, drift) case, end to end. */
static order_case run_case(float order, float f0, float sweep_pct)
{
    static float x[N_IN];
    static float marks[64];
    static float resampled[N_OUT + SPR];
    static float mag_t[N_IN / 2u + 1u];
    static float mag_a[N_OUT / 2u + 1u];
    order_case c;

    const float t_rec = (float)(N_IN - 1u) / FS;
    const float rate = f0 * (sweep_pct / 100.0f) / t_rec;

    for (uint32_t i = 0; i < N_IN; ++i) {
        const float t = (float)i / FS;
        const float theta = f0 * t + 0.5f * rate * t * t;
        x[i] = sinf(TWO_PI * order * theta);
    }
    uint16_t n_marks = 0;
    for (float k = 0.0f; n_marks < 64u; k += 1.0f) {
        const float tk = mark_time(f0, rate, k);
        if (tk > t_rec) {
            break;
        }
        marks[n_marks++] = tk;
    }
    c.drift_pct = pdm_order_speed_drift_pct(marks, n_marks);
    c.mean_hz = pdm_order_mean_hz(marks, n_marks);

    spectrum(x, N_IN, mag_t);
    peak_and_width(mag_t, N_IN / 2u + 1u, (float)N_IN / FS * c.mean_hz, order,
                   &c.peak_time, &c.width_time);

    uint32_t got = 0;
    (void)pdm_order_resample(x, N_IN, FS, marks, n_marks, (uint16_t)SPR,
                             resampled, N_OUT, &got);
    uint32_t n_a = 1024u;
    while (n_a * 2u <= got && n_a < N_OUT) {
        n_a *= 2u;
    }
    c.n_angle = n_a;
    spectrum(resampled, n_a, mag_a);
    peak_and_width(mag_a, n_a / 2u + 1u, (float)n_a / (float)SPR, order,
                   &c.peak_angle, &c.width_angle);
    return c;
}

static void report(const char *label, float order, const order_case *c)
{
    printf("  %s order %2.0f:  time %.4f (%+.2f dB) over %d bins   |   "
           "angle %.4f (%+.2f dB) over %d bin%s   |   gain %.2fx\n",
           label, (double)order,
           (double)c->peak_time, 20.0 * log10((double)(c->peak_time / IDEAL_PEAK)),
           c->width_time,
           (double)c->peak_angle, 20.0 * log10((double)(c->peak_angle / IDEAL_PEAK)),
           c->width_angle, c->width_angle == 1 ? "" : "s",
           (double)(c->peak_angle / c->peak_time));
}

int main(void)
{
    const float f0 = 29.0f;
    const float sweep = 4.0f;

    printf("angular resampling: %.0f Hz +%.0f %% over %.0f ms, %u samples/rev\n",
           (double)f0, (double)sweep, (double)((float)(N_IN - 1u) / FS * 1000.0f),
           (unsigned)SPR);

    /* --- the tacho estimators ------------------------------------------ */
    const order_case c6 = run_case(6.0f, f0, sweep);
    check(fabsf(c6.drift_pct - sweep) < 0.8f, "drift estimate recovers the sweep [%]",
          (double)c6.drift_pct, (double)sweep);
    check(c6.mean_hz > f0 && c6.mean_hz < f0 * 1.05f, "mean rate estimate [Hz]",
          (double)c6.mean_hz, (double)(f0 * 1.02f));

    /* --- low order: resampling is nearly pointless at this record length - */
    report("low ", 6.0f, &c6);
    check(c6.peak_angle > 0.95f * IDEAL_PEAK, "angle domain is near-ideal at order 6",
          (double)(c6.peak_angle / IDEAL_PEAK), 0.95);
    check(c6.peak_time > 0.75f * IDEAL_PEAK,
          "order 6 is barely smeared: a 307 ms record has 3.3 Hz bins",
          (double)(c6.peak_time / IDEAL_PEAK), 0.75);
    /* The honest statement of what resampling buys here, as a ratio. It is
     * small, and the test asserts that it is small -- a test that demanded a
     * large gain at order 6 would have to be rigged to pass. */
    check(c6.peak_angle / c6.peak_time < 1.35f,
          "gain at order 6 is modest (<1.35x), not the headline",
          (double)(c6.peak_angle / c6.peak_time), 1.35);

    /* --- high order: this is where it pays ------------------------------ */
    const order_case c40 = run_case(40.0f, f0, sweep);
    report("high", 40.0f, &c40);
    check(c40.peak_angle > 0.95f * IDEAL_PEAK, "angle domain holds up at order 40",
          (double)(c40.peak_angle / IDEAL_PEAK), 0.95);
    check(c40.peak_time < 0.65f * IDEAL_PEAK,
          "order 40 loses about half its peak to smearing",
          (double)(c40.peak_time / IDEAL_PEAK), 0.65);
    check(c40.peak_angle > 1.7f * c40.peak_time,
          "resampling restores most of it (>1.7x)",
          (double)(c40.peak_angle / c40.peak_time), 1.7);
    check(c40.width_angle < c40.width_time,
          "and narrows the line [bins above -3 dB]",
          (double)c40.width_angle, (double)c40.width_time);
    /* The dependence on order is the actionable finding: it tells an engineer
     * WHEN to enable this, which a single pass/fail number does not. */
    check((c40.peak_angle / c40.peak_time) > 1.4f * (c6.peak_angle / c6.peak_time),
          "the gain grows with order, as the theory says",
          (double)((c40.peak_angle / c40.peak_time) / (c6.peak_angle / c6.peak_time)),
          1.4);

    /* --- negative controls --------------------------------------------- */
    static float marks[64];
    static float x[N_IN];
    static float out[N_OUT];
    for (uint16_t k = 0; k < 10u; ++k) {
        marks[k] = (float)k / f0;
    }
    for (uint32_t i = 0; i < N_IN; ++i) {
        x[i] = sinf(TWO_PI * 6.0f * f0 * (float)i / FS);
    }
    float bad[64];
    for (uint16_t i = 0; i < 10u; ++i) {
        bad[i] = marks[i];
    }
    bad[5] = bad[6];                        /* a doubled tacho pulse: dt == 0 */
    uint32_t bgot = 0;
    check(pdm_order_resample(x, N_IN, FS, bad, 10u, (uint16_t)SPR, out, N_OUT, &bgot)
              == PDM_ERR_ARG,
          "a non-ascending mark list is refused", 0.0, 0.0);
    check(pdm_order_speed_drift_pct(bad, 10u) < 0.0f,
          "drift on a bad mark list is negative, not plausible",
          (double)pdm_order_speed_drift_pct(bad, 10u), -1.0);
    check(pdm_order_resample(x, N_IN, FS, marks, 10u, 255u, out, N_OUT, &bgot)
              == PDM_ERR_ARG,
          "an odd samples_per_rev is refused", 0.0, 0.0);

    /* --- a steady shaft must not be made worse -------------------------- */
    const order_case cs = run_case(6.0f, f0, 0.0f);
    report("flat", 6.0f, &cs);
    check(cs.peak_angle > 0.90f * cs.peak_time,
          "a steady shaft loses <10 % peak to interpolation",
          (double)(cs.peak_angle / cs.peak_time), 0.90);

    printf("\n%s  %d checks\n", s_fail ? "FAIL" : "PASS", s_checks);
    return s_fail ? 1 : 0;
}
