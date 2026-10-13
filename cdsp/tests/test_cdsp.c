/* Native cross-check of cdsp against the pdmcore Python reference.
 *
 * Build and run:  make -C cdsp test
 *
 * Every tolerance below is justified, because a tolerance that is merely "big
 * enough" hides the next regression. The reference is float64; cdsp is float32,
 * so the floor is 2^-24 ~ 6e-8 relative per operation, accumulating as
 * sqrt(n_ops) for well-conditioned sums and linearly for badly-conditioned
 * ones.
 */

#include "pdm_dsp.h"
#include "golden.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_checks = 0;

static void report(const char *what, double got, double want, double tol, int relative)
{
    ++g_checks;
    const double err = fabs(got - want);
    const double scale = relative ? (fabs(want) > 1e-12 ? fabs(want) : 1.0) : 1.0;
    if (err / scale > tol || !isfinite(got)) {
        printf("  FAIL %-44s got %+.8e want %+.8e (err %.2e, tol %.1e)\n",
               what, got, want, err / scale, tol);
        ++g_fail;
    }
}

#define CHECK_REL(name, got, want, tol) report((name), (got), (want), (tol), 1)
#define CHECK_ABS(name, got, want, tol) report((name), (got), (want), (tol), 0)

static void section(const char *name)
{
    printf("%s\n", name);
}

/* ------------------------------------------------------------------------- */
static void test_windows(void)
{
    section("windows");
    static const pdm_window_kind kinds[6] = {
        PDM_WIN_RECT, PDM_WIN_HANN, PDM_WIN_HAMMING,
        PDM_WIN_BLACKMAN, PDM_WIN_BLACKMAN_HARRIS, PDM_WIN_FLATTOP
    };
    static const char *names[6] = {
        "rect", "hann", "hamming", "blackman", "blackman_harris", "flattop"
    };
    static const float *heads[6] = {
        gold_win_rect_head, gold_win_hann_head, gold_win_hamming_head,
        gold_win_blackman_head, gold_win_blackman_harris_head, gold_win_flattop_head
    };
    static float w[GOLD_N_FFT];
    char label[96];

    for (int i = 0; i < 6; ++i) {
        pdm_window_info info;
        if (pdm_window_make(kinds[i], w, GOLD_N_FFT, &info) != PDM_OK) {
            printf("  FAIL pdm_window_make(%s)\n", names[i]);
            ++g_fail;
            continue;
        }
        /* Coefficients: a single cosf per term, so ~1e-7 relative. */
        for (int k = 0; k < 32; ++k) {
            snprintf(label, sizeof(label), "%s coeff[%d]", names[i], k);
            CHECK_ABS(label, w[k], heads[i][k], 2e-6);
        }
        /* Calibration constants are accumulated in double, so they should be
         * good to ~1e-7 even over 256 terms. */
        snprintf(label, sizeof(label), "%s coherent_gain", names[i]);
        CHECK_REL(label, info.coherent_gain, gold_win_cg[i], 2e-6);
        snprintf(label, sizeof(label), "%s enbw_bins", names[i]);
        CHECK_REL(label, info.enbw_bins, gold_win_enbw[i], 2e-6);
    }
}

/* ------------------------------------------------------------------------- */
static float g_twiddle[PDM_RFFT_TWIDDLE_LEN(GOLD_N_FFT)];
static uint16_t g_bitrev[PDM_RFFT_BITREV_LEN(GOLD_N_FFT)];
static pdm_rfft_ctx g_fft;

static void test_rfft(void)
{
    section("real FFT");
    if (pdm_rfft_init(&g_fft, GOLD_N_FFT, g_twiddle, g_bitrev) != PDM_OK) {
        printf("  FAIL pdm_rfft_init\n");
        ++g_fail;
        return;
    }
    static float buf[GOLD_N_FFT];
    static float mag[GOLD_N_BINS];
    memcpy(buf, gold_signal, sizeof(buf));
    if (pdm_rfft(&g_fft, buf) != PDM_OK) {
        printf("  FAIL pdm_rfft\n");
        ++g_fail;
        return;
    }
    pdm_rfft_magnitude(&g_fft, buf, mag);

    /* log2(256) = 8 butterfly stages, so ~sqrt(8) * 6e-8 relative on a
     * well-conditioned bin. Bins near the noise floor are differences of large
     * numbers, so they are checked absolutely against the signal's own scale
     * (max |X| is ~100 here, and 1e-3 absolute is 100 dB below it). */
    double max_mag = 0.0;
    for (int k = 0; k < GOLD_N_BINS; ++k) {
        if (gold_rfft_mag[k] > max_mag) {
            max_mag = gold_rfft_mag[k];
        }
    }
    char label[64];
    for (int k = 0; k < GOLD_N_BINS; ++k) {
        snprintf(label, sizeof(label), "rfft mag[%d]", k);
        CHECK_ABS(label, mag[k], gold_rfft_mag[k], max_mag * 3e-6);
    }

    /* Parseval: sum over the single-sided power must equal n * sum(x^2). */
    double acc = (double)mag[0] * mag[0] + (double)mag[GOLD_N_FFT / 2] * mag[GOLD_N_FFT / 2];
    for (int k = 1; k < GOLD_N_FFT / 2; ++k) {
        acc += 2.0 * (double)mag[k] * mag[k];
    }
    double energy = 0.0;
    for (int i = 0; i < GOLD_N_FFT; ++i) {
        energy += (double)gold_signal[i] * gold_signal[i];
    }
    CHECK_REL("rfft Parseval", acc, (double)GOLD_N_FFT * energy, 1e-5);
}

/* ------------------------------------------------------------------------- */
static void test_spectrum(void)
{
    section("spectrum scaling and band RMS");
    static float buf[GOLD_N_FFT];
    static float wnd[GOLD_N_FFT];
    static float mag[GOLD_N_BINS];
    static float spec[GOLD_N_BINS];
    pdm_window_info info;

    pdm_window_make(PDM_WIN_HANN, wnd, GOLD_N_FFT, &info);

    /* Mean-remove then window, exactly as pdmcore.dsp.spectrum does. */
    double mean = 0.0;
    for (int i = 0; i < GOLD_N_FFT; ++i) {
        mean += (double)gold_signal[i];
    }
    mean /= GOLD_N_FFT;
    for (int i = 0; i < GOLD_N_FFT; ++i) {
        buf[i] = (float)((double)gold_signal[i] - mean) * wnd[i];
    }
    pdm_rfft(&g_fft, buf);
    pdm_rfft_magnitude(&g_fft, buf, mag);

    double max_mag = 0.0;
    for (int k = 0; k < GOLD_N_BINS; ++k) {
        if (gold_hann_raw_mag[k] > max_mag) {
            max_mag = gold_hann_raw_mag[k];
        }
    }
    char label[64];
    for (int k = 0; k < GOLD_N_BINS; ++k) {
        snprintf(label, sizeof(label), "hann raw mag[%d]", k);
        CHECK_ABS(label, mag[k], gold_hann_raw_mag[k], max_mag * 3e-6);
    }

    static const pdm_scaling scalings[4] = {
        PDM_SCALE_PEAK, PDM_SCALE_RMS, PDM_SCALE_POWER, PDM_SCALE_PSD
    };
    static const char *snames[4] = {"peak", "rms", "power", "psd"};
    static const float *golds[4] = {
        gold_spec_peak, gold_spec_rms, gold_spec_power, gold_spec_psd
    };
    static const float *gold_bands[4] = {
        gold_band_rms_peak, gold_band_rms_rms, gold_band_rms_power, gold_band_rms_psd
    };
    const float df = GOLD_FS / (float)GOLD_N_FFT;

    for (int s = 0; s < 4; ++s) {
        if (pdm_spectrum_scale(mag, GOLD_N_BINS, GOLD_N_FFT, GOLD_FS, &info,
                               scalings[s], spec) != PDM_OK) {
            printf("  FAIL pdm_spectrum_scale(%s)\n", snames[s]);
            ++g_fail;
            continue;
        }
        double gmax = 0.0;
        for (int k = 0; k < GOLD_N_BINS; ++k) {
            if (golds[s][k] > gmax) {
                gmax = golds[s][k];
            }
        }
        for (int k = 0; k < GOLD_N_BINS; ++k) {
            snprintf(label, sizeof(label), "spec_%s[%d]", snames[s], k);
            CHECK_ABS(label, spec[k], golds[s][k], gmax * 5e-6);
        }
        /* Band RMS over four bands. The full-band value is the calibration
         * invariant: it must equal the time-domain RMS of the windowed,
         * mean-removed block, for every scaling. */
        float out[4];
        pdm_band_rms_vector(spec, GOLD_N_BINS, df, scalings[s], info.enbw_bins,
                            gold_band_edges, 4, out);
        for (int b = 0; b < 4; ++b) {
            snprintf(label, sizeof(label), "band_rms_%s[%d]", snames[s], b);
            CHECK_REL(label, out[b], gold_bands[s][b], 2e-5);
        }
        snprintf(label, sizeof(label), "band_rms_%s == time RMS", snames[s]);
        CHECK_REL(label, out[0], GOLD_TIME_RMS, 0.02);
    }
}

/* ------------------------------------------------------------------------- */
static void test_time_features(void)
{
    section("time-domain features");
    pdm_moments m;
    pdm_moments_init(&m);

    /* Remove DC first, as the feature extractor does: the 1 g static component
     * on a vertically mounted accelerometer otherwise dominates RMS and makes
     * crest factor meaningless. */
    double mean = 0.0;
    for (int i = 0; i < GOLD_N_BLOCK; ++i) {
        mean += (double)gold_signal[i];
    }
    mean /= GOLD_N_BLOCK;
    static float centred[GOLD_N_BLOCK];
    for (int i = 0; i < GOLD_N_BLOCK; ++i) {
        centred[i] = (float)((double)gold_signal[i] - mean);
    }
    pdm_moments_push_block(&m, centred, GOLD_N_BLOCK);

    pdm_time_features tf;
    if (pdm_time_features_finish(&m, centred, GOLD_N_BLOCK, &tf) != PDM_OK) {
        printf("  FAIL pdm_time_features_finish\n");
        ++g_fail;
        return;
    }

    /* The Python reference reports `mean` from the RAW block; this path feeds
     * it a centred block, so compare the shape statistics only -- `mean` is
     * checked separately below against running_stats, which is also centred. */
    CHECK_REL("rms",          tf.rms,          gold_time_features[1], 1e-5);
    CHECK_REL("std",          tf.std,          gold_time_features[2], 1e-5);
    CHECK_REL("peak",         tf.peak,         gold_time_features[3], 1e-6);
    CHECK_REL("peak_to_peak", tf.peak_to_peak, gold_time_features[4], 1e-6);
    CHECK_REL("crest",        tf.crest,        gold_time_features[5], 1e-5);
    /* m4 accumulates 2048 fourth powers; the recurrence is well-conditioned in
     * double but the stored moments are float32, so ~1e-4 relative. */
    CHECK_REL("kurtosis",     tf.kurtosis,     gold_time_features[6], 2e-4);
    CHECK_REL("skewness",     tf.skewness,     gold_time_features[7], 5e-3);
    CHECK_REL("shape",        tf.shape,        gold_time_features[8], 1e-5);
    CHECK_REL("impulse",      tf.impulse,      gold_time_features[9], 1e-5);
    CHECK_REL("clearance",    tf.clearance,    gold_time_features[10], 1e-5);
    CHECK_REL("margin",       tf.margin,       gold_time_features[11], 1e-5);
    CHECK_REL("zcr",          tf.zcr,          gold_time_features[12], 1e-6);
    CHECK_REL("entropy",      tf.entropy,      gold_time_features[13], 1e-5);

    /* The streaming reference: identical recurrence, so tighter tolerances. */
    CHECK_ABS("running mean",     tf.mean,      gold_running_stats[0], 1e-6);
    CHECK_REL("running std",      tf.std,       gold_running_stats[1], 1e-5);
    CHECK_REL("running rms",      tf.rms,       gold_running_stats[2], 1e-5);
    CHECK_REL("running peak",     tf.peak,      gold_running_stats[3], 1e-6);
    CHECK_REL("running kurtosis", tf.kurtosis,  gold_running_stats[4], 2e-4);
    CHECK_REL("running skewness", tf.skewness,  gold_running_stats[5], 5e-3);
}

/* ------------------------------------------------------------------------- */
static void test_bearing(void)
{
    section("bearing kinematics and harmonic SNR");
    pdm_bearing_geometry g;
    g.n_balls = GOLD_BEARING_N_BALLS;
    g.ball_d = GOLD_BEARING_BALL_D;
    g.pitch_d = GOLD_BEARING_PITCH_D;
    g.contact_angle_deg = GOLD_BEARING_ANGLE;

    pdm_fault_freqs ff;
    pdm_fault_frequencies(&g, GOLD_SHAFT_HZ, &ff);
    CHECK_REL("shaft_hz", ff.shaft_hz, gold_fault_freqs[0], 1e-6);
    CHECK_REL("ftf_hz",   ff.ftf_hz,   gold_fault_freqs[1], 1e-6);
    CHECK_REL("bpfo_hz",  ff.bpfo_hz,  gold_fault_freqs[2], 1e-6);
    CHECK_REL("bpfi_hz",  ff.bpfi_hz,  gold_fault_freqs[3], 1e-6);
    CHECK_REL("bsf_hz",   ff.bsf_hz,   gold_fault_freqs[4], 1e-6);

    /* Harmonic SNR on the planted-BPFO envelope spectrum. The statistic uses a
     * median floor, which is a selection rather than an arithmetic result, so
     * it is exact up to the tolerance-window edges -- hence a tight tolerance
     * and a hard assertion on the discrimination. */
    uint8_t matched = 0;
    const float snr_hit = pdm_harmonic_snr(gold_env_spec, GOLD_ENV_N, GOLD_ENV_DF,
                                           ff.bpfo_hz, 5, 0.02f, 120.0f, &matched);
    const float snr_miss = pdm_harmonic_snr(gold_env_spec, GOLD_ENV_N, GOLD_ENV_DF,
                                            ff.bpfi_hz, 5, 0.02f, 120.0f, NULL);
    CHECK_REL("harmonic SNR at BPFO", snr_hit, GOLD_SNR_BPFO, 0.02);
    CHECK_REL("harmonic SNR at BPFI", snr_miss, GOLD_SNR_BPFI, 0.05);

    /* The property that actually matters: the planted family is separable from
     * the absent one. If this fails, the diagnosis engine is broken even if
     * every number above matched. */
    ++g_checks;
    if (!(snr_hit > 4.0f && snr_miss < 4.0f && snr_hit > 2.0f * snr_miss)) {
        printf("  FAIL BPFO/BPFI discrimination: hit %.2f, miss %.2f\n",
               (double)snr_hit, (double)snr_miss);
        ++g_fail;
    }
    ++g_checks;
    if (matched < 3u) {
        printf("  FAIL matched harmonics: got %u, want >= 3\n", (unsigned)matched);
        ++g_fail;
    }
}

/* ------------------------------------------------------------------------- */
static void test_iso(void)
{
    section("ISO 20816-3 zones");
    char label[64];
    for (int i = 0; i < 8; ++i) {
        const uint8_t z = pdm_iso_zone(gold_iso_inputs[i], PDM_ISO_GROUP2_RIGID);
        snprintf(label, sizeof(label), "iso_zone(%.2f)", (double)gold_iso_inputs[i]);
        CHECK_ABS(label, (double)z, (double)gold_iso_zones[i], 0.0);
    }
}

/* ------------------------------------------------------------------------- */
static void test_model(void)
{
    section("int8 model inference");
    pdm_model_i8 m;
    memset(&m, 0, sizeof(m));
    m.n_layers = 2;
    m.layers[0].n_in = GOLD_MODEL_N_FEAT;
    m.layers[0].n_out = GOLD_MODEL_H1;
    m.layers[0].w = gold_model_w1;
    m.layers[0].w_scale = GOLD_MODEL_S1;
    m.layers[1].n_in = GOLD_MODEL_H1;
    m.layers[1].n_out = GOLD_MODEL_H2;
    m.layers[1].w = gold_model_w2;
    m.layers[1].w_scale = GOLD_MODEL_S2;
    m.centre = gold_model_centre;
    m.in_centre = gold_model_in_centre;
    m.in_scale = gold_model_in_scale;
    m.in_log = NULL;
    m.n_features = GOLD_MODEL_N_FEAT;
    m.leak = GOLD_MODEL_LEAK;

    static float scratch[2 * PDM_MODEL_MAX_WIDTH];
    float score = 0.0f;
    if (pdm_model_score(&m, gold_model_features, scratch, &score) != PDM_OK) {
        printf("  FAIL pdm_model_score\n");
        ++g_fail;
        return;
    }
    /* int8 arithmetic is exact, so the only error is the float de-quantisation
     * at each layer boundary -- a handful of operations. */
    CHECK_REL("model score", score, GOLD_MODEL_SCORE, 1e-5);
}

/* ------------------------------------------------------------------------- */
static void test_filters(void)
{
    section("biquad filters (properties, not golden values)");
    /* The envelope path deliberately differs from the Python reference (rectify
     * + low-pass on the device vs the analytic envelope in the cloud -- see
     * pdm_dsp.h), so it is checked by property rather than by value. */
    pdm_biquad_cascade bp;
    if (pdm_biquad_bandpass(&bp, GOLD_FS, 3000.0f, 6000.0f, 2u) != PDM_OK) {
        printf("  FAIL pdm_biquad_bandpass\n");
        ++g_fail;
        return;
    }

    /* Pass band vs stop band: a tone inside the band must survive, a tone an
     * octave below must be attenuated by >= 20 dB. */
    static float in[4096];
    static float out[4096];
    const float freqs[3] = {4500.0f, 1000.0f, 11000.0f};
    const char *fnames[3] = {"in band 4.5 kHz", "below band 1 kHz", "above band 11 kHz"};
    float gains[3];
    for (int f = 0; f < 3; ++f) {
        for (int i = 0; i < 4096; ++i) {
            in[i] = sinf(6.2831853f * freqs[f] * (float)i / GOLD_FS);
        }
        pdm_biquad_reset(&bp);
        pdm_biquad_bandpass(&bp, GOLD_FS, 3000.0f, 6000.0f, 2u);
        pdm_biquad_run(&bp, in, out, 4096);
        /* Measure on the second half, after the transient has decayed. */
        double acc = 0.0;
        for (int i = 2048; i < 4096; ++i) {
            acc += (double)out[i] * out[i];
        }
        gains[f] = (float)sqrt(acc / 2048.0) * 1.41421356f;
        printf("    %-20s gain %.4f (%.1f dB)\n", fnames[f], (double)gains[f],
               20.0 * log10((double)gains[f] + 1e-12));
    }
    /* Unity in-band gain, not merely "not attenuated". Each section is
     * normalised at the geometric centre frequency at design time, so a
     * cascade of any order passes an in-band tone through at amplitude 1. */
    ++g_checks;
    if (!(gains[0] > 0.80f && gains[0] < 1.25f)) {
        printf("  FAIL in-band gain %.3f, want 0.80..1.25 (unity)\n", (double)gains[0]);
        ++g_fail;
    }
    ++g_checks;
    if (!(gains[1] < 0.1f * gains[0])) {
        printf("  FAIL low stop-band rejection: %.4f vs in-band %.4f\n",
               (double)gains[1], (double)gains[0]);
        ++g_fail;
    }
    ++g_checks;
    if (!(gains[2] < 0.1f * gains[0])) {
        printf("  FAIL high stop-band rejection: %.4f vs in-band %.4f\n",
               (double)gains[2], (double)gains[0]);
        ++g_fail;
    }

    /* Envelope of an amplitude-modulated carrier must contain the modulation
     * rate. This is the property the whole bearing path depends on. */
    const float carrier = 4500.0f;
    const float modulation = 107.0f;      /* a plausible BPFO */
    for (int i = 0; i < 4096; ++i) {
        const float t = (float)i / GOLD_FS;
        in[i] = (1.0f + 0.8f * sinf(6.2831853f * modulation * t))
              * sinf(6.2831853f * carrier * t);
    }
    static float env[4096];
    if (pdm_envelope_rectify(in, 4096, GOLD_FS, 1200.0f, env, 0) != PDM_OK) {
        printf("  FAIL pdm_envelope_rectify\n");
        ++g_fail;
        return;
    }
    /* Correlate the envelope against the known modulation and against a decoy
     * at 1.7x the rate; the true rate must win by a wide margin. */
    double c_true = 0.0, c_decoy = 0.0;
    for (int i = 1024; i < 4096; ++i) {
        const float t = (float)i / GOLD_FS;
        c_true += (double)env[i] * sin(6.2831853 * modulation * t);
        c_decoy += (double)env[i] * sin(6.2831853 * modulation * 1.7 * t);
    }
    printf("    envelope correlation: true %.1f, decoy %.1f\n", fabs(c_true), fabs(c_decoy));
    ++g_checks;
    if (!(fabs(c_true) > 5.0 * fabs(c_decoy) + 1.0)) {
        printf("  FAIL envelope does not recover the modulation rate\n");
        ++g_fail;
    }

    /* Decimation must preserve a low-frequency tone and reduce the length. */
    size_t n_out = 0;
    static float dec[4096];
    for (int i = 0; i < 4096; ++i) {
        in[i] = sinf(6.2831853f * 200.0f * (float)i / GOLD_FS);
    }
    if (pdm_decimate(in, 4096, 8u, GOLD_FS, dec, &n_out) != PDM_OK) {
        printf("  FAIL pdm_decimate\n");
        ++g_fail;
        return;
    }
    ++g_checks;
    if (n_out != 512u) {
        printf("  FAIL decimate length: got %zu, want 512\n", n_out);
        ++g_fail;
    }
    double acc = 0.0;
    for (size_t i = 128; i < n_out; ++i) {
        acc += (double)dec[i] * dec[i];
    }
    const double rms = sqrt(acc / (double)(n_out - 128));
    printf("    decimated 200 Hz tone RMS %.4f (want ~0.707)\n", rms);
    ++g_checks;
    if (!(rms > 0.6 && rms < 0.78)) {
        printf("  FAIL decimation changed the amplitude of an in-band tone\n");
        ++g_fail;
    }
}

/* ------------------------------------------------------------------------- */
static void test_error_paths(void)
{
    section("error handling");
    pdm_window_info info;
    static float w[64];
    ++g_checks;
    if (pdm_window_make(PDM_WIN_HANN, NULL, 64, &info) != PDM_ERR_ARG) {
        printf("  FAIL NULL buffer not rejected\n");
        ++g_fail;
    }
    ++g_checks;
    if (pdm_window_make((pdm_window_kind)99, w, 64, &info) != PDM_ERR_UNSUPPORTED) {
        printf("  FAIL bad window kind not rejected\n");
        ++g_fail;
    }
    pdm_rfft_ctx ctx;
    static float tw[PDM_RFFT_TWIDDLE_LEN(64)];
    static uint16_t br[PDM_RFFT_BITREV_LEN(64)];
    ++g_checks;
    if (pdm_rfft_init(&ctx, 100, tw, br) != PDM_ERR_SIZE) {
        printf("  FAIL non-power-of-two FFT size not rejected\n");
        ++g_fail;
    }
    ++g_checks;
    if (pdm_rfft_init(&ctx, 32, tw, br) != PDM_ERR_SIZE) {
        printf("  FAIL undersized FFT not rejected\n");
        ++g_fail;
    }
    pdm_biquad_cascade c;
    ++g_checks;
    if (pdm_biquad_bandpass(&c, 25600.0f, 6000.0f, 3000.0f, 2u) != PDM_ERR_ARG) {
        printf("  FAIL inverted band not rejected\n");
        ++g_fail;
    }
    ++g_checks;
    if (pdm_biquad_bandpass(&c, 25600.0f, 3000.0f, 20000.0f, 2u) != PDM_ERR_ARG) {
        printf("  FAIL band above Nyquist not rejected\n");
        ++g_fail;
    }
    ++g_checks;
    if (pdm_is_power_of_two(0u) || !pdm_is_power_of_two(4096u) || pdm_is_power_of_two(4095u)) {
        printf("  FAIL pdm_is_power_of_two\n");
        ++g_fail;
    }
}

/* ------------------------------------------------------------------------- */
int main(void)
{
    printf("cdsp cross-check against the pdmcore reference\n");
    printf("==============================================\n");
    test_windows();
    test_rfft();
    test_spectrum();
    test_time_features();
    test_bearing();
    test_iso();
    test_model();
    test_filters();
    test_error_paths();
    printf("==============================================\n");
    if (g_fail == 0) {
        printf("PASS  %d checks\n", g_checks);
        return 0;
    }
    printf("FAIL  %d of %d checks\n", g_fail, g_checks);
    return 1;
}
