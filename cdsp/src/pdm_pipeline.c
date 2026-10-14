/* pdm_pipeline.c -- the tiered feature pipeline, portable across all MCU kits.
 *
 * This file contains no vendor SDK calls. Everything board-specific is behind
 * the `pdm_board_*` interface in board.h, which each kit implements. That split
 * is the reason the same pipeline runs on the STM32U5, the nRF9160, the ESP32-S3,
 * the Alif E7 and the RA8D1 branches, and the reason this file can be compiled
 * and tested on the build host.
 *
 * Tiering. The device does not compute everything every cycle:
 *
 *   tier 0   streaming time-domain indicators            ~0.4 ms
 *   tier 1   + FFT, bands, shape, orders, ISO velocity   ~2 ms
 *   tier 2   + envelope demodulation, bearing evidence   ~5 ms
 *   tier 3   + raw waveform snapshot to the cloud        ~300 ms radio
 *
 * Escalation is driven by the *measured* anomaly score, not by a fixed
 * schedule. The common case terminates at tier 1, which is where the energy
 * saving comes from; the diagnostic quality comes from tier 3 existing at all.
 *
 * Caveat worth stating: the tier-1 score is computed from a partially-filled
 * feature vector (the bearing features are zero until tier 2 runs). The model
 * was trained on complete vectors, so a tier-1 score is biased low. The
 * escalation threshold is therefore a *fraction* of the operating threshold
 * (APP_TIER1_ESCALATE_FRAC), tuned so that tier 2 runs on anything plausible
 * rather than only on things already alarming. Getting this backwards -- using
 * the full threshold at tier 1 -- would make the bearing features unreachable
 * exactly when they are needed.
 */

#include "pdm_pipeline.h"

#include <math.h>
#include <string.h>

#include "app_config.h"
#include "pdm_detector.h"

#ifndef PDM_TABLES_IN_FLASH
#define PDM_TABLES_IN_FLASH 0
#endif

/* -------------------------------------------------------------- static RAM */
/* All buffers static. No malloc anywhere -- see cdsp/README.md.
 *
 * With PDM_TABLES_IN_FLASH the window, twiddle and bit-reversal tables come
 * from a generated `const` header instead of RAM: 46 kB saved on a 4096-point
 * pipeline, which on the 256 kB nRF9160 is the difference between fitting
 * comfortably and not. Generate with cdsp/tools/gen_tables.py. */
#if PDM_TABLES_IN_FLASH
#include "pdm_tables.h"
#if (PDM_TABLES_N_FFT != APP_N_FFT) || (PDM_TABLES_ENV_N_FFT != APP_ENVELOPE_N_FFT)
#error "pdm_tables.h was generated for different FFT sizes than app_config.h asks for"
#endif
#if !defined(APP_WINDOW_ID)
#error "app_config.h must define APP_WINDOW_ID alongside APP_WINDOW when using \
PDM_TABLES_IN_FLASH: the preprocessor cannot compare C enum names"
#endif
#if (PDM_TABLES_WINDOW_ID != APP_WINDOW_ID)
#error "pdm_tables.h was generated for a different window than app_config.h asks for"
#endif
#define s_window      pdm_table_window
#define s_window_env  pdm_table_window_env
#else
static float    s_window[APP_N_FFT];
static float    s_twiddle[PDM_RFFT_TWIDDLE_LEN(APP_N_FFT)];
static uint16_t s_bitrev[PDM_RFFT_BITREV_LEN(APP_N_FFT)];
#endif
static float    s_samples[APP_N_FFT];          /* in-place FFT buffer */
static float    s_mag[APP_N_FFT / 2u + 1u];
static float    s_spec[APP_N_FFT / 2u + 1u];
/* Sized by the envelope stage, which consumes the full acquisition -- not by
 * the spectral block. GCC's -Waggressive-loop-optimizations caught this when it
 * was APP_N_FFT: the loop that fills it runs to APP_ENV_SAMPLES. A silent
 * overflow here would have corrupted whatever followed in .bss. */
static float    s_env[APP_ENV_SAMPLES];
static float    s_env_mag[APP_ENVELOPE_N_FFT / 2u + 1u];
static float    s_env_spec[APP_ENVELOPE_N_FFT / 2u + 1u];
static float    s_scratch[APP_SCRATCH_FLOATS];

static pdm_rfft_ctx      s_fft;
static pdm_rfft_ctx      s_fft_env;
#if !PDM_TABLES_IN_FLASH
static float             s_twiddle_env[PDM_RFFT_TWIDDLE_LEN(APP_ENVELOPE_N_FFT)];
static uint16_t          s_bitrev_env[PDM_RFFT_BITREV_LEN(APP_ENVELOPE_N_FFT)];
static float             s_window_env[APP_ENVELOPE_N_FFT];
#endif
static pdm_window_info   s_win_info;
static pdm_window_info   s_win_info_env;
static pdm_model_i8      s_model;
static pdm_biquad_cascade s_bandpass;

/* Log-spaced band edges, 12 bands from 5 Hz to 0.45*fs. Computed once at init
 * rather than tabulated, so changing APP_FS_HZ cannot leave a stale table. */
static float s_band_edges[2u * PDM_N_BANDS];

static int s_initialised = 0;

/* ------------------------------------------------------------------- utils */
static void build_band_edges(float fs_hz)
{
    const float f_lo = 5.0f;
    const float f_hi = 0.45f * fs_hz;
    const float ratio = powf(f_hi / f_lo, 1.0f / (float)PDM_N_BANDS);
    float f = f_lo;
    for (uint16_t b = 0; b < PDM_N_BANDS; ++b) {
        s_band_edges[2u * b] = f;
        f *= ratio;
        s_band_edges[2u * b + 1u] = f;
    }
}

static void order_band(float shaft_hz, float order, float *lo, float *hi)
{
    /* +-6 %: wide enough to survive the speed drift of a DOL motor and a modest
     * VFD, narrow enough that 1x and 2x do not merge. */
    const float centre = shaft_hz * order;
    *lo = centre * 0.94f;
    *hi = centre * 1.06f;
}

/* ------------------------------------------------------------------- init */
pdm_status pdm_pipeline_init(void)
{
    pdm_status st;

#if PDM_TABLES_IN_FLASH
    /* Tables are const in flash; the window info comes from the generated
     * constants rather than being recomputed. */
    s_win_info.kind = APP_WINDOW;
    s_win_info.n = APP_N_FFT;
    s_win_info.coherent_gain = PDM_TABLES_COHERENT_GAIN;
    s_win_info.enbw_bins = PDM_TABLES_ENBW_BINS;
    s_win_info.sum = PDM_TABLES_SUM;
    s_win_info.sum_sq = PDM_TABLES_SUM_SQ;
    s_win_info_env.kind = APP_WINDOW;
    s_win_info_env.n = APP_ENVELOPE_N_FFT;
    s_win_info_env.coherent_gain = PDM_TABLES_ENV_COHERENT_GAIN;
    s_win_info_env.enbw_bins = PDM_TABLES_ENV_ENBW_BINS;
    s_win_info_env.sum = PDM_TABLES_ENV_SUM;
    s_win_info_env.sum_sq = PDM_TABLES_ENV_SUM_SQ;
    st = pdm_rfft_attach(&s_fft, APP_N_FFT, pdm_table_twiddle, pdm_table_bitrev);
    if (st != PDM_OK) {
        return st;
    }
    st = pdm_rfft_attach(&s_fft_env, APP_ENVELOPE_N_FFT,
                         pdm_table_twiddle_env, pdm_table_bitrev_env);
    if (st != PDM_OK) {
        return st;
    }
#else
    st = pdm_window_make(APP_WINDOW, s_window, APP_N_FFT, &s_win_info);
    if (st != PDM_OK) {
        return st;
    }
    st = pdm_rfft_init(&s_fft, APP_N_FFT, s_twiddle, s_bitrev);
    if (st != PDM_OK) {
        return st;
    }
    st = pdm_window_make(APP_WINDOW, s_window_env, APP_ENVELOPE_N_FFT, &s_win_info_env);
    if (st != PDM_OK) {
        return st;
    }
    st = pdm_rfft_init(&s_fft_env, APP_ENVELOPE_N_FFT, s_twiddle_env, s_bitrev_env);
    if (st != PDM_OK) {
        return st;
    }
#endif
    st = pdm_biquad_bandpass(&s_bandpass, APP_FS_HZ, APP_DEMOD_LO_HZ, APP_DEMOD_HI_HZ, 2u);
    if (st != PDM_OK) {
        return st;
    }
    build_band_edges(APP_FS_HZ);
    pdm_detector_init(&s_model);
    if (s_model.n_features != PDM_N_FEATURES) {
        /* The exported model and this build disagree on the feature count. A
         * mismatch here means the model would read band RMS as kurtosis, so it
         * is a hard failure at init rather than a silent misdiagnosis. */
        return PDM_ERR_SIZE;
    }
    s_initialised = 1;
    return PDM_OK;
}

/* ------------------------------------------------------------- tier 0 + 1 */
static void write_time_features(const pdm_time_features *tf, float *out)
{
    out[0] = tf->mean;
    out[1] = tf->rms;
    out[2] = tf->std;
    out[3] = tf->peak;
    out[4] = tf->peak_to_peak;
    out[5] = tf->crest;
    out[6] = tf->kurtosis;
    out[7] = tf->skewness;
    out[8] = tf->shape;
    out[9] = tf->impulse;
    out[10] = tf->clearance;
    out[11] = tf->margin;
    out[12] = tf->zcr;
    out[13] = tf->entropy;
}

/* Spectral shape over the whole band. Mirrors
 * pdmcore.dsp.spectrum.spectral_shape_features. */
static void write_shape_features(const float *spec, uint16_t n_bins, float df, float *out)
{
    double total = 0.0;
    for (uint16_t k = 0; k < n_bins; ++k) {
        total += (double)spec[k] * spec[k];
    }
    if (total <= 0.0) {
        memset(out, 0, 7u * sizeof(float));
        return;
    }
    double centroid = 0.0;
    for (uint16_t k = 0; k < n_bins; ++k) {
        const double w = (double)spec[k] * spec[k] / total;
        centroid += (double)k * df * w;
    }
    double spread = 0.0, skew = 0.0, kurt = 0.0, entropy = 0.0;
    double log_sum = 0.0, lin_sum = 0.0;
    uint16_t n_pos = 0;
    for (uint16_t k = 0; k < n_bins; ++k) {
        const double p = (double)spec[k] * spec[k];
        const double w = p / total;
        const double d = (double)k * df - centroid;
        spread += d * d * w;
        if (w > 0.0) {
            entropy -= w * log(w);
        }
        if (p > 0.0) {
            log_sum += log(p);
            lin_sum += p;
            ++n_pos;
        }
    }
    spread = sqrt(spread);
    if (spread > 0.0) {
        for (uint16_t k = 0; k < n_bins; ++k) {
            const double w = (double)spec[k] * spec[k] / total;
            const double d = ((double)k * df - centroid) / spread;
            skew += d * d * d * w;
            kurt += d * d * d * d * w;
        }
    }
    /* 95 % energy rolloff. */
    double cum = 0.0;
    float rolloff = (float)(n_bins - 1u) * df;
    for (uint16_t k = 0; k < n_bins; ++k) {
        cum += (double)spec[k] * spec[k] / total;
        if (cum >= 0.95) {
            rolloff = (float)k * df;
            break;
        }
    }
    const double flatness = (n_pos > 0u && lin_sum > 0.0)
        ? exp(log_sum / (double)n_pos) / (lin_sum / (double)n_pos)
        : 0.0;

    out[0] = (float)centroid;
    out[1] = (float)spread;
    out[2] = (float)skew;
    out[3] = (float)kurt;
    out[4] = rolloff;
    out[5] = (float)flatness;
    out[6] = (float)(entropy / log((double)n_bins));
}

/* --------------------------------------------------------------- the stages */
pdm_status pdm_pipeline_run(const float *accel_g, size_t n_samples, float shaft_hz,
                            pdm_pipeline_result *out)
{
    if (!s_initialised) {
        return PDM_ERR_ARG;
    }
    if (accel_g == NULL || out == NULL || n_samples < APP_N_FFT) {
        return PDM_ERR_ARG;
    }
    if (!(shaft_hz > 0.0f)) {
        /* Order and bearing features are meaningless without a speed. Refusing
         * is correct: a plausible-looking vector computed against a wrong speed
         * is worse than no vector. */
        return PDM_ERR_ARG;
    }
    /* Preserve the caller's inputs across the clear. The threshold is pushed
     * down from the cloud and set by the caller before the run; memset-ing the
     * whole struct silently zeroed it, which made every acquisition escalate to
     * tier 2 -- a battery-life bug that produces perfectly correct features and
     * is therefore invisible to a functional test. Caught by
     * test_pipeline.c's "stops at tier 1 when quiet" assertion. */
    const float threshold = out->threshold_hint;
    memset(out, 0, sizeof(*out));
    out->threshold_hint = threshold;
    out->tier = 0u;

    /* ---- tier 0: streaming indicators, no FFT -------------------------- */
    /* Time-domain indicators over the WHOLE record, matching
     * pdmcore.dsp.features.time_features, which is given the full block. Using
     * only the spectral block would make every shape statistic disagree by a
     * few percent -- and kurtosis, which is an impulse counter, by much more,
     * because the impulses are not uniformly distributed across the record. */
    pdm_moments m;
    pdm_moments_init(&m);
    double mean = 0.0;
    for (size_t i = 0; i < APP_ACQ_SAMPLES; ++i) {
        mean += (double)accel_g[i];
        const float a = fabsf(accel_g[i]);
        if (a > APP_FULL_SCALE_G * APP_CLIP_MARGIN) {
            out->flags |= PDM_FRAME_CLIPPED;
        }
    }
    mean /= (double)APP_ACQ_SAMPLES;
    /* DC removal before the shape statistics: the 1 g static component on a
     * vertically mounted accelerometer otherwise dominates RMS and makes crest
     * factor meaningless. Streamed rather than buffered -- the full record does
     * not fit in s_samples, and does not need to. */
    for (size_t i = 0; i < APP_ACQ_SAMPLES; ++i) {
        pdm_moments_push(&m, (float)((double)accel_g[i] - mean));
    }

    pdm_time_features tf;
    pdm_status st = pdm_time_features_finish(&m, NULL, 0u, &tf);
    if (st != PDM_OK) {
        return st;
    }

    /* zcr and entropy are not expressible as moments, so they need a second
     * pass -- over the WHOLE record, matching the reference. Computing them
     * over the spectral block instead is tempting (the centred copy is already
     * in s_samples) and wrong by 3.5 % on zcr, because the zero-crossing rate
     * of an impulsive signal depends on where the impulses fall. The second
     * pass reads the caller's buffer directly and needs no extra RAM. */
    {
        size_t crossings = 0;
        int prev_neg = (((double)accel_g[0] - mean) < 0.0) ? 1 : 0;
        uint32_t hist[32];
        memset(hist, 0, sizeof(hist));
        const float peak = m.peak;
        const float inv = (peak > 0.0f) ? (32.0f / (2.0f * peak)) : 0.0f;
        for (size_t i = 0; i < APP_ACQ_SAMPLES; ++i) {
            const float v = (float)((double)accel_g[i] - mean);
            if (i > 0u) {
                const int neg = (v < 0.0f) ? 1 : 0;
                if (neg != prev_neg) {
                    ++crossings;
                }
                prev_neg = neg;
            }
            if (inv > 0.0f) {
                int b = (int)((v + peak) * inv);
                if (b < 0) {
                    b = 0;
                } else if (b > 31) {
                    b = 31;
                }
                hist[b] += 1u;
            }
        }
        tf.zcr = (float)((double)crossings / (double)(APP_ACQ_SAMPLES - 1u));
        double h = 0.0;
        for (int b = 0; b < 32; ++b) {
            if (hist[b] != 0u) {
                const double pr = (double)hist[b] / (double)APP_ACQ_SAMPLES;
                h -= pr * log(pr);
            }
        }
        tf.entropy = (float)(h / log(32.0));
    }
    write_time_features(&tf, &out->features[PDM_IDX_TIME]);
    /* Report the raw DC level, not the centred zero: it is a useful mounting
     * orientation check and the Python reference does the same. */
    out->features[PDM_IDX_TIME] = (float)mean;

    if (out->flags & PDM_FRAME_CLIPPED) {
        /* A clipped acquisition cannot produce valid amplitudes. Return the
         * time features so the device can still see that something is very
         * wrong, and let the cloud refuse to score it. */
        out->tier = 0u;
        return PDM_OK;
    }

    /* ---- tier 1: averaged spectrum serves bands, shape, orders, velocity */
    /* 50 %-overlapped blocks, accumulated in the POWER domain and scaled once.
     * Averaging magnitudes instead biases broadband content low by pi/4
     * (-1.05 dB) -- the same bug the Python side had, and the reason both sides
     * accumulate power. Each block is mean-removed individually, matching
     * pdmcore.dsp.spectrum's per-block detrend. */
    const uint16_t n_bins = APP_N_FFT / 2u + 1u;
    const size_t step = (size_t)APP_N_FFT * APP_SPEC_OVERLAP_NUM / APP_SPEC_OVERLAP_DEN;
    uint16_t n_avg = 0u;
    for (uint16_t k = 0; k < n_bins; ++k) {
        s_spec[k] = 0.0f;
    }
    for (size_t start = 0; start + APP_N_FFT <= APP_ACQ_SAMPLES; start += step) {
        double blk_mean = 0.0;
        for (size_t i = 0; i < APP_N_FFT; ++i) {
            blk_mean += (double)accel_g[start + i];
        }
        blk_mean /= (double)APP_N_FFT;
        for (size_t i = 0; i < APP_N_FFT; ++i) {
            s_samples[i] = (float)((double)accel_g[start + i] - blk_mean) * s_window[i];
        }
        st = pdm_rfft(&s_fft, s_samples);
        if (st != PDM_OK) {
            return st;
        }
        pdm_rfft_magnitude(&s_fft, s_samples, s_mag);
        /* Fold to single-sided power and accumulate. */
        for (uint16_t k = 0; k < n_bins; ++k) {
            float p = s_mag[k] * s_mag[k];
            if (k != 0u && k != (uint16_t)(APP_N_FFT / 2u)) {
                p *= 2.0f;
            }
            s_spec[k] += p;
        }
        ++n_avg;
    }
    if (n_avg == 0u) {
        return PDM_ERR_SIZE;
    }
    /* Mean power, then one scaling pass. s_spec currently holds accumulated
     * single-sided power, which is exactly what pdm_spectrum_scale's input
     * would be after the fold -- so undo the fold it would apply by passing the
     * per-bin magnitudes it expects. Simpler and less error-prone: convert here. */
    {
        const float s1 = s_win_info.sum;
        for (uint16_t k = 0; k < n_bins; ++k) {
            const float mean_p = s_spec[k] / (float)n_avg;
            s_spec[k] = sqrtf(mean_p) / s1;   /* PDM_SCALE_RMS */
        }
    }
    out->n_averages = n_avg;
    const float df = APP_FS_HZ / (float)APP_N_FFT;

    pdm_band_rms_vector(s_spec, n_bins, df, PDM_SCALE_RMS, s_win_info.enbw_bins,
                        s_band_edges, PDM_N_BANDS, &out->features[PDM_IDX_BAND]);
    write_shape_features(s_spec, n_bins, df, &out->features[PDM_IDX_SHAPE]);

    const float vrms = pdm_velocity_rms_mm_s(s_spec, n_bins, df, PDM_SCALE_RMS,
                                             s_win_info.enbw_bins,
                                             APP_VELOCITY_LO_HZ, APP_VELOCITY_HI_HZ);
    out->features[PDM_IDX_ISO] = vrms;
    out->features[PDM_IDX_ISO + 1u] = (float)pdm_iso_zone(vrms, APP_ISO_CLASS);
    out->velocity_rms_mm_s = vrms;
    out->iso_zone = pdm_iso_zone(vrms, APP_ISO_CLASS);

    static const float k_orders[6] = {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float order_sum = 0.0f;
    for (uint16_t i = 0; i < 6u; ++i) {
        float lo, hi;
        order_band(shaft_hz, k_orders[i], &lo, &hi);
        const float v = pdm_band_rms(s_spec, n_bins, df, PDM_SCALE_RMS,
                                     s_win_info.enbw_bins, lo, hi);
        out->features[PDM_IDX_ORDER + i] = v;
        if (i >= 1u) {
            order_sum += v;
        }
    }
    out->features[PDM_IDX_ORDER + 6u] = order_sum;
    const float one_x = out->features[PDM_IDX_ORDER + 1u];
    out->features[PDM_IDX_ORDER + 7u] =
        (one_x > 0.0f) ? (out->features[PDM_IDX_ORDER] / one_x) : 0.0f;

    out->tier = 1u;

    /* Score the partial vector. Biased low -- the bearing block is still zero --
     * so escalate on a fraction of the operating threshold, not on it. */
    st = pdm_model_score(&s_model, out->features, s_scratch, &out->score);
    if (st != PDM_OK) {
        return st;
    }
    if (out->score < APP_TIER1_ESCALATE_FRAC * out->threshold_hint) {
        return PDM_OK;   /* the common case: done in ~2 ms */
    }

    /* ---- tier 2: envelope demodulation and bearing evidence ------------ */
    /* Band-pass the FULL acquisition, un-windowed and DC-removed.
     *
     * Full, not the spectral block: envelope resolution is set by record
     * duration, and APP_ENV_SAMPLES is chosen for the modulation frequencies of
     * interest (see app_config.h). Using APP_N_FFT here produced 512 envelope
     * samples for a 1024-point transform, so the guard below fired and the
     * pipeline silently reported tier 1 with an all-zero bearing block --
     * indistinguishable from "no bearing fault". Caught by test_pipeline.c.
     *
     * Un-windowed: a window tapers the impacts at the block edges, which is
     * exactly where the fault energy can be. */
    for (size_t i = 0; i < APP_ENV_SAMPLES; ++i) {
        s_env[i] = (float)((double)accel_g[i] - mean);
    }
    pdm_biquad_reset(&s_bandpass);
    (void)pdm_biquad_bandpass(&s_bandpass, APP_FS_HZ, APP_DEMOD_LO_HZ, APP_DEMOD_HI_HZ, 2u);
    pdm_biquad_run(&s_bandpass, s_env, s_env, APP_ENV_SAMPLES);

    /* Squared envelope: better statistical properties than the plain envelope
     * (Randall & Antoni 2011). Rectify-and-low-pass, not the analytic signal --
     * see cdsp/README.md for why, and for the normalisation that keeps the
     * decision identical to the cloud's. */
    st = pdm_envelope_rectify(s_env, APP_ENV_SAMPLES, APP_FS_HZ, APP_ENVELOPE_LP_HZ, s_env, 1);
    if (st != PDM_OK) {
        return st;
    }
    size_t n_env = 0;
    st = pdm_decimate(s_env, APP_ENV_SAMPLES, APP_ENVELOPE_DECIMATE, APP_FS_HZ, s_env, &n_env);
    if (st != PDM_OK) {
        return st;
    }
    const float fs_env = APP_FS_HZ / (float)APP_ENVELOPE_DECIMATE;
    if (n_env < APP_ENVELOPE_N_FFT) {
        /* Unreachable given the compile-time guards at the top of this file.
         * Kept because staying at tier 1 is the right behaviour if it ever
         * becomes reachable: emitting zeros would read as "no bearing fault",
         * which is the dangerous direction. */
        return PDM_OK;
    }
    for (size_t i = 0; i < APP_ENVELOPE_N_FFT; ++i) {
        s_env[i] *= s_window_env[i];
    }
    st = pdm_rfft(&s_fft_env, s_env);
    if (st != PDM_OK) {
        return st;
    }
    const uint16_t n_env_bins = APP_ENVELOPE_N_FFT / 2u + 1u;
    pdm_rfft_magnitude(&s_fft_env, s_env, s_env_mag);
    st = pdm_spectrum_scale(s_env_mag, n_env_bins, APP_ENVELOPE_N_FFT, fs_env,
                            &s_win_info_env, PDM_SCALE_PEAK, s_env_spec);
    if (st != PDM_OK) {
        return st;
    }
    const float df_env = fs_env / (float)APP_ENVELOPE_N_FFT;

    pdm_bearing_geometry geom = {
        APP_BEARING_N_BALLS, APP_BEARING_BALL_D, APP_BEARING_PITCH_D, APP_BEARING_ANGLE_DEG
    };
    pdm_fault_freqs ff;
    pdm_fault_frequencies(&geom, shaft_hz, &ff);

    /* Four modes, in the fixed order the feature contract specifies:
     * outer race, inner race, rolling element, cage. */
    const float mode_f0[4] = {ff.bpfo_hz, ff.bpfi_hz, ff.bsf_hz, ff.ftf_hz};
    const float mode_mod[4] = {0.0f, ff.shaft_hz, ff.ftf_hz, 0.0f};
    for (uint16_t mode = 0; mode < 4u; ++mode) {
        uint8_t matched = 0u;
        const float ratio = pdm_harmonic_snr(s_env_spec, n_env_bins, df_env,
                                             mode_f0[mode], (uint8_t)APP_BEARING_HARMONICS,
                                             0.02f, 120.0f, &matched);
        float peak_snr = 0.0f;
        {
            /* Peak SNR across the harmonics, for the model and for triage. */
            for (uint8_t h = 1u; h <= APP_BEARING_HARMONICS; ++h) {
                uint8_t dummy = 0u;
                const float s1 = pdm_harmonic_snr(s_env_spec, n_env_bins, df_env,
                                                  mode_f0[mode] * (float)h, 1u,
                                                  0.02f, 120.0f, &dummy);
                if (s1 > peak_snr) {
                    peak_snr = s1;
                }
            }
        }
        float sideband = 0.0f;
        if (mode_mod[mode] > 0.0f) {
            sideband = pdm_sideband_ratio(s_env_spec, n_env_bins, df_env,
                                          mode_f0[mode], mode_mod[mode], 2u, 0.02f);
        }
        float *dst = &out->features[PDM_IDX_BEARING + 4u * mode];
        dst[0] = ratio;
        dst[1] = peak_snr;
        dst[2] = (float)matched;
        dst[3] = sideband;
    }

    /* Envelope-domain impulsiveness: cheap, and a strong early indicator even
     * when no kinematic frequency matches (lubrication loss). */
    {
        double em = 0.0;
        for (uint16_t k = 0; k < n_env_bins; ++k) {
            em += (double)s_env_spec[k];
        }
        em /= (double)n_env_bins;
        double var = 0.0, m4 = 0.0, peak = 0.0;
        for (uint16_t k = 0; k < n_env_bins; ++k) {
            const double d = (double)s_env_spec[k] - em;
            var += d * d;
            m4 += d * d * d * d;
            if ((double)s_env_spec[k] > peak) {
                peak = (double)s_env_spec[k];
            }
        }
        var /= (double)n_env_bins;
        m4 /= (double)n_env_bins;
        out->features[PDM_IDX_BEARING + 16u] = (var > 0.0) ? (float)(m4 / (var * var)) : 0.0f;
        out->features[PDM_IDX_BEARING + 17u] = (em > 0.0) ? (float)(peak / em) : 0.0f;
    }
    out->features[PDM_IDX_BEARING + 18u] = APP_DEMOD_LO_HZ;
    out->features[PDM_IDX_BEARING + 19u] = APP_DEMOD_HI_HZ;

    out->tier = 2u;

    /* Re-score the complete vector. This is the score the cloud sees. */
    st = pdm_model_score(&s_model, out->features, s_scratch, &out->score);
    if (st != PDM_OK) {
        return st;
    }
    if (out->score >= APP_TIER2_ESCALATE_FRAC * out->threshold_hint) {
        out->request_snapshot = 1u;
        out->tier = 3u;
    }
    return PDM_OK;
}

uint16_t pdm_pipeline_n_features(void)
{
    return PDM_N_FEATURES;
}

const float *pdm_pipeline_band_edges(void)
{
    return s_band_edges;
}
