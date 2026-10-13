/* pdm_dsp.h -- portable C99 condition-monitoring DSP for microcontrollers.
 *
 * This is the firmware twin of pdmcore.dsp. The point of having two
 * implementations is that a maintenance decision must not depend on *where* a
 * number was computed: the MCU computes the feature vector in the field, the
 * cloud recomputes it from an archived waveform, and the two must agree. They
 * are held together by generated golden vectors (cdsp/tests/golden.h, produced
 * by tools/gen_golden.py from pdmcore) and checked by `make -C cdsp test`,
 * which runs natively on the build host -- so the agreement is verified on
 * every commit without hardware in the loop.
 *
 * Constraints this code is written under, and why:
 *
 *   - **float32 throughout.** Cortex-M4F/M33/M55 have single-precision FPUs and
 *     no double. Writing `double` silently drops you into software emulation and
 *     a 4096-point FFT goes from 1 ms to 40 ms. The Python reference uses
 *     float64, so the golden-vector tolerances are stated in float32 epsilons
 *     (see PDM_GOLDEN_TOL in the tests).
 *   - **No dynamic allocation.** No malloc, no VLAs. Every buffer is caller-
 *     provided. A heap on a device that must run unattended for five years is a
 *     fragmentation bug waiting for the worst possible moment.
 *   - **No stdio, no errno, no locale.** The library must link into a bare-metal
 *     image with -nostdlib if required. Only <math.h>, <string.h>, <stdint.h>.
 *   - **In-place where it matters.** The FFT works in place on the caller's
 *     buffer because on a 256 kB-RAM part a second 4096-float buffer is 6 % of
 *     RAM.
 *   - **No recursion.** Stack depth must be statically known.
 *
 * Portability: C99, no compiler extensions, no unaligned access, no assumptions
 * about endianness except in pdm_payload.c (which documents its own).
 */

#ifndef PDM_DSP_H
#define PDM_DSP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PDM_DSP_VERSION_MAJOR 0
#define PDM_DSP_VERSION_MINOR 3

/* Feature-spec version. Must match pdmcore.FEATURE_SPEC_VERSION or the cloud
 * will refuse the frame. */
#define PDM_FEATURE_SPEC_VERSION 2

/* ---------------------------------------------------------------- status -- */
typedef enum {
    PDM_OK = 0,
    PDM_ERR_ARG = -1,       /* a NULL pointer or an out-of-range size */
    PDM_ERR_SIZE = -2,      /* length not a power of two, or too small */
    PDM_ERR_UNSUPPORTED = -3,
    PDM_ERR_NUMERIC = -4    /* produced a non-finite result */
} pdm_status;

/* --------------------------------------------------------------- windows -- */
typedef enum {
    PDM_WIN_RECT = 0,
    PDM_WIN_HANN,
    PDM_WIN_HAMMING,
    PDM_WIN_BLACKMAN,
    PDM_WIN_BLACKMAN_HARRIS,
    PDM_WIN_FLATTOP,
    PDM_WIN__COUNT
} pdm_window_kind;

/* Window plus the two correction factors that make a reading calibrated.
 * Keeping them with the coefficients is deliberate: a windowed FFT under-reports
 * tone amplitude by 1/coherent_gain and over-reports band energy by enbw_bins,
 * and the two are easy to confuse. */
typedef struct {
    pdm_window_kind kind;
    uint16_t n;
    float coherent_gain;   /* mean(w): divide tone amplitude by this */
    float enbw_bins;       /* n*sum(w^2)/sum(w)^2: divide band power by this */
    float sum;             /* sum(w)   -- S1 */
    float sum_sq;          /* sum(w^2) -- S2 */
} pdm_window_info;

/* Generate `n` periodic window coefficients into `w` (caller-provided).
 * Periodic (2*pi*k/n), not symmetric (2*pi*k/(n-1)): periodic is correct for
 * spectral analysis. Mixing the two gives a small, maddening amplitude bias. */
pdm_status pdm_window_make(pdm_window_kind kind, float *w, uint16_t n,
                           pdm_window_info *info);

/* ------------------------------------------------------------------- FFT -- */
/* Real FFT via the N/2-point complex FFT trick, the same structure CMSIS-DSP
 * uses, so swapping in arm_rfft_fast_f32 on a Cortex-M is a one-line change and
 * the surrounding code does not move.
 *
 * `n` must be a power of two, 64 <= n <= PDM_FFT_MAX.
 *
 * `twiddle` must hold PDM_RFFT_TWIDDLE_LEN(n) floats and `bitrev`
 * PDM_RFFT_BITREV_LEN(n) entries, both initialised once by pdm_rfft_init.
 * Precomputing them costs RAM but removes ~n*log2(n) sinf() calls from the hot
 * path, which on an M33 with no hardware transcendental unit is the difference
 * between 1 ms and 11 ms.
 *
 * USE THE MACROS. The table is two concatenated halves -- n/2 floats of
 * complex-pass twiddles followed by n/2+2 floats of split-pass twiddles -- and
 * sizing it by hand is how this code shipped its first bug: a 3*n/4 allocation
 * silently overflowed into the split table and corrupted exactly bins
 * n/8+1 .. 3n/8 of every transform. The golden-vector test caught it, which is
 * the entire argument for having one. */
#define PDM_FFT_MAX 8192
#define PDM_RFFT_TWIDDLE_LEN(n) ((n) + 2u)
#define PDM_RFFT_BITREV_LEN(n)  ((n) / 2u)

typedef struct {
    uint16_t n;
    const float *twiddle;   /* 3*n/4 floats: cos/sin table for the complex pass */
    uint16_t *bitrev;       /* n/2 entries */
} pdm_rfft_ctx;

/* Fill `twiddle` (3*n/4 floats) and `bitrev` (n/2 uint16) for transform size n. */
pdm_status pdm_rfft_init(pdm_rfft_ctx *ctx, uint16_t n, float *twiddle, uint16_t *bitrev);

/* In-place real FFT of `n` samples in `data`.
 *
 * On return `data` holds n/2+1 complex bins packed as
 *   data[0]   = Re(X[0])      (DC, imaginary part is zero by construction)
 *   data[1]   = Re(X[n/2])    (Nyquist, likewise) -- CMSIS-DSP packing
 *   data[2k], data[2k+1] = Re(X[k]), Im(X[k]) for k = 1 .. n/2-1
 *
 * The DC/Nyquist packing into slots 0 and 1 is what lets the transform be
 * strictly in place with no extra element. It is also the single most common
 * source of off-by-one bugs when reading the output, which is why
 * pdm_rfft_magnitude exists and should be preferred over hand-indexing. */
pdm_status pdm_rfft(const pdm_rfft_ctx *ctx, float *data);

/* Magnitude of the packed output into `mag` (n/2+1 floats).
 * `mag[0]` is |DC|, `mag[n/2]` is |Nyquist|. */
pdm_status pdm_rfft_magnitude(const pdm_rfft_ctx *ctx, const float *packed, float *mag);

/* ------------------------------------------------------------- spectrum -- */
typedef enum {
    PDM_SCALE_PEAK = 0,  /* amplitude of a sinusoid, input units */
    PDM_SCALE_RMS,       /* peak/sqrt(2) -- what ISO limits use */
    PDM_SCALE_POWER,     /* units^2 per bin */
    PDM_SCALE_PSD        /* units^2/Hz -- block-length independent */
} pdm_scaling;

/* Convert |rfft| magnitudes (n/2+1) to a calibrated single-sided spectrum.
 *
 * The single-sided fold is applied to POWER, not magnitude: doubling the
 * magnitude and then squaring over-counts by 2x and makes a white-noise PSD
 * integrate to sqrt(2)*sigma. The Python implementation had exactly that bug
 * before it was caught by a Parseval test; this comment is here so it is not
 * reintroduced on the firmware side. */
pdm_status pdm_spectrum_scale(const float *mag, uint16_t n_bins, uint16_t n_fft,
                              float fs_hz, const pdm_window_info *win,
                              pdm_scaling scaling, float *out);

/* RMS of the content in [lo_hz, hi_hz], in input units.
 *
 * Divides out the ENBW: a windowed spectrum spreads one tone over enbw_bins
 * bins, each calibrated to report that tone's full amplitude, so summing them
 * over-counts by exactly enbw_bins. With the division this one function gives
 * the right answer for a tone AND for broadband noise -- without it, one of the
 * two is wrong by up to 1.9x (flat-top). */
float pdm_band_rms(const float *spec, uint16_t n_bins, float df_hz,
                   pdm_scaling scaling, float enbw_bins,
                   float lo_hz, float hi_hz);

/* Fill `out` with the band RMS of each of `n_bands` [lo,hi) pairs in `edges`
 * (2*n_bands floats). The band-RMS vector is the workhorse edge feature: ~12
 * numbers carry most of the diagnostic content of a 4096-point spectrum at
 * 1/300 of the radio payload. */
pdm_status pdm_band_rms_vector(const float *spec, uint16_t n_bins, float df_hz,
                               pdm_scaling scaling, float enbw_bins,
                               const float *edges, uint16_t n_bands, float *out);

/* Interpolated peak: parabolic fit around the local maximum in
 * [lo_bin, hi_bin]. Recovers most of a Hann window's 1.42 dB scallop loss
 * (to ~0.6 dB) without storing a flat-top window in flash. */
pdm_status pdm_peak_interp(const float *spec, uint16_t n_bins, float df_hz,
                           uint16_t lo_bin, uint16_t hi_bin,
                           float *freq_hz, float *value);

/* ------------------------------------------------- time-domain features -- */
/* Streaming moment accumulator. The MCU cannot iterate a long record several
 * times, so mean/variance/skew/kurtosis come from one pass of Welford-style
 * updates and the indicators are derived at the end. pdmcore.dsp.features
 * .running_stats implements the identical algorithm -- not merely the identical
 * definition -- so the two agree to float32 precision rather than "closely". */
typedef struct {
    uint32_t n;
    float mean, m2, m3, m4;
    float peak;          /* max |x| */
    float sum_abs;
    float sum_sqrt_abs;
    float min_val, max_val;
} pdm_moments;

void  pdm_moments_init(pdm_moments *m);
void  pdm_moments_push(pdm_moments *m, float x);
void  pdm_moments_push_block(pdm_moments *m, const float *x, size_t n);

/* Indicator set. Order and definitions match TIME_FEATURE_NAMES in
 * pdmcore.dsp.features. kurtosis is the NON-EXCESS (Pearson) form: 3.0 for
 * Gaussian noise. Mixing excess and non-excess between edge and cloud shifts
 * every threshold by exactly 3. */
typedef struct {
    float mean, rms, std, peak, peak_to_peak, crest;
    float kurtosis, skewness, shape, impulse, clearance, margin;
    float zcr, entropy;
} pdm_time_features;

/* Derive the indicators from accumulated moments. `zcr` and `entropy` need the
 * block (they are not single-pass), so pass it, or pass NULL to leave those two
 * at zero. */
pdm_status pdm_time_features_finish(const pdm_moments *m, const float *block,
                                    size_t n, pdm_time_features *out);

/* ------------------------------------------------------------- envelope -- */
/* Biquad cascade, Direct Form I with float32 state.
 *
 * Direct Form I rather than the more common DF2T: DF1 keeps the input history
 * explicitly, which makes it far better behaved in float32 for the narrow,
 * high-Q band-passes used for envelope demodulation (a 2 kHz-wide band at
 * 20 kHz sample rate is a pole very close to the unit circle, where DF2T's
 * single state variable loses precision). Costs two extra floats of state per
 * section. */
#define PDM_BIQUAD_MAX_SECTIONS 4

typedef struct {
    float b0, b1, b2, a1, a2;   /* a0 normalised to 1 */
    float x1, x2, y1, y2;
} pdm_biquad;

typedef struct {
    pdm_biquad s[PDM_BIQUAD_MAX_SECTIONS];
    uint8_t n_sections;
} pdm_biquad_cascade;

/* Butterworth band-pass of order 2*n_sections. */
pdm_status pdm_biquad_bandpass(pdm_biquad_cascade *c, float fs_hz,
                               float lo_hz, float hi_hz, uint8_t n_sections);
/* Butterworth low-pass, used as the decimation anti-alias filter. */
pdm_status pdm_biquad_lowpass(pdm_biquad_cascade *c, float fs_hz, float fc_hz,
                              uint8_t n_sections);
void  pdm_biquad_reset(pdm_biquad_cascade *c);
void  pdm_biquad_run(pdm_biquad_cascade *c, const float *in, float *out, size_t n);

/* Rectify-and-smooth envelope: |x| through a low-pass.
 *
 * NOT the Hilbert envelope. The analytic signal needs a 90-degree all-pass over
 * the whole band, which on an MCU means either a long FIR (RAM) or a second FFT
 * (cycles). Full-wave rectification plus a low-pass gives the same modulation
 * spectrum up to a DC term and a known 2/pi scale factor on the fundamental,
 * and the squared-envelope spectrum is unaffected in *shape* -- which is what
 * the bearing-frequency matcher uses. The cloud, which has the cycles, uses the
 * true analytic envelope; the ratio statistics are normalised so the two agree
 * on the decision even though the absolute envelope amplitudes differ by 2/pi.
 * This is documented in docs/02_feature_spec.md as a deliberate edge/cloud
 * divergence, with the normalisation that makes it safe. */
pdm_status pdm_envelope_rectify(const float *x, size_t n, float fs_hz,
                                float lp_hz, float *out, int squared);

/* Decimate by an integer factor after low-pass filtering. Returns the output
 * length in *n_out. The envelope is slowly varying (bearing modulation rates
 * are < 1 kHz), so keeping 25.6 kHz of envelope bandwidth wastes an order of
 * magnitude of FFT resolution where it actually matters. */
pdm_status pdm_decimate(const float *x, size_t n, uint8_t factor,
                        float fs_hz, float *out, size_t *n_out);

/* -------------------------------------------------------------- bearing -- */
typedef struct {
    float n_balls;
    float ball_d;
    float pitch_d;
    float contact_angle_deg;
} pdm_bearing_geometry;

typedef struct {
    float shaft_hz, ftf_hz, bpfo_hz, bpfi_hz, bsf_hz;
} pdm_fault_freqs;

void pdm_fault_frequencies(const pdm_bearing_geometry *g, float shaft_hz,
                           pdm_fault_freqs *out);

/* Harmonic-family score at f0.
 *
 * Returns the MEAN per-harmonic SNR against a local median floor, not the
 * summed amplitude. Summed amplitude over one floor estimate scales with the
 * harmonic count, so the number is uninterpretable on its own and every
 * threshold has to be retuned when n_harmonics changes. With the tolerance
 * windows used here a pure-noise family scores ~2-3, so 4-6 is a real
 * detection. Identical statistic to pdmcore.dsp.bearing.harmonic_energy_ratio.
 *
 * `n_matched` (may be NULL) receives the count of harmonics with SNR >= 4. */
float pdm_harmonic_snr(const float *spec, uint16_t n_bins, float df_hz,
                       float f0_hz, uint8_t n_harmonics, float rel_tol,
                       float floor_width_hz, uint8_t *n_matched);

/* Energy in carrier +- k*modulation relative to the carrier.
 *
 * High for inner-race defects (modulation = shaft rate) and rolling-element
 * defects (modulation = cage rate); low for outer-race defects. The sideband
 * pattern -- not the centre frequency -- is what distinguishes the modes.
 *
 * Returns 0.0 when the carrier does not stand clear of its local floor, because
 * the ratio of two noise samples is unbounded and reports ~6 on a healthy
 * machine. "Not modulated" and "no carrier to modulate" are both 0 here; the
 * cloud distinguishes them via the carrier SNR, which it has. */
float pdm_sideband_ratio(const float *spec, uint16_t n_bins, float df_hz,
                         float carrier_hz, float modulation_hz,
                         uint8_t n_sidebands, float rel_tol);

/* --------------------------------------------------------- ISO 20816-3 -- */
typedef enum {
    PDM_ISO_GROUP2_RIGID = 0,   /* 15-300 kW, rigid support  : 1.4/2.8/4.5 */
    PDM_ISO_GROUP2_FLEX,        /* 15-300 kW, flexible       : 2.3/4.5/7.1 */
    PDM_ISO_GROUP1_RIGID,       /* >300 kW, rigid            : 2.3/4.5/7.1 */
    PDM_ISO_GROUP1_FLEX         /* >300 kW, flexible         : 3.5/7.1/11.0 */
} pdm_iso_class;

/* 0=A, 1=B, 2=C, 3=D. A zone is not a diagnosis and not a trend -- see
 * pdmcore.dsp.iso for why alarms are based on the trended health index. */
uint8_t pdm_iso_zone(float velocity_rms_mm_s, pdm_iso_class cls);

/* Integrate an acceleration spectrum (g) to velocity RMS (mm/s) over a band,
 * by dividing each bin by omega. Frequency-domain integration with an explicit
 * high-pass, never time-domain: a 1 mg offset integrates into a velocity ramp,
 * which is the "ski slope" at the bottom of every badly-made velocity
 * spectrum. */
float pdm_velocity_rms_mm_s(const float *accel_spec_g, uint16_t n_bins,
                            float df_hz, pdm_scaling scaling, float enbw_bins,
                            float lo_hz, float hi_hz);

/* ---------------------------------------------------------------- model -- */
/* int8 quantised MLP for the one-class detector (Deep SVDD).
 *
 * Symmetric per-tensor quantisation: w_float ~= w_int8 * w_scale. No bias terms
 * -- deliberately, and not to save flash: Deep SVDD with biases can collapse to
 * the constant centre and reach zero training loss having learned nothing
 * ("hypersphere collapse", Ruff et al. 2018), so the architecture omits them
 * and the firmware must too or the exported weights will not match. */
#define PDM_MODEL_MAX_LAYERS 4
#define PDM_MODEL_MAX_WIDTH 128

typedef struct {
    uint16_t n_in, n_out;
    const int8_t *w;        /* row-major [n_in][n_out] */
    float w_scale;          /* w_float ~= w_int8 * w_scale */
    /* Scale used to quantise THIS layer's input: q = round(x / in_scale),
     * clipped to [-127, 127]. Calibrated per layer by the exporter from the
     * 99.9th percentile of observed activations.
     *
     * A single fixed scale for every layer looks simpler and is wrong. With a
     * fixed 1/16 the representable range is +-8, which covers a healthy
     * standardised feature but saturates hard on a fault -- measured 95 % score
     * error on strongly faulty samples, because the very activations that carry
     * the fault are the ones clipped. Detection survived (saturation only
     * reduces an already-large score, so there were no decision flips) but the
     * score stopped being usable as a magnitude, which breaks trending and RUL.
     * Per-layer scales cost one float per layer in flash. */
    float in_scale;
} pdm_layer_i8;

typedef struct {
    uint8_t n_layers;
    pdm_layer_i8 layers[PDM_MODEL_MAX_LAYERS];
    const float *centre;    /* hypersphere centre, length = last n_out */
    const float *in_centre; /* feature scaler: median */
    const float *in_scale;  /* feature scaler: robust sigma */
    const uint8_t *in_log;  /* 1 => log-transform this feature first */
    uint16_t n_features;
    float leak;             /* LeakyReLU negative slope */
} pdm_model_i8;

/* Anomaly score: distance from the hypersphere centre. Higher = more anomalous.
 * `scratch` must hold 2*PDM_MODEL_MAX_WIDTH floats. */
pdm_status pdm_model_score(const pdm_model_i8 *m, const float *features,
                           float *scratch, float *score_out);

/* ------------------------------------------------------------- utility -- */
int   pdm_is_power_of_two(uint32_t v);
float pdm_median_f32(float *scratch, uint16_t n);   /* destroys scratch */

#ifdef __cplusplus
}
#endif
#endif /* PDM_DSP_H */
