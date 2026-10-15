"""The current-domain feature contract, and the budget that makes it defensible.

The one calibration problem that vibration does not have
--------------------------------------------------------
In a vibration spectrum the thing being measured and the thing next to it are
within 20-30 dB of each other, so the window is chosen for amplitude accuracy
(flat top) or for resolution (Hann) and the sidelobes never matter.

In a current spectrum the supply fundamental is **40 to 60 dB larger than the
sideband being measured and 0.1 to 3 Hz away from it**. The measurement is then
not limited by noise at all; it is limited by the window's sidelobe at that
offset. Numbers, for a 10 s record (``df = 0.1 Hz``) with the 2sf1 pair 1.33 Hz
out, i.e. 13.3 bins:

======================  =======================  ==========================
window                  leakage at 13.3 bins     verdict for a -45 dB claim
======================  =======================  ==========================
``rect``                about -26 dB             the sideband is invisible
``hann``                about -72 dB             27 dB of margin
``blackman_harris``     about -109 dB            64 dB of margin
======================  =======================  ==========================

(The exact figures are computed, not quoted: :func:`window_leakage_db` evaluates
the window's own DTFT and ``test_mcsa_leakage_budget`` pins them.)

So Hann -- correct everywhere else in this repository -- is adequate at 13 bins
and *not* adequate at the 2 bins a 1 s record gives, where it leaks -31 dB and a
-45 dB sideband is 14 dB below the leakage of the line next to it. That is why
:class:`CurrentFeatureExtractor` computes the budget for its own configuration
and marks the frame rather than reporting a number it cannot support. A detector
fed a leakage artefact learns that the artefact is the fault.

Everything is reported relative to the fundamental
--------------------------------------------------
Sideband amplitudes are stored as dB relative to the measured fundamental, never
in amperes. Three reasons, in increasing order of importance: it cancels the CT
ratio and the burden resistor, so a mis-entered turns ratio cannot create a
fault; it cancels motor size, so one threshold works across a fleet; and it is
the quantity the entire MCSA literature quotes thresholds in (-50 dB healthy,
-45 dB one broken bar, -35 dB several), so a number here can be compared with a
number from a textbook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsp.bearing import BearingGeometry, fault_frequencies, line_snr, sideband_ratio
from ..dsp.spectrum import Spectrum, interpolate_peak, spectrum
from ..dsp.windows import window
from .frequencies import (
    min_record_duration_s,
    sideband_position_error_hz,
    slip_from_shaft_hz,
    slip_relative_error,
)
from .park import ParkVector, clarke_transform, epva_spectrum, sequence_components, unbalance_pct

__all__ = [
    "CurrentFeatureConfig",
    "CurrentFrame",
    "CurrentFeatureExtractor",
    "window_leakage_db",
    "leakage_budget_db",
    "CURRENT_FEATURE_NAMES",
]


def window_leakage_db(window_name: str, offset_bins: float, *, size: int = 1024,
                      pad: int = 64) -> float:
    """Sidelobe envelope of a window at ``offset_bins`` from its main lobe, in dB.

    Evaluated from the window's own DTFT (a ``pad``-times zero-padded FFT of the
    coefficients), taking the local maximum within half a bin so the answer is
    the *envelope* rather than whatever a particular null happened to give. That
    distinction matters: the sidelobe pattern of a cosine-sum window has deep
    nulls between its lobes, and sampling one of them would report 40 dB more
    margin than the measurement actually has.

    Independent of ``size`` to within a fraction of a dB for ``size >= 256``,
    which is why the default is a fixed 1024 rather than the caller's ``n_fft``.
    """
    w = window(window_name, int(size)).coeff
    n = int(size) * int(pad)
    mag = np.abs(np.fft.rfft(w, n=n))
    peak = float(mag[0])
    if peak <= 0:
        return 0.0
    centre = float(offset_bins) * pad
    half = pad // 2
    lo = int(max(0, np.floor(centre - half)))
    hi = int(min(mag.size, np.ceil(centre + half) + 1))
    if hi <= lo:
        return -200.0
    local = float(np.max(mag[lo:hi]))
    return float(20.0 * np.log10(max(local, 1e-16) / peak))


def leakage_budget_db(
    *,
    line_hz: float,
    slip: float,
    df_hz: float,
    window_name: str,
    claim_db: float = -50.0,
    k: int = 1,
) -> Dict[str, float]:
    """Can this configuration support a ``claim_db`` sideband measurement?

    ``margin_db`` is how far the fundamental's leakage at the sideband offset
    sits below the claimed sideband level. Positive is good. A margin of 10 dB
    means the leakage contributes at most ~1 dB of amplitude error to the
    sideband; at 0 dB the "sideband" is half leakage and the measurement means
    nothing.
    """
    offset_hz = 2.0 * int(k) * float(slip) * float(line_hz)
    offset_bins = offset_hz / float(df_hz) if df_hz > 0 else float("inf")
    leak = window_leakage_db(window_name, offset_bins) if np.isfinite(offset_bins) else -200.0
    return {
        "offset_hz": float(offset_hz),
        "offset_bins": float(offset_bins),
        "leakage_db": float(leak),
        "claim_db": float(claim_db),
        "margin_db": float(claim_db - leak),
        "window": window_name,
        "df_hz": float(df_hz),
    }


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
_SUPPLY_NAMES = (
    "supply.f1_hz",
    "supply.fundamental_a",
    "supply.rms_a",
    "supply.thd_pct",
    "supply.h3_db",
    "supply.h5_db",
    "supply.h7_db",
    "supply.i_unbalance_pct",
    "supply.zero_seq_ratio",
)
_ROTOR_NAMES = (
    "rotor_bar.lsb_db",
    "rotor_bar.usb_db",
    "rotor_bar.pair_db",
    "rotor_bar.asymmetry_db",
    "rotor_bar.k2_pair_db",
    "rotor_bar.lsb_snr",
    "rotor_bar.usb_snr",
    "rotor_bar.sideband_ratio",
)
_ECC_NAMES = (
    "ecc.lsb_db",
    "ecc.usb_db",
    "ecc.pair_db",
    "ecc.lsb_snr",
    "ecc.sideband_ratio",
)
_EPVA_NAMES = (
    "epva.index_2sf1_db",
    "epva.index_fr_db",
    "epva.index_2f1_db",
    "epva.ellipticity",
)
_BEARING_MODES = ("outer_race", "inner_race", "rolling_element", "cage")
_BEARING_FIELDS = ("pair_db", "max_snr")
_QUALITY_NAMES = (
    "quality.slip_used",
    "quality.slip_rel_error",
    "quality.sideband_pos_error_hz",
    "quality.df_hz",
    "quality.leakage_margin_db",
    "quality.resolvable",
    "quality.three_phase",
)


def _mode_names() -> Tuple[str, ...]:
    out: List[str] = []
    for m in _BEARING_MODES:
        out += [f"bearing.{m}.{f}" for f in _BEARING_FIELDS]
    return tuple(out)


CURRENT_FEATURE_NAMES: Tuple[str, ...] = (
    _SUPPLY_NAMES + _ROTOR_NAMES + _ECC_NAMES + _EPVA_NAMES + _mode_names() + _QUALITY_NAMES
)


@dataclass
class CurrentFeatureConfig:
    """Everything that affects a current feature vector, in one object.

    ``n_fft`` is not a free parameter here either. It sets ``df``, which sets
    the sideband offset in bins, which sets the leakage margin -- so the
    spectral configuration and the fault being looked for are coupled in a way
    they never are for vibration. :meth:`check` reports whether the two agree.
    """

    fs_hz: float = 10000.0
    n_fft: int = 65536
    window_name: str = "blackman_harris"
    line_hz: float = 50.0
    pole_pairs: int = 2
    #: Harmonic order up to which the supply THD is accumulated.
    thd_order: int = 13
    #: Claimed detection floor, dB relative to the fundamental. -50 dB is the
    #: textbook "healthy" level for the 2sf1 pair, so a configuration that
    #: cannot support -50 dB cannot distinguish healthy from one broken bar.
    claim_db: float = -50.0
    #: Minimum leakage margin before the frame is marked unsupported.
    min_leakage_margin_db: float = 10.0
    bearing: Optional[BearingGeometry] = None
    #: Search half-width for a predicted line, as a multiple of the slip-induced
    #: position error. 1.0 means "search exactly as far as the slip uncertainty
    #: requires" -- wider biases the peak upward, narrower misses the line.
    search_k: float = 1.0
    #: Speed/slip input uncertainty, relative. Propagates into the search width.
    speed_rel_error: float = 0.005
    floor_width_hz: float = 20.0

    def df_hz(self) -> float:
        return float(self.fs_hz) / float(self.n_fft)

    def record_s(self) -> float:
        return float(self.n_fft) / float(self.fs_hz)

    def check(self, slip: float) -> Dict[str, float]:
        """Leakage and resolution budget for this config at this slip."""
        b = leakage_budget_db(
            line_hz=self.line_hz, slip=slip, df_hz=self.df_hz(),
            window_name=self.window_name, claim_db=self.claim_db,
        )
        b["min_duration_s"] = min_record_duration_s(
            self.line_hz, slip, window_name=self.window_name, n_fft=self.n_fft
        )
        b["record_s"] = self.record_s()
        b["resolvable"] = 1.0 if (b["record_s"] >= b["min_duration_s"]
                                  and b["margin_db"] >= self.min_leakage_margin_db) else 0.0
        return b

    def as_dict(self) -> Dict[str, object]:
        return {
            "fs_hz": self.fs_hz,
            "n_fft": self.n_fft,
            "window_name": self.window_name,
            "line_hz": self.line_hz,
            "pole_pairs": self.pole_pairs,
            "thd_order": self.thd_order,
            "claim_db": self.claim_db,
            "min_leakage_margin_db": self.min_leakage_margin_db,
            "bearing": self.bearing.name if self.bearing else None,
            "search_k": self.search_k,
            "speed_rel_error": self.speed_rel_error,
            "floor_width_hz": self.floor_width_hz,
            "df_hz": self.df_hz(),
            "record_s": self.record_s(),
        }


@dataclass
class CurrentFrame:
    """One current feature vector plus what it took to make it."""

    vector: np.ndarray
    names: Tuple[str, ...]
    line_hz: float
    slip: float
    spec_version: int
    extras: Dict[str, float] = field(default_factory=dict)
    spectrum: Optional[Spectrum] = None
    epva: Optional[Spectrum] = None

    def as_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}

    def get(self, name: str) -> float:
        return float(self.vector[self.names.index(name)])

    @property
    def supported(self) -> bool:
        """Whether the configuration can support the claim it is making."""
        return self.get("quality.resolvable") > 0.5


class CurrentFeatureExtractor:
    """Turns one current acquisition into the current feature vector.

    Accepts one phase or three. With one phase the three-phase features
    (unbalance, EPVA, ellipticity) are **zero and flagged**, not estimated:
    there is no way to obtain a space vector from one phase, and filling those
    slots with a single-phase surrogate would let a model trained on
    three-phase data score a single-phase frame.
    """

    def __init__(self, config: Optional[CurrentFeatureConfig] = None):
        self.config = config or CurrentFeatureConfig()
        self.names = CURRENT_FEATURE_NAMES

    # -- helpers -----------------------------------------------------------
    def _measure_f1(self, spec: Spectrum) -> Tuple[float, float]:
        """Fundamental frequency and amplitude, by interpolated peak search.

        Searched in a +-5 Hz window around the configured line frequency rather
        than taken as configured: a 50 Hz grid is 50.00 +- 0.05 Hz but an
        inverter output is whatever it was commanded to be, and every sideband
        offset is measured *from the fundamental*, so an error here moves
        everything.
        """
        f0 = self.config.line_hz
        m = np.abs(spec.freq_hz - f0) <= 5.0
        if not np.any(m):
            return float(f0), 0.0
        idx = int(np.flatnonzero(m)[int(np.argmax(spec.value[m]))])
        f, v = interpolate_peak(spec, idx)
        return float(f), float(v)

    def _line(self, spec: Spectrum, f_hz: float, tol_hz: float) -> Dict[str, float]:
        return line_snr(spec, f_hz, tol_hz=tol_hz, floor_width_hz=self.config.floor_width_hz)

    @staticmethod
    def _db(amp: float, ref: float) -> float:
        if ref <= 0 or amp <= 0:
            return -120.0
        return float(20.0 * np.log10(amp / ref))

    # -- extraction --------------------------------------------------------
    def extract(
        self,
        current: Sequence[float] | Sequence[Sequence[float]],
        *,
        slip: Optional[float] = None,
        shaft_hz: Optional[float] = None,
        fs_hz: Optional[float] = None,
        keep_spectra: bool = False,
    ) -> CurrentFrame:
        """Extract the vector from one acquisition.

        Parameters
        ----------
        current
            One phase ``(n,)`` or three phases ``(3, n)``, in amperes.
        slip, shaft_hz
            Exactly one is required. There is no default and no estimate from
            the data: :func:`pdmcore.mcsa.supply.estimate_slip_from_sidebands`
            exists for that and is a separate, checkable step whose error bound
            goes into the frame.
        """
        cfg = self.config
        fs = float(fs_hz if fs_hz is not None else cfg.fs_hz)
        arr = np.asarray(current, dtype=np.float64)
        if arr.ndim == 1:
            phases = arr[None, :]
        elif arr.ndim == 2 and arr.shape[0] in (1, 3):
            phases = arr
        else:
            raise ValueError(f"current must be (n,), (1,n) or (3,n); got {arr.shape}")
        three_phase = phases.shape[0] == 3

        if (slip is None) == (shaft_hz is None):
            raise ValueError(
                "give exactly one of slip or shaft_hz. Every sideband family in "
                "this vector is positioned by slip; a default would place them "
                "all in the wrong bins and the vector would still look valid."
            )
        if slip is None:
            slip = slip_from_shaft_hz(cfg.line_hz, cfg.pole_pairs, float(shaft_hz))
        slip = float(slip)

        n_fft = int(min(cfg.n_fft, 1 << int(np.floor(np.log2(max(2, phases.shape[1]))))))
        spec = spectrum(phases[0], fs, n_fft=n_fft, window_name=cfg.window_name,
                        scaling="rms", detrend="mean")
        f1, a1 = self._measure_f1(spec)

        values: List[float] = []
        extras: Dict[str, float] = {"f1_measured_hz": f1, "n_fft": float(n_fft)}

        # --- supply ------------------------------------------------------
        rms = float(np.sqrt(np.mean(phases[0] ** 2)))
        h_amp = {}
        tol_h = max(1.5 * spec.df_hz, 0.5)
        for h in range(2, cfg.thd_order + 1):
            h_amp[h] = self._line(spec, f1 * h, tol_h)["amplitude"]
        thd = float(100.0 * np.sqrt(sum(v ** 2 for v in h_amp.values())) / a1) if a1 > 0 else 0.0
        if three_phase:
            pv = clarke_transform(phases[0], phases[1], phases[2], fs)
            seq = sequence_components(phases[0], phases[1], phases[2], fs, f1)
            unb = min(unbalance_pct(seq), 1000.0)
            zratio = pv.zero_sequence_ratio()
        else:
            pv = None
            unb = 0.0
            zratio = 0.0
        values += [f1, a1, rms, thd,
                   self._db(h_amp.get(3, 0.0), a1),
                   self._db(h_amp.get(5, 0.0), a1),
                   self._db(h_amp.get(7, 0.0), a1),
                   unb, zratio]

        # --- rotor bar ---------------------------------------------------
        # Search half-width from the slip uncertainty, floored at the bin width.
        pos_err = (sideband_position_error_hz(f1, slip, cfg.speed_rel_error)
                   if slip > 0 else float("inf"))
        tol = max(1.5 * spec.df_hz, cfg.search_k * min(pos_err, 5.0))
        off1 = 2.0 * slip * f1
        lsb = self._line(spec, f1 - off1, tol)
        usb = self._line(spec, f1 + off1, tol)
        k2l = self._line(spec, f1 - 2.0 * off1, tol)
        k2u = self._line(spec, f1 + 2.0 * off1, tol)
        lsb_db = self._db(lsb["amplitude"], a1)
        usb_db = self._db(usb["amplitude"], a1)
        pair_db = self._db(0.5 * (lsb["amplitude"] + usb["amplitude"]), a1)
        k2_db = self._db(0.5 * (k2l["amplitude"] + k2u["amplitude"]), a1)
        sb = sideband_ratio(spec, f1, off1, n_sidebands=2,
                            rel_tol=max(tol / max(f1, 1e-9), 1e-6),
                            floor_width_hz=cfg.floor_width_hz)
        values += [lsb_db, usb_db, pair_db, lsb_db - usb_db, k2_db,
                   lsb["snr"], usb["snr"], sb["ratio"]]

        # --- eccentricity -------------------------------------------------
        fr = (1.0 - slip) * f1 / float(cfg.pole_pairs)
        tol_r = max(1.5 * spec.df_hz, cfg.search_k * cfg.speed_rel_error * fr)
        el = self._line(spec, f1 - fr, tol_r)
        eu = self._line(spec, f1 + fr, tol_r)
        esb = sideband_ratio(spec, f1, fr, n_sidebands=1,
                             rel_tol=max(tol_r / max(f1, 1e-9), 1e-6),
                             floor_width_hz=cfg.floor_width_hz)
        values += [self._db(el["amplitude"], a1), self._db(eu["amplitude"], a1),
                   self._db(0.5 * (el["amplitude"] + eu["amplitude"]), a1),
                   el["snr"], esb["ratio"]]

        # --- EPVA ---------------------------------------------------------
        if pv is not None:
            ep = epva_spectrum(pv, n_fft=n_fft, window_name=cfg.window_name)
            mean_mod = float(np.mean(pv.modulus))
            def _index(f: float, tolerance: float) -> float:
                a = self._line(ep, f, tolerance)["amplitude"]
                return self._db(a, mean_mod)
            values += [_index(off1, tol), _index(fr, tol_r), _index(2.0 * f1, tol_h),
                       pv.ellipticity()]
        else:
            ep = None
            values += [-120.0, -120.0, -120.0, 0.0]

        # --- bearing via the air gap ---------------------------------------
        if cfg.bearing is not None and fr > 0:
            ff = fault_frequencies(cfg.bearing, fr)
            chars = {
                "outer_race": ff.bpfo_hz,
                "inner_race": ff.bpfi_hz,
                "rolling_element": ff.bsf_hz,
                "cage": ff.ftf_hz,
            }
            for mode in _BEARING_MODES:
                fc = chars[mode]
                tol_b = max(1.5 * spec.df_hz, cfg.search_k * cfg.speed_rel_error * fc + 0.02 * fc)
                amps, snrs = [], []
                for sign in (-1.0, 1.0):
                    f = abs(f1 + sign * fc)
                    if f <= 0 or f > spec.freq_hz[-1]:
                        continue
                    d = self._line(spec, f, tol_b)
                    amps.append(d["amplitude"])
                    snrs.append(d["snr"])
                values += [self._db(float(np.mean(amps)) if amps else 0.0, a1),
                           float(max(snrs)) if snrs else 0.0]
        else:
            values += [0.0] * (len(_BEARING_MODES) * len(_BEARING_FIELDS))

        # --- quality ------------------------------------------------------
        budget = cfg.check(slip)
        rel_err = slip_relative_error(slip, cfg.speed_rel_error) if slip > 0 else float("inf")
        values += [slip, min(rel_err, 99.0), min(pos_err, 999.0), spec.df_hz,
                   budget["margin_db"], budget["resolvable"], 1.0 if three_phase else 0.0]
        extras.update({f"budget_{k}": float(v) for k, v in budget.items()
                       if isinstance(v, (int, float))})

        vec = np.asarray(values, dtype=np.float64)
        if vec.size != len(self.names):
            raise AssertionError(
                f"current feature layout drift: {vec.size} values for {len(self.names)} names"
            )
        if not np.all(np.isfinite(vec)):
            extras["n_nonfinite"] = float(int(np.count_nonzero(~np.isfinite(vec))))
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        from . import CURRENT_FEATURE_SPEC_VERSION

        return CurrentFrame(
            vector=vec,
            names=self.names,
            line_hz=f1,
            slip=slip,
            spec_version=CURRENT_FEATURE_SPEC_VERSION,
            extras=extras,
            spectrum=spec if keep_spectra else None,
            epva=ep if keep_spectra else None,
        )

    def extract_many(self, records: Sequence[np.ndarray], slips: Sequence[float],
                     **kwargs) -> np.ndarray:
        if len(records) != len(slips):
            raise ValueError("records and slips must have the same length")
        rows = [self.extract(r, slip=float(s), **kwargs).vector for r, s in zip(records, slips)]
        return np.vstack(rows) if rows else np.empty((0, len(self.names)))
