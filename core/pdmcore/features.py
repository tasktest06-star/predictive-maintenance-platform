"""The feature contract between firmware and cloud.

This module defines *the* feature vector. Firmware computes it on the MCU and
ships ~60 floats per acquisition; the cloud computes the identical vector from an
archived waveform. Both stamp :data:`pdmcore.FEATURE_SPEC_VERSION` into the
frame, and the ingest service refuses a frame whose version it does not know --
because silently scoring a v1 vector with a v2 model is the kind of failure that
produces confident, wrong work orders for weeks before anyone notices.

Why a fixed vector rather than raw waveforms
--------------------------------------------
A 1 s, 25.6 kHz, 16-bit acquisition is 51 kB. On LTE-M at a realistic 20 kbit/s
that is 20 s of airtime, which is roughly 1 % of a AA-cell annual energy budget
*per acquisition*. The feature vector is 240 B. So the architecture is: features
every cycle, raw waveform only on alarm or on demand ("snapshot on anomaly").
That single decision is what makes a 3-5 year battery life possible, and it is
why the feature set must be rich enough to diagnose from, not just detect.

Cost on the target
------------------
One real FFT of the block, one envelope FFT after decimation, and a single
streaming pass for the time-domain indicators. On a Cortex-M33 at 160 MHz with
CMSIS-DSP, a 4096-point float FFT is ~1.1 ms, so the whole vector lands in
3-6 ms -- see ``docs/02_feature_spec.md`` for the measured per-kit numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import FEATURE_SPEC_VERSION
from .dsp.bearing import BEARING_CATALOG, BearingGeometry, bearing_evidence, fault_frequencies
from .dsp.envelope import DemodBand, envelope_spectrum, select_demodulation_band
from .dsp.features import TIME_FEATURE_NAMES, time_features
from .dsp.integrate import velocity_rms_mm_s
from .dsp.iso import iso_zone
from .dsp.spectrum import Spectrum, band_rms, spectral_shape_features, spectrum

__all__ = [
    "FeatureConfig",
    "FeatureFrame",
    "FeatureExtractor",
    "default_bands_hz",
    "order_bands",
]

_SHAPE_NAMES = ("centroid_hz", "spread_hz", "skew", "kurtosis", "rolloff95_hz", "flatness", "entropy")
_ORDERS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
_BEARING_MODES = ("outer_race", "inner_race", "rolling_element", "cage")
_BEARING_FIELDS = ("ratio", "peak_snr", "n_matched", "sideband_ratio")


def default_bands_hz(fs_hz: float, n_bands: int = 12, f_lo_hz: float = 5.0) -> List[Tuple[float, float]]:
    """Log-spaced band edges from ``f_lo_hz`` to ``0.45*fs``.

    Log spacing, not linear: the information in a vibration spectrum is roughly
    log-distributed, and a linear split wastes 11 of 12 bands on the
    high-frequency region where only broadband energy lives.
    """
    f_hi = 0.45 * fs_hz
    edges = np.geomspace(f_lo_hz, f_hi, n_bands + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bands)]


def order_bands(shaft_hz: float, orders: Sequence[float] = _ORDERS, rel_width: float = 0.06) -> List[Tuple[float, float]]:
    """Narrow bands centred on shaft orders.

    ``rel_width`` of 6 % is wide enough to survive the speed drift of a
    DOL motor and a modest VFD, and narrow enough that 1x and 2x do not merge.
    """
    return [(float(o * shaft_hz * (1 - rel_width)), float(o * shaft_hz * (1 + rel_width))) for o in orders]


@dataclass
class FeatureConfig:
    """Everything that affects the numbers, in one serialisable object.

    Any change here that alters the vector must bump
    :data:`pdmcore.FEATURE_SPEC_VERSION`. The config is persisted next to each
    trained model (see :class:`pdmcore.models.registry.ModelCard`) so a model can
    never be fed a differently configured extractor.
    """

    fs_hz: float = 25600.0
    n_fft: int = 4096
    window_name: str = "hann"
    n_bands: int = 12
    band_f_lo_hz: float = 5.0
    velocity_band_hz: Tuple[float, float] = (10.0, 1000.0)
    iso_class: str = "group2_rigid"
    bearing: Optional[BearingGeometry] = field(default_factory=lambda: BEARING_CATALOG["SKF6205"])
    n_bearing_harmonics: int = 5
    # Fixed demodulation band. None => choose per acquisition with a kurtogram
    # (cloud behaviour). Firmware is always given an explicit band so the MCU
    # runs one fixed filter instead of a band search.
    demod_band_hz: Optional[Tuple[float, float]] = None
    envelope_n_fft: int = 2048
    envelope_decimate_to_hz: Optional[float] = 2500.0
    include_orders: bool = True
    include_bearing: bool = True
    include_shape: bool = True

    def bands(self) -> List[Tuple[float, float]]:
        return default_bands_hz(self.fs_hz, self.n_bands, self.band_f_lo_hz)

    def as_dict(self) -> Dict:
        d = {
            "fs_hz": self.fs_hz,
            "n_fft": self.n_fft,
            "window_name": self.window_name,
            "n_bands": self.n_bands,
            "band_f_lo_hz": self.band_f_lo_hz,
            "velocity_band_hz": list(self.velocity_band_hz),
            "iso_class": self.iso_class,
            "n_bearing_harmonics": self.n_bearing_harmonics,
            "demod_band_hz": list(self.demod_band_hz) if self.demod_band_hz else None,
            "envelope_n_fft": self.envelope_n_fft,
            "envelope_decimate_to_hz": self.envelope_decimate_to_hz,
            "include_orders": self.include_orders,
            "include_bearing": self.include_bearing,
            "include_shape": self.include_shape,
            "spec_version": FEATURE_SPEC_VERSION,
        }
        if self.bearing is not None:
            d["bearing"] = {
                "name": self.bearing.name,
                "n_balls": self.bearing.n_balls,
                "ball_d": self.bearing.ball_d,
                "pitch_d": self.bearing.pitch_d,
                "contact_angle_deg": self.bearing.contact_angle_deg,
            }
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureConfig":
        bearing = None
        if d.get("bearing"):
            b = d["bearing"]
            bearing = BearingGeometry(
                name=b.get("name", "custom"),
                n_balls=int(b["n_balls"]),
                ball_d=float(b["ball_d"]),
                pitch_d=float(b["pitch_d"]),
                contact_angle_deg=float(b.get("contact_angle_deg", 0.0)),
            )
        demod = d.get("demod_band_hz")
        return cls(
            fs_hz=float(d.get("fs_hz", 25600.0)),
            n_fft=int(d.get("n_fft", 4096)),
            window_name=d.get("window_name", "hann"),
            n_bands=int(d.get("n_bands", 12)),
            band_f_lo_hz=float(d.get("band_f_lo_hz", 5.0)),
            velocity_band_hz=tuple(d.get("velocity_band_hz", (10.0, 1000.0))),  # type: ignore[arg-type]
            iso_class=d.get("iso_class", "group2_rigid"),
            bearing=bearing,
            n_bearing_harmonics=int(d.get("n_bearing_harmonics", 5)),
            demod_band_hz=tuple(demod) if demod else None,  # type: ignore[arg-type]
            envelope_n_fft=int(d.get("envelope_n_fft", 2048)),
            envelope_decimate_to_hz=d.get("envelope_decimate_to_hz", 2500.0),
            include_orders=bool(d.get("include_orders", True)),
            include_bearing=bool(d.get("include_bearing", True)),
            include_shape=bool(d.get("include_shape", True)),
        )


@dataclass
class FeatureFrame:
    """One extracted feature vector plus the artefacts used to build it."""

    vector: np.ndarray
    names: Tuple[str, ...]
    shaft_hz: float
    spec_version: int
    extras: Dict[str, float] = field(default_factory=dict)
    spectrum: Optional[Spectrum] = None
    envelope_spectrum: Optional[Spectrum] = None
    demod_band: Optional[DemodBand] = None

    def as_dict(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}

    def get(self, name: str) -> float:
        idx = self.names.index(name)
        return float(self.vector[idx])

    @property
    def n_features(self) -> int:
        return int(self.vector.size)


class FeatureExtractor:
    """Turns a raw block into the platform feature vector.

    The vector layout is fixed by the config and exposed as :attr:`names`, in the
    same order the firmware emits them. ``tools/export_feature_names.py`` on each
    hardware branch regenerates the C header from this list so the two cannot
    drift.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._names: Optional[Tuple[str, ...]] = None

    # -- layout ------------------------------------------------------------
    @property
    def names(self) -> Tuple[str, ...]:
        if self._names is None:
            self._names = tuple(self._build_names())
        return self._names

    def _build_names(self) -> List[str]:
        cfg = self.config
        out: List[str] = [f"time.{n}" for n in TIME_FEATURE_NAMES]
        out += [f"band.{i:02d}_rms" for i in range(cfg.n_bands)]
        if cfg.include_shape:
            out += [f"shape.{n}" for n in _SHAPE_NAMES]
        out += ["iso.velocity_rms_mm_s", "iso.zone_index"]
        if cfg.include_orders:
            out += [f"order.{o:g}x_g" for o in _ORDERS]
            out += ["order.sum_1_5x_g", "order.subharmonic_ratio"]
        if cfg.include_bearing and cfg.bearing is not None:
            for mode in _BEARING_MODES:
                out += [f"bearing.{mode}.{f}" for f in _BEARING_FIELDS]
            out += ["bearing.env_kurtosis", "bearing.env_crest", "bearing.demod_lo_hz", "bearing.demod_hi_hz"]
        return out

    # -- extraction --------------------------------------------------------
    def extract(
        self,
        x: Sequence[float],
        shaft_hz: float,
        *,
        fs_hz: Optional[float] = None,
        keep_spectra: bool = False,
    ) -> FeatureFrame:
        """Extract the feature vector from one acquisition block.

        Parameters
        ----------
        x
            Acceleration block in **g**.
        shaft_hz
            Shaft frequency. Required: without it the order and bearing features
            are meaningless. Supply a tacho measurement, a VFD setpoint read over
            Modbus, or :func:`pdmcore.dsp.ordertrack.estimate_speed_tacholess`.
        keep_spectra
            Attach the spectra to the frame. Useful for plots and debugging;
            firmware obviously does not do this.
        """
        cfg = self.config
        fs = float(fs_hz if fs_hz is not None else cfg.fs_hz)
        x = np.asarray(x, dtype=np.float64).ravel()
        if shaft_hz <= 0:
            raise ValueError("shaft_hz must be positive; order and bearing features depend on it")

        values: List[float] = []
        extras: Dict[str, float] = {}

        # 1. Time-domain indicators (single streaming pass on the MCU).
        tf = time_features(x)
        values.extend(float(v) for v in tf.as_vector())

        # 2. One FFT serves bands, shape, orders and velocity.
        spec = spectrum(x, fs, n_fft=cfg.n_fft, window_name=cfg.window_name, scaling="rms")
        values.extend(float(v) for v in band_rms(spec, cfg.bands()))

        if cfg.include_shape:
            shape = spectral_shape_features(spec)
            values.extend(float(shape[n]) for n in _SHAPE_NAMES)

        vrms = velocity_rms_mm_s(x, fs, band_hz=cfg.velocity_band_hz, n_fft=cfg.n_fft)
        zone = iso_zone(vrms, cfg.iso_class)
        values.append(float(vrms))
        values.append(float("ABCD".index(zone)))
        extras["iso_zone"] = float("ABCD".index(zone))

        # 3. Shaft orders.
        if cfg.include_orders:
            order_amps = []
            for lo, hi in order_bands(shaft_hz):
                order_amps.append(spec.rms_in_band(lo, hi))
            values.extend(float(v) for v in order_amps)
            sum_1_5 = float(np.sum(order_amps[1:]))
            values.append(sum_1_5)
            # Subharmonic ratio: 0.5x relative to 1x. Rises with mechanical
            # looseness and rub, and is near zero on a healthy machine.
            one_x = order_amps[1] if len(order_amps) > 1 else 0.0
            values.append(float(order_amps[0] / one_x) if one_x > 0 else 0.0)

        # 4. Envelope analysis for bearing evidence (second, smaller FFT).
        env_spec = None
        band: Optional[DemodBand] = None
        if cfg.include_bearing and cfg.bearing is not None:
            if cfg.demod_band_hz is not None:
                band = DemodBand(cfg.demod_band_hz[0], cfg.demod_band_hz[1], kurtosis=float("nan"), level=0)
            else:
                band = select_demodulation_band(x, fs)
            env_spec = envelope_spectrum(
                x, fs,
                band_hz=(band.lo_hz, band.hi_hz),
                n_fft=cfg.envelope_n_fft,
                decimate_to_hz=cfg.envelope_decimate_to_hz,
            )
            ev = bearing_evidence(env_spec, cfg.bearing, shaft_hz, n_harmonics=cfg.n_bearing_harmonics)
            for mode in _BEARING_MODES:
                d = ev[mode]
                values.extend(float(d[f]) for f in _BEARING_FIELDS)
            # Envelope-domain impulsiveness: cheap, and a strong early indicator
            # even when no kinematic frequency matches (e.g. lubrication loss).
            env_amp = np.asarray(env_spec.value, dtype=np.float64)
            env_mean = float(np.mean(env_amp))
            env_std = float(np.std(env_amp))
            values.append(float(np.mean(((env_amp - env_mean) / env_std) ** 4)) if env_std > 0 else 0.0)
            values.append(float(np.max(env_amp) / env_mean) if env_mean > 0 else 0.0)
            values.append(float(band.lo_hz))
            values.append(float(band.hi_hz))
            ff = fault_frequencies(cfg.bearing, shaft_hz)
            extras.update({f"truth_{k}": v for k, v in ff.as_dict().items()})

        vec = np.asarray(values, dtype=np.float64)
        if vec.size != len(self.names):
            raise AssertionError(
                f"feature layout drift: produced {vec.size} values for {len(self.names)} names"
            )
        if not np.all(np.isfinite(vec)):
            # Replace rather than raise: one bad acquisition must not stop a
            # fleet pipeline. The count is surfaced so it can be alarmed on.
            bad = int(np.count_nonzero(~np.isfinite(vec)))
            extras["n_nonfinite"] = float(bad)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureFrame(
            vector=vec,
            names=self.names,
            shaft_hz=float(shaft_hz),
            spec_version=FEATURE_SPEC_VERSION,
            extras=extras,
            spectrum=spec if keep_spectra else None,
            envelope_spectrum=env_spec if keep_spectra else None,
            demod_band=band,
        )

    def extract_many(
        self, blocks: Sequence[Sequence[float]], shaft_hz: Sequence[float], **kwargs
    ) -> np.ndarray:
        """Stack vectors for a list of blocks -> ``(n_samples, n_features)``."""
        if len(blocks) != len(shaft_hz):
            raise ValueError("blocks and shaft_hz must have the same length")
        rows = [self.extract(b, f, **kwargs).vector for b, f in zip(blocks, shaft_hz)]
        return np.vstack(rows) if rows else np.empty((0, len(self.names)))
