"""Engineering units, and the refusal to guess them.

The difference between an instrument and a sensor with a number on it is this
file. A MEMS node reports g because its datasheet says so and nobody checks; a
CN0549 channel reports g because a calibration certificate says what its piezo
produced on a reference shaker at a reference frequency.

The chain, in the order the signal travels it:

    ADC code -> volts at the ADC input
             -> volts at the sensor output   (amplifier gain, AC coupling)
             -> g                            (sensitivity, mV/g)
             -> g corrected                  (frequency response, optional)

Every stage is a number someone can get wrong, and most of them produce a
plausible answer when wrong. A 100 mV/g sensor entered as 10 mV/g reports a
healthy machine at 10x its real amplitude -- which crosses every ISO zone
boundary at once and is immediately obvious. A 100 mV/g sensor entered as
102 mV/g is a 2 % error nobody will ever find, which is why the certificate
value and its date are carried in the frame's metadata rather than compiled in.

WHAT THIS MODULE REFUSES TO DO

It will not silently default a sensitivity. An uncalibrated channel yields
`CalibrationError`, and the agent marks the frame rather than shipping volts
labelled as g. Scoring a channel whose units are unknown against a baseline in
g is not a degraded measurement, it is a different quantity -- and the resulting
anomaly score is confident and meaningless.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Beyond this, a "calibration" is stale enough that it should be questioned.
#: ISO 17025 laboratories typically recommend annual recalibration for
#: accelerometers in industrial service; 2 years is the outer limit at which
#: most condition-monitoring programmes still accept a certificate.
CERTIFICATE_WARN_DAYS = 365
CERTIFICATE_STALE_DAYS = 730


class CalibrationError(ValueError):
    """Raised when a channel's units cannot be established.

    Deliberately an error and not a warning. The alternative -- a default
    sensitivity -- produces numbers in the right range with the wrong scale,
    and the whole point of the high-fidelity tier is that its numbers mean
    something.
    """


@dataclass
class ChannelCalibration:
    """One sensor channel's path from ADC code to g.

    Parameters
    ----------
    sensitivity_mv_per_g
        From the certificate, at the reference frequency. For an IEPE piezo
        this is the dominant term and the one worth arguing about.
    adc_full_scale_v
        Differential full scale of the converter, volts peak. For the AD7768-1
        with a 4.096 V reference in bipolar mode this is 4.096.
    adc_bits
        24 for the AD7768-1. Used only to convert codes; the *effective* number
        of bits is lower and is recorded separately because it decides the real
        noise floor.
    gain
        Any amplification between sensor and ADC. The CN0549's front end is
        unity by default; a charge amplifier or a x10 stage goes here.
    reference_hz
        The frequency the certificate's sensitivity was measured at. Usually
        159.2 Hz (1000 rad/s) for accelerometers.
    certificate_date
        ISO date. Used to decide whether to flag the channel as stale, which is
        a programme-management question that shows up as a data-quality one.
    response
        Optional measured frequency response as (hz, relative_gain) pairs,
        normalised to 1.0 at reference_hz. Applied in the frequency domain by
        :meth:`correct_spectrum`. Absent means "assume flat", which is true
        within a few percent across a piezo's usable band and badly false near
        its mounted resonance.
    """

    name: str
    sensitivity_mv_per_g: float
    adc_full_scale_v: float
    adc_bits: int
    gain: float = 1.0
    reference_hz: float = 159.2
    serial: str = ""
    certificate_date: str = ""
    units: str = "g"
    #: Vendor-stated noise floor, g/sqrt(Hz). Not used in the conversion; it is
    #: the number that decides whether a measurement is sensor-limited.
    noise_density_g: float = float("nan")
    response: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (self.sensitivity_mv_per_g > 0.0):
            raise CalibrationError(
                f"{self.name}: sensitivity must be positive, got "
                f"{self.sensitivity_mv_per_g}"
            )
        if not (self.adc_full_scale_v > 0.0):
            raise CalibrationError(f"{self.name}: adc_full_scale_v must be positive")
        if not (8 <= self.adc_bits <= 32):
            raise CalibrationError(f"{self.name}: adc_bits {self.adc_bits} is implausible")
        if not (self.gain > 0.0):
            raise CalibrationError(f"{self.name}: gain must be positive")
        if self.response:
            hz = [p[0] for p in self.response]
            if any(b <= a for a, b in zip(hz, hz[1:])):
                raise CalibrationError(f"{self.name}: response frequencies must ascend")

    # -- derived ----------------------------------------------------------
    @property
    def counts_per_g(self) -> float:
        """ADC codes per g. The single number the whole chain reduces to."""
        volts_per_g = self.sensitivity_mv_per_g * 1e-3 * self.gain
        codes_per_volt = (1 << (self.adc_bits - 1)) / self.adc_full_scale_v
        return volts_per_g * codes_per_volt

    @property
    def full_scale_g(self) -> float:
        """Acceleration at ADC full scale.

        Worth printing at commissioning: a 100 mV/g sensor on a 4.096 V
        converter clips at 41 g, which a hammer tap on the housing exceeds
        easily. Clipping is flagged, so this shows up as lost data rather than
        wrong data -- but only if someone knows to look.
        """
        return self.adc_full_scale_v / (self.sensitivity_mv_per_g * 1e-3 * self.gain)

    @property
    def resolution_g(self) -> float:
        """One LSB in g. The floor the ADC imposes, before sensor noise."""
        return 1.0 / self.counts_per_g

    def certificate_age_days(self, today: Optional[_dt.date] = None) -> Optional[int]:
        if not self.certificate_date:
            return None
        try:
            d = _dt.date.fromisoformat(self.certificate_date)
        except ValueError as exc:
            raise CalibrationError(
                f"{self.name}: certificate_date {self.certificate_date!r} is not an ISO date"
            ) from exc
        return (today or _dt.date.today() - _dt.timedelta(0)).toordinal() - d.toordinal()

    def certificate_status(self, today: Optional[_dt.date] = None) -> str:
        """``"none"``, ``"current"``, ``"due"`` or ``"stale"``."""
        age = self.certificate_age_days(today)
        if age is None:
            return "none"
        if age > CERTIFICATE_STALE_DAYS:
            return "stale"
        if age > CERTIFICATE_WARN_DAYS:
            return "due"
        return "current"

    # -- conversion -------------------------------------------------------
    def to_g(self, codes: Sequence[float]) -> np.ndarray:
        """ADC codes to g.

        Accepts floats as well as integers, because pyadi-iio hands back
        whatever the IIO channel's `scale` attribute produced and that is not
        always an integer. The conversion is linear, so a pre-scaled input just
        needs the matching `adc_full_scale_v`.
        """
        x = np.asarray(codes, dtype=np.float64)
        return (x / self.counts_per_g).astype(np.float64)

    def volts_to_g(self, volts: Sequence[float]) -> np.ndarray:
        """Volts at the ADC input to g, for a source that already scales."""
        v = np.asarray(volts, dtype=np.float64)
        return v / (self.sensitivity_mv_per_g * 1e-3 * self.gain)

    def response_at(self, hz: Sequence[float]) -> np.ndarray:
        """Relative gain at each frequency, 1.0 where there is no data.

        Log-interpolated in frequency and linear in gain, which is how
        accelerometer response curves are published. Outside the measured range
        it holds the end value rather than extrapolating: a piezo's response
        above its measured band rises towards resonance, and an extrapolation
        that guessed the slope would be confidently wrong in the one region
        where the correction matters most.
        """
        f = np.asarray(hz, dtype=np.float64)
        out = np.ones_like(f)
        if not self.response:
            return out
        rf = np.array([p[0] for p in self.response], dtype=np.float64)
        rg = np.array([p[1] for p in self.response], dtype=np.float64)
        pos = f > 0
        out[pos] = np.interp(np.log(f[pos]), np.log(rf), rg, left=rg[0], right=rg[-1])
        return out

    def correct_spectrum(self, freqs: Sequence[float], magnitude: Sequence[float]) -> np.ndarray:
        """Divide out the measured frequency response.

        Applied to a spectrum rather than to the time series on purpose: the
        correction is a smooth gain curve, and a minimum-phase time-domain
        filter that implemented it would add group delay that matters for
        envelope and impact timing. In the frequency domain it is a divide.
        """
        g = self.response_at(freqs)
        m = np.asarray(magnitude, dtype=np.float64)
        return m / np.where(g > 1e-6, g, 1.0)

    def noise_budget(self, bandwidth_hz: float) -> Dict[str, object]:
        """Is this channel sensor-limited or converter-limited, and by how much?

        Compared over a BANDWIDTH, because the two numbers are not the same
        quantity: the sensor's noise is a density (g/sqrt(Hz)) and the ADC's is
        a quantisation step (g). The first version of this comparison put
        `noise_density_g > resolution_g` in a report line, which is
        dimensionally meaningless and happened to give the right verdict for
        the one channel it was tried on.

        Over a bandwidth B:
            sensor RMS = density * sqrt(B)
            ADC RMS    = LSB / sqrt(12)        (uniform quantisation)

        A 24-bit converter on a 7 ug/sqrt(Hz) piezo is sensor-limited by about
        56 dB, which says the expensive converter is buying dynamic range, not
        resolution -- worth knowing before paying for a better one.
        """
        if not (bandwidth_hz > 0.0):
            raise CalibrationError("bandwidth_hz must be positive")
        adc_rms = self.resolution_g / math.sqrt(12.0)
        sensor_rms = (self.noise_density_g * math.sqrt(bandwidth_hz)
                      if not math.isnan(self.noise_density_g) else float("nan"))
        if math.isnan(sensor_rms):
            return {"bandwidth_hz": bandwidth_hz, "adc_rms_g": adc_rms,
                    "sensor_rms_g": float("nan"), "limited_by": "unknown",
                    "margin_db": float("nan")}
        ratio = sensor_rms / adc_rms if adc_rms > 0 else float("inf")
        return {
            "bandwidth_hz": bandwidth_hz,
            "adc_rms_g": adc_rms,
            "sensor_rms_g": sensor_rms,
            "limited_by": "sensor" if ratio > 1.0 else "adc",
            "margin_db": 20.0 * math.log10(ratio if ratio > 0 else 1e-12),
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "serial": self.serial,
            "sensitivity_mv_per_g": self.sensitivity_mv_per_g,
            "gain": self.gain,
            "adc_full_scale_v": self.adc_full_scale_v,
            "adc_bits": self.adc_bits,
            "reference_hz": self.reference_hz,
            "certificate_date": self.certificate_date,
            "certificate_status": self.certificate_status(),
            "counts_per_g": self.counts_per_g,
            "full_scale_g": self.full_scale_g,
            "resolution_g": self.resolution_g,
            "noise_density_g": self.noise_density_g,
            "adc_noise_rms_g": self.resolution_g / math.sqrt(12.0),
            "response_points": len(self.response),
            "units": self.units,
        }


@dataclass
class Calibration:
    """A device's channels, loaded from a JSON file written at commissioning."""

    channels: Dict[str, ChannelCalibration] = field(default_factory=dict)
    source_path: str = ""

    def __getitem__(self, name: str) -> ChannelCalibration:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise CalibrationError(
                f"no calibration for channel {name!r}; known channels: "
                f"{sorted(self.channels) or 'none'}. Refusing to assume a "
                f"sensitivity -- an uncalibrated channel reports a different "
                f"quantity, not a worse one."
            ) from exc

    def __contains__(self, name: object) -> bool:
        return name in self.channels

    def __len__(self) -> int:
        return len(self.channels)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        if not os.path.isfile(path):
            raise CalibrationError(
                f"calibration file {path!r} does not exist. Write one at "
                f"commissioning from the sensor's certificate; see "
                f"docs/hardware/cn0549.md."
            )
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        chans = raw.get("channels", raw)
        if not isinstance(chans, dict) or not chans:
            raise CalibrationError(f"{path!r} contains no channels")
        out: Dict[str, ChannelCalibration] = {}
        for name, spec in chans.items():
            if not isinstance(spec, dict):
                raise CalibrationError(f"{path!r}: channel {name!r} is not an object")
            resp = [tuple(p) for p in spec.get("response", [])]
            out[name] = ChannelCalibration(
                name=name,
                sensitivity_mv_per_g=float(spec["sensitivity_mv_per_g"]),
                adc_full_scale_v=float(spec["adc_full_scale_v"]),
                adc_bits=int(spec["adc_bits"]),
                gain=float(spec.get("gain", 1.0)),
                reference_hz=float(spec.get("reference_hz", 159.2)),
                serial=str(spec.get("serial", "")),
                certificate_date=str(spec.get("certificate_date", "")),
                units=str(spec.get("units", "g")),
                noise_density_g=float(spec.get("noise_density_g", float("nan"))),
                response=resp,  # type: ignore[arg-type]
            )
        return cls(channels=out, source_path=path)

    def report(self, bandwidth_hz: float = 0.0) -> str:
        """Human-readable summary, printed at agent start.

        Printed rather than logged at debug level because a wrong sensitivity is
        the single most consequential configuration error on this tier, and the
        only defence is that someone reads it once.
        """
        lines = [f"calibration: {len(self.channels)} channel(s) from {self.source_path or '<memory>'}"]
        for name in sorted(self.channels):
            c = self.channels[name]
            status = c.certificate_status()
            mark = {"none": "  NO CERTIFICATE", "stale": "  STALE", "due": "  RECALIBRATION DUE"}.get(status, "")
            lines.append(
                f"  {name:10s} {c.sensitivity_mv_per_g:7.2f} mV/g  gain {c.gain:4.1f}  "
                f"full scale {c.full_scale_g:7.1f} g  resolution "
                f"{c.resolution_g * 1e6:7.2f} ug  cert {c.certificate_date or '-'}{mark}"
            )
            if not math.isnan(c.noise_density_g) and bandwidth_hz:
                b = c.noise_budget(bandwidth_hz)
                lines.append(
                    f"             over {bandwidth_hz / 1000.0:.1f} kHz: sensor "
                    f"{float(b['sensor_rms_g']) * 1e6:8.1f} ug RMS, ADC "
                    f"{float(b['adc_rms_g']) * 1e6:6.2f} ug RMS -- "
                    f"{b['limited_by']}-limited by "
                    f"{float(b['margin_db']):.0f} dB"
                )
            elif not math.isnan(c.noise_density_g):
                lines.append(
                    f"             sensor noise {c.noise_density_g * 1e6:.1f} ug/sqrt(Hz), "
                    f"ADC LSB {c.resolution_g * 1e6:.2f} ug "
                    f"(pass bandwidth_hz for the comparison -- a density and a "
                    f"step are not the same quantity)"
                )
        return "\n".join(lines)
