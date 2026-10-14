"""CN0549 acquisition: AD7768-1 over Linux IIO, with the checks that matter.

    TE 8911 piezo (IEPE) -> CN0540 front end -> AD7768-1 24-bit -> Cora Z7

Subclasses the shared `IIOSource` and adds the three things that are specific to
an IEPE measurement chain, each of which exists because of a specific way this
kit can be silently wrong:

1. **The IEPE bias check.** An IEPE sensor sits on a 2-20 mA constant-current
   supply and idles at a known bias voltage, typically 8-14 V. Read the bias
   and you know the state of the cable: near the supply rail means an open
   circuit (cut cable, unmated connector), near zero means a short, in band
   means a live sensor. See `bias_state`.

   This is the single most valuable feature on the kit, and the reason is in
   `experiments.py`: **a disconnected sensor looks healthier than a healthy
   machine.** An AC-coupled front end with no sensor produces converter noise,
   the smoothest signal the system will ever see, and a one-class detector
   trained on healthy data scores it as *less* anomalous than the real thing.
   The most severe possible fault -- no measurement at all -- is invisible in
   the data and needs an out-of-band check.

2. **The anti-alias filter is a choice, not a default.** The AD7768-1 offers a
   sinc5 path (sharp, long group delay) and a wideband path (flat to 0.433 x
   fs/2, short delay). For a magnitude spectrum the wideband filter is right;
   for impact timing and envelope work the group delay matters. Neither is a
   safe default, so `filter_type` is required and `describe()` prints it.

3. **The DC content is real and must be removed deliberately.** This chain is
   DC-coupled through the converter: the bias voltage appears as a large DC
   offset, and leaving it in puts a huge value in bin 0 that leaks into the low
   orders through any window's skirts. The agent's features exclude DC, but the
   clipping check does not -- so the offset is removed here, at the source, and
   the amount removed is reported rather than discarded.

NOT EXERCISED. There is no CN0549 in the environment this was written in. The
register and attribute names are from the pyadi-iio AD7768-1 driver and the
CN0549 reference documentation; treat them as review-quality.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pdmedge.source import BlockMeta, IIOSource, Overrun, SourceError

#: IEPE bias window, volts. Outside this the cable, not the machine, is the
#: story. The window is wide because sensors differ (TE 8911 idles near 12 V,
#: some PCB units near 8) and because a long cable drops a little.
IEPE_BIAS_MIN_V = 6.0
IEPE_BIAS_MAX_V = 16.0
#: Below this, the line is shorted or the supply is off.
IEPE_SHORT_V = 1.0


@dataclass
class BiasState:
    volts: float
    state: str          # "ok" | "open" | "short" | "unknown"
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.state == "ok"


def classify_bias(volts: float, supply_v: float = 24.0) -> BiasState:
    """Cable state from the IEPE bias voltage.

    Pure function, so it is testable without hardware -- which matters, because
    this is the check that catches the failure the data cannot show.
    """
    if not np.isfinite(volts):
        return BiasState(volts, "unknown", "no bias reading available")
    if volts >= supply_v * 0.85:
        return BiasState(volts, "open",
                         f"{volts:.1f} V is at the {supply_v:.0f} V supply rail: "
                         f"open circuit. Cut cable, unmated connector, or no sensor.")
    if volts <= IEPE_SHORT_V:
        return BiasState(volts, "short",
                         f"{volts:.2f} V: shorted line, or the current source is off.")
    if not (IEPE_BIAS_MIN_V <= volts <= IEPE_BIAS_MAX_V):
        return BiasState(volts, "unknown",
                         f"{volts:.1f} V is outside the {IEPE_BIAS_MIN_V:.0f}-"
                         f"{IEPE_BIAS_MAX_V:.0f} V window. Wrong sensor type, a "
                         f"long cable, or a failing current source.")
    return BiasState(volts, "ok", f"{volts:.1f} V")


class CN0549Source(IIOSource):
    """AD7768-1 on the CN0549, with the IEPE chain's own checks."""

    #: The two converter filter paths, and what each costs.
    FILTERS = {
        # Sharp transition, ~0.5/fs stopband, but a long group delay: the sinc5
        # path's impulse response spans tens of samples, which smears impact
        # timing. Right for a magnitude spectrum, wrong for envelope timing.
        "sinc5": "sharp, long group delay -- spectra",
        # Flat to 0.433 x fs/2 with a short impulse response. Right when impact
        # arrival time or envelope shape matters, which is bearing work.
        "wideband": "flat to 0.433 x Nyquist, short delay -- impacts/envelope",
    }

    def __init__(
        self,
        uri: str = "ip:analog.local",
        channels: Tuple[str, ...] = ("voltage0",),
        fs_hz: float = 32000.0,
        filter_type: str = "wideband",
        remove_dc: bool = True,
        bias_supply_v: float = 24.0,
        full_scale_v: float = 4.096,
    ) -> None:
        if filter_type not in self.FILTERS:
            raise SourceError(
                f"filter_type must be one of {sorted(self.FILTERS)}; there is no "
                f"safe default. sinc5 is sharp with a long group delay (spectra); "
                f"wideband is flat to 0.433 x Nyquist with a short delay "
                f"(impacts and envelopes)."
            )
        super().__init__(uri=uri, device="ad7768-1", channels=channels, fs_hz=fs_hz)
        self.filter_type = filter_type
        self.remove_dc = bool(remove_dc)
        self.bias_supply_v = float(bias_supply_v)
        self.full_scale_v = float(full_scale_v)
        self.last_bias: Optional[BiasState] = None
        self.last_dc_v: float = float("nan")

    # -- bring-up ---------------------------------------------------------
    def open(self) -> None:
        super().open()
        dev = self._dev
        if dev is None:  # pragma: no cover
            raise SourceError("IIO device did not open")
        try:  # pragma: no cover - hardware dependent
            dev.filter_type = self.filter_type
        except Exception as exc:  # pragma: no cover
            raise SourceError(
                f"cannot set filter_type={self.filter_type!r}: {exc}. The "
                f"attribute name differs between pyadi-iio versions; check "
                f"`dir(dev)` on the target."
            ) from exc
        bias = self.read_bias_v()
        self.last_bias = classify_bias(bias, self.bias_supply_v)
        if not self.last_bias.usable:
            # Refusing to open is deliberate. A node that starts happily on a
            # cut cable produces months of data showing a beautifully smooth
            # machine, and nobody questions good news.
            raise SourceError(
                f"IEPE bias check failed: {self.last_bias.detail} "
                f"Fix the sensor chain before acquiring -- a disconnected "
                f"sensor scores as healthier than a healthy machine "
                f"(hardware/cn0549/tools/experiments.py)."
            )

    def read_bias_v(self) -> float:  # pragma: no cover - hardware dependent
        """IEPE bias, from the CN0540's monitor channel.

        Returns NaN when the monitor is unavailable, which `classify_bias`
        reports as "unknown" rather than treating as a pass.
        """
        dev = self._dev
        if dev is None:
            return float("nan")
        for attr in ("sensor_bias_voltage", "bias_voltage", "iepe_bias_v"):
            value = getattr(dev, attr, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float("nan")

    def describe(self) -> str:
        """Printed at start. The filter choice and the bias are configuration
        that silently changes what the numbers mean, so they are not logged at
        debug level."""
        bias = self.last_bias
        lines = [
            f"CN0549 on {self.uri}",
            f"  rate         {self.fs_hz:.1f} Hz (read back from the device)",
            f"  filter       {self.filter_type} -- {self.FILTERS[self.filter_type]}",
            f"  full scale   +-{self.full_scale_v:.3f} V differential",
            f"  DC removal   {'on' if self.remove_dc else 'OFF'}",
            f"  IEPE bias    {bias.detail if bias else 'not read'}"
            f"{'' if (bias and bias.usable) else '   <-- CHECK THE CABLE'}",
        ]
        return "\n".join(lines)

    # -- acquisition ------------------------------------------------------
    def read_block(self, n: int, channel: str = "") -> Tuple[np.ndarray, BlockMeta]:  # pragma: no cover
        x, meta = super().read_block(n, channel)
        # Clipping is checked BEFORE DC removal: the converter clips on the
        # total signal including the bias offset, and subtracting the offset
        # first would hide exactly the condition being looked for.
        codes_full_scale = float(1 << 23)
        if np.max(np.abs(x)) >= 0.995 * codes_full_scale:
            meta.clipped = True
        if self.remove_dc:
            dc = float(np.mean(x))
            self.last_dc_v = dc / codes_full_scale * self.full_scale_v
            x = x - dc
        return x, meta


def main(argv: Optional[list] = None) -> int:
    """`python -m cn0549_source --uri ip:analog.local` -- bring-up only."""
    import argparse

    ap = argparse.ArgumentParser(description="CN0549 bring-up check")
    ap.add_argument("--uri", default="ip:analog.local")
    ap.add_argument("--fs-hz", type=float, default=32000.0)
    ap.add_argument("--filter", default="wideband", choices=sorted(CN0549Source.FILTERS))
    args = ap.parse_args(argv)

    src = CN0549Source(uri=args.uri, fs_hz=args.fs_hz, filter_type=args.filter)
    try:
        src.open()
    except SourceError as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        print(src.describe(), file=sys.stderr)
        return 1
    print(src.describe())
    x, meta = src.read_block(4096)
    print(f"  block        {x.size} samples, rate {meta.fs_hz:.1f} Hz "
          f"(measured={meta.fs_measured}), clipped={meta.clipped}")
    print(f"  DC removed   {src.last_dc_v:+.4f} V")
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
