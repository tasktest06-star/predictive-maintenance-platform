"""Agent configuration, read from a file rather than from argv.

A gateway's configuration outlives the shell that started it: it is edited by
whoever commissions the site, read back by whoever debugs it, and must survive a
reboot. So it is a file, it is validated on load, and the validation refuses
rather than defaults.

`tomllib` where available (3.11+), JSON otherwise, chosen by extension. No
third-party dependency for a config loader.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class ConfigError(ValueError):
    pass


@dataclass
class ChannelConfig:
    """One sensor channel on this device."""

    name: str
    #: Shaft rate for the machine this channel watches. Required: order and
    #: bearing features computed against a guessed speed are worse than no
    #: features, because they look plausible and point at the wrong bearing.
    shaft_hz: float
    asset_id: str = ""
    #: Source channel name, if it differs from `name`.
    source_channel: str = ""
    bearing: str = "SKF6205"
    demod_band_hz: Optional[Tuple[float, float]] = None
    #: Already in g (a simulator or a pre-scaled IIO channel), so no
    #: calibration is applied. Set deliberately, never by default.
    units_are_g: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("channel needs a name")
        if not (self.shaft_hz > 0.0):
            raise ConfigError(
                f"channel {self.name!r}: shaft_hz must be positive. A shaft "
                f"speed is not optional -- every order and bearing feature is "
                f"computed against it."
            )
        if self.demod_band_hz is not None:
            lo, hi = self.demod_band_hz
            if not (0.0 < lo < hi):
                raise ConfigError(f"channel {self.name!r}: bad demod band {self.demod_band_hz}")


@dataclass
class AgentConfig:
    device_id: int
    fs_hz: float
    block_samples: int
    interval_s: float = 3600.0
    n_fft: int = 4096
    envelope_n_fft: int = 2048
    channels: List[ChannelConfig] = field(default_factory=list)
    calibration_path: str = ""
    queue_path: str = "/var/lib/pdm/queue.sqlite"
    queue_max_frames: int = 100_000
    #: Frames drained per successful publish. Bounded so a large backlog cannot
    #: monopolise the uplink -- a gateway flushing 50 000 frames at once starves
    #: live data for minutes.
    max_drain: int = 64
    sink: Dict[str, object] = field(default_factory=dict)
    tier: str = "node"            # node | gateway | edge
    labels: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.fs_hz > 0.0):
            raise ConfigError("fs_hz must be positive")
        if self.block_samples < self.n_fft:
            raise ConfigError(
                f"block_samples {self.block_samples} is shorter than n_fft "
                f"{self.n_fft}: the spectral stage would have nothing to average"
            )
        if self.block_samples < self.envelope_n_fft:
            raise ConfigError(
                f"block_samples {self.block_samples} cannot feed a "
                f"{self.envelope_n_fft}-point envelope FFT. Decimation cannot "
                f"create resolution -- only record length can."
            )
        if self.n_fft & (self.n_fft - 1):
            raise ConfigError(f"n_fft {self.n_fft} is not a power of two")
        if not self.channels:
            raise ConfigError("no channels configured")
        names = [c.name for c in self.channels]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate channel names: {names}")
        if self.tier not in ("node", "gateway", "edge"):
            raise ConfigError(f"tier {self.tier!r} is not node, gateway or edge")
        if self.interval_s <= 0:
            raise ConfigError("interval_s must be positive")
        duty = self.block_samples / self.fs_hz / self.interval_s
        if duty > 1.0:
            raise ConfigError(
                f"a {self.block_samples / self.fs_hz:.1f} s block every "
                f"{self.interval_s:.0f} s is a duty cycle of {duty:.1f} -- the "
                f"agent cannot keep up with its own schedule"
            )


def load_config(path: str) -> AgentConfig:
    if not os.path.isfile(path):
        raise ConfigError(f"config {path!r} does not exist")
    raw: Dict[str, object]
    if path.endswith(".toml"):
        try:
            import tomllib  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "TOML config needs Python 3.11+ (tomllib). Use a .json config "
                "on an older interpreter."
            ) from exc
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    chans_raw = raw.get("channels") or []
    if not isinstance(chans_raw, list):
        raise ConfigError("channels must be a list")
    channels = []
    for spec in chans_raw:
        if not isinstance(spec, dict):
            raise ConfigError("each channel must be an object")
        band = spec.get("demod_band_hz")
        channels.append(ChannelConfig(
            name=str(spec.get("name", "")),
            shaft_hz=float(spec.get("shaft_hz", 0.0)),
            asset_id=str(spec.get("asset_id", "")),
            source_channel=str(spec.get("source_channel", "")),
            bearing=str(spec.get("bearing", "SKF6205")),
            demod_band_hz=(float(band[0]), float(band[1])) if band else None,
            units_are_g=bool(spec.get("units_are_g", False)),
        ))

    dev = raw.get("device_id", 0)
    device_id = int(dev, 16) if isinstance(dev, str) else int(dev)
    if device_id == 0:
        raise ConfigError(
            "device_id must be set and non-zero: the ingest service "
            "deduplicates on (device_id, sequence), so a fleet sharing an id "
            "discards all but one node's data"
        )

    return AgentConfig(
        device_id=device_id,
        fs_hz=float(raw.get("fs_hz", 0.0)),
        block_samples=int(raw.get("block_samples", 0)),
        interval_s=float(raw.get("interval_s", 3600.0)),
        n_fft=int(raw.get("n_fft", 4096)),
        envelope_n_fft=int(raw.get("envelope_n_fft", 2048)),
        channels=channels,
        calibration_path=str(raw.get("calibration_path", "")),
        queue_path=str(raw.get("queue_path", "/var/lib/pdm/queue.sqlite")),
        queue_max_frames=int(raw.get("queue_max_frames", 100_000)),
        max_drain=int(raw.get("max_drain", 64)),
        sink=dict(raw.get("sink", {})),  # type: ignore[arg-type]
        tier=str(raw.get("tier", "node")),
        labels=dict(raw.get("labels", {})),  # type: ignore[arg-type]
    )
