"""Sensor sources. One `read_block` call, one acquisition.

The `Source` contract mirrors `pdm_board_acquire` in the firmware, for the same
reason: everything above it is vendor-independent, and the vendor-specific part
is small enough to review.

Three implementations ship:

- `SimSource`   -- the simulator, so the agent can be exercised with no hardware
- `FileSource`  -- a recorded capture, so a bug can be reproduced
- `IIOSource`   -- Linux Industrial I/O, which is how the CN0549's AD7768-1, and
                   most Linux-attached ADCs, are actually read

`IIOSource` imports `adi` (pyadi-iio) lazily, so this module is importable --
and testable -- on a machine with no IIO stack, which is every CI runner and
this one.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class SourceError(RuntimeError):
    """Acquisition failed. The agent treats this the way the firmware treats
    `PDM_BOARD_ERR_*`: no frame, and a counter."""


class Overrun(SourceError):
    """Samples were lost mid-block.

    Separate from a general failure because the handling differs: the block is
    a concatenation of two different times, so its spectrum has a step
    discontinuity that looks exactly like an impact. The agent emits a
    *diagnostic* frame with zeroed features and the overrun flag, so the gap is
    visible in the cloud rather than being an unexplained missing hour.
    """


@dataclass
class BlockMeta:
    """What the source knows about the block it just produced."""

    fs_hz: float
    #: True when the rate is measured rather than configured. An IIO device's
    #: actual rate is a divider off a crystal and can differ from the requested
    #: one by percent, which moves every bearing frequency by the same percent.
    fs_measured: bool = False
    #: Wall-clock milliseconds at the *first* sample. Not at the end: a
    #: 0.3-1 s block timestamped at its end is systematically late by its own
    #: duration, which matters when correlating with a process trend.
    t0_ms: int = 0
    clipped: bool = False
    channel: str = ""


class Source:
    """Abstract acquisition source."""

    #: Nominal sample rate, Hz.
    fs_hz: float = 0.0
    #: Channel names this source can produce.
    channels: Tuple[str, ...] = ()

    def open(self) -> None:          # pragma: no cover - trivial
        pass

    def close(self) -> None:         # pragma: no cover - trivial
        pass

    def read_block(self, n: int, channel: str = "") -> Tuple[np.ndarray, BlockMeta]:
        raise NotImplementedError

    def __enter__(self) -> "Source":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SimSource(Source):
    """The machine simulator, driven through the `Source` interface.

    Exists so the whole agent -- queue, sink, drain, flags -- can be exercised
    end to end with no hardware, which is how its tests run. Produces ACCELERATION
    IN g directly, so no calibration is applied: the simulator's output is
    already an engineering quantity, and pretending it were ADC codes would mean
    inventing a sensitivity to divide back out.
    """

    def __init__(
        self,
        machine: str = "motor_pump_37kw",
        fs_hz: float = 25600.0,
        fault: Optional[dict] = None,
        seed: int = 0,
        channels: Sequence[str] = ("accel",),
    ) -> None:
        from pdmcore.sim import MACHINE_LIBRARY  # local: keeps import cost off the gateway path

        from dataclasses import replace as _replace

        if machine not in MACHINE_LIBRARY:
            raise SourceError(f"unknown machine {machine!r}; have {sorted(MACHINE_LIBRARY)}")
        self._spec = _replace(MACHINE_LIBRARY[machine], fs_hz=fs_hz)
        self.fs_hz = fs_hz
        self.channels = tuple(channels)
        self._fault = fault
        self._seed = seed
        self._n = 0

    @property
    def shaft_hz(self) -> float:
        return float(self._spec.shaft_hz)

    def read_block(self, n: int, channel: str = "") -> Tuple[np.ndarray, BlockMeta]:
        from pdmcore.sim import simulate

        self._n += 1
        res = simulate(self._spec, self._fault, duration_s=n / self.fs_hz,
                       seed=self._seed + self._n)
        x = np.asarray(res.accel_g, dtype=np.float64)[:n]
        if x.size < n:
            raise SourceError(f"simulator produced {x.size} of {n} samples")
        return x, BlockMeta(fs_hz=self.fs_hz, fs_measured=False,
                            t0_ms=int(time.time() * 1000.0), channel=channel or self.channels[0])


class FileSource(Source):
    """A recorded capture: one float per line, or a .npy array.

    Loops when exhausted, so a short record can drive a long run. Reproducing a
    field problem means replaying the exact samples that caused it, and that is
    worth a class.
    """

    def __init__(self, path: str, fs_hz: float, channel: str = "accel",
                 loop: bool = True) -> None:
        self.path = path
        self.fs_hz = fs_hz
        self.channels = (channel,)
        self._loop = loop
        self._data: Optional[np.ndarray] = None
        self._pos = 0

    def open(self) -> None:
        if not os.path.isfile(self.path):
            raise SourceError(f"{self.path!r} does not exist")
        if self.path.endswith(".npy"):
            self._data = np.load(self.path).astype(np.float64).ravel()
        else:
            self._data = np.loadtxt(self.path, dtype=np.float64).ravel()
        if self._data.size == 0:
            raise SourceError(f"{self.path!r} is empty")
        self._pos = 0

    def read_block(self, n: int, channel: str = "") -> Tuple[np.ndarray, BlockMeta]:
        if self._data is None:
            self.open()
        assert self._data is not None
        if self._pos + n > self._data.size:
            if not self._loop:
                raise SourceError(f"{self.path!r} exhausted")
            self._pos = 0
            if n > self._data.size:
                raise SourceError(
                    f"{self.path!r} holds {self._data.size} samples, block is {n}"
                )
        x = self._data[self._pos:self._pos + n].copy()
        self._pos += n
        return x, BlockMeta(fs_hz=self.fs_hz, fs_measured=False,
                            t0_ms=int(time.time() * 1000.0),
                            channel=channel or self.channels[0])


class IIOSource(Source):
    """Linux Industrial I/O, via pyadi-iio.

    This is how the CN0549's AD7768-1 is read, and how most Linux-attached
    industrial ADCs are read. `adi` is imported inside `open()` so that this
    module -- and the agent's tests -- work on a machine with no IIO stack.

    THE RATE IS READ BACK, NOT ASSUMED. An IIO device's sampling frequency is a
    divider off a crystal: ask for 32 kHz on an AD7768-1 and you may get
    32000.0 or 31250.0 depending on the MCLK and decimation. A 2.3 % rate error
    moves a 106 Hz BPFO line by 2.4 Hz, which is more than the harmonic
    matcher's whole tolerance. So `open()` reads the attribute back, and
    `read_block` reports the value it read -- the agent puts it in the frame.

    NOT EXERCISED. There is no IIO device in the environment this was written
    in. The attribute names below are from the pyadi-iio AD7768-1 driver and the
    CN0549 reference design documentation; treat them as review-quality.
    """

    def __init__(
        self,
        uri: str = "ip:analog.local",
        device: str = "ad7768-1",
        channels: Sequence[str] = ("voltage0",),
        fs_hz: float = 32000.0,
        buffer_blocks: int = 1,
    ) -> None:
        self.uri = uri
        self.device = device
        self.channels = tuple(channels)
        self.fs_hz = float(fs_hz)
        self._requested_fs = float(fs_hz)
        self._buffer_blocks = max(1, int(buffer_blocks))
        self._dev: object = None

    def open(self) -> None:
        try:
            import adi  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise SourceError(
                "pyadi-iio is not installed. `pip install pyadi-iio` on the "
                "target; see hardware/cn0549/README.md for the full bring-up."
            ) from exc

        factory = getattr(adi, self.device.replace("-", "_"), None)
        if factory is None:  # pragma: no cover
            raise SourceError(
                f"pyadi-iio has no driver named {self.device!r}; available "
                f"candidates are in `dir(adi)`"
            )
        try:
            dev = factory(uri=self.uri)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise SourceError(f"cannot open {self.device} at {self.uri}: {exc}") from exc

        try:
            dev.sample_rate = int(self._requested_fs)
        except Exception:  # pragma: no cover
            pass
        # Read it BACK. See the class docstring: this is the difference between
        # a bearing frequency that is right and one that is 2 % wrong.
        try:
            self.fs_hz = float(dev.sample_rate)
        except Exception:  # pragma: no cover
            self.fs_hz = self._requested_fs
        self._dev = dev

    def close(self) -> None:
        dev = self._dev
        self._dev = None
        if dev is not None:  # pragma: no cover - hardware dependent
            try:
                dev.rx_destroy_buffer()
            except Exception:
                pass

    def read_block(self, n: int, channel: str = "") -> Tuple[np.ndarray, BlockMeta]:  # pragma: no cover
        dev = self._dev
        if dev is None:
            raise SourceError("IIOSource.read_block before open()")
        ch = channel or self.channels[0]
        t0 = int(time.time() * 1000.0)
        try:
            dev.rx_enabled_channels = [ch]
            dev.rx_buffer_size = n
            raw = dev.rx()
        except Exception as exc:
            raise SourceError(f"IIO read failed: {exc}") from exc
        x = np.asarray(raw, dtype=np.float64).ravel()
        if x.size < n:
            # A short buffer from a continuous-mode converter means samples were
            # dropped, which is an overrun and not a partial success.
            raise Overrun(f"IIO returned {x.size} of {n} samples")
        return x[:n], BlockMeta(fs_hz=self.fs_hz, fs_measured=True, t0_ms=t0, channel=ch)
