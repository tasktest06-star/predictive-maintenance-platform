"""pdmedge -- the Linux-class node and gateway agent.

Three of the eight hardware kits in this repository run Linux rather than bare
metal: the ADI CN0549 (Cora Z7), the Raspberry Pi 5 + Hailo-8L, and the NVIDIA
Jetson Orin Nano. They share an agent the way the five MCU kits share `cdsp`.

The agent is deliberately the same *shape* as `cdsp/src/pdm_app.c`:

    battery/health gate -> acquire -> features -> encode -> publish or queue
                        -> drain the backlog while the link is up -> sleep

and it reuses `pdmcore` for the DSP, the feature spec and the wire format, so a
Linux node's frames are indistinguishable from an MCU node's. That is the point:
the cloud must not need to know which tier a device belongs to.

What is *different* on Linux, and why these modules exist:

- **Calibration is real.** A CN0549 channel has a certificate: a piezo
  sensitivity in mV/g, an amplifier gain, an ADC reference. An uncalibrated
  channel is not a worse measurement, it is a different quantity, and
  `calibration.py` refuses to pretend otherwise.
- **The queue must survive power loss.** An MCU queues to internal flash with
  no filesystem in the way. A gateway holding a site's data in a page cache
  loses it on a power cut, so `queue.py` is SQLite with explicit durability.
- **There is more than one sensor.** A gateway aggregates; the loop is per
  channel, not per device.
"""

from .calibration import Calibration, CalibrationError, ChannelCalibration
from .config import AgentConfig, ChannelConfig, load_config
from .queue import FrameQueue, QueueStats
from .source import FileSource, IIOSource, SimSource, Source, SourceError
from .sink import MqttSink, Sink, SinkError, StdoutSink

__all__ = [
    "AgentConfig",
    "Calibration",
    "CalibrationError",
    "ChannelCalibration",
    "ChannelConfig",
    "FileSource",
    "FrameQueue",
    "IIOSource",
    "MqttSink",
    "QueueStats",
    "Sink",
    "SinkError",
    "SimSource",
    "Source",
    "SourceError",
    "StdoutSink",
    "load_config",
]

__version__ = "0.1.0"
