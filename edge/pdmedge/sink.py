"""Where frames go. MQTT in production, stdout in tests.

The contract is one method, `publish(payload) -> None`, raising `SinkError` on
failure -- because the agent's queue-or-publish decision needs exactly one bit
of information and anything richer invites a sink that reports partial success.

The subtlety is what "accepted" means. `MqttSink` publishes at **QoS 1** and
waits for the broker's PUBACK before returning, so a frame is only dequeued
once a broker has taken responsibility for it. Fire-and-forget (QoS 0) would
make the agent's store-and-forward logic decorative: it would dequeue frames
that never arrived, and the loss would be invisible at both ends.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


class SinkError(RuntimeError):
    """Publication failed. The agent queues the frame and tries later."""


class Sink:
    def connect(self) -> None:       # pragma: no cover - trivial
        pass

    def close(self) -> None:         # pragma: no cover - trivial
        pass

    @property
    def connected(self) -> bool:
        return True

    def publish(self, payload: bytes, topic: str = "") -> None:
        raise NotImplementedError

    def __enter__(self) -> "Sink":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class StdoutSink(Sink):
    """Length-prefixed frames on stdout, like the firmware's virtual device.

    Same framing as `cdsp/tests/virtual_device.c`, so the same test helper
    parses both and a Linux node's output can be fed to the same ingest replay
    path as an MCU node's.
    """

    def __init__(self, stream: object = None) -> None:
        self._stream = stream
        self.published = 0

    def publish(self, payload: bytes, topic: str = "") -> None:
        stream = self._stream or sys.stdout.buffer
        stream.write(len(payload).to_bytes(2, "little"))  # type: ignore[attr-defined]
        stream.write(payload)                             # type: ignore[attr-defined]
        try:
            stream.flush()                                # type: ignore[attr-defined]
        except Exception:                                 # pragma: no cover
            pass
        self.published += 1


class FailingSink(Sink):
    """A sink that fails on a schedule. For testing store-and-forward.

    In the repository rather than in a test file because every tier needs it:
    the firmware has `--tx-fail-every`, and a store-and-forward path that is
    never exercised against a failing link is a feature nobody has tested.
    """

    def __init__(self, fail_every: int = 0, fail_until: int = 0) -> None:
        self.fail_every = int(fail_every)
        self.fail_until = int(fail_until)
        self.calls = 0
        self.published = 0
        self.frames: list = []

    @property
    def connected(self) -> bool:
        return not (self.fail_until and self.calls < self.fail_until)

    def publish(self, payload: bytes, topic: str = "") -> None:
        self.calls += 1
        if self.fail_until and self.calls <= self.fail_until:
            raise SinkError(f"link down (call {self.calls} of {self.fail_until})")
        if self.fail_every and self.calls % self.fail_every == 0:
            raise SinkError(f"link flap on call {self.calls}")
        self.frames.append(payload)
        self.published += 1


class MqttSink(Sink):
    """MQTT at QoS 1, with the PUBACK waited for. `paho-mqtt` imported lazily.

    TLS is on by default and the certificate paths are required when it is: a
    gateway shipping a plant's condition data over an unauthenticated plaintext
    link is a finding, not a configuration choice. `docs/04_security.md` covers
    the per-device X.509 provisioning this expects.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8883,
        topic: str = "pdm/v1/raw",
        client_id: str = "pdm-edge",
        tls: bool = True,
        ca_cert: Optional[str] = None,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
        publish_timeout_s: float = 10.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.topic = topic
        self.client_id = client_id
        self.tls = bool(tls)
        self.ca_cert = ca_cert
        self.certfile = certfile
        self.keyfile = keyfile
        self.username = username
        self.password = password
        self.keepalive = int(keepalive)
        self.publish_timeout_s = float(publish_timeout_s)
        self._client: object = None
        self.published = 0

    def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SinkError("paho-mqtt is not installed") from exc

        if self.tls and not self.ca_cert:
            raise SinkError(
                "tls=True requires ca_cert. Set tls=False explicitly for a "
                "lab bring-up -- but not for a plant: see docs/04_security.md."
            )
        client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)
        if self.username:
            client.username_pw_set(self.username, self.password)
        if self.tls:
            client.tls_set(ca_certs=self.ca_cert, certfile=self.certfile, keyfile=self.keyfile)
        try:
            client.connect(self.host, self.port, keepalive=self.keepalive)
        except OSError as exc:
            raise SinkError(f"cannot connect to {self.host}:{self.port}: {exc}") from exc
        client.loop_start()
        self._client = client

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:  # pragma: no cover
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:
                pass

    @property
    def connected(self) -> bool:
        client = self._client
        return bool(client is not None and getattr(client, "is_connected", lambda: False)())

    def publish(self, payload: bytes, topic: str = "") -> None:  # pragma: no cover
        client = self._client
        if client is None:
            raise SinkError("publish before connect")
        info = client.publish(topic or self.topic, payload, qos=1)  # type: ignore[attr-defined]
        deadline = time.monotonic() + self.publish_timeout_s
        while not info.is_published():
            if time.monotonic() > deadline:
                raise SinkError(f"no PUBACK within {self.publish_timeout_s:.0f} s")
            time.sleep(0.01)
        if info.rc != 0:
            raise SinkError(f"publish returned rc={info.rc}")
        self.published += 1
