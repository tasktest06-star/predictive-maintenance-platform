#!/usr/bin/env python3
"""
victoria_exporter.py — ThingsBoard → VictoriaMetrics archival sidecar.

Continuously polls the ThingsBoard Telemetry API for registered devices,
converts the data to Prometheus exposition format, and pushes it to
VictoriaMetrics via the remote-write endpoint.

This sidecar is necessary because ThingsBoard CE stores telemetry internally
but does not expose a Prometheus scrape endpoint. By mirroring all telemetry
to VictoriaMetrics we enable:

  - Long-term retention beyond ThingsBoard's built-in storage window
  - PromQL queries from Grafana for advanced analytics
  - Multi-site aggregation (multiple ThingsBoard instances → one VM)
  - Alerting rules via Prometheus-compatible tools (vmalert, Grafana Alerts)

Architecture:
    ThingsBoard REST API  →  this sidecar  →  VictoriaMetrics /api/v1/import/prometheus

Usage:
    python victoria_exporter.py \\
        --tb-host localhost --tb-port 8080 \\
        --vm-host localhost --vm-port 8428 \\
        --poll-interval 60

Requirements:
    pip install requests
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
log = logging.getLogger("victoria_exporter")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ThingsBoard telemetry keys to export.
# Extend this list if you add new fields to the sensor payload.
TELEMETRY_KEYS = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_g",
    "temperature",
    "rpm",
    "bpfo_magnitude",
    "health_score",
]

# Metric prefix for all exported time-series in VictoriaMetrics.
METRIC_PREFIX = "pm_sensor"  # pm = predictive maintenance


# ---------------------------------------------------------------------------
# ThingsBoard client
# ---------------------------------------------------------------------------

class ThingsBoardClient:
    """Minimal ThingsBoard REST client for telemetry polling."""

    def __init__(self, host: str, port: int, tls: bool = False, timeout: int = 10):
        scheme = "https" if tls else "http"
        self.base = f"{scheme}://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def login(self, username: str, password: str) -> None:
        resp = self.session.post(
            f"{self.base}/api/auth/login",
            json={"username": username, "password": password},
            timeout=self.timeout
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        self.session.headers["X-Authorization"] = f"Bearer {token}"
        log.info("Authenticated with ThingsBoard as %s", username)

    def get_devices(self, page_size: int = 1000) -> list[dict]:
        """Return all devices (paginated, flattened)."""
        devices = []
        page = 0
        while True:
            resp = self.session.get(
                f"{self.base}/api/tenant/devices",
                params={"pageSize": page_size, "page": page},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            devices.extend(data.get("data", []))
            if data.get("hasNext", False):
                page += 1
            else:
                break
        return devices

    def get_latest_telemetry(self, device_id: str, keys: list[str]) -> dict[str, list[dict]]:
        """
        Returns the latest value for each key.
        Response shape: {"rms": [{"ts": 1700000000000, "value": "2.34"}], ...}
        """
        resp = self.session.get(
            f"{self.base}/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries",
            params={"keys": ",".join(keys)},
            timeout=self.timeout
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()

    def get_device_attributes(self, device_id: str, scope: str = "SERVER_SCOPE") -> dict:
        """Returns server-side attributes as a flat key→value dict."""
        resp = self.session.get(
            f"{self.base}/api/plugins/telemetry/DEVICE/{device_id}/values/attributes/{scope}",
            timeout=self.timeout
        )
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return {item["key"]: item["value"] for item in resp.json()}


# ---------------------------------------------------------------------------
# Prometheus exposition format helpers
# ---------------------------------------------------------------------------

def sanitize_label_value(value: str) -> str:
    """Escape double quotes and backslashes in Prometheus label values."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_prometheus_lines(
    device: dict,
    telemetry: dict[str, list[dict]],
    attributes: dict,
) -> list[str]:
    """
    Convert ThingsBoard telemetry to Prometheus text exposition lines.

    Each metric looks like:
        pm_sensor_rms{device="sensor-vib-001",machine="Pump-Motor-01",...} 2.340 1700000000000
    """
    device_id   = device["id"]["id"]
    device_name = device.get("name", device_id)
    label_parts = [
        f'device="{sanitize_label_value(device_name)}"',
        f'device_id="{sanitize_label_value(device_id)}"',
    ]

    # Add selected device attributes as labels for richer filtering in Grafana
    for attr_key in ("machine", "bearing_type", "criticality", "position", "sensor_type"):
        if attr_key in attributes:
            label_parts.append(
                f'{attr_key}="{sanitize_label_value(str(attributes[attr_key]))}"'
            )

    labels = "{" + ",".join(label_parts) + "}"
    lines = []

    for key in TELEMETRY_KEYS:
        if key not in telemetry or not telemetry[key]:
            continue
        entry = telemetry[key][0]  # latest value
        ts_ms  = entry.get("ts")
        raw_v  = entry.get("value")
        if raw_v is None:
            continue
        try:
            value = float(raw_v)
        except (ValueError, TypeError):
            log.debug("Non-numeric value for %s.%s = %s — skipped", device_name, key, raw_v)
            continue

        metric_name = f"{METRIC_PREFIX}_{key}"
        if ts_ms:
            lines.append(f"{metric_name}{labels} {value} {ts_ms}")
        else:
            lines.append(f"{metric_name}{labels} {value}")

    return lines


# ---------------------------------------------------------------------------
# VictoriaMetrics remote-write
# ---------------------------------------------------------------------------

def push_to_victoria(
    vm_base: str,
    lines: list[str],
    session: requests.Session,
    timeout: int = 10
) -> bool:
    """POST Prometheus-format lines to VictoriaMetrics import endpoint."""
    if not lines:
        return True
    body = "\n".join(lines) + "\n"
    try:
        resp = session.post(
            f"{vm_base}/api/v1/import/prometheus",
            data=body.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=timeout
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        log.error("Failed to push to VictoriaMetrics: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main export loop
# ---------------------------------------------------------------------------

@dataclass
class ExporterStats:
    polls:        int = 0
    devices_seen: int = 0
    metrics_sent: int = 0
    errors:       int = 0


def run_exporter(
    tb: ThingsBoardClient,
    vm_base: str,
    poll_interval: int,
    run_forever: bool = True,
    max_iterations: int = 0
) -> None:
    """Continuously poll ThingsBoard and write to VictoriaMetrics."""
    vm_session = requests.Session()
    stats = ExporterStats()
    iteration = 0

    log.info("Starting export loop  poll_interval=%ds  vm_endpoint=%s", poll_interval, vm_base)

    while run_forever or iteration < max_iterations:
        iteration += 1
        stats.polls += 1
        log.info("Poll #%d — fetching device list...", stats.polls)

        try:
            devices = tb.get_devices()
        except Exception as exc:
            log.error("Cannot fetch device list from ThingsBoard: %s", exc)
            stats.errors += 1
            time.sleep(poll_interval)
            continue

        log.info("Found %d devices", len(devices))
        stats.devices_seen = len(devices)
        all_lines: list[str] = []

        for device in devices:
            device_id   = device["id"]["id"]
            device_name = device.get("name", device_id)

            try:
                telemetry  = tb.get_latest_telemetry(device_id, TELEMETRY_KEYS)
                attributes = tb.get_device_attributes(device_id)
            except Exception as exc:
                log.warning("Skipping %s — API error: %s", device_name, exc)
                stats.errors += 1
                continue

            lines = build_prometheus_lines(device, telemetry, attributes)
            if lines:
                all_lines.extend(lines)
                log.debug("%s — %d metrics", device_name, len(lines))

        # Push everything in one HTTP call to reduce round-trips
        if all_lines:
            ok = push_to_victoria(vm_base, all_lines, vm_session)
            if ok:
                stats.metrics_sent += len(all_lines)
                log.info("Pushed %d metric lines to VictoriaMetrics  (total=%d)",
                         len(all_lines), stats.metrics_sent)
            else:
                stats.errors += 1
        else:
            log.info("No telemetry data to export this cycle")

        log.info("Stats: polls=%d  devices=%d  total_metrics=%d  errors=%d",
                 stats.polls, stats.devices_seen, stats.metrics_sent, stats.errors)

        if run_forever or iteration < max_iterations:
            log.info("Sleeping %ds until next poll...", poll_interval)
            time.sleep(poll_interval)

    log.info("Exporter finished.  Final stats: %s", stats)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThingsBoard → VictoriaMetrics archival sidecar."
    )
    # ThingsBoard connection
    parser.add_argument("--tb-host",     default="localhost",           help="ThingsBoard host (default: localhost)")
    parser.add_argument("--tb-port",     type=int, default=8080,        help="ThingsBoard HTTP port (default: 8080)")
    parser.add_argument("--tb-tls",      action="store_true",           help="Use HTTPS for ThingsBoard")
    parser.add_argument("--tb-user",     default="sysadmin@thingsboard.org", help="ThingsBoard username")
    parser.add_argument("--tb-password", default="sysadmin",            help="ThingsBoard password")

    # VictoriaMetrics connection
    parser.add_argument("--vm-host", default="localhost", help="VictoriaMetrics host (default: localhost)")
    parser.add_argument("--vm-port", type=int, default=8428, help="VictoriaMetrics port (default: 8428)")
    parser.add_argument("--vm-tls",  action="store_true", help="Use HTTPS for VictoriaMetrics")

    # Behaviour
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between each full device poll (default: 60)")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Number of poll iterations before stopping. 0 = forever (default: 0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    tb_scheme = "https" if args.tb_tls else "http"
    vm_scheme = "https" if args.vm_tls else "http"
    vm_base   = f"{vm_scheme}://{args.vm_host}:{args.vm_port}"

    tb = ThingsBoardClient(host=args.tb_host, port=args.tb_port, tls=args.tb_tls)
    try:
        tb.login(args.tb_user, args.tb_password)
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to ThingsBoard at %s:%d. Is the service running?",
                  args.tb_host, args.tb_port)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        log.error("Authentication failed: %s", exc)
        sys.exit(1)

    try:
        run_exporter(
            tb=tb,
            vm_base=vm_base,
            poll_interval=args.poll_interval,
            run_forever=(args.iterations == 0),
            max_iterations=args.iterations
        )
    except KeyboardInterrupt:
        log.info("Exporter stopped by user.")


if __name__ == "__main__":
    main()
