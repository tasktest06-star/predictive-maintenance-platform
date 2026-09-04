#!/usr/bin/env python3
"""
provision_devices.py — ThingsBoard REST API device provisioning script.

Creates device profiles, devices, and asset hierarchy for the predictive
maintenance platform. Run once against a fresh ThingsBoard instance.

Usage:
    python provision_devices.py --host localhost --port 8080
    python provision_devices.py --host thingsboard.example.com --port 443 --tls

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import time
from typing import Optional
import requests


# ---------------------------------------------------------------------------
# ThingsBoard REST client
# ---------------------------------------------------------------------------

class ThingsBoardClient:
    """Thin wrapper around ThingsBoard REST API."""

    def __init__(self, host: str, port: int, tls: bool = False):
        scheme = "https" if tls else "http"
        self.base_url = f"{scheme}://{host}:{port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.token: Optional[str] = None

    def login(self, username: str, password: str) -> None:
        """Authenticate and store JWT token."""
        resp = self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        resp.raise_for_status()
        self.token = resp.json()["token"]
        self.session.headers["X-Authorization"] = f"Bearer {self.token}"
        print(f"[OK] Authenticated as {username}")

    def get(self, path: str, params: dict = None) -> dict:
        resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{self.base_url}{path}", json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str) -> None:
        resp = self.session.delete(f"{self.base_url}{path}", timeout=10)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Device profile definitions
# ---------------------------------------------------------------------------

VIBRATION_SENSOR_PROFILE = {
    "name": "vibration-sensor",
    "type": "DEFAULT",
    "transportType": "MQTT",
    "defaultRuleChainId": None,  # will use the root rule chain — import vibration_alert_rule_chain.json separately
    "description": "Industrial vibration sensor (STM32 + ADXL357). Publishes RMS, kurtosis, crest factor, temperature, RPM, and BPFO magnitude at 60-second intervals.",
    "profileData": {
        "configuration": {
            "type": "DEFAULT"
        },
        "transportConfiguration": {
            "type": "MQTT",
            "deviceTelemetryTopic": "v1/devices/me/telemetry",
            "deviceAttributesTopic": "v1/devices/me/attributes",
            "sparkplug": False,
            "sendAckOnValidationException": False
        },
        "provisionConfiguration": {
            "type": "DISABLED"
        },
        "alarmRules": [
            {
                "alarmType": "VIBRATION_HIGH",
                "createRules": {
                    "WARNING": {
                        "schedule": {"type": "ANY_TIME"},
                        "condition": {
                            "spec": {"type": "SIMPLE"},
                            "condition": [
                                {
                                    "key": {"type": "TIME_SERIES", "key": "rms"},
                                    "valueType": "NUMERIC",
                                    "predicate": {
                                        "operation": "GREATER",
                                        "value": {"userValue": 4.5},
                                        "type": "NUMERIC"
                                    }
                                }
                            ]
                        },
                        "alarmDetails": "RMS velocity exceeded 4.5 mm/s (ISO 10816 Zone B/C boundary)"
                    },
                    "MAJOR": {
                        "schedule": {"type": "ANY_TIME"},
                        "condition": {
                            "spec": {"type": "SIMPLE"},
                            "condition": [
                                {
                                    "key": {"type": "TIME_SERIES", "key": "rms"},
                                    "valueType": "NUMERIC",
                                    "predicate": {
                                        "operation": "GREATER",
                                        "value": {"userValue": 7.1},
                                        "type": "NUMERIC"
                                    }
                                }
                            ]
                        },
                        "alarmDetails": "RMS velocity exceeded 7.1 mm/s (ISO 10816 Zone C mid)"
                    },
                    "CRITICAL": {
                        "schedule": {"type": "ANY_TIME"},
                        "condition": {
                            "spec": {"type": "SIMPLE"},
                            "condition": [
                                {
                                    "key": {"type": "TIME_SERIES", "key": "rms"},
                                    "valueType": "NUMERIC",
                                    "predicate": {
                                        "operation": "GREATER",
                                        "value": {"userValue": 11.2},
                                        "type": "NUMERIC"
                                    }
                                }
                            ]
                        },
                        "alarmDetails": "RMS velocity exceeded 11.2 mm/s — ISO 10816 Zone D (danger)"
                    }
                },
                "clearRule": {
                    "schedule": {"type": "ANY_TIME"},
                    "condition": {
                        "spec": {"type": "SIMPLE"},
                        "condition": [
                            {
                                "key": {"type": "TIME_SERIES", "key": "rms"},
                                "valueType": "NUMERIC",
                                "predicate": {
                                    "operation": "LESS_OR_EQUAL",
                                    "value": {"userValue": 4.5},
                                    "type": "NUMERIC"
                                }
                            }
                        ]
                    },
                    "alarmDetails": "RMS velocity returned to normal range (<= 4.5 mm/s)"
                },
                "propagate": True,
                "propagateRelationTypes": ["Contains"]
            }
        ]
    }
}

ELECTRICAL_MONITOR_PROFILE = {
    "name": "electrical-monitor",
    "type": "DEFAULT",
    "transportType": "MQTT",
    "description": "Electrical current monitor (ACS712). Tracks current draw, voltage, power factor, and thermal overload indicators.",
    "profileData": {
        "configuration": {"type": "DEFAULT"},
        "transportConfiguration": {
            "type": "MQTT",
            "deviceTelemetryTopic": "v1/devices/me/telemetry",
            "deviceAttributesTopic": "v1/devices/me/attributes",
            "sparkplug": False,
            "sendAckOnValidationException": False
        },
        "provisionConfiguration": {"type": "DISABLED"},
        "alarmRules": [
            {
                "alarmType": "OVERCURRENT",
                "createRules": {
                    "WARNING": {
                        "schedule": {"type": "ANY_TIME"},
                        "condition": {
                            "spec": {"type": "SIMPLE"},
                            "condition": [
                                {
                                    "key": {"type": "TIME_SERIES", "key": "current_a"},
                                    "valueType": "NUMERIC",
                                    "predicate": {
                                        "operation": "GREATER",
                                        "value": {"userValue": 0.9},  # 90% of rated current
                                        "type": "NUMERIC"
                                    }
                                }
                            ]
                        },
                        "alarmDetails": "Current draw exceeds 90% of rated capacity"
                    }
                },
                "propagate": True
            }
        ]
    }
}


# ---------------------------------------------------------------------------
# Asset hierarchy helpers
# ---------------------------------------------------------------------------

def create_asset(client: ThingsBoardClient, name: str, asset_type: str, attributes: dict) -> dict:
    """Create an asset and set its server-side attributes."""
    asset = client.post("/api/asset", {
        "name": name,
        "type": asset_type,
        "label": name
    })
    asset_id = asset["id"]["id"]
    # Set attributes
    client.post(f"/api/plugins/telemetry/ASSET/{asset_id}/attributes/SERVER_SCOPE", attributes)
    print(f"  [+] Asset: {name}  (type={asset_type}, id={asset_id})")
    return asset


def create_relation(client: ThingsBoardClient, from_type: str, from_id: str,
                    to_type: str, to_id: str, relation_type: str = "Contains") -> None:
    """Create a directional relationship between two entities."""
    client.post("/api/relation", {
        "from": {"id": from_id, "entityType": from_type},
        "to":   {"id": to_id,   "entityType": to_type},
        "type": relation_type,
        "typeGroup": "COMMON"
    })


def create_device(client: ThingsBoardClient, name: str, profile_id: str,
                  label: str, attributes: dict) -> tuple[dict, str]:
    """Create a device and return (device_dict, access_token)."""
    device = client.post("/api/device", {
        "name": name,
        "label": label,
        "deviceProfileId": {"id": profile_id, "entityType": "DEVICE_PROFILE"}
    })
    device_id = device["id"]["id"]

    # Set server-side attributes (static device metadata)
    client.post(
        f"/api/plugins/telemetry/DEVICE/{device_id}/attributes/SERVER_SCOPE",
        attributes
    )

    # Retrieve auto-generated access token
    creds = client.get(f"/api/device/{device_id}/credentials")
    access_token = creds.get("credentialsId", "N/A")

    print(f"  [+] Device: {name}  token={access_token}")
    return device, access_token


# ---------------------------------------------------------------------------
# Main provisioning workflow
# ---------------------------------------------------------------------------

def provision(client: ThingsBoardClient) -> None:
    print("\n=== Step 1: Create device profiles ===")

    vib_profile = client.post("/api/deviceProfile", VIBRATION_SENSOR_PROFILE)
    vib_profile_id = vib_profile["id"]["id"]
    print(f"  [+] Profile: vibration-sensor  (id={vib_profile_id})")

    elec_profile = client.post("/api/deviceProfile", ELECTRICAL_MONITOR_PROFILE)
    elec_profile_id = elec_profile["id"]["id"]
    print(f"  [+] Profile: electrical-monitor  (id={elec_profile_id})")

    print("\n=== Step 2: Create asset hierarchy ===")
    print("  Factory → Production Line → Machine → Sensor")

    factory = create_asset(client, "Factory-Alpha", "factory", {
        "location": "São Paulo, Brazil",
        "timezone": "America/Sao_Paulo",
        "contact_email": "maintenance@factory-alpha.local"
    })
    factory_id = factory["id"]["id"]

    line_a = create_asset(client, "Production-Line-A", "production_line", {
        "factory": "Factory-Alpha",
        "shift_hours": "06:00-22:00",
        "criticality": "HIGH"
    })
    line_a_id = line_a["id"]["id"]
    create_relation(client, "ASSET", factory_id, "ASSET", line_a_id)

    machine_01 = create_asset(client, "Pump-Motor-01", "machine", {
        "machine_model": "WEG W22 75kW",
        "nominal_rpm": 1780,
        "bearing_type": "SKF 6312",
        "installation_date": "2022-03-15",
        "criticality": "CRITICAL",
        "production_line": "Production-Line-A",
        "maintainer_email": "maintenance@factory-alpha.local"
    })
    machine_01_id = machine_01["id"]["id"]
    create_relation(client, "ASSET", line_a_id, "ASSET", machine_01_id)

    machine_02 = create_asset(client, "Compressor-02", "machine", {
        "machine_model": "Atlas Copco GA37",
        "nominal_rpm": 2950,
        "bearing_type": "FAG 6209",
        "installation_date": "2021-07-20",
        "criticality": "HIGH",
        "production_line": "Production-Line-A",
        "maintainer_email": "maintenance@factory-alpha.local"
    })
    machine_02_id = machine_02["id"]["id"]
    create_relation(client, "ASSET", line_a_id, "ASSET", machine_02_id)

    print("\n=== Step 3: Create sensor devices ===")

    # Vibration sensors for Pump-Motor-01
    sensor_001, tok_001 = create_device(
        client, "sensor-vib-001", vib_profile_id,
        label="Drive-End Bearing — Pump-Motor-01",
        attributes={
            "sensor_type": "vibration",
            "position": "drive_end",
            "bearing_type": "SKF 6312",
            "nominal_rpm": 1780,
            "criticality": "CRITICAL",
            "machine": "Pump-Motor-01"
        }
    )
    create_relation(client, "ASSET", machine_01_id, "DEVICE", sensor_001["id"]["id"])

    sensor_002, tok_002 = create_device(
        client, "sensor-vib-002", vib_profile_id,
        label="Non-Drive-End Bearing — Pump-Motor-01",
        attributes={
            "sensor_type": "vibration",
            "position": "non_drive_end",
            "bearing_type": "SKF 6312",
            "nominal_rpm": 1780,
            "criticality": "HIGH",
            "machine": "Pump-Motor-01"
        }
    )
    create_relation(client, "ASSET", machine_01_id, "DEVICE", sensor_002["id"]["id"])

    # Electrical monitor for Pump-Motor-01
    elec_001, tok_elec_001 = create_device(
        client, "sensor-elec-001", elec_profile_id,
        label="Motor Current Monitor — Pump-Motor-01",
        attributes={
            "sensor_type": "electrical",
            "rated_current_a": 142.0,
            "voltage_v": 380,
            "machine": "Pump-Motor-01"
        }
    )
    create_relation(client, "ASSET", machine_01_id, "DEVICE", elec_001["id"]["id"])

    # Vibration sensor for Compressor-02
    sensor_003, tok_003 = create_device(
        client, "sensor-vib-003", vib_profile_id,
        label="Drive-End Bearing — Compressor-02",
        attributes={
            "sensor_type": "vibration",
            "position": "drive_end",
            "bearing_type": "FAG 6209",
            "nominal_rpm": 2950,
            "criticality": "HIGH",
            "machine": "Compressor-02"
        }
    )
    create_relation(client, "ASSET", machine_02_id, "DEVICE", sensor_003["id"]["id"])

    print("\n=== Provisioning complete ===")
    print("\nDevice access tokens (use these for MQTT publishing):")
    print(f"  sensor-vib-001  : {tok_001}")
    print(f"  sensor-vib-002  : {tok_002}")
    print(f"  sensor-elec-001 : {tok_elec_001}")
    print(f"  sensor-vib-003  : {tok_003}")
    print()
    print("MQTT publish example:")
    print(f"  mosquitto_pub -h localhost -p 1883 \\")
    print(f"    -u '{tok_001}' \\")
    print(f"    -t 'v1/devices/me/telemetry' \\")
    print(f"    -m '{{\"rms\":2.1,\"kurtosis\":2.8,\"crest_factor\":4.1,\"peak_g\":8.6,\"temperature\":42.3,\"rpm\":1778,\"bpfo_magnitude\":0.03}}'")
    print()
    print("Asset hierarchy:")
    print("  Factory-Alpha")
    print("  └── Production-Line-A")
    print("      ├── Pump-Motor-01")
    print("      │   ├── sensor-vib-001  (drive end)")
    print("      │   ├── sensor-vib-002  (non-drive end)")
    print("      │   └── sensor-elec-001 (current monitor)")
    print("      └── Compressor-02")
    print("          └── sensor-vib-003  (drive end)")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision ThingsBoard device profiles, devices, and asset hierarchy."
    )
    parser.add_argument("--host", default="localhost", help="ThingsBoard host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="ThingsBoard HTTP port (default: 8080)")
    parser.add_argument("--tls", action="store_true", help="Use HTTPS")
    parser.add_argument("--username", default="sysadmin@thingsboard.org", help="ThingsBoard admin username")
    parser.add_argument("--password", default="sysadmin", help="ThingsBoard admin password")
    parser.add_argument("--wait", type=int, default=0,
                        help="Wait N seconds for ThingsBoard to start before provisioning")
    args = parser.parse_args()

    if args.wait > 0:
        print(f"Waiting {args.wait}s for ThingsBoard to initialise...")
        time.sleep(args.wait)

    client = ThingsBoardClient(host=args.host, port=args.port, tls=args.tls)

    try:
        client.login(args.username, args.password)
        provision(client)
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to ThingsBoard at {args.host}:{args.port}.")
        print("  Is the service running?  Try: docker compose up -d thingsboard")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"\n[ERROR] HTTP {exc.response.status_code}: {exc.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
