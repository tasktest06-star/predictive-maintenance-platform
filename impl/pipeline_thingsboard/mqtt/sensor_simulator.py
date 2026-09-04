#!/usr/bin/env python3
"""
sensor_simulator.py — MQTT vibration sensor simulator for ThingsBoard.

Publishes realistic vibration telemetry to the ThingsBoard MQTT broker,
simulating both healthy (normal) and faulty bearing conditions.

Telemetry payload fields
------------------------
rms           : float  — RMS velocity in mm/s (ISO 10816 severity indicator)
kurtosis      : float  — statistical kurtosis (>3 indicates impulsive fault)
crest_factor  : float  — peak / RMS (>6 indicates impulsive fault)
peak_g        : float  — peak acceleration in g
temperature   : float  — bearing housing temperature in °C
rpm           : float  — shaft speed in RPM
bpfo_magnitude: float  — ball-pass frequency outer-race harmonic magnitude (fault signature)
sensor_type   : str    — always "vibration" (used by rule engine filter)

Usage examples
--------------
# Normal operation — 10 sensors, one reading per minute
python sensor_simulator.py \\
    --tokens TOKEN_001,TOKEN_002,TOKEN_003 \\
    --broker localhost --port 1883 \\
    --sensors 10 --interval 60

# Inject fault mode on one sensor (elevated kurtosis + BPFO peaks)
python sensor_simulator.py \\
    --tokens TOKEN_001,TOKEN_002,TOKEN_003 \\
    --fault-device sensor_003 \\
    --interval 30

# Load test — 100 sensors, 10-second interval
python sensor_simulator.py \\
    --sensors 100 --interval 10 --broker localhost

Requirements:
    pip install paho-mqtt
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
import paho.mqtt.client as mqtt


# ---------------------------------------------------------------------------
# Telemetry generation
# ---------------------------------------------------------------------------

@dataclass
class SensorState:
    """Tracks per-sensor state so degradation evolves over time."""
    device_name: str
    access_token: str
    nominal_rpm: float = 1780.0
    fault_mode: bool = False
    # Degradation accumulates gradually in fault mode
    degradation: float = 0.0
    # Baseline temperature at startup
    base_temp: float = field(default_factory=lambda: random.uniform(35.0, 45.0))


def generate_normal_telemetry(state: SensorState) -> dict:
    """Generate healthy machine telemetry with realistic noise."""
    rpm = state.nominal_rpm + random.gauss(0, 15)

    # Healthy RMS: ISO 10816 Zone A-B (0.5–4.0 mm/s with small variations)
    rms = random.gauss(2.1, 0.4)
    rms = max(0.3, rms)

    # Normal kurtosis of vibration signal: ~2.5–3.5 (Gaussian-like)
    kurtosis = random.gauss(2.8, 0.3)
    kurtosis = max(2.0, kurtosis)

    # Crest factor: 3–5 for healthy bearings
    crest_factor = random.gauss(3.8, 0.4)
    crest_factor = max(2.0, crest_factor)

    peak_g = rms * crest_factor * 0.001 * rpm / 60  # plausible physics approximation

    # Temperature slowly drifts around baseline
    temperature = state.base_temp + random.gauss(0, 1.5)

    # BPFO magnitude very low in healthy state
    bpfo_magnitude = random.gauss(0.015, 0.005)
    bpfo_magnitude = max(0.001, bpfo_magnitude)

    return {
        "rms":            round(rms, 3),
        "kurtosis":       round(kurtosis, 3),
        "crest_factor":   round(crest_factor, 3),
        "peak_g":         round(peak_g, 4),
        "temperature":    round(temperature, 1),
        "rpm":            round(rpm, 1),
        "bpfo_magnitude": round(bpfo_magnitude, 4),
        "sensor_type":    "vibration",
        "health_mode":    "normal"
    }


def generate_fault_telemetry(state: SensorState, fault_stage: int) -> dict:
    """
    Generate faulty bearing telemetry with progressive degradation.

    fault_stage: 1 = incipient (detectable by kurtosis/BPFO only)
                 2 = developing (RMS begins to rise)
                 3 = advanced (RMS Zone C, loud harmonics)
                 4 = severe (RMS Zone D, imminent failure)
    """
    rpm = state.nominal_rpm + random.gauss(0, 20)

    # Stage-dependent RMS escalation
    rms_base = {1: 2.5, 2: 5.5, 3: 9.0, 4: 14.0}[fault_stage]
    rms = random.gauss(rms_base, rms_base * 0.1)
    rms = max(0.5, rms)

    # Kurtosis rises significantly in early fault stages (impulsive impacts)
    kurtosis_base = {1: 4.5, 2: 6.2, 3: 7.8, 4: 9.5}[fault_stage]
    kurtosis = random.gauss(kurtosis_base, 0.6)
    kurtosis = max(3.0, kurtosis)

    # Crest factor also increases
    crest_factor_base = {1: 5.0, 2: 6.5, 3: 8.0, 4: 10.5}[fault_stage]
    crest_factor = random.gauss(crest_factor_base, 0.5)
    crest_factor = max(3.0, crest_factor)

    peak_g = rms * crest_factor * 0.0012 * rpm / 60

    # Temperature rises from friction increase
    temp_rise = {1: 3.0, 2: 8.0, 3: 18.0, 4: 35.0}[fault_stage]
    temperature = state.base_temp + temp_rise + random.gauss(0, 2.0)

    # BPFO magnitude is the primary early fault indicator
    bpfo_base = {1: 0.12, 2: 0.35, 3: 0.70, 4: 1.20}[fault_stage]
    bpfo_magnitude = random.gauss(bpfo_base, bpfo_base * 0.15)
    bpfo_magnitude = max(0.05, bpfo_magnitude)

    return {
        "rms":            round(rms, 3),
        "kurtosis":       round(kurtosis, 3),
        "crest_factor":   round(crest_factor, 3),
        "peak_g":         round(peak_g, 4),
        "temperature":    round(temperature, 1),
        "rpm":            round(rpm, 1),
        "bpfo_magnitude": round(bpfo_magnitude, 4),
        "sensor_type":    "vibration",
        "health_mode":    f"fault_stage_{fault_stage}"
    }


def get_fault_stage(degradation: float) -> int:
    """Map degradation accumulator to ISO 10816 fault stage."""
    if degradation < 25:  return 1
    if degradation < 60:  return 2
    if degradation < 90:  return 3
    return 4


# ---------------------------------------------------------------------------
# MQTT connection
# ---------------------------------------------------------------------------

def on_connect(client: mqtt.Client, userdata: dict, flags: dict, rc: int) -> None:
    if rc == 0:
        print(f"[MQTT] Connected to broker  ({userdata['broker']}:{userdata['port']})")
    else:
        print(f"[MQTT] Connection failed with code {rc}", file=sys.stderr)


def on_disconnect(client: mqtt.Client, userdata: dict, rc: int) -> None:
    if rc != 0:
        print(f"[MQTT] Unexpected disconnect (rc={rc}). Will attempt reconnect.", file=sys.stderr)


def on_publish(client: mqtt.Client, userdata: dict, mid: int) -> None:
    pass  # suppress per-message noise; summary is printed in the main loop


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    sensors: list[SensorState],
    broker: str,
    port: int,
    interval: float,
    fault_device: Optional[str],
    run_forever: bool = True,
    iterations: int = 0
) -> None:
    """
    Publish telemetry for all sensors at the configured interval.

    Args:
        sensors:      list of SensorState objects
        broker:       MQTT broker hostname
        port:         MQTT broker port
        interval:     seconds between publishes (per sensor)
        fault_device: device name to inject fault into (or None)
        run_forever:  loop indefinitely if True
        iterations:   number of rounds to run (used when run_forever=False)
    """

    # Each sensor gets its own MQTT client (one access token = one connection)
    clients: list[tuple[SensorState, mqtt.Client]] = []
    userdata = {"broker": broker, "port": port}

    for state in sensors:
        client = mqtt.Client(client_id=state.device_name, userdata=userdata)
        client.username_pw_set(username=state.access_token, password=None)
        client.on_connect    = on_connect
        client.on_disconnect = on_disconnect
        client.on_publish    = on_publish
        try:
            client.connect(broker, port, keepalive=60)
            client.loop_start()
        except Exception as exc:
            print(f"[ERROR] Cannot connect {state.device_name}: {exc}", file=sys.stderr)
            continue
        clients.append((state, client))

    if not clients:
        print("[ERROR] No sensors connected. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSimulating {len(clients)} sensors at {interval}s interval.")
    if fault_device:
        print(f"  Fault injection enabled on: {fault_device}")
    print("Press Ctrl+C to stop.\n")

    round_num = 0
    try:
        while run_forever or round_num < iterations:
            round_num += 1
            published = 0
            for state, client in clients:
                # Determine whether this sensor is in fault mode
                is_fault = fault_device and (
                    state.device_name == fault_device or
                    state.device_name.endswith(fault_device)
                )

                if is_fault:
                    state.fault_mode = True
                    # Gradually increase degradation each publish cycle
                    state.degradation = min(100.0, state.degradation + random.uniform(0.5, 2.0))
                    stage = get_fault_stage(state.degradation)
                    payload = generate_fault_telemetry(state, stage)
                else:
                    payload = generate_normal_telemetry(state)

                result = client.publish(
                    topic="v1/devices/me/telemetry",
                    payload=json.dumps(payload),
                    qos=1
                )

                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    published += 1
                    mode = payload.get("health_mode", "normal")
                    print(
                        f"  [{round_num:04d}] {state.device_name:<22s}  "
                        f"rms={payload['rms']:5.2f} mm/s  "
                        f"kurt={payload['kurtosis']:4.2f}  "
                        f"temp={payload['temperature']:4.1f}°C  "
                        f"rpm={payload['rpm']:6.0f}  "
                        f"mode={mode}"
                    )
                else:
                    print(f"  [WARN] Publish failed for {state.device_name} (rc={result.rc})",
                          file=sys.stderr)

            print(f"--- Round {round_num} complete — {published}/{len(clients)} sensors published ---")
            if run_forever or round_num < iterations:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        for _, client in clients:
            client.loop_stop()
            client.disconnect()
        print("All MQTT connections closed.")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MQTT vibration sensor simulator for ThingsBoard."
    )
    parser.add_argument(
        "--tokens",
        default="",
        help="Comma-separated list of ThingsBoard device access tokens. "
             "If omitted, synthetic tokens are generated (useful for load testing "
             "with a ThingsBoard mock)."
    )
    parser.add_argument("--sensors",   type=int,   default=10,        help="Number of sensors to simulate (default: 10)")
    parser.add_argument("--interval",  type=float, default=60.0,      help="Publish interval in seconds (default: 60)")
    parser.add_argument("--broker",    default="localhost",            help="MQTT broker hostname (default: localhost)")
    parser.add_argument("--port",      type=int,   default=1883,      help="MQTT broker port (default: 1883)")
    parser.add_argument(
        "--fault-device",
        default=None,
        metavar="DEVICE_NAME",
        help="Device name to inject fault telemetry into (e.g. sensor_003). "
             "Degradation escalates gradually across publish cycles."
    )
    parser.add_argument(
        "--nominal-rpm",
        type=float,
        default=1780.0,
        help="Nominal shaft speed for all simulated sensors (default: 1780 RPM)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of publish rounds before stopping. 0 = run forever (default: 0)"
    )
    args = parser.parse_args()

    # Build sensor list
    token_list = [t.strip() for t in args.tokens.split(",") if t.strip()] if args.tokens else []

    sensors: list[SensorState] = []
    for i in range(args.sensors):
        name  = f"sensor_{(i + 1):03d}"
        # Use provided token if available, otherwise generate a placeholder
        token = token_list[i] if i < len(token_list) else f"SIMULATED_TOKEN_{(i + 1):03d}"
        sensors.append(SensorState(
            device_name=name,
            access_token=token,
            nominal_rpm=args.nominal_rpm + random.gauss(0, 20),
            base_temp=random.uniform(38.0, 48.0)
        ))

    run_simulation(
        sensors=sensors,
        broker=args.broker,
        port=args.port,
        interval=args.interval,
        fault_device=args.fault_device,
        run_forever=(args.iterations == 0),
        iterations=args.iterations
    )


if __name__ == "__main__":
    main()
