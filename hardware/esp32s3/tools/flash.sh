#!/usr/bin/env bash
# Build and flash the ESP32-S3 node.
#
#   tools/flash.sh [--iis3dwb] [PORT]
#
# Default sensor profile is the ADXL355. Read README.md before choosing:
# the two parts are not substitutes and the choice changes what this node can
# diagnose.
set -euo pipefail

SENSOR_ARG=()
if [ "${1:-}" = "--iis3dwb" ]; then
    SENSOR_ARG=(-DPDM_SENSOR=IIS3DWB); shift
    echo "building the WIDEBAND profile (IIS3DWB)"
else
    echo "building the LOW-FREQUENCY profile (ADXL355) -- bearing evidence will be weak"
fi
PORT="${1:-${ESPPORT:-/dev/ttyUSB0}}"

command -v idf.py >/dev/null || {
    echo "idf.py not found. Run: . \$IDF_PATH/export.sh" >&2; exit 1; }

python3 "$(dirname "$0")/check_config.py" ${SENSOR_ARG:+--define PDM_SENSOR_IIS3DWB}

idf.py set-target esp32s3
idf.py "${SENSOR_ARG[@]}" build
idf.py -p "$PORT" flash

cat <<'NOTE'

Flashed. First-boot checklist:
  1. `idf.py monitor` and watch for the self-test. A failure means the SPI
     wiring: on a DevKitC with jumpers this is nearly always a long MISO lead or
     a missing ground, not a dead sensor.
  2. Expect ONE cycle then deep sleep. The node does not loop -- deep sleep
     restarts from app_main, so each wake is one iteration. `monitor` will go
     quiet; that is correct.
  3. Confirm the RTC sequence survives: the second wake's frame must carry
     sequence 1, not 0. Sequence 0 twice means RTC_DATA_ATTR is not being
     retained and the cloud will deduplicate away every frame.
  4. Wi-Fi association is 240 mA.s. If the node is on a bench supply this does
     not matter; on a cell, confirm the interval is 6 h and not 1 h.
NOTE
