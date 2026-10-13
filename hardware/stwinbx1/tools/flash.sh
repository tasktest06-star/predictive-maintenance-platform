#!/usr/bin/env bash
# Flash the STWIN.box over ST-LINK.
#
#   tools/flash.sh [build/pdm_stwinbx1.bin]
#
# Uses STM32_Programmer_CLI (ships with STM32CubeProgrammer). The explicit
# --start-address is deliberate: the STWIN.box ships with ST's FP-SNS-DATALOG2
# firmware and a bootloader at 0x08000000, and flashing over the bootloader
# leaves a brick that needs the BOOT0 jumper to recover. 0x08000000 is correct
# for a standalone image; if you are keeping ST's bootloader, change this and
# the linker script together.
set -euo pipefail

BIN="${1:-build/pdm_stwinbx1.bin}"
ADDR="${FLASH_ADDR:-0x08000000}"

if ! command -v STM32_Programmer_CLI >/dev/null; then
    echo "STM32_Programmer_CLI not found. Install STM32CubeProgrammer, or use:" >&2
    echo "  openocd -f interface/stlink.cfg -f target/stm32u5x.cfg \\" >&2
    echo "    -c \"program $BIN $ADDR verify reset exit\"" >&2
    exit 1
fi
[ -f "$BIN" ] || { echo "no such file: $BIN (build first)" >&2; exit 1; }

echo "flashing $BIN -> $ADDR"
STM32_Programmer_CLI -c port=SWD mode=UR -d "$BIN" "$ADDR" -v -rst

cat <<'NOTE'

Flashed. First-boot checklist:
  1. Watch the LED. Continuous blinking means pdm_app_init failed -- usually a
     feature-count mismatch between the pipeline and firmware/model/pdm_detector.h.
     Re-run tools/export_model.py and rebuild.
  2. Expect a 100 ms sensor settle plus a 307 ms acquisition on the first cycle.
  3. The node advertises as PDM-STWIN. Until a gateway connects it has no RTC
     time and no measured shaft speed, so frames carry SPEED_ESTIMATED and a
     time-since-boot timestamp.
  4. Confirm the self-test passed: a node that boots and streams without ever
     connecting may have failed it. Check the first frame's flags for
     SELFTEST_FAIL.
NOTE
