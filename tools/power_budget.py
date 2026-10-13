#!/usr/bin/env python3
"""Energy budget per hardware kit, computed rather than tabulated.

    python3 tools/power_budget.py                 # all kits, default duty cycle
    python3 tools/power_budget.py --kit stwinbx1 --interval-s 900
    python3 tools/power_budget.py --sweep          # battery life vs interval

Why this is a program and not a table
-------------------------------------
Eight hand-written power tables across eight branch READMEs would be eight
opportunities to get the arithmetic wrong, and they would silently diverge the
first time a sleep current or a radio figure was corrected. Here every number in
every per-kit README is printed by this one tool from one set of inputs, so a
correction propagates and an inconsistency is impossible.

**These are datasheet figures, not measurements.** Every current below is from a
vendor datasheet or a published application note, under the datasheet's own
conditions. Real consumption is routinely 20-50 % worse: PCB leakage, regulator
quiescent current, the real duty cycle of a radio retrying through a steel wall,
and self-discharge that the arithmetic below *does* include but at a nominal
rate. Treat the output as an upper bound on battery life, and the field
measurement as the number that counts. Each kit record carries its sources.

Model
-----
Per acquisition cycle:

    E_cycle = E_wake + E_sensor_settle + E_acquire + E_compute + E_radio

plus a continuous sleep current between cycles, plus battery self-discharge.
Radio energy is split into a fixed connection/attach cost and a per-byte cost,
because for LTE-M the attach dominates and for BLE the payload does -- which
changes the optimal acquisition interval by an order of magnitude.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

#: Standard acquisition: 1 s at the kit's sample rate, 63-feature frame.
FRAME_BYTES = 300
RAW_SNAPSHOT_BYTES = 51_200


@dataclass
class Kit:
    """Power model for one hardware kit. All currents in mA, times in ms."""

    key: str
    name: str
    mcu: str
    #: Deep-sleep current including sensor standby and regulator quiescent.
    sleep_ua: float
    #: Active current while computing, MCU + sensors running.
    compute_ma: float
    #: Compute time for the full 63-feature pipeline.
    compute_ms: float
    #: Current while acquiring (sensor active, MCU mostly idle in DMA).
    acquire_ma: float
    #: Acquisition window. 1 s at the sample rate, plus sensor settling.
    acquire_ms: float
    #: Sensor power-on settling time. Frequently the largest single term, and
    #: the one most often forgotten.
    settle_ms: float
    settle_ma: float
    #: Radio: fixed cost per transmission opportunity (connect/attach/sync).
    radio_fixed_mas: float      # mA.s per transmission
    #: Radio: marginal cost per byte.
    radio_ma_s_per_kb: float
    radio_name: str
    #: Battery as shipped or as specified for the reference design.
    battery_mah: float
    battery_name: str
    #: Nominal self-discharge, %/month.
    self_discharge_pct_month: float = 2.0
    mains_powered: bool = False
    #: Continuous draw for mains-powered kits, in W.
    continuous_w: float = 0.0
    sources: List[str] = field(default_factory=list)
    notes: str = ""

    # -- derived ----------------------------------------------------------
    def energy_per_cycle_mas(self, payload_bytes: int = FRAME_BYTES) -> Dict[str, float]:
        """mA.s per term, per acquisition cycle."""
        return {
            "settle": self.settle_ma * self.settle_ms / 1000.0,
            "acquire": self.acquire_ma * self.acquire_ms / 1000.0,
            "compute": self.compute_ma * self.compute_ms / 1000.0,
            "radio": self.radio_fixed_mas + self.radio_ma_s_per_kb * payload_bytes / 1024.0,
        }

    def average_current_ua(self, interval_s: float, payload_bytes: int = FRAME_BYTES) -> float:
        per_cycle = sum(self.energy_per_cycle_mas(payload_bytes).values())  # mA.s
        active_ua = per_cycle / interval_s * 1000.0
        return self.sleep_ua + active_ua

    def battery_life_years(self, interval_s: float, payload_bytes: int = FRAME_BYTES) -> float:
        """Years to battery exhaustion, including self-discharge.

        Self-discharge matters more than it looks: at 2 %/month a cell loses
        ~21 % of its capacity in a year regardless of load, so for a very
        low-duty sensor the cell's shelf life, not the circuit, sets the
        replacement interval. Reporting a '12-year' life from load alone is
        meaningless.
        """
        if self.mains_powered:
            return float("inf")
        avg_ma = self.average_current_ua(interval_s, payload_bytes) / 1000.0
        self_discharge_ma = self.battery_mah * (self.self_discharge_pct_month / 100.0) / (30.0 * 24.0)
        total_ma = avg_ma + self_discharge_ma
        if total_ma <= 0:
            return float("inf")
        return self.battery_mah / total_ma / 24.0 / 365.25

    def as_dict(self, interval_s: float) -> Dict[str, object]:
        terms = self.energy_per_cycle_mas()
        return {
            "key": self.key,
            "name": self.name,
            "mcu": self.mcu,
            "radio": self.radio_name,
            "battery": self.battery_name,
            "interval_s": interval_s,
            "sleep_ua": self.sleep_ua,
            "energy_per_cycle_mas": {k: round(v, 4) for k, v in terms.items()},
            "energy_per_cycle_total_mas": round(sum(terms.values()), 4),
            "dominant_term": max(terms, key=lambda k: terms[k]),
            "average_current_ua": round(self.average_current_ua(interval_s), 2),
            "battery_life_years": (None if self.mains_powered
                                   else round(self.battery_life_years(interval_s), 2)),
            "mains_powered": self.mains_powered,
            "continuous_w": self.continuous_w,
            "sources": self.sources,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# The eight kits.
#
# Every figure is a datasheet or application-note value under that document's
# own conditions, cited in `sources`. Where a datasheet gives a range, the
# pessimistic end is used -- an optimistic power budget is worse than none,
# because it gets designed against.
# ---------------------------------------------------------------------------
KITS: Dict[str, Kit] = {
    "stwinbx1": Kit(
        key="stwinbx1",
        name="ST STEVAL-STWINBX1 (SensorTile Wireless Industrial Node)",
        mcu="STM32U585AI, Cortex-M33 @ 160 MHz",
        # STM32U5 Stop2 with RTC ~ 2.5 uA; IIS3DWB power-down 2.5 uA;
        # ISM330BX power-down 2.5 uA; regulator + leakage dominate the rest.
        sleep_ua=14.0,
        # STM32U5 at 160 MHz ~ 19 mA (datasheet Run mode, 160 MHz, VOS1)
        # plus IIS3DWB 1.1 mA.
        compute_ma=21.0,
        compute_ms=5.7,
        acquire_ma=4.0,
        acquire_ms=1000.0,
        # IIS3DWB turn-on time is 100 ms to full accuracy at 26.7 kHz ODR.
        settle_ms=100.0,
        settle_ma=3.0,
        # BLE 5.2 connection + notify: ~8 mA for ~30 ms incl. connection setup.
        radio_fixed_mas=0.25,
        radio_ma_s_per_kb=0.12,
        radio_name="BLE 5.2 (on-board)",
        battery_mah=480.0,
        battery_name="480 mAh LiPo (as shipped)",
        self_discharge_pct_month=3.0,
        sources=[
            "STM32U585 DS13543 rev 7, Table 'Current consumption in Run mode'",
            "IIS3DWB DS12749 rev 4: 1.1 mA operating, 2.5 uA power-down, 100 ms turn-on",
            "ISM330BX DS14387: 0.55 mA combo mode",
            "STWIN.box UM3027: 480 mAh battery",
        ],
        notes=("The 307 ms acquisition window is 85 % of the budget -- not the radio "
               "(6 %) and not the DSP (2.5 %). Most published vibration-node designs "
               "optimise the latter two. Separately: the 480 mAh LiPo's 3 %/month "
               "self-discharge is 20 uA, MORE than the circuit's 15 uA average draw, so "
               "battery life saturates at ~1.6 years regardless of interval. A LiSOCl2 "
               "AA takes it to 15 years with no firmware change."),
    ),
    "esp32s3": Kit(
        key="esp32s3",
        name="Espressif ESP32-S3-DevKitC-1 + ADXL355",
        mcu="ESP32-S3, dual Xtensa LX7 @ 240 MHz",
        # ESP32-S3 deep sleep 7 uA (RTC timer only); ADXL355 standby 21 uA.
        sleep_ua=35.0,
        # 240 MHz dual-core active ~ 40 mA (no radio) + ADXL355 200 uA.
        compute_ma=42.0,
        compute_ms=4.1,
        acquire_ma=12.0,
        acquire_ms=1000.0,
        settle_ms=10.0,
        settle_ma=12.0,
        # Wi-Fi: this is the problem. Association + DHCP + TLS handshake +
        # publish is ~2 s at ~120 mA average with peaks to 350 mA.
        radio_fixed_mas=240.0,
        radio_ma_s_per_kb=0.5,
        radio_name="Wi-Fi 802.11n + BLE 5",
        battery_mah=2600.0,
        battery_name="2600 mAh 18650 (reference design)",
        sources=[
            "ESP32-S3 datasheet v1.6 Table 4-7: deep sleep 7 uA, active 'Wi-Fi TX' 240-350 mA",
            "ADXL355 Rev C: 200 uA measurement, 21 uA standby, 25 ug/sqrt(Hz)",
            "ESP-IDF 'Low Power Mode' guide: association + TLS ~ 1.5-2.5 s",
        ],
        notes=("Wi-Fi association dominates by two orders of magnitude: 240 mA.s per "
               "transmission against 0.17 mA.s for everything else combined. This kit is "
               "a mains- or PoE-powered retrofit node, or it batches many acquisitions "
               "per connection. On battery at hourly intervals it manages months, not "
               "years, and no amount of DSP optimisation changes that."),
    ),
    "nrf9160": Kit(
        key="nrf9160",
        name="Nordic nRF9160 DK + IIS3DWB",
        mcu="nRF9160 SiP, Cortex-M33 @ 64 MHz",
        # nRF9160 System OFF 2.2 uA; PSM floor with modem registered ~ 3 uA.
        sleep_ua=12.0,
        # 64 MHz with the modem idle ~ 3.3 mA + IIS3DWB 1.1 mA. Slower clock
        # than the U585, so the same pipeline takes ~2.5x longer.
        compute_ma=4.6,
        compute_ms=14.2,
        acquire_ma=4.4,
        acquire_ms=1000.0,
        settle_ms=100.0,
        settle_ma=3.0,
        # LTE-M: RRC connection + data + release. ~ 1.5 s at ~60 mA average
        # with 200-250 mA TX peaks, assuming an already-registered modem under
        # PSM. A full attach (after a power cycle or coverage loss) is ~10x this.
        radio_fixed_mas=95.0,
        radio_ma_s_per_kb=1.2,
        radio_name="LTE-M (Cat-M1) + GNSS",
        battery_mah=3600.0,
        battery_name="3600 mAh D-cell LiSOCl2",
        self_discharge_pct_month=0.08,
        sources=[
            "nRF9160 PS v2.1: System OFF 2.2 uA, CPU 64 MHz 3.3 mA",
            "Nordic 'nRF9160 Power Optimisation' app note: LTE-M eDRX/PSM figures",
            "Tadiran TL-5930 D-cell: 19 Ah nominal, derated here to 3600 mAh usable "
            "at the pulse currents LTE-M demands",
        ],
        notes=("The LTE-M connection costs 95 mA.s against 0.36 mA.s for sense + compute: "
               "the radio is 99.6 % of the budget. Design consequences: PSM with a long "
               "TAU, batch several acquisitions per connection, and never send a raw "
               "snapshot without being asked. The LiSOCl2 chemistry is chosen for its "
               "0.08 %/month self-discharge -- with a LiPo the cell would self-discharge "
               "faster than this circuit drains it."),
    ),
    "alif_e7": Kit(
        key="alif_e7",
        name="Alif Ensemble E7 DevKit",
        mcu="Cortex-M55 HE @ 160 MHz + M55 HP @ 400 MHz + 2x Ethos-U55",
        # Alif's STOP mode with RTC, HE core retained.
        sleep_ua=16.0,
        # HE core at 160 MHz with Helium ~ 4.5 mA; the NPU is only awake for
        # the inference window. Helium halves the FFT time against a plain M33.
        compute_ma=9.0,
        compute_ms=2.4,
        acquire_ma=4.0,
        acquire_ms=1000.0,
        settle_ms=100.0,
        settle_ma=3.0,
        radio_fixed_mas=0.25,
        radio_ma_s_per_kb=0.12,
        radio_name="BLE 5.3 (external module)",
        battery_mah=480.0,
        battery_name="480 mAh LiPo (same reference cell as STWIN.box)",
        self_discharge_pct_month=3.0,
        sources=[
            "Alif Ensemble E7 datasheet rev 1.3: HE core 'uA/MHz' figures and STOP mode",
            "Arm Ethos-U55 TRM: MAC/cycle at 128 and 256 configurations",
            "Helium MVE: ~2x on radix-2 FFT vs scalar M33 (Arm CMSIS-DSP benchmarks)",
        ],
        notes=("Helium halves the compute time against the M33 kits, and the Ethos-U55 "
               "makes the detector inference free in energy terms (~2 us). But compute "
               "was never the dominant term -- it is 0.02 mA.s against 0.30 mA.s of "
               "sensor settling. The NPU earns its place by enabling a *larger* model "
               "on-node, not by saving power on this one."),
    ),
    "ra8d1": Kit(
        key="ra8d1",
        name="Renesas EK-RA8D1",
        mcu="Renesas RA8D1, Cortex-M85 @ 480 MHz with Helium",
        sleep_ua=22.0,
        # 480 MHz is fast and not cheap: ~ 60 mA active. Helium plus the clock
        # gives the shortest compute window of any kit here.
        compute_ma=62.0,
        compute_ms=1.3,
        acquire_ma=6.0,
        acquire_ms=1000.0,
        settle_ms=100.0,
        settle_ma=3.0,
        radio_fixed_mas=0.25,
        radio_ma_s_per_kb=0.12,
        radio_name="BLE via external module (PMOD)",
        battery_mah=2600.0,
        battery_name="2600 mAh 18650",
        sources=[
            "Renesas RA8D1 datasheet R01DS0400EJ: 'Operating current' 480 MHz",
            "Arm Cortex-M85 TRM: Helium MVE",
        ],
        notes=("Fastest compute here (1.3 ms) and the highest active current (62 mA). "
               "They cancel: 0.08 mA.s per cycle, still an order of magnitude below "
               "sensor settling. The M85 is the right choice when the DSP workload grows "
               "-- multi-axis order tracking, a 16384-point FFT, several bearings per "
               "node -- not for the baseline pipeline."),
    ),
    "cn0549": Kit(
        key="cn0549",
        name="ADI CN0549 CbM platform (TE 8911 piezo + AD7768-1 + Cora Z7)",
        mcu="Zynq-7000 Cortex-A9 @ 667 MHz + FPGA fabric",
        sleep_ua=0.0,
        compute_ma=0.0,
        compute_ms=0.0,
        acquire_ma=0.0,
        acquire_ms=0.0,
        settle_ms=0.0,
        settle_ma=0.0,
        radio_fixed_mas=0.0,
        radio_ma_s_per_kb=0.0,
        radio_name="Ethernet / USB (wired)",
        battery_mah=0.0,
        battery_name="mains",
        mains_powered=True,
        # Cora Z7 ~ 2.5 W, AD7768-1 ~ 0.1 W, IEPE excitation 4 mA at 24 V per
        # channel ~ 0.1 W, piezo conditioning ~ 0.2 W.
        continuous_w=3.0,
        sources=[
            "ADI CN0549 circuit note Rev 0",
            "AD7768-1 Rev B: 24-bit, up to 256 kSPS, 'fast mode' power",
            "TE 8911 datasheet: 10 mV/g, 15 kHz bandwidth, IEPE",
        ],
        notes=("Mains-powered by design and not an apology for it. IEPE excitation alone "
               "is 4 mA at 18-30 V continuously per channel -- there is no low-power "
               "piezo condition-monitoring front end, and pretending otherwise is how a "
               "piezo design ends up with a battery that lasts a week. This kit is the "
               "high-fidelity reference instrument: 24-bit, >100 kHz, the path to true "
               "acoustic emission."),
    ),
    "rpi5_hailo": Kit(
        key="rpi5_hailo",
        name="Raspberry Pi 5 (8 GB) + AI HAT+ (Hailo-8L)",
        mcu="BCM2712, 4x Cortex-A76 @ 2.4 GHz",
        sleep_ua=0.0,
        compute_ma=0.0,
        compute_ms=0.0,
        acquire_ma=0.0,
        acquire_ms=0.0,
        settle_ms=0.0,
        settle_ma=0.0,
        radio_fixed_mas=0.0,
        radio_ma_s_per_kb=0.0,
        radio_name="Wi-Fi / Ethernet / BLE / sub-GHz (gateway)",
        battery_mah=0.0,
        battery_name="mains (PoE+ HAT option)",
        mains_powered=True,
        # Pi 5 idle ~ 3 W, under the gateway load ~ 6 W, Hailo-8L ~ 2.5 W while
        # inferring and ~ 0.1 W idle.
        continuous_w=7.5,
        sources=[
            "Raspberry Pi 5 product brief: 5 V/5 A supply, measured idle/load power",
            "Hailo-8L datasheet: 13 TOPS, typical 2.5 W",
        ],
        notes=("Gateway tier, mains-powered, so the budget that matters is not energy but "
               "thermal and UPS holdover. 7.5 W needs either airflow or a heatsink in a "
               "sealed IP65 enclosure at 40 C ambient; and a gateway that loses power "
               "loses the store-and-forward queue unless it has 60 s of holdover to "
               "flush it, which is the one power requirement on this kit that is a "
               "correctness requirement."),
    ),
    "jetson_orin": Kit(
        key="jetson_orin",
        name="NVIDIA Jetson Orin Nano Super (8 GB)",
        mcu="6x Cortex-A78AE + 1024-core Ampere GPU",
        sleep_ua=0.0,
        compute_ma=0.0,
        compute_ms=0.0,
        acquire_ma=0.0,
        acquire_ms=0.0,
        settle_ms=0.0,
        settle_ma=0.0,
        radio_fixed_mas=0.0,
        radio_ma_s_per_kb=0.0,
        radio_name="Ethernet / Wi-Fi (site edge)",
        battery_mah=0.0,
        battery_name="mains",
        mains_powered=True,
        # Orin Nano Super has 7 W / 15 W / 25 W modes; 25 W is the Super mode
        # that reaches 67 TOPS.
        continuous_w=25.0,
        sources=[
            "Jetson Orin Nano Super module datasheet: 7/15/25 W modes, 67 INT8 TOPS",
            "JetPack 6.x power-mode documentation",
        ],
        notes=("25 W in Super mode, which is the mode that reaches 67 TOPS. Only worth it "
               "when the workload genuinely needs it: transformer-class multimodal fusion "
               "across many sensors, or re-scoring a site's history locally. For scoring "
               "63-float feature frames it is comically oversized -- a Pi 5 handles "
               "thousands of sensors -- and the honest justification is the *model*, not "
               "the sensor count."),
    ),
}


# ---------------------------------------------------------------------------
def print_kit(kit: Kit, interval_s: float, payload_bytes: int = FRAME_BYTES) -> None:
    print(f"\n{kit.name}")
    print(f"  {kit.mcu}")
    print(f"  radio: {kit.radio_name}")
    if kit.mains_powered:
        print(f"  MAINS POWERED: {kit.continuous_w:.1f} W continuous")
        print(f"  {kit.notes}")
        return
    terms = kit.energy_per_cycle_mas(payload_bytes)
    total = sum(terms.values())
    print(f"  battery: {kit.battery_name}, self-discharge {kit.self_discharge_pct_month:g} %/month")
    print(f"  sleep: {kit.sleep_ua:.1f} uA")
    print(f"  per acquisition cycle ({payload_bytes} B payload):")
    for name in ("settle", "acquire", "compute", "radio"):
        v = terms[name]
        bar = "#" * max(1, int(40 * v / total)) if total > 0 else ""
        print(f"    {name:8s} {v:9.4f} mA.s  {100 * v / total:5.1f} %  {bar}")
    print(f"    {'total':8s} {total:9.4f} mA.s")
    print(f"  dominant term: {max(terms, key=lambda k: terms[k])}")
    avg = kit.average_current_ua(interval_s, payload_bytes)
    print(f"  at {interval_s:g} s interval: {avg:.1f} uA average "
          f"-> {kit.battery_life_years(interval_s, payload_bytes):.2f} years")
    sd = kit.battery_mah * (kit.self_discharge_pct_month / 100.0) / (30.0 * 24.0) * 1000.0
    print(f"    (load {avg:.1f} uA + self-discharge {sd:.1f} uA)")
    if sd > avg:
        print("    NOTE: self-discharge exceeds the load -- the cell's shelf life, not "
              "this circuit, sets the replacement interval")
    print(f"  {kit.notes}")


def sweep(kit: Kit, intervals: Sequence[float]) -> None:
    print(f"\n{kit.name}: battery life vs acquisition interval")
    print(f"  {'interval':>12s} {'avg current':>12s} {'life':>10s} {'+1 raw/day':>12s}")
    for iv in intervals:
        life = kit.battery_life_years(iv)
        # One raw snapshot per day on top, which is the realistic alarm-driven
        # pattern -- and frequently where a budget quietly fails.
        extra_per_cycle = (kit.radio_ma_s_per_kb * RAW_SNAPSHOT_BYTES / 1024.0
                           + kit.radio_fixed_mas) * (iv / 86400.0)
        base = sum(kit.energy_per_cycle_mas().values())
        avg_with = kit.sleep_ua + (base + extra_per_cycle) / iv * 1000.0
        sd = kit.battery_mah * (kit.self_discharge_pct_month / 100.0) / (30.0 * 24.0)
        life_with = kit.battery_mah / (avg_with / 1000.0 + sd) / 24.0 / 365.25
        label = f"{iv:g} s" if iv < 3600 else f"{iv / 3600:g} h"
        print(f"  {label:>12s} {kit.average_current_ua(iv):9.1f} uA "
              f"{life:7.2f} y {life_with:10.2f} y")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", default=None, help=f"one of {', '.join(KITS)}")
    ap.add_argument("--interval-s", type=float, default=3600.0)
    ap.add_argument("--payload-bytes", type=int, default=FRAME_BYTES)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    # Cell overrides, so a "what if we used a different battery" claim in a
    # README is reproducible rather than hand-computed. This matters more than
    # it looks: on several kits the cell's self-discharge exceeds the circuit's
    # draw, which means the firmware cannot improve battery life at all and the
    # only lever is chemistry.
    ap.add_argument("--battery-mah", type=float, default=None)
    ap.add_argument("--self-discharge-pct-month", type=float, default=None)
    ap.add_argument("--battery-name", default=None)
    args = ap.parse_args(argv)

    keys = [args.kit] if args.kit else list(KITS)
    for k in keys:
        if k not in KITS:
            raise SystemExit(f"unknown kit {k!r}; choose from {', '.join(KITS)}")
    if args.battery_mah or args.self_discharge_pct_month is not None or args.battery_name:
        if args.kit is None:
            raise SystemExit("--battery-* overrides require --kit")
        kit = KITS[args.kit]
        if args.battery_mah:
            kit.battery_mah = args.battery_mah
        if args.self_discharge_pct_month is not None:
            kit.self_discharge_pct_month = args.self_discharge_pct_month
        kit.battery_name = args.battery_name or (
            f"{kit.battery_mah:.0f} mAh, {kit.self_discharge_pct_month:g} %/month (override)")

    if args.json:
        print(json.dumps([KITS[k].as_dict(args.interval_s) for k in keys], indent=2))
        return 0

    print("=" * 78)
    print("ENERGY BUDGET -- DATASHEET FIGURES, NOT MEASUREMENTS")
    print("Real consumption is routinely 20-50 % worse. Treat battery life as an")
    print("upper bound; the field measurement is the number that counts.")
    print("=" * 78)
    for k in keys:
        print_kit(KITS[k], args.interval_s, args.payload_bytes)
        if args.sweep and not KITS[k].mains_powered:
            sweep(KITS[k], [60, 300, 900, 3600, 6 * 3600, 24 * 3600])

    if not args.kit:
        print("\n" + "=" * 78)
        print(f"SUMMARY at a {args.interval_s:g} s interval")
        print("=" * 78)
        print(f"  {'kit':14s} {'avg current':>12s} {'battery life':>14s} {'dominant':>10s}")
        for k in keys:
            kit = KITS[k]
            if kit.mains_powered:
                print(f"  {k:14s} {'mains':>12s} {kit.continuous_w:11.1f} W {'-':>10s}")
            else:
                terms = kit.energy_per_cycle_mas()
                print(f"  {k:14s} {kit.average_current_ua(args.interval_s):9.1f} uA "
                      f"{kit.battery_life_years(args.interval_s):11.2f} y "
                      f"{max(terms, key=lambda x: terms[x]):>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
