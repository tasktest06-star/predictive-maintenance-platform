# Predictive maintenance platform — branch index

Eleven branches. `platform/core` is the shared foundation; eight `hw/*` branches
each take it to one hardware kit; `research/*` is the literature work. Nothing
has been flashed to a board or started in Docker — every branch carries its own
honest-status table, and [`docs/07_production_readiness.md`](docs/07_production_readiness.md)
carries the platform-wide one.

## Start here

```bash
# The shared core: DSP, features, models, health chain, wire format, cloud
make test                                     # 116 tests (core + edge)
make -C cdsp test                             # 1113 C checks against the Python reference

# A firmware node, on the host, emitting real frames
make -C cdsp virtual-device
PYTHONPATH=core:tools python3 -m pytest core/tests/test_virtual_device.py -v

# A Linux node or gateway, with no sensor attached
PYTHONPATH=core:edge python3 -m pdmedge --self-test
```

## Branches

Full per-branch detail — what is in each file, what each branch measured, and
what it did not run — is in
**[`docs/08_branch_catalogue.md`](docs/08_branch_catalogue.md)**.

| Branch | Kit | What it is, and the finding worth reading |
|---|---|---|
| **`platform/core`** | — | The shared layer: a calibrated DSP library verified by Parseval invariants, a C twin pinned to it by 1 113 generated checks, a 300-byte wire format, the health/RUL chain, the ingest service and compose stack, and [`edge/`](edge/) — the Linux-tier agent |
| **`research/code-backed-deep-dive`** | — | 2025–26 papers that actually ship code, harvested and filtered rather than curated by hand. The aggregator false positives are removed systematically, and the audit trail is committed |
| `hw/stwinbx1-stm32u5` | ST STEVAL-STWINBX1 | The reference node. **The shipped 480 mAh LiPo's self-discharge exceeds the circuit's draw**, so battery life saturates at 1.6 y no matter what the firmware does; a LiSOCl₂ cell takes it to 15 y with no code change |
| `hw/esp32s3-adxl355` | ESP32-S3 + ADXL355 / IIS3DWB | Two sensor profiles on one board. The ADXL355 is quieter and rolls off at 1 kHz, **below the resonances bearing impacts excite** — flagged weak in its own config |
| `hw/nrf9160-cellular` | Nordic nRF9160 DK | Cellular. **The radio is 98 % of the cycle energy**, which inverts every other kit's optimisation priority; PSM gives 9.6 y. Also the only kit that measures its true sample rate |
| `hw/alif-e7-ethos-u55` | Alif Ensemble E7 | **The NPU optimises 0.0005 % of the energy budget.** A node on this €200 part has slightly *worse* battery life than one on the €90 STM32U5. What it buys is a model class, and that model is not written |
| `hw/renesas-ra8d1-helium` | Renesas EK-RA8D1 | Three axes and a real tachometer. **Order tracking turned out to be nearly pointless at bearing orders** (1.19× at order 6) and worth 1.92× at gear mesh — measured, against a docstring that claimed 8–10 dB |
| `hw/cn0549-piezo-ae` | ADI CN0549 | The reference instrument. **24-bit buys nothing** — every chain here is sensor-limited by 67–718× — and **a cut cable is diagnosed as mechanical looseness at 97 % confidence**, which is why the IEPE bias check refuses to start |
| `hw/rpi5-hailo-gateway` | Pi 5 + Hailo-8L | The gateway tier. **17 µs to score a frame; 781 331 hourly nodes on one core.** Compute is not the constraint, the AI HAT has no job, and the UPS is not the correctness requirement the plan assumed — the storage choice is |
| `hw/jetson-orin-nano` | Jetson Orin Nano Super | **A branch that argues against its own hardware.** Three candidate workloads, all negative by 14× to 13 500×. What survives is data sovereignty, and the on-site training pipeline that implements it caught two "checks that cannot fail" |

## The through-line

Four of the eight kits carry an accelerator or a premium part, and in every case
the measurement said the same thing:

> The feature extractor is ~200× the cost of the model, and the acquisition
> window is ~100× the cost of the extractor. On a battery node, **85 %→63 %** of
> the energy is the time the sensor spends awake.

So an Ethos-U55, a Hailo-8L and an Ampere GPU are all idle on this workload,
independently measured, and all three are justified by **models that do not
exist yet** — which is stated the same way in all three branches rather than
hidden behind a stub that returns a plausible number.

The other recurring result is about process, not silicon: the defects that
survived longest were **checks that could not fail** — a threshold calibrated
and verified on the same data, a noise comparison between a density and a
quantisation step, a figure parsing the wrong rows of its own test's output, a
gate on a rate where the count was 1.8. Each is recorded in the branch that
produced it.

## Known gaps, platform-wide

- **Nothing has been flashed.** No `arm-none-eabi-gcc`, ESP-IDF, nRF Connect
  SDK, Alif SDK or Renesas FSP in this environment. The RA8D1 branch adds a stub
  FSP so its driver at least compiles under `-Werror -Wconversion
  -Wdouble-promotion`; the others are review-quality.
- **No Docker**, so the compose stack has never started.
- **No model has passed validation on measured data.** CWRU, MFPT and MIMII are
  the gate; every detection number in this repository is a number about a
  simulator.
- **Timing and current are datasheet arithmetic**, except the gateway benchmark,
  which is measured on the wrong machine and says so.
- **CI workflows live in [`ci/github/`](ci/README.md)**, not `.github/workflows/`,
  and must be copied into place from a session with `workflow` scope.

See [`docs/07_production_readiness.md`](docs/07_production_readiness.md) for the
full list, including the things that are easy to believe and are not true.
