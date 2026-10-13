# Test strategy

```bash
make test                 # core/ Python suite
make -C cdsp test         # C library vs Python golden vectors
python3 cloud/ingest/app.py --replay    # end-to-end, no infrastructure
make -C slides            # the decks must build
```

## What each layer is responsible for proving

| Layer | Proves | Mechanism |
|---|---|---|
| `core/tests` | The maths is calibrated and the logic is right | Invariants, not snapshots |
| `cdsp/tests` | The MCU computes the same number as the cloud | Generated golden vectors |
| `--replay` | The chain works, including the rejection paths | Simulated run-to-failure |
| Per-kit `tests/` | The kit's own config and budget hold | Property tests + measured numbers |
| CI | All of the above, on 3.9 and 3.12 | `ci/github/core-ci.yml` |

## Invariants, not snapshots

The Python suite is deliberately heavy on properties that must hold rather than
values that happened to be produced. A snapshot test tells you something
changed; an invariant test tells you something is *wrong*.

The load-bearing ones:

**Parseval, for every scaling and every window.** `rms_in_band` over the full
band must equal the time-domain RMS, for `peak`, `rms`, `power` and `psd`, with
`rect`, `hann`, `flattop` and `blackman_harris`. This single test family caught
three separate calibration bugs:

| Bug | Error | Why nothing else caught it |
|---|---|---|
| Single-sided fold applied to magnitude then squared | √2 (3 dB) | A spectrum plot looks identical |
| Block averaging in the magnitude domain | π/4 (−1.05 dB) on broadband only | Tones were correct; only the noise floor was wrong |
| `rms_in_band` not dividing out ENBW | up to 1.9× (flat-top) | Consistent across the codebase, so every internal comparison agreed |

Each was invisible in an integration test and would have been wrong by a
constant factor forever, discovered the first time someone compared the
dashboard to a calibrated handheld meter.

**Window gain constants** against published values (Hann ENBW 1.5, flat-top
3.7702, …), and the scallop-loss ordering.

**Kinematic frequency relations**: BPFO = n·FTF exactly, and orders are
speed-independent.

**Healthy data must be quiet.** A 30-acquisition healthy holdout must produce a
health index below 1.0 and *zero* diagnoses. Measured: median 0.65, max 1.08,
0/30 false diagnoses. A detector is not interesting until it is silent on
healthy data.

**Escalation must happen before the knee.** A 90-day run-to-failure series must
escalate `watch → alarm → critical`, with `watch` well before the simulator's
knee at day 50. Measured: watch day 28.3, alarm day 56.6, critical day 70.4.

**Grouped splits must beat random ones.** `leakage_report` must pass with zero
violations, and the AUC under a unit-grouped split must be *lower* than under a
random one — if they match, the grouping is not working.

## The golden-vector cross-check

`cdsp/tools/gen_golden.py` evaluates `pdmcore` on fixed inputs and emits
`cdsp/tests/golden.h`; `cdsp/tests/test_cdsp.c` recomputes in C and compares.
1 049 checks. It runs natively on the build host, so it gates every commit with
no hardware attached.

Tolerances are per-check and justified in the test — the expected float32
accumulation error for that specific operation, not "close enough". A tolerance
that has to be loosened is a signal.

**This found a real bug that nothing else would have.** The FFT twiddle table
was allocated as `3n/4` floats where the algorithm needs `n+2`. The split-stage
table overflowed into itself and corrupted exactly bins `n/8+1 … 3n/8` of every
transform — bins outside that range were perfect. A tone sweep would have passed.
Parseval over the full band would have passed. A spectrum plot would have looked
plausible. 395 of 1 049 checks failed and pointed straight at it.

## Negative controls in the end-to-end POC

`--replay` ends with four rejection paths, because a POC that only demonstrates
the happy path has not demonstrated the thing that matters:

| Injected | Must | Verified |
|---|---|---|
| Single bit flip in the payload | reject on CRC | ✅ |
| `CLIPPED` flag set | refuse to score | ✅ |
| Replayed sequence number | deduplicate silently | ✅ |
| 115 days of silence | raise `STALE` | ✅ |

## What is not tested, and what that means

Stated plainly, because an untested claim presented as tested is worse than an
acknowledged gap:

- **No hardware is in the loop.** Firmware is structurally complete and the DSP
  is cross-checked, but nothing in this repository has been flashed to a board.
  Timing figures in the per-kit READMEs are derived from cycle counts and vendor
  benchmarks, and are labelled as estimates wherever they are not measured.
- **No cross-compilation in CI.** `arm-none-eabi-gcc` is not available in this
  environment, so the firmware is verified by the host build of the shared C
  library plus review, not by a target build. This is the single largest gap and
  the first thing to close: a per-kit cross-compile job is ~20 lines of CI.
- **The simulator is a signal model, not multibody dynamics.** It reproduces the
  signatures diagnosis relies on; it does not derive them from contact
  mechanics. Every detection number in this repository is a number *on simulated
  data*. Validation against CWRU/MFPT/MIMII and then against pilot data is a
  gate before any model touches a work order, recorded in
  `docs/03_model_cards.md`.
- **No Docker in this environment**, so the compose stack is unexercised. The
  ingest service itself is exercised through `--replay`, which runs the same
  `handle_raw` path the MQTT and HTTP transports call.
- **No load test.** The ~5 000 frames/second figure in
  `docs/00_architecture.md` is an estimate used as a Kafka trigger condition,
  not a measurement.

## CI

`ci/github/core-ci.yml` (in `ci/` rather than `.github/workflows/` because the
push token lacks the `workflow` scope — see `ci/README.md`):

- Python 3.9 and 3.12. 3.9 is the floor because gateway images ship old
  interpreters; 3.12 catches deprecations early.
- **Imports the core with only numpy and scipy installed.** The unsupervised
  detectors that ship on day one must work without sklearn or torch, and this is
  the only way to keep that true.
- Runs the full suite, then a CLI smoke test.
- Lints with ruff.

Not yet in CI and worth adding, in priority order:

1. `make -C cdsp test` — the golden-vector check only guards against drift if it
   runs automatically.
2. Per-kit cross-compile.
3. `make -C slides` — a deck that does not build is a broken deliverable.
