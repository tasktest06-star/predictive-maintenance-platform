# Data contract

Three interfaces have to stay stable across firmware releases, cloud deploys and
customer integrations. Each has a version, and each rejects what it does not
understand rather than guessing.

| Interface | Version | Defined in | Rejects on mismatch |
|---|---|---|---|
| Wire frame | `PAYLOAD_SCHEMA_VERSION = 3` | `core/pdmcore/io/payload.py` | yes — hard error |
| Feature vector | `FEATURE_SPEC_VERSION = 2` | `core/pdmcore/features.py` | yes — hard error |
| MQTT topics | `v1` in the path | this document | unknown topics ignored |

## Wire frame

44-byte header, `n_features` float32 values, 4-byte CRC-32. Little-endian.
300 bytes for the standard 63-feature frame.

```
offset size  field            notes
0      4     magic            b"PDM3"
4      1     schema_version   uint8
5      1     feature_spec     uint8
6      2     n_features       uint16
8      8     device_id        uint64   MCU unique ID, truncated
16     8     timestamp_ms     uint64   Unix ms, UTC
24     4     uptime_s         uint32
28     2     battery_mv       uint16
30     2     rssi_dbm         int16
32     4     shaft_hz         float32
36     4     temperature_c    float32
40     2     flags            uint16   see below
42     2     sequence         uint16   wraps; used for gap detection
44     4*N   features         float32[n_features]
44+4N  4     crc32            uint32   over bytes [0, 44+4N)
```

### Why these choices

**Binary, not JSON.** The same 63 features as JSON are ~1 574 bytes — 5.2× the
payload. On LTE-M that is 5.2× the airtime and therefore 5.2× the radio energy,
which is the dominant term in a battery sensor's budget.

**float32, not float64.** No condition indicator needs more than 7 significant
digits, and float64 would double the payload.

**CRC-32 over the whole frame, even under TLS.** The CRC catches corruption that
happens *before* transmission: a brown-out during the DMA copy out of the
accelerometer FIFO, or a flash-wear bit flip in the store-and-forward queue. TLS
protects the link, not the device.

**`sequence` wraps at 16 bits** and is used for two things: deduplicating
store-and-forward replays, and counting gaps. A sensor silently dropping 30 % of
its frames looks healthy on a trend chart; the gap count is what makes that
visible. 65 536 frames at one per hour is 7.5 years, so the wrap is handled
explicitly (`sequence_gaps`) rather than assumed away.

**`shaft_hz` in the frame, not looked up.** Order and bearing features are
meaningless without the speed they were computed against, and a VFD changes it
between acquisitions. Carrying it makes every frame self-describing.

### Flags

| Bit | Flag | Blocks scoring |
|---:|---|:---:|
| 0 | `CLIPPED` — ADC hit full scale | **yes** |
| 1 | `FIFO_OVERRUN` — samples lost | **yes** |
| 2 | `SELFTEST_FAIL` | **yes** |
| 3 | `LOW_BATTERY` | no |
| 4 | `FALLBACK_CONFIG` — degraded config after a failed update | no |
| 5 | `SPEED_ESTIMATED` — `shaft_hz` is tacholess, not measured | no |
| 6 | `TEMP_OUT_OF_RANGE` | no |
| 7 | `STORED_AND_FORWARDED` — replayed, not live | no |

The first three block scoring because only the device knows them, and scoring a
clipped acquisition produces a confident anomaly on an artefact. That is worse
than having no data: it teaches the operator to ignore alarms.

`SPEED_ESTIMATED` does not block scoring but is recorded, because a tacholess
speed estimate carries a few tenths of a percent of error that widens every
order band — and if a diagnosis is later disputed, whether the speed was
measured or inferred is the first question.

## MQTT topics

```
pdm/v1/<site>/<device_uid>/telemetry      device → cloud, binary frame, QoS 1
pdm/v1/<site>/<device_uid>/snapshot       device → cloud, raw waveform, QoS 1
pdm/v1/<site>/<device_uid>/status         device → cloud, JSON, retained
pdm/v1/<site>/<device_uid>/cmd            cloud → device, JSON, QoS 1
pdm/v1/<site>/<device_uid>/cfg            cloud → device, JSON, retained
```

**QoS 1, not 2.** At-least-once with application-level deduplication on
`(device_id, sequence)` costs one round trip; exactly-once costs four and still
needs the deduplication for the store-and-forward case. The dedup is required
regardless, so QoS 2 buys nothing.

**`cfg` retained.** A device waking from deep sleep must get its current
configuration without the cloud having to be listening for it. Retained
configuration is how a sensor that slept through a reconfiguration catches up.

**`status` retained.** So the dashboard knows a device exists and what state it
was last in, even if the device is currently asleep or dead. This is what makes
a `STALE` alarm possible at all.

### Downlink commands

```json
{"cmd": "snapshot", "id": "...", "duration_s": 1.0, "fs_hz": 25600}
{"cmd": "set_config", "id": "...", "demod_band_hz": [2500, 6000], "interval_s": 3600}
{"cmd": "set_model", "id": "...", "url": "...", "sha256": "...", "version": 7}
{"cmd": "selftest", "id": "..."}
{"cmd": "identify", "id": "...", "seconds": 30}
```

Every command carries an `id` and is acknowledged on `status` with that `id`.
Commands are **idempotent**: re-sending `set_config` with the same values is a
no-op, because a retained message is redelivered on every reconnect and a
non-idempotent command would then re-execute on every wake.

`identify` blinks the LED. It exists because finding one specific sensor among
forty identical ones bolted to a plant wall is otherwise a genuinely hard
problem, and that is a commissioning cost nobody budgets for.

### Configuration pushed down, not decided locally

| Parameter | Decided by | Why |
|---|---|---|
| `demod_band_hz` | cloud kurtogram | Needs the full signal and a band search; the device runs one fixed filter |
| `interval_s` | cloud policy | Balances battery life against detection delay per asset criticality |
| `fs_hz`, `n_fft` | commissioning | Determined by the bearing frequencies and resonance map of that asset |
| alarm thresholds | cloud | Derived from the asset's own commissioning distribution |
| model weights | cloud | Trained on the fleet |

The device decides **nothing** that benefits from fleet context. It executes, it
reports, and it can alarm autonomously on the configuration it has — which is
the property that makes it useful when the network is down.

## Snapshot format

Raw waveforms go as a separate message, not inline:

```
offset size  field
0      4     magic b"PDMW"
4      1     schema_version (3)
5      1     channels
6      2     reserved
8      8     device_id
16     8     timestamp_ms
24     4     fs_hz (float32)
28     4     n_samples_per_channel (uint32)
32     4     scale_g_per_lsb (float32)
36     4     flags
40     2*N*C int16 samples, channel-interleaved
...    4     crc32
```

**int16 with an explicit scale factor, not float32.** The ADC is 16- or 20-bit;
storing float32 doubles the payload to carry zero extra information. The scale
factor makes the samples self-describing, so a snapshot archived for three years
is still interpretable without a device database lookup.

## Compatibility rules

1. **Never reuse a field offset.** Adding a field means a new schema version.
2. **Never reorder features.** Adding one means bumping the feature spec version.
3. **Ingest rejects unknown versions.** It never guesses a layout. An un-upgraded
   fleet goes dark rather than silently wrong, which is the correct failure
   direction — and still an outage, so the rollout has to be planned.
4. **Devices tolerate unknown `cfg` keys** and ignore them, so a cloud deploy
   that adds a parameter does not brick an older fleet.
5. **One deprecated version supported at a time.** Ingest may accept v2 and v3
   during a migration, with v2 frames counted separately so the migration's
   progress is observable.
