# Security

Threat model first, because the interesting risks here are not the obvious ones.

## What an attacker gets, and what they want

A sensor is bolted to a wall in a plant. It is **physically accessible** to
anyone who gets into the building, and extracting one is not a sophisticated
attack. Assume device compromise and design so that one compromised device
cannot do much.

| Asset | Attacker goal | Impact |
|---|---|---|
| Telemetry integrity | Inject a **healthy** reading for a failing machine | **Highest.** Suppresses a real alarm. Can injure someone. |
| Telemetry integrity | Inject an **alarm** for a healthy machine | Unnecessary shutdown; erodes trust in the programme |
| Fleet credentials | Extract one device's key, impersonate the fleet | Mass injection, both directions |
| Downlink | Push hostile configuration or firmware | Brick the fleet, or silently disable monitoring |
| Telemetry confidentiality | Infer production rates from vibration | Commercially sensitive, rarely safety-relevant |
| CMMS | Alter or close work orders | Hides a known fault; audit problem |

**The inverted-priority insight:** a fabricated *healthy* reading is more
dangerous than a fabricated alarm, and it is equally easy to inject. Most IIoT
security writing optimises against the alarm case. The mitigations below are
chosen for the suppression case: per-device identity (so a silent device cannot
be covered for by a forged one), staleness as a first-class alarm (so silence is
itself an event), and sequence-gap accounting (so partial suppression is
visible).

## Mitigations, and where each is implemented

| Control | Status | Where |
|---|---|---|
| Per-device X.509 client certificate | **design** | `mosquitto.conf` TLS listener, commented until certs exist |
| MQTT over TLS 1.2+ off-premises | **design** | `mosquitto.conf` 8883 listener |
| CRC-32 per frame | **implemented** | `pdm_payload.c`, `payload.py` |
| Schema and feature-version rejection | **implemented** | `decode_frame` |
| Device-flag gating (clipped/overrun not scored) | **implemented** | `IngestService.handle_frame` |
| Sequence deduplication | **implemented** | `AssetState.is_duplicate` |
| Gap accounting | **implemented** | `sequence_gaps`, `pdm_frames_missed_total` |
| Staleness alarm | **implemented** | `AlarmStateMachine.check_stale` |
| Signed firmware + rollback | **design** | per-kit secure-boot sections |
| Non-root container, minimal image | **implemented** | `cloud/ingest/Dockerfile` |
| Secrets from env, not baked | **implemented** | `docker-compose.yml`, `.env.example` |

"Design" means specified and not built. It is listed as such rather than implied
to be done.

## Device identity

**Per-device certificates, not a shared fleet password.** A pre-shared secret
across a fleet means one extracted sensor compromises every sensor, in both
directions. The provisioning model:

1. At manufacture, the MCU generates a keypair in hardware (the secure element
   or the MCU's own crypto peripheral) and the private key never leaves it.
2. A CSR goes out over a provisioning channel; the platform CA issues a cert
   bound to the MCU unique ID.
3. The broker is configured `require_certificate true` and
   `use_identity_as_username true`, so the MQTT identity *is* the certificate
   subject and cannot be spoofed by a device that holds a different key.
4. Topic ACLs restrict each device to `pdm/v1/<site>/<its own uid>/#`. A
   compromised sensor can lie about itself and nothing else.

Revocation is by CRL distributed to the broker. On a fleet of thousands,
revocation lists grow; the mitigation is short-lived certificates (90 days) with
automatic renewal over the existing authenticated channel.

## Firmware updates

- **Signed images, verified before activation.** Ed25519 over the image hash;
  the public key is in a write-protected flash region or the secure element.
- **A/B partitions with rollback.** A new image boots once in trial mode; if it
  does not confirm, the bootloader reverts. A sensor that cannot be recovered
  without a ladder is the whole reason this is non-negotiable.
- **Fallback configuration.** A device that fails to apply a configuration
  update runs the last known-good one and raises `FALLBACK_CONFIG`. It keeps
  monitoring in a degraded mode rather than stopping — a sensor that stops
  monitoring because a config push failed has turned a software bug into an
  unmonitored machine.
- **Staged rollout.** Canary, then 5 %, then the fleet, gated on the ingest
  metrics (`pdm_frames_rejected_*`, `pdm_frames_missed_total`). Those metrics
  exist partly for this.

## Multi-tenancy

The platform's strongest available claim is:

> Customer data never participates in another customer's training round.

This is defensible, auditable, and it is why
`research/deep_dive/02_gap_analysis_vs_plans.md` recommends **source-free domain
adaptation** across tenants rather than federated learning. Source-free
adaptation moves only weights; no shared training round, no aggregation server,
no contractual question about whose data trained what. Federated learning stays
inside a single tenant, where the question does not arise.

Isolation is per-tenant: database schema, object-storage prefix, broker ACL
namespace, and separate model registries.

## What is deliberately out of scope for now

- **FedRAMP.** `plans/05` targets it. Nothing here is FedRAMP-ready; the
  realistic path is the multi-tenancy claim above plus a managed control plane,
  and it is a Phase 3 item with a long lead time.
- **ATEX/IECEx.** A hardware certification, not a software one; tracked per kit.
- **Hardware anti-tamper.** A sensor in a plant is physically accessible and the
  cost of making it tamper-evident exceeds the value of what it protects. The
  response is to assume compromise and limit blast radius, which is what the
  per-device identity model does.
- **Encrypting telemetry at rest beyond disk encryption.** Vibration features
  are commercially sensitive but not personal data; the controls are
  proportionate to that.
