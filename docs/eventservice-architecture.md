# EventService Architecture

## Overview

EventService provides pub/sub event distribution for hyprstream using moq-lite
as the transport. Services publish lifecycle events that other services
subscribe to, replacing polling-based status checks. Since the EventService
consolidation epic (#600), the bus is **group-keyed**: publishers can encrypt
events under a per-prefix group key (AES-256-GCM) and sign them (Ed25519),
with O(M) key rotation across M subscribers.

The event bus replaced the legacy ZMQ XPUB/XSUB proxy (`ProxyService`) in epic
#131/#167; the publisher/subscriber API survived, only the transport changed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Single Host                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────── moq-lite Origin ────────────────────────────┐  │
│  │                        (MoqEventOrigin, process-global)               │  │
│  │                                                                        │  │
│  │  Publishers                                        Subscribers         │  │
│  │  ┌────────────────┐                               ┌────────────────┐  │  │
│  │  │WorkerService   │                               │WorkflowService │  │  │
│  │  │RegistryService │──publish──► [Origin] ──sub──►│CLI (wait)      │  │  │
│  │  │ModelService    │                               │ThresholdMonitor│  │  │
│  │  │InferenceService│                               └────────────────┘  │  │
│  │  └────────────────┘                                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Cross-process UDS socket: hyprstream_rpc::paths::event_socket()            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The `event` Service

The event bus is itself a registered service:
`#[service_factory("event")]` in `crates/hyprstream/src/services/factories.rs`.
The factory:

1. Creates a `MoqEventOrigin` and registers it as the process global
   (`init_global_moq_event_origin`)
2. Serves it over the well-known cross-process UDS path
   (`hyprstream_rpc::paths::event_socket()`,
   `serve_event_moq_uds_background`) so other service processes can
   publish/subscribe to the shared bus in multi-process deployments
3. Returns `MoqEventBarrierService` — a minimal `Spawnable` (defined in
   `factories.rs`) that just holds the shutdown barrier so the orchestrator
   tracks lifecycle; the bus itself needs no proxy threads

Other services declare it as a dependency, e.g. WorkerService:
`#[service_factory("worker", depends_on = ["policy", "discovery", "event"])]`.

In the same-process (inproc) deployment every service shares the global origin
directly; the UDS plane is the bridge for the systemd / `--ipc` deployment
where each service is its own process (non-event processes create a local
origin and link it to the shared bus — see `ensure_event_client_origin`).

## Topic Format

```
{service}.{entity}.{event}
```

**Examples:**

| Topic | Description |
|-------|-------------|
| `worker.sandbox123.started` | WorkerService: sandbox started |
| `worker.container456.stopped` | WorkerService: container stopped |
| `registry.repo789.cloned` | RegistryService: repository cloned |
| `model.qwen3.loaded` | ModelService: model loaded |
| `inference.session123.completed` | InferenceService: generation completed |

**Service Ownership:**
- **WorkerService** → `worker.*` events (sandbox, container lifecycle)
- **RegistryService** → `registry.*` events (git operations)
- **ModelService** → `model.*` events (model loading/unloading)
- **InferenceService** → `inference.*` events (generation lifecycle)

**Constraint:** Entity and event names cannot contain dots (used as separator).

## Delivery Semantics & QoS

Delivery/QoS rides the same `StreamOpt` contract as the streaming plane
(`streaming.capnp` / `hyprstream_rpc::stream_info`, #606) rather than a
parallel event-specific QoS type. `MoqEventSubscriber`/`EventSubscriber`
select it via `with_qos(...)`:

| Preset | Semantics |
|--------|-----------|
| `EventLive` (default) | At-most-once, drop-oldest — best-effort lifecycle signals |
| `EventReliable` | At-least-once + retained — events that must not be silently dropped (e.g. `model.loaded`) |

Only `delivery` and `overflow_policy` are enforced client-side today;
`ordering`/`completion` are reserved for a future chained-integrity layer.

| Aspect | Behavior |
|--------|----------|
| Ordering | In-order per publisher; no cross-publisher guarantees |
| Delivery | `StreamOpt`-selected (at-most-once default; at-least-once via `EventReliable`) |
| Persistence | moq's per-track cache retains recent groups (evicted after `MAX_GROUP_AGE`, 5s) |
| Late join | Not purely "live-only" — see below |
| Slow subscriber | `StreamOpt::overflow_policy` (drop-oldest ring by default; `block` for lossless backpressure) |
| Prefix filtering | Dot-separated prefix match (`"worker."` → all worker events) |

### Late Join, Latched State, and Backfill

Three mechanisms serve subscribers that arrive after an event was published:

1. **moq per-track cache** — moq-lite retains recent groups per track (up to
   `MAX_GROUP_AGE`), so a subscriber joining moments late still sees them.
2. **Latched terminal state (EV7, `hyprstream-rpc/src/latch.rs`)** — a
   resource's terminal value (task exit, model-load result, fd close) is
   retained host-side in a `TerminalStore` and served to late watchers
   immediately; `read_then_subscribe` serves the retained value if present,
   else subscribes to the live edge and awaits the terminal event. This
   subsumes `load --wait` and the 9P `/task/<id>/exit` file pattern
   ("file holds the latch").
3. **Firehose backfill (#393)** — on first subscription to a per-OID track,
   `BackfillMode::FirehoseBackfill` replays history from the atproto firehose
   / registry before going live; on any backfill error it degrades gracefully
   to live-only.

### Hybrid State Pattern

For reliable status checking without a latch (e.g. CLI waiting for a
container): query current state via RPC first, subscribe to events for
updates, and handle the race by checking the timestamp in the query response.

## Group-Keyed Encryption (epic #600)

**Sources:** `crates/hyprstream-rpc/src/events.rs` (publisher/subscriber +
`EncryptedEvent` wire codec), `crates/hyprstream-rpc/src/crypto/event_crypto.rs`
(AES-GCM, key wrapping), `crates/hyprstream-rpc/src/crypto/group_key.rs`
(the reusable keyable-group primitive: `GroupKeyRegistry`, `RekeyPolicy`,
`EncryptedEvent`, `WrappedKeyEntry`, rotation constants).

Each `EventPublisher` selects a privacy mode (`EventPrivacy`,
`crypto/event_crypto.rs`):

| Mode | Behavior |
|------|----------|
| `Public` (default for `EventPublisher::new`/`new_with_oid`/`new_oid_only`) | No encryption — payload written to the moq track unmodified, wire-identical to the pre-#600 plaintext bus |
| `ZeroKnowledge` | Group-key encrypted; all events on one broadcast stream — the relay learns nothing about interests |
| `LimitedKnowledge` | Group-key encrypted with a per-prefix keyed routing tag (`lk_tag`) — efficient per-prefix circuits, but the stable tag reveals topology/linkability |

Flipping production publishers to an encrypted default is deferred until
group-membership records (#602) and authenticated join (#604) land.

### Encryption & signing

Encrypted publishers (`EventPublisher::new_encrypted(...)`) bind prefixes via
`register_prefix`, which generates a random 32-byte group key. Each publish:

- Encrypts the payload with **AES-256-GCM** under the current group key
  (fresh OsRng nonce, topic-bound AAD)
- Includes a 16-byte **key commitment** so subscribers can select the right
  key without trial decryption
- **Ed25519-signs** the event (`build_event_sig_message` over
  topic ‖ payload ‖ timestamp); `EventSubscriber::recv` verifies the
  signature before returning the payload. The embedded pubkey is
  self-asserted — tamper-evidence, not identity; publisher↔DID binding is
  EV4 (#604), gated on #446

One ciphertext is written once per publish to a shared moq broadcast track and
fans out natively to every subscriber of that track (group-level
confidentiality: O(1) publish, not O(N) per-subscriber encryption).

The `EncryptedEvent` body is a versioned, length-prefixed binary layout
(internal-only, no capnp schema):

```
[1B version][12B nonce][16B key_commitment]
[4B tag_len][tag][4B ciphertext_len][ciphertext][4B lk_tag_len][lk_tag]
[8B timestamp BE][32B publisher_pubkey][4B signature_len][signature]
```

The topic itself still travels as the moq frame's topic field, unencrypted —
keyed/opaque topic routing is EV3 (#603).

### Key distribution (wrapping)

Subscribers join a prefix (`EventSubscriber::join_prefix`) with an ephemeral
Ristretto255 DH keypair. The publisher wraps the group key per subscriber
(`wrap_for_subscriber` → `derive_wrap_key`/`wrap_group_key`): Ristretto255
ECDH with the subscriber's pubkey, then AES-256-GCM-wraps the group key under
the derived wrap key with length-prefixed AAD binding the subscriber hash and
prefix. Subscribers unwrap with `unwrap_group_key`.

### Rotation (`RekeyPolicy`, O(M) re-wrap)

`EventPublisher::rotate_key(prefix, effective_delay)` generates a new group
key + ephemeral keypair and re-wraps for every known subscriber — **O(M) DH +
wrap operations per rotation** (each `WrappedKeyEntry` carries a random
routing tag, unlinkable across rekeys). The new key is held as a pending
rekey and atomically promoted at `effective_at`; subscribers receive a
`RekeyEvent` and keep the previous key in a `KeyRing` for a grace window
(publisher `GRACE_PERIOD` 120s; subscriber default 30s).

`RekeyPolicy` (`crypto/group_key.rs`) is the latency-vs-cost tradeoff:

| Policy | Behavior |
|--------|----------|
| `Scheduled { interval }` (default: 1h, max lifetime 24h) | Bounded O(M) per interval; revocations deferred to the next rotation |
| `Immediate` | Rotate on every revocation — prompt forward secrecy, but O(M) per revocation (O(M²) over M departures) |
| `Jittered { interval, jitter }` | Scheduled with jitter for timing-attack resistance |

`GroupKeyRegistry` (`crypto/group_key.rs`) is the generic keyable-group
primitive behind this (group registration, membership-resolver-gated `join`,
`begin_rekey`/`maybe_promote_pending`), reusable outside the event bus.

## Components

### EventPublisher / EventSubscriber

The canonical broadcast types (EV1, epic #600). They live in
`crates/hyprstream-rpc/src/events.rs` — alongside the moq transport
(`moq_event.rs`) and crypto (`crypto/event_crypto.rs`) they wire together —
and are re-exported from `crates/hyprstream-workers/src/events/mod.rs` for
back-compat with existing callers.

```rust
use hyprstream_workers::events::{EventPublisher, EventSubscriber};

// Publisher: no origin argument — uses the process-global MoqEventOrigin.
let publisher = EventPublisher::new("worker")?;   // EventPrivacy::Public
publisher.publish("sandbox123", "started", &payload).await?;

// Subscriber
let mut subscriber = EventSubscriber::new()?;
subscriber.subscribe("worker.")?;   // all worker events, prefix match
while let Ok((topic, payload)) = subscriber.recv().await {
    println!("Received: {} ({} bytes)", topic, payload.len());
}
```

**Publisher constructors:** `new(source)` / `new_with_oid(source, oid)` /
`new_oid_only(source, oid)` (plaintext) and `new_encrypted(...)` +
`register_prefix(prefix)` (group-keyed).

**Subscriber methods:** `subscribe(prefix)` / `subscribe_all()` /
`subscribe_oid(oid)` (#393 per-OID track), `with_qos(StreamOpt)`,
`with_backfill(BackfillMode)`, `join_prefix(...)` (encrypted prefixes),
`recv()` / `recv_timeout(duration)`, `take_rekey_receiver()`.

Prefixes not joined via `join_prefix` are treated as `Public`: `recv` returns
their frames unmodified. Joined prefixes are decoded, signature-checked, and
decrypted automatically.

### MoqEventOrigin

**Location:** `crates/hyprstream-rpc/src/moq_event.rs`

The process-global broadcast origin, rooted at `local/events`. Each source
registers a broadcast under `local/events/{source}` with a single `events`
track; per-OID publication events additionally get their own selective track
`local/events/publications/{oid_hash}` (#393) so a node tracking N of M OIDs
reads N tracks, not M.

This is distinct from `MoqStreamOrigin` (`moq_stream.rs`): broadcast fan-out
belongs on the event origin; point-to-point DH-keyed token streaming belongs
on the stream origin.

### Event Types

**Location:** `crates/hyprstream-workers/src/events/types.rs`

Worker events with Cap'n Proto serialization:

```rust
pub enum WorkerEvent {
    SandboxStarted(SandboxStarted),
    SandboxStopped(SandboxStopped),
    ContainerStarted(ContainerStarted),
    ContainerStopped(ContainerStopped),
}
```

## Message Format

Each source publishes to its `local/events/{source}` broadcast's `events`
track. Every frame packs the full dot-separated topic with the payload:

```
[4 bytes topic_len BE][topic UTF-8][payload bytes]
```

For `Public` events the payload is the raw application bytes (typically a
Cap'n Proto `EventEnvelope`); for encrypted events it is the
`EncryptedEvent` body described above. Prefix filtering is applied
subscriber-side by `MoqEventSubscriber` using dot-separated prefix matching.

## Schema Ownership

To avoid circular dependencies, event schemas are distributed:

| Crate | Schema | Contents |
|-------|--------|----------|
| `hyprstream-rpc` | `schema/events.capnp` | `EventEnvelope`, `EventEnvelopeV2`, rekey announcements (`RekeyAnnouncement`, `WrappedKey`), prefix announcements |
| `hyprstream-workers` | `schema/worker.capnp` | `SandboxStarted`/`SandboxStopped`/`ContainerStarted`/`ContainerStopped` (the `WorkerEvent` payloads) |
| `hyprstream-rpc-std` | `schema/service_events.capnp` | `TypedEventEnvelope` + typed service event payloads (generation, metrics, …) |

Consumers deserialize `payload` based on topic prefix (e.g. `worker.*` →
`WorkerEvent` from the workers schema).

## Separation from MetricsService

EventService and MetricsService are **separate concerns**:

| Service | Pattern | Purpose |
|---------|---------|---------|
| EventService | publish/subscribe (moq-lite) | Lifecycle events (broadcast) |
| MetricsService | request/reply RPC | Data queries/inserts |

MetricsService queries are NOT broadcast to EventService. Future work may add
optional CDC for threshold breach events.

## Implemented Features

- **`event` service** — registered factory; moq-lite origin as process global,
  no proxy threads
- **Cross-process UDS plane** — external processes connect via
  `hyprstream_rpc::paths::event_socket()`
- **Group-keyed encryption** — AES-256-GCM + Ed25519 signing, per-prefix
  group keys, O(M) rotation (epic #600 EV1)
- **Latched terminal state** — retained terminal values for late watchers
  (EV7; `latch.rs`)
- **StreamOpt QoS** — `EventLive`/`EventReliable` presets (#606)
- **Per-OID tracks + firehose backfill** — wire-level selectivity and
  cold-start history (#393)
- **Prefix filtering** — dot-separated topic prefixes

## Future Work

- **Encrypted-by-default publishers** (#555) — gated on group-membership
  records (#602) and authenticated join (#604)
- **Keyed/opaque topic routing** (EV3, #603)
- **Publisher↔DID identity binding** (EV4, #604)
- **Multi-host routing** — subscription-driven event propagation via iroh relay
