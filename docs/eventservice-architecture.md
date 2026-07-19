# EventService Architecture

## Overview

EventService provides pub/sub event distribution for hyprstream using moq-lite
as the transport. Services publish lifecycle events that other services
subscribe to, replacing polling-based status checks. Since the EventService
consolidation epic (#600), the bus supports controller-managed confidential
group epochs: publishers encrypt events with sender/track keys derived from a
fresh epoch secret and attest them with hybrid Ed25519 + ML-DSA-65 signatures.
Accepted members receive separate HyKEM/COSE grants, while a forwarding relay
sees only opaque ciphertext.

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

## Controller-Managed Hybrid Encryption (#555)

**Sources:** `crates/hyprstream-rpc/src/events.rs` (publisher/subscriber and
wire codec), `crypto/event_crypto.rs` (epoch-derived object encryption), and
`crypto/group_key.rs` (controller epoch state machine and grants).

Public publishers remain wire-compatible. Confidential `ZeroKnowledge` and
`LimitedKnowledge` publishers fail closed until a controller installs a
committed membership-version/epoch secret. The controller uses an explicit
prepare/commit transaction: every join, leave, revocation, expiry, recipient
rotation, accepted-state advance, or controller change prepares a fresh random
epoch and keyset; commit atomically swaps membership and epoch coordinates.
Abort leaves the committed state untouched.

Each resulting member receives a separate COSE_Encrypt0 grant sealed to its
HyKEM recipient (`X25519 + ML-KEM-768`). Grant AAD binds the group/keyset,
controller and accepted state, subject and capability, recipient key ID,
subscriber-generated blinded routing presentation, publisher Ed25519 and
ML-DSA-65 anchors, retention/opaque-routing policies, membership version,
epoch, and expiry. Signing keys and blinded Ristretto presentations are never
used as KEX inputs. A stock MoQ relay receives neither grants nor epoch secrets.

For each event object, the publisher derives a distinct sender/track AEAD key
and nonce domain from the epoch secret. AES-256-GCM AAD binds track, publisher
key ID, membership version, epoch, and sequence; the nonce is deterministically
derived from that domain plus epoch/sequence. The plaintext signature transcript
also binds topic, timestamp, membership version, epoch, and sequence and requires
a hybrid Ed25519 + ML-DSA-65 composite signature anchored by the controller
grant. Subscribers reject replay, nonce reuse, unknown/future epochs, retired
objects, and events outside a bounded prior-epoch last-issued cutoff.

One opaque ciphertext is published once and forwarded/cacheable byte-identically
by a stock relay. Membership transitions distribute O(M) per-member grants but
do not require per-subscriber event ciphertexts. Revoked members receive no next
grant and cannot decrypt the fresh epoch.

The confidential body is versioned and length-prefixed:

```
[1B version][8B membership_version][8B epoch][8B sequence]
[12B nonce][16B key_commitment]
[4B tag_len][tag][4B ciphertext_len][ciphertext][4B lk_tag_len][lk_tag]
[8B timestamp BE][32B publisher_pubkey][4B signature_len][hybrid signature]
```

The topic remains the MoQ frame topic and is authenticated by the hybrid
signature. `LimitedKnowledge` additionally carries a per-prefix keyed routing
tag; the grant authenticates the selected opaque-routing policy.

Anonymous-member acceptance is intentionally not implemented here; it remains
blocked on #1058 and #1060–#1062.

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
