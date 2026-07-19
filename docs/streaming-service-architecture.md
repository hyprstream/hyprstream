# Streaming Service Architecture

moq-lite streaming plane with HMAC-chained end-to-end authentication and
transport-level AEAD.

> Migration status (#554): the identified hybrid epoch profile is implemented in
> `stream_epoch.rs` and the authenticated request now carries `clientKemPublic`.
> Individual service producers shown below still use the legacy Ristretto path
> until they can consume accepted-current admission/#726 key-release evidence.
> Do not treat this document's legacy flow as the network security target or as
> #554 closure evidence.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMING PLANE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  InferenceService              MoqStreamOrigin             Client            │
│       │                             │                        │              │
│       │─ infer_stream (RPC) ────────────────────────────────►│              │
│       │◄─ StreamInfo {streamId, dhPublic, qos,               │              │
│       │              broadcastPath, announcedAt} ────────────┤              │
│       │                             │                        │              │
│       │  derive DH keys             │                        │              │
│       │  topic = DH-derived hex     │                        │              │
│       │                             │                        │              │
│       │─ publish(broadcast_path) ──►│                        │              │
│       │   [StreamBlocks, HMAC]      │                        │              │
│       │                             │◄─ subscribe ───────────┤              │
│       │                             │   (UDS same-host, or   │              │
│       │                             │    dial announcedAt)   │              │
│       │                             │                        │              │
│       │                             │─ block ───────────────►│              │
│       │                             │─ block ───────────────►│[verify HMAC] │
│       │                             │─ complete ────────────►│              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The streaming plane is moq-lite: InferenceService publishes to the
process-global `MoqStreamOrigin` (`crates/hyprstream-rpc/src/moq_stream.rs`);
clients subscribe — via UDS on the same host, or by dialing a network reach —
and receive blocks through `MoqStreamHandle`. (The former ZMQ PUSH/PULL →
XPUB/XSUB `StreamService` was removed in epic #131/#138.)

Two distinct MoQ origins exist per process — do not conflate them:

| Origin | Purpose |
|--------|---------|
| `MoqStreamOrigin` (`moq_stream.rs`) | Point-to-point token/data streaming: one producer, one consumer, DH-derived keys |
| `MoqEventOrigin` (`moq_event.rs`) | Broadcast fan-out event bus — see [EventService Architecture](./eventservice-architecture.md) |

## Security Model (E2E Authentication)

The moq origin is a **blind router** — it does NOT verify HMACs.

| Layer | Responsibility |
|-------|----------------|
| **InferenceService** | Derives DH keys, produces HMAC chain, signs RPC responses |
| **moq-lite Origin** | Routes by broadcast path, buffers for late subscribers |
| **Client** | Derives same DH keys via ephemeral pubkey, verifies HMAC chain |

### Signing Keys

The node signing key resolves via `HyprConfig::resolve_secrets_dir()` to
`<secrets_dir>/signing-key` (generated on first startup if absent; see
`load_or_generate_signing_key` in `crates/hyprstream/src/cli/policy_handlers.rs`).
A `HYPRSTREAM__SIGNING_KEY` / `config.signing_key` test bypass takes precedence.

Services do **not** share one key. `ServiceContext::service_signing_key(name)`
(`crates/hyprstream-service/src/service/factory.rs`) resolves a per-service key:

1. Independent per-service keypair from the registry (generated in inproc
   mode; loaded from the service's own credentials in multi-process/IPC mode)
2. PolicyService special case — it IS the CA and uses the root key
3. `"multi"` fallback — multi-service IPC mode shares one key across the
   co-located services

Each service registers its verifying key with the trust store
(`hyprstream-service::service::trust_store`) via a CA-signed JWT; verifiers
resolve a signer pubkey back to a service identity through that store.

### Security Properties

| Property | Implementation |
|----------|----------------|
| **Topic unpredictability** | DH-derived (InferenceService ↔ Client ephemeral Ristretto255 keys) |
| **Response auth** | ResponseEnvelope with hybrid COSE composite signature (EdDSA + ML-DSA-65, mandatory) — authenticates `dhPublic` and the QoS contract at PQ strength (#275) |
| **Data integrity** | Chained HMAC verified end-to-end by client |
| **Data confidentiality** | Transport AEAD: payloads sealed under the DH-derived `enc_key` (AES-256-GCM, #321) |
| **QoS integrity** | `StreamOpt` carried inside the signed StreamInfo; clients MUST enforce it |
| **Stream binding** | Claims scope: `publish:stream:{topic}` in StreamInfo |
| **Replay protection** | Nonce cache on SignedEnvelope (RPC layer) |

### Request/Response Signing

All RPC communication uses signed envelopes:

```
Request Flow:
┌────────────────────────────────────────────────────────────┐
│  Client                                          Service   │
│    │                                                │      │
│    │─── SignedEnvelope(RequestEnvelope) ───────────►│      │
│    │       [signed with client's key]               │      │
│    │                                     [verify]   │      │
│    │                                     [process]  │      │
│    │◄── ResponseEnvelope ──────────────────────────│      │
│    │       [COSE composite, service's key]          │      │
│    │  [verify]                                      │      │
└────────────────────────────────────────────────────────────┘
```

## Stream Lifecycle

### 1. Client Initiates Inference Stream

Client calls `infer_stream` via the generated RPC client. The request carries an
ephemeral Ristretto255 public key for DH key exchange:

```
RequestEnvelope {
    payload: InferStreamRequest { model, prompt, ... },
    ephemeral_pubkey: [32 bytes],  // Client's ephemeral DH public key
}
```

### 2. InferenceService Responds with StreamInfo

InferenceService derives the shared stream keys and returns `StreamInfo`
(schema: `crates/hyprstream-rpc/schema/streaming.capnp`):

```capnp
struct StreamInfo {
  streamId      @0 :Text;    # Unique stream identifier ("stream-{uuid}")
  dhPublic      @1 :Data;    # Server's ephemeral Ristretto255 public key (32 bytes)
  qos           @2 :StreamOpt;           # Service-declared QoS contract
  broadcastPath @3 :Text;    # e.g. "local/streams/{topic_hex}"
  announcedAt   @4 :List(Destination);   # Network reaches for the moq plane (#274)
}
```

There is no UDS path or endpoint on the wire. Same-host clients resolve the
moq socket locally via `global_moq_uds_path()`; remote clients dial one of the
`announcedAt` reaches. The enclosing ResponseEnvelope's hybrid COSE composite
signature authenticates `dhPublic` and the QoS contract.

Key derivation (`derive_stream_keys`, `crypto/key_exchange.rs`) produces from
the DH shared secret: `topic` (64 hex chars), `mac_key` (HMAC chain),
`enc_key` (transport AEAD, #321), plus a control-channel topic and MAC key.

### QoS: StreamOpt (client-enforced contract)

`StreamOpt` (#213) is the service-declared delivery/integrity contract, carried
inside the signed StreamInfo. Five axes:

| Axis | Options (fail-closed `@0` default first) |
|------|------------------------------------------|
| `ordering` | `ordered` (gap = fatal) / `unordered { antiReplayWindow }` |
| `delivery` | `atMostOnce` / `atLeastOnce { dedupWindow, resumable }` |
| `completion` | `endOfStream` (terminal frame required before EOF — EOF without one is a truncation attack) / `none` |
| `retention` | `live` / `blocks(N)` / `seconds(N)` — relay late-join buffer |
| `overflowPolicy` | `block` (lossless backpressure) / `dropOldest { highWaterMark }` |

The service asserts the contract it will enforce; **clients MUST enforce the
same options on ingress and MUST disconnect rather than silently downgrade**
if they cannot honour a received option (see `StreamVerifier::with_policy`).
Named Rust presets (`Job`, `Log`, `Pipe`) live in `hyprstream_rpc::stream_info`.

### Network reaches: Destination (#274)

Each `Destination` is one way to reach the moq plane for this stream:

```capnp
struct Destination {
  role      @0 :Role;             # direct (the producer itself) | relay
  transport @1 :TransportConfig;  # union: quic (QuicReach) | iroh (IrohReach)
}
```

Only network-routable reaches are encoded; same-host endpoints
(`inproc`/`ipc`/UDS) are never carried on the wire — a co-located caller
resolves them from local config. An empty reach list means "co-located fast
path only". The node's own reach parameters come from one process-global
source (`NodeStreamReach`, set when the QUIC server binds), so every
StreamInfo producer publishes the same reach list.

### 3. Client Connects via MoqStreamHandle

Generated clients call `MoqStreamHandle::networked` (`moq_stream.rs`):

```rust
let handle = MoqStreamHandle::networked(
    stream_info.announced_at,   // reaches — source of truth for where the stream lives
    &stream_info.qos,           // signed StreamOpt: selects direct-vs-relay topology (#358)
    stream_info.broadcast_path,
    mac_key,                    // derived from the same DH exchange
    enc_key,
    topic,
);

while let Some(payload) = handle.recv_next().await? {
    match payload {
        StreamPayload::Data(bytes) => process(bytes),
        StreamPayload::Complete(meta) => break,
        StreamPayload::Error(e) => return Err(e.into()),
        _ => {}
    }
}
```

The constructor is synchronous; it spawns the background receive task and
resolves the transport internally:

- **Networked reach preferred**: the producer's reach is the source of truth.
  QoS selects topology (`select_reach`, #358): relay-first for
  retained/resumable streams, direct-first for live pipes — a stable reorder
  of the service-advertised reaches only (the client never invents a reach).
- **Local UDS fallback**: only when StreamInfo carries no dialable reach, the
  handle falls back to this process's moq plane via `global_moq_uds_path()`
  (UDS-only / test deployments). The local plane is never preferred over a
  reach — it only carries the producer's broadcast if the producer is
  co-located in the same process (#275).
- Neither available → the handle surfaces a clear error.

`MoqStreamHandle::new(uds_path, broadcast_path, mac_key, enc_key, topic)` is
the direct same-host constructor. Both implement `futures::Stream` and handle
per-block HMAC-chain verification, AEAD decryption, and cancellation via
`cancel()` / `cancel_token()`.

### 4. InferenceService Publishes Blocks

The RPC response (with StreamInfo) is sent immediately; token generation runs
in the background. The producer side is `StreamChannel::publisher(&ctx)`
(`streaming.rs`), which appends into the global `MoqStreamOrigin`, batching
payloads into HMAC-chained `StreamBlock`s and sealing Data/Complete payloads
under `enc_key`.

## Message Types

**Schema:** `crates/hyprstream-rpc/schema/streaming.capnp`

### StreamBlock (the wire unit)

Each moq track object is a serialized `StreamBlock` with a 16-byte truncated
HMAC appended:

```capnp
struct StreamBlock {
    prevMac        @0 :Data;                 # topic[..16] for block 0, mac_{n-1} after
    payloads       @1 :List(StreamPayload);  # Batched payloads
    sequenceNumber @2 :UInt64;   # Producer-assigned monotonic counter (= moq Group id);
                                 # resume/dedup offset and anti-replay/ordering anchor (#219)
    epoch          @3 :UInt64;   # Key-epoch; bumps on re-key/producer restart (#223)
    provenance     @4 :Data;     # Optional per-host hybrid COSE signature (#321):
                                 # proves WHICH mesh host produced the block
}
```

**MAC chain:**
```
mac_0 = HMAC(mac_key, topic_bytes || segments)[..16]     // First block
mac_n = HMAC(mac_key, mac_{n-1}  || segments)[..16]      // Subsequent blocks
```

The MAC covers the whole serialized block, so `sequenceNumber`/`epoch` are
authenticated implicitly. Consumer ordering/replay enforcement (gap-fatal vs
per-Group media) is selected by the stream's `StreamOpt`.

### StreamPayload (Content)

```capnp
struct StreamPayload {
    union {
        data      @0 :Data;           # Generic binary payload (tokens, I/O, etc.)
        complete  @1 :Data;           # App-specific completion metadata (serialized)
        error     @2 :StreamError;    # Error during streaming
        heartbeat @3 :Void;           # Keep-alive
        tagged    @4 :TaggedPayload;  # Encrypted tagged payload with key commitment
    }
}
```

### StreamError

```capnp
struct StreamError {
    message @0 :Text;
    code    @1 :Text;    # "timeout", "cancelled", "internal", etc.
    details @2 :Text;
}
```

### StreamControl (Consumer → Producer)

Cancellation and keep-alive travel on a separate DH-derived control channel
(`ctrl_topic` / `ctrl_mac_key`):

```capnp
struct StreamControl {
    union {
        cancel @0 :Void;
        ping   @1 :Void;
    }
}
```

## moq-lite Origin

**Location:** `crates/hyprstream-rpc/src/moq_stream.rs`

Every process that publishes moq streams initializes its own process-global
`MoqStreamOrigin` at startup — the `streams` service factory, or (in
multi-process deployments) `init_local_moq_stream_plane` in any
stream-publisher service's factory (both idempotent):

```rust
// Startup (streams service factory): register origin + start UDS listener.
// Both functions are synchronous.
let origin = MoqStreamOrigin::standalone()...build();
init_global_moq_origin(origin.clone());          // fn(MoqStreamOrigin) -> bool (idempotent)
serve_moq_uds_background(origin, moq_uds_path);  // fn(MoqStreamOrigin, PathBuf)

// From anywhere in the process
let uds_path = global_moq_uds_path()
    .ok_or_else(|| anyhow!("moq not initialized"))?;
```

`serve_moq_uds_background` binds the socket synchronously (mode `0o600`,
peer-credential checked) before publishing the path, so any caller reading
`global_moq_uds_path()` is guaranteed the socket is ready. Each accepted UDS
connection gets a dedicated moq server session sharing the same live broadcast
tree, serving cross-process local subscribers (e.g. `hyprstream tui attach`).

## Memory Management

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| Stream completion | `complete`/`error` payload | Background task exits |
| Cancellation | `handle.cancel()` or StreamControl `cancel` | Background task exits |
| Connection close | UDS/QUIC peer disconnect | Background task exits |
| Relay retention | `StreamOpt::retention` | moq-lite drops per policy |

The moq origin does not implement per-stream TTLs; streams are expected to
complete or be cancelled. The origin lives in a process-global `OnceLock` for
the process lifetime.

## Integration with Generated Clients

Codegen (`hyprstream-rpc-derive`) generates streaming methods that return
`MoqStreamHandle` via the networked constructor:

```rust
// Generated client method (simplified)
impl InferenceClient {
    pub async fn infer_stream(&self, request: InferStreamRequest) -> Result<MoqStreamHandle> {
        let (ephemeral_sk, ephemeral_pk) = generate_ephemeral_keypair();
        let info: StreamInfo = self.call_rpc(request, ephemeral_pk).await?;

        let keys = derive_stream_keys(&dh(&ephemeral_sk, &info.dh_public), ...)?;
        Ok(MoqStreamHandle::networked(
            info.announced_at, &info.qos, info.broadcast_path,
            keys.mac_key, keys.enc_key, keys.topic,
        ))
    }
}
```

## Files

| File | Purpose |
|------|---------|
| `crates/hyprstream-rpc/src/moq_stream.rs` | `MoqStreamOrigin`, `MoqStreamHandle`, UDS server, reach selection |
| `crates/hyprstream-rpc/src/streaming.rs` | `StreamChannel`, `StreamContext`, `StreamVerifier`, block encoding |
| `crates/hyprstream-rpc/src/stream_info.rs` | `StreamInfo`, `StreamOpt`, `Destination`, presets |
| `crates/hyprstream-rpc/src/moq_event.rs` | `MoqEventOrigin` — the separate broadcast event bus |
| `crates/hyprstream-rpc/schema/streaming.capnp` | Cap'n Proto schema |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for verification |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | DH key derivation (`derive_stream_keys`) |
| `crates/hyprstream-rpc-derive/src/codegen/client.rs` | Generated streaming client methods |

## Related Documentation

- [Cryptography Architecture](./cryptography-architecture.md) — Key exchange and HMAC details
- [RPC Architecture](./rpc-architecture.md) — Overall service communication patterns
- [EventService Architecture](./eventservice-architecture.md) — Event bus (moq-lite)
