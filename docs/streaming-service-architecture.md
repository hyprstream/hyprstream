# Streaming Service Architecture

moq-lite streaming plane with HMAC-chained end-to-end authentication.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMING PLANE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  InferenceService              moq-lite Origin            Client            │
│       │                             │                        │              │
│       │─ infer_stream (RPC) ────────────────────────────────►│              │
│       │◄─ {stream_id, moq_uds_path, moq_broadcast_path} ─────┤              │
│       │                             │                        │              │
│       │  derive DH keys             │                        │              │
│       │  topic = DH-derived hex     │                        │              │
│       │                             │                        │              │
│       │─ publish(broadcast_path) ──►│                        │              │
│       │   [token chunks, HMAC]      │                        │              │
│       │                             │                        │              │
│       │                             │◄─ subscribe(UDS) ──────┤              │
│       │                             │   [broadcast_path]     │              │
│       │                             │                        │              │
│       │                             │─ chunk ───────────────►│              │
│       │                             │─ chunk ───────────────►│[verify HMAC] │
│       │                             │─ StreamComplete ───────►│              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The old ZMQ PUSH/PULL → XPUB/XSUB `StreamService` was removed in epic #131/#138. The
current streaming plane is moq-lite: InferenceService publishes to the process-global
`MoqEventOrigin`; clients subscribe via a UDS connection to the moq socket and receive
chunks through `MoqStreamHandle`.

## Security Model (E2E Authentication)

The moq origin is a **blind router** — it does NOT verify HMACs.

| Layer | Responsibility |
|-------|----------------|
| **InferenceService** | Derives DH keys, produces HMAC chain, signs RPC responses |
| **moq-lite Origin** | Routes by broadcast path, buffers for late subscribers |
| **Client** | Derives same DH keys via ephemeral pubkey, verifies HMAC chain |

### Shared Signing Key

All services use the **same Ed25519 signing key** loaded from:
```
~/.local/share/hyprstream/models/.registry/keys/signing.key
```

This key is:
- **Generated once** on first startup (32 bytes, persisted)
- **Loaded by all services** (systemd units and CLI)
- **Used for both requests and responses** (bidirectional authentication)

### Security Properties

| Property | Implementation |
|----------|----------------|
| **Topic unpredictability** | DH-derived (InferenceService ↔ Client ephemeral key) |
| **Response auth** | ResponseEnvelope with Ed25519 signature (mandatory) |
| **Data integrity** | Chained HMAC-SHA256 verified end-to-end by client |
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
│    │       [signed with server's key]               │      │
│    │  [verify]                                      │      │
└────────────────────────────────────────────────────────────┘
```

## Stream Lifecycle

### 1. Client Initiates Inference Stream

Client calls `infer_stream` via the generated RPC client. The request carries an
ephemeral X25519 public key for DH key exchange:

```
RequestEnvelope {
    payload: InferStreamRequest { model, prompt, ... },
    ephemeral_pubkey: [32 bytes],  // Client's X25519 public key
}
```

### 2. InferenceService Responds with moq Paths

InferenceService derives shared stream keys and returns the moq subscription info:

```rust
// Server-side key derivation
let (topic, mac_key) = derive_client_stream_keys(
    &server_ephemeral_sk,
    &request.ephemeral_pubkey,
    request_id,
)?;

// StreamInfo returned in RPC response
StreamInfo {
    stream_id: uuid,
    moq_uds_path: global_moq_uds_path(),       // /tmp/hyprstream-{pid}/moq.sock
    moq_broadcast_path: format!("streams/{topic}"),
    // endpoint field is empty (reserved, formerly ZMQ address)
}
```

### 3. Client Connects via MoqStreamHandle

```rust
let handle = MoqStreamHandle::new(
    stream_info.moq_uds_path,
    stream_info.moq_broadcast_path,
    mac_key,   // Derived from same DH exchange
    stream_id,
).await?;

// Async iteration
while let Some(payload) = handle.recv_next().await? {
    match payload {
        StreamPayload::Token(text) => process_token(text),
        StreamPayload::Complete(stats) => break,
        StreamPayload::Error(e) => return Err(e.into()),
    }
}
```

`MoqStreamHandle` implements `futures::Stream` and handles:
- UDS connection to moq socket
- moq subscription on `broadcast_path`
- Per-chunk HMAC-SHA256 chain verification
- Cancellation via `cancel()` / `cancel_token()`

### 4. InferenceService Publishes Chunks

While the RPC response is sent immediately (with the moq paths), InferenceService
runs token generation in the background and publishes each token chunk:

```rust
let mut hmac = ChainedStreamHmac::new(mac_key, request_id);
for token in generate_tokens(...) {
    let mac = hmac.compute_next(&token);
    let chunk = StreamChunk { topic: topic.clone(), data: token, hmac: mac };
    moq_origin.publish(&broadcast_path, serialize(&chunk)).await?;
}

// Completion marker
moq_origin.publish(&broadcast_path, serialize(&StreamComplete { stats })).await?;
```

## Message Types

**Schema:** `crates/hyprstream-rpc/schema/streaming.capnp`

### StreamChunk (Single Payload)

Self-contained message with HMAC embedded in capnp structure.

```capnp
struct StreamChunk {
    topic    @0 :Text;       # DH-derived hex string
    data     @1 :Data;       # Serialized StreamPayload
    hmac     @2 :Data;       # Chained HMAC-SHA256 (32 bytes)
    prevHmac @3 :Data;       # Previous chunk HMAC (empty for first)
}
```

**MAC Chain:**
```
mac_0 = HMAC(mac_key, topic_bytes || data_0)       // First chunk
mac_n = HMAC(mac_key, mac_{n-1} || data_n)         // Subsequent chunks
```

### StreamBlock (Batched Payloads)

Batched format for high-throughput paths. Wire format:

```
Frame 0:      topic (DH-derived hex string)
Frame 1..N-1: capnp segments (StreamBlock struct)
Frame N:      mac (16 bytes, truncated HMAC-SHA256)

StreamBlock {
    prevMac:  Data;                    # topic[..16] for block 0, mac_{n-1} for block N
    payloads: List(StreamPayload);     # Multiple payloads batched
}
```

### StreamPayload (Content)

The actual content inside `StreamChunk.data` or `StreamBlock.payloads`:

```capnp
struct StreamPayload {
    streamId @0 :Text;       # Stream identifier for correlation

    union {
        token     @1 :Text;          # Generated token text
        complete  @2 :StreamStats;   # Completion with stats
        error     @3 :StreamError;   # Error during generation
        heartbeat @4 :Void;          # Keep-alive
    }
}
```

#### StreamStats (Completion)

```capnp
struct StreamStats {
    tokensGenerated  @0 :UInt32;
    finishReason     @1 :Text;     # "stop", "length", "eos", "error"
    generationTimeMs @2 :UInt64;
    tokensPerSecond  @3 :Float32;
    perplexity       @4 :Float32;
    avgEntropy       @5 :Float32;
}
```

### StreamInfo (in RPC response)

Returned by streaming RPC calls to tell the client where to subscribe:

```capnp
struct StreamInfo {
    streamId         @0 :Text;   # UUID for correlation
    endpoint         @1 :Text;   # Reserved (empty on moq paths)
    serverPubkey     @2 :Data;   # Server's ephemeral X25519 public key
    moqUdsPath       @4 :Text;   # /tmp/hyprstream-{pid}/moq.sock
    moqBroadcastPath @5 :Text;   # streams/{topic}
}
```

## MoqStreamHandle

**Location:** `crates/hyprstream-rpc/src/moq_stream.rs`

```rust
pub struct MoqStreamHandle {
    // internal: background task + channel
}

impl MoqStreamHandle {
    /// Connect to moq origin and subscribe to broadcast_path.
    pub async fn new(
        uds_path: impl AsRef<Path>,
        broadcast_path: impl Into<String>,
        mac_key: [u8; 32],
        stream_id: String,
    ) -> Result<Self>;

    /// Receive next verified payload.
    pub async fn recv_next(&mut self) -> Result<Option<StreamPayload>>;

    /// Signal cancellation to the background task.
    pub fn cancel(&self);

    /// Get a CancellationToken for integration with tokio-util.
    pub fn cancel_token(&self) -> CancellationToken;

    /// Stream ID for correlation.
    pub fn stream_id(&self) -> &str;
}

impl futures::Stream for MoqStreamHandle {
    type Item = Result<StreamPayload>;
    // ...
}
```

The background task:
1. Opens a UDS connection to `moq_uds_path`
2. Issues a moq SUBSCRIBE for `broadcast_path`
3. For each received object, deserializes and verifies the HMAC chain
4. Sends verified `StreamPayload` values to the foreground via `mpsc`
5. Terminates on `cancel()`, connection close, or `StreamComplete`

## moq-lite Origin

**Location:** `crates/hyprstream-rpc/src/moq_stream.rs`

The process-global origin is initialized at startup:

```rust
// Startup: initialize origin + start UDS listener
let origin = init_global_moq_origin().await?;
serve_moq_uds_background(origin.clone(), shutdown.clone()).await?;

// From anywhere in the process
let uds_path = global_moq_uds_path()
    .ok_or_else(|| anyhow!("moq not initialized"))?;
```

The UDS listener accepts external connections (from clients in the same host,
including subprocess workers) and forwards moq SUBSCRIBE/PUBLISH frames to the
in-process origin.

## Memory Management

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| Stream completion | `StreamComplete` published | Background task exits |
| Cancellation | `handle.cancel()` | Background task exits |
| Connection close | UDS peer disconnect | Background task exits |
| moq buffer limit | Configurable | moq-lite drops oldest cached objects |

The moq origin does not implement per-stream TTLs; streams are expected to complete
or be cancelled by the client. The `MoqEventBarrierService` holds the origin alive
for the process lifetime.

## Integration with Generated Clients

Codegen (`hyprstream-rpc-derive`) generates streaming methods that return `MoqStreamHandle`:

```rust
// Generated client method (simplified)
impl InferenceClient {
    pub async fn infer_stream(
        &self,
        request: InferStreamRequest,
    ) -> Result<MoqStreamHandle> {
        let (ephemeral_sk, ephemeral_pk) = generate_ephemeral_keypair();
        let info: StreamInfo = self.call_rpc(request, ephemeral_pk).await?;

        if info.moq_uds_path.is_empty() {
            bail!("Server did not provide moq transport path — moq transport not initialized");
        }

        let mac_key = derive_client_stream_keys(&ephemeral_sk, &info.server_pubkey, ...)?;
        MoqStreamHandle::new(info.moq_uds_path, info.moq_broadcast_path, mac_key, info.stream_id).await
    }
}
```

## Files

| File | Purpose |
|------|---------|
| `crates/hyprstream-rpc/src/moq_stream.rs` | `MoqStreamHandle`, `MoqEventOrigin`, UDS server |
| `crates/hyprstream-rpc/src/moq_event.rs` | `MoqEventBarrierService`, event bus |
| `crates/hyprstream-rpc/schema/streaming.capnp` | Cap'n Proto schema |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for verification |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | DH key derivation |
| `crates/hyprstream-rpc-derive/src/codegen/client.rs` | Generated streaming client methods |

## Related Documentation

- [Cryptography Architecture](./cryptography-architecture.md) — Key exchange and HMAC details
- [RPC Architecture](./rpc-architecture.md) — Overall service communication patterns
- [EventService Architecture](./eventservice-architecture.md) — Event bus (moq-lite)
