# Streaming Service Architecture

PULL/XPUB queuing proxy with signed registration for end-to-end authenticated streaming.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAM SERVICE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  InferenceService                 StreamService                    Client   │
│       │                                │                              │     │
│       │─ SignedEnvelope(Register) ────►│                              │     │
│       │   [verify sig, check claims]   │                              │     │
│       │                                │                              │     │
│       │─ StreamChunk/Block ──────────►│                              │     │
│       │   [extract topic]              │                              │     │
│       │   [NO HMAC verification]       │                              │     │
│       │                                │─ {topic}{message} ──────────►│     │
│       │                                │   [XPUB prefix routing]      │[verify]
│       │                                │                              │     │
│       │                                │◄─ StreamResume(topic,hmac) ──│     │
│       │                                │   [find hmac in buffer]      │     │
│       │                                │─ {buffered messages...} ────►│     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Location:** `crates/hyprstream-rpc/src/service/streaming.rs`

## Why PUSH/PULL Instead of PUB/XSUB

PUB/SUB drops messages when no subscriber exists, causing a race condition:

```
Traditional PUB/SUB:                    PUSH/PULL Solution:

Publisher                               Publisher
    │                                       │
    │  Starts publishing                    │  Starts publishing
    │  (messages dropped!)                  │  (buffered at HWM)
    ▼                                       ▼
    ...time passes...                   StreamService
    ▼                                       │
Subscriber                                  │  Queues per-topic
    │  Finally subscribes                   ▼
    │  (missed first messages)          Subscriber
                                            │  Subscribes
                                            │  (receives ALL messages)
```

PUSH/PULL solves the race condition:
- **PUSH buffers** at HWM (never drops)
- **StreamService queues** per-topic until subscriber arrives
- **On subscribe**, queued messages are flushed to client

## Security Model (E2E Authentication)

StreamService is a **blind forwarder** - it does NOT verify HMACs.

| Layer | Responsibility |
|-------|----------------|
| **InferenceService** | Derives DH keys, produces HMAC chain |
| **StreamService** | Routes by topic, buffers for retransmit (blind) |
| **Client** | Derives same DH keys, verifies HMAC chain |

### Security Properties

| Property | Implementation |
|----------|----------------|
| **Topic unpredictability** | DH-derived (InferenceService ↔ Client) |
| **Registration auth** | SignedEnvelope with Ed25519 signature |
| **Data integrity** | Chained HMAC verified end-to-end by client |
| **Stream binding** | Claims-based scope: `publish:stream:{topic}` |
| **Replay protection** | Nonce cache on SignedEnvelope |

## Components

### StreamService

The main queuing proxy that routes messages from publishers to subscribers.

```rust
pub struct StreamService {
    name: String,
    context: Arc<zmq::Context>,
    pub_transport: TransportConfig,    // XPUB frontend (client-facing)
    pull_transport: TransportConfig,   // PULL backend (receives from publishers)
    message_ttl: Duration,             // Default: 30s
    max_pending_per_topic: usize,      // Default: 1000
    compact_interval: Duration,        // Default: 5s
    nonce_cache: Arc<InMemoryNonceCache>,
}
```

### StreamState

Unified state for tracking authorization, subscription, and messages per topic.

```rust
struct StreamState {
    /// Expiration from claims (Unix timestamp)
    exp: i64,

    /// Whether a client has subscribed
    subscribed: bool,

    /// Message queue (also serves as retransmit buffer)
    messages: VecDeque<PendingMessage>,
}

struct PendingMessage {
    data: Vec<u8>,           // Original capnp bytes (StreamChunk or StreamBlock)
    received_at: Instant,    // For TTL expiry
    hmac: [u8; 32],          // For retransmit buffer indexing
}
```

## Message Types

**Schema:** `crates/hyprstream-rpc/schema/streaming.capnp`

### StreamRegister

Registration message wrapped in `SignedEnvelope` for authentication.

```
SignedEnvelope {
    envelope: RequestEnvelope {
        payload: StreamRegister {
            topic: "abc123...",  // 64 hex chars (DH-derived)
            exp: 1762974327,     // Unix timestamp
        },
        claims: Claims { scopes: ["publish:stream:abc123..."] },
    },
    signature: [64 bytes],  // Ed25519
}
```

### Wire Formats: StreamChunk vs StreamBlock

Two wire formats are supported for streaming data:

| Format | Use Case | HMAC | Batching |
|--------|----------|------|----------|
| **StreamChunk** | Single payloads | 32 bytes (in capnp) | No |
| **StreamBlock** | Batched payloads | 16 bytes (ZMQ frame) | Yes |

#### StreamChunk (Single Payload)

Self-contained message with HMAC embedded in capnp structure.

```
StreamChunk {
    topic: "abc123...",     // 64 hex chars (DH-derived)
    data: [bytes],          // Serialized StreamPayload
    hmac: [32 bytes],       // Chained HMAC-SHA256 (full)
    prevHmac: [32 bytes],   // Previous chunk's HMAC (empty for first)
}
```

**MAC Chain:**
```
mac_0 = HMAC(key, topic_bytes || data_0)       // First chunk
mac_n = HMAC(key, mac_{n-1} || data_n)         // Subsequent chunks
```

#### StreamBlock (Batched Payloads)

ZMQ multipart message with truncated HMAC as separate frame.

```
Wire format (ZMQ multipart):
  Frame 0:      topic (64 hex chars, DH-derived)
  Frame 1..N-1: capnp segments (StreamBlock struct)
  Frame N:      mac (16 bytes, truncated HMAC-SHA256)

StreamBlock (capnp) {
    prevMac: [16 bytes],           // topic[..16] for block 0, mac_{n-1} for block N
    payloads: List(StreamPayload), // Multiple payloads batched
}
```

**MAC Chain:**
```
Block 0: mac = HMAC(mac_key, topic_bytes || segments)[..16]
Block N: mac = HMAC(mac_key, prev_mac || segments)[..16]
```

### StreamPayload (Content)

The actual content inside `StreamChunk.data` or `StreamBlock.payloads`:

```
StreamPayload {
    streamId: Text,         // Stream identifier for correlation

    union {
        token: Text,              // Generated token text
        complete: StreamStats,    // Stream completion with stats
        error: StreamError,       // Error during generation
        heartbeat: Void,          // Keep-alive (no data)
    }
}
```

#### StreamStats (Completion)

```
StreamStats {
    tokensGenerated: UInt32,
    finishReason: Text,        // "stop", "length", "eos", "error"
    generationTimeMs: UInt64,
    tokensPerSecond: Float32,
    perplexity: Float32,       // Optional quality metric
    avgEntropy: Float32,       // Optional quality metric
}
```

#### StreamError

```
StreamError {
    message: Text,
    code: Text,      // "timeout", "cancelled", "internal", etc.
    details: Text,   // Optional additional context
}
```

### StreamResume

Client request to retransmit chunks/blocks after the given HMAC.

```
StreamResume {
    topic: "abc123...",
    resumeFromHmac: [32 bytes],  // Last valid HMAC received
}
```

StreamService finds this HMAC in its buffer and resends all subsequent messages.

## Message Flow

### 1. Stream Registration

```
InferenceService                        StreamService
      │                                      │
      │  1. Derive DH keys with client       │
      │  2. topic = derive_stream_keys()     │
      │                                      │
      │  SignedEnvelope(StreamRegister)      │
      ├─────────────────────────────────────►│
      │                                      │  3. Verify signature
      │                                      │  4. Check claims.scopes
      │                                      │  5. Create StreamState
      │                                      │
```

### 2. Chunk Publishing (Before Subscriber)

```
InferenceService                        StreamService
      │                                      │
      │  StreamChunk(topic, "Hello", hmac)   │
      ├─────────────────────────────────────►│
      │                                      │  Queue in messages
      │                                      │
      │  StreamChunk(topic, "World", hmac)   │
      ├─────────────────────────────────────►│
      │                                      │  Queue in messages
```

### 3. Subscriber Connects (Flush Queued)

```
                                        StreamService                Client
                                             │                          │
                                             │◄─── SUB("abc123...") ────┤
                                             │                          │
                                             │  Mark subscribed=true    │
                                             │  Flush queued messages   │
                                             │                          │
                                             │─── {topic}{StreamChunk} ►│
                                             │─── {topic}{StreamChunk} ►│
                                             │                          │ Verify HMAC
```

### 4. Live Streaming (After Subscriber)

```
InferenceService                        StreamService                Client
      │                                      │                          │
      │  StreamChunk(topic, data, hmac)      │                          │
      ├─────────────────────────────────────►│                          │
      │                                      │─── {topic}{StreamChunk} ►│
      │                                      │     [immediate forward]  │
      │                                      │                          │ Verify HMAC
```

### 5. Stream Resume (Recovery)

```
                                        StreamService                Client
                                             │                          │
                                             │◄─ StreamResume(topic, ──┤
                                             │      last_valid_hmac)    │
                                             │                          │
                                             │  Find hmac in buffer     │
                                             │  Resend subsequent chunks│
                                             │                          │
                                             │─── {topic}{chunk_n+1} ──►│
                                             │─── {topic}{chunk_n+2} ──►│
```

## Memory Management

### Message Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MESSAGE LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Receive StreamChunk/StreamBlock                                      │
│     ├── Topic not registered? → DROP (unregistered topic)               │
│     └── Topic registered                                                 │
│         ├── subscribed=true → Forward + add to buffer                   │
│         └── subscribed=false → Add to queue/buffer only                 │
│                                                                          │
│  2. Buffer management                                                    │
│     ├── Per-topic limit: 1000 messages (oldest dropped on overflow)     │
│     └── Message TTL: 30 seconds (expired on flush/compact)              │
│                                                                          │
│  3. Stream cleanup                                                       │
│     ├── Claims expired (compact interval) → Remove StreamState          │
│     ├── Unsubscribe (0x00) → Remove StreamState entirely                │
│     └── Empty + not subscribed (compact) → Remove StreamState           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```rust
StreamService::new(...)
    .with_buffer_config(
        max_pending_per_topic: 1000,     // Max messages per topic
        message_ttl: Duration::from_secs(30),  // Message expiry
        compact_interval: Duration::from_secs(5),  // Cleanup frequency
    )
```

### Memory Limits

| Setting | Default | Purpose |
|---------|---------|---------|
| `max_pending_per_topic` | 1000 | Prevents unbounded growth per stream |
| `message_ttl` | 30s | Drops old messages if subscriber never arrives |
| `compact_interval` | 5s | How often to run expiry checks |
| Socket HWM | 100,000 | ZMQ high-water mark for buffering |

## Socket Architecture

### PULL Socket (Backend)

Receives messages from publishers (InferenceService uses PUSH).

```rust
fn setup_pull(&self) -> Result<zmq::Socket> {
    let pull = self.context.socket(zmq::PULL)?;

    // Buffer up to 100K messages (~10 seconds at 10K msg/s)
    pull.set_rcvhwm(100_000)?;

    // Bind with transport layer (handles CurveZMQ)
    self.pull_transport.bind(&mut pull)?;

    // Restrictive permissions for IPC sockets
    #[cfg(unix)]
    if let EndpointType::Ipc { path } = &self.pull_transport.endpoint {
        std::fs::set_permissions(path, Permissions::from_mode(0o600))?;
    }

    Ok(pull)
}
```

### XPUB Socket (Frontend)

Publishes to subscribers with topic-prefix routing.

```rust
fn setup_xpub(&self) -> Result<zmq::Socket> {
    let xpub = self.context.socket(zmq::XPUB)?;

    xpub.set_sndhwm(100_000)?;

    // Receive ALL subscribe/unsubscribe notifications
    xpub.set_xpub_verbose(true)?;

    self.pub_transport.bind(&mut xpub)?;

    Ok(xpub)
}
```

### XPUB Subscription Protocol

```
Client sends:     0x01 + topic_bytes   → Subscribe
Client sends:     0x00 + topic_bytes   → Unsubscribe

Service receives subscription via xpub.recv_bytes()
Service sends:    topic_bytes + message_bytes   → Routed to matching subscribers
```

## Event Loop

The main proxy loop handles three socket types:

```rust
let mut items = [
    pull.as_poll_item(zmq::POLLIN),   // Index 0: Publisher messages
    xpub.as_poll_item(zmq::POLLIN),   // Index 1: Client subscriptions
    ctrl.as_poll_item(zmq::POLLIN),   // Index 2: Shutdown signal
];

loop {
    // Periodic compaction
    if last_compact.elapsed() >= self.compact_interval {
        compact_expired_streams(&mut streams);
    }

    zmq::poll(&mut items, 1000)?;

    // Handle shutdown
    if items[2].is_readable() { /* TERMINATE */ break; }

    // Handle publisher messages
    if items[0].is_readable() {
        match parse_message(&msg) {
            SignedEnvelope(StreamRegister) => handle_register(),
            StreamResume => handle_resume(),
            StreamChunk => handle_chunk(),  // Blind forward
        }
    }

    // Handle client subscriptions
    if items[1].is_readable() {
        match subscription[0] {
            0x01 => handle_subscribe(),   // Flush queued, mark subscribed
            0x00 => handle_unsubscribe(), // Remove StreamState
        }
    }
}
```

## Integration

### With InferenceService

```rust
// InferenceService derives keys and registers stream
let (topic, mac_key) = derive_stream_keys(&shared_secret, ...)?;

// Send registration via PUSH
let register = StreamRegister { topic: topic.clone(), exp };
let signed = sign_envelope(register, &signing_key)?;
push_socket.send(serialize(&signed), 0)?;

// Stream chunks via PUSH
let mut hmac = ChainedStreamHmac::from_bytes(mac_key, request_id);
for token in tokens {
    let mac = hmac.compute_next(&token);
    let chunk = StreamChunk { topic: topic.clone(), data: token, hmac: mac };
    push_socket.send(serialize(&chunk), 0)?;
}
```

### With Client

```rust
// Client subscribes via SUB socket
sub_socket.set_subscribe(topic.as_bytes())?;

// Client verifies HMAC chain
let mut verifier = ChainedStreamHmac::from_bytes(mac_key, request_id);
while let Ok(msg) = sub_socket.recv_bytes(0) {
    let chunk = parse_stream_chunk(&msg)?;
    verifier.verify_next(&chunk.data, &chunk.hmac)?;
    process_token(&chunk.data);
}
```

### Spawnable Trait

StreamService implements `Spawnable` for use with `ServiceSpawner`:

```rust
impl Spawnable for StreamService {
    fn name(&self) -> &str { &self.name }

    fn context(&self) -> &Arc<zmq::Context> { &self.context }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![
            (SocketKind::Sub, self.pub_transport.clone()),   // Clients subscribe
            (SocketKind::Push, self.pull_transport.clone()), // Publishers push
        ]
    }

    fn run(self: Box<Self>, shutdown: Arc<Notify>, on_ready: ...) -> Result<()> {
        let xpub = self.setup_xpub()?;
        let pull = self.setup_pull()?;

        if let Some(tx) = on_ready { tx.send(()); }

        self.run_loop_with_sockets(xpub, pull, shutdown)
    }
}
```

## Transport Security

### CurveZMQ Support

Both frontend and backend sockets support CurveZMQ encryption:

```rust
let service = StreamService::new(
    "inference-stream",
    context.clone(),
    TransportConfig {
        endpoint: EndpointType::Tcp { port: 5556 },
        curve: Some(CurveConfig::server(server_keypair)),
    },
    TransportConfig {
        endpoint: EndpointType::Ipc { path: "/run/hyprstream/stream.sock" },
        curve: None,  // Internal IPC doesn't need encryption
    },
);
```

### IPC Socket Permissions

IPC sockets automatically get restrictive permissions:

```rust
// Unix only
std::fs::set_permissions(path, Permissions::from_mode(0o600))?;
// Owner read/write only
```

## Message Parsing

### SignedEnvelope Detection

```rust
fn is_signed_envelope(msg: &[u8]) -> bool {
    // Parse as capnp
    let envelope = reader.get_root::<signed_envelope::Reader>()?;

    // Check signature is 64 bytes (Ed25519)
    // This distinguishes from StreamChunk (32-byte HMAC)
    envelope.get_signature()?.len() == 64
}
```

### StreamChunk Parsing

```rust
fn parse_stream_chunk(msg: &[u8]) -> Option<(String, Vec<u8>, [u8; 32])> {
    let chunk = reader.get_root::<stream_chunk::Reader>()?;

    let topic = chunk.get_topic()?.to_str()?.to_string();
    let data = chunk.get_data()?.to_vec();
    let hmac = chunk.get_hmac()?;  // 32 bytes

    Some((topic, data, hmac))
}
```

## Files

| File | Purpose |
|------|---------|
| `crates/hyprstream-rpc/src/service/streaming.rs` | StreamService implementation |
| `crates/hyprstream-rpc/schema/streaming.capnp` | Cap'n Proto schema |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for verification |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | DH key derivation for topics |

## Related Documentation

- [Cryptography Architecture](./cryptography-architecture.md) - Key exchange and HMAC details
- [RPC Architecture](./rpc-architecture.md) - Overall service communication patterns
- [Policy Service Architecture](./policy-service-architecture.md) - Claims and scope validation
