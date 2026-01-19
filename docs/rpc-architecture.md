# RPC Architecture

This document describes the ZeroMQ-based RPC infrastructure used by hyprstream for inter-service communication.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYPRSTREAM SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  hyprstream     │    │ hyprstream-rpc  │    │ hyprstream-     │         │
│  │  (main app)     │    │ (RPC infra)     │    │ workers         │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  │                                          │
│  ┌───────────────────────────────┴───────────────────────────────┐         │
│  │                    SHARED ZMQ INFRASTRUCTURE                   │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │         │
│  │  │ REQ/REP     │  │ PUB/SUB     │  │ PUSH/PULL → XPUB    │   │         │
│  │  │ (RPC calls) │  │ (events)    │  │ (inference stream)  │   │         │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## RPC Message Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RPC MESSAGE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLIENT                              SERVICE                                │
│  ┌─────────────────┐                ┌─────────────────┐                    │
│  │ ZmqClient       │                │ ServiceRunner   │                    │
│  │                 │                │                 │                    │
│  │ 1. Build payload│                │                 │                    │
│  │    (Cap'n Proto)│                │                 │                    │
│  │         │       │                │                 │                    │
│  │ 2. RequestEnvelope              │                 │                    │
│  │    + identity   │                │                 │                    │
│  │    + nonce      │                │                 │                    │
│  │         │       │                │                 │                    │
│  │ 3. Sign Ed25519 │                │                 │                    │
│  │    →SignedEnvelope              │                 │                    │
│  │         │       │   TMQ REQ     │                 │                    │
│  │         └───────┼───────────────►│ 4. Receive     │                    │
│  │                 │                │ 5. Verify sig  │                    │
│  │                 │                │ 6. Check nonce │                    │
│  │                 │                │ 7. Dispatch    │                    │
│  │                 │                │    handler()   │                    │
│  │                 │   TMQ REP     │ 8. Respond     │                    │
│  │ 9. Parse resp  ◄├───────────────┤                 │                    │
│  └─────────────────┘                └─────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Security Model

### Envelope Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENVELOPE SECURITY LAYERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SignedEnvelope                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  signature: [u8; 64]    ← Ed25519 over RequestEnvelope          │       │
│  │  signer_pubkey: [u8; 32]                                        │       │
│  │  ┌───────────────────────────────────────────────────────────┐  │       │
│  │  │ RequestEnvelope                                           │  │       │
│  │  │   request_id: u64                                         │  │       │
│  │  │   identity: RequestIdentity                               │  │       │
│  │  │   nonce: [u8; 16]     ← Replay protection                 │  │       │
│  │  │   timestamp: i64      ← Clock skew check                  │  │       │
│  │  │   ephemeral_pubkey    ← Stream HMAC derivation            │  │       │
│  │  │   payload: Vec<u8>    ← Actual request (Cap'n Proto)      │  │       │
│  │  └───────────────────────────────────────────────────────────┘  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  RequestIdentity → casbin_subject()                                         │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Local { user }        → "local:alice"                          │       │
│  │  ApiToken { user, .. } → "token:bob"                            │       │
│  │  Peer { name, .. }     → "peer:gpu-server-1"                    │       │
│  │  Anonymous             → "anonymous"                            │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Security Layers

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Transport** | CURVE encryption | End-to-end encryption for TCP connections |
| **Application** | Ed25519 signatures | Request authentication and integrity |
| **Authorization** | Casbin policy | RBAC/ABAC access control |

### JWT Authorization (Added 2026-01-15)

Services can enforce user-level authorization via JWT tokens embedded in request envelopes. This provides end-to-end user attribution and fine-grained access control.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      JWT AUTHORIZATION FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLIENT                    PolicyService              InferenceService      │
│    │                             │                            │             │
│    │ 1. Request JWT token        │                            │             │
│    ├──issue_token(scopes)───────►│                            │             │
│    │   ["infer:model:qwen-7b"]   │                            │             │
│    │                             │ Check Casbin policy        │             │
│    │                             │ Sign JWT (Ed25519)         │             │
│    │◄──Returns: jwt_token────────┤                            │             │
│    │   (contains Claims)         │                            │             │
│    │                             │                            │             │
│    │ 2. Make RPC call with JWT   │                            │             │
│    │ RequestEnvelope {           │                            │             │
│    │   jwt_token: "hypr_eyJ..."  │                            │             │
│    │   payload: GenerateRequest  │                            │             │
│    │   ... (signed envelope)     │                            │             │
│    │ }                           │                            │             │
│    ├────────────────────────────────────────────────────────►│             │
│    │                             │                            │             │
│    │                             │      3. Validate JWT       │             │
│    │                             │      ctx.validate_jwt()    │             │
│    │                             │      (~200µs stateless)    │             │
│    │                             │                            │             │
│    │                             │      4. Check authorization│             │
│    │                             │      #[authorize(          │             │
│    │                             │        action="infer",     │             │
│    │                             │        resource="model"    │             │
│    │                             │      )]                    │             │
│    │                             │      • Check Casbin        │             │
│    │                             │      • Check JWT scopes    │             │
│    │                             │                            │             │
│    │◄────────────Returns: Result──────────────────────────────┤             │
│    │   (or 403 Forbidden)        │                            │             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Features

**Stateless Validation:**
- JWT signature verification: ~200µs per request
- No server-side token registry or caching
- Eliminates cache-related vulnerabilities (timing attacks, poisoning, stale tokens)

**Structured Scopes:**
- Format: `action:resource:identifier`
- Examples:
  - `infer:model:qwen-7b` - Run inference on specific model
  - `subscribe:stream:abc-123` - Subscribe to specific stream
  - `read:model:*` - Read any model (explicit wildcard)
- Safe wildcards: Only identifier field supports `*`
- Action/resource isolation prevents privilege escalation

**Defense-in-Depth:**
```
Layer 1: Ed25519 envelope signature (service identity)
         ↓
Layer 2: Casbin policy check (RBAC/ABAC)
         ↓
Layer 3: JWT scope validation (least privilege)
```

**Canonical Serialization:**
- Uses Cap'n Proto canonical form for deterministic signatures
- Prevents signature verification failures across platforms/library versions
- Required by Cap'n Proto specification for signing operations

#### Integration with RequestEnvelope

```capnp
# Schema: common.capnp
struct RequestEnvelope {
  requestId @0 :UInt64;
  identity @1 :RequestIdentity;
  payload @2 :Data;
  ephemeralPubkey @3 :Data;
  nonce @4 :Data;
  timestamp @5 :Int64;
  jwtToken @6 :Text;  # Signed JWT token string (added 2026-01-15)
}
```

The `jwtToken` field contains a **signed JWT token string** (not deserialized Claims), ensuring independent signature verification by services.

#### Service Implementation

Services use the `#[authorize]` macro for declarative authorization:

```rust
use hyprstream_rpc::{authorize, register_scopes};

#[register_scopes]  // Auto-registers scopes at compile time
impl InferenceService {
    #[authorize(action = "infer", resource = "model", identifier_field = "model")]
    fn handle_generate(
        &self,
        ctx: &EnvelopeContext,
        request: GenerateRequest,
    ) -> Result<Vec<u8>> {
        // Authorization already validated:
        // 1. JWT signature verified
        // 2. Casbin policy checked
        // 3. JWT scopes validated

        // Just implement business logic
        let user_claims = ctx.user_claims.as_ref().unwrap();
        info!(user = %user_claims.sub, model = %request.model, "Inference authorized");
        // ...
    }
}
```

The `#[authorize]` macro generates code that:
1. Validates JWT token via `ctx.validate_jwt()` (stateless signature verification)
2. Checks Casbin policy for the user
3. Validates JWT scopes match required scope
4. Rejects request if any check fails

#### EnvelopeContext Extensions

```rust
pub struct EnvelopeContext {
    pub request_id: u64,
    pub identity: RequestIdentity,
    // ... existing fields ...

    jwt_token: Option<String>,              // Signed JWT token string
    user_claims: Option<Arc<Claims>>,       // Validated claims (lazy-initialized)
}

impl EnvelopeContext {
    /// Validate JWT token and extract Claims (stateless)
    pub fn validate_jwt(&mut self, verifying_key: &VerifyingKey) -> Result<Arc<Claims>>;

    /// Get validated user claims (None if not yet validated)
    pub fn user_claims(&self) -> Option<&Arc<Claims>>;

    /// Get user subject for Casbin checks (if JWT validated)
    pub fn user_subject(&self) -> Option<String>;

    /// Get effective subject (user if present, otherwise service identity)
    pub fn effective_subject(&self) -> String;
}
```

#### Security Properties

| Property | Implementation | Benefit |
|----------|----------------|---------|
| **Stateless** | JWT signature verification only | Zero server-side state, horizontal scaling |
| **No caching** | Validates every request (~200µs) | Eliminates cache vulnerabilities |
| **Fail-secure** | Empty scopes deny all | No privilege escalation |
| **Action isolation** | Structured scopes | `read:model:*` ≠ `write:model:*` |
| **Resource isolation** | Structured scopes | `infer:model:*` ≠ `infer:stream:*` |
| **Deterministic signatures** | Canonical Cap'n Proto | Cross-platform, version-stable |
| **Defense-in-depth** | 3 validation layers | Envelope + Casbin + JWT scopes |

#### Streaming Authorization

Streaming uses a pre-authorization model where InferenceService authorizes streams before clients subscribe. This eliminates JWT handling at the StreamService level.

```
InferenceService                StreamService                     Client
      │                              │                              │
      │  AUTHORIZE|stream-uuid|exp   │                              │
  ───►│──────────(PUSH)─────────────►│ Creates StreamState          │
      │                              │ {exp, subscribed:false}      │
      │                              │                              │
      │  stream-uuid.chunk.0         │                              │
  ───►│──────────(PUSH)─────────────►│ Queues message               │
      │                              │                              │
      │                              │◄────\x01stream-uuid──────────│
      │                              │ Checks authorized? ✓         │
      │                              │ Flushes queued messages      │
      │                              │────chunk.0──────────────────►│
```

**Key Points:**
- Server generates stream UUID (not client)
- AUTHORIZE message includes claims expiry for automatic GC
- Client subscribes with just `stream-{uuid}` (no JWT in subscription)
- StreamService removes entry on unsubscribe (0x00) to prevent memory leaks
- Periodic compact() removes entries when `claims.exp` passes

#### Complete Documentation

For full implementation details, security analysis, and architectural decisions:

- **Main Plan:** `/home/birdetta/.claude/plans/pure-moseying-wand.md` (2,305 lines)
- **Security Hardening:** `docs/security-hardening-scopes-validation.md` (structured scopes)
- **Canonical Serialization Fix:** `docs/canonical-serialization-security-fix.md` (CRITICAL)
- **Documentation Index:** `docs/JWT-AUTHORIZATION-INDEX.md` (navigation hub)

## Service Spawning Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SERVICE SPAWNING MODES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ServiceSpawner                                                             │
│       │                                                                     │
│       ├──► TOKIO MODE ──────────────────────────────────────────────────►  │
│       │    tokio::spawn(async { service.run_blocking(shutdown) })          │
│       │    ┌──────────────────────────────────────────────┐                │
│       │    │ SpawnedService { ServiceKind::TokioTask }    │                │
│       │    │   └── ServiceHandle (JoinHandle + Notify)    │                │
│       │    └──────────────────────────────────────────────┘                │
│       │                                                                     │
│       ├──► THREAD MODE ─────────────────────────────────────────────────►  │
│       │    std::thread::spawn() + single-threaded tokio runtime            │
│       │    ┌──────────────────────────────────────────────┐                │
│       │    │ SpawnedService { ServiceKind::Thread }       │                │
│       │    │   └── JoinHandle + Arc<Notify>               │                │
│       │    │   USE: !Send types (tch-rs tensors)          │                │
│       │    └──────────────────────────────────────────────┘                │
│       │                                                                     │
│       └──► SUBPROCESS MODE ─────────────────────────────────────────────►  │
│            ProcessSpawner (auto-detects backend)                           │
│            ┌──────────────────────────────────────────────┐                │
│            │         ┌─────────────┐ ┌─────────────┐      │                │
│            │         │ Standalone  │ │ Systemd     │      │                │
│            │         │ Backend     │ │ Backend     │      │                │
│            │         │ (tokio      │ │ (systemd-   │      │                │
│            │         │  process)   │ │  run)       │      │                │
│            │         └─────────────┘ └─────────────┘      │                │
│            │ SpawnedService { ServiceKind::Subprocess }   │                │
│            │   └── PID file + SIGTERM/SIGKILL             │                │
│            └──────────────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Process Spawner Backends

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROCESS SPAWNER BACKENDS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ProcessSpawner::new()                                                      │
│       │                                                                     │
│       ├─► Check: /run/systemd/system exists?                               │
│       │         OR NOTIFY_SOCKET env var?                                  │
│       │         AND which("systemd-run")?                                  │
│       │                                                                     │
│       ├─► YES ──► SystemdBackend                                           │
│       │           ┌───────────────────────────────────────────────┐        │
│       │           │ • Uses systemd-run for transient units        │        │
│       │           │ • Unit: hyprstream-{name}-{uuid}.service      │        │
│       │           │ • Slice: hyprstream-workers.slice             │        │
│       │           │ • Resource limits: MemoryMax, CPUQuota        │        │
│       │           │ • Auto-cleanup: CollectMode=inactive-or-failed│        │
│       │           │ • Stop: systemctl stop <unit>                 │        │
│       │           │ • Status: systemctl is-active --quiet         │        │
│       │           └───────────────────────────────────────────────┘        │
│       │                                                                     │
│       └─► NO ───► StandaloneBackend                                        │
│                   ┌───────────────────────────────────────────────┐        │
│                   │ • Uses tokio::process::Command                │        │
│                   │ • Tracks in DashMap<id, Child>                │        │
│                   │ • kill_on_drop(true) for cleanup              │        │
│                   │ • Stop: SIGTERM → wait 5s → kill_on_drop      │        │
│                   │ • Status: signal 0 (kill with None)           │        │
│                   └───────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Service Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SERVICE LIFECYCLE STATES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────┐                                              │
│                    │ CREATED │                                              │
│                    └────┬────┘                                              │
│                         │ spawn()                                           │
│                         ▼                                                   │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                   SPAWNING                        │                  │
│      │  • Create Notify shutdown signal                  │                  │
│      │  • Register with EndpointRegistry                 │                  │
│      │  • Start task/thread/process                      │                  │
│      │  • Wait for ready signal (socket bound)           │                  │
│      └────────────────────┬─────────────────────────────┘                  │
│                           │ ready signal received                           │
│                           ▼                                                 │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                    ACTIVE                         │                  │
│      │  • Accepting requests (REP socket)                │                  │
│      │  • Publishing events (PUB socket)                 │                  │
│      │  • is_running() == true                           │                  │
│      └────────────────────┬─────────────────────────────┘                  │
│                           │ stop() called                                   │
│                           ▼                                                 │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                   STOPPING                        │                  │
│      │  Tokio:      shutdown.notify_one() → task.await   │                  │
│      │  Thread:     shutdown.notify_one() → join()       │                  │
│      │  Subprocess: SIGTERM → wait → SIGKILL (if stuck)  │                  │
│      └────────────────────┬─────────────────────────────┘                  │
│                           │                                                 │
│                           ▼                                                 │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                   STOPPED                         │                  │
│      │  • is_running() == false                          │                  │
│      │  • ServiceRegistration dropped (auto-unregister)  │                  │
│      │  • PID file cleaned up (subprocess)               │                  │
│      └──────────────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ZMQ Socket Types

| Socket Type | Location | Purpose |
|-------------|----------|---------|
| **REQ/REP** | `hyprstream-rpc/src/service/zmq.rs` | Synchronous RPC calls |
| **PUSH/PULL** | `hyprstream-rpc/src/service/streaming.rs` | Inference → StreamService (guaranteed delivery) |
| **XPUB/SUB** | `hyprstream-rpc/src/service/streaming.rs` | StreamService → Client (topic-based delivery) |
| **XSUB/XPUB** | `hyprstream-rpc/src/service/spawner/service.rs` | Steerable proxy for events |
| **PUB/SUB** | `hyprstream-workers/src/events/` | Topic-based event streaming |
| **DEALER/ROUTER** | `hyprstream/src/services/inference.rs` | Async callback mode |
| **PAIR** | `hyprstream-rpc/src/service/spawner/service.rs` | Proxy shutdown control |

## Streaming Architecture (Inference)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PUSH/PULL STREAMING WITH PRE-AUTHORIZATION               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client              ModelService       InferenceService      StreamService │
│    │                      │                    │                    │       │
│    │  1. infer_stream     │                    │                    │       │
│    ├─────REQ─────────────►│                    │                    │       │
│    │                      │───forward──────────►│                    │       │
│    │                      │                    │                    │       │
│    │                      │                    │  2. Pre-authorize  │       │
│    │                      │                    │  AUTHORIZE|uuid|exp│       │
│    │                      │                    ├────────PUSH───────►│       │
│    │                      │                    │                    │       │
│    │◄────REP: {stream_id, endpoint}────────────┤                    │       │
│    │                      │                    │                    │       │
│    │  3. Subscribe        │                    │                    │       │
│    ├──────────────SUB: "stream-{uuid}"─────────────────────────────►│       │
│    │                      │                    │                    │       │
│    │                      │                    │  (generates tokens)│       │
│    │                      │                    │  stream-uuid.chunk │       │
│    │                      │                    ├────────PUSH───────►│       │
│    │                      │                    │                    │       │
│    │◄─────────────────────XPUB: chunk──────────────────────────────┤       │
│    │◄─────────────────────XPUB: chunk──────────────────────────────┤       │
│    │◄─────────────────────XPUB: StreamComplete─────────────────────┤       │
│    │                      │                    │                    │       │
│    │  4. Unsubscribe      │                    │                    │       │
│    ├──────────────\x00stream-{uuid}────────────────────────────────►│       │
│    │                      │                    │        (removes    │       │
│    │                      │                    │         entry)     │       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why PUSH/PULL Instead of PUB/XSUB

PUB/SUB drops messages when no subscriber exists - causing a race condition where early
chunks are lost before the client subscribes. PUSH/PULL solves this:

| Pattern | Behavior | Use Case |
|---------|----------|----------|
| PUB/SUB | Drops if no subscriber | Broadcast (lossy OK) |
| PUSH/PULL | Buffers at HWM | Work queue (guaranteed) |

StreamService uses PULL to receive from publishers, queues per-topic, then delivers via XPUB.

### Memory Management

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| Unsubscribe (0x00) | Client disconnects | Entry removed immediately |
| Claims expiry | `now > claims.exp` | Entry removed in compact() |
| Message TTL | Message age > 30s | Individual message dropped |
| Per-topic limit | Queue > 1000 msgs | Oldest message dropped |

## Event Bus Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EVENT BUS (XPUB/XSUB PROXY)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PUBLISHERS                  PROXY                    SUBSCRIBERS           │
│  ┌─────────────┐        ┌──────────────┐         ┌─────────────┐           │
│  │WorkerService│──PUB──►│              │──SUB───►│WorkflowSvc  │           │
│  └─────────────┘        │              │         └─────────────┘           │
│  ┌─────────────┐        │ ProxyService │         ┌─────────────┐           │
│  │RegistrySvc  │──PUB──►│              │──SUB───►│  Clients    │           │
│  └─────────────┘        │  (XSUB/XPUB) │         └─────────────┘           │
│  ┌─────────────┐        │              │         ┌─────────────┐           │
│  │InferenceSvc │──PUB──►│              │──SUB───►│  Monitors   │           │
│  └─────────────┘        └──────────────┘         └─────────────┘           │
│                                                                             │
│  Topic Format: {source}.{entity}.{event}                                    │
│  Example: "worker.sandbox-123.started"                                      │
│                                                                             │
│  Subscription Patterns (prefix match):                                      │
│    "worker."              → All worker events                               │
│    "worker.sandbox-123."  → Events for specific sandbox                     │
│    ""                     → All events                                      │
│                                                                             │
│  Transport Modes:                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │   INPROC    │  │     IPC     │  │  SYSTEMD FD │                        │
│  │ (same proc) │  │ (Unix sock) │  │ (activated) │                        │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete Service Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYPRSTREAM SERVICE TOPOLOGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                     MAIN PROCESS                                   │     │
│  │                                                                    │     │
│  │  ┌────────────────┐                                               │     │
│  │  │ EndpointRegistry│◄───── Service discovery (global singleton)   │     │
│  │  └───────┬────────┘                                               │     │
│  │          │                                                        │     │
│  │  ┌───────┴───────────────────────────────────────────────────┐   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │RegistrySvc  │  │ PolicySvc   │  │ ModelSvc    │       │   │     │
│  │  │  │ (Tokio)     │  │ (Tokio)     │  │ (Thread)    │       │   │     │
│  │  │  │ REQ/REP     │  │ REQ/REP     │  │ REQ/REP     │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │InferenceSvc │  │ WorkerSvc   │  │ EventProxy  │       │   │     │
│  │  │  │ (Thread)    │  │ (Tokio)     │  │ (Thread)    │       │   │     │
│  │  │  │ REQ/REP     │  │ REQ/REP     │  │ XPUB/XSUB   │       │   │     │
│  │  │  │ +XPUB       │  │ +PUB        │  │             │       │   │     │
│  │  │  │ (streaming) │  │ (events)    │  │             │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  └───────────────────────────────────────────────────────────┘   │     │
│  │                                                                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│           ┌──────────────────┼──────────────────┐                          │
│           │                  │                  │                          │
│           ▼                  ▼                  ▼                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │ SUBPROCESS     │  │ SUBPROCESS     │  │ SUBPROCESS     │               │
│  │ (Isolated)     │  │ (GPU Worker)   │  │ (Kata VM)      │               │
│  │                │  │                │  │                │               │
│  │ InferenceSvc   │  │ InferenceSvc   │  │ Container      │               │
│  │ (callback mode)│  │ (model loaded) │  │                │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `hyprstream-rpc/src/service/zmq.rs` | Core ZMQ service infrastructure |
| `hyprstream-rpc/src/service/spawner/service.rs` | Unified service spawner |
| `hyprstream-rpc/src/service/streaming.rs` | PUSH/PULL→XPUB streaming proxy with claims-based expiry |
| `hyprstream-rpc/src/envelope.rs` | Signed envelope types |
| `hyprstream-rpc/src/crypto/signing.rs` | Ed25519 signing |
| `hyprstream/src/services/inference.rs` | InferenceService with AUTHORIZE pre-authorization |
| `hyprstream-workers/src/events/publisher.rs` | Event publishing |
| `hyprstream-workers/src/events/subscriber.rs` | Event subscription |
