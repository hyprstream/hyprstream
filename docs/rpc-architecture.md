# RPC Architecture

This document describes the RPC infrastructure used by hyprstream for inter-service communication.

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
│  │                    TRANSPORT PLANE                             │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │         │
│  │  │  inproc     │  │  UDS (ipc)  │  │ QUIC / iroh         │   │         │
│  │  │ (same proc) │  │ (same host) │  │ (remote / P2P)      │   │         │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │                  STREAMING / EVENT PLANE                       │         │
│  │  ┌────────────────────────────────────────────────────────┐   │         │
│  │  │ moq-lite (MoqEventOrigin + MoqStreamHandle)            │   │         │
│  │  │ UDS socket: /tmp/hyprstream-{pid}/moq.sock             │   │         │
│  │  └────────────────────────────────────────────────────────┘   │         │
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
│  │ Generated Client│                │ serve_bridged() │                    │
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
│  │         │       │   UDS frame   │                 │                    │
│  │         └───────┼───────────────►│ 4. Receive     │                    │
│  │                 │                │ 5. Verify sig  │                    │
│  │                 │                │ 6. Check nonce │                    │
│  │                 │                │ 7. Dispatch    │                    │
│  │                 │                │    handler()   │                    │
│  │                 │   UDS frame   │ 8. Respond     │                    │
│  │ 9. Parse resp  ◄├───────────────┤                 │                    │
│  └─────────────────┘                └─────────────────┘                    │
│                                                                             │
│  Transport varies: inproc channel (same process), UDS (same host),         │
│  QUIC/TLS (remote), iroh (P2P). The signed envelope is transport-agnostic. │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Transport Types

| Transport | Use Case | Implementation |
|-----------|----------|----------------|
| **Inproc** | Same process, test harnesses | `InprocChannel` (in-memory) |
| **UDS / ipc** | Same host (default) | `LazyUdsTransport` over Unix domain sockets |
| **QUIC** | Remote, TLS 1.3 | `LazyQuinnTransport` (ALPN: `ALPN_HYPRSTREAM_RPC`) |
| **iroh** | P2P / NAT-traversal | iroh substrate (Ed25519 node identity) |

All transports carry the same `SignedEnvelope`-wrapped Cap'n Proto frames via ZMTP framing. Only the wire transport layer differs; the application security model is identical.

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
| **Transport** | TLS 1.3 (QUIC) / UDS peer credentials (IPC) | Wire confidentiality and peer identity |
| **Application** | Ed25519 signatures | Request authentication and integrity |
| **Authorization** | Casbin policy | RBAC/ABAC access control |

### JWT Authorization

Services enforce user-level authorization via JWT tokens embedded in request envelopes. This provides end-to-end user attribution and fine-grained access control.

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
│    │   jwt_token: "eyJ..."       │                            │             │
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
  - `infer:model:qwen-7b` — Run inference on specific model
  - `subscribe:stream:abc-123` — Subscribe to specific stream
  - `read:model:*` — Read any model (explicit wildcard)
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
  requestId      @0 :UInt64;
  identity       @1 :RequestIdentity;
  payload        @2 :Data;
  ephemeralPubkey @3 :Data;
  nonce          @4 :Data;
  timestamp      @5 :Int64;
  jwtToken       @6 :Text;  # Signed JWT token string
}
```

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

        let user_claims = ctx.user_claims.as_ref().unwrap();
        info!(user = %user_claims.sub, model = %request.model, "Inference authorized");
        // ...
    }
}
```

#### EnvelopeContext Extensions

```rust
pub struct EnvelopeContext {
    pub request_id: u64,
    pub identity: RequestIdentity,

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

Streaming uses a pre-authorization model where InferenceService authorizes streams before clients subscribe. Authorization happens at the RPC call site; the moq subscription path requires no additional JWT handling.

```
InferenceService                moq Origin                        Client
      │                              │                              │
      │  infer_stream(ephemeral_pk)  │                              │
  ◄───┤◄─────────────────────────────────────────────────────────── │
      │  ─── return StreamInfo ──────────────────────────────────── ►│
      │       (moq_uds_path,         │                              │
      │        moq_broadcast_path)   │                              │
      │                              │                              │
      │  publish(broadcast_path, ───►│                              │
      │           chunk + HMAC)      │                              │
      │                              │◄────── subscribe ────────────│
      │                              │────── chunk ────────────────►│
      │                              │            [verify HMAC] ────│
```

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
│            │         └─────────────┘ └─────────────┘      │                │
│            │ SpawnedService { ServiceKind::Subprocess }   │                │
│            │   └── PID file + SIGTERM/SIGKILL             │                │
│            └──────────────────────────────────────────────┘                │
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
│      │  • Wait for ready signal (transport bound)        │                  │
│      └────────────────────┬─────────────────────────────┘                  │
│                           │ ready signal received                           │
│                           ▼                                                 │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                    ACTIVE                         │                  │
│      │  • Accepting requests (serve_bridged loop)        │                  │
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

## Complete Service Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYPRSTREAM SERVICE TOPOLOGY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                     MAIN PROCESS                                   │     │
│  │                                                                    │     │
│  │  ┌────────────────┐  ┌──────────────────────────────────────┐     │     │
│  │  │ EndpointRegistry│  │ moq-lite Origin                     │     │     │
│  │  │ (global singl.) │  │ • Streaming plane (MoqStreamHandle) │     │     │
│  │  └───────┬────────┘  │ • Event bus (MoqEventBarrierService) │     │     │
│  │          │            └──────────────────────────────────────┘     │     │
│  │          │                                                         │     │
│  │  ┌───────┴───────────────────────────────────────────────────┐   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │RegistrySvc  │  │ PolicySvc   │  │ ModelSvc    │       │   │     │
│  │  │  │ (Tokio)     │  │ (Tokio)     │  │ (Thread)    │       │   │     │
│  │  │  │ inproc/UDS  │  │ inproc/UDS  │  │ inproc/UDS  │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │InferenceSvc │  │ WorkerSvc   │  │ OAuthSvc    │       │   │     │
│  │  │  │ (Thread)    │  │ (Tokio)     │  │ (Thread)    │       │   │     │
│  │  │  │ inproc/UDS  │  │ inproc/UDS  │  │ HTTP        │       │   │     │
│  │  │  │ +moq stream │  │ +moq events │  │             │       │   │     │
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
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `hyprstream-rpc/src/service/svc.rs` | Core service infrastructure (`RequestService`, `EnvelopeContext`, `ServiceHandle`) |
| `hyprstream-rpc/src/service/serve.rs` | `serve_bridged()` — bridged serve path |
| `hyprstream-rpc/src/service/dispatch.rs` | Transport-agnostic request dispatch |
| `hyprstream-rpc/src/transport/mod.rs` | `TransportConfig` — endpoint configuration |
| `hyprstream-rpc/src/transport/lazy_uds.rs` | `LazyUdsTransport` — UDS client transport |
| `hyprstream-rpc/src/transport/lazy_quinn.rs` | `LazyQuinnTransport` — QUIC client transport |
| `hyprstream-rpc/src/transport/iroh_substrate.rs` | iroh P2P transport substrate |
| `hyprstream-rpc/src/transport/in_memory.rs` | `InprocChannel` — in-process transport |
| `hyprstream-rpc/src/envelope.rs` | Signed envelope types |
| `hyprstream-rpc/src/crypto/signing.rs` | Ed25519 signing |
| `hyprstream-rpc/src/moq_stream.rs` | `MoqStreamHandle`, moq streaming plane |
| `hyprstream-rpc/src/moq_event.rs` | `MoqEventOrigin`, moq event bus |
| `hyprstream-rpc/src/registry/mod.rs` | `EndpointRegistry`, `SocketKind`, `EndpointMode` |
| `hyprstream-workers/src/events/publisher.rs` | Event publishing |
| `hyprstream-workers/src/events/subscriber.rs` | Event subscription |
