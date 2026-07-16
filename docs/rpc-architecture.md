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
│  │  │ moq-lite (MoqStreamOrigin + MoqEventOrigin)            │   │         │
│  │  │ stream UDS: /tmp/hyprstream-{pid}/moq.sock             │   │         │
│  │  │ event UDS:  event.sock in the runtime dir              │   │         │
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
│  │    + authorization              │                 │                    │
│  │    + nonce      │                │                 │                    │
│  │         │       │                │                 │                    │
│  │ 3. Sign (COSE   │                │                 │                    │
│  │    composite)   │                │                 │                    │
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
| **iroh** | P2P / NAT-traversal | iroh substrate (Ed25519 carrier address; no application identity authority) |

All transports carry the same `SignedEnvelope`-wrapped Cap'n Proto frames via ZMTP 3.1 framing (the wire serialization only — a holdover from the retired ZeroMQ stack; no libzmq is involved). Only the wire transport layer differs; the application security model is identical.

## Security Model

### Envelope Security Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENVELOPE SECURITY LAYERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SignedEnvelope                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  cose: Vec<u8>          ← COSE composite signature (detached):  │       │
│  │                           EdDSA entry (Classical) or            │       │
│  │                           EdDSA + ML-DSA-65 entries (Hybrid)    │       │
│  │  sig: [u8; 64]          ← raw Ed25519 (compat + cnf binding)    │       │
│  │  cnf: [u8; 32]          ← signer's Ed25519 public key           │       │
│  │  encrypted_envelope     ← Option: AES-256-GCM-SIV ciphertext    │       │
│  │  client_ephemeral_public← X25519 ephemeral (encrypted mode)     │       │
│  │  pq_kem_ciphertext      ← Option: ML-KEM-768 (hybrid encryption)│       │
│  │  ┌───────────────────────────────────────────────────────────┐  │       │
│  │  │ RequestEnvelope (signed as serialized bytes)              │  │       │
│  │  │   request_id: u64                                         │  │       │
│  │  │   payload: Vec<u8>    ← Actual request (Cap'n Proto)      │  │       │
│  │  │   iat: i64            ← Expiration check                  │  │       │
│  │  │   nonce: [u8; 16]     ← Replay protection                 │  │       │
│  │  │   authorization       ← union: none/local/federated/idJag │  │       │
│  │  │   delegation_token    ← Optional relayed delegation token │  │       │
│  │  │   wth                 ← Optional SHA-256 WIT binding      │  │       │
│  │  │   client_dh_public    ← Stream HMAC key derivation        │  │       │
│  │  └───────────────────────────────────────────────────────────┘  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Identity → Subject (bare username, empty = anonymous)                      │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  key-derived subject  ← from the verified envelope signer key   │       │
│  │                         (system/inproc callers)                 │       │
│  │  JWT-derived subject  ← from verified token claims:             │       │
│  │    local token        → "alice"                                 │       │
│  │    federated token    → "https://node-a:alice"                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The authoritative authentication mechanism is the `cose` field: a CBOR-encoded
COSE composite signature over the canonical envelope bytes. In Classical mode it
carries one EdDSA entry; in Hybrid mode it carries a nested composite — an inner
EdDSA `COSE_Sign1` plus an outer ML-DSA-65 (FIPS 204) signature over
`signing-data ‖ inner_sig`. The ML-DSA-65 verifying key is never carried in the
envelope; it is resolved by kid from a trust store keyed by the signer's Ed25519
`cnf`, and the outer signature is enforced only for signer identities whose PQ
key is anchored out-of-band (per-identity PQ anchoring — see
[cryptography-architecture.md](cryptography-architecture.md)). The raw `sig`/`cnf`
fields remain populated for the JWT `cnf` key-binding path.

### Security Layers

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Transport** | TLS 1.3 (QUIC) / UDS peer credentials (IPC) | Wire confidentiality and peer identity |
| **Application** | COSE hybrid signatures (inner EdDSA + outer ML-DSA-65, per-identity PQ anchoring) | Request authentication and integrity |
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
│    │   authorization: <JWT>      │                            │             │
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
Layer 1: COSE envelope signature (service identity)
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
  requestId @0 :UInt64;            # Unique request ID for correlation
  payload @1 :Data;                # Serialized inner request (RegistryRequest, etc.)
  iat @2 :Int64;                   # Unix millis, for expiration check
  nonce @3 :Data $fixedSize(16);   # 16 random bytes for replay protection
  authorization @4 :Authorization; # Authorization context (union)
  delegationToken @5 :Text $optional;  # Delegation token relayed by a trusted service
  wth @6 :Data $fixedSize(32) $optional;  # SHA-256 of the WIT JWT (WIMSE binding)
  clientDhPublic @7 :Data $fixedSize(32) $optional;  # Ephemeral DH public key for stream keys
}
```

The `Authorization` union replaces the legacy `identity`/`jwtToken` fields; it is
`none`, `local` (verified `TokenClaims`), `federated` (raw token from a foreign
issuer + claims), or `idJag` (identity-assertion JWT).

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

        let claims = ctx.claims().unwrap();
        info!(user = %claims.sub, model = %request.model, "Inference authorized");
        // ...
    }
}
```

#### EnvelopeContext

`EnvelopeContext` (`crates/hyprstream-rpc/src/service/svc.rs`) is constructed from
a verified `SignedEnvelope` and passed to every handler:

```rust
pub struct EnvelopeContext {
    pub request_id: u64,          // Unique request ID for correlation
    pub cnf: [u8; 32],            // Verified Ed25519 signer key (RFC 7800 cnf)

    claims: Option<Claims>,       // Validated JWT claims (set by verify_claims())
    jwt_token: Option<String>,    // Raw JWT from the envelope's authorization union
    key_derived_subject: Subject, // Subject from the verified signer key (system/inproc)
    jwt_subject: Option<Subject>, // Subject from a verified JWT (local or federated)
}
```

The server dispatch loop runs `verify_claims()` on the context after envelope
verification: it decodes and verifies the JWT, populates `claims`, and resolves
`jwt_subject` (bare `"alice"` for local tokens, `"https://node-a:alice"` for
federated ones). Handlers then use accessors:

```rust
impl EnvelopeContext {
    /// Authorization subject: key-derived subject if present (cryptographically
    /// proven via the signer key), else the verified JWT subject, else anonymous
    pub fn subject(&self) -> Subject;

    /// Validated JWT claims (None until verify_claims() runs)
    pub fn claims(&self) -> Option<&Claims>;

    /// Raw JWT token from the envelope (if present)
    pub fn jwt_token(&self) -> Option<&str>;

    /// Whether a verified user identity is present
    pub fn is_authenticated(&self) -> bool;
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
│  │  │ EndpointRegistry│  │ moq-lite planes (process-global)    │     │     │
│  │  │ (global singl.) │  │ • Streaming plane (MoqStreamOrigin) │     │     │
│  │  └───────┬────────┘  │ • Event bus (MoqEventOrigin)         │     │     │
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
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │DiscoverySvc │  │ MetricsSvc  │  │ McpSvc      │       │   │     │
│  │  │  │ inproc/UDS  │  │ inproc/UDS  │  │ inproc/UDS  │       │   │     │
│  │  │  │             │  │             │  │ +HTTP/SSE   │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────────────────────────┐    │   │     │
│  │  │  │ TuiSvc      │  │ event / streams factories       │    │   │     │
│  │  │  │ inproc/UDS  │  │ (initialize the MoQ event and   │    │   │     │
│  │  │  │             │  │  stream planes for the process) │    │   │     │
│  │  │  └─────────────┘  └─────────────────────────────────┘    │   │     │
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
| `hyprstream-rpc/src/crypto/cose_sign.rs` | COSE composite (EdDSA + ML-DSA-65) sign/verify |
| `hyprstream-rpc/src/moq_stream.rs` | `MoqStreamHandle`, moq streaming plane |
| `hyprstream-rpc/src/moq_event.rs` | `MoqEventOrigin`, moq event bus |
| `hyprstream-rpc/src/registry/mod.rs` | `EndpointRegistry`, `SocketKind`, `EndpointMode` |
| `hyprstream-workers/src/events/{mod,service,token_manager,types}.rs` | Worker event integration (service, token management, event types) |
