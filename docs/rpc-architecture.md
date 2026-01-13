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
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │         │
│  │  │ REQ/REP     │  │ PUB/SUB     │  │ XPUB/XSUB   │           │         │
│  │  │ (RPC calls) │  │ (events)    │  │ (streaming) │           │         │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │         │
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
| **XSUB/XPUB** | `hyprstream-rpc/src/service/spawner/service.rs` | Steerable proxy for events |
| **PUB/SUB** | `hyprstream-workers/src/events/` | Topic-based event streaming |
| **DEALER/ROUTER** | `hyprstream/src/services/inference.rs` | Async callback mode |
| **PAIR** | `hyprstream-rpc/src/service/spawner/service.rs` | Proxy shutdown control |

## Streaming Architecture (Inference)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THREE-PHASE STREAMING PATTERN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client                InferenceService                 XPUB Socket         │
│    │                         │                              │               │
│    │  PHASE 1: Request       │                              │               │
│    ├───REQ: GenerateStream──►│                              │               │
│    │                         │  prepare_stream()            │               │
│    │                         │  → stream_id = "stream-42"   │               │
│    │                         │  → PendingStream             │               │
│    │◄──REP: {stream_id}──────┤                              │               │
│    │      + stream_endpoint  │                              │               │
│    │                         │                              │               │
│    │  PHASE 2: Subscribe     │                              │               │
│    ├─────────SUB: "stream-42"────────────────────────────►  │               │
│    │                         │                              │               │
│    │                         │  PHASE 3: Stream             │               │
│    │                         │  execute_stream(pending)     │               │
│    │                         │  → wait for subscription     │               │
│    │                         │  → generate tokens           │               │
│    │                         │         │                    │               │
│    │◄────────PUB: chunk "Hello"────────┼────────────────────┤               │
│    │◄────────PUB: chunk " world"───────┼────────────────────┤               │
│    │◄────────PUB: chunk "!"────────────┼────────────────────┤               │
│    │◄────────PUB: StreamComplete───────┴────────────────────┤               │
│    │                         │                              │               │
│                                                                             │
│  Topic Format: "stream-{id}"                                                │
│  Multipart: [topic_bytes, payload_bytes]                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
| `hyprstream-rpc/src/envelope.rs` | Signed envelope types |
| `hyprstream-rpc/src/crypto/signing.rs` | Ed25519 signing |
| `hyprstream-workers/src/events/publisher.rs` | Event publishing |
| `hyprstream-workers/src/events/subscriber.rs` | Event subscription |
