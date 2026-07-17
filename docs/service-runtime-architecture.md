# Service Runtime Architecture

Production registry/Discovery trust is provisioned through the fixed OS-owned
startup seam documented in
[`deployment-registry-trust.md`](deployment-registry-trust.md). Service-manager
credential directories and user configuration remain suitable for ordinary
service secrets, but they do not select the registry/PDS authority root.

This document describes the threading model, async patterns, and `Send`/`Sync` constraints for hyprstream services.

## Overview

Hyprstream services run in isolated threads, each with their own tokio runtime. The choice of runtime type depends on whether the service holds `!Send` types (like GPU tensors).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SERVICE RUNTIME PATTERNS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Multi-threaded (Send + Sync)          Single-threaded (!Send)              │
│  ┌─────────────────────────┐           ┌─────────────────────────┐         │
│  │ new_multi_thread()      │           │ new_current_thread()    │         │
│  │                         │           │ + LocalSet              │         │
│  │ ┌─────┐ ┌─────┐ ┌─────┐│           │ ┌─────────────────────┐ │         │
│  │ │ T1  │ │ T2  │ │ T3  ││           │ │ Single thread       │ │         │
│  │ └─────┘ └─────┘ └─────┘│           │ │ (spawn_local)       │ │         │
│  │                         │           │ └─────────────────────┘ │         │
│  │ HTTP servers            │           │ GPU inference           │         │
│  │ gRPC (Flight)           │           │ InferenceService        │         │
│  │ RPC services            │           │                         │         │
│  └─────────────────────────┘           └─────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Service Categories

### Multi-threaded Services (no LocalSet)

These services use `tokio::runtime::Builder::new_multi_thread()` and can leverage multiple CPU cores:

| Service | Protocol | Why Multi-threaded |
|---------|----------|-------------------|
| OAIService | HTTP/HTTPS | Axum server handles concurrent requests |
| FlightService | gRPC | Arrow Flight uses tonic (multi-threaded) |
| RegistryService | UDS RPC | Serves via `serve_bridged()` — async, no !Send types |
| PolicyService | UDS RPC | Same: `serve_bridged()` + generated client handles |
| ModelService | UDS RPC | Same; GPU tensors live in InferenceService subprocess |

### Single-threaded Services (LocalSet required)

These services use `new_current_thread()` + `LocalSet` because they hold `!Send` types:

| Service | !Send Type | Why |
|---------|-----------|-----|
| InferenceService | GPU tensors (tch-rs) | CUDA contexts are thread-bound |
| OAuthService | RPC in handlers | HTTP handlers call `policy_client.issue_token()` |

## Client vs Service Constraints

**Important distinction**: The `spawn_local` constraint applies to the **service side** (when a handler holds `!Send` state), not the client side.

### RPC Client (Send + Sync)

Generated clients are designed for concurrent use across threads:

```rust
// All generated clients wrap Arc<dyn RpcClient>
pub struct PolicyClient {
    inner: Arc<dyn RpcClient>,
}

// RpcClientImpl<Signer, LazyUdsTransport> is Send+Sync:
// - LazyUdsTransport holds a tokio::sync::Mutex<Option<UdsSession>>
// - Signer holds an Arc<SigningKey>
```

Generated clients like `PolicyClient` wrap `Arc<dyn RpcClient>` — they're `Clone`, `Send`, and `Sync`. Multiple threads can share one client instance.

### Service Side

The `spawn_local` requirement comes from services that hold `!Send` state. `serve_bridged()` handles this:

```rust
// serve_bridged() accepts a Box<dyn RequestService> (which may be !Send)
// and runs the dispatch loop on whatever runtime context calls it.
// For !Send services, wrap in LocalSet:
let local = tokio::task::LocalSet::new();
local.run_until(serve_bridged(transport, service, shutdown)).await?;
```

The `RequestService` trait allows `?Send` implementations (e.g., GPU tensors):

```rust
#[async_trait(?Send)]
pub trait RequestService {
    async fn handle(&self, ctx: EnvelopeContext, payload: Vec<u8>) -> Result<Vec<u8>>;
}
```

## Factory Construction

Service factories run synchronously on the main thread. When a factory needs to call async code (like RPC), it must use `block_in_place`:

```rust
// Safe: no RPC calls, just local initialization
let policy_client = PolicyClient::for_service(sk, policy_vk, token);

// Requires LocalSet wrapper if async code makes RPC calls during construction
let model_service = tokio::task::block_in_place(|| {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, ModelService::new(...))  // Makes RPC calls
})?;
```

### When LocalSet is Required in Factories

| Operation | LocalSet Needed? | Reason |
|-----------|-----------------|--------|
| Creating a client (`XxxClient::for_service`) | No | Just struct construction |
| Calling client methods (`.load()`, `.resolve_service_key()`) | Yes | Uses `spawn_local` |
| Local file/DB operations | No | No async RPC involved |
| `tokio::spawn()` | No | Regular async spawn |

## Deferred Initialization Pattern

For operations that need LocalSet but shouldn't block factory construction, defer them to `Spawnable::run()`:

```rust
impl Spawnable for OAIService {
    fn run(self: Box<Self>, shutdown: Arc<Notify>, on_ready: Option<...>) -> Result<...> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        rt.block_on(async move {
            // Signal ready immediately (don't block on slow operations)
            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Spawn deferred work with its own LocalSet
            let server_state = self.server_state.clone();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create runtime");
                let local = tokio::task::LocalSet::new();
                local.block_on(&rt, server_state.preload_configured_models());
            });

            // ... run main service loop ...
        })
    }
}
```

## Service Dependency Graph

The `#[service_factory]` macro supports `depends_on` to ensure services start in order:

```rust
#[service_factory("model", depends_on = ["policy", "registry", "discovery"])]
fn create_model_service(ctx: &ServiceContext) -> Result<Box<dyn Spawnable>> {
    // PolicyService, RegistryService, and DiscoveryService are guaranteed
    // to be running before this factory executes
}
```

This is critical when factories make RPC calls during construction — the target service must be running.

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
│      │  • Wait for ready signal (transport bound)        │                  │
│      └────────────────────┬─────────────────────────────┘                  │
│                           │ ready signal received                           │
│                           ▼                                                 │
│      ┌──────────────────────────────────────────────────┐                  │
│      │                    ACTIVE                         │                  │
│      │  • Accepting requests (UDS / QUIC / iroh)        │                  │
│      │  • Publishing events (moq-lite)                   │                  │
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

## RPC Client Architecture

All generated clients share the same structure:

```
┌─────────────────────────────────────────────────────────────┐
│  PolicyClient / RegistryClient / etc.                       │
│  └── Arc<dyn RpcClient>  (Clone, Send+Sync)                │
│       └── RpcClientImpl<Signer, Transport>                  │
│            ├── signer: Ed25519 signing key                  │
│            ├── transport: LazyUdsTransport /                │
│            │     LazyQuinnTransport (QUIC) / iroh substrate │
│            │     └── tokio::sync::Mutex<Option<Session>>    │
│            ├── request_id: AtomicU64                        │
│            └── default_jwt: Option<String>                  │
└─────────────────────────────────────────────────────────────┘
```

- **Pooling primitive**: `Arc::clone()` — share one client across handlers
- **Thread safety**: each lazy transport wraps its session in `tokio::Mutex`
- **Per-call auth**: `RequestBuilder` allows JWT override without mutation
- **No LocalSet on client side**: The constraint is on the service side only

The transport is chosen by `TransportConfig` (inproc, UDS, QUIC, iroh); all backends follow the same `Send+Sync` pattern via `Mutex`-wrapped session state.

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
│  │  │ (global singl.) │  │ • MoqStreamOrigin — streaming plane │     │     │
│  │  └───────┬────────┘  │   (cross-process UDS:               │     │     │
│  │          │            │    /tmp/hyprstream-{pid}/moq.sock)  │     │     │
│  │          │            │ • MoqEventOrigin — event bus        │     │     │
│  │          │            │   (cross-process UDS: event.sock    │     │     │
│  │          │            │    in the runtime dir)              │     │     │
│  │          │            └──────────────────────────────────────┘     │     │
│  │  ┌───────┴───────────────────────────────────────────────────┐   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │RegistrySvc  │  │ PolicySvc   │  │ ModelSvc    │       │   │     │
│  │  │  │ (Tokio)     │  │ (Tokio)     │  │ (Thread)    │       │   │     │
│  │  │  │ UDS/inproc  │  │ UDS/inproc  │  │ UDS/inproc  │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │InferenceSvc │  │ WorkerSvc   │  │ DiscoverySvc│       │   │     │
│  │  │  │ (Thread)    │  │ (Tokio)     │  │ UDS/inproc  │       │   │     │
│  │  │  │ UDS/inproc  │  │ UDS/inproc  │  │             │       │   │     │
│  │  │  │ +moq stream │  │ +moq events │  │             │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │     │
│  │  │  │ MetricsSvc  │  │ McpSvc      │  │ TuiSvc      │       │   │     │
│  │  │  │ UDS/inproc  │  │ UDS/inproc  │  │ UDS/inproc  │       │   │     │
│  │  │  │             │  │ +HTTP/SSE   │  │             │       │   │     │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘       │   │     │
│  │  │                                                           │   │     │
│  │  │  ┌─────────────────────────────────────────────────┐     │   │     │
│  │  │  │ event / streams factories — initialize the      │     │   │     │
│  │  │  │ MoqEventOrigin / MoqStreamOrigin planes above   │     │   │     │
│  │  │  └─────────────────────────────────────────────────┘     │   │     │
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

## Debugging Runtime Issues

### "there is no reactor running" / "spawn_local called outside LocalSet"

**Cause**: Async code using `spawn_local` running outside a LocalSet context.

**Fix**: Wrap the async block in a LocalSet:
```rust
let local = tokio::task::LocalSet::new();
local.block_on(&rt, async { ... });
```

### "Cannot resolve registry pubkey"

**Cause**: RPC client trying to resolve service key before the target service is running.

**Fix**: Add the dependency to `depends_on` in the factory macro.

### Service hangs during startup

**Cause**: Circular dependency or deadlock in factory construction.

**Debug**: Check `depends_on` graph for cycles. Ensure factories don't block waiting for services that haven't started.

## Summary

| Context | Runtime | LocalSet | Can Make RPC |
|---------|---------|----------|--------------|
| Factory construction | Main thread | No (unless wrapped) | Only with wrapper |
| `Spawnable::run()` multi-thread | `new_multi_thread()` | No | Spawn dedicated thread |
| `Spawnable::run()` single-thread | `new_current_thread()` | Yes | Yes |
| `serve_bridged()` dispatch loop | Caller's runtime | Caller's | Yes |
| HTTP handlers (OAI, Flight) | Multi-thread pool | No | Spawn dedicated thread |
| HTTP handlers (OAuth) | Single thread (LocalSet) | Yes | Yes |
