# Service Runtime Architecture

This document describes the threading model, async patterns, and `Send`/`Sync` constraints for hyprstream services.

## Overview

Hyprstream services run in isolated threads, each with their own tokio runtime. The choice of runtime type depends on whether the service holds `!Send` types (like ZMQ sockets or GPU tensors).

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
│  │ HTTP servers            │           │ ZMQ services            │         │
│  │ gRPC (Flight)           │           │ GPU inference           │         │
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

### Single-threaded Services (LocalSet required)

These services use `new_current_thread()` + `LocalSet` because they hold `!Send` types or make RPC calls in handlers:

| Service | !Send Type | Why |
|---------|-----------|-----|
| InferenceService | GPU tensors (tch-rs) | CUDA contexts are thread-bound |
| ModelService | ZMQ sockets | libzmq sockets are `!Send` |
| RegistryService | ZMQ sockets | ZMQ client for policy checks |
| PolicyService | ZMQ sockets | ZMQ REP socket |
| OAuthService | RPC in handlers | HTTP handlers call `policy_client.issue_token()` |

## Client vs Service Constraints

**Important distinction**: The `spawn_local` constraint applies to the **service side**, not the client side.

### RPC Client (Send + Sync)

The RPC client is designed for concurrent use:

```rust
pub struct ZmqConnection {
    endpoint: String,
    context: Arc<zmq::Context>,
    sender: tokio::sync::Mutex<Option<RequestSender>>,  // Mutex makes it Send+Sync
}

// Safe because socket is only accessed under the Mutex
unsafe impl Send for ZmqConnection {}
unsafe impl Sync for ZmqConnection {}
```

Generated clients like `PolicyClient` wrap `Arc<dyn RpcClient>` — they're `Clone`, `Send`, and `Sync`. Multiple threads can share one client instance.

### Service Side (spawn_local)

The `spawn_local` requirement comes from:
1. **RequestLoop** — uses `spawn_local` to handle connections with `?Send` services
2. **Continuations** — streaming responses spawn continuations via `spawn_local`
3. **ZmqService trait** — allows `?Send` implementations (e.g., GPU tensors)

## Factory Construction

Service factories run synchronously on the main thread. When a factory needs to call async code (like RPC), it must use `block_in_place`:

```rust
// Safe: no RPC calls, just local initialization
let policy_client = PolicyClient::for_service(sk, policy_vk, token);

// Requires LocalSet wrapper if async code makes RPC calls
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
| Local file/DB operations | No | No ZMQ involved |
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
            // ... setup HTTP server ...

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
#[service_factory("model", depends_on = ["policy", "registry", "discovery", "notification"])]
fn create_model_service(ctx: &ServiceContext) -> Result<Box<dyn Spawnable>> {
    // PolicyService, RegistryService, DiscoveryService, and NotificationService
    // are guaranteed to be running before this factory executes
}
```

This is critical when factories make RPC calls during construction — the target service must be running.

## Future: Channel-based RPC Proxy (Service Side)

For services that need to make RPC calls from multi-threaded HTTP handlers (like OAuthService calling PolicyService), a cleaner architecture would use a dedicated I/O thread with channels:

```
┌─────────────────────┐     mpsc      ┌──────────────────────┐
│  HTTP Handler       │ ──────────▶   │  Dedicated I/O       │
│  (Axum thread pool) │               │  thread + LocalSet   │
│                     │ ◀──────────   │  (owns RPC clients)  │
└─────────────────────┘   oneshot     └──────────────────────┘
```

```rust
struct RpcProxy {
    tx: mpsc::Sender<(Request, oneshot::Sender<Response>)>,
}

impl RpcProxy {
    async fn call(&self, req: Request) -> Result<Response> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx.send((req, resp_tx)).await?;
        resp_rx.await?
    }
}
```

Benefits:
- HTTP handlers don't need LocalSet
- True multi-threading for request handling
- Clean separation of I/O concerns

Tradeoffs:
- Extra channel hop (~microseconds latency)
- More complexity in client wrapper
- Streaming responses need additional plumbing

**Current workaround**: OAuthService runs entirely in a LocalSet (`new_current_thread()`), which works but limits concurrency. The mpsc proxy would allow multi-threaded HTTP handling while keeping RPC on a dedicated thread.

## Debugging Runtime Issues

### "there is no reactor running" / "spawn_local called outside LocalSet"

**Cause**: Async code using `spawn_local` (ZMQ client) running outside a LocalSet context.

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
| ZMQ `RequestLoop` | Single thread | Yes | Yes |
| HTTP handlers (OAI, Flight) | Multi-thread pool | No | Spawn dedicated thread |
| HTTP handlers (OAuth) | Single thread (LocalSet) | Yes | Yes |

## RPC Client Architecture

The RPC client is already designed for connection pooling:

```
┌─────────────────────────────────────────────────────────────┐
│  PolicyClient / RegistryClient / etc.                       │
│  └── Arc<dyn RpcClient>  (Clone, Send+Sync)                │
│       └── RpcClientImpl<Signer, Transport>                  │
│            ├── signer: Ed25519 key                          │
│            ├── transport: ZmqConnection (Send+Sync via Mutex)│
│            ├── request_id: AtomicU64                        │
│            └── default_jwt: Option<String>                  │
└─────────────────────────────────────────────────────────────┘
```

- **Pooling primitive**: `Arc::clone()` — share one client across handlers
- **Thread safety**: `ZmqConnection` wraps socket in `tokio::Mutex`
- **Per-call auth**: `RequestBuilder` allows JWT override without mutation
- **No LocalSet on client side**: The constraint is on the service side only
