# EventService Architecture

## Overview

EventService provides pub/sub event distribution for hyprstream using ZeroMQ's XPUB/XSUB proxy pattern. It enables services to publish lifecycle events that other services can subscribe to, replacing polling-based status checks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Single Host                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────── EventService ───────────────────────────────┐ │
│  │                           (XPUB/XSUB Proxy)                            │ │
│  │                                                                         │ │
│  │  Publishers                                        Subscribers          │ │
│  │  ┌────────────────┐                               ┌────────────────┐   │ │
│  │  │WorkerService   │                               │WorkflowService │   │ │
│  │  │RegistryService │──PUB──► [Proxy] ──SUB──────►│CLI (wait)      │   │ │
│  │  │ModelService    │                               │ThresholdMonitor│   │ │
│  │  │InferenceService│                               └────────────────┘   │ │
│  │  └────────────────┘                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why XPUB/XSUB Proxy?

The proxy provides a **single well-known endpoint** for multi-host extensibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Now (Single Host)                Future (Multi-Host)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Publishers ──► [Proxy] ◄── Subscribers                             │
│       ▲              │                                               │
│  inproc://       inproc://                                          │
│                      │                                               │
│                      ▼                                               │
│               [Bridge to Remote]  ◄──tcp://──► [Remote Proxy]       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Clean extension point for multi-host bridging
- Publishers/subscribers don't change when we add remotes
- Centralized subscription management
- Efficient C-level message forwarding via `zmq::proxy_steerable()`

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

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `inproc://hyprstream/events/pub` | Publishers connect here (XSUB binds) |
| `inproc://hyprstream/events/sub` | Subscribers connect here (XPUB binds) |
| `inproc://hyprstream/events/ctrl` | Control socket for graceful shutdown |

## Delivery Semantics

| Aspect | Behavior |
|--------|----------|
| Ordering | In-order per publisher; no cross-publisher guarantees |
| Delivery | At-most-once (fire-and-forget) |
| Persistence | None - events are ephemeral |
| Late join | Subscribers only see events after subscribing |
| Slow subscriber | Messages dropped at HWM, never slows publisher |

### Hybrid State Pattern

For reliable status checking (e.g., CLI waiting for container):

1. Query current state via RPC first
2. Subscribe to events for updates
3. Handle race condition by checking timestamp in query response

## Components

### EventService

Runs the XPUB/XSUB proxy using `ProxyService` from `hyprstream-rpc`.

**Location:** `crates/hyprstream-workers/src/events/service.rs`

```rust
use hyprstream_workers::events::{endpoints, ProxyService, ServiceSpawner, SpawnedService};

// Create proxy with transport configuration
let ctx = Arc::new(zmq::Context::new());
let (pub_transport, sub_transport) = endpoints::inproc_transports();
let proxy = ProxyService::new("events", ctx.clone(), pub_transport, sub_transport);

// Spawn in dedicated thread
let spawner = ServiceSpawner::threaded();
let service: SpawnedService = spawner.spawn(proxy).await?;

// Later, graceful shutdown
service.stop().await?;
```

**Endpoint Modes:**
- `Inproc` - In-process communication (default)
- `Ipc` - Unix domain sockets for distributed mode
- `SystemdFd` - Systemd socket activation

### EventPublisher

TMQ-based async publisher. Each service creates its own instance.

**Location:** `crates/hyprstream-workers/src/events/publisher.rs`

```rust
let mut publisher = EventPublisher::new(&ctx, "worker")?;
publisher.publish("sandbox123", "started", &payload).await?;
```

### EventSubscriber

TMQ-based async subscriber with topic filtering.

**Location:** `crates/hyprstream-workers/src/events/subscriber.rs`

```rust
let mut subscriber = EventSubscriber::new(&ctx)?;
subscriber.subscribe("worker.")?;  // All worker events

while let Ok((topic, payload)) = subscriber.recv().await {
    println!("Received: {} ({} bytes)", topic, payload.len());
}
```

**Additional Methods:**
- `new_at(ctx, endpoint)` - Connect to explicit endpoint (for distributed mode)
- `subscribe_all()` - Subscribe to all topics
- `recv_timeout(duration)` - Receive with timeout
- `try_recv()` - Non-blocking receive

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

pub struct ReceivedEvent {
    pub topic: String,
    pub source: String,
    pub entity_id: String,
    pub event_type: String,
    pub worker_event: Option<WorkerEvent>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

## Message Format

ZMQ multipart message:
- **Frame 1:** Topic bytes (e.g., `worker.sandbox123.started`)
- **Frame 2:** Payload bytes (Cap'n Proto serialized event)

## Schema Ownership

To avoid circular dependencies, event schemas are distributed:

| Crate | Schema | Contents |
|-------|--------|----------|
| `hyprstream-rpc` | `schema/events.capnp` | `EventEnvelope` only (generic) |
| `hyprstream-workers` | `schema/workers.capnp` | `WorkerEvent` types |
| `hyprstream` | `schema/events.capnp` | `InferenceEvent`, `RegistryEvent`, `ModelEvent` |

### EventEnvelope (hyprstream-rpc)

```capnp
struct EventEnvelope {
  id @0 :Data;              # UUID
  timestamp @1 :Int64;      # Unix ms
  source @2 :Text;          # "worker", "registry", "model", "inference"
  topic @3 :Text;           # For ZMQ filtering
  payload @4 :Data;         # Service-specific bytes
  correlationId @5 :Data;   # Optional tracing ID
}
```

Consumers deserialize `payload` based on topic prefix:
- `worker.*` → `WorkerEvent` from workers schema
- `inference.*` → `InferenceEvent` from main schema

## Separation from MetricsService

EventService and MetricsService are **separate concerns**:

| Service | Pattern | Purpose |
|---------|---------|---------|
| EventService | PUB/SUB | Lifecycle events (broadcast) |
| MetricsService | REQ/REP | Data queries/inserts (RPC) |

MetricsService queries are NOT broadcast to EventService. Future work may add optional CDC for threshold breach events.

## Implemented Features

The following features are now implemented:

- **IPC mode** - Unix domain sockets for distributed deployments
- **Systemd socket activation** - Pre-bound file descriptors via `LISTEN_FDS`
- **Endpoint detection** - Automatic transport mode selection

## Future Work

- **Multi-host routing** - Subscription-driven event propagation via bridge
- **Event persistence/replay** - EventArchiver subscriber for debugging
- **CDC events** - Optional threshold breach events from MetricsService
