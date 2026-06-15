# EventService Architecture

## Overview

EventService provides pub/sub event distribution for hyprstream using moq-lite as the transport. It enables services to publish lifecycle events that other services can subscribe to, replacing polling-based status checks.

The event bus replaced the legacy ZMQ XPUB/XSUB proxy (`ProxyService`) in epic #131/#167. Publishers and subscribers have the same API; only the underlying transport changed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Single Host                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────── moq-lite Origin ────────────────────────────┐  │
│  │                    (MoqEventOrigin / MoqEventBarrierService)          │  │
│  │                                                                        │  │
│  │  Publishers                                        Subscribers         │  │
│  │  ┌────────────────┐                               ┌────────────────┐  │  │
│  │  │WorkerService   │                               │WorkflowService │  │  │
│  │  │RegistryService │──publish──► [Origin] ──sub──►│CLI (wait)      │  │  │
│  │  │ModelService    │                               │ThresholdMonitor│  │  │
│  │  │InferenceService│                               └────────────────┘  │  │
│  │  └────────────────┘                                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  UDS socket: /tmp/hyprstream-{pid}/moq.sock                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

## Delivery Semantics

| Aspect | Behavior |
|--------|----------|
| Ordering | In-order per publisher; no cross-publisher guarantees |
| Delivery | At-most-once (fire-and-forget) |
| Persistence | None — events are ephemeral |
| Late join | Subscribers only see events after subscribing |
| Slow subscriber | moq-lite applies backpressure; drops at buffer limit |
| Prefix filtering | Dot-separated prefix match (`"worker."` → all worker events) |

### Hybrid State Pattern

For reliable status checking (e.g., CLI waiting for container):

1. Query current state via RPC first
2. Subscribe to events for updates
3. Handle race condition by checking timestamp in query response

## Components

### MoqEventBarrierService

Holds the `MoqEventOrigin` lifetime and keeps the in-process moq relay alive. Spawned
as a `Spawnable` service by `ServiceSpawner` at startup.

**Location:** `crates/hyprstream-rpc/src/moq_event.rs`

```rust
// Started by the factory before any publisher/subscriber connects
let barrier = MoqEventBarrierService::new(origin.clone());
let spawner = ServiceSpawner::tokio();
let service: SpawnedService = spawner.spawn(barrier).await?;

// Graceful shutdown
service.stop().await?;
```

### EventPublisher

Async publisher backed by `MoqEventOrigin`. Each service creates its own instance.

**Location:** `crates/hyprstream-workers/src/events/publisher.rs`

```rust
let publisher = EventPublisher::new(origin.clone(), "worker");
publisher.publish("sandbox123", "started", &payload).await?;
```

### EventSubscriber

Async subscriber backed by `MoqEventSubscriber`. Dot-separated prefix filtering.

**Location:** `crates/hyprstream-workers/src/events/subscriber.rs`

```rust
let mut subscriber = EventSubscriber::new(origin.clone());
subscriber.subscribe("worker.").await?;  // All worker events

while let Ok((topic, payload)) = subscriber.recv().await {
    println!("Received: {} ({} bytes)", topic, payload.len());
}
```

**Additional methods:**
- `subscribe_all()` — Subscribe to all topics (empty prefix)
- `recv_timeout(duration)` — Receive with timeout
- `try_recv()` — Non-blocking receive

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

Each event is published as:
- **Track name:** topic string (e.g., `worker.sandbox123.started`)
- **Payload:** Cap'n Proto serialized `EventEnvelope`

Prefix filtering is applied by `MoqEventSubscriber` using dot-separated prefix matching.

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
  topic @3 :Text;           # For prefix filtering
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
| EventService | publish/subscribe (moq-lite) | Lifecycle events (broadcast) |
| MetricsService | REQ/REP (UDS RPC) | Data queries/inserts |

MetricsService queries are NOT broadcast to EventService. Future work may add optional CDC for threshold breach events.

## Implemented Features

- **In-process mode** — moq-lite origin runs in the main process; no IPC needed
- **UDS accept path** — External subscribers connect via `/tmp/hyprstream-{pid}/moq.sock`
- **Prefix filtering** — Dot-separated topic prefixes, identical semantics to the former ZMQ pub/sub filter

## Future Work

- **Multi-host routing** — Subscription-driven event propagation via iroh relay
- **Event persistence/replay** — EventArchiver subscriber for debugging
- **CDC events** — Optional threshold breach events from MetricsService
