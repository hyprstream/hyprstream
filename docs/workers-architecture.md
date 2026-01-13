# hyprstream-workers Architecture

Isolated workload execution using Kata Containers with OCI image support and GitHub Actions-compatible workflows.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       hyprstream-workers                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WorkflowService (ZmqService) ⚠️ Partially Implemented              │
│    └── Endpoint: inproc://hyprstream/workflows                      │
│    └── Discovers .github/workflows/*.yml from RegistryService repos │
│    └── Subscribes to event bus for workflow triggers                │
│    └── Spawns pods/containers via WorkerService (RuntimeClient)     │
│                                                                     │
│  WorkerService (ZmqService) ─ Kubernetes CRI aligned                │
│    └── Endpoint: inproc://hyprstream/workers                        │
│    └── RuntimeClient: PodSandbox + Container lifecycle              │
│    └── ImageClient: Pull, List, Remove images                       │
│    └── PodSandbox = Kata VM (maps to CRI sandbox concept)           │
│    └── Container = OCI container within VM                          │
│                                                                     │
│  [Future] gRPC CRI Bridge                                           │
│    └── Thin adapter exposing WorkerService as CRI runtime           │
│    └── Enables kubelet integration (become a K8s runtime)           │
│                                                                     │
│  [Future] Guest Agent                                               │
│    └── vsock + CURVE communication channel                          │
│    └── MCP tool execution, exec sync                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Design Rationale

WorkerService mirrors Kubernetes CRI (`runtime.v1`):

1. **Future gRPC bridge** - Can expose as CRI runtime for kubelet/crictl
2. **Pod abstraction** - Maps naturally to Kata VMs (sandbox = VM, containers inside)
3. **ImageService built-in** - No separate image management needed
4. **Kubernetes-native** - Ready for K8s integration if needed

## Crate Structure

```
crates/hyprstream-workers/
├── Cargo.toml
├── build.rs                      # Cap'n Proto schema compilation
├── schema/
│   └── workers.capnp             # CRI-aligned RuntimeService + ImageService + WorkflowService
├── src/
│   ├── lib.rs
│   ├── config.rs                 # WorkerConfig, PoolConfig, WorkflowConfig
│   ├── error.rs                  # WorkerError enum
│   │
│   ├── runtime/                  # CRI RuntimeClient (PodSandbox + Container)
│   │   ├── mod.rs
│   │   ├── service.rs            # WorkerService (ZmqService impl)
│   │   ├── client.rs             # RuntimeClient trait + RuntimeZmq client
│   │   ├── sandbox.rs            # PodSandbox (Kata VM lifecycle)
│   │   ├── container.rs          # Container lifecycle within sandbox
│   │   ├── pool.rs               # SandboxPool (warm VM management)
│   │   ├── virtiofs.rs           # VM file sharing via virtio-fs
│   │   └── spawner/              # Re-exports from hyprstream-rpc
│   │
│   ├── image/                    # RAFS-backed ImageClient
│   │   ├── mod.rs
│   │   ├── client.rs             # ImageClient trait + ImageZmq client
│   │   ├── store.rs              # RafsStore - chunk CAS + bootstrap metadata
│   │   └── manifest.rs           # OCI manifest fetching and parsing
│   │
│   ├── workflow/                 # WorkflowService (orchestration)
│   │   ├── mod.rs
│   │   ├── service.rs            # WorkflowService (ZmqService impl)
│   │   ├── client.rs             # WorkflowClient trait + WorkflowZmq client
│   │   ├── parser.rs             # GitHub Actions YAML parsing
│   │   ├── triggers.rs           # EventHandler trait + concrete handlers
│   │   ├── subscription.rs       # WorkflowSubscription, indexed routing
│   │   └── runner.rs             # Job/step execution via RuntimeClient
│   │
│   ├── events/                   # Event bus infrastructure
│   │   ├── mod.rs                # Main entry, re-exports from hyprstream-rpc
│   │   ├── service.rs            # ProxyService usage (from hyprstream-rpc)
│   │   ├── publisher.rs          # EventPublisher
│   │   ├── subscriber.rs         # EventSubscriber
│   │   ├── types.rs              # WorkerEvent, SandboxStarted, etc.
│   │   ├── endpoints.rs          # Endpoint detection (inproc/IPC/systemd)
│   │   └── sockopt.rs            # ZMQ socket options
│   │
│   └── dbus/                     # D-Bus bridge for container access
│       ├── mod.rs                # Main module
│       ├── bridge.rs             # ZMQ bridge with policy integration
│       ├── policy.rs             # Resource access control
│       └── protocol.rs           # D-Bus request/response protocol
│
└── cloud-init/                   # VM bootstrap templates
    ├── user-data.template
    └── meta-data.template
```

## Services

### WorkerService (CRI RuntimeService + ImageService)

Implements Kubernetes CRI. PodSandbox = Kata VM, Container = OCI container.

```rust
pub struct WorkerService {
    sandbox_pool: Arc<SandboxPool>,
    rafs_store: Arc<RafsStore>,
    policy_client: PolicyZmqClient,
    runtime_handle: tokio::runtime::Handle,
}
```

#### RuntimeClient Trait (CRI-aligned)

```rust
#[async_trait]
pub trait RuntimeClient {
    // Runtime
    async fn version(&self, version: &str) -> Result<VersionResponse>;
    async fn status(&self, verbose: bool) -> Result<StatusResponse>;

    // Pod Sandbox lifecycle (Kata VM)
    async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> Result<String>;
    async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;
    async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;
    async fn pod_sandbox_status(&self, pod_sandbox_id: &str, verbose: bool) -> Result<PodSandboxStatusResponse>;
    async fn list_pod_sandbox(&self, filter: Option<&PodSandboxFilter>) -> Result<Vec<PodSandbox>>;

    // Container lifecycle
    async fn create_container(&self, pod_sandbox_id: &str, config: &ContainerConfig, sandbox_config: &PodSandboxConfig) -> Result<String>;
    async fn start_container(&self, container_id: &str) -> Result<()>;
    async fn stop_container(&self, container_id: &str, timeout: i64) -> Result<()>;
    async fn remove_container(&self, container_id: &str) -> Result<()>;
    async fn container_status(&self, container_id: &str, verbose: bool) -> Result<ContainerStatusResponse>;
    async fn list_containers(&self, filter: Option<&ContainerFilter>) -> Result<Vec<Container>>;

    // Exec
    async fn exec_sync(&self, container_id: &str, cmd: &[String], timeout: i64) -> Result<ExecSyncResponse>;

    // Stats
    async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> Result<PodSandboxStats>;
    async fn list_pod_sandbox_stats(&self, filter: Option<&PodSandboxStatsFilter>) -> Result<Vec<PodSandboxStats>>;
    async fn container_stats(&self, container_id: &str) -> Result<ContainerStats>;
    async fn list_container_stats(&self, filter: Option<&ContainerStatsFilter>) -> Result<Vec<ContainerStats>>;
}
```

#### ImageClient Trait (CRI-aligned)

```rust
#[async_trait]
pub trait ImageClient {
    async fn list_images(&self, filter: Option<&ImageFilter>) -> Result<Vec<Image>>;
    async fn image_status(&self, image: &ImageSpec, verbose: bool) -> Result<ImageStatusResponse>;
    async fn pull_image(&self, image: &ImageSpec, auth: Option<&AuthConfig>) -> Result<String>;
    async fn remove_image(&self, image: &ImageSpec) -> Result<()>;
    async fn image_fs_info(&self) -> Result<Vec<FilesystemUsage>>;
}
```

### WorkflowService (Orchestration)

Discovers workflows from repos, subscribes to events, spawns via WorkerService.

```rust
pub struct WorkflowService {
    // Per-repo workflow subscriptions (O(1) lookup)
    repo_workflows: RwLock<HashMap<RepoId, Vec<WorkflowSubscription>>>,

    // Per-model event handlers (training, metrics)
    model_handlers: RwLock<HashMap<ModelId, Vec<Box<dyn EventHandler>>>>,

    // ZMQ subscriber socket
    subscriber: zmq::Socket,

    worker_client: WorkerZmq,
    registry_client: RegistryZmq,
    policy_client: PolicyZmqClient,
}
```

#### WorkflowClient Trait ⚠️ Partially Implemented

```rust
#[async_trait]
pub trait WorkflowClient {
    // Discovery and registration
    async fn scan_repo(&self, repo_id: &str) -> Result<Vec<WorkflowDef>>;
    async fn register_workflow(&self, workflow: WorkflowDef) -> Result<WorkflowId>;
    async fn list_workflows(&self) -> Result<Vec<WorkflowInfo>>;

    // Manual triggers
    async fn dispatch(&self, workflow_id: &WorkflowId, inputs: HashMap<String, String>) -> Result<RunId>;

    // Event subscriptions
    async fn subscribe(&self, trigger: EventTrigger, workflow_id: &WorkflowId) -> Result<SubscriptionId>;
    async fn unsubscribe(&self, sub_id: &SubscriptionId) -> Result<()>;

    // Run status
    async fn get_run(&self, run_id: &RunId) -> Result<WorkflowRun>;
    async fn list_runs(&self, workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>>;
}
```

## Image Storage: Nydus RAFS

Images stored using Nydus RAFS format - chunk-level CAS with lazy-loading.

| Aspect | Traditional OCI | Nydus RAFS |
|--------|-----------------|------------|
| **Granularity** | Layer blobs (100s MB) | Chunks (1MB) |
| **Deduplication** | Per-layer | Per-chunk (across all images) |
| **Cold start** | Full download | Lazy-load (~76% faster) |
| **Storage** | ~40% dedup | ~80% dedup |

### Storage Layout

```
images/
├── blobs/sha256/          # Chunk blobs (content-addressed, shared)
├── bootstrap/             # RAFS metadata (registry/repo/tag.meta)
└── refs/                  # Tag → digest symlinks
```

### Architecture

```
WorkerService
    │
    ├── RafsStore (global)
    │     ├── blobs/sha256/...     ← Chunk CAS (deduplicated)
    │     └── bootstrap/...        ← RAFS metadata per image
    │
    └── SandboxService (per VM)
          └── nydus daemon         ← Embedded, virtiofs to guest
                │
                └── Guest VM mounts /run/kata-containers/shared
                    (file access triggers chunk fetch)
```

## Event Bus

XPUB/XSUB proxy pattern for reliable delivery.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Event Bus (XPUB/XSUB Proxy)                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Publishers                    EventBroker                Subscribers  │
│  ┌─────────────┐              ┌───────────┐              ┌──────────┐ │
│  │RegistryService│──XPUB────▶│           │──XSUB──────▶│Workflow- │ │
│  │WorkerService  │           │   Proxy   │              │ Service  │ │
│  │InferenceService│           └───────────┘              └──────────┘ │
│  └─────────────┘                                                       │
└────────────────────────────────────────────────────────────────────────┘

Endpoints:
  - EVENTS_PUB = inproc://hyprstream/events/pub    (publishers connect)
  - EVENTS_SUB = inproc://hyprstream/events/sub    (subscribers connect)
```

### Topic Format (Repo-Scoped)

Events use repo-scoped topics for efficient ZMQ prefix filtering:

```
# Worker events (✅ Implemented)
worker.{entity_id}.{event}     # Sandbox/container lifecycle events

# Repository events (⚠️ Future Work - not yet implemented)
git2db.{repo_id}.clone         # Repository cloned
git2db.{repo_id}.push          # Branch pushed
git2db.{repo_id}.commit        # Commit created
git2db.{repo_id}.merge         # Branch merged
git2db.{repo_id}.tag           # Tag created

# Non-repo events (⚠️ Future Work - not yet implemented)
training.{model_id}.started    # Training started
training.{model_id}.completed  # Training completed
metrics.{model_id}.breach      # Threshold breach
```

### Event Handler Trait

```rust
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Event types this handler processes
    fn handles(&self) -> &[&str];

    /// Fine-grained matching (branch patterns, thresholds, etc.)
    fn matches(&self, event: &EventEnvelope) -> bool;

    /// Process the event
    async fn handle(&self, event: &EventEnvelope) -> Result<HandlerResult>;
}

pub enum HandlerResult {
    Dispatch { workflow_id: WorkflowId, inputs: HashMap<String, String> },
    Rescan { repo_id: String },
    Ignored,
}
```

### Event Routing (Indexed Dispatch) ⚠️ Future Work

> **Note**: The routing code below is the design target. Currently only `worker.*` events
> are implemented. git2db/training/metrics event routing is not yet wired up.

```rust
impl WorkflowService {
    async fn handle_event(&self, event: &EventEnvelope) -> Result<()> {
        // Parse topic: "git2db.{repo_id}.{event_type}"
        let parts: Vec<&str> = event.topic.split('.').collect();

        match parts.as_slice() {
            // Repository events - O(1) lookup by repo_id
            ["git2db", repo_id, event_type] => {
                if *event_type == "clone" || *event_type == "push" {
                    self.rescan_repo(repo_id).await?;
                }
                // Dispatch to repo-specific handlers
                if let Some(subscriptions) = self.repo_workflows.read().await.get(*repo_id) {
                    for sub in subscriptions {
                        if sub.handler.matches(event) {
                            sub.handler.handle(event).await?;
                        }
                    }
                }
            }
            // Model events - O(1) lookup by model_id
            ["training" | "metrics", model_id, _] => {
                if let Some(handlers) = self.model_handlers.read().await.get(*model_id) {
                    for handler in handlers {
                        if handler.matches(event) {
                            handler.handle(event).await?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
```

## Policy Integration

Uses existing Casbin-based PolicyManager with new resources and operations.

### New Resources

```
worker:*           # All worker runtime operations
sandbox:*          # All pod sandboxes (Kata VMs)
sandbox:{id}       # Specific pod sandbox
container:*        # All containers
container:{id}     # Specific container
image:*            # All images
image:{name}       # Specific image
workflow:*         # All workflows
workflow:{path}    # Specific workflow (.github/workflows/train.yml)
tool:*             # All MCP tools
tool:{name}        # Specific tool (tool:bash, tool:read_file)
```

### New Operations

```rust
pub enum Operation {
    // Existing
    Infer,    // i - Run model inference
    Train,    // t - Train/fine-tune
    Query,    // q - Query data
    Write,    // w - Write data
    Serve,    // s - Serve via API
    Manage,   // m - Admin operations
    Context,  // c - Context-augmented generation

    // New for workers
    Execute,  // x - Execute in sandbox
    Subscribe,// b - Subscribe to events
    Publish,  // p - Publish artifacts/models
}
```

### Example Policies

```csv
# Local users have full CRI access
p, local:*, *, worker:*, manage, allow
p, local:*, *, sandbox:*, manage, allow
p, local:*, *, container:*, execute, allow
p, local:*, *, image:*, manage, allow
p, local:*, *, workflow:*, execute, allow

# Trainers can create sandboxes and run training workflows
p, trainer, HfModel, sandbox:*, execute, allow
p, trainer, HfModel, container:*, execute, allow
p, trainer, HfModel, workflow:*train*, execute, allow

# Deny dangerous tools for anonymous
p, anonymous, *, tool:bash, execute, deny
p, anonymous, *, tool:write_file, execute, deny
```

## Workflow Triggers

```rust
pub enum EventTrigger {
    /// Repository events - triggers rescan + registered workflows
    RepositoryEvent {
        event_type: RepoEventType,
        pattern: Option<String>,  // e.g., "main", "feature/*"
    },
    /// Training progress (for auto-checkpoint workflows)
    TrainingProgress { model_id: String, min_step: Option<u32> },
    /// Metrics threshold breach (for auto-tune workflows)
    MetricsBreach { metric_name: String, threshold: f64 },
    /// Manual workflow dispatch
    WorkflowDispatch { inputs: HashMap<String, InputDef> },
    /// Custom topic subscription
    Custom { topic: String, pattern: String },
}

pub enum RepoEventType {
    Clone,       // Repository cloned
    Push,        // Branch pushed
    Commit,      // Commit created
    Merge,       // Branch merged
    PullRequest, // PR opened/updated
    Tag,         // Tag created
}
```

## Example Workflow

```yaml
# .github/workflows/train.yml
name: LoRA Training

on:
  workflow_dispatch:
    inputs:
      model:
        description: "Model reference"
        required: true

jobs:
  train:
    runs-on: hyprstream-gpu
    steps:
      - uses: hyprstream/model-load@v1
        with:
          model: ${{ inputs.model }}

      - uses: hyprstream/lora-train@v1
        with:
          dataset: ./data/train.jsonl
          epochs: 3

      - uses: hyprstream/adapter-save@v1
        with:
          name: "lora-${{ github.run_id }}"
```

## Built-in Actions

| Action | Description |
|--------|-------------|
| `hyprstream/model-load@v1` | Load model into worker |
| `hyprstream/lora-train@v1` | Train LoRA adapter |
| `hyprstream/adapter-save@v1` | Save adapter, commit to git |
| `hyprstream/inference-test@v1` | Run inference tests |

## Security Model

1. **Host-side**: Ed25519 signed envelopes (existing `hyprstream-rpc/envelope.rs`)
2. **Container isolation**: Kata VM boundary provides security
3. **Network**: Default isolated (no network), optional bridge mode

Guest agent communication (vsock + CURVE) is deferred to a future phase.

## ZMQ Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Pattern** | XPUB/XSUB proxy | Dynamic subs, monitoring, future persistence |
| **Topic format** | `{source}.{id}.{event}` | Per-repo/model ZMQ prefix filtering |
| **Subscription storage** | Derived from YAML | No runtime state to lose, git-native |
| **Dispatch** | Indexed by repo_id/model_id | O(1) lookup |
| **Handlers** | EventHandler trait | Testable, composable |
| **Filtering** | Two-tier (ZMQ + handler) | ZMQ prefix fast, handler patterns flexible |
| **Backpressure** | HWM (high water mark) | Standard ZMQ flow control |
| **Durability** | Future work | Comprehensive solution TBD |

## Dependencies

```toml
# OCI registry client
oci-distribution = "0.11"

# Nydus (library mode)
nydus-storage = { git = "https://github.com/dragonflyoss/nydus" }
nydus-rafs = { git = "https://github.com/dragonflyoss/nydus" }
nydus-service = { git = "https://github.com/dragonflyoss/nydus" }
nydus-api = { git = "https://github.com/dragonflyoss/nydus" }
```

## Implementation Phases

### Phase 1: Crate Setup + RafsStore
- Create crate with Cargo.toml, Cap'n Proto schema
- Implement RafsStore (OCI registry, chunk storage, metadata)
- Implement SandboxService (nydus daemon, virtiofs)

### Phase 2: WorkerService (CRI RuntimeService)
- Cap'n Proto schema mirroring CRI runtime.v1
- RuntimeService + ImageService traits with ZMQ clients
- PodSandbox with cloud-hypervisor + virtiofs
- SandboxPool (warm/cold VM management)

### Phase 3: Event Bus Infrastructure
- EventBroker (XPUB/XSUB proxy)
- New endpoints (EVENTS_PUB, EVENTS_SUB)
- Repository events in EventPayload

### Phase 4: WorkflowService
- EventHandler trait + concrete handlers
- WorkflowSubscription with indexed routing
- GitHub Actions YAML parser
- Workflow discovery and initialization
- WorkflowRunner (creates PodSandbox, runs containers)
