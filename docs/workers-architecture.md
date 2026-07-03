# hyprstream-workers Architecture

Isolated workload execution behind a pluggable sandbox-backend seam, with CRI-aligned
runtime/image services and GitHub Actions-compatible workflow orchestration.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       hyprstream-workers                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WorkerService (RequestService) ─ Kubernetes CRI aligned            │
│    └── Cap'n Proto over the bridged transports (inproc/UDS/QUIC)    │
│    └── RuntimeClient: PodSandbox + Container lifecycle              │
│    └── ImageClient: Pull, List, Remove images                       │
│    └── PodSandbox = one sandbox on the selected SandboxBackend      │
│    └── Lifecycle events + FD streams ride the moq-lite plane        │
│                                                                     │
│  WorkflowService (RequestService)                                   │
│    └── Discovers .github/workflows/*.yml from RegistryService repos │
│    └── Subscribes to the moq event plane for workflow triggers      │
│    └── Spawns pods/containers via WorkerService (RuntimeClient)     │
│                                                                     │
│  SandboxBackend registry (inventory, fail-closed)                   │
│    └── nspawn / kata / oci / wasm — selected by worker.backend      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Both services implement `hyprstream_rpc::service::RequestService` (Cap'n Proto wire
format via `generate_rpc_service!`) and are transport-agnostic: the blanket `Spawnable`
impl serves them over the registered transport (inproc / UDS / systemd-fd, plus the
QUIC/iroh dial plane). Endpoints are not hardcoded — socket and runtime paths resolve
through the `hyprstream_rpc::paths` registry (FHS/XDG-compliant: `/run/hyprstream/`
as root, `$XDG_RUNTIME_DIR/hyprstream/` per-user, namespaced by `HYPRSTREAM_INSTANCE`).

## Sandbox Backends

The VM/container lifecycle lives behind the `SandboxBackend` trait
(`src/runtime/backend.rs`): `initialize` → `start` → `stop`/`destroy`, plus
`reset` (warm-pool reuse), `exec_sync`, `get_pids`, `update_resources`, and
optional guest-level `container_stats`. Each backend stashes runtime-specific
state (hypervisor handles, PIDs, machine names, live wasm instances) in an opaque
`SandboxHandle` stored on the `PodSandbox`.

### Registry and fail-closed selection (`src/runtime/selection.rs`)

Backends are **not** a hardcoded enum. Each backend self-registers a
`BackendRegistration` via `inventory::submit!` next to its implementation,
feature-gated by the `#[cfg]` that compiles it in — the registry is exactly the
set of backends built into the binary:

| Backend | Feature | Priority | Auto-selectable | Isolation |
|---------|---------|----------|-----------------|-----------|
| `kata` | `kata-vm` | 100 | yes | Full VM (Cloud Hypervisor / Dragonball) |
| `oci` | `oci` | 20 | yes | Rootless OCI container (podman, #346/#694) |
| `nspawn` | (always) | 10 | yes | systemd-nspawn container, host rootfs |
| `wasm` | `wasm` | 5 | **no** (explicit-name-only) | In-process WebAssembly (#505) |

Selection is config-driven via `WorkerConfig::backend` (default `"auto"`):

- **`"auto"`** — among auto-selectable registrations, pick the highest-priority one
  whose `is_available()` probe passes; the choice is logged. If none qualifies,
  **error** — a workload is never run without isolation.
- **A concrete name** — that registration, iff registered *and* its prerequisites
  are present. Otherwise error; a weaker backend is never substituted (the #486
  fail-open bug). An explicit name resolves regardless of `auto_selectable`, so
  `wasm` is reachable when deliberately requested.

The cardinal rule is **fail-closed**: an unavailable or unknown backend returns an
error, never a silent downgrade to weaker isolation. The in-process `wasm` tier
(shared host address space) is excluded from auto-selection entirely so it cannot
become a silent fallback (#547 zero-standing-privilege model).

### Backend notes

- **kata** (`src/runtime/kata_backend.rs`) — full VM isolation via Kata's
  `Hypervisor` trait (Cloud Hypervisor and Dragonball). In-guest container ops
  (create/start/exec) go through a real kata-agent ttrpc/vsock client
  (`src/runtime/kata_agent.rs`, #344) using the upstream `protocols` crate.
  Each sandbox gets a private VFS namespace (image filesystem + injected mounts)
  served over a per-sandbox Unix socket that the guest attaches as a virtio-fs
  ShareFs device (`src/runtime/sandbox_fs.rs`, #365).
- **oci** (`src/runtime/oci_backend.rs`, #346, GA #694) — drives a **rootless
  OCI runtime** (podman by default; `OciConfig::runtime_bin` is configurable) as a
  CLI shell-out. The CRI `PodSandboxConfig` is threaded through the invocation:
  labels/annotations, cgroup resource flags, and security context all map to
  runtime flags. No root, no daemon; user namespaces + seccomp + a real image
  rootfs.
- **nspawn** (`src/runtime/nspawn.rs`) — lightweight `systemd-nspawn` container
  bind-mounting the host rootfs (`--directory=/`) with ephemeral `/tmp`/`/run`,
  veth network isolation, and GPU pass-through via device bind-mounts.
- **wasm** (`src/runtime/wasm_backend.rs`, #505) — runs the workload as a
  WebAssembly guest inside *this* process via the embedded wasmtime substrate
  (`hyprstream-workers-wasmtime`). No hypervisor, no child process: isolation is
  the wasm sandbox itself — a capability-only `Linker` (Profile A: zero WASI),
  DoS-bounded by fuel + epoch interruption. `reset` drops and reinstantiates the
  guest (cheap vs. a VM reboot).

## Crate Structure

```
crates/hyprstream-workers/
├── Cargo.toml
├── build.rs                      # Cap'n Proto schema compilation
├── schema/
│   ├── worker.capnp              # CRI-aligned RuntimeService + ImageService
│   └── workflow.capnp            # WorkflowService
├── src/
│   ├── lib.rs
│   ├── config.rs                 # WorkerConfig (incl. `backend`), PoolConfig, WorkflowConfig
│   ├── error.rs                  # WorkerError enum
│   ├── paths.rs                  # Worker-specific dirs on top of hyprstream_rpc::paths
│   ├── generated.rs              # generate_rpc_service! output
│   │
│   ├── runtime/                  # CRI RuntimeClient (PodSandbox + Container)
│   │   ├── service.rs            # WorkerService (RequestService impl)
│   │   ├── client.rs             # RuntimeClient trait + typed client
│   │   ├── backend.rs            # SandboxBackend trait + SandboxHandle
│   │   ├── selection.rs          # inventory registry + fail-closed resolve_backend
│   │   ├── kata_backend.rs       # Kata VM backend (kata-vm feature)
│   │   ├── kata_agent.rs         # kata-agent ttrpc/vsock client (#344)
│   │   ├── nspawn.rs             # systemd-nspawn backend (always compiled)
│   │   ├── oci_backend.rs        # rootless podman backend (oci feature, #346)
│   │   ├── wasm_backend.rs       # in-process wasmtime backend (wasm feature, #505)
│   │   ├── sandbox.rs            # PodSandbox lifecycle/state
│   │   ├── sandbox_fs.rs         # per-sandbox VFS composition + vhost-user-fs serve (#365)
│   │   ├── exec_mount.rs         # Plan 9 /exec/instances projection (epic #608)
│   │   ├── container.rs          # Container lifecycle within sandbox
│   │   ├── pool.rs               # SandboxPool (warm sandbox management)
│   │   └── virtiofs.rs           # VM file sharing via virtio-fs
│   │
│   ├── image/                    # ImageStore seam + RAFS backend
│   │   ├── client.rs             # ImageClient trait + typed client
│   │   ├── store_trait.rs        # ImageStore trait + inventory registration (feature-invariant)
│   │   ├── store.rs              # RafsStore — chunk CAS + bootstrap metadata (oci-image)
│   │   ├── image_fs.rs           # ImageFs — mountable OCI/RAFS image filesystem (#633)
│   │   ├── rafs_builder.rs       # RAFS bootstrap building
│   │   ├── store_mount.rs        # image store VFS projection
│   │   └── manifest.rs           # OCI manifest fetching and parsing
│   │
│   ├── workflow/                 # WorkflowService (orchestration)
│   │   ├── service.rs            # WorkflowService (RequestService impl)
│   │   ├── client.rs             # WorkflowClient trait + typed client
│   │   ├── parser.rs             # GitHub Actions YAML parsing
│   │   ├── triggers.rs           # EventHandler trait + concrete handlers
│   │   ├── subscription.rs       # WorkflowSubscription, indexed routing
│   │   ├── adapter.rs / gh_adapter.rs  # event-source adapters
│   │   └── runner.rs             # Job/step execution via RuntimeClient
│   │
│   ├── events/                   # moq-lite event integration (epic #600)
│   │   ├── mod.rs                # re-exports EventPublisher/EventSubscriber from hyprstream-rpc
│   │   ├── service.rs            # event-plane wiring
│   │   ├── token_manager.rs      # event auth tokens
│   │   └── types.rs              # WorkerEvent, SandboxStarted, etc.
│   │
│   └── dbus/                     # D-Bus bridge for container access
│       ├── bridge.rs             # bridge with policy integration
│       ├── policy.rs             # Resource access control
│       └── protocol.rs           # D-Bus request/response protocol
```

## Services

### WorkerService (CRI RuntimeService + ImageService)

Mirrors Kubernetes CRI (`runtime.v1`): PodSandbox = one sandbox on the selected
backend, Container = workload within it. The CRI alignment keeps a future gRPC
CRI bridge (kubelet integration) possible without reshaping the API.

```rust
pub struct WorkerService {
    sandbox_pool: Arc<SandboxPool>,
    image_store: Option<Arc<dyn ImageStore>>,   // None when built without oci-image
    containers: RwLock<HashMap<String, Container>>,
    container_sandbox_map: RwLock<HashMap<String, String>>,
    event_publisher: tokio::sync::Mutex<EventPublisher>,   // moq-backed lifecycle events
    active_fd_streams: Arc<RwLock<HashMap<String, ActiveFdStream>>>,
    stream_channel: Arc<StreamChannel>,          // authenticated FD streaming (moq)
    transport: TransportConfig,
    // ... signing key, cancellation, etc.
}
```

The synchronous CRI request/reply control plane rides the shared Cap'n Proto
bridged transport; the asynchronous surfaces (lifecycle events, terminal
attach/detach FD data) ride the moq-lite streaming plane. There is no ZMQ socket
anywhere in this path.

The `RuntimeClient` and `ImageClient` traits mirror the CRI runtime/image services
(pod sandbox lifecycle, container lifecycle, `exec_sync`, stats; image list /
status / pull / remove / fs-info) — see `src/runtime/client.rs` and
`src/image/client.rs` for the exact signatures.

### WorkflowService (Orchestration)

Discovers workflows from repos, subscribes to the event plane, spawns via
WorkerService:

```rust
pub struct WorkflowService {
    workflows: RwLock<HashMap<WorkflowId, WorkflowDef>>,
    runs: Arc<RwLock<HashMap<RunId, WorkflowRun>>>,
    repo_workflows: RwLock<HashMap<String, Vec<WorkflowSubscription>>>,  // O(1) per-repo lookup
    repo_paths: RwLock<HashMap<String, PathBuf>>,
    subscriptions: RwLock<HashMap<String, WorkflowSubscription>>,
    handlers: RwLock<Vec<Box<dyn EventHandler>>>,
    event_loop_handle: tokio::sync::Mutex<Option<JoinHandle<()>>>,  // moq EventSubscriber loop
    transport: TransportConfig,
    // ... clients, signing key
}
```

The event loop consumes a moq-backed `EventSubscriber` (topic-prefix
subscriptions), dispatches through the `EventHandler` trait, and runs jobs/steps
via `RuntimeClient`.

## Image Storage

### ImageStore trait (feature-invariant seam)

`ImageStore` (`src/image/store_trait.rs`) is the seam between the generated CRI
`ImageHandler` surface (always compiled) and concrete image backends. Backends
register via `inventory::submit!{ ImageBackendRegistration }` — same pattern as
`SandboxBackend`. `WorkerService` holds `Option<Arc<dyn ImageStore>>`; when no
backend is compiled in, CRI image ops return a clear "built without oci-image
support" error rather than failing to compile (#646).

### RafsStore + ImageFs (`oci-image` feature)

The production backend is Nydus RAFS — chunk-level CAS with lazy loading:

| Aspect | Traditional OCI | Nydus RAFS |
|--------|-----------------|------------|
| **Granularity** | Layer blobs (100s MB) | Chunks (1MB) |
| **Deduplication** | Per-layer | Per-chunk (across all images) |
| **Cold start** | Full download | Lazy-load |

```
images/
├── blobs/sha256/          # Chunk blobs (content-addressed, shared)
├── bootstrap/             # RAFS metadata (registry/repo/tag.meta)
└── refs/                  # Tag → digest symlinks
```

**ImageFs** (`src/image/image_fs.rs`, #633) is a universal mountable image
filesystem: an `OverlayFs` whose **lower** is the image's RAFS loaded in-process
(read-only, lazily fetching chunks from the CAS / Dragonfly P2P) and whose
**upper** is a per-sandbox writable directory. "Root" is a mount position, not a
type — an `ImageFs` mounted at `/` is the sandbox rootfs purely because the
namespace recipe put it there. Kata serves it to its guest via virtio-fs today;
it compiles under `oci-image` alone, with no VM toolchain required.

## Plan 9 /exec Projection (epic #608)

`src/runtime/exec_mount.rs` projects `SandboxPool`'s active sandboxes as a Plan
9 `/proc`-style VFS tree:

```
/exec/instances/                  # dynamic dir: active sandbox/instance ids
/exec/instances/<id>/ctl          # write a verb to drive the instance lifecycle
/exec/instances/<id>/status       # read-only: current PodSandboxState (live poll)
/exec/instances/<id>/exit         # read-only: blocks until terminal, then returns status
/exec/instances/<id>/ns           # read-only: mount-prefix/namespace listing
```

Reads of `exit` that start after termination return immediately with the retained
status (read-then-subscribe, no missed completions).

## Engine Crates

The worker-engine crate family (`hyprstream-workers-{engine}`) provides language
substrates that plug into the sandbox/VFS model:

- **`hyprstream-workers-wasmtime`** — the generic, language-agnostic embedded
  wasmtime host. Sandboxes untrusted wasm guests behind capability profiles:
  Profile A is a bespoke `Linker` exposing exactly the chosen host functions
  (`env::host_random` + Subject-scoped `env::vfs_*`) with every other import
  trapped; Profile B is a real WASI preview1 surface whose only filesystem is a
  Subject-scoped VFS `Mount`. Capabilities are Subject-scoped by construction
  (identity lives in the per-call `Store` state, never guest-supplied), with
  fuel + epoch DoS bounds. Guest-side support crates:
  `hyprstream-workers-wasmtime-fsguest`.
- **`hyprstream-workers-tcl`** — a Tcl (molt) shell over the hyprstream VFS
  namespace: VFS builtins (`cat`, `ls`, `echo`, `ctl`, `mount`, …), unknown
  commands fall back to `/bin/{name}` resolution (Plan 9 PATH model). Dangerous
  molt commands are removed at construction; all host filesystem access goes
  through the VFS.
- **`hyprstream-workers-python`** — the one Python-aware layer over the generic
  wasm engine. Runs untrusted Python on a RustPython guest
  (`hyprstream-workers-python-guest`, `wasm32-unknown-unknown`) inside a
  Profile-A sandbox, and mounts a `/lang/python` 9P shell into the namespace.
  `import os; os.system(...)` is inert — there is no syscall surface.

## Event Bus

Worker events ride the moq-lite event plane (`MoqEventOrigin`), which replaced
the former ZMQ XPUB/XSUB proxies (#167, epic #600). `EventPublisher` /
`EventSubscriber` live in `hyprstream-rpc::events` and are re-exported from
`src/events/` for existing callers. Confidentiality is group-keyed: privacy
modes range from `Public` (plaintext) to group-key-encrypted
`ZeroKnowledge`/`LimitedKnowledge`. See `docs/eventservice-architecture.md` for
the full event-plane design.

Worker lifecycle events (`worker.{entity_id}.{event}` — sandbox/container
started/stopped) are published today; WorkflowService subscribes by topic prefix
and dispatches through `EventHandler` implementations.

## Policy Integration

Handlers authorize through the shared Casbin + JWT/UCAN stack (`#[authorize]`
attributes over `action:resource` pairs), the same model as every other
hyprstream RPC service — see `docs/rpc-architecture.md`. Backend selection adds
its own mandatory floor: fail-closed resolution means no policy misconfiguration
can silently downgrade a workload's isolation tier.

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

## Security Model

1. **Host-side**: signed envelopes (hybrid-PQC COSE, `hyprstream-rpc`) on every
   RPC request; transport channel auth (TLS 1.3 on QUIC, UDS peer credentials)
2. **Isolation**: the selected backend's boundary — VM (kata), rootless
   container (oci), nspawn container, or the wasm capability sandbox — chosen
   fail-closed, never silently downgraded
3. **Network**: default isolated, optional bridge mode

## Dependencies

```toml
# Kata Containers runtime-rs (hypervisor abstraction) — kata-vm feature
kata-hypervisor = { git = "https://github.com/kata-containers/kata-containers", tag = "3.31.0", package = "hypervisor", features = ["cloud-hypervisor"] }
kata-types      = { git = "https://github.com/kata-containers/kata-containers", tag = "3.31.0" }
protocols       = { git = "https://github.com/kata-containers/kata-containers", tag = "3.31.0", features = ["async", "with-serde"] }  # kata-agent ttrpc

# Nydus (RAFS image storage) — oci-image feature. Pinned to the exact commit
# for tag v2.4.0; see Cargo.toml for the crates.io cohort-conflict rationale.
nydus-api     = { git = "https://github.com/dragonflyoss/nydus", rev = "5a9d42d8" }
nydus-storage = { git = "https://github.com/dragonflyoss/nydus", rev = "5a9d42d8", features = ["backend-registry", "backend-localfs"] }
nydus-rafs    = { git = "https://github.com/dragonflyoss/nydus", rev = "5a9d42d8" }
nydus-service = { git = "https://github.com/dragonflyoss/nydus", rev = "5a9d42d8" }
```

> Status: the phased build-out that this document once tracked (RafsStore →
> WorkerService → event bus → WorkflowService) is complete; everything above
> describes shipped code.
