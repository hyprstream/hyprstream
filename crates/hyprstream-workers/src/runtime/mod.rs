//! CRI RuntimeClient implementation
//!
//! Provides Kubernetes CRI-aligned APIs for managing pod sandboxes (Kata VMs)
//! and containers. PodSandbox maps to a Kata VM, Container to an OCI container
//! within that VM.
//!
//! # Architecture
//!
//! ```text
//! WorkerService (RequestService)
//!     │
//!     ├── RuntimeClient trait (client-side interface)
//!     │     ├── run_pod_sandbox()    → Creates Kata VM
//!     │     ├── create_container()   → Creates container in VM
//!     │     ├── start_container()    → Starts container
//!     │     └── exec_sync()          → Executes command
//!     │
//!     └── SandboxPool
//!           ├── Warm pool (pre-created VMs)
//!           └── Active sandboxes
//! ```

// Admission control (#525 P2): fail-closed identity, per-Subject/per-group
// quotas, bounded wait-queue, resource-aware fit over the #628 scheduling
// substrate. Consumed by `pool::SandboxPool::acquire`.
mod admission;
pub mod backend;
mod client;
mod container;
// CRI-client sandbox backend (#510) — gated behind `cri`. A tonic gRPC
// client of an external, already-running CRI runtime service (containerd's
// CRI plugin, or CRI-O) — the kubelet role, not an embedded runtime. Sibling
// of `oci_backend` (which drives podman via CLI shell-out) under the same
// SandboxBackend seam; does NOT replace `kata_backend` (embedded Kata stays).
#[cfg(feature = "cri")]
pub mod cri_backend;
// `/exec/instances/` VFS projection of `SandboxPool` (#608 P2 / #610) — a
// `hyprstream_vfs::Mount` impl, so it has no `kata-vm`/`wasm` dependency.
pub mod exec_mount;
// Kata/CH VM backend — gated behind `kata-vm` (pulls the kata/nydus toolchain).
#[cfg(feature = "kata-vm")]
pub mod kata_backend;
// kata-agent ttrpc/vsock client (#344): CreateContainer/StartContainer/
// ExecProcess/WaitProcess against the guest's kata-agent. Used by
// `kata_backend::KataBackend::exec_sync`.
#[cfg(feature = "kata-vm")]
pub mod kata_agent;
pub mod nspawn;
// Rootless OCI container backend (#346) — gated behind `oci`. Drives a rootless
// OCI runtime (podman by default) via CLI shell-out; torch-free, no VM toolchain.
// A subprocess sibling of nspawn under the SandboxBackend seam. Off by default so
// the lean build doesn't assume a container runtime is installed.
#[cfg(feature = "oci")]
pub mod oci_backend;
mod pool;
mod sandbox;
// In-process WebAssembly sandbox backend (#505 P2) — gated behind `wasm`
// (pulls the wasmtime-bearing `hyprstream-workers-wasmtime` substrate). A native in-process
// sibling under the SandboxBackend seam; the default build stays lean + torch-free.
#[cfg(feature = "wasm")]
pub mod wasm_backend;
// Per-sandbox VFS composition + serve (FS-D, #365): composes the image Mount +
// injected mounts into a Namespace and serves it over vhost-user-fs. Needs the
// image filesystem service (`oci-image`) but NOT the VM toolchain, so any
// backend that hands a guest a composed 9P/VFS namespace consumes it (#632).
#[cfg(all(not(target_arch = "wasm32"), feature = "oci-image"))]
pub mod sandbox_fs;
// Canonical backend taxonomy + fail-closed selection (Phase 0 of #508, #507).
pub mod selection;
mod service;
pub mod spawner;
// Wanix-guest workload wiring (#506 deliverable 3): allocate a per-workload UDS,
// serve a tenant's Subject-scoped VFS `Mount` as 9P2000.L over it, and inject the
// socket + env into a sandbox so the native Wanix guest dials back. Native only
// (needs tokio `net` for the 9P UDS server).
#[cfg(not(target_arch = "wasm32"))]
pub mod wanix_workload;

// Generated wire types — canonical OCI/CRI-aligned names (no Gen*/Wire/Enum aliases)
pub use client::{
    // Generated client
    WorkerClient,
    // Response types
    VersionInfo, RuntimeStatus, RuntimeCondition,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    // Stats types
    PodSandboxStats, PodSandboxAttributes, LinuxPodSandboxStats,
    ContainerStats, ContainerAttributes,
    CpuUsage, MemoryUsage, NetworkUsage, NetworkInterfaceUsage,
    ProcessUsage, FilesystemUsage, FilesystemIdentifier,
    Timestamp,
    // Info types
    PodSandboxInfo, ContainerInfo,
    // State enums (generated, with Hash + serde)
    PodSandboxState, ContainerState,
    // Config/request DTOs
    ImageSpec, PodSandboxConfig, PodSandboxStatus, PodSandboxNetworkStatus,
    ContainerConfig, ContainerStatus,
    PodSandboxMetadata, ContainerMetadata,
    DNSConfig, PortMapping, LinuxPodSandboxConfig,
    LinuxSandboxSecurityContext,
    Mount, Device, LinuxContainerConfig,
    LinuxContainerSecurityContext, Capability,
    AuthConfig, StreamInfo,
    ImageInfo, ImageStatusResult,
    LinuxContainerResources,
    // Filter types (generated)
    PodSandboxFilter, ContainerFilter,
    PodSandboxStatsFilter, ContainerStatsFilter,
    // Common types
    KeyValue,
    // Local composite
    StatusResponse,
};
// Backend trait and implementations
pub use backend::{SandboxBackend, SandboxHandle};
#[cfg(feature = "cri")]
pub use cri_backend::{CriBackend, CriConfig, CriHandle};
// `/exec/instances/` Plan9 projection of the pool (#608 P2 / #610)
pub use exec_mount::ExecMount;
#[cfg(feature = "kata-vm")]
pub use kata_backend::{KataBackend, KataHandle};
pub use nspawn::{NspawnBackend, NspawnConfig, NspawnHandle};
#[cfg(feature = "oci")]
pub use oci_backend::{OciBackend, OciConfig, OciHandle};
#[cfg(feature = "wasm")]
pub use wasm_backend::{WasmBackend, WasmConfig, WasmHandle};
// Inventory-based backend registry + fail-closed selection spine (#507)
pub use selection::{
    backend_injects_9p_socket, require_9p_socket_capability, require_fuse_mount_capability,
    resolve_backend, resolve_backend_9p_capable, BackendCtx, BackendRegistration,
};
// Scheduling-substrate explain (#628): the shared SelectionReport<C> trace.
pub use selection::{explain_selection, BackendCandidate};
// Domain entities (business logic only)
pub use container::Container;
pub use pool::{PoolStats, SandboxPool};
pub use sandbox::PodSandbox;
// Admission control (#525 P2) — quota/queue configuration + the request
// annotations it reads (GPU count demand, group key).
pub use admission::{
    AdmissionConfig, DenyUnknownGroupValidator, GroupSelectorValidator, StaticGroupMembership,
    ANN_GPU_REQUEST, ANN_GROUP,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "oci-image"))]
pub use sandbox_fs::{
    InjectedMounts, SandboxFs, SandboxFsLocalMount, SandboxFsServer, VFS_SOCKET_NAME,
};
#[cfg(not(target_arch = "wasm32"))]
pub use wanix_workload::{
    import_guest_namespace, inject_9p_socket, ImportedGuestNamespace, Injected9pServer,
    WanixGuestConfig, WanixInjection, ANN_WANIX_COMMAND, ENV_9P_SOCK, ENV_GUEST_EXPORT_SOCK,
};
pub use service::WorkerService;
// Re-export service infrastructure from hyprstream-rpc for convenience
pub use hyprstream_rpc::service::{EnvelopeContext, ServiceHandle, RequestService};

/// CRI runtime version
pub const RUNTIME_VERSION: &str = "0.1.0";

/// CRI runtime name
pub const RUNTIME_NAME: &str = "hyprstream-workers";
