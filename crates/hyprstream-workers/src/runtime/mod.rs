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

pub mod backend;
mod client;
mod container;
// `/exec/instances/` VFS projection of `SandboxPool` (#608 P2 / #610) — a
// `hyprstream_vfs::Mount` impl, so it has no `kata-vm`/`wasm` dependency.
pub mod exec_mount;
// Kata/CH VM backend — gated behind `kata-vm` (pulls the kata/nydus toolchain).
#[cfg(feature = "kata-vm")]
pub mod kata_backend;
pub mod nspawn;
// Rootless podman OCI sandbox backend (#346, T2-A of #341) — gated behind
// `podman`. Shells out to the `podman` CLI at runtime; no extra build-time
// dependencies, so this is a thin opt-in atop the always-built nspawn shape.
#[cfg(feature = "podman")]
pub mod podman;
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
// `/exec/instances/` Plan9 projection of the pool (#608 P2 / #610)
pub use exec_mount::ExecMount;
#[cfg(feature = "kata-vm")]
pub use kata_backend::{KataBackend, KataHandle};
pub use nspawn::{NspawnBackend, NspawnConfig, NspawnHandle};
#[cfg(feature = "podman")]
pub use podman::{PodmanBackend, PodmanConfig, PodmanHandle};
#[cfg(feature = "wasm")]
pub use wasm_backend::{WasmBackend, WasmConfig, WasmHandle};
// Inventory-based backend registry + fail-closed selection spine (#507)
pub use selection::{resolve_backend, BackendCtx, BackendRegistration};
// Domain entities (business logic only)
pub use container::Container;
pub use pool::{PoolStats, SandboxPool};
pub use sandbox::PodSandbox;
#[cfg(all(not(target_arch = "wasm32"), feature = "oci-image"))]
pub use sandbox_fs::{InjectedMounts, SandboxFs, SandboxFsServer, VFS_SOCKET_NAME};
pub use service::WorkerService;
// Re-export service infrastructure from hyprstream-rpc for convenience
pub use hyprstream_rpc::service::{EnvelopeContext, ServiceHandle, RequestService};

/// CRI runtime version
pub const RUNTIME_VERSION: &str = "0.1.0";

/// CRI runtime name
pub const RUNTIME_NAME: &str = "hyprstream-workers";
