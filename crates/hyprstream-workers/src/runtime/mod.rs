//! CRI RuntimeClient implementation
//!
//! Provides Kubernetes CRI-aligned APIs for managing pod sandboxes (Kata VMs)
//! and containers. PodSandbox maps to a Kata VM, Container to an OCI container
//! within that VM.
//!
//! # Architecture
//!
//! ```text
//! WorkerService (ZmqService)
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
pub mod kata_backend;
pub mod nspawn;
mod pool;
mod sandbox;
mod service;
pub mod spawner;
mod virtiofs;

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
pub use kata_backend::{KataBackend, KataHandle};
pub use nspawn::{NspawnBackend, NspawnConfig, NspawnHandle};
// Domain entities (business logic only)
pub use container::Container;
pub use pool::{PoolStats, SandboxPool};
pub use sandbox::PodSandbox;
pub use service::WorkerService;
// Re-export service infrastructure from hyprstream-rpc for convenience
pub use hyprstream_rpc::service::{EnvelopeContext, ServiceHandle, ZmqService};
pub use virtiofs::{SandboxVirtiofs, SandboxVirtiofsBuilder};

/// CRI runtime version
pub const RUNTIME_VERSION: &str = "0.1.0";

/// CRI runtime name
pub const RUNTIME_NAME: &str = "hyprstream-workers";
