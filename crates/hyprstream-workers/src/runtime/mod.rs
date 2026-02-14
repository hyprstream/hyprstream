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

mod client;
mod container;
mod pool;
mod sandbox;
mod service;
pub mod spawner;
mod virtiofs;

// Generated wire types re-exported from client (originally from worker_client)
// Note: Filter types and Metadata are hand-written in client.rs with extra functionality
pub use client::{
    // Generated client
    GenWorkerClient,
    // Generated response/stats types
    VersionInfo, RuntimeStatus, RuntimeCondition,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    PodSandboxStats, PodSandboxAttributes, LinuxPodSandboxStats,
    ContainerStats as ContainerStatsWire, ContainerAttributes,
    CpuUsage, MemoryUsage, NetworkUsage, NetworkInterfaceUsage,
    ProcessUsage, FilesystemUsage, FilesystemIdentifier,
    Timestamp,
    PodSandboxInfo, ContainerInfo,
    // Generated enum types
    PodSandboxStateEnum, ContainerStateEnum,
    // Generated DTOs (now canonical - no duplicates)
    ImageSpec, PodSandboxConfig, PodSandboxStatus, PodSandboxNetworkStatus,
    ContainerConfig, ContainerStatus,
    DNSConfig, PortMapping, LinuxPodSandboxConfig,
    LinuxSandboxSecurityContext,
    Mount, Device, LinuxContainerConfig,
    LinuxContainerSecurityContext, Capability,
    AuthConfig as AuthConfigWire, StreamInfo,
    ImageInfo, ImageStatusResult,
    // Hand-written types (from client.rs)
    ContainerFilter, ContainerMetadata, ContainerStatsFilter,
    KeyValue,
    PodSandboxFilter, PodSandboxMetadata, PodSandboxStatsFilter,
    RuntimeClient, StatusResponse,
};
// Domain entities (business logic only - not DTOs)
pub use container::{Container, ContainerState};
pub use pool::{PoolStats, SandboxPool};
pub use sandbox::{PodSandbox, PodSandboxState};
pub use service::WorkerService;
// Re-export service infrastructure from hyprstream-rpc for convenience
pub use hyprstream_rpc::service::{EnvelopeContext, ServiceHandle, ZmqService};
pub use virtiofs::{SandboxVirtiofs, SandboxVirtiofsBuilder};

/// CRI runtime version
pub const RUNTIME_VERSION: &str = "0.1.0";

/// CRI runtime name
pub const RUNTIME_NAME: &str = "hyprstream-workers";
