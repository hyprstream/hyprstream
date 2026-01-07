//! CRI RuntimeService implementation
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
//!     ├── RuntimeService trait
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
mod virtiofs;

pub use client::{
    ContainerAttributes, ContainerFilter, ContainerMetadata, ContainerStats,
    ContainerStatsFilter, ContainerStatusResponse, CpuUsage, ExecSyncResponse,
    FilesystemIdentifier, FilesystemUsage, LinuxPodSandboxStats, MemoryUsage,
    NetworkInterfaceUsage, NetworkUsage, PodSandboxAttributes, PodSandboxFilter,
    PodSandboxMetadata, PodSandboxStats, PodSandboxStatsFilter, PodSandboxStatusResponse,
    ProcessUsage, RuntimeCondition, RuntimeService, RuntimeStatus, RuntimeZmq,
    StatusResponse, VersionResponse,
};
pub use container::{
    Container, ContainerConfig, ContainerState, ContainerStatus, ImageSpec as ContainerImageSpec,
    KeyValue,
};
pub use pool::{PoolStats, SandboxPool};
pub use sandbox::{
    PodIP, PodSandbox, PodSandboxConfig, PodSandboxNetworkStatus, PodSandboxState,
    PodSandboxStatus,
};
pub use service::{WorkerService, WORKER_ENDPOINT};
// Re-export service infrastructure from hyprstream-rpc for convenience
pub use hyprstream_rpc::service::{EnvelopeContext, ServiceHandle, ZmqService};
pub use virtiofs::{SandboxVirtiofs, SandboxVirtiofsBuilder};

/// CRI runtime version
pub const RUNTIME_VERSION: &str = "0.1.0";

/// CRI runtime name
pub const RUNTIME_NAME: &str = "hyprstream-workers";
