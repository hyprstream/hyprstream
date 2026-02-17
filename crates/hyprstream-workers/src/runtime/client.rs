//! RuntimeClient trait and ZMQ client
//!
//! Client-side trait for CRI RuntimeService (`runtime.v1`) for future kubelet compatibility.
//! Response types use generated `*Data` types from the Cap'n Proto schema directly,
//! eliminating redundant domain type wrappers and cross-crate conversion.

use crate::error::Result;
use crate::worker_capnp;
use async_trait::async_trait;
use hyprstream_rpc::{FromCapnp, ToCapnp};
use std::collections::HashMap; // Only for StatusResponse.info (local-only, not serialized)

// Re-export generated wire types as the canonical response/stats types.
// These are produced by `generate_rpc_service!("worker")` in crate::generated::worker_client.
pub use crate::generated::worker_client::{
    // Generated client
    WorkerClient as GenWorkerClient,
    // Response types
    VersionInfo, RuntimeStatus, RuntimeCondition,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    // Stats types
    PodSandboxStats, PodSandboxAttributes,
    LinuxPodSandboxStats,
    ContainerStats, ContainerAttributes,
    CpuUsage, MemoryUsage, NetworkUsage,
    NetworkInterfaceUsage, ProcessUsage,
    FilesystemUsage, FilesystemIdentifier,
    // Common types
    KeyValue as GenKeyValue, Timestamp,
    // Info types for list responses
    PodSandboxInfo, ContainerInfo,
    // Scoped status types (generated)
    PodSandboxStatus, ContainerStatus,
    PodSandboxNetworkStatus,
    // Generated enum types
    PodSandboxStateEnum, ContainerStateEnum,
    // Request config types (for client RPC calls)
    PodSandboxConfig, ContainerConfig, ImageSpec,
    DNSConfig, PortMapping, LinuxPodSandboxConfig,
    LinuxSandboxSecurityContext,
    Mount, Device, LinuxContainerConfig,
    LinuxContainerSecurityContext, Capability,
    AuthConfig, StreamInfo,
    ImageInfo, ImageStatusResult,
};
// Note: PodSandboxMetadata, ContainerMetadata, KeyValue, and Filter types
// are hand-written below with extra functionality (serde, #[capnp(skip)], etc.)


/// CRI-aligned RuntimeClient trait
///
/// Client-side interface for CRI RuntimeService (`runtime.v1`) for future kubelet/crictl compatibility.
/// PodSandbox = Kata VM, Container = OCI container within VM.
///
/// Response types use generated `*Data` types directly for zero-conversion
/// cross-crate interoperability.
#[async_trait]
pub trait RuntimeClient: Send + Sync {
    // ─────────────────────────────────────────────────────────────────────
    // Runtime Information
    // ─────────────────────────────────────────────────────────────────────

    /// Get runtime version information
    async fn version(&self, version: &str) -> Result<VersionInfo>;

    /// Get runtime status
    async fn status(&self, verbose: bool) -> Result<StatusResponse>;

    // ─────────────────────────────────────────────────────────────────────
    // Pod Sandbox Lifecycle (Kata VM)
    // ─────────────────────────────────────────────────────────────────────

    /// Create and start a pod sandbox (Kata VM)
    async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> Result<String>;

    /// Stop a running pod sandbox
    async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;

    /// Remove a pod sandbox
    async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()>;

    /// Get pod sandbox status
    async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> Result<PodSandboxStatusResponse>;

    /// List pod sandboxes
    async fn list_pod_sandbox(
        &self,
        filter: Option<&PodSandboxFilter>,
    ) -> Result<Vec<PodSandboxInfo>>;

    // ─────────────────────────────────────────────────────────────────────
    // Container Lifecycle
    // ─────────────────────────────────────────────────────────────────────

    /// Create a container within a pod sandbox
    async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        sandbox_config: &PodSandboxConfig,
    ) -> Result<String>;

    /// Start a created container
    async fn start_container(&self, container_id: &str) -> Result<()>;

    /// Stop a running container
    async fn stop_container(&self, container_id: &str, timeout: i64) -> Result<()>;

    /// Remove a container
    async fn remove_container(&self, container_id: &str) -> Result<()>;

    /// Get container status
    async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> Result<ContainerStatusResponse>;

    /// List containers
    async fn list_containers(
        &self,
        filter: Option<&ContainerFilter>,
    ) -> Result<Vec<ContainerInfo>>;

    // ─────────────────────────────────────────────────────────────────────
    // Exec
    // ─────────────────────────────────────────────────────────────────────

    /// Execute a command in a container synchronously
    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> Result<ExecSyncResult>;

    // ─────────────────────────────────────────────────────────────────────
    // Stats
    // ─────────────────────────────────────────────────────────────────────

    /// Get pod sandbox stats
    async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> Result<PodSandboxStats>;

    /// List pod sandbox stats
    async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>>;

    /// Get container stats
    async fn container_stats(&self, container_id: &str) -> Result<ContainerStats>;

    /// List container stats
    async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Local Composite Types (not in schema, used internally)
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime status response (local composite, not in schema).
///
/// Bundles the generated RuntimeStatus with verbose diagnostic info.
/// The `info` HashMap is NOT serialized over the wire — it's only used locally.
#[derive(Debug, Clone)]
pub struct StatusResponse {
    /// Overall runtime status (generated wire type)
    pub status: RuntimeStatus,
    /// Additional info (if verbose) — local only, not serialized
    pub info: HashMap<String, String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata Types (used in runtime state — need serde derives)
// ─────────────────────────────────────────────────────────────────────────────

/// Pod sandbox metadata
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, ToCapnp, FromCapnp)]
#[capnp(worker_capnp::pod_sandbox_metadata)]
pub struct PodSandboxMetadata {
    /// Pod name
    pub name: String,
    /// Pod UID
    pub uid: String,
    /// Pod namespace
    pub namespace: String,
    /// Attempt number
    pub attempt: u32,
}

/// Container metadata
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, ToCapnp, FromCapnp)]
#[capnp(worker_capnp::container_metadata)]
pub struct ContainerMetadata {
    /// Container name
    pub name: String,
    /// Attempt number
    pub attempt: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Domain Types
// ─────────────────────────────────────────────────────────────────────────────

/// Key-value pair — alias for generated KeyValue (now has ToCapnp/FromCapnp).
pub type KeyValue = GenKeyValue;

// ─────────────────────────────────────────────────────────────────────────────
// Filter Types
// ─────────────────────────────────────────────────────────────────────────────

/// Filter for listing pod sandboxes
#[derive(Debug, Clone, Default, FromCapnp)]
#[capnp(worker_capnp::pod_sandbox_filter)]
pub struct PodSandboxFilter {
    /// Filter by sandbox ID
    pub id: Option<String>,
    /// Filter by state
    #[capnp(skip)]
    pub state: Option<super::PodSandboxState>,
    /// Filter by labels
    pub label_selector: Vec<KeyValue>,
}

/// Filter for listing containers
#[derive(Debug, Clone, Default, FromCapnp)]
#[capnp(worker_capnp::container_filter)]
pub struct ContainerFilter {
    /// Filter by container ID
    pub id: Option<String>,
    /// Filter by pod sandbox ID
    pub pod_sandbox_id: Option<String>,
    /// Filter by state
    #[capnp(skip)]
    pub state: Option<super::ContainerState>,
    /// Filter by labels
    pub label_selector: Vec<KeyValue>,
}

/// Filter for pod sandbox stats
#[derive(Debug, Clone, Default, FromCapnp)]
#[capnp(worker_capnp::pod_sandbox_stats_filter)]
pub struct PodSandboxStatsFilter {
    /// Filter by sandbox ID
    pub id: Option<String>,
    /// Filter by labels
    pub label_selector: Vec<KeyValue>,
}

/// Filter for container stats
#[derive(Debug, Clone, Default, FromCapnp)]
#[capnp(worker_capnp::container_stats_filter)]
pub struct ContainerStatsFilter {
    /// Filter by container ID
    pub id: Option<String>,
    /// Filter by pod sandbox ID
    pub pod_sandbox_id: Option<String>,
    /// Filter by labels
    pub label_selector: Vec<KeyValue>,
}


