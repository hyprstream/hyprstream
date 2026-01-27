//! RuntimeClient trait and ZMQ client
//!
//! Client-side trait for CRI RuntimeService (`runtime.v1`) for future kubelet compatibility.

use crate::error::Result;
use crate::workers_capnp;
use async_trait::async_trait;
use hyprstream_rpc::{FromCapnp, ToCapnp};
use std::collections::HashMap;

use super::{
    PodSandbox, PodSandboxConfig, PodSandboxStatus,
    Container, ContainerConfig, ContainerStatus,
};

/// CRI-aligned RuntimeClient trait
///
/// Client-side interface for CRI RuntimeService (`runtime.v1`) for future kubelet/crictl compatibility.
/// PodSandbox = Kata VM, Container = OCI container within VM.
#[async_trait]
pub trait RuntimeClient: Send + Sync {
    // ─────────────────────────────────────────────────────────────────────
    // Runtime Information
    // ─────────────────────────────────────────────────────────────────────

    /// Get runtime version information
    async fn version(&self, version: &str) -> Result<VersionResponse>;

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
    ) -> Result<Vec<PodSandbox>>;

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
    ) -> Result<Vec<Container>>;

    // ─────────────────────────────────────────────────────────────────────
    // Exec
    // ─────────────────────────────────────────────────────────────────────

    /// Execute a command in a container synchronously
    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> Result<ExecSyncResponse>;

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
// Response Types
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime version response
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::version_info)]
pub struct VersionResponse {
    /// Version of the CRI spec
    pub version: String,
    /// Name of the runtime
    #[capnp(rename = "runtime_name")]
    pub runtime_name: String,
    /// Version of the runtime
    #[capnp(rename = "runtime_version")]
    pub runtime_version: String,
    /// API version of the runtime
    #[capnp(rename = "runtime_api_version")]
    pub runtime_api_version: String,
}

/// Runtime status response
#[derive(Debug, Clone)]
pub struct StatusResponse {
    /// Overall runtime status
    pub status: RuntimeStatus,
    /// Additional info (if verbose)
    pub info: HashMap<String, String>,
}

/// Runtime status
#[derive(Debug, Clone)]
pub struct RuntimeStatus {
    /// Runtime conditions
    pub conditions: Vec<RuntimeCondition>,
}

/// Runtime condition
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::runtime_condition)]
pub struct RuntimeCondition {
    /// Type of condition
    #[capnp(rename = "condition_type")]
    pub condition_type: String,
    /// Status of condition
    pub status: bool,
    /// Reason for condition
    pub reason: String,
    /// Human-readable message
    pub message: String,
}

/// Pod sandbox status response
#[derive(Debug, Clone)]
pub struct PodSandboxStatusResponse {
    /// Status of the pod sandbox
    pub status: PodSandboxStatus,
    /// Additional info (if verbose)
    pub info: HashMap<String, String>,
}

/// Container status response
#[derive(Debug, Clone)]
pub struct ContainerStatusResponse {
    /// Status of the container
    pub status: ContainerStatus,
    /// Additional info (if verbose)
    pub info: HashMap<String, String>,
}

/// Exec sync response
#[derive(Debug, Clone)]
pub struct ExecSyncResponse {
    /// Stdout output
    pub stdout: Vec<u8>,
    /// Stderr output
    pub stderr: Vec<u8>,
    /// Exit code
    pub exit_code: i32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter Types
// ─────────────────────────────────────────────────────────────────────────────

/// Filter for listing pod sandboxes
#[derive(Debug, Clone, Default)]
pub struct PodSandboxFilter {
    /// Filter by sandbox ID
    pub id: Option<String>,
    /// Filter by state
    pub state: Option<super::PodSandboxState>,
    /// Filter by labels
    pub label_selector: HashMap<String, String>,
}

/// Filter for listing containers
#[derive(Debug, Clone, Default)]
pub struct ContainerFilter {
    /// Filter by container ID
    pub id: Option<String>,
    /// Filter by pod sandbox ID
    pub pod_sandbox_id: Option<String>,
    /// Filter by state
    pub state: Option<super::ContainerState>,
    /// Filter by labels
    pub label_selector: HashMap<String, String>,
}

/// Filter for pod sandbox stats
#[derive(Debug, Clone, Default)]
pub struct PodSandboxStatsFilter {
    /// Filter by sandbox ID
    pub id: Option<String>,
    /// Filter by labels
    pub label_selector: HashMap<String, String>,
}

/// Filter for container stats
#[derive(Debug, Clone, Default)]
pub struct ContainerStatsFilter {
    /// Filter by container ID
    pub id: Option<String>,
    /// Filter by pod sandbox ID
    pub pod_sandbox_id: Option<String>,
    /// Filter by labels
    pub label_selector: HashMap<String, String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats Types
// ─────────────────────────────────────────────────────────────────────────────

/// Pod sandbox stats
#[derive(Debug, Clone)]
pub struct PodSandboxStats {
    /// Sandbox attributes
    pub attributes: PodSandboxAttributes,
    /// Linux-specific stats
    pub linux: Option<LinuxPodSandboxStats>,
}

/// Pod sandbox attributes for stats
#[derive(Debug, Clone)]
pub struct PodSandboxAttributes {
    /// Sandbox ID
    pub id: String,
    /// Sandbox metadata
    pub metadata: PodSandboxMetadata,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// Pod sandbox metadata
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::pod_sandbox_metadata)]
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

/// Linux-specific pod sandbox stats
#[derive(Debug, Clone)]
pub struct LinuxPodSandboxStats {
    /// CPU stats
    pub cpu: Option<CpuUsage>,
    /// Memory stats
    pub memory: Option<MemoryUsage>,
    /// Network stats
    pub network: Option<NetworkUsage>,
    /// Process count
    pub process: Option<ProcessUsage>,
}

/// Container stats
#[derive(Debug, Clone)]
pub struct ContainerStats {
    /// Container attributes
    pub attributes: ContainerAttributes,
    /// CPU stats
    pub cpu: Option<CpuUsage>,
    /// Memory stats
    pub memory: Option<MemoryUsage>,
    /// Writable layer stats
    pub writable_layer: Option<FilesystemUsage>,
}

/// Container attributes for stats
#[derive(Debug, Clone)]
pub struct ContainerAttributes {
    /// Container ID
    pub id: String,
    /// Container metadata
    pub metadata: ContainerMetadata,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// Container metadata
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::container_metadata)]
pub struct ContainerMetadata {
    /// Container name
    pub name: String,
    /// Attempt number
    pub attempt: u32,
}

/// CPU usage stats
#[derive(Debug, Clone)]
pub struct CpuUsage {
    /// Timestamp in nanoseconds
    pub timestamp: i64,
    /// Total CPU usage in nanoseconds
    pub usage_core_nano_seconds: u64,
    /// Per-core CPU usage
    pub usage_nano_cores: u64,
}

/// Memory usage stats
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Timestamp in nanoseconds
    pub timestamp: i64,
    /// Working set in bytes
    pub working_set_bytes: u64,
    /// Available bytes
    pub available_bytes: u64,
    /// Usage bytes
    pub usage_bytes: u64,
    /// RSS bytes
    pub rss_bytes: u64,
    /// Page faults
    pub page_faults: u64,
    /// Major page faults
    pub major_page_faults: u64,
}

/// Network usage stats
#[derive(Debug, Clone)]
pub struct NetworkUsage {
    /// Timestamp in nanoseconds
    pub timestamp: i64,
    /// Default interface stats
    pub default_interface: Option<NetworkInterfaceUsage>,
    /// All interfaces
    pub interfaces: Vec<NetworkInterfaceUsage>,
}

/// Network interface usage
#[derive(Debug, Clone)]
pub struct NetworkInterfaceUsage {
    /// Interface name
    pub name: String,
    /// Bytes received
    pub rx_bytes: u64,
    /// Bytes transmitted
    pub tx_bytes: u64,
    /// Receive errors
    pub rx_errors: u64,
    /// Transmit errors
    pub tx_errors: u64,
}

/// Process usage stats
#[derive(Debug, Clone)]
pub struct ProcessUsage {
    /// Timestamp in nanoseconds
    pub timestamp: i64,
    /// Process count
    pub process_count: u64,
}

/// Filesystem usage stats
#[derive(Debug, Clone)]
pub struct FilesystemUsage {
    /// Timestamp in nanoseconds
    pub timestamp: i64,
    /// Filesystem ID
    pub fs_id: Option<FilesystemIdentifier>,
    /// Used bytes
    pub used_bytes: u64,
    /// Inode usage
    pub inodes_used: u64,
}

/// Filesystem identifier
#[derive(Debug, Clone)]
pub struct FilesystemIdentifier {
    /// Mountpoint
    pub mountpoint: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// ZMQ Client Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// ZMQ client for RuntimeClient
///
/// Implements RuntimeClient trait via ZMQ REQ/REP to WorkerService.
pub struct RuntimeZmq {
    // TODO: Add ZmqClient from hyprstream core
    _endpoint: String,
}

impl RuntimeZmq {
    /// Create a new RuntimeZmq client
    pub fn new(endpoint: &str) -> Self {
        Self {
            _endpoint: endpoint.to_owned(),
        }
    }
}

#[async_trait]
impl RuntimeClient for RuntimeZmq {
    async fn version(&self, _version: &str) -> Result<VersionResponse> {
        todo!("Implement ZMQ call")
    }

    async fn status(&self, _verbose: bool) -> Result<StatusResponse> {
        todo!("Implement ZMQ call")
    }

    async fn run_pod_sandbox(&self, _config: &PodSandboxConfig) -> Result<String> {
        todo!("Implement ZMQ call")
    }

    async fn stop_pod_sandbox(&self, _pod_sandbox_id: &str) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn remove_pod_sandbox(&self, _pod_sandbox_id: &str) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn pod_sandbox_status(
        &self,
        _pod_sandbox_id: &str,
        _verbose: bool,
    ) -> Result<PodSandboxStatusResponse> {
        todo!("Implement ZMQ call")
    }

    async fn list_pod_sandbox(
        &self,
        _filter: Option<&PodSandboxFilter>,
    ) -> Result<Vec<PodSandbox>> {
        todo!("Implement ZMQ call")
    }

    async fn create_container(
        &self,
        _pod_sandbox_id: &str,
        _config: &ContainerConfig,
        _sandbox_config: &PodSandboxConfig,
    ) -> Result<String> {
        todo!("Implement ZMQ call")
    }

    async fn start_container(&self, _container_id: &str) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn stop_container(&self, _container_id: &str, _timeout: i64) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn remove_container(&self, _container_id: &str) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn container_status(
        &self,
        _container_id: &str,
        _verbose: bool,
    ) -> Result<ContainerStatusResponse> {
        todo!("Implement ZMQ call")
    }

    async fn list_containers(
        &self,
        _filter: Option<&ContainerFilter>,
    ) -> Result<Vec<Container>> {
        todo!("Implement ZMQ call")
    }

    async fn exec_sync(
        &self,
        _container_id: &str,
        _cmd: &[String],
        _timeout: i64,
    ) -> Result<ExecSyncResponse> {
        todo!("Implement ZMQ call")
    }

    async fn pod_sandbox_stats(&self, _pod_sandbox_id: &str) -> Result<PodSandboxStats> {
        todo!("Implement ZMQ call")
    }

    async fn list_pod_sandbox_stats(
        &self,
        _filter: Option<&PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>> {
        todo!("Implement ZMQ call")
    }

    async fn container_stats(&self, _container_id: &str) -> Result<ContainerStats> {
        todo!("Implement ZMQ call")
    }

    async fn list_container_stats(
        &self,
        _filter: Option<&ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>> {
        todo!("Implement ZMQ call")
    }
}
