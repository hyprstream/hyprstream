//! Container types and lifecycle
//!
//! A Container runs within a PodSandbox (Kata VM).
//! Maps directly to Kubernetes CRI Container concept.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::client::ContainerMetadata;

/// Container state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ContainerState {
    /// Container is being created
    ContainerCreated,
    /// Container is running
    ContainerRunning,
    /// Container has exited
    ContainerExited,
    /// Unknown state
    #[default]
    ContainerUnknown,
}

/// Container configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Container metadata
    pub metadata: ContainerMetadata,

    /// Image to use
    pub image: ImageSpec,

    /// Command to run
    pub command: Vec<String>,

    /// Arguments to command
    pub args: Vec<String>,

    /// Working directory
    pub working_dir: String,

    /// Environment variables
    pub envs: Vec<KeyValue>,

    /// Mounts
    pub mounts: Vec<Mount>,

    /// Devices
    pub devices: Vec<Device>,

    /// Labels
    pub labels: HashMap<String, String>,

    /// Annotations
    pub annotations: HashMap<String, String>,

    /// Log path
    pub log_path: String,

    /// Stdin enabled
    pub stdin: bool,

    /// Stdin once
    pub stdin_once: bool,

    /// TTY enabled
    pub tty: bool,

    /// Linux-specific configuration
    pub linux: Option<LinuxContainerConfig>,
}

/// Image specification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageSpec {
    /// Image reference (e.g., "docker.io/library/alpine:latest")
    pub image: String,

    /// Image annotations
    pub annotations: HashMap<String, String>,

    /// Runtime handler hint
    pub runtime_handler: String,
}

/// Key-value pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValue {
    /// Key
    pub key: String,
    /// Value
    pub value: String,
}

/// Mount configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    /// Container path
    pub container_path: String,

    /// Host path
    pub host_path: String,

    /// Read-only mount
    pub readonly: bool,

    /// SELinux relabel
    pub selinux_relabel: bool,

    /// Propagation mode
    pub propagation: MountPropagation,
}

/// Mount propagation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[allow(clippy::enum_variant_names)] // Wire format compatibility (PROPAGATION_PRIVATE, etc.)
pub enum MountPropagation {
    /// Private propagation
    #[default]
    PropagationPrivate,
    /// Host to container propagation
    PropagationHostToContainer,
    /// Bidirectional propagation
    PropagationBidirectional,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    /// Container path
    pub container_path: String,

    /// Host path
    pub host_path: String,

    /// Permissions (e.g., "rwm")
    pub permissions: String,
}

/// Linux-specific container configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxContainerConfig {
    /// Resource limits
    pub resources: Option<LinuxContainerResources>,

    /// Security context
    pub security_context: Option<LinuxContainerSecurityContext>,
}

/// Linux container resources (mirrors CRI LinuxContainerResources)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxContainerResources {
    /// CPU period in microseconds
    pub cpu_period: i64,

    /// CPU quota in microseconds
    pub cpu_quota: i64,

    /// CPU shares (relative weight)
    pub cpu_shares: i64,

    /// Memory limit in bytes
    pub memory_limit_in_bytes: i64,

    /// OOM score adjustment
    pub oom_score_adj: i64,

    /// CPU set CPUs (e.g., "0-2,6")
    pub cpuset_cpus: String,

    /// CPU set memory nodes
    pub cpuset_mems: String,

    /// Hugepage limits
    pub hugepage_limits: Vec<HugepageLimit>,

    /// Unified cgroup resources
    pub unified: HashMap<String, String>,

    /// Memory swap limit
    pub memory_swap_limit_in_bytes: i64,
}

/// Hugepage limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HugepageLimit {
    /// Page size (e.g., "2MB", "1GB")
    pub page_size: String,
    /// Limit in bytes
    pub limit: u64,
}

/// Linux container security context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxContainerSecurityContext {
    /// User capabilities to add
    pub capabilities: Option<Capability>,

    /// Privileged mode
    pub privileged: bool,

    /// Namespace options
    pub namespace_options: Option<NamespaceOption>,

    /// SELinux options
    pub selinux_options: Option<SELinuxOption>,

    /// Run as user
    pub run_as_user: Option<Int64Value>,

    /// Run as group
    pub run_as_group: Option<Int64Value>,

    /// Run as username
    pub run_as_username: String,

    /// Read-only root filesystem
    pub readonly_rootfs: bool,

    /// Supplemental groups
    pub supplemental_groups: Vec<i64>,

    /// AppArmor profile
    pub apparmor_profile: String,

    /// Seccomp profile path
    pub seccomp_profile_path: String,

    /// No new privileges
    pub no_new_privs: bool,

    /// Masked paths
    pub masked_paths: Vec<String>,

    /// Readonly paths
    pub readonly_paths: Vec<String>,

    /// Seccomp profile
    pub seccomp: Option<SecurityProfile>,

    /// AppArmor profile
    pub apparmor: Option<SecurityProfile>,
}

/// Linux capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capability {
    /// Add capabilities
    pub add_capabilities: Vec<String>,
    /// Drop capabilities
    pub drop_capabilities: Vec<String>,
}

// Re-use types from sandbox module
use super::sandbox::{Int64Value, NamespaceOption, SELinuxOption, SecurityProfile};

/// Container status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStatus {
    /// Container ID
    pub id: String,

    /// Container metadata
    pub metadata: ContainerMetadata,

    /// Current state
    pub state: ContainerState,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Start timestamp
    pub started_at: Option<DateTime<Utc>>,

    /// Finish timestamp
    pub finished_at: Option<DateTime<Utc>>,

    /// Exit code (if exited)
    pub exit_code: i32,

    /// Image reference
    pub image: ImageSpec,

    /// Image reference (digest)
    pub image_ref: String,

    /// Reason for state
    pub reason: String,

    /// Human-readable message
    pub message: String,

    /// Labels
    pub labels: HashMap<String, String>,

    /// Annotations
    pub annotations: HashMap<String, String>,

    /// Mounts
    pub mounts: Vec<Mount>,

    /// Log path
    pub log_path: String,
}

/// Runtime representation of a container
#[derive(Debug, Clone)]
pub struct Container {
    /// Unique container ID
    pub id: String,

    /// Pod sandbox ID this container belongs to
    pub pod_sandbox_id: String,

    /// Container metadata
    pub metadata: ContainerMetadata,

    /// Current state
    pub state: ContainerState,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Image reference
    pub image: ImageSpec,

    /// Labels
    pub labels: HashMap<String, String>,

    /// Annotations
    pub annotations: HashMap<String, String>,

    // Internal fields for process management
    /// Container process ID (if running)
    pub(crate) pid: Option<u32>,

    /// Exit code (if exited)
    pub(crate) exit_code: Option<i32>,
}

impl Container {
    /// Create a new container from configuration
    pub fn new(id: String, pod_sandbox_id: String, config: &ContainerConfig) -> Self {
        Self {
            id,
            pod_sandbox_id,
            metadata: config.metadata.clone(),
            state: ContainerState::ContainerCreated,
            created_at: Utc::now(),
            image: config.image.clone(),
            labels: config.labels.clone(),
            annotations: config.annotations.clone(),
            pid: None,
            exit_code: None,
        }
    }

    /// Create a container from RPC response info
    ///
    /// Used when parsing list_containers response from WorkerService.
    pub fn from_info(
        id: String,
        pod_sandbox_id: String,
        metadata: ContainerMetadata,
        image: ImageSpec,
        state: ContainerState,
        created_at: DateTime<Utc>,
        labels: HashMap<String, String>,
        annotations: HashMap<String, String>,
    ) -> Self {
        Self {
            id,
            pod_sandbox_id,
            metadata,
            state,
            created_at,
            image,
            labels,
            annotations,
            pid: None,
            exit_code: None,
        }
    }

    /// Check if container is running
    pub fn is_running(&self) -> bool {
        self.state == ContainerState::ContainerRunning
    }

    /// Get container status
    pub fn status(&self) -> ContainerStatus {
        ContainerStatus {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            state: self.state,
            created_at: self.created_at,
            started_at: None, // TODO: Track start time
            finished_at: None,
            exit_code: self.exit_code.unwrap_or(0),
            image: self.image.clone(),
            image_ref: String::new(),
            reason: String::new(),
            message: String::new(),
            labels: self.labels.clone(),
            annotations: self.annotations.clone(),
            mounts: Vec::new(),
            log_path: String::new(),
        }
    }
}
