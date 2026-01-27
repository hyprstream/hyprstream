//! Pod Sandbox (Kata VM) types and lifecycle
//!
//! A PodSandbox represents a Kata VM that can host multiple containers.
//! Maps directly to Kubernetes CRI PodSandbox concept.
//!
//! The sandbox uses the Kata Containers `Hypervisor` trait for VM management,
//! providing support for multiple hypervisors (Cloud Hypervisor, QEMU, Dragonball).

use chrono::{DateTime, Utc};
use kata_hypervisor::Hypervisor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::client::PodSandboxMetadata;
use super::virtiofs::SandboxVirtiofs;

/// Pod sandbox state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PodSandboxState {
    /// Sandbox is ready
    SandboxReady,
    /// Sandbox is not ready (stopped)
    SandboxNotReady,
}

impl Default for PodSandboxState {
    fn default() -> Self {
        Self::SandboxNotReady
    }
}

/// Pod sandbox configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PodSandboxConfig {
    /// Sandbox metadata
    pub metadata: PodSandboxMetadata,

    /// Hostname of the sandbox
    pub hostname: String,

    /// Directory for logs
    pub log_directory: String,

    /// DNS configuration
    pub dns_config: Option<DNSConfig>,

    /// Port mappings
    pub port_mappings: Vec<PortMapping>,

    /// Labels
    pub labels: HashMap<String, String>,

    /// Annotations
    pub annotations: HashMap<String, String>,

    /// Linux-specific configuration
    pub linux: Option<LinuxPodSandboxConfig>,
}

/// DNS configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DNSConfig {
    /// DNS servers
    pub servers: Vec<String>,
    /// DNS search domains
    pub searches: Vec<String>,
    /// DNS options
    pub options: Vec<String>,
}

/// Port mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Protocol (TCP, UDP, SCTP)
    pub protocol: Protocol,
    /// Container port
    pub container_port: i32,
    /// Host port
    pub host_port: i32,
    /// Host IP
    pub host_ip: String,
}

/// Network protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Protocol {
    Tcp,
    Udp,
    Sctp,
}

impl Default for Protocol {
    fn default() -> Self {
        Self::Tcp
    }
}

/// Linux-specific pod sandbox configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxPodSandboxConfig {
    /// Cgroup parent
    pub cgroup_parent: String,
    /// Security context
    pub security_context: Option<LinuxSandboxSecurityContext>,
    /// Sysctls
    pub sysctls: HashMap<String, String>,
    /// Overhead (resources for pod infrastructure)
    pub overhead: Option<LinuxContainerResources>,
    /// Resources for the sandbox
    pub resources: Option<LinuxContainerResources>,
}

/// Linux sandbox security context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxSandboxSecurityContext {
    /// Namespace options
    pub namespace_options: Option<NamespaceOption>,
    /// SELinux options
    pub selinux_options: Option<SELinuxOption>,
    /// Run as user
    pub run_as_user: Option<Int64Value>,
    /// Run as group
    pub run_as_group: Option<Int64Value>,
    /// Read-only root filesystem
    pub readonly_rootfs: bool,
    /// Supplemental groups
    pub supplemental_groups: Vec<i64>,
    /// Privileged mode
    pub privileged: bool,
    /// Seccomp profile
    pub seccomp: Option<SecurityProfile>,
    /// AppArmor profile
    pub apparmor: Option<SecurityProfile>,
}

/// Namespace options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceOption {
    /// Network namespace mode
    pub network: NamespaceMode,
    /// PID namespace mode
    pub pid: NamespaceMode,
    /// IPC namespace mode
    pub ipc: NamespaceMode,
    /// Target container ID for sharing
    pub target_id: String,
    /// User namespace options
    pub user_nsmode: NamespaceMode,
}

/// Namespace mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NamespaceMode {
    /// Use pod namespace
    #[default]
    Pod,
    /// Use container namespace
    Container,
    /// Use node namespace
    Node,
    /// Target container namespace
    Target,
}

/// SELinux option
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SELinuxOption {
    /// User label
    pub user: String,
    /// Role label
    pub role: String,
    /// Type label
    pub type_label: String,
    /// Level label
    pub level: String,
}

/// Security profile
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityProfile {
    /// Profile type
    pub profile_type: SecurityProfileType,
    /// Localhost reference
    pub localhost_ref: String,
}

/// Security profile type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SecurityProfileType {
    /// Runtime default
    #[default]
    RuntimeDefault,
    /// Unconfined
    Unconfined,
    /// Localhost
    Localhost,
}

/// Linux container resources
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxContainerResources {
    /// CPU period
    pub cpu_period: i64,
    /// CPU quota
    pub cpu_quota: i64,
    /// CPU shares
    pub cpu_shares: i64,
    /// Memory limit in bytes
    pub memory_limit_in_bytes: i64,
    /// OOM score adjust
    pub oom_score_adj: i64,
    /// CPU set CPUs
    pub cpuset_cpus: String,
    /// CPU set MEMs
    pub cpuset_mems: String,
    /// Huge page limits
    pub hugepage_limits: Vec<HugepageLimit>,
    /// Unified cgroup resources
    pub unified: HashMap<String, String>,
    /// Memory swap limit
    pub memory_swap_limit_in_bytes: i64,
}

/// Hugepage limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HugepageLimit {
    /// Page size
    pub page_size: String,
    /// Limit
    pub limit: u64,
}

/// Int64 value wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int64Value {
    /// Value
    pub value: i64,
}

/// Pod sandbox status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSandboxStatus {
    /// Sandbox ID
    pub id: String,
    /// Sandbox metadata
    pub metadata: PodSandboxMetadata,
    /// Current state
    pub state: PodSandboxState,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Network information
    pub network: Option<PodSandboxNetworkStatus>,
    /// Linux-specific status
    pub linux: Option<LinuxPodSandboxStatus>,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
    /// Runtime handler
    pub runtime_handler: String,
}

/// Pod sandbox network status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PodSandboxNetworkStatus {
    /// IP address
    pub ip: String,
    /// Additional IPs
    pub additional_ips: Vec<PodIP>,
}

/// Pod IP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodIP {
    /// IP address
    pub ip: String,
}

/// Linux pod sandbox status
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinuxPodSandboxStatus {
    /// Namespaces
    pub namespaces: Option<Namespace>,
}

/// Namespace information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Namespace {
    /// Network namespace options
    pub options: Option<NamespaceOption>,
}

/// Runtime representation of a pod sandbox (Kata VM)
///
/// The sandbox owns a hypervisor handle from Kata's runtime-rs which manages
/// the VM lifecycle. This provides support for multiple hypervisors:
/// - Cloud Hypervisor
/// - QEMU
/// - Dragonball
/// - Firecracker
#[derive(Debug)]
pub struct PodSandbox {
    /// Unique sandbox ID
    pub id: String,
    /// Sandbox metadata
    pub metadata: PodSandboxMetadata,
    /// Current state
    pub state: PodSandboxState,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
    /// Runtime handler (e.g., "kata-clh", "kata-qemu")
    pub runtime_handler: String,

    // ─────────────────────────────────────────────────────────────────────────
    // VM Management (via Kata runtime-rs Hypervisor trait)
    // ─────────────────────────────────────────────────────────────────────────

    /// Hypervisor handle for VM lifecycle management
    /// Uses Kata's Hypervisor trait which abstracts Cloud Hypervisor, QEMU, etc.
    pub(crate) hypervisor: Option<Arc<dyn Hypervisor>>,

    /// Path to sandbox runtime directory
    pub(crate) sandbox_path: PathBuf,

    /// Path to VM API socket (for hypervisor communication)
    pub(crate) api_socket: Option<PathBuf>,

    // ─────────────────────────────────────────────────────────────────────────
    // Filesystem (Nydus RAFS via virtiofs)
    // ─────────────────────────────────────────────────────────────────────────

    /// Path to virtiofs socket (for shared filesystem)
    pub(crate) virtiofs_socket: Option<PathBuf>,

    /// VirtioFS daemon serving RAFS to this VM
    pub(crate) virtiofs_daemon: Option<Arc<SandboxVirtiofs>>,

    /// Image ID being served to this sandbox
    pub(crate) image_id: Option<String>,
}

impl Clone for PodSandbox {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            state: self.state,
            created_at: self.created_at,
            labels: self.labels.clone(),
            annotations: self.annotations.clone(),
            runtime_handler: self.runtime_handler.clone(),
            hypervisor: self.hypervisor.clone(),
            sandbox_path: self.sandbox_path.clone(),
            api_socket: self.api_socket.clone(),
            virtiofs_socket: self.virtiofs_socket.clone(),
            virtiofs_daemon: self.virtiofs_daemon.clone(),
            image_id: self.image_id.clone(),
        }
    }
}

impl PodSandbox {
    /// Create a new pod sandbox from configuration
    ///
    /// The sandbox is created in the NotReady state. Call `set_hypervisor()`
    /// and then start the VM to make it ready.
    pub fn new(id: String, config: &PodSandboxConfig, sandbox_path: PathBuf) -> Self {
        Self {
            id,
            metadata: config.metadata.clone(),
            state: PodSandboxState::SandboxNotReady,
            created_at: Utc::now(),
            labels: config.labels.clone(),
            annotations: config.annotations.clone(),
            runtime_handler: "kata".to_owned(),
            hypervisor: None,
            sandbox_path,
            api_socket: None,
            virtiofs_socket: None,
            virtiofs_daemon: None,
            image_id: None,
        }
    }

    /// Create a pod sandbox from RPC response info
    ///
    /// Used when parsing list_pod_sandbox response from WorkerService.
    pub fn from_info(
        id: String,
        metadata: PodSandboxMetadata,
        state: PodSandboxState,
        created_at: DateTime<Utc>,
        labels: HashMap<String, String>,
        annotations: HashMap<String, String>,
        runtime_handler: String,
    ) -> Self {
        Self {
            id,
            metadata,
            state,
            created_at,
            labels,
            annotations,
            runtime_handler,
            hypervisor: None,
            sandbox_path: PathBuf::new(),
            api_socket: None,
            virtiofs_socket: None,
            virtiofs_daemon: None,
            image_id: None,
        }
    }

    /// Check if sandbox is ready
    pub fn is_ready(&self) -> bool {
        self.state == PodSandboxState::SandboxReady
    }

    /// Get the hypervisor handle
    pub fn hypervisor(&self) -> Option<&Arc<dyn Hypervisor>> {
        self.hypervisor.as_ref()
    }

    /// Set the hypervisor handle
    pub fn set_hypervisor(&mut self, hypervisor: Arc<dyn Hypervisor>) {
        self.hypervisor = Some(hypervisor);
    }

    /// Get VM process IDs from the hypervisor
    ///
    /// Returns the PIDs of the hypervisor process(es) if the VM is running.
    pub async fn get_pids(&self) -> Option<Vec<u32>> {
        if let Some(ref hypervisor) = self.hypervisor {
            hypervisor.get_pids().await.ok()
        } else {
            None
        }
    }

    /// Mark sandbox as ready
    pub fn mark_ready(&mut self) {
        self.state = PodSandboxState::SandboxReady;
    }

    /// Mark sandbox as not ready
    pub fn mark_not_ready(&mut self) {
        self.state = PodSandboxState::SandboxNotReady;
    }

    /// Get sandbox status
    pub fn status(&self) -> PodSandboxStatus {
        PodSandboxStatus {
            id: self.id.clone(),
            metadata: self.metadata.clone(),
            state: self.state,
            created_at: self.created_at,
            network: None, // TODO: Populate from VM
            linux: None,
            labels: self.labels.clone(),
            annotations: self.annotations.clone(),
            runtime_handler: self.runtime_handler.clone(),
        }
    }

    /// Get the sandbox runtime directory path
    pub fn sandbox_path(&self) -> &PathBuf {
        &self.sandbox_path
    }

    /// Get the API socket path
    pub fn api_socket(&self) -> Option<&PathBuf> {
        self.api_socket.as_ref()
    }

    /// Set the API socket path
    pub fn set_api_socket(&mut self, path: PathBuf) {
        self.api_socket = Some(path);
    }
}
