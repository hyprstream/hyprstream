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
use std::path::PathBuf;
use std::sync::Arc;

use crate::worker_capnp;

use super::client::KeyValue;
// Use generated PodSandboxMetadata (matches what's in generated PodSandboxConfig/PodSandboxStatus)
use crate::generated::worker_client::PodSandboxMetadata;
use super::virtiofs::SandboxVirtiofs;

/// Pod sandbox state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PodSandboxState {
    /// Sandbox is ready
    SandboxReady,
    /// Sandbox is not ready (stopped)
    #[default]
    SandboxNotReady,
}

impl From<PodSandboxState> for worker_capnp::PodSandboxState {
    fn from(s: PodSandboxState) -> Self {
        match s {
            PodSandboxState::SandboxReady => worker_capnp::PodSandboxState::SandboxReady,
            PodSandboxState::SandboxNotReady => worker_capnp::PodSandboxState::SandboxNotReady,
        }
    }
}

impl From<worker_capnp::PodSandboxState> for PodSandboxState {
    fn from(s: worker_capnp::PodSandboxState) -> Self {
        match s {
            worker_capnp::PodSandboxState::SandboxReady => PodSandboxState::SandboxReady,
            worker_capnp::PodSandboxState::SandboxNotReady => PodSandboxState::SandboxNotReady,
        }
    }
}

// DTO types (PodSandboxConfig, DNSConfig, PortMapping, PodSandboxStatus, etc.)
// are now imported from generated types via super::client

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
    pub labels: Vec<KeyValue>,
    /// Annotations
    pub annotations: Vec<KeyValue>,
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
    /// Create a new pod sandbox from configuration (uses generated PodSandboxConfig)
    ///
    /// The sandbox is created in the NotReady state. Call `set_hypervisor()`
    /// and then start the VM to make it ready.
    pub fn new(id: String, config: &super::client::PodSandboxConfig, sandbox_path: PathBuf) -> Self {
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
        labels: Vec<KeyValue>,
        annotations: Vec<KeyValue>,
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
