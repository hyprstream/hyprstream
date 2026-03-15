//! Pod Sandbox types and lifecycle
//!
//! A PodSandbox represents an isolated execution environment that can host
//! multiple containers. The actual isolation mechanism is provided by a
//! `SandboxBackend` (Kata VM, systemd-nspawn, etc.).

use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::sync::Arc;

use super::backend::SandboxHandle;
use super::client::{KeyValue, PodSandboxState};
use crate::generated::worker_client::PodSandboxMetadata;

/// Runtime representation of a pod sandbox
///
/// The sandbox stores an opaque `backend_handle` from the `SandboxBackend`
/// that started it.  Callers can downcast to the concrete handle type
/// (e.g. `KataHandle`, `NspawnHandle`) via `as_any()`.
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
    /// Runtime handler (e.g., "kata", "nspawn")
    pub runtime_handler: String,

    // ─────────────────────────────────────────────────────────────────────────
    // Backend-managed state
    // ─────────────────────────────────────────────────────────────────────────

    /// Opaque handle from the `SandboxBackend` (holds hypervisor, PIDs, etc.)
    pub(crate) backend_handle: Option<Arc<dyn SandboxHandle>>,

    /// Path to sandbox runtime directory
    pub(crate) sandbox_path: PathBuf,

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
            backend_handle: self.backend_handle.clone(),
            sandbox_path: self.sandbox_path.clone(),
            image_id: self.image_id.clone(),
        }
    }
}

impl PodSandbox {
    /// Create a new pod sandbox from configuration
    ///
    /// The sandbox is created in the NotReady state. The `SandboxBackend`
    /// will populate `backend_handle` when it starts the sandbox.
    pub fn new(id: String, config: &super::client::PodSandboxConfig, sandbox_path: PathBuf) -> Self {
        Self {
            id,
            metadata: config.metadata.clone(),
            state: PodSandboxState::SandboxNotReady,
            created_at: Utc::now(),
            labels: config.labels.clone(),
            annotations: config.annotations.clone(),
            runtime_handler: "kata".to_owned(),
            backend_handle: None,
            sandbox_path,
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
            backend_handle: None,
            sandbox_path: PathBuf::new(),
            image_id: None,
        }
    }

    /// Check if sandbox is ready
    pub fn is_ready(&self) -> bool {
        self.state == PodSandboxState::SandboxReady
    }

    /// Get the backend handle (for downcasting to concrete type)
    pub fn backend_handle(&self) -> Option<&Arc<dyn SandboxHandle>> {
        self.backend_handle.as_ref()
    }

    /// Set the backend handle
    pub fn set_backend_handle(&mut self, handle: Arc<dyn SandboxHandle>) {
        self.backend_handle = Some(handle);
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
}
