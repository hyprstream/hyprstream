//! Pod Sandbox types and lifecycle
//!
//! A PodSandbox represents an isolated execution environment that can host
//! multiple containers. The actual isolation mechanism is provided by a
//! `SandboxBackend` (Kata VM, systemd-nspawn, etc.).

use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::sync::Arc;

use super::admission::ReservationRecord;
use super::backend::SandboxHandle;
use super::client::{KeyValue, LinuxContainerResources, PodSandboxState};
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

    /// Serial console UNIX socket path — set by the backend when the VM starts.
    /// For Cloud Hypervisor: path to the API socket (or a dedicated serial socket).
    /// For nspawn: path to the console socket (if enabled).
    pub(crate) console_socket: Option<PathBuf>,

    /// The `LinuxContainerResources` last actually applied to this sandbox by
    /// the backend (via `start()` or `update_resources()`) — the physical
    /// truth, independent of any pool bookkeeping. `Default` (all-zero /
    /// CRI-"unspecified") for a freshly-booted warm sandbox that has never
    /// had a caller's resources applied.
    ///
    /// #519 fix: `SandboxPool::acquire` compares an incoming request's
    /// `config.linux.resources` against this field before handing out a
    /// warm sandbox, and calls `update_resources` (updating this field in
    /// turn) on a mismatch — instead of silently reusing whatever size the
    /// sandbox happened to be booted with. Deliberately **not** reset when a
    /// sandbox is returned to the warm pool (`reset_sandbox`): resetting it
    /// to `Default` would be a lie (the backend's `reset()` does not resize
    /// the sandbox back down), and the next `acquire()`'s comparison depends
    /// on this reflecting the sandbox's actual current size.
    pub(crate) applied_resources: LinuxContainerResources,

    /// The admission reservation (#525 P2) this sandbox is holding, if it was
    /// obtained through `SandboxPool::acquire`'s admission-controlled path.
    /// `None` for sandboxes constructed outside that path (e.g. `from_info`,
    /// or a warm sandbox that has not yet been claimed). `SandboxPool::release`
    /// gives this back to the `AdmissionTracker` so per-Subject/per-group
    /// counters and resource reservations are released precisely.
    pub(crate) reservation: Option<ReservationRecord>,
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
            console_socket: self.console_socket.clone(),
            applied_resources: self.applied_resources.clone(),
            reservation: self.reservation.clone(),
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
            console_socket: None,
            applied_resources: LinuxContainerResources::default(),
            reservation: None,
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
            console_socket: None,
            applied_resources: LinuxContainerResources::default(),
            reservation: None,
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

    /// Set the image id this sandbox's tenant VFS is composed from.
    ///
    /// Normally populated by the worker runtime from the `CreateSandbox` RPC.
    /// Exposed (doc-hidden) as the seam an out-of-crate boot harness (#721)
    /// needs so that [`KataBackend::start`](super::kata_backend::KataBackend)
    /// composes + serves + attaches the per-sandbox tenant VFS as the guest's
    /// virtio-fs rootfs share — instead of booting the bare VM rootfs with no
    /// container filesystem. When unset, `start()` attaches no share and the
    /// container has no rootfs (the failure #721 fixes).
    #[doc(hidden)]
    pub fn set_image_id(&mut self, image_id: impl Into<String>) {
        self.image_id = Some(image_id.into());
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

    /// Path to the VM serial console UNIX socket, if set by the backend.
    pub fn console_socket(&self) -> Option<&std::path::Path> {
        self.console_socket.as_deref()
    }
}
