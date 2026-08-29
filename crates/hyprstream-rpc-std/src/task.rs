//! Standard Task contract for the existing Worker `/exec` projection.
//!
//! This is an in-process service contract, not a plaintext authority wire
//! schema. A [`TaskAttachmentBinding`] is derived from an opaque, scoped
//! attachment operation grant before a request reaches this API; it cannot be
//! rebuilt from a caller-supplied subject, attachment id, or generation.
//! Task-result association metadata intentionally remains in-process for this
//! spike. It is emitted by a trusted TaskService but is not self-authenticating
//! provenance: its public fields can be constructed by another local caller.
//! A future wire projection must use an explicit output-only audit DTO, never
//! deserialization into an attachment or authority-generation record.

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use hyprstream_rpc::{
    AttachmentError, AttachmentId, AttachmentOperation, AttachmentOperationGrant,
    AuthorityGeneration, Subject,
};
use hyprstream_vfs::AdmittedNamespace;

/// Opaque worker task identifier.  It names a task but grants no authority to
/// it; every service operation also carries a [`TaskAttachmentBinding`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(String);

impl TaskId {
    /// Adapt an already-allocated worker instance id to the standard contract.
    /// Worker allocation is the authority-checked operation; this conversion
    /// never creates a task by itself.
    #[must_use]
    pub fn from_worker_instance(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Stable textual form for the `/exec/instances/<id>` projection.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Namespace commitment recorded by a Task spawn.
///
/// The standard TaskService prototype can still construct this from
/// caller-supplied description bytes, which is asserted metadata only. The
/// attachment-bound `/exec/clone` path instead derives it from an
/// [`AdmittedNamespace`] that is forked and delivered before the Task is made
/// visible. These representations intentionally share a wire digest while the
/// unadmitted TaskService path remains a separately documented limitation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceManifestDigest([u8; 32]);

impl NamespaceManifestDigest {
    /// Hash caller-supplied bytes at the legacy contract-prototype boundary.
    ///
    /// This is an asserted metadata commitment only. The current Worker pool
    /// neither derives a `Namespace`/effective mount description from these
    /// bytes nor passes one to `PodSandboxConfig`, so it is not proof of an
    /// admitted or effective sandbox namespace.
    #[must_use]
    pub fn from_description_bytes(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// Raw digest bytes for the eventual Task wire representation.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Project the digest of a trusted, frozen effective namespace into the Task
/// contract. This conversion is deliberately from [`AdmittedNamespace`], not
/// caller-supplied canonical bytes: the same object that supplied this digest
/// is forked and delivered to the Worker sandbox.
impl From<&AdmittedNamespace> for NamespaceManifestDigest {
    fn from(namespace: &AdmittedNamespace) -> Self {
        Self(*namespace.digest())
    }
}

/// Digest and byte length of content observed at a Task boundary.
///
/// This is content identity, not a materialized or retrievable artifact. The
/// spike does not persist these bytes in CAS; callers needing retrieval must
/// arrange durable materialization through a future artifact boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentDigest {
    digest: [u8; 32],
    byte_len: u64,
}

/// The role content played in a Task result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskContentRole {
    Stdout,
    Stderr,
    Input,
}

/// Content identity plus Task association metadata emitted by a Task service.
///
/// [`ContentDigest`] alone deliberately identifies a byte sequence only and
/// does not promise that its bytes remain available. A trusted Task service
/// may associate its observation with a Task, attachment generation, recorded
/// namespace commitment, and role. For `Stdout`/`Stderr`, `task` is the
/// emitting Task; for `Input`, it is the consuming/admitting Task and does
/// not assert upstream production. The fields are public, so this value is not
/// sealed, signed, or independently verifiable evidence. It intentionally has
/// no serde wire representation: accepting a remote value must never mint
/// local authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskContentRecord {
    pub content: ContentDigest,
    pub task: TaskId,
    pub attachment_id: AttachmentId,
    pub authority_generation: AuthorityGeneration,
    pub namespace_manifest: NamespaceManifestDigest,
    pub role: TaskContentRole,
}

impl ContentDigest {
    /// Derive content identity from exact bytes without storing those bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            digest: *blake3::hash(bytes).as_bytes(),
            byte_len: bytes.len() as u64,
        }
    }

    /// BLAKE3 content digest.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Exact byte length committed alongside the digest.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Task payload supplied to the existing Worker backend.
///
/// `Argv` is deliberately argv-shaped — this contract never adds a shell
/// string parser. A content payload names content identity only; fetching,
/// placement, and durable materialization remain Worker policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskPayload {
    Argv(Vec<String>),
    Content(ContentDigest),
}

/// Backend constraints committed at spawn time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRuntimeConstraints {
    /// Explicit backend class/placement constraint. `None` leaves selection to
    /// the already fail-closed Worker backend selector.
    pub backend_class: Option<String>,
    /// Maximum runtime for an argv task, in seconds.
    pub timeout_secs: Option<u64>,
}

/// VFS/Iroh/MoQ reaches actually handed out for an allocated Task.
///
/// Reach strings are locators, never bearer credentials. Implementations leave
/// a field empty rather than advertising a local-only carrier without endpoint,
/// subscriber, MAC-key, and admission handoff. The spike's Worker projection
/// currently returns only `vfs_path`; Iroh and MoQ delivery are deferred.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskReaches {
    pub vfs_path: String,
    pub iroh_endpoint: Option<String>,
    pub moq_topics: Vec<String>,
}

/// Observable lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Pending,
    Running,
    Exited { code: i32 },
    Cancelled,
}

/// An in-process task result with Task-service content-observation metadata.
///
/// See [`TaskContentRecord`] for its lack of wire serde and independent proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskResult {
    pub task: TaskId,
    pub state: TaskState,
    pub content_records: Vec<TaskContentRecord>,
}

/// Scoped attachment and authority-generation binding for a Task effect.
///
/// The only constructor accepts an [`AttachmentOperationGrant`], whose fields
/// are private. A verified identity by itself cannot construct this binding:
/// production grant issuance remains fail-closed until dispatch MAC and
/// delegated-capability decisions are both represented at this boundary.
#[derive(Clone, Debug)]
pub struct TaskAttachmentBinding {
    grant: AttachmentOperationGrant,
}

impl TaskAttachmentBinding {
    /// Bind a Task effect to a scoped, opaque attachment operation grant.
    #[must_use]
    pub fn from_grant(grant: &AttachmentOperationGrant) -> Self {
        Self {
            grant: grant.clone(),
        }
    }

    /// Attachment correlation identifier.
    #[must_use]
    pub fn attachment_id(&self) -> &AttachmentId {
        self.grant.attachment_id()
    }

    /// Generation that must remain current through the next effect.
    #[must_use]
    pub const fn authority_generation(&self) -> AuthorityGeneration {
        self.grant.generation()
    }

    /// Verified local subject view.
    #[must_use]
    pub fn subject(&self) -> &Subject {
        self.grant.subject()
    }

    /// Fail closed unless this binding is current and carries `operation`.
    pub fn ensure(&self, operation: AttachmentOperation) -> Result<(), TaskError> {
        self.grant.ensure(operation).map_err(TaskError::from)
    }

    /// Opaque scoped grant for a context-aware VFS projection.
    #[must_use]
    pub fn operation_grant(&self) -> &AttachmentOperationGrant {
        &self.grant
    }
}

/// Request to allocate and configure a Task around the existing `/exec` tree.
#[derive(Clone, Debug)]
pub struct TaskSpawnRequest {
    pub parent_task: Option<TaskId>,
    pub attachment: TaskAttachmentBinding,
    /// Contract metadata only; not yet an admitted effective namespace.
    pub namespace_manifest: NamespaceManifestDigest,
    pub payload: TaskPayload,
    pub constraints: TaskRuntimeConstraints,
}

impl TaskSpawnRequest {
    /// Build a root Task request from an authority-bound attachment.
    #[must_use]
    pub fn new(
        attachment: TaskAttachmentBinding,
        namespace_manifest: NamespaceManifestDigest,
        payload: TaskPayload,
        constraints: TaskRuntimeConstraints,
    ) -> Self {
        Self {
            parent_task: None,
            attachment,
            namespace_manifest,
            payload,
            constraints,
        }
    }

    /// Child creation is not implemented by this spike.
    ///
    /// A boolean derived from a lease would not prove a delegated capability
    /// or bind a real parent record, so the contract fails closed until the
    /// verified attachment boundary mints an attenuated parent capability.
    pub fn child_of(
        _parent_task: TaskId,
        _attachment: TaskAttachmentBinding,
        _namespace_manifest: NamespaceManifestDigest,
        _payload: TaskPayload,
        _constraints: TaskRuntimeConstraints,
    ) -> Result<Self, TaskError> {
        Err(TaskError::ChildDelegationRequired)
    }

    /// Validate the scoped Spawn grant immediately before allocation.
    pub fn ensure_spawn_permitted(&self) -> Result<(), TaskError> {
        self.attachment.ensure(AttachmentOperation::TaskSpawn)
    }

    /// Require the composite permissions an argv spawn needs in this Worker
    /// projection: allocating the sandbox and publishing terminal fd output.
    pub fn ensure_argv_spawn_permitted(&self) -> Result<(), TaskError> {
        self.ensure_spawn_permitted()?;
        self.attachment.ensure(AttachmentOperation::TaskPublish)
    }

    /// Validate invariants that must hold even when a request was built with a
    /// struct literal instead of [`Self::child_of`].
    ///
    /// A parent task is rejected until the verified attachment boundary has an
    /// opaque, attenuated parent-capability representation. Implementations
    /// call this at their allocation boundary rather than relying on callers
    /// to use the root constructor.
    pub fn validate_for_spawn(&self) -> Result<(), TaskError> {
        self.ensure_spawn_permitted()?;
        if self.parent_task.is_some() {
            return Err(TaskError::ChildDelegationRequired);
        }
        Ok(())
    }
}

/// Handle returned from spawn/attach, including the projection reaches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskHandle {
    pub task: TaskId,
    pub reaches: TaskReaches,
    pub state: TaskState,
}

/// Signal request kept separate from a task id so every effect also proves its
/// current attachment binding.
#[derive(Clone, Debug)]
pub struct TaskSignalRequest {
    pub task: TaskId,
    pub attachment: TaskAttachmentBinding,
    pub signal: TaskSignal,
}

impl TaskSignalRequest {
    /// Require the composite permissions a terminal lifecycle signal needs.
    /// The Worker projection emits fd completion after `stop`/`kill`/`destroy`,
    /// so `TaskSignal` alone cannot authorize this operation.
    pub fn ensure_permitted(&self) -> Result<(), TaskError> {
        self.attachment.ensure(AttachmentOperation::TaskSignal)?;
        self.attachment.ensure(AttachmentOperation::TaskPublish)
    }
}

/// Lifecycle signals supported by the current Worker projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskSignal {
    Stop,
    Kill,
    Destroy,
}

/// Snapshot metadata including the Task's recorded namespace commitment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub task: TaskId,
    pub namespace_manifest: NamespaceManifestDigest,
    pub state: TaskState,
}

/// Error returned by the transport-neutral Task contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskError {
    StaleAttachment,
    InvalidRequest(String),
    NotFound(TaskId),
    PermissionDenied,
    ChildDelegationRequired,
    Backend(String),
}

impl From<AttachmentError> for TaskError {
    fn from(error: AttachmentError) -> Self {
        match error {
            AttachmentError::StaleAuthority { .. } => Self::StaleAttachment,
            AttachmentError::UnverifiedEnvelopeContext
            | AttachmentError::AnonymousSubject
            | AttachmentError::GenerationExhausted { .. }
            | AttachmentError::OperationNotGranted { .. } => Self::PermissionDenied,
        }
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleAttachment => f.write_str("task attachment authority is stale"),
            Self::InvalidRequest(detail) => write!(f, "invalid task request: {detail}"),
            Self::NotFound(task) => write!(f, "task {task} was not found"),
            Self::PermissionDenied => f.write_str("task operation denied"),
            Self::ChildDelegationRequired => f.write_str("child task requires a delegated binding"),
            Self::Backend(detail) => write!(f, "task backend failed: {detail}"),
        }
    }
}

impl std::error::Error for TaskError {}

/// Standard service surface. Implementations own placement; this crate only
/// owns the cross-runtime contract and the attachment-generation invariant.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait TaskService: Send + Sync {
    /// Allocate/configure a new Task and project it under `/exec/instances`.
    /// An argv request requires both `TaskSpawn` and `TaskPublish` because its
    /// terminal stdout/stderr carrier is part of the allocation operation.
    async fn spawn_task(&self, request: TaskSpawnRequest) -> Result<TaskHandle, TaskError>;

    /// Attach to an existing task with an explicit `TaskAttach` scope.
    async fn attach_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskHandle, TaskError>;

    /// Deliver a lifecycle signal with `TaskSignal` plus `TaskPublish` scopes,
    /// because the Worker projection completes terminal fd streams.
    async fn signal_task(&self, request: TaskSignalRequest) -> Result<(), TaskError>;

    /// Wait for a terminal result with an explicit `TaskRead` scope.
    async fn wait_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskResult, TaskError>;

    /// Return namespace/lifecycle metadata with an explicit `TaskRead` scope.
    async fn snapshot_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskSnapshot, TaskError>;

    /// Return a terminal result with an explicit `TaskRead` scope.
    async fn task_result(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskResult, TaskError>;

    /// Fail closed until the authority boundary mints an attenuated parent
    /// capability and validates the referenced parent record.
    async fn spawn_child_task(&self, request: TaskSpawnRequest) -> Result<TaskHandle, TaskError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::{AttachmentOperation, Subject, VerifiedAttachment};

    fn test_binding(attachment: &VerifiedAttachment) -> TaskAttachmentBinding {
        let grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskAttach,
            AttachmentOperation::TaskSignal,
            AttachmentOperation::TaskRead,
            AttachmentOperation::TaskPublish,
        ]);
        TaskAttachmentBinding::from_grant(&grant)
    }

    #[test]
    fn child_creation_and_forged_parent_lineage_fail_closed() {
        let attachment = VerifiedAttachment::for_test_local_root(Subject::new("alice")).unwrap();
        let binding = test_binding(&attachment);
        let manifest = NamespaceManifestDigest::from_description_bytes(b"/work=cid-a\n");

        let err = TaskSpawnRequest::child_of(
            TaskId::from_worker_instance("parent"),
            binding.clone(),
            manifest.clone(),
            TaskPayload::Argv(vec!["echo".into(), "hello".into()]),
            TaskRuntimeConstraints::default(),
        )
        .unwrap_err();
        assert_eq!(err, TaskError::ChildDelegationRequired);

        let forged_child = TaskSpawnRequest {
            parent_task: Some(TaskId::from_worker_instance("parent")),
            attachment: binding.clone(),
            namespace_manifest: manifest.clone(),
            payload: TaskPayload::Argv(vec!["echo".into(), "forged".into()]),
            constraints: TaskRuntimeConstraints::default(),
        };
        assert_eq!(
            forged_child.validate_for_spawn(),
            Err(TaskError::ChildDelegationRequired)
        );

        let root = TaskSpawnRequest::new(
            binding.clone(),
            manifest,
            TaskPayload::Argv(vec!["echo".into(), "root".into()]),
            TaskRuntimeConstraints::default(),
        );
        assert!(root.validate_for_spawn().is_ok());
        assert!(root.ensure_argv_spawn_permitted().is_ok());
        attachment.revoke().unwrap();
        assert_eq!(
            root.ensure_spawn_permitted(),
            Err(TaskError::StaleAttachment)
        );

        let read_only_attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("reader")).unwrap();
        let read_only_grant =
            read_only_attachment.for_test_operations(&[AttachmentOperation::TaskRead]);
        let read_only_request = TaskSpawnRequest::new(
            TaskAttachmentBinding::from_grant(&read_only_grant),
            NamespaceManifestDigest::from_description_bytes(b"/work=cid-b\n"),
            TaskPayload::Argv(vec!["echo".into(), "denied".into()]),
            TaskRuntimeConstraints::default(),
        );
        assert_eq!(
            read_only_request.validate_for_spawn(),
            Err(TaskError::PermissionDenied)
        );

        let spawn_only_attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("spawner")).unwrap();
        let spawn_only_grant =
            spawn_only_attachment.for_test_operations(&[AttachmentOperation::TaskSpawn]);
        let spawn_only_request = TaskSpawnRequest::new(
            TaskAttachmentBinding::from_grant(&spawn_only_grant),
            NamespaceManifestDigest::from_description_bytes(b"/work=cid-c\n"),
            TaskPayload::Argv(vec!["echo".into(), "needs-publish".into()]),
            TaskRuntimeConstraints::default(),
        );
        assert!(spawn_only_request.validate_for_spawn().is_ok());
        assert_eq!(
            spawn_only_request.ensure_argv_spawn_permitted(),
            Err(TaskError::PermissionDenied)
        );

        let signal_only_attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("signaller")).unwrap();
        let signal_only_grant =
            signal_only_attachment.for_test_operations(&[AttachmentOperation::TaskSignal]);
        let signal_request = TaskSignalRequest {
            task: TaskId::from_worker_instance("task-1"),
            attachment: TaskAttachmentBinding::from_grant(&signal_only_grant),
            signal: TaskSignal::Stop,
        };
        assert_eq!(
            signal_request.ensure_permitted(),
            Err(TaskError::PermissionDenied)
        );
    }

    #[test]
    fn content_records_distinguish_identity_from_task_association() {
        let first = ContentDigest::from_bytes(b"stdout");
        let second = ContentDigest::from_bytes(b"stdout");
        let different = ContentDigest::from_bytes(b"stderr");
        assert_eq!(first, second);
        assert_ne!(first, different);
        assert_eq!(first.byte_len(), 6);

        let attachment = VerifiedAttachment::for_test_local_root(Subject::new("alice")).unwrap();
        let binding = test_binding(&attachment);
        let record = TaskContentRecord {
            content: first,
            task: TaskId::from_worker_instance("task-1"),
            attachment_id: binding.attachment_id().clone(),
            authority_generation: binding.authority_generation(),
            namespace_manifest: NamespaceManifestDigest::from_description_bytes(b"/work=cid-a\n"),
            role: TaskContentRole::Stdout,
        };
        assert_eq!(record.task.as_str(), "task-1");
        assert_eq!(record.role, TaskContentRole::Stdout);
    }
}
