//! Worker service types and helpers.
//!
//! Uses the generated `WorkerClient` from `hyprstream_workers` for schema-driven RPC.
//! All methods are inherent — no domain trait wrappers.
//!
//! Response types use generated types directly — no domain type wrappers.

use hyprstream_rpc::service::AuthorizeFn;
use std::sync::Arc;
use crate::services::generated::policy_client::PolicyCheck;

// Re-export generated types with clean OCI-aligned names (no Gen*/Wire/Enum aliases)
pub use hyprstream_workers::runtime::{
    WorkerClient,
    VersionInfo, RuntimeStatus,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    PodSandboxStats, PodSandboxInfo, ContainerInfo,
    ContainerStats, FilesystemUsage,
    PodSandboxState, ContainerState,
    PodSandboxConfig, ContainerConfig, ImageSpec,
    AuthConfig, StreamInfo,
    ImageInfo, ImageStatusResult,
    StatusResponse, KeyValue,
    // Filter types (generated)
    PodSandboxFilter, ContainerFilter,
    PodSandboxStatsFilter, ContainerStatsFilter,
    Timestamp,
};

// Generated request types for struct-based client calls
pub use hyprstream_workers::generated::worker_client::{
    StatusRequest as WkStatusRequest,
    PodSandboxStatusRequest, CreateContainerRequest, StopContainerRequest,
    ContainerStatusRequest, ExecSyncRequest, AttachRequest,
    ImageFilter, ImageStatusRequest, PullImageRequest,
};

// ============================================================================
// Authorization Helper
// ============================================================================

use crate::services::PolicyClient;

/// Build an `AuthorizeFn` backed by a `PolicyClient`.
///
/// The returned closure is async-compatible (returns a boxed future) so it
/// works on single-threaded runtimes used by ZmqService.
pub fn build_authorize_fn(policy_client: PolicyClient) -> AuthorizeFn {
    Arc::new(move |subject: String, resource: String, operation: String| {
        let client = policy_client.clone();
        Box::pin(async move {
            client.check(&PolicyCheck { subject: subject.clone(), domain: "*".to_owned(), resource: resource.clone(), operation: operation.clone() }).await
        })
    })
}

// WorkerZmqClient / attach_container removed — use generated WorkerClient +
// ContainerRpc trait method directly (DH key exchange encapsulated by codegen).
