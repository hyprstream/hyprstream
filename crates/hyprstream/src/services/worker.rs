//! WorkerZmqClient - typed client for WorkerService
//!
//! Uses the generated `WorkerClient` from `hyprstream_workers` for schema-driven RPC.
//! All methods are inherent — no domain trait wrappers.
//!
//! Response types use generated types directly — no domain type wrappers.

use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service::AuthorizeFn;
use anyhow::Result;
use std::collections::HashMap; // Only for StatusResponse.info (local-only)
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

/// Service name for endpoint registry
const SERVICE_NAME: &str = "worker";

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

// ============================================================================
// WorkerZmqClient
// ============================================================================

/// ZMQ client wrapping the generated `WorkerClient` from the proc macro.
///
/// All CRI runtime and image operations are provided as inherent methods
/// via the generated scoped clients (runtime, sandbox, container, image).
/// The generated client is also available via `Deref` for direct access.
#[derive(Clone)]
pub struct WorkerZmqClient {
    gen: WorkerClient,
}

impl std::ops::Deref for WorkerZmqClient {
    type Target = WorkerClient;
    fn deref(&self) -> &Self::Target { &self.gen }
}

impl WorkerZmqClient {
    /// Create a new WorkerZmqClient (endpoint from registry).
    ///
    /// D9: Uses `try_global()` with inproc fallback if registry not yet initialized.
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = hyprstream_rpc::registry::try_global()
            .map(|r| r.endpoint(SERVICE_NAME, SocketKind::Rep))
            .unwrap_or_else(|| hyprstream_rpc::transport::TransportConfig::inproc(
                format!("hyprstream/{}", SERVICE_NAME)
            ))
            .to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a WorkerZmqClient connected to a specific endpoint.
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            gen: crate::services::core::create_service_client(endpoint, signing_key, identity),
        }
    }

    /// Attach claims for e2e verification. All subsequent calls include these claims.
    pub fn with_claims(self, claims: hyprstream_rpc::auth::Claims) -> Self {
        Self { gen: self.gen.with_claims(claims) }
    }

    /// Attach to container I/O streams.
    ///
    /// Generates an ephemeral DH keypair, performs the attach RPC, and returns
    /// a ready-to-use `StreamHandle` with E2E HMAC verification.
    pub async fn attach(&self, container_id: &str) -> Result<crate::services::rpc_types::StreamHandle> {
        use hyprstream_rpc::crypto::generate_ephemeral_keypair;
        use crate::zmq::global_context;

        let (client_secret, client_pubkey) = generate_ephemeral_keypair();
        let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

        let info = self.gen.container().attach(&AttachRequest {
            container_id: container_id.to_owned(),
            fds: vec![],
        }, client_pubkey_bytes).await?;

        if info.server_pubkey == [0u8; 32] {
            anyhow::bail!(
                "Server did not provide Ristretto255 public key - E2E authentication required"
            );
        }

        crate::services::rpc_types::StreamHandle::new(
            &global_context(),
            info.stream_id,
            &info.endpoint,
            &info.server_pubkey,
            &client_secret,
            &client_pubkey_bytes,
        )
    }

    /// Detach from container I/O streams.
    pub async fn detach(&self, container_id: &str) -> Result<()> {
        self.gen.container().detach(container_id).await?;
        Ok(())
    }

    /// Get runtime status (wraps generated status with local info HashMap).
    pub async fn status(&self, verbose: bool) -> Result<StatusResponse> {
        let data = self.gen.runtime().status(&WkStatusRequest { verbose }).await?;
        Ok(StatusResponse { status: data, info: HashMap::new() })
    }
}
