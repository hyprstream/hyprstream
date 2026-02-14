//! WorkerZmqClient - typed client for WorkerService
//!
//! Uses the generated `GenWorkerClient` from `hyprstream_workers` for schema-driven RPC,
//! implementing `RuntimeClient` and `ImageClient` traits.
//!
//! Response types use generated types directly — no domain type wrappers.

use async_trait::async_trait;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::service::AuthorizeFn;
use anyhow::Result;
use std::collections::HashMap; // Only for StatusResponse.info (local-only)
use std::sync::Arc;

// Domain types from hyprstream-workers (runtime state types that still exist)
pub use hyprstream_workers::image::ImageClient;
pub use hyprstream_workers::runtime::{
    ContainerFilter,
    ContainerState, ContainerStatsFilter,
    KeyValue, PodSandboxFilter,
    PodSandboxState, PodSandboxStatsFilter,
    RuntimeClient, StatusResponse,
};

// Workflow types from hyprstream-workers
pub use hyprstream_workers::workflow::WorkflowClient;

// Generated types from hyprstream-workers (wire format types)
pub use hyprstream_workers::runtime::{
    GenWorkerClient,
    VersionInfo, RuntimeStatus,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    PodSandboxStats, PodSandboxInfo, ContainerInfo,
    ContainerStatsWire, FilesystemUsage,
    PodSandboxStateEnum, ContainerStateEnum,
    // Request config types
    PodSandboxConfig, ContainerConfig, ImageSpec,
    AuthConfigWire, StreamInfo,
    ImageInfo, ImageStatusResult,
};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "worker";

// ============================================================================
// Authorization Helper
// ============================================================================

use crate::services::PolicyClient;
use crate::auth::Operation;

/// Build an `AuthorizeFn` backed by a `PolicyClient`.
///
/// The returned closure is async-compatible (returns a boxed future) so it
/// works on single-threaded runtimes used by ZmqService.
pub fn build_authorize_fn(policy_client: PolicyClient) -> AuthorizeFn {
    Arc::new(move |subject: String, resource: String, operation: String| {
        let client = policy_client.clone();
        Box::pin(async move {
            let op = Operation::from_str(&operation)?;
            client.check_policy_str(&subject, &resource, op).await
        })
    })
}

// ============================================================================
// WorkerZmqClient
// ============================================================================

/// ZMQ client wrapping the generated `GenWorkerClient` from the proc macro.
///
/// Provides `RuntimeClient` and `ImageClient` trait implementations via
/// the generated scoped clients (runtime, sandbox, container, image).
pub struct WorkerZmqClient {
    gen: GenWorkerClient,
}

impl WorkerZmqClient {
    /// Create a new WorkerZmqClient (endpoint from registry).
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a WorkerZmqClient connected to a specific endpoint.
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            gen: crate::services::core::create_service_client(endpoint, signing_key, identity),
        }
    }

    /// Get the underlying generated client (for advanced use).
    pub fn gen(&self) -> &GenWorkerClient {
        &self.gen
    }

    /// Attach to container I/O streams.
    pub async fn attach(&self, container_id: &str, ephemeral_pubkey: [u8; 32]) -> Result<StreamInfo> {
        self.gen.container().attach(container_id, &[], ephemeral_pubkey).await
    }

    /// Detach from container I/O streams.
    pub async fn detach(&self, container_id: &str) -> Result<()> {
        self.gen.container().detach(container_id).await?;
        Ok(())
    }
}

// ============================================================================
// Conversion Helpers: Domain → Generated Data (for RPC requests)
// ============================================================================

fn sandbox_state_to_str(state: &PodSandboxState) -> &str {
    match state {
        PodSandboxState::SandboxReady => "sandbox_ready",
        PodSandboxState::SandboxNotReady => "sandbox_not_ready",
    }
}

fn container_state_to_str(state: &ContainerState) -> &str {
    match state {
        ContainerState::ContainerCreated => "container_created",
        ContainerState::ContainerRunning => "container_running",
        ContainerState::ContainerExited => "container_exited",
        ContainerState::ContainerUnknown => "container_unknown",
    }
}

/// Map anyhow error to WorkerError.
fn worker_err(e: anyhow::Error) -> hyprstream_workers::error::WorkerError {
    hyprstream_workers::error::WorkerError::Internal(e.to_string())
}

// ============================================================================
// RuntimeClient Implementation
// ============================================================================

#[async_trait]
impl RuntimeClient for WorkerZmqClient {
    async fn version(&self, version: &str) -> hyprstream_workers::error::Result<VersionInfo> {
        self.gen.runtime().version(version).await.map_err(worker_err)
    }

    async fn status(&self, verbose: bool) -> hyprstream_workers::error::Result<StatusResponse> {
        let data = self.gen.runtime().status(verbose).await.map_err(worker_err)?;
        Ok(StatusResponse { status: data, info: HashMap::new() })
    }

    async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> hyprstream_workers::error::Result<String> {
        self.gen.sandbox().run(
            config.metadata.clone(), &config.hostname, &config.log_directory, config.dns_config.clone(),
            &config.port_mappings, &config.labels, &config.annotations, config.linux.clone(),
        ).await.map_err(worker_err)
    }

    async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> hyprstream_workers::error::Result<()> {
        self.gen.sandbox().stop(pod_sandbox_id).await.map_err(worker_err)
    }

    async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> hyprstream_workers::error::Result<()> {
        self.gen.sandbox().remove(pod_sandbox_id).await.map_err(worker_err)
    }

    async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<PodSandboxStatusResponse> {
        self.gen.sandbox().status(pod_sandbox_id, verbose).await.map_err(worker_err)
    }

    async fn list_pod_sandbox(
        &self,
        filter: Option<&PodSandboxFilter>,
    ) -> hyprstream_workers::error::Result<Vec<PodSandboxInfo>> {
        let (id, state, labels) = match filter {
            Some(f) => (
                f.id.as_deref().unwrap_or(""),
                f.state.as_ref().map(sandbox_state_to_str).unwrap_or(""),
                f.label_selector.clone(),
            ),
            None => ("", "", vec![]),
        };
        self.gen.sandbox().list(id, state, &labels).await.map_err(worker_err)
    }

    async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        sandbox_config: &PodSandboxConfig,
    ) -> hyprstream_workers::error::Result<String> {
        self.gen.container().create(pod_sandbox_id, config.clone(), sandbox_config.clone())
            .await.map_err(worker_err)
    }

    async fn start_container(&self, container_id: &str) -> hyprstream_workers::error::Result<()> {
        self.gen.container().start(container_id).await.map_err(worker_err)
    }

    async fn stop_container(&self, container_id: &str, timeout: i64) -> hyprstream_workers::error::Result<()> {
        self.gen.container().stop(container_id, timeout).await.map_err(worker_err)
    }

    async fn remove_container(&self, container_id: &str) -> hyprstream_workers::error::Result<()> {
        self.gen.container().remove(container_id).await.map_err(worker_err)
    }

    async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<ContainerStatusResponse> {
        self.gen.container().status(container_id, verbose).await.map_err(worker_err)
    }

    async fn list_containers(
        &self,
        filter: Option<&ContainerFilter>,
    ) -> hyprstream_workers::error::Result<Vec<ContainerInfo>> {
        let (id, pod_sandbox_id, state, labels) = match filter {
            Some(f) => (
                f.id.as_deref().unwrap_or(""),
                f.pod_sandbox_id.as_deref().unwrap_or(""),
                f.state.as_ref().map(container_state_to_str).unwrap_or(""),
                f.label_selector.clone(),
            ),
            None => ("", "", "", vec![]),
        };
        self.gen.container().list(id, pod_sandbox_id, state, &labels)
            .await.map_err(worker_err)
    }

    async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        timeout: i64,
    ) -> hyprstream_workers::error::Result<ExecSyncResult> {
        self.gen.container().exec(container_id, cmd, timeout)
            .await.map_err(worker_err)
    }

    async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> hyprstream_workers::error::Result<PodSandboxStats> {
        self.gen.sandbox().stats(pod_sandbox_id).await.map_err(worker_err)
    }

    async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> hyprstream_workers::error::Result<Vec<PodSandboxStats>> {
        let (id, labels) = match filter {
            Some(f) => (
                f.id.as_deref().unwrap_or(""),
                f.label_selector.clone(),
            ),
            None => ("", vec![]),
        };
        self.gen.sandbox().list_stats(id, &labels).await.map_err(worker_err)
    }

    async fn container_stats(&self, container_id: &str) -> hyprstream_workers::error::Result<ContainerStatsWire> {
        self.gen.container().stats(container_id).await.map_err(worker_err)
    }

    async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> hyprstream_workers::error::Result<Vec<ContainerStatsWire>> {
        let (id, pod_sandbox_id, labels) = match filter {
            Some(f) => (
                f.id.as_deref().unwrap_or(""),
                f.pod_sandbox_id.as_deref().unwrap_or(""),
                f.label_selector.clone(),
            ),
            None => ("", "", vec![]),
        };
        self.gen.container().list_stats(id, pod_sandbox_id, &labels)
            .await.map_err(worker_err)
    }
}

// ============================================================================
// ImageClient Implementation
// ============================================================================

#[async_trait]
impl ImageClient for WorkerZmqClient {
    async fn list_images(
        &self,
        filter: Option<&ImageSpec>,
    ) -> hyprstream_workers::error::Result<Vec<ImageInfo>> {
        let image_data = filter.cloned().unwrap_or_default();
        self.gen.image().list(image_data).await.map_err(worker_err)
    }

    async fn image_status(
        &self,
        image: &ImageSpec,
        verbose: bool,
    ) -> hyprstream_workers::error::Result<ImageStatusResult> {
        self.gen.image().status(image.clone(), verbose)
            .await
            .map_err(|e| hyprstream_workers::error::WorkerError::ImageNotFound(e.to_string()))
    }

    async fn pull_image(
        &self,
        image: &ImageSpec,
        auth: Option<&AuthConfigWire>,
    ) -> hyprstream_workers::error::Result<String> {
        let auth_data = auth.cloned().unwrap_or_default();
        self.gen.image().pull(image.clone(), auth_data, PodSandboxConfig::default())
            .await.map_err(worker_err)
    }

    async fn remove_image(&self, image: &ImageSpec) -> hyprstream_workers::error::Result<()> {
        self.gen.image().remove(&image.image, &image.annotations, &image.runtime_handler)
            .await.map_err(worker_err)
    }

    async fn image_fs_info(
        &self,
    ) -> hyprstream_workers::error::Result<Vec<FilesystemUsage>> {
        self.gen.image().fs_info().await.map_err(worker_err)
    }
}

// ============================================================================
// WorkflowZmqClient
// ============================================================================

/// Service name for workflow endpoint registry
const WORKFLOW_SERVICE_NAME: &str = "workflow";

/// ZMQ client wrapping the generated `GenWorkflowClient` from the proc macro.
///
/// Implements the `WorkflowClient` trait for orchestration operations.
pub struct WorkflowZmqClient {
    gen: hyprstream_workers::workflow::GenWorkflowClient,
}

impl WorkflowZmqClient {
    /// Create a new WorkflowZmqClient (endpoint from registry).
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(WORKFLOW_SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a WorkflowZmqClient connected to a specific endpoint.
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            gen: crate::services::core::create_service_client(endpoint, signing_key, identity),
        }
    }

    /// Get the underlying generated client (for advanced use).
    pub fn gen(&self) -> &hyprstream_workers::workflow::GenWorkflowClient {
        &self.gen
    }
}

use hyprstream_workers::workflow::{
    WorkflowDef, WorkflowInfo, WorkflowRun,
    WorkflowKeyValue,
    WorkflowId, RunId, SubscriptionId,
};

#[async_trait]
impl WorkflowClient for WorkflowZmqClient {
    async fn scan_repo(&self, repo_id: &str) -> hyprstream_workers::error::Result<Vec<WorkflowDef>> {
        self.gen.scan_repo(repo_id).await.map_err(worker_err)
    }

    async fn register_workflow(&self, def: &WorkflowDef) -> hyprstream_workers::error::Result<WorkflowId> {
        self.gen.register(&def.path, &def.repo_id, &def.name, &def.triggers, &def.yaml).await.map_err(worker_err)
    }

    async fn list_workflows(&self) -> hyprstream_workers::error::Result<Vec<WorkflowInfo>> {
        self.gen.list().await.map_err(worker_err)
    }

    async fn dispatch(
        &self,
        workflow_id: &WorkflowId,
        inputs: &[WorkflowKeyValue],
    ) -> hyprstream_workers::error::Result<RunId> {
        self.gen.dispatch(workflow_id, inputs).await.map_err(worker_err)
    }

    async fn subscribe(&self, workflow_id: &WorkflowId) -> hyprstream_workers::error::Result<SubscriptionId> {
        self.gen.subscribe(workflow_id).await.map_err(worker_err)
    }

    async fn unsubscribe(&self, sub_id: &SubscriptionId) -> hyprstream_workers::error::Result<()> {
        self.gen.unsubscribe(sub_id).await.map_err(worker_err)
    }

    async fn get_run(&self, run_id: &RunId) -> hyprstream_workers::error::Result<WorkflowRun> {
        self.gen.get_run(run_id).await.map_err(worker_err)
    }

    async fn list_runs(&self, workflow_id: &WorkflowId) -> hyprstream_workers::error::Result<Vec<WorkflowRun>> {
        self.gen.list_runs(workflow_id).await.map_err(worker_err)
    }
}
