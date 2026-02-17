//! WorkerService - CRI RuntimeClient + ImageClient via ZMQ
//!
//! Implements ZmqService trait for handling CRI-aligned requests.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use tracing::{debug, info, warn};

// Import ZMQ service infrastructure from hyprstream-rpc
use hyprstream_rpc::prelude::SigningKey;
use hyprstream_rpc::service::{AuthorizeFn, EnvelopeContext, ZmqService};
use hyprstream_rpc::streaming::{StreamChannel, StreamPublisher};
use hyprstream_rpc::transport::TransportConfig;

use crate::config::{ImageConfig, PoolConfig};
use crate::error::{Result, WorkerError};
use crate::events::{
    EventPublisher,
    // Event types and serialization helpers
    ContainerStarted, ContainerStopped, SandboxStarted, SandboxStopped,
    serialize_container_started, serialize_container_stopped,
    serialize_sandbox_started, serialize_sandbox_stopped,
};
use crate::image::RafsStore;
// Import generated wire types for handler signatures
use crate::generated::worker_client::{
    ContainerFilter, ContainerStatsFilter,
    PodSandboxFilter, PodSandboxStatsFilter,
    // Request types
    StatusRequest, PodSandboxStatusRequest, StopContainerRequest,
    ContainerStatusRequest, AttachRequest, ImageFilter, ImageStatusRequest,
};
use super::client::{
    StatusResponse, KeyValue,
    // Generated wire types
    VersionInfo, RuntimeStatus, RuntimeCondition,
    ExecSyncResult,
    PodSandboxStats, PodSandboxAttributes, LinuxPodSandboxStats,
    ContainerStats, ContainerAttributes,
    PodSandboxInfo, ContainerInfo,
    PodSandboxStateEnum, ContainerStateEnum, Timestamp, ImageSpec,
    StreamInfo,
};
// Domain entities (business logic)
use super::container::{Container, ContainerState};
use super::pool::SandboxPool;
use super::sandbox::{PodSandbox, PodSandboxState};

// Generated wire types (DTOs with ToCapnp/FromCapnp)
use super::client::{
    PodSandboxConfig, PodSandboxStatus,
    ContainerConfig, ContainerStatus,
};
use super::{RUNTIME_NAME, RUNTIME_VERSION};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "worker";

/// WorkerService handles CRI RuntimeClient and ImageClient requests
///
/// Implements the ZmqService trait for integration with hyprstream's ZMQ infrastructure.
pub struct WorkerService {
    // Business logic
    /// Sandbox pool for VM management
    sandbox_pool: Arc<SandboxPool>,

    /// RAFS store for image management
    rafs_store: Arc<RafsStore>,

    /// Active containers (container_id -> Container)
    containers: RwLock<HashMap<String, Container>>,

    /// Container to sandbox mapping
    container_sandbox_map: RwLock<HashMap<String, String>>,

    /// Event publisher for lifecycle events (sandbox/container started/stopped)
    event_publisher: tokio::sync::Mutex<EventPublisher>,

    /// Active FD streams (stream_id -> ActiveFdStream)
    active_fd_streams: Arc<RwLock<HashMap<String, ActiveFdStream>>>,

    /// StreamChannel for authenticated, async FD streaming
    stream_channel: StreamChannel,

    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,

    /// Optional authorization callback (injected by parent crate)
    authorize_fn: Option<AuthorizeFn>,
}

impl WorkerService {
    /// Create a new WorkerService with full configuration and infrastructure
    ///
    /// Must be called from within a tokio runtime context.
    pub fn new(
        pool_config: PoolConfig,
        image_config: ImageConfig,
        rafs_store: Arc<RafsStore>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> AnyhowResult<Self> {
        let sandbox_pool = Arc::new(SandboxPool::new(
            pool_config,
            image_config,
            Arc::clone(&rafs_store),
        ));

        // Create event publisher for worker lifecycle events
        let event_publisher = EventPublisher::new(&context, "worker")?;

        let stream_channel = StreamChannel::new(Arc::clone(&context), signing_key.clone());

        Ok(Self {
            sandbox_pool,
            rafs_store,
            containers: RwLock::new(HashMap::new()),
            container_sandbox_map: RwLock::new(HashMap::new()),
            event_publisher: tokio::sync::Mutex::new(event_publisher),
            active_fd_streams: Arc::new(RwLock::new(HashMap::new())),
            stream_channel,
            context,
            transport,
            signing_key,
            authorize_fn: None,
        })
    }

    /// Set the authorization callback for policy checks.
    pub fn set_authorize_fn(&mut self, authorize_fn: AuthorizeFn) {
        self.authorize_fn = Some(authorize_fn);
    }

    /// Initialize the service (start warm pool)
    pub async fn initialize(&self) -> Result<()> {
        self.sandbox_pool.initialize().await?;
        tracing::info!("WorkerService initialized");
        Ok(())
    }

    /// Shutdown the service
    pub async fn shutdown(&self) -> Result<()> {
        self.sandbox_pool.shutdown().await?;
        tracing::info!("WorkerService shutdown complete");
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Runtime Information
    // ─────────────────────────────────────────────────────────────────────────

    /// Get runtime version information
    pub async fn version(&self, _version: &str) -> Result<VersionInfo> {
        Ok(VersionInfo {
            version: "v1".to_owned(),
            runtime_name: RUNTIME_NAME.to_owned(),
            runtime_version: RUNTIME_VERSION.to_owned(),
            runtime_api_version: "v1".to_owned(),
        })
    }

    /// Get runtime status
    pub async fn status(&self, verbose: bool) -> Result<StatusResponse> {
        let conditions = vec![
            RuntimeCondition {
                condition_type: "RuntimeReady".to_owned(),
                status: true,
                reason: String::new(),
                message: "Runtime is ready".to_owned(),
            },
            RuntimeCondition {
                condition_type: "NetworkReady".to_owned(),
                status: true,
                reason: String::new(),
                message: "Network is ready".to_owned(),
            },
        ];

        let mut info = HashMap::new();
        if verbose {
            let stats = self.sandbox_pool.stats().await;
            info.insert("warm_pool_size".to_owned(), stats.warm_count.to_string());
            info.insert("active_sandboxes".to_owned(), stats.active_count.to_string());
            info.insert("max_sandboxes".to_owned(), stats.max_sandboxes.to_string());
        }

        Ok(StatusResponse {
            status: RuntimeStatus { conditions },
            info,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pod Sandbox Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /// Create and start a pod sandbox (Kata VM)
    pub async fn run_pod_sandbox(&self, config: &PodSandboxConfig) -> Result<String> {
        let sandbox_id = self.sandbox_pool.acquire(config).await?;
        tracing::info!(sandbox_id = %sandbox_id, "Created pod sandbox");

        // Publish sandbox started event with structured payload
        let event = SandboxStarted {
            sandbox_id: sandbox_id.clone(),
            metadata: serde_json::to_string(&config.metadata).unwrap_or_default(),
            vm_pid: 0, // VM PID not easily accessible here; could be enhanced later
        };
        let payload = serialize_sandbox_started(&event).unwrap_or_default();
        self.publish_event(&sandbox_id, "started", &payload).await;

        Ok(sandbox_id)
    }

    /// Publish a worker lifecycle event (fire-and-forget)
    ///
    /// Topic format: worker.{entity_id}.{event_name}
    async fn publish_event(&self, entity_id: &str, event: &str, payload: &[u8]) {
        let mut publisher = self.event_publisher.lock().await;
        if let Err(e) = publisher.publish(entity_id, event, payload).await {
            tracing::warn!(
                entity_id = %entity_id,
                event = %event,
                error = %e,
                "Failed to publish worker event"
            );
        }
    }

    /// Stop a running pod sandbox
    pub async fn stop_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()> {
        // Stop all containers in the sandbox first
        let containers_to_stop: Vec<String> = {
            let map = self.container_sandbox_map.read().await;
            map.iter()
                .filter(|(_, sid)| *sid == pod_sandbox_id)
                .map(|(cid, _)| cid.clone())
                .collect()
        };

        for container_id in containers_to_stop {
            if let Err(e) = self.stop_container(&container_id, 30).await {
                tracing::warn!(
                    container_id = %container_id,
                    error = %e,
                    "Failed to stop container during sandbox stop"
                );
            }
        }

        // Mark sandbox as not ready (but don't release yet)
        // The actual VM stop happens in remove_pod_sandbox
        tracing::info!(sandbox_id = %pod_sandbox_id, "Stopped pod sandbox");

        // Publish sandbox stopped event with structured payload
        let event = SandboxStopped {
            sandbox_id: pod_sandbox_id.to_owned(),
            reason: "stopped".to_owned(),
            exit_code: 0,
        };
        let payload = serialize_sandbox_stopped(&event).unwrap_or_default();
        self.publish_event(pod_sandbox_id, "stopped", &payload).await;

        Ok(())
    }

    /// Remove a pod sandbox
    pub async fn remove_pod_sandbox(&self, pod_sandbox_id: &str) -> Result<()> {
        // Remove all containers in the sandbox
        let containers_to_remove: Vec<String> = {
            let map = self.container_sandbox_map.read().await;
            map.iter()
                .filter(|(_, sid)| *sid == pod_sandbox_id)
                .map(|(cid, _)| cid.clone())
                .collect()
        };

        for container_id in containers_to_remove {
            if let Err(e) = self.remove_container(&container_id).await {
                tracing::warn!(
                    container_id = %container_id,
                    error = %e,
                    "Failed to remove container during sandbox removal"
                );
            }
        }

        // Release sandbox back to pool
        self.sandbox_pool.release(pod_sandbox_id).await?;
        tracing::info!(sandbox_id = %pod_sandbox_id, "Removed pod sandbox");
        Ok(())
    }

    /// Get pod sandbox status
    ///
    /// Returns the runtime PodSandboxStatus (has ToCapnp for handler serialization)
    /// plus verbose info as KeyValue for the wire format.
    pub async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> Result<(PodSandboxStatus, Vec<KeyValue>)> {
        let sandbox = self
            .sandbox_pool
            .get(pod_sandbox_id)
            .await
            .ok_or_else(|| WorkerError::SandboxNotFound(pod_sandbox_id.to_owned()))?;

        let mut info = Vec::new();
        if verbose {
            // Get PIDs from hypervisor (host-level VM management)
            let pids = sandbox.get_pids().await;
            let pid_str = pids.map_or_else(|| "none".to_owned(), |p| {
                p.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",")
            });
            info.push(KeyValue { key: "vm_pids".to_owned(), value: pid_str });
        }

        Ok((PodSandboxStatus::from(&sandbox), info))
    }

    /// List pod sandboxes (uses hand-written Filter with Option fields)
    pub async fn list_pod_sandbox(
        &self,
        filter: Option<&super::client::PodSandboxFilter>,
    ) -> Result<Vec<PodSandboxInfo>> {
        let mut sandboxes = self.sandbox_pool.list_active().await;

        // Apply filters (hand-written Filter has Option<String> for id)
        if let Some(f) = filter {
            if let Some(id) = &f.id {
                sandboxes.retain(|s| &s.id == id);
            }
            if let Some(state) = f.state {
                sandboxes.retain(|s| s.state == state);
            }
            for kv in &f.label_selector {
                sandboxes.retain(|s| s.labels.iter().any(|l| l.key == kv.key && l.value == kv.value));
            }
        }

        Ok(sandboxes.iter().map(PodSandboxInfo::from).collect())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Container Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a container within a pod sandbox (host-level, no guest agent)
    ///
    /// In host-level mode, creating a container means:
    /// 1. Verify the sandbox (VM) exists and is ready
    /// 2. Ensure the container image is available via RAFS
    /// 3. Track container metadata for lifecycle management
    ///
    /// The container's rootfs will be accessible to the VM via virtiofs.
    pub async fn create_container(
        &self,
        pod_sandbox_id: &str,
        config: &ContainerConfig,
        _sandbox_config: &PodSandboxConfig,
    ) -> Result<String> {
        // Verify sandbox exists and is ready
        let sandbox = self
            .sandbox_pool
            .get(pod_sandbox_id)
            .await
            .ok_or_else(|| WorkerError::SandboxNotFound(pod_sandbox_id.to_owned()))?;

        if !sandbox.is_ready() {
            return Err(WorkerError::SandboxInvalidState {
                sandbox_id: pod_sandbox_id.to_owned(),
                state: "NotReady".to_owned(),
                expected: "Ready".to_owned(),
            });
        }

        // Generate container ID
        let container_id = uuid::Uuid::new_v4().to_string();

        // Ensure image is available (pull if needed via RafsStore)
        // The image will be served to the VM via virtiofs
        if !config.image.image.is_empty() {
            tracing::debug!(
                container_id = %container_id,
                image = %config.image.image,
                "Container image reference recorded (available via virtiofs)"
            );
            // Note: Image pulling is handled by RafsStore at sandbox creation time
            // The virtiofs daemon serves the RAFS filesystem to the VM
        }

        // Create container
        let container = Container::new(container_id.clone(), pod_sandbox_id.to_owned(), config);

        // Store container
        {
            let mut containers = self.containers.write().await;
            containers.insert(container_id.clone(), container);
        }
        {
            let mut map = self.container_sandbox_map.write().await;
            map.insert(container_id.clone(), pod_sandbox_id.to_owned());
        }

        tracing::info!(
            container_id = %container_id,
            sandbox_id = %pod_sandbox_id,
            image = %config.image.image,
            "Created container (host-level, rootfs via virtiofs)"
        );

        Ok(container_id)
    }

    /// Start a created container (host-level, no guest agent)
    ///
    /// In host-level mode, "starting" a container means:
    /// 1. The container's image rootfs is accessible via virtiofs
    /// 2. The VM is running and can access the filesystem
    /// 3. No guest agent required - VM manages its own processes
    pub async fn start_container(&self, container_id: &str) -> Result<()> {
        // Get sandbox for this container
        let sandbox_id = {
            let map = self.container_sandbox_map.read().await;
            map.get(container_id)
                .cloned()
                .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?
        };

        let sandbox = self
            .sandbox_pool
            .get(&sandbox_id)
            .await
            .ok_or_else(|| WorkerError::SandboxNotFound(sandbox_id.clone()))?;

        // Verify sandbox is ready (VM is running)
        if !sandbox.is_ready() {
            return Err(WorkerError::SandboxInvalidState {
                sandbox_id: sandbox_id.clone(),
                state: "NotReady".to_owned(),
                expected: "Ready".to_owned(),
            });
        }

        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        if container.state != ContainerState::ContainerCreated {
            return Err(WorkerError::ContainerInvalidState {
                container_id: container_id.to_owned(),
                state: format!("{:?}", container.state),
                expected: "ContainerCreated".to_owned(),
            });
        }

        // Host-level container start:
        // The VM is already running (sandbox ready), virtiofs provides the rootfs.
        // In this model, "starting" means the container is accessible.
        // The VM's init process or workload handles actual execution.
        //
        // Future enhancement: Use hypervisor.add_device() to hot-add container-specific
        // filesystems or block devices if needed.

        if let Some(ref _hypervisor) = sandbox.hypervisor {
            tracing::debug!(
                container_id = %container_id,
                sandbox_id = %sandbox_id,
                "Container filesystem accessible via VM's virtiofs mount"
            );
        }

        container.state = ContainerState::ContainerRunning;
        let image = container.image.image.clone();
        tracing::info!(
            container_id = %container_id,
            sandbox_id = %sandbox_id,
            "Started container (host-level, VM is running)"
        );

        // Release the lock before publishing to avoid deadlock
        drop(containers);

        // Publish container started event with structured payload
        let event = ContainerStarted {
            container_id: container_id.to_owned(),
            sandbox_id: sandbox_id.clone(),
            image,
        };
        let payload = serialize_container_started(&event).unwrap_or_default();
        self.publish_event(container_id, "started", &payload).await;

        Ok(())
    }

    /// Stop a running container (host-level, no guest agent)
    ///
    /// In host-level mode, "stopping" a container means:
    /// 1. Mark the container as exited
    /// 2. The VM continues running (sandbox lifecycle is separate)
    /// 3. Optionally remove container-specific resources
    pub async fn stop_container(&self, container_id: &str, _timeout: i64) -> Result<()> {
        let mut containers = self.containers.write().await;

        let container = containers
            .get_mut(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        if container.state != ContainerState::ContainerRunning {
            // Already stopped
            return Ok(());
        }

        // Host-level container stop:
        // Without a guest agent, we can't signal processes inside the VM.
        // We simply mark the container as exited - the VM sandbox continues running.
        //
        // In a full implementation with vsock or minimal agent, we could:
        // - Send a shutdown signal via vsock
        // - Use a watchdog/heartbeat mechanism
        //
        // For now, we rely on the VM workload managing its own lifecycle.

        container.state = ContainerState::ContainerExited;
        container.exit_code = Some(0);
        let sandbox_id = container.pod_sandbox_id.clone();
        let exit_code = container.exit_code.unwrap_or(0);
        tracing::info!(
            container_id = %container_id,
            "Stopped container (host-level, container marked as exited)"
        );

        // Release the lock before publishing to avoid deadlock
        drop(containers);

        // Publish container stopped event with structured payload
        let event = ContainerStopped {
            container_id: container_id.to_owned(),
            sandbox_id,
            exit_code,
            reason: "stopped".to_owned(),
        };
        let payload = serialize_container_stopped(&event).unwrap_or_default();
        self.publish_event(container_id, "stopped", &payload).await;

        Ok(())
    }

    /// Remove a container
    pub async fn remove_container(&self, container_id: &str) -> Result<()> {
        // Remove from containers map
        {
            let mut containers = self.containers.write().await;
            let container = containers
                .get(container_id)
                .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

            if container.state == ContainerState::ContainerRunning {
                return Err(WorkerError::ContainerInvalidState {
                    container_id: container_id.to_owned(),
                    state: "ContainerRunning".to_owned(),
                    expected: "ContainerExited or ContainerCreated".to_owned(),
                });
            }

            containers.remove(container_id);
        }

        // Remove from sandbox map
        {
            let mut map = self.container_sandbox_map.write().await;
            map.remove(container_id);
        }

        tracing::info!(container_id = %container_id, "Removed container");
        Ok(())
    }

    /// Get container status
    ///
    /// Returns the runtime ContainerStatus (has ToCapnp for handler serialization)
    /// plus verbose info as KeyValue for the wire format.
    pub async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> Result<(ContainerStatus, Vec<KeyValue>)> {
        let containers = self.containers.read().await;

        let container = containers
            .get(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        let mut info = Vec::new();
        if verbose {
            info.push(KeyValue {
                key: "pid".to_owned(),
                value: container.pid.map_or_else(|| "none".to_owned(), |p| p.to_string()),
            });
        }

        Ok((ContainerStatus::from(container), info))
    }

    /// List containers (uses hand-written Filter with Option fields)
    pub async fn list_containers(
        &self,
        filter: Option<&super::client::ContainerFilter>,
    ) -> Result<Vec<ContainerInfo>> {
        let containers = self.containers.read().await;
        let mut result: Vec<Container> = containers.values().cloned().collect();

        // Apply filters (hand-written Filter has Option fields)
        if let Some(f) = filter {
            if let Some(id) = &f.id {
                result.retain(|c| &c.id == id);
            }
            if let Some(sandbox_id) = &f.pod_sandbox_id {
                result.retain(|c| &c.pod_sandbox_id == sandbox_id);
            }
            if let Some(state) = f.state {
                result.retain(|c| c.state == state);
            }
            for kv in &f.label_selector {
                result.retain(|c| c.labels.iter().any(|l| l.key == kv.key && l.value == kv.value));
            }
        }

        Ok(result.iter().map(ContainerInfo::from).collect())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Exec
    // ─────────────────────────────────────────────────────────────────────────

    /// Execute a command in a container synchronously (host-level, limited support)
    ///
    /// **IMPORTANT**: In host-level mode without a guest agent, exec is not supported.
    /// This returns an error explaining the limitation.
    ///
    /// For workloads that need exec:
    /// - Use vsock with a minimal agent
    /// - Pre-configure the VM image with the workload
    /// - Use serial console (not recommended for production)
    pub async fn exec_sync(
        &self,
        container_id: &str,
        cmd: &[String],
        _timeout: i64,
    ) -> Result<ExecSyncResult> {
        let containers = self.containers.read().await;

        let container = containers
            .get(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        if container.state != ContainerState::ContainerRunning {
            return Err(WorkerError::ContainerInvalidState {
                container_id: container_id.to_owned(),
                state: format!("{:?}", container.state),
                expected: "ContainerRunning".to_owned(),
            });
        }

        // Host-level exec limitation:
        // Without a guest agent, we cannot execute commands inside the VM.
        // The VM is a black box from the host's perspective.

        tracing::warn!(
            container_id = %container_id,
            cmd = ?cmd,
            "Exec not supported in host-level mode (no guest agent)"
        );

        // Return error indicating exec is not supported
        Err(WorkerError::ExecFailed(
            "exec not supported in host-level mode without guest agent. \
             Configure VM image with workload or use vsock agent.".to_owned(),
        ))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stats
    // ─────────────────────────────────────────────────────────────────────────

    /// Get pod sandbox stats
    pub async fn pod_sandbox_stats(&self, pod_sandbox_id: &str) -> Result<PodSandboxStats> {
        let sandbox = self
            .sandbox_pool
            .get(pod_sandbox_id)
            .await
            .ok_or_else(|| WorkerError::SandboxNotFound(pod_sandbox_id.to_owned()))?;

        // TODO: Get actual stats from VM
        Ok(PodSandboxStats::from(&sandbox))
    }

    /// List pod sandbox stats (uses hand-written Filter with Option fields)
    pub async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&super::client::PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>> {
        let sandboxes = self.sandbox_pool.list_active().await;
        let mut results = Vec::new();

        for sandbox in sandboxes {
            // Apply filter (hand-written Filter has Option<String> for id)
            if let Some(f) = filter {
                if let Some(id) = &f.id {
                    if &sandbox.id != id {
                        continue;
                    }
                }
                let mut matches_labels = true;
                for kv in &f.label_selector {
                    if !sandbox.labels.iter().any(|l| l.key == kv.key && l.value == kv.value) {
                        matches_labels = false;
                        break;
                    }
                }
                if !matches_labels {
                    continue;
                }
            }

            results.push(PodSandboxStats::from(&sandbox));
        }

        Ok(results)
    }

    /// Get container stats
    pub async fn container_stats(&self, container_id: &str) -> Result<ContainerStats> {
        let containers = self.containers.read().await;

        let container = containers
            .get(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        // TODO: Get actual stats from container
        Ok(ContainerStats::from(container))
    }

    /// List container stats (uses hand-written Filter with Option fields)
    pub async fn list_container_stats(
        &self,
        filter: Option<&super::client::ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>> {
        let containers = self.containers.read().await;
        let mut results = Vec::new();

        for container in containers.values() {
            // Apply filter (hand-written Filter has Option fields)
            if let Some(f) = filter {
                if let Some(id) = &f.id {
                    if &container.id != id {
                        continue;
                    }
                }
                if let Some(sandbox_id) = &f.pod_sandbox_id {
                    if &container.pod_sandbox_id != sandbox_id {
                        continue;
                    }
                }
                let mut matches_labels = true;
                for kv in &f.label_selector {
                    if !container.labels.iter().any(|l| l.key == kv.key && l.value == kv.value) {
                        matches_labels = false;
                        break;
                    }
                }
                if !matches_labels {
                    continue;
                }
            }

            results.push(ContainerStats::from(container));
        }

        Ok(results)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Terminal Attach/Detach
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Attach to a container's terminal I/O streams (tmux-like)
    ///
    /// Performs DH key exchange and pre-authorization atomically via StreamChannel.
    /// The stream is immediately ready — no separate start_fd_stream() call needed.
    ///
    /// # Flow
    /// 1. Client calls attach() with ephemeral pubkey in envelope
    /// 2. Server does DH + pre-auth + prepares streaming
    /// 3. REP is sent with StreamInfo (stream_id, endpoint, server_pubkey)
    /// 4. Continuation spawns the FD streaming task AFTER client receives REP
    /// 5. Client derives topic/mac_key from DH shared secret and subscribes
    pub async fn prepare_attach(&self, ctx: &EnvelopeContext, container_id: &str) -> Result<(StreamInfo, hyprstream_rpc::service::Continuation)> {
        // Verify container exists
        let containers = self.containers.read().await;
        if !containers.contains_key(container_id) {
            return Err(anyhow::anyhow!("Container not found: {}", container_id).into());
        }
        drop(containers);

        // Extract ephemeral pubkey from SignedEnvelope (required for E2E auth)
        let client_pubkey = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow::anyhow!("Attach requires client ephemeral pubkey for E2E authentication"))?;

        // Forward user claims for StreamService subscription-time validation
        let claims = ctx.claims().cloned();

        // DH + pre-authorization via StreamChannel — atomic, no pending state
        let stream_ctx = self.stream_channel
            .prepare_stream_with_claims(client_pubkey, 600, claims).await
            .map_err(|e| anyhow::anyhow!("Stream preparation failed: {}", e))?;

        let stream_id = stream_ctx.stream_id().to_owned();
        let stream_endpoint = self.stream_channel.stream_endpoint();

        // Create async publisher (uses tmq::push::Push, not raw zmq::PUSH)
        let publisher = self.stream_channel.publisher(&stream_ctx).await
            .map_err(|e| anyhow::anyhow!("Failed to create publisher: {}", e))?;

        // Register active stream before returning
        let cancel_token = CancellationToken::new();
        self.active_fd_streams.write().await.insert(
            stream_id.clone(),
            ActiveFdStream { container_id: container_id.to_owned(), cancel_token: cancel_token.clone() },
        );

        let sandbox_id = self.container_sandbox_map.read().await
            .get(container_id).cloned();
        let sandbox_pool = self.sandbox_pool.clone();
        let container_id_owned = container_id.to_owned();
        let active_streams = self.active_fd_streams.clone();
        let stream_id_for_cleanup = stream_id.clone();

        let stream_info = StreamInfo {
            stream_id,
            endpoint: stream_endpoint,
            server_pubkey: *stream_ctx.server_pubkey(),
        };

        // Continuation: spawns FD streaming task AFTER REP is sent to client
        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            tokio::spawn(async move {
                let result = run_fd_streaming_task(
                    publisher, container_id_owned.clone(), cancel_token, sandbox_pool, sandbox_id,
                ).await;

                // Always clean up active_fd_streams entry on task exit
                active_streams.write().await.remove(&stream_id_for_cleanup);

                if let Err(e) = result {
                    warn!(container_id = %container_id_owned, error = %e, "FD streaming task failed");
                }
            });
        });

        Ok((stream_info, continuation))
    }

    /// Detach from a container's terminal I/O streams
    ///
    /// Cancels all active FD streams for the container.
    pub async fn detach(&self, container_id: &str) -> Result<()> {
        // Find and cancel all active streams for this container
        let mut active_streams = self.active_fd_streams.write().await;
        let streams_to_cancel: Vec<String> = active_streams
            .iter()
            .filter(|(_, active)| active.container_id == container_id)
            .map(|(stream_id, _)| stream_id.clone())
            .collect();

        for stream_id in streams_to_cancel {
            if let Some(active) = active_streams.remove(&stream_id) {
                info!(
                    stream_id = %stream_id,
                    container_id = %container_id,
                    "Cancelling FD stream"
                );
                active.cancel_token.cancel();
            }
        }

        Ok(())
    }
}

/// Active FD stream with cancellation support
struct ActiveFdStream {
    container_id: String,
    cancel_token: CancellationToken,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FD Streaming Task
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the FD streaming task that bridges container I/O to StreamService.
///
/// This task:
/// 1. Uses an async StreamPublisher (DH-derived keys, pre-authorized)
/// 2. Reads from container console (vsock/serial)
/// 3. Publishes data via StreamPublisher with HMAC authentication
async fn run_fd_streaming_task(
    mut publisher: StreamPublisher,
    container_id: String,
    cancel_token: CancellationToken,
    sandbox_pool: Arc<SandboxPool>,
    sandbox_id: Option<String>,
) -> anyhow::Result<()> {
    info!(
        container_id = %container_id,
        topic = %publisher.topic(),
        "Starting FD streaming task"
    );

    // Get sandbox for this container to access console
    let sandbox = if let Some(sid) = sandbox_id {
        sandbox_pool.get(&sid).await
    } else {
        None
    };

    // TODO: Connect to actual container console
    // In a full implementation, this would:
    // - For Cloud Hypervisor: Connect to vsock or serial console via API socket
    // - For QEMU: Connect to monitor socket or virtio-serial
    // - Read stdout/stderr and forward via publisher

    if sandbox.is_none() {
        warn!(
            container_id = %container_id,
            "No sandbox found for container, FD streaming limited"
        );
    }

    // Placeholder: keep task alive until cancellation.
    // In production, replace with actual vsock/serial reading:
    //   let data = console.read().await?;
    //   publisher.publish_data(&data).await?;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                info!(container_id = %container_id, "FD streaming cancelled");
                break;
            }
            _ = interval.tick() => {
                // Placeholder — poll vsock/serial and publish_data() here
            }
        }
    }

    // Send stream completion
    publisher.complete_ref(b"").await
        .map_err(|e| anyhow::anyhow!("Failed to send stream completion: {}", e))?;

    info!(container_id = %container_id, "FD streaming task completed");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Domain → wire type conversions (From impls at service boundary)
// ═══════════════════════════════════════════════════════════════════════════════

impl From<&PodSandbox> for PodSandboxStatus {
    fn from(s: &PodSandbox) -> Self {
        Self {
            id: s.id.clone(),
            metadata: s.metadata.clone(),
            state: match s.state {
                PodSandboxState::SandboxReady => PodSandboxStateEnum::SandboxReady,
                PodSandboxState::SandboxNotReady => PodSandboxStateEnum::SandboxNotReady,
            },
            created_at: Timestamp {
                seconds: s.created_at.timestamp(),
                nanos: s.created_at.timestamp_subsec_nanos() as i32,
            },
            network: Default::default(),
            linux: Default::default(),
            labels: s.labels.clone(),
            annotations: s.annotations.clone(),
            runtime_handler: s.runtime_handler.clone(),
        }
    }
}

impl From<&PodSandbox> for PodSandboxInfo {
    fn from(s: &PodSandbox) -> Self {
        Self {
            id: s.id.clone(),
            metadata: s.metadata.clone(),
            state: match s.state {
                PodSandboxState::SandboxReady => PodSandboxStateEnum::SandboxReady,
                PodSandboxState::SandboxNotReady => PodSandboxStateEnum::SandboxNotReady,
            },
            created_at: Timestamp {
                seconds: s.created_at.timestamp(),
                nanos: s.created_at.timestamp_subsec_nanos() as i32,
            },
            labels: s.labels.clone(),
            annotations: s.annotations.clone(),
            runtime_handler: s.runtime_handler.clone(),
        }
    }
}

impl From<&PodSandbox> for PodSandboxStats {
    fn from(s: &PodSandbox) -> Self {
        Self {
            attributes: PodSandboxAttributes {
                id: s.id.clone(),
                metadata: s.metadata.clone(),
                labels: s.labels.clone(),
                annotations: s.annotations.clone(),
            },
            linux: LinuxPodSandboxStats::default(),
        }
    }
}

impl From<&Container> for ContainerStatus {
    fn from(c: &Container) -> Self {
        Self {
            id: c.id.clone(),
            metadata: c.metadata.clone(),
            state: match c.state {
                ContainerState::ContainerCreated => ContainerStateEnum::ContainerCreated,
                ContainerState::ContainerRunning => ContainerStateEnum::ContainerRunning,
                ContainerState::ContainerExited => ContainerStateEnum::ContainerExited,
                ContainerState::ContainerUnknown => ContainerStateEnum::ContainerUnknown,
            },
            created_at: Timestamp {
                seconds: c.created_at.timestamp(),
                nanos: c.created_at.timestamp_subsec_nanos() as i32,
            },
            started_at: Timestamp { seconds: 0, nanos: 0 },
            finished_at: Timestamp { seconds: 0, nanos: 0 },
            exit_code: c.exit_code.unwrap_or(0),
            image: c.image.clone(),
            image_ref: String::new(),
            reason: String::new(),
            message: String::new(),
            labels: c.labels.clone(),
            annotations: c.annotations.clone(),
            mounts: Vec::new(),
            log_path: String::new(),
        }
    }
}

impl From<&Container> for ContainerInfo {
    fn from(c: &Container) -> Self {
        Self {
            id: c.id.clone(),
            pod_sandbox_id: c.pod_sandbox_id.clone(),
            metadata: c.metadata.clone(),
            image: c.image.clone(),
            image_ref: String::new(),
            state: match c.state {
                ContainerState::ContainerCreated => ContainerStateEnum::ContainerCreated,
                ContainerState::ContainerRunning => ContainerStateEnum::ContainerRunning,
                ContainerState::ContainerExited => ContainerStateEnum::ContainerExited,
                ContainerState::ContainerUnknown => ContainerStateEnum::ContainerUnknown,
            },
            created_at: Timestamp {
                seconds: c.created_at.timestamp(),
                nanos: c.created_at.timestamp_subsec_nanos() as i32,
            },
            labels: c.labels.clone(),
            annotations: c.annotations.clone(),
        }
    }
}

impl From<&Container> for ContainerStats {
    fn from(c: &Container) -> Self {
        Self {
            attributes: ContainerAttributes {
                id: c.id.clone(),
                metadata: c.metadata.clone(),
                labels: c.labels.clone(),
                annotations: c.annotations.clone(),
            },
            cpu: Default::default(),
            memory: Default::default(),
            writable_layer: Default::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Generated scope handler traits (typed inner dispatch)
// ═══════════════════════════════════════════════════════════════════════════════

use crate::generated::worker_client::{
    WorkerHandler, dispatch_worker,
    RuntimeHandler, SandboxHandler, ContainerHandler, ImageHandler,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Typed Scope Handler Implementations
// ═══════════════════════════════════════════════════════════════════════════════

use super::client::{
    PodSandboxStatusResponse, ContainerStatusResponse,
    ImageInfo, ImageStatusResult,
    FilesystemUsage,
};
use crate::generated::worker_client::{
    CreateContainerRequest,
    ExecSyncRequest, PullImageRequest,
};

#[async_trait::async_trait(?Send)]
impl RuntimeHandler for WorkerService {
    async fn handle_version(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<VersionInfo> {
        let resp = self.version(value).await?;
        Ok(resp)
    }

    async fn handle_status(&self, _ctx: &EnvelopeContext, _request_id: u64, request: &StatusRequest) -> AnyhowResult<RuntimeStatus> {
        let resp = self.status(request.verbose).await?;
        Ok(resp.status)
    }
}

#[async_trait::async_trait(?Send)]
impl SandboxHandler for WorkerService {
    async fn handle_run(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &PodSandboxConfig) -> AnyhowResult<String> {
        let sandbox_id = self.run_pod_sandbox(data).await?;
        Ok(sandbox_id)
    }

    async fn handle_stop(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<()> {
        self.stop_pod_sandbox(value).await?;
        Ok(())
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<()> {
        self.remove_pod_sandbox(value).await?;
        Ok(())
    }

    async fn handle_status(&self, _ctx: &EnvelopeContext, _request_id: u64, request: &PodSandboxStatusRequest) -> AnyhowResult<PodSandboxStatusResponse> {
        let (status, info) = self.pod_sandbox_status(&request.pod_sandbox_id, request.verbose).await?;
        Ok(PodSandboxStatusResponse { status, info })
    }

    async fn handle_list(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &PodSandboxFilter) -> AnyhowResult<Vec<PodSandboxInfo>> {
        // Convert generated Filter to internal Filter with domain enums
        let filter = super::client::PodSandboxFilter {
            id: if data.id.is_empty() { None } else { Some(data.id.clone()) },
            state: None, // state filtering handled internally if needed
            label_selector: data.label_selector.clone(),
        };
        let sandboxes = self.list_pod_sandbox(Some(&filter)).await?;
        Ok(sandboxes)
    }

    async fn handle_stats(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<PodSandboxStats> {
        let stats = self.pod_sandbox_stats(value).await?;
        Ok(stats)
    }

    async fn handle_list_stats(&self, _ctx: &EnvelopeContext, _request_id: u64, filter: &PodSandboxStatsFilter) -> AnyhowResult<Vec<PodSandboxStats>> {
        // Convert generated Filter to internal Filter
        let internal_filter = super::client::PodSandboxStatsFilter {
            id: if filter.id.is_empty() { None } else { Some(filter.id.clone()) },
            label_selector: filter.label_selector.clone(),
        };
        let stats = self.list_pod_sandbox_stats(Some(&internal_filter)).await?;
        Ok(stats)
    }
}

#[async_trait::async_trait(?Send)]
impl ContainerHandler for WorkerService {
    async fn handle_create(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &CreateContainerRequest) -> AnyhowResult<String> {
        let container_id = self.create_container(&data.pod_sandbox_id, &data.config, &data.sandbox_config).await?;
        Ok(container_id)
    }

    async fn handle_start(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<()> {
        self.start_container(value).await?;
        Ok(())
    }

    async fn handle_stop(&self, _ctx: &EnvelopeContext, _request_id: u64, request: &StopContainerRequest) -> AnyhowResult<()> {
        self.stop_container(&request.container_id, request.timeout).await?;
        Ok(())
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<()> {
        self.remove_container(value).await?;
        Ok(())
    }

    async fn handle_status(&self, _ctx: &EnvelopeContext, _request_id: u64, request: &ContainerStatusRequest) -> AnyhowResult<ContainerStatusResponse> {
        let (status, info) = self.container_status(&request.container_id, request.verbose).await?;
        Ok(ContainerStatusResponse { status, info })
    }

    async fn handle_list(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &ContainerFilter) -> AnyhowResult<Vec<ContainerInfo>> {
        // Convert generated Filter to internal Filter with domain enums
        let filter = super::client::ContainerFilter {
            id: if data.id.is_empty() { None } else { Some(data.id.clone()) },
            pod_sandbox_id: if data.pod_sandbox_id.is_empty() { None } else { Some(data.pod_sandbox_id.clone()) },
            state: None, // state filtering handled internally if needed
            label_selector: data.label_selector.clone(),
        };
        let containers = self.list_containers(Some(&filter)).await?;
        Ok(containers)
    }

    async fn handle_stats(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<ContainerStats> {
        let stats = self.container_stats(value).await?;
        Ok(stats)
    }

    async fn handle_list_stats(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &ContainerStatsFilter) -> AnyhowResult<Vec<ContainerStats>> {
        // Convert generated Filter to internal Filter
        let filter = super::client::ContainerStatsFilter {
            id: if data.id.is_empty() { None } else { Some(data.id.clone()) },
            pod_sandbox_id: if data.pod_sandbox_id.is_empty() { None } else { Some(data.pod_sandbox_id.clone()) },
            label_selector: data.label_selector.clone(),
        };
        let stats = self.list_container_stats(Some(&filter)).await?;
        Ok(stats)
    }

    async fn handle_exec(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &ExecSyncRequest) -> AnyhowResult<ExecSyncResult> {
        let resp = self.exec_sync(&data.container_id, &data.cmd, data.timeout).await?;
        Ok(resp)
    }

    async fn handle_attach(&self, ctx: &EnvelopeContext, _request_id: u64, request: &AttachRequest) -> AnyhowResult<(StreamInfo, hyprstream_rpc::service::Continuation)> {
        let _ = &request.fds; // fds used for future fd selection
        let (stream_info, continuation) = self.prepare_attach(ctx, &request.container_id).await?;
        Ok((stream_info, continuation))
    }

    async fn handle_detach(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> AnyhowResult<()> {
        self.detach(value).await?;
        Ok(())
    }

}

#[async_trait::async_trait(?Send)]
impl ImageHandler for WorkerService {
    async fn handle_list(&self, _ctx: &EnvelopeContext, _request_id: u64, filter: &ImageFilter) -> AnyhowResult<Vec<ImageInfo>> {
        let _ = filter; // filter not yet used
        let images = self.rafs_store.list_images().await?;
        Ok(images)
    }

    async fn handle_status(&self, _ctx: &EnvelopeContext, _request_id: u64, request: &ImageStatusRequest) -> AnyhowResult<ImageStatusResult> {
        let image_ref = &request.image.image;
        let status = self.rafs_store.image_status(image_ref, request.verbose).await?;
        Ok(status)
    }

    async fn handle_pull(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &PullImageRequest) -> AnyhowResult<String> {
        let image_ref = &data.image.image;
        let auth = if !data.auth.username.is_empty() {
            Some(crate::image::AuthConfig {
                username: data.auth.username.clone(),
                password: data.auth.password.clone(),
                auth: String::new(),
                server_address: String::new(),
                identity_token: String::new(),
                registry_token: String::new(),
            })
        } else {
            None
        };
        let image_id = self.rafs_store.pull_with_auth(image_ref, auth.as_ref()).await?;
        Ok(image_id)
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64, data: &ImageSpec) -> AnyhowResult<()> {
        let image_ref = &data.image;
        self.rafs_store.remove_image(image_ref).await?;
        Ok(())
    }

    async fn handle_fs_info(&self, _ctx: &EnvelopeContext, _request_id: u64) -> AnyhowResult<Vec<FilesystemUsage>> {
        let usage = self.rafs_store.fs_info().await?;
        Ok(usage)
    }
}

/// Super-trait with authorization — all scope traits are implemented above.
#[async_trait::async_trait(?Send)]
impl WorkerHandler for WorkerService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> AnyhowResult<()> {
        if let Some(ref auth_fn) = self.authorize_fn {
            let subject = ctx.subject().to_string();
            let allowed = auth_fn(subject.clone(), resource.to_owned(), operation.to_owned()).await
                .unwrap_or_else(|e| {
                    warn!("Policy check failed for {} on {}: {} - denying access", subject, resource, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
            }
        } else {
            // No authorization configured — fail-closed
            anyhow::bail!("Authorization not configured for worker service")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation — delegates to generated dispatch_worker
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait(?Send)]
impl ZmqService for WorkerService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> AnyhowResult<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
        debug!(
            "Worker request from {} (request_id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_worker(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

#[cfg(test)]
#[allow(clippy::print_stderr)]
mod tests {
    use super::*;
    use crate::runtime::{PodSandboxConfig, ContainerConfig};
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::transport::TransportConfig;
    use tempfile::TempDir;

    /// Create a test WorkerService with temporary directories
    async fn create_test_service() -> std::result::Result<(WorkerService, TempDir), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let base_path = temp_dir.path();

        let image_config = ImageConfig {
            blobs_dir: base_path.join("blobs"),
            bootstrap_dir: base_path.join("bootstrap"),
            cache_dir: base_path.join("cache"),
            refs_dir: base_path.join("refs"),
            runtime_dir: base_path.join("runtime"),
            dragonfly_peer: None,
            fuse_threads: 4,
            chunk_size: 1024 * 1024,
            lazy_loading: true,
            cache_size_bytes: 1024 * 1024 * 1024, // 1GB for tests
        };

        let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);

        let pool_config = PoolConfig {
            runtime_dir: base_path.join("sandboxes"),
            cloud_init_dir: base_path.join("cloud-init"),
            vm_image: base_path.join("vm/rootfs.img"),
            kernel_path: base_path.join("vm/vmlinux"),
            ..PoolConfig::default()
        };

        // Create a zmq context for event publishing
        let context = Arc::new(zmq::Context::new());
        let transport = TransportConfig::inproc("test-worker-service");
        let (signing_key, _verifying_key) = generate_signing_keypair();

        let service = WorkerService::new(pool_config, image_config, rafs_store, context, transport, signing_key)?;
        Ok((service, temp_dir))
    }

    #[tokio::test]
    async fn test_sandbox_lifecycle() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (service, _temp_dir) = create_test_service().await?;

        // Create sandbox - this requires cloud-hypervisor and kata runtime directories
        let config = PodSandboxConfig::default();
        let result = service.run_pod_sandbox(&config).await;

        // Skip test if cloud-hypervisor not installed or kata runtime not available
        match &result {
            Err(WorkerError::VmStartFailed(msg))
                if msg.contains("No such file or directory")
                    || msg.contains("failed to create sandbox directory")
                    || msg.contains("/run/kata")
                    || msg.contains("launch failed") =>
            {
                eprintln!("Skipping test: cloud-hypervisor or kata runtime not available");
                return Ok(());
            }
            _ => {}
        }

        let sandbox_id = result?;

        // Check status
        let (status, _info) = service.pod_sandbox_status(&sandbox_id, false).await?;
        assert_eq!(status.state, PodSandboxStateEnum::SandboxReady);

        // List sandboxes
        let sandboxes = service.list_pod_sandbox(None).await?;
        assert_eq!(sandboxes.len(), 1);

        // Stop and remove
        service.stop_pod_sandbox(&sandbox_id).await?;
        service.remove_pod_sandbox(&sandbox_id).await?;

        // Should be empty
        let sandboxes = service.list_pod_sandbox(None).await?;
        assert_eq!(sandboxes.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_container_lifecycle() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (service, _temp_dir) = create_test_service().await?;

        // Create sandbox first - this requires cloud-hypervisor and kata runtime directories
        let sandbox_config = PodSandboxConfig::default();
        let result = service.run_pod_sandbox(&sandbox_config).await;

        // Skip test if cloud-hypervisor not installed or kata runtime not available
        match &result {
            Err(WorkerError::VmStartFailed(msg))
                if msg.contains("No such file or directory")
                    || msg.contains("failed to create sandbox directory")
                    || msg.contains("/run/kata")
                    || msg.contains("launch failed") =>
            {
                eprintln!("Skipping test: cloud-hypervisor or kata runtime not available");
                return Ok(());
            }
            _ => {}
        }

        let sandbox_id = result?;

        // Create container
        let container_config = ContainerConfig::default();
        let container_id = service
            .create_container(&sandbox_id, &container_config, &sandbox_config)
            .await?;

        // Check status
        let (status, _info) = service.container_status(&container_id, false).await?;
        assert_eq!(status.state, ContainerStateEnum::ContainerCreated);

        // Start container
        service.start_container(&container_id).await?;
        let (status, _info) = service.container_status(&container_id, false).await?;
        assert_eq!(status.state, ContainerStateEnum::ContainerRunning);

        // Stop container
        service.stop_container(&container_id, 30).await?;
        let (status, _info) = service.container_status(&container_id, false).await?;
        assert_eq!(status.state, ContainerStateEnum::ContainerExited);

        // Remove container
        service.remove_container(&container_id).await?;

        // Cleanup
        service.remove_pod_sandbox(&sandbox_id).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_service_version() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (service, _temp_dir) = create_test_service().await?;

        // Version should work without cloud-hypervisor
        let version = service.version("v1").await?;
        assert_eq!(version.runtime_name, super::super::RUNTIME_NAME);
        assert_eq!(version.runtime_api_version, "v1");
        Ok(())
    }

    #[tokio::test]
    async fn test_service_status() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (service, _temp_dir) = create_test_service().await?;

        // Status should work without cloud-hypervisor
        let status = service.status(true).await?;
        assert!(!status.status.conditions.is_empty());
        assert!(status.info.contains_key("warm_pool_size"));
        Ok(())
    }
}
