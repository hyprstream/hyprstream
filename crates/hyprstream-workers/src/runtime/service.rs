//! WorkerService - CRI RuntimeClient + ImageClient via ZMQ
//!
//! Implements ZmqService trait for handling CRI-aligned requests.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use anyhow::Result as AnyhowResult;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use tracing::{debug, info, warn};

// Import ZMQ service infrastructure from hyprstream-rpc
use hyprstream_rpc::prelude::SigningKey;
use hyprstream_rpc::service::{EnvelopeContext, ZmqService};
use hyprstream_rpc::streaming::{BatchingConfig, StreamBuilder};
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
use crate::workers_capnp;

use super::client::{
    ContainerFilter, ContainerStats, ContainerStatsFilter, ContainerStatusResponse,
    ExecSyncResponse, PodSandboxFilter, PodSandboxStats, PodSandboxStatsFilter,
    PodSandboxStatusResponse, RuntimeCondition, RuntimeStatus, StatusResponse, VersionResponse,
};
use super::container::{Container, ContainerConfig, ContainerState};
use super::pool::SandboxPool;
use super::sandbox::{PodSandbox, PodSandboxConfig};
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

    /// Captured runtime handle for async operations in sync handlers
    runtime_handle: tokio::runtime::Handle,

    /// Event publisher for lifecycle events (sandbox/container started/stopped)
    event_publisher: tokio::sync::Mutex<EventPublisher>,

    /// Pending FD streams awaiting DH handshake completion (stream_id -> PendingFdStream)
    pending_fd_streams: RwLock<HashMap<String, PendingFdStream>>,

    /// Active FD streams (stream_id -> ActiveFdStream)
    active_fd_streams: RwLock<HashMap<String, ActiveFdStream>>,

    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
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

        Ok(Self {
            sandbox_pool,
            rafs_store,
            containers: RwLock::new(HashMap::new()),
            container_sandbox_map: RwLock::new(HashMap::new()),
            runtime_handle: tokio::runtime::Handle::current(),
            event_publisher: tokio::sync::Mutex::new(event_publisher),
            pending_fd_streams: RwLock::new(HashMap::new()),
            active_fd_streams: RwLock::new(HashMap::new()),
            context,
            transport,
            signing_key,
        })
    }

    /// Create a new WorkerService with an explicit runtime handle
    ///
    /// Use this when not called from within a tokio runtime context.
    pub fn with_runtime_handle(
        pool_config: PoolConfig,
        image_config: ImageConfig,
        rafs_store: Arc<RafsStore>,
        runtime_handle: tokio::runtime::Handle,
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

        Ok(Self {
            sandbox_pool,
            rafs_store,
            containers: RwLock::new(HashMap::new()),
            container_sandbox_map: RwLock::new(HashMap::new()),
            runtime_handle,
            event_publisher: tokio::sync::Mutex::new(event_publisher),
            pending_fd_streams: RwLock::new(HashMap::new()),
            active_fd_streams: RwLock::new(HashMap::new()),
            context,
            transport,
            signing_key,
        })
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
    pub async fn version(&self, _version: &str) -> Result<VersionResponse> {
        Ok(VersionResponse {
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
                reason: "".to_owned(),
                message: "Runtime is ready".to_owned(),
            },
            RuntimeCondition {
                condition_type: "NetworkReady".to_owned(),
                status: true,
                reason: "".to_owned(),
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
    pub async fn pod_sandbox_status(
        &self,
        pod_sandbox_id: &str,
        verbose: bool,
    ) -> Result<PodSandboxStatusResponse> {
        let sandbox = self
            .sandbox_pool
            .get(pod_sandbox_id)
            .await
            .ok_or_else(|| WorkerError::SandboxNotFound(pod_sandbox_id.to_owned()))?;

        let mut info = HashMap::new();
        if verbose {
            // Get PIDs from hypervisor (host-level VM management)
            let pids = sandbox.get_pids().await;
            let pid_str = pids.map_or_else(|| "none".to_owned(), |p| {
                p.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",")
            });
            info.insert("vm_pids".to_owned(), pid_str);
        }

        Ok(PodSandboxStatusResponse {
            status: sandbox.status(),
            info,
        })
    }

    /// List pod sandboxes
    pub async fn list_pod_sandbox(
        &self,
        filter: Option<&PodSandboxFilter>,
    ) -> Result<Vec<PodSandbox>> {
        let mut sandboxes = self.sandbox_pool.list_active().await;

        // Apply filters
        if let Some(f) = filter {
            if let Some(id) = &f.id {
                sandboxes.retain(|s| &s.id == id);
            }
            if let Some(state) = f.state {
                sandboxes.retain(|s| s.state == state);
            }
            for (key, value) in &f.label_selector {
                sandboxes.retain(|s| s.labels.get(key) == Some(value));
            }
        }

        Ok(sandboxes)
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
    pub async fn container_status(
        &self,
        container_id: &str,
        verbose: bool,
    ) -> Result<ContainerStatusResponse> {
        let containers = self.containers.read().await;

        let container = containers
            .get(container_id)
            .ok_or_else(|| WorkerError::ContainerNotFound(container_id.to_owned()))?;

        let mut info = HashMap::new();
        if verbose {
            info.insert("pid".to_owned(), container.pid.map_or_else(|| "none".to_owned(), |p| p.to_string()));
        }

        Ok(ContainerStatusResponse {
            status: container.status(),
            info,
        })
    }

    /// List containers
    pub async fn list_containers(
        &self,
        filter: Option<&ContainerFilter>,
    ) -> Result<Vec<Container>> {
        let containers = self.containers.read().await;
        let mut result: Vec<Container> = containers.values().cloned().collect();

        // Apply filters
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
            for (key, value) in &f.label_selector {
                result.retain(|c| c.labels.get(key) == Some(value));
            }
        }

        Ok(result)
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
    ) -> Result<ExecSyncResponse> {
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
        Ok(super::client::PodSandboxStats {
            attributes: super::client::PodSandboxAttributes {
                id: sandbox.id.clone(),
                metadata: sandbox.metadata.clone(),
                labels: sandbox.labels.clone(),
                annotations: sandbox.annotations.clone(),
            },
            linux: None,
        })
    }

    /// List pod sandbox stats
    pub async fn list_pod_sandbox_stats(
        &self,
        filter: Option<&PodSandboxStatsFilter>,
    ) -> Result<Vec<PodSandboxStats>> {
        let sandboxes = self.sandbox_pool.list_active().await;
        let mut results = Vec::new();

        for sandbox in sandboxes {
            // Apply filter
            if let Some(f) = filter {
                if let Some(id) = &f.id {
                    if &sandbox.id != id {
                        continue;
                    }
                }
                let mut matches_labels = true;
                for (key, value) in &f.label_selector {
                    if sandbox.labels.get(key) != Some(value) {
                        matches_labels = false;
                        break;
                    }
                }
                if !matches_labels {
                    continue;
                }
            }

            results.push(super::client::PodSandboxStats {
                attributes: super::client::PodSandboxAttributes {
                    id: sandbox.id.clone(),
                    metadata: sandbox.metadata.clone(),
                    labels: sandbox.labels.clone(),
                    annotations: sandbox.annotations.clone(),
                },
                linux: None,
            });
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
        Ok(ContainerStats {
            attributes: super::client::ContainerAttributes {
                id: container.id.clone(),
                metadata: container.metadata.clone(),
                labels: container.labels.clone(),
                annotations: container.annotations.clone(),
            },
            cpu: None,
            memory: None,
            writable_layer: None,
        })
    }

    /// List container stats
    pub async fn list_container_stats(
        &self,
        filter: Option<&ContainerStatsFilter>,
    ) -> Result<Vec<ContainerStats>> {
        let containers = self.containers.read().await;
        let mut results = Vec::new();

        for container in containers.values() {
            // Apply filter
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
                for (key, value) in &f.label_selector {
                    if container.labels.get(key) != Some(value) {
                        matches_labels = false;
                        break;
                    }
                }
                if !matches_labels {
                    continue;
                }
            }

            results.push(ContainerStats {
                attributes: super::client::ContainerAttributes {
                    id: container.id.clone(),
                    metadata: container.metadata.clone(),
                    labels: container.labels.clone(),
                    annotations: container.annotations.clone(),
                },
                cpu: None,
                memory: None,
                writable_layer: None,
            });
        }

        Ok(results)
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // Terminal Attach/Detach
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Attach to a container's terminal I/O streams (tmux-like)
    ///
    /// Returns stream info for DH handshake. Client must call start_fd_stream()
    /// with their public key to complete the handshake before subscribing.
    ///
    /// # Flow
    /// 1. Client calls attach() - receives stream_id, stream_endpoint, server_pubkey
    /// 2. Client calls start_fd_stream(stream_id, client_pubkey) - completes DH handshake
    /// 3. Client derives topic/mac_key from DH shared secret
    /// 4. Client subscribes to DH-derived topic via StreamService
    /// 5. Worker streams FD data via StreamBuilder (HMAC-authenticated)
    pub async fn attach(&self, container_id: &str) -> Result<AttachResponse> {
        // Verify container exists
        let containers = self.containers.read().await;
        if !containers.contains_key(container_id) {
            return Err(anyhow::anyhow!("Container not found: {}", container_id).into());
        }
        drop(containers);

        // Generate ephemeral DH keypair for this stream
        let (server_secret, server_pubkey) = hyprstream_rpc::generate_ephemeral_keypair();
        let server_pubkey_bytes: [u8; 32] = server_pubkey.to_bytes();

        // Generate unique stream ID
        let stream_id = format!("fd-{}-{}", container_id, uuid::Uuid::new_v4());

        // Get stream endpoint from EndpointRegistry (StreamService's PULL socket)
        let stream_endpoint = hyprstream_rpc::registry::global()
            .endpoint("stream", hyprstream_rpc::registry::SocketKind::Pull)
            .to_zmq_string();

        // Store pending stream for DH handshake completion
        let pending = PendingFdStream {
            container_id: container_id.to_owned(),
            server_secret,
            server_pubkey: server_pubkey_bytes,
            stream_endpoint: stream_endpoint.clone(),
        };
        self.pending_fd_streams.write().await.insert(stream_id.clone(), pending);

        Ok(AttachResponse {
            container_id: container_id.to_owned(),
            stream_id,
            stream_endpoint,
            server_pubkey: server_pubkey_bytes,
        })
    }

    /// Complete FD stream DH handshake
    ///
    /// Client provides their public key, server derives shared secret and
    /// starts streaming FD data via StreamBuilder.
    pub async fn start_fd_stream(
        &self,
        stream_id: &str,
        client_pubkey: &[u8; 32],
    ) -> Result<FdStreamAuthResponse> {
        // Get and remove pending stream
        let pending = self.pending_fd_streams.write().await
            .remove(stream_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown stream ID: {}", stream_id))?;

        // Derive stream keys from DH
        let client_pub = hyprstream_rpc::crypto::RistrettoPublic::from_bytes(client_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid client public key"))?;
        let shared_secret = hyprstream_rpc::crypto::ristretto_dh(&pending.server_secret, &client_pub);
        let keys = hyprstream_rpc::crypto::derive_stream_keys(
            &shared_secret,
            client_pubkey,
            &pending.server_pubkey,
        ).map_err(|e| anyhow::anyhow!("Key derivation failed: {}", e))?;

        info!(
            stream_id = %stream_id,
            container_id = %pending.container_id,
            topic = %keys.topic,
            "FD stream authorized, starting I/O bridge"
        );

        // Create cancellation token for the stream
        let cancel_token = CancellationToken::new();

        // Store active stream
        let active = ActiveFdStream {
            container_id: pending.container_id.clone(),
            cancel_token: cancel_token.clone(),
        };
        self.active_fd_streams.write().await.insert(stream_id.to_owned(), active);

        // Spawn FD streaming task
        let context = self.context.clone();
        let stream_endpoint = pending.stream_endpoint.clone();
        let stream_id_owned = stream_id.to_owned();
        let container_id = pending.container_id.clone();
        let mac_key = *keys.mac_key;
        let topic = keys.topic.clone();

        // Get sandbox pool for container console access
        let sandbox_pool = self.sandbox_pool.clone();
        let container_sandbox_map = {
            let map = self.container_sandbox_map.read().await;
            map.get(&container_id).cloned()
        };

        tokio::spawn(async move {
            if let Err(e) = run_fd_streaming_task(
                context,
                stream_endpoint,
                stream_id_owned.clone(),
                container_id.clone(),
                mac_key,
                topic,
                cancel_token,
                sandbox_pool,
                container_sandbox_map,
            ).await {
                warn!(
                    stream_id = %stream_id_owned,
                    container_id = %container_id,
                    error = %e,
                    "FD streaming task failed"
                );
            }
        });

        Ok(FdStreamAuthResponse {
            stream_id: stream_id.to_owned(),
            authorized: true,
        })
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

        // Also remove any pending streams that haven't started yet
        let mut pending_streams = self.pending_fd_streams.write().await;
        pending_streams.retain(|_, pending| pending.container_id != container_id);

        Ok(())
    }
}

/// Response from attach() with stream info for DH handshake
#[derive(Debug, Clone)]
pub struct AttachResponse {
    pub container_id: String,
    pub stream_id: String,
    pub stream_endpoint: String,
    pub server_pubkey: [u8; 32],
}

/// Response from start_fd_stream() confirming authorization
#[derive(Debug, Clone)]
pub struct FdStreamAuthResponse {
    pub stream_id: String,
    pub authorized: bool,
}

/// Pending FD stream awaiting DH handshake completion
struct PendingFdStream {
    container_id: String,
    server_secret: hyprstream_rpc::crypto::RistrettoSecret,
    server_pubkey: [u8; 32],
    stream_endpoint: String,
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
/// 1. Creates a PUSH socket to send to StreamService
/// 2. Creates a StreamBuilder with DH-derived keys
/// 3. Reads from container console (vsock/serial)
/// 4. Sends data via StreamBuilder with HMAC authentication
#[allow(clippy::too_many_arguments)]
async fn run_fd_streaming_task(
    context: Arc<zmq::Context>,
    stream_endpoint: String,
    stream_id: String,
    container_id: String,
    mac_key: [u8; 32],
    topic: String,
    cancel_token: CancellationToken,
    sandbox_pool: Arc<SandboxPool>,
    sandbox_id: Option<String>,
) -> anyhow::Result<()> {
    info!(
        stream_id = %stream_id,
        container_id = %container_id,
        endpoint = %stream_endpoint,
        "Starting FD streaming task"
    );

    // Create PUSH socket to send to StreamService
    let push_socket = context.socket(zmq::PUSH)?;
    push_socket.connect(&stream_endpoint)?;

    // Create StreamBuilder with DH-derived keys
    let config = BatchingConfig {
        min_batch_size: 1,  // Send immediately for interactive terminal
        max_batch_size: 8,
        max_block_bytes: 4096,
        ..Default::default()
    };
    let mut builder = StreamBuilder::new(config, mac_key, topic.clone());

    // Get sandbox for this container to access console
    let sandbox = if let Some(sid) = sandbox_id {
        sandbox_pool.get(&sid).await
    } else {
        None
    };

    // TODO: Connect to actual container console
    // For now, we'll use a placeholder that demonstrates the streaming flow.
    // In a full implementation, this would:
    // - For Cloud Hypervisor: Connect to vsock or serial console via API socket
    // - For QEMU: Connect to monitor socket or virtio-serial
    // - Read stdout/stderr and forward to StreamBuilder

    if sandbox.is_none() {
        warn!(
            stream_id = %stream_id,
            container_id = %container_id,
            "No sandbox found for container, FD streaming limited"
        );
    }

    // Placeholder: Simulate console output for demonstration
    // In production, this would be replaced with actual vsock/serial reading
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                info!(
                    stream_id = %stream_id,
                    container_id = %container_id,
                    "FD streaming cancelled"
                );
                break;
            }
            _ = interval.tick() => {
                // In a real implementation, we would:
                // 1. Poll the vsock/serial for available data
                // 2. Read the data
                // 3. Send via StreamBuilder
                //
                // For now, this is a placeholder that keeps the task alive
                // and demonstrates the cancellation flow.
                //
                // Example of how data would be sent:
                // if let Some(data) = console.try_read()? {
                //     if let Some(frames) = builder.add_data(&data, rate)? {
                //         frames.send(&push_socket)?;
                //     }
                // }
            }
        }
    }

    // Flush any remaining data and send completion
    if let Some(frames) = builder.add_complete(b"")? {
        frames.send(&push_socket)?;
    }

    info!(
        stream_id = %stream_id,
        container_id = %container_id,
        "FD streaming task completed"
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation (uses hyprstream_rpc::service::ZmqService)
// ═══════════════════════════════════════════════════════════════════════════════

impl ZmqService for WorkerService {
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> AnyhowResult<Vec<u8>> {
        debug!(
            "Worker request from {} (request_id={})",
            ctx.casbin_subject(),
            ctx.request_id
        );

        // Check message type prefix
        // - 0x00 or no prefix: RuntimeRequest (for backwards compatibility)
        // - 0x01: ImageRequest
        if payload.is_empty() {
            return Ok(build_error_response(0, "Empty payload"));
        }

        match payload[0] {
            0x01 => {
                // ImageRequest - skip the prefix byte
                tracing::debug!("Routing to ImageRequest handler (0x01 prefix, {} bytes remaining)", payload.len() - 1);
                if let Some(result) = self.try_handle_image_request(&payload[1..]) {
                    return result;
                }
                tracing::warn!("ImageRequest parsing failed, trying RuntimeRequest fallback");
            }
            _ => {
                // RuntimeRequest - raw capnp message without prefix
                // Note: 0x00 is a valid capnp header byte (segment count), not a prefix!
                tracing::debug!("Routing to RuntimeRequest handler ({} bytes)", payload.len());
                if let Some(result) = self.try_handle_runtime_request(payload) {
                    return result;
                }
                tracing::warn!("RuntimeRequest parsing failed");
            }
        }

        // Fallback - shouldn't normally reach here
        tracing::error!("All request parsing failed for {} byte payload", payload.len());
        Ok(build_error_response(0, "Failed to handle request"))
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

impl WorkerService {
    /// Try to handle payload as RuntimeRequest
    fn try_handle_runtime_request(&self, payload: &[u8]) -> Option<AnyhowResult<Vec<u8>>> {
        let reader = match serialize::read_message(payload, ReaderOptions::new()) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("RuntimeRequest: Failed to read message: {}", e);
                return None;
            }
        };

        let req = match reader.get_root::<workers_capnp::runtime_request::Reader>() {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("RuntimeRequest: Failed to get root: {}", e);
                return None;
            }
        };

        let request_id = req.get_id();
        tracing::debug!("RuntimeRequest: parsed request_id={}", request_id);

        use workers_capnp::runtime_request::Which;

        let which = match req.which() {
            Ok(w) => w,
            Err(e) => {
                tracing::debug!("RuntimeRequest: Failed to get discriminant: {} (maybe ImageRequest?)", e);
                return None; // Unknown discriminant - try ImageRequest
            }
        };
        tracing::debug!("RuntimeRequest: discriminant matched successfully");

        // Helper macro to extract capnp values safely
        macro_rules! try_capnp {
            ($expr:expr) => {
                match $expr {
                    Ok(v) => v,
                    Err(e) => return Some(Ok(build_error_response(request_id, &e.to_string()))),
                }
            };
        }

        Some(match which {
            Which::Version(version) => {
                let version_str = try_capnp!(try_capnp!(version).to_str());
                match self.runtime_handle.block_on(self.version(version_str)) {
                    Ok(resp) => Ok(build_version_response(request_id, &resp)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Status(status_req) => {
                let status_req = try_capnp!(status_req);
                let verbose = status_req.get_verbose();
                match self.runtime_handle.block_on(self.status(verbose)) {
                    Ok(resp) => Ok(build_status_response(request_id, &resp)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RunPodSandbox(config) => {
                let _config = try_capnp!(config);
                // TODO: Parse PodSandboxConfig from capnp
                let pod_config = PodSandboxConfig::default();
                match self.runtime_handle.block_on(self.run_pod_sandbox(&pod_config)) {
                    Ok(sandbox_id) => Ok(build_sandbox_id_response(request_id, &sandbox_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StopPodSandbox(pod_sandbox_id) => {
                let id = try_capnp!(try_capnp!(pod_sandbox_id).to_str());
                match self.runtime_handle.block_on(self.stop_pod_sandbox(id)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RemovePodSandbox(pod_sandbox_id) => {
                let id = try_capnp!(try_capnp!(pod_sandbox_id).to_str());
                match self.runtime_handle.block_on(self.remove_pod_sandbox(id)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::PodSandboxStatus(status_req) => {
                let status_req = try_capnp!(status_req);
                let id = try_capnp!(try_capnp!(status_req.get_pod_sandbox_id()).to_str());
                let verbose = status_req.get_verbose();
                match self.runtime_handle.block_on(self.pod_sandbox_status(id, verbose)) {
                    Ok(resp) => Ok(build_sandbox_status_response(request_id, &resp)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListPodSandbox(_filter) => {
                // TODO: Parse filter from capnp
                match self.runtime_handle.block_on(self.list_pod_sandbox(None)) {
                    Ok(sandboxes) => Ok(build_sandboxes_response(request_id, &sandboxes)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::CreateContainer(create_req) => {
                let create_req = try_capnp!(create_req);
                let pod_sandbox_id = try_capnp!(try_capnp!(create_req.get_pod_sandbox_id()).to_str());
                // TODO: Parse ContainerConfig and PodSandboxConfig from capnp
                let container_config = ContainerConfig::default();
                let sandbox_config = PodSandboxConfig::default();
                match self.runtime_handle.block_on(
                    self.create_container(pod_sandbox_id, &container_config, &sandbox_config)
                ) {
                    Ok(container_id) => Ok(build_container_id_response(request_id, &container_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StartContainer(container_id) => {
                let id = try_capnp!(try_capnp!(container_id).to_str());
                match self.runtime_handle.block_on(self.start_container(id)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StopContainer(stop_req) => {
                let stop_req = try_capnp!(stop_req);
                let id = try_capnp!(try_capnp!(stop_req.get_container_id()).to_str());
                let timeout = stop_req.get_timeout();
                match self.runtime_handle.block_on(self.stop_container(id, timeout)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RemoveContainer(container_id) => {
                let id = try_capnp!(try_capnp!(container_id).to_str());
                match self.runtime_handle.block_on(self.remove_container(id)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ContainerStatus(status_req) => {
                let status_req = try_capnp!(status_req);
                let id = try_capnp!(try_capnp!(status_req.get_container_id()).to_str());
                let verbose = status_req.get_verbose();
                match self.runtime_handle.block_on(self.container_status(id, verbose)) {
                    Ok(resp) => Ok(build_container_status_response(request_id, &resp)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListContainers(_filter) => {
                // TODO: Parse filter from capnp
                match self.runtime_handle.block_on(self.list_containers(None)) {
                    Ok(containers) => Ok(build_containers_response(request_id, &containers)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ExecSync(exec_req) => {
                let exec_req = try_capnp!(exec_req);
                let container_id = try_capnp!(try_capnp!(exec_req.get_container_id()).to_str());
                let cmd_list = try_capnp!(exec_req.get_cmd());
                let mut cmd: Vec<String> = Vec::new();
                for i in 0..cmd_list.len() {
                    cmd.push(try_capnp!(try_capnp!(cmd_list.get(i)).to_str()).to_owned());
                }
                let timeout = exec_req.get_timeout();
                match self.runtime_handle.block_on(self.exec_sync(container_id, &cmd, timeout)) {
                    Ok(resp) => Ok(build_exec_result_response(request_id, &resp)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::PodSandboxStats(pod_sandbox_id) => {
                let id = try_capnp!(try_capnp!(pod_sandbox_id).to_str());
                match self.runtime_handle.block_on(self.pod_sandbox_stats(id)) {
                    Ok(stats) => Ok(build_sandbox_stats_response(request_id, &stats)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListPodSandboxStats(_filter) => {
                // TODO: Parse filter from capnp
                match self.runtime_handle.block_on(self.list_pod_sandbox_stats(None)) {
                    Ok(stats) => Ok(build_sandbox_stats_list_response(request_id, &stats)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ContainerStats(container_id) => {
                let id = try_capnp!(try_capnp!(container_id).to_str());
                match self.runtime_handle.block_on(self.container_stats(id)) {
                    Ok(stats) => Ok(build_container_stats_response(request_id, &stats)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListContainerStats(_filter) => {
                // TODO: Parse filter from capnp
                match self.runtime_handle.block_on(self.list_container_stats(None)) {
                    Ok(stats) => Ok(build_container_stats_list_response(request_id, &stats)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Attach(attach_req) => {
                let attach_req = attach_req.ok()?;
                let container_id = attach_req.get_container_id().ok()?.to_str().ok()?.to_owned();
                match self.runtime_handle.block_on(self.attach(&container_id)) {
                    Ok(response) => Ok(build_attach_response(request_id, &response)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Detach(container_id) => {
                let container_id = container_id.ok()?.to_str().ok()?.to_owned();
                match self.runtime_handle.block_on(self.detach(&container_id)) {
                    Ok(()) => Ok(build_success_response(request_id)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StartFdStream(req) => {
                let req = req.ok()?;
                let stream_id = req.get_stream_id().ok()?.to_str().ok()?.to_owned();
                let client_pubkey_data = req.get_client_pubkey().ok()?;
                if client_pubkey_data.len() != 32 {
                    return Some(Ok(build_error_response(request_id, "Invalid client pubkey length")));
                }
                let mut client_pubkey = [0u8; 32];
                client_pubkey.copy_from_slice(client_pubkey_data);
                match self.runtime_handle.block_on(self.start_fd_stream(&stream_id, &client_pubkey)) {
                    Ok(response) => Ok(build_fd_stream_auth_response(request_id, &response)),
                    Err(e) => Ok(build_error_response(request_id, &e.to_string())),
                }
            }
        })
    }

    /// Try to handle payload as ImageRequest
    fn try_handle_image_request(&self, payload: &[u8]) -> Option<AnyhowResult<Vec<u8>>> {
        tracing::debug!("ImageRequest: attempting to parse {} bytes", payload.len());
        let reader = match serialize::read_message(payload, ReaderOptions::new()) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("ImageRequest: Failed to read capnp message: {}", e);
                return None;
            }
        };

        let req = match reader.get_root::<workers_capnp::image_request::Reader>() {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("ImageRequest: Failed to get root: {}", e);
                return None;
            }
        };

        let request_id = req.get_id();
        tracing::debug!("ImageRequest: parsed request_id={}", request_id);

        use workers_capnp::image_request::Which;

        let which = match req.which() {
            Ok(w) => w,
            Err(e) => {
                tracing::debug!("ImageRequest: Unknown variant ({})", e);
                return None;
            }
        };
        tracing::debug!("ImageRequest: variant matched successfully");

        Some(match which {
            Which::ListImages(_filter) => {
                // TODO: Parse filter
                match self.runtime_handle.block_on(self.rafs_store.list_images()) {
                    Ok(images) => Ok(build_images_response(request_id, &images)),
                    Err(e) => Ok(build_image_error_response(request_id, &e.to_string())),
                }
            }

            Which::ImageStatus(status_req) => {
                let status_req = match status_req {
                    Ok(r) => r,
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let image_spec = match status_req.get_image() {
                    Ok(spec) => spec,
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let image_ref = match image_spec.get_image() {
                    Ok(s) => match s.to_str() {
                        Ok(s) => s.to_owned(),
                        Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                    },
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let verbose = status_req.get_verbose();
                match self.runtime_handle.block_on(self.rafs_store.image_status(&image_ref, verbose)) {
                    Ok(status) => Ok(build_image_status_response(request_id, &status)),
                    Err(e) => Ok(build_image_error_response(request_id, &e.to_string())),
                }
            }

            Which::PullImage(pull_req) => {
                let pull_req = match pull_req {
                    Ok(r) => r,
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let image_spec = match pull_req.get_image() {
                    Ok(spec) => spec,
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let image_ref = match image_spec.get_image() {
                    Ok(s) => match s.to_str() {
                        Ok(s) => s.to_owned(),
                        Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                    },
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };

                // Parse optional auth
                let auth = if pull_req.has_auth() {
                    match pull_req.get_auth() {
                        Ok(auth_reader) => {
                            let username = auth_reader.get_username().ok()
                                .and_then(|s| s.to_str().ok())
                                .unwrap_or_default().to_owned();
                            let password = auth_reader.get_password().ok()
                                .and_then(|s| s.to_str().ok())
                                .unwrap_or_default().to_owned();
                            if !username.is_empty() {
                                Some(crate::image::AuthConfig {
                                    username,
                                    password,
                                    ..Default::default()
                                })
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                match self.runtime_handle.block_on(self.rafs_store.pull_with_auth(&image_ref, auth.as_ref())) {
                    Ok(image_id) => Ok(build_image_ref_response(request_id, &image_id)),
                    Err(e) => Ok(build_image_error_response(request_id, &e.to_string())),
                }
            }

            Which::RemoveImage(image_spec) => {
                let image_spec = match image_spec {
                    Ok(spec) => spec,
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                let image_ref = match image_spec.get_image() {
                    Ok(s) => match s.to_str() {
                        Ok(s) => s.to_owned(),
                        Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                    },
                    Err(e) => return Some(Ok(build_image_error_response(request_id, &e.to_string()))),
                };
                match self.runtime_handle.block_on(self.rafs_store.remove_image(&image_ref)) {
                    Ok(()) => Ok(build_image_success_response(request_id)),
                    Err(e) => Ok(build_image_error_response(request_id, &e.to_string())),
                }
            }

            Which::ImageFsInfo(()) => {
                match self.runtime_handle.block_on(self.rafs_store.fs_info()) {
                    Ok(usage) => Ok(build_fs_info_response(request_id, &usage)),
                    Err(e) => Ok(build_image_error_response(request_id, &e.to_string())),
                }
            }
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Response Builders
// ═══════════════════════════════════════════════════════════════════════════════

fn serialize_message<F>(f: F) -> AnyhowResult<Vec<u8>>
where
    F: FnOnce(&mut Builder<capnp::message::HeapAllocator>),
{
    let mut message = Builder::new_default();
    f(&mut message);
    let mut output = Vec::new();
    serialize::write_message(&mut output, &message)?;
    Ok(output)
}

fn build_success_response(request_id: u64) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        resp.set_success(());
    }).unwrap_or_default()
}

fn build_error_response(request_id: u64, message: &str) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut error = resp.init_error();
        error.set_message(message);
        error.set_code("INTERNAL");
        error.set_details("");
    }).unwrap_or_default()
}

fn build_version_response(request_id: u64, version: &VersionResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut ver = resp.init_version();
        ver.set_version(&version.version);
        ver.set_runtime_name(&version.runtime_name);
        ver.set_runtime_version(&version.runtime_version);
        ver.set_runtime_api_version(&version.runtime_api_version);
    }).unwrap_or_default()
}

fn build_attach_response(request_id: u64, attach: &AttachResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut attach_resp = resp.init_attach_response();
        attach_resp.set_container_id(&attach.container_id);
        attach_resp.set_stream_id(&attach.stream_id);
        attach_resp.set_stream_endpoint(&attach.stream_endpoint);
        attach_resp.set_server_pubkey(&attach.server_pubkey);
    }).unwrap_or_default()
}

fn build_fd_stream_auth_response(request_id: u64, auth: &FdStreamAuthResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut auth_resp = resp.init_fd_stream_authorized();
        auth_resp.set_stream_id(&auth.stream_id);
        auth_resp.set_authorized(auth.authorized);
    }).unwrap_or_default()
}

fn build_status_response(request_id: u64, status: &StatusResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let stat = resp.init_status();
        let conditions = &status.status.conditions;
        let mut cond_list = stat.init_conditions(conditions.len() as u32);
        for (i, cond) in conditions.iter().enumerate() {
            let mut c = cond_list.reborrow().get(i as u32);
            c.set_condition_type(&cond.condition_type);
            c.set_status(cond.status);
            c.set_reason(&cond.reason);
            c.set_message(&cond.message);
        }
    }).unwrap_or_default()
}

fn build_sandbox_id_response(request_id: u64, sandbox_id: &str) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        resp.set_sandbox_id(sandbox_id);
    }).unwrap_or_default()
}

fn build_sandbox_status_response(request_id: u64, status_resp: &PodSandboxStatusResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut sandbox_status = resp.init_sandbox_status();
        let mut status = sandbox_status.reborrow().init_status();
        status.set_id(&status_resp.status.id);
        let mut meta = status.reborrow().init_metadata();
        meta.set_name(&status_resp.status.metadata.name);
        meta.set_uid(&status_resp.status.metadata.uid);
        meta.set_namespace(&status_resp.status.metadata.namespace);
        meta.set_attempt(status_resp.status.metadata.attempt);
    }).unwrap_or_default()
}

fn build_sandboxes_response(request_id: u64, sandboxes: &[PodSandbox]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_sandboxes(sandboxes.len() as u32);
        for (i, sandbox) in sandboxes.iter().enumerate() {
            let mut s = list.reborrow().get(i as u32);
            s.set_id(&sandbox.id);
            let mut meta = s.reborrow().init_metadata();
            meta.set_name(&sandbox.metadata.name);
            meta.set_uid(&sandbox.metadata.uid);
            meta.set_namespace(&sandbox.metadata.namespace);
            meta.set_attempt(sandbox.metadata.attempt);
            s.set_created_at(sandbox.created_at.timestamp());
            s.set_runtime_handler(&sandbox.runtime_handler);
        }
    }).unwrap_or_default()
}

fn build_container_id_response(request_id: u64, container_id: &str) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        resp.set_container_id(container_id);
    }).unwrap_or_default()
}

fn build_container_status_response(request_id: u64, status_resp: &ContainerStatusResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut container_status = resp.init_container_status();
        let mut status = container_status.reborrow().init_status();
        status.set_id(&status_resp.status.id);
        let mut meta = status.reborrow().init_metadata();
        meta.set_name(&status_resp.status.metadata.name);
        meta.set_attempt(status_resp.status.metadata.attempt);
    }).unwrap_or_default()
}

fn build_containers_response(request_id: u64, containers: &[Container]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_containers(containers.len() as u32);
        for (i, container) in containers.iter().enumerate() {
            let mut c = list.reborrow().get(i as u32);
            c.set_id(&container.id);
            c.set_pod_sandbox_id(&container.pod_sandbox_id);
            let mut meta = c.reborrow().init_metadata();
            meta.set_name(&container.metadata.name);
            meta.set_attempt(container.metadata.attempt);
            c.set_created_at(container.created_at.timestamp());
        }
    }).unwrap_or_default()
}

fn build_exec_result_response(request_id: u64, exec_resp: &ExecSyncResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut exec = resp.init_exec_result();
        exec.set_stdout(&exec_resp.stdout);
        exec.set_stderr(&exec_resp.stderr);
        exec.set_exit_code(exec_resp.exit_code);
    }).unwrap_or_default()
}

fn build_sandbox_stats_response(request_id: u64, stats: &PodSandboxStats) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut sandbox_stats = resp.init_sandbox_stats();
        let mut attrs = sandbox_stats.reborrow().init_attributes();
        attrs.set_id(&stats.attributes.id);
    }).unwrap_or_default()
}

fn build_sandbox_stats_list_response(request_id: u64, stats_list: &[PodSandboxStats]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_sandbox_stats_list(stats_list.len() as u32);
        for (i, stats) in stats_list.iter().enumerate() {
            let mut s = list.reborrow().get(i as u32);
            let mut attrs = s.reborrow().init_attributes();
            attrs.set_id(&stats.attributes.id);
        }
    }).unwrap_or_default()
}

fn build_container_stats_response(request_id: u64, stats: &ContainerStats) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut container_stats = resp.init_container_stats_result();
        let mut attrs = container_stats.reborrow().init_attributes();
        attrs.set_id(&stats.attributes.id);
    }).unwrap_or_default()
}

fn build_container_stats_list_response(request_id: u64, stats_list: &[ContainerStats]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::runtime_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_container_stats_list(stats_list.len() as u32);
        for (i, stats) in stats_list.iter().enumerate() {
            let mut s = list.reborrow().get(i as u32);
            let mut attrs = s.reborrow().init_attributes();
            attrs.set_id(&stats.attributes.id);
        }
    }).unwrap_or_default()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Image Response Builders
// ═══════════════════════════════════════════════════════════════════════════════

fn build_image_success_response(request_id: u64) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        resp.set_success(());
    }).unwrap_or_default()
}

fn build_image_error_response(request_id: u64, message: &str) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        let mut error = resp.init_error();
        error.set_message(message);
        error.set_code("INTERNAL");
        error.set_details("");
    }).unwrap_or_default()
}

fn build_image_ref_response(request_id: u64, image_ref: &str) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        resp.set_image_ref(image_ref);
    }).unwrap_or_default()
}

fn build_images_response(request_id: u64, images: &[crate::image::Image]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_images(images.len() as u32);
        for (i, img) in images.iter().enumerate() {
            let mut image = list.reborrow().get(i as u32);
            image.set_id(&img.id);
            image.set_size(img.size);
            let mut tags = image.reborrow().init_repo_tags(img.repo_tags.len() as u32);
            for (j, tag) in img.repo_tags.iter().enumerate() {
                tags.set(j as u32, tag);
            }
            let mut digests = image.reborrow().init_repo_digests(img.repo_digests.len() as u32);
            for (j, digest) in img.repo_digests.iter().enumerate() {
                digests.set(j as u32, digest);
            }
        }
    }).unwrap_or_default()
}

fn build_image_status_response(request_id: u64, status: &crate::image::ImageStatusResponse) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        let image_status = resp.init_image_status();
        if let Some(ref img) = status.image {
            let mut image = image_status.init_image();
            image.set_id(&img.id);
            image.set_size(img.size);
            let mut tags = image.reborrow().init_repo_tags(img.repo_tags.len() as u32);
            for (j, tag) in img.repo_tags.iter().enumerate() {
                tags.set(j as u32, tag);
            }
        }
    }).unwrap_or_default()
}

fn build_fs_info_response(request_id: u64, usage: &[crate::image::FilesystemUsage]) -> Vec<u8> {
    serialize_message(|msg| {
        let mut resp = msg.init_root::<workers_capnp::image_response::Builder>();
        resp.set_request_id(request_id);
        let mut list = resp.init_fs_info(usage.len() as u32);
        for (i, fs) in usage.iter().enumerate() {
            let mut fs_usage = list.reborrow().get(i as u32);
            fs_usage.set_timestamp(fs.timestamp);
            let mut fs_id = fs_usage.reborrow().init_fs_id();
            fs_id.set_mountpoint(&fs.fs_id.mountpoint);
            fs_usage.set_used_bytes(fs.used_bytes);
            fs_usage.set_inodes_used(fs.inodes_used);
        }
    }).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::PodSandboxState;
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
        let status = service.pod_sandbox_status(&sandbox_id, false).await?;
        assert_eq!(status.status.state, PodSandboxState::SandboxReady);

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
        let status = service.container_status(&container_id, false).await?;
        assert_eq!(status.status.state, ContainerState::ContainerCreated);

        // Start container
        service.start_container(&container_id).await?;
        let status = service.container_status(&container_id, false).await?;
        assert_eq!(status.status.state, ContainerState::ContainerRunning);

        // Stop container
        service.stop_container(&container_id, 30).await?;
        let status = service.container_status(&container_id, false).await?;
        assert_eq!(status.status.state, ContainerState::ContainerExited);

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
