//! Unified service spawner for ZmqService hosting.
//!
//! Provides a single API for spawning ZmqService implementations with different
//! execution modes (Tokio task, dedicated thread, or subprocess).

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use tokio::sync::Notify;

use super::{ProcessConfig, ProcessSpawner, SpawnedProcess};
use crate::envelope::InMemoryNonceCache;
use crate::error::Result;
use crate::prelude::VerifyingKey;
use crate::registry::{ServiceRegistration, SocketKind};
use crate::service::{ServiceRunner, ZmqService};
use crate::transport::TransportConfig;

// ============================================================================
// Spawnable Trait
// ============================================================================

/// Trait for services that can be spawned by ServiceSpawner.
///
/// Implemented by both REQ/REP handlers and XSUB/XPUB proxies.
/// This provides a unified spawning API regardless of service type.
pub trait Spawnable: Send + 'static {
    /// Service name (for logging and registry).
    fn name(&self) -> &str;

    /// ZMQ context.
    fn context(&self) -> &Arc<zmq::Context>;

    /// Endpoints to register with EndpointRegistry.
    ///
    /// Each tuple is (SocketKind, TransportConfig).
    /// - Handlers typically register one REP endpoint
    /// - Proxies register PUB and SUB endpoints (note socket type inversion)
    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)>;

    /// Run the service on current thread (blocking).
    ///
    /// Called by spawner after thread/process setup.
    /// Should block until shutdown is signaled.
    fn run_blocking(self: Box<Self>, shutdown: Arc<Notify>) -> Result<()>;
}

// ============================================================================
// HandlerService - Wrapper for ZmqService
// ============================================================================

/// Wrapper to make ZmqService spawnable via the Spawnable trait.
///
/// This adapts the existing ZmqService pattern (REQ/REP) to work with
/// the unified ServiceSpawner API.
pub struct HandlerService<S: ZmqService> {
    /// The underlying ZmqService implementation.
    service: S,
    /// Service name (for logging and registry).
    service_name: String,
    /// Transport configuration (endpoint).
    transport: TransportConfig,
    /// ZMQ context.
    context: Arc<zmq::Context>,
    /// Server's Ed25519 verifying key for signature verification.
    verifying_key: VerifyingKey,
    /// Nonce cache for replay protection.
    nonce_cache: Arc<InMemoryNonceCache>,
}

impl<S: ZmqService> HandlerService<S> {
    /// Create a new handler service wrapper.
    pub fn new(
        service: S,
        transport: TransportConfig,
        context: Arc<zmq::Context>,
        verifying_key: VerifyingKey,
    ) -> Self {
        let service_name = service.name().to_string();
        Self {
            service,
            service_name,
            transport,
            context,
            verifying_key,
            nonce_cache: Arc::new(InMemoryNonceCache::new()),
        }
    }

    /// Use a shared nonce cache.
    pub fn with_nonce_cache(mut self, cache: Arc<InMemoryNonceCache>) -> Self {
        self.nonce_cache = cache;
        self
    }
}

impl<S: ZmqService> Spawnable for HandlerService<S> {
    fn name(&self) -> &str {
        &self.service_name
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, self.transport.clone())]
    }

    fn run_blocking(self: Box<Self>, shutdown: Arc<Notify>) -> Result<()> {
        // Create a single-threaded runtime for blocking execution
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("runtime: {}", e)))?;

        rt.block_on(async move {
            // Pass TransportConfig directly for SystemdFd support
            let runner = ServiceRunner::with_nonce_cache(
                self.transport.clone(),
                self.context.clone(),
                self.verifying_key.clone(),
                self.nonce_cache.clone(),
            );

            match runner.run(self.service).await {
                Ok(mut handle) => {
                    // Wait for shutdown signal
                    shutdown.notified().await;
                    handle.stop().await;
                    Ok(())
                }
                Err(e) => Err(crate::error::RpcError::SpawnFailed(e.to_string())),
            }
        })
    }
}

// ============================================================================
// ProxyService - XSUB/XPUB Proxy
// ============================================================================

/// XSUB/XPUB proxy service for event forwarding.
///
/// This implements the ZMQ XSUB/XPUB proxy pattern:
/// - XSUB socket binds and receives from publishers (PUB sockets connect)
/// - XPUB socket binds and sends to subscribers (SUB sockets connect)
///
/// Note the socket type inversion in registry:
/// - XSUB binds → clients use PUB → register as SocketKind::Pub
/// - XPUB binds → clients use SUB → register as SocketKind::Sub
pub struct ProxyService {
    /// Service name (for logging and registry).
    name: String,
    /// ZMQ context.
    context: Arc<zmq::Context>,
    /// Transport for XSUB socket (publishers connect here).
    pub_transport: TransportConfig,
    /// Transport for XPUB socket (subscribers connect here).
    sub_transport: TransportConfig,
}

impl ProxyService {
    /// Create a new proxy service.
    pub fn new(
        name: impl Into<String>,
        context: Arc<zmq::Context>,
        pub_transport: TransportConfig,
        sub_transport: TransportConfig,
    ) -> Self {
        Self {
            name: name.into(),
            context,
            pub_transport,
            sub_transport,
        }
    }
}

impl Spawnable for ProxyService {
    fn name(&self) -> &str {
        &self.name
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        // Note: Socket type inversion for clients
        vec![
            (SocketKind::Pub, self.pub_transport.clone()),  // XSUB → clients use PUB
            (SocketKind::Sub, self.sub_transport.clone()),  // XPUB → clients use SUB
        ]
    }

    fn run_blocking(self: Box<Self>, shutdown: Arc<Notify>) -> Result<()> {
        // Create XSUB socket (receives from publishers)
        let mut xsub = self.context.socket(zmq::XSUB)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XSUB socket: {}", e)))?;

        // Create XPUB socket (sends to subscribers)
        let mut xpub = self.context.socket(zmq::XPUB)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XPUB socket: {}", e)))?;

        // Create CTRL socket for shutdown (PAIR pattern)
        let mut ctrl = self.context.socket(zmq::PAIR)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL socket: {}", e)))?;
        let ctrl_endpoint = format!("inproc://proxy-ctrl-{}", self.name);
        ctrl.bind(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL bind: {}", e)))?;

        // Bind XSUB (publishers connect here)
        // Uses TransportConfig::bind() for proper systemd FD support
        self.pub_transport.bind(&mut xsub)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XSUB bind: {}", e)))?;
        let pub_endpoint = self.pub_transport.zmq_endpoint();

        // Bind XPUB (subscribers connect here)
        // Uses TransportConfig::bind() for proper systemd FD support
        self.sub_transport.bind(&mut xpub)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XPUB bind: {}", e)))?;
        let sub_endpoint = self.sub_transport.zmq_endpoint();

        tracing::info!(
            "Proxy {} started: XSUB={}, XPUB={}",
            self.name,
            pub_endpoint,
            sub_endpoint
        );

        // Spawn shutdown listener
        let ctrl_sender = self.context.socket(zmq::PAIR)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL sender: {}", e)))?;
        ctrl_sender.connect(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL connect: {}", e)))?;

        let name_clone = self.name.clone();
        std::thread::spawn(move || {
            // Block until shutdown is signaled
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("shutdown listener runtime");
            rt.block_on(shutdown.notified());

            // Send termination message to proxy
            tracing::debug!("Sending TERMINATE to proxy {}", name_clone);
            let _ = ctrl_sender.send("TERMINATE", 0);
        });

        // Run steerable proxy (blocks until TERMINATE received)
        zmq::proxy_steerable(&mut xsub, &mut xpub, &mut ctrl)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("proxy: {}", e)))?;

        tracing::info!("Proxy {} stopped", self.name);
        Ok(())
    }
}

/// Mode for spawning services.
#[derive(Debug, Clone)]
pub enum ServiceMode {
    /// Spawn as a tokio task in the current runtime.
    Tokio,

    /// Spawn on a dedicated thread with its own tokio runtime.
    /// Useful for services with !Send types (like tch-rs tensors).
    Thread,

    /// Spawn as a subprocess.
    Subprocess {
        /// Path to the binary.
        binary: PathBuf,
    },
}

/// Unified service spawner.
///
/// Spawns `Spawnable` services with consistent lifecycle management,
/// regardless of execution mode.
///
/// # Example
///
/// ```ignore
/// use hyprstream_rpc::service::spawner::{ServiceSpawner, HandlerService, ProxyService};
/// use hyprstream_rpc::transport::TransportConfig;
///
/// // Spawn a REQ/REP handler
/// let transport = TransportConfig::inproc("my-service");
/// let handler = HandlerService::new(MyZmqService::new(), transport, ctx, verifying_key);
/// let spawner = ServiceSpawner::tokio();
/// let service = spawner.spawn(handler).await?;
///
/// // Spawn an XSUB/XPUB proxy on dedicated thread
/// let proxy = ProxyService::new("events", ctx, pub_transport, sub_transport);
/// let spawner = ServiceSpawner::threaded();
/// let service = spawner.spawn(proxy).await?;
///
/// // Stop the service
/// service.stop().await?;
/// ```
pub struct ServiceSpawner {
    mode: ServiceMode,
    process_spawner: Option<ProcessSpawner>,
}

impl ServiceSpawner {
    /// Create a spawner that runs services as tokio tasks.
    pub fn tokio() -> Self {
        Self {
            mode: ServiceMode::Tokio,
            process_spawner: None,
        }
    }

    /// Create a spawner that runs services on dedicated threads.
    ///
    /// Each service gets its own thread with a single-threaded tokio runtime.
    /// Use this for services with !Send types (like tch-rs tensors).
    pub fn threaded() -> Self {
        Self {
            mode: ServiceMode::Thread,
            process_spawner: None,
        }
    }

    /// Create a spawner that runs services as subprocesses.
    ///
    /// The binary should be a `hyprstream service <name>` command.
    pub fn subprocess(binary: PathBuf) -> Self {
        Self {
            mode: ServiceMode::Subprocess { binary },
            process_spawner: Some(ProcessSpawner::new()),
        }
    }

    /// Create a subprocess spawner with a custom process spawner.
    pub fn subprocess_with(binary: PathBuf, process_spawner: ProcessSpawner) -> Self {
        Self {
            mode: ServiceMode::Subprocess { binary },
            process_spawner: Some(process_spawner),
        }
    }

    /// Get the spawning mode.
    pub fn mode(&self) -> &ServiceMode {
        &self.mode
    }

    /// Spawn any Spawnable service with registry integration.
    ///
    /// This is the unified spawning API that works with both:
    /// - `HandlerService<S: ZmqService>` - REQ/REP handlers
    /// - `ProxyService` - XSUB/XPUB proxies
    ///
    /// The service is automatically registered with the EndpointRegistry and
    /// unregistered when the returned handle is dropped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Spawn a handler
    /// let handler = HandlerService::new(my_zmq_service, transport, ctx, verifying_key);
    /// let service = spawner.spawn(handler).await?;
    ///
    /// // Spawn a proxy
    /// let proxy = ProxyService::new("events", ctx, pub_transport, sub_transport);
    /// let service = spawner.spawn(proxy).await?;
    /// ```
    pub async fn spawn<S: Spawnable>(&self, service: S) -> Result<SpawnedService> {
        // 1. Register with EndpointRegistry (if initialized)
        let registrations = service.registrations();
        let registration = if !registrations.is_empty() {
            ServiceRegistration::multi(service.name(), registrations, None).ok()
        } else {
            None
        };

        // 2. Spawn based on mode
        match &self.mode {
            ServiceMode::Tokio => self.spawn_tokio(service, registration).await,
            ServiceMode::Thread => self.spawn_thread(service, registration).await,
            ServiceMode::Subprocess { binary } => {
                self.spawn_subprocess(service, binary.clone(), registration).await
            }
        }
    }

    /// Spawn a Spawnable as a tokio task.
    async fn spawn_tokio<S: Spawnable>(
        &self,
        service: S,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_string();
        let name_for_spawn = name.clone();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();

        // Spawn task that runs the service
        let join_handle = tokio::spawn(async move {
            if let Err(e) = Box::new(service).run_blocking(shutdown_clone) {
                tracing::error!("Service {} failed: {}", name_for_spawn, e);
            }
        });

        // Create a handle wrapper
        let handle = crate::service::ServiceHandle::from_task(join_handle, shutdown);

        Ok(SpawnedService {
            id: format!("{}-tokio", name),
            kind: ServiceKind::TokioTask {
                handle: Some(handle),
            },
            _registration: registration,
        })
    }

    /// Spawn a Spawnable on a dedicated thread.
    async fn spawn_thread<S: Spawnable>(
        &self,
        service: S,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_string();
        let name_for_thread = name.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();

        // Spawn thread
        let thread_handle = thread::Builder::new()
            .name(format!("{}-service", &name))
            .spawn(move || {
                // Signal ready before blocking
                let _ = ready_tx.send(Ok(()));

                // Run the service (blocking)
                if let Err(e) = Box::new(service).run_blocking(shutdown_clone) {
                    tracing::error!("Service {} failed: {}", name_for_thread, e);
                }
            })
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("thread spawn: {}", e)))?;

        // Wait for ready signal
        match ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(crate::error::RpcError::SpawnFailed(e));
            }
            Err(_) => {
                return Err(crate::error::RpcError::SpawnFailed(
                    "service thread exited before ready".to_string(),
                ));
            }
        }

        Ok(SpawnedService {
            id: format!("{}-thread", name),
            kind: ServiceKind::Thread {
                handle: Some(thread_handle),
                shutdown,
            },
            _registration: registration,
        })
    }

    /// Spawn a Spawnable as a subprocess with PID file tracking.
    async fn spawn_subprocess<S: Spawnable>(
        &self,
        service: S,
        binary: PathBuf,
        registration: Option<ServiceRegistration>,
    ) -> Result<SpawnedService> {
        let name = service.name().to_string();
        let spawner = self
            .process_spawner
            .as_ref()
            .expect("subprocess mode requires process_spawner");

        let process_config =
            ProcessConfig::new(&name, binary).args(["service", &name]);

        let process = spawner.spawn(process_config).await?;

        // Write PID file for lifecycle management
        let pid_file = crate::paths::service_pid_file(&name);
        if let Some(pid) = process.pid() {
            // Ensure runtime directory exists
            if let Some(parent) = pid_file.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&pid_file, pid.to_string()) {
                tracing::warn!("Failed to write PID file {:?}: {}", pid_file, e);
            } else {
                tracing::debug!("Wrote PID {} to {:?}", pid, pid_file);
            }
        }

        Ok(SpawnedService {
            id: process.id.clone(),
            kind: ServiceKind::Subprocess {
                process,
                pid_file,
            },
            _registration: registration,
        })
    }

}

impl Default for ServiceSpawner {
    fn default() -> Self {
        Self::tokio()
    }
}

/// Kind of spawned service (determines cleanup behavior).
pub enum ServiceKind {
    /// Running as a tokio task.
    TokioTask {
        handle: Option<crate::service::ServiceHandle>,
    },

    /// Running on a dedicated thread.
    Thread {
        handle: Option<JoinHandle<()>>,
        shutdown: Arc<Notify>,
    },

    /// Running as a subprocess with PID file tracking.
    Subprocess {
        process: SpawnedProcess,
        /// PID file path (XDG-compliant) for lifecycle management.
        pid_file: PathBuf,
    },
}

/// Handle for a spawned service.
pub struct SpawnedService {
    /// Unique identifier.
    id: String,

    /// Service kind (determines cleanup behavior).
    kind: ServiceKind,

    /// Registry registration (RAII cleanup on drop).
    /// Stored as Option to allow Drop to consume it.
    _registration: Option<ServiceRegistration>,
}

impl SpawnedService {
    /// Get the service ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if the service is running.
    pub fn is_running(&self) -> bool {
        match &self.kind {
            ServiceKind::TokioTask { handle } => {
                handle.as_ref().map(|h| h.is_running()).unwrap_or(false)
            }
            ServiceKind::Thread { handle, .. } => {
                handle.as_ref().map(|h| !h.is_finished()).unwrap_or(false)
            }
            ServiceKind::Subprocess { pid_file, .. } => {
                // Check if PID file exists and process is alive (signal 0)
                if let Ok(pid_str) = std::fs::read_to_string(pid_file) {
                    if let Ok(pid) = pid_str.trim().parse::<i32>() {
                        // Signal 0 checks if process exists without sending a signal
                        return nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(pid),
                            None,
                        )
                        .is_ok();
                    }
                }
                false
            }
        }
    }

    /// Stop the service.
    ///
    /// Idempotent: subsequent calls are no-ops if already stopped.
    pub async fn stop(&mut self) -> Result<()> {
        match &mut self.kind {
            ServiceKind::TokioTask { handle } => {
                if let Some(mut h) = handle.take() {
                    h.stop().await;
                }
            }
            ServiceKind::Thread { handle, shutdown } => {
                // Signal shutdown
                shutdown.notify_one();

                // Wait for thread to finish
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
            ServiceKind::Subprocess { process, pid_file } => {
                // Read PID from file and send SIGTERM
                if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = pid_str.trim().parse::<i32>() {
                        tracing::info!("Sending SIGTERM to subprocess {} (PID {})", process.id, pid);
                        if let Err(e) = nix::sys::signal::kill(
                            nix::unistd::Pid::from_raw(pid),
                            nix::sys::signal::Signal::SIGTERM,
                        ) {
                            tracing::warn!("Failed to send SIGTERM to PID {}: {}", pid, e);
                        }
                    }
                }
                // Clean up PID file
                if let Err(e) = std::fs::remove_file(&pid_file) {
                    tracing::debug!("Failed to remove PID file {:?}: {}", pid_file, e);
                }
            }
        }

        tracing::info!("Service {} stopped", self.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::generate_signing_keypair;
    use crate::service::ZmqService;
    use anyhow::Result as AnyhowResult;

    struct EchoService;

    impl ZmqService for EchoService {
        fn handle_request(
            &self,
            _ctx: &crate::service::EnvelopeContext,
            payload: &[u8],
        ) -> AnyhowResult<Vec<u8>> {
            Ok(payload.to_vec())
        }

        fn name(&self) -> &str {
            "echo"
        }
    }

    #[tokio::test]
    async fn test_tokio_spawner() {
        let context = Arc::new(zmq::Context::new());
        let (_, verifying_key) = generate_signing_keypair();

        let transport = TransportConfig::inproc("test-spawner-tokio");
        let handler = HandlerService::new(EchoService, transport, context, verifying_key);

        let spawner = ServiceSpawner::tokio();
        let mut service = spawner.spawn(handler).await.unwrap();

        assert!(service.is_running());

        service.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_thread_spawner() {
        let context = Arc::new(zmq::Context::new());
        let (_, verifying_key) = generate_signing_keypair();

        let transport = TransportConfig::inproc("test-spawner-thread");
        let handler = HandlerService::new(EchoService, transport, context, verifying_key);

        let spawner = ServiceSpawner::threaded();
        let mut service = spawner.spawn(handler).await.unwrap();

        assert!(service.is_running());

        service.stop().await.unwrap();
    }
}
