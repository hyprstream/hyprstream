//! The `Spawnable` trait for services that can be spawned.
//!
//! This trait lives in `hyprstream-rpc` because types in this crate
//! (StreamService, QuicServiceLoop) implement it directly. The spawner
//! infrastructure that consumes Spawnable lives in `hyprstream-service`.

use std::sync::Arc;
use tokio::sync::Notify;

use crate::error::{Result, RpcError};
use crate::registry::SocketKind;
use crate::service::RequestService;
use crate::transport::TransportConfig;

/// Trait for services that can be spawned by ServiceSpawner.
///
/// Implemented by both REQ/REP handlers and XSUB/XPUB proxies.
/// This provides a unified spawning API regardless of service type.
pub trait Spawnable: Send + 'static {
    /// Service name (for logging and registry).
    fn name(&self) -> &str;

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
    ///
    /// # Arguments
    /// * `shutdown` - Notification to signal service shutdown
    /// * `on_ready` - Optional oneshot sender to signal when socket is bound and ready
    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()>;
}

// ============================================================================
// Blanket Spawnable impl for RequestService
// ============================================================================

/// Every `RequestService + Send + Sync` is automatically `Spawnable`.
///
/// This blanket implementation eliminates the need for wrapper types.
/// Services that implement `RequestService` include their infrastructure
/// (context, transport, verifying_key) and can be spawned directly.
///
/// Services not satisfying `Send + Sync` (e.g., those using `Rc` or `!Send` fields)
/// must implement `Spawnable` directly (as `InferenceServiceConfig` does).
impl<S: RequestService + Send + Sync> Spawnable for S {
    fn name(&self) -> &str {
        RequestService::name(self)
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, RequestService::transport(self).clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()> {
        let transport = RequestService::transport(&*self).clone();
        let signing_key = RequestService::signing_key(&*self);

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| RpcError::SpawnFailed(format!("runtime: {e}")))?;

        // Post-ZMQ serve (#136): bridge the service to a Send processor on its own
        // LocalSet thread, then serve it over its registered transport (inproc →
        // in-memory dial registry; ipc/systemd → UdsRpcServer). No ZMQ ROUTER.
        rt.block_on(async move {
            let nonce_cache = Arc::new(crate::envelope::InMemoryNonceCache::new());
            let bridge = crate::transport::iroh_rpc::LocalServiceBridge::spawn(*self, nonce_cache, 0)
                .map_err(|e| RpcError::SpawnFailed(format!("bridge: {e}")))?;
            let processor: Arc<dyn crate::transport::rpc_session::IrohRequestProcessor> =
                Arc::new(bridge);
            crate::service::serve::serve_bridged(&transport, processor, signing_key, shutdown, on_ready)
                .await
                .map_err(|e| RpcError::SpawnFailed(e.to_string()))
        })
    }
}
