//! The `Spawnable` trait for services that can be spawned.
//!
//! This trait lives in `hyprstream-rpc` because types in this crate
//! (StreamService, QuicServiceLoop) implement it directly. The spawner
//! infrastructure that consumes Spawnable lives in `hyprstream-service`.

use std::sync::Arc;
use tokio::sync::Notify;

use crate::error::{Result, RpcError};
use crate::registry::SocketKind;
use crate::service::{RequestLoop, ZmqService};
use crate::transport::TransportConfig;

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
// Blanket Spawnable impl for ZmqService
// ============================================================================

/// Every `ZmqService + Send + Sync` is automatically `Spawnable`.
///
/// This blanket implementation eliminates the need for wrapper types.
/// Services that implement `ZmqService` include their infrastructure
/// (context, transport, verifying_key) and can be spawned directly.
///
/// Services not satisfying `Send + Sync` (e.g., those using `Rc` or `!Send` fields)
/// must implement `Spawnable` directly (as `InferenceServiceConfig` does).
impl<S: ZmqService + Send + Sync> Spawnable for S {
    fn name(&self) -> &str {
        ZmqService::name(self)
    }

    fn context(&self) -> &Arc<zmq::Context> {
        ZmqService::context(self)
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, ZmqService::transport(self).clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<()> {
        let transport = ZmqService::transport(&*self).clone();
        let context = Arc::clone(ZmqService::context(&*self));
        let signing_key = ZmqService::signing_key(&*self);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| RpcError::SpawnFailed(format!("runtime: {e}")))?;

        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            let runner = RequestLoop::new(transport, context, signing_key);

            match runner.run(*self).await {
                Ok(mut handle) => {
                    if let Some(tx) = on_ready {
                        let _ = tx.send(());
                    }
                    let _ = crate::notify::ready();
                    shutdown.notified().await;
                    handle.stop().await;
                    Ok(())
                }
                Err(e) => Err(RpcError::SpawnFailed(e.to_string())),
            }
        })
    }
}
