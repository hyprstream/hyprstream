//! Bridged serve helper for the post-ZMQ spawn path (#136).
//!
//! The RPC cutover replaces every `RequestLoop::new(..).run(service)` (ZMQ
//! ROUTER) with this: bridge the service to a `Send` processor
//! ([`LocalServiceBridge`](crate::transport::iroh_rpc::LocalServiceBridge)), then
//! serve it over its *registered* transport via the same `process_request`
//! dispatch core the quinn/iroh planes use.
//!
//! Spawn-path transports (from `EndpointRegistry::endpoint(name, Rep)`):
//! - `Inproc{endpoint}` (default daemon mode) → register the processor in the
//!   in-memory dial registry ([`register_inproc`](crate::dial::register_inproc));
//!   **no socket**. The registry holds only a `Weak`, so the strong `processor`
//!   `Arc` is retained here for the service's lifetime.
//! - `Ipc{path}` (`--ipc`) → bind a `UnixListener` at `path`, run
//!   [`UdsRpcServer`].
//! - `SystemdFd{fd,..}` → adopt the systemd-passed listener fd, run [`UdsRpcServer`].
//! - `Quic`/`Iroh` are *dialed*, never bound on the spawn path → error.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use ed25519_dalek::SigningKey;
use tokio::sync::Notify;

use crate::transport::rpc_session::{IrohRequestProcessor, DEFAULT_STREAM_LIMIT};
use crate::transport::uds_server::UdsRpcServer;
use crate::transport::{EndpointType, TransportConfig};

/// Signal startup readiness: fire the `on_ready` oneshot (spawner waits on it)
/// and notify systemd (`Type=notify`).
fn signal_ready(on_ready: Option<tokio::sync::oneshot::Sender<()>>) {
    if let Some(tx) = on_ready {
        let _ = tx.send(());
    }
    let _ = crate::notify::ready();
}

/// Serve a bridged request `processor` over its registered `transport` until
/// `shutdown` fires. See the module docs for the per-transport behaviour.
///
/// `processor` MUST be the bridge wrapping the spawn-path service (built via
/// [`LocalServiceBridge::spawn`](crate::transport::iroh_rpc::LocalServiceBridge::spawn)
/// or `spawn_with`). This fn holds it for the serve lifetime.
pub async fn serve_bridged(
    transport: &TransportConfig,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    shutdown: Arc<Notify>,
    on_ready: Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<()> {
    serve_bridged_with_shutdown_armed(transport, processor, signing_key, shutdown, on_ready, None)
        .await
}

/// Variant of [`serve_bridged`] that acknowledges after the shutdown waiter is
/// registered, before startup readiness is signalled. This closes the window
/// where an owner can notify shutdown after readiness but before the helper has
/// begun waiting on the shared [`Notify`].
pub async fn serve_bridged_with_shutdown_armed(
    transport: &TransportConfig,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    shutdown: Arc<Notify>,
    on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    shutdown_armed: Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<()> {
    match &transport.endpoint {
        EndpointType::Inproc { endpoint } => {
            crate::dial::register_inproc(endpoint.clone(), &processor);
            let notified = shutdown.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(tx) = shutdown_armed {
                let _ = tx.send(());
            }
            signal_ready(on_ready);
            notified.await;
            processor.close_admission();
            crate::dial::unregister_inproc(endpoint);
            // The service owner drains and joins, even with retained clients.
            drop(processor);
            Ok(())
        }
        EndpointType::Ipc { path } => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| anyhow!("create uds dir {}: {e}", parent.display()))?;
            }
            // Clear a stale socket left by a prior unclean shutdown.
            let _ = std::fs::remove_file(path);
            let listener = tokio::net::UnixListener::bind(path)
                .map_err(|e| anyhow!("bind uds {}: {e}", path.display()))?;
            // Restrict to owner-only access (#207): world-accessible sockets would
            // let any local process connect, undermining same-host trust isolation.
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .map_err(|e| anyhow!("set uds perms {}: {e}", path.display()))?;
            }
            run_uds(
                listener,
                processor,
                signing_key,
                shutdown,
                on_ready,
                shutdown_armed,
            )
            .await
        }
        EndpointType::SystemdFd { fd, .. } => {
            use std::os::unix::io::FromRawFd;
            // The fd is the systemd-passed, already-bound server listener.
            let std_listener = unsafe { std::os::unix::net::UnixListener::from_raw_fd(*fd) };
            std_listener
                .set_nonblocking(true)
                .map_err(|e| anyhow!("systemd fd set_nonblocking: {e}"))?;
            let listener = tokio::net::UnixListener::from_std(std_listener)
                .map_err(|e| anyhow!("adopt systemd uds fd: {e}"))?;
            run_uds(
                listener,
                processor,
                signing_key,
                shutdown,
                on_ready,
                shutdown_armed,
            )
            .await
        }
        other => bail!(
            "serve_bridged: {other:?} is not a spawn-path RPC endpoint \
             (Quic/Iroh are dialed via dial(), not bound here)"
        ),
    }
}

/// Run a [`UdsRpcServer`] until it ends or `shutdown` fires (graceful drain).
async fn run_uds(
    listener: tokio::net::UnixListener,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    shutdown: Arc<Notify>,
    on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    shutdown_armed: Option<tokio::sync::oneshot::Sender<()>>,
) -> Result<()> {
    let server = UdsRpcServer::with_capacity(
        listener,
        Arc::clone(&processor),
        signing_key,
        DEFAULT_STREAM_LIMIT,
    );
    let token = server.shutdown_token();
    let limit = server.stream_limit();
    let cap = server.capacity();
    let notified = shutdown.notified();
    tokio::pin!(notified);
    notified.as_mut().enable();
    if let Some(tx) = shutdown_armed {
        let _ = tx.send(());
    }
    signal_ready(on_ready);
    tokio::select! {
        r = server.run() => r,
        _ = notified => {
            let deadline = processor.begin_shutdown(
                tokio::time::Instant::now() + crate::transport::rpc_session::DRAIN_TIMEOUT,
            );
            UdsRpcServer::shutdown_until(&limit, cap, &token, deadline).await;
            Ok(())
        }
    }
}
