//! UDS RPC server — the same-host `ipc` serve path, post-ZMQ.
//!
//! Mirrors [`QuinnRpcServer`](super::quinn_transport::QuinnRpcServer): a
//! `UnixListener` accept loop that hands each accepted connection to the
//! transport-generic [`serve_rpc_connection`] over a
//! [`UdsSession`](super::uds_session::UdsSession). Same DoS discipline — a
//! server-wide concurrent-stream semaphore (drained on shutdown), a
//! per-connection cap, a bounded handshake, and the per-stream read timeout —
//! and the same `process_request` dispatch core, so the ipc plane is identical
//! to the quinn/iroh planes save for the socket underneath.
//!
//! The daemon binds the `UnixListener` (choosing the path + file mode; peer
//! creds are #207) and runs one of these per spawn-path service. A
//! `!Send` service is bridged to the `Send`-bound processor via
//! [`LocalServiceBridge`](super::iroh_rpc::LocalServiceBridge), exactly as the
//! quinn/iroh serve paths do.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use ed25519_dalek::SigningKey;
use tokio::net::UnixListener;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use super::rpc_session::{
    serve_rpc_connection, IrohRequestProcessor, DEFAULT_CONNECTION_LIMIT, DEFAULT_STREAM_LIMIT,
    DRAIN_TIMEOUT, REQUEST_READ_TIMEOUT,
};
use super::uds_session::{accept_uds, PLANE_MOQ, PLANE_RPC};

/// A `UnixListener`-backed RPC server for the same-host `ipc` plane.
pub struct UdsRpcServer {
    listener: UnixListener,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    /// Server-wide concurrent-stream cap, shared across all connections; one
    /// permit per in-flight bidi stream. `shutdown` drains it via `acquire_many`.
    stream_limit: Arc<Semaphore>,
    stream_limit_capacity: u32,
    /// Cap on concurrent accepted connections (#162). Connections beyond the cap
    /// are rejected (dropped), not queued.
    connection_limit: Arc<Semaphore>,
    /// Per-stream request-read timeout (#159 slowloris bound).
    read_timeout: Duration,
    shutdown: CancellationToken,
}

impl UdsRpcServer {
    /// Build a server over an already-bound [`UnixListener`].
    pub fn new<P: IrohRequestProcessor>(
        listener: UnixListener,
        processor: P,
        signing_key: SigningKey,
    ) -> Self {
        Self::with_capacity(listener, Arc::new(processor), signing_key, DEFAULT_STREAM_LIMIT)
    }

    /// Build a server with an explicit server-wide concurrent-stream cap.
    pub fn with_capacity(
        listener: UnixListener,
        processor: Arc<dyn IrohRequestProcessor>,
        signing_key: SigningKey,
        stream_limit: usize,
    ) -> Self {
        Self {
            listener,
            processor,
            signing_key,
            stream_limit: Arc::new(Semaphore::new(stream_limit)),
            stream_limit_capacity: u32::try_from(stream_limit).unwrap_or(u32::MAX),
            connection_limit: Arc::new(Semaphore::new(DEFAULT_CONNECTION_LIMIT)),
            read_timeout: REQUEST_READ_TIMEOUT,
            shutdown: CancellationToken::new(),
        }
    }

    /// Override the concurrent-connection cap (builder style).
    pub fn with_connection_limit(mut self, limit: usize) -> Self {
        self.connection_limit = Arc::new(Semaphore::new(limit));
        self
    }

    /// Override the server-wide concurrent-stream cap (builder style).
    pub fn with_stream_limit(mut self, limit: usize) -> Self {
        self.stream_limit = Arc::new(Semaphore::new(limit));
        self.stream_limit_capacity = u32::try_from(limit).unwrap_or(u32::MAX);
        self
    }

    /// Override the per-stream request-read timeout (#159). Primarily for tests.
    pub fn with_read_timeout(mut self, read_timeout: Duration) -> Self {
        self.read_timeout = read_timeout;
        self
    }

    /// Apply all tunables from an [`super::rpc_session::RpcConfig`] in one call (#197).
    pub fn with_rpc_config(self, cfg: &super::rpc_session::RpcConfig) -> Self {
        self.with_stream_limit(cfg.stream_limit)
            .with_connection_limit(cfg.connection_limit)
            .with_read_timeout(cfg.request_read_timeout)
    }

    /// A handle that, when cancelled, stops the accept loop and per-connection
    /// serve loops. Does not by itself drain in-flight streams — use
    /// [`UdsRpcServer::shutdown`] for a graceful drain.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// The shared concurrent-stream semaphore (pair with [`UdsRpcServer::capacity`]
    /// + [`UdsRpcServer::shutdown_token`] to drain while `run` owns `self`).
    pub fn stream_limit(&self) -> Arc<Semaphore> {
        Arc::clone(&self.stream_limit)
    }

    /// The configured server-wide concurrent-stream capacity.
    pub fn capacity(&self) -> u32 {
        self.stream_limit_capacity
    }

    /// Graceful shutdown: cancel the accept loop, then wait (bounded by
    /// [`DRAIN_TIMEOUT`]) for every in-flight stream to release its permit, then
    /// close the semaphore. Mirrors [`QuinnRpcServer::shutdown`](super::quinn_transport::QuinnRpcServer::shutdown).
    pub async fn shutdown(stream_limit: &Arc<Semaphore>, capacity: u32, token: &CancellationToken) {
        token.cancel();
        match tokio::time::timeout(DRAIN_TIMEOUT, stream_limit.acquire_many(capacity)).await {
            Ok(Ok(permits)) => {
                permits.forget();
                stream_limit.close();
            }
            Ok(Err(_)) => { /* already closed */ }
            Err(_) => {
                tracing::warn!(timeout = ?DRAIN_TIMEOUT, "uds-rpc: drain timed out, forcing teardown");
                stream_limit.close();
            }
        }
    }

    /// Run the accept loop until `shutdown` is cancelled. Each accepted
    /// connection is served on its own task by [`serve_rpc_connection`], sharing
    /// the server-wide [`Semaphore`] so [`UdsRpcServer::shutdown`] can drain
    /// every in-flight stream regardless of which connection it belongs to.
    pub async fn run(self) -> Result<()> {
        loop {
            tokio::select! {
                biased;
                _ = self.shutdown.cancelled() => {
                    tracing::debug!("uds-rpc: accept loop cancelled");
                    return Ok(());
                }
                accepted = self.listener.accept() => {
                    let stream = match accepted {
                        Ok((stream, _addr)) => {
                            // SECURITY (#207): enforce same-uid-only access via
                            // SO_PEERCRED. The socket is already 0o700 so only the
                            // daemon's user can connect, but kernel-level peer-cred
                            // verification adds defense-in-depth against fd-passing
                            // or capability tricks that bypass filesystem permissions.
                            #[cfg(target_os = "linux")]
                            {
                                use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
                                let daemon_uid = nix::unistd::getuid();
                                match getsockopt(&stream, PeerCredentials) {
                                    Ok(cred) if cred.uid() == daemon_uid.as_raw() => {}
                                    Ok(cred) => {
                                        tracing::warn!(
                                            peer_uid = cred.uid(),
                                            daemon_uid = daemon_uid.as_raw(),
                                            "uds-rpc: peer uid mismatch, rejecting connection"
                                        );
                                        continue;
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "uds-rpc: SO_PEERCRED failed, rejecting connection"
                                        );
                                        continue;
                                    }
                                }
                            }
                            stream
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "uds-rpc: listener accept error");
                            continue;
                        }
                    };

                    // Connection cap (#162): reject (drop) at cap rather than queue.
                    let conn_permit = match Arc::clone(&self.connection_limit).try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => {
                            tracing::warn!("uds-rpc: connection cap reached, rejecting connection");
                            drop(stream);
                            continue;
                        }
                    };

                    let processor = Arc::clone(&self.processor);
                    let signing_key = self.signing_key.clone();
                    let stream_limit = Arc::clone(&self.stream_limit);
                    let read_timeout = self.read_timeout;
                    let shutdown = self.shutdown.clone();
                    tokio::spawn(async move {
                        let _conn_permit = conn_permit; // released when this connection ends

                        // Read the plane byte + spin up the yamux session, bounded
                        // so a stalled handshake never pins a connection permit.
                        let (plane, session) = match tokio::time::timeout(
                            super::rpc_session::HANDSHAKE_TIMEOUT,
                            accept_uds(stream),
                        )
                        .await
                        {
                            Ok(Ok(pair)) => pair,
                            Ok(Err(e)) => {
                                tracing::debug!(error = %e, "uds-rpc: accept_uds failed");
                                return;
                            }
                            Err(_) => {
                                tracing::warn!("uds-rpc: plane handshake timed out");
                                return;
                            }
                        };

                        match plane {
                            PLANE_RPC => {
                                if let Err(e) = serve_rpc_connection(
                                    session, processor, signing_key, stream_limit, read_timeout, shutdown,
                                )
                                .await
                                {
                                    tracing::debug!(error = ?e, "uds-rpc: serve connection ended with error");
                                }
                            }
                            PLANE_MOQ => {
                                // moq streaming over UDS is a later increment; the
                                // same UdsSession is moq-capable, but the daemon
                                // does not yet wire a moq Server on this listener.
                                tracing::warn!("uds-rpc: PLANE_MOQ on the RPC listener — not yet served, closing");
                            }
                            other => {
                                tracing::warn!(plane = other, "uds-rpc: unknown plane, closing");
                            }
                        }
                    });
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::lazy_uds::LazyUdsTransport;
    use crate::transport::rpc_session::from_fn;
    use crate::transport_traits::Transport;
    use bytes::Bytes;

    fn temp_sock(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("uds-server-{}-{}", tag, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir.join("rpc.sock")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uds_server_round_trips_via_lazy_client() {
        let path = temp_sock("rt");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let (sk, _vk) = crate::generate_signing_keypair();
        let processor = from_fn(|req: Bytes| async move { Ok(req) }); // echo
        let server = UdsRpcServer::new(listener, processor, sk);
        let token = server.shutdown_token();
        let srv = tokio::spawn(server.run());

        let client = LazyUdsTransport::new(path.clone());
        let r1 = client.send(b"ping".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(r1, b"ping");
        // Reuse the cached session over a second multiplexed stream.
        let r2 = client.send(b"pong".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(r2, b"pong");

        token.cancel();
        let _ = srv.await;
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uds_server_shutdown_drains() {
        let path = temp_sock("drain");
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();

        let (sk, _vk) = crate::generate_signing_keypair();
        let processor = from_fn(|req: Bytes| async move { Ok(req) });
        let server = UdsRpcServer::new(listener, processor, sk);
        let token = server.shutdown_token();
        let stream_limit = server.stream_limit();
        let capacity = server.capacity();
        let srv = tokio::spawn(server.run());

        let client = LazyUdsTransport::new(path.clone());
        assert_eq!(client.send(b"x".to_vec(), Some(4_000)).await.unwrap(), b"x");

        // Graceful drain completes promptly with no in-flight streams.
        UdsRpcServer::shutdown(&stream_limit, capacity, &token).await;
        let _ = srv.await;
        let _ = std::fs::remove_file(&path);
    }
}
