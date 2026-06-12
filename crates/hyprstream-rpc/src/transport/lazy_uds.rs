//! Lazy-connecting UDS RPC transport (same-host `ipc` plane).
//!
//! Mirrors [`LazyQuinnTransport`](super::lazy_quinn::LazyQuinnTransport) and
//! [`LazyIrohTransport`](super::lazy_iroh::LazyIrohTransport): the
//! [`dial`](crate::dial::dial) factory is synchronous, so this holds the socket
//! path and connects on the **first `send()`**, caching the session — preserving
//! sync, zero-I/O construction.
//!
//! The connection is a [`UdsSession`](super::uds_session::UdsSession) on the RPC
//! plane (`PLANE_RPC`), wrapped as a [`SessionRpcTransport`] — the *same* generic
//! client used by the quinn and iroh backends, so the ipc plane carries the
//! identical Cap'n Proto bidi wire protocol with no transport-specific RPC code.
//! This is the post-ZMQ `ipc` transport: same-host processes keep `ipc`,
//! re-platformed off ZMQ framing onto SignedEnvelope-over-UDS.
//!
//! # Identity
//!
//! UDS has no transport-level peer identity; same-host trust + the app-layer
//! `SignedEnvelope` (verified against the response key) are the authentication.
//! Socket-file permissions + `SO_PEERCRED` are defense-in-depth owned by the
//! daemon that binds the listener (see #207). A `None` `server_verifying_key`
//! leaves the response identity unpinned (the envelope sig is still verified),
//! matching the quinn arm.
//!
//! Re-dial/self-heal semantics match the other lazy transports: a per-request
//! timeout keeps the session; a transport-fatal error drops it so the next call
//! re-dials.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use tokio::sync::Mutex;

use super::rpc_session::{RpcPendingStream, RpcPublishStub, SessionRpcTransport};
use super::uds_session::{connect_uds, UdsSession, PLANE_RPC};
use crate::transport_traits::Transport;

/// Default per-request deadline when the caller passes `None`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Ceiling on the lazy connect (UDS connect + yamux/SETUP). The per-request
/// deadline wraps only the request; without this a missing socket would hang the
/// first `send()`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// A UDS RPC transport that connects on first use and caches the session.
pub struct LazyUdsTransport {
    path: PathBuf,
    cached: Mutex<Option<SessionRpcTransport<UdsSession>>>,
}

impl LazyUdsTransport {
    /// Create a lazy transport for the UDS endpoint at `path`. No connection is
    /// made until the first `send()`.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            cached: Mutex::new(None),
        }
    }

    /// Return the cached connected transport, connecting once (single-flight
    /// under the lock) if not yet connected.
    async fn connected(&self) -> Result<SessionRpcTransport<UdsSession>> {
        let mut guard = self.cached.lock().await;
        if let Some(transport) = guard.as_ref() {
            return Ok(transport.clone());
        }
        let session = tokio::time::timeout(CONNECT_TIMEOUT, connect_uds(&self.path, PLANE_RPC))
            .await
            .map_err(|_| {
                anyhow!(
                    "uds connect to {} timed out after {CONNECT_TIMEOUT:?}",
                    self.path.display()
                )
            })?
            .map_err(|e| anyhow!("uds connect to {}: {e}", self.path.display()))?;
        let transport = SessionRpcTransport::new(session);
        *guard = Some(transport.clone());
        Ok(transport)
    }

    /// Drop the cached session so the next `send()` re-connects.
    async fn invalidate(&self) {
        *self.cached.lock().await = None;
    }
}

#[async_trait]
impl Transport for LazyUdsTransport {
    type Sub = RpcPendingStream;
    type Pub = RpcPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let transport = self.connected().await?;

        // Our deadline is authoritative so a per-request timeout (busy peer —
        // keep the session) is distinguishable from a transport-fatal error
        // (re-connect), matching the other lazy transports.
        let deadline = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_TIMEOUT);
        let inner_ceiling_ms = deadline.as_millis().saturating_mul(2).min(i32::MAX as u128) as i32;

        match tokio::time::timeout(deadline, transport.send(payload, Some(inner_ceiling_ms))).await {
            Err(_elapsed) => Err(anyhow!("uds RPC timeout after {deadline:?}")),
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => {
                self.invalidate().await;
                Err(e)
            }
        }
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("lazy UDS RPC transport does not support SUB — streaming is on the moq plane")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("lazy UDS RPC transport does not support PUB — streaming is on the moq plane")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::transport::rpc_session::{
        from_fn, serve_rpc_connection, DEFAULT_STREAM_LIMIT, REQUEST_READ_TIMEOUT,
    };
    use crate::transport::uds_session::accept_uds;
    use bytes::Bytes;
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use tokio_util::sync::CancellationToken;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lazy_uds_connects_on_first_send_and_caches() {
        let dir = std::env::temp_dir().join(format!("lazy-uds-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("rpc.sock");
        let _ = std::fs::remove_file(&path);
        let listener = tokio::net::UnixListener::bind(&path).unwrap();

        // Echo RPC server over UDS.
        let shutdown = CancellationToken::new();
        let shutdown_srv = shutdown.clone();
        let srv = tokio::spawn(async move {
            let (sk, _vk) = crate::generate_signing_keypair();
            let processor = Arc::new(from_fn(|req: Bytes| async move { Ok(req) }));
            let limit = Arc::new(Semaphore::new(DEFAULT_STREAM_LIMIT));
            loop {
                tokio::select! {
                    _ = shutdown_srv.cancelled() => break,
                    accepted = listener.accept() => {
                        let (stream, _) = accepted.unwrap();
                        let (plane, session) = accept_uds(stream).await.unwrap();
                        assert_eq!(plane, PLANE_RPC);
                        let p = Arc::clone(&processor);
                        let l = Arc::clone(&limit);
                        let sk = sk.clone();
                        let sd = shutdown_srv.clone();
                        tokio::spawn(async move {
                            let _ = serve_rpc_connection(
                                session, p, sk, l, REQUEST_READ_TIMEOUT, sd,
                            )
                            .await;
                        });
                    }
                }
            }
        });

        let t = LazyUdsTransport::new(path.clone());
        assert!(t.cached.lock().await.is_none(), "no connection before first send");
        let resp = t.send(b"ping".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(resp, b"ping");
        assert!(t.cached.lock().await.is_some(), "session cached after first send");
        // Second call reuses the cached session (multiplexed stream).
        let resp2 = t.send(b"pong".to_vec(), Some(4_000)).await.unwrap();
        assert_eq!(resp2, b"pong");

        shutdown.cancel();
        let _ = srv.await;
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lazy_uds_missing_socket_errors_without_hang() {
        let path = std::env::temp_dir().join(format!("lazy-uds-absent-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let t = LazyUdsTransport::new(path);
        let res = tokio::time::timeout(Duration::from_secs(5), t.send(b"x".to_vec(), Some(2_000)))
            .await
            .expect("send must complete (with an error), not hang");
        assert!(res.is_err(), "dialing an absent socket must fail");
        assert!(t.cached.lock().await.is_none(), "a failed connect caches nothing");
    }

    #[tokio::test]
    async fn subscribe_publish_bail() {
        let t = LazyUdsTransport::new(PathBuf::from("/nonexistent.sock"));
        assert!(t.subscribe(b"topic").await.is_err());
        assert!(t.publish(b"topic").await.is_err());
    }
}
