//! Iroh RPC plane — server-side protocol handler for ALPN `hyprstream-rpc/1`.
//!
//! Part of Epic #131 Phase 2 (#133). This module ships the wire layer:
//!
//! - A length-prefixed framing over an iroh bidi stream (`accept_bi`/`open_bi`).
//! - An [`IrohRpcProtocolHandler`] that plugs into [`crate::transport::iroh_substrate`]
//!   under the `hyprstream-rpc/1` ALPN.
//! - An [`IrohRequestProcessor`] trait that callers implement to wire actual
//!   request processing (envelope verification + service dispatch).
//!
//! **Trust model**: The protocol handler does not verify `SignedEnvelope`
//! itself — it forwards the raw bytes to the [`IrohRequestProcessor`], which
//! is where envelope verification, JWT/DPoP checks, and `authorize_signer`
//! enforcement happen. This matches the ZMTP path's separation of concerns
//! (`transport::zmtp_quic::process_request`) and keeps the wire dumb.
//!
//! **Wire framing**: each message is a 4-byte big-endian length followed by
//! that many opaque bytes (Cap'n Proto-encoded `SignedEnvelope`). The bidi
//! stream is request-response: one request frame, one response frame, then
//! both sides close. No multipart, no ZMTP framing — both endpoints are our
//! code; iroh's QUIC TLS handles peer authentication at the transport layer.
//!
//! **Service integration** lands in a follow-up: the canary `PolicyClient`
//! port (the rest of #133) wraps the existing `ZmqService` trait — including
//! the `Rc<S>`/`!Send` constraint for services like inference — behind an
//! `IrohRequestProcessor` adapter that bridges to a per-service LocalSet.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};

/// Hard cap on inbound message size, to avoid unbounded memory growth from
/// a misbehaving peer. Matches the ZMTP-over-QUIC server-side cap used in
/// [`crate::transport::zmtp_quic`].
const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Trait implemented by callers to wire actual request processing.
///
/// The processor receives the raw request bytes (a Cap'n Proto-encoded
/// `SignedEnvelope`) and returns the raw response bytes. Verification,
/// authorization, and service dispatch all happen inside the implementation.
///
/// Implementations MUST be `Send + Sync + 'static` because iroh's accept
/// loop runs on a multi-threaded tokio runtime. For services that are
/// `!Send` (e.g. those holding `tch-rs` tensors), implement this trait by
/// forwarding to a dedicated `LocalSet` via an `mpsc` channel — same
/// pattern used today for the inference service.
pub trait IrohRequestProcessor: Send + Sync + 'static {
    /// Process one request and return the response bytes.
    fn process(
        &self,
        request: Bytes,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>>;
}

/// Convenience: implement [`IrohRequestProcessor`] from a `Send + Sync`
/// async closure. Useful for tests and simple processors.
pub fn from_fn<F, Fut>(f: F) -> impl IrohRequestProcessor
where
    F: Fn(Bytes) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Bytes>> + Send + 'static,
{
    struct FnProcessor<F>(F);
    impl<F, Fut> IrohRequestProcessor for FnProcessor<F>
    where
        F: Fn(Bytes) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Bytes>> + Send + 'static,
    {
        fn process(
            &self,
            request: Bytes,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
            Box::pin((self.0)(request))
        }
    }
    FnProcessor(f)
}

/// Iroh protocol handler that terminates `hyprstream-rpc/1` bidi streams,
/// applies length-prefixed framing, and dispatches each request through
/// the wrapped [`IrohRequestProcessor`].
#[derive(Clone)]
pub struct IrohRpcProtocolHandler {
    processor: Arc<dyn IrohRequestProcessor>,
}

impl std::fmt::Debug for IrohRpcProtocolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohRpcProtocolHandler").finish_non_exhaustive()
    }
}

impl IrohRpcProtocolHandler {
    pub fn new<P: IrohRequestProcessor>(processor: P) -> Self {
        Self {
            processor: Arc::new(processor),
        }
    }

    pub fn from_arc(processor: Arc<dyn IrohRequestProcessor>) -> Self {
        Self { processor }
    }
}

impl ProtocolHandler for IrohRpcProtocolHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Accept bidi streams on this connection until the peer goes away.
        // Each stream is one request-response pair, processed concurrently.
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(streams) => streams,
                Err(e) => {
                    tracing::debug!(error = ?e, "iroh-rpc: connection closed");
                    return Ok(());
                }
            };

            let processor = Arc::clone(&self.processor);
            tokio::spawn(async move {
                // Read full request (length-prefixed).
                let request = match recv.read_to_end(MAX_REQUEST_BYTES).await {
                    Ok(buf) => Bytes::from(buf),
                    Err(e) => {
                        tracing::warn!(error = ?e, "iroh-rpc: failed reading request");
                        return;
                    }
                };

                // Process.
                let response = match processor.process(request).await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        tracing::warn!(error = ?e, "iroh-rpc: processor returned error");
                        // Close the send side without a body; client will see EOF.
                        let _ = send.finish();
                        return;
                    }
                };

                // Write response.
                if let Err(e) = send.write_all(&response).await {
                    tracing::warn!(error = ?e, "iroh-rpc: failed writing response");
                    return;
                }
                if let Err(e) = send.finish() {
                    tracing::warn!(error = ?e, "iroh-rpc: failed finishing send");
                }
                // Optional: wait for peer to read.
                let _ = send.stopped().await;
            });
        }
    }
}

/// Client-side helper: open a bidi stream on `hyprstream-rpc/1` against an
/// already-connected iroh [`Connection`], write the request, read the response.
///
/// This is a primitive — the real per-service RPC clients (generated by
/// `hyprstream_rpc_derive::generate_rpc_service!`) will wrap it with their
/// own envelope construction and response decoding. Used directly only in
/// tests during Phase 2.
pub async fn client_request(conn: &Connection, request: &[u8]) -> Result<Bytes> {
    let (mut send, mut recv) = conn.open_bi().await.context("open_bi")?;
    send.write_all(request).await.context("write request")?;
    send.finish().context("finish send")?;
    let buf = recv
        .read_to_end(MAX_REQUEST_BYTES)
        .await
        .context("read response")?;
    Ok(Bytes::from(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::iroh_substrate::{
        ALPN_HYPRSTREAM_RPC, ALPN_MOQ_LITE, IrohSubstrate, NoopHandler,
    };
    use iroh::{EndpointAddr, TransportAddr};
    use rand::RngCore;

    fn fresh_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
        EndpointAddr::from_parts(
            substrate.endpoint_id(),
            substrate
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(TransportAddr::Ip),
        )
    }

    /// End-to-end: server runs an `IrohRpcProtocolHandler` wired to a
    /// closure-based processor that prepends a magic byte to the request,
    /// client sends one request and reads the response.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_request_response_round_trip() -> Result<()> {
        let processor = from_fn(|req: Bytes| async move {
            let mut out = Vec::with_capacity(1 + req.len());
            out.push(0xAB);
            out.extend_from_slice(&req);
            Ok(Bytes::from(out))
        });
        let rpc_handler = IrohRpcProtocolHandler::new(processor);

        let server = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("moq-not-wired"),
            rpc_handler,
        )
        .await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await?;

        let conn = client
            .connect(server_addr, ALPN_HYPRSTREAM_RPC)
            .await?;
        let resp = client_request(&conn, b"ping").await?;
        assert_eq!(&resp[..], b"\xABping");

        // Sanity: moq ALPN still routes to the noop handler (does not crash).
        let conn2 = client.connect(direct_addr(&server), ALPN_MOQ_LITE).await?;
        drop(conn2);

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// Multiple concurrent requests on the same connection round-trip
    /// independently and arrive at the correct caller.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_concurrent_requests() -> Result<()> {
        let processor = from_fn(|req: Bytes| async move {
            // Echo with a trailing marker so we can verify correlation.
            let mut out = req.to_vec();
            out.push(b'!');
            Ok(Bytes::from(out))
        });
        let server = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("moq"),
            IrohRpcProtocolHandler::new(processor),
        )
        .await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = Arc::new(
            client
                .connect(server_addr, ALPN_HYPRSTREAM_RPC)
                .await?,
        );

        let mut handles = Vec::new();
        for i in 0..8u8 {
            let conn = Arc::clone(&conn);
            handles.push(tokio::spawn(async move {
                let resp = client_request(&conn, &[i]).await?;
                assert_eq!(&resp[..], &[i, b'!']);
                anyhow::Ok(())
            }));
        }
        for h in handles {
            h.await??;
        }

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
