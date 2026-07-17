//! `/9p` WebTransport plane path-mux (H1b / #765).
//!
//! Proves the third arm of the QUIC WebTransport path mux
//! (`quinn_transport.rs`): a WebTransport session whose CONNECT URL path is
//! [`NINEP_PATH`] is routed to the injected [`NinePWtHandler`] (NOT the RPC core
//! or the `/moq` plane), the client-opened bidi stream is delivered to the
//! handler as a working `AsyncRead + AsyncWrite` byte stream (the WT analogue of
//! H1a's ws↔9p pump), and the session is served over a **cert-pinned** WT client
//! — exactly the reach layer the self-signed mesh uses.
//!
//! The 9P core (`Translator::serve_connection` + attach-time `Tattach.uname`
//! ticket) lives in `hyprstream-9p` and is covered by its own attach tests; this
//! test isolates the transport arm with an echo handler so no 9P/torch stack is
//! pulled into `hyprstream-rpc`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use rand::RngCore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use hyprstream_rpc::dial::NINEP_PATH;
use hyprstream_rpc::transport::quinn_transport::{
    cert_sha256, connect_pinned_hashes_path, NinePWtHandler, NinePWtStream, QuinnRpcServer,
};

fn fresh_signing_key() -> SigningKey {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    SigningKey::from_bytes(&k)
}

/// Loopback `web_transport_quinn` server with a self-signed cert.
/// Returns (server, bound_addr, leaf_cert_der).
fn build_server() -> Result<(web_transport_quinn::Server, std::net::SocketAddr, Vec<u8>)> {
    let cert_key = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
    let cert_der = cert_key.cert.der().to_vec();
    let key_der = cert_key.key_pair.serialize_der();

    let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.clone())];
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
        key_der,
    ));

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse()?;
    let server = web_transport_quinn::ServerBuilder::new()
        .with_addr(addr)
        .with_certificate(chain, key)
        .map_err(|e| anyhow!("quinn server build: {e}"))?;
    let bound = server.local_addr()?;
    Ok((server, bound, cert_der))
}

/// Handler standing in for the 9P core: records that it was invoked (proving
/// `/9p` routing) and echoes bytes back over the joined WT bidi stream (proving
/// the stream is a working duplex the `Translator` could serve).
struct EchoNineP {
    invoked: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl NinePWtHandler for EchoNineP {
    async fn serve(&self, mut stream: Box<dyn NinePWtStream>) {
        self.invoked.notify_one();
        let mut buf = [0u8; 4096];
        loop {
            match stream.read(&mut buf).await {
                // EOF (peer half-closed) or a read error: clean teardown.
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stream.write_all(&buf[..n]).await.is_err() {
                        break;
                    }
                    let _ = stream.flush().await;
                }
            }
        }
    }
}

/// A cert-pinned WT session on `/9p` reaches the injected handler; the bidi
/// stream round-trips bytes both directions.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn wt_ninep_path_routes_to_handler_and_streams() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let (server, addr, cert) = build_server()?;
    let invoked = Arc::new(tokio::sync::Notify::new());
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    let rpc = QuinnRpcServer::with_capacity(
        server,
        Arc::new(processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_ninep_handler(Arc::new(EchoNineP {
        invoked: invoked.clone(),
    }));
    let shutdown = rpc.shutdown_token();
    let server_task = tokio::spawn(rpc.run());

    // Cert-pinned client dials the `/9p` path (the self-signed-mesh reach).
    let pin = cert_sha256(&cert);
    let session = connect_pinned_hashes_path(addr, &[pin], NINEP_PATH).await?;

    // Open the bidi stream the server's `/9p` arm accepts, and round-trip bytes.
    let (mut send, mut recv) = session.open_bi().await?;
    send.write_all(b"hello-9p-over-wt").await?;

    let mut got = vec![0u8; b"hello-9p-over-wt".len()];
    tokio::time::timeout(Duration::from_secs(5), recv.read_exact(&mut got)).await??;
    assert_eq!(
        &got, b"hello-9p-over-wt",
        "echo must round-trip over the WT bidi stream"
    );

    // The handler was actually reached via the `/9p` path (not RPC/moq).
    tokio::time::timeout(Duration::from_secs(5), invoked.notified()).await?;

    // Clean teardown: dropping the session closes the WT connection; the
    // handler's serve loop sees EOF and returns (server-owned liveness).
    drop(session);
    shutdown.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), server_task).await;
    Ok(())
}

/// A `/9p` session with no handler configured is declined cleanly (the server
/// logs and drops it) without disturbing the accept loop — the plane is simply
/// off. We assert the server task keeps running afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn wt_ninep_without_handler_is_declined() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let (server, addr, cert) = build_server()?;
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    // No `.with_ninep_handler(...)` and no process-global registered.
    let rpc = QuinnRpcServer::with_capacity(
        server,
        Arc::new(processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    );
    let shutdown = rpc.shutdown_token();
    let server_task = tokio::spawn(rpc.run());

    let pin = cert_sha256(&cert);
    // Require the pinned WebTransport handshake to succeed first. This keeps a
    // pin, TLS, or generic connection failure from masquerading as evidence
    // that the `/9p` arm declined the session.
    let session = connect_pinned_hashes_path(addr, &[pin], NINEP_PATH).await?;
    let explicitly_denied = match session.open_bi().await {
        Err(_) => true,
        Ok((mut send, mut recv)) => {
            if send.write_all(b"ping").await.is_err() {
                true
            } else {
                let mut got = [0u8; 4];
                matches!(
                    tokio::time::timeout(Duration::from_secs(3), recv.read_exact(&mut got),).await,
                    Ok(Err(_))
                )
            }
        }
    };
    assert!(
        explicitly_denied,
        "no-handler /9p must be explicitly declined"
    );

    // The accept loop is still alive.
    assert!(
        !server_task.is_finished(),
        "accept loop must survive a declined /9p session"
    );
    shutdown.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), server_task).await;
    Ok(())
}
