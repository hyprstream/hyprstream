//! h3 QPACK decoded-header budget regression tests (CVE class: header
//! accordions; loopback only).
//!
//! Every hyprstream WebTransport server site accepts sessions through
//! `web_transport_quinn`, whose CONNECT header decode lives in the patched
//! `web-transport-proto` copy under `third_party/`. These tests drive a real
//! server endpoint on loopback with a raw QUIC client that hand-rolls the H3
//! handshake, so the malicious header sections hit the exact production decode
//! path (`ConnectRequest::read` -> `qpack::Headers::decode`):
//!
//! 1. a CONNECT whose single HEADERS frame carries more unique tiny fields
//!    than the entry-count cap is rejected cleanly — the connection is
//!    dropped by the server, the accept loop survives, and a legitimate
//!    session round-trips afterwards (no panic anywhere);
//! 2. a static-table accordion (2 wire bytes expanding to the longest
//!    predefined name+value pair, repeated to the 64 KiB frame cap) is
//!    rejected by the decoded-size cap;
//! 3. dynamic-table field lines (the lsquic-style accordion vector in full
//!    QPACK stacks) are rejected outright — this stack never materializes a
//!    dynamic table — and the server stays alive;
//! 4. a large-but-legal header set still connects and round-trips.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use rand::RngCore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use hyprstream_rpc::dial::NINEP_PATH;
use hyprstream_rpc::transport::quinn_transport::{
    cert_sha256, connect_pinned_hashes_path, connect_pinned_hashes_path_with_headers,
    NinePWtHandler, NinePWtStream, QuinnRpcServer,
};

fn fresh_signing_key() -> SigningKey {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    SigningKey::from_bytes(&k)
}

/// Loopback `web_transport_quinn` server with a self-signed cert.
fn build_server() -> Result<(web_transport_quinn::Server, SocketAddr, Vec<u8>)> {
    let cert_key = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
    let cert_der = cert_key.cert.der().to_vec();
    let key_der = cert_key.key_pair.serialize_der();
    let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.clone())];
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
        key_der,
    ));
    let addr: SocketAddr = "127.0.0.1:0".parse()?;
    let server = web_transport_quinn::ServerBuilder::new()
        .with_addr(addr)
        .with_certificate(chain, key)
        .map_err(|e| anyhow!("quinn server build: {e}"))?;
    let bound = server.local_addr()?;
    Ok((server, bound, cert_der))
}

/// Echo handler proving a session reached a serving plane.
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

/// rustls verifier that accepts the loopback self-signed cert (test-only).
#[derive(Debug)]
struct AcceptAnyCert;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::CryptoProvider::get_default()
            .map(|p| p.signature_verification_algorithms.supported_schemes())
            .unwrap_or_default()
    }
}

/// Raw QUIC endpoint speaking the `h3` ALPN — everything below the WebTransport
/// session layer is hand-rolled so malicious frames can be injected.
fn raw_client_endpoint() -> Result<quinn::Endpoint> {
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::ring::default_provider()));
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![web_transport_quinn::ALPN.as_bytes().to_vec()];
    let quic_config = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
    let mut endpoint = quinn::Endpoint::client("127.0.0.1:0".parse()?)?;
    endpoint.set_default_client_config(quinn::ClientConfig::new(Arc::new(quic_config)));
    Ok(endpoint)
}

/// QUIC + H3 SETTINGS handshake (no CONNECT). Mirrors what
/// `web_transport_quinn`'s client does before sending its CONNECT request.
async fn raw_h3_connect(endpoint: &quinn::Endpoint, addr: SocketAddr) -> Result<quinn::Connection> {
    let conn = endpoint
        .connect(addr, "localhost")
        .map_err(|e| anyhow!("raw connect: {e}"))?
        .await
        .map_err(|e| anyhow!("raw quic handshake: {e}"))?;

    let mut settings = web_transport_quinn::proto::Settings::default();
    settings.enable_webtransport(1);
    let mut uni = conn.open_uni().await?;
    settings.write(&mut uni).await?;

    // Consume (and sanity-check) the server's SETTINGS so both sides have
    // completed the exchange the real client performs.
    let mut server_uni = conn.accept_uni().await?;
    let server_settings = web_transport_quinn::proto::Settings::read(&mut server_uni).await?;
    if server_settings.supports_webtransport() == 0 {
        return Err(anyhow!("server did not enable webtransport"));
    }
    Ok(conn)
}

/// Wrap a qpack field section in a HEADERS frame, as it appears on the wire.
fn headers_frame(payload: &[u8]) -> Vec<u8> {
    use web_transport_quinn::proto::{Frame, VarInt};
    let mut buf = Vec::new();
    Frame::HEADERS.encode(&mut buf);
    VarInt::from_u32(payload.len() as u32).encode(&mut buf);
    buf.extend_from_slice(payload);
    buf
}

/// Send one CONNECT request on a fresh raw connection and wait for the server
/// to drop the connection (the observable form of the clean rejection).
async fn attack_connect(endpoint: &quinn::Endpoint, addr: SocketAddr, frame: &[u8]) -> Result<()> {
    let conn = raw_h3_connect(endpoint, addr).await?;
    let (mut send, _recv) = conn.open_bi().await?;
    send.write_all(frame).await?;
    send.finish()?;

    // The server rejects the decode and drops the connection. Any observable
    // close (or stream reset) counts; hanging counts as failure.
    tokio::time::timeout(Duration::from_secs(10), conn.closed())
        .await
        .map_err(|_| anyhow!("server never closed the connection after attack"))?;
    Ok(())
}

/// CONNECT frame carrying more unique tiny fields than the entry-count cap.
/// Names stay short so the encoded section stays under the 64 KiB wire frame
/// cap and the entry-count cap is the guard that rejects the decode. Built
/// with the production encoder so the wire bytes are exactly what a real
/// (hostile) client would send.
fn unique_header_flood_frame(count: usize) -> Result<Vec<u8>> {
    let mut headers = http::HeaderMap::new();
    for i in 0..count {
        headers.insert(
            http::HeaderName::from_bytes(format!("x-f-{:05}", i).as_bytes())?,
            http::HeaderValue::from_static(""),
        );
    }
    let url = url::Url::parse("https://localhost/")?;
    let request = web_transport_quinn::proto::ConnectRequest::from(url).with_headers(headers);
    let mut buf = Vec::new();
    request.encode(&mut buf)?;
    Ok(buf)
}

/// Static-table accordion: 2 wire bytes reference predefined entry 85
/// ("content-security-policy" + its long value), repeated to just under the
/// 64 KiB wire frame cap — ~3.2 MB of decoded header data per request before
/// the budget existed.
fn static_indexed_accordion_frame(refs: usize) -> Vec<u8> {
    let mut payload = vec![0x00, 0x00]; // insert count 0, delta base 0
    for _ in 0..refs {
        payload.extend_from_slice(&[0xFF, 22]); // indexed static field 85
    }
    headers_frame(&payload)
}

/// Dynamic-table field line (indexed, relative index 0): must be rejected
/// outright — this stack never materializes a QPACK dynamic table.
fn dynamic_table_ref_frame() -> Vec<u8> {
    headers_frame(&[0x00, 0x00, 0xBF, 0x00])
}

/// A `/9p` session round-trips bytes (proof the server accept loop survived).
async fn assert_server_serves(
    addr: SocketAddr,
    pin: [u8; 32],
    invoked: &Arc<tokio::sync::Notify>,
) -> Result<()> {
    let session = connect_pinned_hashes_path(addr, &[pin], NINEP_PATH).await?;
    let (mut send, mut recv) = session.open_bi().await?;
    send.write_all(b"still-alive").await?;
    let mut got = vec![0u8; b"still-alive".len()];
    tokio::time::timeout(Duration::from_secs(5), recv.read_exact(&mut got))
        .await
        .map_err(|_| anyhow!("echo timed out"))??;
    assert_eq!(&got, b"still-alive");
    tokio::time::timeout(Duration::from_secs(5), invoked.notified())
        .await
        .map_err(|_| anyhow!("handler never invoked"))?;
    Ok(())
}

/// Malicious CONNECT header sections are rejected cleanly and the server keeps
/// serving: entry-count cap, decoded-size cap (accordion), dynamic-table
/// rejection — then a legitimate session round-trips on the same endpoint.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn malicious_connect_header_budget_rejections_keep_server_alive() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let (server, addr, cert) = build_server()?;
    let invoked = Arc::new(tokio::sync::Notify::new());
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: Bytes| async move { Ok(req) });
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

    let pin = cert_sha256(&cert);
    let endpoint = raw_client_endpoint()?;

    // 4097 unique tiny fields: the entry-count cap cancels decode at field
    // 4097, long before any header map could grow unbounded.
    attack_connect(&endpoint, addr, &unique_header_flood_frame(4_100)?)
        .await
        .map_err(|e| anyhow!("unique-header flood attack failed: {e}"))?;

    // Static-table accordion at the wire frame cap.
    attack_connect(&endpoint, addr, &static_indexed_accordion_frame(30_000))
        .await
        .map_err(|e| anyhow!("static accordion attack failed: {e}"))?;

    // Dynamic-table field line: rejected outright (no dynamic table exists).
    attack_connect(&endpoint, addr, &dynamic_table_ref_frame())
        .await
        .map_err(|e| anyhow!("dynamic-table attack failed: {e}"))?;

    // The accept loop survived all three attacks and still serves sessions.
    assert_server_serves(addr, pin, &invoked)
        .await
        .map_err(|e| anyhow!("server unhealthy after attacks: {e}"))?;

    shutdown.cancel();
    let join = tokio::time::timeout(Duration::from_secs(5), server_task).await;
    let joined = join.map_err(|_| anyhow!("server task did not finish"))?;
    joined.map_err(|e| anyhow!("server task panicked: {e}"))??;
    Ok(())
}

/// A large-but-legal header set still connects and round-trips: the budget
/// rejects floods, not fat-but-legal sections. Sized to the wire path's real
/// legal ceiling — the upstream 64 KiB HEADERS frame cap (pre-existing) —
/// ~700 fields / ~52 KiB wire / ~72 KiB accounted decoded. Budget headroom up
/// to 256 KiB / 4096 fields is exercised at the decode layer in the vendored
/// crate's unit tests (no wire frame cap there).
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn large_legal_header_set_still_connects() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let (server, addr, cert) = build_server()?;
    let invoked = Arc::new(tokio::sync::Notify::new());
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: Bytes| async move { Ok(req) });
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

    let pin = cert_sha256(&cert);
    let mut headers = http::HeaderMap::new();
    let value = "v".repeat(60);
    for i in 0..700u32 {
        headers.insert(
            http::HeaderName::from_bytes(format!("x-pad-{:05}", i).as_bytes())?,
            http::HeaderValue::from_str(&value)?,
        );
    }
    // 700 * (name 11 + value 60 + 32 overhead) = 72_100 accounted decoded
    // bytes, ~52 KiB on the wire: large, legal, under both the budget and the
    // upstream 64 KiB frame cap.

    let session =
        connect_pinned_hashes_path_with_headers(addr, &[pin], NINEP_PATH, headers).await?;
    let (mut send, mut recv) = session.open_bi().await?;
    send.write_all(b"large-but-legal").await?;
    let mut got = vec![0u8; b"large-but-legal".len()];
    tokio::time::timeout(Duration::from_secs(5), recv.read_exact(&mut got))
        .await
        .map_err(|_| anyhow!("echo timed out"))??;
    assert_eq!(&got, b"large-but-legal");
    drop(session);

    shutdown.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), server_task).await;
    Ok(())
}
