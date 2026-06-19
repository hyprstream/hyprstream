//! End-to-end canary: real SignedEnvelope round-trip over the same-host UDS plane.
//!
//! Mirrors `iroh_rpc_e2e.rs` but over a Unix-domain socket, exercising the exact
//! bridged serve path the daemon will use for the post-ZMQ `ipc` transport:
//! 1. [`LocalServiceBridge`] — a (possibly `!Send`) [`RequestService`] on its own
//!    LocalSet thread, exposed as a `Send` processor.
//! 2. [`UdsRpcServer`] — UnixListener accept loop → `serve_rpc_connection` over a
//!    `UdsSession` (PLANE_RPC), feeding the same `process_request` dispatch core.
//! 3. [`LazyUdsTransport`] — client wire (connect-on-first-send, yamux-muxed).
//! 4. [`RpcClientImpl`] — envelope sign + response verify.
//!
//! Verifies that a real `SignedEnvelope` issued by `RpcClientImpl::call` traverses
//! the UDS plane, arrives at the service's `handle_request` with verified
//! identity, and the signed response verifies back at the client — the same
//! exit criterion the iroh canary proves, on the ipc transport.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::RngCore;

use hyprstream_rpc::envelope::InMemoryNonceCache;
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge;
use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
use hyprstream_rpc::transport::uds_server::UdsRpcServer;
use hyprstream_rpc::transport::TransportConfig;
use tokio::net::UnixListener;

/// Minimal `RequestService` for the canary: echoes the request payload with a
/// 1-byte prefix so the test can verify the identity round-trip.
struct EchoService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
}

impl EchoService {
    fn new(signing_key: SigningKey) -> Self {
        Self {
            name: "uds-echo".to_owned(),
            transport: TransportConfig::inproc("uds-echo-unused"),
            signing_key,
        }
    }
}

#[async_trait(?Send)]
impl RequestService for EchoService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let mut out = Vec::with_capacity(1 + payload.len());
        out.push(0xEC);
        out.extend_from_slice(payload);
        Ok((out, None))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

fn fresh_signing_key() -> SigningKey {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    SigningKey::from_bytes(&bytes)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn signed_envelope_round_trip_over_uds() -> Result<()> {
    // As an integration test this compiles hyprstream-rpc in non-test mode, where
    // the uninstalled global verify policy fail-closes to Hybrid (#160). Opt in to
    // the Classical policy this canary validates. `set`-by-first-write: ignore the
    // error if another test in this binary already installed a matching config.
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );

    // ─── Server side: bridge the service, serve it over a UnixListener ────────
    let server_signing = fresh_signing_key();
    let server_verifying: VerifyingKey = server_signing.verifying_key();
    let server_nonce_cache = Arc::new(InMemoryNonceCache::new());

    let bridge = LocalServiceBridge::spawn(
        EchoService::new(server_signing.clone()),
        server_nonce_cache,
        0,
    )?;

    let dir = std::env::temp_dir().join(format!("uds-e2e-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("rpc.sock");
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path)?;

    let server = UdsRpcServer::new(listener, bridge, server_signing.clone());
    let token = server.shutdown_token();
    let srv = tokio::spawn(server.run());

    // ─── Client side: real signed envelope over LazyUdsTransport ─────────────
    let client_signing = fresh_signing_key();
    let rpc = RpcClientImpl::new(
        LocalSigner::new(client_signing.clone()),
        LazyUdsTransport::new(path.clone()),
        Some(server_verifying),
    );

    let response = rpc.call(b"ping-payload".to_vec()).await?;
    assert_eq!(&response[..], b"\xECping-payload");

    // Second call reuses the cached session over a fresh multiplexed bidi stream.
    let response2 = rpc.call(b"second".to_vec()).await?;
    assert_eq!(&response2[..], b"\xECsecond");

    token.cancel();
    let _ = srv.await;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
    Ok(())
}
