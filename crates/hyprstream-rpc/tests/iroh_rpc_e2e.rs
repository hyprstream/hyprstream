//! End-to-end Phase 2 canary: real SignedEnvelope round-trip over iroh.
//!
//! Wires a minimal [`RequestService`] through:
//! 1. [`LocalServiceBridge`] (dedicated LocalSet thread)
//! 2. [`IrohRpcProtocolHandler`] on ALPN `hyprstream-rpc/1`
//! 3. iroh `Connection` (direct dial, no DNS/pkarr)
//! 4. [`IrohTransport`] (client wire)
//! 5. [`RpcClientImpl`] (envelope sign + JWT + response unwrap)
//!
//! Verifies the canary exit criterion's wire-and-envelope half: a real
//! `SignedEnvelope` issued by `RpcClientImpl::call` traverses iroh and
//! arrives at the service's `handle_request` with verified identity, and
//! the signed response verifies back at the client.
//!
//! Full PolicyClient integration (running the whole `services/policy/tests/`
//! suite over iroh) requires `services/factories.rs` wiring — tracked as
//! Phase 2 part 3 of #133.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};

use hyprstream_rpc::envelope::InMemoryNonceCache;
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{
    ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler,
};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;
use hyprstream_rpc::transport::TransportConfig;
use iroh::{EndpointAddr, TransportAddr};
use rand::RngCore;

/// Minimal `RequestService` for the canary: echoes the request payload with a
/// 1-byte prefix so the test can verify identity round-trip.
struct EchoService {
    name: String,
    ctx: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
}

impl EchoService {
    fn new(signing_key: SigningKey) -> Self {
        Self {
            name: "iroh-echo".to_owned(),
            ctx: Arc::new(zmq::Context::new()),
            transport: TransportConfig::inproc("iroh-echo-unused"),
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

    fn context(&self) -> &Arc<zmq::Context> {
        &self.ctx
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

fn fresh_node_key() -> [u8; 32] {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn signed_envelope_round_trip_over_iroh() -> Result<()> {
    // ─── Server side ──────────────────────────────────────────────────────
    let server_signing = fresh_signing_key();
    let server_verifying: VerifyingKey = server_signing.verifying_key();
    let server_nonce_cache = Arc::new(InMemoryNonceCache::new());

    let bridge = LocalServiceBridge::spawn(
        EchoService::new(server_signing.clone()),
        server_nonce_cache,
        0,
    )?;

    let server = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(bridge, server_signing.clone()),
    )
    .await?;
    let server_addr = direct_addr(&server);

    // ─── Client side ──────────────────────────────────────────────────────
    let client_signing = fresh_signing_key();
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;

    let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;
    let transport = IrohTransport::new(conn);
    let rpc = RpcClientImpl::new(
        LocalSigner::new(client_signing.clone()),
        transport,
        Some(server_verifying),
    );

    // Real signed envelope round-trip.
    let response = rpc.call(b"ping-payload".to_vec()).await?;
    assert_eq!(&response[..], b"\xECping-payload");

    // Second call on the same connection to exercise multiple bidi streams.
    let response2 = rpc.call(b"second".to_vec()).await?;
    assert_eq!(&response2[..], b"\xECsecond");

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
