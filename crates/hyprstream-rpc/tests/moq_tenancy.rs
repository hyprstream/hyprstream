//! Tenancy-isolation gap demonstrations for issue #1128 (spike).
//!
//! These tests assert the **current, leaky** behaviour of the quinn
//! WebTransport `/moq` serve path and are expected to PASS against today's
//! code. They exist to document — with a networked, end-to-end repro — that a
//! shared `/moq` endpoint leaks cross-tenant broadcast names (and, in relay
//! mode, accepts cross-tenant publishes) before the fix lands. When the
//! `/moq` CONNECT gains an authenticated peer identity, these assertions must
//! be INVERTED (each subject must see only its own tenant's announces).
//!
//! Root cause (all references against the code under test):
//!
//! - `crates/hyprstream-rpc/src/transport/quinn_transport.rs:435` — every
//!   `/moq` CONNECT is treated as `PeerIdentity::anonymous()` because the
//!   WebTransport CONNECT is not mutually authenticated.
//! - `quinn_transport.rs:447-458` — `MoqAuthzConfig::tenant_for(&anonymous)`
//!   yields `None`, and the `None => consumer` arm at `:457` serves the
//!   PROCESS-GLOBAL UNSCOPED `OriginConsumer` (fail-open). No
//!   `tenant_resolver` can distinguish two anonymous subjects, so per-tenant
//!   announce scoping (`moq_authz::tenant_scoped_consumer`) is dead code on
//!   this path.
//! - Production wiring installs `MoqAuthzConfig::default()` (no resolver):
//!   `crates/hyprstream-service/src/service/spawner/service.rs:139-163`.
//! - Relay mode (`QuinnRpcServer::with_moq_relay`, `quinn_transport.rs:412-413`)
//!   bypasses authz entirely: `moq_net::Server::new().with_origin(...)` both
//!   ingests and re-serves with no peer check at all.
//!
//! Fix direction: authenticate the `/moq` CONNECT (client cert / app-level
//! token — see the TODO at `quinn_transport.rs:441-446`), derive the tenant
//! from that identity, and serve a `tenant_scoped_consumer`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use rand::RngCore;

use hyprstream_rpc::dial::MOQ_PATH;
use hyprstream_rpc::transport::quinn_transport::{cert_sha256, connect_pinned_hashes_path, QuinnRpcServer};

const ALICE_BROADCAST: &str = "alice/streams/run-1/i0";
const BOB_BROADCAST: &str = "bob/streams/run-9/i0";
const MALLORY_BROADCAST: &str = "mallory/injected/i0";

fn fresh_signing_key() -> SigningKey {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    SigningKey::from_bytes(&k)
}

/// Build a hermetic `web_transport_quinn` server on loopback with a self-signed
/// cert. Returns (server, bound_addr, leaf_cert_der).
fn build_server() -> Result<(web_transport_quinn::Server, std::net::SocketAddr, Vec<u8>)> {
    let cert_key = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
    let cert_der = cert_key.cert.der().to_vec();
    let key_der = cert_key.key_pair.serialize_der();

    let chain = vec![rustls::pki_types::CertificateDer::from(cert_der.clone())];
    let key =
        rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(key_der));

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse()?;
    let server = web_transport_quinn::ServerBuilder::new()
        .with_addr(addr)
        .with_certificate(chain, key)
        .map_err(|e| anyhow!("quinn server build: {e}"))?;
    let bound = server.local_addr()?;
    Ok((server, bound, cert_der))
}

/// Connect a raw moq client ("subject") to the shared `/moq` endpoint and
/// return the client-side consumer whose announce cursor reflects everything
/// the server chose to publish to this session.
async fn connect_subject(
    addr: std::net::SocketAddr,
    pin: [u8; 32],
) -> Result<moq_net::OriginConsumer> {
    let client_producer = moq_net::Origin::random().produce();
    let client_consumer = client_producer.consume();
    let moq_client = moq_net::Client::new().with_consume(client_producer);
    let session = connect_pinned_hashes_path(addr, &[pin], MOQ_PATH).await?;
    // Keep the moq session alive for the consumer's lifetime.
    let moq_session = moq_client
        .connect(session)
        .await
        .map_err(|e| anyhow!("moq handshake: {e}"))?;
    tokio::spawn(async move {
        let _ = moq_session.closed().await;
    });
    Ok(client_consumer)
}

/// True once `name` is announced to this consumer within the timeout.
async fn announces(consumer: &moq_net::OriginConsumer, name: &str) -> bool {
    matches!(
        tokio::time::timeout(Duration::from_secs(5), consumer.announced_broadcast(name)).await,
        Ok(Some(_))
    )
}

/// ISSUE #1128 GAP DEMONSTRATION — asserts the CURRENT leaky behaviour.
///
/// Two independent raw moq clients ("subject A" and "subject B") connect to the
/// SAME `/moq` endpoint of one server that hosts broadcasts for two tenants
/// (`alice/...` and `bob/...`) under one origin. Because the serve path treats
/// every peer as anonymous (`quinn_transport.rs:435`) and fails open to the
/// unscoped consumer (`:457`), BOTH subjects enumerate BOTH tenants' broadcast
/// names. A tenant-isolated endpoint would show subject A only `alice/...` —
/// flip these assertions once the `/moq` CONNECT is authenticated.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn shared_endpoint_announces_cross_tenant_gap1128() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    // One process-global origin carrying BOTH tenants' broadcasts.
    let producer = moq_net::Origin::random().produce();
    let consumer = producer.consume();
    // Keep the broadcasts alive for the whole test (dropping unannounces).
    let _alice = producer
        .create_broadcast(ALICE_BROADCAST)
        .ok_or_else(|| anyhow!("create alice broadcast"))?;
    let _bob = producer
        .create_broadcast(BOB_BROADCAST)
        .ok_or_else(|| anyhow!("create bob broadcast"))?;

    // Server built exactly like production wiring: `with_moq_consumer` with the
    // default (resolver-less) MoqAuthzConfig — see service.rs:139-163.
    let (server, addr, cert_der) = build_server()?;
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        Arc::new(processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_consumer(consumer.clone());
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());

    let pin = cert_sha256(&cert_der);

    // Two independent subjects on the SAME shared endpoint.
    let subject_a = connect_subject(addr, pin).await?;
    let subject_b = connect_subject(addr, pin).await?;

    // CURRENT (leaky) behaviour: each subject sees BOTH tenants' announces.
    for (label, consumer) in [("subject A", &subject_a), ("subject B", &subject_b)] {
        assert!(
            announces(consumer, ALICE_BROADCAST).await,
            "{label} sees alice's broadcast on the shared endpoint (current leak)"
        );
        assert!(
            announces(consumer, BOB_BROADCAST).await,
            "{label} sees bob's broadcast on the shared endpoint (current leak) — \
             issue #1128: this must become false once /moq peers are authenticated"
        );
    }

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// ISSUE #1128 GAP DEMONSTRATION (relay mode) — asserts the CURRENT leaky
/// behaviour of `with_moq_relay` (`quinn_transport.rs:412-413`): any anonymous
/// client can PUBLISH a broadcast into the shared relay origin with no
/// authorization at all, and it is re-served to other subscribers.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn relay_mode_client_publishes_without_authz_gap1128() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    // Relay origin starts EMPTY; the client injects the broadcast.
    let relay_producer = moq_net::Origin::random().produce();
    let relay_consumer = relay_producer.consume();

    let (server, addr, cert_der) = build_server()?;
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        Arc::new(processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_relay(relay_producer.clone());
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());

    let pin = cert_sha256(&cert_der);

    // "Mallory": an unauthenticated client announcing into the shared origin.
    let attacker_producer = moq_net::Origin::random().produce();
    let _injected = attacker_producer
        .create_broadcast(MALLORY_BROADCAST)
        .ok_or_else(|| anyhow!("create attacker broadcast"))?;
    let moq_client = moq_net::Client::new().with_origin(attacker_producer);
    let session = connect_pinned_hashes_path(addr, &[pin], MOQ_PATH).await?;
    let moq_session = moq_client
        .connect(session)
        .await
        .map_err(|e| anyhow!("attacker moq handshake: {e}"))?;
    tokio::spawn(async move {
        let _ = moq_session.closed().await;
    });

    // CURRENT (leaky) behaviour: the relay ingests the anonymous publish…
    assert!(
        announces(&relay_consumer, MALLORY_BROADCAST).await,
        "relay origin ingested an anonymous client's broadcast with no authz (current leak)"
    );

    // …and re-serves it to ANY other subscriber of the shared endpoint.
    let victim = connect_subject(addr, pin).await?;
    assert!(
        announces(&victim, MALLORY_BROADCAST).await,
        "a second anonymous subscriber is served mallory's injected broadcast (current leak)"
    );

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}
