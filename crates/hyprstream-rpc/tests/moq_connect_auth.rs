//! #1153 integration: authenticate the `/moq` WebTransport CONNECT and bind
//! tenant scoping.
//!
//! These are the **inverted** assertions of the #1128 spike gap tests
//! (`shared_endpoint_announces_cross_tenant_gap1128` and
//! `relay_mode_client_publishes_without_authz_gap1128`), now that the
//! `/moq` CONNECT is authenticated: a connected peer can only enumerate its
//! own tenant's broadcasts, an unauthenticated CONNECT is refused, and an
//! anonymous relay publish is refused.
//!
//! Boundary under test: a `MoqConnectAuthz` installed on the server verifies
//! a bearer JWT in the CONNECT's `Authorization` header (signature, `alg`,
//! `exp`, `iat`, `aud`) and resolves the tenant from the *verified* subject
//! via a server-side resolver. See
//! [`hyprstream_rpc::transport::moq_connect_auth`].

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use rand::RngCore;

use hyprstream_rpc::auth::claims::Claims;
use hyprstream_rpc::auth::jwt;
use hyprstream_rpc::dial::MOQ_PATH;
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::transport::moq_connect_auth::{MoqConnectAuthz, VerifiedSubjectTenantResolver};
use hyprstream_rpc::transport::quinn_transport::{
    cert_sha256, connect_pinned_hashes_path_with_headers, QuinnRpcServer,
};

const ALICE_BROADCAST: &str = "alice/streams/run-1/i0";
const BOB_BROADCAST: &str = "bob/streams/run-9/i0";
const MALLORY_BROADCAST: &str = "mallory/injected/i0";

fn fresh_signing_key() -> SigningKey {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    SigningKey::from_bytes(&k)
}

fn authz_for(
    verify_key: ed25519_dalek::VerifyingKey,
    mapping: &'static [(&'static str, &'static str)],
) -> MoqConnectAuthz {
    let resolver: VerifiedSubjectTenantResolver = Arc::new(move |sub: &str| {
        mapping
            .iter()
            .find(|(s, _)| *s == sub)
            .map(|(_, t)| t.to_string())
    });
    MoqConnectAuthz::new(verify_key, resolver).with_expected_aud("moq.test")
}

fn mint_aud(signing: &SigningKey, sub: &str) -> String {
    let now = chrono::Utc::now().timestamp();
    let mut claims = Claims::new(sub.to_owned(), now, now + 60);
    claims.aud = Some("moq.test".to_owned());
    jwt::encode(&claims, signing)
}

fn build_server() -> Result<(
    web_transport_quinn::Server,
    std::net::SocketAddr,
    Vec<u8>,
)> {
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

fn bearer_headers(token: &str) -> http::HeaderMap {
    let mut h = http::HeaderMap::new();
    h.insert(
        http::HeaderName::from_static("authorization"),
        http::HeaderValue::try_from(format!("Bearer {token}")).unwrap(),
    );
    h
}

/// Connect an authenticated moq client ("subject") and return its consumer
/// view of the server's announces.
async fn connect_authed(
    addr: std::net::SocketAddr,
    pin: [u8; 32],
    token: &str,
) -> Result<moq_net::OriginConsumer> {
    let client_producer = moq_net::Origin::random().produce();
    let client_consumer = client_producer.consume();
    let moq_client = moq_net::Client::new().with_consume(client_producer);
    let session =
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, bearer_headers(token))
            .await?;
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

fn trivial_processor() -> Arc<dyn hyprstream_rpc::transport::rpc_session::IrohRequestProcessor> {
    Arc::new(hyprstream_rpc::transport::rpc_session::from_fn(
        |req: bytes::Bytes| async move { Ok(req) },
    ))
}

/// #1153: an authenticated peer sees ONLY its own tenant's broadcasts on a
/// shared endpoint; cross-tenant enumeration is structurally impossible.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_peer_sees_only_own_tenant() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    // One shared origin carrying BOTH tenants' broadcasts.
    let producer = moq_net::Origin::random().produce();
    let consumer = producer.consume();
    let _alice = producer
        .create_broadcast(ALICE_BROADCAST)
        .ok_or_else(|| anyhow!("create alice"))?;
    let _bob = producer
        .create_broadcast(BOB_BROADCAST)
        .ok_or_else(|| anyhow!("create bob"))?;

    // Issuer key + authz mapping two subjects to two tenants.
    let issuer = fresh_signing_key();
    let authz = authz_for(
        issuer.verifying_key(),
        &[("did:key:alice", "alice"), ("did:key:bob", "bob")],
    );

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_consumer(consumer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // Subject A (alice) — authenticated, own tenant only.
    let alice_tok = mint_aud(&issuer, "did:key:alice");
    let alice_view = connect_authed(addr, pin, &alice_tok).await?;

    assert!(
        announces(&alice_view, ALICE_BROADCAST).await,
        "alice sees her own tenant's broadcast"
    );
    assert!(
        !announces(&alice_view, BOB_BROADCAST).await,
        "alice must NOT see bob's broadcast (cross-tenant refused) — #1128 gap closed"
    );

    // Subject B (bob) — authenticated, own tenant only.
    let bob_tok = mint_aud(&issuer, "did:key:bob");
    let bob_view = connect_authed(addr, pin, &bob_tok).await?;
    assert!(
        announces(&bob_view, BOB_BROADCAST).await,
        "bob sees his own tenant's broadcast"
    );
    assert!(
        !announces(&bob_view, ALICE_BROADCAST).await,
        "bob must NOT see alice's broadcast (cross-tenant refused)"
    );

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// #1153: an unauthenticated `/moq` CONNECT (no `Authorization` header) is
/// REFUSED — the WebTransport CONNECT never completes. Fail-closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn unauthenticated_connect_is_refused() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let producer = moq_net::Origin::random().produce();
    let consumer = producer.consume();
    let _alice = producer
        .create_broadcast(ALICE_BROADCAST)
        .ok_or_else(|| anyhow!("create alice"))?;

    let issuer = fresh_signing_key();
    let authz = authz_for(issuer.verifying_key(), &[("did:key:alice", "alice")]);

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_consumer(consumer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // No Authorization header → the server refuses the CONNECT before the
    // handshake completes, so the WebTransport connect itself errors. A
    // regression that admitted anonymous peers would return `Ok(_)` here,
    // failing the assertion.
    let empty = http::HeaderMap::new();
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, empty),
    )
    .await;
    match res {
        Ok(Err(_)) => { /* expected: CONNECT refused */ }
        Ok(Ok(_session)) => {
            return Err(anyhow!(
                "unauthenticated CONNECT was ADMITTED (server returned a session) — fail-open regression"
            ));
        }
        Err(_) => return Err(anyhow!("unauthenticated CONNECT hung (no refusal)")),
    }

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// #1153: a forged credential (wrong signature) is REFUSED — verification,
/// not parsing.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn forged_credential_is_refused() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let producer = moq_net::Origin::random().produce();
    let consumer = producer.consume();
    let _alice = producer
        .create_broadcast(ALICE_BROADCAST)
        .ok_or_else(|| anyhow!("create alice"))?;

    let issuer = fresh_signing_key();
    let rogue = fresh_signing_key();
    let authz = authz_for(issuer.verifying_key(), &[("did:key:alice", "alice")]);

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_consumer(consumer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // A token signed by a ROGUE key, but claiming alice's subject — must be
    // refused because the signature does not verify against the issuer key.
    // A regression that only PARSED (not verified) the token would admit the
    // forged subject and return `Ok(_)` here, failing the assertion.
    let forged = mint_aud(&rogue, "did:key:alice");
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, bearer_headers(&forged)),
    )
    .await;
    match res {
        Ok(Err(_)) => {}
        Ok(Ok(_session)) => {
            return Err(anyhow!(
                "forged-credential CONNECT was ADMITTED (signature not verified) — fail-open regression"
            ));
        }
        Err(_) => return Err(anyhow!("forged CONNECT hung (no refusal)")),
    }

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// #1153 (relay mode): an anonymous client can no longer CONNECT to the
/// shared relay endpoint — the inverted assertion of
/// `relay_mode_client_publishes_without_authz_gap1128`. The gate is the same
/// CONNECT-time auth as the consumer path: anonymous → refused before the
/// handshake completes. Asserted at the connect level (deterministic), not via
/// announce timing.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn relay_mode_refuses_anonymous_connect() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let relay_producer = moq_net::Origin::random().produce();
    let relay_consumer = relay_producer.consume();

    let issuer = fresh_signing_key();
    let authz = authz_for(issuer.verifying_key(), &[("did:key:alice", "alice")]);

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_relay(relay_producer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // "Mallory": an UNAUTHENTICATED client attempting to reach the shared
    // relay origin. With #1153 the CONNECT is refused before the handshake
    // completes. A regression that admitted anonymous relay publishers would
    // return `Ok(_)` here, failing the assertion (and the relay would ingest
    // mallory's broadcast — the #1128 relay-mode gap).
    let empty = http::HeaderMap::new();
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, empty),
    )
    .await;
    match res {
        Ok(Err(_)) => {}
        Ok(Ok(_session)) => {
            return Err(anyhow!(
                "anonymous relay CONNECT was ADMITTED (fail-open regression — #1128 relay gap)"
            ));
        }
        Err(_) => return Err(anyhow!("anonymous relay CONNECT hung (no refusal)")),
    }

    // Secondary, non-deterministic-by-timing guard: the relay origin must
    // never ingest an anonymous publish regardless. Bounded so it can't hang
    // the suite; the connect-level refusal above is the authoritative check.
    let _ = relay_consumer;
    assert!(
        !announces(&relay_consumer, MALLORY_BROADCAST).await,
        "relay must NOT ingest an anonymous publish"
    );

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// #1153 CRITICAL 3 (relay authorization): an authenticated publisher may
/// publish into its OWN tenant's namespace through the relay, but NOT into
/// another tenant's namespace. Authentication without authorization was the
/// gap (Alice could publish `bob/...`); the relay now scopes ingest to the
/// verified tenant prefix.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn relay_authenticated_cross_tenant_publish_refused() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let relay_producer = moq_net::Origin::random().produce();
    let relay_consumer = relay_producer.consume();

    let issuer = fresh_signing_key();
    // Mallory is provisioned as tenant "alice" but will attempt to publish
    // into bob's namespace.
    let authz = authz_for(issuer.verifying_key(), &[("did:key:mallory", "alice")]);

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_relay(relay_producer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // Authenticated as alice, attempt to publish BOB_BROADCAST into the shared
    // relay origin. The scoped ingest producer must reject the cross-tenant
    // publish — the relay origin never announces bob/...
    let tok = mint_aud(&issuer, "did:key:mallory");
    let attacker_producer = moq_net::Origin::random().produce();
    let _injected = attacker_producer
        .create_broadcast(BOB_BROADCAST)
        .ok_or_else(|| anyhow!("create bob broadcast"))?;
    let moq_client = moq_net::Client::new().with_origin(attacker_producer);
    let session =
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, bearer_headers(&tok))
            .await?;
    let _moq_session = moq_client.connect(session).await?;
    tokio::spawn(async move {
        let _ = _moq_session.closed().await;
    });

    assert!(
        !announces(&relay_consumer, BOB_BROADCAST).await,
        "relay must NOT ingest a cross-tenant publish (Alice publishing bob/) — \
         CRITICAL 3: authentication without authorization"
    );

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// #1153 CRITICAL 3 positive case: an authenticated publisher CAN publish into
/// its OWN tenant's namespace through the relay (the scope does not over-restrict
/// legitimate same-tenant traffic).
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn relay_authenticated_own_tenant_publish_succeeds() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    let relay_producer = moq_net::Origin::random().produce();
    let relay_consumer = relay_producer.consume();

    let issuer = fresh_signing_key();
    let authz = authz_for(issuer.verifying_key(), &[("did:key:alice", "alice")]);

    let (server, addr, cert_der) = build_server()?;
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        trivial_processor(),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_relay(relay_producer.clone())
    .with_moq_connect_authz(authz);
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());
    let pin = cert_sha256(&cert_der);

    // Authenticated as alice, publish ALICE_BROADCAST — own-tenant, must be
    // ingested and re-served by the relay.
    let tok = mint_aud(&issuer, "did:key:alice");
    let alice_producer = moq_net::Origin::random().produce();
    let _a = alice_producer
        .create_broadcast(ALICE_BROADCAST)
        .ok_or_else(|| anyhow!("create alice broadcast"))?;
    let moq_client = moq_net::Client::new().with_origin(alice_producer);
    let session =
        connect_pinned_hashes_path_with_headers(addr, &[pin], MOQ_PATH, bearer_headers(&tok))
            .await?;
    let _moq_session = moq_client.connect(session).await?;
    tokio::spawn(async move {
        let _ = _moq_session.closed().await;
    });

    assert!(
        announces(&relay_consumer, ALICE_BROADCAST).await,
        "relay must ingest an authenticated own-tenant publish"
    );

    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}

/// Compile-time check that the documented public API surface is wired.
#[test]
fn authz_surface_is_public() {
    fn _assert(
        _a: MoqConnectAuthz,
        _r: VerifiedSubjectTenantResolver,
        _p: PeerIdentity,
    ) {
    }
}
