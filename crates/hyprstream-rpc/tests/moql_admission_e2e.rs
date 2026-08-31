//! #1027 — authenticated inside-carrier `moql` admission, end-to-end.
//!
//! Two real iroh endpoints (one host): the server substrate binds the `moql`
//! ALPN with an [`IrohMoqProtocolHandler`] carrying a
//! [`MoqlAdmissionAuthenticator`]; the client runs the challenge/response
//! ([`prove_moql_admission`]) before the moq handshake.
//!
//! Evidence matrix (positive + one explicit negative per rejection class):
//!
//! - **positive** — `admitted_peer_streams_own_tenant_broadcast`: fresh
//!   Ed25519 + ML-DSA-65 proof against current accepted state admits, and the
//!   admitted subscriber reads its own tenant's frames over the wire.
//! - **cross-tenant** — same test: the admitted alice peer cannot enumerate or
//!   subscribe to bob's broadcasts (structural tenant scoping after admission).
//! - **replay** — `replayed_response_on_a_fresh_challenge_is_rejected`: a
//!   captured valid response replayed verbatim against a new challenge is
//!   rejected (the transcript binds the fresh server nonce); the consumed-
//!   transcript single-use rule is unit-tested in the module.
//! - **expiry** — `expired_accepted_state_is_rejected`.
//! - **rotation / state advance** — `state_advance_invalidates_previous_keys`:
//!   admission succeeds at epoch N, then the accepted state advances to
//!   epoch N+1 with new subject keys and the same proof is rejected.
//! - **NodeId-only** — `nodeid_only_carrier_is_rejected`: a carrier that never
//!   presents inside-carrier proof (or presents its NodeId as identity) is
//!   refused; the tenant resolver is never invoked.
//!
//! Libtorch is NOT required (this is transport, not inference).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh::{EndpointAddr, TransportAddr};
use moq_net::{Client, Group, Origin, OriginConsumer, OriginProducer, Track};
use parking_lot::Mutex;
use rand::RngCore;
use web_transport_iroh::Session;

use hyprstream_rpc::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, MlDsaSigningKey};
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::transport::iroh_moq::{
    IrohMoqProtocolHandler, MoqAuthzConfig,
};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, NoopHandler, ALPN_MOQ_LITE};
use hyprstream_rpc::transport::moql_admission::{
    admission_transcript, decode_challenge, encode_hello, encode_response, prove_moql_admission,
    AcceptedIdentityState, AcceptedStateAuthority, AcceptedSubjectKey, AdmissionHello,
    AdmissionResponse, MoqlAdmissionAuthenticator, MoqlAdmissionProof,
};

const ADMISSION_TIMEOUT: Duration = Duration::from_secs(5);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const CROSS_TENANT_GRACE: Duration = Duration::from_millis(500);

// ─────────────────────────────────────────────────────────────────────────────
// fixtures
// ─────────────────────────────────────────────────────────────────────────────

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

/// A closed-staging peer identity: an accepted `did:at9p` subject whose Ed25519
/// and ML-DSA-65 halves are both held by the test.
struct PeerFixture {
    did: String,
    ed: SigningKey,
    pq: MlDsaSigningKey,
}

fn peer(seed: u8, cid_tag: &str) -> PeerFixture {
    PeerFixture {
        did: format!("did:at9p:{cid_tag}"),
        ed: SigningKey::from_bytes(&[seed; 32]),
        pq: ml_dsa_sk_from_seed(&[seed.wrapping_add(101); 32]),
    }
}

fn proof(peer: &PeerFixture) -> MoqlAdmissionProof {
    MoqlAdmissionProof {
        did: peer.did.clone(),
        ed25519: peer.ed.clone(),
        ml_dsa_65: peer.pq.clone(),
    }
}

/// Project an accepted current state publishing `peer`'s key pair at `epoch`.
fn accepted_state(
    peer: &PeerFixture,
    epoch: u64,
    head_tag: u8,
    expires_at_unix_ms: Option<i64>,
) -> AcceptedIdentityState {
    AcceptedIdentityState {
        epoch,
        head_digest: [head_tag; 64],
        subject_keys: vec![AcceptedSubjectKey {
            ed25519: peer.ed.verifying_key().to_bytes(),
            ml_dsa_65: ml_dsa_sk_to_vk_bytes(&peer.pq),
        }],
        expires_at_unix_ms,
    }
}

/// The daemon-owned accepted-state/currentness authority, fixtured: a mutable
/// map the test can advance (rotation) or lapse (expiry) between admissions.
#[derive(Default)]
struct FixtureAuthority {
    states: Mutex<HashMap<String, AcceptedIdentityState>>,
}

impl FixtureAuthority {
    fn set(&self, did: &str, state: AcceptedIdentityState) {
        self.states.lock().insert(did.to_owned(), state);
    }
}

impl AcceptedStateAuthority for FixtureAuthority {
    fn accepted_state(&self, did: &str) -> Option<AcceptedIdentityState> {
        self.states.lock().get(did).cloned()
    }
}

struct AdmissionServer {
    substrate: IrohSubstrate,
    producer: OriginProducer,
    resolver_calls: Arc<AtomicUsize>,
}

/// Bind a `moql` server whose handler requires #1027 admission. `tenants` is
/// the operator-controlled subject→tenant provisioning map.
async fn admission_server(
    authority: Arc<FixtureAuthority>,
    tenants: HashMap<String, String>,
    timeout: Duration,
) -> Result<AdmissionServer> {
    let resolver_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&resolver_calls);
    let resolver = Arc::new(move |peer: &PeerIdentity| {
        calls.fetch_add(1, Ordering::SeqCst);
        peer.subject
            .as_deref()
            .and_then(|sub| tenants.get(sub).cloned())
    });
    let authenticator = Arc::new(
        MoqlAdmissionAuthenticator::new(authority, resolver).with_timeout(timeout),
    );
    let handler = IrohMoqProtocolHandler::new()
        .with_authz(MoqAuthzConfig::default().with_admission(authenticator));
    let producer = handler.origin_producer().clone();
    let substrate = IrohSubstrate::new(fresh_node_key(), handler, NoopHandler::new("rpc")).await?;
    Ok(AdmissionServer {
        substrate,
        producer,
        resolver_calls,
    })
}

/// Keeps a published broadcast + track announced for the lifetime of the
/// guard (dropping the producers would unannounce them).
struct BroadcastGuard {
    _broadcast: moq_net::BroadcastProducer,
    _track: moq_net::TrackProducer,
}

fn publish_frame(
    producer: &OriginProducer,
    broadcast: &str,
    payload: &'static [u8],
) -> Result<BroadcastGuard> {
    let mut b = producer
        .create_broadcast(broadcast)
        .ok_or_else(|| anyhow!("create_broadcast {broadcast} denied"))?;
    let mut track = b.create_track(Track::new("tokens"))?;
    let mut group = track.create_group(Group::from(0u64))?;
    group.write_frame(Bytes::from_static(payload))?;
    drop(group);
    Ok(BroadcastGuard {
        _broadcast: b,
        _track: track,
    })
}

/// Run the production client sequence: connect the `moql` carrier, prove
/// admission, then run the moq handshake. Returns the live moq session and the
/// client-side consumer over its own origin handle.
async fn admitted_client(
    server_addr: &EndpointAddr,
    proof: &MoqlAdmissionProof,
) -> Result<(IrohSubstrate, moq_net::Session, OriginConsumer)> {
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;
    let conn = client.connect(server_addr.clone(), ALPN_MOQ_LITE).await?;
    prove_moql_admission(&conn, proof, ADMISSION_TIMEOUT)
        .await
        .map_err(|e| anyhow!("admission rejected: {e}"))?;
    let session = Session::raw(conn);
    let client_origin: OriginProducer = Origin::random().produce();
    let consumer = client_origin.consume();
    let moq_client = Client::new().with_consume(client_origin);
    let moq_session = tokio::time::timeout(HANDSHAKE_TIMEOUT, moq_client.connect(session))
        .await
        .map_err(|_| anyhow!("moq handshake timed out"))?
        .map_err(|e| anyhow!("moq handshake rejected: {e}"))?;
    Ok((client, moq_session, consumer))
}

// ─────────────────────────────────────────────────────────────────────────────
// positive + cross-tenant
// ─────────────────────────────────────────────────────────────────────────────

/// Positive: a fresh proof against current accepted Ed25519 + ML-DSA-65 state
/// admits, yields the typed subject/tenant, and the admitted subscriber reads
/// its own tenant's frames. Cross-tenant: the same admitted peer cannot
/// enumerate or subscribe to another tenant's broadcasts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admitted_peer_streams_own_tenant_broadcast() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice = peer(1, "alicecid512");
    let bob = peer(2, "bobcid512");
    authority.set(&alice.did, accepted_state(&alice, 3, 9, None));
    authority.set(&bob.did, accepted_state(&bob, 1, 8, None));

    let tenants: HashMap<String, String> = [
        (alice.did.clone(), "alice".to_owned()),
        (bob.did.clone(), "bob".to_owned()),
    ]
    .into_iter()
    .collect();
    let server = admission_server(authority, tenants, ADMISSION_TIMEOUT).await?;
    let addr = direct_addr(&server.substrate);

    let _alice_broadcast = publish_frame(&server.producer, "alice/run-1", b"alice-tokens")?;
    let _bob_broadcast = publish_frame(&server.producer, "bob/run-9", b"bob-tokens")?;

    // ── alice admits and streams her own tenant's broadcast ─────────────────
    let (client, _moq_session, consumer) = admitted_client(&addr, &proof(&alice)).await?;
    assert!(
        server.resolver_calls.load(Ordering::SeqCst) >= 1,
        "admission must resolve the verified subject to a tenant"
    );
    let bc = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        consumer.announced_broadcast("alice/run-1"),
    )
    .await
    .map_err(|_| anyhow!("timed out waiting for own-tenant announce"))?
    .ok_or_else(|| anyhow!("admitted alice peer must see alice/run-1"))?;
    let mut track = bc.subscribe_track(&Track::new("tokens"))?;
    let mut group = tokio::time::timeout(HANDSHAKE_TIMEOUT, track.next_group())
        .await??
        .ok_or_else(|| anyhow!("next_group None"))?;
    let frame = tokio::time::timeout(HANDSHAKE_TIMEOUT, group.read_frame())
        .await??
        .ok_or_else(|| anyhow!("read_frame None"))?;
    assert_eq!(&frame[..], b"alice-tokens", "positive stream payload");

    // ── cross-tenant denial: bob's broadcast is invisible to alice ──────────
    let cross = tokio::time::timeout(
        CROSS_TENANT_GRACE,
        consumer.announced_broadcast("bob/run-9"),
    )
    .await;
    assert!(
        !matches!(cross, Ok(Some(_))),
        "admitted alice peer must not enumerate/subscribe bob's broadcasts"
    );

    client.shutdown().await?;
    server.substrate.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// replay
// ─────────────────────────────────────────────────────────────────────────────

/// A valid response captured from one admission is bound to that challenge's
/// server nonce; replaying it verbatim against a fresh challenge rejects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replayed_response_on_a_fresh_challenge_is_rejected() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice = peer(3, "replaycid512");
    authority.set(&alice.did, accepted_state(&alice, 5, 11, None));
    let tenants: HashMap<String, String> =
        [(alice.did.clone(), "alice".to_owned())].into_iter().collect();
    let server = admission_server(authority, tenants, ADMISSION_TIMEOUT).await?;
    let addr = direct_addr(&server.substrate);

    // ── capture a valid admission response over one connection ──────────────
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("c-moq"),
        NoopHandler::new("c-rpc"),
    )
    .await?;
    let proof = proof(&alice);
    let captured: Vec<u8> = {
        let conn = client.connect(addr.clone(), ALPN_MOQ_LITE).await?;
        let (mut send, mut recv) = conn.open_bi().await?;
        let hello = AdmissionHello {
            did: proof.did.clone(),
            ed25519_pub: proof.ed25519.verifying_key().to_bytes(),
            client_nonce: [0xAA; 32],
        };
        let hello_bytes = encode_hello(&hello);
        send.write_all(&(hello_bytes.len() as u32).to_be_bytes()).await?;
        send.write_all(&hello_bytes).await?;
        let mut len = [0u8; 4];
        recv.read_exact(&mut len).await?;
        let mut buf = vec![0u8; u32::from_be_bytes(len) as usize];
        recv.read_exact(&mut buf).await?;
        let challenge = decode_challenge(&buf)?;
        // Sign the valid response for THIS challenge, then capture the bytes.
        let t = admission_transcript(
            &hello.did,
            &hello.client_nonce,
            &challenge.server_nonce,
            challenge.epoch,
            &challenge.head_digest,
        );
        use ed25519_dalek::Signer as _;
        let ed_sig: [u8; 64] = proof.ed25519.sign(&t).to_bytes();
        let mut outer = t;
        outer.extend_from_slice(&ed_sig);
        let response = AdmissionResponse {
            ed_sig,
            pq_sig: hyprstream_rpc::crypto::pq::ml_dsa_sign(&proof.ml_dsa_65, &outer),
        };
        let response_bytes = encode_response(&response);
        send.write_all(&(response_bytes.len() as u32).to_be_bytes()).await?;
        send.write_all(&response_bytes).await?;
        send.finish()?;
        // The server admits this one (verdict arrives); close without moq.
        let mut vlen = [0u8; 4];
        recv.read_exact(&mut vlen).await?;
        let mut vbuf = vec![0u8; u32::from_be_bytes(vlen) as usize];
        recv.read_exact(&mut vbuf).await?;
        conn.close(0u32.into(), b"done");
        response_bytes
    };

    // ── replay the captured response verbatim against a FRESH challenge ─────
    let conn = client.connect(addr, ALPN_MOQ_LITE).await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    let hello = AdmissionHello {
        did: proof.did.clone(),
        ed25519_pub: proof.ed25519.verifying_key().to_bytes(),
        client_nonce: [0xAA; 32], // identical hello: only the server nonce differs
    };
    let hello_bytes = encode_hello(&hello);
    send.write_all(&(hello_bytes.len() as u32).to_be_bytes()).await?;
    send.write_all(&hello_bytes).await?;
    let mut len = [0u8; 4];
    recv.read_exact(&mut len).await?;
    let mut buf = vec![0u8; u32::from_be_bytes(len) as usize];
    recv.read_exact(&mut buf).await?;
    let fresh_challenge = decode_challenge(&buf)?;
    assert_ne!(
        fresh_challenge.server_nonce, [0; 32],
        "each challenge must carry a fresh server nonce"
    );
    send.write_all(&(captured.len() as u32).to_be_bytes()).await?;
    send.write_all(&captured).await?;
    send.finish()?;

    // Rejection: the server closes the connection instead of a verdict, and the
    // subsequent moq handshake cannot succeed.
    let session = Session::raw(conn);
    let client_origin = Origin::random().produce();
    let moq_client = Client::new().with_consume(client_origin);
    let handshake = tokio::time::timeout(HANDSHAKE_TIMEOUT, moq_client.connect(session)).await;
    assert!(
        matches!(handshake, Ok(Err(_)) | Err(_)),
        "a replayed admission response must be rejected, not admitted"
    );

    client.shutdown().await?;
    server.substrate.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// expiry
// ─────────────────────────────────────────────────────────────────────────────

/// An accepted state that has lapsed admits no one, even with otherwise-valid
/// keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_accepted_state_is_rejected() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice = peer(4, "expiredcid512");
    let now = hyprstream_rpc::envelope::current_timestamp();
    authority.set(&alice.did, accepted_state(&alice, 7, 12, Some(now - 1_000)));
    let tenants: HashMap<String, String> =
        [(alice.did.clone(), "alice".to_owned())].into_iter().collect();
    let server = admission_server(authority, tenants, ADMISSION_TIMEOUT).await?;
    let addr = direct_addr(&server.substrate);

    let result = admitted_client(&addr, &proof(&alice)).await;
    assert!(
        result.is_err(),
        "an expired accepted state must reject admission"
    );

    server.substrate.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// rotation / state advance
// ─────────────────────────────────────────────────────────────────────────────

/// Admission succeeds against the current accepted state; once the state
/// advances (new epoch, new head, new subject keys — a rotation), the
/// previously-admitted proof is rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn state_advance_invalidates_previous_keys() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice_v1 = peer(5, "rotatecid512");
    authority.set(&alice_v1.did, accepted_state(&alice_v1, 3, 9, None));
    let tenants: HashMap<String, String> = [(alice_v1.did.clone(), "alice".to_owned())]
        .into_iter()
        .collect();
    let server = admission_server(Arc::clone(&authority), tenants, ADMISSION_TIMEOUT).await?;
    let addr = direct_addr(&server.substrate);
    let _alice_broadcast = publish_frame(&server.producer, "alice/run-1", b"alice-tokens")?;

    // ── epoch 3: the proof admits ────────────────────────────────────────────
    let (client1, _s1, _c1) = admitted_client(&addr, &proof(&alice_v1)).await?;
    client1.shutdown().await?;

    // ── state advance: epoch 4 publishes a rotated key set ──────────────────
    let alice_v2 = peer(6, "rotatecid512");
    authority.set(&alice_v1.did, accepted_state(&alice_v2, 4, 10, None));

    // ── the previous proof is now rejected ───────────────────────────────────
    let result = admitted_client(&addr, &proof(&alice_v1)).await;
    assert!(
        result.is_err(),
        "a proof under rotated-out keys must be rejected after state advance"
    );

    // ── the rotated-in key admits ────────────────────────────────────────────
    let (client2, _s2, consumer2) = admitted_client(&addr, &proof(&alice_v2)).await?;
    let seen = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        consumer2.announced_broadcast("alice/run-1"),
    )
    .await;
    assert!(
        matches!(seen, Ok(Some(_))),
        "the current rotated key must admit and see its tenant scope"
    );
    client2.shutdown().await?;

    server.substrate.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeId-only identity
// ─────────────────────────────────────────────────────────────────────────────

/// With admission installed, a carrier that never proves (its NodeId is its
/// only "identity") is refused, and the tenant resolver is never consulted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nodeid_only_carrier_is_rejected() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice = peer(7, "nodeidcid512");
    authority.set(&alice.did, accepted_state(&alice, 1, 1, None));
    let tenants: HashMap<String, String> =
        [(alice.did.clone(), "alice".to_owned())].into_iter().collect();
    let server = admission_server(authority, tenants, Duration::from_secs(2)).await?;
    let addr = direct_addr(&server.substrate);

    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("c-moq"),
        NoopHandler::new("c-rpc"),
    )
    .await?;

    // No admission stream at all: straight to the moq handshake. The server is
    // waiting on the admission exchange, interprets the moq control stream as
    // a malformed admission frame (or times out), and closes — fail-closed.
    let conn = client.connect(addr, ALPN_MOQ_LITE).await?;
    let session = Session::raw(conn);
    let client_origin = Origin::random().produce();
    let moq_client = Client::new().with_consume(client_origin);
    let handshake = tokio::time::timeout(HANDSHAKE_TIMEOUT, moq_client.connect(session)).await;
    assert!(
        matches!(handshake, Ok(Err(_)) | Err(_)),
        "a NodeId-only carrier must not complete the moq handshake"
    );
    assert_eq!(
        server.resolver_calls.load(Ordering::SeqCst),
        0,
        "NodeId-only carrier must never reach the tenant resolver"
    );

    client.shutdown().await?;
    server.substrate.shutdown().await?;
    Ok(())
}

/// A hello whose "identity" is the carrier NodeId dressed as a `did:key` is
/// rejected: NodeId is reach evidence, never an accepted-state identity.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nodeid_dressed_as_did_key_is_rejected() -> Result<()> {
    let authority = Arc::new(FixtureAuthority::default());
    let alice = peer(8, "didkeycid512");
    authority.set(&alice.did, accepted_state(&alice, 1, 1, None));
    let tenants: HashMap<String, String> =
        [(alice.did.clone(), "alice".to_owned())].into_iter().collect();
    let server = admission_server(authority, tenants, Duration::from_secs(2)).await?;
    let addr = direct_addr(&server.substrate);

    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("c-moq"),
        NoopHandler::new("c-rpc"),
    )
    .await?;
    let node_id = *client.endpoint_id().as_bytes();
    let conn = client.connect(addr, ALPN_MOQ_LITE).await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    let hello = AdmissionHello {
        // The carrier NodeId as a did:key: not a did:at9p accepted identity.
        did: hyprstream_rpc::did_key::ed25519_to_did_key(&node_id),
        ed25519_pub: node_id,
        client_nonce: [0xEE; 32],
    };
    let hello_bytes = encode_hello(&hello);
    send.write_all(&(hello_bytes.len() as u32).to_be_bytes()).await?;
    send.write_all(&hello_bytes).await?;
    send.finish()?;

    // The server rejects before any challenge: no challenge frame may arrive,
    // and the moq handshake cannot run on the closed connection.
    let mut len = [0u8; 4];
    let challenge = tokio::time::timeout(HANDSHAKE_TIMEOUT, recv.read_exact(&mut len)).await;
    assert!(
        !matches!(challenge, Ok(Ok(_))),
        "a NodeId-as-did:key hello must be rejected before any challenge is issued"
    );
    let session = Session::raw(conn);
    let client_origin = Origin::random().produce();
    let moq_client = Client::new().with_consume(client_origin);
    let handshake = tokio::time::timeout(HANDSHAKE_TIMEOUT, moq_client.connect(session)).await;
    assert!(
        matches!(handshake, Ok(Err(_)) | Err(_)),
        "a NodeId presented as identity must not complete the moq handshake"
    );
    assert_eq!(
        server.resolver_calls.load(Ordering::SeqCst),
        0,
        "NodeId-as-identity must never reach the tenant resolver"
    );

    client.shutdown().await?;
    server.substrate.shutdown().await?;
    Ok(())
}
