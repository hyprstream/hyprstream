//! moq relay rendezvous (#358).
//!
//! Restores the rendezvous property the ZMQ→moq migration removed: NEITHER the
//! publisher NOR the subscriber dials the other — both rendezvous through a
//! producer-advertised moq **relay**.
//!
//! Topology under test:
//!
//! ```text
//!   producer  ──link UP (with_origin)──▶  RELAY  ◀──dial (Role::Relay reach)──  subscriber
//!             announces broadcastPath           re-serves broadcastPath by track name
//! ```
//!
//! The relay is a bidirectional `web_transport_quinn` `/moq` endpoint running in
//! relay mode (`QuinnRpcServer::with_moq_relay`): it ingests the producer's
//! announced broadcast and re-serves it to the subscriber by the SAME
//! `broadcastPath`. The relay holds NO stream keys — it forwards the chained-HMAC
//! frames opaquely. The test also asserts the relay-forwarded bytes do NOT verify
//! under a wrong key (relay-blind: content integrity is keyed at the source, not
//! relay-trusted).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use rand::RngCore;

use hyprstream_rpc::moq_stream::{
    connect_moq_reach_for_qos, run_relay_announce_link, MoqStreamOrigin, STREAM_TRACK,
};
use hyprstream_rpc::stream_info::{
    Destination, QuicReach, Role, StreamOpt, TransportConfig as ReachTransport,
};
use hyprstream_rpc::streaming::{StreamContext, StreamPayload, StreamVerifier};
use hyprstream_rpc::transport::quinn_transport::{cert_sha256, QuinnRpcServer};

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

/// End-to-end: a producer links its origin UP to a relay; a subscriber dials the
/// relay's `Role::Relay` reach and receives MAC-verified frames — neither side
/// ever learns the other's address.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn moq_relay_rendezvous() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    // ── RELAY: a bidirectional /moq endpoint (ingest + re-serve), no keys ──────
    // Its origin starts EMPTY — the broadcast exists only on the producer until
    // the producer links up and announces it. This proves the relay re-serves an
    // INGESTED broadcast, not one it was seeded with.
    let relay_producer = moq_net::Origin::random().produce();
    let relay_consumer = relay_producer.consume();
    let relay_origin = MoqStreamOrigin::from_pair(relay_producer.clone(), relay_consumer);

    let (relay_server, relay_addr, relay_cert) = build_server()?;
    let relay_processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    let relay_rpc = QuinnRpcServer::with_capacity(
        relay_server,
        Arc::new(relay_processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    // Relay (bidirectional) mode: ingest the producer's broadcast and re-serve it.
    .with_moq_relay(relay_origin.producer().clone());
    let relay_shutdown = relay_rpc.shutdown_token();
    let relay_task = tokio::spawn(relay_rpc.run());

    // The producer-advertised relay reach (this is exactly the `Role::Relay`
    // Destination `producer_reach()` emits when a relay is configured).
    let relay_pin = cert_sha256(&relay_cert);
    let relay_reach = ReachTransport::Quic(QuicReach {
        addr: relay_addr.to_string(),
        server_name: "localhost".to_owned(),
        cert_hashes: vec![relay_pin.to_vec()],
    });

    // ── Stream identity (DH-derived topic + MAC key, shared producer/consumer) ──
    let (_client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
    let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
    let topic = ctx.topic().to_owned();
    let mac_key = *ctx.mac_key();

    // ── PRODUCER: a LOCAL origin, never bound to a network endpoint ────────────
    // The producer is NOT directly reachable (no QuinnRpcServer of its own). Its
    // only path to a subscriber is the relay link.
    let producer_origin = {
        let p = moq_net::Origin::random().produce();
        let c = p.consume();
        MoqStreamOrigin::from_pair(p, c)
    };
    let broadcast_path = producer_origin.broadcast_path(&topic);
    let mut publisher = producer_origin.publisher(&ctx)?;

    // Link the producer's origin UP to the relay (announce broadcasts UP). Drive
    // one connect cycle in the background; it returns when the session closes.
    let link_relay = relay_reach.clone();
    let link_producer = producer_origin.producer().clone();
    let link_task = tokio::spawn(async move {
        let _ = run_relay_announce_link(&link_producer, &link_relay).await;
    });

    // Give the producer→relay link time to handshake + announce the broadcast UP.
    tokio::time::sleep(Duration::from_secs(1)).await;

    // ── SUBSCRIBER: dials ONLY the relay's Role::Relay reach ───────────────────
    // The reach list carries the relay ONLY (no direct producer reach) — a
    // relay-only / anonymized advertisement. The subscriber cannot dial the
    // producer because it was never told where the producer is.
    let reach = vec![Destination { role: Role::Relay, transport: relay_reach.clone() }];

    // Default qos = Retention::Live; with a relay-only reach there is nothing to
    // promote — the relay is the only dialable option, exercising rendezvous.
    let qos = StreamOpt::default();
    let conn = connect_moq_reach_for_qos(&reach, &qos)
        .await
        .map_err(|e| anyhow!("subscriber failed to connect via relay: {e}"))?;

    // Subscribe to the SAME broadcastPath the producer announced — the relay
    // forwards by track name; the path is identical for direct and relay.
    let bc = tokio::time::timeout(
        Duration::from_secs(5),
        conn.consumer.announced_broadcast(&broadcast_path),
    )
    .await
    .map_err(|_| anyhow!("timeout: broadcast not announced via relay"))?
    .ok_or_else(|| anyhow!("broadcast not announced via relay"))?;
    let mut track = bc.subscribe_track(&moq_net::Track::new(STREAM_TRACK))?;

    // ── Publish through the relay; verify the chained-HMAC end-to-end ──────────
    let mut verifier = StreamVerifier::new(mac_key, topic.clone());
    let mut got: Vec<StreamPayload> = Vec::new();

    // Publish one group at a time and drain it so the ordered chain is observed.
    async fn next_frame(
        track: &mut moq_net::TrackConsumer,
        seq: u64,
    ) -> Result<bytes::Bytes> {
        let mut group = tokio::time::timeout(Duration::from_secs(10), track.get_group(seq))
            .await
            .map_err(|_| anyhow!("timed out waiting for relayed group {seq}"))??
            .ok_or_else(|| anyhow!("relay track ended early at group {seq}"))?;
        tokio::time::timeout(Duration::from_secs(10), group.read_frame())
            .await
            .map_err(|_| anyhow!("timed out reading relayed frame {seq}"))??
            .ok_or_else(|| anyhow!("relayed group {seq} had no frame"))
    }

    publisher.publish_data(b"alpha").await?;
    let f0 = next_frame(&mut track, 0).await?;
    let raw_relayed: Vec<u8> = f0.to_vec(); // capture a relayed frame for the blind-relay assertion.
    got.extend(hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &f0)?);

    publisher.publish_data(b"beta").await?;
    let f1 = next_frame(&mut track, 1).await?;
    got.extend(hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &f1)?);

    publisher.complete_ref(b"{}").await?;
    let f2 = next_frame(&mut track, 2).await?;
    got.extend(hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &f2)?);

    assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"alpha"), "got {:?}", got[0]);
    assert!(matches!(&got[1], StreamPayload::Data(d) if d == b"beta"), "got {:?}", got[1]);
    assert!(matches!(&got[2], StreamPayload::Complete(_)), "got {:?}", got[2]);

    // ── Relay-blind: the relay-forwarded bytes do NOT verify without the key ───
    // The relay holds no `mac_key` (and no `enc_key`); content integrity is keyed
    // at the source. A verifier seeded with the WRONG key must reject the exact
    // bytes the relay forwarded — proving the relay cannot read or forge content,
    // only carry opaque frames. (The AEAD `enc_key` seal is the same posture: the
    // relay never derives or holds it.)
    assert!(!raw_relayed.is_empty(), "expected to capture a relayed frame");
    let wrong_key = [0xFFu8; 32];
    let mut blind_verifier = StreamVerifier::new(wrong_key, topic.clone());
    let blind = hyprstream_rpc::moq_stream::verify_moq_frame(&mut blind_verifier, &topic, &raw_relayed);
    assert!(
        blind.is_err(),
        "relay-forwarded bytes must NOT verify without the source key (relay-blind); \
         a wrong-key verifier accepted them, which would mean the relay was trusted for content"
    );

    // Teardown.
    relay_shutdown.cancel();
    link_task.abort();
    let _ = relay_task.await;
    drop(producer_origin);
    Ok(())
}
