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
    connect_moq_reach_for_qos, run_relay_announce_link, MoqStreamOrigin, ProducerReachConfig,
    RelayChoice, NodeStreamReach, STREAM_TRACK,
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
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

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
    // Destination `ProducerReachConfig::reach()` emits when a relay is configured).
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

    // No fixed sleep: the `timeout(announced_broadcast)` below is the deterministic
    // wait — it resolves once the relay origin actually has the announced broadcast,
    // so a wall-clock delay here would only add latency / flake (matches the
    // `relay_choice_only_anonymizes_stream_end_to_end` pattern).

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
    // The producer (`from_dh` ctx) seals each payload under the DH `enc_key` (#321),
    // so the verifier must open it with the SAME key — otherwise `Tagged` frames pass
    // through undecrypted and never become `Data`/`Complete`. Mirrors the production
    // consumers (stream_consumer.rs + moq_stream.rs all call `.with_enc_key`).
    let mut verifier = StreamVerifier::new(mac_key, topic.clone())
        .with_enc_key(*ctx.enc_key().expect("from_dh ctx derives the AEAD enc_key (#321)"));
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

/// End-to-end anonymization (#384): a producer that HAS a direct reach but builds
/// its stream with [`RelayChoice::Only`] advertises a relay-ONLY reach list, and
/// the stream still delivers through the relay. This proves the per-stream relay
/// override actually anonymizes — the direct path exists in the node's config yet
/// is deliberately withheld from the wire for this stream, so a subscriber CANNOT
/// learn or dial the producer's direct address.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn relay_choice_only_anonymizes_stream_end_to_end() -> Result<()> {
    hyprstream_rpc::transport::install_pq_crypto_provider().expect("install PQ provider");

    // ── RELAY: bidirectional /moq endpoint, no keys (same as rendezvous) ───────
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
    .with_moq_relay(relay_origin.producer().clone());
    let relay_shutdown = relay_rpc.shutdown_token();
    let relay_task = tokio::spawn(relay_rpc.run());

    let relay_pin = cert_sha256(&relay_cert);
    let relay_reach = ReachTransport::Quic(QuicReach {
        addr: relay_addr.to_string(),
        server_name: "localhost".to_owned(),
        cert_hashes: vec![relay_pin.to_vec()],
    });

    // ── The node's per-server reach config: a REAL direct Quic reach is present
    //    alongside the relay. A naive build would advertise the direct address. ──
    // The iroh-direct reach is sourced from `ProducerReachConfig.iroh_node_id`
    // (the live field `reach_with_relay` reads) — NOT from `NodeStreamReach`,
    // which carries no node id (#384: that dead field was removed).
    let iroh_node_id = [0x11u8; 32];
    let direct_addr: std::net::SocketAddr = "203.0.113.7:443".parse()?; // TEST-NET-3, unreachable
    let server_cfg = ProducerReachConfig {
        iroh_node_id: Some(iroh_node_id),
        quic_reach: Some(NodeStreamReach {
            addr: direct_addr,
            server_name: "producer.invalid".to_owned(),
            cert_hashes: vec![[0x22u8; 32]],
        }),
        relay: Some(relay_reach.clone()),
    };

    // ── ASSERT the anonymization property at the reach-construction layer ───────
    // ServerDefault would leak the direct reaches; Only(relay) must withhold them.
    let default_reach = server_cfg.reach_with_relay(RelayChoice::ServerDefault);
    assert!(
        default_reach.iter().any(|d| d.role == Role::Direct),
        "ServerDefault must advertise the configured direct reach(es)"
    );
    // Prove the iroh-direct reach comes from the LIVE source
    // (`ProducerReachConfig.iroh_node_id`): the advertised iroh Destination must
    // carry exactly the configured node id. Before #384 a redundant
    // `NodeStreamReach.iroh_node_id` masked which field was actually read.
    assert!(
        default_reach.iter().any(|d| matches!(
            &d.transport,
            ReachTransport::Iroh(r) if r.node_id == iroh_node_id
        )),
        "the iroh-direct reach must be built from ProducerReachConfig.iroh_node_id: {default_reach:?}"
    );
    let only_reach = server_cfg.reach_with_relay(RelayChoice::Only(relay_reach.clone()));
    assert!(
        only_reach.iter().all(|d| d.role == Role::Relay),
        "RelayChoice::Only must omit ALL direct reaches (anonymized): {only_reach:?}"
    );
    assert_eq!(only_reach.len(), 1, "Only advertises exactly the one relay reach");
    assert!(
        only_reach[0].transport == relay_reach,
        "the sole advertised reach is the relay"
    );

    // ── Heterogeneity in one process: a sibling stream stays direct ────────────
    // (Proves two streams on the SAME node/config can differ — the property the
    // process-global singleton made impossible.)
    let sibling_reach = server_cfg.reach_with_relay(RelayChoice::NoRelay);
    assert!(
        sibling_reach.iter().all(|d| d.role == Role::Direct),
        "a sibling RelayChoice::NoRelay stream stays direct-only while Only is anonymized"
    );

    // ── Now drive ACTUAL delivery using ONLY the advertised (relay-only) reach ──
    let (_client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
    let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
    let topic = ctx.topic().to_owned();
    let mac_key = *ctx.mac_key();

    // Producer is a LOCAL origin, never network-bound — reachable ONLY via relay.
    let producer_origin = {
        let p = moq_net::Origin::random().produce();
        let c = p.consume();
        MoqStreamOrigin::from_pair(p, c)
    };
    let broadcast_path = producer_origin.broadcast_path(&topic);
    let mut publisher = producer_origin.publisher(&ctx)?;

    let link_relay = relay_reach.clone();
    let link_producer = producer_origin.producer().clone();
    let link_task = tokio::spawn(async move {
        let _ = run_relay_announce_link(&link_producer, &link_relay).await;
    });
    // Wait for the announce link to propagate the producer's broadcast UP to the
    // relay before the subscriber dials, instead of a fixed sleep that can flake
    // on slow CI. `announced_broadcast` resolves once the relay origin has the
    // broadcast announced.
    tokio::time::timeout(
        Duration::from_secs(10),
        relay_origin.consumer().announced_broadcast(&broadcast_path),
    )
    .await
    .map_err(|_| anyhow!("timeout: producer broadcast not announced to relay"))?
    .ok_or_else(|| anyhow!("relay origin closed before broadcast was announced"))?;

    // The subscriber dials EXACTLY the anonymized reach the producer advertised —
    // the relay-only list. It has no direct address to fall back to.
    let qos = StreamOpt::default();
    let conn = connect_moq_reach_for_qos(&only_reach, &qos)
        .await
        .map_err(|e| anyhow!("subscriber failed to connect via anonymized relay reach: {e}"))?;

    let bc = tokio::time::timeout(
        Duration::from_secs(5),
        conn.consumer.announced_broadcast(&broadcast_path),
    )
    .await
    .map_err(|_| anyhow!("timeout: broadcast not announced via relay"))?
    .ok_or_else(|| anyhow!("broadcast not announced via relay"))?;
    let track = bc.subscribe_track(&moq_net::Track::new(STREAM_TRACK))?;

    // The producer (`from_dh` ctx) seals each payload under the DH `enc_key` (#321),
    // so the verifier must open it with the SAME key — otherwise `Tagged` frames pass
    // through undecrypted and never become `Data`/`Complete`. Mirrors the production
    // consumers (stream_consumer.rs + moq_stream.rs all call `.with_enc_key`).
    let mut verifier = StreamVerifier::new(mac_key, topic.clone())
        .with_enc_key(*ctx.enc_key().expect("from_dh ctx derives the AEAD enc_key (#321)"));
    // Publish FIRST, then read the group (the moq group exists once published) —
    // matches the `next_frame` ordering in `moq_relay_rendezvous`.
    publisher.publish_data(b"anon").await?;
    let mut group = tokio::time::timeout(Duration::from_secs(10), track.get_group(0))
        .await
        .map_err(|_| anyhow!("timed out waiting for relayed group 0"))??
        .ok_or_else(|| anyhow!("relay track ended early"))?;
    let frame = tokio::time::timeout(Duration::from_secs(10), group.read_frame())
        .await
        .map_err(|_| anyhow!("timed out reading relayed frame"))??
        .ok_or_else(|| anyhow!("relayed group 0 had no frame"))?;
    let got = hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &frame)?;
    assert!(
        matches!(got.first(), Some(StreamPayload::Data(d)) if d == b"anon"),
        "anonymized stream must deliver through the relay: {got:?}"
    );

    relay_shutdown.cancel();
    link_task.abort();
    let _ = relay_task.await;
    drop(producer_origin);
    Ok(())
}
