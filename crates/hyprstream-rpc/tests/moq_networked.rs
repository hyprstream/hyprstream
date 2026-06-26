//! Networked moq subscription over `web_transport_quinn` (#274).
//!
//! End-to-end proof of the issue-#274 streaming path: a daemon serves the moq
//! plane on the same `web_transport_quinn` endpoint as RPC (path-dispatched on
//! `/moq`); a client resolves a signed `StreamInfo.reach`, dials `/moq` via
//! [`dial_stream`], subscribes to the broadcast, and receives MAC-verified
//! frames — the same chained-HMAC envelope the in-process path produces.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use futures::StreamExt;
use rand::RngCore;

use hyprstream_rpc::moq_stream::MoqStreamOrigin;
use hyprstream_rpc::stream_info::{QuicReach, Destination, Role, TransportConfig};
use hyprstream_rpc::streaming::{StreamContext, StreamPayload};
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

/// A client resolves `StreamInfo.reach`, dials `/moq` over `web_transport_quinn`
/// via `dial_stream`, subscribes, and receives MAC-verified frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn networked_moq_subscribe_receives_mac_verified_frames() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Server-side moq origin (the shared broadcast tree external subscribers see).
    let producer = moq_net::Origin::random().produce();
    let consumer = producer.consume();
    let origin = MoqStreamOrigin::from_pair(producer, consumer);

    // Derive the stream's DH-bound topic + MAC key; the publisher and the client
    // verifier share them exactly (the client gets them via DH in production).
    let (_client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
    let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
    let topic = ctx.topic().to_owned();
    let mac_key = *ctx.mac_key();
    // #321: the DH path is AEAD-on; the consumer shares the same enc_key.
    let enc_key = *ctx.enc_key().expect("DH StreamContext has an AEAD enc_key");
    let broadcast_path = origin.broadcast_path(&topic);

    // Stand up the daemon's RPC+moq endpoint: RPC core on default path, moq on
    // `/moq` (path-dispatch), publishing the shared origin consumer (#274).
    let (server, addr, cert_der) = build_server()?;
    let processor =
        hyprstream_rpc::transport::rpc_session::from_fn(|req: bytes::Bytes| async move { Ok(req) });
    let rpc_server = QuinnRpcServer::with_capacity(
        server,
        Arc::new(processor),
        fresh_signing_key(),
        hyprstream_rpc::transport::rpc_session::DEFAULT_STREAM_LIMIT,
    )
    .with_moq_consumer(origin.consumer().clone());
    let shutdown = rpc_server.shutdown_token();
    let server_task = tokio::spawn(rpc_server.run());

    // Build the signed-StreamInfo reach the client resolves: a Quic option
    // pointing at the bound endpoint, pinned by the server's leaf-cert SHA-256.
    let pin = cert_sha256(&cert_der);
    let reach = vec![Destination {
        role: Role::Direct,
        transport: TransportConfig::Quic(QuicReach {
            addr: addr.to_string(),
            server_name: "localhost".to_owned(),
            cert_hashes: vec![pin.to_vec()],
        }),
    }];

    // Announce the broadcast first (creates the moq broadcast/track), then start
    // the networked subscriber so it can attach before any group is published.
    let mut pub_ = origin.publisher(&ctx)?;

    // Client: networked subscribe — dials `/moq` via dial_stream, subscribes,
    // verifies the chained HMAC, and yields decoded payloads.
    let qos = hyprstream_rpc::stream_info::StreamOpt::default();
    let mut handle = hyprstream_rpc::moq_stream::MoqStreamHandle::networked(
        reach, &qos, broadcast_path, mac_key, enc_key, topic,
    );

    // Give the background task time to dial, handshake, and subscribe to the
    // track before the producer pushes group 0 — a late joiner that misses the
    // first group cannot verify the chained HMAC. Publish one group at a time and
    // drain it before the next so the subscriber observes the chain in order.
    tokio::time::sleep(Duration::from_secs(1)).await;

    async fn next_payload(
        handle: &mut hyprstream_rpc::moq_stream::MoqStreamHandle,
    ) -> Result<StreamPayload> {
        tokio::time::timeout(Duration::from_secs(10), handle.next())
            .await
            .map_err(|_| anyhow!("timed out waiting for moq frame"))?
            .ok_or_else(|| anyhow!("stream ended early"))?
    }

    let mut got: Vec<StreamPayload> = Vec::new();
    pub_.publish_data(b"alpha").await?;
    got.push(next_payload(&mut handle).await?);
    pub_.publish_data(b"beta").await?;
    got.push(next_payload(&mut handle).await?);
    pub_.complete_ref(b"{}").await?;
    got.push(next_payload(&mut handle).await?);

    assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"alpha"), "first frame, got {:?}", got[0]);
    assert!(matches!(&got[1], StreamPayload::Data(d) if d == b"beta"), "second frame, got {:?}", got[1]);
    assert!(matches!(&got[2], StreamPayload::Complete(_)), "terminal frame, got {:?}", got[2]);

    handle.cancel();
    shutdown.cancel();
    let _ = server_task.await;
    Ok(())
}
