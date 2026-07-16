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

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};
use tokio::sync::Mutex;

use hyprstream_rpc::capnp::FromCapnp;
use hyprstream_rpc::crypto::hybrid_kem::KeyedKemTrustStore;
use hyprstream_rpc::envelope::{InMemoryNonceCache, KeyedPqTrustStore, SignedEnvelope};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, NoopHandler, ALPN_HYPRSTREAM_RPC};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::transport_traits::Transport;
use hyprstream_rpc::ToCapnp;
use iroh::{EndpointAddr, TransportAddr};
use rand::RngCore;

/// Minimal `RequestService` for the canary: echoes the request payload with a
/// 1-byte prefix so the test can verify identity round-trip.
struct EchoService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    invocations: Option<Arc<AtomicUsize>>,
}

impl EchoService {
    fn new(signing_key: SigningKey) -> Self {
        Self {
            name: "iroh-echo".to_owned(),
            transport: TransportConfig::inproc("iroh-echo-unused"),
            signing_key,
            invocations: None,
        }
    }

    fn with_invocations(mut self, invocations: Arc<AtomicUsize>) -> Self {
        self.invocations = Some(invocations);
        self
    }
}

#[async_trait(?Send)]
impl RequestService for EchoService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        if let Some(invocations) = &self.invocations {
            invocations.fetch_add(1, Ordering::SeqCst);
        }
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

fn request_client_signing_key() -> &'static SigningKey {
    static KEY: OnceLock<SigningKey> = OnceLock::new();
    KEY.get_or_init(fresh_signing_key)
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

const SILENT_DROP_DEADLINE: Duration = Duration::from_secs(1);

/// Assert that a network admission drop completes promptly with an empty FIN.
///
/// Real iroh reproduction of every intended drop in this suite yields
/// `Ok(empty)`. No reset/closed error is allowlisted because none is produced
/// by this path. In particular, the transport's own timeout and every unrelated
/// error are failures rather than alternate evidence of a silent drop.
async fn assert_silent_drop<F>(operation: F, message: &str)
where
    F: Future<Output = Result<Vec<u8>>>,
{
    match tokio::time::timeout(SILENT_DROP_DEADLINE, operation).await {
        Err(_) => panic!(
            "{message}: outer deadline elapsed after {SILENT_DROP_DEADLINE:?}; \
             timeout is not evidence of peer close"
        ),
        Ok(Ok(bytes)) => assert!(
            bytes.is_empty(),
            "{message}: unexpected response bytes: {}",
            bytes.len()
        ),
        Ok(Err(error)) => panic!(
            "{message}: unexpected transport error (only prompt Ok(empty) is accepted): {error:#}"
        ),
    }
}

#[tokio::test]
async fn silent_drop_oracle_rejects_nonresponsive_peer_control() {
    let started = Instant::now();
    let assertion = tokio::spawn(assert_silent_drop(
        std::future::pending::<Result<Vec<u8>>>(),
        "nonresponsive peer control",
    ));

    match tokio::time::timeout(SILENT_DROP_DEADLINE * 2, assertion).await {
        Err(_) => panic!("oracle control itself hung"),
        Ok(Ok(())) => panic!("a nonresponsive peer passed the silent-drop assertion"),
        Ok(Err(join_error)) => assert!(join_error.is_panic()),
    }
    assert!(
        started.elapsed() < SILENT_DROP_DEADLINE * 2,
        "timeout control must fail promptly"
    );
}

struct RecordingTransport<T> {
    inner: T,
    last_sent: Arc<Mutex<Option<Vec<u8>>>>,
}

impl<T> RecordingTransport<T> {
    fn new(inner: T) -> (Self, Arc<Mutex<Option<Vec<u8>>>>) {
        let last_sent = Arc::new(Mutex::new(None));
        (
            Self {
                inner,
                last_sent: Arc::clone(&last_sent),
            },
            last_sent,
        )
    }
}

#[async_trait]
impl<T> Transport for RecordingTransport<T>
where
    T: Transport + Send + Sync,
    T::Sub: Send,
    T::Pub: Send,
{
    type Sub = T::Sub;
    type Pub = T::Pub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        *self.last_sent.lock().await = Some(payload.clone());
        self.inner.send(payload, timeout_ms).await
    }

    fn forbids_cleartext_envelope(&self) -> bool {
        self.inner.forbids_cleartext_envelope()
    }

    async fn subscribe(&self, topic: &[u8]) -> Result<Self::Sub> {
        self.inner.subscribe(topic).await
    }

    async fn publish(&self, topic: &[u8]) -> Result<Self::Pub> {
        self.inner.publish(topic).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hykem_encrypted_envelope_round_trip_over_iroh() -> Result<()> {
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
    let client_signing = request_client_signing_key().clone();
    let client_verifying = client_signing.verifying_key();
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;

    let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;
    let (transport, last_sent) = RecordingTransport::new(IrohTransport::new(conn));

    // Request verification: server anchors the client's mesh ML-DSA key.
    let client_pq_sk = derive_mesh_mldsa_key(&client_signing);
    let client_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&client_pq_sk),
    )?;
    let mut request_pq_store = KeyedPqTrustStore::new();
    request_pq_store.bind(client_verifying.to_bytes(), &client_pq_vk);
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(request_pq_store)),
        },
    );

    // Request encryption: client anchors the server's #mesh-kem keyAgreement key.
    let mut request_kem_store = KeyedKemTrustStore::new();
    request_kem_store.bind(
        server_verifying.to_bytes(),
        derive_mesh_kem_recipient(&server_signing)?.public(),
    );

    // Response verification: client anchors the server's mesh ML-DSA key.
    let server_pq_sk = derive_mesh_mldsa_key(&server_signing);
    let server_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&server_pq_sk),
    )?;
    let mut response_pq_store = KeyedPqTrustStore::new();
    response_pq_store.bind(server_verifying.to_bytes(), &server_pq_vk);

    let rpc = RpcClientImpl::new(
        LocalSigner::new(client_signing.clone()),
        transport,
        Some(server_verifying),
    )
    .with_request_kem_store(Arc::new(request_kem_store))
    .with_response_pq_store(Arc::new(response_pq_store));

    // Real encrypted + hybrid-signed envelope round-trip over iroh.
    let response = rpc.call(b"ping-payload".to_vec()).await?;
    assert_eq!(&response[..], b"\xECping-payload");
    let sent = last_sent
        .lock()
        .await
        .clone()
        .ok_or_else(|| anyhow::anyhow!("request bytes not recorded"))?;
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(&sent),
        capnp::message::ReaderOptions::new(),
    )?;
    let signed = SignedEnvelope::read_from(
        reader.get_root::<hyprstream_rpc::common_capnp::signed_envelope::Reader>()?,
    )?;
    assert!(
        signed.is_encrypted(),
        "iroh request must carry encrypted_envelope"
    );
    assert!(
        signed.payload().is_empty(),
        "outer request payload must be redacted when encrypted_envelope is present"
    );
    assert!(
        !sent
            .windows(b"ping-payload".len())
            .any(|w| w == b"ping-payload"),
        "clear request payload must not appear in the serialized SignedEnvelope"
    );

    // Second call on the same connection to exercise multiple bidi streams.
    let response2 = rpc.call(b"second".to_vec()).await?;
    assert_eq!(&response2[..], b"\xECsecond");

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// INV-2 receive-side (#1042) over a real iroh carrier: a validly hybrid-signed
/// **cleartext** envelope, sent raw over the wire (bypassing the client's
/// send-side guard), is authenticated/replay-checked and then silently dropped
/// before the echo handler runs. No attacker-triggered signed error is emitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleartext_envelope_rejected_on_iroh_receive() -> Result<()> {
    use hyprstream_rpc::envelope::{
        current_timestamp, generate_nonce, Authorization, RequestEnvelope,
    };

    // ─── Server side ──────────────────────────────────────────────────────
    let server_signing = fresh_signing_key();
    let server_nonce_cache = Arc::new(InMemoryNonceCache::new());
    let invocations = Arc::new(AtomicUsize::new(0));
    let bridge = LocalServiceBridge::spawn(
        EchoService::new(server_signing.clone()).with_invocations(Arc::clone(&invocations)),
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

    // ─── Client side: send a CLEARTEXT signed envelope raw ────────────────
    let client_signing = request_client_signing_key().clone();
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;
    let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;
    let transport = IrohTransport::new(conn);

    // Anchor the client's ML-DSA key so the envelope is a *validly* hybrid-signed
    // cleartext one — the rejection must be about the carrier, not the signature.
    let client_pq_sk = derive_mesh_mldsa_key(&client_signing);
    let client_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&client_pq_sk),
    )?;
    let mut request_pq_store = KeyedPqTrustStore::new();
    request_pq_store.bind(client_signing.verifying_key().to_bytes(), &client_pq_vk);
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(request_pq_store)),
        },
    );

    let envelope = RequestEnvelope {
        request_id: 99,
        payload: b"ping-payload".to_vec(),
        iat: current_timestamp(),
        nonce: generate_nonce(),
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
    };
    let signed = SignedEnvelope::new_signed_hybrid(envelope, &client_signing, &client_pq_sk);
    let mut wire = Vec::new();
    {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder =
                message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        capnp::serialize::write_message(&mut wire, &message)?;
    }
    assert!(
        !signed.is_encrypted(),
        "test must send a cleartext envelope"
    );

    // Raw send bypasses RpcClientImpl's send-side guard — this exercises the
    // *receive* side directly, the scenario a hostile/downgrading peer creates.
    assert_silent_drop(
        transport.send(wire, Some(8_000)),
        "cleartext must be reset/dropped without a response",
    )
    .await;
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "cleartext must never reach the application handler"
    );

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// A non-empty `encryptedEnvelope` marker is not proof of encryption. Even
/// when the same wire message contains a clear outer payload, the sealed raw
/// processor extension point must be refused on a real iroh carrier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn false_encrypted_marker_never_reaches_custom_processor_over_iroh() -> Result<()> {
    use bytes::Bytes;
    use hyprstream_rpc::envelope::{
        current_timestamp, generate_nonce, Authorization, RequestEnvelope,
    };

    let invocations = Arc::new(AtomicUsize::new(0));
    let invoked = Arc::clone(&invocations);
    let processor = hyprstream_rpc::transport::rpc_session::from_fn(move |_request: Bytes| {
        let invoked = Arc::clone(&invoked);
        async move {
            invoked.fetch_add(1, Ordering::SeqCst);
            Ok(Bytes::from_static(b"must-not-run"))
        }
    });
    let server = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(processor, fresh_signing_key()),
    )
    .await?;
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;
    let conn = client
        .connect(direct_addr(&server), ALPN_HYPRSTREAM_RPC)
        .await?;
    let transport = IrohTransport::new(conn);

    let signer = fresh_signing_key();
    let signed = SignedEnvelope::new_signed(
        RequestEnvelope {
            request_id: 31337,
            payload: b"visible-cleartext-sentinel".to_vec(),
            iat: current_timestamp(),
            nonce: generate_nonce(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
        },
        &signer,
    );
    let mut wire = Vec::new();
    {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder =
                message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
            builder.set_encrypted_envelope(&[0xA5]);
        }
        capnp::serialize::write_message(&mut wire, &message)?;
    }
    assert!(
        wire.windows(b"visible-cleartext-sentinel".len())
            .any(|w| w == b"visible-cleartext-sentinel"),
        "adversarial frame must actually contain the clear outer payload"
    );

    assert_silent_drop(
        transport.send(wire, Some(5_000)),
        "false marker must be silently reset/dropped",
    )
    .await;
    assert_eq!(invocations.load(Ordering::SeqCst), 0);

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// Repeated undecodable unauthenticated requests produce no signed response
/// and never invoke the application handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_garbage_is_silently_dropped_without_handler_work() -> Result<()> {
    let server_signing = fresh_signing_key();
    let invocations = Arc::new(AtomicUsize::new(0));
    let bridge = LocalServiceBridge::spawn(
        EchoService::new(server_signing.clone()).with_invocations(Arc::clone(&invocations)),
        Arc::new(InMemoryNonceCache::new()),
        0,
    )?;
    let server = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(bridge, server_signing),
    )
    .await?;
    let client = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("client-moq"),
        NoopHandler::new("client-rpc"),
    )
    .await?;
    let conn = client
        .connect(direct_addr(&server), ALPN_HYPRSTREAM_RPC)
        .await?;
    let transport = IrohTransport::new(conn);

    for _ in 0..3 {
        assert_silent_drop(
            transport.send(b"not-a-capnp-message".to_vec(), Some(5_000)),
            "garbage must not receive a signed response",
        )
        .await;
    }
    assert_eq!(invocations.load(Ordering::SeqCst), 0);

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
