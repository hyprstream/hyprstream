//! INV-2 receive-side dispatch enforcement (#1042).
//!
//! Proves `service::dispatch::process_request` — the shared envelope-verify →
//! claims → handler pipeline — refuses a cleartext request envelope on an
//! untrusted carrier **before** the application handler runs, while an
//! encrypted (`#mesh-kem`) request on the same carrier dispatches normally.
//! Trusted-local cleartext is covered through the actual inproc/UDS transport
//! tests because trusted contexts are intentionally not publicly mintable.
//!
//! The carrier context is a required parameter supplied by the accept
//! boundary; nothing inside the request bytes participates in the decision,
//! which these tests demonstrate across all publicly constructible untrusted
//! carrier classifications.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;

use hyprstream_rpc::crypto::pq::MlDsaSigningKey;
use hyprstream_rpc::crypto::signing::generate_signing_keypair;
use hyprstream_rpc::envelope::{
    self, Authorization, EnvelopeVerification, RequestEnvelope, ResponseEnvelope, SignedEnvelope,
};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::service::dispatch::process_request;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::transport::carrier::CarrierContext;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::{FromCapnp, ToCapnp};

/// Process-wide fixed keys so every test in this binary shares one global
/// verify config / PQ trust store (installation is first-wins).
struct Keys {
    client_sk: SigningKey,
    client_pq: MlDsaSigningKey,
    server_sk: SigningKey,
}

fn keys() -> &'static Keys {
    static KEYS: OnceLock<Keys> = OnceLock::new();
    KEYS.get_or_init(|| {
        let (client_sk, client_vk) = generate_signing_keypair();
        let (server_sk, _server_vk) = generate_signing_keypair();
        let client_pq = derive_mesh_mldsa_key(&client_sk);

        // Hybrid verify config anchoring the client's ML-DSA key by its
        // Ed25519 kid — same shape the daemon installs at startup.
        let client_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&client_pq),
        )
        .unwrap();
        let mut store = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
        store.bind(client_vk.to_bytes(), &client_pq_vk);
        let _ = envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        });

        Keys {
            client_sk,
            client_pq,
            server_sk,
        }
    })
}

/// Minimal service whose only job is to record whether the application
/// handler was ever reached.
struct SentinelService {
    transport: TransportConfig,
    signing_key: SigningKey,
    invoked: Arc<AtomicBool>,
}

impl SentinelService {
    fn new(signing_key: SigningKey) -> (Self, Arc<AtomicBool>) {
        let invoked = Arc::new(AtomicBool::new(false));
        (
            Self {
                transport: TransportConfig::inproc("inv2-sentinel"),
                signing_key,
                invoked: Arc::clone(&invoked),
            },
            invoked,
        )
    }
}

#[async_trait(?Send)]
impl RequestService for SentinelService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invoked.store(true, Ordering::SeqCst);
        Ok((payload.to_vec(), None))
    }

    fn name(&self) -> &str {
        "inv2-sentinel"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    /// Surface the raw error text (the default returns an empty vec) so the
    /// test can assert *which* rejection path produced the response.
    fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
        error.as_bytes().to_vec()
    }
}

fn request_envelope(payload: &[u8]) -> RequestEnvelope {
    RequestEnvelope {
        request_id: 7,
        payload: payload.to_vec(),
        iat: envelope::current_timestamp(),
        nonce: envelope::generate_nonce(),
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
    }
}

fn to_wire(signed: &SignedEnvelope) -> Vec<u8> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut builder =
            message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
        signed.write_to(&mut builder);
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).unwrap();
    bytes
}

fn decode_response(bytes: &[u8]) -> ResponseEnvelope {
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(bytes),
        capnp::message::ReaderOptions::new(),
    )
    .unwrap();
    let rr = reader
        .get_root::<hyprstream_rpc::common_capnp::response_envelope::Reader>()
        .unwrap();
    ResponseEnvelope::read_from(rr).unwrap()
}

/// A validly hybrid-signed **cleartext** envelope from an authorized signer is
/// rejected on an untrusted carrier before the handler runs. The signature
/// being good is exactly the point: INV-2 is a carrier policy, not a
/// signature check.
#[tokio::test]
async fn cleartext_on_untrusted_carrier_rejected_before_handler() {
    let k = keys();
    let (service, invoked) = SentinelService::new(k.server_sk.clone());
    let nonce_cache = envelope::InMemoryNonceCache::new();

    for carrier in [
        CarrierContext::iroh(),
        CarrierContext::quic(),
        CarrierContext::web_transport(),
        CarrierContext::untrusted_unknown(),
    ] {
        let signed = SignedEnvelope::new_signed_hybrid(
            request_envelope(b"sensitive-cleartext"),
            &k.client_sk,
            &k.client_pq,
        );
        let wire = to_wire(&signed);
        let result = process_request(
            &wire,
            &service,
            EnvelopeVerification::AnySigner,
            &k.server_sk,
            &nonce_cache,
            carrier,
        )
        .await;

        assert!(result.is_err(), "cleartext must be silently rejected");
        assert!(
            !invoked.load(Ordering::SeqCst),
            "handler must never run for cleartext on carrier '{carrier}'"
        );
    }
}

/// An encrypted (`#mesh-kem`) hybrid-signed request on an untrusted carrier
/// passes the INV-2 gate and reaches the handler — the ban is on cleartext,
/// not on the carrier.
#[tokio::test]
async fn encrypted_on_untrusted_carrier_dispatches() {
    let k = keys();
    let (service, invoked) = SentinelService::new(k.server_sk.clone());
    let nonce_cache = envelope::InMemoryNonceCache::new();

    let recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request_envelope(b"sealed-payload"),
        &k.client_sk,
        &k.client_pq,
        &recipient.public(),
    )
    .unwrap();

    let response_bytes = process_request(
        &to_wire(&signed),
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &nonce_cache,
        CarrierContext::iroh(),
    )
    .await
    .unwrap();

    let response = decode_response(&response_bytes);
    assert!(
        invoked.load(Ordering::SeqCst),
        "encrypted request must dispatch"
    );
    assert_eq!(response.request_id, 7);
    assert_eq!(response.payload, b"sealed-payload");
}

/// Undecodable bytes on an untrusted carrier follow the cleartext branch
/// (fail-closed) and never reach the handler.
#[tokio::test]
async fn garbage_on_untrusted_carrier_rejected_before_handler() {
    let k = keys();
    let (service, invoked) = SentinelService::new(k.server_sk.clone());
    let nonce_cache = envelope::InMemoryNonceCache::new();

    let result = process_request(
        b"not-a-capnp-message",
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &nonce_cache,
        CarrierContext::iroh(),
    )
    .await;

    assert!(
        result.is_err(),
        "garbage must be dropped without a signed response"
    );
    assert!(
        !invoked.load(Ordering::SeqCst),
        "garbage must never dispatch"
    );
}
