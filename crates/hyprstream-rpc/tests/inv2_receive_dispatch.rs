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

use parking_lot::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;

use hyprstream_rpc::browser_provisioning::{
    bind_request_payload, BrowserCarrierProfile, BrowserCurrentnessVerifier, BrowserRequestBinding,
    BROWSER_PROVISIONING_VERSION,
};

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
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    invoked: Arc<AtomicBool>,
}

impl SentinelService {
    fn new(signing_key: SigningKey) -> (Self, Arc<AtomicBool>) {
        Self::new_named(signing_key, "inv2-sentinel")
    }

    fn new_named(signing_key: SigningKey, name: &str) -> (Self, Arc<AtomicBool>) {
        let invoked = Arc::new(AtomicBool::new(false));
        (
            Self {
                name: name.to_owned(),
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
        if payload == b"handler-error" {
            anyhow::bail!("intentional handler failure");
        }
        Ok((payload.to_vec(), None))
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

    fn pq_signing_key(&self) -> Option<MlDsaSigningKey> {
        Some(derive_mesh_mldsa_key(&self.signing_key))
    }

    /// Surface the raw error text (the default returns an empty vec) so the
    /// test can assert *which* rejection path produced the response.
    fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
        error.as_bytes().to_vec()
    }
}

struct RecordingService {
    transport: TransportConfig,
    signing_key: SigningKey,
    payloads: Arc<Mutex<Vec<Vec<u8>>>>,
}#[async_trait(?Send)]
impl RequestService for RecordingService {
async fn handle_request(&self,
        _ctx:
        &EnvelopeContext,payload: &[u8],) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.payloads.lock().push(
        payload.to_vec());
        Ok(( payload.to_vec(),None))
}

fn name( &self) -> &str {
        "model"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self
    .signing_key.clone()
    }
}

struct AdvancingCurrentness {
    advanced: AtomicBool,
}

#[async_trait]
impl BrowserCurrentnessVerifier for AdvancingCurrentness {
    async fn ensure_current(&self, binding: &BrowserRequestBinding) -> Result<()> {
        anyhow::ensure!(
            binding.service_name == "model",
            "unexpected service binding");
        anyhow::ensure!(binding.accepted_state_epoch == 9, "unexpected state epoch");
        anyhow::ensure!(
            !self.advanced.load(Ordering::SeqCst),
            "accepted state advanced after browser refetch"
        );
        Ok(())
    }
}

fn browser_currentness() -> &'static Arc<AdvancingCurrentness> {
    static VERIFIER: OnceLock<Arc<AdvancingCurrentness>> = OnceLock::new();
    VERIFIER.get_or_init(|| {
        let verifier = Arc::new(AdvancingCurrentness {
            advanced: AtomicBool::new(false),
        });
        envelope::install_browser_currentness_verifier(verifier.clone()).unwrap();
        verifier
    })
}

fn browser_binding() -> BrowserRequestBinding {
    use base64::Engine as _;
    let encode = |bytes: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    BrowserRequestBinding {
        version: BROWSER_PROVISIONING_VERSION.to_owned(),
        service_name: "model".to_owned(),
        service_did: "did:at9p:test-service".to_owned(),
        service_origin: "https://model.example:443".to_owned(),
        capability: "hyprstream-rpc/1".to_owned(),
        scope: "model".to_owned(),
        carrier_profile: BrowserCarrierProfile::OwnedHybridWebTransport,
        response_key_id: "did:at9p:test-service#response".to_owned(),
        response_key_digest: encode(&[0x11; 32]),
        request_kem_key_id: "did:at9p:test-service#mesh-kem".to_owned(),
        request_kem_digest: encode(&[0x22; 32]),
        accepted_state_digest: encode(&[0x33; 64]),
        accepted_state_epoch: 9,
        expires_at_unix_ms: i64::MAX,
        projection_digest: encode(&[0x44; 32]),
    }
}

fn capnp_application_request(method_discriminator: u16) -> Vec<u8> {
    let mut request = Vec::with_capacity(32);
    request.extend_from_slice(&0u32.to_le_bytes());
    request.extend_from_slice(&3u32.to_le_bytes());
    request.extend_from_slice(&(2u64 << 32).to_le_bytes());
    request.extend_from_slice(&91u64.to_le_bytes());
    request.extend_from_slice(&(method_discriminator as u64).to_le_bytes());
    request
}

/// Two logical services may intentionally share one Ed25519/PQ/KEM identity.
/// The authenticated destination, not the key alone, must prevent forwarding
/// an exact request for A to B.
#[tokio::test]
async fn exact_same_key_request_is_bound_to_destination_service() {
    let k = keys();
    let (service_a, invoked_a) = SentinelService::new_named(k.server_sk.clone(), "service-a");
    let (service_b, invoked_b) = SentinelService::new_named(k.server_sk.clone(), "service-b");
    let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
        hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
    )
    .unwrap();
    let request = request_envelope(b"service-bound-control")
        .with_service_domain("service-a")
        .unwrap()
        .with_response_kem_recipient(response_recipient.public());
    let server_recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        &k.client_sk,
        &k.client_pq,
        &server_recipient.public(),
    )
    .unwrap();
    let wire = to_wire(&signed);

    let forwarded = process_request(
        &wire,
        &service_b,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await;
    assert!(
        forwarded.is_err(),
        "service B must reject A's exact request"
    );
    assert!(!invoked_b.load(Ordering::SeqCst));

    let matching = process_request(
        &wire,
        &service_a,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await
    .expect("unmutated matching-service control succeeds");
    let response = decode_response(&matching);
    assert!(invoked_a.load(Ordering::SeqCst));
    assert!(response.payload.is_empty());
    assert!(response.encrypted_response.is_some());
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
        response_kem_recipient: None,
        service_domain: Some("inv2-sentinel".to_owned()),
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

#[tokio::test]
async fn browser_dispatch_recovers_exact_bytes_and_rejects_post_refetch_advance() {
    let k = keys();
    let verifier = browser_currentness();
    verifier.advanced.store(false, Ordering::SeqCst);
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let service = RecordingService {
        transport: TransportConfig::inproc("browser-recording"),
        signing_key: k.server_sk.clone(),
        payloads: payloads.clone(),
    };
    let server_recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();
    let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
        hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
    )
    .unwrap();
    let binding = browser_binding();
    let application = capnp_application_request(7);

    let make_wire = |request_id| {
        let framed = bind_request_payload(&binding, request_id, "model", 7, &application).unwrap();
        let mut request =
            request_envelope(&framed).with_response_kem_recipient(response_recipient.public());
        request.request_id = request_id;
        request.service_domain = Some("model".to_owned());
        let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
            request,
            &k.client_sk,
            &k.client_pq,
            &server_recipient.public(),
        )
        .unwrap();
        to_wire(&signed)
    };

    process_request(
        &make_wire(23),
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::web_transport(),
    )
    .await
    .unwrap();
    assert_eq!(&*payloads.lock(), std::slice::from_ref(&application));

    // Simulate accepted-current state advancing after the client's last
    // defense-in-depth refetch but before dispatch/key release.
    verifier.advanced.store(true, Ordering::SeqCst);
    let rejected = process_request(
        &make_wire(24),
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::web_transport(),
    )
    .await;
    assert!(rejected.is_err());
    assert_eq!(&*payloads.lock(), &[application]);
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
    let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
        hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
    )
    .unwrap();
    let request = request_envelope(b"sealed-payload")
        .with_response_kem_recipient(response_recipient.public());
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
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
    assert!(response.payload.is_empty());
    assert!(response.encrypted_response.is_some());
}

/// A verified, sealed request that omits the response recipient must be
/// dropped before claims/handler work on every untrusted carrier. The server
/// cannot safely construct either a success or error response, so no
/// cleartext fallback is permitted.
#[tokio::test]
async fn missing_response_recipient_drops_without_handler_on_all_network_carriers() {
    let k = keys();
    let (service, invoked) = SentinelService::new(k.server_sk.clone());
    let nonce_cache = envelope::InMemoryNonceCache::new();
    let server_recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();

    for carrier in [
        CarrierContext::iroh(),
        CarrierContext::quic(),
        CarrierContext::web_transport(),
        CarrierContext::untrusted_unknown(),
    ] {
        let request = request_envelope(b"must-not-dispatch");
        let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
            request,
            &k.client_sk,
            &k.client_pq,
            &server_recipient.public(),
        )
        .unwrap();
        let result = process_request(
            &to_wire(&signed),
            &service,
            EnvelopeVerification::AnySigner,
            &k.server_sk,
            &nonce_cache,
            carrier,
        )
        .await;
        assert!(result.is_err(), "missing recipient must drop on {carrier}");
        assert!(!invoked.load(Ordering::SeqCst));
    }
}

#[tokio::test]
async fn post_admission_handler_error_is_sealed_not_cleartext() {
    let k = keys();
    let (service, invoked) = SentinelService::new(k.server_sk.clone());
    let nonce_cache = envelope::InMemoryNonceCache::new();
    let server_recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();
    let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
        hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
    )
    .unwrap();
    let request =
        request_envelope(b"handler-error").with_response_kem_recipient(response_recipient.public());
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        &k.client_sk,
        &k.client_pq,
        &server_recipient.public(),
    )
    .unwrap();
    let bytes = process_request(
        &to_wire(&signed),
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &nonce_cache,
        CarrierContext::quic(),
    )
    .await
    .unwrap();
    let response = decode_response(&bytes);
    assert!(invoked.load(Ordering::SeqCst));
    assert!(response.payload.is_empty());
    assert!(response.encrypted_response.is_some());
    assert!(!bytes
        .windows(b"intentional handler failure".len())
        .any(|w| { w == b"intentional handler failure" }));
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
