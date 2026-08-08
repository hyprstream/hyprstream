//! #1483: the regression node for the browser-hybrid blind spot.
//!
//! Every hybrid integration test in this crate used to hand-anchor its
//! synthetic client with `store.bind(client_vk, &client_pq_vk)` — the exact
//! production mechanism a browser cannot perform (the admin-anchored store is
//! built from `mesh_peers` and is immutable after install). Those tests
//! validated the composite-signature crypto while assuming away the admission
//! problem, so CI stayed green through changes that broke every real browser.
//!
//! This test is the node that catches that class. It drives the same
//! `process_request` dispatch path the server uses, under the production-
//! mandatory `CryptoPolicy::Hybrid`, with an **empty** admin-anchored store
//! and **no `store.bind()` call anywhere**. The client establishes its own PQ
//! binding through the session overlay's first-contact path, exactly as a
//! browser would.
//!
//! ## What this catches
//!
//! Three named defect classes from the #1483 writeup, each locked by an
//! assertion below:
//!
//! 1. **Hand-bind masking** — `no_hand_bind_round_trip_succeeds` proves a
//!    request dispatches with zero anchored identities, and the store stays
//!    empty after. Remove the overlay consultation from `resolve_pq_anchor`
//!    and this test fails with the production error: *"mandatory Hybrid suite
//!    requires an anchored ML-DSA-65 signer key"*.
//!
//! 2. **Composite-binding defect** — `captured_envelope_cannot_squat` drives
//!    a real outer-layer swap through `process_request` and asserts nothing
//!    is recorded for the victim. The inner-commitment check (#1486 O1 fix)
//!    is the only thing between this and an identity squat; removing it
//!    makes this test fail.
//!
//! 3. **MAC assurance ceiling** — `first_contact_does_not_raise_assurance`
//!    asserts a first-contact binding reports `Classical`, not `PqHybrid`.
//!    A regression that promoted TOFU to `PqHybrid` would be caught here.
//!
//! ## What this does NOT cover (stated honestly)
//!
//! - This is a Rust e2e, not a headless-browser run of the wasm artifact.
//!   The wasm compile/link/execute gate lives in `.github/scripts/browser-
//!   wasm-check.sh` and `browser-wasm-test-ci.sh`, which compile the crate
//!   to `wasm32-unknown-unknown` and run `wasm_browser_fetch.rs` in a real
//!   headless Chromium. A std-only construct on a browser-path wasm surface
//!   fails the compile step; the fetch conformance path is executed in the
//!   browser. What is NOT exercised in a browser is the PQ composite signing
//!   path itself — the browser test covers DPoP token exchange, not hybrid
//!   envelope signing. That gap is noted here rather than hidden.
//!
//! - The carrier is `iroh`, not `browser_web_transport`. The PQ overlay is
//!   consulted in `verify_cose`, which is carrier-independent; the browser
//!   carrier adds provisioning checks (currentness, transcript) that are a
//!   separate concern from PQ admission. Using `iroh` isolates the PQ path.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;

use hyprstream_rpc::auth::mac::VerifiedKeyMaterial;
use hyprstream_rpc::crypto::pq::{
    ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
    MlDsaVerifyingKey,
};
use hyprstream_rpc::crypto::signing::generate_signing_keypair;
use hyprstream_rpc::envelope::{
    self, Authorization, EnvelopeVerification, RequestEnvelope, SignedEnvelope,
};
use hyprstream_rpc::node_identity::derive_mesh_kem_recipient;
use hyprstream_rpc::service::dispatch::process_request;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::session_pq_overlay::{
    install_session_pq_overlay, PqBindingEventSink, PqProvenance, RebindApproval, RebindEvent,
    SessionPqOverlay,
};
use hyprstream_rpc::transport::carrier::CarrierContext;
use hyprstream_rpc::transport::TransportConfig;

// ---------------------------------------------------------------------------
// Process fixtures (first-write-wins globals, built once per binary).
// ---------------------------------------------------------------------------

/// A minimal admin-anchored store that starts EMPTY and is asserted empty
/// after every round-trip. This is the load-bearing fixture: it proves the
/// overlay never writes through to the store the operator controls.
#[derive(Default)]
struct EmptyAnchoredStore {
    bindings: Mutex<std::collections::HashMap<[u8; 32], Vec<u8>>>,
}

impl EmptyAnchoredStore {
    fn assert_empty(&self, label: &str) {
        let map = self.bindings.lock();
        assert!(
            map.is_empty(),
            "{label}: the admin-anchored store must stay empty — the overlay \
             must never write through to it (found {} entries)",
            map.len()
        );
    }
}

impl envelope::PqTrustStore for EmptyAnchoredStore {
    fn ml_dsa_key_for(&self, ed25519_pubkey: &[u8; 32]) -> Option<MlDsaVerifyingKey> {
        self.bindings
            .lock()
            .get(ed25519_pubkey)
            .and_then(|b| ml_dsa_vk_from_bytes(b).ok())
    }
}

/// Counts first-contact events so a test can prove the overlay was consulted
/// rather than the store.
#[derive(Default)]
struct CountingSink {
    first_contact: std::sync::atomic::AtomicUsize,
}

impl PqBindingEventSink for CountingSink {
    fn on_first_contact(&self, _identity: &[u8; 32], _fingerprint: &[u8; 32]) {
        self.first_contact
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    fn on_rebind_refused(&self, _event: &RebindEvent) {}
    fn on_rebind_applied(&self, _approval: &RebindApproval) {}
}

struct Fixtures {
    server_sk: SigningKey,
    overlay: Arc<SessionPqOverlay>,
    sink: Arc<CountingSink>,
    anchored: Arc<EmptyAnchoredStore>,
}

fn fixtures() -> &'static Fixtures {
    static F: OnceLock<Fixtures> = OnceLock::new();
    F.get_or_init(|| {
        let (server_sk, _) = generate_signing_keypair();

        // EMPTY admin-anchored store: no operator enrolled anybody.
        let anchored = Arc::new(EmptyAnchoredStore::default());
        envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(anchored.clone()),
        })
        .expect("verify config installs once per test binary");

        let sink = Arc::new(CountingSink::default());
        let overlay = Arc::new(SessionPqOverlay::new(sink.clone()));
        install_session_pq_overlay(overlay.clone()).expect("overlay installs once per test binary");

        Fixtures {
            server_sk,
            overlay,
            sink,
            anchored,
        }
    })
}

/// A self-generated client — the browser/dynamic shape. Nothing about it is
/// known to the server in advance. There is deliberately no `store.bind`
/// anywhere in this file.
struct BrowserShapedClient {
    ed_sk: SigningKey,
    pq_sk: MlDsaSigningKey,
}

impl BrowserShapedClient {
    fn new(seed: u8) -> Self {
        let (ed_sk, _) = generate_signing_keypair();
        Self {
            ed_sk,
            pq_sk: ml_dsa_sk_from_seed(&[seed; 32]),
        }
    }

    fn identity(&self) -> [u8; 32] {
        self.ed_sk.verifying_key().to_bytes()
    }

    fn pq_vk(&self) -> MlDsaVerifyingKey {
        ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&self.pq_sk)).unwrap()
    }

    /// Sign a request the way a real hybrid client does: encrypted to the
    /// server's `#mesh-kem` recipient because the carrier forbids cleartext.
    fn signed_request(&self, service: &str, payload: &[u8]) -> SignedEnvelope {
        let request = RequestEnvelope {
            request_id: 42,
            payload: payload.to_vec(),
            iat: envelope::current_timestamp(),
            nonce: envelope::generate_nonce(),
            authorization: Authorization::None,
            delegation_token: None,
            wth: None,
            client_dh_public: None,
            client_kem_public: None,
            response_kem_recipient: None,
            service_domain: Some(service.to_owned()),
        };
        let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
            hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .unwrap();
        let request = request.with_response_kem_recipient(response_recipient.public());
        let server_recipient = derive_mesh_kem_recipient(&fixtures().server_sk).unwrap();
        SignedEnvelope::new_signed_encrypted_mesh_kem(
            request,
            &self.ed_sk,
            &self.pq_sk,
            &server_recipient.public(),
        )
        .unwrap()
    }
}

/// A service that records the MAC key material for each request.
struct ProbingService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    invoked: Arc<AtomicBool>,
    key_material: Arc<Mutex<Option<VerifiedKeyMaterial>>>,
}

impl ProbingService {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            transport: TransportConfig::inproc(name),
            signing_key: fixtures().server_sk.clone(),
            invoked: Arc::new(AtomicBool::new(false)),
            key_material: Arc::new(Mutex::new(None)),
        }
    }

    fn was_invoked(&self) -> bool {
        self.invoked.load(Ordering::SeqCst)
    }

    fn observed_key_material(&self) -> Option<VerifiedKeyMaterial> {
        *self.key_material.lock()
    }
}

#[async_trait(?Send)]
impl RequestService for ProbingService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invoked.store(true, Ordering::SeqCst);
        *self.key_material.lock() = Some(ctx.verified_key_material());
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
        Some(hyprstream_rpc::node_identity::derive_mesh_mldsa_key(
            &self.signing_key,
        ))
    }
}

fn to_wire(signed: &SignedEnvelope) -> Vec<u8> {
    use hyprstream_rpc::ToCapnp;
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

async fn dispatch(service: &ProbingService, signed: &SignedEnvelope) -> Result<Vec<u8>> {
    support::install_explicit_dispatch_pep();
    process_request(
        &to_wire(signed),
        service,
        EnvelopeVerification::AnySigner,
        &fixtures().server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await
}

// ---------------------------------------------------------------------------
// Tests — each locks a named defect class from the #1483 writeup.
// ---------------------------------------------------------------------------

/// **Defect class 1 — hand-bind masking.**
///
/// A browser-shaped client with NO operator enrollment dispatches through the
/// real `process_request` under `CryptoPolicy::Hybrid`. The admin-anchored
/// store is empty before and after: the overlay established the binding, and
/// it never wrote through.
///
/// **Causality:** remove the overlay consultation from `resolve_pq_anchor`
/// (envelope.rs) and this test fails with the exact production error:
/// *"mandatory Hybrid suite requires an anchored ML-DSA-65 signer key"*.
/// That is the gap #1483 exists to keep visible.
#[tokio::test]
async fn no_hand_bind_round_trip_succeeds() {
    let f = fixtures();
    let client = BrowserShapedClient::new(0x83);
    let service = ProbingService::new("e2e-1483-accept");

    // Precondition: nobody anchored this identity.
    assert!(f.overlay.provenance_for(&client.identity()).is_none());
    f.anchored.assert_empty("before dispatch");

    dispatch(
        &service,
        &client.signed_request("e2e-1483-accept", b"hello-1483"),
    )
    .await
    .expect("an unenrolled hybrid client must dispatch through the overlay");

    assert!(service.was_invoked());
    assert_eq!(
        f.overlay.provenance_for(&client.identity()),
        Some(PqProvenance::TofuBound),
        "the overlay must have recorded a first-contact binding"
    );
    assert!(
        f.sink.first_contact.load(Ordering::SeqCst) >= 1,
        "the overlay's first-contact path must have been consulted"
    );

    // The load-bearing assertion: the overlay never wrote through.
    f.anchored.assert_empty("after dispatch");
}

/// **Defect class 2 — composite-binding squat (#1486 O1).**
///
/// A captured envelope's outer ML-DSA layer can be re-signed by anyone who
/// holds the bytes. If first contact recorded such a composite, an attacker
/// with no Ed25519 key would squat the sender's binding. The inner-layer
/// commitment is what prevents this, and this test drives the attack through
/// the real dispatch path.
///
/// **Causality:** remove the `pq_bound` commitment check (envelope.rs
/// `FirstContact` arm) and this test fails — the victim's identity gets
/// squatted by the attacker's key.
#[tokio::test]
async fn captured_envelope_cannot_squat() {
    let f = fixtures();
    let victim = BrowserShapedClient::new(0x51);
    let attacker_pq = ml_dsa_sk_from_seed(&[0xEE; 32]);

    let mut captured = victim.signed_request("e2e-1483-squat", b"capture");
    let attacker_kid = swap_outer_layer(&mut captured, &attacker_pq);

    let service = ProbingService::new("e2e-1483-squat");
    dispatch(&service, &captured)
        .await
        .expect_err("a re-outer-signed capture must not dispatch");
    assert!(!service.was_invoked());

    assert!(
        f.overlay.provenance_for(&victim.identity()).is_none(),
        "no binding may be established from a captured envelope"
    );

    // The victim is not locked out.
    let victim_svc = ProbingService::new("e2e-1483-squat-2");
    dispatch(
        &victim_svc,
        &victim.signed_request("e2e-1483-squat-2", b"legitimate"),
    )
    .await
    .expect("the victim still owns their identity");
    assert_eq!(
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &f.overlay.verifying_key_for(&victim.identity()).unwrap()
        ),
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&victim.pq_vk())
    );
    assert_ne!(
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&victim.pq_vk()),
        attacker_kid
    );
    f.anchored.assert_empty("after squat attempt");
}

/// **Defect class 3 — MAC assurance ceiling.**
///
/// A first-contact binding makes a signature verifiable without making the
/// signer more trusted. The MAC key material derived from a TOFU binding
/// must be `Classical`, never `PqHybrid`. A regression that promoted TOFU
/// to `PqHybrid` would be caught here.
#[tokio::test]
async fn first_contact_does_not_raise_assurance() {
    let f = fixtures();
    let client = BrowserShapedClient::new(0x53);
    // (0x53 → seed byte for the assurance test; chosen to avoid collision with
    // the other tests' seeds in this isolated binary.)
    let service = ProbingService::new("e2e-1483-assurance");

    dispatch(
        &service,
        &client.signed_request("e2e-1483-assurance", b"hello"),
    )
    .await
    .unwrap();

    assert_eq!(
        service.observed_key_material(),
        Some(VerifiedKeyMaterial::Classical),
        "a first-contact binding must NOT raise MAC assurance above Classical"
    );
    f.anchored.assert_empty("after assurance check");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rewrite only the outer ML-DSA layer, keeping the sender's inner EdDSA
/// COSE_Sign1 byte-for-byte. Uses no Ed25519 private key — holding the
/// captured bytes is enough.
fn swap_outer_layer(signed: &mut SignedEnvelope, attacker_pq: &MlDsaSigningKey) -> Vec<u8> {
    use hyprstream_rpc::crypto::cose_sign::{decode_composite_for_test, encode_composite_for_test};

    let payload = signed
        .encrypted_envelope
        .clone()
        .expect("mesh-kem shape signs over the ciphertext");
    let aad = hyprstream_rpc::crypto::cose_sign1::build_external_aad(
        envelope::ENVELOPE_SCHEMA_ID,
        envelope::REQUEST_ENVELOPE_TYPE_ID,
    );
    let (inner_bytes, _) = decode_composite_for_test(&signed.cose).unwrap();
    let (ed_sig, _) = hyprstream_rpc::crypto::cose_sign::split_composite(&signed.cose).unwrap();

    let attacker_kid = ml_dsa_sk_to_vk_bytes(attacker_pq);
    let tbs =
        hyprstream_rpc::crypto::cose_sign::outer_tbs(attacker_kid.clone(), &payload, &ed_sig, &aad);
    let attacker_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(attacker_pq, &tbs);
    let attacker_outer =
        hyprstream_rpc::crypto::cose_sign::outer_layer_for_test(attacker_kid.clone(), attacker_sig)
            .unwrap();

    signed.cose = encode_composite_for_test(inner_bytes, Some(attacker_outer)).unwrap();
    attacker_kid
}
