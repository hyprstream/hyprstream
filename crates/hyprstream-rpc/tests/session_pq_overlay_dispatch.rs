//! End-to-end proof that a dynamically-identified client — one no operator
//! could have pre-enrolled — completes a real dispatch round-trip under the
//! production-mandatory Hybrid suite.
//!
//! **There is deliberately no `KeyedPqTrustStore::bind` call in this file.**
//! Every other hybrid integration test in this crate hand-anchors its synthetic
//! client, which is precisely the production mechanism that did not exist; a
//! test that hand-binds proves nothing about the path exercised here. The store
//! installed below is empty and stays empty: it is asserted empty after the
//! round-trip, so the overlay cannot be quietly writing through to it.
//!
//! What is proved:
//!
//! - first contact admits the request and records a binding;
//! - a first-contact binding does NOT raise MAC assurance above Classical;
//! - presenting a different PQ key for a bound identity is refused, surfaced,
//!   and leaves the binding intact;
//! - a self-asserted key the client cannot actually sign with is never recorded;
//! - a PQ-less envelope from a bound identity does not clear the binding;
//! - an out-of-band promotion does raise assurance, and can be revoked.

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
    composite_fingerprint, FirstContactOutcome, PqBindingEventSink, PqProvenance, RebindApproval,
    RebindEvent, SessionPqOverlay,
};
use hyprstream_rpc::transport::carrier::CarrierContext;
use hyprstream_rpc::transport::TransportConfig;

// ---------------------------------------------------------------------------
// Process fixtures. The verify config, the overlay, and the PEP are all
// first-write-wins process globals, so they are built once for the binary.
// ---------------------------------------------------------------------------

/// Records every binding event the overlay publishes, so a test can assert that
/// a refusal was surfaced rather than merely returned.
#[derive(Default)]
struct RecordingSink {
    refused: Mutex<Vec<[u8; 32]>>,
    applied: Mutex<Vec<[u8; 32]>>,
    first_contact: Mutex<Vec<[u8; 32]>>,
}

impl PqBindingEventSink for RecordingSink {
    fn on_rebind_refused(&self, event: &RebindEvent) {
        self.refused.lock().push(*event.identity());
    }
    fn on_rebind_applied(&self, approval: &RebindApproval) {
        self.applied.lock().push(*approval.event().identity());
    }
    fn on_first_contact(&self, identity: &[u8; 32], _fingerprint: &[u8; 32]) {
        self.first_contact.lock().push(*identity);
    }
}

struct Fixtures {
    server_sk: SigningKey,
    overlay: Arc<SessionPqOverlay>,
    sink: Arc<RecordingSink>,
    /// The admin-anchored store, kept so tests can assert it never grew.
    store: Arc<envelope::KeyedPqTrustStore>,
}

fn fixtures() -> &'static Fixtures {
    static F: OnceLock<Fixtures> = OnceLock::new();
    F.get_or_init(|| {
        let (server_sk, _) = generate_signing_keypair();

        // An EMPTY admin-anchored store: no operator enrolled anybody. This is
        // the state a real deployment is in with respect to a browser.
        let store = Arc::new(envelope::KeyedPqTrustStore::new());
        assert!(store.is_empty());
        envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(store.clone()),
        })
        .expect("verify config installs once per test binary");

        let sink = Arc::new(RecordingSink::default());
        let overlay = Arc::new(SessionPqOverlay::new(sink.clone()));
        hyprstream_rpc::session_pq_overlay::install_session_pq_overlay(overlay.clone())
            .expect("overlay installs once per test binary");

        Fixtures {
            server_sk,
            overlay,
            sink,
            store,
        }
    })
}

/// A client whose Ed25519 identity and ML-DSA-65 key are both self-generated —
/// the browser/dynamic shape. Nothing about it is known to the server in
/// advance.
struct DynamicClient {
    ed_sk: SigningKey,
    pq_sk: MlDsaSigningKey,
}

impl DynamicClient {
    fn new(pq_seed: u8) -> Self {
        let (ed_sk, _) = generate_signing_keypair();
        Self {
            ed_sk,
            pq_sk: ml_dsa_sk_from_seed(&[pq_seed; 32]),
        }
    }

    fn identity(&self) -> [u8; 32] {
        self.ed_sk.verifying_key().to_bytes()
    }

    fn pq_vk(&self) -> MlDsaVerifyingKey {
        ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&self.pq_sk)).unwrap()
    }

    /// Sign a request the way a real hybrid client does, encrypted to the
    /// server's `#mesh-kem` recipient because the carrier forbids cleartext.
    fn signed_request(&self, service: &str, payload: &[u8]) -> SignedEnvelope {
        let request = RequestEnvelope {
            request_id: 7,
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

/// Service that records the MAC key material derived for each request it sees.
struct ObservingService {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    invoked: Arc<AtomicBool>,
    key_material: Arc<Mutex<Option<VerifiedKeyMaterial>>>,
    signer: Arc<Mutex<Option<[u8; 32]>>>,
}

impl ObservingService {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            transport: TransportConfig::inproc(name),
            signing_key: fixtures().server_sk.clone(),
            invoked: Arc::new(AtomicBool::new(false)),
            key_material: Arc::new(Mutex::new(None)),
            signer: Arc::new(Mutex::new(None)),
        }
    }

    fn was_invoked(&self) -> bool {
        self.invoked.load(Ordering::SeqCst)
    }

    fn observed_key_material(&self) -> Option<VerifiedKeyMaterial> {
        *self.key_material.lock()
    }

    fn observed_signer(&self) -> Option<[u8; 32]> {
        *self.signer.lock()
    }
}

#[async_trait(?Send)]
impl RequestService for ObservingService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invoked.store(true, Ordering::SeqCst);
        *self.signer.lock() = Some(ctx.cnf);
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

async fn dispatch(service: &ObservingService, signed: &SignedEnvelope) -> Result<Vec<u8>> {
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

fn refusals_for(identity: &[u8; 32]) -> usize {
    fixtures()
        .sink
        .refused
        .lock()
        .iter()
        .filter(|i| *i == identity)
        .count()
}

fn first_contacts_for(identity: &[u8; 32]) -> usize {
    fixtures()
        .sink
        .first_contact
        .lock()
        .iter()
        .filter(|i| *i == identity)
        .count()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The acceptance case: a client the operator never enrolled gets its hybrid
/// signature checked and its request handled — and gains no trust for it.
#[tokio::test]
async fn unenrolled_hybrid_client_dispatches_and_stays_classical() {
    let f = fixtures();
    let client = DynamicClient::new(0xA1);
    let service = ObservingService::new("overlay-first-contact");

    // Precondition: nobody anchored this identity, and nothing knows it yet.
    assert!(f.overlay.provenance_for(&client.identity()).is_none());

    dispatch(&service, &client.signed_request("overlay-first-contact", b"hello"))
        .await
        .expect("first contact from an unenrolled hybrid client must dispatch");

    assert!(service.was_invoked());
    assert_eq!(service.observed_signer(), Some(client.identity()));
    assert_eq!(first_contacts_for(&client.identity()), 1);
    assert_eq!(
        f.overlay.provenance_for(&client.identity()),
        Some(PqProvenance::TofuBound)
    );

    // The load-bearing property: verifiable, not more trusted.
    assert_eq!(
        service.observed_key_material(),
        Some(VerifiedKeyMaterial::Classical),
        "a first-contact binding must not raise MAC assurance"
    );

    // The admin-anchored store was consulted, never written.
    assert!(
        f.store.is_empty(),
        "the overlay must never write through to the admin-anchored store"
    );
}

/// A second request from the same client reuses the established binding without
/// re-recording it.
#[tokio::test]
async fn an_established_binding_serves_later_requests() {
    let f = fixtures();
    let client = DynamicClient::new(0xA2);
    let service = ObservingService::new("overlay-repeat");

    dispatch(&service, &client.signed_request("overlay-repeat", b"one"))
        .await
        .unwrap();
    dispatch(&service, &client.signed_request("overlay-repeat", b"two"))
        .await
        .expect("an established binding keeps serving");

    assert_eq!(
        first_contacts_for(&client.identity()),
        1,
        "the binding is recorded once, not re-recorded per request"
    );
    assert_eq!(refusals_for(&client.identity()), 0);
    assert!(f.store.is_empty());
}

/// The service-worker / cache-replacement case: the same identity comes back
/// holding a different PQ key. It must be refused and surfaced, and the
/// established binding must survive untouched.
#[tokio::test]
async fn a_different_pq_key_for_a_bound_identity_is_refused_and_surfaced() {
    let f = fixtures();
    let mut client = DynamicClient::new(0xA3);
    let service = ObservingService::new("overlay-rebind");

    dispatch(&service, &client.signed_request("overlay-rebind", b"first"))
        .await
        .unwrap();
    let original_key = client.pq_vk();

    // Same Ed25519 identity, different ML-DSA-65 key — the substitution TOFU
    // exists to catch.
    client.pq_sk = ml_dsa_sk_from_seed(&[0xB3; 32]);
    let substituted = ObservingService::new("overlay-rebind-2");
    let err = dispatch(
        &substituted,
        &client.signed_request("overlay-rebind-2", b"second"),
    )
    .await
    .expect_err("a substituted PQ key must not be silently accepted");
    assert!(
        format!("{err:#}").contains("established binding"),
        "rejection must name the rebinding, got: {err:#}"
    );
    assert!(!substituted.was_invoked());

    assert_eq!(
        refusals_for(&client.identity()),
        1,
        "the refusal must reach the event sink, not merely be returned"
    );
    // Last write did NOT win.
    assert_eq!(
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &f.overlay.verifying_key_for(&client.identity()).unwrap()
        ),
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&original_key)
    );
    assert!(f.store.is_empty());
}

/// The rebinding alarm must be attributable. Anyone can copy a public key into
/// an envelope's `cnf`; only the identity's holder can sign as it. A stranger
/// waving a victim's public key around must not be able to inject a security
/// event about that victim.
#[tokio::test]
async fn a_rebinding_alarm_cannot_be_injected_by_a_stranger() {
    let f = fixtures();
    let victim = DynamicClient::new(0xA8);
    let service = ObservingService::new("overlay-alarm");

    dispatch(&service, &victim.signed_request("overlay-alarm", b"bind"))
        .await
        .unwrap();
    let before = refusals_for(&victim.identity());

    // A stranger signs with their own keys, then relabels the envelope as the
    // victim. The PQ key on the wire differs from the victim's binding, so the
    // rebinding branch is reached — but the EdDSA layer does not verify as the
    // victim, so nothing may be raised about them.
    let stranger = DynamicClient::new(0xB8);
    let mut impersonation = stranger.signed_request("overlay-alarm-2", b"not-me");
    impersonation.cnf = victim.identity();

    let target = ObservingService::new("overlay-alarm-2");
    dispatch(&target, &impersonation)
        .await
        .expect_err("a relabelled envelope must be rejected");
    assert!(!target.was_invoked());
    assert_eq!(
        refusals_for(&victim.identity()),
        before,
        "an unauthenticated sender must not be able to raise a rebinding alarm"
    );
    // And the victim's binding is untouched.
    assert_eq!(
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &f.overlay.verifying_key_for(&victim.identity()).unwrap()
        ),
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&victim.pq_vk())
    );
    assert!(f.store.is_empty());
}

/// A PQ key the client asserts but cannot sign with must never be recorded:
/// the binding is committed only after the whole composite verifies against
/// the candidate.
#[tokio::test]
async fn an_unusable_self_asserted_key_is_never_recorded() {
    use hyprstream_rpc::crypto::cose_sign::{assemble_composite_nested, split_composite};

    let f = fixtures();
    let client = DynamicClient::new(0xA4);
    let service = ObservingService::new("overlay-forged");

    let mut signed = client.signed_request("overlay-forged", b"forged");
    let (ed_sig, pq_sig) = split_composite(&signed.cose).unwrap();
    // Keep the real ML-DSA-65 signature, but claim a key it was not made with.
    let squatted = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[0xC4; 32]));
    signed.cose = assemble_composite_nested(
        (signed.cnf.to_vec(), ed_sig),
        Some((squatted.clone(), pq_sig.unwrap())),
    )
    .unwrap();

    dispatch(&service, &signed)
        .await
        .expect_err("a self-asserted key the signer cannot use must be rejected");
    assert!(!service.was_invoked());
    assert!(
        f.overlay.provenance_for(&client.identity()).is_none(),
        "nothing may be recorded for a composite that did not verify"
    );
    assert_eq!(first_contacts_for(&client.identity()), 0);
    assert!(f.store.is_empty());
}

/// Downgrade monotonicity: an envelope arriving without the PQ layer from an
/// already-bound identity is rejected and clears nothing.
#[tokio::test]
async fn a_pq_less_envelope_does_not_clear_an_established_binding() {
    use hyprstream_rpc::crypto::cose_sign::{decode_composite_for_test, encode_composite_for_test};

    let f = fixtures();
    let client = DynamicClient::new(0xA5);
    let service = ObservingService::new("overlay-downgrade");

    dispatch(&service, &client.signed_request("overlay-downgrade", b"bound"))
        .await
        .unwrap();
    let bound_key = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
        &f.overlay.verifying_key_for(&client.identity()).unwrap(),
    );

    let mut stripped = client.signed_request("overlay-downgrade-2", b"downgrade");
    let (inner, _outer) = decode_composite_for_test(&stripped.cose).unwrap();
    stripped.cose = encode_composite_for_test(inner, None).unwrap();

    let downgraded = ObservingService::new("overlay-downgrade-2");
    dispatch(&downgraded, &stripped)
        .await
        .expect_err("the mandatory Hybrid suite must reject a PQ-less envelope");
    assert!(!downgraded.was_invoked());

    assert_eq!(
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &f.overlay.verifying_key_for(&client.identity()).unwrap()
        ),
        bound_key,
        "a PQ-less envelope must not clear an established binding"
    );
    assert_eq!(
        f.overlay.provenance_for(&client.identity()),
        Some(PqProvenance::TofuBound)
    );
    assert!(f.store.is_empty());
}

/// Out-of-band promotion is the only route to `PqHybrid`, and it is reversible.
#[tokio::test]
async fn out_of_band_promotion_raises_assurance_and_can_be_revoked() {
    let f = fixtures();
    let client = DynamicClient::new(0xA6);
    let service = ObservingService::new("overlay-promote");

    dispatch(&service, &client.signed_request("overlay-promote", b"before"))
        .await
        .unwrap();
    assert_eq!(
        service.observed_key_material(),
        Some(VerifiedKeyMaterial::Classical)
    );

    // The operator compares the fingerprint over the PAIR through a channel
    // with different compromise assumptions, then promotes.
    let fingerprint = composite_fingerprint(&client.identity(), &client.pq_vk());
    assert_eq!(f.overlay.fingerprint_for(&client.identity()), Some(fingerprint));
    f.overlay
        .promote_out_of_band(&client.identity(), &fingerprint, "operator-published")
        .expect("a matching out-of-band fingerprint promotes");

    let promoted = ObservingService::new("overlay-promote-2");
    dispatch(&promoted, &client.signed_request("overlay-promote-2", b"after"))
        .await
        .unwrap();
    assert_eq!(
        promoted.observed_key_material(),
        Some(VerifiedKeyMaterial::PqHybrid),
        "an out-of-band verified binding may raise assurance"
    );

    // Revocation through the granting channel takes the assurance back without
    // breaking the live session.
    assert!(f.overlay.revoke_promotion(&client.identity()));
    let revoked = ObservingService::new("overlay-promote-3");
    dispatch(&revoked, &client.signed_request("overlay-promote-3", b"revoked"))
        .await
        .expect("revoking a promotion does not break verifiability");
    assert_eq!(
        revoked.observed_key_material(),
        Some(VerifiedKeyMaterial::Classical),
        "a revoked promotion returns to first-contact continuity"
    );
    assert!(f.store.is_empty());
}

/// Rotation is permitted, but only through the same visible path an attack
/// takes: the refusal event is what an approval is built from.
#[tokio::test]
async fn rotation_travels_the_visible_rebinding_path() {
    let f = fixtures();
    let mut client = DynamicClient::new(0xA7);
    let service = ObservingService::new("overlay-rotate");

    dispatch(&service, &client.signed_request("overlay-rotate", b"old"))
        .await
        .unwrap();

    client.pq_sk = ml_dsa_sk_from_seed(&[0xB7; 32]);
    let rotated_key = client.pq_vk();

    // The rotated key is refused at the verify path first.
    let blocked = ObservingService::new("overlay-rotate-2");
    dispatch(&blocked, &client.signed_request("overlay-rotate-2", b"new"))
        .await
        .expect_err("rotation is not silently accepted");

    // The operator reviews the surfaced event and approves it.
    let FirstContactOutcome::RebindRefused(event) = f
        .overlay
        .observe_first_contact(client.identity(), &rotated_key)
    else {
        panic!("the overlay must surface the rebinding rather than apply it");
    };
    assert!(f
        .overlay
        .apply_rebind(event.approve(PqProvenance::TofuBound, "operator-reviewed rotation")));

    let after = ObservingService::new("overlay-rotate-3");
    dispatch(&after, &client.signed_request("overlay-rotate-3", b"new"))
        .await
        .expect("an approved rotation is accepted");
    assert_eq!(
        after.observed_key_material(),
        Some(VerifiedKeyMaterial::Classical),
        "an approved rotation confers no trust of its own"
    );
    assert!(f.store.is_empty());
}
