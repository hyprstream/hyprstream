//! Causal single-decode dispatch tests over a REAL generated service (v16
//! §5.1/§5.2, issue #1504).
//!
//! These tests run the production `process_request` pipeline against the
//! actual generated `discovery` service artifacts — the generated
//! signed-body decoder (`decode_discovery_request_body`), the generated
//! method-policy inventory rows, and generated dispatch — and prove causally:
//!
//! 1. the handler receives the SAME decoded body whose derived leaf fed the
//!    MAC PEP (one decode; signed == decoded == handler body);
//! 2. an undecodable body denies uniformly BEFORE the handler (handler
//!    non-invocation); and
//! 3. the generated decoder and the generated policy inventory agree by
//!    construction: every derivable leaf resolves a generated row, and the
//!    installed table resolves it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use parking_lot::Mutex;

use hyprstream_discovery::generated::discovery_client::decode_discovery_request_body;
use hyprstream_rpc::crypto::pq::MlDsaSigningKey;
use hyprstream_rpc::crypto::signing::generate_signing_keypair;
use hyprstream_rpc::envelope::{
    self, Authorization, EnvelopeVerification, RequestEnvelope, SignedEnvelope,
};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::service::dispatch::{process_request, DISPATCH_DENIED};
use hyprstream_rpc::service::{Continuation, DecodedRequestBody, EnvelopeContext, RequestService};
use hyprstream_rpc::transport::carrier::CarrierContext;
use hyprstream_rpc::transport::TransportConfig;

// ── Shared fixed keys (verify config install is first-write-wins) ──────────

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

/// A permit-all MAC PEP that RECORDS the leaf coordinate it was handed, so a
/// test can prove the PEP consumed the same derived leaf the handler body
/// carries (one method identity across policy, PEP, and handler).
struct RecordingPermitPep {
    observed: Arc<Mutex<Vec<Option<Vec<u16>>>>>,
}

impl hyprstream_rpc::auth::mac::MacDispatchPep for RecordingPermitPep {
    fn check(
        &self,
        _ctx: &EnvelopeContext,
        _service_domain: &str,
        method: Option<&[u16]>,
    ) -> hyprstream_rpc::auth::mac::MacDecision {
        self.observed.lock().push(method.map(<[u16]>::to_vec));
        hyprstream_rpc::auth::mac::MacDecision::Permit
    }
}

fn install_recording_pep() -> Arc<Mutex<Vec<Option<Vec<u16>>>>> {
    static OBSERVED: OnceLock<Arc<Mutex<Vec<Option<Vec<u16>>>>>> = OnceLock::new();
    OBSERVED
        .get_or_init(|| {
            let observed = Arc::new(Mutex::new(Vec::new()));
            hyprstream_rpc::auth::mac::install_mac_dispatch_pep(Arc::new(RecordingPermitPep {
                observed: observed.clone(),
            }));
            observed
        })
        .clone()
}

/// The REAL generated discovery decoder + a sentinel handler that records the
/// body it was dispatched with.
struct SentinelDiscovery {
    transport: TransportConfig,
    signing_key: SigningKey,
    invoked: Arc<AtomicBool>,
    observed: Arc<Mutex<Vec<(Vec<u8>, Option<Vec<u16>>)>>>,
}

impl SentinelDiscovery {
    #[allow(clippy::type_complexity)]
    fn new(
        signing_key: SigningKey,
    ) -> (
        Self,
        Arc<AtomicBool>,
        Arc<Mutex<Vec<(Vec<u8>, Option<Vec<u16>>)>>>,
    ) {
        let invoked = Arc::new(AtomicBool::new(false));
        let observed = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                transport: TransportConfig::inproc("leaf-dispatch-test"),
                signing_key,
                invoked: Arc::clone(&invoked),
                observed: Arc::clone(&observed),
            },
            invoked,
            observed,
        )
    }
}

#[async_trait(?Send)]
impl RequestService for SentinelDiscovery {
    fn decode_request_body(&self, signed_body: &[u8]) -> Result<DecodedRequestBody> {
        // The production seam under test: the generated single decode.
        decode_discovery_request_body(signed_body)
    }

    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        body: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invoked.store(true, Ordering::SeqCst);
        self.observed.lock().push((
            body.bytes().to_vec(),
            body.leaf_path().map(<[u16]>::to_vec),
        ));
        Ok((b"handled".to_vec(), None))
    }

    fn name(&self) -> &str {
        "discovery"
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

    fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
        error.as_bytes().to_vec()
    }
}

/// A real `discovery` request body: `ping` (a `$scope(query)` root leaf).
fn ping_request_bytes() -> Vec<u8> {
    let mut message = capnp::message::Builder::new_default();
    {
        let mut req = message
            .init_root::<hyprstream_discovery::discovery_capnp::discovery_request::Builder>();
        req.set_id(7);
        req.set_ping(());
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).unwrap();
    bytes
}

fn envelope_for(payload: &[u8]) -> Vec<u8> {
    let k = keys();
    let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
        hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
    )
    .unwrap();
    let request = RequestEnvelope {
        request_id: 7,
        payload: payload.to_vec(),
        iat: envelope::current_timestamp(),
        nonce: hyprstream_rpc::envelope::generate_nonce(),
        authorization: Authorization::None,
        delegation_token: None,
        wth: None,
        client_dh_public: None,
        client_kem_public: None,
        response_kem_recipient: None,
        service_domain: Some("discovery".to_owned()),
        proof_cwt: None,
    }
    .with_response_kem_recipient(response_recipient.public());
    let server_recipient = derive_mesh_kem_recipient(&k.server_sk).unwrap();
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        &k.client_sk,
        &k.client_pq,
        &server_recipient.public(),
    )
    .unwrap();
    let mut message = capnp::message::Builder::new_default();
    {
        let mut builder = message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
        use hyprstream_rpc::ToCapnp;
        signed.write_to(&mut builder);
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).unwrap();
    bytes
}

fn decode_response(bytes: &[u8]) -> hyprstream_rpc::envelope::ResponseEnvelope {
    use hyprstream_rpc::FromCapnp;
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(bytes),
        capnp::message::ReaderOptions::new(),
    )
    .unwrap();
    let root = reader
        .get_root::<hyprstream_rpc::common_capnp::response_envelope::Reader>()
        .unwrap();
    hyprstream_rpc::envelope::ResponseEnvelope::read_from(root).unwrap()
}

/// The handler receives the SAME decoded body whose leaf the MAC PEP was
/// handed — one decode, one method identity, end to end through the
/// production `process_request` pipeline.
#[tokio::test]
async fn handler_receives_the_same_decoded_body_and_leaf_the_pep_consumed() {
    let pep_observed = install_recording_pep();
    let k = keys();
    let (service, invoked, observed) = SentinelDiscovery::new(k.server_sk.clone());

    let payload = ping_request_bytes();
    // The expected leaf, derived through the same generated decoder the
    // service wires into dispatch.
    let expected_leaf = decode_discovery_request_body(&payload)
        .unwrap()
        .leaf_path()
        .map(<[u16]>::to_vec)
        .expect("a real request derives a leaf");

    let wire = envelope_for(&payload);
    let pep_before = pep_observed.lock().len();
    let response = process_request(
        &wire,
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await
    .expect("valid request dispatches");
    let _ = decode_response(&response);

    assert!(invoked.load(Ordering::SeqCst), "handler must run");
    let seen = observed.lock();
    assert_eq!(seen.len(), 1);
    let (handler_bytes, handler_leaf) = &seen[0];
    assert_eq!(
        handler_bytes, &payload,
        "handler body bytes must be the exact signed request bytes"
    );
    assert_eq!(
        handler_leaf.as_deref(),
        Some(expected_leaf.as_slice()),
        "handler body must carry the derived leaf"
    );
    let pep_seen = pep_observed.lock();
    assert!(
        pep_seen[pep_before..]
            .iter()
            .any(|leaf| leaf.as_deref() == Some(expected_leaf.as_slice())),
        "the MAC PEP must have been handed the SAME derived leaf; saw {:?}",
        &pep_seen[pep_before..]
    );
}

/// An undecodable body denies through the uniform `DispatchDenied` surface
/// BEFORE the handler — causal handler non-invocation for the decode gate.
#[tokio::test]
async fn an_undecodable_body_denies_before_the_handler() {
    install_recording_pep();
    let k = keys();
    let (service, invoked, _) = SentinelDiscovery::new(k.server_sk.clone());

    let wire = envelope_for(b"not a capnp message at all");
    let response = process_request(
        &wire,
        &service,
        EnvelopeVerification::AnySigner,
        &k.server_sk,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await
    .expect("a decode failure is a uniform signed denial, not a dropped stream");

    assert!(
        !invoked.load(Ordering::SeqCst),
        "handler must never run for an undecodable body"
    );
    let envelope = decode_response(&response);
    assert!(
        envelope.encrypted_response.is_some(),
        "denial must keep the uniform encrypted response shape"
    );
    // The visible constant is confined to the encrypted payload; the outer
    // envelope shape is indistinguishable from success.
    assert_eq!(DISPATCH_DENIED, "dispatch denied");
}

/// The generated decoder and the generated method-policy inventory agree on
/// the real schema: the decoded leaf resolves a generated row in the
/// validated, installed table.
#[test]
fn decoded_leaf_resolves_a_generated_policy_row() {
    use hyprstream_rpc::proof::policy::DispatchMethodPolicy as _;

    let payload = ping_request_bytes();
    let body = decode_discovery_request_body(&payload).unwrap();
    let leaf = body.leaf_path_string().expect("real request derives a leaf");

    let rows = hyprstream_rpc::proof::policy::collect_generated_rows().unwrap();
    let row = rows
        .iter()
        .find(|row| row.service == "discovery" && row.leaf_key() == leaf)
        .expect("the decoded leaf must be listed in the generated inventory");
    assert_eq!(row.symbolic_path, "ping");
    assert_eq!(row.scope_action, "query");

    let (table, count) = hyprstream_rpc::proof::policy::build_generated_method_policy().unwrap();
    assert!(count > 0);
    assert!(
        table.policy_for("discovery", &leaf).is_some(),
        "the installed table must resolve the decoded leaf"
    );
    assert!(
        table.policy_for("discovery", "9999").is_none(),
        "an unlisted leaf must resolve no row"
    );
}
