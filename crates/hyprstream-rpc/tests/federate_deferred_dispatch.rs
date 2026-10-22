//! Transport-to-handler wiring for the disabled Federate admission branch.
//! The hook below is a test double: H3b signature/current-primary and H2
//! Postgres authority/replay are tested in their own modules, not replaced by
//! this fixture's local deny switches.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::{
    io::Cursor,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use anyhow::{ensure, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signer as _, SigningKey};
use hyprstream_rpc::auth::mac::{install_mac_dispatch_pep, MacDecision, MacDispatchPep};
use hyprstream_rpc::{
    auth::{Claims, ClusterKeySource, InMemoryCredentialRevocationStore, JwtKeySource},
    crypto::{
        pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey},
        CryptoPolicy,
    },
    envelope::{self, EnvelopeVerification, RequestEnvelope, SignedEnvelope},
    node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key},
    proof::{
        admission::{set_global_proof_replay_store, InMemoryProofReplayStore},
        build::{
            build_authenticated_hybrid_request_proof, AuthenticatedHybridProofSigner,
            AuthenticatedRequestProofInput,
        },
        parser::ParsedProof,
        policy::{set_global_method_policy, CryptoSuite, InMemoryMethodPolicy, SignaturePolicy},
        recipient_binding::FederateRecipientBinding,
        verify::VerifiedProof,
    },
    service::{
        dispatch::process_request, Continuation, DecodedRequestBody, EnvelopeContext,
        RequestService,
    },
    transport::{carrier::CarrierContext, TransportConfig},
    ToCapnp,
};
use sha2::{Digest, Sha256};

const HOST: &str = hyprstream_rpc::auth::claims::FEDERATE_STAGING_HOST;
const CLIENT: &str = hyprstream_rpc::auth::claims::FEDERATE_STAGING_CLIENT;
const TEST_DID: &str = "did:plc:3f7ldey5bivhfjapwqi2sljy";
const SOURCE_ISSUER: &str = "https://login.federate.to";
const SCHEMA: u64 = 0xe7339d5d26ab3076;
const CLIENT_DH: [u8; 32] = [0x61; 32];

static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static MAC_ALLOW: AtomicBool = AtomicBool::new(true);

struct TestPep;

impl MacDispatchPep for TestPep {
    fn check(&self, _: &EnvelopeContext, _: &str, _: Option<&[u16]>) -> MacDecision {
        if MAC_ALLOW.load(Ordering::SeqCst) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(hyprstream_rpc::auth::mac::MacDenyReason::NoPepInstalled)
        }
    }
}

struct Fixture {
    ca: SigningKey,
    ca_pq: MlDsaSigningKey,
    server: SigningKey,
    relay: SigningKey,
    relay_pq: MlDsaSigningKey,
    holder: SigningKey,
    body: Vec<u8>,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let ca = SigningKey::from_bytes(&[51; 32]);
        let ca_pq = ml_dsa_sk_from_seed(&[57; 32]);
        let server = SigningKey::from_bytes(&[52; 32]);
        let relay = SigningKey::from_bytes(&[53; 32]);
        let relay_pq = ml_dsa_sk_from_seed(&[54; 32]);
        let holder = SigningKey::from_bytes(&[55; 32]);
        let mut store = envelope::KeyedPqTrustStore::new();
        store.bind(
            relay.verifying_key().to_bytes(),
            &ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&relay_pq)).unwrap(),
        );
        envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
            policy: CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        })
        .expect("isolated integration-test verify config");
        let mut table = InMemoryMethodPolicy::new();
        table.insert(
            "model",
            "1",
            SignaturePolicy::TokenBound {
                suite: CryptoSuite::Hybrid,
            },
        );
        assert!(
            set_global_method_policy(Box::new(table)).is_ok(),
            "isolated method policy"
        );
        install_mac_dispatch_pep(Arc::new(TestPep));
        let _ = set_global_proof_replay_store(Box::new(
            InMemoryProofReplayStore::single_verifier_instance(128),
        ));
        let body = hyprstream_rpc::serialize_message(|message| {
            message
                .init_root::<hyprstream_rpc::common_capnp::error_info::Builder>()
                .set_message("fixed-model");
        })
        .unwrap();
        Fixture {
            ca,
            ca_pq,
            server,
            relay,
            relay_pq,
            holder,
            body,
        }
    })
}

fn token(subject: &str) -> String {
    token_for_host(subject, HOST)
}

fn token_for_host(subject: &str, host: &str) -> String {
    let f = fixture();
    let now = chrono::Utc::now().timestamp();
    let holder_ed = f.holder.verifying_key().to_bytes();
    let holder_pq = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[56; 32]));
    let suite = hyprstream_rpc::auth::signer_suite_thumbprint(
        "hs-cose-sign-ed25519-mldsa65-wns-v1",
        &[&holder_ed, &holder_pq],
    );
    let mut claims = Claims::new(subject.into(), now, now + 120)
        .with_issuer(host.into())
        .with_audience(Some(host.into()))
        .with_client_id(CLIENT)
        .with_tenant("tenant-a".into())
        .with_sid("sid-a")
        .with_scope(Some("infer:model:qwen2.5-0.5b-instruct:main".into()))
        .with_cnf_jwk(&holder_ed)
        .with_federate_signer_suite(URL_SAFE_NO_PAD.encode(suite))
        .with_federate_signer_inventory([7; 32])
        .with_jti();
    claims.hs_signer_pq_key_v1 = Some(URL_SAFE_NO_PAD.encode(holder_pq));
    claims.hs_source_issuer_v1 = Some(SOURCE_ISSUER.into());
    claims.hs_source_subject_v1 = Some("opaque-atproto-connector-subject".into());
    let vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&f.ca_pq)).unwrap();
    let kid = hyprstream_rpc::auth::jwt::composite_kid(&vk, &f.ca.verifying_key());
    let header = format!(r#"{{"alg":"ML-DSA-65-Ed25519","typ":"at+jwt","kid":"{kid}"}}"#);
    let input = format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(header),
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
    );
    let mut signature = hyprstream_rpc::crypto::pq::ml_dsa_sign(&f.ca_pq, input.as_bytes());
    signature.extend_from_slice(&f.ca.sign(input.as_bytes()).to_bytes());
    format!("{input}.{}", URL_SAFE_NO_PAD.encode(signature))
}

type Recipient = hyprstream_rpc::crypto::hybrid_kem::RecipientKeypair;

fn recipients() -> (Recipient, Recipient) {
    let suite = hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
    (
        hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite).unwrap(),
        hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite).unwrap(),
    )
}

fn proof(
    token: &str,
    body: &[u8],
    response: &Recipient,
    stream: &Recipient,
    client_dh_public: [u8; 32],
) -> Vec<u8> {
    proof_with_recipients(token, body, Some(response), Some(stream), client_dh_public)
}

fn proof_with_recipients(
    token: &str,
    body: &[u8],
    response: Option<&Recipient>,
    stream: Option<&Recipient>,
    client_dh_public: [u8; 32],
) -> Vec<u8> {
    let f = fixture();
    let now = chrono::Utc::now().timestamp() as u64;
    let signer = AuthenticatedHybridProofSigner::new(
        f.holder.clone(),
        b"holder-ed".to_vec(),
        ml_dsa_sk_from_seed(&[56; 32]),
        b"holder-pq".to_vec(),
    )
    .unwrap();
    build_authenticated_hybrid_request_proof(
        &AuthenticatedRequestProofInput {
            service_domain: "model",
            credential: token.as_bytes(),
            issued_at: now,
            expires_at: now + 20,
            capnp_schema_id: SCHEMA,
            capnp_body: body,
            response_binding: None,
            federate_recipient_binding: Some(
                &FederateRecipientBinding::from_recipients(
                    response.map(Recipient::public).as_ref(),
                    stream.map(Recipient::public).as_ref(),
                    Some(client_dh_public),
                )
                .unwrap(),
            ),
        },
        &signer,
    )
    .unwrap()
}

fn proof_without_recipient_binding(token: &str, body: &[u8]) -> Vec<u8> {
    let f = fixture();
    let now = chrono::Utc::now().timestamp() as u64;
    let signer = AuthenticatedHybridProofSigner::new(
        f.holder.clone(),
        b"holder-ed".to_vec(),
        ml_dsa_sk_from_seed(&[56; 32]),
        b"holder-pq".to_vec(),
    )
    .unwrap();
    build_authenticated_hybrid_request_proof(
        &AuthenticatedRequestProofInput {
            service_domain: "model",
            credential: token.as_bytes(),
            issued_at: now,
            expires_at: now + 20,
            capnp_schema_id: SCHEMA,
            capnp_body: body,
            response_binding: None,
            federate_recipient_binding: None,
        },
        &signer,
    )
    .unwrap()
}

struct Service {
    transport: TransportConfig,
    key_source: Arc<dyn JwtKeySource>,
    expected_audience: Option<String>,
    revocations: InMemoryCredentialRevocationStore,
    expected_body: Vec<u8>,
    revoked: AtomicBool,
    calls: AtomicUsize,
    policy_admissions: AtomicUsize,
}

impl Service {
    fn new() -> Self {
        Self::for_host(HOST)
    }

    fn for_host(host: &str) -> Self {
        hyprstream_rpc::auth::claims::FederateHost::new(host).unwrap();
        let f = fixture();
        Self {
            transport: TransportConfig::inproc("model"),
            key_source: Arc::new(
                ClusterKeySource::new(f.ca.verifying_key(), host.into()).with_ca_composite_key(
                    ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&f.ca_pq)).unwrap(),
                ),
            ),
            expected_audience: Some(host.into()),
            revocations: InMemoryCredentialRevocationStore::new(),
            expected_body: f.body.clone(),
            revoked: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
            policy_admissions: AtomicUsize::new(0),
        }
    }
}

#[async_trait(?Send)]
impl RequestService for Service {
    fn name(&self) -> &str {
        "model"
    }
    fn transport(&self) -> &TransportConfig {
        &self.transport
    }
    fn signing_key(&self) -> SigningKey {
        fixture().server.clone()
    }
    fn pq_signing_key(&self) -> Option<MlDsaSigningKey> {
        Some(derive_mesh_mldsa_key(&fixture().server))
    }
    fn jwt_key_source(&self) -> Option<Arc<dyn JwtKeySource>> {
        Some(self.key_source.clone())
    }
    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }
    fn credential_revocation_store(
        &self,
    ) -> Option<&dyn hyprstream_rpc::auth::CredentialRevocationStore> {
        Some(&self.revocations)
    }
    fn accept_deferred_federate_credential(&self) -> bool {
        true
    }
    fn decode_request_body(&self, bytes: &[u8]) -> Result<DecodedRequestBody> {
        let message = capnp::serialize::read_message(
            &mut Cursor::new(bytes),
            hyprstream_rpc::service::body::bounded_reader_options(),
        )?;
        DecodedRequestBody::from_message(bytes.to_vec(), message, vec![1])
    }
    async fn admit_deferred_federate_request(
        &self,
        ctx: &EnvelopeContext,
        body: &DecodedRequestBody,
        proof: &ParsedProof,
    ) -> Result<(
        VerifiedProof,
        [u8; 32],
        String,
        String,
        hyprstream_rpc::proof::admission::ProofReplayKey,
        u64,
    )> {
        let (claims, exact_token) = ctx
            .deferred_federate_credential()
            .ok_or_else(|| anyhow::anyhow!("not deferred"))?;
        ensure!(
            claims.sub == TEST_DID && ctx.subject().is_anonymous(),
            "wrong subject or early identity"
        );
        ensure!(
            ctx.cnf != fixture().holder.verifying_key().to_bytes(),
            "relay must be distinct"
        );
        ensure!(body.bytes() == self.expected_body, "wrong resource/body");
        ensure!(
            proof.claims.capnp_schema_id == SCHEMA
                && proof.claims.credential_hash
                    == Some(Sha256::digest(exact_token.as_bytes()).into()),
            "proof does not bind exact credential/schema"
        );
        ensure!(!self.revoked.load(Ordering::SeqCst), "authority revoked");
        let replay_key = hyprstream_rpc::proof::admission::ProofReplayKey {
            signer_thumbprint: [9; 32],
            request_id: proof.claims.request_id,
        };
        Ok((
            VerifiedProof {
                replay_thumbprint: [9; 32],
                primary_principal: Some(TEST_DID.into()),
                primary_suite: "hs-cose-sign-ed25519-mldsa65-wns-v1".into(),
                approvers: vec![],
            },
            fixture().holder.verifying_key().to_bytes(),
            "model:qwen2.5-0.5b-instruct:main".into(),
            "infer".into(),
            replay_key,
            proof.claims.exp,
        ))
    }
    async fn commit_deferred_federate_request(
        &self,
        ctx: &EnvelopeContext,
        resource: &str,
        operation: &str,
        replay_key: &hyprstream_rpc::proof::admission::ProofReplayKey,
        replay_expiry: u64,
    ) -> Result<()> {
        ensure!(ctx.admitted_federate_operation(resource, operation));
        ensure!(replay_expiry > 0);
        ensure!(replay_key.signer_thumbprint == [9; 32]);
        self.policy_admissions.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        _: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        ensure!(
            ctx.subject().name() == Some(TEST_DID)
                && ctx.domain()? == "tenant-a"
                && ctx.authenticated_signer_key().unwrap().to_bytes()
                    == fixture().holder.verifying_key().to_bytes()
                && ctx.admitted_federate_operation("model:qwen2.5-0.5b-instruct:main", "infer"),
            "handler received non-holder or unadmitted identity"
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"served".to_vec(), None))
    }
}

fn wire(
    token: &str,
    proof: Vec<u8>,
    body: &[u8],
    response_recipient: &Recipient,
    stream_recipient: &Recipient,
    client_dh_public: [u8; 32],
) -> Vec<u8> {
    wire_with_recipients(
        token,
        proof,
        body,
        response_recipient,
        Some(stream_recipient),
        client_dh_public,
    )
}

fn wire_with_recipients(
    token: &str,
    proof: Vec<u8>,
    body: &[u8],
    response_recipient: &Recipient,
    stream_recipient: Option<&Recipient>,
    client_dh_public: [u8; 32],
) -> Vec<u8> {
    let f = fixture();
    let server_recipient = derive_mesh_kem_recipient(&f.server).unwrap();
    let mut request = RequestEnvelope::new(body.to_vec())
        .with_jwt_token(token.into())
        .with_proof_cwt(proof)
        .with_service_domain("model")
        .unwrap()
        .with_client_dh_public(client_dh_public)
        .with_response_kem_recipient(response_recipient.public());
    if let Some(stream_recipient) = stream_recipient {
        request = request
            .with_client_kem_public(stream_recipient.public())
            .unwrap();
    }
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        &f.relay,
        &f.relay_pq,
        &server_recipient.public(),
    )
    .unwrap();
    let mut message = capnp::message::Builder::new_default();
    signed.write_to(
        &mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>(),
    );
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).unwrap();
    bytes
}

fn assert_denial_has_no_success_payload(bytes: &[u8]) {
    let reader = capnp::serialize::read_message(
        &mut Cursor::new(bytes),
        capnp::message::ReaderOptions::new(),
    )
    .unwrap();
    let response = reader
        .get_root::<hyprstream_rpc::common_capnp::response_envelope::Reader>()
        .unwrap();
    assert!(response.get_payload().unwrap().is_empty());
    assert!(
        !response.get_encrypted_response().unwrap().is_empty(),
        "the uniform denial remains encrypted"
    );
}

async fn send(service: &Service, bytes: &[u8]) -> Result<Vec<u8>> {
    process_request(
        bytes,
        service,
        EnvelopeVerification::AnySigner,
        &fixture().server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await
}

#[tokio::test]
async fn configured_hosts_dispatch_only_with_the_matching_explicit_audience() -> Result<()> {
    let _guard = TEST_LOCK.lock().await;
    MAC_ALLOW.store(true, Ordering::SeqCst);
    let f = fixture();
    let hosts = ["https://vm-a.example", "https://vm-b.example:8443"];
    for (index, host) in hosts.iter().enumerate() {
        let service = Service::for_host(host);
        let foreign_service = Service::for_host(hosts[1 - index]);
        let mut missing_audience = Service::for_host(host);
        missing_audience.expected_audience = None;
        let mut wrong_audience = Service::for_host(host);
        wrong_audience.expected_audience = Some(hosts[1 - index].into());
        let bearer = token_for_host(TEST_DID, host);
        let (response, stream) = recipients();
        let request = wire(
            &bearer,
            proof(&bearer, &f.body, &response, &stream, CLIENT_DH),
            &f.body,
            &response,
            &stream,
            CLIENT_DH,
        );
        for denied_service in [&foreign_service, &missing_audience, &wrong_audience] {
            let denied = send(denied_service, &request).await?;
            assert_denial_has_no_success_payload(&denied);
            assert_eq!(denied_service.calls.load(Ordering::SeqCst), 0);
            assert_eq!(denied_service.policy_admissions.load(Ordering::SeqCst), 0);
        }
        // Wrong-host/configuration denials must not consume this holder proof.
        send(&service, &request).await?;
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
        assert_eq!(service.policy_admissions.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[tokio::test]
async fn deferred_dispatch_wiring_commits_only_valid_original_holder_request() {
    let _guard = TEST_LOCK.lock().await;
    MAC_ALLOW.store(true, Ordering::SeqCst);
    let f = fixture();
    let service = Service::new();
    let bearer = token(TEST_DID);
    let (response, stream) = recipients();
    let same_proof = proof(&bearer, &f.body, &response, &stream, CLIENT_DH);
    send(
        &service,
        &wire(
            &bearer,
            same_proof.clone(),
            &f.body,
            &response,
            &stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    send(
        &service,
        &wire(&bearer, same_proof, &f.body, &response, &stream, CLIENT_DH),
    )
    .await
    .unwrap();
    assert_eq!(
        service.calls.load(Ordering::SeqCst),
        1,
        "test admission replay denial"
    );

    service.revoked.store(true, Ordering::SeqCst);
    let (response, stream) = recipients();
    send(
        &service,
        &wire(
            &bearer,
            proof(&bearer, &f.body, &response, &stream, CLIENT_DH),
            &f.body,
            &response,
            &stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        service.calls.load(Ordering::SeqCst),
        1,
        "test authority revocation denial"
    );
    service.revoked.store(false, Ordering::SeqCst);

    let wrong_subject = token("bob");
    let (response, stream) = recipients();
    send(
        &service,
        &wire(
            &wrong_subject,
            proof(&wrong_subject, &f.body, &response, &stream, CLIENT_DH),
            &f.body,
            &response,
            &stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        service.calls.load(Ordering::SeqCst),
        1,
        "wrong subject denial"
    );

    let wrong_body = hyprstream_rpc::serialize_message(|message| {
        message
            .init_root::<hyprstream_rpc::common_capnp::error_info::Builder>()
            .set_message("other-model");
    })
    .unwrap();
    let (response, stream) = recipients();
    send(
        &service,
        &wire(
            &bearer,
            proof(&bearer, &wrong_body, &response, &stream, CLIENT_DH),
            &wrong_body,
            &response,
            &stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        service.calls.load(Ordering::SeqCst),
        1,
        "wrong resource/body denial"
    );
}

#[tokio::test]
async fn blind_relay_cannot_substitute_any_holder_bound_recipient() {
    let _guard = TEST_LOCK.lock().await;
    MAC_ALLOW.store(true, Ordering::SeqCst);
    let f = fixture();
    let service = Service::new();
    let bearer = token(TEST_DID);
    let (holder_response, holder_stream) = recipients();
    let holder_proof = proof(
        &bearer,
        &f.body,
        &holder_response,
        &holder_stream,
        CLIENT_DH,
    );

    let (relay_response, _) = recipients();
    let response_substitution = send(
        &service,
        &wire(
            &bearer,
            holder_proof.clone(),
            &f.body,
            &relay_response,
            &holder_stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_denial_has_no_success_payload(&response_substitution);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    let (_, relay_stream) = recipients();
    let stream_substitution = send(
        &service,
        &wire(
            &bearer,
            holder_proof.clone(),
            &f.body,
            &holder_response,
            &relay_stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_denial_has_no_success_payload(&stream_substitution);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    let relay_dh = [0x62; 32];
    let dh_substitution = send(
        &service,
        &wire(
            &bearer,
            holder_proof.clone(),
            &f.body,
            &holder_response,
            &holder_stream,
            relay_dh,
        ),
    )
    .await
    .unwrap();
    assert_denial_has_no_success_payload(&dh_substitution);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    // A signed null is meaningful: if the holder chose the legacy DH-only
    // stream path, the relay cannot add its own stream KEM recipient.
    let null_stream_proof =
        proof_with_recipients(&bearer, &f.body, Some(&holder_response), None, CLIENT_DH);
    let (_, relay_added_stream) = recipients();
    let stream_added_to_signed_null = send(
        &service,
        &wire_with_recipients(
            &bearer,
            null_stream_proof,
            &f.body,
            &holder_response,
            Some(&relay_added_stream),
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_denial_has_no_success_payload(&stream_added_to_signed_null);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    let omitted_claim = send(
        &service,
        &wire(
            &bearer,
            proof_without_recipient_binding(&bearer, &f.body),
            &f.body,
            &holder_response,
            &holder_stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_denial_has_no_success_payload(&omitted_claim);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    send(
        &service,
        &wire(
            &bearer,
            holder_proof,
            &f.body,
            &holder_response,
            &holder_stream,
            CLIENT_DH,
        ),
    )
    .await
    .unwrap();
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn federate_replay_waits_for_mac_and_remains_atomic_for_duplicates() -> Result<()> {
    let _guard = TEST_LOCK.lock().await;
    let f = fixture();
    let service = Service::new();
    let bearer = token(TEST_DID);
    let (response, stream) = recipients();
    let proof_bytes = proof(&bearer, &f.body, &response, &stream, CLIENT_DH);
    let request = wire(&bearer, proof_bytes, &f.body, &response, &stream, CLIENT_DH);

    // A mandatory-MAC denial happens before the service's final fresh Policy
    // admission. It must not consume replay state; correcting authorization
    // lets the identical signed request retry successfully.
    MAC_ALLOW.store(false, Ordering::SeqCst);
    let denied = send(&service, &request).await?;
    assert_denial_has_no_success_payload(&denied);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    assert_eq!(service.policy_admissions.load(Ordering::SeqCst), 0);

    MAC_ALLOW.store(true, Ordering::SeqCst);
    send(&service, &request).await?;
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.policy_admissions.load(Ordering::SeqCst), 1);

    // Two dispatches racing with the same fresh proof share the atomic replay
    // admission store. At most one reaches the handler.
    let (response, stream) = recipients();
    let proof_bytes = proof(&bearer, &f.body, &response, &stream, CLIENT_DH);
    let concurrent = wire(&bearer, proof_bytes, &f.body, &response, &stream, CLIENT_DH);
    let (first, second) = tokio::join!(send(&service, &concurrent), send(&service, &concurrent));
    first?;
    second?;
    assert_eq!(service.calls.load(Ordering::SeqCst), 2);
    Ok(())
}
