//! Real production mint -> encrypted RPC/proof verification -> Model MAC gate.
//! Policy action admission/replay and the outer dispatch PEP are explicit test
//! doubles; this is not a running-service or inference-engine E2E receipt.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::super::{
    federate_exchange,
    federate_host::FederateSigner,
    federate_source::{FederateHost, CLIENT, HOST, ISSUER},
};
use crate::{
    mac::{
        compiled::{
            cose::{HybridPolicySigner, HybridPolicyVerifier},
            sign_policy, CompiledPolicy, PolicyApproval, PolicyLoader,
        },
        lattice::{Assurance, CompartmentSet, Lattice, LatticeVersion, Level, SecurityLabel},
        te::TeMatrix,
        EnrollmentSubjectContextResolver,
    },
    services::{
        generated::model_client::ModelHandler,
        model::{ModelService, ModelServiceConfig},
        oauth::token_exchange::SubjectContextResolver as _,
    },
};
use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::{
    auth::{
        mac::{CredentialClearance, SubjectContextClaims, VerifiedKeyMaterial},
        ClusterKeySource, InMemoryCredentialRevocationStore, JwtKeySource,
    },
    crypto::{
        pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey},
        CryptoPolicy,
    },
    envelope::{self, EnvelopeVerification, RequestEnvelope, SignedEnvelope},
    node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key},
    proof::{
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
use parking_lot::Mutex;
use std::{
    io::Cursor,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

const DID: &str = "did:plc:3f7ldey5bivhfjapwqi2sljy";
const RESOURCE: &str = "model:qwen2.5-0.5b-instruct:main";
const SCHEMA: u64 = 0xe7339d5d26ab3076;

struct ModelBoundary {
    host: FederateHost,
    model: ModelService,
    server: SigningKey,
    transport: TransportConfig,
    keys: Arc<dyn JwtKeySource>,
    revocations: InMemoryCredentialRevocationStore,
    attempts: AtomicUsize,
    calls: AtomicUsize,
    authorization_errors: Mutex<Vec<String>>,
}

#[async_trait(?Send)]
impl RequestService for ModelBoundary {
    fn name(&self) -> &str {
        "model"
    }
    fn transport(&self) -> &TransportConfig {
        &self.transport
    }
    fn signing_key(&self) -> SigningKey {
        self.server.clone()
    }
    fn pq_signing_key(&self) -> Option<MlDsaSigningKey> {
        Some(derive_mesh_mldsa_key(&self.server))
    }
    fn jwt_key_source(&self) -> Option<Arc<dyn JwtKeySource>> {
        Some(self.keys.clone())
    }
    fn expected_audience(&self) -> Option<&str> {
        Some(self.host.as_str())
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
        let verified = super::super::federate_proof::verify_dispatch_fixture(
            ctx,
            "model",
            SCHEMA,
            body.bytes(),
            &self.host,
        )
        .await?;
        anyhow::ensure!(
            ctx.deferred_federate_credential().unwrap().0.sub == DID,
            "fixture grant denied"
        );
        let replay_key = hyprstream_rpc::proof::admission::ProofReplayKey {
            signer_thumbprint: verified.0.replay_thumbprint,
            request_id: proof.claims.request_id,
        };
        Ok((
            verified.0,
            verified.1,
            RESOURCE.into(),
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
        anyhow::ensure!(ctx.admitted_federate_operation(resource, operation));
        anyhow::ensure!(replay_key.request_id != [0; 16]);
        anyhow::ensure!(replay_expiry > chrono::Utc::now().timestamp() as u64);
        Ok(())
    }
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        _: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        // Enter the production Model authorization path for both allowed and
        // denied credentials. The outer dispatch PEP remains a fixture double.
        self.attempts.fetch_add(1, Ordering::SeqCst);
        if let Err(error) = ModelHandler::authorize(&self.model, ctx, RESOURCE, "infer").await {
            self.authorization_errors.lock().push(error.to_string());
            return Err(error);
        }
        let context = ctx
            .security_context()
            .ok_or_else(|| anyhow::anyhow!("verified Model security context missing"))?;
        let required =
            SecurityLabel::new(Level::Internal, Assurance::PqHybrid, CompartmentSet::EMPTY);
        anyhow::ensure!(
            context.can_access(&required),
            "hybrid Federate request proof did not meet the Model method floor"
        );
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"authorized".to_vec(), None))
    }
}

fn enrollment_policy(level: Level, assurance: Assurance) -> CompiledPolicy {
    let lattice = Lattice::new(LatticeVersion(1), []);
    let label = SecurityLabel::new(level, assurance, CompartmentSet::EMPTY);
    let policy = CompiledPolicy::new(TeMatrix::default(), &lattice)
        .with_enrollment([(DID.to_owned(), label)].into_iter().collect());
    let ed = SigningKey::from_bytes(&[81; 32]);
    let pq = ml_dsa_sk_from_seed(&[82; 32]);
    let pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&pq)).unwrap();
    let signed = sign_policy(
        &policy,
        &HybridPolicySigner {
            ed_sk: &ed,
            pq_sk: Some(&pq),
        },
    )
    .unwrap();
    PolicyLoader::new(HybridPolicyVerifier {
        ed_vk: &ed.verifying_key(),
        pq_vk: Some(&pq_vk),
        require_pq: true,
    })
    .with_approval(PolicyApproval {
        generation: policy.generation,
        approved_hash: policy.policy_hash().unwrap(),
    })
    .load(&signed)
    .unwrap()
}

async fn signed_model_request_bytes(
    token: &[u8],
    body: &[u8],
    now: i64,
    holder: &SigningKey,
    holder_pq: &MlDsaSigningKey,
    relay: &SigningKey,
    relay_pq: &MlDsaSigningKey,
    server: &SigningKey,
) -> Result<Vec<u8>> {
    let suite = hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
    let response = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let stream = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let binding = FederateRecipientBinding::from_recipients(
        Some(&response.public()),
        Some(&stream.public()),
        Some([61; 32]),
    )?;
    let holder_ed = holder.verifying_key().to_bytes();
    let holder_pq_bytes = ml_dsa_sk_to_vk_bytes(holder_pq);
    let kids = super::super::federate_source::session_primary_kids(&holder_ed, &holder_pq_bytes);
    let proof_signer = AuthenticatedHybridProofSigner::new(
        holder.clone(),
        kids[0].to_vec(),
        holder_pq.clone(),
        kids[1].to_vec(),
    )?;
    let proof = build_authenticated_hybrid_request_proof(
        &AuthenticatedRequestProofInput {
            service_domain: "model",
            credential: token,
            issued_at: now as u64,
            expires_at: now as u64 + 20,
            capnp_schema_id: SCHEMA,
            capnp_body: body,
            response_binding: None,
            federate_recipient_binding: Some(&binding),
        },
        &proof_signer,
    )?;
    let request = RequestEnvelope::new(body.to_vec())
        .with_jwt_token(std::str::from_utf8(token)?.to_owned())
        .with_proof_cwt(proof)
        .with_service_domain("model")?
        .with_client_dh_public([61; 32])
        .with_response_kem_recipient(response.public())
        .with_client_kem_public(stream.public())?;
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        relay,
        relay_pq,
        &derive_mesh_kem_recipient(server)?.public(),
    )?;
    let mut message = capnp::message::Builder::new_default();
    signed.write_to(
        &mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>(),
    );
    let mut bytes = vec![];
    capnp::serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

#[tokio::test]
async fn federate_minted_token_reaches_real_model_authorization() -> Result<()> {
    // Global crypto/PEP settings are process-lifetime: do not pollute other tests.
    const CHILD: &str = "HYPRSTREAM_MODEL_MINT_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for host in [HOST, "https://vm-a.example", "https://vm-b.example:8443"] {
            let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::oauth::state::federate_model_tests::federate_minted_token_reaches_real_model_authorization", "--nocapture"])
            .env(CHILD, host).status()?;
            anyhow::ensure!(
                status.success(),
                "isolated Model mint regression failed for {host}"
            );
        }
        return Ok(());
    }
    let host = FederateHost::new(&std::env::var(CHILD)?)?;
    struct FixturePep;
    impl hyprstream_rpc::auth::mac::MacDispatchPep for FixturePep {
        fn check(
            &self,
            _: &EnvelopeContext,
            _: &str,
            _: Option<&[u16]>,
        ) -> hyprstream_rpc::auth::mac::MacDecision {
            hyprstream_rpc::auth::mac::MacDecision::Permit
        }
    }
    hyprstream_rpc::auth::mac::install_mac_dispatch_pep(Arc::new(FixturePep));
    let mut methods = InMemoryMethodPolicy::new();
    methods.insert(
        "model",
        "1",
        SignaturePolicy::TokenBound {
            suite: CryptoSuite::Hybrid,
        },
    );
    assert!(set_global_method_policy(Box::new(methods)).is_ok());
    let server = SigningKey::from_bytes(&[72; 32]);
    let relay = SigningKey::from_bytes(&[73; 32]);
    let relay_pq = ml_dsa_sk_from_seed(&[74; 32]);
    let holder = SigningKey::from_bytes(&[75; 32]);
    let holder_pq = ml_dsa_sk_from_seed(&[76; 32]);
    let mut store = envelope::KeyedPqTrustStore::new();
    store.bind(
        relay.verifying_key().to_bytes(),
        &ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&relay_pq))?,
    );
    envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(Arc::new(store)),
    })?;
    let pair = super::tests::oauth_signing_pair(71);
    let (_dir, key_set) = super::tests::composite_signer_fixture(1, vec![pair.clone()]);
    let signer = super::HybridFederateSigner::new(key_set)?;
    let _ = hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(128),
    ));
    let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
        hyprstream_rpc::moq_event::MoqEventOrigin::new(),
    );
    // No infrastructure RPC is permitted on the Federate-admitted path. The
    // private nonexistent socket makes any accidental fallback fail closed.
    let transport = TransportConfig::ipc(_dir.path().join("unused-infrastructure.sock"));
    let policy = hyprstream_rpc_std::policy_client::PolicyClient::for_local_transport_bootstrap(
        &transport,
        server.clone(),
        server.verifying_key(),
        None,
    )?;
    let registry =
        hyprstream_rpc_std::registry_client::RegistryClient::for_local_transport_bootstrap(
            &transport,
            server.clone(),
            server.verifying_key(),
            None,
        )?;
    let model = ModelService::new(
        ModelServiceConfig::default(),
        server.clone(),
        policy,
        registry,
        transport.clone(),
        transport.clone(),
    )
    .await?;
    let service = ModelBoundary {
        host: host.clone(),
        model,
        server,
        transport,
        keys: Arc::new(
            ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
                .with_ca_composite_key(pair.ml_dsa().clone()),
        ),
        revocations: InMemoryCredentialRevocationStore::new(),
        attempts: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        authorization_errors: Mutex::new(Vec::new()),
    };
    let now = chrono::Utc::now().timestamp();
    let session = hyprstream_session_store::Session {
        host: host.as_str().into(),
        sid: "mint-model-sid".into(),
        account_id: "fixture-account".into(),
        subject: DID.into(),
        tenant: "tenant-a".into(),
        client_id: CLIENT.into(),
        resource: host.as_str().into(),
        scopes: vec![format!("infer:{RESOURCE}")],
        grant_revision: "revision-1".into(),
        ed_public: holder.verifying_key().to_bytes(),
        pq_public: ml_dsa_sk_to_vk_bytes(&holder_pq),
        generation: [7; 32],
        collision_inventory_id: [7; 32],
        proof_epoch: 1,
        expires_at: now + 120,
    };
    let hybrid_policy = enrollment_policy(Level::Internal, Assurance::PqHybrid);
    let hybrid_resolver = EnrollmentSubjectContextResolver::new(Arc::new(hybrid_policy.clone()));
    let allowed = federate_exchange::authorized_committed_claims_for_host(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &hybrid_resolver,
        &host,
        None,
    )?;
    let authority_clearance = hybrid_policy.clearance_for(DID).unwrap();
    assert_eq!(
        allowed.clearance,
        Some(CredentialClearance::from_label(authority_clearance)),
        "the JWT carries the authority clearance's level/compartments"
    );
    assert_eq!(
        hybrid_resolver.resolve(DID).unwrap().clearance().assurance,
        Assurance::Classical,
        "the generic OAuth resolver intentionally clamps its in-memory context"
    );
    let hybrid_context = allowed
        .security_context(VerifiedKeyMaterial::PqHybrid)
        .unwrap();
    assert_eq!(hybrid_context.clearance().assurance, Assurance::PqHybrid);
    let classical_context = allowed
        .security_context(VerifiedKeyMaterial::Classical)
        .unwrap();
    assert_eq!(
        classical_context.clearance().assurance,
        Assurance::Classical
    );
    let required = SecurityLabel::new(Level::Internal, Assurance::PqHybrid, CompartmentSet::EMPTY);
    assert!(hybrid_context.can_access(&required));
    assert!(!classical_context.can_access(&required));

    let insufficient = federate_exchange::authorized_committed_claims_for_host(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &EnrollmentSubjectContextResolver::new(Arc::new(enrollment_policy(
            Level::Public,
            Assurance::PqHybrid,
        ))),
        &host,
        None,
    )?;
    assert!(!insufficient
        .security_context(VerifiedKeyMaterial::PqHybrid)
        .unwrap()
        .can_access(&required));
    let missing =
        federate_exchange::committed_claims_for_host(&session, now, ISSUER, "opaque-sub", &host)?;
    assert!(missing.clearance.is_none());
    let public = federate_exchange::authorized_committed_claims_for_host(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &EnrollmentSubjectContextResolver::new(Arc::new(enrollment_policy(
            Level::Public,
            Assurance::PqHybrid,
        ))),
        &host,
        None,
    )?;
    let mut unknown = session.clone();
    unknown.subject = "did:plc:abcdefghijklmnopqrstuvwx".into();
    assert!(federate_exchange::authorized_committed_claims_for_host(
        &unknown,
        now,
        ISSUER,
        "opaque-sub",
        &EnrollmentSubjectContextResolver::new(Arc::new(enrollment_policy(
            Level::Internal,
            Assurance::PqHybrid,
        ))),
        &host,
        None,
    )
    .is_err());
    let body = hyprstream_rpc::serialize_message(|message| {
        message
            .init_root::<hyprstream_rpc::common_capnp::error_info::Builder>()
            .set_message("model-fixture");
    })?;
    for (index, (claims, expected_error)) in [
        (
            &missing,
            "authorization denied: inference caller has no verified MAC clearance",
        ),
        (
            &public,
            "authorization denied: inference caller clearance does not dominate engine label",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let token = signer.sign(claims).await?;
        let bytes = signed_model_request_bytes(
            token.as_bytes(),
            &body,
            now,
            &holder,
            &holder_pq,
            &relay,
            &relay_pq,
            &service.server,
        )
        .await?;
        process_request(
            &bytes,
            &service,
            EnvelopeVerification::AnySigner,
            &service.server,
            &envelope::InMemoryNonceCache::new(),
            CarrierContext::iroh(),
        )
        .await?;
        let errors = service.authorization_errors.lock();
        assert_eq!(
            errors.len(),
            index + 1,
            "authorization attempts={}",
            service.attempts.load(Ordering::SeqCst)
        );
        assert!(errors.last().unwrap().contains(expected_error));
    }
    let token = signer.sign(&allowed).await?;
    let bytes = signed_model_request_bytes(
        token.as_bytes(),
        &body,
        now,
        &holder,
        &holder_pq,
        &relay,
        &relay_pq,
        &service.server,
    )
    .await?;
    let _ = process_request(
        &bytes,
        &service,
        EnvelopeVerification::AnySigner,
        &service.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        service.attempts.load(Ordering::SeqCst),
        3,
        "all signed credentials must reach the real Model authorization handler"
    );
    for variant in 0..4 {
        let mut bad = allowed.clone();
        match variant {
            0 => {
                bad.iss = "https://another-vm.example".into();
                bad.aud = Some("https://another-vm.example".into());
            }
            1 => {
                bad.hs_profile = None;
                bad.hs_signer_suite_v1 = None;
                bad.hs_signer_pq_key_v1 = None;
                bad.hs_signer_inventory_v1 = None;
                bad.hs_source_issuer_v1 = None;
                bad.hs_source_subject_v1 = None;
            }
            2 => {
                bad.hs_session_authority_generation = Some(base64::Engine::encode(
                    &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                    [8; 32],
                ));
            }
            _ => bad.hs_signer_inventory_v1 = Some("malformed".into()),
        }
        let token = signer.sign(&bad).await?;
        let bytes = signed_model_request_bytes(
            token.as_bytes(),
            &body,
            now,
            &holder,
            &holder_pq,
            &relay,
            &relay_pq,
            &service.server,
        )
        .await?;
        process_request(
            &bytes,
            &service,
            EnvelopeVerification::AnySigner,
            &service.server,
            &envelope::InMemoryNonceCache::new(),
            CarrierContext::iroh(),
        )
        .await?;
        assert_eq!(
            service.attempts.load(Ordering::SeqCst),
            3,
            "{host:?} variant {variant} must fail before Model authorization"
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }
    let errors = service.authorization_errors.lock();
    assert_eq!(errors.len(), 2, "the two weaker clearances must be denied");
    Ok(())
}
