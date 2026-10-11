//! Real production mint -> encrypted RPC/proof verification -> Model MAC gate.
//! Policy action admission/replay and the outer dispatch PEP are explicit test
//! doubles; this is not a running-service or inference-engine E2E receipt.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::super::{
    federate_exchange,
    federate_host::FederateSigner,
    federate_source::{CLIENT, HOST, ISSUER},
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
    },
};
use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::{
    auth::{ClusterKeySource, InMemoryCredentialRevocationStore, JwtKeySource},
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
    model: ModelService,
    server: SigningKey,
    transport: TransportConfig,
    keys: Arc<dyn JwtKeySource>,
    revocations: InMemoryCredentialRevocationStore,
    attempts: AtomicUsize,
    calls: AtomicUsize,
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
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        _: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        // This is the actual production Model authorization entry, not a
        // duplicated label comparison or a fabricated authenticated context.
        self.attempts.fetch_add(1, Ordering::SeqCst);
        ModelHandler::authorize(&self.model, ctx, RESOURCE, "infer").await?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"authorized".to_vec(), None))
    }
}

fn resolver(level: Level) -> EnrollmentSubjectContextResolver {
    let lattice = Lattice::new(LatticeVersion(1), []);
    let label = SecurityLabel::new(level, Assurance::Unverified, CompartmentSet::EMPTY);
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
    let policy = PolicyLoader::new(HybridPolicyVerifier {
        ed_vk: &ed.verifying_key(),
        pq_vk: Some(&pq_vk),
        require_pq: true,
    })
    .with_approval(PolicyApproval {
        generation: policy.generation,
        approved_hash: policy.policy_hash().unwrap(),
    })
    .load(&signed)
    .unwrap();
    EnrollmentSubjectContextResolver::new(Arc::new(policy))
}

#[tokio::test]
async fn federate_minted_token_reaches_real_model_authorization() -> Result<()> {
    // Global crypto/PEP settings are process-lifetime: do not pollute other tests.
    const CHILD: &str = "HYPRSTREAM_MODEL_MINT_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::oauth::state::federate_model_tests::federate_minted_token_reaches_real_model_authorization", "--nocapture"])
            .env(CHILD, "1").status()?;
        anyhow::ensure!(status.success(), "isolated Model mint regression failed");
        return Ok(());
    }
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
        model,
        server,
        transport,
        keys: Arc::new(
            ClusterKeySource::new(*pair.ed25519(), HOST.into())
                .with_ca_composite_key(pair.ml_dsa().clone()),
        ),
        revocations: InMemoryCredentialRevocationStore::new(),
        attempts: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    };
    let now = chrono::Utc::now().timestamp();
    let session = hyprstream_session_store::Session {
        host: HOST.into(),
        sid: "mint-model-sid".into(),
        account_id: "fixture-account".into(),
        subject: DID.into(),
        tenant: "tenant-a".into(),
        client_id: CLIENT.into(),
        resource: HOST.into(),
        scopes: vec![format!("infer:{RESOURCE}")],
        grant_revision: "revision-1".into(),
        ed_public: holder.verifying_key().to_bytes(),
        pq_public: ml_dsa_sk_to_vk_bytes(&holder_pq),
        generation: [7; 32],
        collision_inventory_id: [7; 32],
        proof_epoch: 1,
        expires_at: now + 120,
    };
    let allowed = federate_exchange::authorized_committed_claims(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &resolver(Level::Internal),
    )?;
    let insufficient = federate_exchange::authorized_committed_claims(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &resolver(Level::Public),
    )?;
    let missing = federate_exchange::committed_claims(&session, now, ISSUER, "opaque-sub")?;
    let mut unknown = session.clone();
    unknown.subject = "did:plc:abcdefghijklmnopqrstuvwx".into();
    assert!(federate_exchange::authorized_committed_claims(
        &unknown,
        now,
        ISSUER,
        "opaque-sub",
        &resolver(Level::Internal)
    )
    .is_err());
    let body = hyprstream_rpc::serialize_message(|message| {
        message
            .init_root::<hyprstream_rpc::common_capnp::error_info::Builder>()
            .set_message("model-fixture");
    })?;
    for (index, (claims, expected_calls)) in [(missing, 0), (insufficient, 0), (allowed, 1)]
        .into_iter()
        .enumerate()
    {
        let token = signer.sign(&claims).await?;
        let suite = hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
        let response = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
        let stream = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
        let binding = FederateRecipientBinding::from_recipients(
            Some(&response.public()),
            Some(&stream.public()),
            Some([61; 32]),
        )?;
        let kids = super::super::federate_source::session_primary_kids(
            &session.ed_public,
            &session.pq_public,
        );
        let proof_signer = AuthenticatedHybridProofSigner::new(
            holder.clone(),
            kids[0].to_vec(),
            holder_pq.clone(),
            kids[1].to_vec(),
        )?;
        let proof = build_authenticated_hybrid_request_proof(
            &AuthenticatedRequestProofInput {
                service_domain: "model",
                credential: token.as_bytes(),
                issued_at: now as u64,
                expires_at: now as u64 + 20,
                capnp_schema_id: SCHEMA,
                capnp_body: &body,
                response_binding: None,
                federate_recipient_binding: Some(&binding),
            },
            &proof_signer,
        )?;
        let request = RequestEnvelope::new(body.clone())
            .with_jwt_token(token)
            .with_proof_cwt(proof)
            .with_service_domain("model")?
            .with_client_dh_public([61; 32])
            .with_response_kem_recipient(response.public())
            .with_client_kem_public(stream.public())?;
        let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
            request,
            &relay,
            &relay_pq,
            &derive_mesh_kem_recipient(&service.server)?.public(),
        )?;
        let mut message = capnp::message::Builder::new_default();
        signed.write_to(
            &mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>(),
        );
        let mut bytes = vec![];
        capnp::serialize::write_message(&mut bytes, &message)?;
        let _ = process_request(
            &bytes,
            &service,
            EnvelopeVerification::AnySigner,
            &service.server,
            &envelope::InMemoryNonceCache::new(),
            CarrierContext::iroh(),
        )
        .await?;
        assert_eq!(service.calls.load(Ordering::SeqCst), expected_calls);
        assert_eq!(
            service.attempts.load(Ordering::SeqCst),
            index + 1,
            "fixture must reach the real Model MAC gate, not pass vacuously at an earlier denial"
        );
    }
    Ok(())
}
