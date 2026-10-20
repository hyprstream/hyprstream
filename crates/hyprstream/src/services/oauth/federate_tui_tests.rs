//! W1.4 re-signed malformed-fixture negatives at the dispatch boundary:
//! the TUI-class federate credential through REAL envelope/JWT verification.
//!
//! The TUI credential never enters the svc.rs deferred federate branch: it
//! authenticates through normal JWT verification with `cnf.jwk` holder
//! binding. These fixtures prove intended-vs-forbidden services, marker
//! stripping/mixing/substitution, and wrong-proof-key rejection against the
//! real verifier, not only the reservation predicates. Policy action
//! admission and the outer dispatch PEP remain test doubles.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::super::{
    federate_exchange,
    federate_host::FederateSigner as _,
    federate_source::{FederateHost, CLIENT, HOST, ISSUER, TUI_PROFILE},
};
use crate::services::{
    generated::model_client::ModelHandler,
    model::{ModelService, ModelServiceConfig},
};
use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::{
    auth::ClusterKeySource,
    crypto::{
        pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes},
        CryptoPolicy,
    },
    envelope::{self, RequestEnvelope, SignedEnvelope},
    node_identity::derive_mesh_kem_recipient,
    service::{
        dispatch::process_request, Continuation, DecodedRequestBody, EnvelopeContext,
        RequestService,
    },
    transport::{carrier::CarrierContext, TransportConfig},
    ToCapnp as _,
};
use parking_lot::Mutex;
use std::{
    io::Cursor,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

const DID: &str = "did:plc:abcdefghijklmnopqrstuvwx";
const TUI_CLIENT: &str = "hyprstream-tui-staging";
const RESOURCE: &str = "model:qwen2.5-0.5b-instruct:main";

/// A `tui`-named serving boundary: the intended consumer of the TUI class.
/// Deferred federate credentials stay disabled (default): a TUI credential
/// must arrive through normal JWT verification.
struct TuiBoundary {
    host: FederateHost,
    server: SigningKey,
    transport: TransportConfig,
    keys: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore,
    attempts: AtomicUsize,
    calls: AtomicUsize,
    rejections: Mutex<Vec<String>>,
}

#[async_trait(?Send)]
impl RequestService for TuiBoundary {
    fn name(&self) -> &str {
        "tui"
    }
    fn transport(&self) -> &TransportConfig {
        &self.transport
    }
    fn signing_key(&self) -> SigningKey {
        self.server.clone()
    }
    fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
        Some(hyprstream_rpc::node_identity::derive_mesh_mldsa_key(
            &self.server,
        ))
    }
    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
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
    fn decode_request_body(&self, bytes: &[u8]) -> Result<DecodedRequestBody> {
        let message = capnp::serialize::read_message(
            &mut Cursor::new(bytes),
            hyprstream_rpc::service::body::bounded_reader_options(),
        )?;
        DecodedRequestBody::from_message(bytes.to_vec(), message, vec![1])
    }
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        _: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        // The TUI consumer check must pass on the published claims.
        match super::super::federate_proof::verify_tui_credential_claims(
            ctx.claims()
                .ok_or_else(|| anyhow::anyhow!("claims missing"))?,
            &self.host,
            TUI_CLIENT,
        ) {
            Ok(facts) if facts.subject == DID && facts.tenant == "tenant-a" => {}
            other => {
                let message = match other {
                    Err(error) => error.to_string(),
                    Ok(facts) => format!("unexpected subject {}", facts.subject),
                };
                self.rejections.lock().push(message);
                anyhow::bail!("TUI consumer check failed at dispatch");
            }
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"tui-authorized".to_vec(), None))
    }
}

async fn signed_tui_request_bytes(
    token: &[u8],
    body: &[u8],
    holder: &SigningKey,
    holder_pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    server: &SigningKey,
    service_domain: &str,
) -> Result<Vec<u8>> {
    let suite = hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
    let response = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let stream = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let request = RequestEnvelope::new(body.to_vec())
        .with_jwt_token(std::str::from_utf8(token)?.to_owned())
        .with_service_domain(service_domain)?
        .with_client_dh_public([61; 32])
        .with_response_kem_recipient(response.public())
        .with_client_kem_public(stream.public())?;
    // The holder signs the envelope directly with its anchored hybrid pair:
    // the TUI credential's cnf.jwk must match this envelope signer (svc.rs
    // R2 holder binding), and the envelope PQ key must match the trust-store
    // binding for that signer.
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        holder,
        holder_pq,
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
async fn w14_tui_credential_dispatch_intended_and_forbidden_services() -> Result<()> {
    // Global crypto settings are process-lifetime: isolate in a child.
    const CHILD: &str = "HYPRSTREAM_TUI_DISPATCH_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::oauth::state::federate_tui_tests::w14_tui_credential_dispatch_intended_and_forbidden_services", "--nocapture"])
            .env(CHILD, "1")
            .status()?;
        anyhow::ensure!(status.success(), "isolated TUI dispatch regression failed");
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
    assert!(
        hyprstream_rpc::proof::policy::set_global_method_policy(Box::new(
            hyprstream_rpc::proof::policy::InMemoryMethodPolicy::new(),
        ))
        .is_ok()
    );
    let host = FederateHost::new(HOST)?;
    let server = SigningKey::from_bytes(&[72; 32]);
    let holder = SigningKey::from_bytes(&[75; 32]);
    let holder_pq = ml_dsa_sk_from_seed(&[76; 32]);
    // The wrong-proof-keys substitute signer below. Both keys are bound in
    // the ONE trust store up front (install_verify_config is install-once):
    // the relay envelope is then properly trust-anchored, so its rejection
    // is exactly the cnf.jwk holder-binding mismatch, not an unanchored
    // envelope.
    let relay_key = SigningKey::from_bytes(&[93; 32]);
    let relay_pq = ml_dsa_sk_from_seed(&[94; 32]);
    let mut store = envelope::KeyedPqTrustStore::new();
    store.bind(
        holder.verifying_key().to_bytes(),
        &hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&holder_pq))?,
    );
    store.bind(
        relay_key.verifying_key().to_bytes(),
        &hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&relay_pq))?,
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
    // The real ModelService constructor requires the process-global MoQ
    // event origin (as in the www mint-to-Model fixture).
    let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
        hyprstream_rpc::moq_event::MoqEventOrigin::new(),
    );
    // The TUI credential is a NORMAL local at+jwt carrying `sid`; v16 §3.3
    // requires a session registry for local sid-carrying tokens. The www
    // deferred path never runs this check, the TUI path deliberately does.
    if hyprstream_rpc::auth::credential::global_session_registry().is_none() {
        let _ = hyprstream_rpc::auth::credential::set_global_session_registry(Arc::new(
            hyprstream_rpc::auth::credential::InMemorySessionRegistry::new(),
        ));
    }
    // The registry fails closed on UNKNOWN sessions (is_revoked -> true), so
    // the minted session record must be published before dispatch.
    let session_registry = hyprstream_rpc::auth::credential::global_session_registry()
        .expect("session registry just installed")
        .clone();
    let transport = TransportConfig::ipc(_dir.path().join("unused-tui-infrastructure.sock"));
    let tui = TuiBoundary {
        host: host.clone(),
        server: server.clone(),
        transport: transport.clone(),
        keys: Arc::new(
            ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
                .with_ca_composite_key(pair.ml_dsa().clone()),
        ),
        revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        attempts: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        rejections: Mutex::new(Vec::new()),
    };
    let now = chrono::Utc::now().timestamp();
    let session = hyprstream_session_store::Session {
        host: host.as_str().into(),
        sid: "tui-mint-sid".into(),
        account_id: "fixture-account".into(),
        subject: DID.into(),
        tenant: "tenant-a".into(),
        client_id: TUI_CLIENT.into(),
        resource: host.as_str().into(),
        scopes: vec!["connect:tui:session".into()],
        grant_revision: "revision-1".into(),
        ed_public: holder.verifying_key().to_bytes(),
        pq_public: ml_dsa_sk_to_vk_bytes(&holder_pq),
        generation: [7; 32],
        collision_inventory_id: [7; 32],
        proof_epoch: 1,
        expires_at: now + 120,
    };
    // Publish the minted session as active: the dispatch session check
    // rejects revoked AND unknown sids alike.
    hyprstream_rpc::auth::SessionRegistry::register_session(
        session_registry.as_ref(),
        hyprstream_rpc::auth::SessionKey::oidc(host.as_str(), &session.sid),
        hyprstream_rpc::auth::SessionState {
            subject: session.subject.clone(),
            tenant: session.tenant.clone(),
            kind: hyprstream_rpc::auth::SessionKind::Interactive,
            created_at: now,
            expires_at: session.expires_at,
            status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
            clearance_epoch: 0,
        },
    )
    .await?;
    let claims = federate_exchange::committed_claims_for_class(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        &host,
        Some(TUI_CLIENT),
    )?;
    assert!(claims.hs_tui_profile.as_deref() == Some(TUI_PROFILE));
    let body = hyprstream_rpc::serialize_message(|message| {
        message
            .init_root::<hyprstream_rpc::common_capnp::error_info::Builder>()
            .set_message("tui-fixture");
    })?;

    // Intended service: the clean TUI credential reaches the tui handler.
    let token = signer.sign(&claims).await?;
    let bytes =
        signed_tui_request_bytes(token.as_bytes(), &body, &holder, &holder_pq, &server, "tui")
            .await?;
    process_request(
        &bytes,
        &tui,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &tui.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(tui.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(tui.calls.load(Ordering::SeqCst), 1);

    // Wrong proof keys: the envelope signer differs from cnf.jwk. svc.rs
    // rejects the binding before the handler. The substitute signer was
    // trust-anchored for its own PQ half in the single store above, so the
    // failure is exactly the cnf.jwk mismatch, not an unanchored envelope.
    let bytes = signed_tui_request_bytes(
        token.as_bytes(),
        &body,
        &relay_key,
        &relay_pq,
        &server,
        "tui",
    )
    .await?;
    process_request(
        &bytes,
        &tui,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &tui.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(
        tui.attempts.load(Ordering::SeqCst),
        1,
        "cnf mismatch must fail before the handler"
    );
    assert_eq!(tui.calls.load(Ordering::SeqCst), 1);

    // Malformed re-signed fixtures against the same boundary. Two distinct
    // denial layers, both proven per variant:
    // * pre-handler (svc.rs): a www marker OR the reserved www client id
    //   forces the deferred federate branch, which this boundary never
    //   accepts — the handler is never invoked;
    // * handler: a validly signed TUI-shaped credential with stripped or
    //   wrong TUI markers reaches the handler and is denied by the TUI
    //   consumer check (`verify_tui_credential_claims`).
    for variant in 0..6 {
        let mut bad = claims.clone();
        let denied_pre_handler = match variant {
            0 => {
                bad.hs_tui_profile = None;
                false
            }
            1 => {
                bad.hs_tui_profile = None;
                bad.hs_tui_signer_pq_key_v1 = None;
                bad.hs_tui_source_issuer_v1 = None;
                bad.hs_tui_source_subject_v1 = None;
                false
            }
            2 => {
                bad.hs_profile = Some("federate-session-v1".into());
                true
            }
            3 => {
                bad.hs_tui_profile = Some("federate-session-v1".into());
                false
            }
            4 => {
                bad.client_id = Some(CLIENT.into());
                true
            }
            _ => {
                bad.hs_signer_inventory_v1 = Some(base64::Engine::encode(
                    &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                    [8; 32],
                ));
                true
            }
        };
        let token = signer.sign(&bad).await?;
        let bytes =
            signed_tui_request_bytes(token.as_bytes(), &body, &holder, &holder_pq, &server, "tui")
                .await?;
        let before = tui.attempts.load(Ordering::SeqCst);
        process_request(
            &bytes,
            &tui,
            hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
            &tui.server,
            &envelope::InMemoryNonceCache::new(),
            CarrierContext::iroh(),
        )
        .await?;
        if denied_pre_handler {
            assert_eq!(
                tui.attempts.load(Ordering::SeqCst),
                before,
                "variant {variant} must fail in the svc.rs deferred branch, before the tui handler"
            );
        } else {
            assert_eq!(
                tui.attempts.load(Ordering::SeqCst),
                before + 1,
                "variant {variant} must reach the handler and be denied by the TUI consumer check"
            );
        }
        assert_eq!(
            tui.calls.load(Ordering::SeqCst),
            1,
            "variant {variant} authorized a call"
        );
    }

    // Forbidden service: the same TUI credential cannot drive the Model
    // service. It carries no registry|model consumer authority (no scopes,
    // no MAC clearance), so the real Model authorization denies it.
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
    let model_boundary = ModelForbiddenBoundary {
        host: host.clone(),
        model,
        server: server.clone(),
        transport: transport.clone(),
        keys: Arc::new(
            ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
                .with_ca_composite_key(pair.ml_dsa().clone()),
        ),
        revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        attempts: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        authorization_errors: Mutex::new(Vec::new()),
    };
    let token = signer.sign(&claims).await?;
    // The envelope must name the model service domain, or dispatch denies
    // pre-handler on the domain mismatch and never exercises the real Model
    // authorization this phase exists to test.
    let bytes = signed_tui_request_bytes(
        token.as_bytes(),
        &body,
        &holder,
        &holder_pq,
        &server,
        "model",
    )
    .await?;
    process_request(
        &bytes,
        &model_boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &model_boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(model_boundary.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(model_boundary.calls.load(Ordering::SeqCst), 0);
    let errors = model_boundary.authorization_errors.lock();
    assert_eq!(
        errors.len(),
        1,
        "the TUI credential must be denied by real Model authorization"
    );
    Ok(())
}

/// The Model service boundary for the forbidden-service negative: the real
/// Model authorization path runs; the outer dispatch PEP stays a fixture.
struct ModelForbiddenBoundary {
    host: FederateHost,
    model: ModelService,
    server: SigningKey,
    transport: TransportConfig,
    keys: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore,
    attempts: AtomicUsize,
    calls: AtomicUsize,
    authorization_errors: Mutex<Vec<String>>,
}

#[async_trait(?Send)]
impl RequestService for ModelForbiddenBoundary {
    fn name(&self) -> &str {
        "model"
    }
    fn transport(&self) -> &TransportConfig {
        &self.transport
    }
    fn signing_key(&self) -> SigningKey {
        self.server.clone()
    }
    fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
        Some(hyprstream_rpc::node_identity::derive_mesh_mldsa_key(
            &self.server,
        ))
    }
    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
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
    fn decode_request_body(&self, bytes: &[u8]) -> Result<DecodedRequestBody> {
        let message = capnp::serialize::read_message(
            &mut Cursor::new(bytes),
            hyprstream_rpc::service::body::bounded_reader_options(),
        )?;
        DecodedRequestBody::from_message(bytes.to_vec(), message, vec![1])
    }
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        _: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        if let Err(error) = ModelHandler::authorize(&self.model, ctx, RESOURCE, "infer").await {
            self.authorization_errors.lock().push(error.to_string());
            return Err(error);
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"forbidden".to_vec(), None))
    }
}
