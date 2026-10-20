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
    // The deployment-configured class reservation, as startup installs it:
    // stripped class-shaped credentials are classified by pin even without
    // markers, and the dispatch fence denies them pre-handler.
    hyprstream_rpc::auth::install_federate_tui_class_client(Some(TUI_CLIENT))?;
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

    // Malformed re-signed fixtures against the same boundary. Distinct
    // denial layers, each proven per variant:
    // * pre-handler deferred branch (svc.rs): a www marker OR the reserved
    //   www client id, which this boundary never accepts;
    // * pre-handler class fence (dispatch.rs): a marker-stripped but
    //   class-pinned credential (incomplete marker set is never legitimate);
    // * handler: a complete-marker-set credential with a wrong profile
    //   string reaches the handler and is denied by the TUI consumer check
    //   (`verify_tui_credential_claims`).
    for variant in 0..6 {
        let mut bad = claims.clone();
        let denied_pre_handler = match variant {
            0 => {
                bad.hs_tui_profile = None;
                // Classified by the authority-owned pin (markers stripped,
                // client_id intact) and denied by the dispatch fence: the
                // marker set is incomplete and the mint never produces that.
                true
            }
            1 => {
                bad.hs_tui_profile = None;
                bad.hs_tui_signer_pq_key_v1 = None;
                bad.hs_tui_source_issuer_v1 = None;
                bad.hs_tui_source_subject_v1 = None;
                true
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

    // Forbidden services: a production-shaped TUI credential — carrying the
    // subject's enrolled MAC clearance, as the production mint attaches — can
    // never reach Registry/Model dispatch. The svc.rs class fence (any TUI
    // marker + service name != "tui") denies BEFORE the handler, so the
    // denial is attributable to the credential class alone: never to missing
    // clearance, missing Policy grants, or an unavailable Policy.
    let mut forbidden_claims = claims.clone();
    forbidden_claims.clearance = Some(hyprstream_rpc::auth::mac::CredentialClearance::from_label(
        hyprstream_rpc::auth::mac::SecurityLabel::new(
            hyprstream_rpc::auth::mac::Level::Internal,
            hyprstream_rpc::auth::mac::Assurance::PqHybrid,
            hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
        ),
    ));
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
    let model = Arc::new(model);
    for service_name in ["model", "registry"] {
        let boundary = ModelForbiddenBoundary {
            name: service_name,
            host: host.clone(),
            model: Arc::clone(&model),
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
        // The complete-marker credential AND the all-stripped class-pinned
        // shape (the W1.4 all-stripped forbidden-service case) both deny.
        let mut stripped_claims = forbidden_claims.clone();
        stripped_claims.hs_tui_profile = None;
        stripped_claims.hs_tui_signer_pq_key_v1 = None;
        stripped_claims.hs_tui_source_issuer_v1 = None;
        stripped_claims.hs_tui_source_subject_v1 = None;
        for case in [&forbidden_claims, &stripped_claims] {
            let token = signer.sign(case).await?;
            // The envelope names the forbidden service domain, so only the
            // class fence stands between the credential and that handler.
            let bytes = signed_tui_request_bytes(
                token.as_bytes(),
                &body,
                &holder,
                &holder_pq,
                &server,
                service_name,
            )
            .await?;
            process_request(
                &bytes,
                &boundary,
                hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                &boundary.server,
                &envelope::InMemoryNonceCache::new(),
                CarrierContext::iroh(),
            )
            .await?;
            assert_eq!(
                boundary.attempts.load(Ordering::SeqCst),
                0,
                "the class fence must deny the TUI credential before the {service_name} handler"
            );
            assert_eq!(boundary.calls.load(Ordering::SeqCst), 0);
            assert!(
                boundary.authorization_errors.lock().is_empty(),
                "the {service_name} authorization path must never run"
            );
        }
    }
    Ok(())
}

/// The Model service boundary for the forbidden-service negative: the real
/// Model authorization path runs; the outer dispatch PEP stays a fixture.
struct ModelForbiddenBoundary {
    name: &'static str,
    host: FederateHost,
    model: Arc<ModelService>,
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
        self.name
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
        if let Err(error) = ModelHandler::authorize(&*self.model, ctx, RESOURCE, "infer").await {
            self.authorization_errors.lock().push(error.to_string());
            return Err(error);
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((b"forbidden".to_vec(), None))
    }
}

/// Permit-all MAC dispatch PEP for the consumer fixtures (the outer dispatch
/// PEP stays a fixture; the class gate and Casbin layer are what's tested).
struct PermitAllPep;
impl hyprstream_rpc::auth::mac::MacDispatchPep for PermitAllPep {
    fn check(
        &self,
        _: &EnvelopeContext,
        _: &str,
        _: Option<&[u16]>,
    ) -> hyprstream_rpc::auth::mac::MacDecision {
        hyprstream_rpc::auth::mac::MacDecision::Permit
    }
}

/// Process-global fixture setup shared by the real-service consumer tests.
/// Every global is install-once; the tests re-exec themselves for isolation.
/// All envelope signer keys used by the fixture are trust-anchored here so
/// rejections are attributable to the layer under test, never to an
/// unanchored envelope.
fn install_real_service_fixture_globals(
    signers: &[(&SigningKey, &hyprstream_rpc::crypto::pq::MlDsaSigningKey)],
) -> Result<()> {
    hyprstream_rpc::auth::mac::install_mac_dispatch_pep(Arc::new(PermitAllPep));
    assert!(
        hyprstream_rpc::proof::policy::set_global_method_policy(Box::new(
            hyprstream_rpc::proof::policy::InMemoryMethodPolicy::new(),
        ))
        .is_ok()
    );
    let mut store = envelope::KeyedPqTrustStore::new();
    for (signer, signer_pq) in signers {
        store.bind(
            signer.verifying_key().to_bytes(),
            &hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(signer_pq))?,
        );
    }
    envelope::install_verify_config(envelope::EnvelopeVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(Arc::new(store)),
    })?;
    let _ = hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(128),
    ));
    // TuiService consumes the process-global revocation/session authorities
    // (the RequestService defaults), exactly as in production.
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
    if hyprstream_rpc::auth::credential::global_session_registry().is_none() {
        let _ = hyprstream_rpc::auth::credential::set_global_session_registry(Arc::new(
            hyprstream_rpc::auth::credential::InMemorySessionRegistry::new(),
        ));
    }
    Ok(())
}

/// Mint TUI-class claims for `scopes`, publishing the session exactly as
/// the production mint path does (federate_host exchange). Returns the
/// unsigned claims so fixtures can mutate and re-sign them.
async fn mint_real_service_claims(
    host: &FederateHost,
    holder: &SigningKey,
    holder_pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    sid: &str,
    scopes: &[&str],
) -> Result<hyprstream_rpc::auth::Claims> {
    let now = chrono::Utc::now().timestamp();
    let session = hyprstream_session_store::Session {
        host: host.as_str().into(),
        sid: sid.into(),
        account_id: "fixture-account".into(),
        subject: DID.into(),
        tenant: "tenant-a".into(),
        client_id: TUI_CLIENT.into(),
        resource: host.as_str().into(),
        scopes: scopes.iter().map(ToString::to_string).collect(),
        grant_revision: "revision-1".into(),
        ed_public: holder.verifying_key().to_bytes(),
        pq_public: ml_dsa_sk_to_vk_bytes(holder_pq),
        generation: [7; 32],
        collision_inventory_id: [7; 32],
        proof_epoch: 1,
        expires_at: now + 120,
    };
    let registry = hyprstream_rpc::auth::credential::global_session_registry()
        .expect("session registry installed")
        .clone();
    let key = hyprstream_rpc::auth::SessionKey::oidc(host.as_str(), sid);
    if registry.session_state(&key).await.is_none() {
        hyprstream_rpc::auth::SessionRegistry::register_session(
            registry.as_ref(),
            key,
            hyprstream_rpc::auth::SessionState {
                subject: DID.into(),
                tenant: "tenant-a".into(),
                kind: hyprstream_rpc::auth::SessionKind::Interactive,
                created_at: now,
                expires_at: session.expires_at,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await?;
    }
    Ok(federate_exchange::committed_claims_for_class(
        &session,
        now,
        ISSUER,
        "opaque-sub",
        host,
        Some(TUI_CLIENT),
    )?)
}

/// Mint and sign a TUI-class credential for `scopes`.
async fn signed_real_service_credential(
    signer: &super::HybridFederateSigner,
    host: &FederateHost,
    holder: &SigningKey,
    holder_pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    sid: &str,
    scopes: &[&str],
) -> Result<String> {
    let claims = mint_real_service_claims(host, holder, holder_pq, sid, scopes).await?;
    signer.sign(&claims).await
}

/// The Connect request body the real TuiService decodes and dispatches.
fn connect_request_body() -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|message| {
        let mut req =
            message.init_root::<hyprstream_rpc_std::tui_capnp::tui_request::Builder<'_>>();
        req.set_id(1);
        let mut connect = req.reborrow().init_connect();
        connect.set_session_id(0);
        connect.set_display_mode(hyprstream_rpc_std::tui_capnp::DisplayMode::Ansi);
        connect.set_cols(80);
        connect.set_rows(24);
    })
}

/// W1.3 consumer proof: the TUI-class federate credential drives the REAL
/// TuiService dispatch end to end — the class gate and scope map run in the
/// production handler, observed through real service state.
#[tokio::test]
async fn w13_tui_credential_drives_the_real_tui_service() -> Result<()> {
    const CHILD: &str = "HYPRSTREAM_TUI_REAL_SERVICE_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::oauth::state::federate_tui_tests::w13_tui_credential_drives_the_real_tui_service", "--nocapture"])
            .env(CHILD, "1")
            .status()?;
        anyhow::ensure!(
            status.success(),
            "isolated TUI real-service regression failed"
        );
        return Ok(());
    }
    let host = FederateHost::new(HOST)?;
    let server = SigningKey::from_bytes(&[72; 32]);
    let holder = SigningKey::from_bytes(&[75; 32]);
    let holder_pq = ml_dsa_sk_from_seed(&[76; 32]);
    // The deployment-configured class reservation, as startup installs it:
    // the consumer gate classifies marker-stripped credentials by pin even
    // on a TuiService instance the class was not wired into.
    hyprstream_rpc::auth::install_federate_tui_class_client(Some(TUI_CLIENT))?;
    install_real_service_fixture_globals(&[(&holder, &holder_pq)])?;
    let pair = super::tests::oauth_signing_pair(71);
    let (_dir, key_set) = super::tests::composite_signer_fixture(1, vec![pair.clone()]);
    let signer = super::HybridFederateSigner::new(key_set)?;
    let transport = TransportConfig::ipc(_dir.path().join("unused-tui-infrastructure.sock"));
    let keys: Arc<dyn hyprstream_rpc::auth::JwtKeySource> = Arc::new(
        ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
            .with_ca_composite_key(pair.ml_dsa().clone()),
    );
    let build_service = |client: Option<&str>| {
        crate::tui::service::TuiService::new(
            Arc::new(tokio::sync::RwLock::new(crate::tui::TuiState::new(
                80, 24, 1000,
            ))),
            transport.clone(),
            server.clone(),
        )
        .with_expected_audience(host.as_str().to_owned())
        .with_jwt_key_source(keys.clone())
        .with_federate_tui_client(client.map(str::to_owned))
    };
    let tui = build_service(Some(TUI_CLIENT));
    let tui_unconfigured = build_service(None);

    let full = signed_real_service_credential(
        &signer,
        &host,
        &holder,
        &holder_pq,
        "tui-real-sid",
        &[
            "connect:tui:session",
            "query:tui:session",
            "write:tui:session",
        ],
    )
    .await?;
    let watcher = signed_real_service_credential(
        &signer,
        &host,
        &holder,
        &holder_pq,
        "tui-watcher-sid",
        &["query:tui:session"],
    )
    .await?;
    let body = connect_request_body()?;

    // The Connect continuation spawns the frame loop via spawn_local.
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            // Negative: a query-only credential cannot attach (the class gate
            // denies before the handler runs — no session is created).
            let bytes = signed_tui_request_bytes(
                watcher.as_bytes(),
                &body,
                &holder,
                &holder_pq,
                &server,
                "tui",
            )
            .await?;
            process_request(
                &bytes,
                &tui,
                hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                &server,
                &envelope::InMemoryNonceCache::new(),
                CarrierContext::iroh(),
            )
            .await?;
            assert_eq!(
                tui.test_session_count().await,
                0,
                "query-only TUI credential must not attach a viewer"
            );

            // Negative: the same full-scope credential against a service
            // without the configured class default-denies (W1.1 invariant).
            let bytes = signed_tui_request_bytes(
                full.as_bytes(),
                &body,
                &holder,
                &holder_pq,
                &server,
                "tui",
            )
            .await?;
            process_request(
                &bytes,
                &tui_unconfigured,
                hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                &server,
                &envelope::InMemoryNonceCache::new(),
                CarrierContext::iroh(),
            )
            .await?;
            assert_eq!(
                tui_unconfigured.test_session_count().await,
                0,
                "unconfigured TUI class must default-deny at the consumer"
            );

            // Negative: the ALL-STRIPPED class-pinned credential against the
            // unconfigured service. The global reservation pin classifies it
            // even without markers and even without the service wiring, and
            // the dispatch fence denies the incomplete marker set outright
            // (the gate would also deny: no configured class to verify).
            let mut stripped = mint_real_service_claims(
                &host,
                &holder,
                &holder_pq,
                "tui-unconfigured-sid",
                &["connect:tui:session"],
            )
            .await?;
            stripped.hs_tui_profile = None;
            stripped.hs_tui_signer_pq_key_v1 = None;
            stripped.hs_tui_source_issuer_v1 = None;
            stripped.hs_tui_source_subject_v1 = None;
            let token = signer.sign(&stripped).await?;
            let bytes = signed_tui_request_bytes(
                token.as_bytes(),
                &body,
                &holder,
                &holder_pq,
                &server,
                "tui",
            )
            .await?;
            process_request(
                &bytes,
                &tui_unconfigured,
                hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                &server,
                &envelope::InMemoryNonceCache::new(),
                CarrierContext::iroh(),
            )
            .await?;
            assert_eq!(
                tui_unconfigured.test_session_count().await,
                0,
                "all-stripped TUI-class credential must deny on an unconfigured service"
            );

            // Negatives through the REAL service: re-signed malformed
            // fixtures (the class pin and gate deny each; no session is
            // ever created). The session for this sid is registered, so the
            // denial is attributable to the class gate, not the session
            // check.
            let base_claims = mint_real_service_claims(
                &host,
                &holder,
                &holder_pq,
                "tui-real-sid",
                &["connect:tui:session"],
            )
            .await?;
            let mut variants: Vec<hyprstream_rpc::auth::Claims> = Vec::new();
            // All TUI markers stripped: the class pin (iss + configured
            // client_id) still classifies the credential, and the gate
            // denies the incomplete marker set.
            let mut all_stripped = base_claims.clone();
            all_stripped.hs_tui_profile = None;
            all_stripped.hs_tui_signer_pq_key_v1 = None;
            all_stripped.hs_tui_source_issuer_v1 = None;
            all_stripped.hs_tui_source_subject_v1 = None;
            variants.push(all_stripped);
            // Each single-marker omission.
            for omit in 0..4 {
                let mut bad = base_claims.clone();
                match omit {
                    0 => bad.hs_tui_profile = None,
                    1 => bad.hs_tui_signer_pq_key_v1 = None,
                    2 => bad.hs_tui_source_issuer_v1 = None,
                    _ => bad.hs_tui_source_subject_v1 = None,
                }
                variants.push(bad);
            }
            // Mixed: a www marker on a TUI-shaped credential (denied at the
            // svc.rs deferred branch before the handler).
            let mut mixed = base_claims.clone();
            mixed.hs_profile = Some("federate-session-v1".into());
            variants.push(mixed);
            // Wrong client id with the complete marker set.
            let mut wrong_client = base_claims.clone();
            wrong_client.client_id = Some("other-client".into());
            variants.push(wrong_client);
            for (index, bad) in variants.iter().enumerate() {
                let token = signer.sign(bad).await?;
                let bytes = signed_tui_request_bytes(
                    token.as_bytes(),
                    &body,
                    &holder,
                    &holder_pq,
                    &server,
                    "tui",
                )
                .await?;
                process_request(
                    &bytes,
                    &tui,
                    hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                    &server,
                    &envelope::InMemoryNonceCache::new(),
                    CarrierContext::iroh(),
                )
                .await?;
                assert_eq!(
                    tui.test_session_count().await,
                    0,
                    "malformed TUI credential variant {index} attached a viewer"
                );
            }

            // Positive: the full-scope TUI credential attaches a viewer to a
            // real session on the real TuiService.
            let bytes = signed_tui_request_bytes(
                full.as_bytes(),
                &body,
                &holder,
                &holder_pq,
                &server,
                "tui",
            )
            .await?;
            process_request(
                &bytes,
                &tui,
                hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
                &server,
                &envelope::InMemoryNonceCache::new(),
                CarrierContext::iroh(),
            )
            .await?;
            assert_eq!(
                tui.test_session_count().await,
                1,
                "the TUI-class credential must attach a viewer session"
            );
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}

/// A boundary wrapping the REAL PolicyService, recording check results so
/// the dispatch fixtures can observe what the inner service decided.
struct PolicyBoundary {
    host: FederateHost,
    policy: crate::services::policy::PolicyService,
    server: SigningKey,
    transport: TransportConfig,
    keys: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore,
    attempts: AtomicUsize,
    check_results: Mutex<Vec<bool>>,
}

#[async_trait(?Send)]
impl RequestService for PolicyBoundary {
    fn name(&self) -> &str {
        "policy"
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
    fn accept_delegated_bearer(&self, signer_pubkey: &[u8; 32]) -> bool {
        // Delegate the mediator-admission decision to the real PolicyService.
        RequestService::accept_delegated_bearer(&self.policy, signer_pubkey)
    }
    fn credential_revocation_store(
        &self,
    ) -> Option<&dyn hyprstream_rpc::auth::CredentialRevocationStore> {
        Some(&self.revocations)
    }
    fn decode_request_body(&self, bytes: &[u8]) -> Result<DecodedRequestBody> {
        RequestService::decode_request_body(&self.policy, bytes)
    }
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        body: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let (bytes, cont) = self.policy.handle_request(ctx, body).await?;
        // Record the check verdict from the raw policy_response payload.
        let message = capnp::serialize::read_message(
            &mut Cursor::new(&bytes),
            hyprstream_rpc::service::body::bounded_reader_options(),
        )?;
        let response =
            message.get_root::<hyprstream_rpc_std::policy_capnp::policy_response::Reader<'_>>()?;
        if let Ok(hyprstream_rpc_std::policy_capnp::policy_response::Which::CheckResult(allowed)) =
            response.which()
        {
            self.check_results.lock().push(allowed);
        }
        Ok((bytes, cont))
    }
}

/// Build a policy_request body: `check` leaf or the `issueToken` leaf.
fn policy_request_body(check: bool, resource: &str, operation: &str) -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|message| {
        let mut req =
            message.init_root::<hyprstream_rpc_std::policy_capnp::policy_request::Builder<'_>>();
        req.set_id(1);
        if check {
            let mut c = req.reborrow().init_check();
            c.set_subject("did:plc:abcdefghijklmnopqrstuvwx");
            c.set_domain("tenant-a");
            c.set_resource(resource);
            c.set_operation(operation);
        } else {
            req.reborrow().init_issue_token();
        }
    })
}

/// Sign an envelope carrying the TUI credential as a delegated bearer (the
/// TUI service's authorize path shape), or — when `direct` — as the direct
/// envelope JWT.
async fn signed_policy_request_bytes(
    token: &str,
    body: &[u8],
    mediator: &SigningKey,
    mediator_pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    server: &SigningKey,
    direct: bool,
) -> Result<Vec<u8>> {
    let suite = hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768;
    let response = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let stream = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(suite)?;
    let request = RequestEnvelope::new(body.to_vec()).with_service_domain("policy")?;
    let request = if direct {
        request.with_jwt_token(token.to_owned())
    } else {
        request.with_delegation_token(token.to_owned())
    };
    let request = request
        .with_client_dh_public([61; 32])
        .with_response_kem_recipient(response.public())
        .with_client_kem_public(stream.public())?;
    let signed = SignedEnvelope::new_signed_encrypted_mesh_kem(
        request,
        mediator,
        mediator_pq,
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

/// W1.3 mediated authorization: a TUI-class credential reaches Policy's
/// read-only check ONLY as the delegated subject of an enrolled service
/// mediator (the TUI service's authorize path) — never directly, never for
/// Policy's mutation/issuance leaves, never via an unenrolled mediator.
#[tokio::test]
async fn w13_tui_credential_reaches_policy_check_only_delegated() -> Result<()> {
    const CHILD: &str = "HYPRSTREAM_TUI_POLICY_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::oauth::state::federate_tui_tests::w13_tui_credential_reaches_policy_check_only_delegated", "--nocapture"])
            .env(CHILD, "1")
            .status()?;
        anyhow::ensure!(status.success(), "isolated TUI-to-Policy regression failed");
        return Ok(());
    }
    // The deployment-configured class reservation, as startup installs it.
    hyprstream_rpc::auth::install_federate_tui_class_client(Some(TUI_CLIENT))?;
    let host = FederateHost::new(HOST)?;
    let server = SigningKey::from_bytes(&[72; 32]);
    let holder = SigningKey::from_bytes(&[75; 32]);
    let holder_pq = ml_dsa_sk_from_seed(&[76; 32]);
    // The mediator (the tui service's own key pair).
    let mediator = SigningKey::from_bytes(&[77; 32]);
    let mediator_pq = ml_dsa_sk_from_seed(&[78; 32]);
    // The unenrolled-mediator negative anchors its envelope too: the denial
    // must come from delegated-bearer admission, never from an unanchored
    // envelope.
    let stranger = SigningKey::from_bytes(&[90; 32]);
    let stranger_pq = ml_dsa_sk_from_seed(&[91; 32]);
    install_real_service_fixture_globals(&[
        (&holder, &holder_pq),
        (&mediator, &mediator_pq),
        (&stranger, &stranger_pq),
    ])?;
    let pair = super::tests::oauth_signing_pair(71);
    let (_dir, key_set) = super::tests::composite_signer_fixture(1, vec![pair.clone()]);
    let signer = super::HybridFederateSigner::new(key_set)?;
    // The mediator is an enrolled service in the global trust store.
    hyprstream_service::global_trust_store().insert(
        mediator.verifying_key(),
        hyprstream_service::Attestation {
            scopes: ["tui".to_owned()].into_iter().collect(),
            subject: None,
            jwt: None,
            expires_at: 0,
            attested_by: None,
        },
    );
    // The real PolicyService with one exact grant for the DID.
    let manager = Arc::new(crate::auth::PolicyManager::new_in_memory().await?);
    manager
        .add_policy_with_domain(DID, "tenant-a", "tui:session", "write", "allow")
        .await?;
    let git_root = tempfile::tempdir()?;
    let git2db = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(git_root.path()).await?,
    ));
    let policy_transport = TransportConfig::ipc(_dir.path().join("policy-fixture.sock"));
    let policy = crate::services::policy::PolicyService::new(
        manager,
        Arc::new(server.clone()),
        crate::config::TokenConfig::default(),
        git2db,
        policy_transport.clone(),
    )
    .with_jwt_key_source(Arc::new(
        ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
            .with_ca_composite_key(pair.ml_dsa().clone()),
    ))
    .with_default_audience(host.as_str().to_owned());
    let boundary = PolicyBoundary {
        host: host.clone(),
        policy,
        server: server.clone(),
        transport: policy_transport,
        keys: Arc::new(
            ClusterKeySource::new(*pair.ed25519(), host.as_str().into())
                .with_ca_composite_key(pair.ml_dsa().clone()),
        ),
        revocations: hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        attempts: AtomicUsize::new(0),
        check_results: Mutex::new(Vec::new()),
    };
    let token = signed_real_service_credential(
        &signer,
        &host,
        &holder,
        &holder_pq,
        "tui-policy-sid",
        &["write:tui:session"],
    )
    .await?;

    // Positive: delegated check for the granted TUI operation.
    let body = policy_request_body(true, "tui:session", "write")?;
    let bytes =
        signed_policy_request_bytes(&token, &body, &mediator, &mediator_pq, &server, false).await?;
    process_request(
        &bytes,
        &boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(boundary.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        boundary.check_results.lock().as_slice(),
        &[true],
        "the granted TUI operation must pass the real Policy check"
    );

    // The real check runs: a non-granted resource returns false (the fence
    // admits the check; Casbin decides it).
    let body = policy_request_body(true, "model:qwen2.5-0.5b-instruct:main", "infer")?;
    let bytes =
        signed_policy_request_bytes(&token, &body, &mediator, &mediator_pq, &server, false).await?;
    process_request(
        &bytes,
        &boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(boundary.check_results.lock().as_slice(), &[true, false]);

    // Negative: the same credential as a DIRECT (non-delegated) bearer is
    // fenced out of policy.check. Signed by the credential HOLDER (anchored,
    // cnf-bound), so the denial is the class fence — not envelope admission.
    let body = policy_request_body(true, "tui:session", "write")?;
    let bytes =
        signed_policy_request_bytes(&token, &body, &holder, &holder_pq, &server, true).await?;
    process_request(
        &bytes,
        &boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(
        boundary.attempts.load(Ordering::SeqCst),
        2,
        "a direct TUI credential must not reach the policy handler"
    );

    // Negative: delegated TUI credential at a Policy mutation leaf
    // (issueToken) is fenced out.
    let body = policy_request_body(false, "tui:session", "write")?;
    let bytes =
        signed_policy_request_bytes(&token, &body, &mediator, &mediator_pq, &server, false).await?;
    process_request(
        &bytes,
        &boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(
        boundary.attempts.load(Ordering::SeqCst),
        2,
        "a TUI credential must not reach Policy mutation leaves"
    );

    // Negative: an unenrolled mediator cannot present the credential at all
    // (delegated-bearer admission, before the class fence).
    let body = policy_request_body(true, "tui:session", "write")?;
    let bytes =
        signed_policy_request_bytes(&token, &body, &stranger, &stranger_pq, &server, false).await?;
    process_request(
        &bytes,
        &boundary,
        hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
        &boundary.server,
        &envelope::InMemoryNonceCache::new(),
        CarrierContext::iroh(),
    )
    .await?;
    assert_eq!(
        boundary.attempts.load(Ordering::SeqCst),
        2,
        "an unenrolled mediator must not carry a TUI credential to Policy"
    );
    Ok(())
}
