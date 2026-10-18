//! Causal tests for the `authorize_verified` extension on the Flight
//! authorizer boundary (G1).
//!
//! Panicking is correct in unit tests: expect/unwrap mirror the repo's other
//! test modules (ninep mount-ticket, policy exchange-delegated).
#![allow(clippy::unwrap_used, clippy::expect_used)]
//!
//! Positive path: a REAL owned PolicyService (in-memory Casbin via
//! PolicyManager) on a dedicated inproc transport; the user bearer is
//! relayed as the DELEGATED bearer of an attested `service:flight`
//! credential; the policy service verifies the delegated user envelope via
//! a ClusterKeySource that SHARES the resource fixture's composite key set
//! (`with_composite_key_set`, the public seam used by the policy crate's
//! own delegated tests). The verified context is asserted ONLY after
//! policy ALLOW, and the casbin allow rule is written for the VERIFIED
//! USER identity ("alice" / "tenant.test") — never for the mediator
//! credential and never for wire request fields
//! (`PolicyService::handle_check` derives subject/domain from the verified
//! delegated envelope and ignores the wire fields).
//!
//! Fixture wiring mirrors the proven `exchange_delegated_tests` in
//! crates/hyprstream/src/services/policy.rs: isolated CompositeKeySet,
//! TempDir + InprocManager owned by the fixture for the whole test,
//! delegation-edge authorizer installed, session registry active. No
//! process-global key publication. No mock allow of the policy decision
//! itself: the ALLOW/DENY comes from the real Casbin rules.

use super::*;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_flight::{FlightAuthError, FlightAuthorizer as _};
use hyprstream_rpc::auth::{
    ClusterKeySource, CompositeKeyPair, CompositeKeySet, CompositePairRole, CompositePairState,
};
use hyprstream_service::{InprocManager, ServiceManager};
use std::time::Duration;

const ISSUER: &str = "https://issuer.test";
const RESOURCE: &str = "https://resource.test";
const TENANT: &str = "tenant.test";
const OTHER_TENANT: &str = "other.test";
const CLIENT_ID: &str = "hyprstream-oauth-client-1";
/// Bound for every authorize call so a wedged fixture fails fast.
const CALL_TIMEOUT: Duration = Duration::from_secs(10);

fn label() -> hyprstream_rpc::auth::mac::SecurityLabel {
    hyprstream_rpc::auth::mac::SecurityLabel::new(
        hyprstream_rpc::auth::mac::Level::Internal,
        hyprstream_rpc::auth::mac::Assurance::Classical,
        hyprstream_rpc::auth::mac::CompartmentSet::EMPTY,
    )
}

fn session_registry() -> &'static Arc<dyn hyprstream_rpc::auth::SessionRegistry> {
    if hyprstream_rpc::auth::global_session_registry().is_none() {
        let _ = hyprstream_rpc::auth::set_global_session_registry(Arc::new(
            hyprstream_rpc::auth::InMemorySessionRegistry::new(),
        ));
    }
    hyprstream_rpc::auth::global_session_registry().expect("session registry published")
}

async fn register_active_session(sid: &str, tenant: &str, now: i64) {
    let reg = session_registry();
    let key = hyprstream_rpc::auth::SessionKey::oidc(ISSUER, sid);
    let _ = reg
        .register_session(
            key,
            hyprstream_rpc::auth::SessionState {
                subject: "alice".to_owned(),
                tenant: tenant.to_owned(),
                kind: hyprstream_rpc::auth::SessionKind::Interactive,
                created_at: now,
                expires_at: now + 3600,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await;
}

fn ensure_revocation_store() {
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
}

/// The allow edge for the relayed user bearer — same shape as the policy
/// crate's own test authorizer (allow + the edge must name a target method).
struct AllowEdgeAuthorizer;
impl crate::services::policy::DelegationEdgeAuthorizer for AllowEdgeAuthorizer {
    fn authorize(&self, edge: &crate::services::policy::DelegationEdge<'_>) -> bool {
        edge.target_method_id.is_some()
    }
}

/// The authoritative primary resolver for "alice" (cnf thumbprint check);
/// unknown principals deny.
struct AlicePrimaryResolver {
    ed25519: [u8; 32],
}
impl crate::services::policy::PrimaryEnrollmentResolver for AlicePrimaryResolver {
    fn primary_group(
        &self,
        principal: &str,
        _tenant: &str,
    ) -> Option<crate::services::policy::PrimaryGroup> {
        (principal == "alice").then(|| crate::services::policy::PrimaryGroup {
            suite_id: hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519.to_owned(),
            ordered_component_keys: vec![self.ed25519.to_vec()],
        })
    }
}

struct G1Fixture {
    authorizer: TenantFlightAuthorizer,
    ml_dsa: Arc<hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
    ed: Arc<ed25519_dalek::SigningKey>,
    user_ed: Arc<ed25519_dalek::SigningKey>,
    key_set: Arc<CompositeKeySet>,
    // Owned so per-fixture RAII removes exactly this fixture's scratch tree
    // and the spawned policy service outlives the test body.
    _root: tempfile::TempDir,
    _manager: InprocManager,
}

/// Resource verifier state bound to a composite ledger (the public
/// `with_composite_key_set` sharing seam — no process-global publication).
fn resource_state(key_set: &Arc<CompositeKeySet>) -> crate::server::state::ResourceAuthState {
    let ed25519 = ed25519_dalek::SigningKey::from_bytes(&[0x5a; 32]);
    let federation_ed25519 = ed25519_dalek::SigningKey::from_bytes(&[0x5b; 32]);
    let federation_kid = hyprstream_rpc::auth::jwt::kid_for_key(&federation_ed25519);
    let federation_jwks = serde_json::json!({
        "keys": [{
            "kty": "OKP",
            "crv": "Ed25519",
            "x": URL_SAFE_NO_PAD.encode(federation_ed25519.verifying_key().as_bytes()),
            "use": "sig",
            "alg": "EdDSA",
            "kid": federation_kid,
        }]
    });
    let mut trusted = std::collections::HashMap::new();
    trusted.insert(
        "https://federation.test".to_owned(),
        crate::config::TrustedIssuerConfig {
            jwks_uri: Some("https://federation.test/jwks".to_owned()),
            jwks_cache_ttl_secs: 300,
            allow_http: false,
        },
    );
    let federation_fetcher = Arc::new(
        move |_url: &str| -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<serde_json::Value>> + Send>,
        > {
            let federation_jwks = federation_jwks.clone();
            Box::pin(async move { Ok(federation_jwks) })
        },
    );
    let mut state = crate::server::state::ResourceAuthState::new(
        ed25519.verifying_key(),
        RESOURCE.to_owned(),
        ISSUER.to_owned(),
        Arc::new(
            crate::auth::FederationKeyResolver::new(&trusted).with_jwks_fetcher(federation_fetcher),
        ),
    );
    state.composite_key_set = Arc::clone(key_set);
    state.dpop_jti_seen = Arc::new(hyprstream_util::TtlCache::new(8, 16));
    state
}

async fn fixture() -> G1Fixture {
    crate::mac::install_explicit_test_dispatch_pep();
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    ensure_revocation_store();

    let root = tempfile::tempdir().expect("tempdir");
    let ca = ed25519_dalek::SigningKey::from_bytes(&[0x70; 32]);
    let ed = Arc::new(ed25519_dalek::SigningKey::from_bytes(&[0x5a; 32]));
    let user_ed = Arc::new(ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]));
    let (ml_dsa_sk, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
    let ml_dsa = Arc::new(ml_dsa_sk);

    // One isolated composite ledger shared by BOTH boundaries:
    // ResourceAuthState (resource-side verification of the user bearer) and
    // the policy service's ClusterKeySource (delegated-bearer verification).
    let key_set = Arc::new(CompositeKeySet::default());
    let pair = CompositeKeyPair::verifying(
        hyprstream_rpc::auth::composite_kid(&ml_dsa_vk, &ed.verifying_key()),
        ml_dsa_vk,
        ed.verifying_key(),
        CompositePairRole::OAuth,
        CompositePairState::Active,
        0,
        i64::MAX,
    );
    key_set
        .publish(1, "flight-authz-fixture".to_owned(), vec![pair])
        .expect("composite pair publishes");

    let resource_state = resource_state(&key_set);

    // Casbin: the ALLOW rule is for the VERIFIED USER identity (delegated
    // bearer envelope), so a wrong-tenant bearer denies against this same
    // real service with no rule change.
    let policy = Arc::new(
        crate::auth::PolicyManager::new_in_memory()
            .await
            .expect("in-memory policy manager"),
    );
    policy
        .add_policy_with_domain("alice", TENANT, "flight", "decide", "allow")
        .await
        .expect("allow rule");

    let key_source = ClusterKeySource::new(ca.verifying_key(), ISSUER.to_owned())
        .with_composite_key_set(Arc::clone(&key_set));
    let git = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(root.path()).await.expect("git2db"),
    ));
    let transport = hyprstream_rpc::transport::TransportConfig::inproc(format!(
        "flight-authz-{}",
        uuid::Uuid::new_v4()
    ));
    let service = crate::services::policy::PolicyService::new(
        policy,
        Arc::new(ca.clone()),
        crate::config::TokenConfig::default(),
        Arc::clone(&git),
        transport.clone(),
    )
    .with_default_audience(RESOURCE.to_owned())
    .with_jwt_key_source(Arc::new(key_source))
    .with_delegation_edge_authorizer(Arc::new(AllowEdgeAuthorizer))
    .with_primary_enrollment_resolver(Arc::new(AlicePrimaryResolver {
        ed25519: user_ed.verifying_key().to_bytes(),
    }));
    let manager = InprocManager::new();
    manager
        .spawn(Box::new(service))
        .await
        .expect("policy service spawns");

    // The flight RELAY credential: real service JWT (subject service:flight)
    // signed by the fixture authority, attested in the trust store so
    // PolicyService::accept_delegated_bearer admits the relay (its base rule
    // service:flight -> policy:*/check matches this subject).
    let flight_key = ed25519_dalek::SigningKey::from_bytes(&[0x72; 32]);
    let now = chrono::Utc::now().timestamp();
    let service_claims = hyprstream_rpc::auth::Claims::new("service:flight".into(), now, now + 120)
        .with_issuer(ISSUER.to_owned())
        .with_cnf_jwk(flight_key.verifying_key().as_bytes());
    let service_token = hyprstream_rpc::auth::jwt::encode_service_jwt(&service_claims, &ca);
    hyprstream_service::global_trust_store().insert(
        flight_key.verifying_key(),
        hyprstream_service::Attestation {
            scopes: ["flight".to_owned()].into_iter().collect(),
            subject: Some("service:flight".to_owned()),
            jwt: Some(service_token.clone()),
            expires_at: now + 120,
            attested_by: Some(ca.verifying_key().to_bytes()),
        },
    );

    let client = hyprstream_rpc_std::policy_client::PolicyClient::for_local_transport_bootstrap(
        &transport,
        flight_key.clone(),
        ca.verifying_key(),
        Some(service_token),
    )
    .expect("policy client bootstrap");
    let authorizer = TenantFlightAuthorizer::new(resource_state, client);
    G1Fixture {
        authorizer,
        ml_dsa,
        ed,
        user_ed,
        key_set,
        _root: root,
        _manager: manager,
    }
}

impl G1Fixture {
    /// Production user `at+jwt` composite bearer: verified by the resource
    /// boundary AND, as the delegated bearer, by the policy service.
    fn user_bearer(&self, now: i64, tenant: &str, sid: &str) -> String {
        let hs = hyprstream_rpc::auth::service_signer_suite_b64(
            &self.user_ed.verifying_key().to_bytes(),
            None,
        );
        let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
            .with_issuer(ISSUER.to_owned())
            .with_tenant(tenant.to_owned())
            .with_sid(sid)
            .with_client_id(CLIENT_ID)
            .with_scope(Some("read write".to_owned()))
            .with_clearance(label())
            .with_audience(Some(RESOURCE.to_owned()))
            .with_cnf_jwk(&self.user_ed.verifying_key().to_bytes())
            .with_cnf_hs_signer_suite(hs)
            .with_jti();
        crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(&claims, &self.ml_dsa, &self.ed)
    }

    async fn authorize_verified_bounded(
        &self,
        token: Option<&str>,
    ) -> Result<hyprstream_flight::VerifiedFlightContext, FlightAuthError> {
        tokio::time::timeout(
            CALL_TIMEOUT,
            self.authorizer.authorize_verified(
                token.map(|t| format!("Bearer {t}")).as_deref(),
                "flight",
                "decide",
            ),
        )
        .await
        .expect("authorize_verified must not wedge")
    }
}

#[tokio::test]
async fn verified_context_returns_real_jwt_tenant_after_policy_allow() {
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let token = fx.user_bearer(now, TENANT, "sess-1");

    let context = fx
        .authorize_verified_bounded(Some(&token))
        .await
        .expect("real JWT with correct tenant must pass policy allow");
    assert_eq!(context.tenant.as_deref(), Some(TENANT));
    assert_eq!(context.subject, "alice");
    assert_eq!(context.exp, Some(now + 3600));
}

#[tokio::test]
async fn wrong_tenant_user_jwt_is_denied_by_real_policy() {
    // A VALIDLY SIGNED user bearer for a different tenant (own registered
    // session): the policy service verifies the delegated envelope (subject
    // alice, domain other.test) and finds no allow rule for it — the causal
    // wrong-tenant DENY through the real service, with policy and mediator
    // unchanged.
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-other", OTHER_TENANT, now).await;
    let token = fx.user_bearer(now, OTHER_TENANT, "sess-other");

    let result = fx
        .authorize_verified_bounded(Some(&token))
        .await
        .map(|_| ());
    assert_eq!(
        result,
        Err(FlightAuthError::Forbidden(
            "Flight request denied by tenant policy".to_owned()
        ))
    );
}

#[tokio::test]
async fn missing_bearer_denies_identically_on_both_methods() {
    let fx = fixture().await;
    let unauthorized = FlightAuthError::Unauthenticated("Bearer token required".to_owned());
    assert_eq!(
        fx.authorizer.authorize(None, "flight", "decide").await,
        Err(unauthorized.clone())
    );
    assert_eq!(
        fx.authorize_verified_bounded(None).await.map(|_| ()),
        Err(unauthorized)
    );
}

#[tokio::test]
async fn invalid_jwt_denies_identically_on_both_methods() {
    let fx = fixture().await;
    let invalid = FlightAuthError::Unauthenticated("Invalid access token".to_owned());
    assert_eq!(
        fx.authorizer
            .authorize(Some("Bearer not-a-jwt"), "flight", "decide")
            .await,
        Err(invalid.clone())
    );
    assert_eq!(
        fx.authorize_verified_bounded(Some("not-a-jwt"))
            .await
            .map(|_| ()),
        Err(invalid)
    );
}

#[tokio::test]
async fn expired_jwt_denies() {
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let hs = hyprstream_rpc::auth::service_signer_suite_b64(
        &fx.user_ed.verifying_key().to_bytes(),
        None,
    );
    let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now - 120, now - 60)
        .with_issuer(ISSUER.to_owned())
        .with_tenant(TENANT.to_owned())
        .with_sid("sess-1")
        .with_client_id(CLIENT_ID)
        .with_clearance(label())
        .with_audience(Some(RESOURCE.to_owned()))
        .with_cnf_jwk(&fx.user_ed.verifying_key().to_bytes())
        .with_cnf_hs_signer_suite(hs)
        .with_jti();
    let token = crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(&claims, &fx.ml_dsa, &fx.ed);
    assert_eq!(
        fx.authorize_verified_bounded(Some(&token))
            .await
            .map(|_| ()),
        Err(FlightAuthError::Unauthenticated(
            "Invalid access token".to_owned()
        ))
    );
}

#[tokio::test]
async fn wrong_audience_jwt_denies_at_the_resource_boundary() {
    // aud is part of the signed resource binding (RFC 8707): a token minted
    // for another resource must not verify at the resource boundary, so the
    // policy service is never reached.
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let hs = hyprstream_rpc::auth::service_signer_suite_b64(
        &fx.user_ed.verifying_key().to_bytes(),
        None,
    );
    let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
        .with_issuer(ISSUER.to_owned())
        .with_tenant(TENANT.to_owned())
        .with_sid("sess-1")
        .with_client_id(CLIENT_ID)
        .with_clearance(label())
        .with_audience(Some("https://other.resource.test".to_owned()))
        .with_cnf_jwk(&fx.user_ed.verifying_key().to_bytes())
        .with_cnf_hs_signer_suite(hs)
        .with_jti();
    let token = crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(&claims, &fx.ml_dsa, &fx.ed);
    assert_eq!(
        fx.authorize_verified_bounded(Some(&token))
            .await
            .map(|_| ()),
        Err(FlightAuthError::Unauthenticated(
            "Invalid access token".to_owned()
        ))
    );
}

#[tokio::test]
async fn missing_tenant_binding_denies_before_policy() {
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let hs = hyprstream_rpc::auth::service_signer_suite_b64(
        &fx.user_ed.verifying_key().to_bytes(),
        None,
    );
    let mut claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
        .with_issuer(ISSUER.to_owned())
        .with_sid("sess-1")
        .with_client_id(CLIENT_ID)
        .with_clearance(label())
        .with_audience(Some(RESOURCE.to_owned()))
        .with_cnf_jwk(&fx.user_ed.verifying_key().to_bytes())
        .with_cnf_hs_signer_suite(hs)
        .with_jti();
    claims.tenant = None;
    let token = crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(&claims, &fx.ml_dsa, &fx.ed);
    assert_eq!(
        fx.authorize_verified_bounded(Some(&token))
            .await
            .map(|_| ()),
        Err(FlightAuthError::Forbidden(
            "Verified hosted-account tenant binding required".to_owned()
        ))
    );
}

/// An INJECTED failing policy client: every RPC errors, so the authorizer's
/// error-to-deny mapping is exercised without any network. This is a unit
/// fixture for the failure mapping only — it does NOT exercise or prove
/// transport reachability behavior. The call counter proves the request
/// actually REACHED the policy client (verification/session/preflight
/// passed) before the injected failure was mapped to deny.
#[derive(Default)]
struct InjectedFailingPolicyClient {
    request_id: std::sync::atomic::AtomicU64,
    policy_calls: std::sync::atomic::AtomicU64,
}

impl InjectedFailingPolicyClient {
    fn policy_calls(&self) -> u64 {
        self.policy_calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl hyprstream_rpc::RpcClient for InjectedFailingPolicyClient {
    async fn call(&self, _payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_for_service_with_method(
        &self,
        _service_domain: &str,
        _method_discriminator: u16,
        _payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_with_options(
        &self,
        _payload: Vec<u8>,
        _options: hyprstream_rpc::CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_with_options_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
        _options: hyprstream_rpc::CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_with_options_for_service_with_method(
        &self,
        _service_domain: &str,
        _method_discriminator: u16,
        _payload: Vec<u8>,
        _options: hyprstream_rpc::CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        self.policy_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_streaming(
        &self,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_streaming_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn call_streaming_for_service_with_method(
        &self,
        _service_domain: &str,
        _method_discriminator: u16,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn open_stream(
        &self,
        _payload: Vec<u8>,
    ) -> anyhow::Result<Box<dyn hyprstream_rpc::stream_consumer::StreamHandle + 'static>> {
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    async fn open_stream_from_info(
        &self,
        _info: hyprstream_rpc::StreamInfo,
        _local_ephemeral: [u8; 32],
        _remote: [u8; 32],
    ) -> anyhow::Result<Box<dyn hyprstream_rpc::stream_consumer::StreamHandle + 'static>> {
        anyhow::bail!("injected policy-client failure (test fixture)")
    }
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[tokio::test]
async fn injected_policy_client_failure_maps_to_deny() {
    // Separate unit case: an INJECTED failing policy client. Verification,
    // session, and preflight gates run for real; the injected failure then
    // maps to deny. This proves the error-to-deny path only — it does NOT
    // exercise or prove transport reachability; the real ALLOW (correct
    // tenant) and real-policy wrong-tenant DENY cases above are separate.
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let token = fx.user_bearer(now, TENANT, "sess-1");
    // INJECTED failing policy client, held behind an Arc the test can
    // observe: the deny assertion alone would also pass if the request were
    // rejected before the policy layer, so the call counter proves the
    // request actually reached the policy client after the real
    // verification/session/preflight gates.
    let failing = Arc::new(InjectedFailingPolicyClient::default());
    let client = hyprstream_rpc_std::policy_client::PolicyClient::new(failing.clone());
    let authorizer = TenantFlightAuthorizer::new(resource_state(&fx.key_set), client);
    let result = tokio::time::timeout(
        CALL_TIMEOUT,
        authorizer.authorize_verified(Some(&format!("Bearer {token}")), "flight", "decide"),
    )
    .await
    .expect("authorize_verified must not wedge")
    .map(|_| ());
    assert_eq!(
        result,
        Err(FlightAuthError::Forbidden(
            "Flight request denied by tenant policy".to_owned()
        ))
    );
    // The policy RPC was actually invoked (nonzero) and the injected
    // failure mapped to exactly the deny path. No retries are configured in
    // this path, so the causal expectation is exactly one invocation.
    assert_eq!(
        failing.policy_calls(),
        1,
        "the injected policy client must be invoked exactly once before the deny"
    );
}

#[tokio::test]
async fn authorize_behavior_is_unchanged_by_the_verified_extension() {
    // Pre-existing `authorize` on a valid token reaches policy exactly as
    // before G1 and is allowed by the same real service: identical result.
    let fx = fixture().await;
    let now = chrono::Utc::now().timestamp();
    register_active_session("sess-1", TENANT, now).await;
    let token = fx.user_bearer(now, TENANT, "sess-1");
    let result = tokio::time::timeout(
        CALL_TIMEOUT,
        fx.authorizer
            .authorize(Some(&format!("Bearer {token}")), "flight", "decide"),
    )
    .await
    .expect("authorize must not wedge");
    assert_eq!(result, Ok(()));
}
