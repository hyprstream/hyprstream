//! Authenticated identity registration and federation-intake HTTP face.
//!
//! Registration has two distinct routes with the same handle-only body:
//! browser-session self service and bearer-authenticated operator manual
//! registration. The route, never a client field, selects the authority path.
//!
//! Federation intake is deliberately exposed only beside the authenticated
//! self-service route. Its `did:web` arm checks an exact, deployment-owned
//! HTTPS-origin allowlist before [`FederationIntake::intake`] can perform any
//! resolver network I/O.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{ensure, Result};
use async_trait::async_trait;
use axum::extract::{Extension, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use hyprstream_pds::{DidOpSignature, UnsignedGenesisDidOp};
use hyprstream_pds_service::federation_intake::{FederationIntake, InventoryEntry};
use hyprstream_pds_service::hosted_account_mint::{
    HostedAccountGenesisSigner, HostedAccountRegistrationRequest, HostedAccountRegistrationResult,
    HostedPdsAccountMinter,
};
use hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL;
use serde::{Deserialize, Serialize};
use tracing::warn;
use url::Url;

use super::auth::AuthenticatedUser;
use super::session;
use super::state::OAuthState;
use crate::server::middleware::RateLimiter;

/// Handle-only registration payload.
///
/// `deny_unknown_fields` makes `tenant`, `did`, mode, and transport metadata
/// invalid at the JSON boundary instead of silently ignoring them.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisterHostedAccountRequest {
    pub handle: String,
}

/// Browser registration result.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RegisterHostedAccountResponse {
    pub handle: String,
    pub did: String,
    pub pds_endpoint: String,
    pub quic_url: String,
    pub cert_hash: String,
}

impl From<HostedAccountRegistrationResult> for RegisterHostedAccountResponse {
    fn from(result: HostedAccountRegistrationResult) -> Self {
        Self {
            handle: result.handle,
            did: result.did,
            pds_endpoint: result.pds_endpoint,
            quic_url: result.quic_url,
            cert_hash: result.cert_hash,
        }
    }
}

/// Federation-add payload.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FederationIntakeRequest {
    pub did: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegistrationPath {
    SelfService,
    OperatorManual,
}

/// An identity created only after the corresponding HTTP authenticator passes.
#[derive(Clone, Debug)]
pub(crate) struct AuthenticatedIdentityCaller {
    subject: String,
}

impl AuthenticatedIdentityCaller {
    fn new(subject: impl Into<String>) -> Result<Self, IdentityApiError> {
        let subject = subject.into();
        if subject.is_empty() || subject == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::Unauthenticated);
        }
        Ok(Self { subject })
    }

    fn subject(&self) -> &str {
        &self.subject
    }
}

/// Select and use the hybrid genesis signer for one authenticated request.
///
/// Implementations may map self-service callers to user-held key material and
/// operator-manual callers to the deployment-authority workflow. The client
/// cannot select this path in its JSON body.
pub trait RegistrationGenesisSigner: Send + Sync {
    fn sign_genesis(
        &self,
        caller: &str,
        operator_manual: bool,
        unsigned: &UnsignedGenesisDidOp,
    ) -> Result<DidOpSignature>;
}

/// Injectable mint boundary used by the HTTP adapter and causal tests.
pub trait HostedRegistrationMint: Send + Sync {
    fn mint(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
    ) -> Result<HostedAccountRegistrationResult>;
}

impl HostedRegistrationMint for HostedPdsAccountMinter {
    fn mint(
        &self,
        request: &HostedAccountRegistrationRequest,
        signer: &dyn HostedAccountGenesisSigner,
    ) -> Result<HostedAccountRegistrationResult> {
        HostedPdsAccountMinter::mint(self, request, signer)
    }
}

/// Injectable federation-intake boundary.
#[async_trait]
pub trait FederatedIdentityIntake: Send + Sync {
    async fn intake(&self, did: &str) -> Result<InventoryEntry>;
}

#[async_trait]
impl FederatedIdentityIntake for FederationIntake {
    async fn intake(&self, did: &str) -> Result<InventoryEntry> {
        FederationIntake::intake(self, did).await
    }
}

/// Exact HTTPS origins that may be resolved through the `did:web` intake arm.
///
/// Wildcards and suffix matching are intentionally unsupported. An empty set
/// means `did:web` intake is disabled, while fixed-directory `did:plc` intake
/// remains available.
#[derive(Clone, Debug)]
pub struct DidWebOriginAllowlist {
    origins: BTreeSet<String>,
}

impl DidWebOriginAllowlist {
    pub fn new<I, S>(origins: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut normalized = BTreeSet::new();
        for origin in origins {
            let origin = origin.as_ref();
            let parsed = Url::parse(origin)?;
            ensure!(
                parsed.scheme() == "https"
                    && parsed.host_str().is_some_and(|host| !host.contains('*'))
                    && parsed.username().is_empty()
                    && parsed.password().is_none()
                    && parsed.path() == "/"
                    && parsed.query().is_none()
                    && parsed.fragment().is_none(),
                "did:web allowlist entries must be exact HTTPS origins"
            );
            normalized.insert(parsed.origin().ascii_serialization());
        }
        Ok(Self {
            origins: normalized,
        })
    }

    fn require_allowed(&self, did: &str) -> Result<(), IdentityApiError> {
        let document_url = hyprstream_rpc::did_web::did_web_to_url(did)
            .map_err(|_| IdentityApiError::InvalidRequest)?;
        let parsed = Url::parse(&document_url).map_err(|_| IdentityApiError::InvalidRequest)?;
        let origin = parsed.origin().ascii_serialization();
        if !self.origins.contains(&origin) {
            return Err(IdentityApiError::ResolvableHostDenied);
        }
        Ok(())
    }
}

/// Authenticated, rate-limited wire adapter over hosted mint and intake.
pub struct IdentityRegistrationApi {
    minter: Arc<dyn HostedRegistrationMint>,
    signer: Arc<dyn RegistrationGenesisSigner>,
    intake: Arc<dyn FederatedIdentityIntake>,
    did_web_origins: DidWebOriginAllowlist,
    rate_limiter: Arc<RateLimiter>,
}

impl IdentityRegistrationApi {
    #[must_use]
    pub fn new(
        minter: Arc<dyn HostedRegistrationMint>,
        signer: Arc<dyn RegistrationGenesisSigner>,
        intake: Arc<dyn FederatedIdentityIntake>,
        did_web_origins: DidWebOriginAllowlist,
        rate_limiter: Arc<RateLimiter>,
    ) -> Self {
        Self {
            minter,
            signer,
            intake,
            did_web_origins,
            rate_limiter,
        }
    }

    fn register(
        &self,
        caller: &AuthenticatedIdentityCaller,
        path: RegistrationPath,
        request: RegisterHostedAccountRequest,
    ) -> Result<RegisterHostedAccountResponse, IdentityApiError> {
        if path == RegistrationPath::OperatorManual && !caller.subject().starts_with("service:") {
            return Err(IdentityApiError::Forbidden);
        }
        self.check_rate(caller)?;
        if request.handle.is_empty() || request.handle == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::InvalidRequest);
        }

        let mint_request =
            HostedAccountRegistrationRequest::from_client_fields(request.handle, None)
                .map_err(|_| IdentityApiError::InvalidRequest)?;
        let signer = CallerGenesisSigner {
            provider: self.signer.as_ref(),
            caller: caller.subject(),
            operator_manual: path == RegistrationPath::OperatorManual,
        };
        let result = self
            .minter
            .mint(&mint_request, &signer)
            .map_err(IdentityApiError::Backend)?;
        if result.did == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::Backend(anyhow::anyhow!(
                "hosted-account mint returned the reserved unauthenticated DID"
            )));
        }
        Ok(RegisterHostedAccountResponse::from(result))
    }

    async fn intake(
        &self,
        caller: &AuthenticatedIdentityCaller,
        request: FederationIntakeRequest,
    ) -> Result<InventoryEntry, IdentityApiError> {
        self.check_rate(caller)?;
        if request.did == UNAUTHENTICATED_DID_SENTINEL {
            return Err(IdentityApiError::InvalidRequest);
        }
        if request.did.starts_with("did:web:") {
            self.did_web_origins.require_allowed(&request.did)?;
        } else if !hyprstream_rpc::did_plc::is_did_plc(&request.did) {
            return Err(IdentityApiError::InvalidRequest);
        }
        self.intake
            .intake(&request.did)
            .await
            .map_err(IdentityApiError::Backend)
    }

    fn check_rate(&self, caller: &AuthenticatedIdentityCaller) -> Result<(), IdentityApiError> {
        if self.rate_limiter.check_and_increment(caller.subject()) {
            return Err(IdentityApiError::RateLimited);
        }
        Ok(())
    }
}

struct CallerGenesisSigner<'a> {
    provider: &'a dyn RegistrationGenesisSigner,
    caller: &'a str,
    operator_manual: bool,
}

impl HostedAccountGenesisSigner for CallerGenesisSigner<'_> {
    fn sign(&self, unsigned: &UnsignedGenesisDidOp) -> Result<DidOpSignature> {
        self.provider
            .sign_genesis(self.caller, self.operator_manual, unsigned)
    }
}

#[derive(Debug)]
enum IdentityApiError {
    Unauthenticated,
    Forbidden,
    RateLimited,
    ResolvableHostDenied,
    InvalidRequest,
    Backend(anyhow::Error),
}

impl IntoResponse for IdentityApiError {
    fn into_response(self) -> Response {
        let (status, code, description) = match &self {
            Self::Unauthenticated => (
                StatusCode::UNAUTHORIZED,
                "authentication_required",
                "Authenticated caller required",
            ),
            Self::Forbidden => (
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                "Operator authority required",
            ),
            Self::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "Identity mutation rate limit exceeded",
            ),
            Self::ResolvableHostDenied => (
                StatusCode::FORBIDDEN,
                "origin_not_allowed",
                "DID web origin is not permitted",
            ),
            Self::InvalidRequest => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Registration or intake request is invalid",
            ),
            Self::Backend(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Identity operation failed",
            ),
        };
        if let Self::Backend(error) = &self {
            warn!(%error, "identity registration/intake backend failed");
        }
        (
            status,
            Json(serde_json::json!({
                "error": code,
                "error_description": description,
            })),
        )
            .into_response()
    }
}

fn unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "temporarily_unavailable",
            "error_description": "Identity registration is not configured",
        })),
    )
        .into_response()
}

pub(super) async fn require_registration_session(
    State(state): State<Arc<OAuthState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let session = match session::extract_session_id(request.headers()) {
        Some(session_id) => state.sessions.get(&session_id).await,
        None => None,
    };
    let Some(session) = session else {
        return IdentityApiError::Unauthenticated.into_response();
    };
    let caller = match AuthenticatedIdentityCaller::new(session.username) {
        Ok(caller) => caller,
        Err(error) => return error.into_response(),
    };
    request.extensions_mut().insert(caller);
    next.run(request).await
}

pub(super) async fn register_self_service(
    State(state): State<Arc<OAuthState>>,
    Extension(caller): Extension<AuthenticatedIdentityCaller>,
    Json(request): Json<RegisterHostedAccountRequest>,
) -> Response {
    match state.identity_registration_api.as_deref() {
        Some(api) => api
            .register(&caller, RegistrationPath::SelfService, request)
            .map(Json)
            .into_response(),
        None => unavailable_response(),
    }
}

pub(super) async fn register_operator_manual(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(request): Json<RegisterHostedAccountRequest>,
) -> Response {
    let caller = match AuthenticatedIdentityCaller::new(user.user) {
        Ok(caller) => caller,
        Err(error) => return error.into_response(),
    };
    match state.identity_registration_api.as_deref() {
        Some(api) => api
            .register(&caller, RegistrationPath::OperatorManual, request)
            .map(Json)
            .into_response(),
        None => unavailable_response(),
    }
}

pub(super) async fn intake_federated_identity(
    State(state): State<Arc<OAuthState>>,
    Extension(caller): Extension<AuthenticatedIdentityCaller>,
    Json(request): Json<FederationIntakeRequest>,
) -> Response {
    match state.identity_registration_api.as_deref() {
        Some(api) => api.intake(&caller, request).await.map(Json).into_response(),
        None => unavailable_response(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::bail;

    use super::*;

    struct FakeMint {
        calls: AtomicUsize,
    }

    impl HostedRegistrationMint for FakeMint {
        fn mint(
            &self,
            request: &HostedAccountRegistrationRequest,
            _signer: &dyn HostedAccountGenesisSigner,
        ) -> Result<HostedAccountRegistrationResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let did = if request.handle() == "reserved-result" {
                UNAUTHENTICATED_DID_SENTINEL.to_owned()
            } else {
                format!("did:web:{}.accounts.example", request.handle())
            };
            Ok(HostedAccountRegistrationResult {
                handle: format!("at://{}.accounts.example", request.handle()),
                did,
                pds_endpoint: "https://pds.example".to_owned(),
                quic_url: "https://pds.example:4433/wt".to_owned(),
                cert_hash: "zQmPin".to_owned(),
            })
        }
    }

    struct FakeSigner;

    impl RegistrationGenesisSigner for FakeSigner {
        fn sign_genesis(
            &self,
            _caller: &str,
            _operator_manual: bool,
            _unsigned: &UnsignedGenesisDidOp,
        ) -> Result<DidOpSignature> {
            bail!("fake mint must not invoke the signer")
        }
    }

    struct CountingIntake {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl FederatedIdentityIntake for CountingIntake {
        async fn intake(&self, _did: &str) -> Result<InventoryEntry> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            bail!("counting intake reached")
        }
    }

    fn fixture(max_requests: u32) -> (Arc<FakeMint>, Arc<CountingIntake>, IdentityRegistrationApi) {
        let mint = Arc::new(FakeMint {
            calls: AtomicUsize::new(0),
        });
        let intake = Arc::new(CountingIntake {
            calls: AtomicUsize::new(0),
        });
        let api = IdentityRegistrationApi::new(
            mint.clone(),
            Arc::new(FakeSigner),
            intake.clone(),
            DidWebOriginAllowlist::new(["https://federated.example"]).unwrap(),
            Arc::new(RateLimiter::new(max_requests, 60)),
        );
        (mint, intake, api)
    }

    #[test]
    fn handle_only_contract_rejects_client_authority_fields() {
        let error = serde_json::from_value::<RegisterHostedAccountRequest>(
            serde_json::json!({"handle": "alice", "tenant": "attacker"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));

        let error = serde_json::from_value::<RegisterHostedAccountRequest>(
            serde_json::json!({"handle": "alice", "mode": "operator"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn self_service_and_manual_paths_return_exact_registration_contract() {
        let (mint, _intake, api) = fixture(10);
        let self_caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        let response = api
            .register(
                &self_caller,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: "alice".to_owned(),
                },
            )
            .unwrap();
        assert_eq!(response.handle, "at://alice.accounts.example");
        assert_eq!(response.did, "did:web:alice.accounts.example");
        assert_eq!(response.pds_endpoint, "https://pds.example");
        assert_eq!(response.quic_url, "https://pds.example:4433/wt");
        assert_eq!(response.cert_hash, "zQmPin");
        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            serde_json::json!({
                "handle": "at://alice.accounts.example",
                "did": "did:web:alice.accounts.example",
                "pdsEndpoint": "https://pds.example",
                "quicUrl": "https://pds.example:4433/wt",
                "certHash": "zQmPin",
            })
        );

        let operator = AuthenticatedIdentityCaller::new("service:identity-operator").unwrap();
        api.register(
            &operator,
            RegistrationPath::OperatorManual,
            RegisterHostedAccountRequest {
                handle: "bob".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(mint.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn manual_path_rejects_non_service_and_unknown_before_mint() {
        let (mint, _intake, api) = fixture(10);
        let user = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        assert!(matches!(
            api.register(
                &user,
                RegistrationPath::OperatorManual,
                RegisterHostedAccountRequest {
                    handle: "alice".to_owned(),
                },
            ),
            Err(IdentityApiError::Forbidden)
        ));
        assert!(matches!(
            api.register(
                &user,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: UNAUTHENTICATED_DID_SENTINEL.to_owned(),
                },
            ),
            Err(IdentityApiError::InvalidRequest)
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reserved_did_is_rejected_again_at_the_api_output_boundary() {
        let (mint, _intake, api) = fixture(10);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        assert!(matches!(
            api.register(
                &caller,
                RegistrationPath::SelfService,
                RegisterHostedAccountRequest {
                    handle: "reserved-result".to_owned(),
                },
            ),
            Err(IdentityApiError::Backend(_))
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rate_limit_runs_before_mint() {
        let (mint, _intake, api) = fixture(1);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();
        let request = || RegisterHostedAccountRequest {
            handle: "alice".to_owned(),
        };
        api.register(&caller, RegistrationPath::SelfService, request())
            .unwrap();
        assert!(matches!(
            api.register(&caller, RegistrationPath::SelfService, request()),
            Err(IdentityApiError::RateLimited)
        ));
        assert_eq!(mint.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn intake_constrains_did_web_before_resolver_io() {
        let (_mint, intake, api) = fixture(10);
        let caller = AuthenticatedIdentityCaller::new("did:web:alice.example").unwrap();

        assert!(matches!(
            api.intake(
                &caller,
                FederationIntakeRequest {
                    did: "did:web:127.0.0.1%3A8443".to_owned(),
                },
            )
            .await,
            Err(IdentityApiError::ResolvableHostDenied)
        ));
        assert!(matches!(
            api.intake(
                &caller,
                FederationIntakeRequest {
                    did: UNAUTHENTICATED_DID_SENTINEL.to_owned(),
                },
            )
            .await,
            Err(IdentityApiError::InvalidRequest)
        ));
        assert_eq!(intake.calls.load(Ordering::SeqCst), 0);

        let error = api
            .intake(
                &caller,
                FederationIntakeRequest {
                    did: "did:web:federated.example".to_owned(),
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, IdentityApiError::Backend(_)));
        assert_eq!(intake.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn allowlist_accepts_only_exact_https_origins() {
        assert!(DidWebOriginAllowlist::new(["http://example.com"]).is_err());
        assert!(DidWebOriginAllowlist::new(["https://example.com/path"]).is_err());
        assert!(DidWebOriginAllowlist::new(["https://*.example.com"]).is_err());

        let allowlist =
            DidWebOriginAllowlist::new(["https://example.com", "https://example.com:8443"])
                .unwrap();
        assert!(allowlist.require_allowed("did:web:example.com").is_ok());
        assert!(allowlist
            .require_allowed("did:web:example.com%3A8443")
            .is_ok());
        assert!(matches!(
            allowlist.require_allowed("did:web:sub.example.com"),
            Err(IdentityApiError::ResolvableHostDenied)
        ));
    }

    fn oauth_state() -> (Arc<OAuthState>, crate::config::CorsConfig) {
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x51; 32]);
        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x52; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new("/dev/null/identity-registration-test.sock".into()),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let mut config = crate::config::OAuthConfig::default();
        config.external_url = Some("https://pds.example.test".to_owned());
        let cors = config.cors.clone();
        (
            Arc::new(OAuthState::new(
                &config,
                crate::services::PolicyClient::new(make_client()),
                crate::services::DiscoveryClient::new(make_client()),
                signing_key.verifying_key().to_bytes(),
            )),
            cors,
        )
    }

    fn post(path: &str, cookie: Option<String>) -> axum::http::Request<axum::body::Body> {
        let mut request = axum::http::Request::post(path)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(r#"{"handle":"alice"}"#))
            .unwrap();
        if let Some(cookie) = cookie {
            request
                .headers_mut()
                .insert(axum::http::header::COOKIE, cookie.parse().unwrap());
        }
        request
    }

    #[tokio::test]
    async fn live_routes_never_expose_unauthenticated_registration_or_intake() {
        use tower::ServiceExt;

        let (state, cors) = oauth_state();
        let app = super::super::create_app(Arc::clone(&state), &cors);

        for path in [
            "/api/identity/register",
            "/api/identity/intake",
            "/api/identity/register/manual",
        ] {
            let response = app.clone().oneshot(post(path, None)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{path} must authenticate before its handler"
            );
        }

        let session_id = state
            .sessions
            .create("did:web:alice.example".to_owned(), "local".to_owned())
            .await;
        let response = app
            .oneshot(post(
                "/api/identity/register",
                Some(format!("{}={session_id}", session::SESSION_COOKIE_NAME)),
            ))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "an authenticated route with no injected minter must fail closed"
        );
    }
}
