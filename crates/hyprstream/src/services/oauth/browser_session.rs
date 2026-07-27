//! Browser-facing ATProto session exchange and viewer context.
//!
//! The exchange consumes a one-use, DID-signed ATProto service-auth JWT plus
//! its DPoP proof, resolves any local tenant only from the authority-owned
//! hosted-account store, and creates the opaque server-side session used by
//! identity registration and federation intake.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

use super::session;
use super::state::OAuthState;
use super::token_exchange::{verify_atproto_service_jwt, ATPROTO_SESSION_EXCHANGE_NSID};

pub const SESSION_EXCHANGE_PATH: &str = "/api/session/exchange";
pub const WHOAMI_PATH: &str = "/api/session/whoami";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewerContext {
    did: Option<String>,
    kind: ViewerKind,
    tenant: Option<String>,
    can_act_locally: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum ViewerKind {
    Local,
    Federated,
    Unauthenticated,
}

impl ViewerContext {
    fn unauthenticated() -> Self {
        Self {
            did: None,
            kind: ViewerKind::Unauthenticated,
            tenant: None,
            can_act_locally: false,
        }
    }

    fn atproto(did: String, tenant: Option<String>) -> Self {
        let kind = if tenant.is_some() {
            ViewerKind::Local
        } else {
            ViewerKind::Federated
        };
        Self {
            did: Some(did),
            kind,
            can_act_locally: tenant.is_some(),
            tenant,
        }
    }

    fn from_session(session: session::Session) -> Self {
        session
            .atproto_did
            .map(|did| Self::atproto(did, session.verified_tenant))
            .unwrap_or_else(Self::unauthenticated)
    }
}

/// Consume an ATProto service-auth assertion and establish the HttpOnly cookie
/// session expected by the self-service identity mutation routes.
pub async fn exchange(State(state): State<Arc<OAuthState>>, headers: HeaderMap) -> Response {
    let Some(assertion) = bearer_token(&headers) else {
        return session_error(
            StatusCode::UNAUTHORIZED,
            "invalid_token",
            "Authorization: Bearer ATProto service-auth JWT is required",
        );
    };

    let endpoint = format!(
        "{}{SESSION_EXCHANGE_PATH}",
        state.atproto_issuer_url().trim_end_matches('/')
    );
    let Some(dpop) = headers.get("DPoP").and_then(|value| value.to_str().ok()) else {
        return session_error(
            StatusCode::BAD_REQUEST,
            "invalid_dpop_proof",
            "DPoP proof is required",
        );
    };
    let proof = match super::dpop::verify_dpop_proof(dpop, "POST", &endpoint, None) {
        Ok(proof) => proof,
        Err(error) => {
            tracing::warn!(%error, "browser session exchange rejected DPoP proof");
            return session_error(
                StatusCode::BAD_REQUEST,
                "invalid_dpop_proof",
                "DPoP proof verification failed",
            );
        }
    };

    let verified =
        match verify_atproto_service_jwt(&state, assertion, ATPROTO_SESSION_EXCHANGE_NSID).await {
            Ok(verified) => verified,
            Err(error) => {
                tracing::warn!(%error, "browser session exchange rejected ATProto assertion");
                return session_error(StatusCode::UNAUTHORIZED, "invalid_token", &error);
            }
        };

    if !state.check_and_record_dpop_jti(&proof.jti, proof.iat) {
        return session_error(
            StatusCode::BAD_REQUEST,
            "invalid_dpop_proof",
            "DPoP proof already used",
        );
    }
    let Some((issuer, jti, exp)) = verified.atproto_replay.as_ref() else {
        return session_error(
            StatusCode::UNAUTHORIZED,
            "invalid_token",
            "ATProto assertion has no replay identifier",
        );
    };
    if !state.check_and_record_atproto_service_jti(issuer, jti, *exp) {
        return session_error(
            StatusCode::UNAUTHORIZED,
            "invalid_token",
            "ATProto service-auth JWT already used",
        );
    }

    let context = ViewerContext::atproto(verified.sub.clone(), verified.verified_tenant.clone());
    let session_id = state
        .sessions
        .create_atproto(verified.sub, verified.verified_tenant)
        .await;
    let secure = state.atproto_issuer_url().starts_with("https://");
    (
        StatusCode::OK,
        [
            (
                header::SET_COOKIE,
                session::session_cookie(&session_id, secure),
            ),
            (header::CACHE_CONTROL, "no-store, max-age=0".to_owned()),
            (header::PRAGMA, "no-cache".to_owned()),
        ],
        Json(context),
    )
        .into_response()
}

/// Return the viewer authority derived from the opaque server-side session.
/// Missing, invalid, expired, or non-ATProto sessions all collapse to the same
/// public unauthenticated floor.
pub async fn whoami(State(state): State<Arc<OAuthState>>, headers: HeaderMap) -> Response {
    let context = match session::extract_session_id(&headers) {
        Some(session_id) => state
            .sessions
            .get(&session_id)
            .await
            .map(ViewerContext::from_session)
            .unwrap_or_else(ViewerContext::unauthenticated),
        None => ViewerContext::unauthenticated(),
    };
    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store, max-age=0"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(context),
    )
        .into_response()
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .split_once(' ')
        .filter(|(scheme, token)| {
            scheme.eq_ignore_ascii_case("Bearer")
                && !token.is_empty()
                && !token.chars().any(char::is_whitespace)
        })
        .map(|(_, token)| token)
}

fn session_error(status: StatusCode, error: &str, description: &str) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::json!({
            "error": error,
            "error_description": description,
        })),
    )
        .into_response()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use p256::ecdsa::signature::Signer as _;
    use tower::ServiceExt as _;

    use hyprstream_rpc::crypto::CryptoPolicy;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    use hyprstream_rpc::signer::LocalSigner;
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};

    const ISSUER: &str = "https://pds.example.test";
    const LOCAL_DID: &str = "did:web:alice.accounts.example.com";
    const FEDERATED_DID: &str = "did:plc:ar7c4by46qjdydhdevvrndac";
    const TENANT: &str = "tenant-a";

    struct FixtureDidResolver {
        did: String,
        document: serde_json::Value,
    }

    #[async_trait::async_trait]
    impl super::super::state::AtprotoDidDocumentResolver for FixtureDidResolver {
        async fn resolve_document(&self, did: &str) -> anyhow::Result<serde_json::Value> {
            anyhow::ensure!(did == self.did, "fixture DID mismatch");
            Ok(self.document.clone())
        }
    }

    struct PermitAccountReads;

    impl hyprstream_pds_service::AccountRecordReadAuthorizer for PermitAccountReads {
        fn check_read(
            &self,
            _subject: &hyprstream_rpc::Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&hyprstream_rpc::auth::mac::SecurityContext>,
            _object_id: &str,
        ) -> hyprstream_rpc::auth::mac::MacDecision {
            hyprstream_rpc::auth::mac::MacDecision::Permit
        }
    }

    struct SessionFixture {
        state: Arc<OAuthState>,
        cors: crate::config::CorsConfig,
        did: &'static str,
        atproto_key: p256::ecdsa::SigningKey,
        _storage: tempfile::TempDir,
    }

    fn fixture(local: bool) -> SessionFixture {
        let storage = tempfile::TempDir::new().unwrap();
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x71; 32]);
        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x72; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new("/dev/null/browser-session-test.sock".into()),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let mut oauth = crate::config::OAuthConfig::default();
        oauth.external_url = Some(ISSUER.to_owned());
        let cors = crate::config::CorsConfig {
            enabled: false,
            ..oauth.cors.clone()
        };

        let (did, atproto_key, document, hosted_store) = if local {
            let account_ed = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]);
            let (account_pq, account_pq_vk) = hyprstream_crypto::pq::ml_dsa_generate_keypair();
            let hybrid = hyprstream_pds::HybridRotationKey::new(
                account_ed.verifying_key().to_bytes(),
                hyprstream_crypto::pq::ml_dsa_vk_bytes(&account_pq_vk),
            )
            .unwrap();
            let rotations = hyprstream_pds::GenesisRotationKeys::new(
                hyprstream_pds::UserRotationKey::new(hybrid),
                hyprstream_pds::RecoveryKeyEnrollment::Declined,
                hyprstream_pds::HostKeyEnrollment::Absent,
            )
            .unwrap();
            let mint = hyprstream_pds::HostedAccountMint::begin(
                hyprstream_pds::AllocatedAccountName::new("alice", LOCAL_DID).unwrap(),
                rotations,
            )
            .unwrap();
            let account_document = mint.seal_did_document(ISSUER).unwrap();
            let pending = mint
                .prepare_genesis(
                    account_document,
                    hyprstream_pds::did_op::GenesisRepoHead::EmptyRepo,
                )
                .unwrap();
            let signature =
                hyprstream_pds::sign_genesis(pending.unsigned_genesis(), &account_ed, &account_pq)
                    .unwrap();
            let sealed = pending.seal(signature).unwrap();
            let atproto_key = sealed.atproto_signing_key().clone();
            let document = serde_json::from_slice(sealed.did_document().as_bytes()).unwrap();
            let root = SyntheticNode::dir().with_child(
                TENANT,
                SyntheticNode::dir().with_child(
                    "accounts",
                    SyntheticNode::dir().with_child(
                        "alice",
                        SyntheticNode::dir().with_child(
                            "account-record.cbor",
                            SyntheticNode::file(sealed.record_bytes().to_vec()),
                        ),
                    ),
                ),
            );
            let store = Arc::new(hyprstream_pds_service::AccountRecordStore::new(
                Arc::new(SyntheticMount::new(root)),
                Arc::new(PermitAccountReads),
            ));
            (LOCAL_DID, atproto_key, document, Some(store))
        } else {
            let atproto_key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
            let document = did_document(FEDERATED_DID, &atproto_key);
            (FEDERATED_DID, atproto_key, document, None)
        };

        let certified =
            rcgen::generate_simple_self_signed(vec!["pds.example.test".to_owned()]).unwrap();
        let cert_path = storage.path().join("quic-cert.pem");
        let key_path = storage.path().join("quic-key.pem");
        std::fs::write(&cert_path, certified.cert.pem()).unwrap();
        std::fs::write(&key_path, certified.key_pair.serialize_pem()).unwrap();
        let account = crate::account::AccountZoneConfig {
            zone: Some("accounts.example.com".to_owned()),
            ..Default::default()
        };
        let quic = crate::config::QuicConfig {
            enabled: true,
            bind_addr: "127.0.0.1:4433".to_owned(),
            server_name: "pds.example.test".to_owned(),
            cert_path: cert_path.to_string_lossy().into_owned(),
            key_path: key_path.to_string_lossy().into_owned(),
            iroh: false,
            relay: String::new(),
        };
        let registration_api =
            super::super::identity_registration::production_identity_registration_api(
                &oauth,
                &account,
                &quic,
                signing_key.clone(),
                storage.path().join("pds"),
            )
            .unwrap();

        let mut state = OAuthState::new(
            &oauth,
            crate::services::PolicyClient::new(make_client()),
            crate::services::DiscoveryClient::new(make_client()),
            signing_key.verifying_key().to_bytes(),
        )
        .with_atproto_did_resolver(Arc::new(FixtureDidResolver {
            did: did.to_owned(),
            document,
        }))
        .with_identity_registration_api(registration_api);
        if let Some(store) = hosted_store {
            state = state.with_hosted_account_store(store);
        }
        SessionFixture {
            state: Arc::new(state),
            cors,
            did,
            atproto_key,
            _storage: storage,
        }
    }

    fn did_document(did: &str, key: &p256::ecdsa::SigningKey) -> serde_json::Value {
        let mut multikey = vec![0x80, 0x24];
        multikey.extend_from_slice(key.verifying_key().to_encoded_point(true).as_bytes());
        serde_json::json!({
            "id": did,
            "verificationMethod": [{
                "id": format!("{did}#atproto"),
                "type": "Multikey",
                "controller": did,
                "publicKeyMultibase": format!("z{}", bs58::encode(multikey).into_string()),
            }]
        })
    }

    fn service_jwt(fixture: &SessionFixture, jti: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({
                "alg": "ES256", "typ": "JWT", "kid": "#atproto"
            }))
            .unwrap(),
        );
        let now = chrono::Utc::now().timestamp();
        let payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({
                "iss": fixture.did,
                "aud": fixture.state.atproto_service_did().unwrap(),
                "iat": now,
                "exp": now + 60,
                "lxm": ATPROTO_SESSION_EXCHANGE_NSID,
                "jti": jti,
            }))
            .unwrap(),
        );
        let signing_input = format!("{header}.{payload}");
        let signature: p256::ecdsa::Signature = fixture.atproto_key.sign(signing_input.as_bytes());
        format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    fn dpop_proof(jti: &str) -> String {
        let key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let point = key.verifying_key().to_encoded_point(false);
        let header = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({
                "alg": "ES256",
                "typ": "dpop+jwt",
                "jwk": {
                    "kty": "EC",
                    "crv": "P-256",
                    "x": URL_SAFE_NO_PAD.encode(point.x().unwrap()),
                    "y": URL_SAFE_NO_PAD.encode(point.y().unwrap()),
                }
            }))
            .unwrap(),
        );
        let payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&serde_json::json!({
                "htm": "POST",
                "htu": format!("{ISSUER}{SESSION_EXCHANGE_PATH}"),
                "iat": chrono::Utc::now().timestamp(),
                "jti": jti,
            }))
            .unwrap(),
        );
        let signing_input = format!("{header}.{payload}");
        let signature: p256::ecdsa::Signature = key.sign(signing_input.as_bytes());
        format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    async fn body(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn cookie(response: &Response) -> String {
        response.headers()[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_owned()
    }

    async fn exchange_request(
        fixture: &SessionFixture,
        service_jti: &str,
        dpop_jti: &str,
    ) -> Response {
        super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(
                axum::http::Request::post(SESSION_EXCHANGE_PATH)
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {}", service_jwt(fixture, service_jti)),
                    )
                    .header("DPoP", dpop_proof(dpop_jti))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn local_atproto_exchange_cookie_whoami_and_register_end_to_end() {
        let fixture = fixture(true);
        let exchange = exchange_request(&fixture, "local-service", "local-dpop").await;
        assert_eq!(exchange.status(), StatusCode::OK);
        let session_cookie = cookie(&exchange);
        let context = body(exchange).await;
        assert_eq!(context["did"], LOCAL_DID);
        assert_eq!(context["kind"], "local");
        assert_eq!(context["tenant"], TENANT);
        assert_eq!(context["canActLocally"], true);

        let app = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors);
        let whoami = app
            .clone()
            .oneshot(
                axum::http::Request::get(WHOAMI_PATH)
                    .header(header::COOKIE, &session_cookie)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(whoami.status(), StatusCode::OK);
        assert_eq!(body(whoami).await, context);

        let register = app
            .oneshot(
                axum::http::Request::post("/api/identity/register")
                    .header(header::COOKIE, session_cookie)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"handle":"new-account"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(register.status(), StatusCode::OK);
        assert_eq!(
            body(register).await["did"],
            "did:web:new-account.accounts.example.com"
        );
    }

    #[tokio::test]
    async fn federated_exchange_whoami_has_no_local_authority() {
        let fixture = fixture(false);
        let exchange = exchange_request(&fixture, "foreign-service", "foreign-dpop").await;
        assert_eq!(exchange.status(), StatusCode::OK);
        let session_cookie = cookie(&exchange);
        let context = body(exchange).await;
        assert_eq!(context["did"], FEDERATED_DID);
        assert_eq!(context["kind"], "federated");
        assert!(context["tenant"].is_null());
        assert_eq!(context["canActLocally"], false);

        let whoami = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors)
            .oneshot(
                axum::http::Request::get(WHOAMI_PATH)
                    .header(header::COOKIE, session_cookie)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body(whoami).await, context);
    }

    #[tokio::test]
    async fn unauthenticated_whoami_is_public_floor() {
        let fixture = fixture(false);
        let generic_id = fixture
            .state
            .sessions
            .create(FEDERATED_DID.to_owned(), "atproto".to_owned())
            .await;
        let app = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors);
        for cookie in [
            None,
            Some(format!("{}=unknown", session::SESSION_COOKIE_NAME)),
            Some(format!("{}={generic_id}", session::SESSION_COOKIE_NAME)),
        ] {
            let mut request = axum::http::Request::get(WHOAMI_PATH)
                .body(axum::body::Body::empty())
                .unwrap();
            if let Some(cookie) = cookie {
                request
                    .headers_mut()
                    .insert(header::COOKIE, cookie.parse().unwrap());
            }
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                body(response).await,
                serde_json::json!({
                    "did": null,
                    "kind": "unauthenticated",
                    "tenant": null,
                    "canActLocally": false,
                })
            );
        }
    }

    #[tokio::test]
    async fn exchange_requires_dpop_and_consumes_service_assertion_once() {
        let fixture = fixture(false);
        let app = super::super::create_app(Arc::clone(&fixture.state), &fixture.cors);
        let missing_dpop = app
            .clone()
            .oneshot(
                axum::http::Request::post(SESSION_EXCHANGE_PATH)
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {}", service_jwt(&fixture, "missing-dpop")),
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_dpop.status(), StatusCode::BAD_REQUEST);
        assert!(!missing_dpop.headers().contains_key(header::SET_COOKIE));

        let first = exchange_request(&fixture, "one-use-service", "first-dpop").await;
        assert_eq!(first.status(), StatusCode::OK);
        let replay = exchange_request(&fixture, "one-use-service", "fresh-dpop").await;
        assert_eq!(replay.status(), StatusCode::UNAUTHORIZED);
        assert!(!replay.headers().contains_key(header::SET_COOKIE));
        assert_eq!(body(replay).await["error"], "invalid_token");
    }
}
