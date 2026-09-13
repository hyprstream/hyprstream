//! Browser WIT bootstrap endpoint — DISABLED for v16 (frozen A).
//!
//! `POST /oauth/wit` formerly minted a HYBRID `wit+jwt` (a SERVICE credential
//! type) for a USER subject, bound to a caller-SUBMITTED (untrusted) Ed25519
//! public key, with no RFC 9068 `client_id`, no authoritative interactive /
//! non-interactive session classification, and no authoritative Primary
//! signer-suite enrollment.
//!
//! Frozen A requires a hybrid credential to be an `at+jwt` (never a service
//! `wit+jwt`) and a user credential to carry a non-empty `client_id` + an active
//! session; §5/T1 forbids a self-asserted wire key as the confirmation. The
//! `AuthenticatedUser` request extension exposes only identity/token/exp, so
//! this route cannot authoritatively recover any of those and cannot mint a
//! v16-conformant credential. It is therefore **disabled fail-closed** — a
//! browser access token is issued through the typed OAuth issuance profiles
//! that carry those authorities. (Git history holds the removed mint; it is not
//! retained here as unreachable code.)

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde::Deserialize;

use crate::server::middleware::AuthenticatedUser;
use super::state::OAuthState;

#[derive(Deserialize)]
pub struct WitRequest {
    /// Base64url-encoded Ed25519 public key — retained for wire compatibility of
    /// the disabled route; no longer trusted, decoded, or used.
    pub pubkey: String,
}

/// POST /oauth/wit — DISABLED fail-closed for v16 (see the module docs). Returns
/// `501 Not Implemented` for every request, before any credential is minted.
pub async fn issue_browser_wit(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(body): Json<WitRequest>,
) -> Response {
    let _ = (&state, &user, &body);
    (
        StatusCode::NOT_IMPLEMENTED,
        [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
        Json(serde_json::json!({
            "error": "unsupported_grant_type",
            "error_description": "browser WIT bootstrap is disabled: a v16 credential cannot be \
                minted from an untrusted submitted key without an OAuth client_id, an active \
                session, and authoritative signer-suite enrollment; use the OAuth token flow",
        })),
    )
        .into_response()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::OAuthConfig;
    use crate::services::{DiscoveryClient, PolicyClient};
    use axum::extract::Extension;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    use hyprstream_rpc::signer::LocalSigner;
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

    fn test_state() -> Arc<OAuthState> {
        let key = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]);
        let dummy = std::path::PathBuf::from("/dev/null/wit-freeze-test.sock");
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(key.clone()),
                    LazyUdsTransport::new(dummy.clone()),
                    Some(key.verifying_key()),
                )
                .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical),
            )
        };
        Arc::new(
            OAuthState::new(
                &OAuthConfig::default(),
                PolicyClient::new(make_client()),
                DiscoveryClient::new(make_client()),
                key.verifying_key().to_bytes(),
            )
            .with_ca_jwt_key(key),
        )
    }

    fn request() -> WitRequest {
        WitRequest { pubkey: String::new() }
    }

    /// The browser WIT bootstrap route is DISABLED fail-closed for v16: it
    /// returns `501 Not Implemented` with an `unsupported_grant_type` body and
    /// never mints a credential.
    #[tokio::test]
    async fn browser_wit_route_is_disabled_fail_closed() {
        let response = issue_browser_wit(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "alice".to_owned(),
                verified_tenant: Some("tenant-a.example".to_owned()),
                token: None,
                exp: None,
            }),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"].as_str(), Some("unsupported_grant_type"));
    }

    /// The refusal is unconditional — even a path-form subject (which the old
    /// mint rejected only later) is refused before any signing.
    #[tokio::test]
    async fn browser_wit_route_disabled_for_path_form_subject() {
        let response = issue_browser_wit(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "did:web:accounts.example:users:alice".to_owned(),
                verified_tenant: Some("tenant-a.example".to_owned()),
                token: None,
                exp: None,
            }),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
