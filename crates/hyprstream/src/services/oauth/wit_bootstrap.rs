//! Browser WIT bootstrap endpoint.
//!
//! `POST /oauth/wit` — exchanges a PKCE-issued `at+jwt` for a `wit+jwt`
//! that binds the browser's vault (or ephemeral) Ed25519 pubkey via `cnf.jwk`.
//!
//! The browser can then use the WIT in ZMQ envelope calls, giving the same
//! key-bound identity story as service workloads. ExchangeWit becomes
//! available after the browser holds a WIT.
//!
//! Auth: `Authorization: Bearer <at+jwt>` (verified by `require_bearer_token`).
//! Body: `application/json` — `{ "pubkey": "<base64url Ed25519 pubkey>" }`.
//! Response: `{ "wit": "<wit+jwt>", "expires_in": <seconds> }`.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Extension, Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_pds::repo_authority::is_path_form_did_web;
use serde::Deserialize;

use crate::server::middleware::AuthenticatedUser;
use super::state::OAuthState;

/// Browser WIT TTL: 8 hours. Shorter than service WITs (30 days).
const BROWSER_WIT_TTL: i64 = 8 * 3600;

#[derive(Deserialize)]
pub struct WitRequest {
    /// Base64url-encoded 32-byte Ed25519 public key to bind in `cnf.jwk`.
    pub pubkey: String,
}

/// POST /oauth/wit — issue a browser WIT bound to the caller's Ed25519 pubkey.
pub async fn issue_browser_wit(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(body): Json<WitRequest>,
) -> Response {
    if is_path_form_did_web(&user.user) {
        return (
            StatusCode::BAD_REQUEST,
            [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
            Json(serde_json::json!({
                "error": "invalid_grant",
                "error_description": "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)",
            })),
        )
            .into_response();
    }

    let ca_key_arc = state.active_jwt_signing_key().await;
    let ca_key = match ca_key_arc.as_deref() {
        Some(k) => k,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
                Json(serde_json::json!({
                    "error": "temporarily_unavailable",
                    "error_description": "WIT issuance not available — CA signing key not loaded",
                })),
            ).into_response();
        }
    };

    // Decode and validate the submitted Ed25519 public key.
    let pubkey_bytes: [u8; 32] = match URL_SAFE_NO_PAD.decode(&body.pubkey)
        .ok()
        .and_then(|b| b.try_into().ok())
    {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
                Json(serde_json::json!({
                    "error": "invalid_request",
                    "error_description": "pubkey must be base64url-encoded 32-byte Ed25519 public key",
                })),
            ).into_response();
        }
    };

    // Validate the key bytes form a valid Ed25519 point.
    if ed25519_dalek::VerifyingKey::from_bytes(&pubkey_bytes).is_err() {
        return (
            StatusCode::BAD_REQUEST,
            [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
            Json(serde_json::json!({
                "error": "invalid_request",
                "error_description": "pubkey is not a valid Ed25519 public key",
            })),
        ).into_response();
    }

    let sub = &user.user;
    let now = chrono::Utc::now().timestamp();
    let expires_at = now + BROWSER_WIT_TTL;

    let claims = hyprstream_rpc::auth::Claims::new(sub.clone(), now, expires_at)
        .with_issuer(state.issuer_url.clone())
        .with_cnf_jwk(&pubkey_bytes);

    let wit = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, ca_key);

    tracing::info!(sub = %sub, "Browser WIT issued");

    (
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
        Json(serde_json::json!({
            "wit": wit,
            "expires_in": BROWSER_WIT_TTL,
        })),
    ).into_response()
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
        let make_client = || Arc::new(
            RpcClientImpl::new(
                LocalSigner::new(key.clone()),
                LazyUdsTransport::new(dummy.clone()),
                Some(key.verifying_key()),
            )
            .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical),
        );
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
        let public_key = ed25519_dalek::SigningKey::from_bytes(&[0x74; 32]).verifying_key();
        WitRequest {
            pubkey: URL_SAFE_NO_PAD.encode(public_key.as_bytes()),
        }
    }

    #[tokio::test]
    async fn browser_wit_rejects_path_form_authenticated_user_before_signing() {
        let response = issue_browser_wit(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "did:web:accounts.example:users:alice".to_owned(),
                token: None,
                exp: None,
            }),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn browser_wit_allows_ordinary_authenticated_user() {
        let response = issue_browser_wit(
            State(test_state()),
            Extension(AuthenticatedUser {
                user: "alice".to_owned(),
                token: None,
                exp: None,
            }),
            Json(request()),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
    }
}
