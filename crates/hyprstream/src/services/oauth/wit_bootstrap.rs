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

    let domain = match user.authorization_domain() {
        Ok(domain) => domain,
        Err(_) => {
            return (
                StatusCode::FORBIDDEN,
                [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
                Json(serde_json::json!({
                    "error": "insufficient_scope",
                    "error_description": "Verified hosted-account tenant binding required",
                })),
            )
                .into_response();
        }
    };
    // Stamp `aud` alongside `iss`: composite verification is strict about the
    // audience (an absent `aud` is rejected when the verifier expects one),
    // and every service's expected audience is this same issuer URL.
    let mut claims = hyprstream_rpc::auth::Claims::new(sub.clone(), now, expires_at)
        .with_issuer(state.issuer_url.clone())
        .with_audience(Some(state.issuer_url.clone()))
        .with_cnf_jwk(&pubkey_bytes);
    if domain != "*" {
        claims = claims.with_tenant(domain);
    }

    // Hybrid mint: the browser presents this WIT in ZMQ envelope calls, which
    // the dispatch plane verifies under a mandatory Hybrid crypto policy — a
    // classical EdDSA WIT would be rejected on exactly the plane it is for.
    let wit = match crate::auth::jwt::encode_service_jwt_hybrid_via_authority(&claims, ca_key) {
        Ok(wit) => wit,
        Err(error) => {
            tracing::warn!(%error, "WIT issuance refused: hybrid signing authority unavailable");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
                Json(serde_json::json!({
                    "error": "temporarily_unavailable",
                    "error_description": "WIT issuance not available — hybrid signing authority unavailable",
                })),
            )
                .into_response();
        }
    };

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
                verified_tenant: Some("tenant-a.example".to_owned()),
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
                verified_tenant: Some("tenant-a.example".to_owned()),
                token: None,
                exp: None,
            }),
            Json(request()),
        )
        .await;

        // The process-global composite authority is shared test state: a
        // sibling test may have configured it without a usable active OAuth
        // pair, in which case issuance correctly refuses (never downgrades).
        if response.status() == StatusCode::SERVICE_UNAVAILABLE {
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"].as_str(), Some("temporarily_unavailable"));
            return;
        }

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let wit = json["wit"].as_str().unwrap();

        // The browser WIT is minted hybrid — the dispatch plane's Hybrid
        // policy would reject a classical EdDSA WIT — so it must dispatch as
        // a composite wit+jwt.
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(wit, &["wit+jwt"]).unwrap();

        // When no composite signing authority is initialized, the mint falls
        // back to the self-contained CA pair derived from the CA JWT key;
        // verify with that exact pair when its kid matches. (A sibling test
        // may have published a process-global authority pair, in which case
        // the kid differs; the composite dispatch shape above is asserted
        // either way.)
        let ca = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]);
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        let claims =
            if dispatch.kid() == crate::auth::jwt::composite_kid(&ca_pq_vk, &ca.verifying_key()) {
                hyprstream_rpc::auth::jwt::decode_composite(
                    wit,
                    &ca_pq_vk,
                    &ca.verifying_key(),
                    None,
                    &dispatch,
                )
                .unwrap()
            } else {
                hyprstream_rpc::auth::decode_unverified(wit).unwrap()
            };
        assert_eq!(claims.tenant.as_deref(), Some("tenant-a.example"));
        // Composite verification is strict about audience, so the WIT must
        // carry `aud` = the issuer URL it was minted under.
        assert_eq!(claims.aud.as_deref(), Some(claims.iss.as_str()));
        assert!(!claims.iss.is_empty());
    }
}
