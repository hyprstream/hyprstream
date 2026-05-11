//! Bearer token authentication middleware for the OAuth service.
//!
//! Applied as an Axum middleware layer to the protected route group (SCIM
//! user management). Public endpoints (discovery, token, authorize) bypass
//! this layer.
//!
//! On success, inserts `AuthenticatedUser` into request extensions so
//! downstream handlers can identify the caller.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::state::OAuthState;
pub use crate::server::middleware::AuthenticatedUser;

/// Require a valid Bearer or DPoP token on the request.
///
/// Validates the Ed25519-signed JWT against the server's verifying key.
/// When `Authorization: DPoP` is used, also verifies the `DPoP` proof header and
/// checks that `cnf.jkt` in the token matches the proof key's thumbprint.
/// Inserts `AuthenticatedUser` into request extensions on success.
/// Returns 401 if the token is absent, expired, or has an invalid signature.
pub async fn require_bearer_token(
    State(state): State<Arc<OAuthState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let (scheme, token) = match auth_header.as_deref().and_then(|h| {
        if let Some(t) = h.strip_prefix("Bearer ") { Some(("Bearer", t.to_owned())) }
        else if let Some(t) = h.strip_prefix("DPoP ") { Some(("DPoP", t.to_owned())) }
        else { None }
    }) {
        Some(pair) => pair,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "Bearer token required"})),
            )
                .into_response();
        }
    };

    // For DPoP, also grab the DPoP proof header.
    let dpop_proof_str = if scheme == "DPoP" {
        match request.headers().get("DPoP").and_then(|v| v.to_str().ok()) {
            Some(p) => Some(p.to_owned()),
            None => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "DPoP proof header required when using DPoP scheme"})),
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    let vk = match ed25519_dalek::VerifyingKey::from_bytes(&state.verifying_key_bytes) {
        Ok(k) => k,
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "server configuration error")
                .into_response();
        }
    };

    let claims = match hyprstream_rpc::auth::jwt::decode(&token, &vk, None) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "Token invalid or expired"})),
            )
                .into_response();
        }
    };

    // For DPoP: verify the proof and validate cnf.jkt matches the proof key.
    if let Some(ref proof_str) = dpop_proof_str {
        let method = request.method().as_str();
        let uri = request.uri();
        let htu = format!("{}{}",
            uri.scheme_str().map(|s| format!("{s}://")).unwrap_or_default(),
            uri.path(),
        );
        let proof = match super::dpop::verify_dpop_proof(proof_str, method, &htu, Some(&token)) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_dpop_proof", "error_description": e.to_string()})),
                )
                    .into_response();
            }
        };
        // JTI replay prevention for resource requests (RFC 9449 §11.1).
        if !state.check_and_record_dpop_jti(&proof.jti, proof.iat).await {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                axum::Json(serde_json::json!({"error": "invalid_dpop_proof", "error_description": "DPoP proof jti already used"})),
            )
                .into_response();
        }
        // Verify cnf.jkt in token matches the DPoP proof key.
        if let Some(token_jkt) = claims.cnf_jkt() {
            if token_jkt != proof.jkt {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "cnf.jkt does not match DPoP proof key"})),
                )
                    .into_response();
            }
        }
    }

    request.extensions_mut().insert(AuthenticatedUser {
        user: claims.sub,
        token: Some(token),
        exp: Some(claims.exp),
    });

    next.run(request).await
}
