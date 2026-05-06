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

/// Require a valid Bearer token on the request.
///
/// Validates the Ed25519-signed JWT against the server's verifying key.
/// Inserts `AuthenticatedUser` into request extensions on success.
/// Returns 401 if the token is absent, expired, or has an invalid signature.
pub async fn require_bearer_token(
    State(state): State<Arc<OAuthState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let token = match token {
        Some(t) => t.to_owned(),
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "Bearer token required"})),
            )
                .into_response();
        }
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

    request.extensions_mut().insert(AuthenticatedUser {
        user: claims.sub,
        token: Some(token),
        exp: Some(claims.exp),
    });

    next.run(request).await
}
