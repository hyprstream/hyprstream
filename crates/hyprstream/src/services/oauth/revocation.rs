//! OAuth 2.0 Token Revocation (RFC 7009).
//!
//! `POST /oauth/revoke` — revokes a refresh token.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Form;
use serde::Deserialize;

use super::state::OAuthState;

#[derive(Deserialize)]
pub struct RevocationRequest {
    /// The token to revoke.
    pub token: String,
    /// Optional hint: "refresh_token" or "access_token".
    #[serde(default)]
    pub token_type_hint: Option<String>,
}

/// POST /oauth/revoke (RFC 7009)
///
/// Revokes a refresh token. Access tokens are stateless JWTs and cannot be
/// revoked (they expire naturally). Per RFC 7009 Section 2.1, the server
/// MUST respond with 200 OK even if the token is invalid or already revoked.
pub async fn revoke_token(
    State(state): State<Arc<OAuthState>>,
    Form(params): Form<RevocationRequest>,
) -> Response {
    // Try to remove as a refresh token
    {
        let mut tokens = state.refresh_tokens.write().await;
        if tokens.remove(&params.token).is_some() {
            tracing::info!("Revoked refresh token");
        }
    }

    // RFC 7009: always return 200 OK regardless of whether the token was found.
    StatusCode::OK.into_response()
}
