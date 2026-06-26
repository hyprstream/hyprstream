//! OAuth 2.0 Token Revocation (RFC 7009).
//!
//! `POST /oauth/revoke` — revokes refresh tokens and access tokens.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Form;
use serde::Deserialize;

use hyprstream_rpc::auth::JtiBlocklist as _;

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
/// Revokes a refresh token or access token. Access tokens with a `jti` claim
/// are added to the blocklist (checked by `verify_claims` on every request).
/// Per RFC 7009 Section 2.1, the server MUST respond with 200 OK even if the
/// token is invalid or already revoked.
pub async fn revoke_token(
    State(state): State<Arc<OAuthState>>,
    Form(params): Form<RevocationRequest>,
) -> Response {
    let is_access_hint = params.token_type_hint.as_deref() == Some("access_token");

    if !is_access_hint {
        if let Err(e) = state.delete_refresh_token(&params.token).await {
            tracing::warn!(error = %e, "Refresh token store delete failed during revocation");
        } else {
            tracing::info!("Revoked refresh token");
        }
    }

    if is_access_hint || params.token_type_hint.is_none() {
        if let Some(ref blocklist) = state.jti_blocklist {
            match hyprstream_rpc::auth::decode_unverified(&params.token) {
                Ok(claims) => {
                    if let Some(jti) = claims.jti {
                        blocklist.revoke(jti, claims.exp);
                        tracing::info!(sub = %claims.sub, "Revoked access token via jti blocklist");
                    }
                }
                Err(_) => {
                    // Not a valid JWT — may be a refresh token tried above, or invalid.
                    // RFC 7009: always 200.
                }
            }
        }
    }

    StatusCode::OK.into_response()
}
