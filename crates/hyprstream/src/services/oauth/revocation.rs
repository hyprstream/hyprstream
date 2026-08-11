//! OAuth 2.0 Token Revocation (RFC 7009).
//!
//! `POST /oauth/revoke` — revokes refresh tokens and access tokens.

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
        // Use the process-global credential-revocation store.
        if let Some(store) = hyprstream_rpc::auth::global_credential_revocation_store() {
            // Verify the token using the same single verification stack as
            // the auth path (typ, aud, issuer, composite/classical). An
            // unverified token's iss/jti/exp are attacker-controlled; a
            // forged token must not revoke another credential. If
            // verification fails, return 200 per RFC 7009 but do not
            // modify the store.
            match super::auth::validate_oauth_access_token(state.as_ref(), &params.token).await {
                Ok(claims) => {
                    if let Some(ref jti) = claims.jti {
                        let cred_id = hyprstream_rpc::auth::CredentialId::jwt(&claims.iss, jti);
                        store.revoke_credential(cred_id, claims.exp);
                        tracing::info!(sub = %claims.sub, "Revoked access token via credential revocation store");
                    }
                }
                Err(_) => {
                    // Token verification failed — may be expired, wrong key,
                    // or not a JWT. RFC 7009: still 200.
                }
            }
        }
    }

    StatusCode::OK.into_response()
}
