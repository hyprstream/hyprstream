//! RFC 7662 Token Introspection endpoint.
//!
//! `POST /oauth/introspect` — validates a token and returns its metadata.
//! Caller must authenticate with a Bearer token.
//!
//! - JWT access tokens: verified against the server's Ed25519 key.
//! - Opaque refresh tokens: looked up from RocksDB with lazy expiry.
//! - Any failure → `{"active": false}` per RFC 7662 § 2.2.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Form, Json,
};
use serde::Deserialize;

use super::state::OAuthState;

#[derive(Deserialize)]
pub struct IntrospectRequest {
    pub token: String,
    #[serde(default)]
    pub token_type_hint: Option<String>,
}

/// POST /oauth/introspect (RFC 7662)
pub async fn introspect_token(
    State(state): State<Arc<OAuthState>>,
    Form(params): Form<IntrospectRequest>,
) -> Response {
    // JWT tokens contain dots; opaque tokens do not.
    if params.token.contains('.') {
        introspect_jwt(&state, &params.token)
    } else {
        introspect_refresh(&state, &params.token).await
    }
}

fn introspect_jwt(state: &Arc<OAuthState>, token: &str) -> Response {
    let vk = match ed25519_dalek::VerifyingKey::from_bytes(&state.verifying_key_bytes) {
        Ok(k) => k,
        Err(_) => return inactive_response(),
    };

    let claims = match hyprstream_rpc::auth::jwt::decode(token, &vk, None) {
        Ok(c) => c,
        Err(_) => return inactive_response(),
    };

    let now = chrono::Utc::now().timestamp();
    if claims.exp < now {
        return inactive_response();
    }

    (
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
        Json(serde_json::json!({
            "active": true,
            "token_type": "Bearer",
            "sub": claims.sub,
            "iss": claims.iss,
            "exp": claims.exp,
            "iat": claims.iat,
            "aud": claims.aud,
        })),
    )
        .into_response()
}

async fn introspect_refresh(state: &Arc<OAuthState>, token: &str) -> Response {
    match state.get_refresh_token(token).await {
        Ok(Some(entry)) => {
            // #1113: report the account DID as `sub` for atproto conformance,
            // matching the access-token subject. The refresh-token entry is
            // keyed on the internal username; fall back to it if DID
            // derivation fails (introspection is informational — the token
            // path itself fails closed at mint time).
            let sub = state
                .subject_did(&entry.username)
                .unwrap_or_else(|_| entry.username.clone());
            (
                StatusCode::OK,
                [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
                Json(serde_json::json!({
                    "active": true,
                    "token_type": "refresh_token",
                    "sub": sub,
                    "client_id": entry.client_id,
                    "scope": entry.scopes.join(" "),
                    "exp": entry.expires_at_unix,
                })),
            )
                .into_response()
        }
        _ => inactive_response(),
    }
}

fn inactive_response() -> Response {
    (
        StatusCode::OK,
        [(header::CACHE_CONTROL, "no-store"), (header::PRAGMA, "no-cache")],
        Json(serde_json::json!({"active": false})),
    )
        .into_response()
}
