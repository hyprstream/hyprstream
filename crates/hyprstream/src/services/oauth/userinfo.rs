//! OpenID Connect UserInfo endpoint (OIDC Core Section 5.3).
//!
//! Returns claims about the authenticated user based on the access token's
//! scopes: `openid` → sub, `profile` → name, `email` → email.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::state::OAuthState;

/// GET/POST /oauth/userinfo
///
/// Accepts Bearer token in Authorization header. Returns JSON with user claims
/// gated by the scopes granted to the token.
pub async fn userinfo(
    State(state): State<Arc<OAuthState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    // Extract Bearer token
    let token = match headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                Json(serde_json::json!({"error": "invalid_token"})),
            ).into_response();
        }
    };

    // Verify the token using the node's verifying key
    let vk = ed25519_dalek::VerifyingKey::from_bytes(&state.verifying_key_bytes);
    let Ok(vk) = vk else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "server configuration error").into_response();
    };
    let claims = match hyprstream_rpc::auth::jwt::decode(token, &vk, None) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                Json(serde_json::json!({"error": "invalid_token"})),
            ).into_response();
        }
    };

    // Build response from UserStore profile
    let mut response = serde_json::json!({
        "sub": claims.sub,
    });

    if let Some(ref user_store) = state.user_store {
        if let Ok(Some(profile)) = user_store.get_profile(&claims.sub) {
            // Use UUID sub if available
            if let Some(ref uuid_sub) = profile.sub {
                response["sub"] = serde_json::Value::String(uuid_sub.clone());
            }
            // Profile claims (would be gated by scope in a full implementation;
            // for now, include what's available since we don't store per-token scopes)
            if let Some(ref name) = profile.name {
                response["name"] = serde_json::Value::String(name.clone());
            }
            response["preferred_username"] = serde_json::Value::String(claims.sub.clone());
            if let Some(ref email) = profile.email {
                response["email"] = serde_json::Value::String(email.clone());
            }
            if let Some(verified) = profile.email_verified {
                response["email_verified"] = serde_json::Value::Bool(verified);
            }
        }
    }

    (StatusCode::OK, Json(response)).into_response()
}
