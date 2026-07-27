//! OpenID Connect UserInfo endpoint (OIDC Core Section 5.3).
//!
//! Returns claims about the authenticated user based on the access token's
//! scopes: `openid` → sub, `profile` → name, `email` → email.
//!
//! Authentication is enforced by the `require_bearer_token` middleware layer
//! applied to the protected route group. By the time this handler runs, the
//! token has already been validated and the caller's identity is in extensions.

use std::sync::Arc;

use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::auth::AuthenticatedUser;
use super::state::OAuthState;

/// GET/POST /oauth/userinfo
pub async fn userinfo(
    State(state): State<Arc<OAuthState>>,
    Extension(caller): Extension<AuthenticatedUser>,
) -> Response {
    let Some(user_store) = state.user_store_reader() else {
        tracing::error!("UserInfo unavailable: relational user store is not configured");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "temporarily_unavailable"})),
        )
            .into_response();
    };
    let profile = match user_store.get_profile(&caller.user).await {
        Ok(Some(profile)) => profile,
        Ok(None) => {
            tracing::error!(username = %caller.user, "UserInfo subject has no credential profile");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "server_error"})),
            )
                .into_response();
        }
        Err(error) => {
            tracing::error!(username = %caller.user, %error, "UserInfo credential lookup failed");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "temporarily_unavailable"})),
            )
                .into_response();
        }
    };
    let Some(stable_sub) = profile.sub else {
        tracing::error!(username = %caller.user, "UserInfo profile has no stable subject");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "server_error"})),
        )
            .into_response();
    };

    let mut response = serde_json::json!({
        "sub": stable_sub,
        "preferred_username": caller.user,
    });
    if let Some(name) = profile.name {
        response["name"] = serde_json::Value::String(name);
    }
    if let Some(email) = profile.email {
        response["email"] = serde_json::Value::String(email);
    }
    if let Some(verified) = profile.email_verified {
        response["email_verified"] = serde_json::Value::Bool(verified);
    }

    (StatusCode::OK, Json(response)).into_response()
}
