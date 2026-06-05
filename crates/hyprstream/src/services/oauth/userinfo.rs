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
    let mut response = serde_json::json!({
        "sub": caller.user,
    });

    if let Some(user_store) = state.user_store_reader() {
        if let Ok(Some(profile)) = user_store.get_profile(&caller.user).await {
            if let Some(ref uuid_sub) = profile.sub {
                response["sub"] = serde_json::Value::String(uuid_sub.clone());
            }
            response["preferred_username"] = serde_json::Value::String(caller.user.clone());
            if let Some(ref name) = profile.name {
                response["name"] = serde_json::Value::String(name.clone());
            }
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
