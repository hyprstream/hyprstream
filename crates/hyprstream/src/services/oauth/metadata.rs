//! OAuth 2.1 / OpenID Connect discovery metadata.
//!
//! - `GET /.well-known/oauth-authorization-server` — RFC 8414
//! - `GET /.well-known/openid-configuration` — OIDC Discovery 1.0

use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, Json};

use super::state::OAuthState;

/// Base metadata fields shared between RFC 8414 and OIDC discovery.
fn base_metadata(issuer: &str, scopes: &[String]) -> serde_json::Value {
    serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/oauth/authorize", issuer),
        "token_endpoint": format!("{}/oauth/token", issuer),
        "registration_endpoint": format!("{}/oauth/register", issuer),
        "device_authorization_endpoint": format!("{}/oauth/device", issuer),
        "jwks_uri": format!("{}/oauth/jwks", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": scopes,
        "client_id_metadata_document_supported": true,
    })
}

/// GET /.well-known/oauth-authorization-server (RFC 8414)
pub async fn authorization_server_metadata(
    State(state): State<Arc<OAuthState>>,
) -> impl IntoResponse {
    Json(base_metadata(&state.issuer_url, &state.default_scopes))
}

/// GET /.well-known/openid-configuration (OIDC Discovery 1.0)
///
/// Superset of RFC 8414 metadata with OIDC-specific fields:
/// userinfo_endpoint, id_token_signing_alg_values_supported,
/// subject_types_supported, claims_supported.
pub async fn openid_configuration(
    State(state): State<Arc<OAuthState>>,
) -> impl IntoResponse {
    let issuer = &state.issuer_url;
    let mut meta = base_metadata(issuer, &state.default_scopes);

    // OIDC-specific fields
    let obj = meta.as_object_mut().unwrap_or_else(|| unreachable!());
    obj.insert("userinfo_endpoint".into(),
        serde_json::Value::String(format!("{}/oauth/userinfo", issuer)));
    obj.insert("id_token_signing_alg_values_supported".into(),
        serde_json::json!(["EdDSA"]));
    obj.insert("subject_types_supported".into(),
        serde_json::json!(["public"]));
    obj.insert("claims_supported".into(),
        serde_json::json!([
            "sub", "iss", "aud", "exp", "iat", "nonce", "auth_time",
            "name", "email", "email_verified", "preferred_username"
        ]));

    Json(meta)
}
