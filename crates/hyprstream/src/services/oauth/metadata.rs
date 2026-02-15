//! OAuth 2.1 Authorization Server Metadata (RFC 8414).
//!
//! Serves `GET /.well-known/oauth-authorization-server`.

use std::sync::Arc;

use axum::{extract::State, response::IntoResponse, Json};

use super::state::OAuthState;

/// GET /.well-known/oauth-authorization-server (RFC 8414)
pub async fn authorization_server_metadata(
    State(state): State<Arc<OAuthState>>,
) -> impl IntoResponse {
    let issuer = &state.issuer_url;

    Json(serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{}/oauth/authorize", issuer),
        "token_endpoint": format!("{}/oauth/token", issuer),
        "registration_endpoint": format!("{}/oauth/register", issuer),
        "device_authorization_endpoint": format!("{}/oauth/device", issuer),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "urn:ietf:params:oauth:grant-type:device_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": state.default_scopes,
        "client_id_metadata_document_supported": true,
    }))
}
