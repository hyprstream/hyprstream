//! OAuth 2.1 Client Registration (RFC 7591) and Client ID Metadata Documents.
//!
//! Two registration paths:
//! - **Client ID Metadata Documents** (preferred): client_id is an HTTPS URL
//! - **Dynamic Client Registration** (fallback): POST /oauth/register

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use super::state::{OAuthState, RegisteredClient};

/// Dynamic client registration request (RFC 7591)
#[derive(Debug, Deserialize)]
pub struct RegistrationRequest {
    pub redirect_uris: Vec<String>,
    #[serde(default)]
    pub client_name: Option<String>,
}

/// Dynamic client registration response
#[derive(Debug, Serialize)]
pub struct RegistrationResponse {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
}

/// POST /oauth/register — Dynamic Client Registration (RFC 7591)
pub async fn register_client(
    State(state): State<Arc<OAuthState>>,
    Json(req): Json<RegistrationRequest>,
) -> impl IntoResponse {
    // Validate redirect URIs are loopback only
    for uri in &req.redirect_uris {
        if !is_loopback_uri(uri) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid_redirect_uri",
                    "error_description": "Only loopback redirect URIs are allowed (http://127.0.0.1:* or http://localhost:*)"
                })),
            ).into_response();
        }
    }

    if req.redirect_uris.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "At least one redirect_uri is required"
            })),
        ).into_response();
    }

    let client_id = uuid::Uuid::new_v4().to_string();

    let client = RegisteredClient {
        client_id: client_id.clone(),
        redirect_uris: req.redirect_uris.clone(),
        client_name: req.client_name.clone(),
        is_cimd: false,
        registered_at: Instant::now(),
    };

    state.clients.write().await.insert(client_id.clone(), client);

    Json(RegistrationResponse {
        client_id,
        redirect_uris: req.redirect_uris,
        client_name: req.client_name,
    }).into_response()
}

/// Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)
#[derive(Debug, Deserialize)]
pub struct ClientIdMetadataDocument {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    #[serde(default)]
    pub client_name: Option<String>,
}

/// Fetch and validate a Client ID Metadata Document from an HTTPS URL.
///
/// Returns the registered client on success, or an error description on failure.
pub async fn fetch_client_metadata(
    state: &OAuthState,
    client_id_url: &str,
) -> Result<RegisteredClient, String> {
    // SSRF protection: must be HTTPS
    if !client_id_url.starts_with("https://") {
        return Err("Client ID Metadata Document URL must use HTTPS".to_owned());
    }

    // SSRF protection: block private/loopback IPs
    if let Ok(url) = url::Url::parse(client_id_url) {
        if let Some(host) = url.host_str() {
            if is_private_host(host) {
                return Err("Client ID URL must not point to private/loopback addresses".to_owned());
            }
        }
    }

    let response = state.http_client
        .get(client_id_url)
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch client metadata: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Client metadata fetch returned HTTP {}",
            response.status()
        ));
    }

    let doc: ClientIdMetadataDocument = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse client metadata: {}", e))?;

    // Validate client_id in document matches the URL
    if doc.client_id != client_id_url {
        return Err(format!(
            "client_id mismatch: document contains '{}' but URL is '{}'",
            doc.client_id, client_id_url
        ));
    }

    if doc.redirect_uris.is_empty() {
        return Err("Client metadata must include at least one redirect_uri".to_owned());
    }

    Ok(RegisteredClient {
        client_id: client_id_url.to_owned(),
        redirect_uris: doc.redirect_uris,
        client_name: doc.client_name,
        is_cimd: true,
        registered_at: Instant::now(),
    })
}

/// Check if a redirect URI is a loopback address.
pub(crate) fn is_loopback_uri(uri: &str) -> bool {
    if let Ok(url) = url::Url::parse(uri) {
        if url.scheme() != "http" {
            return false;
        }
        match url.host_str() {
            Some("127.0.0.1") | Some("localhost") | Some("[::1]") => true,
            _ => false,
        }
    } else {
        false
    }
}

/// Check if a hostname resolves to a private/loopback address.
fn is_private_host(host: &str) -> bool {
    matches!(
        host,
        "localhost" | "127.0.0.1" | "::1" | "[::1]" | "0.0.0.0"
    ) || host.starts_with("10.")
        || host.starts_with("192.168.")
        || host.starts_with("172.16.")
        || host.starts_with("172.17.")
        || host.starts_with("172.18.")
        || host.starts_with("172.19.")
        || host.starts_with("172.2")
        || host.starts_with("172.30.")
        || host.starts_with("172.31.")
}

/// Validate a redirect_uri against a client's registered URIs.
///
/// Exact match required, except for loopback URIs where port is ignored per RFC 8252.
pub fn validate_redirect_uri(requested: &str, registered: &[String]) -> bool {
    // Try exact match first
    if registered.contains(&requested.to_owned()) {
        return true;
    }

    // For loopback URIs, ignore port per RFC 8252 section 7.3
    if let (Ok(req_url), true) = (url::Url::parse(requested), is_loopback_uri(requested)) {
        for reg in registered {
            if let Ok(reg_url) = url::Url::parse(reg) {
                if is_loopback_uri(reg)
                    && req_url.scheme() == reg_url.scheme()
                    && req_url.host_str() == reg_url.host_str()
                    && req_url.path() == reg_url.path()
                {
                    return true;
                }
            }
        }
    }

    false
}
