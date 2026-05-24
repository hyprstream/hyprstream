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
use std::time::Duration;

/// Maximum number of dynamically-registered (RFC 7591) clients held in
/// `state.clients`. The DCR endpoint is unauthenticated by design (the
/// whole point is bootstrapping new clients), so a hard cap bounds the
/// memory amplifier even when an attacker pummels POST /oauth/register.
/// CIMD clients are NOT counted here — they live in cimd_cache, which
/// has its own capacity bound.
///
/// 1000 is comfortable for an operator with a handful of MCP clients
/// per user and the occasional dev/test churn. Operators expecting more
/// can override via configuration in a future change.
pub const DCR_MAX_CLIENTS: usize = 1000;

/// Maximum DCR field lengths to bound request-payload memory.
pub const DCR_MAX_REDIRECT_URIS: usize = 16;
pub const DCR_MAX_URI_LEN: usize = 2048;
pub const DCR_MAX_NAME_LEN: usize = 256;

/// Dynamic client registration request (RFC 7591 §3.1).
///
/// Only the fields hyprstream actually honors are deserialized; unknown
/// fields are dropped per RFC 7591 §2 ("server MAY ignore them").
#[derive(Debug, Deserialize)]
pub struct RegistrationRequest {
    pub redirect_uris: Vec<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub client_uri: Option<String>,
    #[serde(default)]
    pub logo_uri: Option<String>,
    #[serde(default)]
    pub grant_types: Vec<String>,
    #[serde(default)]
    pub response_types: Vec<String>,
    #[serde(default)]
    pub token_endpoint_auth_method: Option<String>,
    #[serde(default)]
    pub jwks: Option<serde_json::Value>,
    #[serde(default)]
    pub jwks_uri: Option<String>,
}

/// Dynamic client registration response (RFC 7591 §3.2.1).
#[derive(Debug, Serialize)]
pub struct RegistrationResponse {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logo_uri: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub grant_types: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub response_types: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_endpoint_auth_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwks_uri: Option<String>,
}

/// POST /oauth/register — Dynamic Client Registration (RFC 7591)
pub async fn register_client(
    State(state): State<Arc<OAuthState>>,
    Json(req): Json<RegistrationRequest>,
) -> impl IntoResponse {
    if req.redirect_uris.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "At least one redirect_uri is required"
            })),
        ).into_response();
    }

    if req.redirect_uris.len() > DCR_MAX_REDIRECT_URIS {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": format!(
                    "Too many redirect_uris (max {DCR_MAX_REDIRECT_URIS})"
                ),
            })),
        ).into_response();
    }

    // Validate redirect URIs: loopback only, bounded length.
    for uri in &req.redirect_uris {
        if uri.len() > DCR_MAX_URI_LEN {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid_redirect_uri",
                    "error_description": format!("redirect_uri exceeds {DCR_MAX_URI_LEN} chars"),
                })),
            ).into_response();
        }
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

    // Bound the display-name field (rendered on consent screen).
    if let Some(name) = req.client_name.as_deref() {
        if name.len() > DCR_MAX_NAME_LEN {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid_client_metadata",
                    "error_description": format!("client_name exceeds {DCR_MAX_NAME_LEN} chars"),
                })),
            ).into_response();
        }
    }

    // RFC 7591 §2.1: jwks and jwks_uri are mutually exclusive.
    if req.jwks.is_some() && req.jwks_uri.is_some() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "jwks and jwks_uri are mutually exclusive (RFC 7591 §2.1)"
            })),
        ).into_response();
    }

    // If token_endpoint_auth_method requires a key, enforce key presence.
    if matches!(req.token_endpoint_auth_method.as_deref(), Some("private_key_jwt"))
        && req.jwks.is_none() && req.jwks_uri.is_none()
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "invalid_client_metadata",
                "error_description": "private_key_jwt requires jwks or jwks_uri"
            })),
        ).into_response();
    }

    let client_id = uuid::Uuid::new_v4().to_string();

    let client = RegisteredClient {
        client_id: client_id.clone(),
        redirect_uris: req.redirect_uris.clone(),
        client_name: req.client_name.clone(),
        client_uri: req.client_uri.clone(),
        logo_uri: req.logo_uri.clone(),
        grant_types: req.grant_types.clone(),
        response_types: req.response_types.clone(),
        token_endpoint_auth_method: req.token_endpoint_auth_method.clone(),
        jwks: req.jwks.clone(),
        jwks_uri: req.jwks_uri.clone(),
        is_cimd: false,
        registered_at: Instant::now(),
    };

    // Capacity gate: refuse new registrations when at DCR_MAX_CLIENTS.
    // Hold the write lock across the size check + insert so two
    // concurrent registrations can't both squeeze in at len == cap-1.
    {
        let mut clients = state.clients.write().await;
        if clients.len() >= DCR_MAX_CLIENTS {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "registration_capacity_exhausted",
                    "error_description": format!(
                        "DCR client registry at capacity ({DCR_MAX_CLIENTS}). \
                         Operator must rotate stale clients or migrate to CIMD."
                    ),
                })),
            ).into_response();
        }
        clients.insert(client_id.clone(), client);
    }

    Json(RegistrationResponse {
        client_id,
        redirect_uris: req.redirect_uris,
        client_name: req.client_name,
        client_uri: req.client_uri,
        logo_uri: req.logo_uri,
        grant_types: req.grant_types,
        response_types: req.response_types,
        token_endpoint_auth_method: req.token_endpoint_auth_method,
        jwks: req.jwks,
        jwks_uri: req.jwks_uri,
    }).into_response()
}

/// Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00 §4).
#[derive(Debug, Deserialize)]
pub struct ClientIdMetadataDocument {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub client_uri: Option<String>,
    #[serde(default)]
    pub logo_uri: Option<String>,
    #[serde(default)]
    pub grant_types: Vec<String>,
    #[serde(default)]
    pub response_types: Vec<String>,
    #[serde(default)]
    pub token_endpoint_auth_method: Option<String>,
    #[serde(default)]
    pub jwks: Option<serde_json::Value>,
    #[serde(default)]
    pub jwks_uri: Option<String>,
}

/// Resolve a CIMD client: cache-aside lookup, fetch on miss.
///
/// Hot-path entry point used by `/oauth/par` and `/oauth/authorize`.
/// Returns the registered client (cached or freshly fetched + cached).
pub async fn resolve_cimd_client(
    state: &OAuthState,
    client_id_url: &str,
) -> Result<RegisteredClient, String> {
    if let Some(cached) = state.cimd_cache.get(client_id_url).await {
        return Ok(cached);
    }
    let (client, max_age) = fetch_client_metadata(state, client_id_url).await?;
    let ttl = state.cimd_cache.clamp_ttl(max_age);
    state.cimd_cache.insert(client.clone(), ttl).await;
    Ok(client)
}

/// Parse `Cache-Control: max-age=N` from response headers. `None` if
/// absent, malformed, or directive set to `no-store`/`no-cache`/`private`
/// (the cache will fall back to `min_ttl` in that case via `clamp_ttl`).
pub(crate) fn extract_max_age(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::CACHE_CONTROL)?.to_str().ok()?;
    let mut max_age: Option<Duration> = None;
    for part in raw.split(',') {
        let part = part.trim();
        let lower = part.to_ascii_lowercase();
        if lower == "no-store" || lower == "no-cache" || lower == "private" {
            return None;
        }
        if let Some(value) = lower.strip_prefix("max-age=") {
            if let Ok(secs) = value.trim().parse::<u64>() {
                max_age = Some(Duration::from_secs(secs));
            }
        }
    }
    max_age
}

/// Fetch and validate a Client ID Metadata Document from an HTTPS URL.
///
/// Returns `(client, max_age_from_cache_control)`; the latter is passed
/// to `CimdCache::clamp_ttl` by the caller. Callers should usually go
/// through [`resolve_cimd_client`] instead, which handles caching.
pub async fn fetch_client_metadata(
    state: &OAuthState,
    client_id_url: &str,
) -> Result<(RegisteredClient, Option<Duration>), String> {
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

    let max_age = extract_max_age(response.headers());

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

    // RFC 7591 §2.1 (mirrored by CIMD §4): jwks and jwks_uri are mutually exclusive.
    if doc.jwks.is_some() && doc.jwks_uri.is_some() {
        return Err("CIMD must not declare both jwks and jwks_uri".to_owned());
    }

    if matches!(doc.token_endpoint_auth_method.as_deref(), Some("private_key_jwt"))
        && doc.jwks.is_none() && doc.jwks_uri.is_none()
    {
        return Err("CIMD declares token_endpoint_auth_method=private_key_jwt but provides no jwks/jwks_uri".to_owned());
    }

    Ok((
        RegisteredClient {
            client_id: client_id_url.to_owned(),
            redirect_uris: doc.redirect_uris,
            client_name: doc.client_name,
            client_uri: doc.client_uri,
            logo_uri: doc.logo_uri,
            grant_types: doc.grant_types,
            response_types: doc.response_types,
            token_endpoint_auth_method: doc.token_endpoint_auth_method,
            jwks: doc.jwks,
            jwks_uri: doc.jwks_uri,
            is_cimd: true,
            registered_at: Instant::now(),
        },
        max_age,
    ))
}

/// Check if a redirect URI is a loopback address.
pub(crate) fn is_loopback_uri(uri: &str) -> bool {
    if let Ok(url) = url::Url::parse(uri) {
        // Accept both http and https for loopback URIs (https is used with mkcert in dev)
        if url.scheme() != "http" && url.scheme() != "https" {
            return false;
        }
        matches!(url.host_str(), Some("127.0.0.1") | Some("localhost") | Some("[::1]"))
    } else {
        false
    }
}

/// Public re-export of `is_private_host` for sibling modules
/// (client_auth) that need the same SSRF policy for fetches.
pub(crate) fn is_private_host_for_jwks(host: &str) -> bool {
    is_private_host(host)
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cimd_parses_extended_fields() {
        let json = serde_json::json!({
            "client_id": "https://app.example.com/client.json",
            "redirect_uris": ["http://127.0.0.1:3000/cb"],
            "client_name": "Example",
            "client_uri": "https://app.example.com",
            "logo_uri": "https://app.example.com/logo.png",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        });
        let doc: ClientIdMetadataDocument = serde_json::from_value(json).unwrap();
        assert_eq!(doc.client_uri.as_deref(), Some("https://app.example.com"));
        assert_eq!(doc.logo_uri.as_deref(), Some("https://app.example.com/logo.png"));
        assert_eq!(doc.grant_types, vec!["authorization_code", "refresh_token"]);
        assert_eq!(doc.response_types, vec!["code"]);
        assert_eq!(doc.token_endpoint_auth_method.as_deref(), Some("none"));
    }

    #[test]
    fn cimd_drops_unknown_fields() {
        let json = serde_json::json!({
            "client_id": "https://app.example.com/client.json",
            "redirect_uris": ["http://127.0.0.1:3000/cb"],
            "future_extension": "ignored",
            "policy_uri": "https://app.example.com/policy",
        });
        let doc: ClientIdMetadataDocument = serde_json::from_value(json).unwrap();
        assert!(doc.client_name.is_none());
    }

    #[test]
    fn registration_request_jwks_and_jwks_uri_default_empty() {
        let json = serde_json::json!({
            "redirect_uris": ["http://localhost:3000/cb"]
        });
        let req: RegistrationRequest = serde_json::from_value(json).unwrap();
        assert!(req.jwks.is_none());
        assert!(req.jwks_uri.is_none());
        assert!(req.grant_types.is_empty());
    }

    #[test]
    fn registration_response_omits_empty_fields() {
        let resp = RegistrationResponse {
            client_id: "abc".into(),
            redirect_uris: vec!["http://localhost/cb".into()],
            client_name: None,
            client_uri: None,
            logo_uri: None,
            grant_types: vec![],
            response_types: vec![],
            token_endpoint_auth_method: None,
            jwks: None,
            jwks_uri: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("client_name"));
        assert!(!obj.contains_key("jwks"));
        assert!(!obj.contains_key("grant_types"));
        assert!(obj.contains_key("client_id"));
        assert!(obj.contains_key("redirect_uris"));
    }

    #[test]
    fn loopback_uri_rejects_external() {
        assert!(is_loopback_uri("http://127.0.0.1:8080/cb"));
        assert!(is_loopback_uri("http://localhost:8080/cb"));
        assert!(is_loopback_uri("http://[::1]:8080/cb"));
        assert!(!is_loopback_uri("http://example.com/cb"));
        assert!(!is_loopback_uri("ftp://localhost/cb"));
    }

    #[test]
    fn extract_max_age_parses_cache_control() {
        use reqwest::header::{HeaderMap, HeaderValue, CACHE_CONTROL};

        let mut h = HeaderMap::new();
        h.insert(CACHE_CONTROL, HeaderValue::from_static("max-age=3600"));
        assert_eq!(extract_max_age(&h), Some(Duration::from_secs(3600)));

        let mut h = HeaderMap::new();
        h.insert(CACHE_CONTROL, HeaderValue::from_static("public, max-age=600, must-revalidate"));
        assert_eq!(extract_max_age(&h), Some(Duration::from_secs(600)));

        // no-store / no-cache / private force a None result (we fall back to min_ttl).
        let mut h = HeaderMap::new();
        h.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        assert!(extract_max_age(&h).is_none());

        let mut h = HeaderMap::new();
        h.insert(CACHE_CONTROL, HeaderValue::from_static("max-age=300, no-cache"));
        assert!(extract_max_age(&h).is_none());

        // Missing header → None.
        assert!(extract_max_age(&HeaderMap::new()).is_none());
    }

    #[test]
    fn private_host_blocks_rfc1918() {
        assert!(is_private_host("localhost"));
        assert!(is_private_host("127.0.0.1"));
        assert!(is_private_host("10.0.0.1"));
        assert!(is_private_host("192.168.1.1"));
        assert!(!is_private_host("example.com"));
        assert!(!is_private_host("8.8.8.8"));
    }
}
