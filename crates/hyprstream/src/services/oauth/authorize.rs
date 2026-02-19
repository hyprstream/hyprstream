//! OAuth 2.1 Authorization Endpoint.
//!
//! Handles the authorization code flow with PKCE (S256).
//! - GET /oauth/authorize — validates params, renders consent page
//! - POST /oauth/authorize — on approval, generates auth code and redirects

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Redirect, Response},
    Form,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::Deserialize;

use super::registration::{fetch_client_metadata, validate_redirect_uri};
use super::state::{OAuthState, PendingAuthCode};

/// Authorization request query parameters
#[derive(Debug, Deserialize)]
pub struct AuthorizeParams {
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub response_type: String,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    /// RFC 8707 resource indicator
    #[serde(default)]
    pub resource: Option<String>,
}

/// Consent form submission
#[derive(Debug, Deserialize)]
pub struct ConsentForm {
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub scope: String,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub resource: Option<String>,
    pub action: String,
}

/// GET /oauth/authorize — validate params and render consent page
pub async fn authorize_get(
    State(state): State<Arc<OAuthState>>,
    Query(params): Query<AuthorizeParams>,
) -> Response {
    // Validate response_type
    if params.response_type != "code" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "unsupported_response_type",
            "Only response_type=code is supported",
        );
    }

    // Validate code_challenge_method
    if params.code_challenge_method != "S256" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Only code_challenge_method=S256 is supported",
        );
    }

    if params.code_challenge.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "code_challenge is required (PKCE mandatory per OAuth 2.1)",
        );
    }

    // Resolve client
    let (client_name, redirect_uris) = if params.client_id.starts_with("https://") {
        // Client ID Metadata Document flow
        match fetch_client_metadata(&state, &params.client_id).await {
            Ok(client) => {
                let name = client.client_name.clone();
                let uris = client.redirect_uris.clone();
                // Cache the client
                state.clients.write().await.insert(client.client_id.clone(), client);
                (name.unwrap_or_else(|| params.client_id.clone()), uris)
            }
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    &format!("Failed to resolve client metadata: {}", e),
                );
            }
        }
    } else {
        // Dynamic registration lookup
        let clients = state.clients.read().await;
        match clients.get(&params.client_id) {
            Some(client) => (
                client.client_name.clone().unwrap_or_else(|| params.client_id.clone()),
                client.redirect_uris.clone(),
            ),
            None => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    "Unknown client_id. Register first via POST /oauth/register",
                );
            }
        }
    };

    // Validate redirect_uri
    if !validate_redirect_uri(&params.redirect_uri, &redirect_uris) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_redirect_uri",
            "redirect_uri does not match registered URIs",
        );
    }

    // Determine scopes
    let scopes = params.scope.as_deref().unwrap_or(
        &state.default_scopes.join(" ")
    ).to_owned();

    // Extract redirect hostname for display
    let redirect_host = url::Url::parse(&params.redirect_uri)
        .ok()
        .and_then(|u| u.host_str().map(std::borrow::ToOwned::to_owned))
        .unwrap_or_else(|| params.redirect_uri.clone());

    // Render consent page
    let html = render_consent_page(
        &client_name,
        &scopes,
        &redirect_host,
        &params.client_id,
        &params.redirect_uri,
        &params.code_challenge,
        params.state.as_deref().unwrap_or(""),
        params.resource.as_deref().unwrap_or(""),
    );

    Html(html).into_response()
}

/// POST /oauth/authorize — handle consent form submission
pub async fn authorize_post(
    State(state): State<Arc<OAuthState>>,
    Form(form): Form<ConsentForm>,
) -> Response {
    if form.action == "deny" {
        // Redirect with error
        let mut redirect_url = form.redirect_uri.clone();
        redirect_url.push_str("?error=access_denied&error_description=User+denied+access");
        if let Some(ref s) = form.state {
            redirect_url.push_str(&format!("&state={}", s));
        }
        return Redirect::to(&redirect_url).into_response();
    }

    // Generate 32-byte random auth code
    let mut code_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut code_bytes);
    let code = URL_SAFE_NO_PAD.encode(code_bytes);

    let scopes: Vec<String> = form.scope.split_whitespace().map(std::borrow::ToOwned::to_owned).collect();
    let resource = form.resource.as_ref().filter(|s| !s.is_empty()).cloned();

    let pending = PendingAuthCode {
        code: code.clone(),
        client_id: form.client_id.clone(),
        redirect_uri: form.redirect_uri.clone(),
        code_challenge: form.code_challenge.clone(),
        scopes,
        resource,
        created_at: Instant::now(),
        expires_at: Instant::now() + Duration::from_secs(60),
    };

    state.pending_codes.write().await.insert(code.clone(), pending);

    tracing::info!(
        client_id = %form.client_id,
        redirect_uri = %form.redirect_uri,
        "Authorization code issued, redirecting"
    );

    // Redirect with code
    let mut redirect_url = format!("{}?code={}", form.redirect_uri, code);
    if let Some(ref s) = form.state {
        redirect_url.push_str(&format!("&state={}", s));
    }

    Redirect::to(&redirect_url).into_response()
}

fn error_response(status: StatusCode, error: &str, description: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({
            "error": error,
            "error_description": description,
        })),
    ).into_response()
}

/// Render the minimal consent page HTML.
fn render_consent_page(
    client_name: &str,
    scopes: &str,
    redirect_host: &str,
    client_id: &str,
    redirect_uri: &str,
    code_challenge: &str,
    state: &str,
    resource: &str,
) -> String {
    // HTML-escape helper (basic)
    let esc = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorize — hyprstream</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 480px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
  .client {{ color: #0066cc; font-weight: 600; }}
  .scopes {{ background: #f5f5f5; border-radius: 8px; padding: 12px 16px; margin: 16px 0;
             font-family: monospace; font-size: 0.9rem; }}
  .redirect {{ font-size: 0.85rem; color: #666; margin: 12px 0; }}
  .actions {{ display: flex; gap: 12px; margin-top: 24px; }}
  button {{ flex: 1; padding: 12px; border: none; border-radius: 8px; font-size: 1rem;
           cursor: pointer; font-weight: 500; }}
  .approve {{ background: #0066cc; color: #fff; }}
  .approve:hover {{ background: #0052a3; }}
  .deny {{ background: #e8e8e8; color: #333; }}
  .deny:hover {{ background: #d0d0d0; }}
</style>
</head>
<body>
<div class="card">
  <h1>Authorize Application</h1>
  <p><span class="client">{client_name}</span> wants to access your hyprstream resources.</p>
  <div class="scopes">
    <strong>Requested scopes:</strong><br>
    {scopes_display}
  </div>
  <p class="redirect">Will redirect to: <code>{redirect_host}</code></p>
  <form method="post" action="/oauth/authorize">
    <input type="hidden" name="client_id" value="{client_id_val}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri_val}">
    <input type="hidden" name="code_challenge" value="{code_challenge_val}">
    <input type="hidden" name="scope" value="{scope_val}">
    <input type="hidden" name="state" value="{state_val}">
    <input type="hidden" name="resource" value="{resource_val}">
    <div class="actions">
      <button type="submit" name="action" value="deny" class="deny">Deny</button>
      <button type="submit" name="action" value="approve" class="approve">Approve</button>
    </div>
  </form>
</div>
</body>
</html>"#,
        client_name = esc(client_name),
        scopes_display = scopes.split_whitespace()
            .map(|s| format!("<code>{}</code>", esc(s)))
            .collect::<Vec<_>>()
            .join(", "),
        redirect_host = esc(redirect_host),
        client_id_val = esc(client_id),
        redirect_uri_val = esc(redirect_uri),
        code_challenge_val = esc(code_challenge),
        scope_val = esc(scopes),
        state_val = esc(state),
        resource_val = esc(resource),
    )
}
