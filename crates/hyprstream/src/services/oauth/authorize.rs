//! OAuth 2.1 Authorization Endpoint.
//!
//! Handles the authorization code flow with PKCE (S256) and Ed25519 user authentication.
//! - GET /oauth/authorize — validates params, renders Ed25519 challenge form
//! - POST /oauth/authorize — verifies Ed25519 signature, generates auth code and redirects
//!
//! The consent page was replaced with an Ed25519 challenge-response form. The nonce is
//! stored server-side (`OAuthState::pending_nonces`, 5-min TTL) and embedded as a hidden
//! form field. On POST, the server verifies the nonce was issued by itself and consumes it
//! (single-use). Challenge format:
//!   `"{username}:{nonce}:{code_challenge}"`
//! The `code_challenge` binding prevents the signature from being reused against a different
//! PKCE session; the server-side nonce store enforces TTL and prevents replay.

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

use super::challenge;
use super::device::html_escape;
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

/// Consent form submission (Ed25519 challenge-response)
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
    /// Ed25519 nonce (from hidden form field; validated server-side against pending_nonces)
    pub nonce: String,
    /// Username entered by the user
    pub username: String,
    /// Base64-encoded Ed25519 signature
    pub signature: String,
}

/// GET /oauth/authorize — validate params and render Ed25519 challenge form
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

    // Require UserStore for identity verification
    if state.user_store.is_none() {
        return Html(render_error_page(
            "Server Not Configured",
            "This server is not configured for local user authentication. \
             Contact your administrator to set up the user credential store.",
        )).into_response();
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

    // Generate a nonce, store it server-side (5-min TTL, single-use on POST).
    let nonce = issue_nonce(&state).await;

    // Render challenge form
    let html = render_challenge_page(
        &client_name,
        &scopes,
        &redirect_host,
        &params.client_id,
        &params.redirect_uri,
        &params.code_challenge,
        params.state.as_deref().unwrap_or(""),
        params.resource.as_deref().unwrap_or(""),
        &nonce,
        None,
    );

    Html(html).into_response()
}

/// POST /oauth/authorize — verify Ed25519 signature, issue auth code
pub async fn authorize_post(
    State(state): State<Arc<OAuthState>>,
    Form(form): Form<ConsentForm>,
) -> Response {
    // Validate and consume the nonce. Rejects forged nonces and enforces the 5-min TTL.
    // Single-use: remove from pending_nonces whether valid or expired.
    let nonce_valid = {
        let mut nonces = state.pending_nonces.write().await;
        match nonces.remove(&form.nonce) {
            Some(expiry) => Instant::now() < expiry,
            None => false,
        }
    };
    if !nonce_valid {
        // Nonce was not issued by this server, already consumed, or expired.
        // Validate redirect_uri and re-derive client info from the registry.
        // If either fails, return an error page rather than re-rendering with
        // attacker-controlled values.
        let Some((client_name, redirect_host)) =
            derive_display_info(&state, &form.client_id, &form.redirect_uri).await
        else {
            return Html(render_error_page(
                "Invalid Request",
                "Unknown client or redirect URI. Please restart the authorization flow.",
            )).into_response();
        };
        let fresh_nonce = issue_nonce(&state).await;
        let html = render_challenge_page(
            &client_name,
            &form.scope,
            &redirect_host,
            &form.client_id,
            &form.redirect_uri,
            &form.code_challenge,
            form.state.as_deref().unwrap_or(""),
            form.resource.as_deref().unwrap_or(""),
            &fresh_nonce,
            Some("Authorization request expired. Please try again."),
        );
        return Html(html).into_response();
    }

    // Reconstruct challenge: "{username}:{nonce}:{code_challenge}"
    // Binds signature to both the user identity and the PKCE session.
    let challenge_str = format!("{}:{}:{}", form.username, form.nonce, form.code_challenge);

    // Get user store
    let user_store = match state.user_store.as_ref() {
        Some(s) => s,
        None => {
            return Html(render_error_page(
                "Server Not Configured",
                "User credential store not configured on this server.",
            )).into_response();
        }
    };

    // Verify Ed25519 challenge-response
    if let Err(e) = challenge::verify_ed25519_response(
        user_store.as_ref(),
        &form.username,
        &challenge_str,
        &form.signature,
    ) {
        if matches!(e, challenge::ChallengeError::UserStoreError(_)) {
            tracing::error!(username = %form.username, "UserStore lookup error during authorize");
        }
        // Validate redirect_uri and re-derive client display info from the registry.
        // Never trust the POST body for display values; return an error page if
        // the client or redirect_uri is no longer valid.
        let Some((client_name, redirect_host)) =
            derive_display_info(&state, &form.client_id, &form.redirect_uri).await
        else {
            return Html(render_error_page(
                "Invalid Request",
                "Unknown client or redirect URI. Please restart the authorization flow.",
            )).into_response();
        };
        // Issue a fresh nonce so re-render shows a signable challenge.
        let fresh_nonce = issue_nonce(&state).await;
        let html = render_challenge_page(
            &client_name,
            &form.scope,
            &redirect_host,
            &form.client_id,
            &form.redirect_uri,
            &form.code_challenge,
            form.state.as_deref().unwrap_or(""),
            form.resource.as_deref().unwrap_or(""),
            &fresh_nonce,
            Some(e.message()),
        );
        return Html(html).into_response();
    }

    // Signature valid — generate auth code
    let mut code_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut code_bytes);
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
        username: form.username.clone(),
    };

    state.pending_codes.write().await.insert(code.clone(), pending);

    tracing::info!(
        client_id = %form.client_id,
        username = %form.username,
        redirect_uri = %form.redirect_uri,
        "Authorization code issued, redirecting"
    );

    // Build redirect URL using url::Url to ensure proper percent-encoding of `state`.
    let redirect_url = match url::Url::parse(&form.redirect_uri) {
        Ok(mut u) => {
            {
                let mut q = u.query_pairs_mut();
                q.append_pair("code", &code);
                if let Some(ref s) = form.state {
                    q.append_pair("state", s);
                }
            }
            u.to_string()
        }
        Err(_) => {
            // redirect_uri was validated on GET; this should not happen
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Invalid redirect_uri",
            );
        }
    };

    Redirect::to(&redirect_url).into_response()
}

/// Generate a fresh nonce and record it in `pending_nonces` with a 5-min expiry.
async fn issue_nonce(state: &OAuthState) -> String {
    let mut nonce_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);
    let expiry = Instant::now() + Duration::from_secs(300);
    state.pending_nonces.write().await.insert(nonce.clone(), expiry);
    nonce
}

/// Re-derive client display name and redirect host from the registry, validating
/// that `redirect_uri` is registered for the client.
///
/// Returns `None` if the client is not in the registry or `redirect_uri` is not
/// among its registered URIs — callers must return an error page in that case.
/// This ensures that POST body values are never trusted for UI display without
/// first being validated against the canonical server-side registry.
async fn derive_display_info(
    state: &OAuthState,
    client_id: &str,
    redirect_uri: &str,
) -> Option<(String, String)> {
    let clients = state.clients.read().await;
    let client = clients.get(client_id)?;

    // Validate redirect_uri against the client's registered URIs before displaying it.
    if !validate_redirect_uri(redirect_uri, &client.redirect_uris) {
        return None;
    }

    let client_name = client
        .client_name
        .clone()
        .unwrap_or_else(|| client_id.to_owned());
    let redirect_host = url::Url::parse(redirect_uri)
        .ok()
        .and_then(|u| u.host_str().map(std::borrow::ToOwned::to_owned))
        .unwrap_or_else(|| redirect_uri.to_owned());
    Some((client_name, redirect_host))
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

/// Render an error page (HTML) for configuration errors.
fn render_error_page(title: &str, message: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_esc} — hyprstream</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 480px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.4rem; margin: 0 0 8px; color: #cc0000; }}
  p {{ color: #555; line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
  <h1>{title_esc}</h1>
  <p>{message_esc}</p>
</div>
</body>
</html>"#,
        title_esc = html_escape(title),
        message_esc = html_escape(message),
    )
}

/// Render the Ed25519 challenge form (replaces the old Approve/Deny consent page).
#[allow(clippy::too_many_arguments)]
fn render_challenge_page(
    client_name: &str,
    scopes: &str,
    redirect_host: &str,
    client_id: &str,
    redirect_uri: &str,
    code_challenge: &str,
    state: &str,
    resource: &str,
    nonce: &str,
    error: Option<&str>,
) -> String {
    let error_html = match error {
        Some(msg) => format!(r#"<div class="error">{}</div>"#, html_escape(msg)),
        None => String::new(),
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
         max-width: 520px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
  .client {{ color: #0066cc; font-weight: 600; }}
  .scopes {{ background: #f5f5f5; border-radius: 8px; padding: 12px 16px; margin: 16px 0;
             font-family: monospace; font-size: 0.9rem; }}
  .redirect {{ font-size: 0.85rem; color: #666; margin: 12px 0; }}
  .error {{ background: #fff0f0; color: #cc0000; border-radius: 8px; padding: 12px 16px;
            margin: 12px 0; font-size: 0.9rem; }}
  .step {{ background: #f0f6ff; border-left: 3px solid #0066cc; padding: 10px 14px;
           margin: 12px 0; border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #333; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; word-break: break-all; }}
  label {{ display: block; margin-top: 16px; color: #333; font-size: 0.9rem; font-weight: 500; }}
  input[type=text] {{ width: 100%; padding: 10px; font-size: 0.9rem; border: 2px solid #ddd;
                       border-radius: 8px; box-sizing: border-box; margin-top: 4px; }}
  input[type=text]:focus {{ border-color: #0066cc; outline: none; }}
  button {{ width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 1rem;
           cursor: pointer; font-weight: 500; background: #0066cc; color: #fff; margin-top: 24px; }}
  button:hover {{ background: #0052a3; }}
</style>
</head>
<body>
<div class="card">
  <h1>Authorize Application</h1>
  <p><span class="client">{client_name_esc}</span> wants to access your hyprstream resources.</p>
  <div class="scopes">
    <strong>Requested scopes:</strong><br>
    {scopes_display}
  </div>
  <p class="redirect">Will redirect to: <code>{redirect_host_esc}</code></p>
  {error_html}
  <div class="step">
    <strong>Sign the challenge on your workstation:</strong><br/>
    <code>hyprstream sign-challenge --nonce {nonce_esc} --code-challenge {code_challenge_esc}</code><br/>
    Copy the printed <em>Username</em> and <em>Signature</em> below.
  </div>
  <form method="post" action="/oauth/authorize">
    <input type="hidden" name="client_id" value="{client_id_val}">
    <input type="hidden" name="redirect_uri" value="{redirect_uri_val}">
    <input type="hidden" name="code_challenge" value="{code_challenge_val}">
    <input type="hidden" name="scope" value="{scope_val}">
    <input type="hidden" name="state" value="{state_val}">
    <input type="hidden" name="resource" value="{resource_val}">
    <input type="hidden" name="nonce" value="{nonce_val}">
    <label>Username:
      <input type="text" name="username" required autocomplete="username" autofocus/>
    </label>
    <label>Signature (base64, 88 chars):
      <input type="text" name="signature" required size="88" autocomplete="off"
             placeholder="base64-encoded Ed25519 signature"/>
    </label>
    <button type="submit">Authorize</button>
  </form>
</div>
</body>
</html>"#,
        client_name_esc = html_escape(client_name),
        scopes_display = scopes.split_whitespace()
            .map(|s| format!("<code>{}</code>", html_escape(s)))
            .collect::<Vec<_>>()
            .join(", "),
        redirect_host_esc = html_escape(redirect_host),
        error_html = error_html,
        nonce_esc = html_escape(nonce),
        code_challenge_esc = html_escape(code_challenge),
        client_id_val = html_escape(client_id),
        redirect_uri_val = html_escape(redirect_uri),
        code_challenge_val = html_escape(code_challenge),
        scope_val = html_escape(scopes),
        state_val = html_escape(state),
        resource_val = html_escape(resource),
        nonce_val = html_escape(nonce),
    )
}
