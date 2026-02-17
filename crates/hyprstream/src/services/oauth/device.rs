//! RFC 8628 Device Authorization Grant.
//!
//! Endpoints:
//! - POST /oauth/device — Device Authorization Request
//! - GET  /oauth/device/verify — Verification page (user visits in browser)
//! - POST /oauth/device/verify — Verification form submission

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    Form, Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::Deserialize;

use super::state::{DeviceCodeStatus, OAuthState, PendingDeviceCode};

/// Device code TTL in seconds (10 minutes).
const DEVICE_CODE_TTL_SECS: u64 = 600;

/// Minimum polling interval in seconds.
const DEFAULT_POLL_INTERVAL: u64 = 5;

/// Device authorization request (application/x-www-form-urlencoded).
#[derive(Debug, Deserialize)]
pub struct DeviceAuthRequest {
    pub client_id: String,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub resource: Option<String>,
}

/// Query params for the verification page.
#[derive(Debug, Deserialize)]
pub struct VerifyQuery {
    #[serde(default)]
    pub user_code: Option<String>,
}

/// Verification form submission.
#[derive(Debug, Deserialize)]
pub struct VerifyForm {
    pub user_code: String,
    pub action: String,
}

/// POST /oauth/device — Device Authorization Endpoint (RFC 8628 Section 3.1)
pub async fn device_authorize(
    State(state): State<Arc<OAuthState>>,
    Form(params): Form<DeviceAuthRequest>,
) -> Response {
    // Validate client exists
    {
        let clients = state.clients.read().await;
        if !clients.contains_key(&params.client_id) {
            return device_error(
                StatusCode::BAD_REQUEST,
                "invalid_client",
                "Unknown client_id. Register first via POST /oauth/register",
            );
        }
    }

    // Generate device_code (32 bytes, URL-safe base64)
    let mut device_code_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut device_code_bytes);
    let device_code = URL_SAFE_NO_PAD.encode(device_code_bytes);

    // Generate user_code (8 uppercase alphanumeric, formatted XXXX-XXXX)
    let user_code = generate_user_code();

    let scopes: Vec<String> = params
        .scope
        .as_deref()
        .unwrap_or(&state.default_scopes.join(" "))
        .split_whitespace()
        .map(std::borrow::ToOwned::to_owned)
        .collect();

    let resource = params.resource.filter(|s| !s.is_empty());

    let now = Instant::now();
    let pending = PendingDeviceCode {
        device_code: device_code.clone(),
        user_code: user_code.clone(),
        client_id: params.client_id,
        scopes,
        resource,
        status: DeviceCodeStatus::Pending,
        created_at: now,
        expires_at: now + Duration::from_secs(DEVICE_CODE_TTL_SECS),
        interval: DEFAULT_POLL_INTERVAL,
        last_polled: None,
    };

    // Store pending device code
    {
        let mut device_codes = state.pending_device_codes.write().await;
        device_codes.insert(device_code.clone(), pending);
    }
    {
        let mut user_code_map = state.device_code_by_user_code.write().await;
        user_code_map.insert(user_code.clone(), device_code.clone());
    }

    let formatted_code = format_user_code(&user_code);
    let verification_uri = format!("{}/oauth/device/verify", state.issuer_url);
    let verification_uri_complete = format!(
        "{}/oauth/device/verify?user_code={}",
        state.issuer_url, formatted_code
    );

    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::json!({
            "device_code": device_code,
            "user_code": formatted_code,
            "verification_uri": verification_uri,
            "verification_uri_complete": verification_uri_complete,
            "expires_in": DEVICE_CODE_TTL_SECS,
            "interval": DEFAULT_POLL_INTERVAL,
        })),
    )
        .into_response()
}

/// GET /oauth/device/verify — Render verification page
pub async fn verify_get(
    State(state): State<Arc<OAuthState>>,
    Query(query): Query<VerifyQuery>,
) -> Response {
    let prefilled_code = query.user_code.as_deref().unwrap_or("");
    let html = render_verify_page(&state.issuer_url, prefilled_code, None);
    Html(html).into_response()
}

/// POST /oauth/device/verify — Handle verification form submission
pub async fn verify_post(
    State(state): State<Arc<OAuthState>>,
    Form(form): Form<VerifyForm>,
) -> Response {
    // Normalize: strip dash, uppercase
    let normalized = form.user_code.replace('-', "").to_uppercase();

    // Look up device_code from user_code
    let device_code = {
        let user_code_map = state.device_code_by_user_code.read().await;
        user_code_map.get(&normalized).cloned()
    };

    let device_code = match device_code {
        Some(dc) => dc,
        None => {
            let html = render_verify_page(
                &state.issuer_url,
                &form.user_code,
                Some("Invalid or expired code. Please try again."),
            );
            return Html(html).into_response();
        }
    };

    // Update status
    let result = {
        let mut device_codes = state.pending_device_codes.write().await;
        match device_codes.get_mut(&device_code) {
            Some(pending) if pending.is_expired() => {
                device_codes.remove(&device_code);
                let mut user_code_map = state.device_code_by_user_code.write().await;
                user_code_map.remove(&normalized);
                Err("This code has expired. Please request a new one.")
            }
            Some(pending) if pending.status != DeviceCodeStatus::Pending => {
                Err("This code has already been used.")
            }
            Some(pending) => {
                if form.action == "approve" {
                    pending.status = DeviceCodeStatus::Approved;
                } else {
                    pending.status = DeviceCodeStatus::Denied;
                }
                Ok(form.action == "approve")
            }
            None => Err("Invalid or expired code. Please try again."),
        }
    };

    match result {
        Ok(approved) => {
            let html = render_result_page(approved);
            Html(html).into_response()
        }
        Err(msg) => {
            let html = render_verify_page(&state.issuer_url, &form.user_code, Some(msg));
            Html(html).into_response()
        }
    }
}

/// Generate an 8-character uppercase alphanumeric user code.
/// Uses characters that are unambiguous (no 0/O, 1/I/L).
fn generate_user_code() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHJKMNPQRSTVWXYZ23456789";
    let mut rng = rand::thread_rng();
    let mut code = String::with_capacity(8);
    for _ in 0..8 {
        let mut byte = [0u8; 1];
        loop {
            rng.fill_bytes(&mut byte);
            let idx = byte[0] as usize;
            if idx < (256 / CHARSET.len()) * CHARSET.len() {
                code.push(CHARSET[idx % CHARSET.len()] as char);
                break;
            }
        }
    }
    code
}

/// Format user code as XXXX-XXXX.
fn format_user_code(code: &str) -> String {
    if code.len() == 8 {
        format!("{}-{}", &code[..4], &code[4..])
    } else {
        code.to_owned()
    }
}

fn device_error(status: StatusCode, error: &str, description: &str) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::json!({
            "error": error,
            "error_description": description,
        })),
    )
        .into_response()
}

/// Render the verification page HTML.
fn render_verify_page(issuer_url: &str, prefilled_code: &str, error: Option<&str>) -> String {
    let esc = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    };

    let error_html = match error {
        Some(msg) => format!(
            r#"<div class="error">{}</div>"#,
            esc(msg)
        ),
        None => String::new(),
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Device Verification — hyprstream</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 480px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
  p {{ color: #555; line-height: 1.5; }}
  .code-input {{ width: 100%; padding: 14px; font-size: 1.4rem; font-family: monospace;
                 text-align: center; letter-spacing: 4px; text-transform: uppercase;
                 border: 2px solid #ddd; border-radius: 8px; box-sizing: border-box; margin: 16px 0; }}
  .code-input:focus {{ border-color: #0066cc; outline: none; }}
  .error {{ background: #fff0f0; color: #cc0000; border-radius: 8px; padding: 12px 16px;
            margin: 12px 0; font-size: 0.9rem; }}
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
  <h1>Device Verification</h1>
  <p>Enter the code displayed on your device to authorize access to hyprstream.</p>
  {error_html}
  <form method="post" action="{action_url}">
    <input type="text" name="user_code" class="code-input" value="{prefilled}"
           placeholder="XXXX-XXXX" maxlength="9" autocomplete="off" autofocus>
    <div class="actions">
      <button type="submit" name="action" value="deny" class="deny">Deny</button>
      <button type="submit" name="action" value="approve" class="approve">Approve</button>
    </div>
  </form>
</div>
</body>
</html>"#,
        error_html = error_html,
        action_url = esc(&format!("{}/oauth/device/verify", issuer_url)),
        prefilled = esc(prefilled_code),
    )
}

/// Render the result page after approval/denial.
fn render_result_page(approved: bool) -> String {
    let (title, message, color) = if approved {
        (
            "Device Authorized",
            "Your device has been authorized. You can close this window and return to your device.",
            "#0a8a0a",
        )
    } else {
        (
            "Access Denied",
            "You denied access for this device. You can close this window.",
            "#cc0000",
        )
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — hyprstream</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 480px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
           text-align: center; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 16px; color: {color}; }}
  p {{ color: #555; line-height: 1.5; }}
</style>
</head>
<body>
<div class="card">
  <h1>{title}</h1>
  <p>{message}</p>
</div>
</body>
</html>"#,
        title = title,
        message = message,
        color = color,
    )
}
