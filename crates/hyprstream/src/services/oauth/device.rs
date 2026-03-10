//! RFC 8628 Device Authorization Grant.
//!
//! Endpoints:
//! - POST /oauth/device — Device Authorization Request
//! - GET  /oauth/device/verify — Verification page (user visits in browser)
//! - POST /oauth/device/verify — Verification form submission (Ed25519 challenge-response)
//! - GET  /oauth/device/nonce — JSON nonce endpoint for CLI sign-challenge helper

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    Form, Json,
};
use base64::{engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD}, Engine};
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

/// Verification form submission (Ed25519 challenge-response).
#[derive(Debug, Deserialize)]
pub struct VerifyForm {
    pub user_code: String,
    pub username: String,
    pub signature: String,
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

    // Generate nonce (32 bytes, URL-safe base64, no padding → 43 chars)
    let mut nonce_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);

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
        nonce,
        approved_by: None,
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

/// GET /oauth/device/verify — Render verification page with Ed25519 challenge form
pub async fn verify_get(
    State(state): State<Arc<OAuthState>>,
    Query(query): Query<VerifyQuery>,
) -> Response {
    let prefilled_code = query.user_code.as_deref().unwrap_or("").trim();

    if prefilled_code.is_empty() {
        // No user_code provided — show generic entry prompt
        let html = render_entry_page(&state.issuer_url, None);
        return Html(html).into_response();
    }

    // Normalize: strip dash, uppercase
    let normalized = prefilled_code.replace('-', "").to_uppercase();

    // Look up device_code from user_code to get the nonce
    let device_code = {
        let user_code_map = state.device_code_by_user_code.read().await;
        user_code_map.get(&normalized).cloned()
    };

    let device_code = match device_code {
        Some(dc) => dc,
        None => {
            let html = render_entry_page(&state.issuer_url, Some("Invalid or expired code. Please check your device and try again."));
            return Html(html).into_response();
        }
    };

    let nonce = {
        let device_codes = state.pending_device_codes.read().await;
        match device_codes.get(&device_code) {
            Some(pending) if pending.is_expired() => {
                let html = render_entry_page(&state.issuer_url, Some("This code has expired. Please request a new one from your device."));
                return Html(html).into_response();
            }
            Some(pending) => pending.nonce.clone(),
            None => {
                let html = render_entry_page(&state.issuer_url, Some("Invalid or expired code. Please try again."));
                return Html(html).into_response();
            }
        }
    };

    let html = render_verify_page(prefilled_code, &nonce, None);
    Html(html).into_response()
}

/// POST /oauth/device/verify — Handle Ed25519 challenge-response verification
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
                &form.user_code,
                "",
                Some("Invalid or expired code. Please try again."),
            );
            return Html(html).into_response();
        }
    };

    // Retrieve nonce and validate state (without holding write lock during sig verification)
    let nonce = {
        let device_codes = state.pending_device_codes.read().await;
        match device_codes.get(&device_code) {
            Some(pending) if pending.is_expired() => {
                let html = render_verify_page(
                    &form.user_code,
                    "",
                    Some("This code has expired. Please request a new one."),
                );
                return Html(html).into_response();
            }
            Some(pending) if pending.status != DeviceCodeStatus::Pending => {
                let html = render_verify_page(
                    &form.user_code,
                    "",
                    Some("This code has already been used."),
                );
                return Html(html).into_response();
            }
            Some(pending) => pending.nonce.clone(),
            None => {
                let html = render_verify_page(
                    &form.user_code,
                    "",
                    Some("Invalid or expired code. Please try again."),
                );
                return Html(html).into_response();
            }
        }
    };

    // Reconstruct the challenge string: "{username}:{user_code}:{nonce}"
    // user_code here is the normalized (no-dash) form stored internally
    let challenge = format!("{}:{}:{}", form.username, normalized, nonce);

    // Decode the base64 signature
    let sig_bytes = match STANDARD.decode(&form.signature) {
        Ok(b) => b,
        Err(_) => {
            let html = render_verify_page(
                &form.user_code,
                &nonce,
                Some("Invalid signature encoding (expected base64)."),
            );
            return Html(html).into_response();
        }
    };

    // Convert to ed25519 Signature (must be exactly 64 bytes)
    let sig_array: [u8; 64] = match sig_bytes.try_into() {
        Ok(arr) => arr,
        Err(_) => {
            let html = render_verify_page(
                &form.user_code,
                &nonce,
                Some("Invalid signature length (expected 64 bytes / 88 base64 chars)."),
            );
            return Html(html).into_response();
        }
    };
    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

    // Look up user's public key from UserStore
    let user_store = match state.user_store.as_ref() {
        Some(s) => s,
        None => {
            let html = render_verify_page(
                &form.user_code,
                &nonce,
                Some("User credential store not configured on this server."),
            );
            return Html(html).into_response();
        }
    };
    let pubkey = match user_store.get_pubkey(&form.username) {
        Ok(Some(pk)) => pk,
        Ok(None) => {
            let html = render_verify_page(
                &form.user_code,
                &nonce,
                Some("Unknown user. Please contact your administrator."),
            );
            return Html(html).into_response();
        }
        Err(e) => {
            tracing::error!(username = %form.username, error = %e, "UserStore lookup error");
            let html = render_verify_page(
                &form.user_code,
                &nonce,
                Some("Internal error looking up user credentials."),
            );
            return Html(html).into_response();
        }
    };

    // Verify the Ed25519 signature
    if pubkey.verify_strict(challenge.as_bytes(), &signature).is_err() {
        let html = render_verify_page(
            &form.user_code,
            &nonce,
            Some("Invalid signature. Ensure you signed the correct challenge string."),
        );
        return Html(html).into_response();
    }

    // Signature valid — approve the device code and record who approved it
    let result = {
        let mut device_codes = state.pending_device_codes.write().await;
        match device_codes.get_mut(&device_code) {
            Some(pending) if pending.is_expired() => {
                let user_code_inner = pending.user_code.clone();
                device_codes.remove(&device_code);
                let mut user_code_map = state.device_code_by_user_code.write().await;
                user_code_map.remove(&user_code_inner);
                Err("This code has expired. Please request a new one.")
            }
            Some(pending) if pending.status != DeviceCodeStatus::Pending => {
                Err("This code has already been used.")
            }
            Some(pending) => {
                pending.status = DeviceCodeStatus::Approved;
                pending.approved_by = Some(form.username.clone());
                Ok(())
            }
            None => Err("Invalid or expired code. Please try again."),
        }
    };

    match result {
        Ok(()) => {
            let html = render_result_page(true, &form.username);
            Html(html).into_response()
        }
        Err(msg) => {
            let html = render_verify_page(&form.user_code, &nonce, Some(msg));
            Html(html).into_response()
        }
    }
}

/// GET /oauth/device/nonce — JSON nonce endpoint for CLI sign-challenge helper.
///
/// Returns: `{"user_code": "...", "nonce": "...", "challenge_prefix": "USERNAME"}`
/// The full challenge to sign is: `"{username}:{user_code}:{nonce}"`
pub async fn device_nonce(
    State(state): State<Arc<OAuthState>>,
    Query(query): Query<VerifyQuery>,
) -> Response {
    let user_code = match query.user_code.as_deref() {
        Some(c) if !c.is_empty() => c.replace('-', "").to_uppercase(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "user_code query param required"})),
            )
                .into_response();
        }
    };

    let device_code = {
        let user_code_map = state.device_code_by_user_code.read().await;
        user_code_map.get(&user_code).cloned()
    };

    let device_code = match device_code {
        Some(dc) => dc,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "unknown_code", "error_description": "user_code not found or expired"})),
            )
                .into_response();
        }
    };

    let device_codes = state.pending_device_codes.read().await;
    match device_codes.get(&device_code) {
        Some(pending) if pending.is_expired() => (
            StatusCode::GONE,
            Json(serde_json::json!({"error": "expired_token", "error_description": "Device code has expired"})),
        )
            .into_response(),
        Some(pending) => (
            StatusCode::OK,
            [
                (header::CACHE_CONTROL, "no-store"),
                (header::PRAGMA, "no-cache"),
            ],
            Json(serde_json::json!({
                "user_code": user_code,
                "nonce": pending.nonce,
                "challenge_template": "{username}:{user_code}:{nonce}",
            })),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "unknown_code", "error_description": "user_code not found"})),
        )
            .into_response(),
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

/// HTML-escape a string for safe embedding in HTML attributes and text.
pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Render a generic user-code entry page (shown when no user_code provided in URL).
fn render_entry_page(issuer_url: &str, error: Option<&str>) -> String {
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
  button {{ width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 1rem;
           cursor: pointer; font-weight: 500; background: #0066cc; color: #fff; margin-top: 8px; }}
  button:hover {{ background: #0052a3; }}
</style>
</head>
<body>
<div class="card">
  <h1>Device Verification</h1>
  <p>Enter the code displayed on your device to proceed to authorization.</p>
  {error_html}
  <form method="get" action="{action_url}/oauth/device/verify">
    <input type="text" name="user_code" class="code-input"
           placeholder="XXXX-XXXX" maxlength="9" autocomplete="off" autofocus>
    <button type="submit">Continue</button>
  </form>
</div>
</body>
</html>"#,
        error_html = error_html,
        action_url = html_escape(issuer_url),
    )
}

/// Render the Ed25519 challenge-response verification page.
pub(crate) fn render_verify_page(user_code: &str, nonce: &str, error: Option<&str>) -> String {
    let error_html = match error {
        Some(msg) => format!(r#"<div class="error">{}</div>"#, html_escape(msg)),
        None => String::new(),
    };
    // The challenge template (username is substituted by the user's client)
    let challenge_display = format!("{{username}}:{}:{}", user_code.replace('-', "").to_uppercase(), nonce);

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Authorize Device — hyprstream</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 560px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }}
  .card {{ background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
  p {{ color: #555; line-height: 1.5; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; word-break: break-all; }}
  .challenge-box {{ background: #f8f8f8; border: 1px solid #ddd; border-radius: 8px;
                    padding: 12px; margin: 12px 0; font-family: monospace; font-size: 0.85em;
                    word-break: break-all; }}
  .error {{ background: #fff0f0; color: #cc0000; border-radius: 8px; padding: 12px 16px;
            margin: 12px 0; font-size: 0.9rem; }}
  label {{ display: block; margin-top: 16px; color: #333; font-size: 0.9rem; font-weight: 500; }}
  input[type=text] {{ width: 100%; padding: 10px; font-size: 0.9rem; border: 2px solid #ddd;
                       border-radius: 8px; box-sizing: border-box; margin-top: 4px; }}
  input[type=text]:focus {{ border-color: #0066cc; outline: none; }}
  button {{ width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 1rem;
           cursor: pointer; font-weight: 500; background: #0066cc; color: #fff; margin-top: 24px; }}
  button:hover {{ background: #0052a3; }}
  .step {{ background: #f0f6ff; border-left: 3px solid #0066cc; padding: 10px 14px;
           margin: 12px 0; border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #333; }}
</style>
</head>
<body>
<div class="card">
  <h1>Authorize Device</h1>
  <p>Code: <code>{user_code_esc}</code></p>
  {error_html}
  <div class="step">
    <strong>Step 1:</strong> Run on your workstation:<br/>
    <code>hyprstream sign-challenge {user_code_esc}</code><br/>
    This signs the challenge with your Ed25519 key and prints the base64 signature.
  </div>
  <p>Challenge string (for reference):</p>
  <div class="challenge-box">{challenge_display_esc}</div>
  <form method="post" action="/oauth/device/verify">
    <input type="hidden" name="user_code" value="{user_code_esc}"/>
    <label>Username:
      <input type="text" name="username" required autocomplete="username" autofocus/>
    </label>
    <label>Signature (base64, 88 chars):
      <input type="text" name="signature" required size="88" autocomplete="off"
             placeholder="base64-encoded Ed25519 signature"/>
    </label>
    <button type="submit">Authorize Device</button>
  </form>
</div>
</body>
</html>"#,
        user_code_esc = html_escape(user_code),
        error_html = error_html,
        challenge_display_esc = html_escape(&challenge_display),
    )
}

/// Render the result page after successful authorization.
fn render_result_page(approved: bool, username: &str) -> String {
    let (title, message, color) = if approved {
        (
            "Device Authorized",
            format!(
                "Device authorized for user <strong>{}</strong>. You can close this window and return to your device.",
                html_escape(username)
            ),
            "#0a8a0a",
        )
    } else {
        (
            "Access Denied",
            "Access was denied for this device. You can close this window.".to_owned(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer};

    #[test]
    fn test_render_verify_page_contains_nonce() {
        let nonce = "abc123XYZ_nonce_value_here_for_test";
        let html = render_verify_page("ABCD-EFGH", nonce, None);
        assert!(html.contains(nonce), "rendered page must contain the nonce");
        assert!(html.contains("ABCDEFGH"), "rendered page must contain normalized user_code in challenge");
        assert!(html.contains("Authorize Device"), "page must have the heading");
    }

    #[test]
    fn test_render_verify_page_shows_error() {
        let html = render_verify_page("ABCD-EFGH", "somenonce", Some("Invalid signature"));
        assert!(html.contains("Invalid signature"), "error message must appear in page");
    }

    #[test]
    fn test_challenge_string_format() {
        // Challenge must be: "{username}:{user_code_normalized}:{nonce}"
        let username = "alice";
        let user_code = "ABCDEFGH";
        let nonce = "testnonce42";
        let challenge = format!("{}:{}:{}", username, user_code, nonce);
        assert_eq!(challenge, "alice:ABCDEFGH:testnonce42");
    }

    #[test]
    fn test_ed25519_signature_verification_roundtrip() {
        use ed25519_dalek::Verifier;

        // Generate a key pair
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();

        let username = "alice";
        let user_code = "ABCDEFGH";
        let nonce = "randomnonce12345678";
        let challenge = format!("{}:{}:{}", username, user_code, nonce);

        // Sign the challenge
        let signature = signing_key.sign(challenge.as_bytes());

        // Verify with the verifying key
        assert!(
            verifying_key.verify_strict(challenge.as_bytes(), &signature).is_ok(),
            "signature must verify successfully"
        );

        // Tampered challenge should fail
        let tampered = format!("mallory:{}:{}", user_code, nonce);
        assert!(
            verifying_key.verify_strict(tampered.as_bytes(), &signature).is_err(),
            "signature must NOT verify for tampered challenge"
        );
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>alert('xss')</script>"),
                   "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
        assert_eq!(html_escape(r#"say "hi""#), "say &quot;hi&quot;");
    }

    #[test]
    fn test_nonce_is_43_chars() {
        // 32 random bytes → base64url no-pad = ceil(32*8/6) = 43 chars
        let mut nonce_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);
        assert_eq!(nonce.len(), 43, "nonce must be 43 base64url chars");
    }
}
