//! Handler for `hyprstream sign-challenge`.
//!
//! Signs an Ed25519 challenge using the local user identity key stored in the OS keyring.
//!
//! Two modes:
//!
//! 1. **Device flow**: `hyprstream sign-challenge ABCD-EFGH`
//!    - Fetches nonce from `GET /oauth/device/nonce?user_code=ABCD-EFGH`
//!    - Constructs challenge: `"{username}:{user_code_normalized}:{nonce}"`
//!    - Signs and POSTs to `/oauth/device/verify` automatically
//!    - Prints: `"✓ Device authorized for {username}."`
//!
//! 2. **Auth code flow**: `hyprstream sign-challenge --nonce <n> --code-challenge <cc>`
//!    - Constructs challenge: `"{username}:{nonce}:{code_challenge}"`
//!    - Prints: `"Username: {username}\nSignature: {base64}"`
//!    - User pastes these into the browser challenge form.
// CLI handler intentionally prints to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use ed25519_dalek::Signer;


/// Handle `hyprstream sign-challenge [USER_CODE] [--nonce N] [--code-challenge CC]`
pub async fn handle_sign_challenge(
    user_code: Option<String>,
    nonce: Option<String>,
    code_challenge: Option<String>,
    server: Option<String>,
) -> Result<()> {
    // Load user identity key from OS keyring
    let (signing_key, username) = load_user_signing_key()?;

    match (user_code, nonce, code_challenge) {
        // Device flow: sign-challenge ABCD-EFGH
        (Some(user_code), None, None) => {
            handle_device_flow(signing_key, username, user_code, server).await
        }
        // Auth code flow: sign-challenge --nonce N --code-challenge CC
        (None, Some(nonce), Some(code_challenge)) => {
            handle_auth_code_flow(signing_key, username, nonce, code_challenge)
        }
        // Missing args for auth code flow
        (None, Some(_), None) => {
            anyhow::bail!("--code-challenge is required when --nonce is specified")
        }
        (None, None, Some(_)) => {
            anyhow::bail!("--nonce is required when --code-challenge is specified")
        }
        // Both modes specified — ambiguous
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            anyhow::bail!(
                "Specify either USER_CODE (device flow) or --nonce + --code-challenge (auth code flow)"
            )
        }
        // No args at all
        (None, None, None) => {
            anyhow::bail!(
                "Usage:\n  \
                 hyprstream sign-challenge ABCD-EFGH\n  \
                 hyprstream sign-challenge --nonce <nonce> --code-challenge <cc>"
            )
        }
    }
}

/// Device flow: fetch nonce, sign, POST to /oauth/device/verify.
async fn handle_device_flow(
    signing_key: ed25519_dalek::SigningKey,
    username: String,
    user_code: String,
    server: Option<String>,
) -> Result<()> {
    let base_url = resolve_server_url(server);
    let normalized = user_code.replace('-', "").to_uppercase();

    // Fetch the nonce from the server
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let nonce_url = format!(
        "{}/oauth/device/nonce?user_code={}",
        base_url, normalized
    );
    let resp = client
        .get(&nonce_url)
        .send()
        .await
        .context("Failed to reach OAuth server")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Server returned {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await.context("Invalid JSON from nonce endpoint")?;
    let nonce = json["nonce"]
        .as_str()
        .context("Missing 'nonce' in server response")?;

    // Construct and sign challenge: "{username}:{user_code_normalized}:{nonce}"
    let challenge = format!("{}:{}:{}", username, normalized, nonce);
    let signature = signing_key.sign(challenge.as_bytes());
    let sig_b64 = STANDARD.encode(signature.to_bytes());

    // POST to /oauth/device/verify.
    // Send Accept: application/json so the server returns structured JSON errors
    // instead of HTML (which would require fragile HTML parsing).
    let verify_url = format!("{}/oauth/device/verify", base_url);
    let resp = client
        .post(&verify_url)
        .header("Accept", "application/json")
        .form(&[
            ("user_code", user_code.as_str()),
            ("username", username.as_str()),
            ("signature", sig_b64.as_str()),
        ])
        .send()
        .await
        .context("Failed to POST to device verify endpoint")?;

    if resp.status().is_success() {
        println!("✓ Device authorized for {}.", username);
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        // Parse JSON error_description returned by verify_post when Accept: application/json
        let error_msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["error_description"].as_str().map(std::borrow::ToOwned::to_owned))
            .unwrap_or_else(|| format!("Server returned {}", status));
        anyhow::bail!("Authorization failed: {}", error_msg);
    }

    Ok(())
}

/// Auth code flow: sign and print Username + Signature for browser paste.
fn handle_auth_code_flow(
    signing_key: ed25519_dalek::SigningKey,
    username: String,
    nonce: String,
    code_challenge: String,
) -> Result<()> {
    // Construct challenge: "{username}:{nonce}:{code_challenge}"
    let challenge = format!("{}:{}:{}", username, nonce, code_challenge);
    let signature = signing_key.sign(challenge.as_bytes());
    let sig_b64 = STANDARD.encode(signature.to_bytes());

    println!("Username: {}", username);
    println!("Signature: {}", sig_b64);

    Ok(())
}

/// Load the user's Ed25519 signing key from the configured secrets directory.
/// Returns (signing_key, username).
fn load_user_signing_key() -> Result<(ed25519_dalek::SigningKey, String)> {
    let username = hyprstream_rpc::envelope::RequestIdentity::anonymous().user().to_owned();

    if let Some((sk, _vk)) = crate::config::HyprConfig::user_signing_key_bypass()? {
        return Ok((sk, username));
    }

    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir();
    let (sk, _vk) = crate::auth::credentials::load_or_generate_user_signing_key(&secrets_dir)
        .map_err(|e| anyhow::anyhow!(
            "Could not load user signing key: {e}\n\
             Run 'hyprstream wizard' to set up your identity first."
        ))?;

    Ok((sk, username))
}

/// Resolve the OAuth server URL from the option or config/default.
fn resolve_server_url(server: Option<String>) -> String {
    if let Some(s) = server {
        return s;
    }
    crate::config::HyprConfig::load()
        .map(|c| c.oauth.issuer_url())
        .unwrap_or_else(|_| "http://localhost:6791".to_owned())
}

