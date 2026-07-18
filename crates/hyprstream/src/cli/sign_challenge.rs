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
//!    - Derives the SSH-style key fingerprint (`SHA256:...`) of the local
//!      user signing key.
//!    - Constructs challenge: `"{fingerprint}:{nonce}:{code_challenge}"`
//!    - Prints: `"Fingerprint: {fingerprint}\nSignature: {base64}"`
//!    - User pastes these into the browser challenge form. The server
//!      resolves the username from the fingerprint via the pubkey reverse
//!      index — the client never declares an identity.
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
    insecure: bool,
) -> Result<()> {
    // Load user identity key from OS keyring
    let (signing_key, username) = load_user_signing_key()?;

    match (user_code, nonce, code_challenge) {
        // Device flow: sign-challenge ABCD-EFGH
        (Some(user_code), None, None) => {
            handle_device_flow(signing_key, username, user_code, server, insecure).await
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
    insecure: bool,
) -> Result<()> {
    let base_url = resolve_server_url(server);
    let normalized = user_code.replace('-', "").to_uppercase();

    // Build an HTTP client that trusts the local self-signed dev cert by
    // default (Option B), or — when `--insecure` is passed — disables TLS
    // verification entirely (Option A, opt-in with a warning).
    let client = build_oauth_http_client(insecure)?;

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

/// Auth code flow: sign and print Fingerprint + Signature for browser paste.
fn handle_auth_code_flow(
    signing_key: ed25519_dalek::SigningKey,
    _username: String,
    nonce: String,
    code_challenge: String,
) -> Result<()> {
    let verifying_key = signing_key.verifying_key();
    let fingerprint = crate::auth::pubkey_fingerprint(&verifying_key);

    // Construct challenge: "{fingerprint}:{nonce}:{code_challenge}"
    // The server resolves the user from the fingerprint via the pubkey
    // reverse index — no username on the wire.
    let challenge = format!("{}:{}:{}", fingerprint, nonce, code_challenge);
    let signature = signing_key.sign(challenge.as_bytes());
    let sig_b64 = STANDARD.encode(signature.to_bytes());

    println!("Fingerprint: {}", fingerprint);
    println!("Signature: {}", sig_b64);

    Ok(())
}

/// Load the user's Ed25519 signing key from the configured secrets directory.
///
/// Returns `(signing_key, os_username)`. The OS username is kept for the
/// device flow (which embeds it in the challenge string and posts it to
/// `/oauth/device/verify`); the auth-code flow ignores it and uses the
/// signing key's fingerprint instead.
fn load_user_signing_key() -> Result<(ed25519_dalek::SigningKey, String)> {
    // Device flow embeds the username in the challenge string. The OS user
    // matches what the wizard registers; falls back to "anonymous" for the
    // edge case where USER/LOGNAME aren't set.
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "anonymous".to_owned());

    if let Some((sk, _vk)) = crate::config::HyprConfig::user_signing_key_bypass()? {
        return Ok((sk, username));
    }

    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let (sk, _vk) = crate::auth::identity_store::load_or_generate_user_signing_key(&secrets_dir)
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

/// Name of the self-signed TLS certificate secret persisted in the shared
/// secrets directory (see `auth::identity_store::load_or_generate_tls_materials`).
const LOCAL_TLS_CERT_SECRET: &str = "tls-cert";

/// Build the `reqwest::Client` used to reach the OAuth server.
///
/// Two modes, resolving issue #450:
///
/// - **`insecure = true`** (Option A, explicit opt-in via `--insecure`):
///   disables certificate verification entirely with
///   `danger_accept_invalid_certs(true)` and prints a warning to stderr.
///   Use this only against throwaway dev servers where importing the cert
///   is impractical.
///
/// - **`insecure = false`** (default, Option B): if the OAuth server uses the
///   self-signed cert this project generates into the shared secrets
///   directory (the default local-dev / air-gapped mode — see
///   `server::tls` and `auth::identity_store::load_or_generate_tls_materials`),
///   that cert is loaded and added as a trusted root. This makes
///   `sign-challenge` "just work" against a local install *without* weakening
///   TLS for any other host. The system trust store is still consulted for
///   any cert it already trusts (e.g. a real CA-backed deployment), so adding
///   the local dev cert is strictly additive. If the cert cannot be read
///   (e.g. the secrets directory does not exist yet, or the server uses a
///   real CA), we fall through to the system trust store.
fn build_oauth_http_client(insecure: bool) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));

    if insecure {
        // Explicit opt-in: blanket TLS bypass. Warn loudly — this defeats
        // server authentication and must never be a default.
        eprintln!(
            "⚠️  --insecure: TLS certificate verification disabled. \
             Only use against a trusted local dev server."
        );
        builder = builder.danger_accept_invalid_certs(true);
    } else if let Some(cert_der) = load_local_tls_cert() {
        // Trust the project's own self-signed dev cert (stored as DER).
        // `from_pem` is backend-agnostic (works for both native-tls and
        // rustls reqwest backends), so convert DER → PEM here.
        match reqwest::Certificate::from_pem(&der_to_pem_cert(&cert_der)) {
            Ok(cert) => {
                builder = builder.add_root_certificate(cert);
            }
            Err(e) => {
                // Don't hard-fail: a malformed local cert shouldn't block a
                // user pointing at a real-CA server. Fall through to the
                // system trust store.
                tracing::debug!(
                    "could not parse local TLS cert as a root: {} \
                     (falling back to system trust store)",
                    e
                );
            }
        }
    }

    builder.build().context("failed to build OAuth HTTP client")
}

/// Load the local self-signed TLS certificate (DER) from the shared secrets
/// directory, if one is present.
///
/// Returns `None` silently when the secrets directory or the cert is absent —
/// callers fall back to the system trust store.
fn load_local_tls_cert() -> Option<Vec<u8>> {
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir().ok()?;
    crate::auth::identity_store::read_secret(&secrets_dir, LOCAL_TLS_CERT_SECRET)
        .ok()
        .flatten()
}

/// Wrap DER-encoded certificate bytes as a PEM block.
///
/// reqwest's `Certificate::from_pem` is accepted by every TLS backend, while
/// `from_der` is native-tls-only — so we convert here to stay backend-agnostic.
fn der_to_pem_cert(der: &[u8]) -> Vec<u8> {
    use base64::engine::general_purpose::STANDARD as B64;
    // base64 output is always ASCII, so we can work in raw bytes and avoid
    // any UTF-8 validation overhead (or `expect`).
    let b64 = B64.encode(der);
    let mut pem: Vec<u8> = Vec::with_capacity(b64.len() + 64);
    pem.extend_from_slice(b"-----BEGIN CERTIFICATE-----\n");
    for chunk in b64.as_bytes().chunks(64) {
        pem.extend_from_slice(chunk);
        pem.push(b'\n');
    }
    pem.extend_from_slice(b"-----END CERTIFICATE-----\n");
    pem
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn der_to_pem_roundtrips_through_reqwest() {
        // Generate a throwaway self-signed cert and confirm reqwest accepts
        // the PEM we produce from its DER encoding.
        let kp =
            rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).expect("gen key");
        let mut params =
            rcgen::CertificateParams::new(vec!["localhost".to_owned()]).expect("params");
        params.not_before = time::OffsetDateTime::now_utc();
        params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);
        let cert = params.self_signed(&kp).expect("self-signed");
        let der = cert.der().to_vec();

        let pem = der_to_pem_cert(&der);
        assert!(pem.starts_with(b"-----BEGIN CERTIFICATE-----\n"));
        assert!(pem.ends_with(b"-----END CERTIFICATE-----\n"));
        reqwest::Certificate::from_pem(&pem)
            .expect("reqwest must accept the PEM we produce from a valid DER cert");
    }

    #[test]
    fn der_to_pem_wraps_at_64_columns() {
        // 256 bytes of DER → base64 is longer than 64 chars, so we must wrap.
        let der = vec![0u8; 256];
        let pem = der_to_pem_cert(&der);
        let text = std::str::from_utf8(&pem).unwrap();
        for line in text.lines() {
            // Header/footer lines are exempt; body lines must be ≤ 64 chars.
            if line.starts_with("-----") {
                continue;
            }
            assert!(line.len() <= 64, "PEM body line exceeds 64 cols: {line}");
        }
    }
}
