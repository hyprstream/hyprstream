//! Shared helper for service JWT issuance with load-or-renew semantics.
//!
//! Both the wizard and bootstrap manager use this function to avoid
//! duplicating the "load existing JWT, renew if within 7 days of expiry"
//! logic.

use std::path::Path;

use anyhow::Result;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{SigningKey, VerifyingKey};

const NEW_EXPIRY_TTL: i64 = 30 * 86_400;
const RENEW_THRESHOLD: i64 = 7 * 86_400;

/// Load an existing service JWT from disk, or sign a new one if absent or
/// within `RENEW_THRESHOLD` seconds of expiry.
///
/// Does NOT write to disk — callers are responsible for persisting the
/// returned JWT via `identity_store::write_service_jwt`.
pub fn issue_or_load_service_jwt(
    credentials_dir: &Path,
    service_name: &str,
    ca_jwt_key: &SigningKey,
    service_vk: &VerifyingKey,
    now: i64,
) -> Result<String> {
    let existing = super::identity_store::load_service_jwt(credentials_dir, service_name)?;
    let needs_issue = match existing {
        None => true,
        Some(ref jwt) => {
            let exp = decode_jwt_exp(jwt).unwrap_or(0);
            (exp - now) <= RENEW_THRESHOLD
        }
    };

    if !needs_issue {
        if let Some(jwt) = existing {
            return Ok(jwt);
        }
    }

    let expiry = now + NEW_EXPIRY_TTL;
    let claims = hyprstream_rpc::auth::Claims::new(
        format!("service:{service_name}"),
        now,
        expiry,
    )
    .with_cnf_jwk(service_vk.as_bytes());

    Ok(hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, ca_jwt_key))
}

fn decode_jwt_exp(jwt: &str) -> Option<i64> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("exp")?.as_i64()
}
