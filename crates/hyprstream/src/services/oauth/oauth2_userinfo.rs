//! Generic OAuth 2.0 userinfo endpoint fetch and claim normalization.
//!
//! Used by the `oauth2` and `github` provider kinds, which authenticate the user
//! with an opaque access token rather than a signed id_token JWT.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use crate::config::ClaimMapping;

/// Fetch the userinfo endpoint and return a normalized synthetic claims object.
///
/// The returned `serde_json::Value` always contains:
/// - `"sub"` — stable string identifier (coerced from integer if needed)
/// - `"name"` — display name or `null`
/// - `"email"` — email address or `null`
/// - `"email_verified"` — bool (`false` when the field is absent or null)
///
/// This object is structurally identical to what the OIDC path produces from a
/// verified id_token, so all downstream code (identity mapping, provisioning)
/// is unchanged.
pub async fn fetch_oauth2_claims(
    http_client: &reqwest::Client,
    userinfo_endpoint: &str,
    access_token: &str,
    mapping: &ClaimMapping,
) -> Result<serde_json::Value> {
    let response = http_client
        .get(userinfo_endpoint)
        .header("Authorization", format!("Bearer {access_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", "hyprstream")
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("userinfo request failed")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("userinfo endpoint returned {status}: {body}"));
    }

    let raw: serde_json::Value = response
        .json()
        .await
        .context("userinfo response is not valid JSON")?;

    // Extract sub — required; coerce integer to string (e.g. GitHub numeric id).
    let sub = match &raw[&mapping.sub] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        other if other.is_null() => {
            return Err(anyhow!("userinfo '{}' field is null", mapping.sub));
        }
        _ => return Err(anyhow!("userinfo '{}' field missing or wrong type", mapping.sub)),
    };

    // Extract optional string fields — null is acceptable.
    let name = mapping.name.as_deref()
        .map(|field| raw[field].clone())
        .unwrap_or(serde_json::Value::Null);

    let email = mapping.email.as_deref()
        .map(|field| raw[field].clone())
        .unwrap_or(serde_json::Value::Null);

    // Extract email_verified — absent or null field → false.
    let email_verified: bool = mapping.email_verified.as_deref()
        .and_then(|field| {
            let v = &raw[field];
            match v {
                serde_json::Value::Bool(b) => Some(*b),
                serde_json::Value::String(s) => Some(s.eq_ignore_ascii_case("true")),
                _ => None,
            }
        })
        .unwrap_or(false);

    Ok(serde_json::json!({
        "sub": sub,
        "name": name,
        "email": email,
        "email_verified": email_verified,
    }))
}
