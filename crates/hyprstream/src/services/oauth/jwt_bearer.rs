//! RFC 7523 JWT Bearer assertion grant handler.
//!
//! Grant type: `urn:ietf:params:oauth:grant-type:jwt-bearer`
//!
//! Validates a signed JWT assertion (WIT or federated OIDC token) and exchanges
//! it for an OAuth `at+jwt` access token via PolicyService. No refresh token is
//! issued — service-to-service flows are expected to re-assert per RFC 7523 § 4.

use std::sync::Arc;

use axum::{
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::VerifyingKey;

use super::state::OAuthState;
use crate::services::generated::policy_client::IssueToken;

/// Exchange a JWT bearer assertion for an access token (RFC 7523).
pub async fn exchange_jwt_bearer(
    state: &Arc<OAuthState>,
    client_id: &str,
    assertion: &str,
) -> Response {
    // Peek at claims without verifying — used only for issuer/subject routing.
    let unverified = match hyprstream_rpc::auth::decode_unverified(assertion) {
        Ok(c) => c,
        Err(e) => {
            return jwt_bearer_error(StatusCode::BAD_REQUEST, "invalid_grant", &format!("Cannot parse assertion: {e}"));
        }
    };

    let sub = unverified.sub.clone();
    let iss = {
        let iss = unverified.iss.clone();
        if iss.is_empty() {
            return jwt_bearer_error(StatusCode::BAD_REQUEST, "invalid_grant", "Assertion missing 'iss' claim");
        }
        iss
    };

    // nbf (not-before) check: reject assertions that aren't valid yet (RFC 7523 §3).
    {
        let now = chrono::Utc::now().timestamp();
        if let Some(nbf) = decode_jwt_nbf(assertion) {
            if nbf > now + 5 {
                return jwt_bearer_error(StatusCode::BAD_REQUEST, "invalid_grant", "Assertion not yet valid (nbf)");
            }
        }
    }

    // A service may publish drain and lead keys simultaneously. Verify against
    // every current candidate and retain the exact key that succeeded.
    let (claims, verified_service_key) = if sub.starts_with("service:") {
        let service_name = sub.trim_start_matches("service:");
        let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));
        match decode_with_any_local_service_key(assertion, service_name, &token_endpoint) {
            Some((claims, vk)) => (claims, Some(vk)),
            None => {
                return jwt_bearer_error(
                    StatusCode::UNAUTHORIZED,
                    "invalid_grant",
                    &format!("Unknown service: {service_name}"),
                );
            }
        }
    } else {
        // Federated subject — issuer must be explicitly trusted (RFC 7523 §3 allow-listing).
        let issuer_config = match state.trusted_issuers.get(&iss) {
            Some(cfg) => cfg.clone(),
            None => {
                return jwt_bearer_error(
                    StatusCode::UNAUTHORIZED,
                    "invalid_grant",
                    &format!("Issuer not in trusted_issuers allow-list: {iss}"),
                );
            }
        };
        // Fetch JWKS from the issuer's discovery document.
        let vk = match resolve_federated_key(state, &iss, assertion, issuer_config.allow_http).await {
            Ok(vk) => vk,
            Err(e) => {
                return jwt_bearer_error(StatusCode::UNAUTHORIZED, "invalid_grant", &e);
            }
        };
        let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));
        let claims = match hyprstream_rpc::auth::decode_with_key(assertion, &vk, Some(&token_endpoint)) {
            Ok(claims) => claims,
            Err(e) => return jwt_bearer_error(StatusCode::UNAUTHORIZED, "invalid_grant", &format!("Assertion verification failed: {e}")),
        };
        (claims, None)
    };

    let sub = claims.sub.clone();


    // Delegate token issuance to PolicyService (authorization enforced there via Casbin).
    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes: None, // Let PolicyService apply service defaults
            ttl: Some(state.token_ttl),
            audience: claims.aud.clone(),
            subject: Some(sub.clone()),
            // OAuth signs the RPC envelope; PolicyService needs the service
            // assertion key instead for its issued-token confirmation binding.
            user_pub_key: verified_service_key.map(|key| URL_SAFE_NO_PAD.encode(key.to_bytes())),
            dpop_jkt: None,
            issuer: None,
        })
        .await;

    match result {
        Ok(token_info) => {
            tracing::info!(client_id = %client_id, sub = %sub, "JWT bearer token issued");
            let now = chrono::Utc::now().timestamp();
            let expires_in = (token_info.expires_at - now).max(0);
            (
                StatusCode::OK,
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::PRAGMA, "no-cache"),
                ],
                Json(serde_json::json!({
                    "access_token": token_info.token,
                    "token_type": "Bearer",
                    "expires_in": expires_in,
                })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!(sub = %sub, error = %e, "JWT bearer token issuance failed");
            jwt_bearer_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "Failed to issue token")
        }
    }
}

/// Verify a local service assertion against each currently published key.
/// The returned key is the successful verifier, not a sorted-set selection.
pub(super) fn decode_with_any_local_service_key(
    assertion: &str,
    service_name: &str,
    expected_audience: &str,
) -> Option<(hyprstream_rpc::auth::Claims, VerifyingKey)> {
    decode_with_any_service_key_at(
        hyprstream_service::global_trust_store(), assertion, service_name,
        expected_audience, chrono::Utc::now().timestamp(),
    )
}

fn decode_with_any_service_key_at(
    trust: &hyprstream_service::TrustStore,
    assertion: &str,
    service_name: &str,
    expected_audience: &str,
    now: i64,
) -> Option<(hyprstream_rpc::auth::Claims, VerifyingKey)> {
    trust.published_keys_for_scope_at(service_name, now).into_iter().find_map(|candidate| {
        hyprstream_rpc::auth::decode_with_key(assertion, &candidate.verifying_key, Some(expected_audience))
            .ok().map(|claims| (claims, candidate.verifying_key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn attestation(expires_at: i64) -> hyprstream_service::Attestation {
        hyprstream_service::Attestation { scopes: std::iter::once("model".to_owned()).collect(), subject: None, jwt: None, expires_at, attested_by: None }
    }

    #[test]
    fn overlap_accepts_both_keys_then_rejects_retired_key_at_deadline() {
        let trust = hyprstream_service::TrustStore::new();
        let retired = SigningKey::generate(&mut rand::rngs::OsRng);
        let lead = SigningKey::generate(&mut rand::rngs::OsRng);
        let now = chrono::Utc::now().timestamp();
        let retirement = now + 60;
        trust.insert(retired.verifying_key(), attestation(retirement));
        trust.insert(lead.verifying_key(), attestation(now + 600));
        for signer in [&retired, &lead] {
            let assertion = hyprstream_rpc::auth::jwt::encode(
                &hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 300)
                    .with_issuer("https://issuer.example".to_owned())
                    .with_audience(Some("https://issuer.example/oauth/token".to_owned())), signer);
            let (_, verified) = decode_with_any_service_key_at(&trust, &assertion, "model", "https://issuer.example/oauth/token", now)
                .expect("each overlap key must verify its own assertion");
            assert_eq!(verified, signer.verifying_key());
        }
        let retired_assertion = hyprstream_rpc::auth::jwt::encode(
            &hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 300)
                .with_issuer("https://issuer.example".to_owned())
                .with_audience(Some("https://issuer.example/oauth/token".to_owned())), &retired);
        assert!(decode_with_any_service_key_at(&trust, &retired_assertion, "model", "https://issuer.example/oauth/token", retirement).is_none());
    }
}

/// Fetch a verifying key for a federated issuer by resolving JWKS.
/// Matches the `kid` JWT header against keys in the JWKS endpoint.
/// Falls back to the first Ed25519 (OKP/Ed25519) key when no `kid` matches.
/// Shared by jwt_bearer (RFC 7523) and token_exchange (RFC 8693).
pub(super) async fn resolve_federated_key(state: &Arc<OAuthState>, issuer: &str, assertion: &str, allow_http: bool) -> Result<VerifyingKey, String> {
    let metadata = state
        .oidc_discovery
        .get_metadata(issuer, allow_http)
        .await
        .map_err(|e| format!("OIDC discovery failed for {issuer}: {e}"))?;

    let jwks_resp = state
        .http_client
        .get(&metadata.jwks_uri)
        .send()
        .await
        .map_err(|e| format!("JWKS fetch failed: {e}"))?;

    let jwks: serde_json::Value = jwks_resp
        .json()
        .await
        .map_err(|e| format!("JWKS parse failed: {e}"))?;

    // Extract `kid` from JWT header for targeted lookup.
    let header_kid: Option<String> = (|| -> Option<String> {
        let parts: Vec<&str> = assertion.split('.').collect();
        let hdr = URL_SAFE_NO_PAD.decode(parts.first()?).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
        v["kid"].as_str().map(str::to_owned)
    })();

    let keys = jwks["keys"].as_array().ok_or("JWKS missing 'keys' array")?;

    for key in keys {
        // Filter by kid when present.
        if let Some(ref kid) = header_kid {
            if key["kid"].as_str() != Some(kid.as_str()) {
                continue;
            }
        }
        // Only accept OKP/Ed25519 keys.
        if key["kty"].as_str() != Some("OKP") || key["crv"].as_str() != Some("Ed25519") {
            continue;
        }
        if let Some(x_b64) = key["x"].as_str() {
            let bytes: [u8; 32] = URL_SAFE_NO_PAD
                .decode(x_b64)
                .ok()
                .and_then(|b| b.try_into().ok())
                .ok_or("Invalid Ed25519 key in JWKS")?;
            return VerifyingKey::from_bytes(&bytes).map_err(|e| format!("Invalid key bytes: {e}"));
        }
    }

    Err(format!("No matching Ed25519 key in JWKS for issuer {issuer}"))
}

/// Decode `nbf` from a JWT payload without verifying the signature.
fn decode_jwt_nbf(jwt: &str) -> Option<i64> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("nbf")?.as_i64()
}

fn jwt_bearer_error(status: StatusCode, error: &str, description: &str) -> Response {
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
