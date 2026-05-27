//! RFC 8693 OAuth Token Exchange grant handler.
//!
//! Grant type: `urn:ietf:params:oauth:grant-type:token-exchange`
//!
//! Exchanges an existing credential (OIDC ID token, at+jwt, or WIT) for a
//! hyprstream at+jwt. Serves as the HTTP-layer complement to ExchangeWit (ZMQ)
//! and enables the MCP SDK's CrossAppAccessProvider enterprise flow.

use std::sync::Arc;

use axum::{
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use sha2::{Digest, Sha256};

use super::state::OAuthState;
use crate::services::generated::policy_client::IssueToken;

const TOKEN_TYPE_ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
const TOKEN_TYPE_ACCESS_TOKEN: &str = "urn:ietf:params:oauth:token-type:access_token";
const TOKEN_TYPE_JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
const ISSUED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

struct VerifiedSubject {
    sub: String,
    cnf_key_bytes: Option<[u8; 32]>,
    iat: i64,
}

/// POST /oauth/token — token-exchange grant (RFC 8693).
pub async fn exchange_token_exchange(
    state: &Arc<OAuthState>,
    subject_token: &str,
    subject_token_type: &str,
    audience: Option<&str>,
    scope: Option<&str>,
    actor_token: Option<&str>,
    requested_token_type: Option<&str>,
) -> Response {
    // Actor token (delegation) is deferred — RFC 8693 §4.
    if actor_token.is_some() {
        return tx_error(StatusCode::BAD_REQUEST, "invalid_request", "actor_token is not supported");
    }

    // Only access_token is supported as the requested output type.
    if let Some(rt) = requested_token_type {
        if rt != ISSUED_TOKEN_TYPE {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_target",
                "only urn:ietf:params:oauth:token-type:access_token is supported as requested_token_type",
            );
        }
    }

    let verified = match subject_token_type {
        TOKEN_TYPE_ID_TOKEN => verify_id_token(state, subject_token).await,
        TOKEN_TYPE_ACCESS_TOKEN => verify_access_token(state, subject_token),
        TOKEN_TYPE_JWT => verify_jwt(state, subject_token).await,
        _ => return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            &format!("unsupported subject_token_type: {subject_token_type}; supported: id_token, access_token, jwt"),
        ),
    };

    let verified = match verified {
        Ok(v) => v,
        Err(e) => return tx_error(StatusCode::UNAUTHORIZED, "invalid_grant", &e),
    };

    // Replay prevention: SHA-256 of the subject token as the replay key.
    // Covers all token types, regardless of whether they carry a jti claim.
    let token_hash = URL_SAFE_NO_PAD.encode(Sha256::digest(subject_token.as_bytes()));
    if !state.check_and_record_dpop_jti(&token_hash, verified.iat).await {
        return tx_error(StatusCode::BAD_REQUEST, "invalid_grant", "subject_token already used (replay)");
    }

    let requested_scopes: Option<Vec<String>> = scope.map(|s| {
        s.split_whitespace().map(str::to_owned).collect()
    });

    // Encode cnf key bytes for PolicyService (same path as user_pub_key in other flows).
    let user_pub_key = verified.cnf_key_bytes.map(|b| URL_SAFE_NO_PAD.encode(b));

    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes,
            ttl: Some(state.token_ttl),
            audience: audience.map(str::to_owned),
            subject: Some(verified.sub.clone()),
            user_pub_key,
            dpop_jkt: None,
        })
        .await;

    match result {
        Ok(token_info) => {
            let now = chrono::Utc::now().timestamp();
            let expires_in = (token_info.expires_at - now).max(0);
            tracing::info!(sub = %verified.sub, "Token exchange issued at+jwt");
            (
                StatusCode::OK,
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::PRAGMA, "no-cache"),
                ],
                Json(serde_json::json!({
                    "access_token": token_info.token,
                    "issued_token_type": ISSUED_TOKEN_TYPE,
                    "token_type": "Bearer",
                    "expires_in": expires_in,
                })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::error!(sub = %verified.sub, error = %e, "Token exchange issuance failed");
            tx_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "Failed to issue token")
        }
    }
}

/// Verify an OIDC ID token from a trusted issuer (CrossAppAccessProvider path).
///
/// `aud` is not strictly enforced — ID tokens target the OIDC client_id, not our
/// token endpoint. Trust is established by the `iss` being in `trusted_issuers`.
async fn verify_id_token(state: &Arc<OAuthState>, token: &str) -> Result<VerifiedSubject, String> {
    let unverified = hyprstream_rpc::auth::decode_unverified(token)
        .map_err(|e| format!("Cannot parse id_token: {e}"))?;

    let iss = if unverified.iss.is_empty() {
        return Err("id_token missing 'iss' claim".to_owned());
    } else {
        unverified.iss.clone()
    };

    let issuer_cfg = state
        .trusted_issuers
        .get(&iss)
        .ok_or_else(|| format!("Issuer not in trusted_issuers allow-list: {iss}"))?
        .clone();

    check_nbf(token)?;

    let vk = super::jwt_bearer::resolve_federated_key(state, &iss, token, issuer_cfg.allow_http)
        .await
        .map_err(|e| format!("JWKS key resolution failed for {iss}: {e}"))?;

    // No audience check: ID token aud = OIDC client_id, not our token endpoint.
    let claims = hyprstream_rpc::auth::decode_with_key(token, &vk, None)
        .map_err(|e| format!("id_token signature verification failed: {e}"))?;

    if claims.sub.is_empty() {
        return Err("id_token missing 'sub' claim".to_owned());
    }

    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes: None, // ID tokens carry no key binding
        iat: claims.iat,
    })
}

/// Verify an existing hyprstream at+jwt (downscoping / audience narrowing).
fn verify_access_token(state: &OAuthState, token: &str) -> Result<VerifiedSubject, String> {
    let vk = ed25519_dalek::VerifyingKey::from_bytes(&state.verifying_key_bytes)
        .map_err(|_| "server configuration error: invalid verifying key".to_owned())?;

    let claims = hyprstream_rpc::auth::jwt::decode(token, &vk, None)
        .map_err(|e| format!("access_token verification failed: {e}"))?;

    let cnf_key_bytes = claims.cnf_key_bytes();
    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes,
        iat: claims.iat,
    })
}

/// Verify a generic JWT — WIT from local trust store or federated OIDC issuer.
///
/// For `sub: service:*`: global trust store (CA-signed WIT).
/// For other subjects: issuer must be in `trusted_issuers`.
/// Audience must equal the token endpoint URL (same constraint as RFC 7523).
async fn verify_jwt(state: &Arc<OAuthState>, token: &str) -> Result<VerifiedSubject, String> {
    let unverified = hyprstream_rpc::auth::decode_unverified(token)
        .map_err(|e| format!("Cannot parse jwt: {e}"))?;

    let iss = if unverified.iss.is_empty() {
        return Err("jwt missing 'iss' claim".to_owned());
    } else {
        unverified.iss.clone()
    };

    let sub = unverified.sub.clone();
    check_nbf(token)?;

    let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));

    let vk = if sub.starts_with("service:") {
        let svc_name = sub.trim_start_matches("service:");
        hyprstream_service::global_trust_store()
            .resolve_one(svc_name)
            .ok_or_else(|| format!("Unknown service in trust store: {svc_name}"))?
    } else {
        let cfg = state
            .trusted_issuers
            .get(&iss)
            .ok_or_else(|| format!("Issuer not in trusted_issuers allow-list: {iss}"))?
            .clone();
        super::jwt_bearer::resolve_federated_key(state, &iss, token, cfg.allow_http)
            .await
            .map_err(|e| format!("JWKS key resolution failed for {iss}: {e}"))?
    };

    // Full verify with audience = token endpoint (RFC 7523 §3).
    let claims = hyprstream_rpc::auth::decode_with_key(token, &vk, Some(&token_endpoint))
        .map_err(|e| format!("JWT verification failed: {e}"))?;

    let cnf_key_bytes = claims.cnf_key_bytes(); // carry cnf.jwk through (WIT key binding)
    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes,
        iat: claims.iat,
    })
}

/// Decode `nbf` from JWT payload and reject if in the future (±5s clock skew).
fn check_nbf(jwt: &str) -> Result<(), String> {
    let nbf = (|| -> Option<i64> {
        let payload_b64 = jwt.split('.').nth(1)?;
        let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
        let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
        value.get("nbf")?.as_i64()
    })();
    if let Some(nbf) = nbf {
        let now = chrono::Utc::now().timestamp();
        if nbf > now + 5 {
            return Err("token not yet valid (nbf)".to_owned());
        }
    }
    Ok(())
}

fn tx_error(status: StatusCode, error: &str, description: &str) -> Response {
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
