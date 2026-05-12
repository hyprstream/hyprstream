//! OAuth 2.1 Token Endpoint.
//!
//! POST /oauth/token — exchanges authorization code, device code, or refresh token
//! for an access token.
//!
//! Supports:
//! - `grant_type=authorization_code` — PKCE + PolicyClient delegation
//! - `grant_type=urn:ietf:params:oauth:grant-type:device_code` — RFC 8628 device flow
//! - `grant_type=refresh_token` — OAuth 2.1 token refresh with rotation

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::{HeaderMap, header, StatusCode},
    response::{IntoResponse, Response},
    Form, Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use super::state::{DeviceCodeStatus, OAuthState, RefreshTokenEntry};
use crate::services::generated::policy_client::IssueToken;

/// Device code grant type URN (RFC 8628).
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
/// JWT bearer grant type URN (RFC 7523).
const JWT_BEARER_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// Token exchange grant type URN (RFC 8693).
const TOKEN_EXCHANGE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";

/// Token exchange request (application/x-www-form-urlencoded).
///
/// All fields are optional because different grant types use different subsets.
#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: String,
    pub client_id: String,
    // authorization_code fields
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub redirect_uri: Option<String>,
    #[serde(default)]
    pub code_verifier: Option<String>,
    // device_code field
    #[serde(default)]
    pub device_code: Option<String>,
    // refresh_token field
    #[serde(default)]
    pub refresh_token: Option<String>,
    // jwt-bearer assertion (RFC 7523)
    #[serde(default)]
    pub assertion: Option<String>,
    // token-exchange fields (RFC 8693)
    #[serde(default)]
    pub subject_token: Option<String>,
    #[serde(default)]
    pub subject_token_type: Option<String>,
    #[serde(default)]
    pub requested_token_type: Option<String>,
    #[serde(default)]
    pub actor_token: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub audience: Option<String>,
}

/// POST /oauth/token — token exchange
pub async fn exchange_token(
    State(state): State<Arc<OAuthState>>,
    req_headers: HeaderMap,
    Form(params): Form<TokenRequest>,
) -> Response {
    tracing::info!(
        grant_type = %params.grant_type,
        client_id = %params.client_id,
        has_code = params.code.is_some(),
        has_redirect_uri = params.redirect_uri.is_some(),
        has_code_verifier = params.code_verifier.is_some(),
        "Token exchange request received"
    );
    // Extract optional DPoP proof header (RFC 9449).
    let dpop_header: Option<String> = req_headers
        .get("DPoP")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    match params.grant_type.as_str() {
        "authorization_code" => exchange_authorization_code(state, params, dpop_header).await,
        "refresh_token" => exchange_refresh_token(state, params, dpop_header).await,
        gt if gt == DEVICE_CODE_GRANT_TYPE => exchange_device_code(state, params, dpop_header).await,
        gt if gt == JWT_BEARER_GRANT_TYPE => {
            let assertion = match params.assertion {
                Some(a) => a,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "assertion is required"),
            };
            super::jwt_bearer::exchange_jwt_bearer(&state, &params.client_id, &assertion).await
        }
        gt if gt == TOKEN_EXCHANGE_GRANT_TYPE => {
            let subject_token = match params.subject_token {
                Some(t) => t,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "subject_token is required"),
            };
            let subject_token_type = match params.subject_token_type {
                Some(t) => t,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "subject_token_type is required"),
            };
            super::token_exchange::exchange_token_exchange(
                &state,
                &subject_token,
                &subject_token_type,
                params.audience.as_deref(),
                params.scope.as_deref(),
                params.actor_token.as_deref(),
                params.requested_token_type.as_deref(),
            ).await
        }
        _ => token_error(
            StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            "Supported: authorization_code, refresh_token, device_code, jwt-bearer",
        ),
    }
}

/// Verify a DPoP proof for the token endpoint, check JTI replay, validate nonce.
///
/// Returns `None` when no DPoP header is present (DPoP is optional at the token endpoint).
/// Returns `Some(Ok(jkt))` on success; `Some(Err(response))` on failure.
async fn verify_dpop_at_token_endpoint(
    state: &OAuthState,
    dpop_header: Option<&str>,
) -> Option<Result<String, Response>> {
    let proof_str = dpop_header?;
    let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));
    let proof = match super::dpop::verify_dpop_proof(proof_str, "POST", &token_endpoint, None) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("DPoP proof verification failed: {e}");
            return Some(Err(token_error(StatusCode::BAD_REQUEST, "invalid_dpop_proof", &e.to_string())));
        }
    };
    // JTI replay check.
    if !state.check_and_record_dpop_jti(&proof.jti, proof.iat).await {
        tracing::warn!(jti = %proof.jti, "DPoP JTI replay detected");
        return Some(Err(token_error(StatusCode::BAD_REQUEST, "invalid_dpop_proof", "DPoP proof jti already used")));
    }
    // Nonce validation: if proof carries a nonce, it must be one we issued.
    if let Some(ref nonce) = proof.nonce {
        if !state.verify_dpop_nonce(nonce).await {
            tracing::warn!("DPoP nonce invalid or expired");
            return Some(Err(token_error(StatusCode::BAD_REQUEST, "use_dpop_nonce", "DPoP nonce invalid or expired")));
        }
    }
    Some(Ok(proof.jkt))
}

/// Handle authorization_code grant type.
async fn exchange_authorization_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
) -> Response {
    let code = match params.code {
        Some(c) => c,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "code is required"),
    };
    let redirect_uri = match params.redirect_uri {
        Some(r) => r,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "redirect_uri is required"),
    };
    let code_verifier = match params.code_verifier {
        Some(v) => v,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "code_verifier is required"),
    };

    // Look up and remove pending code (single-use)
    let pending = {
        let mut codes = state.pending_codes.write().await;
        codes.remove(&code)
    };

    let pending = match pending {
        Some(p) => p,
        None => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                "Authorization code not found or already used",
            );
        }
    };

    if pending.is_expired() {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "Authorization code has expired",
        );
    }

    if params.client_id != pending.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "client_id does not match",
        );
    }

    if redirect_uri != pending.redirect_uri {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "redirect_uri does not match",
        );
    }

    // PKCE verification
    let computed_challenge = {
        let digest = Sha256::digest(code_verifier.as_bytes());
        URL_SAFE_NO_PAD.encode(digest)
    };

    if computed_challenge.as_bytes().ct_eq(pending.code_challenge.as_bytes()).unwrap_u8() == 0 {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "PKCE code_verifier verification failed",
        );
    }

    // Verify DPoP if present.
    let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref()).await {
        None => None,
        Some(Ok(jkt)) => Some(jkt),
        Some(Err(resp)) => return resp,
    };

    tracing::info!(client_id = %params.client_id, username = %pending.username, "PKCE verified, issuing token");
    let sub = pending.username.clone();
    let vk_ref = pending.verifying_key.as_ref();
    issue_token_with_refresh(&state, &params.client_id, pending.scopes, pending.resource, &sub, pending.oidc_nonce, true, vk_ref, dpop_jkt).await
}

/// Handle refresh_token grant type (OAuth 2.1 with rotation).
async fn exchange_refresh_token(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
) -> Response {
    let refresh_token = match params.refresh_token {
        Some(rt) => rt,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "refresh_token is required"),
    };

    // Look up and atomically consume the refresh token (single-use rotation).
    // get_refresh_token handles lazy expiry; returns None if expired or missing.
    let entry = match state.get_refresh_token(&refresh_token).await {
        Ok(Some(e)) => e,
        Ok(None) => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                "Refresh token not found or already used",
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Refresh token store read failed");
            return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "Token store error");
        }
    };

    // Delete before issuing new token (rotation; prevents replay on store errors).
    if let Err(e) = state.delete_refresh_token(&refresh_token).await {
        tracing::error!(error = %e, "Refresh token store delete failed");
        return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "Token store error");
    }

    if params.client_id != entry.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "client_id does not match",
        );
    }

    // Verify DPoP if present.
    let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref()).await {
        None => None,
        Some(Ok(jkt)) => Some(jkt),
        Some(Err(resp)) => return resp,
    };

    // Reconstruct verifying key from stored bytes (cnf continuity across refreshes).
    let stored_vk: Option<ed25519_dalek::VerifyingKey> = entry.verifying_key_bytes
        .and_then(|b| ed25519_dalek::VerifyingKey::from_bytes(&b).ok());

    // Issue new access token + rotated refresh token. No id_token on refresh (OIDC Core § 12.2).
    issue_token_with_refresh(&state, &entry.client_id, entry.scopes, entry.resource, &entry.username, None, false, stored_vk.as_ref(), dpop_jkt).await
}

/// Handle urn:ietf:params:oauth:grant-type:device_code grant type (RFC 8628 Section 3.4).
async fn exchange_device_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
) -> Response {
    let device_code = match params.device_code {
        Some(dc) => dc,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "device_code is required"),
    };

    let mut device_codes = state.pending_device_codes.write().await;

    let pending = match device_codes.get_mut(&device_code) {
        Some(p) => p,
        None => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                "Device code not found or already used",
            );
        }
    };

    // Check expiration
    if pending.is_expired() {
        let user_code = pending.user_code.clone();
        device_codes.remove(&device_code);
        let mut user_code_map = state.device_code_by_user_code.write().await;
        user_code_map.remove(&user_code);
        return token_error(StatusCode::BAD_REQUEST, "expired_token", "The device code has expired");
    }

    // Validate client_id
    if params.client_id != pending.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "client_id does not match",
        );
    }

    // Rate limiting: check poll interval
    let now = Instant::now();
    if let Some(last) = pending.last_polled {
        if now.duration_since(last).as_secs() < pending.interval {
            return token_error(StatusCode::BAD_REQUEST, "slow_down", "Polling too frequently");
        }
    }
    pending.last_polled = Some(now);

    match pending.status {
        DeviceCodeStatus::Pending => {
            token_error(StatusCode::BAD_REQUEST, "authorization_pending", "The authorization request is still pending")
        }
        DeviceCodeStatus::Denied => {
            let user_code = pending.user_code.clone();
            device_codes.remove(&device_code);
            let mut user_code_map = state.device_code_by_user_code.write().await;
            user_code_map.remove(&user_code);
            token_error(StatusCode::BAD_REQUEST, "access_denied", "The user denied the authorization request")
        }
        DeviceCodeStatus::Approved => {
            let client_id = pending.client_id.clone();
            let scopes = pending.scopes.clone();
            let resource = pending.resource.clone();
            let user_code = pending.user_code.clone();
            // Use the approving user's username as the JWT subject.
            // approved_by must be set when status is Approved; error defensively if missing.
            let approved_by = match pending.approved_by.clone() {
                Some(u) => u,
                None => {
                    return token_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        "Device code approved but no approver identity recorded",
                    );
                }
            };
            let device_vk = pending.verifying_key;
            device_codes.remove(&device_code);
            drop(device_codes);
            let mut user_code_map = state.device_code_by_user_code.write().await;
            user_code_map.remove(&user_code);
            drop(user_code_map);

            // Verify DPoP if present.
            let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref()).await {
                None => None,
                Some(Ok(jkt)) => Some(jkt),
                Some(Err(resp)) => return resp,
            };

            // Device flow: no OIDC nonce and not initial OIDC auth.
            issue_token_with_refresh(&state, &client_id, scopes, resource, &approved_by, None, false, device_vk.as_ref(), dpop_jkt).await
        }
    }
}

/// Generate a cryptographically random refresh token string.
fn generate_refresh_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Issue a JWT access token via PolicyService, plus a rotated refresh token.
///
/// `sub` is the JWT subject (username). Must be non-empty:
/// - authorization_code flow: pass `pending.username` (the Ed25519-authenticated user from the consent page)
/// - refresh_token flow: pass the original sub from the RefreshTokenEntry
/// - device_code flow: pass the approving user's username (from challenge-response)
///
/// `initial_auth`: true for authorization_code exchange (may issue id_token),
/// false for refresh_token and device_code (never issues id_token per OIDC Core § 12.2).
async fn issue_token_with_refresh(
    state: &OAuthState,
    client_id: &str,
    scopes: Vec<String>,
    resource: Option<String>,
    sub: &str,
    oidc_nonce: Option<String>,
    initial_auth: bool,
    user_verifying_key: Option<&ed25519_dalek::VerifyingKey>,
    dpop_jkt: Option<String>,
) -> Response {
    let scope_str = scopes.join(" ");

    // DPoP jkt takes priority; fall back to raw key bytes for cnf.jwk.
    let user_pub_key_b64 = if dpop_jkt.is_none() {
        user_verifying_key.map(|vk| URL_SAFE_NO_PAD.encode(vk.to_bytes()))
    } else {
        None
    };

    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes: Some(scopes.clone()),
            ttl: Some(state.token_ttl),
            audience: resource.clone(),
            subject: Some(sub.to_owned()),
            user_pub_key: user_pub_key_b64,
            dpop_jkt: dpop_jkt.clone(),
        })
        .await;

    match result {
        Ok(token_info) => {
            tracing::info!(client_id = %client_id, "Token issued successfully");
            let now = chrono::Utc::now().timestamp();
            let expires_in = (token_info.expires_at - now).max(0);
            // Issue a fresh DPoP nonce when the client used DPoP (RFC 9449 §8).
            let dpop_nonce = if dpop_jkt.is_some() {
                Some(state.issue_dpop_nonce().await)
            } else {
                None
            };

            // Generate and persist a refresh token (RocksDB).
            let refresh_token = generate_refresh_token();
            {
                let entry = RefreshTokenEntry {
                    client_id: client_id.to_owned(),
                    username: sub.to_owned(),
                    scopes: scopes.clone(),
                    resource,
                    expires_at_unix: now + state.refresh_token_ttl as i64,
                    verifying_key_bytes: user_verifying_key.map(|vk| *vk.as_bytes()),
                };
                if let Err(e) = state.put_refresh_token(&refresh_token, &entry, state.refresh_token_ttl as u64).await {
                    tracing::error!(error = %e, "Failed to persist refresh token");
                }
            }

            // Build OIDC id_token when: scope includes "openid", signing key is available,
            // and this is an initial authorization (not refresh/device per OIDC Core § 12.2).
            let has_openid = scopes.iter().any(|s| s == "openid");
            let id_token = if has_openid && initial_auth && state.signing_key.is_some() {
                let id_exp = now + 300; // 5-minute id_token lifetime
                let mut id_claims = hyprstream_rpc::auth::IdTokenClaims::new(
                    state.issuer_url.clone(),
                    sub.to_owned(),
                    client_id.to_owned(),
                    now,
                    id_exp,
                )
                .with_nonce(oidc_nonce)
                .with_auth_time(now);

                // Add profile claims based on requested scopes.
                if let Some(user_store) = state.user_store_reader() {
                    if let Ok(Some(profile)) = user_store.get_profile(sub).await {
                        if scopes.iter().any(|s| s == "profile") {
                            id_claims.preferred_username = Some(sub.to_owned());
                            id_claims.name = profile.name;
                        }
                        if scopes.iter().any(|s| s == "email") {
                            id_claims.email = profile.email;
                            id_claims.email_verified = profile.email_verified;
                        }
                        // Use stable UUID sub if available.
                        if let Some(uuid_sub) = profile.sub {
                            id_claims.sub = uuid_sub;
                        }
                    }
                }

                // SAFETY: signing_key.is_some() checked in the outer condition.
                let Some(ref sk) = state.signing_key else { unreachable!() };
                let jwt_key = hyprstream_rpc::node_identity::derive_purpose_key(sk, "hyprstream-jwt-v1");
                let id_token_jwt = hyprstream_rpc::auth::jwt::encode_id_token(&id_claims, &jwt_key);
                Some(id_token_jwt)
            } else {
                None
            };

            let mut response_json = serde_json::json!({
                "access_token": token_info.token,
                "token_type": "Bearer",
                "expires_in": expires_in,
                "scope": scope_str,
                "refresh_token": refresh_token,
            });
            if let Some(id_token) = id_token {
                response_json["id_token"] = serde_json::Value::String(id_token);
            }

            let mut resp = (
                StatusCode::OK,
                [
                    (header::CACHE_CONTROL, "no-store"),
                    (header::PRAGMA, "no-cache"),
                ],
                Json(response_json),
            )
                .into_response();
            if let Some(nonce) = dpop_nonce {
                if let Ok(val) = axum::http::HeaderValue::from_str(&nonce) {
                    resp.headers_mut().insert("DPoP-Nonce", val);
                }
            }
            resp
        }
        Err(e) => {
            tracing::error!(client_id = %client_id, error = %e, "Token issuance failed");
            token_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Failed to issue access token",
            )
        }
    }
}

fn token_error(status: StatusCode, error: &str, description: &str) -> Response {
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
    ).into_response()
}
