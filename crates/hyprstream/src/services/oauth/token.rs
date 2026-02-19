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
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Form, Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::state::{DeviceCodeStatus, OAuthState, RefreshTokenEntry};

/// Device code grant type URN (RFC 8628).
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

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
}

/// POST /oauth/token — token exchange
pub async fn exchange_token(
    State(state): State<Arc<OAuthState>>,
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
    match params.grant_type.as_str() {
        "authorization_code" => exchange_authorization_code(state, params).await,
        "refresh_token" => exchange_refresh_token(state, params).await,
        gt if gt == DEVICE_CODE_GRANT_TYPE => exchange_device_code(state, params).await,
        _ => token_error(
            StatusCode::BAD_REQUEST,
            "unsupported_grant_type",
            "Supported: authorization_code, refresh_token, device_code",
        ),
    }
}

/// Handle authorization_code grant type.
async fn exchange_authorization_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
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

    if computed_challenge != pending.code_challenge {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "PKCE code_verifier verification failed",
        );
    }

    tracing::info!(client_id = %params.client_id, "PKCE verified, issuing token");
    issue_token_with_refresh(&state, &params.client_id, pending.scopes, pending.resource).await
}

/// Handle refresh_token grant type (OAuth 2.1 with rotation).
async fn exchange_refresh_token(
    state: Arc<OAuthState>,
    params: TokenRequest,
) -> Response {
    let refresh_token = match params.refresh_token {
        Some(rt) => rt,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", "refresh_token is required"),
    };

    // Look up and remove refresh token (single-use rotation)
    let entry = {
        let mut tokens = state.refresh_tokens.write().await;
        tokens.remove(&refresh_token)
    };

    let entry = match entry {
        Some(e) => e,
        None => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                "Refresh token not found or already used",
            );
        }
    };

    if entry.is_expired() {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "Refresh token has expired",
        );
    }

    if params.client_id != entry.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "client_id does not match",
        );
    }

    // Issue new access token + rotated refresh token with the stored scopes/resource
    issue_token_with_refresh(&state, &entry.client_id, entry.scopes, entry.resource).await
}

/// Handle urn:ietf:params:oauth:grant-type:device_code grant type (RFC 8628 Section 3.4).
async fn exchange_device_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
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
            device_codes.remove(&device_code);
            drop(device_codes);
            let mut user_code_map = state.device_code_by_user_code.write().await;
            user_code_map.remove(&user_code);
            drop(user_code_map);
            issue_token_with_refresh(&state, &client_id, scopes, resource).await
        }
    }
}

/// Generate a cryptographically random refresh token string.
fn generate_refresh_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Issue a JWT access token via PolicyService, plus a rotated refresh token.
async fn issue_token_with_refresh(
    state: &OAuthState,
    client_id: &str,
    scopes: Vec<String>,
    resource: Option<String>,
) -> Response {
    let scope_str = scopes.join(" ");

    let result = state
        .policy_client
        .issue_token(&scopes, state.token_ttl, resource.as_deref().unwrap_or_default(), "")
        .await;

    match result {
        Ok(token_info) => {
            tracing::info!(client_id = %client_id, "Token issued successfully");
            let now = chrono::Utc::now().timestamp();
            let expires_in = (token_info.expires_at - now).max(0);

            // Generate and store a refresh token
            let refresh_token = generate_refresh_token();
            {
                let mut tokens = state.refresh_tokens.write().await;
                tokens.insert(refresh_token.clone(), RefreshTokenEntry {
                    client_id: client_id.to_owned(),
                    scopes,
                    resource,
                    expires_at: Instant::now() + Duration::from_secs(state.refresh_token_ttl as u64),
                });
            }

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
                    "scope": scope_str,
                    "refresh_token": refresh_token,
                })),
            )
                .into_response()
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
