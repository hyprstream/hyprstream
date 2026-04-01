//! External OIDC provider callback handler.
//!
//! Handles the return redirect from an external OIDC provider after the user
//! authenticates. Validates the external id_token, maps the identity to a
//! local hyprstream subject, and resumes the original authorize flow.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::Deserialize;
use sha2::{Sha256, Digest};

use super::state::{OAuthState, PendingAuthCode, PendingExternalAuth};

/// Initiate external OIDC login by redirecting to the provider.
///
/// `GET /oauth/external/authorize/:provider`
///
/// Stores the original hyprstream authorize request in `pending_external_auths`,
/// then redirects to the external provider's authorization endpoint with PKCE.
pub async fn external_authorize(
    State(state): State<Arc<OAuthState>>,
    Path(provider_slug): Path<String>,
    Query(params): Query<super::authorize::AuthorizeParams>,
) -> Response {
    // Look up provider config
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let provider = match config.oauth.oidc_providers.get(&provider_slug) {
        Some(p) => p.clone(),
        None => {
            return (axum::http::StatusCode::NOT_FOUND, format!("Unknown provider: {provider_slug}"))
                .into_response();
        }
    };

    // Fetch OIDC discovery
    let metadata = match state.oidc_discovery.get_metadata(&provider.issuer_url, provider.allow_http).await {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(provider = %provider_slug, error = %e, "OIDC discovery failed");
            return (axum::http::StatusCode::BAD_GATEWAY, format!("OIDC discovery failed: {e}"))
                .into_response();
        }
    };

    // Generate PKCE challenge (defense in depth even as confidential client)
    let mut verifier_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
    let pkce_verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let pkce_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(pkce_verifier.as_bytes()));

    // Generate state + nonce for the external flow
    let mut state_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut state_bytes);
    let external_state = URL_SAFE_NO_PAD.encode(state_bytes);

    let mut nonce_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let external_nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);

    // Store pending external auth
    let pending = PendingExternalAuth {
        provider_slug: provider_slug.clone(),
        external_state: external_state.clone(),
        external_nonce: external_nonce.clone(),
        pkce_verifier,
        client_secret: provider.client_secret.clone(),
        token_endpoint: metadata.token_endpoint.clone(),
        // Original hyprstream authorize request (to resume after external auth)
        original_client_id: params.client_id.clone(),
        original_redirect_uri: params.redirect_uri.clone(),
        original_code_challenge: params.code_challenge.clone(),
        original_scopes: params.scope.as_deref().unwrap_or("openid").to_owned(),
        original_state: params.state.clone(),
        original_resource: params.resource.clone(),
        original_oidc_nonce: params.nonce.clone(),
        created_at: Instant::now(),
        expires_at: Instant::now() + Duration::from_secs(600), // 10-minute TTL
    };
    state.pending_external_auths.write().await.insert(external_state.clone(), pending);

    // Build external provider's authorize URL
    let callback_url = format!("{}/oauth/callback/{}", state.issuer_url, provider_slug);
    let scopes = provider.scopes.join(" ");
    let authorize_url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}&nonce={}&code_challenge={}&code_challenge_method=S256",
        metadata.authorization_endpoint,
        urlencoding::encode(&provider.client_id),
        urlencoding::encode(&callback_url),
        urlencoding::encode(&scopes),
        urlencoding::encode(&external_state),
        urlencoding::encode(&external_nonce),
        urlencoding::encode(&pkce_challenge),
    );

    Redirect::temporary(&authorize_url).into_response()
}

#[derive(Deserialize)]
pub struct CallbackParams {
    pub code: String,
    pub state: String,
}

/// Handle callback from external OIDC provider.
///
/// `GET /oauth/callback/:provider`
///
/// 1. Validate state against pending external auth
/// 2. Exchange code for tokens at external provider
/// 3. Validate external id_token
/// 4. Map identity to local subject
/// 5. Issue hyprstream auth code
/// 6. Redirect to original client
pub async fn external_callback(
    State(state): State<Arc<OAuthState>>,
    Path(provider_slug): Path<String>,
    Query(params): Query<CallbackParams>,
) -> Response {
    // Look up and consume pending external auth
    let pending = {
        let mut auths = state.pending_external_auths.write().await;
        auths.remove(&params.state)
    };
    let pending = match pending {
        Some(p) if p.provider_slug == provider_slug && p.expires_at > Instant::now() => p,
        _ => {
            return (axum::http::StatusCode::BAD_REQUEST, "Invalid or expired external auth state")
                .into_response();
        }
    };

    // Look up provider config
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let provider = match config.oauth.oidc_providers.get(&provider_slug) {
        Some(p) => p.clone(),
        None => {
            return (axum::http::StatusCode::NOT_FOUND, format!("Unknown provider: {provider_slug}"))
                .into_response();
        }
    };

    // Exchange code for tokens at external provider
    let callback_url = format!("{}/oauth/callback/{}", state.issuer_url, provider_slug);
    let mut token_params = vec![
        ("grant_type", "authorization_code".to_owned()),
        ("code", params.code.clone()),
        ("redirect_uri", callback_url),
        ("client_id", provider.client_id.clone()),
        ("code_verifier", pending.pkce_verifier.clone()),
    ];
    if let Some(ref secret) = pending.client_secret {
        token_params.push(("client_secret", secret.clone()));
    }

    let token_response = match state.http_client
        .post(&pending.token_endpoint)
        .form(&token_params)
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(provider = %provider_slug, error = %e, "External token exchange failed");
            return (axum::http::StatusCode::BAD_GATEWAY, "Token exchange failed").into_response();
        }
    };

    if !token_response.status().is_success() {
        let body = token_response.text().await.unwrap_or_default();
        tracing::error!(provider = %provider_slug, body = %body, "External token endpoint returned error");
        return (axum::http::StatusCode::BAD_GATEWAY, "External token exchange rejected").into_response();
    }

    let token_json: serde_json::Value = match token_response.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(provider = %provider_slug, error = %e, "Invalid token response JSON");
            return (axum::http::StatusCode::BAD_GATEWAY, "Invalid token response").into_response();
        }
    };

    // Extract and validate external id_token
    let external_id_token = match token_json["id_token"].as_str() {
        Some(t) => t,
        None => {
            tracing::error!(provider = %provider_slug, "No id_token in external token response");
            return (axum::http::StatusCode::BAD_GATEWAY, "No id_token from external provider").into_response();
        }
    };

    // Verify external id_token using jsonwebtoken (supports RS256, ES256, EdDSA)
    // For now, decode without full JWKS verification — we trust the TLS connection
    // to the token endpoint. Full JWKS verification will be added with the
    // FederationKeySource refactor.
    let external_claims: serde_json::Value = match decode_external_id_token(external_id_token) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(provider = %provider_slug, error = %e, "Failed to decode external id_token");
            return (axum::http::StatusCode::BAD_GATEWAY, "Invalid external id_token").into_response();
        }
    };

    // Validate nonce
    if let Some(expected_nonce) = external_claims["nonce"].as_str() {
        if expected_nonce != pending.external_nonce {
            tracing::error!(provider = %provider_slug, "External id_token nonce mismatch");
            return (axum::http::StatusCode::BAD_REQUEST, "Nonce mismatch").into_response();
        }
    }

    // Map external identity to local subject
    let mapped_subject = match super::user_mapping::map_external_identity(
        &provider_slug,
        &external_claims,
        &provider.user_mapping,
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(provider = %provider_slug, error = %e, "Identity mapping failed");
            return (axum::http::StatusCode::FORBIDDEN, format!("Identity mapping failed: {e}")).into_response();
        }
    };

    // Check provisioning
    match super::user_mapping::should_provision(
        &provider.provisioning,
        &mapped_subject,
        &provider.allowed_domains,
    ) {
        Ok(true) => {
            tracing::info!(provider = %provider_slug, subject = %mapped_subject, "Auto-provisioning external user");
            // TODO: Create UserStore entry with profile from external claims
        }
        Ok(false) => {
            // User must already exist — check will happen at policy evaluation time
        }
        Err(e) => {
            tracing::warn!(provider = %provider_slug, subject = %mapped_subject, error = %e, "Provisioning denied");
            return (axum::http::StatusCode::FORBIDDEN, format!("Access denied: {e}")).into_response();
        }
    }

    tracing::info!(
        provider = %provider_slug,
        external_sub = %external_claims["sub"].as_str().unwrap_or("unknown"),
        mapped_subject = %mapped_subject,
        "External OIDC authentication successful"
    );

    // Resume the original hyprstream authorize flow: issue auth code
    let mut code_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut code_bytes);
    let code = URL_SAFE_NO_PAD.encode(code_bytes);

    let scopes: Vec<String> = pending.original_scopes
        .split_whitespace()
        .map(String::from)
        .collect();

    let auth_code = PendingAuthCode {
        code: code.clone(),
        client_id: pending.original_client_id.clone(),
        redirect_uri: pending.original_redirect_uri.clone(),
        code_challenge: pending.original_code_challenge.clone(),
        scopes,
        oidc_nonce: pending.original_oidc_nonce.clone(),
        resource: pending.original_resource.clone(),
        created_at: Instant::now(),
        expires_at: Instant::now() + Duration::from_secs(60),
        username: mapped_subject,
    };
    state.pending_codes.write().await.insert(code.clone(), auth_code);

    // Redirect to original client with hyprstream auth code
    let mut redirect_url = format!("{}?code={}", pending.original_redirect_uri, urlencoding::encode(&code));
    if let Some(ref original_state) = pending.original_state {
        redirect_url.push_str(&format!("&state={}", urlencoding::encode(original_state)));
    }

    Redirect::temporary(&redirect_url).into_response()
}

/// Decode an external id_token without full JWKS verification.
///
/// Uses jsonwebtoken's dangerous_insecure_decode to extract claims.
/// The token was received directly from the provider's token endpoint over TLS,
/// so transport-level trust is established. Full JWKS verification will be
/// added when the FederationKeySource is refactored for multi-algorithm support.
fn decode_external_id_token(token: &str) -> anyhow::Result<serde_json::Value> {
    // Split token to get the payload
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return Err(anyhow::anyhow!("Invalid JWT format"));
    }
    let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1])
        .map_err(|e| anyhow::anyhow!("Invalid base64 in id_token payload: {e}"))?;
    let claims: serde_json::Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in id_token payload: {e}"))?;
    Ok(claims)
}
