//! External OAuth/OIDC provider callback handler.
//!
//! Handles the return redirect from an external provider after the user
//! authenticates. Supports two provider kinds:
//!
//! - `oidc`   — full OpenID Connect: discovery + id_token JWT verification
//! - `oauth2` — generic OAuth 2.0: fixed endpoints + userinfo HTTP call
//!
//! After claims are obtained (via either path), identity mapping, provisioning,
//! and auth code issuance are identical regardless of provider kind.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::Deserialize;
use sha2::{Sha256, Digest};

use crate::auth::user_store::UserProfile;
use crate::config::ProviderKind;
use super::state::{OAuthState, PendingAuthCode, PendingExternalAuth};

/// Initiate external OAuth/OIDC login by redirecting to the provider.
///
/// `GET /oauth/external/authorize/:provider`
///
/// Dispatches on `provider.kind`:
/// - `oidc`   — fetches OIDC discovery, sends PKCE + nonce
/// - `oauth2` — uses configured endpoints, omits nonce,
///   skips PKCE when `pkce_supported = false`
pub async fn external_authorize(
    State(state): State<Arc<OAuthState>>,
    Path(provider_slug): Path<String>,
    Query(params): Query<super::authorize::AuthorizeParams>,
) -> Response {
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let provider = match config.oauth.oidc_providers.get(&provider_slug) {
        Some(p) => p.clone(),
        None => {
            return (axum::http::StatusCode::NOT_FOUND, format!("Unknown provider: {provider_slug}"))
                .into_response();
        }
    };

    let mut state_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut state_bytes);
    let external_state = URL_SAFE_NO_PAD.encode(state_bytes);

    let callback_url = format!("{}/oauth/callback/{}", state.issuer_url, provider_slug);

    match provider.kind {
        ProviderKind::Oidc => {
            let issuer = match provider.issuer_url.as_deref() {
                Some(u) => u,
                None => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "oidc provider missing issuer_url").into_response(),
            };
            let metadata = match state.oidc_discovery.get_metadata(issuer, provider.allow_http).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(provider = %provider_slug, error = %e, "OIDC discovery failed");
                    return (axum::http::StatusCode::BAD_GATEWAY,
                        format!("OIDC discovery failed: {e}")).into_response();
                }
            };

            let mut verifier_bytes = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
            let pkce_verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
            let pkce_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(pkce_verifier.as_bytes()));

            let mut nonce_bytes = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
            let external_nonce = URL_SAFE_NO_PAD.encode(nonce_bytes);

            let scopes = provider.effective_scopes().join(" ");
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

            let pending = PendingExternalAuth {
                provider_slug: provider_slug.clone(),
                external_state: external_state.clone(),
                external_nonce,
                provider_kind: ProviderKind::Oidc,
                pkce_supported: true,
                pkce_verifier,
                client_secret: provider.client_secret.clone(),
                token_endpoint: metadata.token_endpoint.clone(),
                original_client_id: params.client_id.clone(),
                original_redirect_uri: params.redirect_uri.clone(),
                original_code_challenge: params.code_challenge.clone(),
                original_scopes: params.scope.as_deref().unwrap_or("openid").to_owned(),
                original_state: params.state.clone(),
                original_resource: params.resource.clone(),
                original_oidc_nonce: params.nonce.clone(),
                created_at: Instant::now(),
                expires_at: Instant::now() + Duration::from_secs(600),
            };
            state.pending_external_auths.write().await.insert(external_state.clone(), pending);
            Redirect::temporary(&authorize_url).into_response()
        }

        ProviderKind::OAuth2 => {
            let auth_endpoint = match provider.effective_authorization_endpoint() {
                Some(u) => u,
                None => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "oauth2 provider missing authorization_endpoint").into_response(),
            };
            let token_endpoint = match provider.effective_token_endpoint_url() {
                Some(u) => u.to_owned(),
                None => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "oauth2 provider missing token_endpoint_url").into_response(),
            };

            let pkce_supported = provider.effective_pkce_supported();
            let (pkce_verifier, pkce_challenge) = if pkce_supported {
                let mut verifier_bytes = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
                let v = URL_SAFE_NO_PAD.encode(verifier_bytes);
                let c = URL_SAFE_NO_PAD.encode(Sha256::digest(v.as_bytes()));
                (v, Some(c))
            } else {
                (String::new(), None)
            };

            let scopes = provider.effective_scopes().join(" ");
            let mut authorize_url = format!(
                "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}",
                auth_endpoint,
                urlencoding::encode(&provider.client_id),
                urlencoding::encode(&callback_url),
                urlencoding::encode(&scopes),
                urlencoding::encode(&external_state),
            );
            if let Some(ref challenge) = pkce_challenge {
                authorize_url.push_str(&format!(
                    "&code_challenge={}&code_challenge_method=S256",
                    urlencoding::encode(challenge)
                ));
            }

            let pending = PendingExternalAuth {
                provider_slug: provider_slug.clone(),
                external_state: external_state.clone(),
                external_nonce: String::new(),
                provider_kind: provider.kind.clone(),
                pkce_supported,
                pkce_verifier,
                client_secret: provider.client_secret.clone(),
                token_endpoint,
                original_client_id: params.client_id.clone(),
                original_redirect_uri: params.redirect_uri.clone(),
                original_code_challenge: params.code_challenge.clone(),
                original_scopes: params.scope.as_deref().unwrap_or("openid").to_owned(),
                original_state: params.state.clone(),
                original_resource: params.resource.clone(),
                original_oidc_nonce: params.nonce.clone(),
                created_at: Instant::now(),
                expires_at: Instant::now() + Duration::from_secs(600),
            };
            state.pending_external_auths.write().await.insert(external_state.clone(), pending);
            Redirect::temporary(&authorize_url).into_response()
        }
    }
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
    ];
    if pending.pkce_supported {
        token_params.push(("code_verifier", pending.pkce_verifier.clone()));
    }
    if let Some(ref secret) = pending.client_secret {
        token_params.push(("client_secret", secret.clone()));
    }

    let mut request = state.http_client
        .post(&pending.token_endpoint)
        .form(&token_params)
        .timeout(Duration::from_secs(10));

    // GitHub (and many generic OAuth 2.0 providers) default to form-encoded responses;
    // Accept: application/json ensures we always get parseable JSON.
    if matches!(pending.provider_kind, ProviderKind::OAuth2) {
        request = request.header("Accept", "application/json");
    }

    let token_response = match request.send().await {
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

    // Obtain normalised claims — dispatching on provider kind.
    // Both paths produce the same structure: { sub, name, email, email_verified }.
    let external_claims: serde_json::Value = match pending.provider_kind {
        ProviderKind::Oidc => {
            // Full OpenID Connect: verify the id_token JWT against provider JWKS.
            let external_id_token = match token_json["id_token"].as_str() {
                Some(t) => t,
                None => {
                    tracing::error!(provider = %provider_slug, "No id_token in external token response");
                    return (axum::http::StatusCode::BAD_GATEWAY, "No id_token from external provider").into_response();
                }
            };

            let issuer = match provider.issuer_url.as_deref() {
                Some(u) => u,
                None => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "oidc provider missing issuer_url").into_response(),
            };

            // Verify external id_token signature against the provider's JWKS.
            //
            // Fetches the JWKS from the provider's discovery endpoint, selects the
            // appropriate key (by kid or algorithm), and verifies the JWT signature.
            // Supports RS256, ES256, and EdDSA algorithms.
            let metadata = match state.oidc_discovery.get_metadata(issuer, provider.allow_http).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(provider = %provider_slug, error = %e, "OIDC discovery for JWKS failed");
                    return (axum::http::StatusCode::BAD_GATEWAY, "OIDC discovery for JWKS verification failed").into_response();
                }
            };

            let verified = match crate::auth::id_token_verify::verify_id_token(
                external_id_token,
                &metadata.jwks_uri,
                issuer,
                &provider.client_id,
                &state.http_client,
            ).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!(provider = %provider_slug, error = %e, "External id_token JWKS verification failed");
                    return (axum::http::StatusCode::BAD_GATEWAY, "External id_token verification failed").into_response();
                }
            };

            let claims = verified.claims;

            // Validate issuer (OIDC Core Section 3.1.3.7)
            if claims["iss"].as_str() != Some(issuer) {
                tracing::error!(
                    provider = %provider_slug,
                    expected = %issuer,
                    got = %claims["iss"],
                    "External id_token issuer mismatch"
                );
                return (axum::http::StatusCode::BAD_REQUEST, "Issuer mismatch").into_response();
            }

            // Validate audience (OIDC Core Section 3.1.3.6)
            let aud_valid = match &claims["aud"] {
                serde_json::Value::String(s) => s == &provider.client_id,
                serde_json::Value::Array(arr) => arr.iter().any(|v| v.as_str() == Some(&provider.client_id)),
                _ => false,
            };
            if !aud_valid {
                tracing::error!(provider = %provider_slug, "External id_token audience mismatch");
                return (axum::http::StatusCode::BAD_REQUEST, "Audience mismatch").into_response();
            }

            // Validate expiration (OIDC Core Section 3.1.3.6)
            if let Some(exp) = claims["exp"].as_i64() {
                let now = chrono::Utc::now().timestamp();
                let skew = provider.clock_skew_seconds as i64;
                if now > exp + skew {
                    tracing::error!(provider = %provider_slug, "External id_token expired");
                    return (axum::http::StatusCode::BAD_REQUEST, "Token expired").into_response();
                }
            } else {
                tracing::error!(provider = %provider_slug, "External id_token missing exp claim");
                return (axum::http::StatusCode::BAD_REQUEST, "Missing exp claim").into_response();
            }

            // Validate nonce (REQUIRED when sent in auth request — OIDC Core Section 3.1.3.6)
            match claims["nonce"].as_str() {
                Some(nonce) if nonce == pending.external_nonce => { /* valid */ }
                Some(_) => {
                    tracing::error!(provider = %provider_slug, "External id_token nonce mismatch");
                    return (axum::http::StatusCode::BAD_REQUEST, "Nonce mismatch").into_response();
                }
                None => {
                    tracing::error!(provider = %provider_slug, "External id_token missing nonce (required)");
                    return (axum::http::StatusCode::BAD_REQUEST, "Missing nonce").into_response();
                }
            }

            claims
        }

        ProviderKind::OAuth2 => {
            // Generic OAuth 2.0 / GitHub: exchange gave us an opaque access_token.
            // Fetch user identity from the provider's userinfo endpoint.
            let access_token = match token_json["access_token"].as_str() {
                Some(t) => t,
                None => {
                    tracing::error!(provider = %provider_slug, "No access_token in external token response");
                    return (axum::http::StatusCode::BAD_GATEWAY, "No access_token from external provider").into_response();
                }
            };

            let userinfo_url = match provider.effective_userinfo_endpoint() {
                Some(u) => u.to_owned(),
                None => {
                    tracing::error!(provider = %provider_slug, "OAuth2 provider missing userinfo_endpoint");
                    return (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        "oauth2 provider missing userinfo_endpoint").into_response();
                }
            };

            let mapping = provider.effective_claim_mapping();
            match super::oauth2_userinfo::fetch_oauth2_claims(
                &state.http_client,
                &userinfo_url,
                access_token,
                &mapping,
            ).await {
                Ok(claims) => claims,
                Err(e) => {
                    tracing::error!(provider = %provider_slug, error = %e, "OAuth2 userinfo fetch failed");
                    return (axum::http::StatusCode::BAD_GATEWAY,
                        format!("Userinfo fetch failed: {e}")).into_response();
                }
            }
        }
    };

    // Map external identity to local subject
    let mapped_subject = match super::user_mapping::map_external_identity(
        &provider_slug,
        &external_claims,
        &provider.user_mapping,
        &state.issuer_url,
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
            provision_federated_user(
                &state,
                &provider_slug,
                &mapped_subject,
                &external_claims,
                &provider.default_scopes,
            ).await;
        }
        Ok(false) => {
            // Deny mode: user must already exist; reject unknown subjects early.
            if let Some(ref user_svc) = state.user_service {
                match user_svc.store().get_profile(&mapped_subject).await {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        tracing::warn!(
                            provider = %provider_slug,
                            subject = %mapped_subject,
                            "Deny mode: user not registered — rejecting"
                        );
                        return (
                            axum::http::StatusCode::FORBIDDEN,
                            format!("Access denied: '{mapped_subject}' is not registered. Contact your administrator."),
                        ).into_response();
                    }
                    Err(e) => {
                        tracing::error!(subject = %mapped_subject, error = %e, "User lookup failed");
                        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "User lookup failed").into_response();
                    }
                }
            }
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
        verifying_key: None, // external OIDC — no local Ed25519 key binding
        dpop_jkt: None, // external OIDC flow — no PAR DPoP binding
    };
    state.pending_codes.write().await.insert(code.clone(), auth_code);

    // Redirect to original client with hyprstream auth code
    let mut redirect_url = format!("{}?code={}", pending.original_redirect_uri, urlencoding::encode(&code));
    if let Some(ref original_state) = pending.original_state {
        redirect_url.push_str(&format!("&state={}", urlencoding::encode(original_state)));
    }

    Redirect::temporary(&redirect_url).into_response()
}

/// Create a UserStore entry and write Casbin rules for a first-time federated login.
///
/// Idempotent: if the subject already has a profile, skips all writes silently.
/// Non-fatal: errors are logged but do not abort the login flow.
async fn provision_federated_user(
    state: &OAuthState,
    provider_slug: &str,
    subject: &str,
    external_claims: &serde_json::Value,
    default_scopes: &[String],
) {
    let Some(ref user_svc) = state.user_service else {
        tracing::warn!(provider = %provider_slug, "No user store — skipping federated provisioning");
        return;
    };
    let store = user_svc.store();

    // Check idempotency before writing.
    let already_exists = match store.get_profile(subject).await {
        Ok(Some(_)) => true,
        Ok(None) => false,
        Err(e) => {
            tracing::error!(subject = %subject, error = %e, "Failed to check user existence");
            return;
        }
    };

    if already_exists {
        tracing::debug!(subject = %subject, "Federated user already provisioned — skipping");
        return;
    }

    // First login: register and populate profile from external claims.
    if let Err(e) = store.register(subject).await {
        tracing::error!(subject = %subject, error = %e, "Failed to register federated user");
        return;
    }
    let profile = UserProfile {
        sub: None,
        name: external_claims["name"].as_str().map(str::to_owned),
        email: external_claims["email"].as_str().map(str::to_owned),
        email_verified: external_claims["email_verified"].as_bool(),
        active: Some(true),
        external_id: external_claims["sub"].as_str().map(str::to_owned),
        atproto_did: None,
    };
    if let Err(e) = store.set_profile(subject, profile).await {
        tracing::warn!(subject = %subject, error = %e, "Failed to set profile for federated user");
    }

    // Write Casbin rules for default_scopes.
    // Scope format: "action:resource_type:resource_id" — e.g. "infer:model:*", "read:*:*"
    let Some(pm) = crate::auth::global_policy_manager() else {
        tracing::warn!(provider = %provider_slug, "PolicyManager not available — default_scopes not applied");
        return;
    };

    // Self-ownership rules: always grant access to the user's own namespace
    // (user:{sub}:*) so JIT users can at minimum read/write their own profile
    // and settings — regardless of what default_scopes the provider supplies.
    // Addresses the zero-capabilities-on-first-login gap (#182).
    let self_ns = format!("user:{subject}:*");
    if let Err(e) = pm.add_policy_with_domain(subject, "*", &self_ns, "*", "allow").await {
        tracing::warn!(subject = %subject, error = %e, "Failed to write self-ownership Casbin rule");
    }

    // If no scopes are configured by the provider, fall back to viewer-level
    // defaults so the user can at minimum query models and registry entries.
    // Operators can tighten or widen via policy templates after provisioning.
    let effective_scopes: Vec<String> = if default_scopes.is_empty() {
        vec!["query:model:*".to_owned(), "query:registry:*".to_owned()]
    } else {
        default_scopes.to_vec()
    };

    for scope in &effective_scopes {
        let parts: Vec<&str> = scope.splitn(3, ':').collect();
        let (action, resource) = match parts.as_slice() {
            [action, rtype, rid] if *rtype == "*" && *rid == "*" => (*action, "*".to_owned()),
            [action, rtype, rid] => (*action, format!("{rtype}:{rid}")),
            [action] => (*action, "*".to_owned()),
            _ => continue,
        };
        if let Err(e) = pm.add_policy_with_domain(subject, "*", &resource, action, "allow").await {
            tracing::error!(subject = %subject, scope = %scope, error = %e, "Failed to write Casbin rule");
        }
    }
    if let Err(e) = pm.save().await {
        tracing::error!(subject = %subject, error = %e, "Failed to persist Casbin rules after provisioning");
    }

    tracing::info!(
        provider = %provider_slug,
        subject = %subject,
        scopes = ?effective_scopes,
        "Federated user provisioned"
    );
}
