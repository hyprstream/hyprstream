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

use crate::auth::user_store::UserProfilePatch;
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

/// Reject an upstream error using only public response metadata.
fn check_token_response(
    response: reqwest::Response,
    provider_slug: &str,
) -> Result<reqwest::Response, (axum::http::StatusCode, &'static str)> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    // Never read or log the body, even at DEBUG/TRACE: error fields and provider
    // diagnostics can contain credentials, and arbitrary providers have no safe
    // redaction contract. Keep only the configured provider and HTTP status.
    tracing::error!(
        provider = %provider_slug,
        status = status.as_u16(),
        "External token endpoint returned error"
    );
    Err((
        axum::http::StatusCode::BAD_GATEWAY,
        "External token exchange rejected",
    ))
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

    let token_response = match check_token_response(token_response, &provider_slug) {
        Ok(response) => response,
        Err(response) => return response.into_response(),
    };

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

    // Map external identity to a candidate local username. This is a *hint* —
    // the authoritative username is resolved by the credential store, not by
    // the mapping function.
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

    // Resolve the issuer to use for the normalized (issuer, subject) binding.
    // Prefer the provider's configured issuer_url, then the id_token `iss`
    // claim, then the provider slug as a last-resort synthetic issuer.
    let binding_issuer = provider.issuer_url.as_deref()
        .or_else(|| external_claims["iss"].as_str())
        .unwrap_or(&provider_slug)
        .to_owned();
    let external_sub = match external_claims["sub"].as_str() {
        Some(s) if !s.is_empty() => s.to_owned(),
        _ => {
            tracing::error!(provider = %provider_slug, "External token missing required 'sub' claim — rejecting");
            return (axum::http::StatusCode::BAD_REQUEST, "External token missing required 'sub' claim").into_response();
        }
    };

    // Resolve the authoritative local username via the credential store.
    // In auto-provision mode this atomically binds (issuer, subject) and may
    // create the local account. In deny mode the binding must already exist.
    // Either way: never continue with the pre-binding candidate if the store
    // resolves a different authoritative username.
    let authoritative_username = match super::user_mapping::should_provision(
        &provider.provisioning,
        &mapped_subject,
        &provider.allowed_domains,
    ) {
        Ok(true) => {
            // Auto-provision: atomically resolve-or-bind the external identity.
            match provision_federated_user(
                &state,
                &binding_issuer,
                &external_sub,
                &mapped_subject,
                &external_claims,
                &provider.default_scopes,
            ).await {
                Ok(username) => username,
                Err(response) => return response,
            }
        }
        Ok(false) => {
            // Deny mode: the user must already exist. Resolve the normalized
            // (issuer, subject) binding first, then fall back to the mapped
            // username for legacy profiles that pre-date the binding table.
            if let Some(ref user_svc) = state.user_service {
                let store = user_svc.store();
                // Try normalized binding first.
                match store.get_external_identity_user(&binding_issuer, &external_sub).await {
                    Ok(Some(username)) => username,
                    Ok(None) => {
                        // No normalized binding — try legacy username lookup.
                        match store.get_profile(&mapped_subject).await {
                            Ok(Some(_)) => mapped_subject.clone(),
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
                    Err(e) => {
                        tracing::error!(issuer = %binding_issuer, subject = %external_sub, error = %e, "External identity lookup failed");
                        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "User lookup failed").into_response();
                    }
                }
            } else {
                mapped_subject.clone()
            }
        }
        Err(e) => {
            tracing::warn!(provider = %provider_slug, subject = %mapped_subject, error = %e, "Provisioning denied");
            return (axum::http::StatusCode::FORBIDDEN, format!("Access denied: {e}")).into_response();
        }
    };

    tracing::info!(
        provider = %provider_slug,
        external_sub = %external_claims["sub"].as_str().unwrap_or("unknown"),
        mapped_subject = %mapped_subject,
        authoritative_username = %authoritative_username,
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
        username: authoritative_username,
        verifying_key: None, // external OIDC — no local Ed25519 key binding
        dpop_jkt: None, // external OIDC flow — no PAR DPoP binding
        client_assertion_jkt: None, // external OIDC flow — no PAR client-auth binding
    };
    state.pending_codes.write().await.insert(code.clone(), auth_code);

    // Redirect to original client with hyprstream auth code
    let mut redirect_url = format!("{}?code={}", pending.original_redirect_uri, urlencoding::encode(&code));
    if let Some(ref original_state) = pending.original_state {
        redirect_url.push_str(&format!("&state={}", urlencoding::encode(original_state)));
    }

    Redirect::temporary(&redirect_url).into_response()
}

/// Atomically resolve or bind an external IdP identity and provision profile
/// data for a first-time federated login.
///
/// Returns the authoritative local username — which may differ from the
/// candidate when the `(issuer, subject)` binding already existed. The caller
/// MUST use the returned username, never the pre-binding candidate.
///
/// Fail-closed: any credential-store error rejects the login. Casbin policy
/// writes are best-effort (logged, not fatal) because they are a separate
/// subsystem from the credential store.
async fn provision_federated_user(
    state: &OAuthState,
    issuer: &str,
    external_sub: &str,
    candidate_username: &str,
    external_claims: &serde_json::Value,
    default_scopes: &[String],
) -> Result<String, Response> {
    let Some(ref user_svc) = state.user_service else {
        tracing::error!("Federated provisioning failed: credential store is not configured");
        return Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "credential store is not configured",
        )
            .into_response());
    };
    let store = user_svc.store();

    // Atomically resolve-or-bind the normalized (issuer, subject) identity.
    // This is the authoritative step — it creates the local account and binding
    // in one operation, or returns the existing binding's username.
    let resolution = match store
        .resolve_or_bind_external_idp(issuer, external_sub, candidate_username)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                issuer = %issuer,
                external_sub = %external_sub,
                candidate = %candidate_username,
                error = %e,
                "Failed to resolve or bind external identity — rejecting login"
            );
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to provision federated identity",
            )
                .into_response());
        }
    };

    // Populate profile fields from external claims only for newly provisioned
    // accounts. Existing accounts retain their configured profile.
    if resolution.provisioned {
        let profile = UserProfilePatch {
            name: external_claims["name"].as_str().map(str::to_owned).map(Some),
            email: external_claims["email"].as_str().map(str::to_owned).map(Some),
            email_verified: external_claims["email_verified"].as_bool().map(Some),
            active: Some(Some(true)),
            external_id: external_claims["sub"].as_str().map(str::to_owned).map(Some),
            ..Default::default()
        };
        if let Err(e) = store.set_profile(&resolution.username, profile).await {
            tracing::error!(
                username = %resolution.username,
                error = %e,
                "Failed to set profile for newly provisioned federated user — rejecting login"
            );
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to provision federated user profile",
            )
                .into_response());
        }
    }

    // Write Casbin rules for default_scopes (best-effort — separate subsystem).
    // Scope format: "action:resource_type:resource_id" — e.g. "infer:model:*", "read:*:*"
    let subject = &resolution.username;
    if let Some(pm) = crate::auth::global_policy_manager() {
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
            subject = %subject,
            provisioned = resolution.provisioned,
            scopes = ?effective_scopes,
            "Federated user provisioned"
        );
    } else {
        tracing::warn!("PolicyManager not available — default_scopes not applied for {subject}");
    }

    Ok(resolution.username)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod token_response_tests {
    use super::check_token_response;
    use axum::{http::StatusCode, response::IntoResponse, routing::post, Router};
    use parking_lot::Mutex;
    use std::io::Write;
    use std::sync::Arc;
    use tracing::Level;

    #[derive(Clone, Default)]
    struct LogCapture(Arc<Mutex<Vec<u8>>>);

    impl Write for LogCapture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    async fn upstream_response(status: StatusCode, body: String) -> reqwest::Response {
        let app = Router::new().route("/token", post(move || async move { (status, body) }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock token endpoint");
        let address = listener.local_addr().expect("mock endpoint address");
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let response = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("test client")
            .post(format!("http://{address}/token"))
            .send()
            .await
            .expect("mock token response");
        server.abort();
        response
    }

    fn capture_check(
        response: reqwest::Response,
        level: Level,
    ) -> (Result<reqwest::Response, axum::response::Response>, String) {
        let capture = LogCapture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .without_time()
            .with_max_level(level)
            .with_writer(move || writer.clone())
            .finish();
        let result = tracing::subscriber::with_default(subscriber, || {
            check_token_response(response, "test-provider")
        })
        .map_err(IntoResponse::into_response);
        let logs = String::from_utf8(capture.0.lock().clone()).expect("UTF-8 structured logs");
        (result, logs)
    }

    #[tokio::test]
    async fn token_response_errors_log_only_public_metadata() {
        let bodies = [
            serde_json::json!({
                "access_token": "access-secret-canary",
                "refresh_token": "refresh-secret-canary",
                "id_token": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJzZWNyZXQifQ.signature-canary",
                "code": "authorization-code-canary",
                "client_secret": "client-secret-canary",
                "error": "provider-defined-secret-canary",
                "error_description": "Bearer diagnostic-secret-canary",
                "diagnostics": { "nested": "nested-secret-canary" }
            })
            .to_string(),
            "Bearer opaque-token-canary\nclient_secret=secret-canary&code=code-canary".repeat(1024),
            "".to_owned(),
        ];
        for status in [
            StatusCode::FOUND,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::SERVICE_UNAVAILABLE,
        ] {
            for body in &bodies {
                // TRACE additionally proves that diagnostics weren't merely
                // moved to a lower severity. Each normal threshold is covered.
                for level in [Level::ERROR, Level::WARN, Level::INFO, Level::TRACE] {
                    let upstream = upstream_response(status, body.clone()).await;
                    let (result, logs) = capture_check(upstream, level);
                    let rejection = result.expect_err("reject non-success response");
                    assert_eq!(rejection.status(), StatusCode::BAD_GATEWAY);
                    let rejection_body = axum::body::to_bytes(rejection.into_body(), 1024)
                        .await
                        .expect("read rejection body");
                    assert_eq!(&rejection_body[..], b"External token exchange rejected");
                    let events: Vec<serde_json::Value> = logs
                        .lines()
                        .map(|line| serde_json::from_str(line).expect("JSON log event"))
                        .collect();
                    assert_eq!(events.len(), 1, "one safe error event: {logs}");
                    assert_eq!(events[0]["level"], "ERROR");
                    // Exact field allowlist catches raw bodies, unknown error
                    // codes, nested credentials, and future diagnostic fields.
                    assert_eq!(
                        events[0]["fields"],
                        serde_json::json!({
                            "message": "External token endpoint returned error",
                            "provider": "test-provider",
                            "status": status.as_u16(),
                        })
                    );
                    assert!(!logs.contains("canary"), "secret in logs: {logs}");
                }
            }
        }
    }

    #[tokio::test]
    async fn token_response_success_preserves_body_without_logging() {
        let body = r#"{"access_token":"success-secret-canary"}"#;
        let upstream = upstream_response(StatusCode::OK, body.to_owned()).await;
        let (result, logs) = capture_check(upstream, Level::TRACE);
        let response = result.expect("successful response passes through");
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.text().await.expect("untouched token body"), body);
        assert!(logs.is_empty(), "successful token response logged: {logs}");
    }
}
