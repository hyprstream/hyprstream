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
use hyprstream_pds::repo_authority::is_path_form_did_web;
use hyprstream_rpc::auth::{JwkThumbprintInput, jwk_thumbprint};
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
    // private_key_jwt client auth (RFC 7521 §4.2, RFC 7523 §2.2)
    #[serde(default)]
    pub client_assertion: Option<String>,
    #[serde(default)]
    pub client_assertion_type: Option<String>,
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

    // Extract __Secure-VaultDevice cookie for silent device-user linking.
    let vault_device_cookie: Option<String> = req_headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies.split(';').find_map(|part| {
                let part = part.trim();
                part.strip_prefix("__Secure-VaultDevice=").map(str::to_owned)
            })
        });

    // Client authentication gate. For confidential clients (those that
    // declared token_endpoint_auth_method=private_key_jwt at
    // registration), verify the client_assertion JWT against the
    // client's JWKS BEFORE dispatching the grant. Public clients
    // (auth_method=none or unset) skip this — PKCE substitutes for
    // client auth per OAuth 2.1.
    //
    // On success this yields the RFC 7638 thumbprint of the assertion
    // key (#1146 T3.3) — the grant handlers persist it into the refresh
    // entry so the session stays bound to that exact key.
    let client_assertion_jkt = match enforce_client_authentication(&state, &params).await {
        Ok(jkt) => jkt,
        Err(resp) => return resp,
    };

    match params.grant_type.as_str() {
        "authorization_code" => exchange_authorization_code(state, params, dpop_header, vault_device_cookie, client_assertion_jkt).await,
        "refresh_token" => exchange_refresh_token(state, params, dpop_header, vault_device_cookie, client_assertion_jkt).await,
        gt if gt == DEVICE_CODE_GRANT_TYPE => exchange_device_code(state, params, dpop_header, vault_device_cookie, client_assertion_jkt).await,
        gt if gt == JWT_BEARER_GRANT_TYPE => {
            let assertion = match params.assertion {
                Some(a) => a,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("assertion is required")),
            };
            super::jwt_bearer::exchange_jwt_bearer(&state, &params.client_id, &assertion).await
        }
        gt if gt == TOKEN_EXCHANGE_GRANT_TYPE => {
            let subject_token = match params.subject_token {
                Some(t) => t,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("subject_token is required")),
            };
            let subject_token_type = match params.subject_token_type {
                Some(t) => t,
                None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("subject_token_type is required")),
            };
            // S6 (#572): the UCAN grant path. Routed on a dedicated
            // `subject_token_type` so it cannot be confused with the OIDC/WIT
            // exchange flows. Requires sender-binding (DPoP) — ZSP.
            if subject_token_type == crate::mac::exchange::UCAN_GRANT_TOKEN_TYPE {
                // #698 Decision D: resolves a delegated actor's clearance off the
                // signed policy's enrollment table, floored at Classical
                // assurance; falls back to DenyUnlabeledResolver (fail-closed) on
                // a node that hasn't installed a compiled policy.
                let resolver = crate::mac::exchange_enrollment_resolver();
                return super::token_exchange::exchange_ucan_grant(
                    &state,
                    &subject_token,
                    dpop_header.as_deref(),
                    params.scope.as_deref(),
                    params.audience.as_deref(),
                    resolver.as_ref(),
                )
                .await;
            }
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
            Some("Supported: authorization_code, refresh_token, device_code, jwt-bearer"),
        ),
    }
}

/// The immutable scope set of the grant being redeemed, used to derive
/// the profile-correct issuer for assertion-audience checking.
async fn bound_grant_scopes(
    state: &OAuthState,
    params: &TokenRequest,
) -> Option<Vec<String>> {
    match params.grant_type.as_str() {
        "authorization_code" => {
            let code = params.code.as_deref()?;
            let pending = state.pending_codes.read().await;
            let entry = pending.get(code)?;
            (entry.client_id == params.client_id).then(|| entry.scopes.clone())
        }
        "refresh_token" => {
            let refresh_token = params.refresh_token.as_deref()?;
            let entry = state.get_refresh_token(refresh_token).await.ok().flatten()?;
            (entry.client_id == params.client_id).then_some(entry.scopes)
        }
        gt if gt == DEVICE_CODE_GRANT_TYPE => {
            let device_code = params.device_code.as_deref()?;
            let pending = state.pending_device_codes.read().await;
            let entry = pending.get(device_code)?;
            (entry.client_id == params.client_id).then(|| entry.scopes.clone())
        }
        _ => None,
    }
}

/// The client-assertion key thumbprint bound to the grant being redeemed
/// (#1146 T3.3): PAR-bound for `authorization_code`, issuance-bound for
/// `refresh_token`. `None` when the grant carries no binding.
async fn bound_assertion_jkt(
    state: &OAuthState,
    params: &TokenRequest,
) -> Option<String> {
    match params.grant_type.as_str() {
        "authorization_code" => {
            let code = params.code.as_deref()?;
            let pending = state.pending_codes.read().await;
            let entry = pending.get(code)?;
            (entry.client_id == params.client_id)
                .then(|| entry.client_assertion_jkt.clone())
                .flatten()
        }
        "refresh_token" => {
            let refresh_token = params.refresh_token.as_deref()?;
            let entry = state.get_refresh_token(refresh_token).await.ok().flatten()?;
            (entry.client_id == params.client_id)
                .then_some(entry.client_assertion_jkt)
                .flatten()
        }
        _ => None,
    }
}

/// Enforce token endpoint client authentication.
///
/// Resolves the registered client (DCR or CIMD-cached), then:
///   - If the client requires `private_key_jwt`: `client_assertion` and
///     `client_assertion_type` MUST be present and verify successfully.
///   - If the client does NOT require it but an assertion was sent
///     anyway: we still verify it (best-effort, defense in depth).
///   - Unknown client_ids return invalid_client.
///
/// Returns the assertion key's RFC 7638 thumbprint when an assertion was
/// verified (`Ok(Some(jkt))`), `Ok(None)` when no assertion was presented;
/// the caller proceeds with the grant. Returns Err(Response) with the
/// appropriate token error response on failure.
async fn enforce_client_authentication(
    state: &OAuthState,
    params: &TokenRequest,
) -> Result<Option<String>, Response> {
    // CIMD client_ids are HTTPS URLs; DCR client_ids are UUIDs.
    //
    // For CIMD: on cache miss (entry expired between PAR/authorize
    // admission and now), we MUST re-resolve via resolve_cimd_client,
    // not fall through. Otherwise a client that declared
    // token_endpoint_auth_method=private_key_jwt could pass through
    // without an assertion check just because its cache entry aged
    // out — a security regression.
    //
    // resolve_cimd_client re-runs the federation:register policy check
    // and re-fetches metadata, so a revocation that happened during
    // the user's consent step is honored at token time.
    let client = if params.client_id.starts_with("https://") {
        match state.cimd_cache.get(&params.client_id).await {
            Some(c) => Some(c),
            None => super::registration::resolve_cimd_client(state, &params.client_id)
                .await
                .ok(),
        }
    } else {
        state.clients.read().await.get(&params.client_id).cloned()
    };

    let Some(client) = client else {
        // Genuinely unknown client_id. For grants that don't need a
        // RegisteredClient (jwt-bearer, token_exchange) the downstream
        // handler dispatches without it. For grants that DO need one
        // (authorization_code, refresh_token, device_code), the
        // downstream handler's own client_id check on the pending entry
        // will reject — but only if the IDs don't match. Permitting
        // pass-through here is safe because:
        //   - CIMD path above already attempts re-resolve, so a real
        //     CIMD client_id has had two chances to be found.
        //   - DCR clients with UUID IDs are only "absent" if they were
        //     never registered, in which case no pending entry exists.
        return Ok(None);
    };

    let needs_auth = super::client_auth::requires_private_key_jwt(&client);
    let has_assertion = params.client_assertion.is_some() && params.client_assertion_type.is_some();

    if needs_auth && !has_assertion {
        // Client-actionable: the client knows their own
        // token_endpoint_auth_method registration; we can tell them
        // what's missing from the request they sent.
        return Err(token_error(
            StatusCode::UNAUTHORIZED,
            "invalid_client",
            Some("client_assertion required"),
        ));
    }

    if has_assertion {
        let assertion = params.client_assertion.as_deref().unwrap_or_else(|| unreachable!());
        let atype = params.client_assertion_type.as_deref().unwrap_or_else(|| unreachable!());
        // Derive the assertion audience from the immutable grant, not from
        // caller-supplied scopes. atproto grants use the origin-only issuer
        // advertised in RFC 8414 metadata; generic grants preserve the
        // configured path-bearing issuer.
        //
        // #1146 T1.2: the accepted audience is the AS ISSUER — alone. The
        // atproto OAuth profile mandates the issuer; RFC 7523 §3 also
        // permits the token endpoint URL, but atproto does not, and
        // `private_key_jwt` client auth ships with this conformance chain,
        // so no deployed legacy client needs the endpoint form. Accepting
        // both would leave the endpoint URL valid as an assertion target
        // (replayable at any endpoint applying the same broadened rule).
        // PAR accepts the issuer only as well (`par.rs`).
        let issuer = match bound_grant_scopes(state, params).await {
            Some(scopes) => state.issuer_for_scopes(&scopes),
            None => state.issuer_url.clone(),
        };
        let expected_audiences = [issuer];
        match super::client_auth::verify_client_assertion(
            state, &client, atype, assertion, &expected_audiences,
        ).await {
            Ok(verified) => {
                // #1146 T3.3: when the grant was bound to a specific
                // assertion key (at PAR for authorization_code, at
                // issuance for refresh_token), the presented assertion
                // MUST verify under the same key — refresh cannot switch
                // to another registered key, and removing the bound key
                // from the client's JWKS revokes the session.
                if let Some(bound_jkt) = bound_assertion_jkt(state, params).await {
                    if bound_jkt != verified.key_jkt {
                        tracing::warn!(
                            client_id = %params.client_id,
                            "client_assertion key does not match the grant-bound key"
                        );
                        return Err(token_error(
                            StatusCode::UNAUTHORIZED,
                            "invalid_client",
                            None,
                        ));
                    }
                }
                return Ok(Some(verified.key_jkt));
            }
            Err(e) => {
                // Full reason goes to logs; the public response stays
                // opaque so we don't leak the validation logic / order
                // (iss/sub/aud/exp/signature) to probing attackers.
                tracing::warn!(
                    client_id = %params.client_id,
                    error = %e,
                    "client_assertion verification failed"
                );
                return Err(token_error(
                    StatusCode::UNAUTHORIZED,
                    "invalid_client",
                    None,
                ));
            }
        }
    }

    Ok(None)
}

/// Verify a DPoP proof for the token endpoint, check JTI replay, enforce
/// server-side nonces per RFC 9449 §8.
///
/// Returns `None` when no DPoP header is present (DPoP is optional at the
/// token endpoint).
/// Returns `Some(Ok(jkt))` on success; `Some(Err(response))` on failure.
///
/// Nonce enforcement policy:
/// - First proof from a given `jkt` (no prior nonce issuance recorded for
///   this key) is accepted as a bootstrap and a fresh nonce is issued in
///   the response `DPoP-Nonce` header.
/// - Subsequent proofs from the same `jkt` MUST include a server-issued
///   nonce; otherwise we reject with `error: "use_dpop_nonce"` and a fresh
///   nonce in the response header.
/// - Any presented nonce MUST be one we issued and not expired (5-min
///   sliding window).
async fn verify_dpop_at_token_endpoint(
    state: &OAuthState,
    dpop_header: Option<&str>,
    issuer: &str,
) -> Option<Result<String, Response>> {
    let proof_str = dpop_header?;
    let token_endpoint = format!("{}/oauth/token", issuer.trim_end_matches('/'));
    let proof = match super::dpop::verify_dpop_proof(proof_str, "POST", &token_endpoint, None) {
        Ok(p) => p,
        Err(e) => {
            // The DPoP module's error variants describe internal
            // checks (signature, jkt match, htm/htu, alg). Keep them
            // out of the public response; logs carry the detail.
            tracing::warn!("DPoP proof verification failed: {e}");
            return Some(Err(token_error(StatusCode::BAD_REQUEST, "invalid_dpop_proof", None)));
        }
    };
    // JTI replay check.
    if !state.check_and_record_dpop_jti(&proof.jti, proof.iat) {
        tracing::warn!(jti = %proof.jti, "DPoP JTI replay detected");
        return Some(Err(token_error(StatusCode::BAD_REQUEST, "invalid_dpop_proof", Some("DPoP proof jti already used"))));
    }

    // RFC 9449 §8 nonce enforcement.
    let client_needs_nonce = state.dpop_client_requires_nonce(&proof.jkt).await;
    match (client_needs_nonce, proof.nonce.as_deref()) {
        (true, None) => {
            // Subsequent request without nonce — reject with fresh nonce.
            let fresh = state.issue_dpop_nonce().await;
            state.mark_dpop_client_nonced(&proof.jkt).await;
            tracing::warn!(jkt = %proof.jkt, "DPoP nonce required but proof omitted it");
            return Some(Err(use_dpop_nonce_error(&fresh, "DPoP proof must include a server-issued nonce")));
        }
        (_, Some(presented)) => {
            // Whether bootstrap or subsequent: a presented nonce must be one
            // we issued. If invalid/expired, reject and rotate.
            if !state.verify_dpop_nonce(presented).await {
                let fresh = state.issue_dpop_nonce().await;
                state.mark_dpop_client_nonced(&proof.jkt).await;
                tracing::warn!(jkt = %proof.jkt, "DPoP nonce invalid or expired");
                return Some(Err(use_dpop_nonce_error(&fresh, "DPoP nonce invalid or expired")));
            }
        }
        (false, None) => {
            // Bootstrap: first request from this jkt with no nonce. Accept;
            // a fresh nonce will be issued on the success path (see
            // `issue_token_with_refresh`).
        }
    }

    Some(Ok(proof.jkt))
}

/// Build a `400 use_dpop_nonce` response with the current nonce in the
/// `DPoP-Nonce` header (RFC 9449 §8).
///
/// atproto contract (#1113): `@atproto/oauth-client-browser` retries a token
/// request on `400` + `{"error":"use_dpop_nonce"}` + a fresh `DPoP-Nonce`
/// header — the client re-signs the DPoP proof carrying the new nonce and
/// replays the request. This is the 400 form (not the resource-server 401
/// form used in `auth.rs`); both carry the same `DPoP-Nonce` header.
fn use_dpop_nonce_error(nonce: &str, description: &str) -> Response {
    // `description` here is always client-actionable ("nonce required",
    // "nonce expired") — that's the whole point of RFC 9449 §8: tell
    // the client to retry with the nonce we just issued in the header.
    let mut resp = token_error(StatusCode::BAD_REQUEST, "use_dpop_nonce", Some(description));
    if let Ok(val) = axum::http::HeaderValue::from_str(nonce) {
        resp.headers_mut().insert("DPoP-Nonce", val);
    }
    resp
}

/// Handle authorization_code grant type.
async fn exchange_authorization_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
    vault_device_cookie: Option<String>,
    client_assertion_jkt: Option<String>,
) -> Response {
    let code = match params.code {
        Some(c) => c,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("code is required")),
    };
    let redirect_uri = match params.redirect_uri {
        Some(r) => r,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("redirect_uri is required")),
    };
    let code_verifier = match params.code_verifier {
        Some(v) => v,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("code_verifier is required")),
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
                Some("Authorization code not found or already used"),
            );
        }
    };

    if pending.is_expired() {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("Authorization code has expired"),
        );
    }

    if params.client_id != pending.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("client_id does not match"),
        );
    }

    if redirect_uri != pending.redirect_uri {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("redirect_uri does not match"),
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
            Some("PKCE code_verifier verification failed"),
        );
    }

    // Verify DPoP if present.
    let token_issuer = state.issuer_for_scopes(&pending.scopes);
    let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref(), &token_issuer).await {
        None => None,
        Some(Ok(jkt)) => Some(jkt),
        Some(Err(resp)) => return resp,
    };

    // #1113 rev2 finding 3 + 6: the atproto profile (granted scope set
    // includes `atproto`) is a STRICT path — DPoP is mandatory and the proof
    // key MUST match the `jkt` bound at PAR (stored on the auth code). Non-
    // atproto flows keep DPoP optional and their existing behavior.
    if super::state::atproto_profile_active(&pending.scopes) {
        match (dpop_jkt.as_ref(), pending.dpop_jkt.as_ref()) {
            (Some(proof_jkt), Some(bound_jkt)) if proof_jkt == bound_jkt => {
                // DPoP key matches the PAR binding — proceed.
            }
            _ => {
                tracing::warn!(
                    client_id = %params.client_id,
                    proof_jkt = ?dpop_jkt,
                    bound_jkt = ?pending.dpop_jkt,
                    "atproto profile requires a DPoP proof matching the PAR-bound key"
                );
                return token_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_dpop_proof",
                    Some("atproto profile requires a DPoP proof from the key bound at PAR"),
                );
            }
        }
    }

    tracing::info!(client_id = %params.client_id, username = %pending.username, "PKCE verified, issuing token");
    let sub = pending.username.clone();
    let vk_ref = pending.verifying_key.as_ref();
    issue_token_with_refresh(&state, &params.client_id, pending.scopes, pending.resource, &sub, pending.oidc_nonce, true, vk_ref, dpop_jkt, client_assertion_jkt, vault_device_cookie).await
}

/// Handle refresh_token grant type (OAuth 2.1 with rotation).
async fn exchange_refresh_token(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
    vault_device_cookie: Option<String>,
    client_assertion_jkt: Option<String>,
) -> Response {
    let refresh_token = match params.refresh_token {
        Some(rt) => rt,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("refresh_token is required")),
    };

    let entry = match state.get_refresh_token(&refresh_token).await {
        Ok(Some(e)) => e,
        Ok(None) => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Refresh token not found or already used"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Refresh token store read failed");
            return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", None);
        }
    };

    // #1113 rev2 F3: validate client_id and DPoP BEFORE consuming the
    // single-use refresh token. Consuming first and failing second strands
    // the legitimate session on a missing/invalid proof.
    if params.client_id != entry.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("client_id does not match"),
        );
    }

    // MAC #547 / B1 (#673): a UCAN-grant refresh is re-evaluated through the S6
    // gate chain with a MANDATORY fresh DPoP proof — never this generic OAuth
    // rotation path (which treats DPoP as optional and does not re-check the
    // ceiling). Preserve its existing fail-closed, single-use behavior.
    if entry.ucan_grant.is_some() {
        let claimed = match state.take_refresh_token(&refresh_token).await {
            Ok(Some(claimed)) => claimed,
            Ok(None) => {
                return token_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_grant",
                    Some("Refresh token not found or already used"),
                );
            }
            Err(e) => {
                tracing::error!(error = %e, "Refresh token store take failed");
                return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", None);
            }
        };
        let Some(claimed_grant) = claimed.ucan_grant.as_ref() else {
            tracing::error!("UCAN refresh token changed before atomic claim");
            return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", None);
        };
        return super::token_exchange::exchange_ucan_grant_refresh(
            &state,
            claimed_grant,
            dpop_header.as_deref(),
        )
        .await;
    }

    // Verify DPoP before consuming the refresh token. A use_dpop_nonce
    // response must leave the credential available for the RFC 9449 retry.
    let token_issuer = state.issuer_for_scopes(&entry.scopes);
    let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref(), &token_issuer).await {
        None => None,
        Some(Ok(jkt)) => Some(jkt),
        Some(Err(resp)) => return resp,
    };

    // A sender-constrained token cannot be refreshed by another key or without
    // a proof.
    if !refresh_dpop_matches(entry.dpop_jkt.as_deref(), dpop_jkt.as_deref()) {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_dpop_proof",
            Some("DPoP proof must use the key bound to this refresh token"),
        );
    }

    // The atproto profile is always DPoP-mandatory and sender-bound, even if
    // a malformed legacy refresh entry somehow lacks a stored thumbprint.
    if super::state::atproto_profile_active(&entry.scopes) {
        match (&dpop_jkt, &entry.dpop_jkt) {
            (Some(proof_jkt), Some(bound_jkt)) if proof_jkt == bound_jkt => {}
            _ => {
                return token_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_dpop_proof",
                    Some("atproto profile requires a DPoP proof from the key bound at issuance"),
                );
            }
        }
    }

    // Atomically claim only after all retryable DPoP validation succeeds.
    // A successful claim prevents every other OAuth replica from minting with
    // this single-use refresh credential.
    let claimed = match state.take_refresh_token(&refresh_token).await {
        Ok(Some(claimed)) => claimed,
        Ok(None) => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Refresh token not found or already used"),
            );
        }
        Err(e) => {
            tracing::error!(error = %e, "Refresh token store take failed");
            return token_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", None);
        }
    };

    // Reconstruct verifying key from the atomically claimed record (cnf continuity across refreshes).
    let stored_vk: Option<ed25519_dalek::VerifyingKey> = claimed.verifying_key_bytes
        .and_then(|b| ed25519_dalek::VerifyingKey::from_bytes(&b).ok());

    // #1146 T3.3: carry the assertion-key binding forward across rotation.
    // enforce_client_authentication already verified the presented
    // assertion against this binding; for a legacy entry without one,
    // ratchet onto the key that just verified.
    let carried_assertion_jkt = claimed.client_assertion_jkt.clone().or(client_assertion_jkt);

    // Issue new access token + rotated refresh token. No id_token on refresh (OIDC Core § 12.2).
    issue_token_with_refresh(&state, &claimed.client_id, claimed.scopes, claimed.resource, &claimed.username, None, false, stored_vk.as_ref(), dpop_jkt, carried_assertion_jkt, vault_device_cookie).await
}

/// Handle urn:ietf:params:oauth:grant-type:device_code grant type (RFC 8628 Section 3.4).
async fn exchange_device_code(
    state: Arc<OAuthState>,
    params: TokenRequest,
    dpop_header: Option<String>,
    vault_device_cookie: Option<String>,
    client_assertion_jkt: Option<String>,
) -> Response {
    let device_code = match params.device_code {
        Some(dc) => dc,
        None => return token_error(StatusCode::BAD_REQUEST, "invalid_request", Some("device_code is required")),
    };

    let mut device_codes = state.pending_device_codes.write().await;

    let pending = match device_codes.get_mut(&device_code) {
        Some(p) => p,
        None => {
            return token_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                Some("Device code not found or already used"),
            );
        }
    };

    // Check expiration
    if pending.is_expired() {
        let user_code = pending.user_code.clone();
        device_codes.remove(&device_code);
        let mut user_code_map = state.device_code_by_user_code.write().await;
        user_code_map.remove(&user_code);
        return token_error(StatusCode::BAD_REQUEST, "expired_token", Some("The device code has expired"));
    }

    // Validate client_id
    if params.client_id != pending.client_id {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some("client_id does not match"),
        );
    }

    // Rate limiting: check poll interval
    let now = Instant::now();
    if let Some(last) = pending.last_polled {
        if now.duration_since(last).as_secs() < pending.interval {
            return token_error(StatusCode::BAD_REQUEST, "slow_down", Some("Polling too frequently"));
        }
    }
    match pending.status {
        DeviceCodeStatus::Pending => {
            pending.last_polled = Some(now);
            token_error(StatusCode::BAD_REQUEST, "authorization_pending", Some("The authorization request is still pending"))
        }
        DeviceCodeStatus::Denied => {
            let user_code = pending.user_code.clone();
            device_codes.remove(&device_code);
            let mut user_code_map = state.device_code_by_user_code.write().await;
            user_code_map.remove(&user_code);
            token_error(StatusCode::BAD_REQUEST, "access_denied", Some("The user denied the authorization request"))
        }
        DeviceCodeStatus::Approved => {
            let client_id = pending.client_id.clone();
            let scopes = pending.scopes.clone();
            let resource = pending.resource.clone();
            // Use the approving user's username as the JWT subject.
            // approved_by must be set when status is Approved; error defensively if missing.
            let approved_by = match pending.approved_by.clone() {
                Some(u) => u,
                None => {
                    // Internal invariant violation; do not surface
                    // server state shape to the polling client.
                    tracing::error!(
                        device_code_prefix = %&device_code[..8.min(device_code.len())],
                        "Device code approved but no approver identity recorded"
                    );
                    return token_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        None,
                    );
                }
            };
            let device_vk = pending.verifying_key;
            drop(device_codes);

            // Verify DPoP before consuming the device code. A
            // `use_dpop_nonce` response must leave it available for retry.
            let token_issuer = state.issuer_for_scopes(&scopes);
            let dpop_jkt = match verify_dpop_at_token_endpoint(&state, dpop_header.as_deref(), &token_issuer).await {
                None => None,
                Some(Ok(jkt)) => Some(jkt),
                Some(Err(resp)) => return resp,
            };

            // A PDS attachment client records the host's iroh did:key at
            // registration. Its token request must demonstrate possession of
            // exactly that Ed25519 key via DPoP before a credential is minted.
            let registered_node_did = {
                let clients = state.clients.read().await;
                clients
                    .get(&client_id)
                    .and_then(|client| client.hyprstream_node_did.clone())
            };
            if !registered_host_dpop_matches(
                registered_node_did.as_deref(),
                dpop_jkt.as_deref(),
            ) {
                tracing::warn!(%client_id, "PDS host DID and DPoP key do not match");
                return token_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_dpop_proof",
                    Some("DPoP proof must use the registered host key"),
                );
            }

            // Atomically claim the approved code only after validation. A
            // concurrent poller cannot mint a second token after this point.
            let user_code = {
                let mut device_codes = state.pending_device_codes.write().await;
                let Some(current) = device_codes.get(&device_code) else {
                    return token_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_grant",
                        Some("Device code not found or already used"),
                    );
                };
                if current.is_expired() {
                    let user_code = current.user_code.clone();
                    device_codes.remove(&device_code);
                    drop(device_codes);
                    let mut user_code_map = state.device_code_by_user_code.write().await;
                    user_code_map.remove(&user_code);
                    return token_error(
                        StatusCode::BAD_REQUEST,
                        "expired_token",
                        Some("The device code has expired"),
                    );
                }
                if current.status != DeviceCodeStatus::Approved {
                    return token_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_grant",
                        Some("Device code not found or already used"),
                    );
                }
                let user_code = current.user_code.clone();
                device_codes.remove(&device_code);
                user_code
            };
            let mut user_code_map = state.device_code_by_user_code.write().await;
            user_code_map.remove(&user_code);
            drop(user_code_map);

            // Device flow: no OIDC nonce and not initial OIDC auth.
            issue_token_with_refresh(&state, &client_id, scopes, resource, &approved_by, None, false, device_vk.as_ref(), dpop_jkt, client_assertion_jkt, vault_device_cookie).await
        }
    }
}

fn registered_host_dpop_matches(registered_node_did: Option<&str>, dpop_jkt: Option<&str>) -> bool {
    let Some(node_did) = registered_node_did else {
        return true;
    };
    let Ok(node_key) = hyprstream_crypto::did_key::did_key_to_ed25519(node_did) else {
        return false;
    };
    let expected_jkt = jwk_thumbprint(&JwkThumbprintInput::Ed25519 { x: &node_key });
    dpop_jkt == Some(expected_jkt.as_str())
}

fn refresh_dpop_matches(expected_jkt: Option<&str>, presented_jkt: Option<&str>) -> bool {
    expected_jkt.is_none() || expected_jkt == presented_jkt
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
/// `sub` is the internal account username used for server-side bookkeeping
/// (refresh-token entry, profile lookup, device link). The emitted JWT's
/// `sub` claim is the atproto-conformant DID derived from it via
/// [`OAuthState::subject_did`] (#1113). Must be non-empty:
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
    client_assertion_jkt: Option<String>,
    vault_device_cookie: Option<String>,
) -> Response {
    // ── #1159 freeze: path-form account subject guard ───────────────────────
    // This is the CENTRAL mint boundary: the authorization_code, refresh_token,
    // and device_code flows all funnel through here for both the access-token
    // `subject` and the persisted rotated `RefreshTokenEntry.username`. Blocking
    // `UserMappingStrategy::DidWeb` at config time stops NEW OIDC callbacks from
    // constructing a path-form subject, but it does not touch subjects already
    // durably persisted in refresh-token (and Valkey user-profile) stores. A
    // pre-upgrade refresh token carries `did:web:{authority}:users:{name}` as its
    // `username`; without this guard, refresh would mint a fresh access token for
    // that subject AND persist a newly rotated ~30-day refresh token carrying it,
    // keeping the path-form chain alive indefinitely.
    //
    // Fail-closed here bounds the hole: a stored path-form subject can no longer
    // be minted or rotated, so each pre-upgrade refresh token dies at its next
    // refresh attempt or at natural expiry — it cannot extend the lifetime. The
    // accounts themselves are reminted to host-form by E4 (#1176), not by this
    // freeze; this PR does NOT scan or revoke stored records at startup, because
    // that durable scan is exactly the out-of-band work #1176 owns and a partial
    // scan here would risk the data migration the freeze deliberately defers.
    // The predicate is the shared `is_path_form_did_web` so host-form
    // `did:web:alice.example.com`, plain usernames, and every non-did:web subject
    // pass through unchanged.
    if is_path_form_did_web(sub) {
        return token_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            Some(
                "token subject is a frozen path-form did:web account identifier; \
                 host-form account minting is not available yet (#1159)",
            ),
        );
    }

    let scope_str = scopes.join(" ");
    let atproto_profile = super::state::atproto_profile_active(&scopes);

    // DPoP jkt takes priority; fall back to raw key bytes for cnf.jwk.
    let user_pub_key_b64 = if dpop_jkt.is_none() {
        user_verifying_key.map(|vk| URL_SAFE_NO_PAD.encode(vk.to_bytes()))
    } else {
        None
    };

    // #1113 rev2 finding 6 / r4 finding 1: the DID subject / account-
    // eligibility check is CONDITIONAL on the atproto profile. Non-atproto
    // flows (device, WIT, generic authorization-code) keep their existing
    // username subject byte-for-byte — the atproto strict path only activates
    // when the granted scope set includes `atproto`.
    //
    // atproto profile: the access token's `sub` MUST be the account's real
    // atproto DID. The eligibility check inspects the ACCOUNT/KEY MAPPING
    // (not the subject string): at9p-backed accounts are rejected; an account
    // with a mapped atproto DID gets that DID as `sub`; an account without a
    // mapped DID fails closed (#1124). The returned DID is already form-
    // validated by evaluate_atproto_eligibility → subject_did_for.
    let jwt_sub = if atproto_profile {
        match state.check_atproto_account_eligibility(sub).await {
            Ok(mapped_did) => mapped_did,
            Err(e) => {
                tracing::warn!(local_subject = %sub, error = %e, "rejecting atproto token issuance: account not eligible");
                return token_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    Some("account has no atproto identity; provisioning tracked in #1124"),
                );
            }
        }
    } else {
        sub.to_owned()
    };
    let token_issuer = state.issuer_for_scopes(&scopes);

    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes: Some(scopes.clone()),
            ttl: Some(state.token_ttl),
            audience: resource.clone(),
            subject: Some(jwt_sub.clone()),
            user_pub_key: user_pub_key_b64,
            dpop_jkt: dpop_jkt.clone(),
            issuer: Some(token_issuer.clone()),
        })
        .await;

    match result {
        Ok(token_info) => {
            tracing::info!(client_id = %client_id, "Token issued successfully");

            // Silently link device → user when __Secure-VaultDevice cookie accompanies the request.
            if let (Some(cookie_val), Some(ref ds), Some(ref sk)) =
                (&vault_device_cookie, &state.device_store, &state.signing_key)
            {
                let sk_bytes = sk.to_bytes();
                if let Some(pubkey) =
                    crate::auth::device_challenge::verify_vault_device_cookie(&sk_bytes, cookie_val)
                {
                    let fingerprint = bs58::encode(&pubkey).into_string();
                    let ds = ds.clone();
                    let sub_owned = sub.to_owned();
                    tokio::spawn(async move {
                        if let Err(e) = ds.link_device_user(&fingerprint, &sub_owned).await {
                            tracing::warn!(fingerprint = %fingerprint, error = %e, "device-user link failed");
                        } else {
                            tracing::debug!(fingerprint = %fingerprint, sub = %sub_owned, "device linked to user");
                        }
                    });
                }
            }

            let now = chrono::Utc::now().timestamp();
            let expires_in = (token_info.expires_at - now).max(0);
            // Issue a fresh DPoP nonce when the client used DPoP (RFC 9449 §8).
            // Also record that this jkt has now been issued a nonce so future
            // proofs are required to carry one.
            let dpop_nonce = if let Some(ref jkt) = dpop_jkt {
                let n = state.issue_dpop_nonce().await;
                state.mark_dpop_client_nonced(jkt).await;
                Some(n)
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
                    dpop_jkt: dpop_jkt.clone(),
                    client_assertion_jkt: client_assertion_jkt.clone(),
                    ucan_grant: None, // generic OAuth refresh; not a UCAN grant (MAC #547 B1)
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
                    token_issuer,
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

            let mut response_json = build_token_response_json(
                atproto_profile,
                &token_info.token,
                &jwt_sub,
                &scope_str,
                expires_in,
                &refresh_token,
            );
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
                None,
            )
        }
    }
}

/// Build the token-endpoint success response JSON (#1113 rev2 finding 1).
///
/// Factored pure so the atproto `atprotoOAuthTokenResponseSchema` shape and the
/// legacy Bearer shape are both unit-testable without constructing an
/// `OAuthState` + PolicyClient. When `atproto_profile` is true the response
/// carries `token_type: "DPoP"` and a top-level `sub` (the account DID), per
/// the schema the stock `@atproto/oauth-client-browser` parses.
pub(crate) fn build_token_response_json(
    atproto_profile: bool,
    access_token: &str,
    sub: &str,
    scope_str: &str,
    expires_in: i64,
    refresh_token: &str,
) -> serde_json::Value {
    if atproto_profile {
        serde_json::json!({
            "access_token": access_token,
            "token_type": "DPoP",
            "expires_in": expires_in,
            "scope": scope_str,
            "sub": sub,
            "refresh_token": refresh_token,
        })
    } else {
        serde_json::json!({
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": scope_str,
            "refresh_token": refresh_token,
        })
    }
}

/// Build an OAuth 2.1 §5.3 token-endpoint error response.
///
/// `description` is `Some(...)` only when the message is **client-
/// actionable** (refers to something the client themselves can fix in
/// their request — missing field, wrong assertion type, redirect_uri
/// mismatch). For server-internal failures (policy denials, dependency
/// outages, claim mismatches, signature failures) pass `None`: the
/// public response carries only the OAuth error code, and the caller
/// logs the full reason via tracing for operators.
///
/// Rationale: returning details like "PolicyService unreachable" or
/// "iss does not match client_id" leaks internal IAM topology and
/// validation order to unauthenticated callers. OAuth 2.1 §5.3
/// permits `error_description` to be omitted; standard practice
/// among hardened IdPs is to omit it for security-sensitive failures.
fn token_error(status: StatusCode, error: &str, description: Option<&str>) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(token_error_body(error, description)),
    ).into_response()
}

/// Pure JSON-body construction for `token_error`. Split out so unit
/// tests can assert on the body shape (notably: `error_description`
/// is omitted entirely when `description` is `None`, not serialized
/// as `null`).
fn token_error_body(error: &str, description: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert("error".to_owned(), serde_json::Value::String(error.to_owned()));
    if let Some(d) = description {
        body.insert(
            "error_description".to_owned(),
            serde_json::Value::String(d.to_owned()),
        );
    }
    serde_json::Value::Object(body)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn token_error_body_omits_description_when_none() {
        let body = token_error_body("invalid_client", None);
        let obj = body.as_object().unwrap();
        assert_eq!(obj.get("error").and_then(|v| v.as_str()), Some("invalid_client"));
        assert!(
            !obj.contains_key("error_description"),
            "description field MUST be absent when None — leaks happen via null/empty too"
        );
        assert_eq!(obj.len(), 1, "no extra fields");
    }

    #[test]
    fn token_error_body_includes_description_when_some() {
        let body = token_error_body("invalid_request", Some("code is required"));
        let obj = body.as_object().unwrap();
        assert_eq!(obj.get("error_description").and_then(|v| v.as_str()), Some("code is required"));
    }

    #[test]
    fn registered_pds_host_requires_its_dpop_key() {
        let key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let did = hyprstream_crypto::did_key::ed25519_to_did_key(key.verifying_key().as_bytes());
        let matching_jkt = jwk_thumbprint(&JwkThumbprintInput::Ed25519 {
            x: key.verifying_key().as_bytes(),
        });

        assert!(registered_host_dpop_matches(
            Some(&did),
            Some(&matching_jkt),
        ));
        assert!(!registered_host_dpop_matches(Some(&did), None));
        assert!(!registered_host_dpop_matches(Some(&did), Some("other-key")));
        assert!(!registered_host_dpop_matches(Some("did:key:zinvalid"), Some(&matching_jkt)));
        assert!(registered_host_dpop_matches(None, None));
    }

    #[test]
    fn dpop_bound_refresh_requires_the_original_key() {
        assert!(refresh_dpop_matches(None, None));
        assert!(refresh_dpop_matches(None, Some("new-key")));
        assert!(refresh_dpop_matches(Some("bound-key"), Some("bound-key")));
        assert!(!refresh_dpop_matches(Some("bound-key"), None));
        assert!(!refresh_dpop_matches(Some("bound-key"), Some("other-key")));
    }

    #[test]
    fn token_error_body_never_carries_internal_substrings() {
        // Regression guard: no caller in token.rs should ever pass a
        // description containing these internal-state markers. The
        // sweep in 2026-05-26 stripped them all to None. If a future
        // change adds one back, this test won't catch it directly —
        // but the documented rule for token_error makes it a code-
        // review item. The test here pins the API: `None` yields an
        // empty object, so any leak would have to be explicit in a
        // caller's `Some(...)`.
        let body = token_error_body("invalid_client", None);
        let serialized = body.to_string();
        for forbidden in &[
            "PolicyService",
            "federation:register",
            "Failed to fetch",
            "iss does not match",
            "Token store error",
        ] {
            assert!(
                !serialized.contains(forbidden),
                "opaque response unexpectedly leaks `{forbidden}`: {serialized}"
            );
        }
    }

    /// #1113: the DPoP nonce-retry contract that
    /// `@atproto/oauth-client-browser` relies on — HTTP 400 with
    /// `error: use_dpop_nonce` AND a `DPoP-Nonce` response header so the
    /// client can re-sign and replay the proof.
    #[test]
    fn use_dpop_nonce_error_carries_atproto_retry_contract() {
        // Draw the test nonce from OsRng exactly as production
        // (`OAuthState::issue_dpop_nonce`) does. A hard-coded literal here
        // tripped CodeQL `rust/hard-coded-cryptographic-value` (#1121) — a
        // false positive (test-only value, never a crypto sink), but the
        // literal-free form is also a more faithful test of the real shape.
        use rand::Rng;
        let nonce = URL_SAFE_NO_PAD.encode(rand::rngs::OsRng.gen::<[u8; 16]>());
        let resp = use_dpop_nonce_error(&nonce, "DPoP proof must include a server-issued nonce");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            resp.headers().get("DPoP-Nonce").and_then(|v| v.to_str().ok()),
            Some(nonce.as_str()),
            "DPoP-Nonce header MUST carry the fresh nonce for client retry"
        );
    }

    /// #1113: a nonce value that is not a valid HTTP header token still
    /// yields a 400 `use_dpop_nonce` body — the nonce header is omitted
    /// (graceful) but the error contract the client keys on is preserved.
    #[test]
    fn use_dpop_nonce_error_keeps_400_on_unheaderable_nonce() {
        // Map OsRng bytes into the C0 control range (0x00–0x1F): every byte
        // is then invalid in an HTTP header value, so
        // `HeaderValue::from_str` MUST reject it and the header is omitted.
        // (The previous "not valid token" literal contained only spaces,
        // which ARE valid header bytes — the header was silently inserted
        // and the graceful-omission branch went untested. The literal also
        // tripped CodeQL `rust/hard-coded-cryptographic-value`, #1121.)
        use rand::Rng;
        let bytes: Vec<u8> = rand::rngs::OsRng
            .gen::<[u8; 8]>()
            .into_iter()
            .map(|b| b % 0x20)
            .collect();
        let nonce = String::from_utf8(bytes).expect("C0 control bytes are valid UTF-8");
        let resp = use_dpop_nonce_error(&nonce, "expired");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            resp.headers().get("DPoP-Nonce").is_none(),
            "unheaderable nonce MUST be omitted from the response"
        );
    }

    /// #1113 rev2 finding 1: the atproto token response matches
    /// atprotoOAuthTokenResponseSchema — `token_type: "DPoP"`, a top-level
    /// `sub` that is a valid atproto DID (did:plc or host-form did:web), and
    /// a scope containing `atproto`. Uses a valid `did:plc` subject (the
    /// spec-valid form; path-form did:web is rejected by @atproto/did — #1124).
    #[test]
    fn atproto_token_response_matches_schema_shape() {
        let json = build_token_response_json(
            true,
            "access-jwt",
            "did:plc:abcdefghijklmnqrstuvwx2p",
            "atproto transition:generic",
            3600,
            "rt",
        );
        assert_eq!(json["token_type"].as_str(), Some("DPoP"));
        let sub = json["sub"].as_str().expect("top-level sub required");
        // Valid atproto DID: did:plc (or host-form did:web with NO path).
        assert!(
            sub.starts_with("did:plc:") || {
                if let Some(rest) = sub.strip_prefix("did:web:") {
                    rest.split(':').count() <= 2 && !rest.contains('/')
                } else {
                    false
                }
            },
            "atproto sub must be did:plc or host-form did:web (no path), got {sub}"
        );
        let scope = json["scope"].as_str().expect("scope required");
        assert!(scope.split_whitespace().any(|s| s == "atproto"), "scope missing atproto");
        assert_eq!(json["access_token"].as_str(), Some("access-jwt"));
        assert_eq!(json["refresh_token"].as_str(), Some("rt"));
    }

    /// #1113 rev2 finding 6 (regression): a non-atproto grant (e.g. device
    /// flow, scopes without `atproto`) keeps the legacy Bearer shape with NO
    /// top-level `sub` — existing principals are byte-for-byte unchanged.
    #[test]
    fn non_atproto_token_response_is_bearer_without_sub() {
        let json = build_token_response_json(
            false,
            "access-jwt",
            "alice",
            "read:*:*",
            3600,
            "rt",
        );
        assert_eq!(json["token_type"].as_str(), Some("Bearer"));
        assert!(
            json.get("sub").is_none(),
            "non-atproto response MUST NOT carry a top-level sub"
        );
    }

    /// #1113 rev2 finding 3: PAR↔token DPoP binding — a proof whose `jkt`
    /// does not match the PAR-bound thumbprint must be rejected. The binding
    /// check is a plain equality compare; pin the comparator semantics.
    #[test]
    fn dpop_jkt_binding_rejects_mismatched_key() {
        let bound: Option<String> = Some("jkt-A".into());
        let proof: Option<String> = Some("jkt-B".into());
        // The token endpoint accepts only when both are present AND equal.
        let matches = matches!(
            (proof.as_ref(), bound.as_ref()),
            (Some(a), Some(b)) if a == b
        );
        assert!(!matches, "mismatched jkt must not bind");
        let bound2: Option<String> = Some("jkt-A".into());
        let matches2 = matches!(
            (proof.as_ref(), bound2.as_ref()),
            (Some(a), Some(b)) if a == b
        );
        assert!(!matches2);
        let proof_same: Option<String> = Some("jkt-A".into());
        let matches3 = matches!(
            (proof_same.as_ref(), bound.as_ref()),
            (Some(a), Some(b)) if a == b
        );
        assert!(matches3, "equal jkt must bind");
    }

    // ── #1159 freeze: pre-upgrade path-form refresh token must not rotate ──

    /// A `TokenStore` that counts `put` calls (rotation writes) and otherwise
    /// behaves like a small in-memory map. Used to prove a path-form refresh
    /// fails *without* persisting a rotated token.
    struct RecordingTokenStore {
        inner: parking_lot::Mutex<std::collections::HashMap<String, RefreshTokenEntry>>,
        puts: std::sync::atomic::AtomicUsize,
    }

    impl RecordingTokenStore {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: parking_lot::Mutex::new(std::collections::HashMap::new()),
                puts: std::sync::atomic::AtomicUsize::new(0),
            })
        }
        fn put_count(&self) -> usize {
            self.puts.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl crate::services::oauth::token_store::TokenStore for RecordingTokenStore {
        async fn put(&self, token: &str, entry: &RefreshTokenEntry, _ttl: u64) -> anyhow::Result<()> {
            self.puts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.lock().insert(token.to_owned(), entry.clone());
            Ok(())
        }
        async fn get(&self, token: &str) -> anyhow::Result<Option<RefreshTokenEntry>> {
            Ok(self.inner.lock().get(token).cloned())
        }
        async fn take(&self, token: &str) -> anyhow::Result<Option<RefreshTokenEntry>> {
            Ok(self.inner.lock().remove(token))
        }
        async fn delete(&self, token: &str) -> anyhow::Result<()> {
            self.inner.lock().remove(token);
            Ok(())
        }
    }

    /// Build a minimal `OAuthState` over dummy RPC clients (LazyUdsTransport at
    /// /dev/null — never opened) and a recording refresh-token store.
    async fn freeze_test_state(store: Arc<RecordingTokenStore>) -> Arc<OAuthState> {
        use crate::config::OAuthConfig;
        use crate::services::{DiscoveryClient, PolicyClient};
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x76; 32]);
        let vk = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]).verifying_key();
        let dummy = std::path::PathBuf::from("/dev/null/freeze-test.sock");
        let mk_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(key.clone()),
                    LazyUdsTransport::new(dummy.clone()),
                    Some(vk),
                )
                .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical),
            )
        };
        let mut state = OAuthState::new(
            &OAuthConfig::default(),
            PolicyClient::new(mk_client()),
            DiscoveryClient::new(mk_client()),
            [0x76; 32],
        );
        state.with_token_store_impl(store as Arc<dyn crate::services::oauth::token_store::TokenStore>);
        Arc::new(state)
    }

    /// The central mint guard: a path-form subject is rejected with
    /// `invalid_grant` and no rotated refresh token is persisted.
    #[tokio::test]
    async fn path_form_subject_rejected_at_mint_boundary_without_rotation() {
        let store = RecordingTokenStore::new();
        let state = freeze_test_state(Arc::clone(&store)).await;

        let resp = issue_token_with_refresh(
            &state,
            "client-1",
            vec!["openid".to_owned()],
            None,
            "did:web:accounts.example:users:alice", // path-form — frozen under #1159
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"].as_str(), Some("invalid_grant"));
        assert!(
            v["error_description"]
                .as_str()
                .unwrap()
                .contains("frozen path-form did:web"),
            "body was: {v}",
        );
        assert_eq!(
            store.put_count(),
            0,
            "guard must return before persisting a rotated refresh token",
        );
    }

    /// End-to-end refresh regression: a pre-upgrade refresh token whose stored
    /// subject is path-form is consumed (taken) on refresh but NOT rotated —
    /// no access token is minted and no replacement refresh token is written,
    /// so the chain dies here rather than rotating indefinitely.
    #[tokio::test]
    async fn pre_upgrade_path_form_refresh_token_is_consumed_not_rotated() {
        let store = RecordingTokenStore::new();
        let state = freeze_test_state(Arc::clone(&store)).await;

        // Seed a pre-upgrade refresh token carrying a path-form subject, exactly
        // as a `didweb`-configured deployment would have persisted before #1159.
        let path_form_entry = RefreshTokenEntry {
            client_id: "client-1".to_owned(),
            username: "did:web:accounts.example:users:alice".to_owned(),
            scopes: vec!["openid".to_owned()],
            resource: None,
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
            verifying_key_bytes: None,
            dpop_jkt: None,
            client_assertion_jkt: None,
            ucan_grant: None,
        };
        state
            .put_refresh_token("legacy-refresh", &path_form_entry, 3600)
            .await
            .unwrap();
        assert_eq!(store.put_count(), 1, "seeding writes one entry");

        let params = TokenRequest {
            grant_type: "refresh_token".to_owned(),
            refresh_token: Some("legacy-refresh".to_owned()),
            client_id: "client-1".to_owned(),
            code: None,
            redirect_uri: None,
            code_verifier: None,
            device_code: None,
            assertion: None,
            client_assertion: None,
            client_assertion_type: None,
            subject_token: None,
            subject_token_type: None,
            requested_token_type: None,
            actor_token: None,
            scope: None,
            audience: None,
        };
        let resp = exchange_refresh_token(Arc::clone(&state), params, None, None, None).await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"].as_str(), Some("invalid_grant"));
        assert!(v["error_description"].as_str().unwrap().contains("frozen"));

        // No rotated refresh token was persisted (put_count still equals the
        // single seed write), and the legacy token is gone (taken on refresh) —
        // the chain is dead, not extended.
        assert_eq!(
            store.put_count(),
            1,
            "refresh must not persist a rotated refresh token for a path-form subject",
        );
        assert!(
            state.get_refresh_token("legacy-refresh").await.unwrap().is_none(),
            "the path-form refresh token must be consumed (taken), not left reusable",
        );
    }
}
