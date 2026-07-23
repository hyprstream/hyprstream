//! Pushed Authorization Requests (RFC 9126).
//!
//! POST /oauth/par lets the client push the authorization request parameters
//! out-of-band, then navigate the browser to `/oauth/authorize?request_uri=...`
//! with a short, opaque reference. Required by atproto OAuth; also avoids URL
//! length and referer-leakage problems with long inline parameters.
//!
//! Flow:
//! 1. Client POSTs the same parameters it would pass to GET /oauth/authorize.
//! 2. Server validates them up front (same checks the GET handler does:
//!    response_type, PKCE method, client metadata resolution, redirect_uri).
//! 3. On success, the server generates a URN of the form
//!    `urn:ietf:params:oauth:request_uri:<random-id>` (RFC 9126 §2.2,
//!    RFC-standard form used by atproto OAuth clients) and stores the
//!    validated `AuthorizeParams` snapshot keyed by it. 60s TTL.
//! 4. Returns `{ "request_uri": "...", "expires_in": 60 }` with HTTP 201.
//!
//! The /oauth/authorize GET handler resolves `request_uri` by consuming the
//! entry (single-use per spec) and proceeding as if the params had been inline.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Form, Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::authorize::AuthorizeParams;
use super::registration::{resolve_cimd_client, validate_redirect_uri};
use super::state::{OAuthState, PushedAuthRequest};

/// PAR request lifetime. RFC 9126 recommends short-lived (e.g. <= 60s).
pub const PAR_TTL_SECS: u64 = 60;

/// PAR form body. Same shape as `AuthorizeParams`, but accepted as a form POST.
///
/// Per RFC 9126 §2.1 the body is `application/x-www-form-urlencoded` with the
/// same parameters that would appear on the authorize URL.
#[derive(Debug, Deserialize)]
pub struct ParForm {
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub response_type: String,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub resource: Option<String>,
    #[serde(default)]
    pub nonce: Option<String>,
    /// RFC 7523 client assertion — REQUIRED at PAR for confidential
    /// (`private_key_jwt`) clients: the atproto OAuth profile authenticates
    /// confidential clients during the authorization request, which for a
    /// PAR-mandatory profile is here (#1146 T3.3).
    #[serde(default)]
    pub client_assertion: Option<String>,
    #[serde(default)]
    pub client_assertion_type: Option<String>,
}

/// PAR success response body (RFC 9126 §2.2).
#[derive(Debug, Serialize)]
pub struct ParResponse {
    pub request_uri: String,
    pub expires_in: u64,
}

/// PAR error body (RFC 9126 §2.3 — same shape as OAuth token errors).
#[derive(Debug, Serialize)]
struct ParError {
    error: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_description: Option<String>,
}

/// Build a PAR error response. `description` is `Some(...)` only when
/// the message is client-actionable (refers to the client's own
/// request). Server-internal failures pass `None`; the public response
/// then carries only the OAuth error code. See `token_error` for the
/// same rationale at the token endpoint.
fn par_error(status: StatusCode, error: &'static str, description: Option<&str>) -> Response {
    (
        status,
        Json(ParError {
            error,
            error_description: description.map(str::to_owned),
        }),
    )
        .into_response()
}

/// POST /oauth/par
///
/// Validates the pushed authorization request and returns a `request_uri` the
/// client should navigate the browser to via `/oauth/authorize?request_uri=...`.
pub async fn push_authorization_request(
    State(state): State<Arc<OAuthState>>,
    req_headers: HeaderMap,
    Form(form): Form<ParForm>,
) -> Response {
    // Validate response_type
    if form.response_type != "code" {
        return par_error(
            StatusCode::BAD_REQUEST,
            "unsupported_response_type",
            Some("Only response_type=code is supported"),
        );
    }

    // Validate PKCE method
    if form.code_challenge_method != "S256" {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("Only code_challenge_method=S256 is supported"),
        );
    }
    if form.code_challenge.is_empty() {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            Some("code_challenge is required (PKCE mandatory per OAuth 2.1)"),
        );
    }

    // Resolve client (CIMD or dynamically registered) and pull redirect URIs.
    let registered_client = if form.client_id.starts_with("https://") {
        match resolve_cimd_client(&state, &form.client_id).await {
            Ok(client) => Some(client),
            Err(e) => {
                // The resolver's error spans federation:register policy
                // denial, PolicyService RPC outage, and CIMD doc fetch
                // failures — all of which describe internal trust
                // state to the unauthenticated PAR caller. Log full
                // detail; respond opaquely.
                tracing::warn!(
                    client_id = %form.client_id,
                    error = %e,
                    "CIMD resolution failed during PAR"
                );
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    None,
                );
            }
        }
    } else {
        let clients = state.clients.read().await;
        match clients.get(&form.client_id) {
            Some(client) => Some(client.clone()),
            None => {
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    Some("Unknown client_id. Register first via POST /oauth/register"),
                );
            }
        }
    };

    let registered_client = match registered_client {
        Some(c) => c,
        None => {
            return par_error(
                StatusCode::BAD_REQUEST,
                "invalid_client",
                None,
            );
        }
    };

    if !validate_redirect_uri(&form.redirect_uri, &registered_client.redirect_uris) {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_redirect_uri",
            Some("redirect_uri does not match registered URIs"),
        );
    }

    // #1113 rev2 finding 4: validate requested scope against server-supported
    // ∩ client-declared. Garbage / undeclared tokens → invalid_scope.
    let requested_scopes: Vec<String> = form
        .scope
        .as_deref()
        .unwrap_or(&state.default_scopes.join(" "))
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let client_declared = registered_client.declared_scopes();
    let require_atproto = super::state::atproto_profile_active(&requested_scopes);
    let granted_scopes = match super::state::validate_requested_scopes(
        &requested_scopes,
        &state.server_supported_scopes(),
        if client_declared.is_empty() { None } else { Some(&client_declared) },
        require_atproto,
    ) {
        Ok(g) => g,
        Err(_) => {
            return par_error(
                StatusCode::BAD_REQUEST,
                "invalid_scope",
                Some("Requested scope is not supported or not declared by the client"),
            );
        }
    };
    // `atproto` activates the strict profile — echo back the granted scope
    // string into the stored snapshot so authorize/token see the exact set.
    let granted_scope_str = granted_scopes.join(" ");

    // #1146 T3.3 (+T1.2): authenticate confidential clients at PAR. The
    // atproto OAuth profile authenticates confidential clients during the
    // authorization request — which, for a PAR-mandatory profile, is here.
    // The assertion audience is the AS ISSUER (atproto mandate; the token
    // endpoint additionally accepts the RFC 7523 endpoint-URL form, PAR
    // does not). The verified key's RFC 7638 thumbprint is bound into the
    // snapshot so token redemption and refresh must use the SAME key.
    let client_assertion_jkt = {
        let needs_auth = super::client_auth::requires_private_key_jwt(&registered_client);
        let has_assertion =
            form.client_assertion.is_some() && form.client_assertion_type.is_some();
        if needs_auth && !has_assertion {
            return par_error(
                StatusCode::UNAUTHORIZED,
                "invalid_client",
                Some("client_assertion required"),
            );
        }
        if has_assertion {
            let assertion = form.client_assertion.as_deref().unwrap_or_else(|| unreachable!());
            let atype = form
                .client_assertion_type
                .as_deref()
                .unwrap_or_else(|| unreachable!());
            let issuer = state.issuer_for_scopes(&granted_scopes);
            match super::client_auth::verify_client_assertion(
                &state,
                &registered_client,
                atype,
                assertion,
                std::slice::from_ref(&issuer),
            )
            .await
            {
                Ok(verified) => Some(verified.key_jkt),
                Err(e) => {
                    // Opaque response, detailed log — same rationale as
                    // the token endpoint's invalid_client handling.
                    tracing::warn!(
                        client_id = %form.client_id,
                        error = %e,
                        "client_assertion verification failed at PAR"
                    );
                    return par_error(
                        StatusCode::UNAUTHORIZED,
                        "invalid_client",
                        None,
                    );
                }
            }
        } else {
            None
        }
    };

    // #1113 rev2 finding 3: atproto profile requires DPoP. Capture and verify
    // the DPoP proof at PAR, bind its `jkt` into the authorization request so
    // the token endpoint can require a proof from the SAME key. Non-atproto
    // requests keep DPoP optional (RFC 9449 general behavior).
    let dpop_jkt = if require_atproto {
        let dpop_header = req_headers
            .get("DPoP")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let dpop_header = match dpop_header {
            Some(h) => h,
            None => {
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    Some("atproto profile requires a DPoP proof header"),
                );
            }
        };
        let atproto_issuer = state.atproto_issuer_url();
        let par_endpoint = format!("{}/oauth/par", atproto_issuer.trim_end_matches('/'));
        match super::dpop::verify_dpop_proof(&dpop_header, "POST", &par_endpoint, None) {
            Ok(proof) => {
                // Record the jkt so future token-endpoint proofs from this
                // key are nonce-bound (RFC 9449 §8). Single-use jti check.
                if !state.check_and_record_dpop_jti(&proof.jti, proof.iat) {
                    return par_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_dpop_proof",
                        Some("DPoP proof jti already used"),
                    );
                }
                state.mark_dpop_client_nonced(&proof.jkt).await;
                Some(proof.jkt)
            }
            Err(e) => {
                tracing::warn!(error = %e, "DPoP proof verification failed at PAR");
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_dpop_proof",
                    None,
                );
            }
        }
    } else {
        None
    };

    // Build the AuthorizeParams snapshot to store.
    let params = AuthorizeParams {
        client_id: form.client_id,
        redirect_uri: form.redirect_uri,
        code_challenge: form.code_challenge,
        code_challenge_method: form.code_challenge_method,
        response_type: form.response_type,
        state: form.state,
        scope: Some(granted_scope_str),
        resource: form.resource,
        nonce: form.nonce,
        dpop_jkt: dpop_jkt.clone(),
        client_assertion_jkt,
    };

    // Generate a random URN. RFC 9126 §2.2 / atproto OAuth use the standard
    // `urn:ietf:params:oauth:request_uri:` namespace.
    let mut id_bytes = [0u8; 24];
    rand::rngs::OsRng.fill_bytes(&mut id_bytes);
    let id = URL_SAFE_NO_PAD.encode(id_bytes);
    let request_uri = format!("urn:ietf:params:oauth:request_uri:{id}");

    let expires_at = Instant::now() + std::time::Duration::from_secs(PAR_TTL_SECS);
    state.pending_par_requests.write().await.insert(
        request_uri.clone(),
        PushedAuthRequest {
            params,
            expires_at,
        },
    );

    // #1113 rev2 finding 3: when DPoP was bound, issue a fresh server nonce
    // and return it in the `DPoP-Nonce` header so the client can include it
    // in subsequent proofs (matches the atproto client's PAR response handling).
    let dpop_nonce = if dpop_jkt.is_some() {
        Some(state.issue_dpop_nonce().await)
    } else {
        None
    };

    let mut resp = (
        StatusCode::CREATED,
        Json(ParResponse {
            request_uri,
            expires_in: PAR_TTL_SECS,
        }),
    )
        .into_response();
    if let Some(nonce) = dpop_nonce {
        if let Ok(val) = axum::http::HeaderValue::from_str(&nonce) {
            resp.headers_mut().insert("DPoP-Nonce", val);
        }
    }
    resp
}
