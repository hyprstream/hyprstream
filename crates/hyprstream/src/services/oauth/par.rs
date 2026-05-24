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
    http::StatusCode,
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
    error_description: String,
}

fn par_error(status: StatusCode, error: &'static str, description: impl Into<String>) -> Response {
    (
        status,
        Json(ParError {
            error,
            error_description: description.into(),
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
    Form(form): Form<ParForm>,
) -> Response {
    // Validate response_type
    if form.response_type != "code" {
        return par_error(
            StatusCode::BAD_REQUEST,
            "unsupported_response_type",
            "Only response_type=code is supported",
        );
    }

    // Validate PKCE method
    if form.code_challenge_method != "S256" {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "Only code_challenge_method=S256 is supported",
        );
    }
    if form.code_challenge.is_empty() {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "code_challenge is required (PKCE mandatory per OAuth 2.1)",
        );
    }

    // Resolve client (CIMD or dynamically registered) and pull redirect URIs.
    let redirect_uris = if form.client_id.starts_with("https://") {
        match resolve_cimd_client(&state, &form.client_id).await {
            Ok(client) => client.redirect_uris,
            Err(e) => {
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    format!("Failed to resolve client metadata: {e}"),
                );
            }
        }
    } else {
        let clients = state.clients.read().await;
        match clients.get(&form.client_id) {
            Some(client) => client.redirect_uris.clone(),
            None => {
                return par_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_client",
                    "Unknown client_id. Register first via POST /oauth/register",
                );
            }
        }
    };

    if !validate_redirect_uri(&form.redirect_uri, &redirect_uris) {
        return par_error(
            StatusCode::BAD_REQUEST,
            "invalid_redirect_uri",
            "redirect_uri does not match registered URIs",
        );
    }

    // Build the AuthorizeParams snapshot to store.
    let params = AuthorizeParams {
        client_id: form.client_id,
        redirect_uri: form.redirect_uri,
        code_challenge: form.code_challenge,
        code_challenge_method: form.code_challenge_method,
        response_type: form.response_type,
        state: form.state,
        scope: form.scope,
        resource: form.resource,
        nonce: form.nonce,
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

    (
        StatusCode::CREATED,
        Json(ParResponse {
            request_uri,
            expires_in: PAR_TTL_SECS,
        }),
    )
        .into_response()
}
