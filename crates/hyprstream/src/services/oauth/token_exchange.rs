//! RFC 8693 OAuth Token Exchange grant handler.
//!
//! Grant type: `urn:ietf:params:oauth:grant-type:token-exchange`
//!
//! Exchanges an existing credential (OIDC ID token, at+jwt, or WIT) for a
//! hyprstream at+jwt. Serves as the HTTP-layer complement to ExchangeWit (ZMQ)
//! and enables the MCP SDK's CrossAppAccessProvider enterprise flow.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq as _;

use super::state::{DpopJtiAdmission, OAuthState};
use crate::mac::exchange::{GrantDecision, GrantError, GrantRequest, GrantedAccess};
use crate::services::generated::policy_client::{IssueToken, IssueTokenProfile};
use hyprstream_pds::repo_authority::is_path_form_did_web;
// #1425: the browser RFC 8693 sender-bound contract. The public client_id is
// the single source of truth shared with the WASM client (`hyprstream-rpc`),
// so the wire contract cannot drift between client and server.
use hyprstream_rpc::wasm_token_exchange::BROWSER_PUBLIC_CLIENT_ID;

const TOKEN_TYPE_ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
const TOKEN_TYPE_ACCESS_TOKEN: &str = "urn:ietf:params:oauth:token-type:access_token";
const TOKEN_TYPE_JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
const ISSUED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";
pub const ATPROTO_EXCHANGE_NSID: &str = "ai.hyprstream.identity.exchangeUcan";
pub const ATPROTO_SESSION_EXCHANGE_NSID: &str = "ai.hyprstream.identity.exchangeSession";
pub(super) const MAX_ATPROTO_SERVICE_TOKEN_LIFETIME: i64 = 3600;
pub(super) const MAX_ATPROTO_EXCHANGE_TOKEN_TTL: u32 = 300;

pub(super) struct VerifiedSubject {
    pub(super) sub: String,
    cnf_key_bytes: Option<[u8; 32]>,
    /// RFC 9449 `cnf.jkt` sender-binding thumbprint on the subject token, when
    /// present. A DPoP-bound access-token subject MUST be exchanged under a
    /// token-endpoint proof from the **same** key (#1425 P1: the subject
    /// token's sender constraint must be preserved, not re-bound to an
    /// attacker's key).
    cnf_jkt: Option<String>,
    iat: i64,
    /// Authority from a locally validated OAuth access-token grant. Identity
    /// tokens and generic JWTs do not carry a server-authorized OAuth grant.
    granted_scopes: Option<Vec<String>>,
    pub(super) verified_tenant: Option<String>,
    pub(super) atproto_replay: Option<(String, String, i64)>,
    require_clearance: bool,
    ttl_ceiling: Option<u32>,
}

/// POST /oauth/token — token-exchange grant (RFC 8693).
pub async fn exchange_token_exchange(
    state: &Arc<OAuthState>,
    subject_token: &str,
    subject_token_type: &str,
    audience: Option<&str>,
    scope: Option<&str>,
    actor_token: Option<&str>,
    output_dpop_jkt: Option<String>,
    requested_token_type: Option<&str>,
    tenant: Option<&str>,
    client_id: &str,
) -> Response {
    // Actor token (delegation) is deferred — RFC 8693 §4.
    if actor_token.is_some() {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "actor_token is not supported",
        );
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
        TOKEN_TYPE_ACCESS_TOKEN => verify_access_token(state, subject_token).await,
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

    let requested_scopes = match attenuate_exchange_scopes(&verified, scope) {
        Ok(scopes) => scopes,
        Err(description) => {
            return tx_error(StatusCode::BAD_REQUEST, "invalid_scope", description);
        }
    };

    let tenant = match exchange_tenant(&verified, tenant) {
        Ok(tenant) => tenant,
        Err(description) => {
            return tx_error(StatusCode::BAD_REQUEST, "invalid_target", description);
        }
    };

    // The PolicyService check remains the shared signing boundary for every
    // RPC issuer; this gives RFC 8693 callers a concrete OAuth error before a
    // legacy subject can reach that RPC boundary.
    if is_path_form_did_web(&verified.sub) {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)",
        );
    }
    let subject_identity = crate::server::middleware::AuthenticatedUser {
        user: verified.sub.clone(),
        verified_tenant: tenant.clone(),
        token: None,
        exp: None,
    };
    if subject_identity.authorization_domain().is_err() {
        return tx_error(
            StatusCode::UNAUTHORIZED,
            "invalid_grant",
            "subject token has no valid verified hosted-account tenant binding",
        );
    }

    let fresh = if let Some((issuer, jti, exp)) = verified.atproto_replay.as_ref() {
        state.check_and_record_atproto_service_jti(issuer, jti, *exp)
    } else {
        let token_hash = URL_SAFE_NO_PAD.encode(Sha256::digest(subject_token.as_bytes()));
        state.check_and_record_dpop_jti(&token_hash, verified.iat)
    };
    if !fresh {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "subject_token already used (replay)",
        );
    }

    // Service assertions carry the exact key which verified their signature.
    let user_pub_key = verified.cnf_key_bytes.map(|b| URL_SAFE_NO_PAD.encode(b));
    let output_issuer =
        state.issuer_for_scopes(requested_scopes.as_deref().unwrap_or_default());

    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes,
            ttl: Some(state.token_ttl.min(
                verified.ttl_ceiling.unwrap_or(state.token_ttl),
            )),
            audience: audience.map(str::to_owned),
            subject: Some(verified.sub.clone()),
            user_pub_key,
            dpop_jkt: output_dpop_jkt,
            // RFC 8693/XRPC credentials cross a network boundary. They must
            // never inherit the PolicyService's empty-issuer local-IPC profile.
            issuer: Some(output_issuer),
            tenant,
            require_clearance: verified.require_clearance,
            session_id: None,
            issuance_profile: if verified.sub.starts_with("service:") {
                IssueTokenProfile::Service
            } else {
                IssueTokenProfile::Rfc8693
            },
            // RFC 9068 §2.2.1: the exchanging OAuth client on the user
            // `at+jwt`; the service (`wit+jwt`) form carries none.
            client_id: (!verified.sub.starts_with("service:")).then(|| client_id.to_owned()),
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
            tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Failed to issue token",
            )
        }
    }
}

/// The canonical Hyprstream RPC service resource/audience the browser
/// sender-bound exchange mints for: the AS origin (`issuer` without path).
/// The browser presents the exchanged token back to RPC services on this
/// origin, so the mint is restricted to exactly it (#1425 audience/resource
/// restriction). Any caller-supplied `audience` MUST match this value —
/// substitution is rejected as `invalid_target`.
fn browser_exchange_audience(state: &OAuthState) -> String {
    state.atproto_issuer_url()
}

/// POST /oauth/token — the **browser** RFC 8693 token-exchange grant composed
/// with RFC 9449 DPoP (#1425).
///
/// This is the sender-bound contract the WASM `VfsShell` uses: it exchanges an
/// atproto/external subject token for a short-lived at+jwt access token that is
/// **bound to the browser's DPoP key** (`cnf.jkt`) and **audience-restricted**
/// to the Hyprstream RPC service. It is distinct from the generic
/// [`exchange_token_exchange`] (bearer, optional DPoP) and the UCAN grant path.
///
/// **Public client rule (RFC 9700):** `client_id` is the well-known public
/// browser client ([`BROWSER_PUBLIC_CLIENT_ID`]) — it identifies the browser
/// client and is **not a client secret and not a proof of identity**. There is
/// no secret; the sender binding comes entirely from the verified DPoP proof.
///
/// **Sender binding (RFC 9449):** a fresh DPoP proof is **mandatory** at the
/// token endpoint. The proof's method (`POST`), URI (`{issuer}/oauth/token`),
/// `iat`, `jti` (single-use, replay-rejected), and server-issued nonce are all
/// verified here; the proof key's `jkt` becomes the minted token's `cnf.jkt`,
/// and the response carries `token_type: DPoP`. The resource layer
/// (`auth.rs`) then refuses to accept the token as a plain Bearer and requires
/// a matching proof + `ath` on every use, so the binding is enforced end to
/// end — a stolen DPoP-bound token is unusable without the key.
///
/// **Audience/resource restriction:** the issued token targets
/// [`browser_exchange_audience`] (the AS origin / RPC service resource). A
/// caller-supplied `audience` that differs is rejected (`invalid_target`).
///
/// **Access-token only:** no refresh token is issued (RFC 8693 browser/public
/// client rotation is a separately reviewed, metadata-advertised policy).
///
/// **Assurance boundary:** this composes Classical browser authentication
/// (Ed25519/ES256 DPoP) with RFC 8693/9449. It does **not** claim `PqHybrid`
/// identity assurance; hybrid RPC transport signing is independent of this
/// Classical sender binding.
pub(super) async fn exchange_browser_token_exchange(
    state: &Arc<OAuthState>,
    subject_token: &str,
    subject_token_type: &str,
    dpop_header: Option<&str>,
    audience: Option<&str>,
    resource: Option<&str>,
    scope: Option<&str>,
    requested_token_type: Option<&str>,
    actor_token: Option<&str>,
    actor_token_type: Option<&str>,
    tenant: Option<&str>,
    client_id: &str,
) -> Response {
    // Public client rule: the browser client_id is a public identifier. It is
    // not authenticated here — there is no secret. The routing in `token.rs`
    // only reaches this handler for the well-known browser client, so a
    // mismatch is a contract violation, not an auth failure.
    if client_id != BROWSER_PUBLIC_CLIENT_ID {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "browser token-exchange requires the public browser client_id",
        );
    }

    // ── P2: Explicitly reject RFC 8693 fields this contract does not support ──
    // The browser exchange is a leaf path, not a general-purpose delegation
    // flow. Fields that are silently ignored in the generic handler MUST be
    // explicitly rejected here so there is no second, ambiguous interpretation.
    if actor_token.is_some() {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "actor_token (delegation) is not supported for browser token-exchange",
        );
    }
    // #1425 r2 P2: `actor_token_type` is rejected independently of
    // `actor_token` — a request naming the type but omitting the token (or
    // sending only the type) must not be treated as if it carried no actor
    // field at all. Checked here, before proof admission or subject-token
    // consumption, so neither the DPoP replay registry nor the subject-token
    // single-use registry is ever touched by a request this contract refuses.
    if actor_token_type.is_some() {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "actor_token_type is not supported for browser token-exchange",
        );
    }
    if let Some(rtt) = requested_token_type {
        if rtt != ISSUED_TOKEN_TYPE {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_target",
                "browser token-exchange issues access_token only",
            );
        }
    }
    if tenant.is_some() {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "tenant is not a browser token-exchange parameter (authority is resolved from the subject token)",
        );
    }

    // ── P2: Audience/resource restriction (#1425) ────────────────────────────
    // Checked before DPoP verification and subject-token consumption so a
    // malformed request fails fast. The browser token targets the Hyprstream
    // RPC service resource (the canonical AS origin the metadata advertises).
    // RFC 8707 `resource` and RFC 8693 `audience` both scope the issued token;
    // for this contract they MUST both equal (or be omitted, defaulting to)
    // the RPC service origin.
    let rpc_resource = browser_exchange_audience(state);
    for requested in [audience, resource].into_iter().flatten() {
        if requested != rpc_resource {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_target",
                "browser exchange audience/resource must be the Hyprstream RPC service resource",
            );
        }
    }

    // ── 1. Mandatory DPoP proof (RFC 9449 sender binding) ───────────────────
    let Some(proof_str) = dpop_header else {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "browser token-exchange requires a DPoP proof (sender-binding)",
        );
    };
    // The DPoP htu is the canonical token endpoint the AS advertises
    // (`{origin}/oauth/token`), matching the browser's absolute request URI
    // and the RFC 8414 `token_endpoint` value — not the possibly path-bearing
    // configured `issuer_url`.
    let token_endpoint = format!("{}/oauth/token", state.atproto_issuer_url());
    let proof = match super::dpop::verify_dpop_proof(proof_str, "POST", &token_endpoint, None) {
        Ok(p) => p,
        Err(error) => {
            tracing::warn!(%error, "browser exchange rejected invalid DPoP proof");
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_dpop_proof",
                "DPoP proof verification failed",
            );
        }
    };
    // Single-use jti: a replayed proof is rejected (RFC 9449 §6.1).
    let admission = state.check_and_record_dpop_jti_admission(&proof.jti, proof.iat);
    if !admission.is_inserted() {
        if admission == DpopJtiAdmission::Duplicate {
            tracing::debug!("browser exchange: DPoP JTI replay rejected");
        }
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_dpop_proof",
            "DPoP proof rejected",
        );
    }
    // Server-nonce policy (RFC 9449 §8): once a key has been issued a nonce,
    // every subsequent proof MUST carry a valid server nonce. A bootstrap
    // (first proof from this jkt) is accepted and a fresh nonce is issued in
    // the success response.
    let client_needs_nonce = state.dpop_client_requires_nonce(&proof.jkt).await;
    let nonce_to_return: Option<String> = match (client_needs_nonce, proof.nonce.as_deref()) {
        (true, None) => {
            let fresh = state.issue_dpop_nonce().await;
            state.mark_dpop_client_nonced(&proof.jkt).await;
            tracing::warn!(jkt = %proof.jkt, "browser exchange: DPoP nonce required but omitted");
            return dpop_nonce_tx_error(
                &fresh,
                "use_dpop_nonce",
                "DPoP proof must include a server-issued nonce",
            );
        }
        (_, Some(presented)) => {
            if !state.verify_dpop_nonce(presented).await {
                let fresh = state.issue_dpop_nonce().await;
                state.mark_dpop_client_nonced(&proof.jkt).await;
                tracing::warn!(jkt = %proof.jkt, "browser exchange: DPoP nonce invalid/expired");
                return dpop_nonce_tx_error(
                    &fresh,
                    "use_dpop_nonce",
                    "DPoP nonce invalid or expired",
                );
            }
            None
        }
        (false, None) => {
            // Bootstrap: accept; issue a fresh nonce on the success path.
            Some(state.issue_dpop_nonce().await)
        }
    };

    // ── 3. Verify the subject token ──────────────────────────────────────────
    // P2: ID tokens are NOT accepted for the browser contract. `verify_id_token`
    // validates no audience/client binding, so trusting an issuer alone is not
    // sufficient to establish the token was minted for this browser client.
    // The browser exchanges a `jwt` (atproto/external service assertion) or an
    // `access_token` (subject to same-key cnf.jkt confirmation below).
    let verified = match subject_token_type {
        TOKEN_TYPE_ACCESS_TOKEN => verify_access_token(state, subject_token).await,
        TOKEN_TYPE_JWT => verify_jwt(state, subject_token).await,
        TOKEN_TYPE_ID_TOKEN => Err(
            "id_token is not supported for browser token-exchange \
             (no audience/client binding)"
                .to_owned(),
        ),
        _ => Err(
            "unsupported subject_token_type for browser exchange; \
             supported: access_token, jwt"
                .to_owned(),
        ),
    };
    let verified = match verified {
        Ok(v) => v,
        Err(e) => return tx_error(StatusCode::UNAUTHORIZED, "invalid_grant", &e),
    };

    // ── 4. Scope + tenant + subject invariants (shared with the generic path) ──
    let requested_scopes = match attenuate_exchange_scopes(&verified, scope) {
        Ok(s) => s,
        Err(description) => return tx_error(StatusCode::BAD_REQUEST, "invalid_scope", description),
    };
    let tenant = match exchange_tenant(&verified, None) {
        Ok(t) => t,
        Err(description) => return tx_error(StatusCode::BAD_REQUEST, "invalid_target", description),
    };
    // ── Subject-token sender-constraint preservation (#1425 P1) ──────────────
    // A DPoP-bound subject access token (cnf.jkt present) MUST be exchanged
    // under a token-endpoint proof from the **same** key. Without this check,
    // a stolen sender-constrained token string can be re-bound to an
    // attacker-owned key — defeating the source token's sender constraint.
    // The one-use subject-token hash only makes this a race, not proof of
    // possession. The check fires BEFORE the replay registry consumes the
    // subject token so a mismatched key is rejected without burning it.
    if let Some(ref subject_jkt) = verified.cnf_jkt {
        if subject_jkt
            .as_bytes()
            .ct_eq(proof.jkt.as_bytes())
            .unwrap_u8()
            == 0
        {
            tracing::warn!(
                sub = %verified.sub,
                subject_cnf_jkt = %subject_jkt,
                proof_jkt = %proof.jkt,
                "browser exchange: DPoP proof key does not match the subject token's cnf.jkt"
            );
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_dpop_proof",
                "DPoP proof key must match the subject token's sender binding (cnf.jkt)",
            );
        }
    }
    if is_path_form_did_web(&verified.sub) {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "path-form did:web account subjects are frozen; \
             host-form account minting is not available yet (#1159)",
        );
    }
    let subject_identity = crate::server::middleware::AuthenticatedUser {
        user: verified.sub.clone(),
        verified_tenant: tenant.clone(),
        token: None,
        exp: None,
    };
    if subject_identity.authorization_domain().is_err() {
        return tx_error(
            StatusCode::UNAUTHORIZED,
            "invalid_grant",
            "subject token has no valid verified hosted-account tenant binding",
        );
    }
    // Replay-protect the subject token exactly as the generic path does.
    let fresh = if let Some((issuer, jti, exp)) = verified.atproto_replay.as_ref() {
        state.check_and_record_atproto_service_jti(issuer, jti, *exp)
    } else {
        let token_hash = URL_SAFE_NO_PAD.encode(Sha256::digest(subject_token.as_bytes()));
        state.check_and_record_dpop_jti(&token_hash, verified.iat)
    };
    if !fresh {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "subject_token already used (replay)",
        );
    }

    let requested_scopes_ref = requested_scopes.as_deref().unwrap_or_default();
    let output_issuer = state.issuer_for_scopes(requested_scopes_ref);

    // ── 5. Mint the sender-bound (cnf.jkt) access token via PolicyService ────
    // `dpop_jkt = proof.jkt` is the cnf.jkt binding. No refresh token is
    // issued (access-token only; rotation is a separately reviewed policy).
    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes,
            ttl: Some(state.token_ttl.min(verified.ttl_ceiling.unwrap_or(state.token_ttl))),
            audience: Some(rpc_resource),
            subject: Some(verified.sub.clone()),
            user_pub_key: None,
            dpop_jkt: Some(proof.jkt.clone()),
            // RFC 8693 credentials cross a network boundary; never inherit the
            // PolicyService's empty-issuer local-IPC profile.
            issuer: Some(output_issuer),
            tenant,
            require_clearance: verified.require_clearance,
            session_id: None,
            issuance_profile: if verified.sub.starts_with("service:") {
                IssueTokenProfile::Service
            } else {
                IssueTokenProfile::Rfc8693
            },
            // RFC 9068 §2.2.1: the exchanging OAuth client on the user
            // `at+jwt`; the service (`wit+jwt`) form carries none.
            client_id: (!verified.sub.starts_with("service:")).then(|| client_id.to_owned()),
        })
        .await;

    let token_info = match result {
        Ok(ti) => ti,
        Err(e) => {
            tracing::error!(sub = %verified.sub, error = %e, "browser exchange issuance failed");
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Failed to issue token",
            );
        }
    };

    // ── 6. DPoP response (access-token only) + fresh nonce for the next proof ─
    let now = chrono::Utc::now().timestamp();
    let expires_in = (token_info.expires_at - now).max(0);
    // Mark this jkt as nonce-eligible so the next exchange/refresh proof from
    // it is required to carry a server nonce.
    state.mark_dpop_client_nonced(&proof.jkt).await;
    let nonce = match nonce_to_return {
        Some(n) => n,
        None => state.issue_dpop_nonce().await,
    };
    tracing::info!(sub = %verified.sub, "Browser sender-bound token exchanged (cnf.jkt, DPoP)");
    let mut resp = (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::json!({
            "access_token": token_info.token,
            "issued_token_type": ISSUED_TOKEN_TYPE,
            "token_type": "DPoP",
            "expires_in": expires_in,
        })),
    )
        .into_response();
    if let Ok(val) = axum::http::HeaderValue::from_str(&nonce) {
        resp.headers_mut().insert("DPoP-Nonce", val);
    }
    resp
}

/// Build a `400` OAuth error response that also carries a fresh `DPoP-Nonce`
/// header (RFC 9449 §8 retry contract) for the browser exchange path.
fn dpop_nonce_tx_error(nonce: &str, error: &str, description: &str) -> Response {
    let mut resp = tx_error(StatusCode::BAD_REQUEST, error, description);
    if let Ok(val) = axum::http::HeaderValue::from_str(nonce) {
        resp.headers_mut().insert("DPoP-Nonce", val);
    }
    resp
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
        sub: hyprstream_rpc::Subject::federated(&iss, &claims.sub)
            .name()
            .ok_or_else(|| "id_token subject is empty".to_owned())?
            .to_owned(),
        cnf_key_bytes: None, // ID tokens carry no key binding
        cnf_jkt: None,
        iat: claims.iat,
        granted_scopes: None,
        verified_tenant: None,
        atproto_replay: None,
        require_clearance: false,
        ttl_ceiling: None,
    })
}

/// Verify an existing hyprstream at+jwt through the same algorithm, audience,
/// issuer, and revocation checks used by protected OAuth routes.
async fn verify_access_token(state: &OAuthState, token: &str) -> Result<VerifiedSubject, String> {
    let claims = super::auth::validate_oauth_access_token(state, token)
        .await
        .map_err(|e| format!("access_token verification failed: {e}"))?;

    let granted_scopes = claims
        .scope
        .as_ref()
        .map(|_| claims.granted_scopes().map(str::to_owned).collect());
    let cnf_key_bytes = claims.cnf_key_bytes();
    let cnf_jkt = claims.cnf_jkt().map(str::to_owned);
    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes,
        cnf_jkt,
        iat: claims.iat,
        granted_scopes,
        verified_tenant: claims.tenant,
        atproto_replay: None,
        require_clearance: false,
        ttl_ceiling: None,
    })
}

/// Verify a generic JWT — WIT from local trust store or federated OIDC issuer.
///
/// For `sub: service:*`: global trust store (CA-signed WIT).
/// For other subjects: issuer must be in `trusted_issuers`.
/// Audience must equal the token endpoint URL (same constraint as RFC 7523).
async fn verify_jwt(state: &Arc<OAuthState>, token: &str) -> Result<VerifiedSubject, String> {
    let issuer_hint = token
        .split('.')
        .nth(1)
        .and_then(|payload| URL_SAFE_NO_PAD.decode(payload).ok())
        .and_then(|payload| serde_json::from_slice::<serde_json::Value>(&payload).ok())
        .and_then(|payload| payload.get("iss").and_then(serde_json::Value::as_str).map(str::to_owned));
    if issuer_hint.as_deref().is_some_and(|issuer| {
        issuer.starts_with("did:plc:") || issuer.starts_with("did:web:")
    }) {
        return verify_atproto_service_jwt(state, token, ATPROTO_EXCHANGE_NSID).await;
    }

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

    let (claims, service_signing_key) = if sub.starts_with("service:") {
        let svc_name = sub.trim_start_matches("service:");
        let (claims, vk) =
            super::jwt_bearer::decode_with_any_local_service_key(token, svc_name, &token_endpoint)
                .ok_or_else(|| format!("JWT verification failed for service: {svc_name}"))?;
        (claims, Some(vk.to_bytes()))
    } else {
        let cfg = state
            .trusted_issuers
            .get(&iss)
            .ok_or_else(|| format!("Issuer not in trusted_issuers allow-list: {iss}"))?
            .clone();
        let vk = super::jwt_bearer::resolve_federated_key(state, &iss, token, cfg.allow_http)
            .await
            .map_err(|e| format!("JWKS key resolution failed for {iss}: {e}"))?;
        let claims = hyprstream_rpc::auth::decode_with_key(token, &vk, Some(&token_endpoint))
            .map_err(|e| format!("JWT verification failed: {e}"))?;
        (claims, None)
    };

    let mut claims = claims;
    let atproto_issuer = state.atproto_issuer_url();
    let local_issuers = [atproto_issuer.as_str(), state.issuer_url.as_str()];
    claims.strip_federated_tenant(&local_issuers);
    let subject = claims.subject(&local_issuers);
    subject
        .validate()
        .map_err(|error| format!("invalid JWT subject: {error}"))?;
    let cnf_key_bytes = service_signing_key.or_else(|| claims.cnf_key_bytes());
    Ok(VerifiedSubject {
        sub: subject
            .name()
            .ok_or_else(|| "JWT subject is empty".to_owned())?
            .to_owned(),
        cnf_key_bytes,
        cnf_jkt: claims.cnf_jkt().map(str::to_owned),
        iat: claims.iat,
        granted_scopes: None,
        verified_tenant: claims.tenant,
        atproto_replay: None,
        require_clearance: false,
        ttl_ceiling: None,
    })
}

#[derive(Deserialize)]
struct AtprotoJwtHeader {
    alg: String,
    #[serde(default)]
    typ: Option<String>,
    #[serde(default)]
    kid: Option<String>,
}

#[derive(Deserialize)]
struct AtprotoServiceClaims {
    iss: String,
    aud: String,
    exp: i64,
    iat: i64,
    lxm: String,
    jti: String,
}

pub(super) async fn verify_atproto_service_jwt(
    state: &Arc<OAuthState>,
    token: &str,
    expected_lxm: &str,
) -> Result<VerifiedSubject, String> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err("ATProto service JWT must contain exactly three segments".to_owned());
    }
    let header: AtprotoJwtHeader = decode_jwt_json(parts[0], "header")?;
    let claims: AtprotoServiceClaims = decode_jwt_json(parts[1], "claims")?;
    if !matches!(header.alg.as_str(), "ES256" | "ES256K") {
        return Err("ATProto service JWT alg must be ES256 or ES256K".to_owned());
    }
    if header.typ.as_deref().is_some_and(|typ| !typ.eq_ignore_ascii_case("JWT")) {
        return Err("ATProto service JWT typ must be JWT when present".to_owned());
    }
    super::state::subject_did_for(&state.issuer_url, &claims.iss)
        .map_err(|error| format!("invalid ATProto issuer DID: {error}"))?;
    let expected_audience = state.atproto_service_did()
        .ok_or_else(|| "OAuth issuer cannot be represented as a host service DID".to_owned())?;
    if claims.aud != expected_audience {
        return Err(format!("ATProto service JWT aud must equal host DID {expected_audience}"));
    }
    if claims.lxm != expected_lxm {
        return Err(format!("ATProto service JWT lxm must equal {expected_lxm}"));
    }
    let now = chrono::Utc::now().timestamp();
    if claims.exp <= now {
        return Err("ATProto service JWT is expired".to_owned());
    }
    if claims.iat > now.saturating_add(5) || claims.exp <= claims.iat {
        return Err("ATProto service JWT has an invalid iat/exp interval".to_owned());
    }
    let _lifetime = claims
        .exp
        .checked_sub(claims.iat)
        .filter(|lifetime| *lifetime > 0 && *lifetime <= MAX_ATPROTO_SERVICE_TOKEN_LIFETIME)
        .ok_or_else(|| "ATProto service JWT has an invalid iat/exp interval".to_owned())?;
    if claims.jti.is_empty() || claims.jti.len() > 256 {
        return Err("ATProto service JWT jti must be 1..=256 bytes".to_owned());
    }

    let document = state.atproto_did_resolver.resolve_document(&claims.iss).await
        .map_err(|error| format!("ATProto DID resolution failed: {error}"))?;
    let fragment = header.kid.as_deref().unwrap_or("#atproto");
    let key_id = if fragment.starts_with('#') {
        format!("{}{fragment}", claims.iss)
    } else {
        fragment.to_owned()
    };
    if !key_id.starts_with(&format!("{}#", claims.iss)) {
        return Err("ATProto signing kid must be a fragment of the issuer DID".to_owned());
    }
    let methods = document.get("verificationMethod")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "ATProto DID document has no verificationMethod array".to_owned())?;
    let mut matches = methods.iter().filter(|method| {
        method.get("id").and_then(serde_json::Value::as_str) == Some(key_id.as_str())
    });
    let method = matches.next()
        .ok_or_else(|| format!("ATProto signing method {key_id} not found"))?;
    if matches.next().is_some() {
        return Err(format!("ATProto signing method {key_id} is ambiguous"));
    }
    if method.get("type").and_then(serde_json::Value::as_str) != Some("Multikey")
        || method.get("controller").and_then(serde_json::Value::as_str)
            != Some(claims.iss.as_str())
    {
        return Err("ATProto signing method must be a subject-controlled Multikey".to_owned());
    }
    let multikey = method.get("publicKeyMultibase")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "ATProto signing method has no publicKeyMultibase".to_owned())?;
    verify_atproto_ecdsa(
        &header.alg,
        multikey,
        format!("{}.{}", parts[0], parts[1]).as_bytes(),
        parts[2],
    )?;
    // The service assertion proves the DID, not a tenant. Resolve any local
    // tenant binding only after signature verification and only from the
    // authority-owned hosted-account record store.
    let verified_tenant = state.hosted_account_tenant(&claims.iss).await?;

    Ok(VerifiedSubject {
        sub: claims.iss.clone(),
        cnf_key_bytes: None,
        cnf_jkt: None,
        iat: claims.iat,
        granted_scopes: Some(vec!["transition:generic".to_owned()]),
        verified_tenant,
        atproto_replay: Some((claims.iss, claims.jti, claims.exp)),
        require_clearance: true,
        ttl_ceiling: Some(MAX_ATPROTO_EXCHANGE_TOKEN_TTL),
    })
}

fn decode_jwt_json<T: serde::de::DeserializeOwned>(
    segment: &str,
    name: &str,
) -> Result<T, String> {
    let bytes = URL_SAFE_NO_PAD.decode(segment)
        .map_err(|_| format!("ATProto service JWT {name} is not base64url"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("ATProto service JWT {name} is invalid: {error}"))
}

fn verify_atproto_ecdsa(
    alg: &str,
    multikey: &str,
    signing_input: &[u8],
    signature_segment: &str,
) -> Result<(), String> {
    use p256::ecdsa::signature::Verifier as _;

    let encoded = multikey.strip_prefix('z')
        .ok_or_else(|| "ATProto Multikey must use base58btc".to_owned())?;
    let key = bs58::decode(encoded).into_vec()
        .map_err(|_| "ATProto Multikey is invalid base58btc".to_owned())?;
    let signature = URL_SAFE_NO_PAD.decode(signature_segment)
        .map_err(|_| "ATProto service JWT signature is not base64url".to_owned())?;
    match alg {
        "ES256" => {
            let raw = key.strip_prefix(&[0x80, 0x24])
                .ok_or_else(|| "ES256 requires a p256-pub Multikey".to_owned())?;
            let verifying_key = p256::ecdsa::VerifyingKey::from_sec1_bytes(raw)
                .map_err(|_| "invalid P-256 ATProto signing key".to_owned())?;
            let signature = p256::ecdsa::Signature::from_slice(&signature)
                .map_err(|_| "invalid ES256 service JWT signature encoding".to_owned())?;
            verifying_key.verify(signing_input, &signature)
                .map_err(|_| "ATProto service JWT signature verification failed".to_owned())
        }
        "ES256K" => {
            let raw = key.strip_prefix(&[0xe7, 0x01])
                .ok_or_else(|| "ES256K requires a secp256k1-pub Multikey".to_owned())?;
            let verifying_key = k256::ecdsa::VerifyingKey::from_sec1_bytes(raw)
                .map_err(|_| "invalid secp256k1 ATProto signing key".to_owned())?;
            let signature = k256::ecdsa::Signature::from_slice(&signature)
                .map_err(|_| "invalid ES256K service JWT signature encoding".to_owned())?;
            verifying_key.verify(signing_input, &signature)
                .map_err(|_| "ATProto service JWT signature verification failed".to_owned())
        }
        _ => Err("unsupported ATProto service JWT algorithm".to_owned()),
    }
}

fn exchange_tenant(
    subject: &VerifiedSubject,
    requested: Option<&str>,
) -> Result<Option<String>, &'static str> {
    let requested = requested.map(str::trim);
    if requested.is_some_and(|tenant| {
        tenant.is_empty() || tenant == "*" || tenant.len() > 128
            || tenant.chars().any(char::is_control)
    }) {
        return Err("tenant must be a concrete, non-empty domain");
    }
    match (subject.verified_tenant.as_deref(), requested) {
        (Some(verified), Some(requested)) if verified != requested => {
            Err("requested tenant differs from the verified source-token tenant")
        }
        (Some(verified), _) => Ok(Some(verified.to_owned())),
        // Tenant selection is an authority assertion, not an exchange
        // parameter. Federated identity proofs establish identity but do not
        // bind the subject to a local Casbin domain. Only a tenant preserved
        // from a verified local-issuer token or resolved from a hosted account
        // record may reach PolicyService.
        (None, Some(_)) => Err("subject token has no verified local tenant binding"),
        (None, None) if subject.require_clearance => {
            Err("tenant is required for the ATProto exchange")
        }
        (None, None) => Ok(None),
    }
}

#[derive(Deserialize)]
pub struct AtprotoExchangeRequest {
    pub tenant: String,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub audience: Option<String>,
}

/// XRPC adapter over the RFC 8693 exchange core.
pub async fn exchange_atproto_ucan(
    State(state): State<Arc<OAuthState>>,
    headers: HeaderMap,
    Json(request): Json<AtprotoExchangeRequest>,
) -> Response {
    let Some(assertion) = headers.get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, token)| {
            scheme.eq_ignore_ascii_case("Bearer")
                && !token.is_empty()
                && !token.chars().any(char::is_whitespace)
        })
        .map(|(_, token)| token)
    else {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            "InvalidToken",
            "Authorization: Bearer service JWT is required",
        );
    };
    let endpoint = format!(
        "{}/xrpc/ai.hyprstream.identity.exchangeUcan",
        state.atproto_issuer_url().trim_end_matches('/')
    );
    let output_dpop_jkt = match headers
        .get("DPoP")
        .and_then(|value| value.to_str().ok())
        .and_then(|proof| super::dpop::verify_dpop_proof(proof, "POST", &endpoint, None).ok())
    {
        Some(proof) => Some(proof.jkt),
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                "InvalidRequest",
                "a valid DPoP proof is required to bind the exchanged credential",
            );
        }
    };
    let response = exchange_token_exchange(
        &state,
        assertion,
        TOKEN_TYPE_JWT,
        request.audience.as_deref(),
        request.scope.as_deref(),
        None,
        output_dpop_jkt,
        Some(ISSUED_TOKEN_TYPE),
        Some(&request.tenant),
        // The atproto XRPC exchange is a public-client (UCAN-authenticated)
        // flow; the public client identifier is the RFC 9068 `client_id` for a
        // user `at+jwt` output (unused for a service `wit+jwt` output).
        BROWSER_PUBLIC_CLIENT_ID,
    ).await;
    if response.status().is_success() {
        return response;
    }

    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let oauth_error = body.as_ref()
        .and_then(|value| value.get("error"))
        .and_then(serde_json::Value::as_str);
    let message = body.as_ref()
        .and_then(|value| value.get("error_description"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("ATProto credential exchange failed");
    let error = match oauth_error {
        Some("invalid_grant") => "InvalidToken",
        Some("invalid_scope" | "invalid_target" | "invalid_request") => "InvalidRequest",
        _ if status.is_server_error() => "InternalServerError",
        _ => "InvalidRequest",
    };
    xrpc_error(status, error, message)
}

fn xrpc_error(status: StatusCode, error: &str, message: &str) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(serde_json::json!({
            "error": error,
            "message": message,
        })),
    )
        .into_response()
}

/// Bound an RFC 8693 scope request to authority already present on the verified
/// subject access token. The signed subject-token grant is the ceiling: callers
/// may retain or narrow it, never add authority. Subject types without a local
/// OAuth grant cannot mint a scope claim through token exchange.
fn attenuate_exchange_scopes(
    subject: &VerifiedSubject,
    requested: Option<&str>,
) -> Result<Option<Vec<String>>, &'static str> {
    let Some(granted) = subject.granted_scopes.as_ref() else {
        return if requested.is_some() {
            Err("subject token carries no OAuth grant to authorize requested scope")
        } else {
            Ok(None)
        };
    };

    let Some(requested) = requested else {
        return Ok(Some(granted.clone()));
    };
    let requested: Vec<&str> = requested
        .split_whitespace()
        .map(|scope| {
            super::state::normalize_scope_token(scope)
                .ok_or("requested scope contains an invalid token")
        })
        .collect::<Result<_, _>>()?;
    if requested.is_empty() {
        return Err("requested scope must not be empty");
    }
    if requested
        .iter()
        .any(|scope| !granted.iter().any(|allowed| allowed == scope))
    {
        return Err("requested scope exceeds the subject token grant");
    }

    Ok(Some(
        granted
            .iter()
            .filter(|allowed| requested.contains(&allowed.as_str()))
            .cloned()
            .collect(),
    ))
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

// ─── S6: UCAN grant → access/refresh tokens (#572) ─────────────────────────
//
// This is the HTTP-layer wiring over `mac::exchange::evaluate_grant`. The
// security-critical logic (chain validation, ceiling-subset, MAC clearance,
// sender-binding) lives in `mac::exchange`; this handler is the RFC 8693
// adapter that decodes the grant, verifies the DPoP proof, resolves the
// subject's MAC context, calls `evaluate_grant`, and mints the short-ttl
// sender-bound token the decision authorizes.
//
// **S8 (#574) activation:** the MAC *clearance* now rides the verified `Claims`
// (the `clearance` field, signed by the issuing node) and the subject's
// assurance is derived from the DPoP proof key + the kid-anchored PQ trust
// store binding. The concrete `ClaimsSubjectContextResolver` does the real
// two-input S1 derivation. The caller supplies the resolver (built from the
// subject's verified claims + DPoP-derived key material); passing
// `DenyUnlabeledResolver` keeps the deny-by-default posture for a node that has
// not configured subject-clearance resolution.

/// Resolve a UCAN audience DID → the S1 [`SecurityContext`] it presents at grant
/// time (clearance clamped to verified key material).
///
/// **S8 (#574) ships the real derivation.** The concrete
/// [`ClaimsSubjectContextResolver`] reads the authority-asserted `clearance`
/// off verified [`Claims`] and derives assurance from the `VerifiedKeyMaterial`
/// the caller resolved from the verified crypto (DPoP proof key + PQ trust
/// store binding). The two-input `security_context(key_material)` clamps the
/// assurance axis DOWN to what the verified key supports — the load-bearing
/// #548 invariant.
///
/// SECURITY: the resolution MUST be from *authority-asserted* clearance (a
/// field the issuing node signed), never from a self-asserted claim in the
/// UCAN. The UCAN is the grant; the clearance is independent state the MAC
/// model holds about the subject. `None` (→ `UnlabeledSubject` → deny) remains
/// the fail-closed posture for a subject the resolver has no verified claims
/// for.
pub trait SubjectContextResolver: Send + Sync {
    /// The clearance context for `audience_did`, or `None` if the subject is
    /// unlabeled / unverified. `None` ⇒ `evaluate_grant` denies.
    fn resolve(&self, audience_did: &str) -> Option<hyprstream_rpc::auth::mac::SecurityContext>;
}

/// A no-op resolver that always returns `None` — the explicit fail-closed
/// choice for a node that has not configured subject-clearance resolution.
/// Production under MAC SHOULD wire [`ClaimsSubjectContextResolver`] instead.
pub struct DenyUnlabeledResolver;

impl SubjectContextResolver for DenyUnlabeledResolver {
    fn resolve(&self, _audience_did: &str) -> Option<hyprstream_rpc::auth::mac::SecurityContext> {
        // Fail-closed: no clearance is known ⇒ no token. See the trait docs.
        None
    }
}

/// **S8 (#574):** the concrete `SubjectContextResolver` that does the real
/// two-input MAC context derivation.
///
/// Construct it with the subject's verified [`Claims`] (carrying the
/// authority-asserted `clearance` field) and the [`VerifiedKeyMaterial`]
/// derived from the verified crypto (the DPoP proof key + the kid-anchored PQ
/// trust store binding, exactly what
/// [`EnvelopeContext::verified_key_material`](hyprstream_rpc::service::EnvelopeContext::verified_key_material)
/// computes). [`SubjectContextResolver::resolve`] then returns the assembled
/// [`SecurityContext`] — clearance clamped to the crypto-derived assurance.
///
/// The resolver matches `audience_did` against the claims' subject (the
/// principal the clearance was issued to). A mismatch ⇒ `None` ⇒ deny: the
/// clearance cannot be borrowed across identities.
///
/// This is what flips S6 from deny-everything to actually-issuing-tokens for
/// verified subjects. For a fully-verified PQ subject (PqHybrid key material +
/// a clearance that dominates the object label), `evaluate_grant` permits; for
/// a classical-key subject on a PQ-required object, the clamped assurance
/// floors to Classical and the dominance check denies (fail-closed).
pub struct ClaimsSubjectContextResolver {
    /// The audience DID this resolver's claims are bound to. `resolve()` checks
    /// the grant's audience matches before returning the context (anti-borrow).
    audience_did: String,
    /// The subject's verified claims, carrying the authority-asserted clearance.
    claims: hyprstream_rpc::auth::Claims,
    /// Assurance derived from the verified crypto (DPoP key + PQ anchor).
    key_material: hyprstream_rpc::auth::mac::VerifiedKeyMaterial,
}

impl ClaimsSubjectContextResolver {
    /// Construct a resolver for a single subject. `claims` is the subject's
    /// verified JWT claims (signed by the issuing node, so the `clearance` is
    /// authority-asserted). `key_material` is the assurance derived from the
    /// verified DPoP proof key + PQ trust store binding. `audience_did` is the
    /// DID the clearance was issued to; a grant whose audience differs is
    /// denied (the clearance cannot cross identities).
    pub fn new(
        audience_did: impl Into<String>,
        claims: hyprstream_rpc::auth::Claims,
        key_material: hyprstream_rpc::auth::mac::VerifiedKeyMaterial,
    ) -> Self {
        Self {
            audience_did: audience_did.into(),
            claims,
            key_material,
        }
    }
}

impl SubjectContextResolver for ClaimsSubjectContextResolver {
    fn resolve(&self, audience_did: &str) -> Option<hyprstream_rpc::auth::mac::SecurityContext> {
        use hyprstream_rpc::auth::mac::SubjectContextClaims as _;
        // Anti-borrow: the clearance was issued to `self.audience_did`; a grant
        // presented for a DIFFERENT audience cannot use this subject's clearance.
        if audience_did != self.audience_did.as_str() {
            return None;
        }
        // The two-input S1 derivation: clearance (from Claims) + assurance (from
        // verified key material). SecurityContext::from_clearance clamps the
        // assurance axis DOWN to what the crypto supports — no silent upgrade.
        self.claims.security_context(self.key_material)
    }
}

/// The two principals a minted grant token carries (#680/#681). For a delegated
/// grant this splits the single `sub` into the delegator (source of authority)
/// and the actor (the presenter that signs the downstream envelope) — the
/// confused-deputy fix: the token records "actor acting for delegator" instead
/// of collapsing to one identity.
struct TokenPrincipals {
    /// The minted token's `sub`: the delegator (UCAN chain-root issuer) for a
    /// delegated grant, or the subject (the self-issued root's DID) for a root
    /// grant. This is always a UCAN principal DID — never the RFC 8707 resource
    /// indicator (which stays on the `aud` claim).
    sub: String,
    /// The minted token's RFC 8693 §4.1 `act` claim: the actor (grant audience /
    /// presenter). `None` for a self-issued root grant (single principal).
    act: Option<hyprstream_rpc::auth::ActClaim>,
}

/// Derive the subject's MAC [`SecurityContext`] for a (possibly delegated) UCAN
/// grant, plus the [`TokenPrincipals`] the mint stamps on the token (#680/#681).
///
/// - **Self-issued root grant** (`root_issuer == audience`): single-principal.
///   Context is `resolve(audience)`; `sub = audience`, `act = None`.
/// - **Delegated grant** (`root_issuer != audience`): two-principal (#681). The
///   effective context is `meet(delegator = root_issuer, actor = audience)` via
///   [`SecurityContext::delegated`](hyprstream_rpc::auth::mac::SecurityContext::delegated)
///   — fail-closed if *either* principal is unresolved (no default clearance for
///   a missing principal). `sub = delegator`, `act = { sub: actor, clearance:
///   actor's }`, so the token is minted as "actor acting for delegator".
///
/// Pure over the resolver (a trait) so the delegation logic is unit-testable
/// without the async HTTP mint path.
fn resolve_grant_subject(
    grant: &hyprstream_rpc::auth::ucan::token::Ucan,
    resolver: &dyn SubjectContextResolver,
) -> ResolvedGrantSubject {
    let audience = grant.audience().as_str();
    let delegator = grant.root_issuer().as_str();

    if delegator == audience {
        // Self-issued root — no delegation. Single principal = the subject.
        return ResolvedGrantSubject {
            subject: resolver.resolve(audience),
            on_behalf_of: None,
            principals: TokenPrincipals {
                sub: audience.to_owned(),
                act: None,
            },
        };
    }

    // Delegation: resolve BOTH principals and take the fail-closed meet (#681).
    let delegator_ctx = resolver.resolve(delegator);
    let actor_ctx = resolver.resolve(audience);
    let ctx = hyprstream_rpc::auth::mac::SecurityContext::delegated(
        delegator_ctx.as_ref(),
        actor_ctx.as_ref(),
    );
    let principals = TokenPrincipals {
        sub: delegator.to_owned(),
        act: Some(hyprstream_rpc::auth::ActClaim {
            sub: audience.to_owned(),
            // The actor's authority-asserted clearance, carried so a downstream
            // reader can re-take the meet off the one verified token. `None` if
            // the actor is unresolved — in which case `ctx` above is already
            // `None` and the grant fails closed before minting.
            clearance: actor_ctx
                .map(|c| hyprstream_rpc::auth::mac::CredentialClearance::from_label(*c.clearance())),
            act: None,
        }),
    };
    ResolvedGrantSubject {
        subject: ctx,
        // The delegator (authority source), recorded on the grant audit record
        // as `on_behalf_of` (#680/#681). Carried separately from the met
        // `subject` so the audit trail names both principals of a delegated
        // decision. `None` if the delegator is unresolved (the grant already
        // fails closed via `subject` above).
        on_behalf_of: delegator_ctx,
        principals,
    }
}

/// Outcome of [`resolve_grant_subject`]: the effective (met) subject context the
/// grant gates evaluate, the delegator context for two-principal audit
/// attribution (#680/#681), and the principals the mint stamps on the token.
struct ResolvedGrantSubject {
    /// Effective context: `meet(delegator, actor)` for a delegated grant, or the
    /// sole principal for a self-issued root grant. `None` ⇒ fail closed.
    subject: Option<hyprstream_rpc::auth::mac::SecurityContext>,
    /// The delegator principal, for the audit record's `on_behalf_of`. `None`
    /// for a self-issued root grant (single principal) or an unresolved delegator.
    on_behalf_of: Option<hyprstream_rpc::auth::mac::SecurityContext>,
    /// The delegator/actor split the mint stamps on the token.
    principals: TokenPrincipals,
}

/// POST /oauth/token — UCAN grant (RFC 8693 token-exchange,
/// `subject_token_type = urn:hyprstream:token-type:ucan-grant`).
///
/// Accepts a CBOR-encoded UCAN as the `subject_token` (the subset-grant) and
/// mints a **short-ttl, sender-bound** access token (+ refresh token when the
/// store is configured) for the requested access — never the whole grant (ZSP).
///
/// `dpop_header` is the `DPoP` proof header; it is MANDATORY for this grant
/// type (ZSP: no bearer). The proof signature/htm/htu/replay are verified here
/// via the same `verify_dpop_proof` the other grant types use; the resulting
/// `jkt` is the sender-binding thumbprint. Passing `None` ⇒ `invalid_request`.
///
/// `subject_resolver` supplies the MAC clearance; `DenyUnlabeledResolver`
/// denies until S8 ships the real resolver.
///
/// Fail-closed: every `GrantError` maps to a concrete OAuth error. There is no
/// fallback path; authority-unreachable is a denial.
pub async fn exchange_ucan_grant(
    state: &Arc<OAuthState>,
    subject_token: &str,
    dpop_header: Option<&str>,
    requested_scope: Option<&str>,
    audience: Option<&str>,
    subject_resolver: &dyn SubjectContextResolver,
) -> Response {
    // ── 1. DPoP sender-binding is MANDATORY (ZSP) ──────────────────────────
    // No proof ⇒ no token. A bearer token minted from a grant re-introduces
    // standing access — the exact thing ZSP removes.
    let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));
    let dpop_jkt = match dpop_header.and_then(|h| {
        super::dpop::verify_dpop_proof(h, "POST", &token_endpoint, None)
            .ok()
            .map(|p| p.jkt)
    }) {
        Some(jkt) => jkt,
        None => {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "UCAN grant token-exchange requires a valid DPoP proof (sender-binding)",
            );
        }
    };

    // ── 2. Decode the CBOR UCAN grant ──────────────────────────────────────
    let ucan_bytes = match URL_SAFE_NO_PAD.decode(subject_token) {
        Ok(b) => b,
        Err(_) => {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "subject_token is not valid base64url CBOR",
            );
        }
    };
    let grant = match hyprstream_rpc::auth::ucan::token::Ucan::from_cbor(&ucan_bytes) {
        Ok(u) => u,
        Err(e) => {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_grant",
                &format!("subject_token is not a valid UCAN: {e}"),
            );
        }
    };

    // ── 3. Resolve the subject's MAC context (S8 seam; deny until then). ────
    //    #680/#681: a delegated grant (chain root ≠ audience) resolves TWO
    //    principals and takes the fail-closed meet(delegator, actor); a
    //    self-issued root grant stays single-principal. `principals` carries the
    //    delegator/actor split the mint stamps onto the token.
    let ResolvedGrantSubject {
        subject: subject_ctx,
        on_behalf_of,
        principals,
    } = resolve_grant_subject(&grant, subject_resolver);

    // ── 4. Parse the requested access off the form fields ──────────────────
    // TODO(#572-object-label): wire the manifest/TE object-label resolver so
    //   the S1 floor can be evaluated for real. Until then `object_label` is
    //   `None`, which makes the S1 object-label gate deny — the conservative
    //   direction.
    let request = match parse_grant_request(requested_scope, audience) {
        Ok(r) => r,
        Err(msg) => return tx_error(StatusCode::BAD_REQUEST, "invalid_scope", &msg),
    };

    // ── 5. The single fail-closed S6 path ──────────────────────────────────
    // A no-op UcanVerifier is NOT acceptable — signatures MUST verify. The
    // verifier is built from the trust store's anchored ML-DSA-65 keys (the
    // same `register_pq_trust` binding the rest of the TCB uses). Until that
    // binding is wired into the OAuth state, the grant path fails closed here
    // rather than trusting an unverified chain. There is NO fallback path.
    //
    // TODO(#572-verifier): construct the UcanVerifier from the trust store's
    //   anchored ML-DSA-65 keys. Until that wiring lands, the HTTP grant path
    //   denies every request at this gate — the conservative direction. The
    //   core `evaluate_grant` (with full happy-path + denial coverage) is
    //   exercised through its own tests using a real `UcanVerifier`.
    let Some(verifier) = crate::mac::exchange_ucan_verifier(state) else {
        return tx_error(
            StatusCode::FORBIDDEN,
            "server_error",
            "UCAN grant verification is not configured on this node",
        );
    };
    // MAC #547 / B2 (#674): the grant path is a security decision, and S7's
    // complete-mediation guarantee is only real if every decision is on the
    // audit trail. No sink configured ⇒ fail closed (deny) rather than mint
    // an unaudited token — mirroring `AuditedAvc`'s rule at the TE path.
    let Some(sink) = state.audit_sink.as_deref() else {
        return tx_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "MAC grant-path audit trail is not configured on this node",
        );
    };
    let now = chrono::Utc::now().timestamp().max(0) as u64;
    let decision = crate::mac::exchange::audited_evaluate_grant(
        &grant,
        &verifier,
        now,
        &request,
        subject_ctx.as_ref(),
        on_behalf_of.as_ref(),
        true, // sender-bound: dpop_jkt is present
        sink,
    );

    let granted = match decision {
        Ok(GrantDecision::Permit(g)) => g,
        Ok(GrantDecision::Escalate { .. }) => {
            // Over-ceiling: the escalation tier (TODO #572-escalation) is not
            // wired; return insufficient_scope. Do NOT auto-mint.
            return tx_error(
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                "grant request exceeds ceiling; escalation amendment required",
            );
        }
        Err(e) => return grant_error_response(e),
    };

    // ── 6. Mint the short-ttl sender-bound access token (ZSP) ──────────────
    // Persist the grant re-evaluation context so a refresh re-runs the S6 gate
    // chain (B1 #673) rather than free-re-minting. The grant CID binds the
    // stored blob to exactly this grant.
    let grant_refresh = super::state::UcanGrantRefresh {
        grant_cbor_b64: subject_token.to_owned(),
        grant_cid: blake3::hash(&ucan_bytes).to_hex().to_string(),
        requested_scope: requested_scope.map(str::to_owned),
        audience: audience.map(str::to_owned),
    };
    // #698 issuer path: thread the resolved clearance into the mint so the
    // token carries it for downstream PEP enforcement. `subject_ctx` is the
    // met context from resolve_grant_subject — its clearance is the
    // authority-assigned label the enrollment table holds for this subject.
    let subject_clearance = subject_ctx.map(|c| *c.clearance());
    mint_grant_token(
        state,
        &granted,
        &dpop_jkt,
        Some(grant_refresh),
        principals,
        subject_clearance,
    )
    .await
}

/// Re-evaluate a UCAN grant on refresh and re-mint (MAC #547 / B1 #673).
///
/// ZSP: a UCAN-grant refresh is NOT a free re-mint. The generic OAuth 2.1
/// refresh path would rotate the token without re-checking the grant and would
/// treat DPoP as optional; both break the S6 discipline. This path instead:
///
/// 1. requires a **fresh** DPoP proof (mandatory sender-binding — matching the
///    initial mint), and
/// 2. re-presents the persisted grant to [`crate::mac::exchange::audited_evaluate_refresh`], which runs the
///    same gate chain as mint against the *current* `now` and verifier state —
///    so a ceiling that has since been amended/revoked, or a grant that has
///    since expired, now denies.
///
/// The caller (`exchange_refresh_token`) has already atomically consumed
/// (rotated) the presented refresh token before delegating here. Every failure
/// is fail-closed. On permit, a new sender-bound access token + rotated refresh
/// token are minted, re-persisting the grant context for the next refresh.
pub(crate) async fn exchange_ucan_grant_refresh(
    state: &Arc<OAuthState>,
    ucan_grant: &super::state::UcanGrantRefresh,
    dpop_header: Option<&str>,
) -> Response {
    // 1. Fresh DPoP is MANDATORY (ZSP sender-binding) — same as the mint path.
    let token_endpoint = format!("{}/oauth/token", state.issuer_url.trim_end_matches('/'));
    let dpop_jkt = match dpop_header.and_then(|h| {
        super::dpop::verify_dpop_proof(h, "POST", &token_endpoint, None)
            .ok()
            .map(|p| p.jkt)
    }) {
        Some(jkt) => jkt,
        None => {
            return tx_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "UCAN grant refresh requires a valid DPoP proof (sender-binding)",
            );
        }
    };

    // 2. Re-present the persisted grant. Verify the stored blob's content id
    //    first — a corrupted/substituted grant fails closed.
    let ucan_bytes = match URL_SAFE_NO_PAD.decode(&ucan_grant.grant_cbor_b64) {
        Ok(b) => b,
        Err(_) => {
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "stored UCAN grant is not valid base64url",
            );
        }
    };
    if blake3::hash(&ucan_bytes).to_hex().to_string() != ucan_grant.grant_cid {
        return tx_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "stored UCAN grant failed its content-id check",
        );
    }
    let grant = match hyprstream_rpc::auth::ucan::token::Ucan::from_cbor(&ucan_bytes) {
        Ok(u) => u,
        Err(e) => {
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &format!("stored UCAN grant no longer decodes: {e}"),
            );
        }
    };

    // Rebuild the S6 request from the persisted requested access.
    let request = match parse_grant_request(
        ucan_grant.requested_scope.as_deref(),
        ucan_grant.audience.as_deref(),
    ) {
        Ok(r) => r,
        Err(msg) => return tx_error(StatusCode::BAD_REQUEST, "invalid_scope", &msg),
    };

    // Resolve the subject's MAC context exactly as the mint path does (#698
    // Decision D: enrollment-table resolver, actor assurance floored at
    // Classical), including the #680/#681 two-principal delegated meet —
    // refresh must not be more permissive than mint.
    let resolver = crate::mac::exchange_enrollment_resolver();
    let ResolvedGrantSubject {
        subject: subject_ctx,
        on_behalf_of,
        principals,
    } = resolve_grant_subject(&grant, resolver.as_ref());

    // The verifier: same trust-store-anchored construction as mint. Absent ⇒
    // fail closed (no unverified chain).
    let Some(verifier) = crate::mac::exchange_ucan_verifier(state) else {
        return tx_error(
            StatusCode::FORBIDDEN,
            "server_error",
            "UCAN grant verification is not configured on this node",
        );
    };
    // MAC #547 / B2 (#674): refresh is a grant-path decision exactly like mint
    // — it must be on the audit trail too, with the same fail-closed rule (no
    // sink configured ⇒ deny rather than re-mint unaudited).
    let Some(sink) = state.audit_sink.as_deref() else {
        return tx_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "MAC grant-path audit trail is not configured on this node",
        );
    };
    let now = chrono::Utc::now().timestamp().max(0) as u64;
    let decision = crate::mac::exchange::audited_evaluate_refresh(
        &grant,
        &verifier,
        now,
        &request,
        subject_ctx.as_ref(),
        on_behalf_of.as_ref(),
        true, // sender-bound: a fresh DPoP proof was just verified above
        sink,
    );

    let granted = match decision {
        Ok(GrantDecision::Permit(g)) => g,
        Ok(GrantDecision::Escalate { .. }) => {
            return tx_error(
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                "grant request exceeds ceiling; escalation amendment required",
            );
        }
        Err(e) => return grant_error_response(e),
    };

    // Re-mint + rotate, re-persisting the grant context for the next refresh.
    // #698: same clearance threading as mint — refresh must not be more
    // permissive (B1/#673).
    let subject_clearance = subject_ctx.map(|c| *c.clearance());
    mint_grant_token(
        state,
        &granted,
        &dpop_jkt,
        Some(ucan_grant.clone()),
        principals,
        subject_clearance,
    )
    .await
}

/// Map an S6 [`GrantError`] to a concrete OAuth 2.1 error response. Every variant
/// is fail-closed; no variant maps to a permissive outcome.
fn grant_error_response(e: GrantError) -> Response {
    let (status, code, desc) = match &e {
        GrantError::Chain(_) => (
            StatusCode::UNAUTHORIZED,
            "invalid_grant",
            "UCAN grant chain failed validation".to_owned(),
        ),
        // Over-ceiling and insufficient-clearance both surface as
        // `insufficient_scope`: the request is not within the grant/label the
        // subject is authorized for. Distinct GrantError variants (different
        // gates) but the same OAuth error shape for the client.
        GrantError::OverCeiling { .. } | GrantError::InsufficientClearance => {
            (StatusCode::FORBIDDEN, "insufficient_scope", e.to_string())
        }
        GrantError::MissingSenderBinding => {
            (StatusCode::BAD_REQUEST, "invalid_request", e.to_string())
        }
        GrantError::UnlabeledSubject | GrantError::EmptyGrant => {
            (StatusCode::FORBIDDEN, "invalid_grant", e.to_string())
        }
        // B2 (#674): a would-be Permit that could not be durably audited.
        // Fail-closed, not the client's fault — surfaced as server_error
        // (matches the "audit trail not configured" preflight response).
        GrantError::AuditUnavailable => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            e.to_string(),
        ),
    };
    tx_error(status, code, &desc)
}

/// Parse the RFC 8693 `scope` + `audience` form fields into the S6
/// [`GrantRequest`].
///
/// `scope` is the S3 `action:resource:identifier` triple (the requested access);
/// `audience` is the RFC 8707 resource indicator (optional). The object label
/// is `None` here (resolved separately — see TODO(#572-object-label)).
fn parse_grant_request(
    scope: Option<&str>,
    audience: Option<&str>,
) -> Result<GrantRequest, String> {
    use hyprstream_rpc::auth::ucan::capability::{Ability, Caveats, Resource};

    let scope_str = scope.ok_or_else(|| "scope is required for UCAN grant".to_owned())?;
    let parsed = hyprstream_rpc::auth::Scope::parse(scope_str)
        .map_err(|e| format!("invalid scope '{scope_str}': {e}"))?;
    Ok(GrantRequest {
        // S3 Scope(action, resource, identifier) → S5 Capability(resource, ability).
        // The `resource` URI is assembled as `mac://<resource>/<identifier>`;
        // `*` identifier maps to the wildcard. This mapping is the S3↔S5
        // vocabulary seam (deferred to #582); this is the conservative
        // structural projection.
        resource: Resource::new(format!("mac://{}/{}", parsed.resource, parsed.identifier)),
        ability: Ability::new(parsed.action),
        caveats: Caveats::default(),
        audience: audience.map(str::to_owned),
        object_label: None, // TODO(#572-object-label): manifest/TE resolver.
    })
}

/// Mint the short-ttl, sender-bound access token for a permitted grant.
///
/// ZSP: the token encodes the **requested subset** (the [`GrantedAccess`]),
/// never the whole grant. It is bound to the DPoP `jkt` (`cnf.jkt`) and carries
/// a short ttl. A refresh token is stored when a token DB is configured.
///
/// **S8 (#574) + Fu3/#677:** the minted token is signed with the **hybrid**
/// composite JWT signature (EdDSA + ML-DSA-65, `alg: "ML-DSA-65-Ed25519"`)
/// under a Hybrid policy, matching the hybrid signature on the UCAN grant and
/// approval it consumed. The minted token is the same kind of
/// confidentiality/integrity-critical authority artifact the rest of the MAC
/// stack signs hybridly. Under Hybrid policy with no provisioned ML-DSA-65 key
/// the mint **fails closed** (no silent classical downgrade); under Classical
/// policy the token is signed with the pinned classical Ed25519 suite.
async fn mint_grant_token(
    state: &Arc<OAuthState>,
    granted: &GrantedAccess,
    dpop_jkt: &str,
    grant_refresh: Option<super::state::UcanGrantRefresh>,
    principals: TokenPrincipals,
    // #698 issuer path: the clearance the enrollment resolver derived for
    // this subject, stamped on the minted token so downstream PEP lanes
    // (#1268/#1269) can enforce MAC without re-resolving from the enrollment
    // table. `None` (unenrolled/unverified) ⇒ no clearance claim ⇒ the
    // downstream `ClaimsSubjectContextResolver` denies (fail-closed).
    subject_clearance: Option<hyprstream_rpc::auth::mac::SecurityLabel>,
) -> Response {
    let now = chrono::Utc::now().timestamp();
    let ttl = state
        .token_ttl
        .min(crate::mac::exchange::MAX_ACCESS_TOKEN_TTL_SECS);
    let expires_at = now + ttl as i64;

    // #680/#681: the token subject is the DELEGATOR (source of authority) for a
    // delegated grant, or the subject for a root grant — always a UCAN principal
    // DID, never the RFC 8707 resource indicator (which stays on `aud`). For a
    // delegation, the actor (presenter that signs downstream) is carried in the
    // `act` claim below, so the token records "actor acting for delegator"
    // rather than collapsing to one identity (the confused-deputy fix).
    let sub = principals.sub.clone();

    // #1159 freeze: UCAN issuance signs directly instead of using
    // PolicyService, so it must enforce the same concrete-subject invariant at
    // its own mint-and-refresh-persistence boundary. Return before signing or
    // storing a refresh token so a legacy path-form chain cannot be extended.
    if is_path_form_did_web(&sub) {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)",
        );
    }

    let scope_str = format!(
        "{}@{}",
        granted.capability.ability, granted.capability.resource
    );

    // Fu1/#677 (was TODO(#572-scope-claim)): carry the attenuated capability
    // subset in the `cap` claim so the downstream PEP (S2) enforces the minted
    // least-authority on the wire, and a refresh can only re-grant this subset
    // — not just log it. The cnf.jkt binding + short ttl remain load-bearing.
    let mut claims = hyprstream_rpc::auth::Claims::new(sub.clone(), now, expires_at)
        .with_issuer(state.issuer_url.clone())
        .with_audience(granted.audience.clone())
        .with_cap(scope_str)
        .with_jti();
    // #680/#681: stamp the actor (delegate) so the two-principal delegation is
    // verifiable downstream (`meet(delegator, actor)`, assurance from the actor
    // who signs the envelope). Absent for a self-issued root grant.
    if let Some(actor) = principals.act {
        claims = claims.with_act(actor);
    }
    // #698 issuer path: stamp the authority-resolved clearance on the minted
    // token. This is the wire that was missing — the enrollment resolver
    // resolved DID→clearance for the S6 gate, then threw it away. Now the
    // minted JWT carries it under the hybrid signature, so a downstream PEP
    // (ClaimsSubjectContextResolver, #1268/#1269) reads it from verified
    // Claims without re-resolving from the enrollment table. `None`
    // (unenrolled) ⇒ no claim ⇒ downstream deny (fail-closed).
    if let Some(clearance) = subject_clearance {
        claims = claims.with_clearance(clearance);
    }

    // DPoP sender-binding via cnf.jkt (RFC 9449 §6). ZSP: no cnf ⇒ bearer ⇒
    // rejected. We set jkt directly from the verified proof.
    claims = claims.with_cnf_jkt_thumbprint(dpop_jkt.to_owned());

    // S8 (#574) + Fu3/#677: sign via the mandatory hybrid composite (EdDSA +
    // ML-DSA-65). If no ML-DSA-65 signing key is provisioned, refuse to mint
    // rather than silently downgrade to a classical-only token — mirroring
    // `PolicyService::sign_token` and
    // `CoseAuditSigner`. The UCAN grant and approval this token was minted from
    // are already hybrid-signed, so a classical-only minted token would break
    // that chain of hybrid authority.
    let snapshot = match hyprstream_rpc::auth::global_composite_key_set().mint_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::error!("composite authority unavailable or stale; refusing to mint: {error}");
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "hybrid token authority is not current",
            );
        }
    };
    let signing = snapshot
        .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::OAuth)
        .and_then(hyprstream_rpc::auth::CompositeKeyPair::signing_keys);
    let token = match signing {
        Some((pq, ed)) => crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(&claims, &pq, &ed),
        None => {
            tracing::error!(
                "no authorized active OAuth composite pair; refusing to mint (fail-closed)"
            );
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "hybrid token signing pair not provisioned",
            );
        }
    };

    // Optional refresh token (ZSP: refresh re-runs evaluate_refresh, not a free
    // re-mint). Stored only when a token DB is configured. B1 (#673): the grant
    // re-evaluation context is persisted with the refresh token so the refresh
    // path re-presents the grant to the S6 gate chain.
    let refresh_token = if state.token_db.is_some() {
        Some(issue_grant_refresh_token(state, &sub, expires_at, grant_refresh).await)
    } else {
        None
    };

    let mut body = serde_json::json!({
        "access_token": token,
        "issued_token_type": ISSUED_TOKEN_TYPE,
        "token_type": "DPoP", // sender-bound, not Bearer
        "expires_in": ttl,
    });
    if let Some(rt) = refresh_token {
        body["refresh_token"] = serde_json::Value::String(rt);
    }

    tracing::info!(sub = %sub, ttl, "UCAN grant token minted (sender-bound, short-ttl)");
    (
        StatusCode::OK,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        Json(body),
    )
        .into_response()
}

/// Issue an opaque refresh token for a UCAN grant, stored in the token DB.
///
/// The refresh token does NOT carry authority itself — it is a handle that lets
/// the presenter re-present the grant (which the refresh path re-validates via
/// `evaluate_refresh`). ZSP: refresh is re-evaluated, never automatic.
async fn issue_grant_refresh_token(
    state: &Arc<OAuthState>,
    sub: &str,
    access_expires_at: i64,
    grant_refresh: Option<super::state::UcanGrantRefresh>,
) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use rand::RngCore as _;

    // 256-bit opaque token.
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let refresh = URL_SAFE_NO_PAD.encode(bytes);

    // The verifying key binding for cnf continuity on refresh is re-established
    // by the re-presented DPoP proof at refresh time (jkt is a thumbprint, not
    // a raw key), so verifying_key_bytes stays None here.
    let _ = access_expires_at;
    let entry = super::state::RefreshTokenEntry {
        client_id: format!("ucan-grant:{sub}"),
        username: sub.to_owned(),
        scopes: vec!["urn:hyprstream:grant-type:ucan".to_owned()],
        resource: None,
        expires_at_unix: chrono::Utc::now().timestamp() + state.refresh_token_ttl as i64,
        verifying_key_bytes: None,
        dpop_jkt: None,
        client_assertion_jkt: None,
        ucan_grant: grant_refresh,
        session_id: None,
    };

    if let Some(db) = &state.token_db {
        if let Err(e) = db
            .put(&refresh, &entry, state.refresh_token_ttl as u64)
            .await
        {
            tracing::warn!(error = %e, "failed to persist UCAN grant refresh token");
        }
    }
    refresh
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! S8 (#574) activation tests for the concrete `SubjectContextResolver`.
    //! These cover the two input MAC context derivation (clearance from Claims
    //! plus assurance from verified key material) and the fail closed
    //! properties: anti borrow, classical key on PQ object denial, and
    //! unlabeled denial.
    use super::*;
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial,
    };

    #[test]
    fn atproto_ecdsa_accepts_p256_and_secp256k1_multikeys() {
        use p256::ecdsa::signature::Signer as _;

        let message = b"header.payload";
        let p256_key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let mut p256_multikey = vec![0x80, 0x24];
        p256_multikey.extend_from_slice(
            p256_key.verifying_key().to_encoded_point(true).as_bytes(),
        );
        let p256_signature: p256::ecdsa::Signature = p256_key.sign(message);
        verify_atproto_ecdsa(
            "ES256",
            &format!("z{}", bs58::encode(p256_multikey).into_string()),
            message,
            &URL_SAFE_NO_PAD.encode(p256_signature.to_bytes()),
        )
        .unwrap();

        let k256_key = k256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let mut k256_multikey = vec![0xe7, 0x01];
        k256_multikey.extend_from_slice(
            k256_key.verifying_key().to_encoded_point(true).as_bytes(),
        );
        let k256_signature: k256::ecdsa::Signature = k256_key.sign(message);
        verify_atproto_ecdsa(
            "ES256K",
            &format!("z{}", bs58::encode(k256_multikey).into_string()),
            message,
            &URL_SAFE_NO_PAD.encode(k256_signature.to_bytes()),
        )
        .unwrap();
    }

    #[test]
    fn exchange_tenant_requires_verified_local_binding() {
        let local_subject = VerifiedSubject {
            sub: "did:plc:abcdefghijklmnqrstuvwx2p".to_owned(),
            cnf_key_bytes: None,
            cnf_jkt: None,
            iat: 1,
            granted_scopes: None,
            verified_tenant: Some("tenant-source".to_owned()),
            atproto_replay: None,
            require_clearance: false,
            ttl_ceiling: None,
        };
        assert_eq!(
            exchange_tenant(&local_subject, None).unwrap().as_deref(),
            Some("tenant-source")
        );
        assert_eq!(
            exchange_tenant(&local_subject, Some("tenant-source"))
                .unwrap()
                .as_deref(),
            Some("tenant-source")
        );
        assert!(exchange_tenant(&local_subject, Some("tenant-other")).is_err());

        let external_enrolled_subject = VerifiedSubject {
            sub: "did:plc:externalenrolledsubject".to_owned(),
            cnf_key_bytes: None,
            cnf_jkt: None,
            iat: 1,
            granted_scopes: Some(vec!["transition:generic".to_owned()]),
            verified_tenant: None,
            atproto_replay: None,
            require_clearance: true,
            ttl_ceiling: Some(MAX_ATPROTO_EXCHANGE_TOKEN_TTL),
        };
        assert_eq!(
            exchange_tenant(&external_enrolled_subject, Some("tenant-foreign")),
            Err("subject token has no verified local tenant binding"),
            "an enrolled external DID cannot turn a client assertion into a local tenant binding"
        );
    }

    fn mint_test_state() -> Arc<OAuthState> {
        use crate::config::OAuthConfig;
        use crate::services::{DiscoveryClient, PolicyClient};
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x65; 32]);
        let dummy = std::path::PathBuf::from("/dev/null/path-form-mint-test.sock");
        let mk_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(key.clone()),
                    LazyUdsTransport::new(dummy.clone()),
                    Some(key.verifying_key()),
                )
                .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical),
            )
        };
        Arc::new(OAuthState::new(
            &OAuthConfig::default(),
            PolicyClient::new(mk_client()),
            DiscoveryClient::new(mk_client()),
            key.verifying_key().to_bytes(),
        ))
    }

    #[tokio::test]
    async fn ucan_mint_rejects_path_form_subject_before_signing() {
        use hyprstream_rpc::auth::ucan::{Ability, Capability, Resource};

        let granted = GrantedAccess {
            capability: Capability::new(Resource::new("mac://model/demo"), Ability::new("read")),
            audience: None,
        };
        let response = mint_grant_token(
            &mint_test_state(),
            &granted,
            "test-dpop-jkt",
            None,
            TokenPrincipals {
                sub: "did:web:accounts.example:users:alice".to_owned(),
                act: None,
            },
            // No clearance — this test checks the path-form freeze, not MAC.
            None,
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"].as_str(), Some("invalid_grant"));
        assert!(value["error_description"]
            .as_str()
            .unwrap()
            .contains("frozen"));
        assert!(value.get("access_token").is_none());
        assert!(value.get("refresh_token").is_none());
    }

    /// A compartment bitset from bit indices.
    fn comps(bits: &[u32]) -> CompartmentSet {
        bits.iter().copied().collect()
    }

    /// A PqHybrid-cleared subject mints a context whose assurance is PqHybrid.
    /// This is the activation: a fully-verified PQ subject gets a real context,
    /// not a denial.
    #[test]
    fn pqhybrid_subject_resolves_to_pqhybrid_context() {
        let did = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        let clearance = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, comps(&[0, 1]));
        let claims =
            hyprstream_rpc::auth::Claims::new("sub".to_owned(), 1, 2).with_clearance(clearance);
        let resolver =
            ClaimsSubjectContextResolver::new(did, claims, VerifiedKeyMaterial::PqHybrid);

        let ctx = resolver.resolve(did).expect("PqHybrid subject resolves");
        assert_eq!(ctx.assurance(), Assurance::PqHybrid);
        assert_eq!(ctx.level(), Level::Secret);
    }

    /// Anti-borrow: a grant whose audience DID differs from the one the
    /// clearance was issued to MUST be denied (`None`). The clearance cannot
    /// cross identities.
    #[test]
    fn resolver_denies_mismatched_audience_did() {
        let owner = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        let impostor = "did:key:z6MkmFpYUWaBjIA4ZJarQtz5FaGGCLpJ4xjXQqRuV4Dx4q6P";
        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let claims =
            hyprstream_rpc::auth::Claims::new("sub".to_owned(), 1, 2).with_clearance(clearance);
        let resolver =
            ClaimsSubjectContextResolver::new(owner, claims, VerifiedKeyMaterial::PqHybrid);

        assert!(
            resolver.resolve(impostor).is_none(),
            "a grant for a different audience MUST NOT borrow this subject's clearance"
        );
    }

    /// **Fail-closed (the #548 invariant at the resolver):** a Classical-key
    /// subject carrying a PqHybrid clearance MUST clamp to Classical assurance.
    /// The resolver does not grant assurance the key does not back.
    #[test]
    fn classical_key_clamps_pqhybrid_clearance_down() {
        let did = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        // Policy mistakenly assigned PqHybrid...
        let claimed = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let claims =
            hyprstream_rpc::auth::Claims::new("sub".to_owned(), 1, 2).with_clearance(claimed);
        // ...but the verified key is Classical:
        let resolver =
            ClaimsSubjectContextResolver::new(did, claims, VerifiedKeyMaterial::Classical);

        let ctx = resolver.resolve(did).expect("labeled subject resolves");
        assert_eq!(
            ctx.assurance(),
            Assurance::Classical,
            "Classical key must clamp a PqHybrid clearance down (no silent upgrade)"
        );

        // Consequently the MAC floor DENIES a PqHybrid object.
        let pq_object =
            SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        assert!(
            !ctx.can_access(&pq_object),
            "Classical-assurance subject MUST be denied on a PqHybrid object (fail-closed)"
        );
    }

    /// A subject with no clearance claim ⇒ `None` ⇒ the S1 monitor denies.
    #[test]
    fn unlabeled_subject_resolves_to_none() {
        let did = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        // No `with_clearance` ⇒ no clearance field.
        let claims = hyprstream_rpc::auth::Claims::new("sub".to_owned(), 1, 2);
        let resolver =
            ClaimsSubjectContextResolver::new(did, claims, VerifiedKeyMaterial::PqHybrid);

        assert!(
            resolver.resolve(did).is_none(),
            "unlabeled subject MUST resolve to None (S1 deny)"
        );
    }

    /// `DenyUnlabeledResolver` remains the explicit deny-by-default choice.
    #[test]
    fn deny_unlabeled_resolver_always_denies() {
        let r = DenyUnlabeledResolver;
        assert!(r.resolve("did:key:anything").is_none());
    }

    // ── #698: issuer path — enrollment → Claims.clearance → PEP resolver ────
    //
    // These tests prove the issuer path contract: the clearance the
    // enrollment resolver derives gets stamped on Claims (via
    // `with_clearance`, as `mint_grant_token` now does) and is readable by
    // the downstream `ClaimsSubjectContextResolver` the #1268/#1269 PEP lanes
    // consume. They exercise the real types end-to-end without the JWT signing
    // infrastructure (which is orthogonal — the Claims construction is what
    // carries the clearance, and it happens before signing).

    use crate::mac::{CompiledPolicy, EnrollmentSubjectContextResolver, TeMatrix};
    use hyprstream_rpc::auth::mac::{Lattice as LatticeType, LatticeVersion};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// Build a compiled policy with one enrolled DID at the given clearance.
    fn policy_with_enrollment(did: &str, clearance: SecurityLabel) -> Arc<CompiledPolicy> {
        let lattice = LatticeType::new(LatticeVersion(1), []);
        let policy = CompiledPolicy::new(TeMatrix::default(), &lattice)
            .with_enrollment(BTreeMap::from([(did.to_owned(), clearance)]));
        Arc::new(policy)
    }

    /// **Issuer path round-trip (happy path):** an enrolled DID resolves to a
    /// clearance via `EnrollmentSubjectContextResolver`, the issuer stamps it
    /// on Claims (exactly as `mint_grant_token` now does via `with_clearance`),
    /// and the downstream `ClaimsSubjectContextResolver` reads it back with the
    /// correct level and compartments. This is the full provenance chain the PEP
    /// lanes depend on.
    #[test]
    fn issuer_path_stamps_clearance_readable_by_pep_resolver() {
        let did = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        let enrolled = SecurityLabel::new(Level::Secret, Assurance::PqHybrid, comps(&[0, 1]));

        // Step 1: enrollment resolver derives the clearance (what the grant
        // path does via `exchange_enrollment_resolver()`).
        let resolver = EnrollmentSubjectContextResolver::new(policy_with_enrollment(did, enrolled));
        let ctx = resolver
            .resolve(did)
            .expect("enrolled DID must resolve to a SecurityContext");
        let clearance = *ctx.clearance();

        // Step 2: issuer stamps it on Claims (what `mint_grant_token` now does
        // via `claims.with_clearance(clearance)`).
        let claims =
            hyprstream_rpc::auth::Claims::new(did.to_owned(), 1, 9999).with_clearance(clearance);

        // Step 3: downstream PEP resolver reads it back from verified Claims.
        let pep_resolver =
            ClaimsSubjectContextResolver::new(did, claims, VerifiedKeyMaterial::Classical);
        let pep_ctx = pep_resolver
            .resolve(did)
            .expect("PEP resolver must read the stamped clearance");

        // The level + compartments survive the round-trip.
        assert_eq!(pep_ctx.level(), Level::Secret);
        assert_eq!(pep_ctx.compartments(), comps(&[0, 1]));
        // Decision D: assurance is Classical (the resolver floored it).
        assert_eq!(pep_ctx.assurance(), Assurance::Classical);
    }

    /// **Issuer path fail-closed:** an unenrolled DID resolves to `None` at the
    /// enrollment resolver, so the issuer passes `None` to `mint_grant_token`,
    /// so no clearance is stamped on Claims, so the downstream PEP resolver
    /// returns `None` → deny. This is the fail-closed chain the whole MAC model
    /// depends on.
    #[test]
    fn issuer_path_unenrolled_did_produces_no_clearance_pep_denies() {
        let enrolled_did = "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o";
        let stranger = "did:key:z6MkmFpYUWaBjIA4ZJarQtz5FaGGCLpJ4xjXQqRuV4Dx4q6P";
        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY);

        // The enrollment table only knows `enrolled_did`.
        let resolver =
            EnrollmentSubjectContextResolver::new(policy_with_enrollment(enrolled_did, clearance));

        // The stranger is NOT enrolled → None (fail-closed at the resolver).
        assert!(
            resolver.resolve(stranger).is_none(),
            "unenrolled DID must not resolve to a clearance"
        );

        // The issuer has no clearance to stamp (None) → Claims carry no clearance.
        let claims = hyprstream_rpc::auth::Claims::new(stranger.to_owned(), 1, 9999);
        assert!(
            claims.clearance.is_none(),
            "no clearance stamped for an unenrolled subject"
        );

        // The downstream PEP resolver denies.
        let pep_resolver =
            ClaimsSubjectContextResolver::new(stranger, claims, VerifiedKeyMaterial::PqHybrid);
        assert!(
            pep_resolver.resolve(stranger).is_none(),
            "PEP resolver MUST deny a subject with no stamped clearance"
        );
    }

    /// **Delegated grant — met clearance is what gets stamped:** for a delegated
    /// grant, `resolve_grant_subject` takes `meet(delegator, actor)`. The
    /// clearance stamped on the minted token is the MET clearance (the
    /// effective one), not the delegator's higher one — so the downstream PEP
    /// sees the least-privilege label.
    #[test]
    fn issuer_path_stamps_met_clearance_not_delegator_higher() {
        let mcp_did = "did:key:zMcp".to_owned();

        // Delegator: Secret / {0,1}. Actor: Confidential / {1}.
        // meet = Confidential / {1}.
        let user_clearance =
            SecurityLabel::new(Level::Secret, Assurance::Classical, comps(&[0, 1]));
        let mcp_clearance =
            SecurityLabel::new(Level::Confidential, Assurance::Classical, comps(&[1]));

        // Both enrolled in their own resolver (simulating two enrollment entries).
        let user_ctx =
            SecurityContext::from_clearance(user_clearance, VerifiedKeyMaterial::Classical);
        let mcp_ctx =
            SecurityContext::from_clearance(mcp_clearance, VerifiedKeyMaterial::Classical);
        let met = SecurityContext::delegated(Some(&user_ctx), Some(&mcp_ctx))
            .expect("both principals resolved");

        // The issuer stamps the MET clearance (what `exchange_ucan_grant`
        // passes as `subject_ctx.clearance()` to `mint_grant_token`).
        let stamped = *met.clearance();
        let claims =
            hyprstream_rpc::auth::Claims::new(mcp_did.clone(), 1, 9999).with_clearance(stamped);

        // The PEP resolver reads it back.
        let pep_resolver =
            ClaimsSubjectContextResolver::new(&mcp_did, claims, VerifiedKeyMaterial::Classical);
        let pep_ctx = pep_resolver
            .resolve(&mcp_did)
            .expect("met clearance resolves");

        // It's the meet (Confidential / {1}), NOT the delegator's Secret / {0,1}.
        assert_eq!(
            pep_ctx.level(),
            Level::Confidential,
            "met level, not delegator's"
        );
        assert_eq!(
            pep_ctx.compartments(),
            comps(&[1]),
            "met compartments, not delegator's"
        );
    }

    /// B1 (#673): a persisted refresh token from BEFORE this field existed (no
    /// `ucan_grant` key) MUST still deserialize — as `None` — so existing stored
    /// tokens keep working and are simply treated as generic (non-UCAN-grant)
    /// refresh tokens. Guards the `#[serde(default)]` on the new field.
    #[test]
    fn refresh_entry_without_ucan_grant_field_deserializes_as_none() {
        let legacy = r#"{
            "client_id": "abc",
            "username": "alice",
            "scopes": ["openid"],
            "resource": null,
            "expires_at_unix": 9999999999,
            "verifying_key_bytes": null
        }"#;
        let entry: super::super::state::RefreshTokenEntry =
            serde_json::from_str(legacy).expect("legacy refresh entry must still deserialize");
        assert!(
            entry.ucan_grant.is_none(),
            "a legacy entry is a generic refresh token, never a UCAN grant"
        );
    }

    /// B1 (#673): a UCAN-grant refresh entry round-trips through serde with its
    /// re-evaluation context intact — the grant blob, its content id, and the
    /// requested access the refresh path re-presents to `evaluate_refresh`.
    #[test]
    fn ucan_grant_refresh_entry_roundtrips() {
        let entry = super::super::state::RefreshTokenEntry {
            client_id: "ucan-grant:did:key:zAlice".to_owned(),
            username: "did:key:zAlice".to_owned(),
            scopes: vec!["urn:hyprstream:grant-type:ucan".to_owned()],
            resource: None,
            expires_at_unix: 9999999999,
            verifying_key_bytes: None,
            dpop_jkt: None,
            client_assertion_jkt: None,
            ucan_grant: Some(super::super::state::UcanGrantRefresh {
                grant_cbor_b64: "Zm9vYmFy".to_owned(),
                grant_cid: blake3::hash(b"the-grant").to_hex().to_string(),
                requested_scope: Some("read:model:llama".to_owned()),
                audience: Some("https://api.example".to_owned()),
            }),
            session_id: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: super::super::state::RefreshTokenEntry = serde_json::from_str(&json).unwrap();
        let ug = back.ucan_grant.expect("ucan_grant survives round-trip");
        assert_eq!(ug.grant_cbor_b64, "Zm9vYmFy");
        assert_eq!(
            ug.grant_cid,
            blake3::hash(b"the-grant").to_hex().to_string()
        );
        assert_eq!(ug.requested_scope.as_deref(), Some("read:model:llama"));
    }

    // ── #680/#681: two-principal delegated grant derivation ─────────────────

    use hyprstream_rpc::auth::mac::SecurityContext;
    use hyprstream_rpc::auth::ucan::token::{Did, Ucan, UcanPayload};
    use std::collections::HashMap;

    fn did(seed: u8) -> Did {
        Did::from_ed25519(&[seed; 32])
    }

    /// A minimal UCAN (payload only; `resolve_grant_subject` reads
    /// `audience()`/`root_issuer()` off the structure, not the signature).
    fn ucan(issuer: Did, audience: Did, proofs: Vec<Ucan>) -> Ucan {
        Ucan {
            payload: UcanPayload {
                issuer,
                audience,
                capabilities: vec![],
                not_before: None,
                expiration: Some(9_999_999_999),
                nonce: vec![],
            },
            proofs,
            signature: vec![],
        }
    }

    fn ctx(level: Level, bits: &[u32], km: VerifiedKeyMaterial) -> SecurityContext {
        SecurityContext::new(level, comps(bits), km)
    }

    /// A resolver that maps specific DIDs to contexts; everything else `None`.
    struct MapResolver(HashMap<String, SecurityContext>);
    impl SubjectContextResolver for MapResolver {
        fn resolve(&self, did: &str) -> Option<SecurityContext> {
            self.0.get(did).cloned()
        }
    }

    /// A self-issued root grant (issuer == audience) is single-principal: `sub`
    /// is the subject, no `act`, and the context is a plain `resolve(audience)`.
    #[test]
    fn root_grant_is_single_principal() {
        let subject = did(1);
        let subject_did = subject.as_str().to_owned();
        let grant = ucan(subject.clone(), subject, vec![]);
        let resolver = MapResolver(HashMap::from([(
            subject_did.clone(),
            ctx(Level::Secret, &[0], VerifiedKeyMaterial::PqHybrid),
        )]));

        let ResolvedGrantSubject {
            subject: context,
            on_behalf_of,
            principals,
        } = resolve_grant_subject(&grant, &resolver);
        assert!(principals.act.is_none(), "root grant has no actor");
        assert!(
            on_behalf_of.is_none(),
            "a single-principal grant records no delegator (audit on_behalf_of = None)"
        );
        assert_eq!(principals.sub, subject_did);
        assert_eq!(context.map(|c| c.level()), Some(Level::Secret));
    }

    /// A delegated grant (root issuer ≠ audience) splits into delegator (`sub`)
    /// and actor (`act`), and the derived context is the fail-closed
    /// `meet(delegator, actor)` — never either principal alone (#681).
    #[test]
    fn delegated_grant_splits_principals_and_meets() {
        let user = did(2); // delegator / source of authority
        let mcp = did(3); // actor / presenter
        let user_did = user.as_str().to_owned();
        let mcp_did = mcp.as_str().to_owned();
        // user delegates directly to mcp (issuer=user, audience=mcp).
        let grant = ucan(user, mcp, vec![]);
        let resolver = MapResolver(HashMap::from([
            // delegator: Secret / {0,1}
            (
                user_did.clone(),
                ctx(Level::Secret, &[0, 1], VerifiedKeyMaterial::PqHybrid),
            ),
            // actor: Confidential / {1,2}
            (
                mcp_did.clone(),
                ctx(Level::Confidential, &[1, 2], VerifiedKeyMaterial::PqHybrid),
            ),
        ]));

        let ResolvedGrantSubject {
            subject: context,
            on_behalf_of,
            principals,
        } = resolve_grant_subject(&grant, &resolver);
        assert_eq!(principals.sub, user_did, "sub is the delegator");
        let act = principals.act.expect("delegation carries an actor");
        assert_eq!(act.sub, mcp_did, "act is the actor");
        // meet = min level (Confidential) ∩ compartments ({1}).
        let context = context.expect("both principals resolved ⇒ Some");
        assert_eq!(context.level(), Level::Confidential, "min level");
        assert_eq!(
            context.compartments(),
            comps(&[1]),
            "compartment intersection"
        );
        // The audit `on_behalf_of` carries the DELEGATOR (the user's own
        // clearance), distinct from the met `subject` context above — this is
        // the two-principal attribution (#445/#681): the met context is what the
        // gates saw, the delegator is who the actor acted for.
        let obo = on_behalf_of.expect("a delegated grant records its delegator");
        assert_eq!(
            obo.level(),
            Level::Secret,
            "on_behalf_of is the delegator's own level"
        );
        assert_eq!(
            obo.compartments(),
            comps(&[0, 1]),
            "on_behalf_of is the delegator's own compartments, not the meet"
        );
    }

    /// The delegated derivation fails closed if EITHER principal is unresolved
    /// — no default clearance for a missing principal (#681 AC).
    #[test]
    fn delegated_grant_fails_closed_if_either_principal_unresolved() {
        let user = did(4);
        let mcp = did(5);
        let user_did = user.as_str().to_owned();
        let grant = ucan(user, mcp, vec![]);
        // Only the delegator resolves; the actor does not.
        let resolver = MapResolver(HashMap::from([(
            user_did,
            ctx(Level::Secret, &[0], VerifiedKeyMaterial::PqHybrid),
        )]));

        let ResolvedGrantSubject {
            subject: context,
            on_behalf_of,
            principals,
        } = resolve_grant_subject(&grant, &resolver);
        assert!(
            context.is_none(),
            "missing actor clearance ⇒ deny (no default clearance)"
        );
        // The principal split is still reported (the actor clearance is None),
        // but the None context makes evaluate_grant deny before any mint.
        assert!(principals.act.is_some());
        // The delegator resolved, so it is still available for audit even though
        // the decision itself fails closed on the unresolved actor.
        assert!(
            on_behalf_of.is_some(),
            "the resolved delegator is still recorded"
        );
    }

    /// Multi-hop delegation roots the delegator at the CHAIN ROOT issuer, not
    /// the nearest hop (RFC 8693 §4.1 — authority flows from the root).
    #[test]
    fn multihop_delegation_roots_at_chain_root() {
        let user = did(6); // ultimate authority (chain root)
        let intermediate = did(7);
        let mcp = did(8); // final presenter
        let root_proof = ucan(user.clone(), intermediate.clone(), vec![]);
        let leaf = ucan(intermediate, mcp.clone(), vec![root_proof]);

        // No clearances resolved — we only assert the principal identities.
        let resolver = MapResolver(HashMap::new());
        let ResolvedGrantSubject {
            on_behalf_of,
            principals,
            ..
        } = resolve_grant_subject(&leaf, &resolver);
        assert!(
            on_behalf_of.is_none(),
            "no clearance resolved ⇒ no delegator context (fails closed on subject anyway)"
        );
        assert_eq!(
            principals.sub,
            user.as_str(),
            "delegator is the chain-root issuer, not the intermediate"
        );
        assert_eq!(
            principals.act.expect("delegation").sub,
            mcp.as_str(),
            "actor is the final audience/presenter"
        );
    }
}
