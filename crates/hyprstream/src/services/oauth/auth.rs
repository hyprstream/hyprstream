//! Bearer token authentication middleware for the OAuth service.
//!
//! Applied as an Axum middleware layer to the protected route group (SCIM
//! user management). Public endpoints (discovery, token, authorize) bypass
//! this layer.
//!
//! On success, inserts `AuthenticatedUser` into request extensions so
//! downstream handlers can identify the caller.

use std::sync::Arc;

use subtle::ConstantTimeEq;
use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

use super::state::OAuthState;
pub use crate::server::middleware::AuthenticatedUser;

/// Require a valid Bearer or DPoP token on the request.
///
/// Routes JWT verification by the protected `alg` header, then validates the
/// exact OAuth-self audience and one of this node's exact profile issuers.
/// When `Authorization: DPoP` is used, also verifies the `DPoP` proof header and
/// checks that `cnf.jkt` in the token matches the proof key's thumbprint.
/// Inserts `AuthenticatedUser` into request extensions on success.
/// Returns 401 if the token is absent, expired, or has an invalid signature.
pub async fn require_bearer_token(
    State(state): State<Arc<OAuthState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let (scheme, token) = match auth_header.as_deref().and_then(|h| {
        if let Some(t) = h.strip_prefix("Bearer ") { Some(("Bearer", t.to_owned())) }
        else if let Some(t) = h.strip_prefix("DPoP ") { Some(("DPoP", t.to_owned())) }
        else { None }
    }) {
        Some(pair) => pair,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "Bearer token required"})),
            )
                .into_response();
        }
    };

    // For DPoP, also grab the DPoP proof header.
    let dpop_proof_str = if scheme == "DPoP" {
        match request.headers().get("DPoP").and_then(|v| v.to_str().ok()) {
            Some(p) => Some(p.to_owned()),
            None => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "DPoP proof header required when using DPoP scheme"})),
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    let claims = match validate_oauth_access_token(&state, &token).await {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "Token invalid or expired"})),
            )
                .into_response();
        }
    };

    // DPoP-bound tokens (cnf.jkt) MUST NOT be accepted as plain Bearer (RFC 9449 §7).
    // A stolen DPoP-bound token presented as Bearer bypasses key-possession binding.
    if claims.cnf_jkt().is_some() && dpop_proof_str.is_none() {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "DPoP error=\"invalid_token\"")],
            axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "DPoP-bound token must use Authorization: DPoP scheme"})),
        )
            .into_response();
    }

    // For DPoP: verify the proof and validate cnf.jkt matches the proof key.
    if let Some(ref proof_str) = dpop_proof_str {
        let method = request.method().as_str();
        let uri = request.uri();
        // RFC 9449 htu is the absolute target URI without query or fragment.
        // Axum server requests normally carry origin-form URIs, so reconstruct
        // the authority from the operator-configured canonical OAuth origin.
        let htu = format!("{}{}", state.atproto_issuer_url(), uri.path());
        let proof = match super::dpop::verify_dpop_proof(proof_str, method, &htu, Some(&token)) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_dpop_proof", "error_description": e.to_string()})),
                )
                    .into_response();
            }
        };
        // JTI replay prevention for resource requests (RFC 9449 §11.1).
        if !state.check_and_record_dpop_jti(&proof.jti, proof.iat) {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                axum::Json(serde_json::json!({"error": "invalid_dpop_proof", "error_description": "DPoP proof jti already used"})),
            )
                .into_response();
        }

        // RFC 9449 §8 nonce enforcement (resource server).
        // Once this jkt has been issued a nonce by us, all subsequent proofs
        // must include a valid server-issued nonce.
        let needs_nonce = state.dpop_client_requires_nonce(&proof.jkt).await;
        let nonce_ok = match proof.nonce.as_deref() {
            Some(n) => state.verify_dpop_nonce(n).await,
            None => false,
        };
        if needs_nonce && !nonce_ok {
            let fresh = state.issue_dpop_nonce().await;
            state.mark_dpop_client_nonced(&proof.jkt).await;
            let mut resp = (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "DPoP error=\"use_dpop_nonce\"")],
                axum::Json(serde_json::json!({
                    "error": "use_dpop_nonce",
                    "error_description": "DPoP proof must include a server-issued nonce",
                })),
            )
                .into_response();
            if let Ok(val) = axum::http::HeaderValue::from_str(&fresh) {
                resp.headers_mut().insert("DPoP-Nonce", val);
            }
            return resp;
        }
        if proof.nonce.is_some() && !nonce_ok {
            // Presented nonce but it was bogus — same rejection.
            let fresh = state.issue_dpop_nonce().await;
            state.mark_dpop_client_nonced(&proof.jkt).await;
            let mut resp = (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "DPoP error=\"use_dpop_nonce\"")],
                axum::Json(serde_json::json!({
                    "error": "use_dpop_nonce",
                    "error_description": "DPoP nonce invalid or expired",
                })),
            )
                .into_response();
            if let Ok(val) = axum::http::HeaderValue::from_str(&fresh) {
                resp.headers_mut().insert("DPoP-Nonce", val);
            }
            return resp;
        }
        // Verify cnf.jkt in token matches the DPoP proof key.
        if let Some(token_jkt) = claims.cnf_jkt() {
            if token_jkt.as_bytes().ct_eq(proof.jkt.as_bytes()).unwrap_u8() == 0 {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer error=\"invalid_token\"")],
                    axum::Json(serde_json::json!({"error": "invalid_token", "error_description": "cnf.jkt does not match DPoP proof key"})),
                )
                    .into_response();
            }
        }
    }

    let user = AuthenticatedUser {
        user: claims.sub,
        verified_tenant: claims.tenant,
        token: Some(token),
        exp: Some(claims.exp),
    };
    if user.authorization_domain().is_err() {
        return (
            StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "insufficient_scope",
                "error_description": "Verified hosted-account tenant binding required"
            })),
        )
            .into_response();
    }
    request.extensions_mut().insert(user);

    next.run(request).await
}

/// Restrict cross-tenant OAuth administration to a tenantless service authority.
///
/// This layer runs inside [`require_bearer_token`], after the verified identity
/// has been inserted. Tenant-bound users may use self-service OAuth routes, but
/// cannot enumerate or mutate the global SCIM store or mint service workload
/// identities. Introspection has its own same-tenant check.
pub async fn require_global_service_authority(request: Request, next: Next) -> Response {
    let authorized = request
        .extensions()
        .get::<AuthenticatedUser>()
        .is_some_and(|user| user.authorization_domain().as_deref() == Ok("*"));
    if !authorized {
        return (
            StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "insufficient_scope",
                "error_description": "global service authority required"
            })),
        )
            .into_response();
    }
    next.run(request).await
}

/// Verify a JWT access token against the OAuth server's complete local signing
/// surface. PolicyService mints composite tokens, while classical EdDSA remains
/// accepted for explicitly classical deployments and legacy rotation slots.
///
/// Audience and issuer are mandatory here. The canonical origin is the OAuth
/// server's RFC 9728 resource identifier, while the configured issuer remains
/// an accepted local audience alias for default-audience tokens minted before
/// the resource indicator is known. No other audience is accepted.
pub(super) async fn validate_oauth_access_token(
    state: &OAuthState,
    token: &str,
) -> Result<hyprstream_rpc::auth::Claims, &'static str> {
    let claims = verify_access_token(state, token, AudiencePolicy::Exact).await?;

    // Fail-closed revocation: if the token carries a jti but no store is
    // configured, reject — a potentially-revoked credential must not be
    // admitted when revocation status cannot be checked. The credential
    // profile makes jti REQUIRED on locally issued tokens (issuer locality
    // is enforced above), so a missing jti is a hard rejection, not a skip.
    match claims.jti.as_deref() {
        Some(jti) => {
            let cred_id = hyprstream_rpc::auth::CredentialId::jwt(&claims.iss, jti);
            match hyprstream_rpc::auth::global_credential_revocation_store() {
                Some(store) => {
                    if store.is_revoked(&cred_id).await {
                        return Err("JWT revoked");
                    }
                }
                None => return Err("revocation store unavailable"),
            }
        }
        None => return Err("local credential missing required jti"),
    }

    // Session check (v16 §3.3): a token carrying a session ID is rejected
    // when the session is revoked, expired, unknown, or cannot be checked.
    let session_key = match claims.session_key() {
        Ok(key) => key,
        Err(_) => return Err("malformed session claims"),
    };
    if let Some(session_key) = session_key {
        match hyprstream_rpc::auth::global_session_registry() {
            Some(registry) => {
                if registry.is_revoked(&session_key).await {
                    return Err("session revoked");
                }
            }
            None => return Err("session registry unavailable"),
        }
    }
    Ok(claims)
}

/// Audience policy for [`verify_access_token`].
///
/// - `Exact`: the token's `aud` must match the resource server's canonical
///   audience or issuer URL. Used by resource-server auth (OAI, MCP).
/// - `AnyLocalIssuer`: the token's `aud` may be any RFC 8707 resource
///   audience; the issuer must still be local. Used by the revocation
///   endpoint (RFC 7009), which must be able to revoke resource-audience
///   tokens, not just OAuth-self-audience tokens.
pub(super) enum AudiencePolicy {
    Exact,
    AnyLocalIssuer,
}

/// One signature/profile/local-issuer verification stack with caller-
/// selected audience policy. Both [`validate_oauth_access_token`] (resource
/// auth, exact audience) and the revocation endpoint (any local-issuer
/// resource audience) call this function. No unverified authority
/// derivation: signature, typ, and local-issuer checks are always enforced.
pub(super) async fn verify_access_token(
    state: &OAuthState,
    token: &str,
    audience_policy: AudiencePolicy,
) -> Result<hyprstream_rpc::auth::Claims, &'static str> {
    let header = hyprstream_rpc::auth::parse_protected_header(token)
        .map_err(|_| "JWT header invalid")?;
    if !hyprstream_rpc::auth::is_rfc9068_access_token_type(&header.typ) {
        return Err("JWT type invalid");
    }

    let canonical_audience = state.atproto_issuer_url();
    let unverified = hyprstream_rpc::auth::decode_unverified(token)
        .map_err(|_| "JWT claims invalid")?;

    // Audience policy dispatch. Both paths enforce a non-empty audience.
    let expected_audience = match (&audience_policy, unverified.aud.as_deref()) {
        (AudiencePolicy::Exact, Some(audience))
            if audience == canonical_audience || audience == state.issuer_url.as_str() =>
        {
            audience.to_owned()
        }
        (AudiencePolicy::AnyLocalIssuer, Some(audience)) if !audience.is_empty() => {
            audience.to_owned()
        }
        _ => return Err("JWT audience invalid"),
    };

    let claims = match header.alg.as_str() {
        "ML-DSA-65-Ed25519" => {
            let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(
                token,
                hyprstream_rpc::auth::RFC9068_ACCESS_TOKEN_TYPES,
            )
            .map_err(|_| "composite JWT dispatch invalid")?;
            let snapshot = hyprstream_rpc::auth::global_composite_key_set().snapshot();
            let pair = snapshot
                .pair(dispatch.kid())
                .ok_or("composite JWT kid unknown")?;
            hyprstream_rpc::auth::jwt::decode_composite(
                token,
                pair.ml_dsa(),
                pair.ed25519(),
                Some(&expected_audience),
                &dispatch,
            )
            .map_err(|_| "composite JWT validation failed")?
        }
        "EdDSA" => {
            let key = state
                .jwt_bearer_verifying_key()
                .await
                .ok_or("EdDSA verification key unavailable")?;
            hyprstream_rpc::auth::jwt::decode(token, &key, Some(&expected_audience))
                .map_err(|_| "EdDSA JWT validation failed")?
        }
        _ => return Err("JWT algorithm unsupported"),
    };

    // Local-issuer enforcement is always applied, regardless of audience
    // policy. A federated token is never accepted through either path.
    let issuer_is_local = claims.iss == canonical_audience || claims.iss == state.issuer_url;
    if !issuer_is_local {
        return Err("JWT issuer invalid");
    }
    Ok(claims)
}
