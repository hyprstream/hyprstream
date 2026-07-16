//! RFC 8693 OAuth Token Exchange grant handler.
//!
//! Grant type: `urn:ietf:params:oauth:grant-type:token-exchange`
//!
//! Exchanges an existing credential (OIDC ID token, at+jwt, or WIT) for a
//! hyprstream at+jwt. Serves as the HTTP-layer complement to ExchangeWit (ZMQ)
//! and enables the MCP SDK's CrossAppAccessProvider enterprise flow.

use std::sync::Arc;

use axum::{
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use sha2::{Digest, Sha256};

use super::state::OAuthState;
use crate::mac::exchange::{GrantDecision, GrantError, GrantRequest, GrantedAccess};
use crate::services::generated::policy_client::IssueToken;

const TOKEN_TYPE_ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
const TOKEN_TYPE_ACCESS_TOKEN: &str = "urn:ietf:params:oauth:token-type:access_token";
const TOKEN_TYPE_JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
const ISSUED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

struct VerifiedSubject {
    sub: String,
    cnf_key_bytes: Option<[u8; 32]>,
    iat: i64,
}

/// POST /oauth/token — token-exchange grant (RFC 8693).
pub async fn exchange_token_exchange(
    state: &Arc<OAuthState>,
    subject_token: &str,
    subject_token_type: &str,
    audience: Option<&str>,
    scope: Option<&str>,
    actor_token: Option<&str>,
    requested_token_type: Option<&str>,
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
        TOKEN_TYPE_ACCESS_TOKEN => verify_access_token(state, subject_token),
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

    // Replay prevention: SHA-256 of the subject token as the replay key.
    // Covers all token types, regardless of whether they carry a jti claim.
    let token_hash = URL_SAFE_NO_PAD.encode(Sha256::digest(subject_token.as_bytes()));
    if !state.check_and_record_dpop_jti(&token_hash, verified.iat) {
        return tx_error(
            StatusCode::BAD_REQUEST,
            "invalid_grant",
            "subject_token already used (replay)",
        );
    }

    let requested_scopes: Option<Vec<String>> =
        scope.map(|s| s.split_whitespace().map(str::to_owned).collect());

    // Encode cnf key bytes for PolicyService (same path as user_pub_key in other flows).
    let user_pub_key = verified.cnf_key_bytes.map(|b| URL_SAFE_NO_PAD.encode(b));

    let result = state
        .policy_client
        .issue_token(&IssueToken {
            requested_scopes,
            ttl: Some(state.token_ttl),
            audience: audience.map(str::to_owned),
            subject: Some(verified.sub.clone()),
            user_pub_key,
            dpop_jkt: None,
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
        sub: claims.sub,
        cnf_key_bytes: None, // ID tokens carry no key binding
        iat: claims.iat,
    })
}

/// Verify an existing hyprstream at+jwt (downscoping / audience narrowing).
fn verify_access_token(state: &OAuthState, token: &str) -> Result<VerifiedSubject, String> {
    let vk = ed25519_dalek::VerifyingKey::from_bytes(&state.verifying_key_bytes)
        .map_err(|_| "server configuration error: invalid verifying key".to_owned())?;

    let claims = hyprstream_rpc::auth::jwt::decode(token, &vk, None)
        .map_err(|e| format!("access_token verification failed: {e}"))?;

    let cnf_key_bytes = claims.cnf_key_bytes();
    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes,
        iat: claims.iat,
    })
}

/// Verify a generic JWT — WIT from local trust store or federated OIDC issuer.
///
/// For `sub: service:*`: global trust store (CA-signed WIT).
/// For other subjects: issuer must be in `trusted_issuers`.
/// Audience must equal the token endpoint URL (same constraint as RFC 7523).
async fn verify_jwt(state: &Arc<OAuthState>, token: &str) -> Result<VerifiedSubject, String> {
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

    let vk = if sub.starts_with("service:") {
        let svc_name = sub.trim_start_matches("service:");
        hyprstream_service::global_trust_store()
            .resolve_one(svc_name)
            .ok_or_else(|| format!("Unknown service in trust store: {svc_name}"))?
    } else {
        let cfg = state
            .trusted_issuers
            .get(&iss)
            .ok_or_else(|| format!("Issuer not in trusted_issuers allow-list: {iss}"))?
            .clone();
        super::jwt_bearer::resolve_federated_key(state, &iss, token, cfg.allow_http)
            .await
            .map_err(|e| format!("JWKS key resolution failed for {iss}: {e}"))?
    };

    // Full verify with audience = token endpoint (RFC 7523 §3).
    let claims = hyprstream_rpc::auth::decode_with_key(token, &vk, Some(&token_endpoint))
        .map_err(|e| format!("JWT verification failed: {e}"))?;

    let cnf_key_bytes = claims.cnf_key_bytes(); // carry cnf.jwk through (WIT key binding)
    Ok(VerifiedSubject {
        sub: claims.sub,
        cnf_key_bytes,
        iat: claims.iat,
    })
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
            clearance: actor_ctx.map(|c| *c.clearance()),
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
    mint_grant_token(state, &granted, &dpop_jkt, Some(grant_refresh), principals).await
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
    mint_grant_token(state, &granted, &dpop_jkt, Some(ucan_grant.clone()), principals).await
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
        GrantError::AuditUnavailable => {
            (StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string())
        }
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
    // DPoP sender-binding via cnf.jkt (RFC 9449 §6). ZSP: no cnf ⇒ bearer ⇒
    // rejected. We set jkt directly from the verified proof.
    claims.cnf = Some(hyprstream_rpc::auth::Cnf {
        jwk: None,
        jkt: Some(dpop_jkt.to_owned()),
    });

    // S8 (#574) + Fu3/#677: sign via the hybrid composite (EdDSA + ML-DSA-65)
    // under a Hybrid policy; under Classical use the pinned classical Ed25519
    // suite. **Hybrid fails closed:** if the node is Hybrid but no ML-DSA-65
    // signing key is provisioned, refuse to mint rather than silently downgrade
    // to a classical-only token — mirroring `PolicyService::sign_token` and
    // `CoseAuditSigner`. The UCAN grant and approval this token was minted from
    // are already hybrid-signed, so a classical-only minted token would break
    // that chain of hybrid authority.
    let policy = hyprstream_rpc::envelope::envelope_policy_from_env();
    let token = if policy.uses_pq() {
        let snapshot = match hyprstream_rpc::auth::global_composite_key_set().mint_snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::error!("composite authority unavailable or stale; refusing to mint: {error}");
                return tx_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "hybrid token authority is not current");
            }
        };
        let signing = snapshot
            .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::OAuth)
            .and_then(hyprstream_rpc::auth::CompositeKeyPair::signing_keys);
        match signing {
            Some((pq, ed)) => crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
                &claims, &pq, &ed,
            ),
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
        }
    } else {
        let Some(ed_sk) = state.active_jwt_signing_key().await else {
            tracing::error!("no active JWT signing key configured; cannot mint UCAN grant token");
            return tx_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "token signing not configured",
            );
        };
        crate::auth::jwt::encode(&claims, &ed_sk)
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

#[cfg(test)]
pub(crate) async fn mint_grant_token_through_production_path() -> anyhow::Result<String> {
    use hyprstream_rpc::auth::ucan::{Ability, Capability, Resource};

    let client_key = hyprstream_rpc::prelude::SigningKey::from_bytes(&[0x6b; 32]);
    let processor: Arc<dyn hyprstream_rpc::transport::rpc_session::IrohRequestProcessor> =
        Arc::new(hyprstream_rpc::transport::rpc_session::from_fn(|_| async {
            Ok(bytes::Bytes::new())
        }));
    hyprstream_rpc::dial::register_inproc("multiprocess-unused-policy", &processor);
    hyprstream_rpc::dial::register_inproc("multiprocess-unused-discovery", &processor);
    let policy_client = crate::services::PolicyClient::for_endpoint(
        "inproc://multiprocess-unused-policy",
        client_key.clone(),
        client_key.verifying_key(),
        None,
    )?;
    let discovery_client = crate::services::DiscoveryClient::for_endpoint(
        "inproc://multiprocess-unused-discovery",
        client_key,
        ed25519_dalek::SigningKey::from_bytes(&[0x6c; 32]).verifying_key(),
        None,
    )?;
    let mut config = crate::config::OAuthConfig::default();
    config.external_url = Some("https://multiprocess.test".to_owned());
    config.token_ttl_seconds = 60;
    let state = Arc::new(OAuthState::new(
        &config,
        policy_client,
        discovery_client,
        [0x6d; 32],
    ));
    let granted = GrantedAccess {
        capability: Capability::new(
            Resource::new("mac://multiprocess/resource"),
            Ability::new("read"),
        ),
        audience: Some("multiprocess".to_owned()),
    };
    let envelope_signer = ed25519_dalek::SigningKey::from_bytes(&[0x75; 32]);
    let dpop_jkt = hyprstream_rpc::auth::jwk_thumbprint(
        &hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 {
            x: &envelope_signer.verifying_key().to_bytes(),
        },
    );
    let response = mint_grant_token(
        &state,
        &granted,
        &dpop_jkt,
        None,
        TokenPrincipals {
            sub: "did:key:multiprocess-oauth".to_owned(),
            act: None,
        },
    )
    .await;
    anyhow::ensure!(
        response.status() == StatusCode::OK,
        "OAuth production mint returned {}",
        response.status()
    );
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await?;
    let json: serde_json::Value = serde_json::from_slice(&body)?;
    json.get("access_token")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("OAuth production response omitted access_token"))
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
        ucan_grant: grant_refresh,
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
            ucan_grant: Some(super::super::state::UcanGrantRefresh {
                grant_cbor_b64: "Zm9vYmFy".to_owned(),
                grant_cid: blake3::hash(b"the-grant").to_hex().to_string(),
                requested_scope: Some("read:model:llama".to_owned()),
                audience: Some("https://api.example".to_owned()),
            }),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: super::super::state::RefreshTokenEntry = serde_json::from_str(&json).unwrap();
        let ug = back.ucan_grant.expect("ucan_grant survives round-trip");
        assert_eq!(ug.grant_cbor_b64, "Zm9vYmFy");
        assert_eq!(ug.grant_cid, blake3::hash(b"the-grant").to_hex().to_string());
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
        assert_eq!(context.compartments(), comps(&[1]), "compartment intersection");
        // The audit `on_behalf_of` carries the DELEGATOR (the user's own
        // clearance), distinct from the met `subject` context above — this is
        // the two-principal attribution (#445/#681): the met context is what the
        // gates saw, the delegator is who the actor acted for.
        let obo = on_behalf_of.expect("a delegated grant records its delegator");
        assert_eq!(obo.level(), Level::Secret, "on_behalf_of is the delegator's own level");
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
        assert!(on_behalf_of.is_some(), "the resolved delegator is still recorded");
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
            on_behalf_of, principals, ..
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
