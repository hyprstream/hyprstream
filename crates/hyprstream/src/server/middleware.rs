//! Middleware for authentication, logging, and request processing

use crate::auth::jwt;
use crate::server::state::ServerState;
use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use hyprstream_rpc::auth::JtiBlocklist as _;
use std::time::Instant;
use std::time::Duration;
use subtle::ConstantTimeEq as _;
use tracing::{debug, info, warn};

/// Authenticated identity extracted from token
#[derive(Clone)]
pub struct AuthenticatedUser {
    /// Username (from JWT sub claim)
    pub user: String,
    /// Original JWT token for e2e verification through service chain.
    /// SECURITY: Never logged — custom Debug impl redacts this field.
    pub token: Option<String>,
    /// JWT expiration timestamp (from the validated JWT claims).
    pub exp: Option<i64>,
}

impl std::fmt::Debug for AuthenticatedUser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthenticatedUser")
            .field("user", &self.user)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("exp", &self.exp)
            .finish()
    }
}

/// JWT authentication middleware
///
/// Validates JWT tokens via Ed25519 signature verification.
/// Accepts `Authorization: Bearer <token>` and `Authorization: DPoP <token>`.
///
/// When the token carries a `cnf.jkt` claim (DPoP-bound token per RFC 9449),
/// the `Authorization: DPoP` scheme MUST be used and a valid `DPoP` proof
/// header MUST accompany the request — plain Bearer is rejected to prevent
/// token replay after theft.
///
/// Revoked tokens (via `POST /oauth/revoke`) are rejected via JTI blocklist.
///
/// On success, inserts `AuthenticatedUser` into request extensions.
pub async fn auth_middleware(
    State(state): State<ServerState>,
    mut request: Request,
    next: Next,
) -> Response {
    let client_ip = extract_client_ip(&request);
    let method = request.method().as_str().to_owned();
    let path = request.uri().path().to_owned();

    // Accept both Bearer and DPoP schemes (RFC 6750, RFC 9449)
    let auth_hdr = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let (scheme, token) = match auth_hdr.as_deref().and_then(|h| {
        if h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer ") {
            Some(("bearer", h[7..].trim().to_owned()))
        } else if h.len() > 5 && h[..5].eq_ignore_ascii_case("dpop ") {
            Some(("dpop", h[5..].trim().to_owned()))
        } else {
            None
        }
    }) {
        Some(pair) => pair,
        None => {
            let www_authenticate = build_www_authenticate(&state);
            return unauthorized_response("Authentication required", &www_authenticate);
        }
    };

    // Build WWW-Authenticate header for 401 responses (RFC 9728)
    let www_authenticate = build_www_authenticate(&state);

    // Verify the token to validated claims via the shared auth chain
    // (malformed check, local-vs-federation issuer routing, signature +
    // audience validation, JTI revocation). Reused verbatim by the 9P mount
    // ticket validator (H1a / #764). DPoP sender-binding is enforced below,
    // after we hold the claims, because it is header/scheme-specific.
    let local_issuers: &[&str] = &[&*state.oauth_issuer_url];
    let claims = match verify_token_claims(&state, &token).await {
        Ok(c) => c,
        Err(reason) => {
            warn!(client_ip = %client_ip, method = %method, path = %path, reason, "Auth failure");
            return unauthorized_response("Authentication failed", &www_authenticate);
        }
    };

    // DPoP binding enforcement (RFC 9449).
    // A token with cnf.jkt MUST be presented with Authorization: DPoP + DPoP proof header.
    // Accepting it as Bearer would let a stolen token replay without proof of key possession.
    if let Some(expected_jkt) = claims.cnf_jkt() {
        if scheme != "dpop" {
            warn!(sub = %claims.sub, "DPoP-bound token presented with Bearer scheme — rejected");
            return unauthorized_response("Authentication failed", &www_authenticate);
        }
        let dpop_proof = match request.headers().get("DPoP").and_then(|v| v.to_str().ok()) {
            Some(p) => p.to_owned(),
            None => {
                debug!("DPoP-bound token missing DPoP proof header");
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
        };
        let method = request.method().as_str().to_owned();
        let path = request.uri().path().to_owned();
        // Build absolute htu from the server's resource_url (scheme + host) + request path.
        // Axum doesn't populate scheme/host in the URI for HTTPS behind a TLS terminator,
        // so we use the pre-configured resource_url as the authority base.
        let htu = format!("{}{}", state.resource_url.trim_end_matches('/'), path);
        let proof = match crate::services::oauth::dpop::verify_dpop_proof(
            &dpop_proof,
            &method,
            &htu,
            Some(&token),
        ) {
            Ok(p) => p,
            Err(e) => {
                debug!("DPoP proof verification failed: {}", e);
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
        };
        // Replay prevention: each DPoP jti is accepted at most once (within
        // iat ±60s window). Atomic check-and-record on the shared TtlCache.
        {
            let now = chrono::Utc::now().timestamp();
            let ttl_secs = ((proof.iat + 120) - now).max(0) as u64;
            if !state
                .dpop_jti_seen
                .insert_if_absent(proof.jti.clone(), (), Duration::from_secs(ttl_secs))
            {
                debug!("DPoP proof jti already used: {}", proof.jti);
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
        }
        // cnf.jkt must match the DPoP proof key thumbprint.
        if expected_jkt.as_bytes().ct_eq(proof.jkt.as_bytes()).unwrap_u8() == 0 {
            warn!(sub = %claims.sub, "cnf.jkt mismatch — DPoP proof key does not match token binding");
            return unauthorized_response("Authentication failed", &www_authenticate);
        }
    }

    debug!("JWT validated for user: {}", claims.sub);
    let subject = claims.subject(local_issuers);
    // Validate local subjects for safe characters. Federated subjects
    // (containing "://") bypass this check — they are validated at JWT decode time.
    if let Err(e) = subject.validate() {
        debug!("JWT sub validation failed: {}", e);
        return unauthorized_response("Authentication failed", &www_authenticate);
    }
    let user_str = match subject.name() {
        Some(n) => n.to_owned(),
        None => {
            debug!("JWT has empty sub");
            return unauthorized_response("Authentication failed", &www_authenticate);
        }
    };
    let user = AuthenticatedUser {
        user: user_str,
        token: Some(token),
        exp: Some(claims.exp),
    };
    request.extensions_mut().insert(user);
    next.run(request).await
}

/// Request logging middleware
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    if status.is_client_error() || status.is_server_error() {
        warn!("{} {} {} ({:?})", method, uri, status, duration);
    } else {
        info!("{} {} {} ({:?})", method, uri, status, duration);
    }

    response
}

/// Fixed-window per-subject rate limiter.
///
/// Tracks request counts per authenticated subject (JWT sub).
/// The window resets cleanly at the end of each window period.
pub struct RateLimiter {
    /// subject → (request_count, window_start_unix)
    counters: parking_lot::RwLock<std::collections::HashMap<String, (u32, i64)>>,
    max_requests: u32,
    window_secs: i64,
}

impl RateLimiter {
    pub fn new(max_requests: u32, window_secs: i64) -> Self {
        Self {
            counters: parking_lot::RwLock::new(std::collections::HashMap::new()),
            max_requests,
            window_secs,
        }
    }

    /// Check + increment. Returns `true` if the caller is over quota.
    pub fn check_and_increment(&self, subject: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let mut map = self.counters.write();
        // Evict stale entries to bound memory growth
        if map.len() > 10_000 {
            let window_secs = self.window_secs;
            map.retain(|_, (_, ws)| now - *ws < window_secs * 2);
        }
        let entry = map.entry(subject.to_owned()).or_insert((0, now));
        if now - entry.1 >= self.window_secs {
            // New window: reset
            *entry = (1, now);
            false
        } else if entry.0 >= self.max_requests {
            true // over quota
        } else {
            entry.0 += 1;
            false
        }
    }
}

/// Rate limiting middleware — enforces per-authenticated-subject request quotas.
///
/// Runs after `auth_middleware` (which inserts `AuthenticatedUser`).
/// Defaults: 300 requests per 60-second window (~5 req/s sustained).
/// Returns 429 when the window quota is exceeded.
pub async fn rate_limit_middleware(
    State(state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    let subject = request
        .extensions()
        .get::<AuthenticatedUser>()
        .map(|u| u.user.clone())
        .unwrap_or_else(|| "anonymous".to_owned());

    if state.rate_limiter.check_and_increment(&subject) {
        warn!(subject = %subject, "Rate limit exceeded");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded — retry after window expires",
        )
            .into_response();
    }

    next.run(request).await
}

/// Verify a bearer/ticket token to its validated [`jwt::Claims`] using the
/// shared server auth chain.
///
/// This is the token-verification core factored out of [`auth_middleware`] so
/// the 9P mount-ticket validator (H1a / #764) reuses the *exact* same chain
/// rather than re-implementing one:
///
/// 1. malformed-token rejection,
/// 2. local-vs-federation issuer routing (local `verifying_key` or
///    `federation_resolver`),
/// 3. Ed25519 / ML-DSA signature + audience validation via `jwt::decode`,
/// 4. JTI revocation against the shared blocklist (RFC 7009).
///
/// It does **not** enforce DPoP sender-binding: that is header/scheme-specific
/// and stays in [`auth_middleware`]. Returns `Err(reason)` with a short,
/// log-safe reason string on any failure (never leaks token contents).
/// RFC 7638 JWK thumbprint (the JWKS `kid`) for an Ed25519 verifying key.
fn ed25519_kid(key: &ed25519_dalek::VerifyingKey) -> String {
    hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 {
        x: key.as_bytes(),
    })
}

/// Verify a locally-issued JWT against the node's full published key set: the
/// cluster CA key plus every currently-published Ed25519 rotation slot (the same
/// key set `/oauth/jwks` publishes). Accepts the token if ANY of these trusted,
/// node-published keys verifies it — standard JWKS/`kid`-rotation semantics.
///
/// Security invariant: only keys the node itself publishes are ever tried. A
/// token-supplied (`jwk`/embedded) key is NEVER admitted; the `kid` header is
/// used only as an ordering *hint* to try the matching published key first, and
/// falls through to the other trusted keys. Audience validation is unchanged —
/// `jwt::decode` still enforces strict local-issuer audience matching.
fn decode_local_multi_key(
    token: &str,
    ca_key: &ed25519_dalek::VerifyingKey,
    published_keys: &[ed25519_dalek::VerifyingKey],
    expected_aud: Option<&str>,
) -> Result<jwt::Claims, &'static str> {
    // Candidate set: CA key first, then all published rotation slots. This is the
    // exact set the JWKS endpoint publishes; nothing else is trusted.
    let mut candidates: Vec<&ed25519_dalek::VerifyingKey> =
        Vec::with_capacity(1 + published_keys.len());
    candidates.push(ca_key);
    candidates.extend(published_keys.iter());

    // If the JOSE header carries a kid, try the matching published key first
    // (fast path for the common rotation case). This is a stable reordering
    // only — every trusted candidate is still attempted, so a token whose kid
    // is absent/mismatched but which is validly signed by a published key still
    // verifies. The kid is the RFC 7638 JWK thumbprint the JWKS `kid` uses.
    if let Some(token_kid) = extract_kid_from_token(token) {
        candidates.sort_by_key(|k| u8::from(ed25519_kid(k) != token_kid));
    }

    for key in candidates {
        if let Ok(claims) = jwt::decode(token, key, expected_aud) {
            return Ok(claims);
        }
    }
    Err("JWT validation failed")
}

pub(crate) async fn verify_token_claims(
    state: &ServerState,
    token: &str,
) -> Result<jwt::Claims, &'static str> {
    if !token.contains('.') {
        return Err("malformed token");
    }

    // Extract iss for key routing (local node key vs trusted federation peer).
    let iss = extract_iss_from_token(token);
    let local_issuers: &[&str] = &[&*state.oauth_issuer_url];
    let claims = if hyprstream_rpc::auth::is_local_iss(&iss, local_issuers) {
        // Local-issuer tokens may be signed by EITHER the cluster CA key OR any
        // currently-published rotation slot (the OAuth token endpoint signs with
        // `active_jwt_signing_key()` → the rotation active slot). Validate against
        // the same key set the /oauth/jwks endpoint publishes (CA + rotation
        // slots), accepting if ANY trusted, published key verifies (#777).
        let published = {
            let guard = state
                .published_jwt_verifying_keys
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.clone()
        };
        decode_local_multi_key(
            token,
            &state.verifying_key,
            &published,
            Some(&state.resource_url),
        )?
    } else {
        if !state.federation_resolver.is_trusted(&iss) {
            return Err("untrusted federation issuer");
        }
        match state.federation_resolver.get_key(&iss).await {
            Ok(key) => jwt::decode_with_key(token, &key, Some(&state.resource_url))
                .map_err(|_| "JWT validation failed")?,
            Err(_) => return Err("federation key resolution failed"),
        }
    };

    // JTI revocation check (RFC 7009) — shared blocklist with the OAuth
    // revocation endpoint.
    if let Some(ref jti) = claims.jti {
        if state.jti_blocklist.is_revoked(jti) {
            return Err("revoked token");
        }
    }

    Ok(claims)
}

/// Build WWW-Authenticate header value with resource_metadata URL (RFC 9728).
fn build_www_authenticate(state: &ServerState) -> String {
    let resource_metadata_url = format!(
        "{}/.well-known/oauth-protected-resource",
        state.resource_url
    );
    format!(
        "Bearer resource_metadata=\"{}\"",
        resource_metadata_url,
    )
}

/// Return a 401 response with WWW-Authenticate header.
fn unauthorized_response(message: &str, www_authenticate: &str) -> Response {
    let mut response = (StatusCode::UNAUTHORIZED, message.to_owned()).into_response();
    if let Ok(val) = HeaderValue::from_str(www_authenticate) {
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            val,
        );
    }
    response
}

/// Extract the `iss` claim from a JWT payload without signature verification.
/// Exported for use in other middleware-like contexts (e.g. MCP inline auth).
/// Returns an empty string on any parse failure.
pub fn extract_iss_from_token(token: &str) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() < 2 {
        return String::new();
    }
    // Guard against adversarially large payloads (real JWT payloads are < 1KB)
    if parts[1].len() > 8192 {
        return String::new();
    }
    let payload_bytes = match URL_SAFE_NO_PAD.decode(parts[1]) {
        Ok(b) => b,
        Err(_) => return String::new(),
    };
    let payload: serde_json::Value = match serde_json::from_slice(&payload_bytes) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    payload.get("iss")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned()
}

/// Extract `kid` from a JWT header without full validation.
pub fn extract_kid_from_token(token: &str) -> Option<String> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let header_b64 = token.split('.').next()?;
    if header_b64.len() > 4096 {
        return None;
    }
    let header_bytes = URL_SAFE_NO_PAD.decode(header_b64).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).ok()?;
    header.get("kid").and_then(|v| v.as_str()).map(str::to_owned)
}

/// Extract the client IP from `X-Forwarded-For` or `X-Real-IP` headers.
/// Falls back to `"unknown"` when no header is present (e.g. direct connection
/// where ConnectInfo is not available in this middleware form).
fn extract_client_ip(request: &Request) -> String {
    if let Some(xff) = request.headers().get("x-forwarded-for") {
        if let Ok(s) = xff.to_str() {
            // Take the leftmost (original client) IP
            if let Some(ip) = s.split(',').next() {
                return ip.trim().to_owned();
            }
        }
    }
    if let Some(xri) = request.headers().get("x-real-ip") {
        if let Ok(s) = xri.to_str() {
            return s.trim().to_owned();
        }
    }
    "unknown".to_owned()
}

/// CORS middleware configuration
pub fn cors_layer(config: &crate::server::state::CorsConfig) -> tower_http::cors::CorsLayer {
    use axum::http::{HeaderName, HeaderValue, Method};
    use tower_http::cors::{Any, CorsLayer};

    let mut cors = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .max_age(std::time::Duration::from_secs(config.max_age));

    // Configure allowed headers
    if config.permissive_headers {
        // Permissive mode — allow any request header. Deliberately scoped to the
        // public, secret-free DID-document routes (CorsConfig::did_document); must
        // not be enabled for the broad public router. Logged at warn! so the
        // relaxed posture stays auditable to operators.
        cors = cors.allow_headers(Any);
        warn!("⚠️  CORS: allowing ALL request headers (permissive_headers=true)");
    } else {
        // Explicit mode - only allow specific headers for security
        cors = cors.allow_headers([
            // Standard HTTP headers
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::USER_AGENT,
            header::ACCEPT,
            header::ACCEPT_LANGUAGE,
            header::ACCEPT_ENCODING,
            header::REFERER,
            header::ORIGIN,
            header::CONNECTION,
            // OpenAI SDK headers (x-stainless-*)
            HeaderName::from_static("x-stainless-arch"),
            HeaderName::from_static("x-stainless-lang"),
            HeaderName::from_static("x-stainless-os"),
            HeaderName::from_static("x-stainless-package-version"),
            HeaderName::from_static("x-stainless-retry-count"),
            HeaderName::from_static("x-stainless-runtime"),
            HeaderName::from_static("x-stainless-runtime-version"),
            HeaderName::from_static("x-stainless-timeout"),
            // Additional OpenAI headers
            HeaderName::from_static("openai-organization"),
            HeaderName::from_static("openai-project"),
            HeaderName::from_static("openai-beta"),
            // Request ID tracking
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("x-trace-id"),
        ]);
    }

    // Configure allowed origins
    if config.allowed_origins.contains(&"*".to_owned()) {
        // Allow all origins with Any (handles wildcard properly)
        cors = cors.allow_origin(Any);
        // Never allow credentials with wildcard
        cors = cors.allow_credentials(false);
    } else if config.allowed_origins.is_empty() {
        // If no origins specified, use default localhost origins
        let default_origins = vec![
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ];
        for origin in default_origins {
            if let Ok(header_value) = origin.parse::<HeaderValue>() {
                cors = cors.allow_origin(header_value);
            }
        }
        // Configure credentials based on config
        if config.allow_credentials {
            cors = cors.allow_credentials(true);
        }
    } else {
        // Allow specific origins
        for origin in &config.allowed_origins {
            if let Ok(header_value) = origin.parse::<HeaderValue>() {
                cors = cors.allow_origin(header_value);
            }
        }
        // Configure credentials based on config
        if config.allow_credentials {
            cors = cors.allow_credentials(true);
        }
    }

    cors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_iss_from_token() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

        // Craft a JWT with iss in payload: header.payload.sig
        // payload = base64url({"iss":"https://node-a","sub":"alice","exp":9999999999,"iat":0})
        let payload = URL_SAFE_NO_PAD.encode(
            r#"{"iss":"https://node-a","sub":"alice","exp":9999999999,"iat":0}"#
        );
        let token = format!("eyJ0eXAiOiJKV1QiLCJhbGciOiJFZERTQSJ9.{}.fakesig", payload);
        assert_eq!(extract_iss_from_token(&token), "https://node-a");

        // Token without iss
        let payload2 = URL_SAFE_NO_PAD.encode(
            r#"{"sub":"alice","exp":9999999999,"iat":0}"#
        );
        let token2 = format!("eyJhbGciOiJFZERTQSJ9.{}.fakesig", payload2);
        assert_eq!(extract_iss_from_token(&token2), "");

        // Garbage input
        assert_eq!(extract_iss_from_token("notavalidtoken"), "");
    }
}

/// Rotation/JWKS-aware local-token validation (#777).
///
/// Proves `decode_local_multi_key` accepts a token signed by ANY currently
/// published key (CA or rotation slot) and rejects unpublished keys / wrong
/// audience, and that the shared jti-revocation check still rejects a
/// validly-decoded token.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod rotation_aware_tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    const AUD: &str = "https://node-a/resource";

    fn now() -> i64 {
        chrono::Utc::now().timestamp()
    }

    fn new_key() -> SigningKey {
        SigningKey::generate(&mut rand::rngs::OsRng)
    }

    /// Build a locally-issued at+JWT signed by `key`, with the given audience and
    /// optional explicit jti (`jwt::encode` auto-assigns a random jti otherwise).
    fn signed_token(key: &SigningKey, aud: &str, jti: Option<&str>) -> String {
        let n = now();
        let mut claims = jwt::Claims::new("alice".to_owned(), n, n + 3600)
            .with_audience(Some(aud.to_owned()));
        if let Some(j) = jti {
            claims.jti = Some(j.to_owned());
        }
        jwt::encode(&claims, key)
    }

    #[test]
    fn rotation_active_slot_verifies() {
        // Token signed by a rotation slot that is NOT the CA key — this is the
        // exact case that 401'd before #777.
        let ca = new_key();
        let rotation = new_key();
        let token = signed_token(&rotation, AUD, None);
        let published = vec![rotation.verifying_key()];

        let claims =
            decode_local_multi_key(&token, &ca.verifying_key(), &published, Some(AUD)).unwrap();
        assert_eq!(claims.sub, "alice");
    }

    #[test]
    fn ca_key_signed_token_still_verifies() {
        // CA-signed token must still verify — even with no rotation slots published.
        let ca = new_key();
        let token = signed_token(&ca, AUD, None);

        let claims = decode_local_multi_key(&token, &ca.verifying_key(), &[], Some(AUD)).unwrap();
        assert_eq!(claims.sub, "alice");
    }

    #[test]
    fn unpublished_random_key_rejected() {
        // A token signed by a key the node never published must be rejected — the
        // core security invariant (only trusted, published keys are ever tried).
        let ca = new_key();
        let rotation = new_key();
        let attacker = new_key();
        let token = signed_token(&attacker, AUD, None);
        let published = vec![rotation.verifying_key()];

        let err = decode_local_multi_key(&token, &ca.verifying_key(), &published, Some(AUD))
            .unwrap_err();
        assert_eq!(err, "JWT validation failed");
    }

    #[test]
    fn wrong_audience_rejected() {
        // Correct (published) key but wrong audience → still rejected. Strict
        // local-issuer audience validation is unchanged.
        let ca = new_key();
        let rotation = new_key();
        let token = signed_token(&rotation, "https://evil/other", None);
        let published = vec![rotation.verifying_key()];

        let err = decode_local_multi_key(&token, &ca.verifying_key(), &published, Some(AUD))
            .unwrap_err();
        assert_eq!(err, "JWT validation failed");
    }

    #[test]
    fn revoked_jti_still_rejected() {
        // A rotation-signed token decodes cleanly, but the shared jti-blocklist
        // check `verify_token_claims` performs after decode still rejects it.
        use hyprstream_rpc::auth::{InMemoryJtiBlocklist, JtiBlocklist as _};

        let ca = new_key();
        let rotation = new_key();
        let token = signed_token(&rotation, AUD, Some("jti-777"));
        let published = vec![rotation.verifying_key()];

        let claims =
            decode_local_multi_key(&token, &ca.verifying_key(), &published, Some(AUD)).unwrap();
        assert_eq!(claims.jti.as_deref(), Some("jti-777"));

        let blocklist = InMemoryJtiBlocklist::new();
        blocklist.revoke("jti-777".to_owned(), now() + 3600);
        // This mirrors the exact post-decode check in `verify_token_claims`.
        assert!(blocklist.is_revoked(claims.jti.as_deref().unwrap()));
    }

    #[test]
    fn kid_hint_finds_rotation_slot_behind_ca() {
        // The rotation slot is listed after the CA candidate; the kid-preference
        // reorder must still locate and verify against it (it also verifies
        // without any kid, since every candidate is tried).
        let ca = new_key();
        let rotation = new_key();
        let token = signed_token(&rotation, AUD, None);
        // Token header carries the rotation key's RFC 7638 thumbprint as kid.
        assert_eq!(
            extract_kid_from_token(&token).as_deref(),
            Some(ed25519_kid(&rotation.verifying_key()).as_str()),
        );
        let published = vec![rotation.verifying_key()];

        let claims =
            decode_local_multi_key(&token, &ca.verifying_key(), &published, Some(AUD)).unwrap();
        assert_eq!(claims.sub, "alice");
    }
}
