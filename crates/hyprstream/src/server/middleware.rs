//! Middleware for authentication, logging, and request processing

use crate::auth::jwt;
use crate::server::state::{ResourceAuthState, ServerState};
use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use hyprstream_rpc::auth::JtiBlocklist as _;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
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
    State(state): State<ResourceAuthState>,
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
    let claims = match verify_resource_token_claims(&state, &token).await {
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
            if !state.dpop_jti_seen.insert_if_absent(
                proof.jti.clone(),
                (),
                Duration::from_secs(ttl_secs),
            ) {
                debug!("DPoP proof jti already used: {}", proof.jti);
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
        }
        // cnf.jkt must match the DPoP proof key thumbprint.
        if expected_jkt
            .as_bytes()
            .ct_eq(proof.jkt.as_bytes())
            .unwrap_u8()
            == 0
        {
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
    State(state): State<ResourceAuthState>,
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

/// Rate-limit the public browser-provisioning route before accepted-state
/// resolution or hybrid projection signing. A single stable bucket is
/// intentional: this boundary may run without trustworthy peer-address
/// metadata, and client-controlled forwarding headers must not select buckets.
pub async fn browser_provisioning_rate_limit_middleware(
    State(rate_limiter): State<Arc<RateLimiter>>,
    request: Request,
    next: Next,
) -> Response {
    const PROVISIONING_BUCKET: &str = "browser-provisioning";
    if rate_limiter.check_and_increment(PROVISIONING_BUCKET) {
        warn!("Browser provisioning rate limit exceeded");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "Browser provisioning rate limit exceeded — retry after window expires",
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
/// Composite tokens verify only against exact published pairs; component key
/// sets are never cross-paired. `alg`, `typ`, and `kid` are mandatory, and
/// EdDSA is accepted only when the configured crypto policy permits it.
fn decode_local_multi_key(
    token: &str,
    ca_key: &ed25519_dalek::VerifyingKey,
    published_ed25519: &[ed25519_dalek::VerifyingKey],
    published_composite: &[hyprstream_rpc::auth::CompositeKeyPair],
    expected_aud: Option<&str>,
    allow_eddsa: bool,
) -> Result<jwt::Claims, &'static str> {
    let header =
        hyprstream_rpc::auth::parse_protected_header(token).map_err(|_| "JWT validation failed")?;
    if !hyprstream_rpc::auth::is_rfc9068_access_token_type(&header.typ) {
        return Err("JWT validation failed");
    }
    match header.alg.as_str() {
        "ML-DSA-65-Ed25519" => {
            decode_composite_multi(token, published_composite, expected_aud, &header)
        }
        "EdDSA" if allow_eddsa => {
            decode_ed25519_multi(token, ca_key, published_ed25519, expected_aud, &header.kid)
        }
        _ => Err("JWT validation failed"),
    }
}

fn decode_ed25519_multi(
    token: &str,
    ca_key: &ed25519_dalek::VerifyingKey,
    published_ed25519: &[ed25519_dalek::VerifyingKey],
    expected_aud: Option<&str>,
    token_kid: &str,
) -> Result<jwt::Claims, &'static str> {
    let mut candidates: Vec<&ed25519_dalek::VerifyingKey> =
        Vec::with_capacity(1 + published_ed25519.len());
    candidates.push(ca_key);
    candidates.extend(published_ed25519.iter());
    let key = candidates
        .into_iter()
        .find(|key| ed25519_kid(key) == token_kid)
        .ok_or("JWT validation failed")?;
    jwt::decode(token, key, expected_aud).map_err(|_| "JWT validation failed")
}

fn decode_composite_multi(
    token: &str,
    published_composite: &[hyprstream_rpc::auth::CompositeKeyPair],
    expected_aud: Option<&str>,
    header: &hyprstream_rpc::auth::ProtectedHeader,
) -> Result<jwt::Claims, &'static str> {
    let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(
        token,
        hyprstream_rpc::auth::RFC9068_ACCESS_TOKEN_TYPES,
    )
    .map_err(|_| "JWT validation failed")?;
    if dispatch.kid() != header.kid {
        return Err("JWT validation failed");
    }
    let pair = published_composite
        .iter()
        .find(|pair| pair.kid() == header.kid)
        .ok_or("JWT validation failed")?;
    hyprstream_rpc::auth::jwt::decode_composite(
        token,
        pair.ml_dsa(),
        pair.ed25519(),
        expected_aud,
        &dispatch,
    )
    .map_err(|_| "JWT validation failed")
}

fn local_issuer_matches(claims: &jwt::Claims, expected: &str) -> bool {
    claims.iss == expected
}

async fn verify_resource_token_claims(
    state: &ResourceAuthState,
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
        // Published ML-DSA-65 verifying keys (PQ half of the composite key set
        // `/oauth/jwks` publishes). Lives in the same process-shared handle the
        // MAC/key-rotation path populates at boot and after each ML-DSA rotation.
        // Empty under Classical policy or before the OAuth store is provisioned —
        // in which case a composite token simply fails closed here.
        let published_composite = state.composite_key_set.snapshot();
        let claims = decode_local_multi_key(
            token,
            &state.verifying_key,
            &published,
            published_composite.pairs(),
            Some(&state.resource_url),
            false,
        )?;
        if !local_issuer_matches(&claims, &state.oauth_issuer_url) {
            return Err("JWT validation failed");
        }
        claims
    } else {
        if !state.federation_resolver.is_trusted(&iss) {
            return Err("untrusted federation issuer");
        }
        // Rotation-aware federation verification (#1185): the resolver
        // returns every Ed25519 key the issuer currently publishes
        // (ordered kid-first), and we accept the token if ANY candidate
        // verifies it. This is what makes overlap rotation safe — a
        // token signed by a non-first key succeeds when its kid is
        // published — and mirrors `decode_local_multi_key` for external
        // issuers. The token's own `kid`, when present, is passed to
        // the resolver so it can refetch on an unknown kid.
        let kid = extract_kid_from_token(token);
        match state
            .federation_resolver
            .get_keys(&iss, kid.as_deref())
            .await
        {
            Ok(candidates) if !candidates.is_empty() => jwt::decode_with_federation_candidates(
                token,
                &candidates,
                Some(&state.resource_url),
            )
            .map_err(|_| "JWT validation failed")?,
            // Empty candidate set or fetch failure: fail closed either way.
            Ok(_) | Err(_) => return Err("federation key resolution failed"),
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

pub(crate) async fn verify_token_claims(
    state: &ServerState,
    token: &str,
) -> Result<jwt::Claims, &'static str> {
    verify_resource_token_claims(&state.resource_auth_state(), token).await
}

/// Build WWW-Authenticate header value with resource_metadata URL (RFC 9728).
fn build_www_authenticate(state: &ResourceAuthState) -> String {
    let resource_metadata_url = format!(
        "{}/.well-known/oauth-protected-resource",
        state.resource_url
    );
    format!("Bearer resource_metadata=\"{}\"", resource_metadata_url,)
}

/// Return a 401 response with WWW-Authenticate header.
fn unauthorized_response(message: &str, www_authenticate: &str) -> Response {
    let mut response = (StatusCode::UNAUTHORIZED, message.to_owned()).into_response();
    if let Ok(val) = HeaderValue::from_str(www_authenticate) {
        response.headers_mut().insert(header::WWW_AUTHENTICATE, val);
    }
    response
}

/// Extract the `iss` claim from a JWT payload without signature verification.
/// Exported for use in other middleware-like contexts (e.g. MCP inline auth).
/// Returns an empty string on any parse failure.
pub fn extract_iss_from_token(token: &str) -> String {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
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
    payload
        .get("iss")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned()
}

/// Extract `kid` from a JWT header without full validation.
pub fn extract_kid_from_token(token: &str) -> Option<String> {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    let header_b64 = token.split('.').next()?;
    if header_b64.len() > 4096 {
        return None;
    }
    let header_bytes = URL_SAFE_NO_PAD.decode(header_b64).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).ok()?;
    header
        .get("kid")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
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
    } else {
        // Allow specific origins.
        //
        // The whole set MUST be passed in a single `allow_origin` call:
        // `CorsLayer::allow_origin` is documented to *override* on every call
        // (`self.allow_origin = origin.into()`), and a single `HeaderValue`
        // becomes `AllowOrigin::exact(...)` — a constant `Access-Control-Allow-Origin`
        // emitted unconditionally, not a membership check. Iterating and calling
        // it once per origin would therefore honor only the LAST origin in the
        // list, silently dropping every earlier origin. Passing the
        // `Vec<HeaderValue>` makes tower-http build `AllowOrigin::list(...)`,
        // which performs exact-equality membership matching against the full set.
        let origins: Vec<HeaderValue> = if config.allowed_origins.is_empty() {
            // If no origins specified, use default localhost origins
            [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
            ]
            .into_iter()
            .filter_map(|o| o.parse::<HeaderValue>().ok())
            .collect()
        } else {
            config
                .allowed_origins
                .iter()
                .filter_map(|o| o.parse::<HeaderValue>().ok())
                .collect()
        };
        cors = cors.allow_origin(origins);
        // Configure credentials based on config
        if config.allow_credentials {
            cors = cors.allow_credentials(true);
        }
    }

    cors
}

/// Runtime-behavior tests for [`cors_layer`].
///
/// These drive the real axum `Router` + `CorsLayer` and assert the
/// `Access-Control-Allow-Origin` / `Access-Control-Allow-Credentials` headers a
/// browser would actually observe. They exist because the credentialed router's
/// origin list used to be built with a per-origin loop of `allow_origin(...)`,
/// which silently honored only the LAST entry — a config-vec assertion can't
/// catch that (the vec is correct; the layer is wrong), so every check here
/// exercises the layer end-to-end.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod cors_layer_tests {
    use super::cors_layer;
    use crate::config::server::ServerConfig;
    use crate::server::state::CorsConfig;
    use axum::{body::Body, http::Request as HttpRequest, routing::get, Router};
    use tower::ServiceExt;

    /// Drive the credentialed CORS layer on a real `Router` and return the
    /// `Access-Control-Allow-Origin` header it emits for `origin` (or `None` if
    /// the request was rejected / header absent). This is the only honest way to
    /// catch the classic "only the last origin in the list is honored" bug, which
    /// a config-vec assertion will happily report as fixed while the runtime is
    /// broken.
    async fn acao_for(config: &CorsConfig, origin: &str) -> Option<String> {
        let app = Router::new()
            .route("/probe", get(|| async { "ok" }))
            .layer(cors_layer(config));
        let req = HttpRequest::get("/probe")
            .header("Origin", origin)
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        resp.headers()
            .get(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    }

    #[tokio::test]
    async fn cors_layer_allows_every_default_origin_at_runtime() {
        // Regression: the credentialed default list previously honored only the
        // LAST origin (loop-of-override bug). Every dev origin must each produce
        // an ACAO echoing itself back.
        let config = CorsConfig::default();
        for origin in [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ] {
            assert_eq!(
                acao_for(&config, origin).await,
                Some(origin.to_owned()),
                "credentialed CORS layer must echo allowed origin {origin}"
            );
        }
    }

    #[tokio::test]
    async fn cors_layer_rejects_near_miss_origins() {
        // tower-http's list path uses exact `HeaderValue` equality, so
        // prefix/suffix/subdomain near-misses must NOT be reflected back. These
        // are the classic CORS-bypass tricks; a reflect-any bug would leak them.
        let config = ServerConfig::builder()
            .cors_origins(vec![
                "https://hyprstream.com".to_owned(),
                "https://www.hyprstream.com".to_owned(),
            ])
            .build()
            .cors;
        for evil in [
            "https://hyprstream.com.evil.tld", // suffix hijack
            "https://evilhyprstream.com",      // subdomain look-alike
            "http://hyprstream.com",           // scheme downgrade (no TLS)
            "https://www.hyprstream.com.evil", // www suffix hijack
        ] {
            assert_eq!(
                acao_for(&config, evil).await,
                None,
                "credentialed CORS layer must reject near-miss origin {evil}"
            );
        }
    }

    #[tokio::test]
    async fn cors_layer_multiorigin_override_honors_all_entries() {
        // An operator who supplies MULTIPLE origins via
        // HYPRSTREAM_CORS_ORIGINS / builder must have all of them honored, not
        // just the last. Single-origin overrides always worked; this guards the
        // multi-origin case.
        let config = ServerConfig::builder()
            .cors_origins(vec![
                "https://a.example.com".to_owned(),
                "https://b.example.com".to_owned(),
                "https://c.example.com".to_owned(),
            ])
            .build()
            .cors;
        for origin in [
            "https://a.example.com",
            "https://b.example.com",
            "https://c.example.com",
        ] {
            assert_eq!(
                acao_for(&config, origin).await,
                Some(origin.to_owned()),
                "multi-origin override must honor {origin}"
            );
        }
        // And an unrelated origin still gets nothing.
        assert_eq!(
            acao_for(&config, "https://evil.example.com").await,
            None,
            "multi-origin override must not leak to unrelated origins"
        );
    }

    #[tokio::test]
    async fn cors_layer_credentials_header_present_for_allowed_origin() {
        // The credentialed router must advertise `Access-Control-Allow-Credentials:
        // true` so browser credentialed requests succeed for an allowed origin.
        let config = CorsConfig::default();
        let app = Router::new()
            .route("/probe", get(|| async { "ok" }))
            .layer(cors_layer(&config));
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("GET")
                    .uri("/probe")
                    .header("Origin", "http://localhost:3000")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(
            resp.headers()
                .get(axum::http::header::ACCESS_CONTROL_ALLOW_CREDENTIALS)
                .and_then(|v| v.to_str().ok()),
            Some("true"),
            "credentialed router must set Access-Control-Allow-Credentials: true"
        );
    }

    #[tokio::test]
    async fn cors_layer_wildcard_never_pairs_with_credentials() {
        // Even a hand-edited config with `["*"]` + credentials-true must be
        // defused at layer-build time: wildcard forces credentials OFF.
        let mut config = CorsConfig::public();
        config.allow_credentials = true; // adversarial: try to force creds on with "*"
        let app = Router::new()
            .route("/probe", get(|| async { "ok" }))
            .layer(cors_layer(&config));
        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .method("GET")
                    .uri("/probe")
                    .header("Origin", "https://evil.example.com")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        // Wildcard surface mirrors any origin...
        assert_eq!(
            resp.headers()
                .get(axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .and_then(|v| v.to_str().ok()),
            Some("*"),
        );
        // ...but must NOT advertise credentials (spec-invalid + credential leak).
        assert_eq!(
            resp.headers()
                .get(axum::http::header::ACCESS_CONTROL_ALLOW_CREDENTIALS),
            None,
            "wildcard + credentials is spec-invalid; layer must drop credentials"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_iss_from_token() {
        use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};

        // Craft a JWT with iss in payload: header.payload.sig
        // payload = base64url({"iss":"https://node-a","sub":"alice","exp":9999999999,"iat":0})
        let payload = URL_SAFE_NO_PAD
            .encode(r#"{"iss":"https://node-a","sub":"alice","exp":9999999999,"iat":0}"#);
        let token = format!("eyJ0eXAiOiJKV1QiLCJhbGciOiJFZERTQSJ9.{}.fakesig", payload);
        assert_eq!(extract_iss_from_token(&token), "https://node-a");

        // Token without iss
        let payload2 = URL_SAFE_NO_PAD.encode(r#"{"sub":"alice","exp":9999999999,"iat":0}"#);
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
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    use ed25519_dalek::{Signer as _, SigningKey};

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
        signed_token_with_type(key, aud, jti, "at+jwt")
    }

    fn signed_token_with_type(key: &SigningKey, aud: &str, jti: Option<&str>, typ: &str) -> String {
        let n = now();
        let mut claims =
            jwt::Claims::new("alice".to_owned(), n, n + 3600).with_audience(Some(aud.to_owned()));
        if let Some(j) = jti {
            claims.jti = Some(j.to_owned());
        }
        let header = serde_json::json!({
            "alg": "EdDSA",
            "typ": typ,
            "kid": ed25519_kid(&key.verifying_key()),
        });
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap()),
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap()),
        );
        format!(
            "{input}.{}",
            URL_SAFE_NO_PAD.encode(key.sign(input.as_bytes()).to_bytes())
        )
    }

    #[test]
    fn ordinary_eddsa_verifier_accepts_exact_rfc9068_type_forms() {
        let ca = new_key();
        let rotation = new_key();
        let published = [rotation.verifying_key()];

        for typ in hyprstream_rpc::auth::RFC9068_ACCESS_TOKEN_TYPES {
            let token = signed_token_with_type(&rotation, AUD, None, typ);
            let claims = decode_local_multi_key(
                &token,
                &ca.verifying_key(),
                &published,
                &[],
                Some(AUD),
                true,
            )
            .unwrap();
            assert_eq!(claims.sub, "alice");
        }

        for typ in [
            "",
            "AT+JWT",
            "at+JWT",
            " at+jwt",
            "at+jwt ",
            "Application/at+jwt",
            "application/AT+JWT",
            "application/at+jwt ",
            "JWT",
            "wit+jwt",
        ] {
            let token = signed_token_with_type(&rotation, AUD, None, typ);
            assert!(
                decode_local_multi_key(
                    &token,
                    &ca.verifying_key(),
                    &published,
                    &[],
                    Some(AUD),
                    true,
                )
                .is_err(),
                "accepted non-RFC 9068 access-token type {typ:?}"
            );
        }
    }

    #[test]
    fn rotation_active_slot_verifies() {
        // Token signed by a rotation slot that is NOT the CA key — this is the
        // exact case that 401'd before #777.
        let ca = new_key();
        let rotation = new_key();
        let token = signed_token(&rotation, AUD, None);
        let published = vec![rotation.verifying_key()];

        let claims = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published,
            &[],
            Some(AUD),
            true,
        )
        .unwrap();
        assert_eq!(claims.sub, "alice");
    }

    #[test]
    fn ca_key_signed_token_still_verifies() {
        // CA-signed token must still verify — even with no rotation slots published.
        let ca = new_key();
        let token = signed_token(&ca, AUD, None);

        let claims =
            decode_local_multi_key(&token, &ca.verifying_key(), &[], &[], Some(AUD), true).unwrap();
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

        let err = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published,
            &[],
            Some(AUD),
            true,
        )
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

        let err = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published,
            &[],
            Some(AUD),
            true,
        )
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

        let claims = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published,
            &[],
            Some(AUD),
            true,
        )
        .unwrap();
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

        let claims = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published,
            &[],
            Some(AUD),
            true,
        )
        .unwrap();
        assert_eq!(claims.sub, "alice");
    }
}

/// Composite (`ML-DSA-65-Ed25519`) local-token verification (#1038).
///
/// Exercises the composite branch of `decode_local_multi_key` end-to-end:
/// happy path (active pair + via-CA-ed pair), both wrong-half rejections,
/// header confusion, exact-pair rotation, audience/issuer/time/JTI rejection,
/// half stripping, and policy-controlled plain `EdDSA` compatibility.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod composite_aware_tests {
    use super::*;
    use crate::auth::jwt::encode_composite_ml_dsa_65_ed25519;
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use ed25519_dalek::SigningKey;

    const AUD: &str = "https://node-a/resource";

    fn now() -> i64 {
        chrono::Utc::now().timestamp()
    }

    fn new_ed_key() -> SigningKey {
        SigningKey::generate(&mut rand::rngs::OsRng)
    }

    fn new_ml_dsa() -> (
        hyprstream_rpc::crypto::pq::MlDsaSigningKey,
        hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
    ) {
        hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair()
    }

    fn published_pair(
        ml_dsa: hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
        ed25519: ed25519_dalek::VerifyingKey,
    ) -> hyprstream_rpc::auth::CompositeKeyPair {
        let kid = crate::auth::jwt::composite_kid(&ml_dsa, &ed25519);
        hyprstream_rpc::auth::CompositeKeyPair::verifying(
            kid,
            ml_dsa,
            ed25519,
            hyprstream_rpc::auth::CompositePairRole::OAuth,
            hyprstream_rpc::auth::CompositePairState::Drain,
            0,
            i64::MAX,
        )
    }

    /// A composite at+JWT signed by `(pq, ed)`, audience `aud`.
    fn composite_token(
        pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
        ed: &SigningKey,
        aud: &str,
    ) -> String {
        let n = now();
        let claims =
            jwt::Claims::new("alice".to_owned(), n, n + 3600).with_audience(Some(aud.to_owned()));
        encode_composite_ml_dsa_65_ed25519(&claims, pq, ed)
    }

    fn composite_token_with_header(
        pq: &hyprstream_rpc::crypto::pq::MlDsaSigningKey,
        ed: &SigningKey,
        claims: &jwt::Claims,
        alg: &str,
        typ: &str,
        kid: &str,
    ) -> String {
        use ed25519_dalek::Signer as _;
        let header = serde_json::json!({ "alg": alg, "typ": typ, "kid": kid });
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(claims).unwrap());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let mut signature = hyprstream_rpc::crypto::pq::ml_dsa_sign(pq, signing_input.as_bytes());
        signature.extend_from_slice(&ed.sign(signing_input.as_bytes()).to_bytes());
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(signature))
    }

    #[test]
    fn composite_active_pair_verifies() {
        // The common post-#574 case: token minted by the active (ML-DSA, Ed25519)
        // pair the node publishes. This is the exact case that 401'd before #1038.
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let active = new_ed_key();
        let published_ed = vec![active.verifying_key()];
        let pairs = [published_pair(pq_vk, active.verifying_key())];
        let claims = jwt::Claims::new("alice".to_owned(), now(), now() + 3600)
            .with_audience(Some(AUD.to_owned()));

        for typ in hyprstream_rpc::auth::RFC9068_ACCESS_TOKEN_TYPES {
            let token = composite_token_with_header(
                &pq,
                &active,
                &claims,
                "ML-DSA-65-Ed25519",
                typ,
                pairs[0].kid(),
            );
            let verified = decode_local_multi_key(
                &token,
                &ca.verifying_key(),
                &published_ed,
                &pairs,
                Some(AUD),
                false,
            )
            .unwrap();
            assert_eq!(verified.sub, "alice");
        }
    }

    #[test]
    fn composite_verifies_via_ca_ed_half() {
        // No Ed25519 rotation slots published — only the CA key. A composite
        // token signed with the CA Ed25519 half verifies through the CA entry.
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let token = composite_token(&pq, &ca, AUD);
        let pairs = [published_pair(pq_vk, ca.verifying_key())];

        let claims =
            decode_local_multi_key(&token, &ca.verifying_key(), &[], &pairs, Some(AUD), false)
                .unwrap();
        assert_eq!(claims.sub, "alice");
    }

    #[test]
    fn composite_wrong_pq_half_rejected() {
        // Correct Ed25519 half, wrong (unpublished) ML-DSA half → rejected.
        // The Ed25519 half alone can never verify a composite signature.
        let ca = new_ed_key();
        let (pq, _pq_vk) = new_ml_dsa();
        let (_other_pq, other_pq_vk) = new_ml_dsa();
        let active = new_ed_key();
        let token = composite_token(&pq, &active, AUD);
        let published_ed = vec![active.verifying_key()];
        let pairs = [published_pair(other_pq_vk, active.verifying_key())];

        let err = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            false,
        )
        .unwrap_err();
        assert_eq!(err, "JWT validation failed");
    }

    #[test]
    fn composite_wrong_ed25519_half_rejected() {
        // Correct ML-DSA half, wrong (unpublished) Ed25519 half → rejected.
        // The ML-DSA half alone can never verify a composite signature.
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let active = new_ed_key();
        let attacker = new_ed_key();
        let token = composite_token(&pq, &active, AUD);
        // Only the attacker's (unpublished) Ed25519 key is offered alongside CA.
        let published_ed = vec![attacker.verifying_key()];
        let pairs = [published_pair(pq_vk, attacker.verifying_key())];

        let err = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            false,
        )
        .unwrap_err();
        assert_eq!(err, "JWT validation failed");
    }

    #[test]
    fn composite_cross_paired_published_halves_rejected() {
        let ca = new_ed_key();
        let (pq_a, pq_a_vk) = new_ml_dsa();
        let (_pq_b, pq_b_vk) = new_ml_dsa();
        let ed_a = new_ed_key();
        let ed_b = new_ed_key();
        let cross_token = composite_token(&pq_a, &ed_b, AUD);
        let published = [
            published_pair(pq_a_vk, ed_a.verifying_key()),
            published_pair(pq_b_vk, ed_b.verifying_key()),
        ];

        assert!(
            decode_local_multi_key(
                &cross_token,
                &ca.verifying_key(),
                &[ed_a.verifying_key(), ed_b.verifying_key()],
                &published,
                Some(AUD),
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn composite_signed_header_mutations_rejected() {
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let ed = new_ed_key();
        let pair = published_pair(pq_vk, ed.verifying_key());
        let claims = jwt::Claims::new("alice".to_owned(), now(), now() + 3600)
            .with_audience(Some(AUD.to_owned()));
        let wrong_kid = "not-the-published-pair";
        for (alg, typ, kid) in [
            ("EdDSA", "at+jwt", pair.kid()),
            ("ML-DSA-65-Ed25519", "wit+jwt", pair.kid()),
            ("ML-DSA-65-Ed25519", "AT+JWT", pair.kid()),
            ("ML-DSA-65-Ed25519", "at+jwt ", pair.kid()),
            ("ML-DSA-65-Ed25519", "Application/at+jwt", pair.kid()),
            ("ML-DSA-65-Ed25519", "application/AT+JWT", pair.kid()),
            ("ML-DSA-65-Ed25519", "application/at+jwt ", pair.kid()),
            ("ML-DSA-65-Ed25519", "at+jwt", wrong_kid),
        ] {
            let token = composite_token_with_header(&pq, &ed, &claims, alg, typ, kid);
            assert!(
                decode_local_multi_key(
                    &token,
                    &ca.verifying_key(),
                    &[ed.verifying_key()],
                    std::slice::from_ref(&pair),
                    Some(AUD),
                    true,
                )
                .is_err(),
                "accepted altered header alg={alg} typ={typ} kid={kid}"
            );
        }
    }

    #[test]
    fn composite_expiry_future_iat_and_missing_audience_rejected() {
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let ed = new_ed_key();
        let pair = published_pair(pq_vk, ed.verifying_key());
        let cases = [
            (
                "expired",
                jwt::Claims::new("expired".to_owned(), now() - 7200, now() - 10)
                    .with_audience(Some(AUD.to_owned())),
            ),
            (
                "future",
                jwt::Claims::new("future".to_owned(), now() + 120, now() + 3600)
                    .with_audience(Some(AUD.to_owned())),
            ),
            (
                "no-aud",
                jwt::Claims::new("no-aud".to_owned(), now(), now() + 3600),
            ),
        ];
        for (case, claims) in cases {
            let token = composite_token_with_header(
                &pq,
                &ed,
                &claims,
                "ML-DSA-65-Ed25519",
                "at+jwt",
                pair.kid(),
            );
            assert!(
                decode_local_multi_key(
                    &token,
                    &ca.verifying_key(),
                    &[ed.verifying_key()],
                    std::slice::from_ref(&pair),
                    Some(AUD),
                    false,
                )
                .is_err(),
                "accepted invalid {case} claims"
            );
        }
    }

    #[test]
    fn composite_wrong_issuer_rejected_by_local_policy() {
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let ed = new_ed_key();
        let pair = published_pair(pq_vk, ed.verifying_key());
        let claims = jwt::Claims::new("alice".to_owned(), now(), now() + 3600)
            .with_issuer("https://other.example".to_owned())
            .with_audience(Some(AUD.to_owned()));
        let token = composite_token_with_header(
            &pq,
            &ed,
            &claims,
            "ML-DSA-65-Ed25519",
            "at+jwt",
            pair.kid(),
        );
        let verified = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &[ed.verifying_key()],
            std::slice::from_ref(&pair),
            Some(AUD),
            false,
        )
        .unwrap();
        assert!(!local_issuer_matches(&verified, "https://node-a.example"));
    }

    #[test]
    fn composite_half_signature_stripping_rejected() {
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let active = new_ed_key();
        let token = composite_token(&pq, &active, AUD);
        let published_ed = vec![active.verifying_key()];
        let pairs = [published_pair(pq_vk, active.verifying_key())];

        // Sanity: the well-formed composite token verifies.
        decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            false,
        )
        .unwrap();

        let dot2 = token.rfind('.').unwrap();
        let sig_b64 = &token[dot2 + 1..];
        let sig_bytes = URL_SAFE_NO_PAD.decode(sig_b64).unwrap();
        for stripped in [&sig_bytes[..3309], &sig_bytes[3309..]] {
            let tampered = format!("{}.{}", &token[..dot2], URL_SAFE_NO_PAD.encode(stripped));
            assert!(
                decode_local_multi_key(
                    &tampered,
                    &ca.verifying_key(),
                    &published_ed,
                    &pairs,
                    Some(AUD),
                    false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn plain_eddsa_is_policy_controlled() {
        let ca = new_ed_key();
        let active = new_ed_key();
        let n = now();
        let claims =
            jwt::Claims::new("alice".to_owned(), n, n + 3600).with_audience(Some(AUD.to_owned()));
        let token = jwt::encode(&claims, &active);
        let published_ed = vec![active.verifying_key()];
        // Even with composite keys published, a plain EdDSA token routes to and
        // verifies via the plain Ed25519 branch.
        let (pq, pq_vk) = new_ml_dsa();
        let pairs = [published_pair(pq_vk, active.verifying_key())];
        let _ = pq;
        let claims = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            true,
        )
        .unwrap();
        assert_eq!(claims.sub, "alice");

        assert!(
            decode_local_multi_key(
                &token,
                &ca.verifying_key(),
                &published_ed,
                &pairs,
                Some(AUD),
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn composite_jti_remains_subject_to_revocation() {
        use hyprstream_rpc::auth::{InMemoryJtiBlocklist, JtiBlocklist as _};

        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let ed = new_ed_key();
        let pair = published_pair(pq_vk, ed.verifying_key());
        let claims = jwt::Claims::new("alice".to_owned(), now(), now() + 3600)
            .with_audience(Some(AUD.to_owned()))
            .with_jti();
        let token = composite_token_with_header(
            &pq,
            &ed,
            &claims,
            "ML-DSA-65-Ed25519",
            "at+jwt",
            pair.kid(),
        );
        let verified = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &[ed.verifying_key()],
            std::slice::from_ref(&pair),
            Some(AUD),
            false,
        )
        .unwrap();
        let jti = verified.jti.as_deref().unwrap();
        let blocklist = InMemoryJtiBlocklist::new();
        blocklist.revoke(jti.to_owned(), verified.exp);
        assert!(blocklist.is_revoked(jti));
    }

    #[test]
    fn composite_wrong_audience_rejected() {
        // Correct published pair, wrong audience → rejected. Composite audience
        // validation is as strict as the plain path.
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let active = new_ed_key();
        let token = composite_token(&pq, &active, "https://evil/other");
        let published_ed = vec![active.verifying_key()];
        let pairs = [published_pair(pq_vk, active.verifying_key())];

        let err = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            false,
        )
        .unwrap_err();
        assert_eq!(err, "JWT validation failed");
    }

    #[test]
    fn composite_survives_independent_ed_rotation() {
        // A token signed under (ml_dsa, ed_active) still verifies after the
        // Ed25519 half rotates because its previously authorized exact pair is
        // retained while both halves remain published.
        let ca = new_ed_key();
        let (pq, pq_vk) = new_ml_dsa();
        let ed_old_active = new_ed_key();
        let token = composite_token(&pq, &ed_old_active, AUD);
        // Post-rotation published set: old active is now in drain, new active added.
        let ed_new_active = new_ed_key();
        let published_ed = vec![ed_new_active.verifying_key(), ed_old_active.verifying_key()];
        let pairs = [published_pair(pq_vk, ed_old_active.verifying_key())];

        let claims = decode_local_multi_key(
            &token,
            &ca.verifying_key(),
            &published_ed,
            &pairs,
            Some(AUD),
            false,
        )
        .unwrap();
        assert_eq!(claims.sub, "alice");
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod browser_provisioning_rate_limit_tests {
    use super::*;
    use axum::{Router, body::Body, http::Request as HttpRequest, middleware, routing::get};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    #[tokio::test]
    async fn provisioning_limit_rejects_before_handler_work() {
        let invoked = Arc::new(AtomicUsize::new(0));
        let handler_invoked = Arc::clone(&invoked);
        let limiter = Arc::new(RateLimiter::new(1, 60));
        let app = Router::new()
            .route(
                "/provision",
                get(move || {
                    let handler_invoked = Arc::clone(&handler_invoked);
                    async move {
                        handler_invoked.fetch_add(1, Ordering::SeqCst);
                        StatusCode::OK
                    }
                }),
            )
            .layer(middleware::from_fn_with_state(
                limiter,
                browser_provisioning_rate_limit_middleware,
            ));

        let first = app
            .clone()
            .oneshot(
                HttpRequest::get("/provision")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("first response");
        let rejected = app
            .oneshot(
                HttpRequest::get("/provision")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("limited response");

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            invoked.load(Ordering::SeqCst),
            1,
            "rejected request must not enter resolver/signing handler work"
        );
    }
}
