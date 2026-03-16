//! Middleware for authentication, logging, and request processing

use crate::auth::jwt;
use crate::server::state::ServerState;
use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::time::Instant;
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
/// Validates JWT tokens (eyJ...) via Ed25519 signature verification.
///
/// On success, inserts `AuthenticatedUser` into request extensions.
/// JWT `sub` claim contains bare username (e.g., "alice").
pub async fn auth_middleware(
    State(state): State<ServerState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Get JWT token from Authorization header (RFC 6750: Bearer scheme is case-insensitive)
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| {
            if h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer ") {
                Some(h[7..].trim())
            } else {
                None
            }
        });

    // SECURITY: No anonymous access — require Bearer token
    let Some(token) = token else {
        let www_authenticate = build_www_authenticate(&state);
        return unauthorized_response("Authentication required", &www_authenticate);
    };

    // Build WWW-Authenticate header for 401 responses (RFC 9728)
    let www_authenticate = build_www_authenticate(&state);

    // Try JWT validation (stateless)
    if token.contains('.') {
        // Extract iss from token payload without signature verification
        let iss = extract_iss_from_token(token);

        // Local token: iss is empty (old token) OR matches our OAuth issuer URL OR matches our resource URL (belt-and-suspenders)
        let is_local = iss.is_empty() || iss == state.oauth_issuer_url || iss == state.resource_url;
        let result = if is_local {
            // Local token: verify with local key
            jwt::decode(token, &state.verifying_key, Some(&state.resource_url))
        } else {
            // Federated token: pre-check trust before acquiring async lock
            if !state.federation_resolver.is_trusted(&iss) {
                debug!("Untrusted federation issuer: {}", iss);
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
            match state.federation_resolver.get_key(&iss).await {
                Ok(key) => jwt::decode_with_key(token, &key, Some(&state.resource_url)),
                Err(e) => {
                    debug!("Federation key resolution failed for issuer {}: {}", iss, e);
                    return unauthorized_response("Authentication failed", &www_authenticate);
                }
            }
        };

        match result {
            Ok(claims) => {
                debug!("JWT validated for user: {}", claims.sub);
                let is_local_claims = claims.iss.is_empty() || claims.iss == state.oauth_issuer_url || claims.iss == state.resource_url;
                let user_str = if is_local_claims {
                    claims.sub.clone()
                } else {
                    // Federated subject: "{iss}:{sub}" for Casbin policy matching
                    format!("{}:{}", claims.iss, claims.sub)
                };
                // Validate local subjects for safe characters. Federated subjects
                // (containing "://") bypass this check — they are validated at JWT decode time.
                let subject = hyprstream_rpc::Subject::new(&user_str);
                if let Err(e) = subject.validate() {
                    debug!("JWT sub validation failed: {}", e);
                    return unauthorized_response("Authentication failed", &www_authenticate);
                }
                let user = AuthenticatedUser {
                    user: user_str,
                    token: Some(token.to_owned()),
                    exp: Some(claims.exp),
                };
                request.extensions_mut().insert(user);
                return next.run(request).await;
            }
            Err(e) => {
                debug!("JWT validation failed: {}", e);
                return unauthorized_response("Authentication failed", &www_authenticate);
            }
        }
    }

    debug!("Invalid token format");
    unauthorized_response("Authentication failed", &www_authenticate)
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

/// Rate limiting middleware (simplified)
pub async fn rate_limit_middleware(
    State(_state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    next.run(request).await
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
        // Permissive mode - allow any header (development/debugging only)
        cors = cors.allow_headers(Any);
        warn!("⚠️  CORS: Allowing ALL headers (permissive mode)");
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
