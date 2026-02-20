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
    // Get JWT token from Authorization header
    let token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(str::trim);

    // If no token provided, allow anonymous access
    let Some(token) = token else {
        return next.run(request).await;
    };

    // Build WWW-Authenticate header for 401 responses (RFC 9728)
    let www_authenticate = build_www_authenticate(&state);

    // Try JWT validation (stateless)
    if token.contains('.') {
        match jwt::decode(token, &state.verifying_key) {
            Ok(claims) => {
                debug!("JWT validated for user: {}", claims.sub);
                let user = AuthenticatedUser {
                    user: claims.sub.clone(),
                    token: Some(token.to_owned()),
                    exp: Some(claims.exp),
                };
                request.extensions_mut().insert(user);
                return next.run(request).await;
            }
            Err(jwt::JwtError::Expired) => {
                debug!("JWT expired");
                return unauthorized_response("Token expired", &www_authenticate);
            }
            Err(jwt::JwtError::InvalidSignature) => {
                debug!("JWT signature invalid");
                return unauthorized_response("Invalid token signature", &www_authenticate);
            }
            Err(e) => {
                debug!("JWT validation failed: {}", e);
                return unauthorized_response("Invalid token", &www_authenticate);
            }
        }
    }

    warn!("Invalid token format");
    unauthorized_response("Invalid token format", &www_authenticate)
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
