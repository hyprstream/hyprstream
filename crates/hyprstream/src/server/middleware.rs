//! Middleware for authentication, logging, and request processing

use crate::server::state::ServerState;
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::time::Instant;
use tracing::{info, warn};

/// API key authentication middleware
pub async fn auth_middleware(
    State(state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    // Check if API key is configured
    if let Some(ref expected_key) = state.config.api_key {
        // Get API key from Authorization header
        let auth_header = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let api_key = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .or_else(|| {
                // Also check X-API-Key header
                request
                    .headers()
                    .get("x-api-key")
                    .and_then(|v| v.to_str().ok())
            });

        match api_key {
            Some(key) if key == expected_key => {
                // Valid API key, proceed
            }
            Some(_) => {
                warn!("Invalid API key attempt");
                return (StatusCode::UNAUTHORIZED, "Invalid API key").into_response();
            }
            None => {
                return (StatusCode::UNAUTHORIZED, "API key required").into_response();
            }
        }
    }

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

/// Rate limiting middleware (simplified)
pub async fn rate_limit_middleware(
    State(_state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    next.run(request).await
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
            // Custom API headers
            HeaderName::from_static("x-api-key"),
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
    if config.allowed_origins.contains(&"*".to_string()) {
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
