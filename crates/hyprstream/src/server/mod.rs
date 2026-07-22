//! Hyprstream server implementation with OpenAI-compatible API
//!
//! This module provides the main HTTP server with:
//! - OpenAI-compatible API endpoints at /oai/v1
//! - Model management at /models

use anyhow::Result;
use axum::{middleware as axum_middleware, response::IntoResponse, routing::get, Json, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use axum::http::StatusCode;
use tower_http::{timeout::TimeoutLayer, trace::TraceLayer};
use tracing::info;

pub mod middleware;
pub mod routes;
pub mod state;
pub mod tls;

pub use middleware::AuthenticatedUser;
use state::ServerState;
use axum::Extension;

/// Extract user identity from optional authenticated user extension.
/// Returns "anonymous" if no authentication present.
pub fn extract_user(auth_user: Option<&Extension<AuthenticatedUser>>) -> String {
    auth_user
        .map(|Extension(u)| u.user.clone())
        .unwrap_or_else(|| "anonymous".to_owned())
}

/// Create the main application router
pub fn create_app(state: ServerState) -> Router {
    // H1b (#765): register the 9P-over-WebTransport handler for the QUIC
    // path-mux `/9p` arm, co-located with H1a's axum `/9p` WS route below so
    // both planes share one `ServerState` (export mount + ticket validator).
    // Idempotent; the per-service QUIC endpoint (built in `hyprstream-service`,
    // which can't depend on this crate) picks it up via the process-global.
    routes::ninep::register_ninep_wt_handler(state.clone());

    // Clone config for middleware
    let cors_config = state.config.cors.clone();
    let timeout_duration = Duration::from_secs(state.config.request_timeout_secs);
    let resource_auth_state = state.resource_auth_state();

    // Public browser provisioning is independently rate-limited before the
    // handler can resolve accepted state or perform hybrid signing.
    let browser_provisioning_routes = Router::new()
        .route(
            "/.well-known/hyprstream/browser-provisioning/:service",
            get(routes::browser_provisioning::browser_provisioning),
        )
        .layer(axum_middleware::from_fn_with_state(
            Arc::clone(&state.browser_provisioning_rate_limiter),
            middleware::browser_provisioning_rate_limit_middleware,
        ));

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_protected_resource_metadata),
        )
        // 9P-over-WebSocket export (H1a / #764). Public routes: the mount
        // ticket rides the URL query (browser WS can't set headers) and is
        // validated inside the /9p handler, so these bypass `auth_middleware`
        // (which requires an Authorization header).
        .route(
            "/.well-known/export9p",
            get(routes::ninep::export9p_metadata),
        )
        // Wire-plane discovery table (#821 / epic #809): enumerates every
        // non-file wire plane (9p, moq, …) → its current path + carriers, the
        // source of truth a client resolves the `/9p` selector against instead
        // of a hardcoded constant. Companions (does not replace) export9p.
        .route(
            "/.well-known/planes",
            get(routes::ninep::wire_planes_metadata),
        )
        .route("/9p", get(routes::ninep::ninep_ws))
        .merge(browser_provisioning_routes);

    // Protected routes (auth required)
    let protected_routes = Router::new()
        // OpenAI-compatible API routes at /oai/v1
        .nest("/oai/v1", routes::openai::create_router())
        // Model management routes
        .nest("/models", routes::models::create_router())
        // rate_limit added first (inner) → runs after auth sees the authenticated subject
        .layer(axum_middleware::from_fn_with_state(
            resource_auth_state.clone(),
            middleware::rate_limit_middleware,
        ))
        .layer(axum_middleware::from_fn_with_state(
            resource_auth_state,
            middleware::auth_middleware,
        ));

    let mut app = public_routes
        .merge(protected_routes)
        // Add middleware (order matters: timeout should be before state)
        .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, timeout_duration))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Add CORS middleware if enabled (should be outermost)
    if cors_config.enabled {
        app = app.layer(middleware::cors_layer(&cors_config));
    }

    app
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "hyprstream",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Protected Resource Metadata (RFC 9728) for the OAI server.
///
/// Advertises the OAuth authorization server that protects this resource.
async fn oauth_protected_resource_metadata() -> impl IntoResponse {
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let oai_url = config.oai.resource_url();
    let oauth_issuer = config.oauth.issuer_url();

    let mut meta = crate::services::oauth::protected_resource_metadata(
        &oai_url,
        &oauth_issuer,
    );
    meta.resource_name = Some("HyprStream OpenAI-Compatible API".to_owned());
    meta.scopes_supported = Some(vec!["infer:model:*".into(), "read:model:*".into()]);
    Json(meta)
}

/// Start the HTTP server (plain, no TLS).
pub async fn start_server(addr: SocketAddr, state: ServerState) -> Result<()> {
    let app = create_app(state);

    info!("Starting Hyprstream server on {}", addr);
    info!("OpenAI-compatible API available at http://{}/oai/v1", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the HTTPS server with TLS.
///
/// Resolves TLS configuration from the global `[tls]` section, with optional
/// per-service cert/key overrides. Uses `serve_app()` for HTTPS or falls back to HTTP.
pub async fn start_server_tls(
    addr: SocketAddr,
    state: ServerState,
    shutdown: Arc<Notify>,
) -> Result<()> {
    let hypr_config = crate::config::HyprConfig::load().unwrap_or_default();
    let rustls_config = tls::resolve_rustls_config(
        &hypr_config.tls,
        hypr_config.oai.tls_cert.as_ref(),
        hypr_config.oai.tls_key.as_ref(),
    )
    .await?;

    let scheme = if rustls_config.is_some() { "https" } else { "http" };
    info!("OpenAI-compatible API available at {scheme}://{addr}/oai/v1");

    let app = create_app(state);
    tls::serve_app(addr, app, rustls_config, shutdown, "Hyprstream")
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))
}
