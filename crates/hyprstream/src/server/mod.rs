//! Hyprstream server implementation with OpenAI-compatible API
//!
//! This module provides the main HTTP server with:
//! - OpenAI-compatible API endpoints at /oai/v1
//! - LoRA adapter management at /lora
//! - Model management at /models
//! - Training service at /training

use anyhow::Result;
use axum::{response::IntoResponse, routing::get, Json, Router};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpListener;
use tower_http::{timeout::TimeoutLayer, trace::TraceLayer};
use tracing::info;

pub mod middleware;
pub mod model_cache;
pub mod routes;
pub mod state;

use state::ServerState;

/// Create the main application router
pub fn create_app(state: ServerState) -> Router {
    // Clone config for middleware
    let cors_config = state.config.cors.clone();
    let timeout_duration = Duration::from_secs(state.config.request_timeout_secs);

    let mut app = Router::new()
        // Health check endpoint
        .route("/", get(health_check))
        .route("/health", get(health_check))
        // OpenAI-compatible API routes at /oai/v1
        .nest("/oai/v1", routes::openai::create_router())
        // Model management routes
        .nest("/models", routes::models::create_router())
        // Training service routes
        .nest("/training", routes::training::create_router())
        // Add middleware (order matters: timeout should be before state)
        .layer(TimeoutLayer::new(timeout_duration))
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

/// Start the HTTP server
pub async fn start_server(addr: SocketAddr, state: ServerState) -> Result<()> {
    let app = create_app(state);

    info!("Starting Hyprstream server on {}", addr);
    info!("OpenAI-compatible API available at http://{}/oai/v1", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the HTTPS server with TLS
pub async fn start_server_tls(
    addr: SocketAddr,
    state: ServerState,
    _cert_path: &str,
    _key_path: &str,
) -> Result<()> {
    info!("TLS server support not yet implemented");
    start_server(addr, state).await
}
