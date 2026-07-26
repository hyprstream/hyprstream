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

/// Extract the subject and its authority-verified hosted-account policy domain.
pub fn extract_policy_identity(
    auth_user: Option<&Extension<AuthenticatedUser>>,
) -> Result<(String, String), &'static str> {
    let Extension(user) = auth_user.ok_or("authenticated identity missing")?;
    Ok((user.user.clone(), user.authorization_domain()?))
}

/// Create the main application router
pub fn create_app(state: ServerState) -> Router {
    // Validate the complete cross-face registry before constructing any live
    // route. Duplicate ids/routes, empty justifications, or a face/handler
    // mismatch are startup-fatal rather than test-only findings.
    if let Err(error) = crate::mac::public_exemptions::validate() {
        panic!("invalid public exemption registry: {error}");
    }

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

    // Public routes (no auth required).
    //
    // #1273 / epic #1267: the unmediated public set is the reviewed
    // `mac::public_exemptions` registry. These routes are built FROM the
    // registry — adding an unmediated route requires appending a reviewed
    // `PublicExemption` entry plus a handler arm below (two review surfaces).
    // Every route NOT here stays on the protected (auth-mediated) router; the
    // default is mediated, never permissive. See `mac::public_exemptions`.
    let (public_routes, browser_provisioning_routes) =
        build_public_routes_from_registry(Arc::clone(&state.browser_provisioning_rate_limiter));

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
        .merge(browser_provisioning_routes)
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

/// Build the main-app public (unmediated) routers from the reviewed
/// `mac::public_exemptions` registry (#1273 / epic #1267).
///
/// Returns `(public_routes, browser_provisioning_routes)`. The caller merges
/// them and applies `.with_state(state)`. Wiring is driven by iterating the
/// registry, so an unmediated route cannot appear without a reviewed
/// `PublicExemption` entry plus a handler arm in the exhaustive match below.
fn build_public_routes_from_registry(
    browser_provisioning_rate_limiter: Arc<crate::server::middleware::RateLimiter>,
) -> (Router<ServerState>, Router<ServerState>) {
    use crate::mac::public_exemptions::{for_face, HttpFace, PublicRouteHandler, RouteMethod};

    let mut public_routes: Router<ServerState> = Router::new();
    let mut browser_provisioning_routes: Router<ServerState> = Router::new();

    for exemption in for_face(HttpFace::MainApp) {
        // Every reviewed MainApp public route is GET today. Fail closed if a
        // non-GET route is added: extend this builder with method dispatch and
        // update the registry snapshot test in the same PR.
        assert_eq!(
            exemption.method,
            RouteMethod::Get,
            "non-GET MainApp public route {:?} requires method dispatch in \
             build_public_routes_from_registry",
            exemption.id,
        );
        // Exhaustive match: wiring a new public route requires a handler arm
        // here AND a registry entry — the double review touch that keeps the
        // exempt set from silently growing.
        match exemption.handler {
            PublicRouteHandler::HealthCheck => {
                public_routes = public_routes.route(exemption.path, get(health_check));
            }
            PublicRouteHandler::OauthProtectedResourceMetadata => {
                public_routes = public_routes
                    .route(exemption.path, get(oauth_protected_resource_metadata));
            }
            PublicRouteHandler::Export9pMetadata => {
                public_routes =
                    public_routes.route(exemption.path, get(routes::ninep::export9p_metadata));
            }
            PublicRouteHandler::WirePlanesMetadata => {
                public_routes =
                    public_routes.route(exemption.path, get(routes::ninep::wire_planes_metadata));
            }
            PublicRouteHandler::NinepWebSocket => {
                public_routes = public_routes.route(exemption.path, get(routes::ninep::ninep_ws));
            }
            PublicRouteHandler::BrowserProvisioning => {
                // Public browser provisioning is independently rate-limited
                // before the handler can resolve accepted state or perform
                // hybrid signing; it rides its own sub-router + layer.
                browser_provisioning_routes = browser_provisioning_routes
                    .route(
                        exemption.path,
                        get(routes::browser_provisioning::browser_provisioning),
                    )
                    .layer(axum_middleware::from_fn_with_state(
                        Arc::clone(&browser_provisioning_rate_limiter),
                        middleware::browser_provisioning_rate_limit_middleware,
                    ));
            }
            PublicRouteHandler::At9pVerify => {
                // Lives on a separate credential-free face; validate() ensures
                // it never appears under HttpFace::MainApp. Crashing here is the
                // fail-closed response to a misconfigured registry.
                unreachable!(
                    "At9pVerify belongs to a separate face; mac::public_exemptions::validate() \
                     should have prevented it appearing on HttpFace::MainApp"
                );
            }
        }
    }

    (public_routes, browser_provisioning_routes)
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
        &hypr_config.account,
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::mac::public_exemptions::{self, for_face, HttpFace};
    use std::collections::BTreeSet;

    /// Extract the concrete paths configured in an Axum router's live path
    /// table. Axum has no public route-enumeration API; its `Debug`
    /// implementation deliberately includes the path router's `Node.paths`
    /// map. A dependency change that removes or changes that representation
    /// fails this load-bearing test closed until the extractor is reviewed.
    fn live_router_paths<S>(router: &Router<S>) -> BTreeSet<String> {
        let debug = format!("{router:?}");
        let path_router = debug
            .split_once("path_router: ")
            .map(|(_, rest)| rest)
            .and_then(|rest| {
                rest.split_once(", fallback_router: ")
                    .map(|(paths, _)| paths)
            })
            .expect("Axum Router Debug must expose the live path_router");
        let paths = path_router
            .split_once("node: Node { paths: {")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.split_once("} }").map(|(paths, _)| paths))
            .expect("Axum PathRouter Debug must expose the live Node.paths map");

        paths
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|value| value.starts_with('/'))
            .map(str::to_owned)
            .collect()
    }

    /// #1273 load-bearing drift gate: the concrete, live public route table
    /// must equal the reviewed registry exactly.
    ///
    /// This is intentionally stronger than checking that every registry entry
    /// builds. Adding `.route("/bypass", ...)` directly to either public router
    /// changes `live_router_paths` without changing the registry and fails this
    /// equality. Conversely, a registry entry without a live route also fails.
    /// Since the public set is exact, protected paths remain outside it and are
    /// still mediated by `auth_middleware` when `create_app` merges the
    /// protected router.
    #[test]
    fn public_exemptions_match_live_routes() {
        public_exemptions::validate().expect("registry must be self-consistent before wiring");

        let rate_limiter = Arc::new(middleware::RateLimiter::new(u32::MAX, 3600));
        let (public, browser_provisioning) = build_public_routes_from_registry(rate_limiter);
        let live = live_router_paths(&public.merge(browser_provisioning));
        let registered = for_face(HttpFace::MainApp)
            .map(|exemption| exemption.path.to_owned())
            .collect::<BTreeSet<_>>();

        assert_eq!(
            live, registered,
            "live unmediated main-app routes diverged from PUBLIC_EXEMPTIONS"
        );
        assert!(
            !live.contains("/models") && !live.contains("/oai/v1"),
            "protected route roots must never appear in the public router"
        );

        let at9p = crate::services::at9p_verify::credential_free_router(
            crate::services::at9p_verify::VerifyFaceState {
                max_skew_seconds: 300,
                max_challenge_bytes: 256,
            },
        );
        let live_at9p = live_router_paths(&at9p);
        let registered_at9p = for_face(HttpFace::At9pVerify)
            .map(|exemption| exemption.path.to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            live_at9p, registered_at9p,
            "live unmediated at9p routes diverged from PUBLIC_EXEMPTIONS"
        );
    }
}
