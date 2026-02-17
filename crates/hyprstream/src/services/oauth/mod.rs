//! OAuth 2.1 Authorization Server for hyprstream.
//!
//! Provides OAuth 2.1 (draft-ietf-oauth-v2-1-13) authorization serving both
//! MCP (:6790) and OAI (:6789) as protected resources.
//!
//! # Standards
//!
//! - OAuth 2.1 (draft-ietf-oauth-v2-1-13)
//! - RFC 8414 (Authorization Server Metadata)
//! - RFC 7591 (Dynamic Client Registration)
//! - RFC 8707 (Resource Indicators)
//! - RFC 8628 (Device Authorization Grant)
//! - RFC 9728 (Protected Resource Metadata)
//! - Client ID Metadata Documents (draft-ietf-oauth-client-id-metadata-document-00)
//!
//! # Architecture
//!
//! ```text
//! OAuth Server (:6791)
//!   /.well-known/oauth-authorization-server  → metadata
//!   /oauth/register                          → dynamic registration
//!   /oauth/authorize                         → consent + auth code
//!   /oauth/token                             → PKCE + PolicyClient delegation
//!   /oauth/device                            → device authorization (RFC 8628)
//!   /oauth/device/verify                     → user verification page
//! ```

pub mod authorize;
pub mod device;
pub mod metadata;
pub mod registration;
pub mod state;
pub mod token;

use std::sync::Arc;

use anyhow::Result;
use axum::{routing::{get, post}, Router};
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service::spawner::Spawnable;
use hyprstream_rpc::transport::TransportConfig;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tracing::{error, info};

use crate::config::OAuthConfig;
use crate::services::PolicyClient;
use state::OAuthState;

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "oauth";

/// Create the OAuth Axum router.
pub fn create_app(state: Arc<OAuthState>) -> Router {
    Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(metadata::authorization_server_metadata),
        )
        .route("/oauth/register", post(registration::register_client))
        .route(
            "/oauth/authorize",
            get(authorize::authorize_get).post(authorize::authorize_post),
        )
        .route("/oauth/token", post(token::exchange_token))
        .route("/oauth/device", post(device::device_authorize))
        .route(
            "/oauth/device/verify",
            get(device::verify_get).post(device::verify_post),
        )
        .with_state(state)
}

/// OAuth 2.1 Authorization Server service.
///
/// Runs an Axum HTTP server with OAuth endpoints. Token issuance is delegated
/// to PolicyService via ZMQ.
pub struct OAuthService {
    config: OAuthConfig,
    policy_client: PolicyClient,
    context: Arc<zmq::Context>,
    control_transport: TransportConfig,
    #[allow(dead_code)]
    verifying_key: ed25519_dalek::VerifyingKey,
}

impl OAuthService {
    pub fn new(
        config: OAuthConfig,
        policy_client: PolicyClient,
        context: Arc<zmq::Context>,
        control_transport: TransportConfig,
        verifying_key: ed25519_dalek::VerifyingKey,
    ) -> Self {
        Self {
            config,
            policy_client,
            context,
            control_transport,
            verifying_key,
        }
    }
}

impl Spawnable for OAuthService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, self.control_transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        rt.block_on(async move {
            let addr_str = format!("{}:{}", self.config.host, self.config.port);
            let addr: std::net::SocketAddr = addr_str.parse().map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(format!("Invalid address: {e}"))
            })?;

            // Create shared state
            let state = Arc::new(OAuthState::new(&self.config, self.policy_client));
            state.spawn_code_sweeper();

            // Create router
            let app = create_app(state);

            // Bind
            let listener = TcpListener::bind(addr).await.map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(format!("HTTP bind failed: {e}"))
            })?;

            info!("OAuth 2.1 server listening on http://{}", addr);
            info!(
                "Authorization server metadata at http://{}/.well-known/oauth-authorization-server",
                addr
            );

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            let _ = hyprstream_rpc::notify::ready();

            let shutdown_clone = shutdown.clone();
            let server_result = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown_clone.notified().await;
                    info!("OAuthService received shutdown signal");
                })
                .await;

            if let Err(e) = server_result {
                error!("OAuthService HTTP server error: {}", e);
            }

            info!("OAuthService stopped");
            Ok(())
        })
    }
}

/// Create a Protected Resource Metadata JSON response (RFC 9728).
///
/// Used by MCP and OAI servers to advertise their OAuth authorization server.
pub fn protected_resource_metadata(resource_url: &str, oauth_issuer_url: &str) -> serde_json::Value {
    serde_json::json!({
        "resource": resource_url,
        "authorization_servers": [oauth_issuer_url],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::registration::validate_redirect_uri;

    #[test]
    fn test_protected_resource_metadata() {
        let meta = protected_resource_metadata(
            "http://localhost:6790",
            "http://localhost:6791",
        );
        assert_eq!(meta["resource"], "http://localhost:6790");
        assert_eq!(meta["authorization_servers"][0], "http://localhost:6791");
    }

    #[test]
    fn test_validate_redirect_uri_exact_match() {
        let registered = vec!["http://127.0.0.1:3000/callback".to_owned()];
        assert!(validate_redirect_uri("http://127.0.0.1:3000/callback", &registered));
        // Loopback URIs: port is ignored per RFC 8252, so different port still matches
        assert!(validate_redirect_uri("http://127.0.0.1:4000/callback", &registered));
        // Different path should NOT match
        assert!(!validate_redirect_uri("http://127.0.0.1:3000/other", &registered));
    }

    #[test]
    fn test_validate_redirect_uri_loopback_port_ignored() {
        let registered = vec!["http://127.0.0.1:3000/callback".to_owned()];
        // Different port on loopback should match per RFC 8252
        assert!(validate_redirect_uri("http://127.0.0.1:9999/callback", &registered));
        // Non-loopback should require exact match
        let non_loopback = vec!["https://example.com:3000/callback".to_owned()];
        assert!(!validate_redirect_uri("https://example.com:4000/callback", &non_loopback));
    }

    #[test]
    fn test_pkce_s256() {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        use sha2::{Digest, Sha256};

        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let expected_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";

        let computed = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        assert_eq!(computed, expected_challenge);
    }
}
