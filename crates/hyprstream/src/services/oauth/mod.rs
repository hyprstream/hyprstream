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
//!   /oauth/jwks                              → JSON Web Key Set (RFC 7517)
//!   /oauth/device                            → device authorization (RFC 8628)
//!   /oauth/device/verify                     → user verification page
//! ```

pub mod authorize;
pub mod device;
pub mod jwks;
pub mod metadata;
pub mod registration;
pub mod state;
pub mod token;

use std::sync::Arc;

use anyhow::Result;
use axum::{routing::{get, post}, Router};
use hyprstream_rpc::registry::SocketKind;
use hyprstream_service::Spawnable;
use hyprstream_rpc::transport::TransportConfig;
use tokio::sync::Notify;
use tracing::info;

use crate::config::OAuthConfig;
use crate::services::PolicyClient;
use state::OAuthState;

/// Service name for registry and logging
pub const SERVICE_NAME: &str = "oauth";

/// Create the OAuth Axum router.
///
/// CORS is outermost (applied last) so OPTIONS preflights are handled before
/// any inner middleware (like logging) runs. This fixes the previous ordering
/// where logging was outermost and CORS was inner.
pub fn create_app(state: Arc<OAuthState>, cors_config: &crate::config::CorsConfig) -> Router {
    let mut router = Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(metadata::authorization_server_metadata),
        )
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_self_protected_resource_metadata),
        )
        .route("/oauth/register", post(registration::register_client))
        .route(
            "/oauth/authorize",
            get(authorize::authorize_get).post(authorize::authorize_post),
        )
        .route("/oauth/token", post(token::exchange_token))
        .route("/oauth/jwks", get(jwks::jwks))
        .route("/oauth/device", post(device::device_authorize))
        .route(
            "/oauth/device/verify",
            get(device::verify_get).post(device::verify_post),
        )
        .route("/oauth/device/nonce", get(device::device_nonce))
        .layer(axum::middleware::from_fn(|req: axum::extract::Request, next: axum::middleware::Next| async move {
            let method = req.method().clone();
            let uri = req.uri().clone();
            tracing::info!(%method, %uri, "OAuth request");
            next.run(req).await
        }));

    // CORS outermost (added last = runs first on request)
    if cors_config.enabled {
        router = router.layer(crate::server::middleware::cors_layer(cors_config));
    }

    router.with_state(state)
}

/// RFC 9728 Protected Resource Metadata for the OAuth server itself.
async fn oauth_self_protected_resource_metadata() -> axum::Json<ProtectedResourceMetadata> {
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let issuer_url = config.oauth.issuer_url();
    let mut meta = protected_resource_metadata(&issuer_url, &issuer_url);
    meta.resource_name = Some("HyprStream OAuth 2.1 Authorization Server".to_owned());
    meta.scopes_supported = Some(vec![
        "openid".into(),
        "read:*:*".into(),
        "write:*:*".into(),
        "infer:model:*".into(),
    ]);
    axum::Json(meta)
}

/// OAuth 2.1 Authorization Server service.
///
/// Runs an Axum HTTP server with OAuth endpoints. Token issuance is delegated
/// to PolicyService via ZMQ.
///
/// **Important**: The PolicyClient is created lazily inside `run()` rather than
/// in the factory. This is because OAuthService runs in its own tokio runtime
/// (separate thread), and ZMQ async I/O (TMQ) registers socket file descriptors
/// with the current runtime's epoll. A PolicyClient created in the main runtime
/// would hang when polled from the OAuth runtime.
pub struct OAuthService {
    config: OAuthConfig,
    /// Global TLS configuration (passed from factory, avoids re-loading config)
    tls_config: crate::config::TlsConfig,
    /// Signing key for creating the PolicyClient inside `run()`.
    signing_key: hyprstream_rpc::prelude::SigningKey,
    context: Arc<zmq::Context>,
    control_transport: TransportConfig,
    #[allow(dead_code)]
    verifying_key: ed25519_dalek::VerifyingKey,
}

impl OAuthService {
    pub fn new(
        config: OAuthConfig,
        tls_config: crate::config::TlsConfig,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        context: Arc<zmq::Context>,
        control_transport: TransportConfig,
        verifying_key: ed25519_dalek::VerifyingKey,
    ) -> Self {
        Self {
            config,
            tls_config,
            signing_key,
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

            // Resolve TLS configuration (tls_config passed from factory, not re-loaded)
            let rustls_config = crate::server::tls::resolve_rustls_config(
                &self.tls_config,
                self.config.tls_cert.as_ref(),
                self.config.tls_key.as_ref(),
            )
            .await
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("TLS config: {e}")))?;

            let scheme = if rustls_config.is_some() { "https" } else { "http" };

            // Create PolicyClient HERE, inside the OAuth runtime, so that ZMQ
            // async I/O (TMQ) registers socket FDs with THIS runtime's epoll.
            // Creating it in the factory (main runtime) would cause hangs.
            let policy_client = PolicyClient::new(
                self.signing_key,
                hyprstream_rpc::RequestIdentity::local(),
            );

            // Attempt to load the user credential store for Ed25519 device verification.
            // Failure is non-fatal; the verify endpoint will report "not configured" instead.
            let user_store: Option<Arc<dyn crate::auth::user_store::UserStore + Send + Sync>> = {
                let credentials_dir = crate::config::HyprConfig::load()
                    .map(|c| c.config_dir().join("credentials"))
                    .unwrap_or_else(|_| {
                        dirs::config_dir()
                            .unwrap_or_else(|| std::path::PathBuf::from("/etc/hyprstream"))
                            .join("hyprstream")
                            .join("credentials")
                    });
                match crate::auth::user_store::LocalKeyStore::load(&credentials_dir) {
                    Ok(store) => {
                        info!("User credential store loaded from {:?}", credentials_dir);
                        Some(Arc::new(store))
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Could not load user credential store (device verify will require manual setup): {}",
                            e
                        );
                        None
                    }
                }
            };

            // Create shared state
            let mut oauth_state = OAuthState::new(
                &self.config,
                policy_client,
                self.verifying_key.to_bytes(),
            );
            if let Some(store) = user_store {
                oauth_state = oauth_state.with_user_store(store);
            }
            let state = Arc::new(oauth_state);
            state.spawn_code_sweeper();

            // Create router with configurable CORS
            let app = create_app(state, &self.config.cors);

            info!(
                "Authorization server metadata at {scheme}://{addr}/.well-known/oauth-authorization-server",
            );

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            let _ = hyprstream_rpc::notify::ready();

            // Run HTTP(S) server with graceful shutdown
            crate::server::tls::serve_app(addr, app, rustls_config, shutdown, "OAuthService").await
        })
    }
}

/// RFC 9728 Protected Resource Metadata.
///
/// Typed representation of the JSON served at `/.well-known/oauth-protected-resource`.
/// Used by MCP, OAI, Flight, and QUIC services to advertise their OAuth authorization server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProtectedResourceMetadata {
    pub resource: String,
    pub authorization_servers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_methods_supported: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes_supported: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_documentation: Option<String>,
}

/// Create a Protected Resource Metadata response (RFC 9728).
///
/// Used by MCP and OAI servers to advertise their OAuth authorization server.
pub fn protected_resource_metadata(resource_url: &str, oauth_issuer_url: &str) -> ProtectedResourceMetadata {
    ProtectedResourceMetadata {
        resource: resource_url.to_owned(),
        authorization_servers: vec![oauth_issuer_url.to_owned()],
        bearer_methods_supported: Some(vec!["header".to_owned()]),
        scopes_supported: None,
        resource_name: None,
        resource_documentation: None,
    }
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
        assert_eq!(meta.resource, "http://localhost:6790");
        assert_eq!(meta.authorization_servers[0], "http://localhost:6791");
        assert_eq!(meta.bearer_methods_supported.as_deref(), Some(&["header".to_owned()][..]));
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
