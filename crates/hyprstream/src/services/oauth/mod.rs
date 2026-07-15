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

pub mod auth;
pub mod authorize;
pub mod challenge;
pub mod cimd_cache;
pub mod client_auth;
pub mod device;
pub mod did_document;
pub mod federation_entity;
pub mod dpop;
pub mod introspection;
pub mod jwks;
pub mod jwt_bearer;
pub mod login_page;
pub mod metadata;
pub mod mount_ticket;
pub mod oauth2_userinfo;
pub mod oidc_callback;
pub mod oidc_discovery;
pub mod par;
pub mod registration;
pub mod revocation;
pub mod scim;
pub mod scim_types;
pub mod session;
pub mod spiffe;
pub mod state;
pub mod token;
pub mod token_exchange;
pub mod token_store;
pub mod user_mapping;
pub mod wit_bootstrap;
pub mod user_service;
pub mod userinfo;
pub mod device_enrollment;
pub mod rpc_handler;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::Spawnable;
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
    // ── Public routes ──────────────────────────────────────────────────────────
    // No Bearer token required. Includes all OAuth flow endpoints (clients are
    // unauthenticated when they arrive) and SCIM discovery (RFC 7644 §4).
    let public_router = Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(metadata::authorization_server_metadata),
        )
        .route(
            "/.well-known/oauth-protected-resource",
            get(oauth_self_protected_resource_metadata),
        )
        .route(
            "/.well-known/openid-configuration",
            get(metadata::openid_configuration),
        )
        .route("/.well-known/spiffe/bundle", get(spiffe::spiffe_bundle))
        .route(
            "/.well-known/openid-federation",
            get(federation_entity::entity_configuration),
        )
        .route("/oauth/register", post(registration::register_client))
        .route(
            "/oauth/authorize",
            get(authorize::authorize_get).post(authorize::authorize_post),
        )
        .route("/oauth/par", post(par::push_authorization_request))
        .route("/oauth/token", post(token::exchange_token))
        .route("/oauth/spiffe/wit", post(spiffe::exchange_workload_wit))
        .route("/oauth/jwks", get(jwks::jwks))
        .route("/oauth/device", post(device::device_authorize))
        .route(
            "/oauth/device/verify",
            get(device::verify_get).post(device::verify_post),
        )
        .route("/oauth/device/nonce", get(device::device_nonce))
        .route("/api/device/challenge", post(device_enrollment::device_challenge_handler))
        .route("/api/device/enroll", post(device_enrollment::device_enroll_handler))
        .route("/oauth/revoke", post(revocation::revoke_token))
        .route("/oauth/logout", post(handle_logout))
        .route(
            "/oauth/external/authorize/:provider",
            get(oidc_callback::external_authorize),
        )
        .route(
            "/oauth/callback/:provider",
            get(oidc_callback::external_callback),
        )
        // SCIM discovery endpoints — RFC 7644 §4 requires these to be unauthenticated
        .route("/scim/v2/Schemas", get(scim::schemas))
        .route("/scim/v2/ResourceTypes", get(scim::resource_types))
        .route(
            "/scim/v2/ServiceProviderConfig",
            get(scim::service_provider_config),
        );

    // ── Protected routes ───────────────────────────────────────────────────────
    // All require a valid Bearer token (validated by require_bearer_token).
    // Inserts AuthenticatedUser into request extensions for downstream handlers.
    let protected_router = Router::new()
        .route("/oauth/introspect", post(introspection::introspect_token))
        .route("/oauth/wit", post(wit_bootstrap::issue_browser_wit))
        .route(
            "/oauth/mount-ticket",
            post(mount_ticket::issue_mount_ticket),
        )
        .route(
            "/oauth/spiffe/service-svid",
            post(spiffe::issue_service_svid),
        )
        .route(
            "/oauth/userinfo",
            get(userinfo::userinfo).post(userinfo::userinfo),
        )
        // SCIM 2.0 user management (RFC 7644)
        .route(
            "/scim/v2/Users",
            get(scim::list_users).post(scim::create_user),
        )
        .route(
            "/scim/v2/Users/:id",
            get(scim::get_user)
                .put(scim::replace_user)
                .delete(scim::delete_user),
        )
        .route(
            "/scim/v2/Users/:id/keys",
            get(scim::list_user_keys).post(scim::add_user_key),
        )
        .route(
            "/scim/v2/Users/:id/keys/:fingerprint",
            axum::routing::delete(scim::remove_user_key),
        )
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::require_bearer_token,
        ));

    // ── DID-document routes ──────────────────────────────────────────────────────
    // Public, secret-free GET endpoints (did:web + atproto handle resolution).
    // Per W3C DID Core a DID document is an open identity anchor that must be
    // fetchable cross-origin with arbitrary request headers, so these get their own
    // permissive-header CORS layer (CorsConfig::did_document) — kept separate from
    // the broad public router so the relaxation never reaches secret-bearing
    // endpoints like /oauth/token (closes #472).
    let mut did_router = Router::new()
        // Phase 0c — did:web document endpoints
        .route("/.well-known/did.json", get(did_document::root_did_document))
        // atproto handle→DID HTTP resolution (#500) — plaintext bare DID
        .route("/.well-known/atproto-did", get(did_document::atproto_did))
        .route("/users/:username/did.json", get(did_document::user_did_document))
        .route("/clients/:client_id/did.json", get(did_document::client_did_document));

    let mut api_router = public_router.merge(protected_router);

    // CORS innermost per route-group, before the shared logging layer.
    // The broad router keeps the restrictive header allowlist; only the DID routes
    // opt into permissive headers, and only their paths carry that CORS layer.
    if cors_config.enabled {
        api_router = api_router.layer(crate::server::middleware::cors_layer(cors_config));
        did_router = did_router.layer(crate::server::middleware::cors_layer(
            &crate::config::CorsConfig::did_document(),
        ));
    }

    let router = api_router.merge(did_router).layer(axum::middleware::from_fn(
        |req: axum::extract::Request, next: axum::middleware::Next| async move {
            let method = req.method().clone();
            let uri = req.uri().clone();
            tracing::info!(%method, %uri, "OAuth request");
            next.run(req).await
        },
    ));

    router.with_state(state)
}

/// POST /oauth/logout — clear session and redirect.
async fn handle_logout(
    State(state): State<Arc<OAuthState>>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    if let Some(session_id) = session::extract_session_id(&headers) {
        state.sessions.remove(&session_id).await;
    }
    (
        axum::http::StatusCode::OK,
        [(
            axum::http::header::SET_COOKIE,
            session::clear_session_cookie(),
        )],
        "Logged out",
    )
        .into_response()
}

/// RFC 9728 Protected Resource Metadata for the OAuth server itself.
///
/// Returns the DiscoveryService QUIC endpoint as the `resource` URL so that
/// browsers can bootstrap a WebTransport connection via DiscoveryService,
/// then resolve all other service endpoints from there.
async fn oauth_self_protected_resource_metadata(
    State(_state): State<Arc<OAuthState>>,
) -> axum::Json<ProtectedResourceMetadata> {
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    let issuer_url = config.oauth.issuer_url();

    // Use issuer URL as the resource
    let resource = issuer_url.clone();

    let mut meta = protected_resource_metadata(&resource, &issuer_url);
    meta.resource_name = Some("HyprStream OAuth 2.1 Authorization Server".to_owned());
    meta.scopes_supported = Some(vec![
        "openid".into(),
        "read:*:*".into(),
        "write:*:*".into(),
        "infer:model:*".into(),
    ]);

    // Include the QUIC TLS cert hash so browsers can pin the self-signed certificate.
    if let Ok((cert_chain, _)) = config.quic.load_tls_materials() {
        meta.x_cert_hash = Some(hyprstream_rpc::transport::zmtp_quic::cert_hash(
            &cert_chain[0],
        ));
    }

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
/// Build the OAuth service's reach-only iroh endpoint.
///
/// Both inbound ALPNs are refused until #1027/#726 supplies independently
/// verified application/session proof. In particular, the local user-CRUD
/// handler is never installed behind `AnySigner`, and anonymous MoQ peers never
/// receive the process-global origin. The endpoint remains usable as the shared
/// outbound dialer. Native-only.
async fn build_oauth_iroh_substrate(
    transport_secret: [u8; 32],
) -> anyhow::Result<hyprstream_rpc::transport::iroh_substrate::IrohSubstrate> {
    use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler};

    // `presets::N0` discovery (pkarr publisher + n0 DNS) is enabled by
    // `IrohSubstrate::new`; that reach state grants no inbound authority.
    IrohSubstrate::new(
        transport_secret,
        RefuseHandler::new("OAuth MoQ disabled pending verified session proof (#1027/#726)"),
        RefuseHandler::new("OAuth RPC disabled pending verified application proof (#1027)"),
    )
    .await
}

pub struct OAuthService {
    config: OAuthConfig,
    /// Global TLS configuration (passed from factory, avoids re-loading config)
    tls_config: crate::config::TlsConfig,
    /// Global QUIC configuration for cert-hash publication in DID doc (#185).
    quic_config: Option<crate::config::QuicConfig>,
    /// Signing key for creating the PolicyClient inside `run()`.
    signing_key: hyprstream_rpc::prelude::SigningKey,
    control_transport: TransportConfig,
    #[allow(dead_code)]
    verifying_key: ed25519_dalek::VerifyingKey,
    /// JWT verifying key (CA key) for JWKS endpoint. This is the key that verifies
    /// JWTs signed by PolicyService, derived from the root signing key.
    jwt_verifying_key: [u8; 32],
    /// Shared JTI blocklist (same Arc as PolicyService) for cross-plane revocation.
    jti_blocklist: Option<Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>>,
}

impl OAuthService {
    pub fn new(
        config: OAuthConfig,
        tls_config: crate::config::TlsConfig,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        control_transport: TransportConfig,
        verifying_key: ed25519_dalek::VerifyingKey,
        jwt_verifying_key: ed25519_dalek::VerifyingKey,
    ) -> Self {
        Self {
            config,
            tls_config,
            quic_config: None,
            signing_key,
            control_transport,
            verifying_key,
            jwt_verifying_key: jwt_verifying_key.to_bytes(),
            jti_blocklist: None,
        }
    }

    /// Attach the global QUIC configuration for DID-doc cert-hash publication (#185).
    pub fn with_quic_config(mut self, quic: crate::config::QuicConfig) -> Self {
        self.quic_config = Some(quic);
        self
    }

    /// Attach the shared JTI blocklist (same Arc as PolicyService).
    pub fn with_jti_blocklist(mut self, bl: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>) -> Self {
        self.jti_blocklist = Some(bl);
        self
    }
}

impl Spawnable for OAuthService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, self.control_transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        // Use single-threaded runtime + LocalSet because HTTP handlers make ZMQ RPC
        // calls (e.g., policy_client.issue_token()), and ZMQ clients use spawn_local.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
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

            // Create RPC clients HERE, inside the OAuth runtime, so that ZMQ
            // async I/O (TMQ) registers socket FDs with THIS runtime's epoll.
            // Creating them in the factory (main runtime) would cause hangs.

            // Bootstrap: Get service verifying keys from trust store.
            // The trust store is populated during startup by depends_on services.
            let policy_vk = match hyprstream_service::global_trust_store().resolve_one("policy") {
                Some(vk) => vk,
                None => {
                    return Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                        "trust store has no policy key — startup must populate it".to_owned(),
                    ));
                }
            };
            let policy_client = PolicyClient::for_service(
                self.signing_key.clone(),
                policy_vk,
                None,
            ).map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(
                format!("failed to create PolicyClient: {e}"),
            ))?;

            // Get discovery key from trust store (populated by depends_on = ["discovery"]).
            // Using trust store avoids RPC calls which require LocalSet context.
            let discovery_vk = match hyprstream_service::global_trust_store().resolve_one("discovery") {
                Some(vk) => vk,
                None => {
                    return Err(hyprstream_rpc::error::RpcError::SpawnFailed(
                        "trust store has no discovery key — ensure discovery is in depends_on".to_owned(),
                    ));
                }
            };
            let discovery_client = crate::services::DiscoveryClient::for_service(
                self.signing_key.clone(),
                discovery_vk,
                None,
            ).map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(
                format!("failed to create DiscoveryClient: {e}"),
            ))?;

            let credentials_dir = crate::auth::identity_store::credentials_dir().map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(
                    format!("failed to resolve credentials directory: {e}"),
                )
            })?;

            // Load the user store and token store based on configured backend.
            // Failure is non-fatal; endpoints will report "not configured" instead.
            use crate::config::CredentialsBackend;
            let credentials_config = crate::config::HyprConfig::load()
                .map(|c| c.credentials)
                .unwrap_or_default();

            let mut device_store_opt: Option<Arc<dyn crate::auth::DeviceStore>> = None;
            let user_store: Option<Arc<dyn crate::auth::user_store::UserStore>> = match credentials_config.backend {
                CredentialsBackend::Rocksdb => {
                    match crate::auth::RocksDbUserStore::open(&credentials_dir) {
                        Ok(store) => {
                            info!("User store (RocksDB) opened at {:?}", credentials_dir);
                            let arc = Arc::new(store);
                            device_store_opt = Some(arc.clone() as Arc<dyn crate::auth::DeviceStore>);
                            Some(arc as Arc<dyn crate::auth::user_store::UserStore>)
                        }
                        Err(e) => {
                            tracing::warn!("Could not open user store (endpoints will report 'not configured'): {}", e);
                            None
                        }
                    }
                }
                CredentialsBackend::Valkey => {
                    #[cfg(feature = "valkey")]
                    {
                        let url = &credentials_config.valkey.url;
                        match crate::auth::ValkeyUserStore::connect(url).await {
                            Ok(store) => {
                                info!("User store (Valkey) connected at {url}");
                                Some(Arc::new(store))
                            }
                            Err(e) => {
                                tracing::warn!("Could not connect user store (Valkey): {e}");
                                None
                            }
                        }
                    }
                    #[cfg(not(feature = "valkey"))]
                    {
                        tracing::error!("credentials.backend = \"valkey\" but binary was not compiled with --features valkey");
                        None
                    }
                }
            };

            // Load the CA JWT signing key for browser WIT issuance (POST /oauth/wit).
            // Also seed the signing key store from the same root key.
            let secrets_dir = credentials_dir.clone();
            let oauth_config_arc = Arc::new(self.config.clone());

            // Load or initialize ES256 + ML-DSA rotation stores (independent of CA key).
            // Uses global singletons — shared with PolicyService and ServiceContext.
            let es256_store = crate::auth::key_rotation::global_es256_key_store(
                &secrets_dir,
                &self.config,
            );
            info!("ES256 (P-256) signing key rotation store loaded");

            let ml_dsa_store = crate::auth::key_rotation::global_ml_dsa_key_store(
                &secrets_dir,
                &self.config,
            );
            info!("ML-DSA-65 signing key rotation store loaded");
            crate::auth::key_rotation::refresh_ml_dsa_verifying_keys(&ml_dsa_store).await;
            let crypto_policy = hyprstream_rpc::envelope::envelope_policy_from_env();

            // MAC #547 / B2 (#674): construct the S6 grant-path audit sink. A
            // startup-time snapshot of the active ML-DSA-65 key (rotation
            // during the process lifetime is a follow-up — S7's audit key
            // provisioning is otherwise the same "load once at boot" shape as
            // the rest of this factory). The node's enforced crypto policy
            // decides whether the PQ layer is REQUIRED (fail-closed if the key
            // snapshot comes back empty) or ignored (Classical).
            let audit_sink: Option<Arc<dyn crate::mac::audit::AuditSink>> = {
                let audit_pq_key = ml_dsa_store.active_key().await;
                let signer = crate::mac::audit::cose::OwnedCoseAuditSigner::new(
                    Arc::new(self.signing_key.clone()),
                    audit_pq_key,
                    crypto_policy,
                );
                // Fail-closed the SAME way as "store failed to open": a signer
                // that can't sign under the enforced policy is not a usable
                // audit sink. Do NOT install it — installing it anyway would
                // let every real request silently fail two audit-write
                // attempts deep (mint's Permit downgrade, then the best-effort
                // fallback deny record, both via this same broken signer) with
                // only this one startup log line as any indication, instead of
                // the immediate, obvious "audit trail is not configured" error
                // the grant path already gives when `audit_sink` is `None`.
                if !signer.can_sign() {
                    tracing::error!(
                        "MAC audit signer cannot sign under the enforced Hybrid crypto \
                         policy (no ML-DSA-65 key provisioned) — the S6 grant path will \
                         fail closed (deny) on every request until a key is provisioned"
                    );
                    None
                } else {
                    match crate::mac::audit::WalAuditStore::open(
                        credentials_dir.join("mac-audit"),
                        signer,
                    ) {
                        Ok(store) => {
                            info!("MAC grant-path audit store opened (WAL, hash-chained, checkpointed)");
                            Some(Arc::new(store) as Arc<dyn crate::mac::audit::AuditSink>)
                        }
                        Err(e) => {
                            // Fail-closed by omission: `audit_sink` stays `None`, and
                            // the grant path denies every request rather than mint
                            // unaudited tokens (see `exchange_ucan_grant`).
                            tracing::error!(
                                "MAC grant-path audit store failed to open — the S6 grant \
                                 path will fail closed (deny) on every request: {e:?}"
                            );
                            None
                        }
                    }
                }
            };

            let (ca_jwt_key, signing_key_store) = match crate::auth::identity_store::load_ca_signing_key(&credentials_dir) {
                Ok(root_key) => {
                    let key = hyprstream_rpc::node_identity::derive_purpose_key(&root_key, "hyprstream-jwt-v1");
                    info!("CA JWT signing key loaded — POST /oauth/wit available");

                    // Load or initialize the multi-slot Ed25519 rotation store.
                    let store_arc = crate::auth::key_rotation::global_ed25519_key_store(
                        &secrets_dir,
                        &self.config,
                    );

                    // Publish the current Ed25519 rotation-slot verifying keys to the
                    // process-shared handle so the OAI/HTTP validator (`verify_token_claims`)
                    // accepts tokens signed by any active/lead/drain slot from startup —
                    // before the first rotation tick. Mirrors the /oauth/jwks key set.
                    crate::auth::key_rotation::refresh_ed25519_verifying_keys(&store_arc).await;
                    crate::auth::key_rotation::initialize_composite_key_set(
                        &secrets_dir,
                        &store_arc,
                        &ml_dsa_store,
                        Arc::new(key.clone()),
                        self.config.drain_secs(),
                    ).await.map_err(|error| {
                        hyprstream_rpc::error::RpcError::SpawnFailed(format!(
                            "composite key-set initialization failed: {error:#}"
                        ))
                    })?;

                    // Spawn the background rotation task (rotates all algorithm stores).
                    crate::auth::key_rotation::spawn_rotation_task(
                        Arc::clone(&oauth_config_arc),
                        secrets_dir.clone(),
                        Arc::clone(&store_arc),
                        crate::auth::key_rotation::RotationStores {
                            es256: Some(Arc::clone(&es256_store)),
                            ml_dsa: Some(Arc::clone(&ml_dsa_store)),
                            composite_ca_key: Arc::new(key.clone()),
                        },
                    );
                    info!("JWT signing key rotation task started (active_days={}, lead_days={}, drain_days={})",
                        self.config.jwt_key_active_days,
                        self.config.jwt_key_lead_days,
                        self.config.jwt_key_drain_days,
                    );

                    (Some(key), Some(store_arc))
                }
                Err(e) => {
                    tracing::warn!("Cannot load CA signing key — POST /oauth/wit unavailable and key rotation disabled: {e}");
                    (None, None)
                }
            };

            // Create shared state — JWKS serves the CA JWT verifying key (from PolicyService)
            let jwt_verifying_key = self.jwt_verifying_key;
            let mut oauth_state = OAuthState::new(
                &self.config,
                policy_client,
                discovery_client.clone(),
                jwt_verifying_key,
            );
            if let Some(store) = user_store {
                oauth_state = oauth_state.with_user_store(store);
            }
            if let Some(ds) = device_store_opt {
                oauth_state = oauth_state.with_device_store(ds);
            }
            oauth_state = oauth_state
                .with_signing_key(self.signing_key.clone(), crypto_policy)
                .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("{e:#}")))?;
            if let Some(key) = ca_jwt_key {
                oauth_state = oauth_state.with_ca_jwt_key(key);
            }
            if let Some(store) = signing_key_store {
                oauth_state = oauth_state.with_signing_key_store(store);
            }
            oauth_state = oauth_state.with_es256_key_store(Arc::clone(&es256_store));
            {
                oauth_state = oauth_state.with_ml_dsa_key_store(Arc::clone(&ml_dsa_store));
            }
            if let Some(sink) = audit_sink {
                oauth_state = oauth_state.with_audit_sink(sink);
            }
            if let Some(bl) = self.jti_blocklist {
                oauth_state = oauth_state.with_jti_blocklist(bl);
            }
            // Populate legacy JWKS nbf/exp from signing-key file mtime (used when store absent).
            let key_nbf = crate::auth::identity_store::node_signing_key_mtime(&credentials_dir);
            oauth_state = oauth_state.with_jwt_key_timestamps(key_nbf, key_nbf + 14 * 86400);

            // Open persistent refresh token store (non-fatal — tokens simply don't survive restart).
            let token_db_path = credentials_dir.join("oauth-tokens");
            let token_store: Option<Arc<dyn crate::services::oauth::token_store::TokenStore>> = match credentials_config.backend {
                CredentialsBackend::Rocksdb => {
                    match crate::services::oauth::token_store::RocksDbTokenStore::open(&token_db_path) {
                        Ok(s) => {
                            info!("Refresh token store (RocksDB) opened at {:?}", token_db_path);
                            Some(Arc::new(s))
                        }
                        Err(e) => {
                            tracing::warn!("Could not open refresh token store (tokens will not survive restart): {}", e);
                            None
                        }
                    }
                }
                CredentialsBackend::Valkey => {
                    #[cfg(feature = "valkey")]
                    {
                        let url = &credentials_config.valkey.url;
                        match crate::services::oauth::token_store::ValkeyTokenStore::connect(url).await {
                            Ok(s) => {
                                info!("Refresh token store (Valkey) connected at {url}");
                                Some(Arc::new(s))
                            }
                            Err(e) => {
                                tracing::warn!("Could not connect refresh token store (Valkey): {e}");
                                None
                            }
                        }
                    }
                    #[cfg(not(feature = "valkey"))]
                    {
                        None
                    }
                }
            };
            if let Some(store) = token_store {
                oauth_state.with_token_store_impl(store);
            }

            // Resolve discovery URL at startup with LocalSet (RPC calls need LocalSet context).
            // Cache it so HTTP handlers don't need to make RPC calls.
            #[allow(clippy::expect_used)]
            let cached_discovery_url = {
                let dc = discovery_client;
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("failed to create discovery resolve runtime");
                    let local = tokio::task::LocalSet::new();
                    local.block_on(&rt, async {
                        match dc.get_endpoints("discovery").await {
                            Ok(service_endpoints) => service_endpoints
                                .endpoints
                                .iter()
                                .find(|ep| ep.socket_kind == "quic")
                                .and_then(|ep| {
                                    let stripped = ep.endpoint.strip_prefix("quic://")?;
                                    let parts: Vec<&str> = stripped.splitn(3, ':').collect();
                                    if parts.len() >= 3 {
                                        Some(format!("https://{}:{}", parts[0], parts[2]))
                                    } else {
                                        None
                                    }
                                }),
                            Err(_) => None,
                        }
                    })
                })
                .join()
                .ok()
                .flatten()
            };
            if let Some(_url) = cached_discovery_url {
                // Discovery URL caching removed - use issuer URL directly
            }

            // Publish QUIC cert hash in DID doc (#185): close the two-trust-roots / TOFU gap.
            // The leaf cert hash (SHA-256 of cert DER) is stored in OAuthState and
            // included in the `#quic` service entry at `GET /.well-known/did.json`.
            if let Some(ref quic_cfg) = self.quic_config {
                if quic_cfg.enabled {
                    if let Ok((cert_chain, _)) = quic_cfg.load_tls_materials() {
                        if let Some(leaf) = cert_chain.first() {
                            let mut hash = [0u8; 32];
                            let digest = ring::digest::digest(&ring::digest::SHA256, leaf);
                            hash.copy_from_slice(digest.as_ref());
                            // Derive the public QUIC URI: same host as issuer, port from QUIC bind addr.
                            let quic_port = quic_cfg.socket_addr().map(|a| a.port()).unwrap_or(4433);
                            let issuer_host = url::Url::parse(&oauth_state.issuer_url)
                                .ok()
                                .and_then(|u| u.host_str().map(str::to_owned))
                                .unwrap_or_else(|| "localhost".to_owned());
                            let quic_uri = format!("https://{issuer_host}:{quic_port}");
                            oauth_state = oauth_state.with_quic_transport(quic_uri, vec![hash]);
                        }
                    }

                    // OAuth's iroh inbound ALPNs are deliberately refused until
                    // fresh application/session proof exists (#1027/#726), so do
                    // not advertise them as an available DID service.
                }
            }

            let state = Arc::new(oauth_state);
            state.spawn_code_sweeper();

            // Phase 0.5 Stage D — publish OIDF entity statement to DiscoveryService
            // at startup AND periodically thereafter. Periodic re-publish keeps
            // the cached statement fresh as signing keys rotate and the embedded
            // JWKS changes; entity statements carry a 24h exp so any longer gap
            // leaves federation peers falling through to HTTPS unnecessarily.
            //
            // Non-fatal on failure: HTTPS fallback continues to work either way.
            {
                let publish_state = state.clone();
                // Re-publish at 1/4 of the entity-statement exp (24h) so we
                // refresh the cached statement well before consumers reject it
                // as expired. Concretely: every 6h. Initial publish happens
                // immediately on the first iteration of the loop.
                let republish_interval = std::time::Duration::from_secs(6 * 3600);
                tokio::task::spawn_local(async move {
                    let mut tick = tokio::time::interval(republish_interval);
                    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        tick.tick().await;
                        federation_entity::publish_entity_statement_to_discovery(
                            publish_state.clone(),
                        )
                        .await;
                    }
                });
            }

            // Create router with configurable CORS
            let app = create_app(state.clone(), &self.config.cors);

            info!(
                "Authorization server metadata at {scheme}://{addr}/.well-known/oauth-authorization-server",
            );

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            let _ = hyprstream_rpc::notify::ready();

            // User-CRUD RPC serve, alongside the HTTP server. #136: bridged
            // dispatch over the registered transport (inproc/ipc) instead of the
            // ZMQ ROUTER. A dedicated `serve_shutdown` stops it once the HTTP
            // server exits, so the task joins cleanly.
            let control_transport = self.control_transport.clone();
            let rpc_signing_key = self.signing_key.clone();
            let rpc_state = state.clone();
            // Bind OAuth's domain-separated outbound iroh carrier when enabled.
            // Its refused inbound ALPNs are not advertised in the DID document.
            let iroh_enabled = self.quic_config.as_ref().is_some_and(|q| q.enabled && q.iroh);
            let serve_shutdown = Arc::new(Notify::new());
            let serve_shutdown_task = Arc::clone(&serve_shutdown);
            let rpc_loop = tokio::task::spawn_local(async move {
                let handler = rpc_handler::OAuthRpcHandler::new(
                    rpc_state,
                    control_transport.clone(),
                    rpc_signing_key.clone(),
                );
                let nonce_cache = Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new());
                let bridge = match hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge::spawn(
                    handler,
                    nonce_cache,
                    0,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("OAuth RPC bridge spawn error: {}", e);
                        return;
                    }
                };
                let processor: Arc<dyn hyprstream_rpc::transport::rpc_session::IrohRequestProcessor> =
                    Arc::new(bridge);

                // Reach-only iroh endpoint. Both inbound ALPNs refuse before the
                // OAuth user-CRUD bridge or global MoQ origin can be reached.
                let _iroh_substrate = if iroh_enabled {
                    let transport_key = hyprstream_rpc::node_identity::derive_purpose_key(
                        &rpc_signing_key,
                        "hyprstream-iroh-transport-v1",
                    );
                    match build_oauth_iroh_substrate(transport_key.to_bytes()).await {
                        Ok(substrate) => {
                            let _ = hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(
                                substrate.owned_client_endpoint(),
                            );
                            Some(substrate)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "OAuth iroh substrate bind failed; continuing without iroh: {e}"
                            );
                            None
                        }
                    }
                } else {
                    None
                };

                if let Err(e) = hyprstream_rpc::service::serve::serve_bridged(
                    &control_transport,
                    processor,
                    rpc_signing_key,
                    serve_shutdown_task,
                    None,
                )
                .await
                {
                    tracing::error!("OAuth RPC serve error: {}", e);
                }

                // Drain the iroh substrate (accept loop + handlers) on shutdown.
                if let Some(substrate) = _iroh_substrate {
                    if let Err(e) = substrate.shutdown().await {
                        tracing::warn!("OAuth iroh substrate shutdown error: {e}");
                    }
                }
            });

            // Run HTTP(S) server with graceful shutdown
            let _ = crate::server::tls::serve_app(addr, app, rustls_config, shutdown, "OAuthService").await;

            // HTTP server stopped — stop the RPC serve and join it. notify_one
            // (not notify_waiters) stores a permit if the serve task hasn't yet
            // armed its `notified()` await, so the signal can't be missed even if
            // the HTTP server exited before the RPC task reached serve_bridged.
            serve_shutdown.notify_one();
            let _ = rpc_loop.await;

            Ok(())
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
    /// Base64 SHA-256 hash of the TLS certificate for WebTransport cert pinning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_cert_hash: Option<String>,
}

/// Create a Protected Resource Metadata response (RFC 9728).
///
/// Used by MCP and OAI servers to advertise their OAuth authorization server.
pub fn protected_resource_metadata(
    resource_url: &str,
    oauth_issuer_url: &str,
) -> ProtectedResourceMetadata {
    ProtectedResourceMetadata {
        resource: resource_url.to_owned(),
        authorization_servers: vec![oauth_issuer_url.to_owned()],
        bearer_methods_supported: Some(vec!["header".to_owned()]),
        scopes_supported: None,
        resource_name: None,
        resource_documentation: None,
        x_cert_hash: None,
    }
}

#[cfg(test)]
mod tests {
    use super::registration::validate_redirect_uri;
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oauth_iroh_refuses_arbitrary_rpc_and_anonymous_moq_before_dispatch(
    ) -> anyhow::Result<()> {
        use hyprstream_rpc::transport::iroh_substrate::{
            ALPN_HYPRSTREAM_RPC, ALPN_MOQ_LITE, IrohSubstrate, NoopHandler,
        };
        use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
        use hyprstream_rpc::ToCapnp as _;
        use iroh::{EndpointAddr, TransportAddr};
        use moq_net::Client;

        let oauth_signer = hyprstream_rpc::prelude::SigningKey::generate(&mut rand::rngs::OsRng);
        let transport_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &oauth_signer,
            "hyprstream-iroh-transport-v1",
        );
        let server = build_oauth_iroh_substrate(transport_key.to_bytes()).await?;
        assert_ne!(
            server.endpoint_id().as_bytes(),
            oauth_signer.verifying_key().as_bytes(),
            "transport address must be domain-separated from the OAuth identity key"
        );
        let addr = EndpointAddr::from_parts(
            server.endpoint_id(),
            server
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(TransportAddr::Ip),
        );
        let client_key = hyprstream_rpc::prelude::SigningKey::generate(&mut rand::rngs::OsRng);
        let client = IrohSubstrate::new(
            client_key.to_bytes(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await?;

        // A valid envelope signed by an arbitrary, unenrolled key never reaches
        // OAuthRpcHandler: the
        // ALPN closes before any request processor/user CRUD dispatch exists.
        let signed_crud = SignedEnvelope::new_signed(
            RequestEnvelope::anonymous(b"oauth-user-crud".to_vec()),
            &client_key,
        );
        let mut message = capnp::message::Builder::new_default();
        signed_crud.write_to(
            &mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>(),
        );
        let mut signed_crud_bytes = Vec::new();
        capnp::serialize::write_message(&mut signed_crud_bytes, &message)?;
        let rpc = client.connect(addr.clone(), ALPN_HYPRSTREAM_RPC).await?;
        if let Ok((mut send, mut recv)) = rpc.open_bi().await {
            let _ = send.write_all(&signed_crud_bytes).await;
            let _ = send.finish();
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                recv.read_to_end(1024),
            )
            .await;
            assert!(
                !matches!(response, Ok(Ok(ref bytes)) if !bytes.is_empty()),
                "refused OAuth RPC ALPN must not produce a user-CRUD response"
            );
        }

        let moq = client.connect(addr, ALPN_MOQ_LITE).await?;
        let session = web_transport_iroh::Session::raw(moq);
        assert!(
            Client::new().connect(session).await.is_err(),
            "anonymous OAuth MoQ carrier must not obtain publish/subscribe access"
        );

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    #[test]
    fn test_protected_resource_metadata() {
        let meta = protected_resource_metadata("http://localhost:6790", "http://localhost:6791");
        assert_eq!(meta.resource, "http://localhost:6790");
        assert_eq!(meta.authorization_servers[0], "http://localhost:6791");
        assert_eq!(
            meta.bearer_methods_supported.as_deref(),
            Some(&["header".to_owned()][..])
        );
    }

    #[test]
    fn test_validate_redirect_uri_exact_match() {
        let registered = vec!["http://127.0.0.1:3000/callback".to_owned()];
        assert!(validate_redirect_uri(
            "http://127.0.0.1:3000/callback",
            &registered
        ));
        // Loopback URIs: port is ignored per RFC 8252, so different port still matches
        assert!(validate_redirect_uri(
            "http://127.0.0.1:4000/callback",
            &registered
        ));
        // Different path should NOT match
        assert!(!validate_redirect_uri(
            "http://127.0.0.1:3000/other",
            &registered
        ));
    }

    #[test]
    fn test_validate_redirect_uri_loopback_port_ignored() {
        let registered = vec!["http://127.0.0.1:3000/callback".to_owned()];
        // Different port on loopback should match per RFC 8252
        assert!(validate_redirect_uri(
            "http://127.0.0.1:9999/callback",
            &registered
        ));
        // Non-loopback should require exact match
        let non_loopback = vec!["https://example.com:3000/callback".to_owned()];
        assert!(!validate_redirect_uri(
            "https://example.com:4000/callback",
            &non_loopback
        ));
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
