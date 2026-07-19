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
pub mod device_enrollment;
pub mod did_document;
pub mod dpop;
pub mod federation_entity;
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
pub mod rpc_handler;
pub mod scim;
pub mod scim_types;
pub mod session;
pub mod spiffe;
pub mod state;
pub mod token;
pub mod token_exchange;
pub mod token_store;
pub mod user_mapping;
pub mod user_service;
pub mod userinfo;
pub mod wit_bootstrap;
pub mod xrpc;

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
        .route(
            "/api/device/challenge",
            post(device_enrollment::device_challenge_handler),
        )
        .route(
            "/api/device/enroll",
            post(device_enrollment::device_enroll_handler),
        )
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

    // ── com.atproto XRPC read slice (#1112) ────────────────────────────────
    // Four public read endpoints, conditionally mounted when the operator
    // opts in via `OAuthConfig::xrpc_read_slice`. Session endpoints
    // (createSession/getSession) are NOT here — they arrive with #1113/#948.
    // Route table lives in `xrpc::xrpc_routes()` — single source of truth.
    let public_router = if state.xrpc_read_slice {
        public_router.merge(xrpc::xrpc_routes())
    } else {
        public_router
    };

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
        .route(
            "/.well-known/did.json",
            get(did_document::root_did_document),
        )
        // atproto handle→DID HTTP resolution (#500) — plaintext bare DID
        .route("/.well-known/atproto-did", get(did_document::atproto_did))
        .route(
            "/users/:username/did.json",
            get(did_document::user_did_document),
        )
        .route(
            "/clients/:client_id/did.json",
            get(did_document::client_did_document),
        );

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

    let router = api_router
        .merge(did_router)
        .layer(axum::middleware::from_fn(
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
    State(state): State<Arc<OAuthState>>,
) -> axum::Json<ProtectedResourceMetadata> {
    // #1113 r4 F5: use the CANONICAL issuer from OAuthState (same
    // normalization as token claims/redirect), NOT raw config.issuer_url().
    // A configured trailing-slash/path must be consistent everywhere.
    let issuer_url = state.issuer_url.clone();

    // Use issuer URL as the resource (PDS = its own AS).
    let resource = issuer_url.clone();

    let mut meta = protected_resource_metadata(&resource, &issuer_url);
    meta.resource_name = Some("HyprStream OAuth 2.1 Authorization Server".to_owned());
    meta.scopes_supported = Some(self_resource_scopes());

    // Include the QUIC TLS cert hash so browsers can pin the self-signed certificate.
    let config = crate::config::HyprConfig::load().unwrap_or_default();
    if let Ok((cert_chain, _)) = config.quic.load_tls_materials() {
        meta.x_cert_hash = Some(hyprstream_rpc::transport::zmtp_quic::cert_hash(
            &cert_chain[0],
        ));
    }

    axum::Json(meta)
}

/// Scopes advertised by the OAuth service's own RFC 9728 protected-resource
/// document. Includes the atproto transition scopes (#1113) so stock atproto
/// clients can discover them at the resource (PDS = its own AS).
fn self_resource_scopes() -> Vec<String> {
    vec![
        "openid".into(),
        "read:*:*".into(),
        "write:*:*".into(),
        "infer:model:*".into(),
        "pds:attach".into(),
        "atproto".into(),
        "transition:generic".into(),
    ]
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
    pub fn with_jti_blocklist(
        mut self,
        bl: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>,
    ) -> Self {
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
            let policy_client = PolicyClient::for_local_bootstrap(
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
            let discovery_client = crate::services::DiscoveryClient::for_local_bootstrap(
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
            let crypto_policy = hyprstream_rpc::envelope::mandatory_envelope_policy();

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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::get_unwrap)]
mod tests {
    use super::registration::validate_redirect_uri;
    use super::*;

    fn encode_form(fields: &[(&str, &str)]) -> String {
        let mut serializer = url::form_urlencoded::Serializer::new(String::new());
        serializer.extend_pairs(fields.iter().copied());
        serializer.finish()
    }

    async fn post_form(
        app: &Router,
        uri: &str,
        fields: &[(&str, &str)],
        dpop: Option<&str>,
        accept_json: bool,
    ) -> axum::response::Response {
        use tower::ServiceExt as _;

        let mut builder = axum::http::Request::post(uri).header(
            axum::http::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        );
        if let Some(proof) = dpop {
            builder = builder.header("DPoP", proof);
        }
        if accept_json {
            builder = builder.header(axum::http::header::ACCEPT, "application/json");
        }
        app.clone()
            .oneshot(
                builder
                    .body(axum::body::Body::from(encode_form(fields)))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn get(app: &Router, uri: &str) -> axum::response::Response {
        use tower::ServiceExt as _;

        app.clone()
            .oneshot(
                axum::http::Request::get(uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn response_text(response: axum::response::Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn html_hidden_value<'a>(html: &'a str, name: &str) -> &'a str {
        let prefix = format!(r#"name="{name}" value=""#);
        let rest = html.split_once(&prefix).unwrap().1;
        rest.split_once('"').unwrap().0
    }

    fn dpop_proof(
        signing_key: &p256::ecdsa::SigningKey,
        htu: &str,
        jti: &str,
        nonce: Option<&str>,
    ) -> String {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        use p256::ecdsa::signature::Signer as _;

        let point = signing_key.verifying_key().to_encoded_point(false);
        let header = serde_json::json!({
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": {
                "kty": "EC",
                "crv": "P-256",
                "x": URL_SAFE_NO_PAD.encode(point.x().unwrap()),
                "y": URL_SAFE_NO_PAD.encode(point.y().unwrap()),
            }
        });
        let mut payload = serde_json::json!({
            "jti": jti,
            "htm": "POST",
            "htu": htu,
            "iat": chrono::Utc::now().timestamp(),
        });
        if let Some(value) = nonce {
            payload["nonce"] = serde_json::Value::String(value.to_owned());
        }
        let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
        let signing_input = format!("{header}.{payload}");
        let signature: p256::ecdsa::Signature = signing_key.sign(signing_input.as_bytes());
        format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    fn jwt_claims(token: &str) -> serde_json::Value {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

        let payload = token.split('.').nth(1).expect("JWT payload");
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).expect("JWT payload base64"))
            .expect("JWT payload JSON")
    }

    // Rust mirror of @atproto/oauth-types 0.7.5
    // atprotoOAuthTokenResponseSchema. The upstream schema extends the base
    // OAuth response with literal DPoP, required atproto DID sub, required
    // atproto-containing scope, and an absent id_token.
    #[derive(serde::Deserialize)]
    struct AtprotoOAuthTokenResponse075 {
        access_token: String,
        token_type: String,
        scope: String,
        refresh_token: Option<String>,
        expires_in: Option<f64>,
        sub: String,
    }

    fn parse_atproto_token_response_075(
        value: serde_json::Value,
    ) -> anyhow::Result<AtprotoOAuthTokenResponse075> {
        anyhow::ensure!(
            value.get("id_token").is_none(),
            "atproto OAuth token response must not contain id_token"
        );
        let parsed: AtprotoOAuthTokenResponse075 = serde_json::from_value(value)?;
        anyhow::ensure!(parsed.token_type == "DPoP", "token_type must be DPoP");
        anyhow::ensure!(
            parsed.scope.bytes().all(|byte| {
                byte == b' '
                    || byte == b'!'
                    || (b'#'..=b'[').contains(&byte)
                    || (b']'..=b'~').contains(&byte)
            }),
            "scope is not an OAuth scope value"
        );
        anyhow::ensure!(
            parsed.scope.split(' ').any(|scope| scope == "atproto"),
            "scope must contain atproto"
        );
        super::state::subject_did_for("https://schema.invalid", &parsed.sub)?;
        Ok(parsed)
    }

    // Publish a disk-backed composite Policy signing pair so this test uses
    // the real PolicyService JWT issuance path. The tiny authority directory
    // remains because the process-global mint barrier checks disk each time.
    fn configure_test_policy_signing_authority() -> anyhow::Result<()> {
        use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};

        let authority_dir = tempfile::TempDir::new()?.keep();
        let ledger = authority_dir.join("ledger.json");
        let committed = authority_dir.join("committed");
        let prefix = authority_dir.join("committed-ledger");
        let lock = authority_dir.join("ledger.lock");
        let key_set = hyprstream_rpc::auth::global_composite_key_set();
        let version = key_set.snapshot().version() + 1;
        let digest = format!("oauth-handler-r5-{version}");
        let generation = serde_json::to_vec(&serde_json::json!({
            "version": version,
            "component_digest": digest,
        }))?;
        std::fs::write(&ledger, &generation)?;
        std::fs::write(&committed, &generation)?;
        std::fs::write(
            authority_dir.join(format!("committed-ledger-{version}-{digest}.json")),
            &generation,
        )?;

        let (ml_signing, ml_verifying) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ed_signing = Arc::new(ed25519_dalek::SigningKey::from_bytes(&[0x61; 32]));
        let kid = hyprstream_rpc::auth::composite_kid(&ml_verifying, &ed_signing.verifying_key());
        let now = chrono::Utc::now().timestamp();
        let pair = CompositeKeyPair::signing(
            kid,
            Arc::new(ml_signing),
            ed_signing,
            CompositePairRole::Policy,
            CompositePairState::Active,
            now - 60,
            now + 86_400,
        );
        key_set.configure_authority(ledger, committed, prefix, lock);
        key_set.publish(version, digest, vec![pair])?;
        Ok(())
    }

    /// #1113 r5 handler-level conformance suite. Every HTTP exchange goes
    /// through the production create_app router and token issuance traverses
    /// a real in-process PolicyService.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oauth_handler_atproto_and_legacy_conformance() -> anyhow::Result<()> {
        use base64::{
            engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
            Engine as _,
        };
        use ed25519_dalek::Signer as _;
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
        use hyprstream_rpc::transport::TransportConfig;
        use hyprstream_service::{InprocManager, ServiceManager as _};
        use sha2::{Digest as _, Sha256};

        use super::state::RegisteredClient;
        use super::token_store::RocksDbTokenStore;
        use crate::auth::rocksdb_store::RocksDbUserStore;
        use crate::auth::{PolicyManager, UserProfile, UserStore};
        use crate::services::{DiscoveryClient, PolicyClient, PolicyService};

        const ISSUER: &str = "https://pds.example.test";
        const CLIENT_ID: &str = "handler-client";
        const REDIRECT_URI: &str = "https://client.example.test/callback";
        const MAPPED_DID: &str = "did:plc:abcdefghijklmnqrstuvwx2p";
        const PKCE_VERIFIER: &str = "r5-handler-pkce-verifier-abcdefghijklmnopqrstuvwxyz012345";

        let _ = hyprstream_rpc::envelope::install_verify_config(
            hyprstream_rpc::envelope::EnvelopeVerifyConfig {
                policy: CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        let _ = hyprstream_rpc::envelope::install_response_verify_config(
            hyprstream_rpc::envelope::ResponseVerifyConfig {
                policy: CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        configure_test_policy_signing_authority()?;

        let service_key = ed25519_dalek::SigningKey::from_bytes(&[0x62; 32]);
        let policy_tag = format!("oauth-handler-r5-{}", uuid::Uuid::new_v4());
        let git_dir = tempfile::TempDir::new()?;
        let git2db = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(git_dir.path()).await?,
        ));
        let policy_service = PolicyService::new(
            Arc::new(PolicyManager::permissive().await?),
            Arc::new(service_key.clone()),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc(&policy_tag),
        )
        .with_default_audience(ISSUER.to_owned());
        let manager = InprocManager::new();
        let mut policy_handle = manager.spawn(Box::new(policy_service)).await?;
        let policy_client = PolicyClient::for_local_endpoint_bootstrap(
            &format!("inproc://{policy_tag}"),
            service_key.clone(),
            service_key.verifying_key(),
            None,
        )?;

        let user_dir = tempfile::TempDir::new()?;
        let user_store = Arc::new(RocksDbUserStore::open(user_dir.path())?);
        let user_key = ed25519_dalek::SigningKey::from_bytes(&[0x63; 32]);
        user_store.register("alice").await?;
        let fingerprint = user_store
            .add_pubkey(
                "alice",
                user_key.verifying_key(),
                Some("handler-test".to_owned()),
            )
            .await?;
        user_store
            .set_profile(
                "alice",
                UserProfile {
                    atproto_did: Some(MAPPED_DID.to_owned()),
                    ..Default::default()
                },
            )
            .await?;
        assert_eq!(
            user_store
                .get_profile("alice")
                .await?
                .and_then(|profile| profile.atproto_did),
            Some(MAPPED_DID.to_owned())
        );

        let dummy_key = ed25519_dalek::SigningKey::from_bytes(&[0x64; 32]);
        let dummy_remote = ed25519_dalek::SigningKey::from_bytes(&[0x65; 32]).verifying_key();
        let dummy_rpc = Arc::new(
            RpcClientImpl::new(
                LocalSigner::new(dummy_key),
                LazyUdsTransport::new("/dev/null/oauth-handler-discovery.sock".into()),
                Some(dummy_remote),
            )
            .with_response_verify_policy(CryptoPolicy::Classical),
        );
        let mut config = crate::config::OAuthConfig::default();
        // A configured path and trailing slash must normalize everywhere.
        config.external_url = Some(format!("{ISSUER}/configured/path/"));
        let mut oauth_state = OAuthState::new(
            &config,
            policy_client,
            DiscoveryClient::new(dummy_rpc),
            service_key.verifying_key().to_bytes(),
        )
        .with_user_store(user_store);
        let token_dir = tempfile::TempDir::new()?;
        oauth_state.with_token_store_impl(Arc::new(RocksDbTokenStore::open(
            token_dir.path().join("refresh.db"),
        )?));
        let state = Arc::new(oauth_state);
        state.clients.write().await.insert(
            CLIENT_ID.to_owned(),
            RegisteredClient {
                client_id: CLIENT_ID.to_owned(),
                redirect_uris: vec![REDIRECT_URI.to_owned()],
                client_name: Some("Handler Conformance Client".to_owned()),
                client_uri: None,
                logo_uri: None,
                grant_types: vec![
                    "authorization_code".to_owned(),
                    "refresh_token".to_owned(),
                    "urn:ietf:params:oauth:grant-type:device_code".to_owned(),
                ],
                response_types: vec!["code".to_owned()],
                token_endpoint_auth_method: Some("none".to_owned()),
                jwks: None,
                jwks_uri: None,
                hyprstream_node_did: None,
                scope: Some("atproto read:*:*".to_owned()),
                is_cimd: false,
                registered_at: std::time::Instant::now(),
            },
        );
        let cors = crate::config::server::CorsConfig {
            enabled: false,
            ..Default::default()
        };
        let app = create_app(Arc::clone(&state), &cors);

        // Device authorization rejects unknown clients at the real endpoint.
        let unknown_device = post_form(
            &app,
            "/oauth/device",
            &[("client_id", "unknown-client"), ("scope", "read:*:*")],
            None,
            false,
        )
        .await;
        assert_eq!(unknown_device.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(unknown_device).await["error"],
            "invalid_client"
        );

        // PAR binds the atproto request to a real ES256 DPoP key.
        let dpop_key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(PKCE_VERIFIER.as_bytes()));
        let par_proof = dpop_proof(
            &dpop_key,
            &format!("{ISSUER}/oauth/par"),
            "handler-par-jti",
            None,
        );
        let par = post_form(
            &app,
            "/oauth/par",
            &[
                ("client_id", CLIENT_ID),
                ("redirect_uri", REDIRECT_URI),
                ("code_challenge", &code_challenge),
                ("code_challenge_method", "S256"),
                ("response_type", "code"),
                ("state", "client-state"),
                ("scope", "atproto"),
                ("resource", ISSUER),
            ],
            Some(&par_proof),
            false,
        )
        .await;
        assert_eq!(par.status(), axum::http::StatusCode::CREATED);
        let par_nonce = par
            .headers()
            .get("DPoP-Nonce")
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_owned();
        let par_json = response_json(par).await;
        let request_uri = par_json["request_uri"].as_str().unwrap();

        // GET authorize consumes the PAR and renders the real consent callback.
        let authorize_uri = format!(
            "/oauth/authorize?{}",
            encode_form(&[("request_uri", request_uri), ("client_id", CLIENT_ID)])
        );
        let authorize = get(&app, &authorize_uri).await;
        assert_eq!(authorize.status(), axum::http::StatusCode::OK);
        let authorize_html = response_text(authorize).await;
        let authorize_nonce = html_hidden_value(&authorize_html, "nonce").to_owned();
        let challenge = format!("{fingerprint}:{authorize_nonce}:{code_challenge}");
        let signature = STANDARD.encode(user_key.sign(challenge.as_bytes()).to_bytes());
        let callback = post_form(
            &app,
            "/oauth/authorize",
            &[
                ("client_id", CLIENT_ID),
                ("redirect_uri", REDIRECT_URI),
                ("code_challenge", &code_challenge),
                ("scope", "atproto"),
                ("state", "client-state"),
                ("resource", ISSUER),
                ("nonce", &authorize_nonce),
                ("fingerprint", &fingerprint),
                ("signature", &signature),
            ],
            None,
            false,
        )
        .await;
        assert_eq!(callback.status(), axum::http::StatusCode::SEE_OTHER);
        let location = callback
            .headers()
            .get(axum::http::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .unwrap();
        let location = url::Url::parse(location)?;
        let callback_params: std::collections::HashMap<_, _> =
            location.query_pairs().into_owned().collect();
        assert_eq!(callback_params.get("iss").map(String::as_str), Some(ISSUER));
        assert_eq!(
            callback_params.get("state").map(String::as_str),
            Some("client-state")
        );
        let code = callback_params.get("code").unwrap();

        // Exchange the code and parse the emitted response against the exact
        // @atproto/oauth-types 0.7.5 token-response constraints.
        let token_proof = dpop_proof(
            &dpop_key,
            &format!("{ISSUER}/oauth/token"),
            "handler-token-jti",
            Some(&par_nonce),
        );
        let token = post_form(
            &app,
            "/oauth/token",
            &[
                ("grant_type", "authorization_code"),
                ("client_id", CLIENT_ID),
                ("code", code),
                ("redirect_uri", REDIRECT_URI),
                ("code_verifier", PKCE_VERIFIER),
            ],
            Some(&token_proof),
            false,
        )
        .await;
        assert_eq!(token.status(), axum::http::StatusCode::OK);
        let token_nonce = token
            .headers()
            .get("DPoP-Nonce")
            .and_then(|value| value.to_str().ok())
            .unwrap()
            .to_owned();
        let token_response = parse_atproto_token_response_075(response_json(token).await)?;
        assert_eq!(token_response.sub, MAPPED_DID);
        assert!(token_response.expires_in.is_some());
        let refresh_token = token_response.refresh_token.unwrap();
        let claims = jwt_claims(&token_response.access_token);
        assert_eq!(claims["iss"], ISSUER);
        assert_eq!(claims["sub"], MAPPED_DID);
        assert_eq!(claims["aud"], ISSUER);

        // A missing refresh proof is rejected before consumption. Retrying
        // the same refresh token with the bound key and nonce must succeed.
        let stranded_attempt = post_form(
            &app,
            "/oauth/token",
            &[
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", &refresh_token),
            ],
            None,
            false,
        )
        .await;
        assert_eq!(
            stranded_attempt.status(),
            axum::http::StatusCode::BAD_REQUEST
        );
        assert_eq!(
            response_json(stranded_attempt).await["error"],
            "invalid_dpop_proof"
        );
        assert!(state.get_refresh_token(&refresh_token).await?.is_some());
        let refresh_proof = dpop_proof(
            &dpop_key,
            &format!("{ISSUER}/oauth/token"),
            "handler-refresh-jti",
            Some(&token_nonce),
        );
        let refreshed = post_form(
            &app,
            "/oauth/token",
            &[
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", &refresh_token),
            ],
            Some(&refresh_proof),
            false,
        )
        .await;
        assert_eq!(refreshed.status(), axum::http::StatusCode::OK);
        let refreshed_json = response_json(refreshed).await;
        assert_eq!(refreshed_json["token_type"], "DPoP");
        assert_eq!(refreshed_json["sub"], MAPPED_DID);
        assert_ne!(refreshed_json["refresh_token"], refresh_token);
        assert!(state.get_refresh_token(&refresh_token).await?.is_none());
        let refreshed_claims = jwt_claims(refreshed_json["access_token"].as_str().unwrap());
        assert_eq!(refreshed_claims["sub"], MAPPED_DID);

        // A real generic device flow retains the local username byte-for-byte
        // and the legacy Bearer response shape.
        let device = post_form(
            &app,
            "/oauth/device",
            &[
                ("client_id", CLIENT_ID),
                ("scope", "read:*:*"),
                ("resource", ISSUER),
            ],
            None,
            false,
        )
        .await;
        assert_eq!(device.status(), axum::http::StatusCode::OK);
        let device_json = response_json(device).await;
        let device_code = device_json["device_code"].as_str().unwrap();
        let user_code = device_json["user_code"].as_str().unwrap();
        let nonce_uri = format!(
            "/oauth/device/nonce?{}",
            encode_form(&[("user_code", user_code)])
        );
        let device_nonce = response_json(get(&app, &nonce_uri).await).await;
        let normalized_code = user_code.replace('-', "");
        let device_challenge = format!(
            "alice:{normalized_code}:{}",
            device_nonce["nonce"].as_str().unwrap()
        );
        let device_signature =
            STANDARD.encode(user_key.sign(device_challenge.as_bytes()).to_bytes());
        let approved = post_form(
            &app,
            "/oauth/device/verify",
            &[
                ("user_code", user_code),
                ("username", "alice"),
                ("signature", &device_signature),
            ],
            None,
            true,
        )
        .await;
        assert_eq!(approved.status(), axum::http::StatusCode::OK);
        let device_token = post_form(
            &app,
            "/oauth/token",
            &[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", CLIENT_ID),
                ("device_code", device_code),
            ],
            None,
            false,
        )
        .await;
        assert_eq!(device_token.status(), axum::http::StatusCode::OK);
        let device_token_json = response_json(device_token).await;
        assert_eq!(device_token_json["token_type"], "Bearer");
        assert!(device_token_json.get("sub").is_none());
        let device_claims = jwt_claims(device_token_json["access_token"].as_str().unwrap());
        assert_eq!(device_claims["sub"], "alice");
        assert_eq!(device_claims["iss"], ISSUER);
        assert_eq!(device_claims["aud"], ISSUER);

        policy_handle.stop().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oauth_iroh_refuses_arbitrary_rpc_and_anonymous_moq_before_dispatch(
    ) -> anyhow::Result<()> {
        use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
        use hyprstream_rpc::transport::iroh_substrate::{
            IrohSubstrate, NoopHandler, ALPN_HYPRSTREAM_RPC, ALPN_MOQ_LITE,
        };
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
            let response =
                tokio::time::timeout(std::time::Duration::from_secs(2), recv.read_to_end(1024))
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

    /// #1113: the PDS is its own AS, so the self protected-resource document
    /// must advertise the atproto transition scopes AND point
    /// `authorization_servers` at itself (resource == issuer).
    #[test]
    fn test_self_protected_resource_advertises_atproto_profile() {
        // PDS = its own AS: resource and authorization_servers[0] are the same.
        let issuer = "https://pds.example.com";
        let meta = protected_resource_metadata(issuer, issuer);
        assert_eq!(meta.resource, issuer);
        assert_eq!(meta.authorization_servers, vec![issuer.to_owned()]);

        // atproto transition scopes are advertised.
        let scopes = self_resource_scopes();
        assert!(
            scopes.contains(&"atproto".to_owned()),
            "missing atproto: {scopes:?}"
        );
        assert!(
            scopes.contains(&"transition:generic".to_owned()),
            "missing transition:generic: {scopes:?}"
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
