//! Service factory functions for inventory-based registration.
//!
//! This module contains all `#[service_factory]` decorated functions that
//! automatically register services with the inventory system.
//!
//! # Pattern
//!
//! Same pattern as:
//! - `#[register_scopes]` for authorization scopes
//! - `DriverFactory` in git2db for storage drivers
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::service::{get_factory, ServiceContext};
//!
//! let ctx = ServiceContext::new(...);
//! let factory = get_factory("policy").unwrap();
//! let spawnable = (factory.factory)(&ctx)?;
//! manager.spawn(spawnable).await?;
//! ```

use std::sync::Arc;

use anyhow::Context;
use git2db::Git2DB;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as global_registry, SocketKind};
use hyprstream_service::{ProxyService, ServiceContext, Spawnable};
use hyprstream_rpc::service_factory;
use hyprstream_workers::endpoints;
use tokio::sync::RwLock;
use tracing::info;

use crate::auth::PolicyManager;
use crate::config::{HyprConfig, TokenConfig};
use crate::services::{DiscoveryService, McpService, McpConfig, PolicyService, PolicyClient, RegistryService, RegistryClient};
use crate::zmq::global_context;

/// Load HyprConfig, falling back to default on error.
fn load_config() -> HyprConfig {
    HyprConfig::load().unwrap_or_default()
}

/// Shared Git2DB registry instance. Lazily initialized by the first factory
/// that needs it. Both PolicyService and RegistryService share this instance.
static SHARED_GIT2DB: std::sync::OnceLock<Arc<RwLock<Git2DB>>> = std::sync::OnceLock::new();

/// Get or initialize the shared Git2DB registry for the given models directory.
fn get_or_init_git2db(models_dir: &std::path::Path) -> anyhow::Result<Arc<RwLock<Git2DB>>> {
    if let Some(existing) = SHARED_GIT2DB.get() {
        return Ok(Arc::clone(existing));
    }

    let registry = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(Git2DB::open(models_dir))
    }).context("Failed to initialize shared Git2DB registry")?;

    let shared = Arc::new(RwLock::new(registry));
    // If another thread beat us, that's fine — use theirs
    Ok(Arc::clone(SHARED_GIT2DB.get_or_init(|| shared)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for EventService (XPUB/XSUB proxy for event distribution)
#[service_factory("event")]
fn create_event_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating EventService");

    let mode = if ctx.is_ipc() {
        endpoints::EndpointMode::Ipc
    } else {
        endpoints::EndpointMode::Inproc
    };

    let (pub_transport, sub_transport) = endpoints::detect_transports(mode);
    let proxy = ProxyService::new("events", global_context(), pub_transport, sub_transport);

    Ok(Box::new(proxy))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Policy Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for PolicyService (Casbin policy management)
#[service_factory("policy", schema = "../../schema/policy.capnp", metadata = crate::services::generated::policy_client::schema_metadata)]
fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating PolicyService");

    let policies_dir = ctx.models_dir().join(".registry").join("policies");

    // Get shared Git2DB instance (initializes .registry as git repo if needed)
    let git2db = get_or_init_git2db(ctx.models_dir())?;

    // Create policy manager (blocking since we're in sync context)
    let policy_manager = Arc::new(
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let pm = PolicyManager::new(&policies_dir).await?;
                // Idempotent migration: ensure required bootstrap rules are present.
                // These rules are in DEFAULT_POLICY_CSV for new installs; existing
                // deployments need them added once.
                let rules = pm.get_policy().await;
                let has_system = rules.iter().any(|r| r.first().map(|s| s == "system").unwrap_or(false));
                let has_anon_tui = rules.iter().any(|r| {
                    r.len() >= 3 && r[0] == "anonymous" && r[2] == "tui:*"
                });
                if !has_system {
                    let _ = pm.add_policy_with_domain("system", "*", "*", "*", "allow").await;
                    tracing::info!("policy migration: added 'system' full-access grant");
                }
                if !has_anon_tui {
                    let _ = pm.add_policy_with_domain("anonymous", "*", "tui:*", "*", "allow").await;
                    tracing::info!("policy migration: added 'anonymous' TUI access grant");
                }
                if !has_system || !has_anon_tui {
                    let _ = pm.save().await;
                }
                Ok::<_, anyhow::Error>(pm)
            })
        })
        .context("Failed to initialize policy manager")?,
    );

    // Spawn file watcher for policy hot-reload
    let pm_clone = Arc::clone(&policy_manager);
    let policy_csv = policies_dir.join("policy.csv");
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            super::policy::watch_policy_file(pm_clone, policy_csv).await;
        });
    });

    let config = load_config();
    let mut policy_service = PolicyService::new(
        policy_manager,
        Arc::new(ctx.signing_key().clone()),
        TokenConfig::default(),
        git2db,
        global_context(),
        ctx.transport("policy", SocketKind::Rep),
    );
    if let Some(issuer) = ctx.oauth_issuer_url() {
        policy_service = policy_service.with_default_audience(issuer.to_owned());
        policy_service = policy_service.with_local_issuer_url(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        policy_service = policy_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(policy_service, config.policy.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for RegistryService (git2db model registry)
#[service_factory("registry", schema = "../../schema/registry.capnp", metadata = crate::services::generated::registry_client::schema_metadata)]
fn create_registry_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating RegistryService");

    let config = load_config();

    // Create policy client for authorization checks
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    // Create registry service with infrastructure (blocking since we're in sync context)
    let mut registry_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(RegistryService::new(
            ctx.models_dir(),
            policy_client,
            global_context(),
            ctx.transport("registry", SocketKind::Rep),
            ctx.signing_key().clone(),
        ))
    })?;
    if let Some(issuer) = ctx.oauth_issuer_url() {
        registry_service = registry_service.with_expected_audience(issuer.to_owned());
        registry_service = registry_service.with_local_issuer_url(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        registry_service = registry_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(registry_service, config.registry.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streams Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for StreamService (PULL/XPUB queuing proxy with JWT validation)
#[service_factory("streams")]
fn create_streams_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating StreamService with JWT validation and queuing");

    use crate::services::StreamService;

    // XPUB frontend - clients subscribe via SUB
    let pub_transport = global_registry().endpoint("streams", SocketKind::Sub);
    // PULL backend - publishers connect via PUSH
    let pull_transport = global_registry().endpoint("streams", SocketKind::Push);

    let stream_service = StreamService::new(
        "streams",
        global_context(),
        pub_transport,
        pull_transport,
        ctx.verifying_key(),
    );

    Ok(Box::new(stream_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Model Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for ModelService (model lifecycle management)
#[service_factory("model", schema = "../../schema/model.capnp", metadata = crate::services::generated::model_client::schema_metadata)]
fn create_model_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating ModelService");

    use crate::services::{ModelService, ModelServiceConfig};

    let config = load_config();

    // Create policy client for authorization checks
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    // Create registry client
    let registry_client: RegistryClient = RegistryClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::anonymous(),
    );

    let mut model_service = ModelService::new(
        ModelServiceConfig::default(),
        ctx.signing_key().clone(),
        policy_client,
        registry_client,
        global_context(),
        ctx.transport("model", SocketKind::Rep),
    );
    if let Some(issuer) = ctx.oauth_issuer_url() {
        model_service = model_service.with_expected_audience(issuer.to_owned());
        model_service = model_service.with_local_issuer_url(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        model_service = model_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(model_service, config.model.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Worker Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for WorkerService (Kata container/sandbox management)
///
/// Note: This service requires worker configuration. If not configured,
/// the factory will use sensible defaults.
#[service_factory("worker")]
fn create_worker_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating WorkerService");

    use hyprstream_workers::config::{BackendType, ImageConfig, PoolConfig};
    use hyprstream_workers::image::RafsStore;
    use hyprstream_workers::{WorkerService, SandboxBackend, KataBackend, NspawnBackend, NspawnConfig};

    let config = load_config();
    let worker_quic_port = config.worker.as_ref().and_then(|w| w.quic_port);
    let backend_type = config.worker.as_ref()
        .map(|w| w.backend)
        .unwrap_or_default();

    info!("WorkerService using {} backend", backend_type);

    // Use default paths based on XDG directories
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("hyprstream");
    let runtime_dir = dirs::runtime_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("hyprstream");

    let kata_boot_path = std::env::var("KATA_BOOT_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/opt/kata/share/kata-containers"));

    let pool_config = PoolConfig {
        warm_pool_size: 0,
        runtime_dir: runtime_dir.join("sandboxes"),
        kernel_path: kata_boot_path.join("vmlinux.container"),
        vm_image: kata_boot_path.join("kata-containers.img"),
        cloud_init_dir: data_dir.join("cloud-init"),
        ..PoolConfig::default()
    };

    let image_config = ImageConfig {
        blobs_dir: data_dir.join("images/blobs"),
        bootstrap_dir: data_dir.join("images/bootstrap"),
        refs_dir: data_dir.join("images/refs"),
        cache_dir: data_dir.join("images/cache"),
        runtime_dir: runtime_dir.join("nydus"),
        ..ImageConfig::default()
    };

    let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);

    // Construct the sandbox backend based on configuration
    let backend: Arc<dyn SandboxBackend> = match backend_type {
        BackendType::Kata => Arc::new(KataBackend::new(image_config, Arc::clone(&rafs_store))),
        BackendType::Nspawn => Arc::new(NspawnBackend::new(NspawnConfig::default())),
    };

    // Service includes infrastructure - directly Spawnable via blanket impl
    let mut worker_service = WorkerService::new(
        pool_config,
        backend,
        rafs_store,
        global_context(),
        ctx.transport("worker", SocketKind::Rep),
        ctx.signing_key().clone(),
    )?;

    // Wire up policy-backed authorization
    let policy_client = crate::services::PolicyClient::new(
        ctx.signing_key().clone(),
        hyprstream_rpc::RequestIdentity::anonymous(),
    );
    worker_service.set_authorize_fn(super::worker::build_authorize_fn(policy_client));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        worker_service.set_expected_audience(issuer.to_owned());
        worker_service.set_local_issuer_url(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        worker_service.set_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(worker_service, worker_quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// OAI Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for OAIService (OpenAI-compatible HTTP API)
///
/// This service provides the HTTP API for inference requests.
/// It communicates with ModelService and PolicyService via ZMQ.
#[service_factory("oai")]
fn create_oai_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAIService");

    use crate::server::state::ServerState;
    use crate::services::generated::model_client::ModelClient;
    use crate::services::OAIService;

    // Load full config for OAI settings
    let config = load_config();

    // Create ZMQ clients for Model and Policy services
    let model_client = ModelClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    // Create registry client
    let registry_client: RegistryClient = RegistryClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::anonymous(),
    );

    // Create server state (blocking since we're in sync context)
    let resource_url = config.oai.resource_url();
    let oauth_issuer_url = config.oauth.issuer_url();
    let server_state = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(ServerState::new(
            config.server.clone(),
            model_client,
            policy_client,
            registry_client,
            ctx.signing_key().clone(),
            resource_url,
            oauth_issuer_url,
            &config.oauth.trusted_issuers,
        ))
    })
    .context("Failed to create server state")?;

    let oai_service = OAIService::new(
        config.oai.clone(),
        config.tls.clone(),
        server_state,
        global_context(),
        ctx.transport("oai", SocketKind::Rep),
        ctx.verifying_key(),
    );

    Ok(Box::new(oai_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flight Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for FlightService (Arrow Flight SQL server)
///
/// This service provides Flight SQL protocol for dataset queries.
/// It optionally uses RegistryClient for dataset lookup.
#[service_factory("flight")]
fn create_flight_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating FlightService");

    use crate::services::FlightService;

    // Load full config for Flight settings
    let config = load_config();

    // Create registry client for dataset lookup (if default_dataset is configured)
    // RegistryClient already implements hyprstream_metrics::RegistryClient
    let registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>> =
        if config.flight.default_dataset.is_some() {
            let zmq_client: RegistryClient = RegistryClient::new(
                ctx.signing_key().clone(),
                RequestIdentity::anonymous(),
            );
            Some(Arc::new(zmq_client))
        } else {
            None
        };

    let flight_service = FlightService::new(
        config.flight.clone(),
        registry_client,
        global_context(),
        ctx.transport("flight", SocketKind::Rep),
        ctx.verifying_key(),
    );

    Ok(Box::new(flight_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// OAuth Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for OAuthService (OAuth 2.1 Authorization Server)
///
/// This service provides OAuth 2.1 authorization for MCP and OAI services.
/// It delegates token issuance to PolicyService over ZMQ.
#[service_factory("oauth")]
fn create_oauth_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAuthService");

    use crate::services::OAuthService;

    let config = load_config();

    // Pass signing key instead of a pre-created PolicyClient.
    // OAuthService runs in its own tokio runtime (separate thread), so the
    // PolicyClient must be created inside that runtime for ZMQ async I/O to work.
    let oauth_service = OAuthService::new(
        config.oauth.clone(),
        config.tls.clone(),
        ctx.signing_key().clone(),
        global_context(),
        ctx.transport("oauth", SocketKind::Rep),
        ctx.verifying_key(),
    );

    Ok(Box::new(oauth_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// MCP Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for McpService (Model Context Protocol)
///
/// This service provides an MCP-compliant interface for AI coding assistants
/// (Claude Code, Cursor, etc.) to interact with hyprstream via:
/// - ZMQ control plane (for internal service communication)
/// - HTTP/SSE (for external MCP clients)
///
/// Note: The HTTP/SSE server is spawned as a background task in the factory.
#[service_factory("mcp", schema = "../../schema/mcp.capnp", metadata = crate::services::generated::mcp_client::schema_metadata)]
fn create_mcp_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating McpService");

    // Load full config for MCP settings
    let config = load_config();

    // Create McpConfig for the service
    let oauth_issuer = ctx.oauth_issuer_url().map(str::to_owned);
    let federation_key_source = ctx.federation_key_source();
    let mcp_config = McpConfig {
        verifying_key: ctx.verifying_key(),
        zmq_context: global_context(),
        signing_key: ctx.signing_key().clone(),
        transport: ctx.transport("mcp", SocketKind::Rep),
        ctx: None, // ServiceContext not yet available as Arc — handlers use signing_key directly
        expected_audience: Some(config.mcp.resource_url()),
        local_issuer_url: oauth_issuer.clone(),
        federation_key_source: federation_key_source.clone(),
    };

    // Clone config for HTTP/SSE server before consuming it for ZMQ service
    let mcp_config_clone = mcp_config.clone();

    // Create the service (includes ZMQ infrastructure)
    let mcp_service = McpService::new(mcp_config)?;

    // Spawn rmcp HTTP/SSE server as background task
    let mcp_host = config.mcp.host.clone();
    let http_port = config.mcp.http_port;
    let mcp_cors_config = config.mcp.cors.clone();
    let mcp_tls_config = config.tls.clone();
    let mcp_tls_cert = config.mcp.tls_cert.clone();
    let mcp_tls_key = config.mcp.tls_key.clone();
    // Use the shared FederationKeySource from ServiceContext if available,
    // otherwise fall back to a locally-constructed resolver from config.
    let mcp_federation_resolver: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource> =
        if let Some(fed) = federation_key_source {
            fed
        } else {
            std::sync::Arc::new(
                crate::auth::FederationKeyResolver::new(&config.oauth.trusted_issuers)
            )
        };
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            use rmcp::transport::streamable_http_server::{
                StreamableHttpServerConfig, StreamableHttpService,
            };

            use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

            let session_mgr = std::sync::Arc::new(LocalSessionManager::default());
            let verifying_key = mcp_config_clone.verifying_key;
            let service: StreamableHttpService<McpService, LocalSessionManager> =
                StreamableHttpService::new(
                    move || McpService::new(mcp_config_clone.clone()).map_err(|e| {
                        std::io::Error::other(e.to_string())
                    }),
                    session_mgr,
                    StreamableHttpServerConfig::default(),
                );
            // Add protected resource metadata (RFC 9728) for OAuth discovery
            let mcp_full_config = crate::config::HyprConfig::load().unwrap_or_default();
            let mcp_resource_url = mcp_full_config.mcp.resource_url();
            let mcp_oauth_issuer = mcp_full_config.oauth.issuer_url();
            let www_authenticate = format!(
                "Bearer resource_metadata=\"{}/.well-known/oauth-protected-resource\"",
                mcp_resource_url
            );
            let router = axum::Router::new()
                .route(
                    "/.well-known/oauth-protected-resource",
                    axum::routing::get({
                        let mcp_resource_url = mcp_resource_url.clone();
                        let mcp_oauth_issuer = mcp_oauth_issuer.clone();
                        move || async move {
                            let mut meta = crate::services::oauth::protected_resource_metadata(
                                &mcp_resource_url,
                                &mcp_oauth_issuer,
                            );
                            meta.resource_name = Some("HyprStream MCP Server".to_owned());
                            meta.scopes_supported = Some(vec![
                                "read:model:*".into(),
                                "infer:model:*".into(),
                                "write:model:*".into(),
                            ]);
                            axum::Json(meta)
                        }
                    }),
                )
                .nest_service("/mcp", service)
                .layer(axum::middleware::from_fn({
                    let mcp_resource_url = mcp_resource_url.clone();
                    let mcp_oauth_issuer_clone = mcp_oauth_issuer.clone();
                    let mcp_federation_resolver = mcp_federation_resolver.clone();
                    move |req: axum::extract::Request, next: axum::middleware::Next| {
                        let www_authenticate = www_authenticate.clone();
                        let mcp_resource_url = mcp_resource_url.clone();
                        let mcp_oauth_issuer = mcp_oauth_issuer_clone.clone();
                        let federation_resolver = mcp_federation_resolver.clone();
                        async move {
                            use axum::http::{header, StatusCode};
                            use axum::response::IntoResponse;
                            let method = req.method().clone();
                            let uri = req.uri().clone();
                            // Allow OAuth discovery endpoint without auth
                            if req.uri().path().starts_with("/.well-known/") {
                                tracing::debug!(%method, %uri, "MCP discovery request (no auth required)");
                                return next.run(req).await;
                            }
                            let has_auth_header = req.headers().contains_key(header::AUTHORIZATION);
                            let auth_value = req.headers()
                                .get(header::AUTHORIZATION)
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_owned);
                            // RFC 6750: Bearer scheme is case-insensitive
                            let token = auth_value.as_deref()
                                .and_then(|h| {
                                    if h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer ") {
                                        Some(h[7..].trim())
                                    } else {
                                        None
                                    }
                                });
                            match token {
                                Some(t) => {
                                    let iss = crate::server::middleware::extract_iss_from_token(t);
                                    let local_issuers: &[&str] = &[mcp_oauth_issuer.as_str()];
                                    let result = if hyprstream_rpc::auth::is_local_iss(&iss, local_issuers) {
                                        crate::auth::jwt::decode(t, &verifying_key, Some(mcp_resource_url.as_str()))
                                    } else {
                                        match federation_resolver.get_key(&iss).await {
                                            Ok(key) => crate::auth::jwt::decode_with_key(t, &key, Some(mcp_resource_url.as_str())),
                                            Err(e) => {
                                                tracing::debug!(%method, %uri, issuer = %iss, error = %e, "MCP federation key resolution failed");
                                                let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                                if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                                    res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                                }
                                                return res;
                                            }
                                        }
                                    };
                                    match result {
                                        Ok(claims) => {
                                            tracing::debug!(%method, %uri, sub = %claims.sub, "MCP auth OK");
                                            next.run(req).await
                                        }
                                        Err(e) => {
                                            tracing::warn!(%method, %uri, error = %e, "MCP auth REJECTED");
                                            let mut res = (StatusCode::UNAUTHORIZED, "Invalid or expired token").into_response();
                                            if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                                res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                            }
                                            res
                                        }
                                    }
                                },
                                None => {
                                    tracing::info!(%method, %uri, has_auth_header, "MCP auth MISSING token");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Authentication required").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    res
                                }
                            }
                        }
                    }
                }));

            // CORS must be outermost layer (added last) so OPTIONS preflights
            // are handled before auth middleware rejects them.
            let router = if mcp_cors_config.enabled {
                router.layer(crate::server::middleware::cors_layer(&mcp_cors_config))
            } else {
                router
            };

            let addr: std::net::SocketAddr = format!("{}:{}", mcp_host, http_port)
                .parse()
                .unwrap_or_else(|_| ([0, 0, 0, 0], http_port).into());

            // Resolve TLS configuration for MCP HTTP server.
            // If the user explicitly configured cert/key paths and TLS fails,
            // refuse to start (don't silently degrade to HTTP).
            let has_explicit_tls = mcp_tls_cert.is_some() || mcp_tls_key.is_some()
                || mcp_tls_config.cert_path.is_some() || mcp_tls_config.key_path.is_some();

            let rustls_config = match crate::server::tls::resolve_rustls_config(
                &mcp_tls_config,
                mcp_tls_cert.as_ref(),
                mcp_tls_key.as_ref(),
            ).await {
                Ok(cfg) => cfg,
                Err(e) => {
                    if has_explicit_tls {
                        tracing::error!(
                            "MCP TLS config error with explicit cert/key paths: {} — refusing to start without TLS", e
                        );
                        return;
                    }
                    tracing::warn!("MCP TLS config error (self-signed): {} — falling back to HTTP", e);
                    None
                }
            };

            let scheme = if rustls_config.is_some() { "https" } else { "http" };
            tracing::info!("MCP HTTP/SSE server listening on {scheme}://{addr}");

            match rustls_config {
                Some(tls) => {
                    // MCP HTTP is fire-and-forget (no Arc<Notify> shutdown signal),
                    // so no Handle is wired for graceful shutdown. The process exit
                    // will terminate this task. OAI/OAuth use serve_app() instead.
                    if let Err(e) = axum_server::bind_rustls(addr, tls)
                        .serve(router.into_make_service())
                        .await
                    {
                        tracing::error!("MCP HTTPS server error: {}", e);
                    }
                }
                None => {
                    let listener = match tokio::net::TcpListener::bind(addr).await {
                        Ok(l) => l,
                        Err(e) => {
                            tracing::error!("Failed to bind MCP HTTP/SSE on {}: {}", addr, e);
                            return;
                        }
                    };
                    if let Err(e) = axum::serve(listener, router).await {
                        tracing::error!("MCP HTTP/SSE server error: {}", e);
                    }
                }
            }
        });
    });
    info!("McpService created (HTTP/SSE on {}:{})", config.mcp.host, http_port);

    Ok(ctx.into_spawnable_quic(mcp_service, config.mcp.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUI Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for TuiService (terminal multiplexer display server)
///
/// This service provides a terminal multiplexer with session persistence,
/// multi-pane layouts, and remote access via ZMQ RPC and WebTransport.
#[service_factory("tui", schema = "../../schema/tui.capnp")]
fn create_tui_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating TuiService");

    use crate::tui::{TuiState, service::TuiService};

    let config = load_config();
    let tui_config = &config.tui;

    let state = Arc::new(RwLock::new(TuiState::new(
        80,
        24,
        tui_config.scrollback_lines,
    )));

    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    let mut tui_service = TuiService::new(
        state,
        global_context(),
        ctx.transport("tui", SocketKind::Rep),
        ctx.signing_key().clone(),
    ).with_policy_client(policy_client);

    if let Some(issuer) = ctx.oauth_issuer_url() {
        tui_service = tui_service.with_local_issuer_url(issuer.to_owned());
        tui_service = tui_service.with_expected_audience(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        tui_service = tui_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(tui_service, tui_config.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Discovery Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for DiscoveryService (endpoint registry over ZMQ RPC)
///
/// This service exposes the EndpointRegistry so remote clients can discover
/// registered services, their endpoints, socket kinds, and schemas.
#[service_factory("discovery", schema = "../../../hyprstream-discovery/schema/discovery.capnp", metadata = hyprstream_discovery::generated::discovery_client::schema_metadata)]
fn create_discovery_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating DiscoveryService");

    let config = load_config();

    // Create policy-based authorization provider
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());
    let auth_provider = crate::services::discovery::PolicyAuthProvider::new(policy_client);

    let mut discovery_service = DiscoveryService::new(
        Arc::new(ctx.signing_key().clone()),
        global_context(),
        ctx.transport("discovery", SocketKind::Rep),
    ).with_auth_provider(Box::new(auth_provider));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        discovery_service = discovery_service.with_oauth_issuer(issuer.to_owned());
        // Use the issuer URL as the audience for discovery tokens
        discovery_service = discovery_service.with_expected_audience(issuer.to_owned());
    }
    // TODO: DiscoveryService federation key source support
    // (federation_key_source not yet implemented on DiscoveryService)

    Ok(ctx.into_spawnable_quic(discovery_service, config.discovery.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Notification Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for NotificationService (blind relay with broadcast encryption)
#[service_factory("notification", schema = "../../schema/notification.capnp", metadata = crate::services::generated::notification_client::schema_metadata)]
fn create_notification_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating NotificationService");

    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    let mut notification_service = crate::services::NotificationService::new(
        Arc::new(ctx.signing_key().clone()),
        global_context(),
        ctx.transport("notification", SocketKind::Rep),
    ).with_policy_client(policy_client);
    if let Some(issuer) = ctx.oauth_issuer_url() {
        notification_service = notification_service.with_local_issuer_url(issuer.to_owned());
        notification_service = notification_service.with_expected_audience(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        notification_service = notification_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable(notification_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Metrics Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for MetricsService (DuckDB-backed time-series ingest + DataFusion query)
#[service_factory("metrics", schema = "../../schema/metrics.capnp", metadata = crate::services::generated::metrics_client::schema_metadata)]
fn create_metrics_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating MetricsService");

    use crate::services::MetricsService;
    use hyprstream_metrics::query::QueryOrchestrator;
    use hyprstream_metrics::storage::duckdb::DuckDbBackend;
    use hyprstream_metrics::StorageBackend as _;

    let config = load_config();
    let mc = &config.metrics;

    let backend = Arc::new(
        DuckDbBackend::new(mc.db_path.clone(), Default::default(), None)
            .map_err(|e| anyhow::anyhow!("DuckDbBackend init: {e}"))?,
    );

    let orchestrator = Arc::new(
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async {
                    let schema = hyprstream_metrics::metrics::get_metrics_schema();
                    backend
                        .create_table("metrics", &schema)
                        .await
                        .map_err(|e| anyhow::anyhow!("metrics table init: {e}"))?;
                    QueryOrchestrator::new(backend as Arc<dyn hyprstream_metrics::StorageBackend>)
                        .await
                        .map_err(|e| anyhow::anyhow!("QueryOrchestrator init: {e}"))
                })
        })
        .map_err(|e| anyhow::anyhow!("metrics service init: {e}"))?,
    );

    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::anonymous());

    let mut metrics_service = MetricsService::new(
        orchestrator,
        global_context(),
        ctx.transport("metrics", SocketKind::Rep),
        ctx.signing_key().clone(),
        policy_client,
    );
    if let Some(issuer) = ctx.oauth_issuer_url() {
        metrics_service = metrics_service.with_expected_audience(issuer.to_owned());
        metrics_service = metrics_service.with_local_issuer_url(issuer.to_owned());
    }
    if let Some(fed) = ctx.federation_key_source() {
        metrics_service = metrics_service.with_federation_key_source(fed);
    }

    Ok(ctx.into_spawnable_quic(metrics_service, mc.quic_port))
}
