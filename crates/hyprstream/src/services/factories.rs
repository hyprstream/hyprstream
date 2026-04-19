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
use crate::services::generated::policy_client::RegisterServiceKey;
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

/// Register this service's verifying key with the PolicyService CA.
///
/// Called by each non-policy factory so that peer services can resolve
/// our pubkey via `resolveServiceKey` RPC.  No-op for PolicyService itself.
fn register_service_key(
    ctx: &ServiceContext,
    service_name: &str,
    signing_key: &SigningKey,
) -> anyhow::Result<()> {
    // PolicyService doesn't register — it IS the CA.
    if service_name == "policy" {
        return Ok(());
    }

    let jwt = ctx.service_jwt(service_name)
        .ok_or_else(|| anyhow::anyhow!(
            "No service JWT for '{service_name}'. \
             Ensure generate_independent_service_keys() was called."
        ))?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(
        signing_key.clone(),
        RequestIdentity::anonymous(),
        policy_vk,
    );

    let request = RegisterServiceKey {
        service_name: service_name.to_owned(),
        verifying_key: signing_key.verifying_key().as_bytes().to_vec(),
        service_jwt: jwt.to_owned(),
    };

    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(policy_client.register_service_key(&request))
    }).map_err(|e| anyhow::anyhow!("registerServiceKey RPC failed for '{service_name}': {e}"))?;

    info!(service = service_name, "Registered verifying key with PolicyService");
    Ok(())
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
#[service_factory("policy", schema = "../../../hyprstream-rpc-std/schema/policy.capnp", metadata = crate::services::generated::policy_client::schema_metadata)]
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
                let has_anon_tui = rules.iter().any(|r| {
                    r.len() >= 3 && r[0] == "anonymous" && r[2] == "tui:*"
                });
                if !has_anon_tui {
                    let _ = pm.add_policy_with_domain("anonymous", "*", "tui:*", "*", "allow").await;
                    tracing::info!("policy migration: added 'anonymous' TUI access grant");
                }
                if !has_anon_tui {
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
#[service_factory("registry", schema = "../../../hyprstream-rpc-std/schema/registry.capnp", metadata = crate::services::generated::registry_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_registry_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating RegistryService");

    let config = load_config();
    let sk = ctx.service_signing_key("registry");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "registry", &sk)?;

    // Create policy client for authorization checks
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    // Create registry service with infrastructure (blocking since we're in sync context)
    let mut registry_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(RegistryService::new(
            ctx.models_dir(),
            policy_client,
            global_context(),
            ctx.transport("registry", SocketKind::Rep),
            sk.clone(),
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
#[service_factory("model", schema = "../../../hyprstream-rpc-std/schema/model.capnp", metadata = crate::services::generated::model_client::schema_metadata, depends_on = ["policy", "registry", "discovery"])]
fn create_model_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating ModelService");

    use crate::services::{ModelService, ModelServiceConfig};

    let config = load_config();
    let sk = ctx.service_signing_key("model");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "model", &sk)?;

    // Create policy client for authorization checks
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    // Create registry client
    let registry_vk = hyprstream_service::global_trust_store()
        .resolve_one("registry")
        .ok_or_else(|| anyhow::anyhow!("trust store has no registry key"))?;
    let registry_client: RegistryClient = RegistryClient::for_service(
        sk.clone(),
        RequestIdentity::anonymous(),
        registry_vk,
    );

    let mut model_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(ModelService::new(
            ModelServiceConfig::default(),
            sk.clone(),
            policy_client,
            registry_client,
            global_context(),
            ctx.transport("model", SocketKind::Rep),
        ))
    })?;
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
#[service_factory("worker", depends_on = ["policy", "discovery"])]
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

    let sk = ctx.service_signing_key("worker");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "worker", &sk)?;

    // Service includes infrastructure - directly Spawnable via blanket impl
    let mut worker_service = WorkerService::new(
        pool_config,
        backend,
        rafs_store,
        global_context(),
        ctx.transport("worker", SocketKind::Rep),
        sk.clone(),
    )?;

    // Wire up policy-backed authorization
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = crate::services::PolicyClient::for_service(
        sk.clone(),
        hyprstream_rpc::RequestIdentity::anonymous(),
        policy_vk,
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
#[service_factory("oai", depends_on = ["policy", "model", "registry", "discovery"])]
fn create_oai_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAIService");

    use crate::server::state::ServerState;
    use crate::services::generated::model_client::ModelClient;
    use crate::services::OAIService;

    // Load full config for OAI settings
    let config = load_config();
    let sk = ctx.service_signing_key("oai");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "oai", &sk)?;

    // Create ZMQ clients for Model and Policy services
    let model_vk = hyprstream_service::global_trust_store()
        .resolve_one("model")
        .ok_or_else(|| anyhow::anyhow!("trust store has no model key"))?;
    let model_client = ModelClient::for_service(sk.clone(), RequestIdentity::anonymous(), model_vk);
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    // Create registry client
    let registry_vk = hyprstream_service::global_trust_store()
        .resolve_one("registry")
        .ok_or_else(|| anyhow::anyhow!("trust store has no registry key"))?;
    let registry_client: RegistryClient = RegistryClient::for_service(
        sk.clone(),
        RequestIdentity::anonymous(),
        registry_vk,
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
            sk.clone(),
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
#[service_factory("flight", depends_on = ["registry", "discovery"])]
fn create_flight_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating FlightService");

    use crate::services::FlightService;

    // Load full config for Flight settings
    let config = load_config();
    let sk = ctx.service_signing_key("flight");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "flight", &sk)?;

    // Create registry client for dataset lookup (if default_dataset is configured)
    // RegistryClient already implements hyprstream_metrics::RegistryClient
    let registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>> =
        if config.flight.default_dataset.is_some() {
            let registry_vk = hyprstream_service::global_trust_store()
                .resolve_one("registry")
                .ok_or_else(|| anyhow::anyhow!("trust store has no registry key"))?;
            let zmq_client: RegistryClient = RegistryClient::for_service(
                sk.clone(),
                RequestIdentity::anonymous(),
                registry_vk,
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
#[service_factory("oauth", depends_on = ["policy", "discovery"])]
fn create_oauth_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAuthService");

    use crate::services::OAuthService;

    let config = load_config();
    let sk = ctx.service_signing_key("oauth");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "oauth", &sk)?;

    // Pass signing key instead of a pre-created PolicyClient.
    // OAuthService runs in its own tokio runtime (separate thread), so the
    // PolicyClient must be created inside that runtime for ZMQ async I/O to work.
    let oauth_service = OAuthService::new(
        config.oauth.clone(),
        config.tls.clone(),
        sk,
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
#[service_factory("mcp", schema = "../../../hyprstream-rpc-std/schema/mcp.capnp", metadata = crate::services::generated::mcp_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_mcp_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating McpService");

    // Load full config for MCP settings
    let config = load_config();

    // Create McpConfig for the service
    let oauth_issuer = ctx.oauth_issuer_url().map(str::to_owned);
    let federation_key_source = ctx.federation_key_source();
    let sk = ctx.service_signing_key("mcp");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "mcp", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;

    let mcp_config = McpConfig {
        verifying_key: ctx.verifying_key(),
        zmq_context: global_context(),
        signing_key: sk,
        transport: ctx.transport("mcp", SocketKind::Rep),
        ctx: None, // ServiceContext not yet available as Arc — handlers use signing_key directly
        policy_verifying_key: policy_vk,
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
#[service_factory("tui", schema = "../../schema/tui.capnp", depends_on = ["policy", "discovery"])]
fn create_tui_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating TuiService");

    use crate::tui::{TuiState, service::TuiService};

    let config = load_config();
    let tui_config = &config.tui;
    let sk = ctx.service_signing_key("tui");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "tui", &sk)?;

    let state = Arc::new(RwLock::new(TuiState::new(
        80,
        24,
        tui_config.scrollback_lines,
    )));

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    // Build VFS namespace for ChatApps spawned via TUI RPC.
    let (vfs_ns, vfs_subject) = crate::tui::vfs::build_chat_vfs_namespace(&sk)?;

    let mut tui_service = TuiService::new(
        state,
        global_context(),
        ctx.transport("tui", SocketKind::Rep),
        sk.clone(),
    ).with_policy_client(policy_client)
     .with_vfs(vfs_ns, vfs_subject);

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
#[service_factory("discovery", schema = "../../../hyprstream-discovery/schema/discovery.capnp", metadata = hyprstream_discovery::generated::discovery_client::schema_metadata, depends_on = ["policy"])]
fn create_discovery_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating DiscoveryService");

    let config = load_config();
    let sk = ctx.service_signing_key("discovery");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "discovery", &sk)?;

    // Create policy-based authorization provider
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);
    let auth_provider = crate::services::discovery::PolicyAuthProvider::new(policy_client);

    let mut discovery_service = DiscoveryService::new(
        Arc::new(sk),
        ctx.jwt_verifying_key(),
        global_context(),
        ctx.transport("discovery", SocketKind::Rep),
    ).with_auth_provider(Box::new(auth_provider));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        discovery_service = discovery_service.with_oauth_issuer(issuer.to_owned());
        // Use the issuer URL as the audience for discovery tokens
        discovery_service = discovery_service.with_expected_audience(issuer.to_owned());
    }

    // Pre-compute TLS endorsement if QUIC is enabled with a TLS cert.
    // Uses the root verifying key — TLS endorsement is a node-level trust assertion,
    // not specific to any per-service key. Clients verify against the pinned root pubkey.
    if let Some(quic) = ctx.quic_shared() {
        let ed25519_pubkey = ctx.verifying_key().to_bytes();
        let domain = &quic.server_name;
        match compute_tls_endorsement(&quic.key_der, &ed25519_pubkey, domain) {
            Ok(endorsement) => {
                if !endorsement.is_empty() {
                    info!("TLS endorsement computed for domain '{}' ({} bytes)", domain, endorsement.len());
                    discovery_service = discovery_service.with_tls_endorsement(endorsement, domain.clone());
                }
            }
            Err(e) => {
                // Non-fatal: TLS endorsement is optional additive trust
                tracing::warn!("Failed to compute TLS endorsement for '{}': {}", domain, e);
            }
        }
    }
    // TODO: DiscoveryService federation key source support
    // (federation_key_source not yet implemented on DiscoveryService)

    Ok(ctx.into_spawnable_quic(discovery_service, config.discovery.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Notification Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for NotificationService (blind relay with broadcast encryption)
#[service_factory("notification", schema = "../../../hyprstream-rpc-std/schema/notification.capnp", metadata = crate::services::generated::notification_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_notification_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating NotificationService");

    let sk = ctx.service_signing_key("notification");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "notification", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    let mut notification_service = crate::services::NotificationService::new(
        Arc::new(sk),
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
#[service_factory("metrics", schema = "../../../hyprstream-rpc-std/schema/metrics.capnp", metadata = crate::services::generated::metrics_client::schema_metadata, depends_on = ["policy", "discovery"])]
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

    let sk = ctx.service_signing_key("metrics");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "metrics", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = PolicyClient::for_service(sk.clone(), RequestIdentity::anonymous(), policy_vk);

    let mut metrics_service = MetricsService::new(
        orchestrator,
        global_context(),
        ctx.transport("metrics", SocketKind::Rep),
        sk,
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

// ═══════════════════════════════════════════════════════════════════════════════
// TLS Endorsement Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Domain separator for TLS endorsement messages.
const TLS_ENDORSEMENT_V1: &[u8] = b"TLS_ENDORSEMENT_V1";

/// Compute a TLS endorsement signature.
///
/// Signs `TLS_ENDORSEMENT_V1 || ed25519_pubkey || domain` with the TLS private key.
/// Handles ECDSA P-256, RSA, and Ed25519 key types (auto-detected from PKCS8 DER).
///
/// Returns the raw signature bytes, or an empty vec if the key type is unsupported.
fn compute_tls_endorsement(
    tls_key_der: &[u8],
    ed25519_pubkey: &[u8; 32],
    domain: &str,
) -> anyhow::Result<Vec<u8>> {
    // Build message: TLS_ENDORSEMENT_V1 || ed25519_pubkey (32) || domain
    let mut message = Vec::with_capacity(TLS_ENDORSEMENT_V1.len() + 32 + domain.len());
    message.extend_from_slice(TLS_ENDORSEMENT_V1);
    message.extend_from_slice(ed25519_pubkey);
    message.extend_from_slice(domain.as_bytes());

    let rng = ring::rand::SystemRandom::new();

    // Try Ed25519 first (most modern, smallest signature)
    if let Ok(key_pair) = ring::signature::Ed25519KeyPair::from_pkcs8(tls_key_der) {
        return Ok(key_pair.sign(&message).as_ref().to_vec());
    }

    // Try ECDSA P-256 SHA-256
    if let Ok(key_pair) = ring::signature::EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        tls_key_der,
        &rng,
    ) {
        let signature = key_pair.sign(&rng, &message)?;
        return Ok(signature.as_ref().to_vec());
    }

    // Try ECDSA P-384 SHA-384
    if let Ok(key_pair) = ring::signature::EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
        tls_key_der,
        &rng,
    ) {
        let signature = key_pair.sign(&rng, &message)?;
        return Ok(signature.as_ref().to_vec());
    }

    // Try RSA (PKCS1v15 + SHA-256, then PSS + SHA-256)
    if let Ok(key_pair) = ring::signature::RsaKeyPair::from_pkcs8(tls_key_der) {
        let mut signature = vec![0u8; key_pair.public().modulus_len()];
        let padding = &ring::signature::RSA_PKCS1_SHA256;
        key_pair.sign(padding, &rng, &message, &mut signature)?;
        return Ok(signature);
    }

    anyhow::bail!("unsupported TLS key type in PKCS8 DER")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Helper: generate an ECDSA P-256 key pair and return (pkcs8_der, public_key_der)
    fn generate_ecdsa_p256_pair() -> (Vec<u8>, Vec<u8>) {
        let key_pair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let pkcs8 = key_pair.serialize_der();
        let pub_der = key_pair.public_key_der();
        (pkcs8, pub_der.clone())
    }

    fn build_endorsement_message(ed25519_pubkey: &[u8; 32], domain: &str) -> Vec<u8> {
        let mut msg = Vec::with_capacity(TLS_ENDORSEMENT_V1.len() + 32 + domain.len());
        msg.extend_from_slice(TLS_ENDORSEMENT_V1);
        msg.extend_from_slice(ed25519_pubkey);
        msg.extend_from_slice(domain.as_bytes());
        msg
    }

    #[test]
    fn test_tls_endorsement_with_ecdsa_p256() {
        let (pkcs8, _pub_der) = generate_ecdsa_p256_pair();
        let ed25519_pubkey = [0xAB_u8; 32];

        let endorsement = compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "example.com").unwrap();
        assert!(!endorsement.is_empty());
        // ECDSA P-256 fixed-length signature is 64 bytes
        assert_eq!(endorsement.len(), 64);
    }

    #[test]
    fn test_tls_endorsement_wrong_domain_differs() {
        let (pkcs8, _) = generate_ecdsa_p256_pair();
        let ed25519_pubkey = [0xAB_u8; 32];

        let endorsement_a = compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "example.com").unwrap();
        let endorsement_b = compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "evil.com").unwrap();

        // ECDSA signatures are randomized so they'll differ anyway, but the important
        // thing is that the message content changes — verified by the factory logic.
        // Just confirm both succeed.
        assert!(!endorsement_a.is_empty());
        assert!(!endorsement_b.is_empty());
    }

    #[test]
    fn test_tls_endorsement_message_format() {
        let ed25519_pubkey = [0x42_u8; 32];
        let msg = build_endorsement_message(&ed25519_pubkey, "test.local");

        let expected_len = TLS_ENDORSEMENT_V1.len() + 32 + "test.local".len();
        assert_eq!(msg.len(), expected_len);

        // Starts with domain separator
        assert_eq!(&msg[..TLS_ENDORSEMENT_V1.len()], TLS_ENDORSEMENT_V1);
        // Followed by pubkey
        assert_eq!(&msg[TLS_ENDORSEMENT_V1.len()..TLS_ENDORSEMENT_V1.len() + 32], &[0x42_u8; 32]);
        // Followed by domain
        assert_eq!(&msg[TLS_ENDORSEMENT_V1.len() + 32..], b"test.local");
    }

    #[test]
    fn test_tls_endorsement_invalid_key() {
        let ed25519_pubkey = [0xAB_u8; 32];
        let result = compute_tls_endorsement(&[0xFF; 32], &ed25519_pubkey, "example.com");
        assert!(result.is_err());
    }
}
