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
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as global_registry, SocketKind};
use hyprstream_rpc::service::factory::ServiceContext;
use hyprstream_rpc::service::spawner::{ProxyService, Spawnable};
use hyprstream_rpc::service_factory;
use hyprstream_workers::endpoints;
use tracing::info;

use crate::auth::PolicyManager;
use crate::config::TokenConfig;
use crate::services::{McpService, McpConfig, PolicyService, PolicyClient, RegistryClient, RegistryService, RegistryZmqClient};
use crate::zmq::global_context;

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

    // Create policy manager (blocking since we're in sync context)
    let policy_manager = Arc::new(
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(PolicyManager::new(&policies_dir))
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

    // Service includes infrastructure - directly Spawnable via blanket impl
    let policy_service = PolicyService::new(
        policy_manager,
        Arc::new(ctx.signing_key().clone()),
        TokenConfig::default(),
        global_context(),
        ctx.transport("policy", SocketKind::Rep),
    );

    Ok(Box::new(policy_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for RegistryService (git2db model registry)
#[service_factory("registry", schema = "../../schema/registry.capnp", metadata = crate::services::generated::registry_client::schema_metadata)]
fn create_registry_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating RegistryService");

    // Create policy client for authorization checks
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    // Create registry service with infrastructure (blocking since we're in sync context)
    let registry_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(RegistryService::new(
            ctx.models_dir(),
            policy_client,
            global_context(),
            ctx.transport("registry", SocketKind::Rep),
            ctx.signing_key().clone(),
        ))
    })?;

    Ok(Box::new(registry_service))
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

    // Create policy client for authorization checks
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    // Create registry client
    let registry_client: Arc<dyn RegistryClient> = Arc::new(RegistryZmqClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::local(),
    ));

    // Service includes infrastructure - directly Spawnable via blanket impl
    let model_service = ModelService::new(
        ModelServiceConfig::default(),
        ctx.signing_key().clone(),
        policy_client,
        registry_client,
        global_context(),
        ctx.transport("model", SocketKind::Rep),
    );

    Ok(Box::new(model_service))
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

    use hyprstream_workers::config::{ImageConfig, PoolConfig};
    use hyprstream_workers::image::RafsStore;
    use hyprstream_workers::WorkerService;

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

    // Service includes infrastructure - directly Spawnable via blanket impl
    let mut worker_service = WorkerService::new(
        pool_config,
        image_config,
        rafs_store,
        global_context(),
        ctx.transport("worker", SocketKind::Rep),
        ctx.signing_key().clone(),
    )?;

    // Wire up policy-backed authorization
    let policy_client = crate::services::PolicyClient::new(
        ctx.signing_key().clone(),
        hyprstream_rpc::RequestIdentity::local(),
    );
    worker_service.set_authorize_fn(super::worker::build_authorize_fn(policy_client));

    Ok(Box::new(worker_service))
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

    use crate::config::HyprConfig;
    use crate::server::state::ServerState;
    use crate::services::{ModelZmqClient, OAIService};

    // Load full config for OAI settings
    let config = HyprConfig::load().unwrap_or_default();

    // Create ZMQ clients for Model and Policy services
    let model_client = ModelZmqClient::new(ctx.signing_key().clone(), RequestIdentity::local());
    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    // Create registry client
    let registry_client: Arc<dyn RegistryClient> = Arc::new(RegistryZmqClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::local(),
    ));

    // Create server state (blocking since we're in sync context)
    let resource_url = config.oai.resource_url();
    let server_state = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(ServerState::new(
            config.server.clone(),
            model_client,
            policy_client,
            registry_client,
            ctx.signing_key().clone(),
            resource_url,
        ))
    })
    .context("Failed to create server state")?;

    let oai_service = OAIService::new(
        config.oai.clone(),
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

    use crate::config::HyprConfig;
    use crate::services::FlightService;

    // Load full config for Flight settings
    let config = HyprConfig::load().unwrap_or_default();

    // Create registry client for dataset lookup (if default_dataset is configured)
    // RegistryZmqClient already implements hyprstream_metrics::RegistryClient
    let registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>> =
        if config.flight.default_dataset.is_some() {
            let zmq_client = RegistryZmqClient::new(
                ctx.signing_key().clone(),
                RequestIdentity::local(),
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

    use crate::config::HyprConfig;
    use crate::services::OAuthService;

    let config = HyprConfig::load().unwrap_or_default();

    let policy_client = PolicyClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    let oauth_service = OAuthService::new(
        config.oauth.clone(),
        policy_client,
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

    use crate::config::HyprConfig;

    // Load full config for MCP settings
    let config = HyprConfig::load().unwrap_or_default();

    // Create McpConfig for the service
    let mcp_config = McpConfig {
        verifying_key: ctx.verifying_key(),
        zmq_context: global_context(),
        signing_key: ctx.signing_key().clone(),
        transport: ctx.transport("mcp", SocketKind::Rep),
        ctx: None, // ServiceContext not yet available as Arc — handlers use signing_key directly
        expected_audience: Some(config.mcp.resource_url()),
    };

    // Clone config for HTTP/SSE server before consuming it for ZMQ service
    let mcp_config_clone = mcp_config.clone();

    // Create the service (includes ZMQ infrastructure)
    let mcp_service = McpService::new(mcp_config)?;

    // Spawn rmcp HTTP/SSE server as background task
    let mcp_host = config.mcp.host.clone();
    let http_port = config.mcp.http_port;
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            use rmcp::transport::streamable_http_server::{
                StreamableHttpServerConfig, StreamableHttpService,
            };

            use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

            let session_mgr = std::sync::Arc::new(LocalSessionManager::default());
            let service: StreamableHttpService<McpService, LocalSessionManager> =
                StreamableHttpService::new(
                    move || McpService::new(mcp_config_clone.clone()).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                    }),
                    session_mgr,
                    StreamableHttpServerConfig::default(),
                );
            // Add protected resource metadata (RFC 9728) for OAuth discovery
            let mcp_full_config = crate::config::HyprConfig::load().unwrap_or_default();
            let mcp_resource_url = mcp_full_config.mcp.resource_url();
            let mcp_oauth_issuer = mcp_full_config.oauth.issuer_url();
            let router = axum::Router::new()
                .route(
                    "/.well-known/oauth-protected-resource",
                    axum::routing::get(move || async move {
                        axum::Json(crate::services::oauth::protected_resource_metadata(
                            &mcp_resource_url,
                            &mcp_oauth_issuer,
                        ))
                    }),
                )
                .nest_service("/mcp", service);

            let addr = format!("{}:{}", mcp_host, http_port);
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("Failed to bind MCP HTTP/SSE on {}: {}", addr, e);
                    return;
                }
            };
            tracing::info!("MCP HTTP/SSE server listening on {}", addr);
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!("MCP HTTP/SSE server error: {}", e);
            }
        });
    });
    info!("McpService created (HTTP/SSE on {}:{})", config.mcp.host, http_port);

    Ok(Box::new(mcp_service))  // Gets auto-Spawnable via ZmqService
}
