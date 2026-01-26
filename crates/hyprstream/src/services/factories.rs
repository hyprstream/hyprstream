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
use crate::services::{PolicyService, PolicyZmqClient, RegistryClient, RegistryService, RegistryZmqClient};
use crate::storage::ModelStorage;
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
#[service_factory("policy")]
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
#[service_factory("registry")]
fn create_registry_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating RegistryService");

    // Create policy client for authorization checks
    let policy_client = PolicyZmqClient::new(ctx.signing_key().clone(), RequestIdentity::local());

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
#[service_factory("model")]
fn create_model_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating ModelService");

    use crate::services::{ModelService, ModelServiceConfig};

    // Create policy client for authorization checks
    let policy_client = PolicyZmqClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    // Create registry client
    let registry_client = Arc::new(RegistryZmqClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::local(),
    )) as Arc<dyn RegistryClient>;

    // Create model storage
    let model_storage = Arc::new(ModelStorage::new(
        registry_client,
        ctx.models_dir().to_path_buf(),
    ));

    // Service includes infrastructure - directly Spawnable via blanket impl
    let model_service = ModelService::new(
        ModelServiceConfig::default(),
        ctx.signing_key().clone(),
        policy_client,
        model_storage,
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
    let worker_service = WorkerService::new(
        pool_config,
        image_config,
        rafs_store,
        global_context(),
        ctx.transport("worker", SocketKind::Rep),
        ctx.signing_key().clone(),
    )?;

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
    let policy_client = PolicyZmqClient::new(ctx.signing_key().clone(), RequestIdentity::local());

    // Create registry client
    let registry_client = Arc::new(RegistryZmqClient::new(
        ctx.signing_key().clone(),
        RequestIdentity::local(),
    )) as Arc<dyn RegistryClient>;

    // Create model storage
    let model_storage = Arc::new(ModelStorage::new(
        registry_client,
        ctx.models_dir().to_path_buf(),
    ));

    // Create server state (blocking since we're in sync context)
    let server_state = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(ServerState::new(
            config.server.clone(),
            model_client,
            policy_client,
            model_storage,
            ctx.signing_key().clone(),
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
