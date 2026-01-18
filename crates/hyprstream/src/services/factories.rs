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
        ctx.verifying_key(),
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
            ctx.verifying_key(),
        ))
    })?;

    Ok(Box::new(registry_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streams Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for StreamService (XPUB/XSUB proxy with JWT validation)
#[service_factory("streams")]
fn create_streams_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating StreamService with JWT validation");

    use crate::services::StreamService;

    let pub_transport = global_registry().endpoint("streams", SocketKind::Sub);
    let sub_transport = global_registry().endpoint("streams", SocketKind::Pub);

    let stream_service = StreamService::new(
        "streams",
        global_context(),
        pub_transport,
        sub_transport,
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
        ctx.verifying_key(),
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
        ctx.verifying_key(),
    )?;

    Ok(Box::new(worker_service))
}
