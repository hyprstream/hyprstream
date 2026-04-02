//! Pluggable service orchestration for hyprstream.
//!
//! This crate provides service lifecycle management:
//! - **Spawner**: `Spawnable` trait, `ServiceSpawner`, process backends
//! - **Factory**: `ServiceFactory`, `ServiceContext`, inventory-based registration
//! - **Manager**: `ServiceManager` trait, systemd/standalone implementations
//! - **Metadata**: `MethodMeta`, `ParamMeta` for schema introspection
//!
//! # Architecture
//!
//! ```text
//! hyprstream-rpc       (transport: ZmqService, RequestLoop, Resolver)
//!     ↑
//! hyprstream-service   (orchestration: spawner, factory, manager)
//!     ↑
//! hyprstream           (application: factory functions, CLI)
//! ```

pub mod service;
pub mod notify;

// Top-level re-exports for convenience
pub use service::spawner::{
    DualSpawnable, LoadBalancerService, ProcessBackend, ProcessConfig, ProcessKind,
    ProcessSpawner, ProxyService, ServiceKind, ServiceMode, ServiceSpawner,
    Spawnable, SpawnedProcess, SpawnedService, SpawnerBackend, StandaloneBackend,
    SystemdBackend, InprocManager, UnifiedServiceConfig,
};

pub use service::factory::{
    get_factory, list_factories, QuicSharedConfig, ServiceClient, ServiceContext, ServiceFactory,
    ServiceFactoryFn,
};

pub use service::manager::{detect as detect_service_manager, ServiceManager, StandaloneManager};

#[cfg(feature = "systemd")]
pub use service::manager::SystemdManager;

#[cfg(feature = "systemd")]
pub use service::manager::systemd::encrypt_credentials_if_available;

pub use service::metadata::{MethodMeta, ParamMeta, SchemaMetadataFn, ScopedSchemaMetadataFn, ScopedClientTreeNode};

pub use service::ordering::startup_stages;
