//! Service factory infrastructure for inventory-based service registration.
//!
//! This module provides the `ServiceFactory` type and `ServiceContext` for
//! implementing the same inventory pattern used for `ScopeDefinition` and
//! `DriverFactory`.
//!
//! # Example
//!
//! ```ignore
//! use hyprstream_rpc::service::factory::{ServiceContext, ServiceFactory};
//! use hyprstream_rpc_derive::service_factory;
//!
//! #[service_factory("policy")]
//! fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
//!     // Services include infrastructure and are directly Spawnable
//!     let policy = PolicyService::new(
//!         ...,
//!         ctx.zmq_context(),
//!         ctx.transport("policy", SocketKind::Rep),
//!         ctx.verifying_key(),
//!     );
//!     Ok(Box::new(policy))
//! }
//! ```

use std::sync::Arc;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::registry::{global as global_registry, SocketKind};
use crate::service::spawner::Spawnable;
use crate::transport::TransportConfig;

/// Context for service creation.
///
/// Contains all shared resources needed by services during initialization.
/// Passed to factory functions registered via `#[service_factory]`.
pub struct ServiceContext {
    /// ZMQ context (shared across all services)
    zmq_context: Arc<zmq::Context>,

    /// Server's signing key (for JWT generation)
    signing_key: SigningKey,

    /// Server's verifying key (for envelope/JWT verification)
    verifying_key: VerifyingKey,

    /// Whether running in IPC mode (vs inproc)
    ipc: bool,

    /// Models directory path
    models_dir: std::path::PathBuf,
}

impl ServiceContext {
    /// Create a new service context.
    pub fn new(
        zmq_context: Arc<zmq::Context>,
        signing_key: SigningKey,
        verifying_key: VerifyingKey,
        ipc: bool,
        models_dir: std::path::PathBuf,
    ) -> Self {
        Self {
            zmq_context,
            signing_key,
            verifying_key,
            ipc,
            models_dir,
        }
    }

    /// Get the shared ZMQ context.
    pub fn zmq_context(&self) -> Arc<zmq::Context> {
        self.zmq_context.clone()
    }

    /// Get the signing key.
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// Get the verifying key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }

    /// Check if running in IPC mode.
    pub fn is_ipc(&self) -> bool {
        self.ipc
    }

    /// Get models directory path.
    pub fn models_dir(&self) -> &std::path::Path {
        &self.models_dir
    }

    /// Get transport config for a service endpoint from the registry.
    ///
    /// This looks up the endpoint from the global EndpointRegistry.
    pub fn endpoint(&self, service: &str, kind: SocketKind) -> TransportConfig {
        global_registry().endpoint(service, kind)
    }

    /// Get unified transport config for a service.
    ///
    /// In IPC mode, returns a Unix socket path in the runtime directory.
    /// In inproc mode, returns the endpoint from the global registry.
    ///
    /// This unifies the transport resolution logic that was previously
    /// duplicated across factory functions.
    pub fn transport(&self, service: &str, kind: SocketKind) -> TransportConfig {
        if self.ipc {
            let runtime_dir = crate::paths::runtime_dir();
            TransportConfig::ipc(runtime_dir.join(format!("{}.sock", service)))
        } else {
            global_registry().endpoint(service, kind)
        }
    }
}

/// Factory function signature for creating services.
///
/// Takes a `ServiceContext` and returns a boxed `Spawnable` service.
pub type ServiceFactoryFn = fn(&ServiceContext) -> anyhow::Result<Box<dyn Spawnable>>;

/// Service factory for inventory-based registration.
///
/// Services register their factory function using `#[service_factory("name")]`,
/// which generates an `inventory::submit!` for this type.
///
/// # Pattern
///
/// Same pattern as:
/// - `ScopeDefinition` with `#[register_scopes]` for authorization scopes
/// - `DriverFactory` in git2db for storage drivers
pub struct ServiceFactory {
    /// Service name (matches config.services.startup entries)
    pub name: &'static str,

    /// Factory function that creates the service
    pub factory: ServiceFactoryFn,
}

impl ServiceFactory {
    /// Create a new service factory.
    ///
    /// Called by the `#[service_factory]` macro-generated code.
    pub const fn new(name: &'static str, factory: ServiceFactoryFn) -> Self {
        Self { name, factory }
    }
}

// Collect all registered factories
inventory::collect!(ServiceFactory);

/// Get a service factory by name.
///
/// Looks up the factory from compile-time registered factories.
///
/// # Example
///
/// ```ignore
/// let factory = get_factory("policy").ok_or_else(|| anyhow!("Unknown service: policy"))?;
/// let spawnable = (factory.factory)(&ctx)?;
/// manager.spawn(spawnable).await?;
/// ```
pub fn get_factory(name: &str) -> Option<&'static ServiceFactory> {
    inventory::iter::<ServiceFactory>().find(|f| f.name == name)
}

/// List all registered service factories.
///
/// Useful for introspection and help text.
pub fn list_factories() -> impl Iterator<Item = &'static ServiceFactory> {
    inventory::iter::<ServiceFactory>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_factory_creation() {
        fn dummy_factory(_ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
            Err(anyhow::anyhow!("dummy"))
        }

        let factory = ServiceFactory::new("test", dummy_factory);
        assert_eq!(factory.name, "test");
    }
}
