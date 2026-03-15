//! Service endpoint resolution abstraction.
//!
//! Defines the `Resolver` trait for converting service names to transport
//! endpoints. The trait is async to support federation (network-based
//! resolution) in addition to local registry lookups.
//!
//! # Architecture
//!
//! ```text
//! hyprstream-rpc:     trait Resolver + SocketKind + pluggable global
//! hyprstream-discovery: EndpointRegistry implements Resolver (local)
//! future:              DiscoveryClient implements Resolver (remote/federated)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::{Resolver, SocketKind};
//!
//! async fn connect(resolver: &dyn Resolver) -> Result<()> {
//!     let endpoint = resolver.resolve("inference", SocketKind::Rep).await?;
//!     // Use endpoint...
//!     Ok(())
//! }
//! ```

use std::sync::Arc;

use parking_lot::RwLock;

use crate::registry::SocketKind;
use crate::transport::TransportConfig;

/// Async endpoint resolver.
///
/// Implementations convert a (service_name, socket_kind) pair into a
/// concrete `TransportConfig`. The async signature supports both local
/// (in-memory) and remote (network RPC) resolution.
///
/// # Implementors
///
/// - `EndpointRegistry` — local, in-process resolution (sync internally)
/// - `DiscoveryService` — authoritative resolution after bootstrap
/// - `DiscoveryClient` — remote resolution via DiscoveryService RPC (future)
#[async_trait::async_trait]
pub trait Resolver: Send + Sync {
    /// Resolve a service endpoint.
    ///
    /// Returns the transport configuration for the given service and socket
    /// type. Returns `Err` if the service is unknown and no default can be
    /// generated.
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig>;
}

// ============================================================================
// Pluggable global resolver (replaceable)
// ============================================================================

static GLOBAL_RESOLVER: RwLock<Option<Arc<dyn Resolver>>> = RwLock::new(None);

/// Set the global resolver.
///
/// Can be called multiple times — each call replaces the previous resolver.
/// During bootstrap, `registry::init()` installs a `GlobalRegistryResolver`.
/// Once `DiscoveryService` starts, it replaces this with itself.
///
/// # Example
///
/// ```ignore
/// // Bootstrap: registry installs itself
/// hyprstream_rpc::registry::init(mode, runtime_dir);
///
/// // After DiscoveryService starts, it replaces the bootstrap resolver:
/// hyprstream_rpc::resolver::set_global(discovery_service.clone());
/// ```
pub fn set_global(resolver: Arc<dyn Resolver>) {
    *GLOBAL_RESOLVER.write() = Some(resolver);
}

/// Get the global resolver (non-panicking).
///
/// Returns `None` before `set_global()` has been called.
pub fn try_global() -> Option<Arc<dyn Resolver>> {
    GLOBAL_RESOLVER.read().clone()
}

/// Get the global resolver.
///
/// # Panics
///
/// Panics if `set_global()` has not been called.
#[deprecated(note = "use try_global() instead for graceful degradation (D9)")]
pub fn global() -> Arc<dyn Resolver> {
    #[allow(clippy::expect_used)] // Intentional panic for programming error
    GLOBAL_RESOLVER.read().clone().expect("Global resolver not initialized — call resolver::set_global() first")
}
