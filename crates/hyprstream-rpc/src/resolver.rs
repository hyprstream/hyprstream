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

use anyhow::anyhow;
use parking_lot::RwLock;

use crate::registry::SocketKind;
use crate::transport::{EndpointType, TransportConfig};

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

/// Resolver profile names used at service startup.
///
/// Profiles make locality an explicit deployment decision. Generated clients
/// still ask for `(service_name, socket_kind)`; the installed resolver profile
/// decides whether that name maps to an in-process handle, a same-host Unix
/// socket, or a network-discovered peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolverProfile {
    /// Single-process / hermetic-test profile.
    LocalInproc,
    /// Local IPC profile.
    Ipc,
    /// Multi-host production profile backed by DiscoveryService/PDS records.
    NetworkDiscovery,
}

/// Network profile wrapper for a DiscoveryService/PDS-backed resolver.
///
/// The wrapped resolver owns service discovery. This wrapper enforces the
/// network-profile invariant: same-host endpoints (`inproc`, `ipc`,
/// `systemd-fd`) are not valid routable service reach.
pub struct NetworkDiscoveryResolver {
    inner: Arc<dyn Resolver>,
}

impl NetworkDiscoveryResolver {
    pub fn new(inner: Arc<dyn Resolver>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl Resolver for NetworkDiscoveryResolver {
    async fn resolve(&self, name: &str, kind: SocketKind) -> anyhow::Result<TransportConfig> {
        let transport = self.inner.resolve(name, kind).await?;
        match &transport.endpoint {
            EndpointType::Inproc { .. }
            | EndpointType::Ipc { .. }
            | EndpointType::SystemdFd { .. } => Err(anyhow!(
                "network-discovery resolver returned same-host endpoint for service '{name}' ({kind:?})"
            )),
            EndpointType::Quic { .. } | EndpointType::Iroh { .. } => Ok(transport),
        }
    }
}

// ============================================================================
// Pluggable global resolver (replaceable)
// ============================================================================

static GLOBAL_RESOLVER: RwLock<Option<Arc<dyn Resolver>>> = RwLock::new(None);

/// Set the global resolver.
///
/// Can be called multiple times — each call replaces the previous resolver.
/// During bootstrap, `registry::init()` installs an explicit local resolver
/// profile. A networked deployment can replace that with a
/// [`NetworkDiscoveryResolver`] wrapping DiscoveryService or a DiscoveryClient.
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
    GLOBAL_RESOLVER
        .read()
        .clone()
        .expect("Global resolver not initialized — call resolver::set_global() first")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticResolver(TransportConfig);

    #[async_trait::async_trait]
    impl Resolver for StaticResolver {
        async fn resolve(&self, _name: &str, _kind: SocketKind) -> anyhow::Result<TransportConfig> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn network_discovery_resolver_rejects_inproc_reach() {
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(
            TransportConfig::inproc("hyprstream/policy"),
        )));

        let err = match resolver.resolve("policy", SocketKind::Rep).await {
            Ok(endpoint) => panic!("network resolver accepted local endpoint: {endpoint:?}"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("same-host endpoint"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn network_discovery_resolver_rejects_same_host_reach() {
        for local in [
            TransportConfig::ipc("/run/hyprstream/policy.sock"),
            TransportConfig::systemd_fd(3, "/run/hyprstream/policy.sock"),
        ] {
            let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(local)));
            let err = match resolver.resolve("policy", SocketKind::Rep).await {
                Ok(endpoint) => {
                    panic!("network resolver accepted same-host endpoint: {endpoint:?}")
                }
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("same-host endpoint"),
                "unexpected error: {err}"
            );
        }
    }

    #[tokio::test]
    async fn network_discovery_resolver_accepts_quic_reach() {
        let addr = "127.0.0.1:9443"
            .parse()
            .unwrap_or_else(|err| panic!("test socket address must parse: {err}"));
        let quic = TransportConfig::quic(addr, "policy.local");
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(quic.clone())));

        let endpoint = resolver
            .resolve("policy", SocketKind::Rep)
            .await
            .unwrap_or_else(|err| panic!("network resolver rejected QUIC endpoint: {err}"));
        assert_eq!(endpoint.endpoint_string(), quic.endpoint_string());
    }

    #[tokio::test]
    async fn network_discovery_resolver_accepts_iroh_reach() {
        let iroh = TransportConfig::iroh([7; 32], Vec::new(), None);
        let resolver = NetworkDiscoveryResolver::new(Arc::new(StaticResolver(iroh.clone())));

        let endpoint = resolver
            .resolve("policy", SocketKind::Rep)
            .await
            .unwrap_or_else(|err| panic!("network resolver rejected iroh endpoint: {err}"));
        assert_eq!(endpoint, iroh);
    }
}
