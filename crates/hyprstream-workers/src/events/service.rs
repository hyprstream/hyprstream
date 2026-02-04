//! EventService - XPUB/XSUB proxy for event distribution
//!
//! **Migration**: Use `ProxyService` + `ServiceSpawner` instead of the legacy functions.
//!
//! # New API (Recommended)
//!
//! ```ignore
//! use hyprstream_workers::events::{endpoints, ProxyService, ServiceSpawner, SpawnedService};
//! use hyprstream_workers::events::endpoints::EndpointMode;
//!
//! // Detect transport configuration
//! let (pub_transport, sub_transport) = endpoints::detect_transports(EndpointMode::Auto);
//!
//! // Create proxy service
//! let ctx = Arc::new(zmq::Context::new());
//! let proxy = ProxyService::new("events", ctx.clone(), pub_transport, sub_transport);
//!
//! // Spawn on dedicated thread
//! let spawner = ServiceSpawner::threaded();
//! let service: SpawnedService = spawner.spawn(proxy).await?;
//!
//! // Stop when done
//! service.stop().await?;
//! ```
//!
//! # Endpoint Modes
//!
//! - **Inproc** (default): In-process transport for monolithic mode
//! - **IPC**: Unix domain sockets for distributed processes
//! - **Systemd FD**: Pre-bound file descriptors from socket activation

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use hyprstream_rpc::service::spawner::{ProxyService, ServiceSpawner};
    use crate::events::endpoints;

    #[tokio::test]
    async fn test_proxy_service() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ctx = Arc::new(zmq::Context::new());
        let (pub_t, sub_t) = endpoints::inproc_transports();

        let proxy = ProxyService::new("events-test", ctx, pub_t, sub_t);
        let spawner = ServiceSpawner::threaded();
        let mut service = spawner.spawn(proxy).await?;

        assert!(service.is_running());

        service.stop().await?;
        Ok(())
    }
}
