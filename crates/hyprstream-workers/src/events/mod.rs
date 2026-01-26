//! Event bus infrastructure
//!
//! Provides XPUB/XSUB proxy for reliable event delivery between services.
//! See `docs/eventservice-architecture.md` for detailed documentation.
//!
//! # Architecture
//!
//! ```text
//! Publishers                    EventService                Subscribers
//! ┌─────────────┐              ┌───────────┐              ┌──────────┐
//! │WorkerService │──PUB──────►│           │──SUB───────►│Workflow- │
//! │RegistryService│            │   Proxy   │              │ Service  │
//! │InferenceService│           └───────────┘              └──────────┘
//! └─────────────┘
//! ```
//!
//! # Endpoint Modes
//!
//! The EventService supports three endpoint configurations:
//!
//! - **Inproc** (default): In-process transport for monolithic mode
//! - **IPC**: Unix domain sockets for distributed processes
//! - **Systemd FD**: Pre-bound file descriptors from socket activation
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::{
//!     endpoints, ProxyService, ServiceSpawner, SpawnedService,
//!     EventPublisher, EventSubscriber,
//! };
//! use hyprstream_workers::events::endpoints::EndpointMode;
//!
//! // Detect transport configuration
//! let (pub_transport, sub_transport) = endpoints::detect_transports(EndpointMode::Auto);
//!
//! // Create and spawn proxy service
//! let ctx = Arc::new(zmq::Context::new());
//! let proxy = ProxyService::new("events", ctx.clone(), pub_transport, sub_transport);
//! let spawner = ServiceSpawner::threaded();
//! let service = spawner.spawn(proxy).await?;
//!
//! // Create a publisher
//! let mut publisher = EventPublisher::new(&ctx, "worker")?;
//! publisher.publish("sandbox123", "started", &payload).await?;
//!
//! // Create a subscriber
//! let mut subscriber = EventSubscriber::new(&ctx)?;
//! subscriber.subscribe("worker.")?;
//! while let Ok((topic, payload)) = subscriber.recv().await {
//!     println!("Received: {}", topic);
//! }
//!
//! // Graceful shutdown
//! service.stop().await?;
//! ```

pub mod endpoints;
mod publisher;
mod service;
pub mod sockopt;
mod subscriber;
mod types;

pub use publisher::EventPublisher;
pub use subscriber::EventSubscriber;

// Re-export spawner types for the recommended API:
// let proxy = ProxyService::new("events", ctx, pub_transport, sub_transport);
// let service = ServiceSpawner::threaded().spawn(proxy).await?;
pub use hyprstream_rpc::service::spawner::{ProxyService, ServiceSpawner, SpawnedService};

// Re-export endpoint types for convenience
pub use endpoints::EndpointMode;

// Re-export event types
pub use types::{
    // Individual event structs (with ToCapnp/FromCapnp)
    ContainerStarted, ContainerStopped, SandboxStarted, SandboxStopped,
    // Union enum for type-safe handling
    WorkerEvent,
    // EventSubscriber integration
    ReceivedEvent,
    // Serialization helpers
    serialize_container_started, serialize_container_stopped,
    serialize_sandbox_started, serialize_sandbox_stopped,
};

// Inproc endpoint constants (for backward compatibility)
pub use endpoints::{PUB as EVENTS_PUB, SUB as EVENTS_SUB};
