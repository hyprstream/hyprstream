//! Event bus infrastructure — moq-lite backed (#167).
//!
//! Provides fan-out event delivery between services using the moq-lite streaming
//! plane. Replaces the former ZMQ XPUB/XSUB ProxyService.
//!
//! # Architecture
//!
//! ```text
//! Publishers                 MoqEventOrigin (global)       Subscribers
//! ┌─────────────┐           ┌──────────────────────┐      ┌──────────┐
//! │WorkerService │──moq────►│ local/events/worker  │─────►│Workflow- │
//! │RegistryService│          │ local/events/system  │      │ Service  │
//! │                │         │ local/events/registry│      └──────────┘
//! └─────────────┘           └──────────────────────┘
//! ```
//!
//! `EventPublisher`/`EventSubscriber` are the canonical broadcast types
//! (EV1, EventService consolidation epic #600) and live in
//! `hyprstream-rpc::events` alongside the moq transport and crypto they wire
//! together. Import them from that canonical module.
//! Default privacy mode is `EventPrivacy::Public` (plaintext, wire-identical
//! to the pre-EV1 behavior of this crate's old standalone wrapper);
//! `EventPrivacy::ZeroKnowledge`/`LimitedKnowledge` group-key encrypted modes
//! are also available — see `hyprstream_rpc::events` docs.
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::events::{EventPublisher, EventSubscriber};
//!
//! // Create a publisher (no ZMQ context needed)
//! let publisher = EventPublisher::new("worker")?;
//! publisher.publish("sandbox123", "started", &payload).await?;
//!
//! // Create a subscriber
//! let mut subscriber = EventSubscriber::new()?;
//! subscriber.subscribe("worker.")?;
//! while let Ok((topic, payload)) = subscriber.recv().await {
//!     println!("Received: {}", topic);
//! }
//! ```

pub mod token_manager;
mod types;

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

pub use token_manager::EventTokenManager;
