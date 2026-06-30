//! Event bus infrastructure — moq-lite backed (#167).
//!
//! Provides fan-out event delivery between services using the moq-lite streaming
//! plane (Live preset: at-most-once, unbounded). Replaces the former ZMQ
//! XPUB/XSUB ProxyService.
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
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::{EventPublisher, EventSubscriber};
//!
//! // Create a publisher (no ZMQ context needed)
//! let mut publisher = EventPublisher::new("worker")?;
//! publisher.publish("sandbox123", "started", &payload).await?;
//!
//! // Create a subscriber
//! let mut subscriber = EventSubscriber::new()?;
//! subscriber.subscribe("worker.")?;
//! while let Ok((topic, payload)) = subscriber.recv().await {
//!     println!("Received: {}", topic);
//! }
//! ```

mod publisher;
pub mod secure_publisher;
pub mod secure_subscriber;
mod subscriber;
pub mod token_manager;
mod types;

// The generic keyable-group primitive (GroupKeyRegistry, GroupRef,
// MembershipResolver, RekeyPolicy, EncryptedEvent, …) lives in
// `hyprstream-rpc::crypto::group_key` — re-exported here for consumers that
// reach it via `hyprstream_workers::events`.
pub use hyprstream_rpc::crypto::group_key::{
    DenyAllResolver, GroupKeyRegistry, GroupMembership, GroupRef, MembershipResolver,
};

pub use publisher::EventPublisher;
pub use subscriber::EventSubscriber;

// Re-export event types
pub use types::{
    // Serialization helpers
    serialize_container_started,
    serialize_container_stopped,
    serialize_sandbox_started,
    serialize_sandbox_stopped,
    // Individual event structs (with ToCapnp/FromCapnp)
    ContainerStarted,
    ContainerStopped,
    // EventSubscriber integration
    ReceivedEvent,
    SandboxStarted,
    SandboxStopped,
    // Union enum for type-safe handling
    WorkerEvent,
};

// Secure event transport (Phase 7)
pub use secure_publisher::{EncryptedEvent, RekeyPolicy, RotationResult, SecureEventPublisher};
pub use secure_subscriber::SecureEventSubscriber;
pub use token_manager::EventTokenManager;
