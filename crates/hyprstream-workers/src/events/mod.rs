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
//! (EV1, EventService consolidation epic #600) — they now live in
//! `hyprstream-rpc::events` alongside the moq transport and crypto they wire
//! together, and are re-exported here for back-compat with existing callers.
//! Default privacy mode is `EventPrivacy::Public` (plaintext, wire-identical
//! to the pre-EV1 behavior of this crate's old standalone wrapper);
//! `EventPrivacy::ZeroKnowledge`/`LimitedKnowledge` group-key encrypted modes
//! are also available — see `hyprstream_rpc::events` docs.
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::{EventPublisher, EventSubscriber};
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

pub use hyprstream_rpc::events::{
    EncryptedEvent, EventPublisher, EventSubscriber, RekeyEvent, RekeyPolicy, RotationResult,
    WrappedKeyEntry,
};

// The generic keyable-group primitive (GroupKeyRegistry, GroupRef,
// MembershipResolver, GroupMembership, DenyAllResolver) lives in
// `hyprstream_rpc::crypto::group_key` — re-exported here for consumers that
// historically imported event primitives via this module. (EncryptedEvent /
// RekeyPolicy / RotationResult / WrappedKeyEntry are canonical there too, and
// are also reachable via the `hyprstream_rpc::events` re-export above.)
pub use hyprstream_rpc::crypto::group_key::{
    DenyAllResolver, GroupKeyRegistry, GroupMembership, GroupRef, MembershipResolver,
};

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
