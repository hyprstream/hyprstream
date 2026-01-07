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
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::{start_event_service, EventPublisher, EventSubscriber, endpoints};
//!
//! // Start the event service (typically in main)
//! let ctx = global_context();
//! let handle = start_event_service(ctx.clone())?;
//!
//! // Create a publisher in your service
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
//! handle.stop()?;
//! ```

mod publisher;
mod service;
mod subscriber;

pub use publisher::EventPublisher;
pub use service::{start_event_service, EventServiceHandle};
pub use subscriber::EventSubscriber;

/// Event bus endpoints
pub mod endpoints {
    /// Publishers connect here (XSUB binds)
    pub const PUB: &str = "inproc://hyprstream/events/pub";

    /// Subscribers connect here (XPUB binds)
    pub const SUB: &str = "inproc://hyprstream/events/sub";

    /// Control socket for graceful shutdown (PAIR)
    pub const CTRL: &str = "inproc://hyprstream/events/ctrl";
}

// Legacy aliases for backward compatibility
#[deprecated(note = "Use endpoints::PUB instead")]
pub const EVENTS_PUB: &str = endpoints::PUB;

#[deprecated(note = "Use endpoints::SUB instead")]
pub const EVENTS_SUB: &str = endpoints::SUB;
