//! Event bus infrastructure
//!
//! Provides XPUB/XSUB proxy for reliable event delivery between services.
//!
//! # Architecture
//!
//! ```text
//! Publishers                    EventBroker                Subscribers
//! ┌─────────────┐              ┌───────────┐              ┌──────────┐
//! │RegistryService│──XPUB────▶│           │──XSUB──────▶│Workflow- │
//! │WorkerService  │           │   Proxy   │              │ Service  │
//! │InferenceService│           └───────────┘              └──────────┘
//! └─────────────┘
//!
//! Endpoints:
//!   - EVENTS_PUB = inproc://hyprstream/events/pub
//!   - EVENTS_SUB = inproc://hyprstream/events/sub
//! ```

mod broker;

pub use broker::EventBroker;

/// Event bus publish endpoint (publishers connect here)
pub const EVENTS_PUB: &str = "inproc://hyprstream/events/pub";

/// Event bus subscribe endpoint (subscribers connect here)
pub const EVENTS_SUB: &str = "inproc://hyprstream/events/sub";
