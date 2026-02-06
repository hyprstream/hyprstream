//! Service-side RPC infrastructure.
//!
//! This module provides:
//! - `ZmqService`, `RequestLoop`, `ZmqClient` - ZMQ REQ/REP service infrastructure
//! - `EnvelopeContext` - Verified request context passed to handlers
//! - `ServiceHandle` - Handle for managing running services
//! - `RpcService`, `RpcHandler` - Lower-level RPC traits
//! - `spawner` - Process and service spawning abstractions
//! - `manager` - Service lifecycle management (systemd/standalone)

mod traits;
mod zmq;
pub mod spawner;
pub mod manager;
pub mod streaming;
pub mod factory;

pub use traits::{RpcHandler, RpcRequest, RpcService};
pub use zmq::{CallOptions, EnvelopeContext, ServiceHandle, RequestLoop, ZmqClient, ZmqService};
pub use streaming::StreamService;
pub use spawner::{InprocManager, Spawnable, SpawnedService};
pub use factory::{get_factory, list_factories, ServiceClient, ServiceContext, ServiceFactory};

// Re-export service manager types
pub use manager::{detect as detect_service_manager, ServiceManager, StandaloneManager};
#[cfg(feature = "systemd")]
pub use manager::SystemdManager;
