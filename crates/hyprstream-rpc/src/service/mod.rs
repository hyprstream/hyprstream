//! Service-side RPC infrastructure.
//!
//! This module provides:
//! - `ZmqService`, `RequestLoop`, `ZmqClient` - ZMQ REQ/REP service infrastructure
//! - `EnvelopeContext` - Verified request context passed to handlers
//! - `ServiceHandle` - Handle for managing running services
//! - `RpcService`, `RpcHandler` - Lower-level RPC traits
//!
//! Spawner, factory, and manager have moved to the `hyprstream-service` crate.
//! Metadata types remain here (used by proc macro codegen across all crates).

mod traits;
mod zmq;
pub mod streaming;
pub mod spawnable;
pub mod metadata;
pub mod doc;

pub use traits::{RpcHandler, RpcRequest, RpcService};
#[allow(deprecated)]
pub use zmq::{AuthorizeFn, Continuation, EnvelopeContext, QuicLoopConfig, ServiceHandle, RequestLoop, UnifiedRequestLoop, ZmqService};
pub use streaming::StreamService;
pub use spawnable::Spawnable;
pub use metadata::{MethodMeta, ParamMeta, SchemaMetadataFn, ScopedSchemaMetadataFn, ScopedClientTreeNode};
pub use doc::DocFs;
