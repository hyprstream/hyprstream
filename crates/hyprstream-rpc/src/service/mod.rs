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

pub use traits::{RpcHandler, RpcRequest, RpcService};
#[allow(deprecated)]
pub use zmq::{AuthorizeFn, CallOptions, Continuation, EnvelopeContext, QuicLoopConfig, ServiceHandle, RequestLoop, UnifiedRequestLoop, ZmqClient, ZmqService};
pub use streaming::StreamService;
pub use spawnable::Spawnable;
pub use metadata::{MethodMeta, ParamMeta, SchemaMetadataFn, ScopedSchemaMetadataFn, ScopedClientTreeNode};

/// Trait for generated service clients that can be constructed from a base `ZmqClient`.
///
/// Implement this trait to enable `ServiceContext::typed_client::<T>()`.
/// Generated clients implement this automatically via `generate_rpc_service!`.
pub trait ServiceClient: Sized {
    /// The service name as registered in the endpoint registry.
    const SERVICE_NAME: &'static str;

    /// Construct this client from a base `ZmqClient`.
    fn from_zmq(client: ZmqClient) -> Self;
}
