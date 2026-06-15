//! Service-side RPC infrastructure.
//!
//! This module provides:
//! - `RequestService` - RPC service trait
//! - `EnvelopeContext` - Verified request context passed to handlers
//! - `ServiceHandle` - Handle for managing running services
//! - `RpcService`, `RpcHandler` - Lower-level RPC traits
//!
//! Spawner, factory, and manager have moved to the `hyprstream-service` crate.
//! Metadata types remain here (used by proc macro codegen across all crates).

mod traits;
mod zmq;
pub mod dispatch;
pub mod serve;
pub mod spawnable;
pub mod metadata;
pub mod doc;

pub use traits::{RpcHandler, RpcRequest, RpcService};
/// Transport-neutral request dispatch core (#148) — shared by all front-ends.
pub use dispatch::process_request;
pub use zmq::{AuthorizeFn, Continuation, EnvelopeContext, QuicLoopConfig, ServiceHandle, RequestService};
pub use spawnable::Spawnable;
pub use metadata::{MethodMeta, ParamMeta, SchemaMetadataFn, ScopedSchemaMetadataFn, ScopedClientTreeNode};
pub use doc::DocFs;
