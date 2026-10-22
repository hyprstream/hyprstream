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

pub mod body;
pub mod dispatch;
pub mod doc;
pub mod metadata;
pub mod serve;
pub mod spawnable;
mod svc;
mod traits;

pub use body::DecodedRequestBody;
/// Transport-neutral request dispatch core (#148) — shared by all front-ends.
pub use dispatch::process_request;
pub use doc::DocFs;
pub use metadata::{
    MethodMeta, ParamMeta, SchemaMetadataFn, ScopedClientTreeNode, ScopedSchemaMetadataFn,
};
pub use spawnable::Spawnable;
pub use svc::{
    AuthorizeFn, Continuation, EnvelopeContext, QuicLoopConfig, RequestService, ServiceHandle,
};
pub use traits::{RpcHandler, RpcRequest, RpcService};
