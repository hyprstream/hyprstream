//! Service-side RPC infrastructure.
//!
//! This module provides:
//! - `ZmqService`, `ServiceRunner`, `ZmqClient` - ZMQ REQ/REP service infrastructure
//! - `EnvelopeContext` - Verified request context passed to handlers
//! - `ServiceHandle` - Handle for managing running services
//! - `RpcService`, `RpcHandler` - Lower-level RPC traits

mod traits;
mod zmq;

pub use traits::{RpcHandler, RpcRequest, RpcService};
pub use zmq::{EnvelopeContext, ServiceHandle, ServiceRunner, ZmqClient, ZmqService};
