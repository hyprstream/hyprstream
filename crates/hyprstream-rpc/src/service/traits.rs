//! Service traits for RPC handlers.

use anyhow::Result;

/// Trait for RPC service implementations.
///
/// This is the core service trait that handles incoming requests.
/// Implementations should parse the request, dispatch to handlers,
/// and build responses.
pub trait RpcService: Send + 'static {
    /// Handle an incoming request and return a response.
    ///
    /// The request and response are Cap'n Proto serialized bytes.
    fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>>;

    /// Get the service name for logging/debugging.
    fn name(&self) -> &str;
}

/// Trait for individual RPC method handlers.
///
/// Each RPC method can be implemented as a separate handler,
/// allowing for cleaner separation of concerns.
pub trait RpcHandler<Req, Resp>: Send + Sync {
    /// Handle a typed request and return a typed response.
    fn handle(&self, request: Req) -> Result<Resp>;
}

/// Marker trait for request types that can identify their expected response.
///
/// Used by generic clients to ensure type safety between requests and responses.
pub trait RpcRequest {
    /// The response type expected for this request.
    type Response;

    /// Get the request variant name (for logging/debugging).
    fn variant_name(&self) -> &'static str;
}
