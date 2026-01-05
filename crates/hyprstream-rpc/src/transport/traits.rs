//! Transport traits for RPC communication.

use anyhow::Result;
use async_trait::async_trait;

/// Synchronous transport trait.
///
/// Implementations provide blocking request/response communication.
pub trait Transport: Send + Sync {
    /// Send a request and wait for response.
    fn call(&self, request: Vec<u8>) -> Result<Vec<u8>>;

    /// Check if transport is connected.
    fn is_connected(&self) -> bool;
}

/// Asynchronous transport trait.
///
/// Implementations provide async request/response communication.
#[async_trait]
pub trait AsyncTransport: Send + Sync {
    /// Send a request and await response.
    async fn call(&self, request: Vec<u8>) -> Result<Vec<u8>>;

    /// Check if transport is connected.
    fn is_connected(&self) -> bool;
}
