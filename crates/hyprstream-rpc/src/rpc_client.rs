//! Transport-agnostic RPC client trait.
//!
//! Implemented by both native (ZmqClient) and browser (WtClient/RpcSession) transports.
//! Generated client structs can use this trait to be transport-agnostic.

use std::future::Future;
use std::pin::Pin;

use anyhow::Result;

/// Transport-agnostic async RPC client.
///
/// The generated client structs (`RegistryClient`, `ModelClient`, etc.) can use this
/// trait for sending requests. Implement it for your transport layer:
///
/// - Native: `ZmqClientBase` (via `hyprstream_rpc::service`)
/// - Browser: `RpcSession` (via `hyprstream_rpc_std::wasm_exports`)
pub trait RpcClient: Send + Sync {
    /// Send a request and return the response bytes.
    fn call(&self, payload: Vec<u8>) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + '_>>;

    /// Send a streaming request (with ephemeral DH pubkey) and return the response bytes.
    fn call_streaming(&self, payload: Vec<u8>, _ephemeral_pubkey: [u8; 32]) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + '_>> {
        self.call(payload)
    }

    /// Get the next request ID (monotonically increasing).
    fn next_id(&self) -> u64;
}
