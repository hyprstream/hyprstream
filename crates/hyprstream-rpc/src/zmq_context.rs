//! Global ZMQ context singleton for hyprstream-rpc.
//!
//! Provides a process-wide ZMQ context that all clients and services share.
//! This enables `inproc://` connectivity between any ZMQ sockets in the process.

use once_cell::sync::Lazy;
use std::sync::Arc;

/// Environment variable for configuring ZMQ IO threads.
const ZMQ_IO_THREADS_ENV: &str = "HYPRSTREAM_ZMQ_IO_THREADS";

/// Default number of IO threads for ZMQ context.
const DEFAULT_IO_THREADS: i32 = 1;

/// Global ZMQ context singleton.
static ZMQ_CONTEXT: Lazy<Arc<zmq::Context>> = Lazy::new(|| {
    let io_threads = std::env::var(ZMQ_IO_THREADS_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IO_THREADS);

    let ctx = zmq::Context::new();
    if let Err(e) = ctx.set_io_threads(io_threads) {
        tracing::warn!("failed to set ZMQ IO threads to {io_threads}: {e}");
    }

    Arc::new(ctx)
});

/// Get the global ZMQ context.
///
/// The context is initialized lazily on first access.
/// All ZMQ sockets should be created from this context to ensure
/// proper resource sharing and `inproc://` connectivity.
pub fn global_context() -> Arc<zmq::Context> {
    Arc::clone(&ZMQ_CONTEXT)
}

/// Create a service client using the global ZMQ context.
///
/// Derives the server verifying key from the signing key (self-signed scenario).
/// For non-self-signed cases, use `ZmqClient::new` directly.
pub fn create_service_client_base(
    endpoint: &str,
    signing_key: crate::SigningKey,
    identity: crate::RequestIdentity,
) -> crate::service::ZmqClient {
    let server_verifying_key = signing_key.verifying_key();
    crate::service::ZmqClient::new(
        endpoint, global_context(), signing_key, server_verifying_key, identity,
    )
}
