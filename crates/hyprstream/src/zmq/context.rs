//! Global ZMQ context management
//!
//! Delegates to `hyprstream_rpc::zmq_context` for the actual singleton.
//! This module provides backward-compatible re-exports.

use std::sync::Arc;

/// Get the global ZMQ context.
///
/// Delegates to `hyprstream_rpc::zmq_context::global_context()`.
pub fn global_context() -> Arc<zmq::Context> {
    hyprstream_rpc::zmq_context::global_context()
}

/// Initialize the global ZMQ context with custom IO thread count.
///
/// This should be called early in main() before any ZMQ sockets are created.
/// Sets the environment variable so the lazy singleton picks up the value.
pub fn init_with_threads(io_threads: i32) {
    std::env::set_var("HYPRSTREAM_ZMQ_IO_THREADS", io_threads.to_string());
    // Touch the singleton to initialize it
    let _ = global_context();
}

/// Get the configured number of IO threads.
pub fn io_thread_count() -> i32 {
    std::env::var("HYPRSTREAM_ZMQ_IO_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_context_creation() {
        let ctx = global_context();
        let socket = ctx.socket(zmq::PUB);
        assert!(socket.is_ok());
    }

    #[test]
    fn test_context_is_shared() {
        let ctx1 = global_context();
        let ctx2 = global_context();
        assert!(Arc::ptr_eq(&ctx1, &ctx2));
    }
}
