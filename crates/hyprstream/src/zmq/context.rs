//! Global ZMQ context management
//!
//! Provides a global singleton ZMQ context with configurable IO thread count.
//! The context is shared across all ZMQ sockets in the application.

use once_cell::sync::Lazy;
use std::sync::Arc;
use tracing::{debug, warn};

/// Environment variable for configuring ZMQ IO threads
const ZMQ_IO_THREADS_ENV: &str = "HYPRSTREAM_ZMQ_IO_THREADS";

/// Default number of IO threads for ZMQ context
const DEFAULT_IO_THREADS: i32 = 1;

/// Global ZMQ context singleton.
///
/// IO thread count tuning: ~1 thread per gigabyte of expected event throughput.
/// For most deployments, 1-2 threads is sufficient. High-throughput scenarios
/// (>1GB/s of messages) may benefit from additional threads.
///
/// Configuration:
/// - Environment variable: `HYPRSTREAM_ZMQ_IO_THREADS`
/// - CLI argument: `--zmq-io-threads <N>` (sets env var before init)
/// - Config file: `zmq.io_threads` in `hyprstream.toml`
/// - Default: 1
static ZMQ_CONTEXT: Lazy<Arc<zmq::Context>> = Lazy::new(|| {
    let io_threads = std::env::var(ZMQ_IO_THREADS_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IO_THREADS);

    debug!("initializing global ZMQ context with {} IO thread(s)", io_threads);

    let ctx = zmq::Context::new();
    if let Err(e) = ctx.set_io_threads(io_threads) {
        warn!("Failed to set ZMQ IO threads to {}: {}, using default", io_threads, e);
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

/// Initialize the global ZMQ context with custom IO thread count.
///
/// This should be called early in main() before any ZMQ sockets are created.
/// If the context is already initialized, this is a no-op.
///
/// # Arguments
/// * `io_threads` - Number of IO threads for the context
pub fn init_with_threads(io_threads: i32) {
    // Set env var before accessing the lazy static
    std::env::set_var(ZMQ_IO_THREADS_ENV, io_threads.to_string());
    // Touch the lazy static to initialize it
    let _ = &*ZMQ_CONTEXT;
}

/// Get the configured number of IO threads.
///
/// Returns the value from the environment variable, or the default (1).
pub fn io_thread_count() -> i32 {
    std::env::var(ZMQ_IO_THREADS_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IO_THREADS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_context_creation() {
        let ctx = global_context();
        // Verify we can create a socket
        let socket = ctx.socket(zmq::PUB);
        assert!(socket.is_ok());
    }

    #[test]
    fn test_context_is_shared() {
        let ctx1 = global_context();
        let ctx2 = global_context();
        // Both should point to the same context
        assert!(Arc::ptr_eq(&ctx1, &ctx2));
    }
}
