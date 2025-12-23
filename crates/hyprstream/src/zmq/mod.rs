//! ZeroMQ infrastructure for hyprstream
//!
//! This module provides the ZMQ infrastructure for messaging:
//! - Global singleton context with configurable IO threads
//! - Re-exports for convenience
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::zmq::global_context;
//!
//! // Get the global context
//! let ctx = global_context();
//!
//! // Create sockets from the context
//! let pub_socket = ctx.socket(zmq::PUB)?;
//! let sub_socket = ctx.socket(zmq::SUB)?;
//! ```
//!
//! # Configuration
//!
//! IO thread count can be configured via:
//! - Environment variable: `HYPRSTREAM_ZMQ_IO_THREADS`
//! - CLI argument: `--zmq-io-threads <N>`
//! - Config file: `zmq.io_threads` in `hyprstream.toml`
//!
//! Default is 1 IO thread. For high-throughput scenarios (>1GB/s),
//! consider 2-4 threads.

mod context;

pub use context::{global_context, init_with_threads, io_thread_count};

// Re-export common ZMQ types for convenience
pub use zmq::{Context, Error as ZmqError, Message, Socket, SocketType};

/// ZMQ socket types as constants for convenience
pub mod socket_types {
    pub const PUB: zmq::SocketType = zmq::PUB;
    pub const SUB: zmq::SocketType = zmq::SUB;
    pub const REQ: zmq::SocketType = zmq::REQ;
    pub const REP: zmq::SocketType = zmq::REP;
    pub const DEALER: zmq::SocketType = zmq::DEALER;
    pub const ROUTER: zmq::SocketType = zmq::ROUTER;
    pub const PUSH: zmq::SocketType = zmq::PUSH;
    pub const PULL: zmq::SocketType = zmq::PULL;
    pub const XPUB: zmq::SocketType = zmq::XPUB;
    pub const XSUB: zmq::SocketType = zmq::XSUB;
    pub const PAIR: zmq::SocketType = zmq::PAIR;
}

/// Standard endpoint constants
pub mod endpoints {
    /// Default inproc endpoint for events
    pub const EVENTS: &str = "inproc://hyprstream/events";

    /// Default inproc endpoint for registry service
    pub const REGISTRY: &str = "inproc://hyprstream/registry";

    /// Default inproc endpoint for inference service
    pub const INFERENCE: &str = "inproc://hyprstream/inference";

    /// Default IPC endpoint for sandbox processes
    pub const EVENTS_IPC: &str = "ipc:///tmp/hyprstream-events";

    /// Default TCP endpoint for remote subscribers
    pub const EVENTS_TCP: &str = "tcp://0.0.0.0:5555";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify the module exports work
        let ctx = global_context();
        assert!(ctx.socket(socket_types::PUB).is_ok());
    }
}
