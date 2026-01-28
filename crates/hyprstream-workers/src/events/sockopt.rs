//! Raw ZMQ socket option helpers
//!
//! Provides access to ZMQ socket options not exposed by the high-level `zmq` crate:
//! - `ZMQ_USE_FD` - Systemd socket activation (re-exported from hyprstream-rpc)
//! - `ZMQ_RCVTIMEO` - Receive timeout (milliseconds)
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::sockopt::set_use_fd;
//!
//! // Get file descriptor from systemd (LISTEN_FDS)
//! let fd: RawFd = 3;  // FDs start at 3 (SD_LISTEN_FDS_START)
//!
//! // Attach to ZMQ socket - socket is now bound, do NOT call bind()
//! set_use_fd(&socket, fd)?;
//!
//! // IMPORTANT: Skip bind() and proceed directly to use
//! ```

use std::ffi::c_int;

// Re-export set_use_fd and supports_use_fd from hyprstream-rpc
pub use hyprstream_rpc::transport::sockopt::{set_use_fd, supports_use_fd};

/// ZMQ_RCVTIMEO socket option
///
/// Sets the timeout for receive operations in milliseconds.
/// - `-1` = infinite (default, blocks forever)
/// - `0` = immediate return (non-blocking)
/// - `> 0` = wait up to N milliseconds, then return EAGAIN
///
/// Value: 27 (defined in zmq.h)
const ZMQ_RCVTIMEO: c_int = 27;

/// Set ZMQ_RCVTIMEO on a socket to configure receive timeout
///
/// This sets the timeout for receive operations. When a receive operation
/// times out, it returns `EAGAIN` error.
///
/// # Arguments
///
/// * `socket` - ZMQ socket to configure
/// * `timeout_ms` - Timeout in milliseconds:
///   - `-1` = infinite (default, blocks forever)
///   - `0` = immediate return (non-blocking)
///   - `> 0` = wait up to N milliseconds
///
/// # Example
///
/// ```ignore
/// let mut socket = ctx.socket(zmq::SUB)?;
/// set_rcvtimeo(&mut socket, 30_000)?;  // 30 second timeout
/// ```
///
/// # Errors
///
/// Returns an error if the socket option cannot be set.
pub fn set_rcvtimeo(socket: &mut zmq::Socket, timeout_ms: i32) -> zmq::Result<()> {
    let size = std::mem::size_of::<i32>();
    let rc = unsafe {
        zmq_sys::zmq_setsockopt(
            socket.as_mut_ptr(),
            ZMQ_RCVTIMEO,
            &timeout_ms as *const i32 as *const std::ffi::c_void,
            size,
        )
    };
    if rc == -1 {
        Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }))
    } else {
        Ok(())
    }
}

/// Get the current libzmq version as a string
pub fn zmq_version_string() -> String {
    let (major, minor, patch) = zmq::version();
    format!("{major}.{minor}.{patch}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_use_fd() {
        let supported = supports_use_fd();
        let version = zmq_version_string();
        println!("libzmq version: {version}, ZMQ_USE_FD supported: {supported}");
    }

    #[test]
    fn test_zmq_rcvtimeo_constant() {
        assert_eq!(ZMQ_RCVTIMEO, 27);
    }

    #[test]
    fn test_set_rcvtimeo() {
        let ctx = zmq::Context::new();
        let mut socket = ctx.socket(zmq::SUB).unwrap();
        set_rcvtimeo(&mut socket, 1000).unwrap();
        set_rcvtimeo(&mut socket, -1).unwrap();
        set_rcvtimeo(&mut socket, 0).unwrap();
    }
}
