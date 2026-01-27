//! Raw ZMQ socket option helpers
//!
//! Provides access to ZMQ socket options not exposed by the high-level `zmq` crate:
//! - `ZMQ_USE_FD` - Systemd socket activation (attach pre-bound file descriptor)
//!
//! # Why raw pointer access?
//!
//! The `zmq` crate (rust-zmq) doesn't expose `ZMQ_USE_FD` in its high-level API.
//! We access it directly via `zmq_sys` using raw pointers.

use std::ffi::c_int;
use std::os::unix::io::RawFd;

/// ZMQ_USE_FD socket option (added in libzmq 4.3.4)
///
/// Allows attaching a pre-bound file descriptor to a ZMQ socket.
/// After setting this option, the socket is already bound - do NOT call bind().
const ZMQ_USE_FD: c_int = 89;

/// Set ZMQ_USE_FD on a socket to use a pre-existing file descriptor
///
/// This enables systemd socket activation for ZMQ sockets. After calling this,
/// the socket is considered bound and ready for use.
///
/// # Arguments
///
/// * `socket` - ZMQ socket to configure
/// * `fd` - Pre-bound file descriptor from systemd
///
/// # Important
///
/// After calling this, do NOT call `bind()` - the socket is already bound by systemd.
///
/// # Errors
///
/// Returns an error if:
/// - libzmq version < 4.3.4 (ZMQ_USE_FD not supported)
/// - The file descriptor is invalid
/// - The socket type doesn't support this option
pub fn set_use_fd(socket: &mut zmq::Socket, fd: RawFd) -> zmq::Result<()> {
    let size = std::mem::size_of::<RawFd>();
    let rc = unsafe {
        zmq_sys::zmq_setsockopt(
            socket.as_mut_ptr(),
            ZMQ_USE_FD,
            &fd as *const RawFd as *const std::ffi::c_void,
            size,
        )
    };
    if rc == -1 {
        Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }))
    } else {
        Ok(())
    }
}

/// Check if libzmq version supports ZMQ_USE_FD (requires 4.3.4+)
pub fn supports_use_fd() -> bool {
    let (major, minor, patch) = zmq::version();
    (major, minor, patch) >= (4, 3, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_use_fd() {
        let supported = supports_use_fd();
        let (major, minor, patch) = zmq::version();
        println!(
            "libzmq version: {major}.{minor}.{patch}, ZMQ_USE_FD supported: {supported}"
        );
    }

    #[test]
    fn test_zmq_use_fd_constant() {
        assert_eq!(ZMQ_USE_FD, 89);
    }
}
