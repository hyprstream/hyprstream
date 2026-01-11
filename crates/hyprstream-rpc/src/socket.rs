//! Socket activation helpers
//!
//! Wraps systemd socket activation for receiving file descriptors
//! passed by systemd when a socket is activated.

use std::os::unix::io::RawFd;

/// Check if running under systemd socket activation
///
/// Returns true if:
/// 1. System was booted with systemd
/// 2. `LISTEN_FDS` environment variable is set
#[cfg(feature = "systemd")]
pub fn is_socket_activated() -> bool {
    systemd::daemon::booted().unwrap_or(false) && std::env::var("LISTEN_FDS").is_ok()
}

#[cfg(not(feature = "systemd"))]
pub fn is_socket_activated() -> bool {
    false
}

/// Get socket activation file descriptors
///
/// Returns a vector of file descriptors passed by systemd.
/// The first fd is always 3 (SD_LISTEN_FDS_START), subsequent ones
/// are 4, 5, etc.
///
/// Returns `None` if not running under socket activation.
#[cfg(feature = "systemd")]
pub fn listen_fds() -> Option<Vec<RawFd>> {
    systemd::daemon::listen_fds(false)
        .ok()
        .map(|fds| fds.iter().collect())
}

#[cfg(not(feature = "systemd"))]
pub fn listen_fds() -> Option<Vec<RawFd>> {
    None
}

/// Get the number of listening file descriptors
///
/// Returns 0 if not running under socket activation.
#[cfg(feature = "systemd")]
pub fn listen_fds_count() -> usize {
    std::env::var("LISTEN_FDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[cfg(not(feature = "systemd"))]
pub fn listen_fds_count() -> usize {
    0
}

/// Check if a specific fd name matches
///
/// When multiple sockets are passed, `LISTEN_FDNAMES` contains
/// colon-separated names. This checks if a given fd index matches
/// the expected name.
#[cfg(feature = "systemd")]
pub fn fd_name_matches(fd_index: usize, expected_name: &str) -> bool {
    if let Ok(names) = std::env::var("LISTEN_FDNAMES") {
        if let Some(name) = names.split(':').nth(fd_index) {
            return name == expected_name;
        }
    }
    false
}

#[cfg(not(feature = "systemd"))]
pub fn fd_name_matches(_fd_index: usize, _expected_name: &str) -> bool {
    false
}
