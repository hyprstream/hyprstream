//! Service notification helpers (re-exported from hyprstream-rpc)
//!
//! Wraps systemd sd_notify protocol for service lifecycle notifications.
//! When the `systemd` feature is disabled, all functions are no-ops.

pub use hyprstream_rpc::notify::*;
