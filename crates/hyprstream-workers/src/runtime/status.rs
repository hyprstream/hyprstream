//! Runtime-local status helpers.
//!
//! The reusable CRI client and all wire types live in
//! `hyprstream_rpc_std::worker_client`; this module only retains the local
//! composite status wrapper used by the service implementation.

use std::collections::HashMap; // Only for StatusResponse.info (local-only, not serialized)

/// Runtime status response (local composite, not in schema).
///
/// Bundles the generated RuntimeStatus with verbose diagnostic info.
/// The `info` HashMap is NOT serialized over the wire — it's only used locally.
#[derive(Debug, Clone)]
pub struct StatusResponse {
    /// Overall runtime status (generated wire type)
    pub status: hyprstream_rpc_std::worker_client::RuntimeStatus,
    /// Additional info (if verbose) — local only, not serialized
    pub info: HashMap<String, String>,
}
