//! StreamService re-exports
//!
//! The `start_stream_service` helper has been removed in favor of the unified
//! `ServiceManager::spawn()` API. See main.rs for usage examples.

pub use hyprstream_rpc::service::StreamService;
