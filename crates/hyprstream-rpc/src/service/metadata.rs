//! Re-export metadata types from the cross-platform `crate::metadata` module.
//!
//! Kept for backward compatibility — existing code that does
//! `use hyprstream_rpc::service::metadata::*` continues to work.

pub use crate::_metadata::*;
