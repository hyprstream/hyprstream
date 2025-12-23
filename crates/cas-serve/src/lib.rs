//! cas-serve library
//!
//! Provides the protocol types for cas-serve communication.
//! Used by both the cas-serve binary and SshClient implementations.

pub mod protocol;

pub use protocol::{ErrorCode, Request, Response};
