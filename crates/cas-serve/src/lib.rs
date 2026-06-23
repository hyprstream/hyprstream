//! cas-serve library
//!
//! Provides the protocol types, CDC chunker, and reconstruction shard types
//! for cas-serve communication. Used by both the cas-serve binary and
//! SshClient / remote-control implementations.

pub mod chunker;
pub mod mdb_shard;
pub mod protocol;
pub mod shard;

pub use protocol::{ErrorCode, Request, Response};
