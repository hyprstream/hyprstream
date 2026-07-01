//! Shared generic utility primitives for hyprstream.
//!
//! Home for small, dependency-light, domain-agnostic data structures used
//! across crates. Kept a leaf crate so any hyprstream crate can depend on it
//! without pulling domain weight.
//!
//! Currently exports:
//! * [`TtlCache`] — generic per-entry-TTL cache with lazy version-tagged
//!   eviction and a capacity bound.

pub mod ttl_cache;

pub use ttl_cache::TtlCache;
