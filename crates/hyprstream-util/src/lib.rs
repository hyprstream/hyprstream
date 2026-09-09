// Rustdoc quality lane pilot (docs/rustdoc-quality.md): this crate opted in
// to `missing_docs` for its public API. The lint is scoped here rather than in
// `[workspace.lints]` so each crate measures its own baseline and opts in
// deliberately. Warnings are non-blocking; see docs/rustdoc-quality.md.
#![warn(missing_docs)]

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

pub use ttl_cache::{InsertIfAbsentNoEvictResult, TtlCache};
