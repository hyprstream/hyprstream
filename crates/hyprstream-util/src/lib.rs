// Rustdoc quality lane pilot (docs/rustdoc-quality.md): this crate opted in
// to `missing_docs` for its public API. The lint is scoped here rather than in
// `[workspace.lints]` so each crate measures its own baseline and opts in
// deliberately.
//
// The `not(clippy)` guard isolates the lint from Clippy: clippy-driver runs
// rustc's lint passes, so a plain `warn(missing_docs)` would be promoted to a
// hard error by the repository-wide `cargo clippy … -- -D warnings` (the PR
// Clippy job and the pre-commit hook) as soon as any public item is
// undocumented — turning the lane's "warnings are non-blocking evidence"
// contract false as it expands. Under clippy the attribute is inert; under
// plain rustc/rustdoc it stays active, so `cargo doc` emits the warnings and
// the scoped strict-evidence run (`RUSTDOCFLAGS="-D warnings"`) still
// escalates them. See docs/rustdoc-quality.md for the warning policy and the
// Clippy/rustdoc isolation probe results.
#![cfg_attr(not(clippy), warn(missing_docs))]

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
