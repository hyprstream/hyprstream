//! Cap'n Proto generated schemas
//!
//! Canonical generated Cap'n Proto Rust code for the public RPC contracts lives
//! in `hyprstream-rpc` and `hyprstream-rpc-std`; this module contains only
//! application-specific stream payloads and compatibility-free helpers:
//! - Events (pub/sub messaging)
//! - Inference (generation requests/responses)
//! - Registry (git repository management)
//!
//! The schemas are compiled from `.capnp` files in the `schema/` directory
//! by the build.rs script using capnpc.
//!
//! Consumers should import the canonical modules from `hyprstream_rpc_std`.
