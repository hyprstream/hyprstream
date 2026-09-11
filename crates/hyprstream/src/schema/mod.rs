//! Cap'n Proto generated schemas
//!
//! This module re-exports the generated Cap'n Proto Rust code for:
//! - Events (pub/sub messaging)
//! - Inference (generation requests/responses)
//! - Registry (git repository management)
//!
//! The schemas are compiled from `.capnp` files in the `schema/` directory
//! by the build.rs script using capnpc.
//!
//! Consumers should import the canonical modules from `hyprstream_rpc_std`.
