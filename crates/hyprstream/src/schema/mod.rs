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
//! Note: The generated modules are included at crate root level due to
//! capnpc path resolution. This module provides re-exports for convenience.

// Re-export the crate-level modules
pub use crate::events_capnp;
pub use crate::inference_capnp;
pub use crate::registry_capnp;
