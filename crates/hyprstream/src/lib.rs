//! Adaptive ML inference server.
//!
//! This crate provides the core functionality for:
//! - Real-time weight updates for neural networks
//! - Dynamic adaptive ML inference
//! - Hardware-accelerated storage
//! - Memory-mapped disk persistence
//! - FlightSQL interface for embeddings and similarity search

#[cfg(not(feature = "encrypted-account-admission"))]
compile_error!(
    "security invariant: every Hyprstream build requires the `encrypted-account-admission` \
     policy so the production UserStore cannot fall back to plaintext storage"
);

#[cfg(all(feature = "metrics", feature = "pglite"))]
compile_error!(
    "features `metrics` and `pglite` are mutually exclusive: DuckDB and PGlite \
     export overlapping PostgreSQL C symbols; build the credential/PDS service \
     with `credential-pds`, and the Metrics/inference/Flight service with `metrics`"
);

// Re-export capnp modules from hyprstream-rpc (compiled once, shared by all crates)
pub use hyprstream_rpc::annotations_capnp;
pub use hyprstream_rpc::common_capnp;
pub use hyprstream_rpc::streaming_capnp;
pub use hyprstream_rpc::optional_capnp;
pub use hyprstream_rpc::nine_capnp;

pub mod api;
pub mod archetypes;
pub mod auth;
pub mod account;
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod events;
pub mod git;
pub mod inference;
pub mod mac;
pub mod runtime;
pub mod schema;
pub mod server;
pub mod services;
pub mod storage;
#[cfg(unix)]
pub mod systemd;
pub mod training;
pub mod tui;

// Storage exports removed
pub use runtime::{
    FinishReason,
    GenerationResult,
    RuntimeConfig,
    RuntimeEngine,
    TorchEngine,
};

// Export init function from runtime
pub use runtime::create_engine as init;

// Event types exports
pub use events::{EventEnvelope, EventPayload, EventSource};
