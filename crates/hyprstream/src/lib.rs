//! Adaptive ML inference server.
//!
//! This crate provides the core functionality for:
//! - Real-time weight updates for neural networks
//! - Dynamic adaptive ML inference
//! - Hardware-accelerated storage
//! - Memory-mapped disk persistence
//! - FlightSQL interface for embeddings and similarity search

#[cfg(not(feature = "credential-pds"))]
compile_error!(
    "security invariant: every Hyprstream build requires the `credential-pds` \
     feature so the production UserStore cannot fall back to plaintext storage"
);

#[cfg(all(feature = "metrics", feature = "pglite"))]
compile_error!(
    "features `metrics` and `pglite` are mutually exclusive: DuckDB and PGlite \
     export overlapping PostgreSQL C symbols; build the credential/PDS service \
     with `credential-pds`, and metrics/inference/Flight as a separate service"
);

// Re-export capnp modules from hyprstream-rpc (compiled once, shared by all crates)
pub use hyprstream_rpc::annotations_capnp;
pub use hyprstream_rpc::common_capnp;
pub use hyprstream_rpc::streaming_capnp;
pub use hyprstream_rpc::optional_capnp;
pub use hyprstream_rpc::nine_capnp;

// Cap'n Proto service modules — re-exported from hyprstream-rpc-std (MIT)
pub use hyprstream_rpc_std::service_events_capnp as events_capnp;
pub use hyprstream_rpc_std::inference_capnp;
pub use hyprstream_rpc_std::registry_capnp;
pub use hyprstream_rpc_std::policy_capnp;
pub use hyprstream_rpc_std::model_capnp;
pub use hyprstream_rpc_std::mcp_capnp;
pub use hyprstream_rpc_std::metrics_capnp;
pub use hyprstream_rpc_std::oauth_capnp;

// TUI-specific Cap'n Proto modules (remain in hyprstream)
pub mod tui_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/tui_capnp.rs"));
}

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
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    RuntimeConfig,
    RuntimeEngine,
    TorchEngine,
};

// Export init function from runtime
pub use runtime::create_engine as init;

// Event types exports
pub use events::{EventEnvelope, EventPayload, EventSource};
