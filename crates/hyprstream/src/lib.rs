//! Adaptive ML inference server.
//!
//! This crate provides the core functionality for:
//! - Real-time weight updates for neural networks
//! - Dynamic adaptive ML inference
//! - Hardware-accelerated storage
//! - Memory-mapped disk persistence
//! - FlightSQL interface for embeddings and similarity search

// Cap'n Proto generated modules (must be at crate root for path resolution)
// Note: common_capnp is in hyprstream-rpc crate (envelope types)
pub mod events_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/events_capnp.rs"));
}

pub mod inference_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/inference_capnp.rs"));
}

pub mod registry_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/registry_capnp.rs"));
}

pub mod policy_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/policy_capnp.rs"));
}

pub mod model_capnp {
    #![allow(dead_code, unused_imports)]
    #![allow(clippy::all, clippy::unwrap_used, clippy::expect_used, clippy::match_same_arms)]
    #![allow(clippy::semicolon_if_nothing_returned, clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/model_capnp.rs"));
}

pub mod adapters;
pub mod api;
pub mod archetypes;
pub mod auth;
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod events;
pub mod git;
pub mod inference;
pub mod lora;
pub mod runtime;
pub mod schema;
pub mod server;
pub mod services;
pub mod storage;
#[cfg(unix)]
pub mod systemd;
pub mod training;
pub mod zmq;

// Storage exports removed
pub use runtime::{
    AdaptationMode,
    // REMOVED: AdaptationTrigger, AdaptationType, ConversationContext, ConversationResponse,
    // ConversationRouter, ConversationSession, ConversationTurn, ModelPool, ModelState,
    // PoolStats, RoutingConfig - dead code from conversation_router
    AdapterMetrics,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    RuntimeConfig,
    RuntimeEngine,
    TorchEngine,
    UserFeedback,
    XLoRAAdapter,
    XLoRARoutingStrategy,
};

// Export TorchEngine as HyprStreamEngine for backward compatibility
pub use runtime::TorchEngine as HyprStreamEngine;

// Export init function from runtime
pub use runtime::create_engine as init;

// Event types exports
pub use events::{EventEnvelope, EventPayload, EventSource};
