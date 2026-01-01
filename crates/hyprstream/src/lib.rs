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
#[allow(dead_code)]
#[allow(unused_imports)]
pub mod events_capnp {
    include!(concat!(env!("OUT_DIR"), "/events_capnp.rs"));
}

#[allow(dead_code)]
#[allow(unused_imports)]
pub mod inference_capnp {
    include!(concat!(env!("OUT_DIR"), "/inference_capnp.rs"));
}

#[allow(dead_code)]
#[allow(unused_imports)]
pub mod registry_capnp {
    include!(concat!(env!("OUT_DIR"), "/registry_capnp.rs"));
}

#[allow(dead_code)]
#[allow(unused_imports)]
pub mod policy_capnp {
    include!(concat!(env!("OUT_DIR"), "/policy_capnp.rs"));
}

#[allow(dead_code)]
#[allow(unused_imports)]
pub mod model_capnp {
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
