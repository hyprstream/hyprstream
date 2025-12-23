//! Adaptive ML inference server.
//!
//! This crate provides the core functionality for:
//! - Real-time weight updates for neural networks
//! - Dynamic adaptive ML inference
//! - Hardware-accelerated storage
//! - Memory-mapped disk persistence
//! - FlightSQL interface for embeddings and similarity search

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
pub mod server;
pub mod services;
pub mod storage;
pub mod training;
pub mod zmq;

// Storage exports removed
pub use runtime::{
    AdaptationMode,
    AdaptationTrigger,
    AdaptationType,
    AdapterMetrics,
    ConversationContext,
    ConversationResponse,
    // Model Evolution System exports
    ConversationRouter,
    ConversationSession,
    ConversationTurn,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    ModelPool,
    ModelState,
    PoolStats,
    RoutingConfig,
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

// Event bus exports
pub use events::{EventBus, EventEnvelope, EventPayload, EventSource, SinkRegistry};
