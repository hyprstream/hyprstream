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
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod git;
pub mod inference;
pub mod lora;
pub mod runtime;
pub mod training;
pub mod server;
pub mod storage;

// Storage exports removed
pub use runtime::{
    RuntimeEngine, TorchEngine, LoRAEngineWrapper,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason, RuntimeConfig,
    XLoRAAdapter, AdaptationMode, UserFeedback, XLoRARoutingStrategy, AdapterMetrics,
    // Model Evolution System exports
    ConversationRouter, ConversationSession, ConversationTurn, ConversationResponse,
    ConversationContext, ModelPool, AdaptationType, AdaptationTrigger, ModelState,
    PoolStats, RoutingConfig
};

// Export TorchEngine as HyprStreamEngine for backward compatibility
pub use runtime::TorchEngine as HyprStreamEngine;

// Export init function from runtime 
pub use runtime::create_engine as init;
