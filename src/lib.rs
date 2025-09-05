//! VDB-first adaptive ML inference server with dynamic sparse weight adjustments.
//!
//! This crate provides the core functionality for:
//! - Real-time sparse weight updates for 99% sparse neural networks
//! - Dynamic adaptive ML inference with streaming weight adjustments
//! - Hardware-accelerated VDB storage with neural compression
//! - Memory-mapped disk persistence with zero-copy operations
//! - FlightSQL interface for embeddings and similarity search

pub mod adapters;
pub mod api;
pub mod auth;
pub mod cli;
pub mod config;
pub mod constants;
pub mod error;
pub mod inference;
pub mod runtime;
pub mod server;
pub mod storage;
pub mod utils;

pub use storage::{
    VDBSparseStorage, SparseStorageConfig, SparseStorage,
    SparseWeightUpdate, EmbeddingMatch, SparseStorageError
};
pub use runtime::{
    RuntimeEngine, TorchEngine, LoRAEngineWrapper, SparseLoRAAdapter, LoRAConfig,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason, RuntimeConfig,
    XLoRAAdapter, AdaptationMode, UserFeedback, XLoRARoutingStrategy, AdapterMetrics,
    // Model Evolution System exports
    ConversationRouter, ConversationSession, ConversationTurn, ConversationResponse,
    ConversationContext, ModelPool, AdaptationType, AdaptationTrigger, ModelState,
    PoolStats, RoutingConfig, create_conversation_router
};

// Export TorchEngine as HyprStreamEngine for backward compatibility
pub use runtime::TorchEngine as HyprStreamEngine;

// Export init function from runtime 
pub use runtime::create_engine as init;
