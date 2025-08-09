//! VDB-first adaptive ML inference server with dynamic sparse weight adjustments.
//!
//! This crate provides the core functionality for:
//! - Real-time sparse weight updates for 99% sparse neural networks
//! - Dynamic adaptive ML inference with streaming weight adjustments
//! - Hardware-accelerated VDB storage with neural compression
//! - Memory-mapped disk persistence with zero-copy operations
//! - FlightSQL interface for embeddings and similarity search

pub mod aggregation;
pub mod adapters;
pub mod api;
pub mod cli;
pub mod config;
pub mod error;
pub mod inference;
pub mod metrics;
pub mod models;
pub mod query;
pub mod runtime;
pub mod service;
pub mod storage;
pub mod utils;

pub use query::{
    DataFusionExecutor, DataFusionPlanner, ExecutorConfig, OptimizationHint, Query, QueryExecutor,
    QueryPlanner,
};
pub use service::{EmbeddingFlightService, MetricFlightSqlService};
pub use storage::{
    VDBSparseStorage, SparseStorageConfig, SparseStorage,
    SparseWeightUpdate, EmbeddingMatch, SparseStorageError
};
pub use runtime::{
    RuntimeEngine, LlamaCppEngine, LoRAEngineWrapper, SparseLoRAAdapter, LoRAConfig,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason, RuntimeConfig
};
