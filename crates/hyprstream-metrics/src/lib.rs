//! Core library for high-performance metric storage and analysis.
//!
//! This crate provides the core functionality for:
//! - Efficient metric storage and retrieval
//! - Real-time aggregation and analysis
//! - Model management and versioning
//! - Git-based checkpoint and recovery

pub mod aggregation;
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod metrics;
pub mod models;
pub mod query;
pub mod storage;
pub mod utils;

pub use aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
pub use checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata, RecoveryManager,
    RecoveryStatus,
};
pub use query::{
    CachedStatement, DataFusionExecutor, DataFusionPlanner, ExecutorConfig, Query,
    QueryOrchestrator,
};
pub use storage::StorageBackend;
pub use storage::context::{ContextRecord, ContextStore, SearchResult, context_schema, DEFAULT_EMBEDDING_DIM};

// Re-export Arrow types from DuckDB for consistency
pub use duckdb::arrow;
