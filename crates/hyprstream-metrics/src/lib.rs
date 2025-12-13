//! Core library for high-performance metric storage and analysis.
//!
//! This crate provides the core functionality for:
//! - Efficient metric storage and retrieval
//! - Real-time aggregation and analysis
//! - Model management and versioning

pub mod aggregation;
pub mod config;
pub mod error;
pub mod metrics;
pub mod models;
pub mod query;
pub mod storage;
pub mod utils;

pub use aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
pub use query::{DataFusionExecutor, DataFusionPlanner, ExecutorConfig, Query};
pub use storage::StorageBackend;

// Re-export Arrow types from DuckDB for consistency
pub use duckdb::arrow;
