//! Core library for high-performance metric storage and analysis.
//!
//! This crate provides the core functionality for:
//! - Efficient metric storage and retrieval
//! - Real-time aggregation and analysis
//! - Flexible query execution
//! - Model management and versioning

pub mod aggregation;
pub mod error;
pub mod cli;
pub mod metrics;
pub mod models;
pub mod query;
pub mod service;
pub mod storage;
pub mod config;
pub mod utils;

pub use query::{
    DataFusionExecutor, DataFusionPlanner, ExecutorConfig, OptimizationHint, Query, QueryExecutor,
    QueryPlanner,
};
pub use service::FlightSqlServer;
pub use storage::{StorageBackend, StorageBackendType};
