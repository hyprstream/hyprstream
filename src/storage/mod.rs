//! Storage backends for metric data persistence and caching.
//!
//! This module provides multiple storage backend implementations:
//! - `duckdb`: High-performance embedded database for caching and local storage
//! - `adbc`: Arrow Database Connectivity for external database integration
//! - `cached`: Two-tier storage with configurable caching layer
//!
//! Each backend implements the `StorageBackend` trait, providing a consistent
//! interface for metric storage and retrieval operations.

pub mod adbc;
pub mod duckdb;
pub mod cache;

use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::metrics::aggregation::{AggregateFunction, GroupBy, AggregateResult};
use async_trait::async_trait;
use std::collections::HashMap;
use tonic::Status;

/// Storage backend trait for metric data persistence.
///
/// This trait defines the interface that all storage backends must implement.
/// It provides methods for:
/// - Initialization and configuration
/// - Metric data insertion
/// - Metric data querying
/// - SQL query preparation and execution
/// - Aggregation of metrics
#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    /// Initialize the storage backend.
    async fn init(&self) -> Result<(), Status>;

    /// Insert metrics into storage.
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;

    /// Query metrics from storage.
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;

    /// Prepare a SQL query and return a handle.
    /// The handle is backend-specific and opaque to the caller.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Execute a prepared SQL query using its handle.
    /// The handle must have been obtained from prepare_sql.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;

    /// Aggregate metrics using the specified function and grouping.
    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status>;

    /// Create a new instance with the given options.
    /// The connection string and options are backend-specific.
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized;
}
