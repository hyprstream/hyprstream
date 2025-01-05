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
pub mod cached;
pub mod duckdb;

use crate::metrics::MetricRecord;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use tonic::Status;

/// Trait defining the interface for metric storage backends.
///
/// This trait must be implemented by all storage backends to provide:
/// - Metric data persistence
/// - Query capabilities
/// - SQL statement preparation and execution
/// - Arrow RecordBatch conversion
///
/// Implementations should ensure efficient handling of time-series data
/// and support for windowed aggregations.
#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    /// Initializes the storage backend.
    ///
    /// This method should:
    /// - Set up any necessary database connections
    /// - Create required tables and schemas
    /// - Initialize caching mechanisms if applicable
    async fn init(&self) -> Result<(), Status>;

    /// Inserts a batch of metric records into storage.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Vector of MetricRecord instances to insert
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;

    /// Queries metrics from a given timestamp.
    ///
    /// This method should efficiently retrieve metrics using time-based filtering
    /// and apply any configured caching strategies.
    ///
    /// # Arguments
    ///
    /// * `from_timestamp` - Unix timestamp to query from
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;

    /// Prepares a SQL statement for execution.
    ///
    /// This method should:
    /// 1. Parse and validate the SQL query
    /// 2. Create an execution plan
    /// 3. Return a serialized handle for later execution
    ///
    /// # Arguments
    ///
    /// * `query` - SQL query string to prepare
    ///
    /// # Returns
    ///
    /// * `Result<Vec<u8>, Status>` - Serialized statement handle on success
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Executes a prepared SQL statement.
    ///
    /// This method should:
    /// 1. Deserialize the statement handle
    /// 2. Execute the prepared statement
    /// 3. Return results as MetricRecords
    ///
    /// The implementation should use zero-copy operations where possible
    /// to optimize performance.
    ///
    /// # Arguments
    ///
    /// * `statement_handle` - Serialized statement handle from prepare_sql
    ///
    /// # Returns
    ///
    /// * `Result<Vec<MetricRecord>, Status>` - Query results as MetricRecords
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;

    /// Converts an Arrow RecordBatch to MetricRecords.
    ///
    /// This is provided as a default implementation to ensure consistent
    /// conversion across all storage backends. It handles:
    /// - Type checking and conversion
    /// - Null value handling
    /// - Efficient batch processing
    ///
    /// # Arguments
    ///
    /// * `batch` - Arrow RecordBatch containing metric data
    ///
    /// # Returns
    ///
    /// * `Result<Vec<MetricRecord>, Status>` - Converted metric records
    fn record_batch_to_metrics(&self, batch: &RecordBatch) -> Result<Vec<MetricRecord>, Status> {
        let metric_ids = batch
            .column_by_name("metric_id")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Int64Array>())
            .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

        let sums = batch
            .column_by_name("value_running_window_sum")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

        let avgs = batch
            .column_by_name("value_running_window_avg")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

        let counts = batch
            .column_by_name("value_running_window_count")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Int64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

        let mut metrics = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            metrics.push(MetricRecord {
                metric_id: metric_ids.value(i).to_string(),
                timestamp: timestamps.value(i),
                value_running_window_sum: sums.value(i),
                value_running_window_avg: avgs.value(i),
                value_running_window_count: counts.value(i),
            });
        }

        Ok(metrics)
    }
}
