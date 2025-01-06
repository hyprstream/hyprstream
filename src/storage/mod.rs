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

use crate::config::Credentials;
use crate::metrics::MetricRecord;
use arrow_array::RecordBatch;
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
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Creates a new instance of the storage backend.
    ///
    /// # Arguments
    ///
    /// * `connection_string` - The connection string for the database
    /// * `options` - Additional options for configuring the connection
    /// * `credentials` - Optional credentials for authentication
    ///
    /// # Returns
    ///
    /// * `Result<Self, Status>` - Configured backend or error
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized;

    /// Initializes the storage backend.
    ///
    /// This method should:
    /// 1. Create necessary tables and indexes
    /// 2. Set up connection pools if needed
    /// 3. Verify connectivity and permissions
    async fn init(&self) -> Result<(), Status>;

    /// Inserts a batch of metrics into storage.
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;

    /// Queries metrics from a given timestamp.
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;

    /// Prepares a SQL statement for execution.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Executes a prepared SQL statement.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;
}
