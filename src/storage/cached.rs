//! Two-tier cached storage backend implementation.
//!
//! This module provides a caching layer on top of any storage backend:
//! - Fast access to frequently used data
//! - Write-through caching for consistency
//! - Configurable cache duration
//! - Support for any StorageBackend as cache or store
//!
//! The implementation follows standard caching patterns while ensuring
//! data consistency between cache and backing store.

use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use std::sync::Arc;
use tonic::Status;

/// Two-tier storage backend with caching support.
///
/// This backend provides:
/// - Fast access to recent data through caching
/// - Write-through caching for data consistency
/// - Configurable cache duration
/// - Support for any StorageBackend implementation
///
/// The implementation uses two storage backends:
/// 1. A fast cache (e.g., in-memory DuckDB)
/// 2. A persistent store (e.g., PostgreSQL via ADBC)
pub struct CachedStorageBackend {
    /// Fast storage backend for caching
    cache: Arc<dyn StorageBackend>,
    /// Persistent storage backend for data
    store: Arc<dyn StorageBackend>,
    /// Cache entry lifetime in seconds
    cache_duration: i64,
}

impl CachedStorageBackend {
    /// Creates a new cached storage backend.
    ///
    /// This method sets up a two-tier storage system with:
    /// - A fast cache layer for frequent access
    /// - A persistent backing store
    /// - Configurable cache duration
    ///
    /// # Arguments
    ///
    /// * `cache` - Fast storage backend for caching
    /// * `store` - Persistent storage backend
    /// * `cache_duration` - Cache entry lifetime in seconds
    pub fn new(
        cache: Arc<dyn StorageBackend>,
        store: Arc<dyn StorageBackend>,
        cache_duration: i64,
    ) -> Self {
        Self {
            cache,
            store,
            cache_duration,
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for CachedStorageBackend {
    /// Initializes both cache and backing store.
    ///
    /// This method ensures both storage layers are properly
    /// initialized and ready for use.
    async fn init(&self) -> Result<(), Status> {
        // Initialize both cache and backing store
        self.cache.init().await?;
        self.store.init().await?;
        Ok(())
    }

    /// Inserts metrics into both cache and backing store.
    ///
    /// This method implements write-through caching:
    /// 1. Writes to cache for fast access
    /// 2. Writes to backing store for persistence
    ///
    /// # Arguments
    ///
    /// * `metrics` - Vector of MetricRecord instances to insert
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Insert into both cache and backing store
        self.cache.insert_metrics(metrics.clone()).await?;
        self.store.insert_metrics(metrics).await?;
        Ok(())
    }

    /// Queries metrics with caching support.
    ///
    /// This method implements a cache-first query strategy:
    /// 1. Attempts to read from cache
    /// 2. On cache miss, reads from backing store
    /// 3. Updates cache with results from backing store
    ///
    /// # Arguments
    ///
    /// * `from_timestamp` - Unix timestamp to query from
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Try cache first
        match self.cache.query_metrics(from_timestamp).await {
            Ok(metrics) if !metrics.is_empty() => Ok(metrics),
            _ => {
                // Cache miss or error, query backing store
                let metrics = self.store.query_metrics(from_timestamp).await?;
                // Update cache with results
                if !metrics.is_empty() {
                    self.cache.insert_metrics(metrics.clone()).await?;
                }
                Ok(metrics)
            }
        }
    }

    /// Prepares a SQL statement on the backing store.
    ///
    /// This method bypasses the cache and prepares statements
    /// directly on the backing store, as prepared statements
    /// are typically used for complex queries.
    ///
    /// # Arguments
    ///
    /// * `query` - SQL query to prepare
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // Prepare on backing store only
        self.store.prepare_sql(query).await
    }

    /// Executes a prepared SQL statement on the backing store.
    ///
    /// This method bypasses the cache and executes statements
    /// directly on the backing store, ensuring consistent results
    /// for complex queries.
    ///
    /// # Arguments
    ///
    /// * `statement_handle` - Handle of the prepared statement
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        // Execute on backing store only
        self.store.query_sql(statement_handle).await
    }
}
