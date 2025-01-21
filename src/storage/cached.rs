//! Two-tier cached storage backend implementation.
//!
//! This module provides a caching layer on top of any storage backend:
//! - Fast access to frequently used data
//! - Write-through caching for consistency
//! - Configurable cache duration
//! - Support for any StorageBackend as cache or store
//!
//! # Configuration
//!
//! The cached backend can be configured using the following options:
//!
//! ```toml
//! # Primary storage configuration
//! [engine]
//! engine = "adbc"
//! connection = "postgresql://localhost:5432"
//! options = {
//!     driver_path = "/usr/local/lib/libadbc_driver_postgresql.so",
//!     username = "postgres",
//!     database = "metrics"
//! }
//!
//! # Cache configuration
//! [cache]
//! enabled = true
//! engine = "duckdb"
//! connection = ":memory:"
//! max_duration_secs = 3600
//! options = {
//!     threads = "2"
//! }
//! ```
//!
//! Or via command line:
//!
//! ```bash
//! hyprstream \
//!   --engine adbc \
//!   --engine-connection "postgresql://localhost:5432" \
//!   --engine-options driver_path=/usr/local/lib/libadbc_driver_postgresql.so \
//!   --engine-options username=postgres \
//!   --enable-cache \
//!   --cache-engine duckdb \
//!   --cache-connection ":memory:" \
//!   --cache-options threads=2 \
//!   --cache-max-duration 3600
//! ```
//!
//! The implementation follows standard caching patterns while ensuring
//! data consistency between cache and backing store.

use crate::config::Credentials;
use crate::storage::view::ViewDefinition;
use crate::storage::{StorageBackend, adbc::AdbcBackend, duckdb::DuckDbBackend};
use std::sync::Arc;
use std::collections::HashMap;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
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
    /// Maximum cache entry lifetime in seconds
    max_duration_secs: u64,
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
    /// * `max_duration_secs` - Maximum cache entry lifetime in seconds
    pub fn new(
        cache: Arc<dyn StorageBackend>,
        store: Arc<dyn StorageBackend>,
        max_duration_secs: u64,
    ) -> Self {
        Self {
            cache,
            store,
            max_duration_secs,
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
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
        // Execute on backing store only
        self.store.query_sql(statement_handle).await
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        // Parse cache duration from options
        let max_duration_secs = options
            .get("max_duration_secs")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);

        // Create cache backend
        let default_engine = "duckdb".to_string();
        let default_connection = ":memory:".to_string();
        let cache_engine = options.get("cache_engine").unwrap_or(&default_engine);
        let cache_connection = options.get("cache_connection").unwrap_or(&default_connection);
        let cache_options: HashMap<String, String> = options
            .iter()
            .filter(|(k, _)| k.starts_with("cache_"))
            .map(|(k, v)| (k[6..].to_string(), v.clone()))
            .collect();

        let cache: Arc<dyn StorageBackend> = match cache_engine.as_str() {
            "duckdb" => Arc::new(DuckDbBackend::new_with_options(
                cache_connection,
                &cache_options,
                None,
            )?),
            "adbc" => Arc::new(AdbcBackend::new_with_options(
                cache_connection,
                &cache_options,
                None,
            )?),
            _ => return Err(Status::invalid_argument("Invalid cache engine type")),
        };

        // Create store backend
        let store = Arc::new(AdbcBackend::new_with_options(
            connection_string,
            options,
            credentials,
        )?);

        Ok(Self::new(cache, store, max_duration_secs))
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        // Create table in both cache and store
        self.cache.create_table(table_name, schema).await?;
        self.store.create_table(table_name, schema).await?;
        Ok(())
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        // Insert into both cache and store
        self.cache.insert_into_table(table_name, batch.clone()).await?;
        self.store.insert_into_table(table_name, batch).await?;
        Ok(())
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        // Create view in both cache and store
        self.cache.create_view(name, definition.clone()).await?;
        self.store.create_view(name, definition).await?;
        Ok(())
    }

    async fn get_view(&self, name: &str) -> Result<crate::storage::view::ViewMetadata, Status> {
        // Try cache first
        match self.cache.get_view(name).await {
            Ok(view) => Ok(view),
            Err(_) => {
                // On cache miss, get from store and update cache
                let view = self.store.get_view(name).await?;
                self.cache.create_view(name, view.definition.clone()).await?;
                Ok(view)
            }
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        // List views from store (source of truth)
        self.store.list_views().await
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        // Drop from both cache and store
        self.cache.drop_view(name).await?;
        self.store.drop_view(name).await?;
        Ok(())
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        // Drop from both cache and store
        self.cache.drop_table(table_name).await?;
        self.store.drop_table(table_name).await?;
        Ok(())
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        // List tables from store (source of truth)
        self.store.list_tables().await
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        // Get schema from store (source of truth)
        self.store.get_table_schema(table_name).await
    }
}
