 use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::{HyprStorageBackend, adbc::AdbcBackend, duckdb::DuckDbBackend};
use std::sync::Arc;
use std::collections::HashMap;
use tonic::Status;
use arrow_array::ArrayRef;

/// Two-tier storage backend with caching support.
///
/// This backend provides:
/// - Fast access to recent data through caching
/// - Write-through caching for data consistency
/// - Configurable cache duration
/// - Support for any HyprStorageBackend implementation
///
/// The implementation uses two storage backends:
/// 1. A fast cache (e.g., in-memory DuckDB)
/// 2. A persistent store (e.g., PostgreSQL via ADBC)
pub struct CachedStorageBackend {
    /// Fast storage backend for caching
    cache: Arc<dyn HyprStorageBackend>,
    /// Persistent storage backend for data
    store: Arc<dyn HyprStorageBackend>,
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
        cache: Arc<dyn HyprStorageBackend>,
        store: Arc<dyn HyprStorageBackend>,
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
impl HyprStorageBackend for CachedStorageBackend {
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

    /// Insert metrics into both cache and backing store.
    ///
    /// This method implements write-through caching by:
    /// 1. Writing to the backing store first
    /// 2. Writing to the cache on success
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Write to backing store first
        self.store.insert_metrics(metrics.clone()).await?;
        // Write to cache on success
        self.cache.insert_metrics(metrics).await?;
        Ok(())
    }

    /// Query metrics from cache with fallback to backing store.
    ///
    /// This method:
    /// 1. Attempts to read from cache first
    /// 2. Falls back to backing store on cache miss
    /// 3. Updates cache with data from backing store
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Try cache first
        match self.cache.query_metrics(from_timestamp).await {
            Ok(metrics) => Ok(metrics),
            Err(_) => {
                // Cache miss - query backing store
                let metrics = self.store.query_metrics(from_timestamp).await?;
                // Update cache
                if !metrics.is_empty() {
                    self.cache.insert_metrics(metrics.clone()).await?;
                }
                Ok(metrics)
            }
        }
    }

    /// Prepare a SQL query for execution.
    ///
    /// This method delegates to the backing store since
    /// SQL preparation is not cached.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        self.store.prepare_sql(query).await
    }

    /// Execute a prepared SQL query.
    ///
    /// This method delegates to the backing store since
    /// SQL execution is not cached.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        self.store.query_sql(statement_handle).await
    }

    /// Create a new instance with the given options.
    ///
    /// This method sets up both cache and backing store with
    /// appropriate configurations.
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized,
    {
        // Get cache engine type
        let cache_engine = options
            .get("cache_engine")
            .ok_or_else(|| Status::invalid_argument("Missing cache_engine option"))?;

        // Create cache backend
        let cache: Arc<dyn HyprStorageBackend> = match cache_engine.as_str() {
            "duckdb" => Arc::new(DuckDbBackend::new_with_options(
                ":memory:",
                options,
                credentials,
            )?),
            "adbc" => Arc::new(AdbcBackend::new_with_options(
                connection_string,
                options,
                credentials,
            )?),
            _ => return Err(Status::invalid_argument("Invalid cache engine type")),
        };

        // Create store backend
        let store_engine = options
            .get("store_engine")
            .ok_or_else(|| Status::invalid_argument("Missing store_engine option"))?;

        let store: Arc<dyn HyprStorageBackend> = match store_engine.as_str() {
            "duckdb" => Arc::new(DuckDbBackend::new_with_options(
                connection_string,
                options,
                credentials,
            )?),
            "adbc" => Arc::new(AdbcBackend::new_with_options(
                connection_string,
                options,
                credentials,
            )?),
            _ => return Err(Status::invalid_argument("Invalid store engine type")),
        };

        // Get cache duration
        let max_duration_secs = options
            .get("cache_duration")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);

        Ok(Self::new(cache, store, max_duration_secs))
    }

    async fn aggregate_metrics(
        &self,
        function: crate::aggregation::AggregateFunction,
        group_by: &crate::aggregation::GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<crate::aggregation::AggregateResult>, Status> {
        // Delegate to backing store
        self.store
            .aggregate_metrics(function, group_by, from_timestamp, to_timestamp)
            .await
    }

    async fn create_table(&self, table_name: &str, schema: &arrow_schema::Schema) -> Result<(), Status> {
        // Create in both cache and store
        self.cache.create_table(table_name, schema).await?;
        self.store.create_table(table_name, schema).await
    }

    async fn insert_into_table(&self, table_name: &str, batch: arrow_array::RecordBatch) -> Result<(), Status> {
        // Use zero-copy by sharing Arc references
        let schema = Arc::clone(&batch.schema());
        let arrays: Vec<ArrayRef> = batch.columns()
            .iter()
            .map(|col| Arc::clone(col))
            .collect();

        let shared_batch = arrow_array::RecordBatch::try_new(schema, arrays)
            .map_err(|e| Status::internal(format!("Failed to create shared batch: {}", e)))?;

        // Write-through using shared references
        self.store.insert_into_table(table_name, shared_batch.clone()).await?;
        self.cache.insert_into_table(table_name, shared_batch).await
    }

    async fn query_table(&self, table_name: &str, projection: Option<Vec<String>>) -> Result<arrow_array::RecordBatch, Status> {
        // Try cache first
        match self.cache.query_table(table_name, projection.clone()).await {
            Ok(batch) => {
                // Return zero-copy reference from cache
                let schema = Arc::clone(&batch.schema());
                let arrays: Vec<ArrayRef> = batch.columns()
                    .iter()
                    .map(|col| Arc::clone(col))
                    .collect();

                arrow_array::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| Status::internal(format!("Failed to create batch from cache: {}", e)))
            },
            Err(_) => {
                // Cache miss - query store
                let batch = self.store.query_table(table_name, projection).await?;
                
                // Create shared batch for cache update
                let schema = Arc::clone(&batch.schema());
                let arrays: Vec<ArrayRef> = batch.columns()
                    .iter()
                    .map(|col| Arc::clone(col))
                    .collect();

                let shared_batch = arrow_array::RecordBatch::try_new(schema.clone(), arrays.clone())
                    .map_err(|e| Status::internal(format!("Failed to create shared batch: {}", e)))?;

                // Update cache with shared reference
                self.cache
                    .insert_into_table(table_name, shared_batch)
                    .await?;

                // Return zero-copy result
                arrow_array::RecordBatch::try_new(schema, arrays)
                    .map_err(|e| Status::internal(format!("Failed to create result batch: {}", e)))
            }
        }
    }

    async fn create_aggregation_view(&self, view: &crate::storage::table_manager::HyprAggregationView) -> Result<(), Status> {
        // Create in both
        self.cache.create_aggregation_view(view).await?;
        self.store.create_aggregation_view(view).await
    }

    async fn query_aggregation_view(&self, view_name: &str) -> Result<arrow_array::RecordBatch, Status> {
        // Try cache first
        match self.cache.query_aggregation_view(view_name).await {
            Ok(batch) => Ok(batch),
            Err(_) => {
                // Cache miss - query store
                let batch = self.store.query_aggregation_view(view_name).await?;
                Ok(batch)
            }
        }
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        // Drop from both
        self.cache.drop_table(table_name).await?;
        self.store.drop_table(table_name).await
    }

    async fn drop_aggregation_view(&self, view_name: &str) -> Result<(), Status> {
        // Drop from both
        self.cache.drop_aggregation_view(view_name).await?;
        self.store.drop_aggregation_view(view_name).await
    }

    fn table_manager(&self) -> &crate::storage::table_manager::HyprTableManager {
        // Use store's table manager
        self.store.table_manager()
    }
}
