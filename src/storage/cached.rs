use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use std::sync::Arc;
use tonic::Status;

pub struct CachedStorageBackend {
    cache: Arc<dyn StorageBackend>,
    store: Arc<dyn StorageBackend>,
    cache_duration: i64,
}

impl CachedStorageBackend {
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
    async fn init(&self) -> Result<(), Status> {
        // Initialize both cache and backing store
        self.cache.init().await?;
        self.store.init().await?;
        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Insert into both cache and backing store
        self.cache.insert_metrics(metrics.clone()).await?;
        self.store.insert_metrics(metrics).await?;
        Ok(())
    }

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

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // Prepare on backing store only
        self.store.prepare_sql(query).await
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        // Execute on backing store only
        self.store.query_sql(statement_handle).await
    }
}
