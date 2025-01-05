use crate::storage::{MetricRecord, StorageBackend};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tonic::Status;

#[derive(Clone)]
pub struct CachedStorageBackend {
    cache: Arc<dyn StorageBackend>,
    backing_store: Arc<dyn StorageBackend>,
    cache_duration_secs: i64,
}

impl CachedStorageBackend {
    pub fn new(
        cache: Arc<dyn StorageBackend>,
        backing_store: Arc<dyn StorageBackend>,
        cache_duration_secs: i64,
    ) -> Self {
        Self {
            cache,
            backing_store,
            cache_duration_secs,
        }
    }

    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }
}

#[async_trait]
impl StorageBackend for CachedStorageBackend {
    async fn init(&self) -> Result<(), Status> {
        // Initialize both cache and backing store
        self.cache.init().await?;
        self.backing_store.init().await?;
        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Insert into backing store first
        self.backing_store.insert_metrics(metrics.clone()).await?;
        
        // Then update cache, reusing the metrics vector
        if let Err(e) = self.cache.insert_metrics(metrics).await {
            tracing::warn!("Failed to update cache: {}", e);
        }

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Calculate cache invalidation time
        let current_time = Self::current_timestamp();
        let cache_valid_after = current_time - self.cache_duration_secs;

        // If querying data newer than cache_valid_after, try cache first
        if from_timestamp >= cache_valid_after {
            match self.cache.query_metrics(from_timestamp).await {
                Ok(results) => Ok(results),
                Err(_) => {
                    // Cache miss or error, query backing store
                    let results = self.backing_store.query_metrics(from_timestamp).await?;

                    // Update cache with new results
                    if let Err(e) = self.cache.insert_metrics(results.clone()).await {
                        tracing::warn!("Failed to update cache: {}", e);
                    }

                    Ok(results)
                }
            }
        } else {
            // For older data, go directly to backing store
            self.backing_store.query_metrics(from_timestamp).await
        }
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // Prepare on backing store only - cache doesn't need prepared statements
        self.backing_store.prepare_sql(query).await
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        // Execute directly on backing store - prepared statements bypass cache
        self.backing_store.query_sql(statement_handle).await
    }
}
