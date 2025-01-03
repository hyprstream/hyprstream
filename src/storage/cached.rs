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
        // Initialize both storage backends
        self.cache.init().await?;
        self.backing_store.init().await?;
        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Write to both cache and backing store
        self.cache.insert_metrics(metrics.clone()).await?;
        self.backing_store.insert_metrics(metrics).await?;
        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // First try to get from cache
        let cache_results = self.cache.query_metrics(from_timestamp).await;

        match cache_results {
            Ok(results) if !results.is_empty() => {
                // Cache hit
                Ok(results)
            }
            _ => {
                // Cache miss - get from backing store
                let results = self.backing_store.query_metrics(from_timestamp).await?;

                if !results.is_empty() {
                    // Update cache with results
                    // Only cache data that's within our cache window
                    let cache_cutoff = Self::current_timestamp() - self.cache_duration_secs;
                    let to_cache: Vec<MetricRecord> = results
                        .iter()
                        .filter(|r| r.timestamp >= cache_cutoff)
                        .cloned()
                        .collect();

                    if !to_cache.is_empty() {
                        // Don't propagate cache update errors to the client
                        let _ = self.cache.insert_metrics(to_cache).await;
                    }
                }

                Ok(results)
            }
        }
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // Delegate to backing store
        self.backing_store.prepare_sql(query).await
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        // Delegate to backing store
        self.backing_store.query_sql(statement_handle).await
    }
} 