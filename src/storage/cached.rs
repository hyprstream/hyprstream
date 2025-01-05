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
        // Insert into both cache and backing store
        let cache_fut = self.cache.insert_metrics(metrics.clone());
        let backing_fut = self.backing_store.insert_metrics(metrics);
        
        // Execute both operations concurrently
        let (cache_res, backing_res) = tokio::join!(cache_fut, backing_fut);
        
        // If backing store fails, we must fail the operation
        backing_res?;
        
        // If cache fails, log warning but continue
        if let Err(e) = cache_res {
            tracing::warn!("Failed to update cache: {}", e);
        }
        
        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Try cache first
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
