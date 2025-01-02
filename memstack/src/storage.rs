use async_trait::async_trait;
use tonic::Status;

#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub metric_id: String,
    pub timestamp: i64,
    pub value_running_window_sum: f64,
    pub value_running_window_avg: f64,
    pub value_running_window_count: i64,
}

#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    async fn init(&self) -> Result<(), Status>;
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;
}

// DuckDB Implementation
pub mod duckdb {
    use super::*;
    use ::duckdb::{params, Connection};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    pub struct DuckDbBackend {
        conn: Arc<Mutex<Connection>>,
    }

    impl DuckDbBackend {
        pub fn new() -> Self {
            let conn = Connection::open_in_memory().unwrap();
            Self {
                conn: Arc::new(Mutex::new(conn)),
            }
        }
    }

    #[async_trait]
    impl StorageBackend for DuckDbBackend {
        async fn init(&self) -> Result<(), Status> {
            let conn = self.conn.lock().await;
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT NOT NULL,
                    timestamp BIGINT NOT NULL,
                    valueRunningWindowSum DOUBLE NOT NULL,
                    valueRunningWindowAvg DOUBLE NOT NULL,
                    valueRunningWindowCount BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_metrics_metric_id ON metrics(metric_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_combined ON metrics(metric_id, timestamp);",
            )
            .map_err(|e| Status::internal(format!("Failed to create table and indexes: {}", e)))
        }

        async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
            let mut conn = self.conn.lock().await;
            let tx = conn
                .transaction()
                .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

            // Prepare the statement once for better performance
            let mut stmt = tx.prepare(
                "INSERT INTO metrics (metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount) 
                 VALUES (?1, ?2, ?3, ?4, ?5)"
            ).map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

            for metric in metrics {
                stmt.execute(params![
                    metric.metric_id,
                    metric.timestamp,
                    metric.value_running_window_sum,
                    metric.value_running_window_avg,
                    metric.value_running_window_count,
                ])
                .map_err(|e| Status::internal(format!("Failed to insert metric: {}", e)))?;
            }

            tx.commit()
                .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))
        }

        async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
            let conn = self.conn.lock().await;
            let mut stmt = conn.prepare(
                "SELECT metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount 
                 FROM metrics 
                 WHERE timestamp >= ?1 
                 ORDER BY timestamp ASC, metric_id ASC
                 LIMIT 100"
            ).map_err(|e| Status::internal(format!("Failed to prepare query: {}", e)))?;

            let rows = stmt
                .query_map([from_timestamp], |row| {
                    Ok(MetricRecord {
                        metric_id: row.get(0)?,
                        timestamp: row.get(1)?,
                        value_running_window_sum: row.get(2)?,
                        value_running_window_avg: row.get(3)?,
                        value_running_window_count: row.get(4)?,
                    })
                })
                .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;

            let mut results = Vec::new();
            for row in rows {
                results
                    .push(row.map_err(|e| Status::internal(format!("Row mapping failed: {}", e)))?);
            }
            Ok(results)
        }
    }
}

// Redis Implementation
pub mod redis {
    use super::*;
    use ::redis::{cmd, Client, Connection, RedisError};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    pub struct RedisBackend {
        conn: Arc<Mutex<Connection>>,
    }

    impl RedisBackend {
        pub fn new(redis_url: &str) -> Result<Self, redis::RedisError> {
            let client = Client::open(redis_url)?;
            let conn = client.get_connection()?;
            Ok(Self {
                conn: Arc::new(Mutex::new(conn)),
            })
        }
    }

    #[async_trait]
    impl StorageBackend for RedisBackend {
        async fn init(&self) -> Result<(), Status> {
            Ok(()) // Redis doesn't need initialization
        }

        async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
            let mut conn = self.conn.lock().await;

            for metric in metrics {
                let key = format!("metric:{}:{}", metric.metric_id, metric.timestamp);
                let value = format!(
                    "{}:{}:{}:{}",
                    metric.metric_id,
                    metric.value_running_window_sum,
                    metric.value_running_window_avg,
                    metric.value_running_window_count
                );

                redis::cmd("SET").arg(&key).arg(&value).execute(&mut *conn);
                //#.map_err(|e| Status::internal(format!("Failed to insert metric: {}", e)));
            }

            Ok(())
        }

        async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
            let mut conn = self.conn.lock().await;

            // Get all keys matching the pattern
            let keys: Vec<String> = redis::cmd("KEYS")
                .arg("metric:*")
                .query(&mut *conn)
                .map_err(|e| Status::internal(format!("Failed to query keys: {}", e)))?;

            let mut results = Vec::new();
            for key in keys {
                let value: String = redis::cmd("GET")
                    .arg(&key)
                    .query(&mut *conn)
                    .map_err(|e| Status::internal(format!("Failed to get value: {}", e)))?;

                let parts: Vec<&str> = value.split(':').collect();
                if parts.len() == 4 {
                    let timestamp = key
                        .split(':')
                        .nth(2)
                        .and_then(|ts| ts.parse().ok())
                        .unwrap_or(0);

                    if timestamp >= from_timestamp {
                        results.push(MetricRecord {
                            metric_id: parts[0].to_string(),
                            timestamp,
                            value_running_window_sum: parts[1].parse().unwrap_or(0.0),
                            value_running_window_avg: parts[2].parse().unwrap_or(0.0),
                            value_running_window_count: parts[3].parse().unwrap_or(0),
                        });
                    }
                }
            }

            results.sort_by_key(|m| m.timestamp);
            Ok(results.into_iter().take(100).collect())
        }
    }
}

// Add this new module after the existing redis and duckdb modules
pub mod cached {
    use super::*;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

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
    }
}
