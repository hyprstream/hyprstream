//! DuckDB storage backend implementation.
//!
//! This module provides a high-performance storage backend using DuckDB,
//! an embedded analytical database. The implementation supports:
//! - In-memory and persistent storage options
//! - Efficient batch operations
//! - SQL query capabilities
//! - Time-based filtering
//!
//! # Configuration
//!
//! The DuckDB backend can be configured using the following options:
//!
//! ```toml
//! [engine]
//! engine = "duckdb"
//! connection = ":memory:"  # Use ":memory:" for in-memory or file path
//! options = {
//!     threads = "4",      # Optional: Number of threads (default: 4)
//!     read_only = "false" # Optional: Read-only mode (default: false)
//! }
//! ```
//!
//! Or via command line:
//!
//! ```bash
//! hyprstream \
//!   --engine duckdb \
//!   --engine-connection ":memory:" \
//!   --engine-options threads=4 \
//!   --engine-options read_only=false
//! ```
//!
//! DuckDB is particularly well-suited for analytics workloads and
//! provides excellent performance for both caching and primary storage.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use duckdb::{Connection, Config};
use tokio::sync::{Mutex, RwLock};
use tonic::Status;
use crate::metrics::MetricRecord;
use crate::config::Credentials;
use crate::storage::StorageBackend;
use async_trait::async_trait;

/// DuckDB-based storage backend for metrics.
#[derive(Clone)]
pub struct DuckDbBackend {
    conn: Arc<Mutex<Connection>>,
    connection_string: String,
    options: HashMap<String, String>,
    ttl: Option<u64>,
    last_eviction: Arc<RwLock<SystemTime>>,
    min_eviction_interval: Duration,
}

#[async_trait]
impl StorageBackend for DuckDbBackend {
    async fn init(&self) -> Result<(), Status> {
        self.create_tables().await
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Evict expired entries before inserting new ones
        self.evict_expired().await?;

        let mut query = String::from("INSERT INTO metrics (timestamp, metric_id, value_running_window_sum, value_running_window_avg, value_running_window_count) VALUES ");
        let mut first = true;

        for metric in metrics {
            if !first {
                query.push_str(", ");
            }
            first = false;

            query.push_str(&format!(
                "({}, '{}', {}, {}, {})",
                metric.timestamp,
                metric.metric_id,
                metric.value_running_window_sum,
                metric.value_running_window_avg,
                metric.value_running_window_count
            ));
        }

        self.execute(&query).await
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Evict expired entries before querying
        self.evict_expired().await?;

        let query = format!(
            "SELECT timestamp, metric_id, value_running_window_sum, value_running_window_avg, value_running_window_count \
             FROM metrics WHERE timestamp >= {}",
            from_timestamp
        );

        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&query)
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut rows = stmt.query([])
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut metrics = Vec::new();
        while let Some(row) = rows.next().map_err(|e| Status::internal(e.to_string()))? {
            let metric = MetricRecord {
                timestamp: row.get(0).map_err(|e| Status::internal(e.to_string()))?,
                metric_id: row.get(1).map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_sum: row.get(2).map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_avg: row.get(3).map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_count: row.get(4).map_err(|e| Status::internal(e.to_string()))?,
            };
            metrics.push(metric);
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // DuckDB doesn't support prepared statement handles in the same way as ADBC,
        // so we just store the SQL string as bytes
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let sql = std::str::from_utf8(statement_handle)
            .map_err(|e| Status::internal(e.to_string()))?;
        self.query_metrics(sql.parse().unwrap_or(0)).await
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut all_options = options.clone();
        if let Some(creds) = credentials {
            all_options.insert("username".to_string(), creds.username.clone());
            all_options.insert("password".to_string(), creds.password.clone());
        }

        let ttl = all_options.get("ttl")
            .and_then(|s| s.parse().ok())
            .map(|ttl| if ttl == 0 { None } else { Some(ttl) })
            .unwrap_or(None);

        Self::new(connection_string.to_string(), all_options, ttl)
    }
}

impl DuckDbBackend {
    /// Creates a new DuckDB backend instance.
    pub fn new(connection_string: String, options: HashMap<String, String>, ttl: Option<u64>) -> Result<Self, Status> {
        let config = Config::default();
        let conn = Connection::open_with_flags(&connection_string, config)
            .map_err(|e| Status::internal(e.to_string()))?;

        let backend = Self {
            conn: Arc::new(Mutex::new(conn)),
            connection_string,
            options,
            ttl,
            last_eviction: Arc::new(RwLock::new(SystemTime::now())),
            min_eviction_interval: Duration::from_secs(60), // Minimum 60s between evictions
        };

        // Initialize tables
        let backend_clone = backend.clone();
        tokio::spawn(async move {
            if let Err(e) = backend_clone.create_tables().await {
                eprintln!("Failed to create tables: {}", e);
            }
        });

        Ok(backend)
    }

    /// Creates a new DuckDB backend with an in-memory database.
    pub fn new_in_memory() -> Result<Self, Status> {
        Self::new(":memory:".to_string(), HashMap::new(), Some(0))
    }

    /// Evicts expired entries from the cache based on TTL.
    /// Uses a rate limiter to prevent too frequent evictions.
    async fn evict_expired(&self) -> Result<(), Status> {
        if let Some(ttl) = self.ttl {
            if ttl == 0 {
                return Ok(()); // TTL of 0 means no expiration
            }

            // Check if enough time has passed since last eviction
            let now = SystemTime::now();
            let last = *self.last_eviction.read().await;
            if now.duration_since(last).unwrap_or(Duration::from_secs(0)) < self.min_eviction_interval {
                return Ok(());
            }

            // Update last eviction time before performing eviction
            *self.last_eviction.write().await = now;

            let cutoff = now
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - ttl;

            // Use an optimized DELETE with USING clause for better index utilization
            let query = format!(
                "DELETE FROM metrics USING (
                    SELECT timestamp 
                    FROM metrics 
                    WHERE timestamp < {} 
                    LIMIT 10000
                ) as expired 
                WHERE metrics.timestamp = expired.timestamp",
                cutoff
            );

            // Spawn eviction in background
            let conn = self.conn.clone();
            tokio::spawn(async move {
                let conn_guard = conn.lock().await;
                if let Err(e) = conn_guard.execute_batch(&query) {
                    eprintln!("Background eviction error: {}", e);
                }
            });
        }
        Ok(())
    }

    /// Creates the necessary tables in the database.
    async fn create_tables(&self) -> Result<(), Status> {
        let create_table = r#"
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp BIGINT NOT NULL,
                metric_id VARCHAR NOT NULL,
                value_running_window_sum DOUBLE NOT NULL,
                value_running_window_avg DOUBLE NOT NULL,
                value_running_window_count BIGINT NOT NULL
            )
        "#;

        self.execute(create_table).await?;

        // Create a more optimized index for TTL-based eviction
        let create_index = r#"
            CREATE INDEX IF NOT EXISTS metrics_timestamp_idx ON metrics(timestamp) WITH (prefetch_blocks = 8)
        "#;

        self.execute(create_index).await
    }

    /// Executes a SQL query.
    async fn execute(&self, query: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(query)
            .map_err(|e| Status::internal(e.to_string()))
    }
}
