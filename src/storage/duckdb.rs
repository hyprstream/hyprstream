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

use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use duckdb::{Connection, Config};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

/// DuckDB-based storage backend for metrics.
///
/// This backend provides:
/// - High-performance storage using DuckDB
/// - Support for both in-memory and persistent storage
/// - Efficient batch operations for inserts and queries
/// - SQL query capabilities with time-based filtering
///
/// The implementation uses connection pooling and prepared statements
/// for optimal performance.
pub struct DuckDbBackend {
    /// Connection pool
    conn: Arc<Mutex<Connection>>,
    /// Connection string (file path or ":memory:")
    connection_string: String,
    /// Connection options
    options: HashMap<String, String>,
}

impl DuckDbBackend {
    /// Creates a new DuckDB backend with an in-memory database.
    ///
    /// This is useful for:
    /// - Temporary storage
    /// - Caching layers
    /// - Testing environments
    pub fn new_in_memory() -> Self {
        Self {
            conn: Arc::new(Mutex::new(Connection::open_in_memory().unwrap())),
            connection_string: ":memory:".to_string(),
            options: HashMap::new(),
        }
    }

    /// Creates a new DuckDB backend with the specified connection string.
    ///
    /// # Arguments
    ///
    /// * `connection_string` - The connection string to use. Can be ":memory:" for an in-memory
    ///   database or a path to a file for persistent storage.
    ///
    /// # Returns
    ///
    /// A Result containing either the backend or a Status error.
    pub fn new(connection_string: &str) -> Result<Self, Status> {
        let config = Config::default();
        Self::new_with_config(connection_string, config)
    }

    /// Creates the necessary database tables and indexes.
    ///
    /// This method:
    /// 1. Creates the metrics table if it doesn't exist
    /// 2. Creates indexes for efficient querying
    /// 3. Sets up the schema for metric storage
    async fn create_tables(&self) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE NOT NULL,
                value_running_window_avg DOUBLE NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_metric_id ON metrics(metric_id);",
        )
        .map_err(|e| Status::internal(e.to_string()))?;
        Ok(())
    }

    /// Gets a connection from the pool.
    ///
    /// This method provides thread-safe access to the DuckDB connection.
    async fn get_connection(&self) -> Result<tokio::sync::MutexGuard<'_, Connection>, Status> {
        Ok(self.conn.lock().await)
    }

    /// Converts a vector of MetricRecords to an Arrow RecordBatch.
    ///
    /// This method efficiently converts metrics to Arrow's columnar format
    /// for batch operations.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Vector of MetricRecord instances to convert
    ///
    /// # Returns
    ///
    /// * `Result<RecordBatch, Status>` - Arrow RecordBatch or error
    fn metrics_to_record_batch(&self, metrics: Vec<MetricRecord>) -> Result<RecordBatch, Status> {
        let schema = Schema::new(vec![
            Arc::new(Field::new("metric_id", DataType::Utf8, false)),
            Arc::new(Field::new("timestamp", DataType::Int64, false)),
            Arc::new(Field::new(
                "value_running_window_sum",
                DataType::Float64,
                false,
            )),
            Arc::new(Field::new(
                "value_running_window_avg",
                DataType::Float64,
                false,
            )),
            Arc::new(Field::new(
                "value_running_window_count",
                DataType::Int64,
                false,
            )),
        ]);

        let metric_ids = StringArray::from_iter(metrics.iter().map(|m| Some(m.metric_id.as_str())));
        let timestamps = Int64Array::from_iter(metrics.iter().map(|m| Some(m.timestamp)));
        let sums =
            Float64Array::from_iter(metrics.iter().map(|m| Some(m.value_running_window_sum)));
        let avgs =
            Float64Array::from_iter(metrics.iter().map(|m| Some(m.value_running_window_avg)));
        let counts =
            Int64Array::from_iter(metrics.iter().map(|m| Some(m.value_running_window_count)));

        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(metric_ids),
                Arc::new(timestamps),
                Arc::new(sums),
                Arc::new(avgs),
                Arc::new(counts),
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))
    }

    /// Executes a SQL query and returns the results as MetricRecords.
    ///
    /// This method:
    /// 1. Prepares the SQL statement
    /// 2. Executes the query
    /// 3. Converts results to MetricRecords
    ///
    /// # Arguments
    ///
    /// * `sql` - SQL query to execute
    ///
    /// # Returns
    ///
    /// * `Result<Vec<MetricRecord>, Status>` - Query results or error
    async fn execute_query(&self, sql: &str) -> Result<Vec<MetricRecord>, Status> {
        let conn = self.get_connection().await?;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut rows = stmt
            .query([])
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut metrics = Vec::new();
        while let Some(row) = rows.next().map_err(|e| Status::internal(e.to_string()))? {
            metrics.push(MetricRecord {
                metric_id: row.get(0).map_err(|e| Status::internal(e.to_string()))?,
                timestamp: row.get(1).map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_sum: row
                    .get(2)
                    .map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_avg: row
                    .get(3)
                    .map_err(|e| Status::internal(e.to_string()))?,
                value_running_window_count: row
                    .get(4)
                    .map_err(|e| Status::internal(e.to_string()))?,
            });
        }

        Ok(metrics)
    }

    /// Performs an efficient batch upsert operation.
    ///
    /// This method:
    /// 1. Creates a transaction
    /// 2. Prepares an upsert statement
    /// 3. Efficiently processes the batch
    /// 4. Commits the transaction
    ///
    /// The implementation uses string reuse and efficient batch processing
    /// to minimize allocations and maximize performance.
    ///
    /// # Arguments
    ///
    /// * `batch` - Arrow RecordBatch containing the records to upsert
    async fn upsert_batch(&self, batch: &RecordBatch) -> Result<(), Status> {
        let mut conn = self.get_connection().await?;
        let tx = conn
            .transaction()
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut stmt = tx
            .prepare(
                "INSERT OR REPLACE INTO metrics (
                    metric_id, timestamp, value_running_window_sum,
                    value_running_window_avg, value_running_window_count
                ) VALUES (?, ?, ?, ?, ?)",
            )
            .map_err(|e| Status::internal(e.to_string()))?;

        // Pre-allocate strings for numeric values to avoid repeated allocations
        let mut timestamp_str = String::new();
        let mut sum_str = String::new();
        let mut avg_str = String::new();
        let mut count_str = String::new();

        for i in 0..batch.num_rows() {
            // Clear and reuse strings
            timestamp_str.clear();
            sum_str.clear();
            avg_str.clear();
            count_str.clear();

            // Format values into the reused strings
            use std::fmt::Write;
            write!(
                timestamp_str,
                "{}",
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(i)
            )
            .map_err(|e| Status::internal(e.to_string()))?;
            write!(
                sum_str,
                "{}",
                batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(i)
            )
            .map_err(|e| Status::internal(e.to_string()))?;
            write!(
                avg_str,
                "{}",
                batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(i)
            )
            .map_err(|e| Status::internal(e.to_string()))?;
            write!(
                count_str,
                "{}",
                batch
                    .column(4)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(i)
            )
            .map_err(|e| Status::internal(e.to_string()))?;

            stmt.execute([
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(i),
                &timestamp_str,
                &sum_str,
                &avg_str,
                &count_str,
            ])
            .map_err(|e| Status::internal(e.to_string()))?;
        }

        tx.commit().map_err(|e| Status::internal(e.to_string()))?;
        Ok(())
    }

    /// Create connection options from configuration
    fn create_connection_options(&self) -> HashMap<String, String> {
        let mut opts = self.options.clone();

        // Set defaults if not specified
        if !opts.contains_key("threads") {
            opts.insert("threads".to_string(), "4".to_string());
        }
        if !opts.contains_key("read_only") {
            opts.insert("read_only".to_string(), "false".to_string());
        }

        opts
    }

    /// Creates a new DuckDB backend with configuration.
    pub fn new_with_config(connection_string: &str, config: Config) -> Result<Self, Status> {
        let conn = if connection_string == ":memory:" {
            Connection::open_in_memory_with_flags(config)
        } else {
            Connection::open_with_flags(connection_string, config)
        }
        .map_err(|e| Status::internal(format!("Failed to open connection: {}", e)))?;
        
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            connection_string: connection_string.to_string(),
            options: HashMap::new(),
        })
    }

    async fn execute(&self, query: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(query)
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;
        Ok(())
    }

    async fn query(&self, query: &str) -> Result<RecordBatch, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(query)
            .map_err(|e| Status::internal(format!("Failed to prepare query: {}", e)))?;
        let mut rows = stmt.query([])
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;
        
        // Convert rows to RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Utf8, false),
        ]));
        
        let mut values = Vec::new();
        while let Ok(Some(row)) = rows.next() {
            let value: String = row.get(0)
                .map_err(|e| Status::internal(format!("Failed to get value: {}", e)))?;
            values.push(value);
        }
        
        let array = StringArray::from(values);
        Ok(RecordBatch::try_new(schema, vec![Arc::new(array)])
            .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?)
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        _credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut config = Config::default();
        if let Some(threads) = options.get("threads").and_then(|s| s.parse().ok()) {
            config = config.threads(threads)
                .map_err(|e| Status::internal(format!("Failed to set threads: {}", e)))?;
        }
        if let Some(read_only) = options.get("read_only").and_then(|s| s.parse().ok()) {
            config = config.access_mode(if read_only {
                duckdb::AccessMode::ReadOnly
            } else {
                duckdb::AccessMode::ReadWrite
            })
            .map_err(|e| Status::internal(format!("Failed to set access mode: {}", e)))?;
        }

        let mut backend = Self::new_with_config(connection_string, config)?;
        backend.options = options.clone();
        Ok(backend)
    }
}

#[async_trait]
impl StorageBackend for DuckDbBackend {
    /// Initializes the DuckDB backend.
    ///
    /// Creates necessary tables and indexes for metric storage.
    async fn init(&self) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        
        // Create metrics table if it doesn't exist
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE PRECISION NOT NULL,
                value_running_window_avg DOUBLE PRECISION NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            )
            "#,
        )
        .map_err(|e| Status::internal(format!("Failed to create metrics table: {}", e)))?;

        Ok(())
    }

    /// Inserts a batch of metrics into storage.
    ///
    /// Converts metrics to a RecordBatch and performs an efficient
    /// batch upsert operation.
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let batch = self.metrics_to_record_batch(metrics)?;
        self.upsert_batch(&batch).await
    }

    /// Queries metrics from a given timestamp.
    ///
    /// Executes an optimized SQL query using the timestamp index.
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let sql = format!(
            "SELECT metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count \
             FROM metrics WHERE timestamp >= {}",
            from_timestamp
        );
        let sql_bytes = self.prepare_sql(&sql).await?;
        self.query_sql(&sql_bytes).await
    }

    /// Prepares a SQL statement for execution.
    ///
    /// Note: DuckDB doesn't support prepared statements in the same way as ADBC,
    /// so we store the SQL string as bytes.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        Ok(query.as_bytes().to_vec())
    }

    /// Executes a prepared SQL statement.
    ///
    /// Deserializes the statement handle and executes the query.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let sql = std::str::from_utf8(statement_handle)
            .map_err(|e| Status::internal(format!("Invalid UTF-8 in statement handle: {}", e)))?;
        self.execute_query(sql).await
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        _credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut config = Config::default();
        if let Some(threads) = options.get("threads").and_then(|s| s.parse().ok()) {
            config = config.threads(threads)
                .map_err(|e| Status::internal(format!("Failed to set threads: {}", e)))?;
        }
        if let Some(read_only) = options.get("read_only").and_then(|s| s.parse().ok()) {
            config = config.access_mode(if read_only {
                duckdb::AccessMode::ReadOnly
            } else {
                duckdb::AccessMode::ReadWrite
            })
            .map_err(|e| Status::internal(format!("Failed to set access mode: {}", e)))?;
        }

        let mut backend = Self::new_with_config(connection_string, config)?;
        backend.options = options.clone();
        Ok(backend)
    }
}
