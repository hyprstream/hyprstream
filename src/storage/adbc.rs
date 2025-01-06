//! ADBC (Arrow Database Connectivity) storage backend implementation.
//!
//! This module provides a storage backend using ADBC, enabling:
//! - Connection to any ADBC-compliant database
//! - High-performance data transport using Arrow's columnar format
//! - Connection pooling and prepared statements
//! - Support for various database systems (PostgreSQL, MySQL, etc.)
//!
//! # Configuration
//!
//! The ADBC backend can be configured using the following options:
//!
//! ```toml
//! [engine]
//! engine = "adbc"
//! # Base connection without credentials
//! connection = "postgresql://localhost:5432/metrics"
//! options = {
//!     driver_path = "/usr/local/lib/libadbc_driver_postgresql.so",  # Required: Path to ADBC driver
//!     pool_max = "10",                                            # Optional: Maximum pool connections
//!     pool_min = "1",                                             # Optional: Minimum pool connections
//!     connect_timeout = "30"                                      # Optional: Connection timeout in seconds
//! }
//! ```
//!
//! For security, credentials should be provided via environment variables:
//! ```bash
//! export HYPRSTREAM_DB_USERNAME=postgres
//! export HYPRSTREAM_DB_PASSWORD=secret
//! ```
//!
//! Or via command line:
//!
//! ```bash
//! hyprstream \
//!   --engine adbc \
//!   --engine-connection "postgresql://localhost:5432/metrics" \
//!   --engine-options driver_path=/usr/local/lib/libadbc_driver_postgresql.so \
//!   --engine-options pool_max=10
//! ```
//!
//! The implementation is optimized for efficient data transfer and
//! query execution using Arrow's native formats.

use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use adbc_core::{
    driver_manager::{ManagedConnection, ManagedDriver},
    options::{AdbcVersion, OptionDatabase, OptionValue},
    Connection, Database, Driver, Statement, Optionable,
};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tonic::Status;

pub struct AdbcBackend {
    conn: Arc<Mutex<ManagedConnection>>,
    statement_counter: AtomicU64,
    prepared_statements: Arc<Mutex<Vec<(u64, String)>>>,
    ttl: u64,
}

impl AdbcBackend {
    pub fn new(driver_path: &str, connection: Option<&str>, credentials: Option<&Credentials>) -> Result<Self, Status> {
        let mut driver = ManagedDriver::load_dynamic_from_filename(
            driver_path,
            None,
            AdbcVersion::V100,
        ).map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let mut database = driver.new_database()
            .map_err(|e| Status::internal(format!("Failed to create database: {}", e)))?;

        // Set connection string if provided
        if let Some(conn_str) = connection {
            database.set_option(OptionDatabase::Uri, OptionValue::String(conn_str.to_string()))
                .map_err(|e| Status::internal(format!("Failed to set connection string: {}", e)))?;
        }

        // Set credentials if provided
        if let Some(creds) = credentials {
            // Set username and password from credentials
            database.set_option(OptionDatabase::Username, OptionValue::String(creds.username.clone()))
                .map_err(|e| Status::internal(format!("Failed to set username: {}", e)))?;

            database.set_option(OptionDatabase::Password, OptionValue::String(creds.password.clone()))
                .map_err(|e| Status::internal(format!("Failed to set password: {}", e)))?;
        }

        let connection = database.new_connection()
            .map_err(|e| Status::internal(format!("Failed to create connection: {}", e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(connection)),
            statement_counter: AtomicU64::new(0),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
            ttl: 0,
        })
    }

    async fn get_connection(&self) -> Result<tokio::sync::MutexGuard<'_, ManagedConnection>, Status> {
        Ok(self.conn.lock().await)
    }

    async fn execute_statement(&self, conn: &mut ManagedConnection, query: &str, batch: Option<RecordBatch>) -> Result<(), Status> {
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(query)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        if let Some(params) = batch {
            stmt.bind(params)
                .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;
        }

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute statement: {}", e)))?;

        Ok(())
    }

    async fn execute_query(&self, conn: &mut ManagedConnection, query: &str, params: Option<&RecordBatch>) -> Result<Vec<RecordBatch>, Status> {
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(query)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        if let Some(batch) = params {
            stmt.bind(batch.clone())
                .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;
        }

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut batches = Vec::new();
        while let Some(batch_result) = reader.next() {
            match batch_result {
                Ok(batch) => batches.push(batch),
                Err(e) => return Err(Status::internal(format!("Failed to get next batch: {}", e))),
            }
        }

        Ok(batches)
    }

    fn prepare_timestamp_param(timestamp: i64) -> Result<RecordBatch, Status> {
        let schema = Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
        ]);

        let timestamps: ArrayRef = Arc::new(Int64Array::from(vec![timestamp]));
        
        RecordBatch::try_new(Arc::new(schema), vec![timestamps])
            .map_err(|e| Status::internal(format!("Failed to create parameter batch: {}", e)))
    }

    fn prepare_params(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]);

        let metric_ids = StringArray::from_iter_values(metrics.iter().map(|m| m.metric_id.as_str()));
        let timestamps = Int64Array::from_iter_values(metrics.iter().map(|m| m.timestamp));
        let sums = Float64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_sum));
        let avgs = Float64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_avg));
        let counts = Int64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_count));

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(metric_ids),
            Arc::new(timestamps),
            Arc::new(sums),
            Arc::new(avgs),
            Arc::new(counts),
        ];

        RecordBatch::try_new(Arc::new(schema), arrays)
            .map_err(|e| Status::internal(format!("Failed to create parameter batch: {}", e)))
    }

    /// Evicts expired entries from the cache.
    async fn evict_expired(&self) -> Result<(), Status> {
        if self.ttl == 0 {
            return Ok(());
        }

        let mut conn = self.get_connection().await?;
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| Status::internal(format!("Failed to get current time: {}", e)))?
            .as_secs() as i64;

        let expiry_time = current_time - self.ttl as i64;
        
        self.execute_statement(&mut conn, 
            "DELETE FROM metrics WHERE timestamp < ?", 
            Some(Self::prepare_timestamp_param(expiry_time)?)
        ).await
    }
}

#[async_trait]
impl StorageBackend for AdbcBackend {
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let driver_path = options.get("driver_path")
            .ok_or_else(|| Status::invalid_argument("driver_path is required"))?;

        let mut backend = Self::new(
            driver_path,
            Some(connection_string),
            credentials,
        )?;

        // Set TTL if provided
        if let Some(ttl) = options.get("ttl").and_then(|s| s.parse().ok()) {
            backend.ttl = ttl;
        }

        Ok(backend)
    }

    async fn init(&self) -> Result<(), Status> {
        let mut conn = self.get_connection().await?;
        
        // Create metrics table if it doesn't exist
        self.execute_statement(&mut conn, r#"
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE PRECISION NOT NULL,
                value_running_window_avg DOUBLE PRECISION NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            )
        "#, None).await?;

        // Create index on timestamp for efficient eviction
        self.execute_statement(&mut conn, 
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            None
        ).await?;

        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        if metrics.is_empty() {
            return Ok(());
        }

        // Evict expired entries before inserting new ones
        self.evict_expired().await?;

        let mut conn = self.get_connection().await?;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        // Build parameterized insert statement
        let placeholders: Vec<_> = (0..metrics.len())
            .map(|i| format!("(${},${},${},${},${})", i*5+1, i*5+2, i*5+3, i*5+4, i*5+5))
            .collect();

        let query = format!(
            r#"
            INSERT INTO metrics (
                metric_id,
                timestamp,
                value_running_window_sum,
                value_running_window_avg,
                value_running_window_count
            ) VALUES {}
            "#,
            placeholders.join(",")
        );

        stmt.set_sql_query(&query)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        // Bind parameters
        let batch = Self::prepare_params(&metrics)?;
        stmt.bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to insert metrics: {}", e)))?;

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Evict expired entries before querying
        self.evict_expired().await?;

        let mut conn = self.get_connection().await?;
        
        let query = r#"
            SELECT
                metric_id,
                timestamp,
                value_running_window_sum,
                value_running_window_avg,
                value_running_window_count
            FROM metrics
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        "#;

        let params = Self::prepare_timestamp_param(from_timestamp)?;
        let batches = self.execute_query(&mut conn, query, Some(&params)).await?;

        let mut metrics = Vec::new();
        for batch in batches {
            let metric_ids = batch.column_by_name("metric_id")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

            let timestamps = batch.column_by_name("timestamp")
                .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

            let sums = batch.column_by_name("value_running_window_sum")
                .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

            let avgs = batch.column_by_name("value_running_window_avg")
                .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

            let counts = batch.column_by_name("value_running_window_count")
                .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

            for i in 0..batch.num_rows() {
                metrics.push(MetricRecord {
                    metric_id: metric_ids.value(i).to_string(),
                    timestamp: timestamps.value(i),
                    value_running_window_sum: sums.value(i),
                    value_running_window_avg: avgs.value(i),
                    value_running_window_count: counts.value(i),
                });
            }
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let handle = self.statement_counter.fetch_add(1, Ordering::SeqCst);
        let mut statements = self.prepared_statements.lock().await;
        statements.push((handle, query.to_string()));
        Ok(handle.to_le_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let handle = u64::from_le_bytes(
            statement_handle.try_into()
                .map_err(|_| Status::invalid_argument("Invalid statement handle"))?
        );

        let statements = self.prepared_statements.lock().await;
        let sql = statements
            .iter()
            .find(|(h, _)| *h == handle)
            .map(|(_, sql)| sql.as_str())
            .ok_or_else(|| Status::invalid_argument("Statement handle not found"))?;

        let mut conn = self.get_connection().await?;
        let batches = self.execute_query(&mut conn, sql, None).await?;

        let mut metrics = Vec::new();
        for batch in batches {
            let metric_ids = batch.column_by_name("metric_id")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

            let timestamps = batch.column_by_name("timestamp")
                .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

            let sums = batch.column_by_name("value_running_window_sum")
                .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

            let avgs = batch.column_by_name("value_running_window_avg")
                .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

            let counts = batch.column_by_name("value_running_window_count")
                .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

            for i in 0..batch.num_rows() {
                metrics.push(MetricRecord {
                    metric_id: metric_ids.value(i).to_string(),
                    timestamp: timestamps.value(i),
                    value_running_window_sum: sums.value(i),
                    value_running_window_avg: avgs.value(i),
                    value_running_window_count: counts.value(i),
                });
            }
        }

        Ok(metrics)
    }
}
