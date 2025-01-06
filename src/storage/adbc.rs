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

use adbc_core::{
    driver_manager::{ManagedConnection, ManagedDriver},
    options::{AdbcVersion, OptionDatabase, OptionValue},
    Connection, Database, Driver, Statement, Optionable,
};
use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tonic::Status;
use crate::metrics::aggregation::{
    AggregateFunction, GroupBy, AggregateResult, build_aggregate_query
};
use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use crate::storage::cache::{CacheManager, CacheEviction};

pub struct AdbcBackend {
    conn: Arc<Mutex<ManagedConnection>>,
    statement_counter: AtomicU64,
    prepared_statements: Arc<Mutex<Vec<(u64, String)>>>,
    cache_manager: CacheManager,
}

#[async_trait]
impl CacheEviction for AdbcBackend {
    async fn execute_eviction(&self, query: &str) -> Result<(), Status> {
        let conn = self.conn.clone();
        let query = query.to_string(); // Clone for background task
        tokio::spawn(async move {
            let mut conn_guard = conn.lock().await;
            if let Err(e) = conn_guard.new_statement()
                .and_then(|mut stmt| {
                    stmt.set_sql_query(&query)?;
                    stmt.execute_update()
                }) {
                eprintln!("Background eviction error: {}", e);
            }
        });
        Ok(())
    }
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
            cache_manager: CacheManager::new(None), // Initialize without TTL
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
            // Create a new statement for binding parameters
            let mut bind_stmt = conn.new_statement()
                .map_err(|e| Status::internal(format!("Failed to create bind statement: {}", e)))?;

            // Set the parameters using SQL directly
            let mut param_values = Vec::new();
            for i in 0..params.num_rows() {
                for j in 0..params.num_columns() {
                    let col = params.column(j);
                    match col.data_type() {
                        DataType::Int64 => {
                            let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                            param_values.push(array.value(i).to_string());
                        }
                        DataType::Float64 => {
                            let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                            param_values.push(array.value(i).to_string());
                        }
                        DataType::Utf8 => {
                            let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                            param_values.push(format!("'{}'", array.value(i)));
                        }
                        _ => return Err(Status::internal("Unsupported parameter type")),
                    }
                }
            }

            let params_sql = format!("VALUES ({})", param_values.join(", "));
            bind_stmt.set_sql_query(&params_sql)
                .map_err(|e| Status::internal(format!("Failed to set parameters: {}", e)))?;

            let mut bind_result = bind_stmt.execute()
                .map_err(|e| Status::internal(format!("Failed to execute parameter binding: {}", e)))?;

            while let Some(batch_result) = bind_result.next() {
                let _ = batch_result.map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;
            }
        }

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute statement: {}", e)))?;

        Ok(())
    }

    async fn execute_query(&self, conn: &mut ManagedConnection, query: &str, params: Option<RecordBatch>) -> Result<Vec<MetricRecord>, Status> {
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(query)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        if let Some(batch) = params {
            // Create a new statement for binding parameters
            let mut bind_stmt = conn.new_statement()
                .map_err(|e| Status::internal(format!("Failed to create bind statement: {}", e)))?;

            // Set the parameters using SQL directly
            let mut param_values = Vec::new();
            for i in 0..batch.num_rows() {
                for j in 0..batch.num_columns() {
                    let col = batch.column(j);
                    match col.data_type() {
                        DataType::Int64 => {
                            let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                            param_values.push(array.value(i).to_string());
                        }
                        DataType::Float64 => {
                            let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                            param_values.push(array.value(i).to_string());
                        }
                        DataType::Utf8 => {
                            let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                            param_values.push(format!("'{}'", array.value(i)));
                        }
                        _ => return Err(Status::internal("Unsupported parameter type")),
                    }
                }
            }

            let params_sql = format!("VALUES ({})", param_values.join(", "));
            bind_stmt.set_sql_query(&params_sql)
                .map_err(|e| Status::internal(format!("Failed to set parameters: {}", e)))?;

            let mut bind_result = bind_stmt.execute()
                .map_err(|e| Status::internal(format!("Failed to execute parameter binding: {}", e)))?;

            while let Some(batch_result) = bind_result.next() {
                let _ = batch_result.map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;
            }
        }

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result.map_err(|e| Status::internal(format!("Failed to get next batch: {}", e)))?;
            
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

    fn prepare_timestamp_param(timestamp: i64) -> Result<RecordBatch, Status> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
        ]));

        let timestamps: ArrayRef = Arc::new(Int64Array::from(vec![timestamp]));
        
        RecordBatch::try_new(schema, vec![timestamps])
            .map_err(|e| Status::internal(format!("Failed to create parameter batch: {}", e)))
    }

    fn prepare_params(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]));

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

        RecordBatch::try_new(schema, arrays)
            .map_err(|e| Status::internal(format!("Failed to create parameter batch: {}", e)))
    }
}

#[async_trait]
impl StorageBackend for AdbcBackend {
    async fn init(&self) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        
        // Create metrics table if it doesn't exist
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(r#"
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE PRECISION NOT NULL,
                value_running_window_avg DOUBLE PRECISION NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            )
        "#).map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create table: {}", e)))?;

        // Create index for efficient eviction
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)"
        ).map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;

        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        if metrics.is_empty() {
            return Ok(());
        }

        // Check if eviction is needed
        if let Some(cutoff) = self.cache_manager.should_evict().await? {
            let query = self.cache_manager.eviction_query(cutoff);
            self.execute_eviction(&query).await?;
        }

        let mut conn = self.conn.lock().await;
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

        // Prepare parameters
        let batch = Self::prepare_params(&metrics)?;

        // Create a new statement for binding parameters
        let mut bind_stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create bind statement: {}", e)))?;

        // Set the parameters using SQL directly
        let mut param_values = Vec::new();
        for i in 0..batch.num_rows() {
            for j in 0..batch.num_columns() {
                let col = batch.column(j);
                match col.data_type() {
                    DataType::Int64 => {
                        let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        param_values.push(array.value(i).to_string());
                    }
                    DataType::Float64 => {
                        let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        param_values.push(array.value(i).to_string());
                    }
                    DataType::Utf8 => {
                        let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                        param_values.push(format!("'{}'", array.value(i)));
                    }
                    _ => return Err(Status::internal("Unsupported parameter type")),
                }
            }
        }

        let params_sql = format!("VALUES ({})", param_values.join(", "));
        bind_stmt.set_sql_query(&params_sql)
            .map_err(|e| Status::internal(format!("Failed to set parameters: {}", e)))?;

        let mut bind_result = bind_stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute parameter binding: {}", e)))?;

        while let Some(batch_result) = bind_result.next() {
            let _ = batch_result.map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;
        }

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to insert metrics: {}", e)))?;

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Check if eviction is needed
        if let Some(cutoff) = self.cache_manager.should_evict().await? {
            let query = self.cache_manager.eviction_query(cutoff);
            self.execute_eviction(&query).await?;
        }

        let mut conn = self.conn.lock().await;
        
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
        self.execute_query(&mut conn, query, Some(params)).await
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

        let mut conn = self.conn.lock().await;
        self.execute_query(&mut conn, sql, None).await
    }

    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        // Check if eviction is needed
        if let Some(cutoff) = self.cache_manager.should_evict().await? {
            let query = self.cache_manager.eviction_query(cutoff);
            self.execute_eviction(&query).await?;
        }

        let query = build_aggregate_query(function, group_by, from_timestamp, to_timestamp);
        let mut conn = self.conn.lock().await;
        let metrics = self.execute_query(&mut conn, &query, None).await?;

        let mut results = Vec::new();
        for metric in metrics {
            let result = AggregateResult {
                metric_id: Some(metric.metric_id),
                timestamp: metric.timestamp,
                window_start: metric.timestamp,
                window_end: to_timestamp.unwrap_or(i64::MAX),
                value: metric.value_running_window_sum,
            };
            results.push(result);
        }

        Ok(results)
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let driver_path = options.get("driver_path")
            .ok_or_else(|| Status::invalid_argument("driver_path is required"))?;

        let mut driver = ManagedDriver::load_dynamic_from_filename(
            driver_path,
            None,
            AdbcVersion::V100,
        ).map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let mut database = driver.new_database()
            .map_err(|e| Status::internal(format!("Failed to create database: {}", e)))?;

        // Set connection string
        database.set_option(OptionDatabase::Uri, OptionValue::String(connection_string.to_string()))
            .map_err(|e| Status::internal(format!("Failed to set connection string: {}", e)))?;

        // Set credentials if provided
        if let Some(creds) = credentials {
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
            cache_manager: CacheManager::new(None), // Initialize without TTL
        })
    }
}
