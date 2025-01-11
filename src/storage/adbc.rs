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
    driver_manager::{ManagedConnection, ManagedDriver, ManagedStatement},
    options::{AdbcVersion, OptionDatabase, OptionValue},
    Connection, Database, Driver, Statement, Optionable,
};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, StringArray, RecordBatch,
};
use arrow_schema::{Schema, Field, DataType};
use arrow_convert::{ArrowField, ArrowSerialize};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tonic::Status;

use crate::storage::arrow_utils::RecordBatchExt;
use crate::aggregation::build_aggregate_query;
use crate::storage::cache::{CacheManager, CacheEviction};
use crate::storage::{
    HyprStorageBackend, HyprBatchAggregation, HyprMetricRecord,
    HyprAggregateFunction, HyprGroupBy, HyprAggregateResult,
    HyprTimeWindow, HyprTableManager, HyprAggregationView,
    HyprCredentials
};
use arrow_convert::serialize::TryIntoArrow;
use crate::storage::arrow_utils::BatchConverter;

/// Safe conversion of values to ADBC option values
trait IntoAdbcOptionValue {
    fn into_adbc_option_value(self) -> Result<OptionValue, Status>;
}

impl IntoAdbcOptionValue for String {
    fn into_adbc_option_value(self) -> Result<OptionValue, Status> {
        Ok(OptionValue::String(self))
    }
}

impl IntoAdbcOptionValue for i64 {
    fn into_adbc_option_value(self) -> Result<OptionValue, Status> {
        Ok(OptionValue::Int64(self))
    }
}

impl IntoAdbcOptionValue for f64 {
    fn into_adbc_option_value(self) -> Result<OptionValue, Status> {
        Ok(OptionValue::Float64(self))
    }
}

/// ADBC storage backend implementation
#[derive(Clone)]
pub struct AdbcBackend {
    driver: Arc<ManagedDriver>,
    conn: Arc<Mutex<ManagedConnection>>,
    table_manager: Arc<HyprTableManager>,
    cache_manager: Arc<CacheManager>,
    query_counter: Arc<AtomicU64>,
}

impl AdbcBackend {
    /// Create a new ADBC backend instance with options
    pub fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&HyprCredentials>,
    ) -> Result<Self, Status> {
        let driver_path = options.get("driver_path")
            .ok_or_else(|| Status::invalid_argument("driver_path is required"))?;

        let mut driver_options = HashMap::new();
        driver_options.insert("uri".to_string(), connection_string.to_string().into_adbc_option_value()?);
        
        if let Some(creds) = credentials {
            driver_options.insert("username".to_string(), creds.username.clone().into_adbc_option_value()?);
            driver_options.insert("password".to_string(), creds.password.clone().into_adbc_option_value()?);
        }

        let driver = ManagedDriver::load_dynamic_from_filename(
            driver_path,
            None,
            AdbcVersion::V100,
        ).map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let conn = driver.connect(&driver_options)
            .map_err(|e| Status::internal(format!("Failed to create ADBC connection: {}", e)))?;

        Ok(Self {
            driver: Arc::new(driver),
            conn: Arc::new(Mutex::new(conn)),
            table_manager: Arc::new(HyprTableManager::new()),
            cache_manager: Arc::new(CacheManager::new(Some(3600))), // 1 hour TTL
            query_counter: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Execute a query and convert the result to a RecordBatch
    async fn execute_query(&self, sql: &str) -> Result<RecordBatch, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let adbc_batch = reader.next()
            .ok_or_else(|| Status::internal("No data returned"))?
            .map_err(|e| Status::internal(format!("Failed to get batch: {}", e)))?;

        BatchConverter::convert_to_record_batch(Box::new(adbc_batch))
    }

    /// Bind values to a statement
    fn bind_values<T: IntoAdbcOptionValue>(stmt: &mut ManagedStatement, values: &[T]) -> Result<(), Status> {
        let option_values: Result<Vec<_>, _> = values.iter()
            .map(|v| v.clone().into_adbc_option_value())
            .collect();
        
        stmt.bind(&option_values?)
            .map_err(|e| Status::internal(format!("Failed to bind values: {}", e)))
    }
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
                tracing::error!("Background eviction error: {}", e);
            }
        });
        Ok(())
    }
}

#[async_trait]
impl HyprStorageBackend for AdbcBackend {
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&HyprCredentials>,
    ) -> Result<Self, Status> {
        let driver_path = options.get("driver_path")
            .ok_or_else(|| Status::invalid_argument("driver_path is required"))?;

        let mut driver_options = HashMap::new();
        driver_options.insert("uri".to_string(), connection_string.to_string().into_adbc_option_value()?);
        
        if let Some(creds) = credentials {
            driver_options.insert("username".to_string(), creds.username.clone().into_adbc_option_value()?);
            driver_options.insert("password".to_string(), creds.password.clone().into_adbc_option_value()?);
        }

        let driver = ManagedDriver::load_dynamic_from_filename(
            driver_path,
            None,
            AdbcVersion::V100,
        ).map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let conn = driver.connect(&driver_options)
            .map_err(|e| Status::internal(format!("Failed to create ADBC connection: {}", e)))?;

        Ok(Self {
            driver: Arc::new(driver),
            conn: Arc::new(Mutex::new(conn)),
            table_manager: Arc::new(HyprTableManager::new()),
            cache_manager: Arc::new(CacheManager::new(Some(3600))), // 1 hour TTL
            query_counter: Arc::new(AtomicU64::new(0)),
        })
    }

    async fn init(&self) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        
        // Create metrics table
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
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

            CREATE TABLE IF NOT EXISTS metric_aggregations (
                metric_id VARCHAR NOT NULL,
                window_start BIGINT NOT NULL,
                window_end BIGINT NOT NULL,
                running_sum DOUBLE PRECISION NOT NULL,
                running_count BIGINT NOT NULL,
                min_value DOUBLE PRECISION NOT NULL,
                max_value DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (metric_id, window_start, window_end)
            );

            CREATE INDEX IF NOT EXISTS idx_aggregations_window 
            ON metric_aggregations(window_start, window_end);
        "#).map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create tables: {}", e)))?;

        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<HyprMetricRecord>) -> Result<(), Status> {
        if metrics.is_empty() {
            return Ok(());
        }

        // Check if eviction is needed
        if let Some(cutoff) = self.cache_manager.should_evict().await? {
            let query = self.cache_manager.eviction_query(cutoff);
            self.execute_eviction(&query).await?;
        }

        // Begin transaction
        let mut conn = self.conn.lock().await;
        
        // Insert metrics
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        let sql = "INSERT INTO metrics (metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count) VALUES (?, ?, ?, ?, ?)";
        stmt.set_sql_query(sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        for metric in metrics {
            stmt.bind_values(&[
                OptionValue::Utf8(metric.metric_id),
                OptionValue::Int64(metric.timestamp),
                OptionValue::Float64(metric.value_running_window_sum),
                OptionValue::Float64(metric.value_running_window_avg),
                OptionValue::Int64(metric.value_running_window_count),
            ]).map_err(|e| Status::internal(format!("Failed to bind values: {}", e)))?;

            stmt.execute_update()
                .map_err(|e| Status::internal(format!("Failed to execute insert: {}", e)))?;
        }

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<HyprMetricRecord>, Status> {
        let mut conn = self.conn.lock().await;
        
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "SELECT metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count 
            FROM metrics
            WHERE timestamp >= ?
             ORDER BY timestamp ASC"
        ).map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.bind_values(&[OptionValue::Int64(from_timestamp)])
            .map_err(|e| Status::internal(format!("Failed to bind timestamp: {}", e)))?;

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result
                .map_err(|e| Status::internal(format!("Failed to get next batch: {}", e)))?;
            
            for row_idx in 0..batch.num_rows() {
                metrics.push(HyprMetricRecord {
                    metric_id: batch.column(0).as_any().downcast_ref::<StringArray>()
                        .ok_or_else(|| Status::internal("Invalid metric_id column"))?
                        .value(row_idx).to_string(),
                    timestamp: batch.column(1).as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid timestamp column"))?
                        .value(row_idx),
                    value_running_window_sum: batch.column(2).as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?
                        .value(row_idx),
                    value_running_window_avg: batch.column(3).as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?
                        .value(row_idx),
                    value_running_window_count: batch.column(4).as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?
                        .value(row_idx),
                });
            }
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let handle = self.query_counter.fetch_add(1, Ordering::SeqCst);
        Ok(handle.to_le_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<HyprMetricRecord>, Status> {
        let handle = u64::from_le_bytes(
            statement_handle.try_into()
                .map_err(|_| Status::invalid_argument("Invalid statement handle"))?
        );

        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&format!("SELECT * FROM metrics WHERE id = {}", handle))
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result
                .map_err(|e| Status::internal(format!("Failed to get next batch: {}", e)))?;
            
            for row_idx in 0..batch.num_rows() {
                metrics.push(HyprMetricRecord {
                    metric_id: batch.column(0).as_any().downcast_ref::<StringArray>()
                        .ok_or_else(|| Status::internal("Invalid metric_id column"))?
                        .value(row_idx).to_string(),
                    timestamp: batch.column(1).as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid timestamp column"))?
                        .value(row_idx),
                    value_running_window_sum: batch.column(2).as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?
                        .value(row_idx),
                    value_running_window_avg: batch.column(3).as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?
                        .value(row_idx),
                    value_running_window_count: batch.column(4).as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?
                        .value(row_idx),
                });
            }
        }

        Ok(metrics)
    }

    async fn aggregate_metrics(
        &self,
        function: HyprAggregateFunction,
        group_by: &HyprGroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<HyprAggregateResult>, Status> {
        let mut conn = self.conn.lock().await;

        let query = build_aggregate_query(
            "metrics",
            function,
            group_by,
            &["metric_id", "timestamp", "value_running_window_sum", "value_running_window_avg", "value_running_window_count"],
            Some(from_timestamp),
            to_timestamp,
        );

        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&query)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut results = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result
                .map_err(|e| Status::internal(format!("Failed to get next batch: {}", e)))?;
            
            for row_idx in 0..batch.num_rows() {
                results.push(HyprAggregateResult {
                    value: batch.column(2).as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid value column"))?
                        .value(row_idx),
                    timestamp: batch.column(1).as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid timestamp column"))?
                        .value(row_idx),
                });
            }
        }

        Ok(results)
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        let mut sql = format!("CREATE TABLE IF NOT EXISTS {} (", table_name);
        let fields: Vec<String> = schema.fields().iter().map(|f| {
            format!("{} {}", 
                f.name(),
                match f.data_type() {
                    DataType::Int8 => "TINYINT",
                    DataType::Int16 => "SMALLINT",
                    DataType::Int32 => "INTEGER",
                    DataType::Int64 => "BIGINT",
                    DataType::Float32 => "FLOAT",
                    DataType::Float64 => "DOUBLE PRECISION",
                    DataType::Boolean => "BOOLEAN",
                    DataType::Utf8 => "VARCHAR",
                    DataType::Binary => "BLOB",
                    _ => "VARCHAR",
                }
            )
        }).collect();
        sql.push_str(&fields.join(", "));
        sql.push_str(")");

        stmt.set_sql_query(&sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create table: {}", e)))?;

        Ok(())
    }

    async fn create_aggregation_view(&self, view: &HyprAggregationView) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        let window_sql = match &view.window {
            HyprTimeWindow::Sliding { window, slide } => {
                format!("WINDOW {} PRECEDING SLIDE {}", window.as_secs(), slide.as_secs())
            },
            HyprTimeWindow::Fixed(timestamp) => {
                format!("WINDOW FIXED AT {}", timestamp)
            },
            HyprTimeWindow::None => String::new(),
        };

        let sql = format!(
            "CREATE VIEW {} AS SELECT * FROM {} {}",
            view.name, view.source_table, window_sql
        );

        stmt.set_sql_query(&sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create view: {}", e)))?;

        Ok(())
    }

    async fn query_aggregation_view(&self, view_name: &str) -> Result<RecordBatch, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&format!("SELECT * FROM {}", view_name))
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        reader.next()
            .ok_or_else(|| Status::internal("No data returned"))?
            .map_err(|e| Status::internal(format!("Failed to get batch: {}", e)))
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&format!("DROP TABLE IF EXISTS {}", table_name))
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to drop table: {}", e)))?;

        Ok(())
    }

    async fn drop_aggregation_view(&self, view_name: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&format!("DROP VIEW IF EXISTS {}", view_name))
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to drop view: {}", e)))?;

        Ok(())
    }

    fn table_manager(&self) -> &HyprTableManager {
        &self.table_manager
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        let sql = format!("INSERT INTO {} VALUES ({})",
            table_name,
            (0..batch.num_columns()).map(|_| "?").collect::<Vec<_>>().join(", ")
        );

        stmt.set_sql_query(&sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        for row_idx in 0..batch.num_rows() {
            let mut values = Vec::new();
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                let value = match col.data_type() {
                    DataType::Int64 => {
                        let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        OptionValue::Int64(array.value(row_idx))
                    }
                    DataType::Float64 => {
                        let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        OptionValue::Float64(array.value(row_idx))
                    }
                    DataType::Utf8 => {
                        let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                        OptionValue::String(array.value(row_idx).to_string())
                    }
                    _ => return Err(Status::internal("Unsupported column type")),
                };
                values.push(value);
            }

            Self::bind_values(&mut stmt, &values)?;
            stmt.execute_update()
                .map_err(|e| Status::internal(format!("Failed to execute insert: {}", e)))?;
        }

        Ok(())
    }

    async fn query_table(&self, table_name: &str, projection: Option<Vec<String>>) -> Result<RecordBatch, Status> {
        let schema = self.table_manager.get_table_schema(table_name).await?;
        
        let projection = projection.unwrap_or_else(|| {
            schema.fields().iter().map(|f| f.name().clone()).collect()
        });

        let sql = format!(
            "SELECT {} FROM {}",
            projection.join(", "),
            table_name
        );

        self.execute_query(&sql).await
    }
}
