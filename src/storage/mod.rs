//! Storage backends for metric data persistence and caching.
//!
//! This module provides multiple storage backend implementations:
//! - `duckdb`: High-performance embedded database for caching and local storage
//! - `adbc`: Arrow Database Connectivity for external database integration
//! - `cached`: Two-tier storage with configurable caching layer
//!
//! Each backend implements the `StorageBackend` trait, providing a consistent
//! interface for metric storage and retrieval operations.

pub mod adbc;
pub mod cache;
pub mod duckdb;
pub mod table_manager;
pub mod view;

use crate::aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
use crate::cli::commands::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::view::{ViewDefinition, ViewMetadata};
use arrow_array::{
    Array, Float64Array, Int64Array, RecordBatch, StringArray,
    builder::{Float64Builder, Int64Builder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

/// Utility functions for storage operations
pub struct StorageUtils;

impl StorageUtils {
    /// Generate SQL for creating a table with the given schema
    pub fn generate_create_table_sql(table_name: &str, schema: &Schema) -> Result<String, Status> {
        let mut sql = format!("CREATE TABLE IF NOT EXISTS {} (", table_name);
        let mut first = true;

        for field in schema.fields() {
            if !first {
                sql.push_str(", ");
            }
            first = false;

            sql.push_str(&format!(
                "{} {}",
                field.name(),
                match field.data_type() {
                    DataType::Int64 => "BIGINT",
                    DataType::Float64 => "DOUBLE PRECISION",
                    DataType::Utf8 => "VARCHAR",
                    _ => return Err(Status::invalid_argument(format!(
                        "Unsupported data type: {:?}",
                        field.data_type()
                    ))),
                }
            ));
        }

        sql.push_str(")");
        Ok(sql)
    }

    /// Generate SQL for inserting data into a table
    pub fn generate_insert_sql(table_name: &str, column_count: usize) -> String {
        let placeholders = vec!["?"; column_count].join(", ");
        format!("INSERT INTO {} VALUES ({})", table_name, placeholders)
    }

    /// Generate SQL for inserting metric records
    pub fn generate_metric_insert_sql() -> String {
        "INSERT INTO metrics (metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count) VALUES (?, ?, ?, ?, ?)"
            .to_string()
    }

    /// Generate SQL for querying metrics
    pub fn generate_metric_query_sql(from_timestamp: i64) -> String {
        format!(
            "SELECT * FROM metrics WHERE timestamp >= {} ORDER BY timestamp ASC",
            from_timestamp
        )
    }

    /// Generate SQL for aggregating metrics
    pub fn generate_metric_aggregation_sql(
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> String {
        let mut sql = format!(
            "SELECT {}, {}({}) as value",
            group_by.columns.join(", "),
            function,
            "value_running_window_avg" // Use avg for now
        );

        sql.push_str(" FROM metrics");
        sql.push_str(&format!(" WHERE timestamp >= {}", from_timestamp));

        if let Some(to) = to_timestamp {
            sql.push_str(&format!(" AND timestamp <= {}", to));
        }

        sql.push_str(&format!(" GROUP BY {}", group_by.columns.join(", ")));
        sql
    }

    /// Generate SQL for selecting data from a table
    pub fn generate_select_sql(table_name: &str, projection: Option<Vec<String>>) -> String {
        let columns = projection.map(|cols| cols.join(", ")).unwrap_or_else(|| "*".to_string());
        format!("SELECT {} FROM {}", columns, table_name)
    }

    /// Get the standard schema for metric records
    pub fn get_metric_schema() -> Schema {
        Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ])
    }

    /// Create a RecordBatch from metrics
    pub fn create_metric_batch(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
        let mut builders = (
            StringBuilder::new(),
            Int64Builder::new(),
            Float64Builder::new(),
            Float64Builder::new(),
            Int64Builder::new(),
        );

        for metric in metrics {
            builders.0.append_value(&metric.metric_id);
            builders.1.append_value(metric.timestamp);
            builders.2.append_value(metric.value_running_window_sum);
            builders.3.append_value(metric.value_running_window_avg);
            builders.4.append_value(metric.value_running_window_count);
        }

        RecordBatch::try_new(
            Arc::new(Self::get_metric_schema()),
            vec![
                Arc::new(builders.0.finish()),
                Arc::new(builders.1.finish()),
                Arc::new(builders.2.finish()),
                Arc::new(builders.3.finish()),
                Arc::new(builders.4.finish()),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create batch: {}", e)))
    }

    /// Convert a RecordBatch to metrics
    pub fn batch_to_metrics(batch: &RecordBatch) -> Result<Vec<MetricRecord>, Status> {
        let mut metrics = Vec::new();

        for row in 0..batch.num_rows() {
            metrics.push(MetricRecord {
                metric_id: batch
                    .column_by_name("metric_id")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| Status::internal("Invalid metric_id column"))?
                    .value(row)
                    .to_string(),
                timestamp: batch
                    .column_by_name("timestamp")
                    .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                    .ok_or_else(|| Status::internal("Invalid timestamp column"))?
                    .value(row),
                value_running_window_sum: batch
                    .column_by_name("value_running_window_sum")
                    .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                    .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?
                    .value(row),
                value_running_window_avg: batch
                    .column_by_name("value_running_window_avg")
                    .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
                    .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?
                    .value(row),
                value_running_window_count: batch
                    .column_by_name("value_running_window_count")
                    .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                    .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?
                    .value(row),
            });
        }

        Ok(metrics)
    }

    /// Generate SQL for creating a view
    pub fn generate_view_sql(name: &str, definition: &ViewDefinition) -> String {
        format!("CREATE VIEW {} AS {}", name, definition.to_sql())
    }
}

/// Batch-level aggregation state for efficient updates
#[derive(Debug, Clone)]
pub struct BatchAggregation {
    /// The metric ID this aggregation belongs to
    pub metric_id: String,
    /// Start of the time window
    pub window_start: i64,
    /// End of the time window
    pub window_end: i64,
    /// Running sum within the window
    pub running_sum: f64,
    /// Running count within the window
    pub running_count: i64,
    /// Minimum value in the window
    pub min_value: f64,
    /// Maximum value in the window
    pub max_value: f64,
    /// Schema for the aggregation
    pub schema: Arc<Schema>,
    /// Column to aggregate
    pub value_column: String,
    /// Grouping specification
    pub group_by: GroupBy,
    /// Time window specification
    pub window: Option<TimeWindow>,
}

impl BatchAggregation {
    pub fn new_window(
        metric_id: String,
        window_start: i64,
        window_end: i64,
        schema: Arc<Schema>,
        value_column: String,
        group_by: GroupBy,
        window: Option<TimeWindow>,
    ) -> Self {
        Self {
            metric_id,
            window_start,
            window_end,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            schema,
            value_column,
            group_by,
            window,
        }
    }

    pub fn new_from_metric(
        metric_id: String,
        window_start: i64,
        window_end: i64,
        window: TimeWindow,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("timestamp", DataType::Int64, false),
        ]));
        let group_by = GroupBy {
            columns: vec!["metric".to_string()],
            time_column: Some("timestamp".to_string()),
        };
        Self {
            metric_id,
            window_start,
            window_end,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            schema,
            value_column: "value".to_string(),
            group_by,
            window: Some(window),
        }
    }

    pub fn new(
        schema: Arc<Schema>,
        value_column: String,
        group_by: GroupBy,
        window: Option<TimeWindow>,
    ) -> Self {
        Self {
            metric_id: String::new(),
            window_start: 0,
            window_end: 0,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            schema,
            value_column,
            group_by,
            window,
        }
    }

    pub fn build_query(&self, table_name: &str) -> String {
        crate::aggregation::build_aggregate_query(
            table_name,
            AggregateFunction::Avg,
            &self.group_by,
            &[&self.value_column],
            None,
            None,
        )
    }
}

/// Storage backend trait for metric data persistence.
#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    /// Initialize the storage backend.
    async fn init(&self) -> Result<(), Status>;

    /// Insert metrics into storage.
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;

    /// Query metrics from storage.
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;

    /// Prepare a SQL query and return a handle.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Execute a prepared SQL query using its handle.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;

    /// Aggregate metrics using the specified function and grouping.
    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status>;

    /// Create a new instance with the given options.
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized;

    /// Create a new table with the given schema
    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status>;

    /// Insert data into a table
    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status>;

    /// Query data from a table
    async fn query_table(
        &self,
        table_name: &str,
        projection: Option<Vec<String>>,
    ) -> Result<RecordBatch, Status>;

    /// Create a view with the given definition
    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status>;

    /// Get view metadata
    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status>;

    /// List all views
    async fn list_views(&self) -> Result<Vec<String>, Status>;

    /// List all tables
    async fn list_tables(&self) -> Result<Vec<String>, Status>;

    /// Get schema for a table
    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status>;

    /// Drop a view
    async fn drop_view(&self, name: &str) -> Result<(), Status>;

    /// Drop a table
    async fn drop_table(&self, table_name: &str) -> Result<(), Status>;

    /// Update batch-level aggregations.
    async fn update_batch_aggregations(
        &self,
        batch: &[MetricRecord],
        window: TimeWindow,
    ) -> Result<Vec<BatchAggregation>, Status> {
        let mut aggregations = HashMap::new();

        for metric in batch {
            let (window_start, window_end) = window.window_bounds(metric.timestamp);
            let key = (metric.metric_id.clone(), window_start, window_end);

            let agg = aggregations.entry(key).or_insert_with(|| {
                BatchAggregation::new_from_metric(
                    metric.metric_id.clone(),
                    window_start,
                    window_end,
                    window,
                )
            });

            agg.running_sum += metric.value_running_window_sum;
            agg.running_count += 1;
            agg.min_value = agg.min_value.min(metric.value_running_window_sum);
            agg.max_value = agg.max_value.max(metric.value_running_window_sum);
        }

        Ok(aggregations.into_values().collect())
    }

    /// Insert batch-level aggregations.
    async fn insert_batch_aggregations(
        &self,
        aggregations: Vec<BatchAggregation>,
    ) -> Result<(), Status> {
        let mut batch = Vec::new();
        for agg in aggregations {
            batch.push(MetricRecord {
                metric_id: agg.metric_id,
                timestamp: agg.window_start,
                value_running_window_sum: agg.running_sum,
                value_running_window_avg: agg.running_sum / agg.running_count as f64,
                value_running_window_count: agg.running_count,
            });
        }
        self.insert_metrics(batch).await
    }
}

#[derive(Clone)]
pub enum StorageBackendType {
    Adbc(adbc::AdbcBackend),
    DuckDb(duckdb::DuckDbBackend),
}

impl AsRef<dyn StorageBackend> for StorageBackendType {
    fn as_ref(&self) -> &(dyn StorageBackend + 'static) {
        match self {
            StorageBackendType::Adbc(backend) => backend,
            StorageBackendType::DuckDb(backend) => backend,
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for StorageBackendType {
    async fn init(&self) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.init().await,
            StorageBackendType::DuckDb(backend) => backend.init().await,
        }
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.insert_metrics(metrics).await,
            StorageBackendType::DuckDb(backend) => backend.insert_metrics(metrics).await,
        }
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.query_metrics(from_timestamp).await,
            StorageBackendType::DuckDb(backend) => backend.query_metrics(from_timestamp).await,
        }
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.prepare_sql(query).await,
            StorageBackendType::DuckDb(backend) => backend.prepare_sql(query).await,
        }
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.query_sql(statement_handle).await,
            StorageBackendType::DuckDb(backend) => backend.query_sql(statement_handle).await,
        }
    }

    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => {
                backend
                    .aggregate_metrics(function, group_by, from_timestamp, to_timestamp)
                    .await
            }
            StorageBackendType::DuckDb(backend) => {
                backend
                    .aggregate_metrics(function, group_by, from_timestamp, to_timestamp)
                    .await
            }
        }
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized,
    {
        let engine_type = options
            .get("engine")
            .ok_or_else(|| Status::invalid_argument("Missing engine type"))?;

        match engine_type.as_str() {
            "adbc" => Ok(StorageBackendType::Adbc(
                adbc::AdbcBackend::new_with_options(connection_string, options, credentials)?,
            )),
            "duckdb" => Ok(StorageBackendType::DuckDb(
                duckdb::DuckDbBackend::new_with_options(connection_string, options, credentials)?,
            )),
            _ => Err(Status::invalid_argument("Invalid engine type")),
        }
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.create_table(table_name, schema).await,
            StorageBackendType::DuckDb(backend) => backend.create_table(table_name, schema).await,
        }
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.insert_into_table(table_name, batch).await,
            StorageBackendType::DuckDb(backend) => backend.insert_into_table(table_name, batch).await,
        }
    }

    async fn query_table(
        &self,
        table_name: &str,
        projection: Option<Vec<String>>,
    ) -> Result<RecordBatch, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.query_table(table_name, projection).await,
            StorageBackendType::DuckDb(backend) => backend.query_table(table_name, projection).await,
        }
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.create_view(name, definition).await,
            StorageBackendType::DuckDb(backend) => backend.create_view(name, definition).await,
        }
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.get_view(name).await,
            StorageBackendType::DuckDb(backend) => backend.get_view(name).await,
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.list_views().await,
            StorageBackendType::DuckDb(backend) => backend.list_views().await,
        }
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.drop_view(name).await,
            StorageBackendType::DuckDb(backend) => backend.drop_view(name).await,
        }
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.drop_table(table_name).await,
            StorageBackendType::DuckDb(backend) => backend.drop_table(table_name).await,
        }
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.list_tables().await,
            StorageBackendType::DuckDb(backend) => backend.list_tables().await,
        }
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.get_table_schema(table_name).await,
            StorageBackendType::DuckDb(backend) => backend.get_table_schema(table_name).await,
        }
    }

    async fn update_batch_aggregations(
        &self,
        batch: &[MetricRecord],
        window: TimeWindow,
    ) -> Result<Vec<BatchAggregation>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.update_batch_aggregations(batch, window).await,
            StorageBackendType::DuckDb(backend) => backend.update_batch_aggregations(batch, window).await,
        }
    }

    async fn insert_batch_aggregations(
        &self,
        aggregations: Vec<BatchAggregation>,
    ) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.insert_batch_aggregations(aggregations).await,
            StorageBackendType::DuckDb(backend) => backend.insert_batch_aggregations(aggregations).await,
        }
    }
}
