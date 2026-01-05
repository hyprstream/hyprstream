use crate::aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
use crate::metrics::{MetricRecord, create_record_batch, encode_record_batch};
use crate::storage::StorageBackend;
use duckdb::arrow::array::{Float64Array, Int64Array, StringArray};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

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
}

/// Storage operations specific to metrics data.
#[async_trait::async_trait]
pub trait MetricsStorage: Send + Sync + 'static {
    /// Get the underlying storage backend
    fn backend(&self) -> &dyn StorageBackend;

    /// Initialize metrics storage (create tables etc.)
    async fn init(&self) -> Result<(), Status> {
        let schema = Self::get_metrics_schema();
        self.backend().create_table("metrics", &schema).await
    }

    /// Insert metrics into storage
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let batch = create_record_batch(&metrics)?;
        self.backend().insert_into_table("metrics", batch).await
    }

    /// Query metrics from storage
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let sql = Self::generate_metric_query_sql(from_timestamp);
        let handle = self.backend().prepare_sql(&sql).await?;
        let batch = self.backend().query_sql(&handle).await?;
        encode_record_batch(&batch)
    }

    /// Aggregate metrics using the specified function and grouping
    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        let sql = Self::generate_metric_aggregation_sql(function, group_by, from_timestamp, to_timestamp);
        let handle = self.backend().prepare_sql(&sql).await?;
        let batch = self.backend().query_sql(&handle).await?;

        let mut results = Vec::with_capacity(batch.num_rows());
        
        // Extract column arrays
        let value_col = batch
            .column_by_name("value")
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value column"))?;

        // Get group by columns
        let mut group_cols = Vec::new();
        for col_name in &group_by.columns {
            let col = batch
                .column_by_name(col_name)
                .ok_or_else(|| Status::internal(format!("Missing group by column: {}", col_name)))?;
            group_cols.push(col);
        }

        // Build results
        for row in 0..batch.num_rows() {
            let mut group_values = HashMap::new();
            for (col_name, col) in group_by.columns.iter().zip(&group_cols) {
                let value = if let Some(str_col) = col.as_any().downcast_ref::<StringArray>() {
                    str_col.value(row).to_string()
                } else if let Some(int_col) = col.as_any().downcast_ref::<Int64Array>() {
                    int_col.value(row).to_string()
                } else if let Some(float_col) = col.as_any().downcast_ref::<Float64Array>() {
                    float_col.value(row).to_string()
                } else {
                    return Err(Status::internal(format!("Unsupported group by column type: {}", col_name)));
                };
                group_values.insert(col_name.clone(), value);
            }

            // Extract timestamp if it's part of the group by
            let timestamp = if let Some(time_col) = &group_by.time_column {
                if let Some(col) = batch.column_by_name(time_col) {
                    if let Some(int_col) = col.as_any().downcast_ref::<Int64Array>() {
                        Some(int_col.value(row))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            results.push(AggregateResult {
                value: value_col.value(row),
                group_values,
                timestamp,
            });
        }

        Ok(results)
    }

    /// Get the standard schema for metric records
    fn get_metrics_schema() -> Schema {
        Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ])
    }

    /// Generate SQL for inserting metric records
    fn generate_metric_insert_sql() -> String {
        "INSERT INTO metrics (metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count) VALUES (?, ?, ?, ?, ?)"
            .to_string()
    }

    /// Generate SQL for querying metrics
    fn generate_metric_query_sql(from_timestamp: i64) -> String {
        format!(
            "SELECT * FROM metrics WHERE timestamp >= {} ORDER BY timestamp ASC",
            from_timestamp
        )
    }

    /// Generate SQL for aggregating metrics
    fn generate_metric_aggregation_sql(
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
}

/// A wrapper around StorageBackend that implements MetricsStorage
pub struct MetricsStorageImpl<B: StorageBackend> {
    backend: B,
}

impl<B: StorageBackend> MetricsStorageImpl<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

#[async_trait::async_trait]
impl<B: StorageBackend> MetricsStorage for MetricsStorageImpl<B> {
    fn backend(&self) -> &dyn StorageBackend {
        &self.backend
    }
}