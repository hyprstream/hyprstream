pub mod arrow_utils;
pub mod zerocopy;
pub mod simd;

#[cfg(test)]
mod benchmarks {
    use super::simd::SimdOps;
    use std::time::Instant;

    #[test]
    fn bench_simd_operations() {
        let ops = SimdOps::new();
        let size = 1_000_000;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(i as f32);
        }

        // Benchmark SIMD abs
        let start = Instant::now();
        for chunk in data.chunks(4) {
            if chunk.len() == 4 {
                let _ = ops.f32x4_abs(chunk);
            }
        }
        let simd_time = start.elapsed();

        // Benchmark scalar abs
        let start = Instant::now();
        for x in &data {
            let _ = x.abs();
        }
        let scalar_time = start.elapsed();

        println!("SIMD abs time: {:?}", simd_time);
        println!("Scalar abs time: {:?}", scalar_time);
        println!("Speedup: {:.2}x", scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
    }
}
pub mod adbc;
pub mod duckdb;
pub mod vector;
pub mod table_manager;
pub mod cache;
pub mod cached;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use std::collections::HashMap;
use tonic::Status;

use crate::storage::table_manager::{HyprTableManager, HyprAggregationView};

/// Batch-level aggregation state for efficient updates
#[derive(Debug, Clone)]
pub struct HyprBatchAggregation {
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
}

#[async_trait::async_trait]
pub trait HyprStorageBackend: Send + Sync + 'static {
    /// Initialize the storage backend.
    async fn init(&self) -> Result<(), Status>;

    /// Insert metrics into storage.
    async fn insert_metrics(&self, metrics: Vec<HyprMetricRecord>) -> Result<(), Status>;

    /// Query metrics from storage.
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<HyprMetricRecord>, Status>;

    /// Prepare a SQL query and return a handle.
    /// The handle is backend-specific and opaque to the caller.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Execute a prepared SQL query using its handle.
    /// The handle must have been obtained from prepare_sql.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<HyprMetricRecord>, Status>;

    /// Aggregate metrics using the specified function and grouping.
    async fn aggregate_metrics(
        &self,
        function: HyprAggregateFunction,
        group_by: &HyprGroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<HyprAggregateResult>, Status>;

    /// Create a new instance with the given options.
    /// The connection string and options are backend-specific.
    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&HyprCredentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized;

    /// Create a new table with the given schema
    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status>;

    /// Insert data into a table
    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status>;

    /// Query data from a table
    async fn query_table(&self, table_name: &str, projection: Option<Vec<String>>) -> Result<RecordBatch, Status>;

    /// Create an aggregation view
    async fn create_aggregation_view(&self, view: &HyprAggregationView) -> Result<(), Status>;

    /// Query data from an aggregation view
    async fn query_aggregation_view(&self, view_name: &str) -> Result<RecordBatch, Status>;

    /// Drop a table
    async fn drop_table(&self, table_name: &str) -> Result<(), Status>;

    /// Drop an aggregation view
    async fn drop_aggregation_view(&self, view_name: &str) -> Result<(), Status>;

    /// Get the table manager instance
    fn table_manager(&self) -> &HyprTableManager;

    /// Update batch-level aggregations.
    /// This is called during batch writes to maintain running aggregations.
    async fn update_batch_aggregations(
        &self,
        batch: &[HyprMetricRecord],
        window: HyprTimeWindow,
    ) -> Result<Vec<HyprBatchAggregation>, Status> {
        // Default implementation that processes the batch and updates aggregations
        let mut aggregations = HashMap::new();

        for metric in batch {
            let (window_start, window_end) = window.window_bounds(metric.timestamp);
            let key = (metric.metric_id.clone(), window_start, window_end);

            let agg = aggregations.entry(key).or_insert_with(|| HyprBatchAggregation {
                metric_id: metric.metric_id.clone(),
                window_start,
                window_end,
                running_sum: 0.0,
                running_count: 0,
                min_value: f64::INFINITY,
                max_value: f64::NEG_INFINITY,
            });

            // Update running aggregations
            agg.running_sum += metric.value_running_window_sum;
            agg.running_count += 1;
            agg.min_value = agg.min_value.min(metric.value_running_window_sum);
            agg.max_value = agg.max_value.max(metric.value_running_window_sum);
        }

        Ok(aggregations.into_values().collect())
    }

    /// Insert batch-level aggregations.
    /// This is called after update_batch_aggregations to persist the aggregations.
    async fn insert_batch_aggregations(
        &self,
        aggregations: Vec<HyprBatchAggregation>,
    ) -> Result<(), Status> {
        // Default implementation that stores aggregations in a separate table
        let mut batch = Vec::new();
        for agg in aggregations {
            batch.push(HyprMetricRecord {
                metric_id: agg.metric_id,
                timestamp: agg.window_start,
                value_running_window_sum: agg.running_sum,
                value_running_window_avg: agg.running_sum / agg.running_count as f64,
                value_running_window_count: agg.running_count as i16,
            });
        }
        self.insert_metrics(batch).await
    }
}

/// Available storage backend implementations
#[derive(Clone)]
pub enum HyprStorageBackendType {
    Adbc(adbc::AdbcBackend),
    DuckDb(duckdb::DuckDbBackend),
    Vector(vector::HyprVectorBackend),
}

#[async_trait::async_trait]
impl HyprStorageBackend for HyprStorageBackendType {
    async fn init(&self) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.init().await,
            HyprStorageBackendType::DuckDb(backend) => backend.init().await,
            HyprStorageBackendType::Vector(backend) => backend.init().await,
        }
    }

    async fn insert_metrics(&self, metrics: Vec<HyprMetricRecord>) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.insert_metrics(metrics).await,
            HyprStorageBackendType::DuckDb(backend) => backend.insert_metrics(metrics).await,
            HyprStorageBackendType::Vector(backend) => backend.insert_metrics(metrics).await,
        }
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<HyprMetricRecord>, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.query_metrics(from_timestamp).await,
            HyprStorageBackendType::DuckDb(backend) => backend.query_metrics(from_timestamp).await,
            HyprStorageBackendType::Vector(backend) => backend.query_metrics(from_timestamp).await,
        }
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.prepare_sql(query).await,
            HyprStorageBackendType::DuckDb(backend) => backend.prepare_sql(query).await,
            HyprStorageBackendType::Vector(backend) => backend.prepare_sql(query).await,
        }
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<HyprMetricRecord>, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.query_sql(statement_handle).await,
            HyprStorageBackendType::DuckDb(backend) => backend.query_sql(statement_handle).await,
            HyprStorageBackendType::Vector(backend) => backend.query_sql(statement_handle).await,
        }
    }

    async fn aggregate_metrics(
        &self,
        function: HyprAggregateFunction,
        group_by: &HyprGroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<HyprAggregateResult>, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => {
                backend.aggregate_metrics(function, group_by, from_timestamp, to_timestamp).await
            },
            HyprStorageBackendType::DuckDb(backend) => {
                backend.aggregate_metrics(function, group_by, from_timestamp, to_timestamp).await
            },
            HyprStorageBackendType::Vector(backend) => {
                backend.aggregate_metrics(function, group_by, from_timestamp, to_timestamp).await
            },
        }
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&HyprCredentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized,
    {
        let engine_type = options.get("engine")
            .ok_or_else(|| Status::invalid_argument("Missing engine type"))?;

        match engine_type.as_str() {
            "adbc" => Ok(HyprStorageBackendType::Adbc(
                adbc::AdbcBackend::new_with_options(connection_string, options, credentials)?
            )),
            "duckdb" => Ok(HyprStorageBackendType::DuckDb(
                duckdb::DuckDbBackend::new_with_options(connection_string, options, credentials)?
            )),
            "vector" => Ok(HyprStorageBackendType::Vector(
                vector::HyprVectorBackend::new_with_options(connection_string, options, credentials)?
            )),
            _ => Err(Status::invalid_argument("Invalid engine type")),
        }
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.create_table(table_name, schema).await,
            HyprStorageBackendType::DuckDb(backend) => backend.create_table(table_name, schema).await,
            HyprStorageBackendType::Vector(backend) => backend.create_table(table_name, schema).await,
        }
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.insert_into_table(table_name, batch).await,
            HyprStorageBackendType::DuckDb(backend) => backend.insert_into_table(table_name, batch).await,
            HyprStorageBackendType::Vector(backend) => backend.insert_into_table(table_name, batch).await,
        }
    }

    async fn query_table(&self, table_name: &str, projection: Option<Vec<String>>) -> Result<RecordBatch, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.query_table(table_name, projection).await,
            HyprStorageBackendType::DuckDb(backend) => backend.query_table(table_name, projection).await,
            HyprStorageBackendType::Vector(backend) => backend.query_table(table_name, projection).await,
        }
    }

    async fn create_aggregation_view(&self, view: &HyprAggregationView) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.create_aggregation_view(view).await,
            HyprStorageBackendType::DuckDb(backend) => backend.create_aggregation_view(view).await,
            HyprStorageBackendType::Vector(backend) => backend.create_aggregation_view(view).await,
        }
    }

    async fn query_aggregation_view(&self, view_name: &str) -> Result<RecordBatch, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.query_aggregation_view(view_name).await,
            HyprStorageBackendType::DuckDb(backend) => backend.query_aggregation_view(view_name).await,
            HyprStorageBackendType::Vector(backend) => backend.query_aggregation_view(view_name).await,
        }
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.drop_table(table_name).await,
            HyprStorageBackendType::DuckDb(backend) => backend.drop_table(table_name).await,
            HyprStorageBackendType::Vector(backend) => backend.drop_table(table_name).await,
        }
    }

    async fn drop_aggregation_view(&self, view_name: &str) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.drop_aggregation_view(view_name).await,
            HyprStorageBackendType::DuckDb(backend) => backend.drop_aggregation_view(view_name).await,
            HyprStorageBackendType::Vector(backend) => backend.drop_aggregation_view(view_name).await,
        }
    }

    fn table_manager(&self) -> &HyprTableManager {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.table_manager(),
            HyprStorageBackendType::DuckDb(backend) => backend.table_manager(),
            HyprStorageBackendType::Vector(backend) => backend.table_manager(),
        }
    }

    async fn update_batch_aggregations(
        &self,
        batch: &[HyprMetricRecord],
        window: HyprTimeWindow,
    ) -> Result<Vec<HyprBatchAggregation>, Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.update_batch_aggregations(batch, window).await,
            HyprStorageBackendType::DuckDb(backend) => backend.update_batch_aggregations(batch, window).await,
            HyprStorageBackendType::Vector(backend) => backend.update_batch_aggregations(batch, window).await,
        }
    }

    async fn insert_batch_aggregations(
        &self,
        aggregations: Vec<HyprBatchAggregation>,
    ) -> Result<(), Status> {
        match self {
            HyprStorageBackendType::Adbc(backend) => backend.insert_batch_aggregations(aggregations).await,
            HyprStorageBackendType::DuckDb(backend) => backend.insert_batch_aggregations(aggregations).await,
            HyprStorageBackendType::Vector(backend) => backend.insert_batch_aggregations(aggregations).await,
        }
    }
}

// Re-export types with Hypr prefix
pub use crate::metrics::MetricRecord as HyprMetricRecord;
pub use crate::config::Credentials as HyprCredentials;
pub use crate::aggregation::{
    AggregateFunction as HyprAggregateFunction,
    GroupBy as HyprGroupBy,
    AggregateResult as HyprAggregateResult,
    TimeWindow as HyprTimeWindow,
};
