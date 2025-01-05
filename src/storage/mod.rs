pub mod adbc;
pub mod cached;
pub mod duckdb;

use crate::metrics::MetricRecord;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use tonic::Status;

#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    async fn init(&self) -> Result<(), Status>;
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;

    /// Prepare a SQL statement. Returns a serialized statement handle.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Execute a prepared SQL statement using its serialized handle.
    /// The implementation should ensure zero-copy operations where possible.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;

    /// Convert a RecordBatch to MetricRecords - provided as a default implementation
    /// to ensure consistent conversion across backends
    fn record_batch_to_metrics(&self, batch: &RecordBatch) -> Result<Vec<MetricRecord>, Status> {
        let metric_ids = batch
            .column_by_name("metric_id")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Int64Array>())
            .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

        let sums = batch
            .column_by_name("value_running_window_sum")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

        let avgs = batch
            .column_by_name("value_running_window_avg")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

        let counts = batch
            .column_by_name("value_running_window_count")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::Int64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

        let mut metrics = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            metrics.push(MetricRecord {
                metric_id: metric_ids.value(i).to_string(),
                timestamp: timestamps.value(i),
                value_running_window_sum: sums.value(i),
                value_running_window_avg: avgs.value(i),
                value_running_window_count: counts.value(i),
            });
        }

        Ok(metrics)
    }
}
