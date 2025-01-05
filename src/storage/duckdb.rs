use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb::Connection;
use std::sync::{Arc, Mutex};
use tonic::Status;

use crate::storage::MetricRecord;

pub struct DuckDbBackend {
    conn: Arc<Mutex<Connection>>,
}

impl DuckDbBackend {
    pub fn new() -> Self {
        Self {
            conn: Arc::new(Mutex::new(Connection::open_in_memory().unwrap())),
        }
    }

    async fn create_tables(&self) -> Result<(), Status> {
        let conn = self.conn.lock().unwrap();
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

    async fn get_connection(&self) -> Result<std::sync::MutexGuard<'_, Connection>, Status> {
        Ok(self.conn.lock().unwrap())
    }

    fn metrics_to_record_batch(&self, metrics: Vec<MetricRecord>) -> Result<RecordBatch, Status> {
        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
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
}

#[async_trait::async_trait]
impl super::StorageBackend for DuckDbBackend {
    async fn init(&self) -> Result<(), Status> {
        self.create_tables().await
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let batch = self.metrics_to_record_batch(metrics)?;
        self.upsert_batch(&batch).await
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let sql = format!(
            "SELECT metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count \
             FROM metrics WHERE timestamp >= {}",
            from_timestamp
        );
        let sql_bytes = self.prepare_sql(&sql).await?;
        self.query_sql(&sql_bytes).await
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // DuckDB doesn't support prepared statements in the same way as ADBC
        // We'll store the SQL string as bytes
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        // Convert bytes back to SQL string
        let sql = std::str::from_utf8(statement_handle)
            .map_err(|e| Status::internal(format!("Invalid UTF-8 in statement handle: {}", e)))?;

        self.execute_query(sql).await
    }
}
