use crate::storage::{MetricRecord, StorageBackend};
use async_trait::async_trait;
use duckdb::{params, Connection};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

pub struct DuckDbBackend {
    conn: Arc<Mutex<Connection>>,
}

impl DuckDbBackend {
    pub fn new() -> Self {
        let conn = Connection::open_in_memory().unwrap();
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }
}

#[async_trait]
impl StorageBackend for DuckDbBackend {
    async fn init(&self) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                valueRunningWindowSum DOUBLE NOT NULL,
                valueRunningWindowAvg DOUBLE NOT NULL,
                valueRunningWindowCount BIGINT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_metrics_metric_id ON metrics(metric_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_combined ON metrics(metric_id, timestamp);",
        )
        .map_err(|e| Status::internal(format!("Failed to create table and indexes: {}", e)))
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        let mut stmt = tx.prepare(
            "INSERT INTO metrics (metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount) 
             VALUES (?1, ?2, ?3, ?4, ?5)"
        ).map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        for metric in metrics {
            stmt.execute(params![
                metric.metric_id,
                metric.timestamp,
                metric.value_running_window_sum,
                metric.value_running_window_avg,
                metric.value_running_window_count,
            ])
            .map_err(|e| Status::internal(format!("Failed to insert metric: {}", e)))?;
        }

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount 
             FROM metrics 
             WHERE timestamp >= ?1 
             ORDER BY timestamp ASC, metric_id ASC
             LIMIT 100"
        ).map_err(|e| Status::internal(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map([from_timestamp], |row| {
                Ok(MetricRecord {
                    metric_id: row.get(0)?,
                    timestamp: row.get(1)?,
                    value_running_window_sum: row.get(2)?,
                    value_running_window_avg: row.get(3)?,
                    value_running_window_count: row.get(4)?,
                })
            })
            .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Status::internal(format!("Row mapping failed: {}", e)))?);
        }
        Ok(results)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        // For DuckDB, we can validate and prepare the SQL directly
        let conn = self.conn.lock().await;
        conn.prepare(query)
            .map_err(|e| Status::invalid_argument(format!("Invalid SQL: {}", e)))?;

        // Return the validated SQL as the handle
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let sql = String::from_utf8(statement_handle.to_vec())
            .map_err(|e| Status::internal(format!("Invalid SQL handle: {}", e)))?;

        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(MetricRecord {
                    metric_id: row.get(0)?,
                    timestamp: row.get(1)?,
                    value_running_window_sum: row.get(2)?,
                    value_running_window_avg: row.get(3)?,
                    value_running_window_count: row.get(4)?,
                })
            })
            .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Status::internal(format!("Row mapping failed: {}", e)))?);
        }
        Ok(results)
    }
}
