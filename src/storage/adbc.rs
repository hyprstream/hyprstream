use crate::storage::{MetricRecord, StorageBackend};
use adbc_core::{
    Connection, Database, Statement,
    connection::{GetInfoResult, ObjectDepth},
    options::{Optionable, OptionConnection},
    schema::Schema,
    RecordBatchReader,
};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use std::{collections::HashSet, sync::Arc};
use tokio::sync::Mutex;
use tonic::Status;

pub struct AdbcBackend<C: Connection> {
    conn: Arc<Mutex<C>>,
}

impl<C: Connection> AdbcBackend<C> {
    pub fn new(connection: C) -> Self {
        Self {
            conn: Arc::new(Mutex::new(connection)),
        }
    }

    async fn get_statement(&self) -> Result<C::StatementType, Status> {
        let mut conn = self.conn.lock().await;
        conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))
    }
}

#[async_trait]
impl<C: Connection + Send + Sync + 'static> StorageBackend for AdbcBackend<C> {
    async fn init(&self) -> Result<(), Status> {
        let mut stmt = self.get_statement().await?;

        // Create the metrics table
        stmt.set_sql_query(
            "CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                valueRunningWindowSum DOUBLE NOT NULL,
                valueRunningWindowAvg DOUBLE NOT NULL,
                valueRunningWindowCount BIGINT NOT NULL
            )"
        ).map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to create table: {}", e)))?;

        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let mut stmt = self.get_statement().await?;

        // Prepare the insert statement
        stmt.set_sql_query(
            "INSERT INTO metrics (
                metric_id, timestamp, valueRunningWindowSum, 
                valueRunningWindowAvg, valueRunningWindowCount
            ) VALUES (?, ?, ?, ?, ?)"
        ).map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        // Insert each metric
        for metric in metrics {
            stmt.bind_parameters(&[
                metric.metric_id.into(),
                metric.timestamp.into(),
                metric.value_running_window_sum.into(),
                metric.value_running_window_avg.into(),
                metric.value_running_window_count.into(),
            ]).map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

            stmt.execute()
                .map_err(|e| Status::internal(format!("Failed to execute insert: {}", e)))?;
        }

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(
            "SELECT metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount 
             FROM metrics 
             WHERE timestamp >= ? 
             ORDER BY timestamp ASC, metric_id ASC
             LIMIT 100"
        ).map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        stmt.bind_parameters(&[from_timestamp.into()])
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        let result = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(row) = result.fetch_next()
            .map_err(|e| Status::internal(format!("Failed to fetch row: {}", e)))? {
            
            metrics.push(MetricRecord {
                metric_id: row.get_string(0)
                    .map_err(|e| Status::internal(format!("Failed to get metric_id: {}", e)))?,
                timestamp: row.get_int64(1)
                    .map_err(|e| Status::internal(format!("Failed to get timestamp: {}", e)))?,
                value_running_window_sum: row.get_float64(2)
                    .map_err(|e| Status::internal(format!("Failed to get sum: {}", e)))?,
                value_running_window_avg: row.get_float64(3)
                    .map_err(|e| Status::internal(format!("Failed to get avg: {}", e)))?,
                value_running_window_count: row.get_int64(4)
                    .map_err(|e| Status::internal(format!("Failed to get count: {}", e)))?,
            });
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(query)
            .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        // Get the prepared statement handle
        stmt.prepare()
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let sql = String::from_utf8(statement_handle.to_vec())
            .map_err(|e| Status::internal(format!("Invalid SQL handle: {}", e)))?;

        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(&sql)
            .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        let result = stmt.execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(row) = result.fetch_next()
            .map_err(|e| Status::internal(format!("Failed to fetch row: {}", e)))? {
            
            metrics.push(MetricRecord {
                metric_id: row.get_string(0)
                    .map_err(|e| Status::internal(format!("Failed to get metric_id: {}", e)))?,
                timestamp: row.get_int64(1)
                    .map_err(|e| Status::internal(format!("Failed to get timestamp: {}", e)))?,
                value_running_window_sum: row.get_float64(2)
                    .map_err(|e| Status::internal(format!("Failed to get sum: {}", e)))?,
                value_running_window_avg: row.get_float64(3)
                    .map_err(|e| Status::internal(format!("Failed to get avg: {}", e)))?,
                value_running_window_count: row.get_int64(4)
                    .map_err(|e| Status::internal(format!("Failed to get count: {}", e)))?,
            });
        }

        Ok(metrics)
    }
} 