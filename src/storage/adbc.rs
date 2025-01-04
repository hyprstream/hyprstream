use crate::storage::{MetricRecord, StorageBackend};
use crate::config::AdbcConfig;
use adbc_core::{
    Connection, Database, Driver,
    driver_manager::{ManagedDriver, ManagedConnection, ManagedDatabase, ManagedStatement},
    options::{AdbcVersion, OptionDatabase, OptionValue},
    Statement,
};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, RecordBatchReader, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

pub struct AdbcBackend {
    conn: Arc<Mutex<ManagedConnection>>,
}

impl AdbcBackend {
    pub fn new(config: &AdbcConfig) -> Result<Self, Status> {
        let mut driver = ManagedDriver::load_dynamic_from_filename(&config.driver_path, None, AdbcVersion::V100)
            .map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let opts = vec![
            (OptionDatabase::Uri, config.url.as_str().into()),
            (OptionDatabase::Username, config.username.as_str().into()),
            (OptionDatabase::Password, config.password.as_str().into()),
            (OptionDatabase::Other("database".into()), config.database.as_str().into()),
            (
                OptionDatabase::Other("pool.max_connections".into()),
                config.pool.max_connections.to_string().as_str().into(),
            ),
            (
                OptionDatabase::Other("pool.min_connections".into()),
                config.pool.min_connections.to_string().as_str().into(),
            ),
            (
                OptionDatabase::Other("pool.acquire_timeout".into()),
                config.pool.acquire_timeout_secs.to_string().as_str().into(),
            ),
        ];

        let mut database = driver
            .new_database_with_opts(opts)
            .map_err(|e| Status::internal(format!("Failed to create database: {}", e)))?;

        let connection = database
            .new_connection()
            .map_err(|e| Status::internal(format!("Failed to create connection: {}", e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(connection)),
        })
    }

    async fn get_statement(&self) -> Result<ManagedStatement, Status> {
        let mut conn = self.conn.lock().await;
        conn.new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))
    }

    fn consume_reader(reader: &mut impl RecordBatchReader) -> Result<(), Status> {
        while let Some(Ok(batch)) = reader.next() {
            // Consume batch
            let _ = batch;
        }
        Ok(())
    }

    fn record_batch_to_metrics(batch: RecordBatch) -> Result<Vec<MetricRecord>, Status> {
        let mut metrics = Vec::with_capacity(batch.num_rows());

        // Get column arrays
        let metric_id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Status::internal("Failed to cast metric_id column"))?;
        let timestamp_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| Status::internal("Failed to cast timestamp column"))?;
        let sum_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Status::internal("Failed to cast sum column"))?;
        let avg_col = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Status::internal("Failed to cast avg column"))?;
        let count_col = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| Status::internal("Failed to cast count column"))?;

        // Convert each row to a MetricRecord
        for row_idx in 0..batch.num_rows() {
            let metric = MetricRecord {
                metric_id: metric_id_col.value(row_idx).to_string(),
                timestamp: timestamp_col.value(row_idx),
                value_running_window_sum: sum_col.value(row_idx),
                value_running_window_avg: avg_col.value(row_idx),
                value_running_window_count: count_col.value(row_idx),
            };
            metrics.push(metric);
        }

        Ok(metrics)
    }
}

#[async_trait]
impl StorageBackend for AdbcBackend {
    async fn init(&self) -> Result<(), Status> {
        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(
            "CREATE TABLE IF NOT EXISTS metrics (
                metric_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                valueRunningWindowSum DOUBLE PRECISION NOT NULL,
                valueRunningWindowAvg DOUBLE PRECISION NOT NULL,
                valueRunningWindowCount BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            )",
        )
        .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to create table: {}", e)))?;
        Self::consume_reader(&mut reader)?;

        // Create each index with a new statement to avoid mutable borrow issues
        for query in &[
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_metric_id ON metrics(metric_id)",
        ] {
            let mut stmt = self.get_statement().await?;
            stmt.set_sql_query(query)
                .map_err(|e| Status::internal(format!("Failed to set index query: {}", e)))?;
            let mut reader = stmt
                .execute()
                .map_err(|e| Status::internal(format!("Failed to create index: {}", e)))?;
            Self::consume_reader(&mut reader)?;
        }

        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        if metrics.is_empty() {
            return Ok(());
        }

        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(
            "INSERT INTO metrics (
                metric_id, timestamp, valueRunningWindowSum, 
                valueRunningWindowAvg, valueRunningWindowCount
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (metric_id, timestamp) DO UPDATE SET
                valueRunningWindowSum = EXCLUDED.valueRunningWindowSum,
                valueRunningWindowAvg = EXCLUDED.valueRunningWindowAvg,
                valueRunningWindowCount = EXCLUDED.valueRunningWindowCount",
        )
        .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        for metric in metrics {
            let params: Vec<Arc<dyn arrow_array::Array>> = vec![
                Arc::new(StringArray::from(vec![metric.metric_id])),
                Arc::new(Int64Array::from(vec![metric.timestamp])),
                Arc::new(Float64Array::from(vec![metric.value_running_window_sum])),
                Arc::new(Float64Array::from(vec![metric.value_running_window_avg])),
                Arc::new(Int64Array::from(vec![metric.value_running_window_count])),
            ];

            let schema = Schema::new(vec![
                Field::new("metric_id", DataType::Utf8, false),
                Field::new("timestamp", DataType::Int64, false),
                Field::new("valueRunningWindowSum", DataType::Float64, false),
                Field::new("valueRunningWindowAvg", DataType::Float64, false),
                Field::new("valueRunningWindowCount", DataType::Int64, false),
            ]);

            let batch = RecordBatch::try_new(Arc::new(schema), params)
                .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

            stmt.bind(batch)
                .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

            let mut reader = stmt
                .execute()
                .map_err(|e| Status::internal(format!("Failed to execute insert: {}", e)))?;
            Self::consume_reader(&mut reader)?;
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
             LIMIT 100",
        )
        .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        let params: Vec<Arc<dyn arrow_array::Array>> =
            vec![Arc::new(Int64Array::from(vec![from_timestamp]))];

        let schema = Schema::new(vec![Field::new("timestamp", DataType::Int64, false)]);

        let batch = RecordBatch::try_new(Arc::new(schema), params)
            .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        stmt.bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(Ok(batch)) = reader.next() {
            let batch_metrics = Self::record_batch_to_metrics(batch)?;
            metrics.extend(batch_metrics);
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let mut stmt = self.get_statement().await?;

        stmt.set_sql_query(query)
            .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        // Prepare the statement
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

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(Ok(batch)) = reader.next() {
            let batch_metrics = Self::record_batch_to_metrics(batch)?;
            metrics.extend(batch_metrics);
        }

        Ok(metrics)
    }
}
