use crate::config::AdbcConfig;
use crate::metrics::MetricRecord;
use crate::storage::StorageBackend;
use adbc_core::{
    driver_manager::{ManagedConnection, ManagedDriver},
    options::{AdbcVersion, OptionDatabase},
    Connection, Database, Driver, Statement,
};
use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

pub struct AdbcBackend {
    conn: Arc<Mutex<ManagedConnection>>,
    statement_counter: AtomicU64,
    prepared_statements: Arc<Mutex<Vec<(u64, String)>>>,
}

impl AdbcBackend {
    pub fn new(config: &AdbcConfig) -> Result<Self, Status> {
        let mut driver =
            ManagedDriver::load_dynamic_from_filename(&config.driver_path, None, AdbcVersion::V100)
                .map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let opts = vec![
            (OptionDatabase::Uri, config.url.as_str().into()),
            (OptionDatabase::Username, config.username.as_str().into()),
            (OptionDatabase::Password, config.password.as_str().into()),
            (
                OptionDatabase::Other("database".into()),
                config.database.as_str().into(),
            ),
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
            statement_counter: AtomicU64::new(0),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
        })
    }

    async fn get_connection(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, ManagedConnection>, Status> {
        Ok(self.conn.lock().await)
    }

    async fn create_tables(&self) -> Result<(), Status> {
        let mut conn = self.get_connection().await?;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE PRECISION NOT NULL,
                value_running_window_avg DOUBLE PRECISION NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            )",
        )
        .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute create table: {}", e)))?;

        Ok(())
    }

    fn metrics_to_record_batch(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
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
}

#[async_trait]
impl StorageBackend for AdbcBackend {
    async fn init(&self) -> Result<(), Status> {
        self.create_tables().await
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let batch = Self::metrics_to_record_batch(&metrics)?;

        let mut conn = self.get_connection().await?;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "INSERT INTO metrics (
                metric_id, timestamp, value_running_window_sum,
                value_running_window_avg, value_running_window_count
            ) VALUES (?, ?, ?, ?, ?)",
        )
        .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind values: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute insert: {}", e)))?;

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let mut conn = self.get_connection().await?;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "SELECT metric_id, timestamp, value_running_window_sum, value_running_window_avg, value_running_window_count 
             FROM metrics WHERE timestamp >= ?",
        )
        .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let param_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "timestamp",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from_iter_values([from_timestamp]))],
        )
        .map_err(|e| Status::internal(e.to_string()))?;

        stmt.bind(param_batch)
            .map_err(|e| Status::internal(format!("Failed to bind values: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result
                .map_err(|e| Status::internal(format!("Failed to get record batch: {}", e)))?;
            metrics.extend(self.record_batch_to_metrics(&batch)?);
        }

        if metrics.is_empty() {
            return Err(Status::not_found(
                "No metrics found for the given timestamp",
            ));
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let handle = self.statement_counter.fetch_add(1, Ordering::SeqCst);
        let mut statements = self.prepared_statements.lock().await;
        statements.push((handle, query.to_string()));

        Ok(handle.to_le_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let handle = u64::from_le_bytes(
            statement_handle
                .try_into()
                .map_err(|_| Status::invalid_argument("Invalid statement handle"))?,
        );

        let statements = self.prepared_statements.lock().await;
        let sql = statements
            .iter()
            .find(|(h, _)| *h == handle)
            .map(|(_, sql)| sql.as_str())
            .ok_or_else(|| Status::invalid_argument("Statement handle not found"))?;

        let mut conn = self.get_connection().await?;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        while let Some(batch_result) = reader.next() {
            let batch = batch_result
                .map_err(|e| Status::internal(format!("Failed to get record batch: {}", e)))?;
            metrics.extend(self.record_batch_to_metrics(&batch)?);
        }

        Ok(metrics)
    }
}
