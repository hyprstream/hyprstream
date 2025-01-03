use async_trait::async_trait;
use tonic::Status;

pub mod adbc;
pub mod duckdb;
pub mod cached;

#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub metric_id: String,
    pub timestamp: i64,
    pub value_running_window_sum: f64,
    pub value_running_window_avg: f64,
    pub value_running_window_count: i64,
}

#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    async fn init(&self) -> Result<(), Status>;
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status>;
    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status>;
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status>;
}

pub use self::adbc::AdbcBackend;
pub use self::duckdb::DuckDbBackend;
pub use self::cached::CachedStorageBackend;
