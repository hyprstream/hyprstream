//! Storage backends for SQL database operations.
//!
//! This module provides multiple storage backend implementations:
//! - `duckdb`: High-performance embedded SQL database
//! - `adbc`: Arrow Database Connectivity for external database integration (optional)
//! - `cached`: Two-tier storage with configurable caching layer
//! - `datafusion_provider`: DataFusion TableProvider bridge for DuckDB
//! - `context`: Embedding storage for RAG/CAG functionality
//!
//! Each backend implements the `StorageBackend` trait, providing a consistent
//! interface for SQL operations like table management, querying, and data insertion.

pub mod cache;
pub mod cached;
pub mod context;
pub mod datafusion_provider;
pub mod duckdb;
pub mod table_manager;
pub mod view;

#[cfg(feature = "adbc")]
pub mod adbc;

use self::{
    cached::CachedStorageBackend,
    duckdb::DuckDbBackend,
    view::{ViewDefinition, ViewMetadata},
};
#[cfg(feature = "adbc")]
use self::adbc::AdbcBackend;

pub use datafusion_provider::{DuckDBExec, DuckDBTableProvider};

use ::duckdb::arrow::array::RecordBatch;
use ::duckdb::arrow::datatypes::{DataType, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

/// Credentials for database authentication
#[derive(Debug, Clone, Default)]
pub struct Credentials {
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Clone)]
pub enum StorageBackendType {
    DuckDb(DuckDbBackend),
    #[cfg(feature = "adbc")]
    Adbc(AdbcBackend),
    Cached(CachedStorageBackend),
}

/// Utility functions for SQL operations
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

    /// Generate SQL for selecting data from a table
    pub fn generate_select_sql(table_name: &str, projection: Option<Vec<String>>) -> String {
        let columns = projection.map(|cols| cols.join(", ")).unwrap_or_else(|| "*".to_string());
        format!("SELECT {} FROM {}", columns, table_name)
    }

    /// Generate SQL for creating a view
    pub fn generate_view_sql(name: &str, definition: &ViewDefinition) -> String {
        format!("CREATE VIEW {} AS {}", name, definition.to_sql())
    }
}

/// Storage backend trait for SQL database operations.
#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    /// Initialize the storage backend.
    async fn init(&self) -> Result<(), Status>;

    /// Prepare a SQL query and return a handle.
    /// The handle can be used with query_sql to execute the prepared statement.
    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status>;

    /// Execute a prepared SQL query using its handle.
    /// Returns results as an Arrow record batch.
    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status>;

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

    /// Insert a record batch into a table
    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status>;

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

    /// Export a table to Parquet format
    async fn export_to_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status>;

    /// Import a table from Parquet format
    async fn import_from_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status>;
}

impl AsRef<dyn StorageBackend> for StorageBackendType {
    fn as_ref(&self) -> &(dyn StorageBackend + 'static) {
        match self {
            #[cfg(feature = "adbc")]
            StorageBackendType::Adbc(backend) => backend,
            StorageBackendType::DuckDb(backend) => backend,
            StorageBackendType::Cached(backend) => backend,
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for StorageBackendType {
    async fn init(&self) -> Result<(), Status> {
        self.as_ref().init().await
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        self.as_ref().prepare_sql(query).await
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
        self.as_ref().query_sql(statement_handle).await
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
            #[cfg(feature = "adbc")]
            "adbc" => Ok(StorageBackendType::Adbc(
                adbc::AdbcBackend::new_with_options(connection_string, options, credentials)?,
            )),
            "duckdb" => Ok(StorageBackendType::DuckDb(
                duckdb::DuckDbBackend::new_with_options(connection_string, options, credentials)?,
            )),
            "cached" => {
                // Create cache backend (in-memory DuckDB)
                let cache = Arc::new(DuckDbBackend::new_in_memory()?);

                // Create store backend based on store_engine option
                let store_engine = options.get("store_engine")
                    .ok_or_else(|| Status::invalid_argument("Missing store_engine for cached backend"))?;

                let store: Arc<dyn StorageBackend> = match store_engine.as_str() {
                    #[cfg(feature = "adbc")]
                    "adbc" => Arc::new(AdbcBackend::new_with_options(connection_string, options, credentials)?),
                    "duckdb" => Arc::new(DuckDbBackend::new_with_options(connection_string, options, credentials)?),
                    _ => return Err(Status::invalid_argument("Invalid store_engine type")),
                };

                Ok(StorageBackendType::Cached(CachedStorageBackend::new(
                    cache,
                    store,
                    options.get("max_duration_secs")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(3600),
                )))
            }
            _ => Err(Status::invalid_argument("Invalid engine type")),
        }
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        self.as_ref().create_table(table_name, schema).await
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        self.as_ref().insert_into_table(table_name, batch).await
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        self.as_ref().create_view(name, definition).await
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        self.as_ref().get_view(name).await
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        self.as_ref().list_views().await
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        self.as_ref().drop_view(name).await
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        self.as_ref().drop_table(table_name).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        self.as_ref().list_tables().await
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        self.as_ref().get_table_schema(table_name).await
    }

    async fn export_to_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status> {
        self.as_ref().export_to_parquet(table_name, path).await
    }

    async fn import_from_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status> {
        self.as_ref().import_from_parquet(table_name, path).await
    }
}
