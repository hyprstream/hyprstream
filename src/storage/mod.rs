//! Storage backends for SQL database operations.
//!
//! This module provides multiple storage backend implementations:
//! - `duckdb`: High-performance embedded SQL database
//! - `adbc`: Arrow Database Connectivity for external database integration
//! - `cached`: Two-tier storage with configurable caching layer
//!
//! Each backend implements the `StorageBackend` trait, providing a consistent
//! interface for SQL operations like table management, querying, and data insertion.

pub mod adbc;
pub mod cache;
pub mod duckdb;
pub mod table_manager;
pub mod view;

use crate::cli::commands::config::Credentials;
use crate::storage::view::{ViewDefinition, ViewMetadata};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

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

    /// Insert data into a table
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
}

#[derive(Clone)]
pub enum StorageBackendType {
    Adbc(adbc::AdbcBackend),
    DuckDb(duckdb::DuckDbBackend),
}

impl AsRef<dyn StorageBackend> for StorageBackendType {
    fn as_ref(&self) -> &(dyn StorageBackend + 'static) {
        match self {
            StorageBackendType::Adbc(backend) => backend,
            StorageBackendType::DuckDb(backend) => backend,
        }
    }
}

#[async_trait::async_trait]
impl StorageBackend for StorageBackendType {
    async fn init(&self) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.init().await,
            StorageBackendType::DuckDb(backend) => backend.init().await,
        }
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.prepare_sql(query).await,
            StorageBackendType::DuckDb(backend) => backend.prepare_sql(query).await,
        }
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.query_sql(statement_handle).await,
            StorageBackendType::DuckDb(backend) => backend.query_sql(statement_handle).await,
        }
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
            "adbc" => Ok(StorageBackendType::Adbc(
                adbc::AdbcBackend::new_with_options(connection_string, options, credentials)?,
            )),
            "duckdb" => Ok(StorageBackendType::DuckDb(
                duckdb::DuckDbBackend::new_with_options(connection_string, options, credentials)?,
            )),
            _ => Err(Status::invalid_argument("Invalid engine type")),
        }
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.create_table(table_name, schema).await,
            StorageBackendType::DuckDb(backend) => backend.create_table(table_name, schema).await,
        }
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.insert_into_table(table_name, batch).await,
            StorageBackendType::DuckDb(backend) => backend.insert_into_table(table_name, batch).await,
        }
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.create_view(name, definition).await,
            StorageBackendType::DuckDb(backend) => backend.create_view(name, definition).await,
        }
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.get_view(name).await,
            StorageBackendType::DuckDb(backend) => backend.get_view(name).await,
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.list_views().await,
            StorageBackendType::DuckDb(backend) => backend.list_views().await,
        }
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.drop_view(name).await,
            StorageBackendType::DuckDb(backend) => backend.drop_view(name).await,
        }
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.drop_table(table_name).await,
            StorageBackendType::DuckDb(backend) => backend.drop_table(table_name).await,
        }
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.list_tables().await,
            StorageBackendType::DuckDb(backend) => backend.list_tables().await,
        }
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        match self {
            StorageBackendType::Adbc(backend) => backend.get_table_schema(table_name).await,
            StorageBackendType::DuckDb(backend) => backend.get_table_schema(table_name).await,
        }
    }
}
