//! Storage backend abstraction for SQL database operations.
//!
//! The `StorageBackend` trait is the seam between the metrics API (this
//! crate) and the serving engine (`hyprstream-metrics`, AGPL): consumers link
//! these types and traits, while the serving process wires in an engine
//! implementation such as DuckDB.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

pub mod view;

use self::view::{ViewDefinition, ViewMetadata};
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
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

/// Utility functions for SQL operations
pub struct StorageUtils;

impl StorageUtils {
    /// Generate SQL for creating a table with the given schema
    pub fn generate_create_table_sql(table_name: &str, schema: &Schema) -> Result<String, Status> {
        let mut sql = format!("CREATE TABLE IF NOT EXISTS {table_name} (");
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
                    DataType::Binary => "BLOB",
                    _ => return Err(Status::invalid_argument(format!(
                        "Unsupported data type: {:?}",
                        field.data_type()
                    ))),
                }
            ));
        }

        sql.push(')');
        Ok(sql)
    }

    /// Generate SQL for inserting data into a table
    pub fn generate_insert_sql(table_name: &str, column_count: usize) -> String {
        let placeholders = vec!["?"; column_count].join(", ");
        format!("INSERT INTO {table_name} VALUES ({placeholders})")
    }

    /// Generate SQL for selecting data from a table
    pub fn generate_select_sql(table_name: &str, projection: Option<Vec<String>>) -> String {
        let columns = projection.map(|cols| cols.join(", ")).unwrap_or_else(|| "*".to_owned());
        format!("SELECT {columns} FROM {table_name}")
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

    /// Execute a DDL/administrative statement that produces no rows
    /// (e.g. `ALTER TABLE ...`). The default routes through prepare/query;
    /// backends with a dedicated DDL path should override it.
    async fn execute_ddl(&self, sql: &str) -> Result<(), Status> {
        let handle = self.prepare_sql(sql).await?;
        self.query_sql(&handle).await?;
        Ok(())
    }

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
