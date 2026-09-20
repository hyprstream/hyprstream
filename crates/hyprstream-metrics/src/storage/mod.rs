//! Storage backends for SQL database operations.
//!
//! This module provides multiple storage backend implementations:
//! - `duckdb`: High-performance embedded SQL database
//! - `adbc`: Arrow Database Connectivity for external database integration (optional)
//! - `cached`: Two-tier storage with configurable caching layer
//! - `datafusion_provider`: DataFusion TableProvider bridge for DuckDB
//! - `context`: Embedding storage for RAG/CAG functionality
//!
//! The `StorageBackend` trait itself, the view types, and the SQL string
//! utilities live in the Apache-2.0 `hyprstream-metrics-api` crate and are
//! re-exported here; each backend here implements that trait.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

pub mod cache;
pub mod cached;
pub mod context;
pub mod datafusion_provider;
pub mod duckdb;
pub mod table_manager;

#[cfg(feature = "adbc")]
pub mod adbc;

pub use hyprstream_metrics_api::storage::view;
pub use hyprstream_metrics_api::storage::{Credentials, StorageBackend, StorageUtils};

use self::{
    cached::CachedStorageBackend,
    duckdb::DuckDbBackend,
    view::{ViewDefinition, ViewMetadata},
};
#[cfg(feature = "adbc")]
use self::adbc::AdbcBackend;

pub use datafusion_provider::{DuckDBExec, DuckDBTableProvider};

use ::duckdb::arrow::array::RecordBatch;
use ::duckdb::arrow::datatypes::Schema;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

#[derive(Clone)]
pub enum StorageBackendType {
    DuckDb(DuckDbBackend),
    #[cfg(feature = "adbc")]
    Adbc(AdbcBackend),
    Cached(CachedStorageBackend),
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
                )))
            }
            _ => Err(Status::invalid_argument("Invalid engine type")),
        }
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        self.as_ref().create_table(table_name, schema).await
    }

    async fn execute_ddl(&self, sql: &str) -> Result<(), Status> {
        self.as_ref().execute_ddl(sql).await
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
