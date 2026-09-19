// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use crate::storage::cache::{CacheEviction, CacheManager};
use crate::storage::view::{ViewDefinition, ViewMetadata};
use crate::storage::{Credentials, StorageBackend};
use duckdb::arrow::array::{Array, ArrayRef, BinaryArray, Float64Array, Int64Array, RecordBatch, StringArray};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use async_trait::async_trait;
use duckdb::{params, Config, Connection};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;
use crate::error::DuckDbErrorWrapper;
use tonic::Status;

/// DuckDB-based storage backend
#[derive(Clone)]
pub struct DuckDbBackend {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    connection_string: String,
    #[allow(dead_code)]
    options: HashMap<String, String>,
    #[allow(dead_code)]
    cache_manager: CacheManager,
}

impl DuckDbBackend {
    pub fn new(
        connection_string: String,
        options: HashMap<String, String>,
        ttl: Option<u64>,
    ) -> Result<Self, Status> {
        let config = Config::default();
        let conn = Connection::open_with_flags(&connection_string, config)
            .map_err(|e| Status::internal(e.to_string()))?;

        // Initialize tables synchronously
        // Note: Using VARCHAR for view_definition instead of JSON to avoid
        // requiring the json extension. We handle JSON via serde anyway.
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS view_metadata (
                view_name VARCHAR PRIMARY KEY,
                source_table VARCHAR NOT NULL,
                view_definition VARCHAR NOT NULL,
                created_at BIGINT NOT NULL
            );
            "#,
        )
        .map_err(|e| Status::internal(format!("Failed to create tables: {e}")))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            connection_string,
            options,
            cache_manager: CacheManager::new(ttl),
        })
    }

    pub fn new_in_memory() -> Result<Self, Status> {
        Self::new(":memory:".to_owned(), HashMap::new(), Some(0))
    }

    async fn execute_statement(&self, sql: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(sql)
            .map_err(|e| Status::internal(format!("Failed to execute statement: {e}")))
    }
}

#[async_trait]
impl StorageBackend for DuckDbBackend {
    async fn init(&self) -> Result<(), Status> {
        // Tables are already created in new()
        Ok(())
    }


    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
        let sql = std::str::from_utf8(statement_handle)
            .map_err(|e| Status::internal(format!("Invalid SQL statement: {e}")))?;
        
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {e}")))?;

        // Use query_arrow to get RecordBatch directly. The stream may yield
        // several chunks (e.g. one per radix partition of a grouped
        // aggregate), so concatenate all of them rather than taking the first.
        let batches: Vec<RecordBatch> = stmt
            .query_arrow(params![])
            .map_err(|e| Status::internal(format!("Failed to execute query: {e}")))?
            .collect();

        if let Some(first) = batches.first() {
            let schema = first.schema();
            duckdb::arrow::compute::concat_batches(&schema, batches.iter())
                .map_err(|e| Status::internal(format!("Failed to concatenate result chunks: {e}")))
        } else {
                // Create empty batch with schema from statement
                let schema = Arc::new(Schema::new(
                    stmt.column_names()
                        .iter()
                        .map(|col| {
                            Field::new(
                                col,
                                match col.to_uppercase().as_str() {
                                    "INTEGER" | "BIGINT" => DataType::Int64,
                                    "DOUBLE" | "REAL" | "FLOAT" => DataType::Float64,
                                    // VARCHAR, TEXT, and all other types fallback to string
                                    _ => DataType::Utf8,
                                },
                                true,
                            )
                        })
                        .collect::<Vec<Field>>(),
                ));
                let empty_arrays: Vec<ArrayRef> = schema
                    .fields()
                    .iter()
                    .map(|field| match field.data_type() {
                        DataType::Int64 => Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                        DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                        _ => Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef,
                    })
                    .collect();
                RecordBatch::try_new(schema, empty_arrays)
                    .map_err(|e| Status::internal(format!("Failed to create empty batch: {e}")))
        }
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut all_options = options.clone();
        if let Some(creds) = credentials {
            if let Some(ref username) = creds.username {
                all_options.insert("username".to_owned(), username.clone());
            }
            if let Some(ref password) = creds.password {
                all_options.insert("password".to_owned(), password.clone());
            }
        }

        let ttl = all_options
            .get("ttl")
            .and_then(|s| s.parse().ok())
            .map(|ttl| if ttl == 0 { None } else { Some(ttl) })
            .unwrap_or(None);

        Self::new(connection_string.to_owned(), all_options, ttl)
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        let sql = crate::storage::StorageUtils::generate_create_table_sql(table_name, schema)?;
        self.execute_statement(&sql).await
    }

    async fn execute_ddl(&self, sql: &str) -> Result<(), Status> {
        self.execute_statement(sql).await
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {e}")))?;

        let sql = crate::storage::StorageUtils::generate_insert_sql(table_name, batch.num_columns());

        let mut stmt = tx
            .prepare(&sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {e}")))?;

        for row_idx in 0..batch.num_rows() {
            let mut params: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                match col.data_type() {
                    DataType::Int64 => {
                        let array = col.as_any().downcast_ref::<Int64Array>()
                            .ok_or_else(|| Status::internal("Failed to downcast to Int64Array"))?;
                        if array.is_null(row_idx) {
                            params.push(Box::new(Option::<i64>::None));
                        } else {
                            params.push(Box::new(array.value(row_idx)));
                        }
                    }
                    DataType::Float64 => {
                        let array = col.as_any().downcast_ref::<Float64Array>()
                            .ok_or_else(|| Status::internal("Failed to downcast to Float64Array"))?;
                        if array.is_null(row_idx) {
                            params.push(Box::new(Option::<f64>::None));
                        } else {
                            params.push(Box::new(array.value(row_idx)));
                        }
                    }
                    DataType::Utf8 => {
                        let array = col.as_any().downcast_ref::<StringArray>()
                            .ok_or_else(|| Status::internal("Failed to downcast to StringArray"))?;
                        if array.is_null(row_idx) {
                            params.push(Box::new(Option::<String>::None));
                        } else {
                            params.push(Box::new(array.value(row_idx).to_owned()));
                        }
                    }
                    DataType::Binary => {
                        let array = col.as_any().downcast_ref::<BinaryArray>()
                            .ok_or_else(|| Status::internal("Failed to downcast to BinaryArray"))?;
                        if array.is_null(row_idx) {
                            params.push(Box::new(Option::<Vec<u8>>::None));
                        } else {
                            params.push(Box::new(array.value(row_idx).to_vec()));
                        }
                    }
                    _ => return Err(Status::invalid_argument("Unsupported data type")),
                }
            }

            let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(std::convert::AsRef::as_ref).collect();
            stmt.execute(param_refs.as_slice())
                .map_err(|e| Status::internal(format!("Failed to insert row: {e}")))?;
        }

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {e}")))?;

        Ok(())
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {e}")))?;

        // Create SQL view
        let create_view_sql = crate::storage::StorageUtils::generate_view_sql(name, &definition);
        tx.execute(&create_view_sql, params![])
            .map_err(|e| Status::internal(format!("Failed to create view: {e}")))?;

        // Store view metadata
        let metadata = ViewMetadata {
            name: name.to_owned(),
            definition: definition.clone(),
            created_at: SystemTime::now(),
        };

        let definition_json = serde_json::to_string(&definition)
            .map_err(|e| Status::internal(format!("Failed to serialize view definition: {e}")))?;

        let created_at_secs = i64::try_from(
            metadata.created_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|e| Status::internal(format!("Failed to get timestamp: {e}")))?
                .as_secs()
        ).unwrap_or(i64::MAX);

        tx.execute(
            "INSERT INTO view_metadata (view_name, source_table, view_definition, created_at) VALUES (?, ?, ?, ?)",
            params![
                name,
                definition.source_table,
                definition_json,
                created_at_secs
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to store view metadata: {e}")))?;

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {e}")))?;

        Ok(())
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT * FROM view_metadata WHERE view_name = ?")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {e}")))?;

        let mut rows = stmt
            .query_map(params![name], |row| {
                let definition_json: String = row.get(2)?;
                let definition: ViewDefinition = serde_json::from_str(&definition_json)
                    .map_err(|e| duckdb::Error::InvalidParameterName(format!("Invalid view definition: {e}")))?;

                let secs = row.get::<_, i64>(3)?;
                let created_at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(u64::try_from(secs).unwrap_or(0));

                Ok(ViewMetadata {
                    name: row.get(0)?,
                    definition,
                    created_at,
                })
            })
            .map_err(|e| Status::internal(format!("Failed to execute query: {e}")))?;

        if let Some(row) = rows.next() {
            row.map_err(|e| Status::internal(format!("Failed to read row: {e}")))
        } else {
            Err(Status::not_found(format!("View {name} not found")))
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT view_name FROM view_metadata")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {e}")))?;

        let rows = stmt
            .query_map(params![], |row| row.get::<_, String>(0))
            .map_err(|e| Status::internal(format!("Failed to execute query: {e}")))?;

        let mut views = Vec::new();
        for row in rows {
            views.push(row.map_err(|e| Status::internal(format!("Failed to read row: {e}")))?);
        }

        Ok(views)
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {e}")))?;

        // Drop the view
        tx.execute(&format!("DROP VIEW IF EXISTS {name}"), params![])
            .map_err(|e| Status::internal(format!("Failed to drop view: {e}")))?;

        // Remove metadata
        tx.execute(
            "DELETE FROM view_metadata WHERE view_name = ?",
            params![name],
        )
        .map_err(|e| Status::internal(format!("Failed to remove view metadata: {e}")))?;

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {e}")))?;

        Ok(())
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        let sql = format!("DROP TABLE IF EXISTS {table_name}");
        self.execute_statement(&sql).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {e}")))?;

        let rows = stmt
            .query_map(params![], |row| row.get::<_, String>(0))
            .map_err(|e| Status::internal(format!("Failed to execute query: {e}")))?;

        let mut tables = Vec::new();
        for row in rows {
            tables.push(row.map_err(|e| Status::internal(format!("Failed to read row: {e}")))?);
        }

        Ok(tables)
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table_name})"))
            .map_err(|e| Into::<Status>::into(DuckDbErrorWrapper(e)))?;

        let mut fields = Vec::new();
        let rows = stmt.query_map(params![], |row| {
            let name: String = row.get(1)?;
            let type_str: String = row.get(2)?;
            let nullable: bool = row.get(3)?;

            let data_type = match type_str.to_uppercase().as_str() {
                "INTEGER" | "BIGINT" => DataType::Int64,
                "DOUBLE" | "REAL" => DataType::Float64,
                _ => DataType::Utf8,
            };

            Ok((name, data_type, !nullable))
        })
        .map_err(|e| Into::<Status>::into(DuckDbErrorWrapper(e)))?;

        for row in rows {
            let (name, data_type, required) = row.map_err(|e| Into::<Status>::into(DuckDbErrorWrapper(e)))?;
            fields.push(Field::new(name, data_type, required));
        }

        Ok(Arc::new(Schema::new(fields)))
    }

    async fn export_to_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Status::internal(format!("Failed to create export directory: {e}"))
            })?;
        }

        let sql = format!(
            "COPY {} TO '{}' (FORMAT PARQUET)",
            table_name,
            path.display()
        );
        self.execute_statement(&sql).await
    }

    async fn import_from_parquet(
        &self,
        table_name: &str,
        path: &std::path::Path,
    ) -> Result<(), Status> {
        if !path.exists() {
            return Err(Status::not_found(format!(
                "Parquet file not found: {}",
                path.display()
            )));
        }

        let sql = format!(
            "COPY {} FROM '{}' (FORMAT PARQUET)",
            table_name,
            path.display()
        );
        self.execute_statement(&sql).await
    }
}

#[async_trait]
impl CacheEviction for DuckDbBackend {
    async fn execute_eviction(&self, query: &str) -> Result<(), Status> {
        self.execute_statement(query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: query_sql must concatenate every arrow chunk — DuckDB emits
    /// at most STANDARD_VECTOR_SIZE (2048) rows per chunk, so a first-chunk-only
    /// read silently truncates any result set larger than that.
    #[tokio::test]
    async fn test_query_sql_returns_all_rows_beyond_one_chunk() -> Result<(), Status> {
        const ROWS: i64 = 5_000;
        let backend = DuckDbBackend::new_in_memory()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        backend.create_table("big", &schema).await?;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..ROWS)) as ArrayRef,
                Arc::new(StringArray::from_iter(
                    (0..ROWS).map(|i| (i % 7 == 0).then(|| format!("n{i}"))),
                )) as ArrayRef,
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))?;
        backend.insert_into_table("big", batch).await?;

        let handle = backend.prepare_sql("SELECT * FROM big").await?;
        let result = backend.query_sql(&handle).await?;
        assert_eq!(
            result.num_rows() as i64,
            ROWS,
            "plain SELECT larger than one chunk must return every row"
        );

        // Grouped results stream one chunk per radix partition.
        let handle = backend
            .prepare_sql("SELECT id, COUNT(*) AS n FROM big GROUP BY id")
            .await?;
        let result = backend.query_sql(&handle).await?;
        assert_eq!(
            result.num_rows() as i64,
            ROWS,
            "GROUP BY must return every group"
        );
        Ok(())
    }
}
