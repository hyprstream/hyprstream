use crate::storage::cache::{CacheEviction, CacheManager};
use crate::storage::view::{ViewDefinition, ViewMetadata};
use crate::storage::{Credentials, StorageBackend};
use adbc_core::{
    driver_manager::ManagedConnection,
    Connection, Database, Driver, Optionable, Statement,
};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_array::builder::{StringBuilder, Int64Builder};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;
use tonic::Status;

#[derive(Clone)]
pub struct AdbcBackend {
    conn: Arc<Mutex<ManagedConnection>>,
    statement_counter: Arc<AtomicU64>,
    prepared_statements: Arc<Mutex<Vec<(u64, String)>>>,
    #[allow(dead_code)]
    cache_manager: CacheManager,
}

impl AdbcBackend {
    pub fn new(
        driver_path: &str,
        connection: Option<&str>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut driver = adbc_core::driver_manager::ManagedDriver::load_dynamic_from_filename(
            driver_path,
            None,
            adbc_core::options::AdbcVersion::V100,
        )
        .map_err(|e| Status::internal(format!("Failed to load ADBC driver: {}", e)))?;

        let mut database = Driver::new_database(&mut driver)
            .map_err(|e| Status::internal(format!("Failed to create database: {}", e)))?;

        if let Some(conn_str) = connection {
            database
                .set_option(
                    adbc_core::options::OptionDatabase::Uri,
                    adbc_core::options::OptionValue::String(conn_str.to_string()),
                )
                .map_err(|e| Status::internal(format!("Failed to set connection string: {}", e)))?;
        }

        if let Some(creds) = credentials {
            database
                .set_option(
                    adbc_core::options::OptionDatabase::Username,
                    adbc_core::options::OptionValue::String(creds.username.clone()),
                )
                .map_err(|e| Status::internal(format!("Failed to set username: {}", e)))?;

            database
                .set_option(
                    adbc_core::options::OptionDatabase::Password,
                    adbc_core::options::OptionValue::String(creds.password.clone()),
                )
                .map_err(|e| Status::internal(format!("Failed to set password: {}", e)))?;
        }

        let connection = database
            .new_connection()
            .map_err(|e| Status::internal(format!("Failed to create connection: {}", e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(connection)),
            statement_counter: Arc::new(AtomicU64::new(0)),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
            cache_manager: CacheManager::new(None),
        })
    }

    async fn execute_statement(&self, sql: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(sql)
            .map_err(|e| Status::internal(format!("Failed to set SQL query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute statement: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl CacheEviction for AdbcBackend {
    async fn execute_eviction(&self, query: &str) -> Result<(), Status> {
        self.execute_statement(query).await
    }
}

#[async_trait]
impl StorageBackend for AdbcBackend {
    async fn init(&self) -> Result<(), Status> {
        self.execute_statement(
            r#"
            CREATE TABLE IF NOT EXISTS view_metadata (
                view_name VARCHAR PRIMARY KEY,
                source_table VARCHAR NOT NULL,
                view_definition JSON NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
            "#,
        )
        .await
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        let handle = self.statement_counter.fetch_add(1, Ordering::SeqCst);
        let mut statements = self.prepared_statements.lock().await;
        statements.push((handle, query.to_string()));
        Ok(handle.to_le_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
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

        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        // Get schema before executing query
        let schema = Arc::new(Schema::new(
            stmt.get_parameter_schema()
                .map_err(|e| Status::internal(format!("Failed to get schema: {}", e)))?
                .fields()
                .to_vec(),
        ));

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        if let Some(batch) = reader.next() {
            batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))
        } else {
            // Use already obtained schema for empty batch
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
                .map_err(|e| Status::internal(format!("Failed to create empty batch: {}", e)))
        }
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let driver_path = options
            .get("driver_path")
            .ok_or_else(|| Status::invalid_argument("driver_path is required"))?;

        Self::new(driver_path, Some(connection_string), credentials)
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        let sql = crate::storage::StorageUtils::generate_create_table_sql(table_name, schema)?;
        self.execute_statement(&sql).await
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        let sql = crate::storage::StorageUtils::generate_insert_sql(table_name, batch.num_columns());

        stmt.set_sql_query(&sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to execute statement: {}", e)))?;

        Ok(())
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        // Create SQL view
        let create_view_sql = crate::storage::StorageUtils::generate_view_sql(name, &definition);
        
        stmt.set_sql_query(&create_view_sql)
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to create view: {}", e)))?;

        // Store view metadata
        let metadata = ViewMetadata {
            name: name.to_string(),
            definition: definition.clone(),
            created_at: SystemTime::now(),
        };

        let definition_json = serde_json::to_string(&definition)
            .map_err(|e| Status::internal(format!("Failed to serialize view definition: {}", e)))?;

        let mut meta_stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        meta_stmt
            .set_sql_query(
                "INSERT INTO view_metadata (view_name, source_table, view_definition, created_at) VALUES (?, ?, ?, ?)",
            )
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut view_name_builder = StringBuilder::new();
        view_name_builder.append_value(name);

        let mut source_table_builder = StringBuilder::new();
        source_table_builder.append_value(&definition.source_table);

        let mut definition_json_builder = StringBuilder::new();
        definition_json_builder.append_value(&definition_json);

        let mut created_at_builder = Int64Builder::new();
        created_at_builder.append_value(
            metadata.created_at.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as i64
        );

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("view_name", DataType::Utf8, false),
                Field::new("source_table", DataType::Utf8, false),
                Field::new("view_definition", DataType::Utf8, false),
                Field::new("created_at", DataType::Int64, false),
            ])),
            vec![
                Arc::new(view_name_builder.finish()),
                Arc::new(source_table_builder.finish()),
                Arc::new(definition_json_builder.finish()),
                Arc::new(created_at_builder.finish()),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create batch: {}", e)))?;

        meta_stmt
            .bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        meta_stmt
            .execute_update()
            .map_err(|e| Status::internal(format!("Failed to store view metadata: {}", e)))?;

        Ok(())
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query("SELECT * FROM view_metadata WHERE view_name = ?")
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut view_name_builder = StringBuilder::new();
        view_name_builder.append_value(name);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("view_name", DataType::Utf8, false)])),
            vec![Arc::new(view_name_builder.finish())],
        )
        .map_err(|e| Status::internal(format!("Failed to create batch: {}", e)))?;

        stmt.bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        if let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;
            
            let definition_json = batch
                .column_by_name("view_definition")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid view_definition column"))?
                .value(0);

            let definition: ViewDefinition = serde_json::from_str(definition_json)
                .map_err(|e| Status::internal(format!("Failed to deserialize view definition: {}", e)))?;

            let created_at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(
                batch
                    .column_by_name("created_at")
                    .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                    .ok_or_else(|| Status::internal("Invalid created_at column"))?
                    .value(0) as u64
            );

            Ok(ViewMetadata {
                name: name.to_string(),
                definition,
                created_at,
            })
        } else {
            Err(Status::not_found(format!("View {} not found", name)))
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query("SELECT view_name FROM view_metadata")
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut views = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;
            let view_names = batch
                .column_by_name("view_name")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid view_name column"))?;

            for i in 0..view_names.len() {
                views.push(view_names.value(i).to_string());
            }
        }

        Ok(views)
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        // Drop the view
        stmt.set_sql_query(&format!("DROP VIEW IF EXISTS {}", name))
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        stmt.execute_update()
            .map_err(|e| Status::internal(format!("Failed to drop view: {}", e)))?;

        // Remove metadata
        let mut meta_stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        meta_stmt
            .set_sql_query("DELETE FROM view_metadata WHERE view_name = ?")
            .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut view_name_builder = StringBuilder::new();
        view_name_builder.append_value(name);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("view_name", DataType::Utf8, false)])),
            vec![Arc::new(view_name_builder.finish())],
        )
        .map_err(|e| Status::internal(format!("Failed to create batch: {}", e)))?;

        meta_stmt
            .bind(batch)
            .map_err(|e| Status::internal(format!("Failed to bind parameters: {}", e)))?;

        meta_stmt
            .execute_update()
            .map_err(|e| Status::internal(format!("Failed to remove view metadata: {}", e)))?;

        Ok(())
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        let sql = format!("DROP TABLE IF EXISTS {}", table_name);
        self.execute_statement(&sql).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        )
        .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut tables = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;
            let table_names = batch
                .column_by_name("table_name")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid table_name column"))?;

            for i in 0..table_names.len() {
                tables.push(table_names.value(i).to_string());
            }
        }

        Ok(tables)
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        let mut conn = self.conn.lock().await;
        let mut stmt = conn
            .new_statement()
            .map_err(|e| Status::internal(format!("Failed to create statement: {}", e)))?;

        stmt.set_sql_query(&format!(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '{}'",
            table_name
        ))
        .map_err(|e| Status::internal(format!("Failed to set query: {}", e)))?;

        let mut reader = stmt
            .execute()
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut fields = Vec::new();
        while let Some(batch) = reader.next() {
            let batch = batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;

            let names = batch
                .column_by_name("column_name")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid column_name column"))?;

            let types = batch
                .column_by_name("data_type")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid data_type column"))?;

            let nullables = batch
                .column_by_name("is_nullable")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Status::internal("Invalid is_nullable column"))?;

            for i in 0..batch.num_rows() {
                let name = names.value(i);
                let type_str = types.value(i);
                let nullable = nullables.value(i) == "YES";

                let data_type = match type_str.to_uppercase().as_str() {
                    "BIGINT" | "INTEGER" => DataType::Int64,
                    "DOUBLE PRECISION" | "REAL" => DataType::Float64,
                    _ => DataType::Utf8,
                };

                fields.push(Field::new(name, data_type, !nullable));
            }
        }

        Ok(Arc::new(Schema::new(fields)))
    }

    async fn export_to_parquet(
        &self,
        _table_name: &str,
        _path: &std::path::Path,
    ) -> Result<(), Status> {
        Err(Status::unimplemented(
            "Parquet export not supported for ADBC backend. Use DuckDB for checkpointing.",
        ))
    }

    async fn import_from_parquet(
        &self,
        _table_name: &str,
        _path: &std::path::Path,
    ) -> Result<(), Status> {
        Err(Status::unimplemented(
            "Parquet import not supported for ADBC backend. Use DuckDB for checkpointing.",
        ))
    }
}
