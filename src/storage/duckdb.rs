use crate::aggregation::{AggregateFunction, AggregateResult, GroupBy};
use crate::metrics::MetricRecord;
use crate::storage::cache::{CacheEviction, CacheManager};
use crate::storage::view::{ViewDefinition, ViewMetadata};
use crate::storage::{Credentials, StorageBackend};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
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
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                value_running_window_sum DOUBLE NOT NULL,
                value_running_window_avg DOUBLE NOT NULL,
                value_running_window_count BIGINT NOT NULL,
                PRIMARY KEY (metric_id, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

            CREATE TABLE IF NOT EXISTS view_metadata (
                view_name VARCHAR PRIMARY KEY,
                source_table VARCHAR NOT NULL,
                view_definition JSON NOT NULL,
                created_at BIGINT NOT NULL
            );
            "#,
        )
        .map_err(|e| Status::internal(format!("Failed to create tables: {}", e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            connection_string,
            options,
            cache_manager: CacheManager::new(ttl),
        })
    }

    pub fn new_in_memory() -> Result<Self, Status> {
        Self::new(":memory:".to_string(), HashMap::new(), Some(0))
    }

    async fn execute_statement(&self, sql: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(sql)
            .map_err(|e| Status::internal(format!("Failed to execute statement: {}", e)))
    }
}

#[async_trait]
impl StorageBackend for DuckDbBackend {
    async fn init(&self) -> Result<(), Status> {
        // Tables are already created in new()
        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        for metric in metrics {
            tx.execute(
                &crate::storage::StorageUtils::generate_metric_insert_sql(),
                params![
                    metric.metric_id,
                    metric.timestamp,
                    metric.value_running_window_sum,
                    metric.value_running_window_avg,
                    metric.value_running_window_count,
                ],
            )
            .map_err(|e| Status::internal(format!("Failed to insert metric: {}", e)))?;
        }

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                &crate::storage::StorageUtils::generate_metric_query_sql(from_timestamp),
            )
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(params![from_timestamp], |row| {
                Ok(MetricRecord {
                    metric_id: row.get(0)?,
                    timestamp: row.get(1)?,
                    value_running_window_sum: row.get(2)?,
                    value_running_window_avg: row.get(3)?,
                    value_running_window_count: row.get(4)?,
                })
            })
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))?);
        }

        Ok(metrics)
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        let sql = std::str::from_utf8(statement_handle)
            .map_err(|e| Status::internal(format!("Invalid SQL statement: {}", e)))?;
        
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(params![], |row| {
                Ok(MetricRecord {
                    metric_id: row.get(0)?,
                    timestamp: row.get(1)?,
                    value_running_window_sum: row.get(2)?,
                    value_running_window_avg: row.get(3)?,
                    value_running_window_count: row.get(4)?,
                })
            })
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))?);
        }

        Ok(metrics)
    }

    fn new_with_options(
        connection_string: &str,
        options: &HashMap<String, String>,
        credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        let mut all_options = options.clone();
        if let Some(creds) = credentials {
            all_options.insert("username".to_string(), creds.username.clone());
            all_options.insert("password".to_string(), creds.password.clone());
        }

        let ttl = all_options
            .get("ttl")
            .and_then(|s| s.parse().ok())
            .map(|ttl| if ttl == 0 { None } else { Some(ttl) })
            .unwrap_or(None);

        Self::new(connection_string.to_string(), all_options, ttl)
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        let sql = crate::storage::StorageUtils::generate_create_table_sql(table_name, schema)?;
        self.execute_statement(&sql).await
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        let sql = crate::storage::StorageUtils::generate_insert_sql(table_name, batch.num_columns());

        let mut stmt = tx
            .prepare(&sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        for row_idx in 0..batch.num_rows() {
            let mut params: Vec<Box<dyn duckdb::ToSql>> = Vec::new();
            for col_idx in 0..batch.num_columns() {
                let col = batch.column(col_idx);
                match col.data_type() {
                    DataType::Int64 => {
                        let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        params.push(Box::new(array.value(row_idx)));
                    }
                    DataType::Float64 => {
                        let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        params.push(Box::new(array.value(row_idx)));
                    }
                    DataType::Utf8 => {
                        let array = col.as_any().downcast_ref::<StringArray>().unwrap();
                        params.push(Box::new(array.value(row_idx).to_string()));
                    }
                    _ => return Err(Status::invalid_argument("Unsupported data type")),
                }
            }

            let param_refs: Vec<&dyn duckdb::ToSql> = params.iter().map(|p| p.as_ref()).collect();
            stmt.execute(param_refs.as_slice())
                .map_err(|e| Status::internal(format!("Failed to insert row: {}", e)))?;
        }

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn query_table(
        &self,
        table_name: &str,
        projection: Option<Vec<String>>,
    ) -> Result<RecordBatch, Status> {
        // Get schema first
        let schema = self.get_table_schema(table_name).await?;

        // Then execute query
        let sql = crate::storage::StorageUtils::generate_select_sql(table_name, projection);
        tracing::debug!("Executing query: {}", sql);

        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        // Execute query and collect results
        let mut rows = stmt
            .query(params![])
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut values: Vec<Vec<String>> = Vec::new();
        while let Some(row) = rows.next().map_err(|e| Status::internal(e.to_string()))? {
            let mut row_values = Vec::new();
            for i in 0..schema.fields().len() {
                row_values.push(row.get::<_, String>(i).unwrap_or_default());
            }
            values.push(row_values);
        }

        // If no rows, return empty batch
        if values.is_empty() {
            let empty_arrays: Vec<ArrayRef> = schema
                .fields()
                .iter()
                .map(|field| match field.data_type() {
                    DataType::Int64 => Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
                    DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
                    _ => Arc::new(StringArray::from(Vec::<String>::new())) as ArrayRef,
                })
                .collect();
            return Ok(RecordBatch::try_new(schema, empty_arrays)
                .map_err(|e| Status::internal(format!("Failed to create empty batch: {}", e)))?);
        }

        // Convert string values to appropriate types
        let arrays: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let column_values: Vec<_> = values.iter().map(|row| &row[i]).collect();
                match field.data_type() {
                    DataType::Int64 => {
                        let parsed: Vec<i64> = column_values
                            .iter()
                            .map(|s| s.parse().unwrap_or_default())
                            .collect();
                        Arc::new(Int64Array::from(parsed)) as ArrayRef
                    }
                    DataType::Float64 => {
                        let parsed: Vec<f64> = column_values
                            .iter()
                            .map(|s| s.parse().unwrap_or_default())
                            .collect();
                        Arc::new(Float64Array::from(parsed)) as ArrayRef
                    }
                    _ => {
                        let str_values: Vec<&str> = column_values.iter().map(|s| s.as_str()).collect();
                        Arc::new(StringArray::from(str_values)) as ArrayRef
                    }
                }
            })
            .collect();

        Ok(RecordBatch::try_new(schema, arrays)
            .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?)
    }

    async fn create_view(&self, name: &str, definition: ViewDefinition) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        // Create SQL view
        let create_view_sql = crate::storage::StorageUtils::generate_view_sql(name, &definition);
        tx.execute(&create_view_sql, params![])
            .map_err(|e| Status::internal(format!("Failed to create view: {}", e)))?;

        // Store view metadata
        let metadata = ViewMetadata {
            name: name.to_string(),
            definition: definition.clone(),
            created_at: SystemTime::now(),
        };

        let definition_json = serde_json::to_string(&definition)
            .map_err(|e| Status::internal(format!("Failed to serialize view definition: {}", e)))?;

        tx.execute(
            "INSERT INTO view_metadata (view_name, source_table, view_definition, created_at) VALUES (?, ?, ?, ?)",
            params![
                name,
                definition.source_table,
                definition_json,
                metadata.created_at.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as i64
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to store view metadata: {}", e)))?;

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn get_view(&self, name: &str) -> Result<ViewMetadata, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT * FROM view_metadata WHERE view_name = ?")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let mut rows = stmt
            .query_map(params![name], |row| {
                let definition_json: String = row.get(2)?;
                let definition: ViewDefinition = serde_json::from_str(&definition_json)
                    .map_err(|e| duckdb::Error::InvalidParameterName(format!("Invalid view definition: {}", e)))?;

                let created_at = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(row.get::<_, i64>(3)? as u64);

                Ok(ViewMetadata {
                    name: row.get(0)?,
                    definition,
                    created_at,
                })
            })
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        if let Some(row) = rows.next() {
            row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))
        } else {
            Err(Status::not_found(format!("View {} not found", name)))
        }
    }

    async fn list_views(&self) -> Result<Vec<String>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT view_name FROM view_metadata")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(params![], |row| row.get::<_, String>(0))
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut views = Vec::new();
        for row in rows {
            views.push(row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))?);
        }

        Ok(views)
    }

    async fn drop_view(&self, name: &str) -> Result<(), Status> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        // Drop the view
        tx.execute(&format!("DROP VIEW IF EXISTS {}", name), params![])
            .map_err(|e| Status::internal(format!("Failed to drop view: {}", e)))?;

        // Remove metadata
        tx.execute(
            "DELETE FROM view_metadata WHERE view_name = ?",
            params![name],
        )
        .map_err(|e| Status::internal(format!("Failed to remove view metadata: {}", e)))?;

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        let sql = format!("DROP TABLE IF EXISTS {}", table_name);
        self.execute_statement(&sql).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(params![], |row| row.get::<_, String>(0))
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut tables = Vec::new();
        for row in rows {
            tables.push(row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))?);
        }

        Ok(tables)
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({})", table_name))
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

    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        group_by: &GroupBy,
        from_timestamp: i64,
        to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        let sql = crate::storage::StorageUtils::generate_metric_aggregation_sql(
            function,
            group_by,
            from_timestamp,
            to_timestamp
        );

        // Execute query
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| Status::internal(format!("Failed to prepare statement: {}", e)))?;

        let rows = stmt
            .query_map(params![], |row| {
                let mut group_values = HashMap::new();
                for (i, col) in group_by.columns.iter().enumerate() {
                    group_values.insert(col.clone(), row.get::<_, String>(i)?);
                }
                let value = row.get::<_, f64>(group_by.columns.len())?;
                Ok(AggregateResult {
                    value,
                    group_values,
                    timestamp: None, // TODO: Add timestamp support
                })
            })
            .map_err(|e| Status::internal(format!("Failed to execute query: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Status::internal(format!("Failed to read row: {}", e)))?);
        }

        Ok(results)
    }
}

#[async_trait]
impl CacheEviction for DuckDbBackend {
    async fn execute_eviction(&self, query: &str) -> Result<(), Status> {
        self.execute_statement(query).await
    }
}
