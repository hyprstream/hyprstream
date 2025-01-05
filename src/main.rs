use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_server::FlightServiceServer,
    sql::{
        server::FlightSqlService, CommandGetCatalogs, CommandGetDbSchemas, CommandGetSqlInfo,
        CommandGetTableTypes, CommandGetTables, CommandPreparedStatementQuery, SqlInfo,
        TicketStatementQuery,
    },
    FlightData, Ticket,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, Stream};
use sqlparser::ast::{BinaryOperator, Expr, Value};
use std::pin::Pin;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod storage;
use storage::{MetricRecord, StorageBackend};

mod config;

#[derive(Clone)]
struct FlightServiceImpl {
    backend: Arc<storage::cached::CachedStorageBackend>,
}

type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

#[async_trait]
impl FlightSqlService for FlightServiceImpl {
    type FlightService = Self;

    async fn do_get_sql_info(
        &self,
        _cmd: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Return basic SQL info
        Ok(Response::new(Box::pin(stream::empty())))
    }

    async fn do_get_tables<'a>(
        &'a self,
        _cmd: CommandGetTables,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for table metadata
        let schema = Schema::new(vec![
            Field::new("catalog_name", DataType::Utf8, true),
            Field::new("schema_name", DataType::Utf8, true),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("table_type", DataType::Utf8, false),
        ]);

        // Create a record batch with our metrics table info
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(vec![Some("memstack")])),
                Arc::new(StringArray::from(vec![Some("public")])),
                Arc::new(StringArray::from(vec!["metrics"])),
                Arc::new(StringArray::from(vec!["TABLE"])),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        let (header, body) = encode_record_batch(&batch)?;

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightData {
                flight_descriptor: None,
                data_header: header.into(),
                data_body: body.into(),
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn do_get_table_types<'a>(
        &'a self,
        _cmd: CommandGetTableTypes,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for table types
        let schema = Schema::new(vec![Field::new("table_type", DataType::Utf8, false)]);

        // Create a record batch with our table types
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(StringArray::from(vec!["TABLE"]))],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        let (header, body) = encode_record_batch(&batch)?;

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightData {
                flight_descriptor: None,
                data_header: header.into(),
                data_body: body.into(),
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn do_get_catalogs<'a>(
        &'a self,
        _cmd: CommandGetCatalogs,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for catalogs
        let schema = Schema::new(vec![Field::new("catalog_name", DataType::Utf8, true)]);

        // Create a record batch with our catalog info
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(StringArray::from(vec![Some("memstack")]))],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        let (header, body) = encode_record_batch(&batch)?;

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightData {
                flight_descriptor: None,
                data_header: header.into(),
                data_body: body.into(),
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn do_get_schemas<'a>(
        &'a self,
        _cmd: CommandGetDbSchemas,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for database schemas
        let schema = Schema::new(vec![
            Field::new("catalog_name", DataType::Utf8, true),
            Field::new("schema_name", DataType::Utf8, false),
        ]);

        // Create a record batch with our schema info
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(vec![Some("memstack")])),
                Arc::new(StringArray::from(vec!["public"])),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        let (header, body) = encode_record_batch(&batch)?;

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightData {
                flight_descriptor: None,
                data_header: header.into(),
                data_body: body.into(),
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn do_get_statement(
        &self,
        query: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Parse the SQL query to validate timestamp conditions
        if let Ok(ast) = sqlparser::parser::Parser::parse_sql(
            &sqlparser::dialect::GenericDialect {},
            std::str::from_utf8(&query.statement_handle)
                .map_err(|e| Status::internal(format!("Invalid UTF-8 in statement: {}", e)))?,
        ) {
            if let Some(first_stmt) = ast.first() {
                if let sqlparser::ast::Statement::Query(box_query) = first_stmt {
                    if let sqlparser::ast::SetExpr::Select(select) = box_query.body.as_ref() {
                        if let Some(selection) = &select.selection {
                            // Validate timestamp conditions
                            if !is_valid_timestamp_condition(selection) {
                                return Err(Status::invalid_argument(
                                    "Query must include valid timestamp conditions",
                                ));
                            }
                            
                            // Extract timestamp for optimization
                            if let Some(from_timestamp) = extract_timestamp_condition(selection) {
                                // Query the metrics using the backend's SQL capabilities
                                let data = self.backend.query_metrics(from_timestamp).await?;
                                let batch = create_record_batch(data)?;
                                let (header, body) = encode_record_batch(&batch)?;

                                return Ok(Response::new(Box::pin(stream::once(async move {
                                    Ok(FlightData {
                                        flight_descriptor: None,
                                        data_header: header.into(),
                                        data_body: body.into(),
                                        app_metadata: Bytes::new(),
                                    })
                                }))));
                            }
                        }
                    }
                }
            }
        }

        Err(Status::invalid_argument("Invalid SQL query"))
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Convert bytes to string for the prepared statement handle
        let statement_handle =
            std::str::from_utf8(&query.prepared_statement_handle).map_err(|e| {
                Status::internal(format!("Invalid UTF-8 in prepared statement handle: {}", e))
            })?;

        // Let the backend prepare the statement
        let prepared_handle = self.backend.prepare_sql(statement_handle).await?;

        // Return the schema for the prepared statement
        let schema = get_metrics_schema();
        let mut schema_buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut schema_buffer, &schema)
                .map_err(|e| Status::internal(format!("Failed to create schema writer: {}", e)))?;
            writer
                .finish()
                .map_err(|e| Status::internal(format!("Failed to write schema: {}", e)))?;
        }

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightData {
                flight_descriptor: None,
                data_header: schema_buffer.into(),
                data_body: prepared_handle.into(),
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn register_sql_info(&self, _info_id: i32, _info: &SqlInfo) {
        // No-op for now
    }
}

// Helper functions
fn create_record_batch(records: Vec<MetricRecord>) -> Result<RecordBatch, Status> {
    let schema = Schema::new(vec![
        Field::new("metric_id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value_running_window_sum", DataType::Float64, false),
        Field::new("value_running_window_avg", DataType::Float64, false),
        Field::new("value_running_window_count", DataType::Int64, false),
    ]);

    let metric_ids = StringArray::from(
        records
            .iter()
            .map(|r| r.metric_id.as_str())
            .collect::<Vec<&str>>(),
    );
    let timestamps: Int64Array = records.iter().map(|r| r.timestamp).collect();
    let sums: Float64Array = records.iter().map(|r| r.value_running_window_sum).collect();
    let avgs: Float64Array = records.iter().map(|r| r.value_running_window_avg).collect();
    let counts: Int64Array = records
        .iter()
        .map(|r| r.value_running_window_count)
        .collect();

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
    .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))
}

fn encode_record_batch(batch: &RecordBatch) -> Result<(Vec<u8>, Vec<u8>), Status> {
    let mut schema_buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut schema_buffer, batch.schema().as_ref())
            .map_err(|e| Status::internal(format!("Failed to create schema writer: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("Failed to write schema: {}", e)))?;
    }

    let mut data_buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut data_buffer, batch.schema().as_ref())
            .map_err(|e| Status::internal(format!("Failed to create writer: {}", e)))?;
        writer
            .write(batch)
            .map_err(|e| Status::internal(format!("Failed to write batch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("Failed to finish writing: {}", e)))?;
    }

    Ok((schema_buffer, data_buffer))
}

// Helper function to get the metrics schema
fn get_metrics_schema() -> Schema {
    Schema::new(vec![
        Field::new("metric_id", DataType::Utf8, false),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("value_running_window_sum", DataType::Float64, false),
        Field::new("value_running_window_avg", DataType::Float64, false),
        Field::new("value_running_window_count", DataType::Int64, false),
    ])
}

/// Extract timestamp condition from an expression
fn extract_timestamp_condition(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::BinaryOp { left, op, right } => {
            // Check if this is a timestamp comparison
            if let Expr::Identifier(ident) = left.as_ref() {
                if ident.value.to_lowercase() == "timestamp" {
                    match (op, right.as_ref()) {
                        (BinaryOperator::GtEq, Expr::Value(Value::Number(n, _))) => n.parse().ok(),
                        (BinaryOperator::LtEq, Expr::Value(Value::Number(n, _))) => n.parse().ok(),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Check if an expression is a valid timestamp condition
fn is_valid_timestamp_condition(expr: &Expr) -> bool {
    match expr {
        Expr::BinaryOp { left, op, right } => {
            // Check if this is a timestamp comparison
            if let Expr::Identifier(ident) = left.as_ref() {
                if ident.value.to_lowercase() == "timestamp" {
                    match (op, right.as_ref()) {
                        (BinaryOperator::GtEq, Expr::Value(Value::Number(_, _)))
                        | (BinaryOperator::LtEq, Expr::Value(Value::Number(_, _))) => true,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let settings = config::Settings::new()?;
    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;

    // Initialize backends
    let duckdb = Arc::new(storage::duckdb::DuckDbBackend::new());

    // Create ADBC backend with configuration
    let adbc = Arc::new(
        storage::adbc::AdbcBackend::new(&settings.adbc)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)))?,
    );

    // Create cached backend using DuckDB as cache and ADBC as backing store
    let backend = Arc::new(storage::cached::CachedStorageBackend::new(
        duckdb,                       // Use DuckDB as the cache
        adbc,                         // Use ADBC as the backing store
        settings.cache.duration_secs, // Cache duration from config
    ));

    // Initialize the backend
    backend.init().await?;

    let service = FlightServiceImpl {
        backend: backend.clone(),
    };

    println!("Flight server listening on {}", addr);
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
