//! Arrow Flight SQL service implementation for high-performance data transport.
//!
//! This module provides the core Flight SQL service implementation that enables:
//! - High-performance data queries via Arrow Flight protocol
//! - Support for vectorized data operations
//! - Real-time metric aggregation queries
//! - Time-windowed data access
//!
//! The service implementation is designed to work with multiple storage backends
//! while maintaining consistent query semantics and high performance.

use crate::metrics::{create_record_batch, encode_record_batch, METRICS_SCHEMA};
use crate::storage::StorageBackend;
use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow_flight::{
    flight_service_server::FlightService,
    sql::{
        server::FlightSqlService, CommandGetCatalogs, CommandGetDbSchemas, CommandGetSqlInfo,
        CommandGetTableTypes, CommandGetTables, CommandPreparedStatementQuery, SqlInfo,
        TicketStatementQuery,
    },
    FlightData, Ticket,
};
use bytes;
use futures::{stream, Stream};
use sqlparser::ast::{BinaryOperator, Expr, Value};
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status};

type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

/// Implementation of the Arrow Flight SQL service.
///
/// This service provides:
/// - SQL query execution over Arrow Flight protocol
/// - Real-time metric aggregation and windowed queries
/// - Integration with configurable storage backends
/// - High-performance data transport using Arrow's columnar format
#[derive(Clone)]
pub struct FlightServiceImpl {
    /// Storage backend for executing queries and storing data
    backend: Arc<dyn StorageBackend>,
}

impl FlightServiceImpl {
    /// Creates a new Flight SQL service with the specified storage backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - Arc-wrapped storage backend implementing the StorageBackend trait
    pub fn new(backend: Arc<dyn StorageBackend>) -> Self {
        Self { backend }
    }

    /// Extracts timestamp conditions from SQL expressions for query optimization.
    ///
    /// This function analyzes SQL expressions to identify timestamp-based filters,
    /// enabling efficient time-window queries.
    ///
    /// # Arguments
    ///
    /// * `expr` - SQL expression to analyze
    ///
    /// # Returns
    ///
    /// * `Option<i64>` - Extracted timestamp value if found
    fn extract_timestamp_condition(expr: &Expr) -> Option<i64> {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                // Check if this is a timestamp comparison
                if let Expr::Identifier(ident) = left.as_ref() {
                    if ident.value.to_lowercase() == "timestamp" {
                        match (op, right.as_ref()) {
                            (BinaryOperator::GtEq, Expr::Value(Value::Number(n, _))) => {
                                n.parse().ok()
                            }
                            (BinaryOperator::LtEq, Expr::Value(Value::Number(n, _))) => {
                                n.parse().ok()
                            }
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

    /// Validates timestamp conditions in SQL expressions.
    ///
    /// This function checks if a SQL expression contains valid timestamp-based
    /// filters, ensuring queries can be properly optimized for time-window access.
    ///
    /// # Arguments
    ///
    /// * `expr` - SQL expression to validate
    ///
    /// # Returns
    ///
    /// * `bool` - True if the expression contains valid timestamp conditions
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
}

#[tonic::async_trait]
impl FlightSqlService for FlightServiceImpl {
    type FlightService = Self;

    /// Handles SQL information requests.
    ///
    /// Currently returns an empty stream as basic SQL info is sufficient
    /// for most client operations.
    async fn do_get_sql_info(
        &self,
        _cmd: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        Ok(Response::new(Box::pin(stream::empty())))
    }

    /// Returns information about available tables.
    ///
    /// Provides metadata about the metrics table including:
    /// - Catalog name
    /// - Schema name
    /// - Table name
    /// - Table type
    async fn do_get_tables<'a>(
        &'a self,
        _cmd: CommandGetTables,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for table metadata
        let schema = Schema::new(vec![
            Arc::new(Field::new("catalog_name", DataType::Utf8, true)),
            Arc::new(Field::new("schema_name", DataType::Utf8, true)),
            Arc::new(Field::new("table_name", DataType::Utf8, false)),
            Arc::new(Field::new("table_type", DataType::Utf8, false)),
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
                app_metadata: bytes::Bytes::new(),
            })
        }))))
    }

    /// Returns information about available table types.
    ///
    /// Currently only supports the "TABLE" type for metrics storage.
    async fn do_get_table_types<'a>(
        &'a self,
        _cmd: CommandGetTableTypes,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for table types
        let schema = Schema::new(vec![Arc::new(Field::new(
            "table_type",
            DataType::Utf8,
            false,
        ))]);

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
                app_metadata: bytes::Bytes::new(),
            })
        }))))
    }

    /// Returns information about available catalogs.
    ///
    /// Currently supports a single "memstack" catalog for metrics storage.
    async fn do_get_catalogs<'a>(
        &'a self,
        _cmd: CommandGetCatalogs,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for catalogs
        let schema = Schema::new(vec![Arc::new(Field::new(
            "catalog_name",
            DataType::Utf8,
            true,
        ))]);

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
                app_metadata: bytes::Bytes::new(),
            })
        }))))
    }

    /// Returns information about available database schemas.
    ///
    /// Currently supports a single "public" schema in the "memstack" catalog.
    async fn do_get_schemas<'a>(
        &'a self,
        _cmd: CommandGetDbSchemas,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        // Create a schema for database schemas
        let schema = Schema::new(vec![
            Arc::new(Field::new("catalog_name", DataType::Utf8, true)),
            Arc::new(Field::new("schema_name", DataType::Utf8, false)),
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
                app_metadata: bytes::Bytes::new(),
            })
        }))))
    }

    /// Executes a SQL statement and returns the results.
    ///
    /// This method:
    /// 1. Validates the SQL query contains proper timestamp conditions
    /// 2. Extracts timestamp filters for optimization
    /// 3. Queries the storage backend with the optimized parameters
    /// 4. Returns results in Arrow Flight format
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
                            if !Self::is_valid_timestamp_condition(selection) {
                                return Err(Status::invalid_argument(
                                    "Query must include valid timestamp conditions",
                                ));
                            }

                            // Extract timestamp for optimization
                            if let Some(from_timestamp) =
                                Self::extract_timestamp_condition(selection)
                            {
                                // Query the metrics using the backend's SQL capabilities
                                let data = self.backend.query_metrics(from_timestamp).await?;
                                let batch = create_record_batch(data)?;
                                let (header, body) = encode_record_batch(&batch)?;

                                return Ok(Response::new(Box::pin(stream::once(async move {
                                    Ok(FlightData {
                                        flight_descriptor: None,
                                        data_header: header.into(),
                                        data_body: body.into(),
                                        app_metadata: bytes::Bytes::new(),
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

    /// Handles prepared statement execution.
    ///
    /// This method:
    /// 1. Prepares the SQL statement using the storage backend
    /// 2. Returns the schema for the prepared statement results
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
        let schema = METRICS_SCHEMA.clone();
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
                app_metadata: bytes::Bytes::new(),
            })
        }))))
    }

    /// Registers SQL information for the service.
    ///
    /// Currently a no-op as basic SQL info is sufficient.
    async fn register_sql_info(&self, _info_id: i32, _info: &SqlInfo) {
        // No-op for now
    }
}
