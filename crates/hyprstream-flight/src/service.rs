use hyprstream_metrics::storage::{
    view::ViewDefinition,
    StorageBackend, StorageBackendType,
};
use hyprstream_metrics::query::QueryOrchestrator;
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    sql::{
        server::FlightSqlService,
        CommandStatementQuery, TicketStatementQuery,
        ActionCreatePreparedStatementRequest, ActionCreatePreparedStatementResult,
        ActionClosePreparedStatementRequest, CommandPreparedStatementQuery,
        SqlInfo, ProstMessageExt,
    },
    Action, FlightDescriptor, FlightInfo, Ticket, HandshakeRequest, HandshakeResponse, FlightEndpoint,
    encode::FlightDataEncoderBuilder, error::FlightError,
};
use futures::{Stream, StreamExt, TryStreamExt};
use prost::Message;
use std::pin::Pin;
use arrow_schema::Schema;
use serde::Deserialize;
use serde_json;
use std::sync::{atomic::AtomicU64, Arc, Mutex};
use tonic::{Request, Response, Status, Streaming};

#[derive(Debug, Deserialize)]
struct CreateTableCmd {
    name: String,
    schema_bytes: Vec<u8>,
}

/// Command types for operations
#[derive(Debug)]
enum Command {
    Table(TableCommand),
    Sql(SqlCommand),
}

#[derive(Debug)]
enum SqlCommand {
    Execute(String),
    Query(String),
}

impl Command {
    fn from_json(cmd: &[u8]) -> Result<Self, Status> {
        let value: serde_json::Value = serde_json::from_slice(cmd)
            .map_err(|e| Status::invalid_argument(format!("Invalid JSON: {}", e)))?;

        match value.get("type").and_then(|t| t.as_str()) {
            Some("table") => {
                let cmd_bytes = serde_json::to_vec(&value["data"])
                    .map_err(|e| Status::internal(format!("Failed to serialize command: {}", e)))?;
                let cmd = TableCommand::from_json(&cmd_bytes)?;
                Ok(Command::Table(cmd))
            }
            Some("sql.execute") => {
                let sql =
                    String::from_utf8(value["data"].as_str().unwrap_or("").as_bytes().to_vec())
                        .map_err(|e| {
                            Status::invalid_argument(format!("Invalid SQL string: {}", e))
                        })?;
                Ok(Command::Sql(SqlCommand::Execute(sql)))
            }
            Some("sql.query") => {
                let sql =
                    String::from_utf8(value["data"].as_str().unwrap_or("").as_bytes().to_vec())
                        .map_err(|e| {
                            Status::invalid_argument(format!("Invalid SQL string: {}", e))
                        })?;
                Ok(Command::Sql(SqlCommand::Query(sql)))
            }
            _ => Err(Status::invalid_argument("Invalid command type")),
        }
    }
}

/// Command types for table and view operations
#[derive(Debug)]
enum TableCommand {
    CreateTable { name: String, schema: Arc<Schema> },
    CreateView { name: String, definition: ViewDefinition },
    DropTable(String),
    DropView(String),
}

impl TableCommand {
    fn from_json(cmd: &[u8]) -> Result<Self, Status> {
        let value: serde_json::Value = serde_json::from_slice(cmd)
            .map_err(|e| Status::invalid_argument(format!("Invalid JSON: {}", e)))?;

        match value.get("type").and_then(|t| t.as_str()) {
            Some("create_table") => {
                let cmd: CreateTableCmd =
                    serde_json::from_value(value["data"].clone()).map_err(|e| {
                        Status::invalid_argument(format!("Invalid create table command: {}", e))
                    })?;

                let schema = arrow_ipc::reader::StreamReader::try_new(
                    std::io::Cursor::new(&cmd.schema_bytes[..]),
                    None,
                )
                .map_err(|e| Status::invalid_argument(format!("Invalid schema bytes: {}", e)))?
                .schema()
                .clone();

                Ok(TableCommand::CreateTable {
                    name: cmd.name,
                    schema,
                })
            }
            Some("create_view") => {
                let definition: ViewDefinition =
                    serde_json::from_value(value["data"]["definition"].clone()).map_err(|e| {
                        Status::invalid_argument(format!("Invalid view definition: {}", e))
                    })?;
                let name = value["data"]["name"]
                    .as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing view name"))?
                    .to_string();
                Ok(TableCommand::CreateView { name, definition })
            }
            Some("drop_table") => {
                let name = value["data"]["name"]
                    .as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing table name"))?;
                Ok(TableCommand::DropTable(name.to_string()))
            }
            Some("drop_view") => {
                let name = value["data"]["name"]
                    .as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing view name"))?;
                Ok(TableCommand::DropView(name.to_string()))
            }
            _ => Err(Status::invalid_argument("Invalid command type")),
        }
    }
}

/// Flight SQL server implementation
#[derive(Clone)]
pub struct FlightSqlServer {
    backend: Arc<StorageBackendType>,
    orchestrator: Arc<QueryOrchestrator>,
    #[allow(dead_code)]
    statement_counter: Arc<AtomicU64>,
    prepared_statements: Arc<Mutex<Vec<(u64, String)>>>,
}

#[tonic::async_trait]
impl FlightSqlService for FlightSqlServer {
    type FlightService = Self;

    async fn register_sql_info(&self, _id: i32, _info: &SqlInfo) {}

    async fn do_handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>>, Status> {
        let mut stream = request.into_inner();
        let response_stream = async_stream::try_stream! {
            while let Some(request) = stream.next().await {
                let request = request?;
                yield HandshakeResponse {
                    protocol_version: request.protocol_version,
                    payload: request.payload,
                };
            }
        };
        Ok(Response::new(Box::pin(response_stream)))
    }

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        // Use QueryOrchestrator to prepare and get schema
        let cached_stmt = self.orchestrator.prepare(&query.query).await
            .map_err(|e| Status::internal(format!("Query preparation failed: {}", e)))?;

        let schema = cached_stmt.schema.clone();
        let handle = cached_stmt.handle;

        // Store SQL with handle for do_get (for compatibility with Flight SQL protocol)
        {
            let mut statements = self.prepared_statements.lock()
                .map_err(|e| Status::internal(format!("Lock error: {}", e)))?;
            statements.push((handle, query.query));
        }

        // Create ticket with statement handle
        let ticket = TicketStatementQuery {
            statement_handle: handle.to_le_bytes().to_vec().into(),
        };

        let flight_descriptor = request.into_inner();

        // Create FlightInfo with schema and endpoint
        let info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .map_err(|e| Status::internal(format!("Unable to serialize schema: {}", e)))?
            .with_descriptor(flight_descriptor)
            .with_endpoint(FlightEndpoint::new()
                .with_ticket(Ticket {
                    ticket: ticket.as_any().encode_to_vec().into(),
                }))
            .with_total_records(-1)
            .with_total_bytes(-1)
            .with_ordered(false);

        Ok(Response::new(info))
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        // Get statement handle from ticket
        let handle = u64::from_le_bytes(
            ticket.statement_handle.to_vec()
                .try_into()
                .map_err(|_| Status::invalid_argument("Invalid statement handle"))?
        );

        // Get cached statement from orchestrator
        let cached_stmt = self.orchestrator.get_statement(handle).await
            .ok_or_else(|| Status::invalid_argument("Statement handle not found"))?;

        let schema = cached_stmt.schema.clone();

        // Execute using the orchestrator's cached physical plan
        let result_stream = self.orchestrator.execute(&cached_stmt).await
            .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;

        // Convert DataFusion stream to Flight stream
        let stream = result_stream
            .map_err(|e| FlightError::ExternalError(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))));

        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_flight_info_prepared_statement(
        &self,
        _cmd: CommandPreparedStatementQuery,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info_prepared_statement not implemented"))
    }

    async fn do_get_prepared_statement(
        &self,
        _query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        Err(Status::unimplemented("do_get_prepared_statement not implemented"))
    }

    async fn do_action_create_prepared_statement(
        &self,
        _query: ActionCreatePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        Err(Status::unimplemented("do_action_create_prepared_statement not implemented"))
    }

    async fn do_action_close_prepared_statement(
        &self,
        _query: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        Err(Status::unimplemented("do_action_close_prepared_statement not implemented"))
    }
}

impl FlightSqlServer {
    pub async fn new(backend: StorageBackendType) -> Result<Self, Status> {
        let backend = Arc::new(backend);

        // Create query orchestrator with the storage backend
        let orchestrator = QueryOrchestrator::new(backend.clone() as Arc<dyn StorageBackend>)
            .await
            .map_err(|e| Status::internal(format!("Failed to create query orchestrator: {}", e)))?;

        Ok(Self {
            backend,
            orchestrator: Arc::new(orchestrator),
            statement_counter: Arc::new(AtomicU64::new(0)),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn into_service(self) -> FlightServiceServer<Self> {
        FlightServiceServer::new(self)
    }

    async fn handle_table_command(&self, cmd: TableCommand) -> Result<Vec<u8>, Status> {
        match cmd {
            TableCommand::CreateTable { name, schema } => {
                self.backend.create_table(&name, &schema).await?;
                Ok(vec![])
            }
            TableCommand::CreateView { name, definition } => {
                self.backend.create_view(&name, definition).await?;
                Ok(vec![])
            }
            TableCommand::DropTable(name) => {
                self.backend.drop_table(&name).await?;
                Ok(vec![])
            }
            TableCommand::DropView(name) => {
                self.backend.drop_view(&name).await?;
                Ok(vec![])
            }
        }
    }

    #[allow(dead_code)]
    async fn handle_action(&self, request: Request<Action>) -> Result<Vec<u8>, Status> {
        let action = request.into_inner();
        match Command::from_json(&action.body)? {
            Command::Table(cmd) => {
                let result = self.handle_table_command(cmd).await;
                // Refresh orchestrator's table registrations after schema changes
                if result.is_ok() {
                    let _ = self.orchestrator.refresh_tables().await;
                }
                result
            }
            Command::Sql(SqlCommand::Execute(sql)) => {
                // Execute statement via orchestrator
                self.orchestrator.query_collect(&sql).await
                    .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;
                Ok(vec![])
            }
            Command::Sql(SqlCommand::Query(sql)) => {
                // Prepare statement via orchestrator
                let cached_stmt = self.orchestrator.prepare(&sql).await
                    .map_err(|e| Status::internal(format!("Query preparation failed: {}", e)))?;

                let handle = cached_stmt.handle;

                // Store SQL with handle for do_get compatibility
                {
                    let mut statements = self.prepared_statements.lock()
                        .map_err(|e| Status::internal(format!("Lock error: {}", e)))?;
                    statements.push((handle, sql.clone()));
                }

                // Return handle as ticket for DoGet
                Ok(handle.to_le_bytes().to_vec())
            }
        }
    }

    /// Get reference to the query orchestrator
    pub fn orchestrator(&self) -> &Arc<QueryOrchestrator> {
        &self.orchestrator
    }
}
