use crate::storage::view::ViewDefinition;
use crate::storage::{StorageBackend, StorageBackendType};
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo, HandshakeRequest,
    HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use arrow_schema::Schema;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::Mutex;
use tonic::{Request, Response, Status, Streaming};

// Add conversion trait for Arrow errors
trait ArrowErrorExt {
    fn to_status(self) -> Status;
}

impl ArrowErrorExt for arrow::error::ArrowError {
    fn to_status(self) -> Status {
        Status::internal(format!("Arrow error: {}", self))
    }
}

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
    statement_counter: Arc<AtomicU64>,
    prepared_statements: Arc<Mutex<Vec<String>>>,
}

impl FlightSqlServer {
    pub fn new(backend: StorageBackendType) -> Self {
        Self {
            backend: Arc::new(backend),
            statement_counter: Arc::new(AtomicU64::new(0)),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
        }
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

    async fn handle_action(&self, request: Request<Action>) -> Result<Vec<u8>, Status> {
        let action = request.into_inner();
        match Command::from_json(&action.body)? {
            Command::Table(cmd) => self.handle_table_command(cmd).await,
            Command::Sql(SqlCommand::Execute(sql)) => {
                let statement_handle = self.backend.prepare_sql(&sql).await?;
                self.backend.query_sql(&statement_handle).await?;
                Ok(vec![])
            }
            Command::Sql(SqlCommand::Query(sql)) => {
                let statement_handle = self.backend.prepare_sql(&sql).await?;
                let results = self.backend.query_sql(&statement_handle).await?;
                serde_json::to_vec(&results)
                    .map_err(|e| Status::internal(format!("Failed to serialize results: {}", e)))
            }
        }
    }
}

#[tonic::async_trait]
impl FlightService for FlightSqlServer {
    type HandshakeStream =
        Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type ListFlightsStream =
        Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoActionStream =
        Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send + 'static>>;
    type ListActionsStream =
        Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;
    type DoExchangeStream =
        Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
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

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let views = self.backend.list_views().await?;

        let stream = futures::stream::iter(views.into_iter().map(|name| {
            Ok(FlightInfo {
                schema: Bytes::new(),
                flight_descriptor: Some(FlightDescriptor {
                    r#type: 0,
                    cmd: Bytes::new(),
                    path: vec![name],
                }),
                endpoint: vec![],
                total_records: -1,
                total_bytes: -1,
                ordered: false,
                app_metadata: Bytes::new(),
            })
        }));

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();

        let cmd = match descriptor.cmd.to_vec().as_slice() {
            [] => return Err(Status::invalid_argument("Empty command")),
            cmd => TableCommand::from_json(cmd)
                .map_err(|e| Status::invalid_argument(format!("Invalid command: {}", e)))?,
        };

        let schema = match cmd {
            TableCommand::CreateTable { schema, .. } => schema,
            TableCommand::CreateView { definition, .. } => definition.schema,
            _ => return Err(Status::invalid_argument("Command does not return schema")),
        };

        let generator = IpcDataGenerator::default();
        let options = IpcWriteOptions::default();
        let mut dictionary_tracker = DictionaryTracker::new(false);
        let schema_data = generator.schema_to_bytes_with_dictionary_tracker(
            &schema,
            &mut dictionary_tracker,
            &options,
        );

        Ok(Response::new(SchemaResult {
            schema: Bytes::from(schema_data.ipc_message),
        }))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info not implemented"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn do_get(
        &self,
        _request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        Err(Status::unimplemented("do_get not implemented"))
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put not implemented"))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let result = self.handle_action(request).await?;

        let stream = futures::stream::once(async move {
            Ok(arrow_flight::Result {
                body: Bytes::from(result),
            })
        });

        Ok(Response::new(Box::pin(stream)))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "CreateTable".to_string(),
                description: "Create a new table".to_string(),
            },
            ActionType {
                r#type: "CreateView".to_string(),
                description: "Create a new view".to_string(),
            },
            ActionType {
                r#type: "DropTable".to_string(),
                description: "Drop an existing table".to_string(),
            },
            ActionType {
                r#type: "DropView".to_string(),
                description: "Drop an existing view".to_string(),
            },
        ];

        let stream = futures::stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }
}
