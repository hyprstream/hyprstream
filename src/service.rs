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

use crate::metrics::{encode_record_batch, MetricRecord};
use crate::models::storage::TimeSeriesModelStorage;
use crate::models::{Model, ModelStorage};
use crate::query::{
    DataFusionExecutor, DataFusionPlanner, ExecutorConfig, OptimizationHint, Query, QueryExecutor,
    QueryPlanner,
};
use crate::storage::table_manager::AggregationView;
use crate::storage::{StorageBackend, StorageBackendType};
use arrow_array::{builder::Float32Builder, ArrayRef, Float32Array};
use arrow_flight::{
    flight_service_server::FlightService, Action, ActionType, Criteria, Empty, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, PollInfo, PutResult,
    SchemaResult, Ticket,
};
use arrow_ipc::writer::IpcDataGenerator;
use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::Schema;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
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
    Model(ModelCommand),
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
            Some("model") => {
                let cmd: ModelCommand =
                    serde_json::from_value(value["data"].clone()).map_err(|e| {
                        Status::invalid_argument(format!("Invalid model command: {}", e))
                    })?;
                Ok(Command::Model(cmd))
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
    CreateAggregationView(AggregationView),
    DropTable(String),
    DropAggregationView(String),
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
            Some("create_aggregation_view") => {
                let view: AggregationView =
                    serde_json::from_value(value["data"].clone()).map_err(|e| {
                        Status::invalid_argument(format!("Invalid view command: {}", e))
                    })?;
                Ok(TableCommand::CreateAggregationView(view))
            }
            Some("drop_table") => {
                let name = value["data"]["name"]
                    .as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing table name"))?;
                Ok(TableCommand::DropTable(name.to_string()))
            }
            Some("drop_aggregation_view") => {
                let name = value["data"]["name"]
                    .as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing view name"))?;
                Ok(TableCommand::DropAggregationView(name.to_string()))
            }
            _ => Err(Status::invalid_argument("Invalid command type")),
        }
    }
}

/// Model management commands
#[derive(Debug, Deserialize)]
enum ModelCommand {
    StoreModel {
        model: Model,
    },
    LoadModel {
        model_id: String,
        version: Option<String>,
    },
    ListModels,
    ListVersions {
        model_id: String,
    },
    DeleteVersion {
        model_id: String,
        version: String,
    },
}

impl ModelCommand {
    fn from_json(bytes: &[u8]) -> Result<Self, Status> {
        serde_json::from_slice(bytes)
            .map_err(|e| Status::invalid_argument(format!("Invalid model command: {}", e)))
    }
}

/// Tracks progress of model data transfers
#[derive(Debug)]
pub struct TransferProgress {
    total_bytes: u64,
    transferred_bytes: AtomicU64,
    start_time: Instant,
}

impl TransferProgress {
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            transferred_bytes: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn update(&self, bytes: usize) {
        self.transferred_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub fn progress(&self) -> f64 {
        let transferred = self.transferred_bytes.load(Ordering::Relaxed) as f64;
        transferred / self.total_bytes as f64
    }

    pub fn transfer_rate(&self) -> f64 {
        let transferred = self.transferred_bytes.load(Ordering::Relaxed) as f64;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        transferred / elapsed
    }
}

/// Authentication token with expiry
#[derive(Debug, Clone)]
struct AuthToken {
    token: String,
    expiry: Instant,
}

/// Flight SQL service for model management and data operations
///
/// Provides high-performance streaming operations for:
/// - Model storage and retrieval
/// - Weight transfer with progress tracking
/// - Authentication and authorization
///
/// # Examples
///
/// ```rust
/// use hyprstream_core::service::FlightSqlService;
///
/// // Store a model
/// let request = Request::new(Action {
///     r#type: "model.store".to_string(),
///     body: serde_json::to_vec(&model)?,
/// });
/// let response = service.do_action(request).await?;
///
/// // Stream model weights
/// let request = Request::new(Ticket {
///     ticket: serde_json::to_vec(&ModelCommand::LoadModel {
///         model_id: "model1".to_string(),
///         version: Some("v1".to_string()),
///     })?,
/// });
/// let mut stream = service.do_get(request).await?.into_inner();
/// while let Some(chunk) = stream.next().await {
///     let data = chunk?.data_header;
///     // Process weight data...
/// }
/// ```
#[derive(Clone)]
pub struct FlightSqlService {
    backend: Arc<StorageBackendType>,
    model_storage: Arc<Box<dyn ModelStorage>>,
    statement_counter: Arc<AtomicU64>,
    prepared_statements: Arc<Mutex<Vec<String>>>,
    planner: Arc<DataFusionPlanner>,
    executor: Arc<DataFusionExecutor>,
}

impl FlightSqlService {
    pub fn new(backend: StorageBackendType) -> Self {
        let backend = Arc::new(backend);
        let model_storage = Box::new(TimeSeriesModelStorage::new(backend.clone()));

        // Initialize query planner and executor
        let planner = Arc::new(DataFusionPlanner::new());
        let executor = Arc::new(DataFusionExecutor::new(ExecutorConfig::default()));

        Self {
            backend,
            model_storage: Arc::new(model_storage),
            statement_counter: Arc::new(AtomicU64::new(0)),
            prepared_statements: Arc::new(Mutex::new(Vec::new())),
            planner,
            executor,
        }
    }

    pub fn into_server(self) -> arrow_flight::flight_service_server::FlightServiceServer<FlightSqlService> {
        arrow_flight::flight_service_server::FlightServiceServer::new(self)
    }

    // SQL execution methods
    pub async fn execute(&self, sql: String) -> Result<(), Status> {
        let action = Action {
            r#type: "sql.execute".to_string(),
            body: sql.into_bytes().into(),
        };
        self.handle_action(Request::new(action)).await?;
        Ok(())
    }

    pub async fn query_sql(&self, sql: String) -> Result<Vec<MetricRecord>, Status> {
        let action = Action {
            r#type: "sql.query".to_string(),
            body: sql.into_bytes().into(),
        };
        let result = self.handle_action(Request::new(action)).await?;
        serde_json::from_slice(&result)
            .map_err(|e| Status::internal(format!("Failed to parse query result: {}", e)))
    }

    async fn authenticate<T>(&self, request: &Request<T>) -> Result<AuthToken, Status> {
        let token = request
            .metadata()
            .get("authorization")
            .ok_or_else(|| Status::unauthenticated("Missing authorization token"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("Invalid token format"))?;

        // Validate token format
        if !token.starts_with("Bearer ") {
            return Err(Status::unauthenticated("Invalid token format"));
        }

        let token = token[7..].to_string();

        // TODO: Validate token with auth service
        // For now, create a token valid for 1 hour
        Ok(AuthToken {
            token,
            expiry: Instant::now() + Duration::from_secs(3600),
        })
    }

    /// Validates a model command and returns appropriate error status
    ///
    /// # Errors
    ///
    /// - `Status::unauthenticated`: Token is missing, invalid, or expired
    /// - `Status::invalid_argument`: Invalid model parameters
    /// - `Status::not_found`: Model or version not found
    /// - `Status::resource_exhausted`: Storage capacity exceeded
    fn validate_command(&self, cmd: &ModelCommand, token: &AuthToken) -> Result<(), Status> {
        // Check token expiry
        if token.expiry < Instant::now() {
            return Err(Status::unauthenticated("Token expired"));
        }

        match cmd {
            ModelCommand::StoreModel { model } => {
                if model.layers.is_empty() {
                    return Err(Status::invalid_argument(
                        "Model must have at least one layer",
                    ));
                }
                model.validate()?;
            }
            ModelCommand::LoadModel { model_id, .. }
            | ModelCommand::ListVersions { model_id }
            | ModelCommand::DeleteVersion { model_id, .. } => {
                if model_id.is_empty() {
                    return Err(Status::invalid_argument("Model ID cannot be empty"));
                }
            }
            ModelCommand::ListModels => {}
        }
        Ok(())
    }

    async fn store_with_progress(
        &self,
        model: &Model,
        progress: Arc<TransferProgress>,
    ) -> Result<(), Status> {
        // Track serialization progress
        let bytes = serde_json::to_vec(model)
            .map_err(|e| Status::internal(format!("Failed to serialize model: {}", e)))?;
        progress.update(bytes.len());

        // Store model with progress updates
        for layer in &model.layers {
            let layer_bytes = serde_json::to_vec(layer)
                .map_err(|e| Status::internal(format!("Failed to serialize layer: {}", e)))?;
            progress.update(layer_bytes.len());
        }

        self.model_storage.store_model(model).await
    }

    async fn handle_model_command(&self, request: Request<Action>) -> Result<Vec<u8>, Status> {
        // Authenticate request
        let token = self.authenticate(&request).await?;

        let cmd = ModelCommand::from_json(&request.into_inner().body)?;

        // Add validation
        self.validate_command(&cmd, &token)?;

        match cmd {
            ModelCommand::StoreModel { model } => {
                // Create progress tracker
                let progress = Arc::new(TransferProgress::new(model.estimated_size()));

                // Store with progress tracking
                self.store_with_progress(&model, progress).await?;

                Ok(vec![])
            }
            ModelCommand::LoadModel { model_id, version } => {
                let model = self
                    .model_storage
                    .load_model(&model_id, version.as_deref())
                    .await?;
                serde_json::to_vec(&model)
                    .map_err(|e| Status::internal(format!("Failed to serialize model: {}", e)))
            }
            ModelCommand::ListModels => {
                let models = self.model_storage.list_models().await?;
                serde_json::to_vec(&models)
                    .map_err(|e| Status::internal(format!("Failed to serialize models: {}", e)))
            }
            ModelCommand::ListVersions { model_id } => {
                let versions = self.model_storage.list_versions(&model_id).await?;
                serde_json::to_vec(&versions)
                    .map_err(|e| Status::internal(format!("Failed to serialize versions: {}", e)))
            }
            ModelCommand::DeleteVersion { model_id, version } => {
                self.model_storage
                    .delete_version(&model_id, &version)
                    .await?;
                Ok(vec![])
            }
        }
    }

    // Move handle_table_command before do_action
    async fn handle_table_command(&self, cmd: TableCommand) -> Result<Vec<u8>, Status> {
        match cmd {
            TableCommand::CreateTable { name, schema } => {
                self.backend.create_table(&name, &schema).await?;
                Ok(vec![])
            }
            TableCommand::CreateAggregationView(view) => {
                self.backend.create_aggregation_view(&view).await?;
                Ok(vec![])
            }
            TableCommand::DropTable(name) => {
                self.backend.drop_table(&name).await?;
                Ok(vec![])
            }
            TableCommand::DropAggregationView(name) => {
                self.backend.drop_aggregation_view(&name).await?;
                Ok(vec![])
            }
        }
    }

    // Fix handle_action to properly handle model commands
    async fn handle_action(&self, request: Request<Action>) -> Result<Vec<u8>, Status> {
        let (metadata, extensions, action) = request.into_parts();
        match Command::from_json(&action.body)? {
            Command::Table(table_cmd) => self.handle_table_command(table_cmd).await,
            Command::Model(model_cmd) => {
                // Validate auth token
                let token = self
                    .authenticate(&Request::from_parts(
                        metadata.clone(),
                        extensions.clone(),
                        (),
                    ))
                    .await?;

                match &model_cmd {
                    ModelCommand::StoreModel { model } => {
                        self.validate_command(&model_cmd, &token)?;
                        self.model_storage.store_model(model).await?;
                        Ok(vec![])
                    }
                    ModelCommand::LoadModel { model_id, version } => {
                        let model = self
                            .model_storage
                            .load_model(&model_id, version.as_deref())
                            .await?;
                        serde_json::to_vec(&model).map_err(|e| {
                            Status::internal(format!("Failed to serialize model: {}", e))
                        })
                    }
                    ModelCommand::ListModels => {
                        let models = self.model_storage.list_models().await?;
                        serde_json::to_vec(&models).map_err(|e| {
                            Status::internal(format!("Failed to serialize models: {}", e))
                        })
                    }
                    ModelCommand::ListVersions { model_id } => {
                        let versions = self.model_storage.list_versions(&model_id).await?;
                        serde_json::to_vec(&versions).map_err(|e| {
                            Status::internal(format!("Failed to serialize versions: {}", e))
                        })
                    }
                    ModelCommand::DeleteVersion { model_id, version } => {
                        self.model_storage
                            .delete_version(&model_id, &version)
                            .await?;
                        Ok(vec![])
                    }
                }
            }
            Command::Sql(sql_cmd) => {
                match sql_cmd {
                    SqlCommand::Execute(sql) => {
                        // Create query with execution hints
                        let query = Query {
                            sql,
                            schema_hint: None,
                            hints: vec![OptimizationHint::PreferPredicatePushdown],
                        };

                        // Plan and execute the query
                        let plan = self.planner.plan_query(&query).await.map_err(|e| {
                            Status::internal(format!("Query planning failed: {}", e))
                        })?;

                        self.executor.execute_collect(plan).await.map_err(|e| {
                            Status::internal(format!("Query execution failed: {}", e))
                        })?;

                        Ok(vec![])
                    }
                    SqlCommand::Query(sql) => {
                        // Create query with optimization hints
                        let query = Query {
                            sql,
                            schema_hint: None,
                            hints: vec![
                                OptimizationHint::PreferPredicatePushdown,
                                OptimizationHint::OptimizeForVectorOps,
                            ],
                        };

                        // Plan and execute the query
                        let plan = self.planner.plan_query(&query).await.map_err(|e| {
                            Status::internal(format!("Query planning failed: {}", e))
                        })?;

                        let results = self.executor.execute_collect(plan).await.map_err(|e| {
                            Status::internal(format!("Query execution failed: {}", e))
                        })?;

                        // Convert results to MetricRecords using the existing conversion function
                        let records: Vec<MetricRecord> = results
                            .into_iter()
                            .filter_map(|batch| encode_record_batch(&batch).ok())
                            .flatten()
                            .collect();

                        serde_json::to_vec(&records).map_err(|e| {
                            Status::internal(format!("Failed to serialize query results: {}", e))
                        })
                    }
                }
            }
        }
    }

    // Optimize large model transfers
    async fn stream_model_weights(
        &self,
        model_id: &str,
        version: Option<&str>,
        progress: Arc<TransferProgress>,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let model = self.model_storage.load_model(model_id, version).await?;

        const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks

        // Create a stream that yields Result<FlightData, Status>
        let stream = futures::stream::iter(model.layers).flat_map(move |layer| {
            let mut chunks = Vec::new();
            let mut current_chunk = Vec::with_capacity(CHUNK_SIZE);

            // Process weights into chunks
            for weights in layer.weights {
                if let Some(array) = weights.as_any().downcast_ref::<Float32Array>() {
                    // Zero-copy access to weight data
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            array.values().as_ptr() as *const u8,
                            array.len() * std::mem::size_of::<f32>(),
                        )
                    };

                    // Split into optimal chunk sizes
                    for chunk in bytes.chunks(CHUNK_SIZE) {
                        if current_chunk.len() + chunk.len() > CHUNK_SIZE {
                            chunks.push(Ok(FlightData {
                                data_header: Bytes::from(current_chunk.split_off(0)),
                                ..Default::default()
                            }));
                        }
                        current_chunk.extend_from_slice(chunk);
                        progress.update(chunk.len());
                    }
                } else {
                    chunks.push(Err(Status::internal("Invalid weight array type")));
                    break;
                }
            }

            // Push final chunk if any
            if !current_chunk.is_empty() {
                chunks.push(Ok(FlightData {
                    data_header: Bytes::from(current_chunk),
                    ..Default::default()
                }));
            }

            futures::stream::iter(chunks)
        });

        // Create boxed stream with correct type
        let stream: Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>> =
            Box::pin(stream);

        Ok(Response::new(stream))
    }

    async fn upload_model_weights(
        &self,
        mut stream: Streaming<FlightData>,
        model_id: String,
        version: Option<String>,
        progress: Arc<TransferProgress>,
    ) -> Result<Response<<Self as FlightService>::DoPutStream>, Status> {
        let mut weights = Vec::new();
        let mut current_layer = Vec::new();
        let mut total_size = 0;

        // Receive chunks and build weight arrays efficiently
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let data = chunk.data_header;

            // Zero-copy slice to f32 array
            let float_data = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const f32,
                    data.len() / std::mem::size_of::<f32>(),
                )
            };

            current_layer.extend_from_slice(float_data);
            total_size += data.len();
            progress.update(data.len());

            // Check if layer is complete
            if current_layer.len() >= 1024 * 1024 {
                // 1M elements per layer
                let mut builder = Float32Builder::with_capacity(current_layer.len());
                builder.append_slice(&current_layer);
                weights.push(Arc::new(builder.finish()) as ArrayRef);
                current_layer.clear();
            }
        }

        // Handle remaining data
        if !current_layer.is_empty() {
            let mut builder = Float32Builder::with_capacity(current_layer.len());
            builder.append_slice(&current_layer);
            weights.push(Arc::new(builder.finish()) as ArrayRef);
        }

        // Update model with new weights
        let mut model = self
            .model_storage
            .load_model(&model_id, version.as_deref())
            .await?;
        model.update_weights(weights)?;
        self.model_storage.store_model(&model).await?;

        let stream = futures::stream::once(async move {
            Ok(PutResult {
                app_metadata: Bytes::from(format!("Uploaded {} bytes", total_size)),
            })
        });

        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl FlightService for FlightSqlService {
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
            TableCommand::CreateAggregationView(view) => {
                let source_schema = self
                    .backend
                    .table_manager()
                    .get_table_schema(&view.source_table)
                    .await
                    .map_err(|_| Status::not_found("Source table not found"))?;
                Arc::new(source_schema)
            }
            _ => return Err(Status::invalid_argument("Command does not return schema")),
        };

        let generator = IpcDataGenerator::default();
        let options = IpcWriteOptions::default();
        let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
        let schema_data = generator.schema_to_bytes_with_dictionary_tracker(
            &schema,
            &mut dictionary_tracker,
            &options,
        );

        Ok(Response::new(SchemaResult {
            schema: Bytes::from(schema_data.ipc_message),
        }))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let cmd = serde_json::from_slice::<ModelCommand>(&request.get_ref().ticket)
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket: {}", e)))?;

        match cmd {
            ModelCommand::LoadModel { model_id, version } => {
                let progress = Arc::new(TransferProgress::new(0)); // Size will be set later
                self.stream_model_weights(&model_id, version.as_deref(), progress)
                    .await
            }
            _ => Err(Status::invalid_argument("Invalid command for do_get")),
        }
    }

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
        let tables = self.backend.table_manager().list_tables().await;

        let stream = futures::stream::iter(tables.into_iter().map(|table| {
            Ok(FlightInfo {
                schema: Bytes::new(),
                flight_descriptor: Some(FlightDescriptor {
                    r#type: 0,
                    cmd: Bytes::new(),
                    path: vec![table],
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

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();

        let table_name = descriptor
            .path
            .first()
            .ok_or_else(|| Status::invalid_argument("No table name provided"))?;

        let schema = self
            .backend
            .table_manager()
            .get_table_schema(table_name)
            .await
            .map_err(|_| Status::not_found(format!("Table {} not found", table_name)))?;

        let options = IpcWriteOptions::default();
        let generator = IpcDataGenerator::default();
        let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
        let schema_data = generator.schema_to_bytes_with_dictionary_tracker(
            &schema,
            &mut dictionary_tracker,
            &options,
        );

        Ok(Response::new(FlightInfo {
            schema: Bytes::from(schema_data.ipc_message),
            flight_descriptor: Some(descriptor),
            endpoint: vec![],
            total_records: -1,
            total_bytes: -1,
            ordered: false,
            app_metadata: Bytes::new(),
        }))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<<Self as FlightService>::DoPutStream>, Status> {
        // Fix metadata access and error handling
        let cmd_bytes = request
            .metadata()
            .get("x-command")
            .ok_or_else(|| Status::invalid_argument("Missing x-command metadata"))?
            .as_bytes();

        let cmd = serde_json::from_slice::<ModelCommand>(cmd_bytes)
            .map_err(|e| Status::invalid_argument(format!("Invalid command: {}", e)))?;

        match cmd {
            ModelCommand::StoreModel { model } => {
                let progress = Arc::new(TransferProgress::new(model.estimated_size()));
                self.upload_model_weights(
                    request.into_inner(),
                    model.id.clone(),
                    Some(model.version.clone()),
                    progress,
                )
                .await
            }
            _ => Err(Status::invalid_argument("Invalid command for do_put")),
        }
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
                r#type: "CreateAggregationView".to_string(),
                description: "Create a new aggregation view".to_string(),
            },
            ActionType {
                r#type: "DropTable".to_string(),
                description: "Drop an existing table".to_string(),
            },
            ActionType {
                r#type: "DropAggregationView".to_string(),
                description: "Drop an existing aggregation view".to_string(),
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
}
