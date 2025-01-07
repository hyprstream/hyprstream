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

use crate::metrics::create_record_batch;
use crate::storage::StorageBackend;
use crate::storage::table_manager::AggregationView;
use crate::metrics::MetricRecord;
use crate::aggregation::build_aggregate_query;
use arrow_flight::{
    flight_service_server::FlightService,
    Action, ActionType, Criteria, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
    Empty, PollInfo,
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};
use serde::Deserialize;
use arrow_ipc::writer::IpcWriteOptions;
use arrow_ipc::writer::IpcDataGenerator;
use arrow_schema::{Field, DataType, Schema};
use arrow_array::{
    Array, ArrayRef, StringArray, Int64Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Float32Array, BooleanArray,
    BinaryArray, TimestampNanosecondArray,
};

/// Command types for table and view operations
#[derive(Debug)]
enum TableCommand {
    CreateTable {
        name: String,
        schema: Arc<Schema>,
    },
    CreateAggregationView(AggregationView),
    DropTable(String),
    DropAggregationView(String),
}

impl TableCommand {
    fn from_json(cmd: &[u8]) -> Result<Self, Status> {
        #[derive(Deserialize)]
        struct CreateTableCmd {
            name: String,
            schema_bytes: Vec<u8>,
        }

        let value: serde_json::Value = serde_json::from_slice(cmd)
            .map_err(|e| Status::invalid_argument(format!("Invalid JSON: {}", e)))?;

        match value.get("type").and_then(|t| t.as_str()) {
            Some("create_table") => {
                let cmd: CreateTableCmd = serde_json::from_value(value["data"].clone())
                    .map_err(|e| Status::invalid_argument(format!("Invalid create table command: {}", e)))?;
                
                let schema = arrow_ipc::reader::StreamReader::try_new(
                    std::io::Cursor::new(&cmd.schema_bytes[..]),
                    None,
                ).map_err(|e| Status::invalid_argument(format!("Invalid schema bytes: {}", e)))?
                .schema().clone();
                
                Ok(TableCommand::CreateTable {
                    name: cmd.name,
                    schema,
                })
            }
            Some("create_aggregation_view") => {
                let view: AggregationView = serde_json::from_value(value["data"].clone())
                    .map_err(|e| Status::invalid_argument(format!("Invalid view command: {}", e)))?;
                Ok(TableCommand::CreateAggregationView(view))
            }
            Some("drop_table") => {
                let name = value["data"]["name"].as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing table name"))?;
                Ok(TableCommand::DropTable(name.to_string()))
            }
            Some("drop_aggregation_view") => {
                let name = value["data"]["name"].as_str()
                    .ok_or_else(|| Status::invalid_argument("Missing view name"))?;
                Ok(TableCommand::DropAggregationView(name.to_string()))
            }
            _ => Err(Status::invalid_argument("Invalid command type")),
        }
    }
}

#[derive(Clone)]
pub struct FlightSqlService {
    backend: Arc<Box<dyn StorageBackend>>,
}

impl FlightSqlService {
    pub fn new(backend: Box<dyn StorageBackend>) -> Self {
        Self { 
            backend: Arc::new(backend)
        }
    }
}

#[tonic::async_trait]
impl FlightService for FlightSqlService {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send + 'static>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

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
                let source_schema = self.backend.table_manager().get_table_schema(&view.source_table).await
                    .map_err(|_| Status::not_found("Source table not found"))?;
                Arc::new(source_schema)
            }
            _ => return Err(Status::invalid_argument("Command does not return schema")),
        };

        let generator = IpcDataGenerator::default();
        let options = IpcWriteOptions::default();
        let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
        let schema_data = generator.schema_to_bytes_with_dictionary_tracker(&schema, &mut dictionary_tracker, &options);

        Ok(Response::new(SchemaResult {
            schema: Bytes::from(schema_data.ipc_message),
        }))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        
        let _query = std::str::from_utf8(&ticket.ticket)
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket data: {}", e)))?;

        let metrics = self.backend.query_metrics(0).await?;
        let batch = create_record_batch(&metrics)?;
        
        let stream = futures::stream::once(async move {
            let generator = IpcDataGenerator::default();
            let options = IpcWriteOptions::default();
            let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
            let (encoded_dictionaries, encoded_batch) = generator.encoded_batch(&batch, &mut dictionary_tracker, &options)
                .map_err(|e| Status::internal(format!("Failed to encode batch: {}", e)))?;

            // First send schema message
            let schema_data = generator.schema_to_bytes_with_dictionary_tracker(&batch.schema(), &mut dictionary_tracker, &options);
            let mut flight_data = vec![FlightData {
                flight_descriptor: None,
                data_header: Bytes::from(schema_data.ipc_message),
                data_body: Bytes::new(),
                app_metadata: Bytes::new(),
            }];

            // Then send dictionary batches if any
            for dict_batch in encoded_dictionaries {
                flight_data.push(FlightData {
                    flight_descriptor: None,
                    data_header: Bytes::from(dict_batch.ipc_message),
                    data_body: Bytes::from(dict_batch.arrow_data),
                    app_metadata: Bytes::new(),
                });
            }

            // Finally send the record batch
            flight_data.push(FlightData {
                flight_descriptor: None,
                data_header: Bytes::from(encoded_batch.ipc_message),
                data_body: Bytes::from(encoded_batch.arrow_data),
                app_metadata: Bytes::new(),
            });

            Ok(flight_data.remove(0))
        });

        Ok(Response::new(Box::pin(stream)))
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
        
        let table_name = descriptor.path.first()
            .ok_or_else(|| Status::invalid_argument("No table name provided"))?;
            
        let schema = self.backend.table_manager().get_table_schema(table_name).await
            .map_err(|_| Status::not_found(format!("Table {} not found", table_name)))?;
            
        let options = IpcWriteOptions::default();
        let mut generator = IpcDataGenerator::default();
        let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
        let schema_data = generator.schema_to_bytes_with_dictionary_tracker(&schema, &mut dictionary_tracker, &options);
        
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
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();
        let mut metrics = Vec::new();
        
        while let Some(data) = stream.next().await {
            let data = data?;
            
            let reader = arrow_ipc::reader::StreamReader::try_new(
                std::io::Cursor::new(data.data_body),
                None,
            ).map_err(|e| Status::internal(format!("Failed to create IPC reader: {}", e)))?;
            
            for batch in reader {
                let batch = batch.map_err(|e| Status::internal(format!("Failed to read record batch: {}", e)))?;
                let batch_metrics = MetricRecord::try_from_record_batch(&batch)
                    .map_err(|e| Status::internal(format!("Failed to convert batch: {}", e)))?;
                metrics.extend(batch_metrics);
            }
        }
        
        self.backend.insert_metrics(metrics).await?;
        
        let stream = futures::stream::iter(vec![Ok(PutResult::default())]);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        
        let cmd = TableCommand::from_json(&action.body)?;
        
        match cmd {
            TableCommand::CreateTable { name, schema } => {
                self.backend.create_table(&name, &schema).await?;
            }
            TableCommand::CreateAggregationView(view) => {
                let columns = vec!["metric_id", "timestamp", "value_running_window_sum", 
                                 "value_running_window_avg", "value_running_window_count"];
                let _sql = build_aggregate_query(
                    &view.source_table,
                    view.function,
                    &view.group_by,
                    &columns,
                    None,
                    None
                );
                self.backend.create_aggregation_view(&view).await?;
            }
            TableCommand::DropTable(name) => {
                self.backend.drop_table(&name).await?;
            }
            TableCommand::DropAggregationView(name) => {
                self.backend.drop_aggregation_view(&name).await?;
            }
        }
        
        let stream = futures::stream::iter(vec![Ok(arrow_flight::Result::default())]);
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
}
