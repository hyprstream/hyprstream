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

use crate::metrics::{create_record_batch, encode_record_batch, get_metrics_schema};
use crate::storage::StorageBackend;
use arrow_flight::{
    flight_service_server::FlightService,
    Action, ActionType, Criteria, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
    Location, FlightEndpoint, Empty, Result as FlightResult,
};
use arrow_ipc::{writer::StreamWriter, reader::StreamReader};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};

pub struct FlightServiceImpl {
    backend: Arc<dyn StorageBackend>,
    cache: Option<Arc<dyn StorageBackend>>,
}

impl FlightServiceImpl {
    pub fn new(backend: Arc<dyn StorageBackend>) -> Self {
        Self {
            backend,
            cache: None,
        }
    }

    pub fn new_with_cache(backend: Arc<dyn StorageBackend>, cache: Arc<dyn StorageBackend>) -> Self {
        Self {
            backend,
            cache: Some(cache),
        }
    }
}

type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
type DoActionStream = Pin<Box<dyn Stream<Item = Result<FlightResult, Status>> + Send + 'static>>;
type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for FlightServiceImpl {
    type HandshakeStream = HandshakeStream;
    type ListFlightsStream = ListFlightsStream;
    type DoGetStream = DoGetStream;
    type DoPutStream = DoPutStream;
    type DoExchangeStream = DoExchangeStream;
    type DoActionStream = DoActionStream;
    type ListActionsStream = ListActionsStream;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let output = vec![Ok(HandshakeResponse {
            protocol_version: 0,
            payload: Bytes::new(),
        })];
        Ok(Response::new(Box::pin(tokio_stream::iter(output))))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Ok(Response::new(Box::pin(tokio_stream::iter(vec![]))))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Ok(Response::new(FlightInfo {
            schema: Bytes::new(),
            flight_descriptor: None,
            endpoint: vec![],
            total_records: -1,
            total_bytes: -1,
            app_metadata: Bytes::new(),
            ordered: false,
        }))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = get_metrics_schema();
        let mut schema_buffer = Vec::new();
        let mut writer = StreamWriter::try_new(&mut schema_buffer, &schema)
            .map_err(|e| Status::internal(format!("Failed to create writer: {}", e)))?;

        writer.finish()
            .map_err(|e| Status::internal(format!("Failed to finish writer: {}", e)))?;

        Ok(Response::new(SchemaResult {
            schema: schema_buffer.into(),
        }))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let metrics = self.backend.query_sql(&ticket.ticket).await?;
        let batch = create_record_batch(&metrics)?;

        let schema = get_metrics_schema();
        let mut schema_buffer = Vec::new();
        let mut writer = StreamWriter::try_new(&mut schema_buffer, &schema)
            .map_err(|e| Status::internal(format!("Failed to create writer: {}", e)))?;

        writer.write(&batch)
            .map_err(|e| Status::internal(format!("Failed to write batch: {}", e)))?;

        writer.finish()
            .map_err(|e| Status::internal(format!("Failed to finish writer: {}", e)))?;

        let mut data_buffer = Vec::new();
        let mut writer = StreamWriter::try_new(&mut data_buffer, &schema)
            .map_err(|e| Status::internal(format!("Failed to create writer: {}", e)))?;

        writer.write(&batch)
            .map_err(|e| Status::internal(format!("Failed to write batch: {}", e)))?;

        writer.finish()
            .map_err(|e| Status::internal(format!("Failed to finish writer: {}", e)))?;

        let flight_data = FlightData {
            data_header: schema_buffer.into(),
            data_body: data_buffer.into(),
            app_metadata: Bytes::new(),
            ..Default::default()
        };

        Ok(Response::new(Box::pin(tokio_stream::iter(vec![Ok(flight_data)]))))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();
        let mut metrics = Vec::new();

        while let Some(data) = stream.next().await {
            let data = data?;
            let schema = get_metrics_schema();
            let reader = StreamReader::try_new(&data.data_body[..], None)
                .map_err(|e| Status::internal(format!("Failed to create reader: {}", e)))?;

            for batch in reader {
                let batch = batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;
                let mut batch_metrics = encode_record_batch(&batch)?;
                metrics.append(&mut batch_metrics);
            }
        }

        self.backend.insert_metrics(metrics).await?;

        Ok(Response::new(Box::pin(tokio_stream::iter(vec![Ok(PutResult {
            app_metadata: Bytes::new(),
        })]))))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(Box::pin(tokio_stream::iter(vec![]))))
    }
}
