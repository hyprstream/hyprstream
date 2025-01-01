use arrow::array::{Float64Array, Int64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::ipc::writer::IpcWriteOptions;
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, FlightData, FlightDescriptor, FlightInfo, HandshakeRequest,
    HandshakeResponse, PutResult, SchemaResult, Ticket, Empty as FlightEmpty,
};
use bytes::Bytes;
use duckdb::Connection;
use futures::stream::{self, Stream};
use std::sync::Arc;
use std::pin::Pin;
use tokio::sync::Mutex;
use tonic::{transport::Server, Request, Response, Status};
mod storage;
use storage::{StorageBackend, MetricRecord};

// Replace the Db struct with StorageWrapper
#[derive(Clone)]
struct StorageWrapper {
    backend: Arc<dyn StorageBackend>,
}

// Update FlightServiceImpl to use StorageWrapper
#[derive(Clone)]
struct FlightServiceImpl {
    storage: StorageWrapper,
}

#[tonic::async_trait]
impl FlightService for FlightServiceImpl {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send + 'static>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let output = stream::once(async move {
            Ok(HandshakeResponse {
                protocol_version: 0,
                ..Default::default()
            })
        });
        Ok(Response::new(Box::pin(output)))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let output = stream::empty();
        Ok(Response::new(Box::pin(output)))
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
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema not implemented"))
    }

    /*async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket: {}", e)))?;*/

    // In the do_get implementation, replace the db.query with:
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket: {}", e)))?;
        
        // Parse start and end timestamps from comma-separated ticket
        let timestamps: Vec<&str> = ticket.split(',').collect();
        if timestamps.len() != 2 {
            return Err(Status::invalid_argument("Ticket must contain start,end timestamps"));
        }

        let start_timestamp = timestamps[0].parse::<i64>()
            .map_err(|e| Status::invalid_argument(format!("Invalid start timestamp: {}", e)))?;
        let end_timestamp = timestamps[1].parse::<i64>()
            .map_err(|e| Status::invalid_argument(format!("Invalid end timestamp: {}", e)))?;

        let records = self.storage.backend.query_metrics(start_timestamp).await?;
        
        // Filter records within time window
        let records: Vec<_> = records.into_iter()
            .filter(|r| r.timestamp >= start_timestamp && r.timestamp <= end_timestamp)
            .collect();

        // Convert records to Arrow RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("valueRunningWindowSum", DataType::Float64, false),
            Field::new("valueRunningWindowAvg", DataType::Float64, false),
            Field::new("valueRunningWindowCount", DataType::Int64, false),
        ]));

        let (metric_id, timestamp, sum, avg, count): (Vec<String>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>) = 
            records.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |mut acc, record| {
                    acc.0.push(record.metric_id);
                    acc.1.push(record.timestamp);
                    acc.2.push(record.value_running_window_sum);
                    acc.3.push(record.value_running_window_avg);
                    acc.4.push(record.value_running_window_count);
                    acc
                }
            );

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(metric_id)),
                Arc::new(TimestampNanosecondArray::from(timestamp)),
                Arc::new(Float64Array::from(sum)),
                Arc::new(Float64Array::from(avg)),
                Arc::new(Int64Array::from(count)),
            ],
        )
        .map_err(|e| Status::internal(format!("RecordBatch creation failed: {}", e)))?;

        let flight_data = flight_data_from_batch(&batch);
        let output = stream::iter(flight_data.into_iter().map(Ok));
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let output = stream::once(async move { Ok(PutResult::default()) });
        println!("do_put called");
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let output = stream::empty();
        Ok(Response::new(Box::pin(output)))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let output = stream::empty();
        Ok(Response::new(Box::pin(output)))
    }

    async fn list_actions(
        &self,
        _request: Request<arrow_flight::Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let output = stream::empty();
        Ok(Response::new(Box::pin(output)))
    }
}

// Helper function to convert RecordBatch to FlightData
fn flight_data_from_batch(batch: &RecordBatch) -> Vec<FlightData> {
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(Vec::new(), &batch.schema())
        .expect("Failed to create IPC writer");
    
    writer.write(batch).expect("Failed to write batch");
    writer.finish().expect("Failed to finish writing");
    
    let data = writer.into_inner().expect("Failed to get written data");
    
    vec![FlightData {
        //data_header: writer.schema().to_ipc_message(),
        data_body: data.into(),
        app_metadata: Bytes::new(),
        ..Default::default()
    }]
}

// Main entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;
    
    // Initialize both backends
    let duckdb_backend = storage::duckdb::DuckDbBackend::new();
    let redis_backend = storage::redis::RedisBackend::new("redis://127.0.0.1/")?;
    
    // Create cached storage with DuckDB as cache and Redis as backing store
    let backend = storage::cached::CachedStorageBackend::new(
        Arc::new(duckdb_backend),
        Arc::new(redis_backend),
        3600, // Cache data for 1 hour
    );
    
    let storage = StorageWrapper {
        backend: Arc::new(backend),
    };

    // Initialize the storage
    storage.backend.init().await?;

    let service = FlightServiceImpl { storage };

    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
