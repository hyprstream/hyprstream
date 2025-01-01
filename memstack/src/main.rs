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

// Shared DuckDB connection pool
#[derive(Clone)]
struct Db {
    conn: Arc<Mutex<Connection>>,
}

impl Db {
    fn new() -> Self {
        let conn = Connection::open_in_memory().unwrap();
        Db {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    async fn execute_query(&self, query: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        conn.execute_batch(query).map_err(|e| Status::internal(format!("Query failed: {}", e)))
    }

    async fn query(&self, query: &str) -> Result<Vec<(String, i64, f64, f64, i64)>, Status> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(query).map_err(|e| Status::internal(format!("Prepare failed: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .map_err(|e| Status::internal(format!("Query execution failed: {}", e)))?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| Status::internal(format!("Row mapping failed: {}", e)))?);
        }
        Ok(results)
    }
}

// FlightService implementation
#[derive(Clone)]
struct FlightServiceImpl {
    db: Db,
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

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = String::from_utf8(request.into_inner().ticket.to_vec())
            .map_err(|e| Status::invalid_argument(format!("Invalid ticket: {}", e)))?;

        let query = format!(
            "SELECT metric_id, timestamp, valueRunningWindowSum, valueRunningWindowAvg, valueRunningWindowCount \
            FROM metrics WHERE timestamp >= '{}' LIMIT 100;",
            ticket
        );

        let rows = self.db.query(&query).await?;
        
        // Convert rows to Arrow RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("valueRunningWindowSum", DataType::Float64, false),
            Field::new("valueRunningWindowAvg", DataType::Float64, false),
            Field::new("valueRunningWindowCount", DataType::Int64, false),
        ]));

        let (metric_id, timestamp, sum, avg, count): (Vec<String>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>) = 
            rows.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
                |mut acc, row| {
                    acc.0.push(row.0);
                    acc.1.push(row.1);
                    acc.2.push(row.2);
                    acc.3.push(row.3);
                    acc.4.push(row.4);
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
    let db = Db::new();

    // Create DuckDB table
    db.execute_query(
        "CREATE TABLE metrics (
            metric_id TEXT,
            timestamp TIMESTAMP,
            valueRunningWindowSum DOUBLE,
            valueRunningWindowAvg DOUBLE,
            valueRunningWindowCount INTEGER
        );",
    )
    .await?;

    let service = FlightServiceImpl { db };

    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
