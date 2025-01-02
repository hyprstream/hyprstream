use arrow::array::{Float64Array, Int64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, Result as FlightResult, SchemaResult,
    Ticket,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream, Stream};
use std::io::Cursor;
use std::pin::Pin;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status, Streaming};

mod storage;
use storage::{MetricRecord, StorageBackend};

#[derive(Clone)]
struct FlightServiceImpl {
    backend: Arc<storage::cached::CachedStorageBackend>,
}

#[async_trait]
impl FlightService for FlightServiceImpl {
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send + 'static>>;
    type DoExchangeStream =
        Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;
    type ListActionsStream =
        Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send + 'static>>;
    type ListFlightsStream =
        Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send + 'static>>;
    type HandshakeStream =
        Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send + 'static>>;
    type DoActionStream =
        Pin<Box<dyn Stream<Item = Result<FlightResult, Status>> + Send + 'static>>;

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let timestamp = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|e| Status::invalid_argument(e.to_string()))?
            .parse::<i64>()
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let data = self.backend.query_metrics(timestamp).await?;
        let batch = create_record_batch(data)?;
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

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        match action.r#type.as_str() {
            "ExecuteSql" => {
                let sql = String::from_utf8(action.body.to_vec())
                    .map_err(|e| Status::invalid_argument(format!("Invalid SQL string: {}", e)))?;

                if let Some((start_time, _end_time)) = parse_sql_timestamps(&sql) {
                    let data = self.backend.query_metrics(start_time).await?;
                    let batch = create_record_batch(data)?;
                    let (_header, body) = encode_record_batch(&batch)?;

                    Ok(Response::new(Box::pin(stream::once(async move {
                        Ok(FlightResult { body: body.into() })
                    }))))
                } else {
                    Err(Status::invalid_argument("Could not parse SQL query"))
                }
            }
            _ => Err(Status::unimplemented("Action type not supported")),
        }
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info not implemented"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]);

        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema)
                .map_err(|e| Status::internal(format!("Failed to create schema writer: {}", e)))?;
            writer
                .finish()
                .map_err(|e| Status::internal(format!("Failed to write schema: {}", e)))?;
        }

        Ok(Response::new(SchemaResult {
            schema: buffer.into(),
        }))
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();

        let first_message = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("Empty stream"))?;

        let reader = StreamReader::try_new(Cursor::new(&first_message.data_header[..]), None)
            .map_err(|e| Status::internal(format!("Invalid schema: {}", e)))?;
        let schema = Arc::new(reader.schema());

        let mut records = Vec::new();

        // Process incoming record batches
        while let Some(data) = stream.message().await? {
            let mut reader = StreamReader::try_new(
                Cursor::new(&data.data_body),
                None, // We don't need to specify projection since we want all columns
            )
            .map_err(|e| Status::internal(format!("Failed to create reader: {}", e)))?;

            while let Some(batch) = reader.next() {
                let batch =
                    batch.map_err(|e| Status::internal(format!("Failed to read batch: {}", e)))?;

                // Convert Arrow batch to MetricRecords
                for row_idx in 0..batch.num_rows() {
                    let metric_id = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| Status::internal("Invalid metric_id column"))?
                        .value(row_idx)
                        .to_string();

                    let timestamp = batch
                        .column(1)
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .ok_or_else(|| Status::internal("Invalid timestamp column"))?
                        .value(row_idx);

                    let value_sum = batch
                        .column(2)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid sum column"))?
                        .value(row_idx);

                    let value_avg = batch
                        .column(3)
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| Status::internal("Invalid avg column"))?
                        .value(row_idx);

                    let value_count = batch
                        .column(4)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| Status::internal("Invalid count column"))?
                        .value(row_idx);

                    records.push(MetricRecord {
                        metric_id,
                        timestamp,
                        value_running_window_sum: value_sum,
                        value_running_window_avg: value_avg,
                        value_running_window_count: value_count,
                    });
                }
            }
        }

        // Insert records into storage backend
        self.backend.insert_metrics(records).await?;

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(PutResult {
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions not implemented"))
    }

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("handshake not implemented"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]);

        let mut schema_bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut schema_bytes, &schema)
                .map_err(|e| Status::internal(format!("Failed to create schema writer: {}", e)))?;
            writer
                .finish()
                .map_err(|e| Status::internal(format!("Failed to write schema: {}", e)))?;
        }

        Ok(Response::new(Box::pin(stream::once(async move {
            Ok(FlightInfo {
                schema: schema_bytes.into(),
                flight_descriptor: None,
                endpoint: vec![],
                total_records: -1,
                total_bytes: -1,
                ordered: false,
                app_metadata: Bytes::new(),
            })
        }))))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }
}

// Helper functions
fn parse_sql_timestamps(sql: &str) -> Option<(i64, i64)> {
    let sql = sql.to_lowercase();
    if let (Some(start_idx), Some(end_idx)) =
        (sql.find("timestamp >= "), sql.find("and timestamp <= "))
    {
        let start_time = sql[start_idx + 12..]
            .split_whitespace()
            .next()?
            .parse::<i64>()
            .ok()?;

        let end_time = sql[end_idx + 15..]
            .split_whitespace()
            .next()?
            .parse::<i64>()
            .ok()?;

        Some((start_time, end_time))
    } else {
        None
    }
}

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;

    // Initialize backends
    let duckdb = Arc::new(storage::duckdb::DuckDbBackend::new());
    let redis = Arc::new(storage::redis::RedisBackend::new("redis://127.0.0.1/")?);

    // Create cached backend
    let backend = Arc::new(storage::cached::CachedStorageBackend::new(
        duckdb, redis, 3600, // Cache duration in seconds
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
