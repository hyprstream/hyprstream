//! Inference service providing FlightSQL interface for inference requests

use crate::inference::{InferenceInput, InferenceOutput, InferenceAPI};
#[cfg(feature = "vdb")]
use crate::storage::vdb::hardware_accelerated::HardwareVDBStorage;

use std::sync::Arc;
use std::collections::HashMap;
use arrow_array::{RecordBatch, StringArray, Float32Array, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use arrow_flight::{
    flight_service_server::FlightService, Action, ActionType, Criteria, Empty, FlightData,
    FlightDescriptor, FlightInfo, HandshakeRequest, HandshakeResponse, PutResult, SchemaResult,
    Ticket,
};
use futures::stream::{self, Stream};
use tonic::{Request, Response, Status, Streaming};
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Inference request via FlightSQL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Session ID for inference
    pub session_id: String,
    
    /// Input prompt or query
    pub prompt: String,
    
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    
    /// Temperature for sampling
    pub temperature: Option<f32>,
    
    /// Top-p for nucleus sampling
    pub top_p: Option<f32>,
    
    /// Whether to stream the response
    pub stream: Option<bool>,
    
    /// Additional parameters
    pub parameters: Option<HashMap<String, serde_json::Value>>,
}

/// Inference response via FlightSQL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated text
    pub text: String,
    
    /// Number of tokens generated
    pub tokens_generated: usize,
    
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    
    /// Active adapters and their contributions
    pub adapter_contributions: HashMap<String, f32>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Flight service for inference requests
pub struct InferenceFlightService {
    /// Inference API backend
    inference_api: Arc<InferenceAPI>,
    
    /// VDB storage for adapter management (only available with VDB feature)
    #[cfg(feature = "vdb")]
    vdb_storage: Arc<HardwareVDBStorage>,
    
    /// Service statistics
    stats: tokio::sync::RwLock<ServiceStats>,
}

/// Service statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceStats {
    pub total_requests: u64,
    pub active_streams: u64,
    pub avg_response_time_ms: f64,
    pub total_tokens_generated: u64,
    pub errors: u64,
    pub last_request_time: i64,
}

impl InferenceFlightService {
    /// Create new inference flight service
    pub fn new(
        inference_api: Arc<InferenceAPI>,
        #[cfg(feature = "vdb")] vdb_storage: Arc<HardwareVDBStorage>,
    ) -> Self {
        Self {
            inference_api,
            #[cfg(feature = "vdb")]
            vdb_storage,
            stats: tokio::sync::RwLock::new(ServiceStats::default()),
        }
    }
    
    /// Process inference request
    async fn process_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = std::time::Instant::now();
        
        // Convert to internal inference input
        let input = InferenceInput {
            prompt: Some(request.prompt),
            input_ids: None,
            max_tokens: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            stream: request.stream.unwrap_or(false),
        };
        
        // Run inference
        let output = self.inference_api.infer(&request.session_id, input).await?;
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.total_tokens_generated += output.tokens_generated as u64;
            
            let response_time = start_time.elapsed().as_millis() as f64;
            stats.avg_response_time_ms = (stats.avg_response_time_ms * (stats.total_requests - 1) as f64 + response_time)
                / stats.total_requests as f64;
            
            stats.last_request_time = chrono::Utc::now().timestamp();
        }
        
        // Create response with metadata
        let mut metadata = HashMap::new();
        metadata.insert("session_id".to_string(), serde_json::Value::String(request.session_id));
        metadata.insert("inference_time_ms".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(output.latency_ms).unwrap_or(serde_json::Number::from(0))
        ));
        
        Ok(InferenceResponse {
            text: output.text,
            tokens_generated: output.tokens_generated,
            latency_ms: output.latency_ms,
            adapter_contributions: output.adapter_contribution,
            metadata,
        })
    }
    
    /// Convert inference response to Arrow RecordBatch
    fn response_to_record_batch(&self, responses: Vec<InferenceResponse>) -> Result<RecordBatch> {
        let mut texts = Vec::new();
        let mut tokens_generated = Vec::new();
        let mut latencies = Vec::new();
        
        for response in responses {
            texts.push(Some(response.text));
            tokens_generated.push(response.tokens_generated as i64);
            latencies.push(response.latency_ms as f32);
        }
        
        let schema = Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("tokens_generated", DataType::Int64, false),
            Field::new("latency_ms", DataType::Float32, false),
        ]);
        
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(texts)),
                Arc::new(Int64Array::from(tokens_generated)),
                Arc::new(Float32Array::from(latencies)),
            ],
        )?;
        
        Ok(batch)
    }
    
    /// Get service statistics
    pub async fn get_stats(&self) -> ServiceStats {
        self.stats.read().await.clone()
    }
    
    /// Reset service statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = ServiceStats::default();
    }
}

#[tonic::async_trait]
impl FlightService for InferenceFlightService {
    type HandshakeStream = std::pin::Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;
    type ListFlightsStream = std::pin::Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send>>;
    type DoGetStream = std::pin::Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
    type DoPutStream = std::pin::Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send>>;
    type DoActionStream = std::pin::Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;
    type ListActionsStream = std::pin::Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send>>;
    type DoExchangeStream = std::pin::Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
    
    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let response = HandshakeResponse::default();
        let stream = stream::once(async { Ok(response) });
        Ok(Response::new(Box::pin(stream)))
    }
    
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let flights = vec![
            FlightInfo {
                schema: vec![], // Schema bytes would go here
                flight_descriptor: Some(FlightDescriptor {
                    r#type: 1, // CMD
                    cmd: b"inference".to_vec(),
                    path: vec![],
                }),
                endpoint: vec![],
                total_records: -1,
                total_bytes: -1,
            }
        ];
        
        let stream = stream::iter(flights.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }
    
    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        
        match &descriptor.r#type {
            1 => { // CMD
                if descriptor.cmd == b"inference" {
                    let info = FlightInfo {
                        schema: vec![], // Would contain actual schema
                        flight_descriptor: Some(descriptor),
                        endpoint: vec![],
                        total_records: -1,
                        total_bytes: -1,
                    };
                    return Ok(Response::new(info));
                }
            }
            _ => {}
        }
        
        Err(Status::not_found("Flight not found"))
    }
    
    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();
        
        if descriptor.cmd == b"inference" {
            let schema = Schema::new(vec![
                Field::new("text", DataType::Utf8, true),
                Field::new("tokens_generated", DataType::Int64, false),
                Field::new("latency_ms", DataType::Float32, false),
            ]);
            
            let schema_bytes = {
                use arrow_ipc::writer::IpcWriteOptions;
                let mut buf = Vec::new();
                {
                    let options = IpcWriteOptions::default();
                    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buf, &schema)?;
                    writer.finish()?;
                }
                buf
            };
            
            return Ok(Response::new(SchemaResult { schema: schema_bytes }));
        }
        
        Err(Status::not_found("Schema not found"))
    }
    
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        
        // Parse inference request from ticket
        let request_str = String::from_utf8(ticket.ticket)
            .map_err(|_| Status::invalid_argument("Invalid ticket format"))?;
            
        let inference_request: InferenceRequest = serde_json::from_str(&request_str)
            .map_err(|_| Status::invalid_argument("Invalid inference request"))?;
        
        // Process inference
        match self.process_inference(inference_request).await {
            Ok(response) => {
                let batch = self.response_to_record_batch(vec![response])
                    .map_err(|e| Status::internal(format!("Failed to create batch: {}", e)))?;
                
                // Convert to FlightData
                let mut dictionary_tracker = arrow_ipc::writer::DictionaryTracker::new(false);
                let (encoded_data, encoded_schema) = {
                    use arrow_ipc::writer::{IpcDataGenerator, IpcWriteOptions};
                    let options = IpcWriteOptions::default();
                    let data_gen = IpcDataGenerator::default();
                    let encoded_dictionaries = data_gen.encoded_batch(&batch, &mut dictionary_tracker, &options)
                        .map_err(|e| Status::internal(format!("Failed to encode batch: {}", e)))?;
                    
                    let schema_message = arrow_ipc::writer::encode_schema(batch.schema().as_ref());
                    let encoded_schema = data_gen.schema_to_bytes(&batch.schema(), &schema_message);
                    
                    (encoded_dictionaries, encoded_schema)
                };
                
                let mut flight_data = Vec::new();
                
                // Add schema
                flight_data.push(Ok(FlightData {
                    flight_descriptor: None,
                    data_header: encoded_schema,
                    app_metadata: vec![],
                    data_body: vec![],
                }));
                
                // Add data
                flight_data.push(Ok(FlightData {
                    flight_descriptor: None,
                    data_header: encoded_data.ipc_message,
                    app_metadata: vec![],
                    data_body: encoded_data.arrow_data,
                }));
                
                let stream = stream::iter(flight_data);
                Ok(Response::new(Box::pin(stream)))
            }
            Err(e) => Err(Status::internal(format!("Inference failed: {}", e))),
        }
    }
    
    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("DoPut not implemented"))
    }
    
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        
        match action.r#type.as_str() {
            "get_stats" => {
                let stats = self.get_stats().await;
                let stats_json = serde_json::to_vec(&stats)
                    .map_err(|e| Status::internal(format!("Failed to serialize stats: {}", e)))?;
                
                let result = arrow_flight::Result {
                    body: stats_json,
                };
                
                let stream = stream::once(async { Ok(result) });
                Ok(Response::new(Box::pin(stream)))
            }
            "reset_stats" => {
                self.reset_stats().await;
                
                let result = arrow_flight::Result {
                    body: b"Stats reset successfully".to_vec(),
                };
                
                let stream = stream::once(async { Ok(result) });
                Ok(Response::new(Box::pin(stream)))
            }
            _ => Err(Status::unimplemented(format!("Action '{}' not implemented", action.r#type))),
        }
    }
    
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "get_stats".to_string(),
                description: "Get service statistics".to_string(),
            },
            ActionType {
                r#type: "reset_stats".to_string(),
                description: "Reset service statistics".to_string(),
            },
        ];
        
        let stream = stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }
    
    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange not implemented"))
    }
}

use chrono;