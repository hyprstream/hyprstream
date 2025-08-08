//! FlightSQL service for inference with LoRA adapters

use crate::api::{ApiState, LoRAEndpoint};
use crate::service::embedding_flight::EmbeddingFlightService;
use crate::storage::vdb::sparse_storage::SparseStorage;

use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
};
use arrow_schema::{DataType, Field, Schema};
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};

/// Inference request for FlightSQL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// LoRA adapter ID to use
    pub lora_id: String,
    
    /// Input prompt or messages
    pub input: InferenceInputType,
    
    /// Generation parameters
    pub params: GenerationParams,
}

/// Input types for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInputType {
    /// Simple text prompt
    Prompt(String),
    
    /// Chat messages (OpenAI format)
    Messages(Vec<ChatMessage>),
    
    /// Token IDs
    TokenIds(Vec<i64>),
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(2048),
            temperature: Some(1.0),
            top_p: Some(1.0),
            stream: Some(false),
        }
    }
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub lora_id: String,
    pub output: String,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub finish_reason: String,
}

/// FlightSQL service for inference operations
pub struct InferenceFlightService {
    /// API state with LoRA registry and training service
    api_state: ApiState,
    
    /// Base embedding service for non-inference operations
    embedding_service: EmbeddingFlightService,
}

impl InferenceFlightService {
    /// Create new inference FlightSQL service
    pub fn new(api_state: ApiState) -> Self {
        // Create embedding service as fallback
        let embedding_service = EmbeddingFlightService::new(
            api_state.vdb_storage.clone() as Arc<dyn SparseStorage>
        );
        
        Self {
            api_state,
            embedding_service,
        }
    }
    
    /// Parse inference request from FlightSQL ticket
    fn parse_inference_request(&self, ticket: &Ticket) -> Result<InferenceRequest, Status> {
        let query_str = std::str::from_utf8(&ticket.ticket)
            .map_err(|_| Status::invalid_argument("Invalid ticket encoding"))?;
        
        serde_json::from_str(query_str)
            .map_err(|e| Status::invalid_argument(format!("Invalid inference request: {}", e)))
    }
    
    /// Handle inference request
    async fn handle_inference(&self, request: InferenceRequest) -> Result<InferenceResponse, Status> {
        let start = std::time::Instant::now();
        
        // Check if LoRA exists
        let endpoint = {
            let endpoints = self.api_state.endpoints.read().await;
            endpoints.get(&request.lora_id)
                .ok_or_else(|| Status::not_found(format!("LoRA {} not found", request.lora_id)))?
                .clone()
        };
        
        // Create inference session
        let session_id = self.api_state.training_service
            .create_inference_session(&request.lora_id, vec![request.lora_id.clone()])
            .await
            .map_err(|e| Status::internal(format!("Failed to create session: {}", e)))?;
        
        // Convert request to inference input
        let input = self.convert_to_inference_input(request.input, request.params)?;
        
        // Run inference
        let output = self.api_state.training_service
            .infer(&session_id, input)
            .await
            .map_err(|e| Status::internal(format!("Inference failed: {}", e)))?;
        
        // Close session
        let _ = self.api_state.training_service
            .close_inference_session(&session_id)
            .await;
        
        // Return response
        Ok(InferenceResponse {
            lora_id: request.lora_id,
            output: output.text,
            tokens_generated: output.tokens_generated,
            latency_ms: start.elapsed().as_millis() as f64,
            finish_reason: "stop".to_string(),
        })
    }
    
    /// Convert inference input to internal format
    fn convert_to_inference_input(
        &self,
        input: InferenceInputType,
        params: GenerationParams,
    ) -> Result<crate::inference::InferenceInput, Status> {
        let prompt = match input {
            InferenceInputType::Prompt(text) => text,
            InferenceInputType::Messages(messages) => {
                messages.iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            InferenceInputType::TokenIds(tokens) => {
                return Ok(crate::inference::InferenceInput {
                    prompt: None,
                    input_ids: Some(tokens),
                    max_tokens: params.max_tokens.unwrap_or(2048),
                    temperature: params.temperature.unwrap_or(1.0),
                    top_p: params.top_p.unwrap_or(1.0),
                    stream: params.stream.unwrap_or(false),
                });
            }
        };
        
        Ok(crate::inference::InferenceInput {
            prompt: Some(prompt),
            input_ids: None,
            max_tokens: params.max_tokens.unwrap_or(2048),
            temperature: params.temperature.unwrap_or(1.0),
            top_p: params.top_p.unwrap_or(1.0),
            stream: params.stream.unwrap_or(false),
        })
    }
    
    /// Schema for inference results
    fn inference_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("lora_id", DataType::Utf8, false),
            Field::new("output", DataType::Utf8, false),
            Field::new("tokens_generated", DataType::UInt64, false),
            Field::new("latency_ms", DataType::Float64, false),
            Field::new("finish_reason", DataType::Utf8, false),
        ])
    }
}

#[tonic::async_trait]
impl FlightService for InferenceFlightService {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;

    /// Handshake for inference service
    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        self.embedding_service.handshake(request).await
    }

    /// List available inference operations
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let descriptor = FlightDescriptor::new_path(vec!["inference".to_string()]);
        let mut inference_flight = FlightInfo::new()
            .with_descriptor(descriptor)
            .with_total_records(-1)
            .with_total_bytes(-1);
            
        // Set schema
        inference_flight = inference_flight.try_with_schema(&self.inference_schema())
            .map_err(|e| Status::internal(format!("Failed to set schema: {}", e)))?;

        let stream = stream::iter(vec![Ok(inference_flight)]);
        Ok(Response::new(Box::pin(stream)))
    }

    /// Execute inference requests
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        
        // Try to parse as inference request first
        match self.parse_inference_request(&ticket) {
            Ok(inference_request) => {
                println!("🤖 Processing inference request for LoRA: {}", inference_request.lora_id);
                
                let response = self.handle_inference(inference_request).await?;
                
                // Convert response to FlightData
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| Status::internal(format!("Serialization error: {}", e)))?;
                
                let flight_data = FlightData::new()
                    .with_data_body(response_json.into_bytes());
                
                let stream = stream::iter(vec![Ok(flight_data)]);
                Ok(Response::new(Box::pin(stream)))
            }
            Err(_) => {
                // Fallback to embedding service
                self.embedding_service.do_get(Request::new(ticket)).await
            }
        }
    }

    /// Get schema for inference operations
    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();
        
        // Check if it's an inference schema request
        if descriptor.path.get(0).map(|p| p.as_str()) == Some("inference") {
            let schema = self.inference_schema();
            let schema_message = arrow_ipc::writer::IpcWriteOptions::default();
            let encoded_data = arrow_ipc::writer::IpcDataGenerator::default();
            let encoded_schema = encoded_data.schema_to_bytes(&schema, &schema_message);
            let schema_bytes = encoded_schema.ipc_message.into();
            let schema_result = SchemaResult { schema: schema_bytes };
            Ok(Response::new(schema_result))
        } else {
            // Fallback to embedding service
            self.embedding_service.get_schema(Request::new(descriptor)).await
        }
    }

    /// Not implemented for inference service
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        self.embedding_service.do_put(request).await
    }

    /// List available inference actions
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "inference".to_string(),
                description: "Run inference with LoRA adapter".to_string(),
            },
            ActionType {
                r#type: "list_loras".to_string(),
                description: "List available LoRA adapters".to_string(),
            },
            ActionType {
                r#type: "create_lora".to_string(),
                description: "Create new LoRA adapter".to_string(),
            },
        ];

        let stream = stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    /// Execute inference actions
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        
        match action.r#type.as_str() {
            "list_loras" => {
                let layers = self.api_state.lora_registry.list_all().await
                    .map_err(|e| Status::internal(format!("Failed to list LoRAs: {}", e)))?;
                
                let result = arrow_flight::Result {
                    body: serde_json::to_vec(&layers)
                        .map_err(|e| Status::internal(format!("Serialization error: {}", e)))?
                        .into(),
                };
                
                let stream = stream::iter(vec![Ok(result)]);
                Ok(Response::new(Box::pin(stream)))
            }
            
            "get_training_stats" => {
                let stats = self.api_state.training_service.get_stats().await;
                
                let result = arrow_flight::Result {
                    body: serde_json::to_vec(&stats)
                        .map_err(|e| Status::internal(format!("Serialization error: {}", e)))?
                        .into(),
                };
                
                let stream = stream::iter(vec![Ok(result)]);
                Ok(Response::new(Box::pin(stream)))
            }
            
            _ => {
                // Fallback to embedding service
                self.embedding_service.do_action(Request::new(action)).await
            }
        }
    }

    /// Not implemented for inference service
    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        self.embedding_service.do_exchange(request).await
    }

    /// Get flight info for inference operations
    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        
        if descriptor.path.get(0).map(|p| p.as_str()) == Some("inference") {
            let mut flight_info = FlightInfo::new()
                .with_descriptor(descriptor)
                .with_total_records(-1)
                .with_total_bytes(-1);
                
            flight_info = flight_info.try_with_schema(&self.inference_schema())
                .map_err(|e| Status::internal(format!("Failed to set schema: {}", e)))?;

            Ok(Response::new(flight_info))
        } else {
            self.embedding_service.get_flight_info(Request::new(descriptor)).await
        }
    }

    /// Poll flight info (not implemented)
    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        self.embedding_service.poll_flight_info(request).await
    }
}

/// Create FlightSQL server for inference service
pub fn create_inference_flight_server(
    api_state: ApiState,
) -> FlightServiceServer<InferenceFlightService> {
    let service = InferenceFlightService::new(api_state);
    FlightServiceServer::new(service)
}