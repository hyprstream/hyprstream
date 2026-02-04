@0xd4e5f6a7b8c9d0e1;

# Cap'n Proto schema for model service
#
# ModelService manages the lifecycle of InferenceService instances.
# It handles model loading, unloading, and routes inference requests
# to the appropriate InferenceService based on model reference.
#
# Endpoint: inproc://hyprstream/model

struct ModelRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Model lifecycle
    load @1 :LoadModelRequest;
    unload @2 :UnloadModelRequest;
    list @3 :Void;
    status @4 :ModelStatusRequest;

    # Inference routing (convenience methods that route to InferenceService)
    # The request field contains serialized InferenceRequest bytes
    infer @5 :InferRequest;
    inferStream @6 :InferRequest;

    # Stream authorization handshake (routes to InferenceService)
    # Client must call this after inferStream to authorize SUB subscription
    startStream @9 :StartStreamRequest;

    # Health/Lifecycle
    healthCheck @7 :Void;

    # Template application (routes to InferenceService's ApplyChatTemplate)
    applyChatTemplate @8 :ApplyChatTemplateRequest;
  }
}

struct ModelResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload (union of response types)
  union {
    ok @1 :Void;
    error @2 :ErrorInfo;
    loaded @3 :LoadedModelResponse;
    list @4 :ModelListResponse;
    status @5 :ModelStatusResponse;
    # Inference responses contain serialized InferenceResponse bytes
    inferResult @6 :Data;
    streamStarted @7 :StreamInfo;
    health @8 :ModelHealthStatus;
    # Templated prompt string
    templateResult @9 :Text;
    # Stream authorization confirmation
    streamAuthorized @10 :StreamAuthInfo;
  }
}

# Error information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
}

# =============================================================================
# Stream Setup (aligns with streaming.capnp::StreamInfo)
# =============================================================================

# Stream info for streaming responses (includes server pubkey for E2E auth)
# Note: Matches streaming.capnp::StreamInfo for consistency
struct StreamInfo {
  streamId @0 :Text;
  endpoint @1 :Text;
  serverPubkey @2 :Data;  # Server's ephemeral Ristretto255 public key (32 bytes) for DH
}

# KV cache quantization type
enum KVQuantType {
  none @0;      # No quantization (full precision)
  int8 @1;      # 8-bit integer quantization
  nf4 @2;       # 4-bit NormalFloat quantization
  fp4 @3;       # 4-bit FloatingPoint quantization
}

# Load model request with optional runtime configuration
struct LoadModelRequest {
  modelRef @0 :Text;       # e.g., "qwen3-small:main"
  maxContext @1 :UInt32;   # Maximum context length (0 = use default)
  kvQuant @2 :KVQuantType; # KV cache quantization type
}

# Unload model request
struct UnloadModelRequest {
  modelRef @0 :Text;
}

# Model status request
struct ModelStatusRequest {
  modelRef @0 :Text;
}

# Inference request (routes to InferenceService)
# The request field contains serialized GenerationRequest bytes
struct InferRequest {
  modelRef @0 :Text;
  request @1 :Data;  # Serialized GenerationRequest (Cap'n Proto bytes)
}

# Start stream request (authorizes SUB subscription)
# Client must call this after inferStream to authorize subscription
# Note: Aligns with streaming.capnp::StreamStartRequest
struct StartStreamRequest {
  modelRef @0 :Text;      # Model that started the stream
  streamId @1 :Text;      # Stream ID from inferStream response (e.g., "stream-uuid")
  clientPubkey @2 :Data;  # Client's ephemeral Ristretto255 public key (32 bytes) for DH
}

# Stream authorization response
# Note: Aligns with streaming.capnp::StreamAuthResponse
struct StreamAuthInfo {
  streamId @0 :Text;
  serverPubkey @1 :Data;  # Server's ephemeral Ristretto255 public key (if not in StreamInfo)
}

# Response when model is loaded
struct LoadedModelResponse {
  modelRef @0 :Text;
  endpoint @1 :Text;  # inproc://hyprstream/inference/{model_ref}
}

# List of loaded models
struct ModelListResponse {
  models @0 :List(LoadedModelInfo);
}

# Information about a loaded model
struct LoadedModelInfo {
  modelRef @0 :Text;
  endpoint @1 :Text;
  loadedAt @2 :Int64;      # Unix timestamp (millis)
  lastUsed @3 :Int64;      # Unix timestamp (millis)
  memoryBytes @4 :UInt64;  # GPU/CPU memory usage
  sessionCount @5 :UInt32; # Active session count
}

# Model status response
struct ModelStatusResponse {
  loaded @0 :Bool;
  memoryBytes @1 :UInt64;
  sessionCount @2 :UInt32;
  endpoint @3 :Text;       # Only set if loaded
}

# Model service health status
struct ModelHealthStatus {
  status @0 :Text;
  loadedModelCount @1 :UInt32;
  maxModels @2 :UInt32;
  totalMemoryBytes @3 :UInt64;
}

# Chat message for template application
struct ChatMessage {
  role @0 :Text;     # "system", "user", "assistant"
  content @1 :Text;  # Message content
}

# Apply chat template request
struct ApplyChatTemplateRequest {
  modelRef @0 :Text;
  messages @1 :List(ChatMessage);
  addGenerationPrompt @2 :Bool;  # Whether to add assistant prompt at end
}

# =============================================================================
# Callback Protocol (InferenceService → ModelService)
# =============================================================================
#
# InferenceService spawns, connects DEALER to ModelService's ROUTER callback
# socket, and sends Register. ModelService then uses the same connection for
# commands (LoadModel, Infer, Shutdown). This eliminates race conditions.

# Sent by InferenceService when it connects back to ModelService
struct Register {
  id @0 :Text;              # Instance ID (e.g., "inference-a1b2c3d4")
  streamEndpoint @1 :Text;  # XPUB endpoint for token streaming
}

# Response to Register (optional, for acknowledgment)
struct RegisterResponse {
  success @0 :Bool;
  error @1 :Text;
}

# Command wrapper for ModelService → InferenceService
struct InferenceCommand {
  union {
    loadModel @0 :LoadModelCommand;
    shutdown @1 :Void;
    # Infer uses existing InferenceRequest via request field
    infer @2 :Data;  # Serialized InferenceRequest
  }
}

# Load model command sent over callback connection
struct LoadModelCommand {
  modelRef @0 :Text;    # e.g., "qwen3-small:main"
  modelPath @1 :Text;   # Resolved path to model directory
}

# Response to LoadModelCommand
struct LoadModelCommandResponse {
  success @0 :Bool;
  error @1 :Text;
}
