@0xd4e5f6a7b8c9d0e1;

using import "annotations.capnp".mcpDescription;
using import "annotations.capnp".paramDescription;

# Cap'n Proto schema for model service
#
# ModelService manages the lifecycle of InferenceService instances.
# It handles model loading, unloading, and routes inference requests
# to the appropriate InferenceService based on model reference.
#
# Endpoint: inproc://hyprstream/model
#
# Convention: Request variants use camelCase names. Response variants
# use the same name suffixed with "Result" to avoid Cap'n Proto naming
# collisions. The code generator strips "Result" to pair them.
# Session-scoped ops are nested under `session`/`sessionResult`.

struct ModelRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    load @1 :LoadModelRequest $mcpDescription("Load a model into memory for inference");
    unload @2 :UnloadModelRequest $mcpDescription("Unload a model from memory to free resources");
    list @3 :Void $mcpDescription("List all models currently loaded in memory");
    healthCheck @4 :Void $mcpDescription("Check model service health and status");

    # Session-scoped operations (requires modelRef)
    session @5 :ModelSessionRequest;
  }
}

# Session-scoped request: operations that target a specific loaded model.
# Generator detects the non-union field (modelRef) + inner union pattern
# and produces a ModelSessionClient with modelRef curried in.
struct ModelSessionRequest {
  modelRef @0 :Text;
  union {
    status @1 :Void $mcpDescription("Get detailed status information about a model");
    infer @2 :InferRequest $mcpDescription("Run inference on a loaded model (non-streaming)");
    inferStream @3 :InferRequest $mcpDescription("Run inference on a loaded model (streaming)");
    startStream @4 :StartStreamRequest $mcpDescription("Authorize a streaming subscription (client must call after inferStream)");
    applyChatTemplate @5 :ApplyChatTemplateRequest $mcpDescription("Apply chat template to messages for a loaded model");

    # LoRA adapter operations
    createLora @6 :CreateLoraRequest $mcpDescription("Create a new LoRA adapter on a loaded model");
    loadLora @7 :Text $mcpDescription("Load a LoRA adapter from a safetensors file");
    saveLora @8 :Text $mcpDescription("Save the current LoRA adapter to a safetensors file");
    unloadLora @9 :Void $mcpDescription("Unload the current LoRA adapter from memory");
    hasLora @10 :Void $mcpDescription("Check if a LoRA adapter is currently loaded");
  }
}

struct ModelResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload — variants suffixed with "Result" to pair with request
  union {
    error @1 :ErrorInfo;
    loadResult @2 :LoadedModelResponse;
    unloadResult @3 :Void;
    listResult @4 :ModelListResponse;
    healthCheckResult @5 :ModelHealthStatus;
    sessionResult @6 :ModelSessionResponse;
  }
}

# Session-scoped response: inner union variants match request names exactly.
struct ModelSessionResponse {
  union {
    error @0 :ErrorInfo;
    status @1 :ModelStatusResponse;
    infer @2 :InferResult;
    inferStream @3 :StreamInfo;
    startStream @4 :StreamAuthResponse;
    applyChatTemplate @5 :Text;
    createLora @6 :Void;
    loadLora @7 :Void;
    saveLora @8 :Void;
    unloadLora @9 :Void;
    hasLora @10 :Bool;
  }
}

# Inference result — flattened from inference.capnp::GenerationResult
# for transparent MCP/JSON bridging.
struct InferResult {
  text @0 :Text;
  tokensGenerated @1 :UInt32;
  finishReason @2 :Text;         # "max_tokens", "stop_token", "end_of_sequence", "error", "stop"
  generationTimeMs @3 :UInt64;
  tokensPerSecond @4 :Float32;
  prefillTokens @5 :UInt32;
  prefillTimeMs @6 :UInt64;
  prefillTokensPerSec @7 :Float32;
  inferenceTokens @8 :UInt32;
  inferenceTimeMs @9 :UInt64;
  inferenceTokensPerSec @10 :Float32;
}

# Error information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
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
  modelRef @0 :Text $paramDescription("Model reference in format name:branch (e.g., 'qwen3-small:main')");
  maxContext @1 :UInt32 $paramDescription("Maximum context length (0 = use default)");
  kvQuant @2 :KVQuantType $paramDescription("KV cache quantization type");
}

# Unload model request
struct UnloadModelRequest {
  modelRef @0 :Text;
}

# Inference request (routes to InferenceService)
# Fields match inference.capnp::GenerationRequest for transparent MCP/JSON bridging.
# Note: modelRef removed — curried into ModelSessionClient
struct InferRequest {
  prompt @0 :Text;
  maxTokens @1 :UInt32;
  temperature @2 :Float32;
  topP @3 :Float32;
  topK @4 :UInt32;
  repeatPenalty @5 :Float32;
  repeatLastN @6 :UInt32;
  stopTokens @7 :List(Text);
  seed @8 :UInt32;
  images @9 :List(Data);
  timeoutMs @10 :UInt64;
}

# Start stream request (authorizes SUB subscription)
# Client must call this after inferStream to authorize subscription
# Note: modelRef removed — curried into ModelSessionClient
struct StartStreamRequest {
  streamId @0 :Text;      # Stream ID from inferStream response (e.g., "stream-uuid")
  clientPubkey @1 :Data;  # Client's ephemeral Ristretto255 public key (32 bytes) for DH
}

# Stream authorization response
# Note: Aligns with streaming.capnp::StreamAuthResponse
struct StreamAuthResponse {
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
# Note: modelRef removed — curried into ModelSessionClient
struct ApplyChatTemplateRequest {
  messages @0 :List(ChatMessage);
  addGenerationPrompt @1 :Bool;  # Whether to add assistant prompt at end
}

# LoRA adapter configuration for creation
struct CreateLoraRequest {
  rank @0 :UInt32 $paramDescription("LoRA rank (e.g., 8, 16, 32)");
  alpha @1 :Float32 $paramDescription("LoRA alpha scaling factor");
  dropout @2 :Float32 $paramDescription("Dropout rate during training");
  targetModules @3 :List(Text) $paramDescription("Model layers to apply LoRA (e.g., ['q_proj','v_proj'])");
  learningRate @4 :Float32 $paramDescription("Learning rate for training (default: 1e-4)");
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
