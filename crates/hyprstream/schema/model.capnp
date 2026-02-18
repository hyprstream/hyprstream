@0xd4e5f6a7b8c9d0e1;

using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".mcpScope;
using import "/streaming.capnp".StreamInfo;

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
# Scoped ops are nested under ttt/peft/infer with matching Result variants.

struct ModelRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    load @1 :LoadModelRequest $mcpDescription("Load a model into memory for inference") $mcpScope(write);
    unload @2 :UnloadModelRequest $mcpDescription("Unload a model from memory to free resources") $mcpScope(write);
    list @3 :Void $mcpDescription("List all models currently loaded in memory");
    healthCheck @4 :Void $mcpDescription("Check model service health and status");

    # Scoped interfaces (require modelRef)
    ttt @5 :TttRequest;     # Test-time training operations
    peft @6 :PeftRequest;   # PEFT adapter management
    infer @7 :InferRequest;  # Inference operations
  }
}

# =============================================================================
# TTT (Test-Time Training) scoped client
# =============================================================================

# TTT-scoped request: test-time training operations on a loaded model.
# Generator detects the non-union field (modelRef) + inner union pattern
# and produces a TttClient with modelRef curried in.
struct TttRequest {
  modelRef @0 :Text;
  union {
    create @1 :CreateLoraRequest
      $mcpDescription("Create a new LoRA adapter on a loaded model");
    train @2 :TrainStepRequest
      $mcpDescription("Run TTT gradient steps on input text WITHOUT generating a response. Pure training — use for pre-training on domain text before asking questions. Returns loss metrics and recommendation. If autoCommit is false, call commitAdaptation or rollbackAdaptation.");
    trainStream @3 :TrainStepRequest
      $mcpDescription("Stream TTT training on input text. Returns progress and results via streaming. Use for long-running training that would timeout via trainStep.");
    commit @4 :Void
      $mcpDescription("Commit a pending TTT adaptation after reviewing metrics from infer/inferStream. Must be called within 30 seconds of the inference response. The adaptation becomes permanent for this tenant's delta.");
    rollback @5 :Void
      $mcpDescription("Rollback a pending TTT adaptation, reverting delta to pre-inference state. Call within 30 seconds if recommendation was false or quality was poor.");
    reset @6 :Void
      $mcpDescription("Clear a tenant's accumulated delta, resetting to base model weights.");
    status @7 :Void
      $mcpDescription("Get status of a tenant's accumulated TTT delta: step count, loss improvement, drift metrics. Use to decide if adaptations should be saved permanently via saveAdaptation.");
    save @8 :SaveAdaptationRequest
      $mcpDescription("Save accumulated TTT adaptations as a permanent LoRA adapter file using DO-Merge strategy. Call getDeltaStatus first to verify quality. The adapter is committed to the model's git repository.");
    snapshot @9 :Void
      $mcpDescription("Snapshot current delta to content-addressed storage without merging into an adapter.");
    export @10 :TttExportRequest
      $mcpDescription("Export TTT delta as a PEFT-compatible adapter directory (adapter_config.json + adapter_model.safetensors). The exported adapter can be loaded via peft.load.");
  }
}

# =============================================================================
# PEFT (Parameter-Efficient Fine-Tuning) scoped client
# =============================================================================

# PEFT-scoped request: adapter management operations on a loaded model.
# Works with PEFT-compatible adapter directories (adapter_config.json +
# adapter_model.safetensors).
struct PeftRequest {
  modelRef @0 :Text;
  union {
    load @1 :Text
      $mcpDescription("Load a PEFT adapter from a directory (relative path within model worktree, e.g. 'adapters/my-adapter')");
    unload @2 :Void
      $mcpDescription("Unload the current LoRA adapter from memory");
    has @3 :Void
      $mcpDescription("Check if a LoRA adapter is currently loaded");
    check @4 :Text
      $mcpDescription("Validate a path as a PEFT adapter directory and return adapter info (rank, alpha, target modules, base model)");
    merge @5 :PeftMergeRequest
      $mcpDescription("Merge a PEFT adapter into the base model weights (experimental)");
  }
}

# =============================================================================
# Infer (Inference) scoped client
# =============================================================================

# Inference-scoped request: generation and model query operations.
struct InferRequest {
  modelRef @0 :Text;
  union {
    generate @1 :GenerateRequest
      $mcpDescription("Run inference with automatic domain adaptation. When TTT is enabled, the model adapts to your prompt before responding. If autoCommit is false (default), the adaptation is PENDING — check onlineTrainingMetrics.recommendation in the response, then call commitAdaptation (if true) or rollbackAdaptation (if false). Pending adaptations auto-rollback after 30 seconds.");
    generateStream @2 :GenerateRequest
      $mcpDescription("Stream inference with automatic domain adaptation. The final SSE chunk includes usage.online_training metrics. If autoCommit is false, the adaptation is PENDING — call commitAdaptation or rollbackAdaptation based on the recommendation field. Pending adaptations auto-rollback after 30 seconds.");
    applyChatTemplate @3 :ApplyChatTemplateRequest
      $mcpDescription("Apply chat template to messages for a loaded model");
    status @4 :Void
      $mcpDescription("Get detailed status information about a model including online training configuration");
  }
}

# =============================================================================
# Response
# =============================================================================

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
    tttResult @6 :TttResponse;
    peftResult @7 :PeftResponse;
    inferResult @8 :InferResponse;
  }
}

# TTT scoped response
struct TttResponse {
  union {
    error @0 :ErrorInfo;
    create @1 :Void;
    train @2 :TrainStepResponse;
    trainStream @3 :StreamInfo;
    commit @4 :Void;
    rollback @5 :Void;
    reset @6 :Void;
    status @7 :GetDeltaStatusResponse;
    save @8 :SaveAdaptationResponse;
    snapshot @9 :SnapshotDeltaResponse;
    export @10 :TttExportResponse;
  }
}

# PEFT scoped response
struct PeftResponse {
  union {
    error @0 :ErrorInfo;
    load @1 :Void;
    unload @2 :Void;
    has @3 :Bool;
    check @4 :PeftAdapterInfo;
    merge @5 :Void;
  }
}

# Infer scoped response
struct InferResponse {
  union {
    error @0 :ErrorInfo;
    generate @1 :InferResult;
    generateStream @2 :StreamInfo;
    applyChatTemplate @3 :Text;
    status @4 :ModelStatusResponse;
  }
}

# =============================================================================
# Inference result — flattened from inference.capnp::GenerationResult
# for transparent MCP/JSON bridging.
# =============================================================================

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

# Generation request (routes to InferenceService)
# Fields match inference.capnp::GenerationRequest for transparent MCP/JSON bridging.
struct GenerateRequest {
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

  # Per-request TTT control (all optional — omit for server defaults)
  tttEnabled @11 :Bool $paramDescription("Override: enable/disable TTT for this request");
  tttGradientSteps @12 :UInt32 $paramDescription("Override: number of gradient steps (0 = skip)");
  tttLearningRate @13 :Float32 $paramDescription("Override: learning rate");
  autoCommit @14 :Bool $paramDescription("If true, server auto-commits based on its recommendation. If false (default), adaptation is pending until client commits.");
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

# Online Training (Test-Time Training) configuration
#
# Shows current online training settings for a loaded model.
# Online training adapts the model to input style/domain before generation.
struct OnlineTrainingConfig {
  enabled @0 :Bool;            # Whether online training is enabled
  learningRate @1 :Float64;    # Learning rate for adaptation (e.g., 0.0003)
  gradientSteps @2 :UInt32;    # Number of gradient steps per input (e.g., 3)
  maxGradNorm @3 :Float64;     # Maximum gradient norm for clipping (e.g., 1.0)
  minInputLength @4 :UInt32;   # Minimum tokens required to trigger (e.g., 32)
  maxTttContext @5 :UInt32;    # Maximum tokens to process (truncates if longer)
}

# Model status response
struct ModelStatusResponse {
  loaded @0 :Bool;
  memoryBytes @1 :UInt64;
  sessionCount @2 :UInt32;
  endpoint @3 :Text;       # Only set if loaded

  # Online training configuration (if model loaded)
  onlineTrainingConfig @4 :OnlineTrainingConfig;
}

# Model service health status
struct ModelHealthStatus {
  status @0 :Text;
  loadedModelCount @1 :UInt32;
  maxModels @2 :UInt32;
  totalMemoryBytes @3 :UInt64;
}

# Tool call data for threading through RPC
struct ToolCallData {
  id @0 :Text;
  callType @1 :Text;        # "function"
  functionName @2 :Text;
  arguments @3 :Text;        # JSON string (opaque, deserialized at consumption point)
}

# Chat message for template application
struct ChatMessage {
  role @0 :Text;     # "system", "user", "assistant", "tool"
  content @1 :Text;  # Message content (empty string = None)
  toolCalls @2 :List(ToolCallData);
  toolCallId @3 :Text;  # For "tool" role messages (empty string = None)
}

# Apply chat template request
struct ApplyChatTemplateRequest {
  messages @0 :List(ChatMessage);
  addGenerationPrompt @1 :Bool;  # Whether to add assistant prompt at end
  toolsJson @2 :Text;  # JSON-serialized tools array (empty string = no tools)
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
# Training Loop Control (TTT commit/rollback/train)
# =============================================================================

struct TrainStepRequest {
  input @0 :Text $paramDescription("Text to train on (NTP loss)");
  gradientSteps @1 :UInt32 $paramDescription("Number of gradient steps (default: 3)");
  learningRate @2 :Float32 $paramDescription("Learning rate override (0 = use default)");
  autoCommit @3 :Bool $paramDescription("If true, auto-commit if quality gate passes");
}

struct TrainStepResponse {
  avgLoss @0 :Float32;
  lossImprovement @1 :Float32;
  stepsPerformed @2 :UInt32;
  adaptationTimeMs @3 :UInt64;
  initialPerplexity @4 :Float32;
  finalPerplexity @5 :Float32;
  recommendation @6 :Bool;    # Server's commit/rollback recommendation
  committed @7 :Bool;         # Whether it was auto-committed
  gradientClipped @8 :Bool;
}

# =============================================================================
# Persistence Operations (delta status/save/snapshot)
# =============================================================================

struct GetDeltaStatusResponse {
  exists @0 :Bool;
  accumulatedSteps @1 :UInt64;
  maxAccumulatedSteps @2 :UInt64;
  requestCount @3 :UInt64;
  avgLossImprovement @4 :Float32;
  memoryBytes @5 :UInt64;
  lastSnapshotHash @6 :Text;
  deltaNormRatios @7 :List(ModuleNormRatio);
  hasPending @8 :Bool;    # Whether there's an uncommitted adaptation
}

struct ModuleNormRatio {
  moduleName @0 :Text;
  ratio @1 :Float32;
}

struct SaveAdaptationRequest {
  name @0 :Text $paramDescription("Adapter name for the saved file");
  mergeStrategy @1 :Text $paramDescription("Merge strategy: 'replace', 'additive', 'do_merge' (default)");
  mergeWeight @2 :Float32 $paramDescription("Merge weight 0.0-1.0 (default: 0.3)");
  commitMessage @3 :Text $paramDescription("Non-empty triggers git commit");
}

struct SaveAdaptationResponse {
  adapterName @0 :Text;
  adapterPath @1 :Text;
  contentHash @2 :Text;
  mergeStrategy @3 :Text;
}

struct SnapshotDeltaResponse {
  contentHash @0 :Text;
  sizeBytes @1 :UInt64;
  accumulatedSteps @2 :UInt64;
  requestCount @3 :UInt64;
}

# =============================================================================
# TTT Export (delta → PEFT adapter)
# =============================================================================

struct TttExportRequest {
  name @0 :Text $paramDescription("PEFT adapter directory name");
  commitMessage @1 :Text $paramDescription("Git commit message (optional)");
}

struct TttExportResponse {
  adapterPath @0 :Text;
  contentHash @1 :Text;
}

# =============================================================================
# PEFT Adapter Info / Merge
# =============================================================================

struct PeftAdapterInfo {
  name @0 :Text;              # Directory name
  path @1 :Text;              # Relative path within worktree
  rank @2 :UInt32;
  loraAlpha @3 :Float32;
  targetModules @4 :List(Text);
  baseModel @5 :Text;         # base_model_name_or_path from config
}

struct PeftMergeRequest {
  adapterName @0 :Text $paramDescription("Adapter directory name to merge");
  weight @1 :Float32 $paramDescription("Merge weight 0.0-1.0 (default: 1.0 = full merge)");
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
