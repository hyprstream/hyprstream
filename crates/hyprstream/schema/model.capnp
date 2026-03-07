@0xd4e5f6a7b8c9d0e1;

using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".optional;
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
# Scoped ops are nested under ttt/adapter/infer with matching Result variants.

struct ModelRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    load @1 :LoadModelRequest $mcpDescription("Load a model into memory for inference") $mcpScope(write);
    unload @2 :UnloadModelRequest $mcpDescription("Unload a model from memory to free resources") $mcpScope(write);
    status @3 :StatusRequest $mcpDescription("Get status of all loaded/loading models (empty modelRef) or a specific model") $mcpScope(query);
    healthCheck @4 :Void $mcpDescription("Check model service health and status") $mcpScope(query);

    # Scoped interfaces (require modelRef)
    ttt @5 :TttRequest;         # Test-time training operations
    adapter @6 :AdapterRequest; # Adapter management (load/unload/inspect/merge)
    infer @7 :InferRequest;     # Inference operations
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
    init @1 :InitLoraRequest
      $mcpDescription("Initialize the training infrastructure (LoRA parameters, optimizer, delta pool) on a loaded model. Required before ttt.train or TTT-enabled inference. Configure rank, alpha, target modules, and learning rate.");
    train @2 :TrainStepRequest
      $mcpDescription("Run TTT gradient steps on input text WITHOUT generating a response. Pure training — use for pre-training on domain text before asking questions. Returns loss metrics and recommendation. If autoCommit is false, call ttt.commit or ttt.rollback.");
    trainStream @3 :TrainStepRequest
      $mcpDescription("Stream TTT training on input text. Returns progress and results via streaming. Use for long-running training that would timeout via ttt.train.");
    commit @4 :Void
      $mcpDescription("Commit a pending TTT adaptation to the tenant delta accumulator. Call after reviewing metrics from infer.generateStream. Must be called within 30 seconds of the inference response.");
    rollback @5 :Void
      $mcpDescription("Rollback a pending TTT adaptation, restoring the tenant delta accumulator to its pre-inference state. Call within 30 seconds if recommendation was false or quality was poor.");
    reset @6 :Void
      $mcpDescription("Clear the tenant delta accumulator, resetting all accumulated training to zero.");
    status @7 :Void
      $mcpDescription("Get tenant delta accumulator metrics: step count, loss improvement, drift. Use to decide if adaptations should be persisted via ttt.save or ttt.export.");
    save @8 :SaveAdaptationRequest
      $mcpDescription("Merge the tenant delta accumulator into an on-disk adapter file using a configurable merge strategy (replace/additive/do_merge). For incremental refinement of existing adapters. Call ttt.status first to verify quality. The result is committed to the model's git repository.");
    snapshot @9 :Void
      $mcpDescription("Snapshot the tenant delta accumulator to content-addressed storage without merging into an adapter file.");
    export @10 :TttExportRequest
      $mcpDescription("Export the tenant delta accumulator as a standalone PEFT-compatible adapter directory (adapter_config.json + adapter_model.safetensors). For interop with HuggingFace and external tools. The exported adapter can be reloaded via adapter.load.");
    writeTttConfig @11 :WriteTttConfigRequest
      $mcpDescription("Write hyprstream_training configuration to the model worktree's config.json and optionally reload. Required before ttt.train or TTT-enabled inference. Sets training mode to test_time_training and configures LoRA rank/alpha, target modules, and learning rate.");
  }
}

# =============================================================================
# Adapter scoped client (base_delta register)
# =============================================================================

# Adapter-scoped request: manage the base_delta register (the loaded adapter
# applied to all inference) and inspect on-disk adapter files.
#
# The base_delta is a LoRA weight delta loaded from a PEFT-compatible adapter
# directory. It is applied to all inference requests. If a per-tenant TTT
# delta also exists, the two are composed at inference time.
struct AdapterRequest {
  modelRef @0 :Text;
  union {
    load @1 :Text
      $mcpDescription("Load a PEFT adapter from disk into the base_delta register. Applied to all inference until unloaded. Path is relative within the model worktree (e.g. 'adapters/my-adapter').");
    unload @2 :Void
      $mcpDescription("Clear the base_delta register, removing the loaded adapter from GPU/CPU memory.");
    status @3 :Void
      $mcpDescription("Check if a LoRA adapter is currently loaded in the base_delta register.");
    inspect @4 :Text
      $mcpDescription("Validate an on-disk PEFT adapter directory and return its metadata (rank, alpha, target modules, base model). Does not load anything into memory.");
    merge @5 :AdapterMergeRequest
      $mcpDescription("Read a PEFT adapter from disk and merge it into the currently loaded base_delta register using a configurable merge strategy (replace/additive/do_merge). Requires an adapter already loaded via adapter.load.");
  }
}

# =============================================================================
# Infer (Inference) scoped client
# =============================================================================

# Inference-scoped request: generation and model query operations.
struct InferRequest {
  modelRef @0 :Text;
  union {
    generateStream @1 :GenerateRequest
      $mcpDescription("Run inference with automatic domain adaptation. When TTT is enabled, the model adapts to your prompt before responding. If autoCommit is false (default), the adaptation is PENDING — check onlineTrainingMetrics.recommendation in the response, then call ttt.commit (if true) or ttt.rollback (if false). Pending adaptations auto-rollback after 30 seconds.");
    applyChatTemplate @2 :ApplyChatTemplateRequest
      $mcpDescription("Apply chat template to messages for a loaded model");
    status @3 :Void
      $mcpDescription("Get detailed status information about a model including online training configuration");
    embed @4 :EmbedRequest
      $mcpDescription("Compute embeddings for one or more images. Returns embedding vectors from the model's vision encoder (e.g. SigLIP). Synchronous — returns all embeddings in a single response.");
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
    statusResult @4 :List(ModelStatusEntry);
    healthCheckResult @5 :ModelHealthStatus;
    tttResult @6 :TttResponse;
    adapterResult @7 :AdapterResponse;
    inferResult @8 :InferResponse;
  }
}

# TTT scoped response
struct TttResponse {
  union {
    error @0 :ErrorInfo;
    init @1 :Void;
    train @2 :TrainStepResponse;
    trainStream @3 :StreamInfo;
    commit @4 :Void;
    rollback @5 :Void;
    reset @6 :Void;
    status @7 :GetDeltaStatusResponse;
    save @8 :SaveAdaptationResponse;
    snapshot @9 :SnapshotDeltaResponse;
    export @10 :TttExportResponse;
    writeTttConfig @11 :Void;
  }
}

# Adapter scoped response
struct AdapterResponse {
  union {
    error @0 :ErrorInfo;
    load @1 :Void;
    unload @2 :Void;
    status @3 :Bool;
    inspect @4 :AdapterInfo;
    merge @5 :Void;
  }
}

# Infer scoped response
struct InferResponse {
  union {
    error @0 :ErrorInfo;
    generateStream @1 :StreamInfo;
    applyChatTemplate @2 :Text;
    status @3 :ModelStatusResponse;
    embed @4 :EmbedResponse;
  }
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
  maxContext @1 :UInt32 $optional $paramDescription("Maximum context length (0 = use default)");
  kvQuant @2 :KVQuantType $optional $paramDescription("KV cache quantization type");
}

# Unload model request
struct UnloadModelRequest {
  modelRef @0 :Text;
}

# Generation request (routes to InferenceService)
# Fields match inference.capnp::GenerationRequest for transparent MCP/JSON bridging.
struct GenerateRequest {
  prompt @0 :Text;
  maxTokens @1 :UInt32 $optional;
  temperature @2 :Float32 $optional;
  topP @3 :Float32 $optional;
  topK @4 :UInt32 $optional;
  repeatPenalty @5 :Float32 $optional;
  repeatLastN @6 :UInt32 $optional;
  stopTokens @7 :List(Text) $optional;
  seed @8 :UInt32 $optional;
  images @9 :List(Data) $optional;
  timeoutMs @10 :UInt64 $optional;

  # Per-request TTT control (all optional — omit for server defaults)
  tttEnabled @11 :Bool $paramDescription("Override: enable/disable TTT for this request");
  tttGradientSteps @12 :UInt32 $optional $paramDescription("Override: number of gradient steps (0 = skip)");
  tttLearningRate @13 :Float32 $optional $paramDescription("Override: learning rate");
  autoCommit @14 :Bool $paramDescription("If true, server auto-commits based on its recommendation. If false (default), adaptation is pending until client commits.");
}

# Embedding request for vision models (e.g. SigLIP)
struct EmbedRequest {
  # Raw image bytes (PNG/JPEG/RGB) — one entry per image
  images @0 :List(Data);
}

# Embedding response
struct EmbedResponse {
  # One embedding vector per input image
  embeddings @0 :List(List(Float32));
  # Embedding dimensionality (e.g. 384, 768)
  dimensions @1 :UInt32;
}

# Response when model is loaded
struct LoadedModelResponse {
  modelRef @0 :Text;
  endpoint @1 :Text;  # inproc://hyprstream/inference/{model_ref}
}

# Status request: empty modelRef = all known models; non-empty = specific model (0 or 1 result)
struct StatusRequest {
  modelRef @0 :Text;
}

# Status entry for a single model (loaded or loading)
# Absence from the list means unloaded.
struct ModelStatusEntry {
  modelRef   @0 :Text;
  status     @1 :Text;    # "loaded" | "loading"
  endpoint   @2 :Text;    # empty if loading
  loadedAt   @3 :Int64;   # ms elapsed since load (0 if loading)
  lastUsed   @4 :Int64;   # ms elapsed since last use (0 if loading)
  onlineTrainingConfig @5 :OnlineTrainingConfig;
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
  toolsJson @2 :Text $optional;  # JSON-serialized tools array (empty string = no tools)
}

# LoRA training initialization configuration
struct InitLoraRequest {
  rank @0 :UInt32 $paramDescription("LoRA rank (e.g., 8, 16, 32)");
  alpha @1 :Float32 $optional $paramDescription("LoRA alpha scaling factor");
  dropout @2 :Float32 $optional $paramDescription("Dropout rate during training");
  targetModules @3 :List(Text) $paramDescription("Model layers to apply LoRA (e.g., ['q_proj','v_proj'])");
  learningRate @4 :Float32 $optional $paramDescription("Learning rate for training (default: 1e-4)");
}

# =============================================================================
# Training Loop Control (TTT commit/rollback/train)
# =============================================================================

struct TrainStepRequest {
  input @0 :Text $paramDescription("Text to train on (NTP loss)");
  gradientSteps @1 :UInt32 $optional $paramDescription("Number of gradient steps (default: 3)");
  learningRate @2 :Float32 $optional $paramDescription("Learning rate override (0 = use default)");
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
  mergeStrategy @1 :Text $optional $paramDescription("Merge strategy: 'replace', 'additive', 'do_merge' (default)");
  mergeWeight @2 :Float32 $optional $paramDescription("Merge weight 0.0-1.0 (default: 0.3)");
  commitMessage @3 :Text $optional $paramDescription("Non-empty triggers git commit");
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
  commitMessage @1 :Text $optional $paramDescription("Git commit message (optional)");
}

struct TttExportResponse {
  adapterPath @0 :Text;
  contentHash @1 :Text;
}

# Write TTT configuration to model's config.json
struct WriteTttConfigRequest {
  learningRate @0 :Float64;
  gradientSteps @1 :UInt32;
  maxGradNorm @2 :Float64;
  minInputLength @3 :UInt32;
  maxTttContext @4 :UInt32;
  loraRank @5 :UInt32;
  loraAlpha @6 :Float32;
  targetModules @7 :List(Text);
  autoReload @8 :Bool;
}

# =============================================================================
# Adapter Info / Merge
# =============================================================================

struct AdapterInfo {
  name @0 :Text;              # Directory name
  path @1 :Text;              # Relative path within worktree
  rank @2 :UInt32;
  loraAlpha @3 :Float32;
  targetModules @4 :List(Text);
  baseModel @5 :Text;         # base_model_name_or_path from config
}

struct AdapterMergeRequest {
  adapterName @0 :Text $paramDescription("Adapter directory name to merge");
  weight @1 :Float32 $optional $paramDescription("Merge weight 0.0-1.0 (default: 1.0 = full merge)");
  strategy @2 :Text $optional $paramDescription("Merge strategy: 'replace', 'additive', or 'do_merge' (default: 'do_merge')");
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
