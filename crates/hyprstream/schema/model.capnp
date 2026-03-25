@0xd4e5f6a7b8c9d0e1;

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".optional;
using import "/annotations.capnp".serdeRename;
using import "/streaming.capnp".StreamInfo;
using import "/inference.capnp".ChatMessage;
using import "/inference.capnp".ToolCall;
using import "/inference.capnp".ToolCallFunction;
# Shared types — defined once in inference.capnp, imported here
using import "/inference.capnp".AdaptationStrategy;
using import "/inference.capnp".GenerationRequest;
using import "/inference.capnp".TrainStepRequest;
using import "/inference.capnp".TrainStepResult;
using import "/inference.capnp".SaveAdaptationRequest;
using import "/inference.capnp".SaveAdaptationResult;
using import "/inference.capnp".DeltaStatusResult;
using import "/inference.capnp".SnapshotDeltaResult;
using import "/inference.capnp".ExportPeftRequest;
using import "/inference.capnp".ExportPeftResult;
using import "/inference.capnp".ModuleNormRatio;
using import "/inference.capnp".ChatTemplateRequest;
using import "/inference.capnp".EmbedImagesRequest;
using import "/inference.capnp".EmbedImagesResponse;
using import "/inference.capnp".LoraConfig;
using import "/inference.capnp".MergeLoraRequest;
using Opt = import "/optional.capnp";
using Nine = import "/nine.capnp";

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

    # 9P filesystem access (scoped by modelRef)
    fs @8 :ModelFsRequest;
  }
}

# 9P filesystem scope for model service.
# Exposes model status, defaults, ctl, and chat/* as synthetic files.
struct ModelFsRequest {
  modelRef @0 :Text $paramDescription("Model reference (e.g., 'qwen3:main'). Empty for root directory.");
  request @1 :Nine.NpRequest;
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
    init @1 :LoraConfig
      $mcpDescription("Initialize the training infrastructure (LoRA parameters, optimizer, delta pool) on a loaded model. Required before ttt.train or TTT-enabled inference. Configure rank, alpha, target modules, and learning rate.");
    train @2 :TrainStepRequest
      $mcpDescription("Run TTT gradient steps on input text WITHOUT generating a response. Pure training — use for pre-training on domain text before asking questions. Returns loss metrics and recommendation. Use adaptationStrategy=speculative to keep pending, then call ttt.writeback or ttt.evict.");
    trainStream @3 :TrainStepRequest
      $mcpDescription("Stream TTT training on input text. Returns progress and results via streaming. Use for long-running training that would timeout via ttt.train.");
    writeback @4 :Void
      $mcpDescription("Write back a pending TTT adaptation to the tenant delta accumulator. Call after reviewing onlineTrainingMetrics.recommendation from infer.generateStream. Must be called within the pending rollback window (default 60 seconds, configurable via pending_rollback_ms). If the window expires, the adaptation is auto-evicted and this call will return an error.");
    evict @5 :Void
      $mcpDescription("Evict (discard) a pending TTT adaptation, restoring the tenant delta accumulator to its pre-adaptation state. Call within the pending rollback window (default 60 seconds, configurable via pending_rollback_ms) if recommendation was false or output quality was poor. If multiple adaptations were stacked, evict restores to the state before the earliest pending adaptation.");
    zero @6 :Void
      $mcpDescription("Zero the tenant delta accumulator, clearing all accumulated training. Use after ttt.save or ttt.export to free capacity.");
    status @7 :Void
      $mcpDescription("Get tenant delta accumulator metrics: step count, loss improvement, drift. Use to decide if adaptations should be persisted via ttt.save or ttt.export.");
    save @8 :SaveAdaptationRequest
      $mcpDescription("Merge the tenant delta accumulator into an on-disk adapter file using a configurable merge strategy (replace/additive/do_merge). For incremental refinement of existing adapters. Call ttt.status first to verify quality. The result is committed to the model's git repository.");
    snapshot @9 :Void
      $mcpDescription("Snapshot the tenant delta accumulator to content-addressed storage without merging into an adapter file.");
    export @10 :ExportPeftRequest
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
    merge @5 :MergeLoraRequest
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
    generateStream @1 :GenerationRequest
      $mcpDescription("Run inference with automatic domain adaptation. When TTT is enabled, the model adapts to your prompt BEFORE generating — the response is always produced using the adapted weights, even when adaptationStrategy=speculative. Check onlineTrainingMetrics.recommendation in the response, then call ttt.writeback (if true) or ttt.evict (if false) within the pending rollback window (default 60 seconds). If you call generateStream again before resolving a pending adaptation, the new adaptation stacks on top and evict will restore to the state before the first pending adaptation. Pending adaptations auto-evict after the timeout if writeback/evict is not called.");
    applyChatTemplate @2 :ChatTemplateRequest
      $mcpDescription("Apply chat template to messages for a loaded model");
    status @3 :Void
      $mcpDescription("Get detailed status information about a model including online training configuration");
    embed @4 :EmbedImagesRequest
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
    fsResult @9 :Nine.NpResponse;
  }
}

# TTT scoped response
struct TttResponse {
  union {
    error @0 :ErrorInfo;
    init @1 :Void;
    train @2 :TrainStepResult;
    trainStream @3 :StreamInfo;
    writeback @4 :Void;
    evict @5 :Void;
    zero @6 :Void;
    status @7 :DeltaStatusResult;
    save @8 :SaveAdaptationResult;
    snapshot @9 :SnapshotDeltaResult;
    export @10 :ExportPeftResult;
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
    embed @4 :EmbedImagesResponse;
  }
}

# KV cache quantization type
enum KVQuantType {
  none @0;      # No quantization (full precision)
  int8 @1;      # 8-bit integer quantization
  nf4 @2;       # 4-bit NormalFloat quantization
  fp4 @3;       # 4-bit FloatingPoint quantization
}

struct OptionKVQuantType { union { none @0 :Void; some @1 :KVQuantType; } }

# Load model request with optional runtime configuration
struct LoadModelRequest {
  modelRef @0 :Text $paramDescription("Model reference in format name:branch (e.g., 'qwen3-small:main')");
  maxContext @1 :Opt.OptionUint32 $paramDescription("Maximum context length (None = use default)");
  kvQuant @2 :OptionKVQuantType $paramDescription("KV cache quantization type");
}

# Unload model request
struct UnloadModelRequest {
  modelRef @0 :Text;
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

# Generation parameter defaults from the model's generation_config.json.
# All fields are optional — None means the model has no opinion (use server default).
struct GenerationDefaults {
  temperature   @0 :Opt.OptionFloat32;
  topP          @1 :Opt.OptionFloat32;
  topK          @2 :Opt.OptionUint32;
  maxTokens     @3 :Opt.OptionUint32;
  repeatPenalty @4 :Opt.OptionFloat32;
  stopTokens    @5 :List(Text);
  doSample      @6 :Opt.OptionBool;
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
  generationDefaults   @6 :GenerationDefaults;
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

# ChatMessage, ToolCall, ToolCallFunction, ChatTemplateRequest,
# EmbedImagesRequest, EmbedImagesResponse imported from inference.capnp

# LoraConfig and MergeLoraRequest imported from inference.capnp

# =============================================================================
# Training Loop Control, Persistence, TTT Export
# =============================================================================
#
# TrainStepRequest, TrainStepResult, SaveAdaptationRequest, SaveAdaptationResult,
# DeltaStatusResult, SnapshotDeltaResult, ExportPeftRequest, ExportPeftResult,
# ModuleNormRatio, and AdaptationStrategy are all imported from inference.capnp.

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
