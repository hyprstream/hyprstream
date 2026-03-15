@0xa8c9e2f1d3b5a7c0;

using import "/common.capnp".ErrorInfo;
using import "/streaming.capnp".StreamInfo;
using import "/annotations.capnp".optional;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".serdeRename;
using Opt = import "/optional.capnp";

# Cap'n Proto schema for inference service
#
# The inference service uses REQ/REP pattern for request handling.
# Streaming uses PUB/SUB with stream IDs for chunk delivery.
#
# Streaming Architecture:
#   - Wire format types are in hyprstream-rpc/schema/streaming.capnp
#   - Streaming payloads use generic StreamPayload (data/complete/error/heartbeat)
#   - Completion metadata serialized as JSON InferenceComplete (see rpc_types.rs)

enum AdaptationStrategy {
  autoWriteback @0;
  # Write back to delta if recommendation positive, evict if negative.
  autoEvict @1;
  # Always evict after generation — eval/benchmark mode.
  speculative @2;
  # Keep pending; client calls ttt.writeback or ttt.evict explicitly.
  writebackIfAbove @3;
  # Write back if loss_improvement exceeds writebackThreshold, else evict.
}

struct InferenceRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    generateStream @1 :GenerationRequest $mcpScope(infer);
    modelInfo @2 :Void $mcpScope(query);
    isReady @3 :Void $mcpScope(query);
    applyChatTemplate @4 :ChatTemplateRequest $mcpScope(query);

    # LoRA operations
    createLora @5 :LoraConfig $mcpScope(write);
    loadLora @6 :Text $mcpScope(write);       # path
    saveLora @7 :Text $mcpScope(write);       # path
    unloadLora @8 :Void $mcpScope(write);
    hasLora @9 :Void $mcpScope(query);

    # Session operations
    setSession @10 :Text $mcpScope(write);    # session_id
    clearSession @11 :Void $mcpScope(write);
    releaseSession @12 :Text $mcpScope(write);

    # Health/Lifecycle
    healthCheck @13 :Void $mcpScope(query);
    shutdown @14 :Void $mcpScope(manage);

    # Training loop control — tenant-aware TTT (identity from auth envelope)
    tttWriteback @15 :Void $mcpScope(train);
    tttEvict @16 :Void $mcpScope(train);
    trainStep @17 :TrainStepRequest $mcpScope(train);
    tttZero @18 :Void $mcpScope(manage);

    # Persistence operations (identity from auth envelope)
    getDeltaStatus @19 :Void $mcpScope(query);
    saveAdaptation @20 :SaveAdaptationRequest $mcpScope(write);
    snapshotDelta @21 :Void $mcpScope(write);

    # Streaming training (returns immediately, results via PUB/SUB)
    trainStepStream @22 :TrainStepRequest $mcpScope(train);

    # Export delta as PEFT adapter directory (identity from auth envelope)
    exportPeftAdapter @23 :ExportPeftRequest $mcpScope(write);

    # Merge an on-disk adapter into the loaded base_delta
    mergeLora @24 :MergeLoraRequest $mcpScope(write);

    # Streaming variants — return StreamInfo immediately, results via PUB/SUB.
    # Use these instead of the non-streaming versions for operations that may
    # involve significant compute or I/O (GPU alloc, disk writes, merges).
    createLoraStream @25 :LoraConfig $mcpScope(write);
    loadLoraStream @26 :Text $mcpScope(write);          # path
    saveLoraStream @27 :Text $mcpScope(write);          # path
    saveAdaptationStream @28 :SaveAdaptationRequest $mcpScope(write);
    snapshotDeltaStream @29 :Void $mcpScope(write);
    exportPeftAdapterStream @30 :ExportPeftRequest $mcpScope(write);
    mergeLoraStream @31 :MergeLoraRequest $mcpScope(write);

    # Vision embeddings (synchronous — returns all embeddings in one response)
    embed @32 :EmbedImagesRequest $mcpScope(infer);

    # TTN layer profile (returns JSON-encoded LayerProfile for diagnostics/tooling)
    getLayerProfile @33 :Void $mcpScope(query);
  }
}

struct InferenceResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload (union of response types)
  # Convention: response variant name = request variant name + "Result"
  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    generateStreamResult @3 :StreamInfo;
    modelInfoResult @4 :ModelInfo;
    isReadyResult @5 :Bool;
    applyChatTemplateResult @6 :Text;
    createLoraResult @7 :Void;
    loadLoraResult @8 :Void;
    saveLoraResult @9 :Void;
    unloadLoraResult @10 :Void;
    hasLoraResult @11 :Bool;
    setSessionResult @12 :Void;
    clearSessionResult @13 :Void;
    releaseSessionResult @14 :Void;
    healthCheckResult @15 :HealthStatus;

    # Training loop control responses
    tttWritebackResult @16 :Void;
    tttEvictResult @17 :Void;
    trainStepResult @18 :TrainStepResult;
    tttZeroResult @19 :Void;

    # Persistence responses
    getDeltaStatusResult @20 :DeltaStatusResult;
    saveAdaptationResult @21 :SaveAdaptationResult;
    snapshotDeltaResult @22 :SnapshotDeltaResult;

    # Streaming training response
    trainStepStreamResult @23 :StreamInfo;

    # Export PEFT adapter response
    exportPeftAdapterResult @24 :ExportPeftResult;

    # Merge LoRA response
    mergeLoraResult @25 :Void;

    # Streaming variant responses — all return StreamInfo
    createLoraStreamResult @26 :StreamInfo;
    loadLoraStreamResult @27 :StreamInfo;
    saveLoraStreamResult @28 :StreamInfo;
    saveAdaptationStreamResult @29 :StreamInfo;
    snapshotDeltaStreamResult @30 :StreamInfo;
    exportPeftAdapterStreamResult @31 :StreamInfo;
    mergeLoraStreamResult @32 :StreamInfo;

    # Embed result
    embedResult @33 :EmbedImagesResponse;

    # TTN layer profile result
    getLayerProfileResult @34 :LayerProfileResult;
  }
}

struct GenerationRequest {
  prompt @0 :Text;
  maxTokens @1 :Opt.OptionUint32;
  temperature @2 :Opt.OptionFloat32;
  topP @3 :Opt.OptionFloat32;
  topK @4 :Opt.OptionUint32;
  repeatPenalty @5 :Opt.OptionFloat32;
  repeatLastN @6 :Opt.OptionUint32;
  stopTokens @7 :List(Text) $optional;
  seed @8 :Opt.OptionUint32;
  images @9 :List(Data) $optional;
  timeoutMs @10 :Opt.OptionUint64;

  # Per-request TTT control (all optional — omit for server defaults)
  tttEnabled @11 :Bool
    $paramDescription("Override: enable/disable TTT for this request");
  tttGradientSteps @12 :Opt.OptionUint32
    $paramDescription("Override: number of gradient steps (None = skip)");
  tttLearningRate @13 :Opt.OptionFloat32
    $paramDescription("Override: learning rate");
  adaptationStrategy @14 :AdaptationStrategy
    $paramDescription("How to handle the adaptation result. autoWriteback: accept if recommendation positive, evict if negative. autoEvict: always evict (eval mode). speculative: keep pending, client calls ttt.writeback or ttt.evict. writebackIfAbove: accept if loss_improvement exceeds writebackThreshold.");
  writebackThreshold @15 :Float32 $optional
    $paramDescription("Loss improvement threshold for writebackIfAbove strategy. Ignored for other strategies.");
}

# Quality metrics for self-supervised training
struct QualityMetrics {
  perplexity @0 :Float32;
  avgEntropy @1 :Float32;
  entropyVariance @2 :Float32;
  repetitionRatio @3 :Float32;
}

# Online Training Metrics
#
# Online training adapts the model to input context BEFORE generation
# using next-token prediction loss. Metrics show adaptation effectiveness.
#
# Populated when:
# - Online training enabled in model config (mode = "test_time_training")
# - Input length >= min_input_length (default: 32 tokens)
# - LoRA adapter is loaded
struct OnlineTrainingMetrics {
  avgLoss @0 :Float32;              # Average loss across gradient steps
  lossImprovement @1 :Float32;       # Loss reduction (initial - final)
  stepsPerformed @2 :UInt32;         # Gradient steps executed
  adaptationTimeMs @3 :UInt64;       # Time spent on adaptation (ms)
  skipped @4 :Bool;                  # Whether adaptation was skipped
  skipReason @5 :Text;               # Why skipped (if applicable)

  # Advanced metrics for ML practitioners (expert recommendation)
  avgGradNorm @6 :Float32;           # Average gradient norm across steps
  maxGradNorm @7 :Float32;           # Maximum gradient norm observed
  gradientClipped @8 :Bool;          # Whether gradients were clipped
  tokensUsed @9 :UInt32;             # Tokens actually used for adaptation
  tokensProvided @10 :UInt32;        # Total tokens in input
  wasTruncated @11 :Bool;            # Whether input was truncated (>max_ttt_context)

  # Tenant-aware TTT metrics
  initialPerplexity @12 :Float32;    # Perplexity before adaptation
  finalPerplexity @13 :Float32;      # Perplexity after adaptation
  recommendation @14 :Bool;          # Server's commit/rollback recommendation
  gatedSteps @15 :UInt32;            # Steps determined by perplexity gating
  pending @16 :Bool;                 # Whether adaptation awaits client commit/rollback
}

enum FinishReason {
  maxTokens @0;
  stopToken @1;
  endOfSequence @2;
  error @3;
  stop @4;
}

# =============================================================================
# Legacy types removed:
#   - InferencePayload: Replaced by generic StreamPayload (streaming.capnp)
#   - InferenceComplete (capnp): The Rust/JSON version in rpc_types.rs is the
#     actual wire format, serialized into StreamPayload::Complete as JSON bytes.
#   - InferenceStats: Legacy, was replaced by InferenceComplete.
# =============================================================================

# Chat Template

struct ChatTemplateRequest {
  messages @0 :List(ChatMessage);
  addGenerationPrompt @1 :Bool;
  toolsJson @2 :Text $optional;  # JSON-serialized tools array (empty string = no tools)
}

# Tool call function details
struct ToolCallFunction {
  name @0 :Text;
  arguments @1 :Text;       # JSON string
}

# Tool call — nested structure matching openai_compat::ToolCall
struct ToolCall {
  id @0 :Text;
  toolType @1 :Text $serdeRename("type");   # "function"
  function @2 :ToolCallFunction;
}

struct ChatMessage {
  role @0 :Text;
  content @1 :Text;
  toolCalls @2 :List(ToolCall);
  toolCallId @3 :Text;
}

# LoRA Configuration

struct LoraConfig {
  rank @0 :UInt32;
  alpha @1 :Opt.OptionFloat32 $paramDescription("LoRA alpha scaling factor");
  dropout @2 :Opt.OptionFloat32 $paramDescription("Dropout rate during training");
  targetModules @3 :List(Text);
  learningRate @4 :Opt.OptionFloat32 $paramDescription("Learning rate for training (default: 1e-4)");
}

# Model Info

struct ModelInfo {
  name @0 :Text;
  architecture @1 :Text;
  vocabSize @2 :UInt32;
  hiddenSize @3 :UInt32;
  numHiddenLayers @4 :Opt.OptionUint32;
  numAttentionHeads @5 :Opt.OptionUint32;
  contextLength @6 :UInt32;
  quantization @7 :Text $optional;
  hasVision @8 :Bool;
  loraLoaded @9 :Bool;
  parameters @10 :Opt.OptionUint64;
  intermediateSize @11 :Opt.OptionUint32;
  numKeyValueHeads @12 :Opt.OptionUint32;
  headDim @13 :Opt.OptionUint32;
}

# Health Status

struct HealthStatus {
  status @0 :Text;
  modelLoaded @1 :Bool;
  kvCacheUsagePercent @2 :Float32;
  gpuMemoryUsedMb @3 :UInt32;
  gpuMemoryTotalMb @4 :UInt32;
}

# =============================================================================
# Training Loop Control (TTT operations)
# =============================================================================

struct TrainStepRequest {
  input @0 :Text;
  gradientSteps @1 :Opt.OptionUint32;
  learningRate @2 :Opt.OptionFloat32;
  adaptationStrategy @3 :AdaptationStrategy
    $paramDescription("How to handle the adaptation result. Same semantics as GenerationRequest.adaptationStrategy.");
  writebackThreshold @4 :Float32 $optional
    $paramDescription("Loss improvement threshold for writebackIfAbove strategy.");
}

struct TrainStepResult {
  avgLoss @0 :Float32;
  lossImprovement @1 :Float32;
  stepsPerformed @2 :UInt32;
  adaptationTimeMs @3 :UInt64;
  initialPerplexity @4 :Float32;
  finalPerplexity @5 :Float32;
  recommendation @6 :Bool;
  committed @7 :Bool;
  gradientClipped @8 :Bool;
}

struct SaveAdaptationRequest {
  name @0 :Text;
  mergeStrategy @1 :Text $optional;
  mergeWeight @2 :Opt.OptionFloat32;
  commitMessage @3 :Text $optional;
  gitCommit @4 :Bool $optional
    $paramDescription("If true, stage and commit the written file to the model repository after saving.");
}

struct SaveAdaptationResult {
  adapterName @0 :Text;
  adapterPath @1 :Text;
  contentHash @2 :Text;
  mergeStrategy @3 :Text;
}

struct DeltaStatusResult {
  exists @0 :Bool;
  accumulatedSteps @1 :UInt64
    $paramDescription("Total gradient steps committed to this delta. Counts toward maxAccumulatedSteps.");
  maxAccumulatedSteps @2 :UInt64
    $paramDescription("Hard capacity ceiling. When accumulatedSteps reaches this value, further TTT adaptation is silently skipped. Call saveLora/exportPeftAdapter then resetDelta to free capacity.");
  requestCount @3 :UInt64;
  avgLossImprovement @4 :Float32;
  memoryBytes @5 :UInt64;
  lastSnapshotHash @6 :Text;
  deltaNormRatios @7 :List(ModuleNormRatio);
  hasPending @8 :Bool
    $paramDescription("Whether a pending adaptation awaits commitAdaptation or rollbackAdaptation. Point-in-time snapshot — may become stale if the timeout expires.");
}

struct ModuleNormRatio {
  moduleName @0 :Text;
  ratio @1 :Float32;
}

struct SnapshotDeltaResult {
  contentHash @0 :Text;
  sizeBytes @1 :UInt64;
  accumulatedSteps @2 :UInt64;
  requestCount @3 :UInt64;
}

# Export PEFT Adapter

struct ExportPeftRequest {
  name @0 :Text;
  commitMessage @1 :Text $optional;
  gitCommit @2 :Bool $optional
    $paramDescription("If true, stage and commit the exported adapter directory to the model repository.");
}

struct ExportPeftResult {
  adapterPath @0 :Text;
  contentHash @1 :Text;
}

# Merge LoRA Request (adapter.merge → inference internal)

struct MergeLoraRequest {
  adapterPath @0 :Text $paramDescription("Adapter directory name or relative path to merge");
  weight @1 :Opt.OptionFloat32 $paramDescription("Merge weight 0.0-1.0 (default: 1.0 = full merge)");
  strategy @2 :Text $optional $paramDescription("Merge strategy: 'replace', 'additive', or 'do_merge' (default: 'do_merge')");
}

# Vision embedding request (raw image bytes)
struct EmbedImagesRequest {
  images @0 :List(Data);   # raw image bytes (PNG/JPEG/RGB)
}

# Vision embedding response
struct EmbedImagesResponse {
  embeddings @0 :List(List(Float32));  # one vector per image
  dimensions @1 :UInt32;               # embedding dimensionality
}

# TTN layer profile result (JSON-encoded to avoid complex capnp map types)
struct LayerProfileResult {
  json @0 :Text;   # JSON-encoded LayerProfile (serde_json::to_string_pretty)
}

# Structured capacity error for TTT delta limits
struct CapacityError {
  currentSteps @0 :UInt32;
  maxSteps @1 :UInt32;
  message @2 :Text;
}
