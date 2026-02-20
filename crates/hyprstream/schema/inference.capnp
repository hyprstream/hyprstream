@0xa8c9e2f1d3b5a7c0;

using import "/streaming.capnp".StreamInfo;
using import "/annotations.capnp".optional;

# Cap'n Proto schema for inference service
#
# The inference service uses REQ/REP pattern for request handling.
# Streaming uses PUB/SUB with stream IDs for chunk delivery.
#
# Streaming Architecture:
#   - Wire format types are in hyprstream-rpc/schema/streaming.capnp
#   - Streaming payloads use generic StreamPayload (data/complete/error/heartbeat)
#   - Completion metadata serialized as JSON InferenceComplete (see rpc_types.rs)

struct InferenceRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    generateStream @1 :GenerationRequest;
    modelInfo @2 :Void;
    isReady @3 :Void;
    applyChatTemplate @4 :ChatTemplateRequest;

    # LoRA operations
    createLora @5 :LoraConfig;
    loadLora @6 :Text;       # path
    saveLora @7 :Text;       # path
    unloadLora @8 :Void;
    hasLora @9 :Void;

    # Session operations
    setSession @10 :Text;    # session_id
    clearSession @11 :Void;
    releaseSession @12 :Text;

    # Health/Lifecycle
    healthCheck @13 :Void;
    shutdown @14 :Void;

    # Training loop control — tenant-aware TTT (identity from auth envelope)
    commitAdaptation @15 :Void;
    rollbackAdaptation @16 :Void;
    trainStep @17 :TrainStepRequest;
    resetDelta @18 :Void;

    # Persistence operations (identity from auth envelope)
    getDeltaStatus @19 :Void;
    saveAdaptation @20 :SaveAdaptationRequest;
    snapshotDelta @21 :Void;

    # Streaming training (returns immediately, results via PUB/SUB)
    trainStepStream @22 :TrainStepRequest;

    # Export delta as PEFT adapter directory (identity from auth envelope)
    exportPeftAdapter @23 :ExportPeftRequest;
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
    commitAdaptationResult @16 :Void;
    rollbackAdaptationResult @17 :Void;
    trainStepResult @18 :TrainStepResult;
    resetDeltaResult @19 :Void;

    # Persistence responses
    getDeltaStatusResult @20 :DeltaStatusResult;
    saveAdaptationResult @21 :SaveAdaptationResult;
    snapshotDeltaResult @22 :SnapshotDeltaResult;

    # Streaming training response
    trainStepStreamResult @23 :StreamInfo;

    # Export PEFT adapter response
    exportPeftAdapterResult @24 :ExportPeftResult;
  }
}

struct GenerationRequest {
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

struct ToolCallData {
  id @0 :Text;
  callType @1 :Text;        # "function"
  functionName @2 :Text;
  arguments @3 :Text;        # JSON string (opaque, deserialized at consumption point)
}

struct ChatMessage {
  role @0 :Text;
  content @1 :Text;
  toolCalls @2 :List(ToolCallData);
  toolCallId @3 :Text;
}

# LoRA Configuration

struct LoraConfig {
  rank @0 :UInt32;
  alpha @1 :Float32;
  dropout @2 :Float32;
  targetModules @3 :List(Text);
  learningRate @4 :Float32;
}

# Model Info

struct ModelInfo {
  modelId @0 :Text;
  architecture @1 :Text;
  vocabSize @2 :UInt32;
  hiddenSize @3 :UInt32;
  numLayers @4 :UInt32;
  numHeads @5 :UInt32;
  maxSequenceLength @6 :UInt32;
  quantization @7 :Text;
  hasVision @8 :Bool;
  loraLoaded @9 :Bool;
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
  gradientSteps @1 :UInt32 $optional;
  learningRate @2 :Float32 $optional;
  autoCommit @3 :Bool;
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
  mergeWeight @2 :Float32 $optional;
  commitMessage @3 :Text $optional;
}

struct SaveAdaptationResult {
  adapterName @0 :Text;
  adapterPath @1 :Text;
  contentHash @2 :Text;
  mergeStrategy @3 :Text;
}

struct DeltaStatusResult {
  exists @0 :Bool;
  accumulatedSteps @1 :UInt64;
  maxAccumulatedSteps @2 :UInt64;
  requestCount @3 :UInt64;
  avgLossImprovement @4 :Float32;
  memoryBytes @5 :UInt64;
  lastSnapshotHash @6 :Text;
  deltaNormRatios @7 :List(ModuleNormRatio);
  hasPending @8 :Bool;
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
}

struct ExportPeftResult {
  adapterPath @0 :Text;
  contentHash @1 :Text;
}

# Error Information

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}
