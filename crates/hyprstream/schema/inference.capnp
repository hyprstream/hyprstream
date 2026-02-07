@0xa8c9e2f1d3b5a7c0;

# Cap'n Proto schema for inference service
#
# The inference service uses REQ/REP pattern for request handling.
# Streaming uses PUB/SUB with stream IDs for chunk delivery.
#
# Streaming Architecture:
#   - Wire format types are in hyprstream-rpc/schema/streaming.capnp
#   - This file defines inference-specific payload types
#   - InferencePayload gets serialized into streaming.capnp::StreamBlock.payloads

struct InferenceRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    generate @1 :GenerationRequest;
    generateStream @2 :GenerationRequest;
    modelInfo @3 :Void;
    isReady @4 :Void;
    applyChatTemplate @5 :ChatTemplateRequest;

    # LoRA operations
    createLora @6 :LoraConfig;
    loadLora @7 :Text;       # path
    saveLora @8 :Text;       # path
    unloadLora @9 :Void;
    hasLora @10 :Void;

    # Session operations
    setSession @11 :Text;    # session_id
    clearSession @12 :Void;
    releaseSession @13 :Text;

    # Health/Lifecycle
    healthCheck @14 :Void;
    shutdown @15 :Void;

    # Stream authorization handshake
    # Client calls this after generateStream to authorize subscription
    # Future: will include client public key for DH key exchange
    startStream @16 :StartStreamRequest;
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
    generateResult @3 :GenerationResult;
    generateStreamResult @4 :StreamInfo;
    modelInfoResult @5 :ModelInfo;
    isReadyResult @6 :Bool;
    applyChatTemplateResult @7 :Text;
    createLoraResult @8 :Void;
    loadLoraResult @9 :Void;
    saveLoraResult @10 :Void;
    unloadLoraResult @11 :Void;
    hasLoraResult @12 :Bool;
    setSessionResult @13 :Void;
    clearSessionResult @14 :Void;
    releaseSessionResult @15 :Void;
    healthCheckResult @16 :HealthStatus;

    # Stream authorization response
    startStreamResult @17 :StreamAuthResponse;
  }
}

struct GenerationRequest {
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

struct GenerationResult {
  text @0 :Text;
  tokensGenerated @1 :UInt32;
  finishReason @2 :FinishReason;
  generationTimeMs @3 :UInt64;
  tokensPerSecond @4 :Float32;
  qualityMetrics @5 :QualityMetrics;
  # Prefill metrics (processing prompt)
  prefillTokens @6 :UInt32;
  prefillTimeMs @7 :UInt64;
  prefillTokensPerSec @8 :Float32;
  # Inference metrics (generating tokens)
  inferenceTokens @9 :UInt32;
  inferenceTimeMs @10 :UInt64;
  inferenceTokensPerSec @11 :Float32;

  # Online training adaptation metrics (optional)
  onlineTrainingMetrics @12 :OnlineTrainingMetrics;
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
}

enum FinishReason {
  maxTokens @0;
  stopToken @1;
  endOfSequence @2;
  error @3;
  stop @4;
}

# =============================================================================
# Stream Setup (aligns with streaming.capnp::StreamInfo)
# =============================================================================

# Response when starting a stream - contains info needed to subscribe
# Note: Matches streaming.capnp::StreamInfo for consistency
struct StreamInfo {
  streamId @0 :Text;
  endpoint @1 :Text;
  serverPubkey @2 :Data;  # Server's ephemeral Ristretto255 public key (32 bytes) for DH
}

# Stream authorization handshake
# Note: Matches streaming.capnp::StartStreamRequest/StreamAuthResponse

struct StartStreamRequest {
  streamId @0 :Text;
  clientPubkey @1 :Data;  # Client's ephemeral Ristretto255 public key (32 bytes)
}

struct StreamAuthResponse {
  streamId @0 :Text;
  serverPubkey @1 :Data;  # Server's ephemeral Ristretto255 public key (if not in StreamInfo)
}

# =============================================================================
# Inference Payload (serialized into streaming.capnp::StreamBlock.payloads)
# =============================================================================

# The actual inference payload - gets serialized into wire format StreamBlock.payloads
# This is the application-layer content, not the wire format.
struct InferencePayload {
  streamId @0 :Text;

  union {
    token @1 :Text;                   # Generated token text
    complete @2 :InferenceStats;      # Generation complete with stats
    error @3 :ErrorInfo;              # Error during generation
  }
}

# Inference-specific completion statistics
#
# Serialized into StreamPayload.complete (streaming.capnp) as raw bytes.
# Contains full generation metrics including prefill and inference breakdown.
struct InferenceComplete {
  # Overall metrics
  tokensGenerated @0 :UInt32;
  finishReason @1 :Text;          # "stop", "length", "eos", "error"
  generationTimeMs @2 :UInt64;
  tokensPerSecond @3 :Float32;

  # Prefill metrics (processing the prompt)
  prefillTokens @4 :UInt32;
  prefillTimeMs @5 :UInt64;
  prefillTokensPerSec @6 :Float32;

  # Inference metrics (generating new tokens, excluding prefill)
  inferenceTokens @7 :UInt32;
  inferenceTimeMs @8 :UInt64;
  inferenceTokensPerSec @9 :Float32;
  inferenceTokensPerSecEma @10 :Float32;  # EMA for adaptive batching

  # Optional quality metrics (0.0 means not set)
  perplexity @11 :Float32;
  avgEntropy @12 :Float32;

  # Online training metrics in streaming completion
  onlineTrainingMetrics @13 :OnlineTrainingMetrics;
}

# Legacy inference stats (kept for backwards compatibility)
struct InferenceStats {
  tokensGenerated @0 :UInt32;
  finishReason @1 :FinishReason;
  generationTimeMs @2 :UInt64;
  tokensPerSecond @3 :Float32;
  qualityMetrics @4 :QualityMetrics;  # Inference-specific quality metrics
}

# Chat Template

struct ChatTemplateRequest {
  messages @0 :List(ChatMessage);
  addGenerationPrompt @1 :Bool;
}

struct ChatMessage {
  role @0 :Text;
  content @1 :Text;
}

# LoRA Configuration

struct LoraConfig {
  rank @0 :UInt32;
  alpha @1 :Float32;
  dropout @2 :Float32;
  targetModules @3 :List(Text);
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

# Error Information

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}
