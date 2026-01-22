@0xa8c9e2f1d3b5a7c0;

# Cap'n Proto schema for inference service
#
# The inference service uses REQ/REP pattern for request handling.
# Streaming uses PUB/SUB with stream IDs for chunk delivery.
#
# Streaming Architecture:
#   - Wire format types are in hyprstream-rpc/schema/streaming.capnp
#   - This file defines inference-specific payload types
#   - InferencePayload gets serialized into streaming.capnp::StreamChunk.data

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
  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    generationResult @3 :GenerationResult;
    streamStarted @4 :StreamInfo;
    modelInfo @5 :ModelInfo;
    ready @6 :Bool;
    templateResult @7 :Text;
    loraCreated @8 :Void;
    loraLoaded @9 :Void;
    loraSaved @10 :Void;
    loraUnloaded @11 :Void;
    hasLoraResult @12 :Bool;
    sessionSet @13 :Void;
    sessionCleared @14 :Void;
    sessionReleased @15 :Void;
    health @16 :HealthStatus;

    # Stream authorization response
    # Future: will include server public key for DH key exchange
    streamAuthorized @17 :StreamAuthResponse;
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
}

# Quality metrics for self-supervised training
struct QualityMetrics {
  perplexity @0 :Float32;
  avgEntropy @1 :Float32;
  entropyVariance @2 :Float32;
  repetitionRatio @3 :Float32;
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
# Note: Matches streaming.capnp::StreamStartRequest/StreamAuthResponse

struct StartStreamRequest {
  streamId @0 :Text;
  clientPubkey @1 :Data;  # Client's ephemeral Ristretto255 public key (32 bytes)
}

struct StreamAuthResponse {
  streamId @0 :Text;
  serverPubkey @1 :Data;  # Server's ephemeral Ristretto255 public key (if not in StreamInfo)
}

# =============================================================================
# Inference Payload (serialized into streaming.capnp::StreamChunk.data)
# =============================================================================

# The actual inference payload - gets serialized into wire format StreamChunk.data
# This is the application-layer content, not the wire format.
struct InferencePayload {
  streamId @0 :Text;

  union {
    token @1 :Text;                   # Generated token text
    complete @2 :InferenceStats;      # Generation complete with stats
    error @3 :ErrorInfo;              # Error during generation
  }
}

# Inference-specific completion statistics (extends streaming.capnp::StreamStats)
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
