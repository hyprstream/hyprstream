@0xcf8bf3d833f22b0d;

using import "/common.capnp".ErrorInfo;
using import "/streaming.capnp".StreamInfo;
using import "/annotations.capnp".optional;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".dispatchMac;
using import "/annotations.capnp".mutationSemantics;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".serdeRename;
using Opt = import "/optional.capnp";
# System One typed decision surface (P3.1): the E2 #1663 data contract
# (distributions, VersionTriple with explicit calibration-null) is imported
# from decision.capnp; service-owned wrapper types live here.
using Decision = import "/decision.capnp";

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
    generateStream @1 :GenerationRequest $scope(infer) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    modelInfo @2 :Void $scope(query) $dispatchMac("internal:pq-hybrid");
    isReady @3 :Void $scope(query) $dispatchMac("internal:pq-hybrid");
    applyChatTemplate @4 :ChatTemplateRequest $scope(query) $dispatchMac("internal:pq-hybrid");

    # LoRA operations
    # createLora/loadLora bump `lora_generation` (fetch_add) on success; that
    # counter feeds GuardStatus and adaptation_state.resolve, so a replayed
    # call invalidates pending adaptation work instead of converging. Retry
    # safety requires a caller-supplied application idempotency key plus a
    # recorded result per key (the v16 §4.8 caller-key/result-record
    # dependency); neither the payload key nor the result record exists
    # today — this declares the missing activation work.
    createLora @5 :LoraConfig $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    loadLora @6 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");       # path
    saveLora @7 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");       # path
    unloadLora @8 :Void $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");
    hasLora @9 :Void $scope(query) $dispatchMac("internal:pq-hybrid");

    # Session operations
    setSession @10 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");    # session_id
    clearSession @11 :Void $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");
    releaseSession @12 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # Health/Lifecycle
    healthCheck @13 :Void $scope(query) $dispatchMac("internal:pq-hybrid");
    shutdown @14 :Void $scope(manage) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # Training loop control — tenant-aware TTT (identity from auth envelope)
    tttWriteback @15 :Void $scope(train) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");
    tttEvict @16 :Void $scope(train) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");
    # A retried trainStep trains on the same input twice — the per-subject
    # TTT delta advances and future inference diverges. Retry safety requires
    # a caller-supplied application key (matching trainStepStream's existing
    # classification); no key machinery is implemented here.
    trainStep @17 :TrainStepRequest $scope(train) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    tttZero @18 :Void $scope(manage) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # Persistence operations (identity from auth envelope)
    getDeltaStatus @19 :Void $scope(query) $dispatchMac("internal:pq-hybrid");
    saveAdaptation @20 :SaveAdaptationRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    snapshotDelta @21 :Void $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # Streaming training (returns immediately, results via PUB/SUB)
    trainStepStream @22 :TrainStepRequest $scope(train) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");

    # Export delta as PEFT adapter directory (identity from auth envelope)
    exportPeftAdapter @23 :ExportPeftRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # Merge an on-disk adapter into the loaded base_delta
    mergeLora @24 :MergeLoraRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");

    # Streaming variants — return StreamInfo immediately, results via PUB/SUB.
    # Use these instead of the non-streaming versions for operations that may
    # involve significant compute or I/O (GPU alloc, disk writes, merges).
    # Every *_stream variant calls setup_stream: it allocates a fresh
    # third-party interop stream under the client's ephemeral pubkey and
    # schedules the work continuation before the reply is observed, so a
    # replayed request duplicates the allocation while the original stream
    # stays active. Retry safety requires a caller-supplied application
    # idempotency key plus a recorded result (the same allocation/continuation
    # effect already classified key-required for trainStepStream,
    # saveAdaptationStream and mergeLoraStream); the key and result record do
    # not exist today — this declares the missing activation work.
    createLoraStream @25 :LoraConfig $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    loadLoraStream @26 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");          # path
    saveLoraStream @27 :Text $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");          # path
    saveAdaptationStream @28 :SaveAdaptationRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    snapshotDeltaStream @29 :Void $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    exportPeftAdapterStream @30 :ExportPeftRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");
    mergeLoraStream @31 :MergeLoraRequest $scope(write) $dispatchMac("internal:pq-hybrid") $mutationSemantics("idempotency-key-required");

    # Vision embeddings (synchronous — returns all embeddings in one response)
    embed @32 :EmbedImagesRequest $scope(infer) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");

    # TTN layer profile (returns JSON-encoded LayerProfile for diagnostics/tooling)
    getLayerProfile @33 :Void $scope(query) $dispatchMac("internal:pq-hybrid");

    # Typed Clef decision inference (text/JSON only). Parses the ORIGINAL raw
    # Clef JSON body server-side through the ordered ClefRequest visitor;
    # returns E2 decision.capnp distributions with calibration=null.
    # Pure function of (admitted weights, request body): no generation state,
    # no KV/GDN reuse, no session mutation — replay yields the same answer.
    decide @34 :DecisionRequest $scope(decide) $dispatchMac("internal:pq-hybrid") $mutationSemantics("naturally-idempotent");
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

    # Decision inference result
    decideResult @35 :DecisionResponse;
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
  maxTokens @3 :Opt.OptionUint32
      $paramDescription("Tokens reserved for generation. When set, oldest non-system messages are truncated so the rendered prompt fits in max_seq_len - maxTokens.");
  enableThinking @4 :Opt.OptionBool
      $paramDescription("Enable reasoning/thinking mode. True: emit <think> block. False: suppress thinking. Unset: use model template default.");
  templateVarsJson @5 :Text $optional
      $paramDescription("JSON object of additional template variables merged into the render context. Use for model-specific knobs not covered by first-class fields (e.g. {\"add_vision_id\": true}).");
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

# =============================================================================
# Typed decision serving (P3.1) — Clef joint-schema decisions
# =============================================================================

# One text/JSON Clef decision request. The body is the ORIGINAL raw Clef
# wire JSON: the server parses it directly with the ordered ClefRequest
# visitor (question insertion order preserved; duplicate request fields and
# question IDs rejected; unsupported media fields rejected; numeric domain
# enforced lexically BEFORE any Value conversion). It is deliberately NOT
# pre-parsed into an unordered Value map on either side of the wire.
struct DecisionRequest {
  clefJson @0 :Text
    $paramDescription("Original raw Clef request JSON body (model/state/questions record). Bounded server-side before the ordered parse.");

  # Optional explicit state-token budget override. The server clamps it to the
  # serving profile: callers may LOWER the default, never raise it.
  maxStateTokens @1 :Opt.OptionUint32
    $paramDescription("Maximum state tokens admitted before deterministic truncation. Unset/clamped to server default.");

  # Optional wall-clock budget for this decision. Clamped by the server to a
  # hard maximum; enforcement is cooperative (checked before admission, per
  # decoder layer, and around head evaluation). An in-flight libtorch kernel
  # completes before the next check; client disconnect alone does not stop a
  # running CPU forward.
  timeoutMs @2 :Opt.OptionUint32
    $paramDescription("Soft deadline in milliseconds for this decision request. Clamped to the server maximum.");
}

# Actual identity of the serving path that produced the batch. These are
# OBSERVED values from the admitted immutable artifact — never directory
# names, request echoes, or hardcoded revisions. Decision serving requires a
# retained pinned artifact; artifactCommit is its immutable content identity.
# Digests are computed once from the same bytes the runtime loaded. calibration
# identity stays inside DecisionBatch.version (OptionText; none = uncalibrated).
struct DecisionProvenance {
  encodingVersion @0 :Text
    $paramDescription("Clef encoding contract version actually used (e.g. cloudflare-clef-8b2e5fd-1).");
  modelRef @1 :Text
    $paramDescription("Admitted worker model reference (resolved identity), e.g. name:branch. Empty only for unbound standalone fixtures.");
  artifactCommit @2 :Text
    $paramDescription("Pinned artifact commit OID (immutable content identity of the whole model tree). Decision serving fails closed without one.");
  headWeightsSha256 @3 :Text
    $paramDescription("SHA-256 of the joint head weights actually loaded.");
  headConfigSha256 @4 :Text
    $paramDescription("SHA-256 of the joint head config actually loaded.");
  tokenizerSha256 @5 :Text
    $paramDescription("SHA-256 of tokenizer.json in the same immutable tree; empty when the bundle ships none.");
  backboneConfigSha256 @6 :Text
    $paramDescription("SHA-256 of the backbone config.json in the same immutable tree; empty when absent.");
  quantization @7 :Text
    $paramDescription("Quantization format actually declared by config.json (e.g. float-quantized-channel-w8-dynact); empty when unquantized.");
  headTensorCount @8 :UInt32
    $paramDescription("Number of strict-validated head tensors loaded (122 for the official joint head config).");
  computeProfile @9 :Text
    $paramDescription("Compute profile actually used, e.g. cpu-float-quantized-channel-w8-dynact or cpu-float32. No hidden precision fallback.");
}

# The limits actually applied to this record (schema overflow is rejected,
# never silently truncated).
struct DecisionAppliedLimits {
  maxLength @0 :UInt32
    $paramDescription("Total encoding token budget applied.");
  maxStateTokens @1 :Opt.OptionUint32
    $paramDescription("State-token budget applied after None resolution.");
  stateTokensAdmitted @2 :UInt32
    $paramDescription("State tokens present in the encoded sequence AFTER truncation.");
  sequenceTokens @3 :UInt32
    $paramDescription("Final input token count of the encoded record.");
  stateTruncated @4 :Bool
    $paramDescription("Whether the state segment was truncated to fit the budget.");
}

struct DecisionResponse {
  # Typed decision distributions per the E2 decision.capnp contract:
  # rows bind answers by request question IDs; VersionTriple carries
  # schema/model identity with calib = none (uncalibrated native output).
  batch @0 :Decision.DecisionBatch;
  provenance @1 :DecisionProvenance;
  limits @2 :DecisionAppliedLimits;
}
