@0xb3d7f8e4a2c1e5f9;

# Cap'n Proto schema for hyprstream events
#
# Events are published via ZMQ PUB/SUB pattern.
# The EventEnvelope wraps all event types for unified handling.

struct EventEnvelope {
  # Unique event ID (UUID bytes)
  id @0 :Data;

  # Timestamp (Unix milliseconds)
  timestamp @1 :Int64;

  # Correlation ID for request tracing (optional UUID)
  correlationId @2 :Data;

  # Event source
  source @3 :EventSource;

  # Topic for ZMQ prefix filtering
  topic @4 :Text;

  # Event payload (union of event types)
  payload @5 :EventPayload;
}

enum EventSource {
  inference @0;
  metrics @1;
  training @2;
  git2db @3;
  system @4;
  editor @5;
}

struct EventPayload {
  union {
    # Inference events
    generationComplete @0 :GenerationComplete;
    generationFailed @1 :GenerationFailed;
    generationStarted @2 :GenerationStarted;

    # Metrics events
    thresholdBreach @3 :ThresholdBreach;
    metricsSnapshot @4 :MetricsSnapshot;

    # Training events
    trainingStarted @5 :TrainingStarted;
    trainingProgress @6 :TrainingProgress;
    trainingCompleted @7 :TrainingCompleted;
    checkpointSaved @8 :CheckpointSaved;

    # Git/Repository events
    repositoryCloned @9 :RepositoryCloned;
    commitCreated @10 :CommitCreated;
    branchCreated @11 :BranchCreated;
    worktreeCreated @12 :WorktreeCreated;

    # System events
    serverStarted @13 :ServerStarted;
    serverStopping @14 :ServerStopping;
    healthCheck @15 :HealthCheck;

    # Editor/CRDT events
    crdtSync @16 :CrdtSync;
  }
}

# Inference Events

struct GenerationComplete {
  modelId @0 :Text;
  sessionId @1 :Text;
  metrics @2 :GenerationMetrics;
}

struct GenerationFailed {
  modelId @0 :Text;
  sessionId @1 :Text;
  error @2 :Text;
  errorCode @3 :Text;
}

struct GenerationStarted {
  modelId @0 :Text;
  sessionId @1 :Text;
  requestId @2 :Text;
}

struct GenerationMetrics {
  perplexity @0 :Float32;
  avgEntropy @1 :Float32;
  entropyVariance @2 :Float32;
  repetitionRatio @3 :Float32;
  tokenCount @4 :UInt32;
  tokensPerSecond @5 :Float32;
  generationTimeMs @6 :UInt64;
}

# Metrics Events

struct ThresholdBreach {
  modelId @0 :Text;
  metric @1 :Text;
  threshold @2 :Float64;
  actual @3 :Float64;
  zScore @4 :Float64;
}

struct MetricsSnapshot {
  modelId @0 :Text;
  metrics @1 :List(MetricEntry);
}

struct MetricEntry {
  name @0 :Text;
  value @1 :Float64;
}

# Training Events

struct TrainingStarted {
  modelId @0 :Text;
  adapterName @1 :Text;
  epochs @2 :UInt32;
  learningRate @3 :Float32;
}

struct TrainingProgress {
  modelId @0 :Text;
  adapterName @1 :Text;
  epoch @2 :UInt32;
  step @3 :UInt32;
  loss @4 :Float32;
  learningRate @5 :Float32;
}

struct TrainingCompleted {
  modelId @0 :Text;
  adapterName @1 :Text;
  finalLoss @2 :Float32;
  totalSteps @3 :UInt32;
  durationMs @4 :UInt64;
}

struct CheckpointSaved {
  modelId @0 :Text;
  adapterName @1 :Text;
  checkpointPath @2 :Text;
  step @3 :UInt32;
}

# Git/Repository Events

struct RepositoryCloned {
  repoId @0 :Text;
  name @1 :Text;
  url @2 :Text;
  worktreePath @3 :Text;
}

struct CommitCreated {
  repoId @0 :Text;
  commitOid @1 :Text;
  message @2 :Text;
  author @3 :Text;
}

struct BranchCreated {
  repoId @0 :Text;
  branchName @1 :Text;
  baseBranch @2 :Text;
}

struct WorktreeCreated {
  repoId @0 :Text;
  worktreePath @1 :Text;
  branchName @2 :Text;
}

# System Events

struct ServerStarted {
  version @0 :Text;
  endpoints @1 :List(Text);
  features @2 :List(Text);
}

struct ServerStopping {
  reason @0 :Text;
}

struct HealthCheck {
  status @0 :Text;
  uptimeMs @1 :UInt64;
  activeRequests @2 :UInt32;
  memoryUsageMb @3 :UInt32;
}

# Editor/CRDT Events

struct CrdtSync {
  docId @0 :Text;        # "{repo_id}:{worktree}:{path}"
  actorId @1 :Data;
  changeBytes @2 :Data;
}
