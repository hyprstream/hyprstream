@0xa40b1708e6b364c8;

# Cap'n Proto schema for MetricsService
#
# Wraps hyprstream-metrics (DuckDB/DataFusion) for ingest, structured queries,
# streaming query results, view management, and health checks over ZMQ REQ/REP.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".dispatchMac;
using import "/annotations.capnp".mutationSemantics;
using import "/annotations.capnp".mcpDescription;
using import "/streaming.capnp".StreamInfo;

# Mirrors hyprstream_metrics::metrics::MetricRecord (schema v2).
# Fields reflect running-window aggregation storage plus labels/tenant.
# labels encodes key/value label pairs (reusing GroupEntry); tenantId empty = node-global.
struct MetricRecord {
  metricId          @0 :Text;
  timestamp         @1 :Int64;
  valueWindowSum    @2 :Float64;   # value_running_window_sum
  valueWindowAvg    @3 :Float64;   # value_running_window_avg
  valueWindowCount  @4 :Int64;     # value_running_window_count
  labels            @5 :List(GroupEntry);  # v2: metric labels; empty = unlabeled
  tenantId          @6 :Text;              # v2: tenant key; empty = no tenant
}

enum AggregationFunc {
  rawSql @0;   # Retained for wire compatibility; raw SQL is rejected by query RPCs
  count  @1;
  sum    @2;
  avg    @3;
  min    @4;
  max    @5;
}

struct MetricQuery {
  sql             @0 :Text;          # Deprecated raw SQL field; non-empty values are rejected
  metricId        @1 :Text;          # Structured filter (translated to WHERE clause)
  windowSecs      @2 :UInt32;        # Fixed time window seconds (0 = none)
  aggregation     @3 :AggregationFunc;
  groupBy         @4 :List(Text);    # Maps to GroupBy.columns
  limitRows       @5 :UInt32;        # 0 = no limit
  ephemeralPubkey @6 :Data;          # queryStream only: Ristretto255 client ephemeral pubkey
}

# Maps to AggregateResult { value: f64, group_values: HashMap<String,String>, timestamp: Option<i64> }.
struct AggregateRow {
  value       @0 :Float64;
  groupValues @1 :List(GroupEntry);
  timestamp   @2 :Int64;              # 0 if None
}

struct GroupEntry {
  key   @0 :Text;
  value @1 :Text;
}

# ViewSpec carries a MetricQuery that is translated to ViewDefinition on the server.
struct ViewSpec {
  name  @0 :Text;
  query @1 :MetricQuery;
}

# ViewInfo.sql is populated from ViewDefinition.to_sql() server-side.
struct ViewInfo {
  name      @0 :Text;
  sql       @1 :Text;
  createdAt @2 :Int64;   # Unix timestamp seconds
}

struct HealthInfo {
  ok          @0 :Bool;
  rowCount    @1 :UInt64;
  backendType @2 :Text;   # "duckdb"
}

struct IngestRequest {
  records @0 :List(MetricRecord);
}

struct MetricsRequest {
  id @0 :UInt64;

  union {
    ingest       @1 :IngestRequest
      $scope(write) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Ingest metric records into the time-series store");

    query        @2 :MetricQuery
      $scope(query) $dispatchMac("internal:pq-hybrid") $mcpDescription("Execute a structured metrics aggregation query; raw SQL is rejected");

    # Prepares a server-side third-party interop stream context under the
    # client's ephemeral pubkey and schedules the query continuation before
    # the reply is observed, so a replay duplicates the allocation/work — the
    # same allocation/continuation effect already classified key-required for
    # the other streaming leaves. Retry safety requires a caller-supplied
    # application idempotency key plus a recorded result; neither exists
    # today — this declares the missing activation work.
    queryStream  @3 :MetricQuery
      $scope(query) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Stream structured metrics query results as Arrow IPC RecordBatch chunks; raw SQL is rejected");

    createView   @4 :ViewSpec
      $scope(manage) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Create a materialized view over the metrics table");

    listViews    @5 :Void
      $scope(query) $dispatchMac("internal:pq-hybrid") $mcpDescription("List all materialized views");

    dropView     @6 :Text
      $scope(manage) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Drop a materialized view by name");

    health       @7 :Void
      $scope(query) $dispatchMac("internal:pq-hybrid") $mcpDescription("Check service health and row count");

    # R6 durable decision/outcome surface. Tenant authority never travels on
    # the wire: handlers derive the tenant domain from the VERIFIED envelope
    # context (ctx.domain()) and authorize in that domain — a global metrics
    # grant never implies tenant-store permission. Payload is bounded and
    # schema-validated before any durable write.
    recordDecision @8 :DecisionRecordRequest
      $scope(write) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Store one decision batch durably with full identity pins and stable per-row ids; append-only");

    recordOutcome @9 :DecisionOutcomeRequest
      $scope(write) $mutationSemantics("idempotency-key-required") $dispatchMac("internal:pq-hybrid") $mcpDescription("Record one realized outcome linked to an exact decision row and question; append-only, corrections supersede");

    queryOutcomes @10 :DecisionOutcomeQuery
      $scope(query) $dispatchMac("internal:pq-hybrid") $mcpDescription("Join realized outcomes with decision pins for the caller tenant under bounded SQL limits");
  }
}

struct MetricsResponse {
  requestId @0 :UInt64;

  union {
    ingestResult      @1 :Void;
    queryResult       @2 :List(AggregateRow);
    queryStreamResult @3 :StreamInfo;
    createViewResult  @4 :Void;
    listViewsResult   @5 :List(ViewInfo);
    dropViewResult    @6 :Void;
    healthResult      @7 :HealthInfo;
    error             @8 :ErrorInfo;
    recordDecisionResult @9 :DecisionRecordResult;
    recordOutcomeResult  @10 :DecisionOutcomeResult;
    queryOutcomesResult  @11 :DecisionOutcomeQueryResult;
  }
}

# ═══════════════════════════════════════════════════════════════════════════
# R6 durable decision/outcome types (hyprstream-metrics-api decision_outcome)
# ═══════════════════════════════════════════════════════════════════════════

struct DecisionRecordRequest {
  schemaFingerprint @0 :Text;
  modelId           @1 :Text;
  encodingId        @2 :Text;
  quantization      @3 :Text;
  computeProfile    @4 :Text;
  calibVersion      @5 :Text;   # empty = raw (uncalibrated) inference
  payload           @6 :Data;   # Arrow IPC stream; bounded decode server-side
}

struct DecisionRecordResult {
  decisionId @0 :Text;
  rowIds     @1 :List(Text);  # stable per-input-row ids, payload order
}

enum ExpectedCalib {
  any     @0;   # no calibration assertion
  raw     @1;   # decision row must be uncalibrated (calibVersion NULL)
  version @2;   # exact calibVersion match
}

struct DecisionOutcomeRequest {
  decisionId             @0 :Text;
  decisionRowId          @1 :Text;
  questionId             @2 :Text;
  # Complete-identity assertions (R6B); empty text = not asserted.
  expectedSchemaFingerprint @3 :Text;
  expectedModelId           @4 :Text;
  expectedEncodingId        @5 :Text;
  expectedQuantization      @6 :Text;
  expectedComputeProfile    @7 :Text;
  expectedCalib             @8 :ExpectedCalib;
  expectedCalibVersion      @9 :Text;
  # Realized outcome; empty label = absent, hasValue gates the numeric score
  # channel (0.0 is a legitimate value and must be distinguishable from
  # absent).
  outcomeLabel           @10 :Text;
  outcomeValue           @11 :Float64;
  hasOutcomeValue        @12 :Bool;
  outcomeSource          @13 :Text;
  supersedesOutcomeId    @14 :Text;  # empty = new observation, not a correction
}

struct DecisionOutcomeResult {
  outcomeId @0 :Text;
}

struct DecisionOutcomeQuery {
  schemaFingerprint @0 :Text;   # empty = all
  modelId           @1 :Text;   # empty = all
  questionId        @2 :Text;   # empty = all
  calibFilter       @3 :ExpectedCalib;
  calibVersion      @4 :Text;   # only for calibFilter = version
  includeSuperseded @5 :Bool;
  limit             @6 :UInt32; # 0 = server default; hard-capped server-side
  afterCreatedAt    @7 :Int64;  # ASC keyset; zero + empty id = first page
  afterOutcomeId    @8 :Text;   # bounded stored id; cursor is not authority
}

struct DecisionOutcomeJoinRow {
  outcomeId          @0 :Text;
  tenantId           @1 :Text;
  decisionId         @2 :Text;
  decisionRowId      @3 :Text;
  questionId         @4 :Text;
  outcomeLabel       @5 :Text;   # empty = none
  outcomeValue       @6 :Float64;
  hasOutcomeValue    @7 :Bool;
  outcomeSource      @8 :Text;
  supersedesOutcomeId @9 :Text;  # empty = none
  createdAt          @10 :Int64;
  schemaFingerprint  @11 :Text;
  modelId            @12 :Text;
  encodingId         @13 :Text;
  quantization       @14 :Text;
  computeProfile     @15 :Text;
  calibVersion       @16 :Text;  # empty = raw decision row
  decisionCreatedAt  @17 :Int64;
}

struct DecisionOutcomeQueryResult {
  rows @0 :List(DecisionOutcomeJoinRow);
  hasMore @1 :Bool;
  nextCreatedAt @2 :Int64;     # zero when hasMore = false
  nextOutcomeId @3 :Text;     # empty when hasMore = false
}
