@0xb4e2f8d3a9c1e705;

# Cap'n Proto schema for MetricsService
#
# Wraps hyprstream-metrics (DuckDB/DataFusion) for ingest, structured queries,
# streaming query results, view management, and health checks over ZMQ REQ/REP.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;
using import "/streaming.capnp".StreamInfo;

# Mirrors hyprstream_metrics::metrics::MetricRecord.
# Fields reflect running-window aggregation storage — no labels.
struct MetricRecord {
  metricId          @0 :Text;
  timestamp         @1 :Int64;
  valueWindowSum    @2 :Float64;   # value_running_window_sum
  valueWindowAvg    @3 :Float64;   # value_running_window_avg
  valueWindowCount  @4 :Int64;     # value_running_window_count
}

enum AggregationFunc {
  rawSql @0;   # No aggregation — use sql field or unmodified SELECT
  count  @1;
  sum    @2;
  avg    @3;
  min    @4;
  max    @5;
}

struct MetricQuery {
  sql             @0 :Text;          # Raw SQL (takes priority when non-empty)
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
      $mcpScope(write) $mcpDescription("Ingest metric records into the time-series store");

    query        @2 :MetricQuery
      $mcpScope(query) $mcpDescription("Execute a structured or raw SQL aggregation query");

    queryStream  @3 :MetricQuery
      $mcpScope(query) $mcpDescription("Stream query results as Arrow IPC RecordBatch chunks");

    createView   @4 :ViewSpec
      $mcpScope(manage) $mcpDescription("Create a materialized view over the metrics table");

    listViews    @5 :Void
      $mcpScope(query) $mcpDescription("List all materialized views");

    dropView     @6 :Text
      $mcpScope(manage) $mcpDescription("Drop a materialized view by name");

    health       @7 :Void
      $mcpScope(query) $mcpDescription("Check service health and row count");
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
  }
}
