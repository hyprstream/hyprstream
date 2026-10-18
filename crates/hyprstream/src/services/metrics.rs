//! MetricsService — ZMQ RPC wrapper over hyprstream-metrics.
//!
//! Exposes DuckDB-backed time-series ingest, structured/raw SQL queries,
//! streaming Arrow IPC results, view management, and health checks over
//! the standard Hyprstream ZMQ REQ/REP + Cap'n Proto pattern.

use std::sync::Arc;
use std::time::Duration;
use std::time::UNIX_EPOCH;

use anyhow::Result;
use async_trait::async_trait;
use hyprstream_metrics::aggregation::{AggregateFunction, GroupBy, TimeWindow};
use hyprstream_metrics::metrics::decision_outcome::{
    CalibExpectation, CalibFilter, DecisionPins, DecisionRowStore, ExpectedPins, NewOutcome,
    OutcomeStore, TrustedTenant,
};
use hyprstream_metrics::metrics::{create_record_batch, get_metrics_schema, MetricRecord};
use hyprstream_metrics::query::QueryOrchestrator;
use hyprstream_metrics::storage::view::{AggregationSpec, ViewDefinition};
use hyprstream_metrics::StorageBackend;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::streaming::StreamChannel;
use hyprstream_rpc::transport::TransportConfig;
use tracing::{error, trace};

use crate::services::generated::metrics_client::{
    dispatch_metrics, serialize_response, MetricsHandler,
};
use crate::services::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc_std::metrics_client::{
    AggregateRow, AggregationFunc, DecisionOutcomeJoinRow, DecisionOutcomeQuery,
    DecisionOutcomeQueryResult, DecisionOutcomeRequest, DecisionOutcomeResult,
    DecisionRecordRequest, DecisionRecordResult, ErrorInfo, ExpectedCalib, GroupEntry, HealthInfo,
    IngestRequest, MetricQuery, MetricsResponseVariant, StreamInfo, ViewInfo, ViewSpec,
};
use hyprstream_rpc_std::policy_client::PolicyCheck;
use hyprstream_rpc_std::policy_client::PolicyClient;

// ============================================================================
// MetricsInner
// ============================================================================

struct MetricsInner {
    orchestrator: Arc<QueryOrchestrator>,
    stream_channel: StreamChannel,
    // stream_endpoint is resolved lazily from EndpointRegistry on first query_stream call.
    // This avoids requiring the registry to be initialized at service construction time.
}

/// Resources of the R6 decision-store surface. These authorize in the
/// VERIFIED tenant domain (`ctx.domain()`); legacy metrics methods keep the
/// global domain. Method-specific legacy grants (`metrics:Query`,
/// `metrics:Ingest`) do not implicitly authorize the new methods — resource
/// scopes separate the surfaces. Explicit administrator wildcard grants
/// (Casbin `p.dom == "*"` on a decision resource) remain global authority by
/// design and are never part of a tenant-only role.
fn is_decision_store_resource(resource: &str) -> bool {
    matches!(
        resource,
        "metrics:RecordDecision" | "metrics:RecordOutcome" | "metrics:QueryOutcomes"
    )
}

// ============================================================================
// MetricsService
// ============================================================================

/// ZMQ RPC service wrapping hyprstream-metrics.
pub struct MetricsService {
    inner: Arc<MetricsInner>,
    // R6 decision stores live OUTSIDE the shared inner so the producer-seam
    // builder can attach them without interior mutability on the Arc.
    decision_rows: Option<Arc<DecisionRowStore<dyn StorageBackend>>>,
    outcomes: Option<Arc<OutcomeStore<dyn StorageBackend>>>,
    policy_client: PolicyClient,
    transport: TransportConfig,
    signing_key: SigningKey,
    expected_audience: Option<String>,
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
}

impl MetricsService {
    pub fn new(
        orchestrator: Arc<QueryOrchestrator>,
        transport: TransportConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
    ) -> Self {
        let stream_channel = StreamChannel::new(signing_key.clone())
            .with_reach_config(hyprstream_rpc::moq_stream::ProducerReachConfig::default());
        Self {
            inner: Arc::new(MetricsInner {
                orchestrator,
                stream_channel,
            }),
            decision_rows: None,
            outcomes: None,
            policy_client,
            transport,
            signing_key,
            expected_audience: None,
            jwt_key_source: None,
        }
    }

    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        self.jwt_key_source = Some(src);
        self
    }

    /// Attach the R6 durable decision/outcome stores over the metrics
    /// backend. Call [`decision_outcome::create_decision_outcome_tables`] on
    /// the same backend before/at startup; the legacy global metrics RPCs are
    /// unaffected when this builder is not used.
    pub fn with_decision_stores(mut self, backend: Arc<dyn StorageBackend>) -> Self {
        self.decision_rows = Some(Arc::new(DecisionRowStore::new(Arc::clone(&backend))));
        self.outcomes = Some(Arc::new(OutcomeStore::new(backend)));
        self
    }

    fn decision_stores(
        &self,
    ) -> Result<(
        &DecisionRowStore<dyn StorageBackend>,
        &OutcomeStore<dyn StorageBackend>,
    )> {
        match (&self.decision_rows, &self.outcomes) {
            (Some(rows), Some(outcomes)) => Ok((rows, outcomes)),
            _ => anyhow::bail!("decision store not wired on this metrics service"),
        }
    }
}

// ============================================================================
// SQL helpers
// ============================================================================

/// Validate a column/table identifier: only `[a-zA-Z_][a-zA-Z0-9_]*`.
/// Returns an error if the name contains characters that could enable SQL injection.
fn validate_identifier(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("identifier must not be empty");
    }
    let valid = name.chars().enumerate().all(|(i, c)| {
        if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || c == '_'
        }
    });
    if !valid {
        anyhow::bail!(
            "invalid identifier {:?}: only [a-zA-Z_][a-zA-Z0-9_]* allowed",
            name
        );
    }
    Ok(())
}

fn build_sql(q: &MetricQuery) -> Result<String> {
    if !q.sql.is_empty() {
        anyhow::bail!(
            "raw SQL is not supported for metrics query RPCs; use a structured metrics query"
        );
    }

    // Validate group_by column names before interpolating them into SQL.
    for col in &q.group_by {
        validate_identifier(col)?;
    }

    // All aggregates are cast to DOUBLE so `column_by_name("value")` always
    // downcasts to Float64Array regardless of the underlying aggregate type.
    let agg = match q.aggregation {
        AggregationFunc::RawSql | AggregationFunc::Count => "CAST(COUNT(*) AS DOUBLE)",
        AggregationFunc::Sum => "SUM(value_running_window_sum)",
        AggregationFunc::Avg => "AVG(value_running_window_avg)",
        AggregationFunc::Min => "MIN(value_running_window_avg)",
        AggregationFunc::Max => "MAX(value_running_window_avg)",
    };

    let group_select = if q.group_by.is_empty() {
        String::new()
    } else {
        format!(", {}", q.group_by.join(", "))
    };

    // For aggregate queries: SELECT value + group_by columns only.
    // timestamp is NOT included — it is not in GROUP BY and can't appear in SELECT with aggregates.
    // AggregateRow.timestamp will be 0 for structured aggregate queries.
    // NOTE: timestamp is stored and compared in milliseconds (Unix epoch ms).
    // The MetricRecord.timestamp field, ingest path, and window filter all use ms.
    let mut sql = format!("SELECT {agg} AS value{group_select} FROM metrics");

    if !q.metric_id.is_empty() {
        sql.push_str(&format!(
            " WHERE metric_id = '{}'",
            q.metric_id.replace('\'', "''")
        ));
    }

    if q.window_secs > 0 {
        let clause = if q.metric_id.is_empty() {
            "WHERE"
        } else {
            "AND"
        };
        // timestamp is stored as milliseconds; window_secs * 1000 for ms comparison
        sql.push_str(&format!(
            " {clause} timestamp >= epoch_ms(now()) - ({} * 1000)",
            q.window_secs
        ));
    }

    if !q.group_by.is_empty() {
        sql.push_str(&format!(" GROUP BY {}", q.group_by.join(", ")));
    }

    if q.limit_rows > 0 {
        sql.push_str(&format!(" LIMIT {}", q.limit_rows));
    }

    Ok(sql)
}

// ============================================================================
// R6 wire conversion helpers
// ============================================================================

/// Maximum decision payload accepted over the wire (the store's own bound;
/// enforced BEFORE any Arrow parse).
const MAX_PAYLOAD_BYTES: usize = hyprstream_metrics::metrics::decision_outcome::MAX_PAYLOAD_BYTES;

fn non_empty_pin(value: &str, field: &str) -> Result<String> {
    if value.is_empty() {
        anyhow::bail!("{field} must not be empty");
    }
    Ok(value.to_owned())
}

fn optional_pin(value: &str, _field: &str) -> Result<Option<String>> {
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value.to_owned()))
    }
}

/// Convert a `join_outcomes` result batch into wire rows (documented column
/// contract of `DecisionRowStore::join_outcomes`).
fn join_rows(
    batch: &hyprstream_metrics::arrow::array::RecordBatch,
) -> Result<Vec<DecisionOutcomeJoinRow>> {
    use hyprstream_metrics::arrow::array::{Array, Float64Array, Int64Array, StringArray};
    // Empty results may carry no columns at all (DuckDB empty-batch path);
    // zero rows means zero join output regardless of schema shape.
    if batch.num_rows() == 0 {
        return Ok(Vec::new());
    }
    let str_col = |name: &str| -> Result<&StringArray> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow::anyhow!("join result: missing column {name}"))
    };
    let outcome_id = str_col("outcome_id")?;
    let tenant_id = str_col("tenant_id")?;
    let decision_id = str_col("decision_id")?;
    let decision_row_id = str_col("decision_row_id")?;
    let question_id = str_col("question_id")?;
    let outcome_label = str_col("outcome_label")?;
    let outcome_source = str_col("outcome_source")?;
    let supersedes = str_col("supersedes_outcome_id")?;
    let calib = str_col("calib_version")?;
    let values = batch
        .column_by_name("outcome_value")
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| anyhow::anyhow!("join result: missing column outcome_value"))?;
    let created_at = batch
        .column_by_name("created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| anyhow::anyhow!("join result: missing column created_at"))?;
    let decision_created_at = batch
        .column_by_name("decision_created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| anyhow::anyhow!("join result: missing column decision_created_at"))?;

    let mut rows = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        rows.push(DecisionOutcomeJoinRow {
            outcome_id: outcome_id.value(i).to_owned(),
            tenant_id: tenant_id.value(i).to_owned(),
            decision_id: decision_id.value(i).to_owned(),
            decision_row_id: decision_row_id.value(i).to_owned(),
            question_id: question_id.value(i).to_owned(),
            outcome_label: if outcome_label.is_null(i) {
                String::new()
            } else {
                outcome_label.value(i).to_owned()
            },
            outcome_value: if values.is_null(i) {
                0.0
            } else {
                values.value(i)
            },
            has_outcome_value: !values.is_null(i),
            outcome_source: outcome_source.value(i).to_owned(),
            supersedes_outcome_id: if supersedes.is_null(i) {
                String::new()
            } else {
                supersedes.value(i).to_owned()
            },
            created_at: created_at.value(i),
            schema_fingerprint: batch
                .column_by_name("schema_fingerprint")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| anyhow::anyhow!("join result: missing schema_fingerprint"))?
                .value(i)
                .to_owned(),
            model_id: str_col("model_id")?.value(i).to_owned(),
            encoding_id: str_col("encoding_id")?.value(i).to_owned(),
            quantization: str_col("quantization")?.value(i).to_owned(),
            compute_profile: str_col("compute_profile")?.value(i).to_owned(),
            calib_version: if calib.is_null(i) {
                String::new()
            } else {
                calib.value(i).to_owned()
            },
            decision_created_at: decision_created_at.value(i),
        });
    }
    Ok(rows)
}

// ============================================================================
// MetricsHandler (generated dispatch trait)
// ============================================================================

#[async_trait(?Send)]
impl MetricsHandler for MetricsService {
    async fn authorize(
        &self,
        ctx: &EnvelopeContext,
        resource: &str,
        operation: &str,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        // R6 decision-store methods authorize in the VERIFIED tenant domain
        // derived from the envelope; requests without a verified tenant fail
        // closed here. Legacy metrics methods stay node-global: the
        // MetricRecord tenant column is a data label, and authorization
        // identity still comes only from the verified envelope. Explicit
        // administrator wildcard grants (domain "*" on a decision resource)
        // keep their global-authority semantics; legacy method-specific
        // grants never implicitly cover the new decision resources.
        let domain = if is_decision_store_resource(resource) {
            ctx.domain()?
        } else {
            "*".to_owned()
        };
        let request = PolicyCheck {
            subject: subject.clone(),
            domain,
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        let allowed = crate::services::policy::check_with_holder_evidence(
            &self.policy_client,
            &request,
            ctx.jwt_token(),
            &ctx.subject(),
            ctx.original_holder_evidence(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Policy check error for {}: {}", subject, e))?;
        if allowed {
            Ok(())
        } else {
            anyhow::bail!(
                "Unauthorized: {} cannot {} on {}",
                subject,
                operation,
                resource
            )
        }
    }

    async fn handle_ingest(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &IngestRequest,
    ) -> Result<MetricsResponseVariant> {
        // Validate before building the batch so callers get a clear rejection.
        for (i, r) in data.records.iter().enumerate() {
            if r.metric_id.is_empty() {
                anyhow::bail!("record[{i}]: metric_id must not be empty");
            }
        }

        let rust_records: Vec<MetricRecord> = data
            .records
            .iter()
            .map(|r| MetricRecord {
                metric_id: r.metric_id.clone(),
                timestamp: r.timestamp,
                value_running_window_sum: r.value_window_sum,
                value_running_window_avg: r.value_window_avg,
                value_running_window_count: r.value_window_count,
                labels: r
                    .labels
                    .iter()
                    .map(|e| (e.key.clone(), e.value.clone()))
                    .collect(),
                tenant_id: if r.tenant_id.is_empty() {
                    None
                } else {
                    Some(r.tenant_id.clone())
                },
            })
            .collect();

        let batch = create_record_batch(&rust_records).map_err(|e| anyhow::anyhow!("{e}"))?;

        self.inner
            .orchestrator
            .storage()
            .insert_into_table("metrics", batch)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(MetricsResponseVariant::IngestResult)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // R6 durable decision/outcome handlers. Tenant comes ONLY from the
    // verified envelope domain; the wire never carries tenant authority.
    // ═══════════════════════════════════════════════════════════════════════

    async fn handle_record_decision(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DecisionRecordRequest,
    ) -> Result<MetricsResponseVariant> {
        let tenant = TrustedTenant::new(ctx.domain()?)?;
        let (rows, _) = self.decision_stores()?;
        // Bound the payload BEFORE any Arrow parse (bounded decode, R6E).
        anyhow::ensure!(
            data.payload.len() <= MAX_PAYLOAD_BYTES,
            "decision payload exceeds {MAX_PAYLOAD_BYTES} bytes"
        );
        let pins = DecisionPins {
            schema_fingerprint: non_empty_pin(&data.schema_fingerprint, "schemaFingerprint")?,
            model_id: non_empty_pin(&data.model_id, "modelId")?,
            encoding_id: non_empty_pin(&data.encoding_id, "encodingId")?,
            quantization: non_empty_pin(&data.quantization, "quantization")?,
            compute_profile: non_empty_pin(&data.compute_profile, "computeProfile")?,
            calib_version: optional_pin(&data.calib_version, "calibVersion")?,
        };
        // Bounded decode: framing + declared (decompressed) buffer sizes are
        // verified within MAX_PAYLOAD_BYTES BEFORE the decoder allocates;
        // valid compression is preserved.
        let batch = hyprstream_metrics::metrics::decision_batch::decode_batch_ipc_bounded(
            &data.payload,
            MAX_PAYLOAD_BYTES,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let stored = rows
            .put(&tenant, &pins, &batch)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(MetricsResponseVariant::RecordDecisionResult(
            DecisionRecordResult {
                decision_id: stored.decision_id,
                row_ids: stored.row_ids,
            },
        ))
    }

    async fn handle_record_outcome(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DecisionOutcomeRequest,
    ) -> Result<MetricsResponseVariant> {
        let tenant = TrustedTenant::new(ctx.domain()?)?;
        let (_, outcomes) = self.decision_stores()?;
        let expected_pins = if data.expected_schema_fingerprint.is_empty()
            && data.expected_model_id.is_empty()
            && data.expected_encoding_id.is_empty()
            && data.expected_quantization.is_empty()
            && data.expected_compute_profile.is_empty()
        {
            None
        } else {
            Some(ExpectedPins {
                schema_fingerprint: optional_pin(
                    &data.expected_schema_fingerprint,
                    "expectedSchemaFingerprint",
                )?,
                model_id: optional_pin(&data.expected_model_id, "expectedModelId")?,
                encoding_id: optional_pin(&data.expected_encoding_id, "expectedEncodingId")?,
                quantization: optional_pin(&data.expected_quantization, "expectedQuantization")?,
                compute_profile: optional_pin(
                    &data.expected_compute_profile,
                    "expectedComputeProfile",
                )?,
            })
        };
        let expected_calib = match data.expected_calib {
            ExpectedCalib::Any => CalibExpectation::Any,
            ExpectedCalib::Raw => CalibExpectation::Raw,
            ExpectedCalib::Version => CalibExpectation::Version(non_empty_pin(
                &data.expected_calib_version,
                "expectedCalibVersion",
            )?),
        };
        let outcome = NewOutcome {
            decision_id: non_empty_pin(&data.decision_id, "decisionId")?,
            decision_row_id: non_empty_pin(&data.decision_row_id, "decisionRowId")?,
            question_id: non_empty_pin(&data.question_id, "questionId")?,
            expected_pins,
            expected_calib,
            outcome_label: optional_pin(&data.outcome_label, "outcomeLabel")?,
            outcome_value: if data.has_outcome_value {
                Some(data.outcome_value)
            } else {
                None
            },
            outcome_source: non_empty_pin(&data.outcome_source, "outcomeSource")?,
            supersedes_outcome_id: optional_pin(
                &data.supersedes_outcome_id,
                "supersedesOutcomeId",
            )?,
        };
        let record = outcomes
            .record(&tenant, &outcome)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(MetricsResponseVariant::RecordOutcomeResult(
            DecisionOutcomeResult {
                outcome_id: record.outcome_id,
            },
        ))
    }

    async fn handle_query_outcomes(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DecisionOutcomeQuery,
    ) -> Result<MetricsResponseVariant> {
        let tenant = TrustedTenant::new(ctx.domain()?)?;
        let (_, outcomes) = self.decision_stores()?;
        let calib = match data.calib_filter {
            ExpectedCalib::Any => CalibFilter::All,
            ExpectedCalib::Raw => CalibFilter::RawOnly,
            ExpectedCalib::Version => {
                CalibFilter::Version(non_empty_pin(&data.calib_version, "calibVersion")?)
            }
        };
        let query = hyprstream_metrics::metrics::decision_outcome::DecisionOutcomeQuery {
            schema_fingerprint: optional_pin(&data.schema_fingerprint, "schemaFingerprint")?,
            model_id: optional_pin(&data.model_id, "modelId")?,
            question_id: optional_pin(&data.question_id, "questionId")?,
            calib,
            include_superseded: data.include_superseded,
            limit: data.limit,
        };
        let cursor = hyprstream_metrics::metrics::decision_outcome::OutcomeCursor::from_wire(
            data.after_created_at,
            &data.after_outcome_id,
        )?;
        let page = outcomes
            .join_outcomes_page(&tenant, &query, cursor.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let rows = join_rows(&page.rows)?;
        let (has_more, next_created_at, next_outcome_id) = match page.next_cursor {
            Some(cursor) => (true, cursor.created_at(), cursor.outcome_id().to_owned()),
            None => (false, 0, String::new()),
        };
        Ok(MetricsResponseVariant::QueryOutcomesResult(
            DecisionOutcomeQueryResult {
                rows,
                has_more,
                next_created_at,
                next_outcome_id,
            },
        ))
    }

    async fn handle_query(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        q: &MetricQuery,
    ) -> Result<MetricsResponseVariant> {
        let sql = build_sql(q)?;
        // Use DuckDB directly — DataFusionPlanner registers tables as empty MemTables
        // so orchestrator.query_collect() always returns no rows for non-empty tables.
        let storage = self.inner.orchestrator.storage();
        let handle = storage
            .prepare_sql(&sql)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let batch = storage
            .query_sql(&handle)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let batches = [batch];

        let mut rows: Vec<AggregateRow> = Vec::new();
        for batch in &batches {
            let n = batch.num_rows();
            let value_col = batch.column_by_name("value").and_then(|c| {
                c.as_any()
                    .downcast_ref::<hyprstream_metrics::arrow::array::Float64Array>()
            });
            let ts_col = batch.column_by_name("timestamp").and_then(|c| {
                c.as_any()
                    .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
            });

            for i in 0..n {
                let value = value_col.map(|c| c.value(i)).unwrap_or(0.0);
                let timestamp = ts_col.map(|c| c.value(i)).unwrap_or(0);

                let mut group_values: Vec<GroupEntry> = Vec::new();
                for col_name in &q.group_by {
                    let col = batch.column_by_name(col_name.as_str()).ok_or_else(|| {
                        anyhow::anyhow!("group_by column {:?} not found in result", col_name)
                    })?;
                    let val =
                        col.as_any()
                            .downcast_ref::<hyprstream_metrics::arrow::array::StringArray>()
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                            "group_by column {:?} has type {:?}; only Utf8/String columns are \
                             supported for group_by (metrics schema: metric_id is Utf8; \
                             timestamp and value_running_window_count are Int64 — not groupable)",
                            col_name, col.data_type()
                        )
                            })?
                            .value(i)
                            .to_owned();
                    group_values.push(GroupEntry {
                        key: col_name.clone(),
                        value: val,
                    });
                }

                rows.push(AggregateRow {
                    value,
                    group_values,
                    timestamp,
                });
            }
        }

        Ok(MetricsResponseVariant::QueryResult(rows))
    }

    async fn handle_query_stream(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        q: &MetricQuery,
    ) -> Result<(StreamInfo, Continuation)> {
        if q.ephemeral_pubkey.is_empty() {
            anyhow::bail!("queryStream requires a 32-byte ephemeral_pubkey (Ristretto255)");
        }
        let sql = build_sql(q)?;

        let stream_ctx = self
            .inner
            .stream_channel
            .prepare_third_party_interop_stream(&q.ephemeral_pubkey, 600)
            .await?;

        let broadcast_path = hyprstream_rpc::moq_stream::global_moq_origin()
            .map(|o| o.broadcast_path(stream_ctx.topic()))
            .unwrap_or_default();
        let stream_info = StreamInfo {
            stream_id: stream_ctx.stream_id().to_owned(),
            dh_public: *stream_ctx.server_pubkey(),
            broadcast_path,
            announced_at: stream_ctx.reach(), // #384: per-stream reach via ctx
            moql_server_identity: stream_ctx.moql_server_identity(),
            ..Default::default()
        };
        let inner = Arc::clone(&self.inner);

        let continuation: Continuation = Box::pin(async move {
            // Clone Arc again so inner is not borrowed through run_stream's receiver
            let inner2 = Arc::clone(&inner);
            if let Err(e) = inner
                .stream_channel
                .run_stream(&stream_ctx, move |mut publisher| {
                    let inner = Arc::clone(&inner2);
                    let sql = sql.clone();
                    async move {
                        let result = async {
                            tokio::time::timeout(
                                std::time::Duration::from_secs(30),
                                publisher.wait_for_consumer(),
                            )
                            .await
                            .map_err(|_| {
                                anyhow::anyhow!("queryStream consumer demand timeout")
                            })??;
                            // Use DuckDB directly — DataFusionPlanner registers tables
                            // as empty MemTables, so orchestrator.query() returns no rows.
                            let storage = inner.orchestrator.storage();
                            let handle = storage
                                .prepare_sql(&sql)
                                .await
                                .map_err(|e| anyhow::anyhow!("{e}"))?;
                            let batch = storage
                                .query_sql(&handle)
                                .await
                                .map_err(|e| anyhow::anyhow!("{e}"))?;

                            // Emit a valid Arrow IPC stream (schema + rows + EOS).
                            // Always use the actual query result schema — aggregate queries
                            // produce a different schema than the raw metrics table schema.
                            // Zero-row results still emit a valid schema + EOS header.
                            let result_schema = batch.schema();
                            let mut buf = Vec::new();
                            let mut writer =
                                hyprstream_metrics::arrow::ipc::writer::StreamWriter::try_new(
                                    &mut buf,
                                    &result_schema,
                                )
                                .map_err(|e| anyhow::anyhow!("IPC writer: {e}"))?;

                            if batch.num_rows() > 0 {
                                writer
                                    .write(&batch)
                                    .map_err(|e| anyhow::anyhow!("IPC write: {e}"))?;
                            }

                            writer
                                .finish()
                                .map_err(|e| anyhow::anyhow!("IPC finish: {e}"))?;
                            drop(writer);
                            publisher
                                .publish_data(&buf)
                                .await
                                .map_err(|e| anyhow::anyhow!("publish_data: {e}"))?;

                            Ok::<(), anyhow::Error>(())
                        }
                        .await;

                        (publisher, result)
                    }
                })
                .await
            {
                error!("metrics query_stream error: {e}");
            }
        });

        Ok((stream_info, continuation))
    }

    async fn handle_create_view(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        spec: &ViewSpec,
    ) -> Result<MetricsResponseVariant> {
        // View name is interpolated into CREATE VIEW {name} AS ... without quoting.
        validate_identifier(&spec.name)?;
        let q = &spec.query;

        let agg_fn = match q.aggregation {
            AggregationFunc::RawSql => None,
            AggregationFunc::Count => Some(AggregateFunction::Count),
            AggregationFunc::Sum => Some(AggregateFunction::Sum),
            AggregationFunc::Avg => Some(AggregateFunction::Avg),
            AggregationFunc::Min => Some(AggregateFunction::Min),
            AggregationFunc::Max => Some(AggregateFunction::Max),
        };

        let aggregations = match agg_fn {
            None => vec![],
            Some(f) => vec![AggregationSpec {
                column: "value_running_window_sum".to_owned(),
                function: f,
            }],
        };

        let group_by = if q.group_by.is_empty() {
            None
        } else {
            Some(GroupBy {
                columns: q.group_by.clone(),
                time_column: None,
            })
        };

        let window = if q.window_secs > 0 {
            Some(TimeWindow::Fixed(Duration::from_secs(q.window_secs as u64)))
        } else {
            None
        };

        // Validate group_by identifiers before building ViewDefinition — interpolated into SQL.
        for col in &q.group_by {
            validate_identifier(col)?;
        }
        // metric_id is a filter value, not a table name. The metrics table is always "metrics".
        let source_table = "metrics".to_owned();

        // raw SQL view creation is not supported — ViewDefinition has no raw-SQL constructor.
        // Reject both: explicit raw SQL string, and RawSql aggregation with no aggregation spec.
        if !q.sql.is_empty() || q.aggregation == AggregationFunc::RawSql {
            anyhow::bail!("create_view with raw SQL or RawSql aggregation is not supported; use a structured aggregation function");
        }

        let schema = Arc::new(get_metrics_schema());

        let def = ViewDefinition::new(
            source_table,
            q.group_by.clone(),
            aggregations,
            group_by,
            window,
            schema,
        );

        self.inner
            .orchestrator
            .storage()
            .create_view(&spec.name, def)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(MetricsResponseVariant::CreateViewResult)
    }

    async fn handle_list_views(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<MetricsResponseVariant> {
        let names = self
            .inner
            .orchestrator
            .storage()
            .list_views()
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut infos: Vec<ViewInfo> = Vec::with_capacity(names.len());
        for name in &names {
            let meta = self
                .inner
                .orchestrator
                .storage()
                .get_view(name)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let created_at = meta
                .created_at
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            infos.push(ViewInfo {
                name: name.clone(),
                sql: meta.definition.to_sql(),
                created_at,
            });
        }

        Ok(MetricsResponseVariant::ListViewsResult(infos))
    }

    async fn handle_drop_view(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        value: &str,
    ) -> Result<MetricsResponseVariant> {
        // View name is interpolated into DROP VIEW IF EXISTS {name} without quoting.
        validate_identifier(value)?;
        self.inner
            .orchestrator
            .storage()
            .drop_view(value)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(MetricsResponseVariant::DropViewResult)
    }

    async fn handle_health(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<MetricsResponseVariant> {
        // Use DuckDB directly — DataFusionPlanner registers tables as empty MemTables.
        let storage = self.inner.orchestrator.storage();
        let result = async {
            let handle = storage
                .prepare_sql("SELECT COUNT(*) AS cnt FROM metrics")
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            storage
                .query_sql(&handle)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
        .await;
        let result = result.map(|b| vec![b]);

        let (ok, row_count) = match result {
            Ok(batches) => {
                // DuckDB may return Int64 or UInt64 for COUNT(*) depending on version/context.
                let col = batches.first().and_then(|b| b.column_by_name("cnt"));
                let cnt = match col {
                    None => {
                        tracing::warn!("health: COUNT query returned no 'cnt' column");
                        0
                    }
                    Some(c) => {
                        let v = c
                            .as_any()
                            .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
                            .map(|a| a.value(0) as u64)
                            .or_else(|| {
                                c.as_any()
                                    .downcast_ref::<hyprstream_metrics::arrow::array::UInt64Array>()
                                    .map(|a| a.value(0))
                            });
                        if v.is_none() {
                            tracing::warn!(
                                dtype = ?c.data_type(),
                                "health: COUNT column has unexpected Arrow type; defaulting to 0"
                            );
                        }
                        v.unwrap_or(0)
                    }
                };
                (true, cnt)
            }
            Err(_) => (false, 0),
        };

        Ok(MetricsResponseVariant::HealthResult(HealthInfo {
            ok,
            row_count,
            backend_type: "duckdb".to_owned(),
        }))
    }
}

// ============================================================================
// RequestService
// ============================================================================

#[async_trait(?Send)]
impl RequestService for MetricsService {
    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> anyhow::Result<hyprstream_rpc::service::DecodedRequestBody> {
        // The ONE bounded decode (v16 §5.2): the generated decoder derives
        // the full method leaf and returns the decoded message that policy,
        // MAC, and dispatch below all consume.
        crate::services::generated::metrics_client::decode_metrics_request_body(signed_body)
    }

    fn producer_reach_config_handle(
        &self,
    ) -> Option<hyprstream_rpc::moq_stream::ProducerReachConfigHandle> {
        Some(self.inner.stream_channel.reach_config_handle())
    }

    fn moq_origin_handle(&self) -> Option<hyprstream_rpc::moq_stream::MoqStreamOriginHandle> {
        Some(self.inner.stream_channel.moq_origin_handle())
    }

    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        trace!(
            "Metrics request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_metrics(self, ctx, body).await
    }

    fn name(&self) -> &str {
        "metrics"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = MetricsResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use hyprstream_metrics::arrow::array::RecordBatch;
    use std::sync::Arc;

    use hyprstream_metrics::query::QueryOrchestrator;
    use hyprstream_metrics::storage::duckdb::DuckDbBackend;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::moq_stream::{init_global_moq_origin, MoqStreamOrigin, STREAM_TRACK};
    use hyprstream_rpc::stream_consumer::StreamPayload;
    use hyprstream_rpc::streaming::{derive_client_stream_keys, StreamVerifier};
    use hyprstream_rpc::transport::TransportConfig;
    use hyprstream_service::{InprocManager, ServiceManager};
    use moq_net::Track;

    use crate::auth::PolicyManager;
    use crate::services::PolicyService;
    use hyprstream_rpc_std::metrics_client::{
        AggregationFunc, IngestRequest, MetricQuery, MetricRecord as CMetricRecord, MetricsClient,
        ViewSpec,
    };
    use hyprstream_rpc_std::policy_client::PolicyClient;
    /// Spin up an in-memory MetricsService and return its typed client, manager, and backend.
    async fn start_metrics_service(
        tag: &str,
    ) -> (MetricsClient, InprocManager, Arc<DuckDbBackend>) {
        start_metrics_service_on_path(tag, ":memory:".to_owned(), true).await
    }

    async fn start_metrics_service_on_path(
        tag: &str,
        connection_string: String,
        query_allowed: bool,
    ) -> (MetricsClient, InprocManager, Arc<DuckDbBackend>) {
        let (signing_key, _vk) = generate_signing_keypair();
        let policy_manager = if query_allowed {
            PolicyManager::permissive()
                .await
                .expect("permissive policy manager")
        } else {
            let manager = PolicyManager::new_in_memory()
                .await
                .expect("deny-by-default policy manager");
            // Permit the metrics service to ask PolicyService, while leaving
            // metrics:Query and metrics:QueryStream themselves ungranted.
            manager
                .add_policy("service:metrics", "policy:Check", "check")
                .await
                .expect("policy check grant");
            manager
        };
        start_metrics_service_with_access(
            tag,
            connection_string,
            policy_manager,
            signing_key.clone(),
            signing_key.clone(),
            signing_key,
            "service:metrics",
            "service:metrics",
            "service:metrics",
        )
        .await
    }

    /// Build a local transport fixture with three independently keyed roles:
    /// Policy, Metrics, and a service-mediated query caller. PolicyService
    /// evaluates the verified Metrics mediator identity, so this fixture
    /// documents mediated authorization rather than caller-bound authorization.
    async fn start_metrics_service_with_access(
        tag: &str,
        connection_string: String,
        policy_manager: PolicyManager,
        policy_signing_key: SigningKey,
        metrics_signing_key: SigningKey,
        client_signing_key: SigningKey,
        policy_subject: &str,
        metrics_subject: &str,
        client_subject: &str,
    ) -> (MetricsClient, InprocManager, Arc<DuckDbBackend>) {
        let _ = init_global_moq_origin(MoqStreamOrigin::standalone().build());
        crate::mac::install_explicit_test_dispatch_pep();
        // Tests use Classical (EdDSA-only) keys — install Classical verify
        // policy on both request and response arms so the global
        // fail-closed Hybrid defaults don't reject them.
        let _ = hyprstream_rpc::envelope::install_verify_config(
            hyprstream_rpc::envelope::EnvelopeVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        let _ = hyprstream_rpc::envelope::install_response_verify_config(
            hyprstream_rpc::envelope::ResponseVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );

        for (key, subject) in [
            (&policy_signing_key, policy_subject),
            (&metrics_signing_key, metrics_subject),
            (&client_signing_key, client_subject),
        ] {
            hyprstream_service::global_trust_store().insert(
                key.verifying_key(),
                hyprstream_service::Attestation {
                    scopes: std::collections::HashSet::new(),
                    subject: Some(subject.to_owned()),
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                },
            );
        }

        let policy_tag = format!("test-policy-{tag}");
        let policy_manager = Arc::new(policy_manager);
        let git2db = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(tempfile::TempDir::new().unwrap().path())
                .await
                .expect("git2db open"),
        ));
        let policy_svc = PolicyService::new(
            policy_manager,
            Arc::new(policy_signing_key.clone()),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc(&policy_tag),
        );
        let manager = InprocManager::new();
        manager
            .spawn(Box::new(policy_svc))
            .await
            .expect("spawn policy service");

        // The Metrics service authenticates to Policy as its own service
        // identity; the query caller is a separate signer on the Metrics RPC.
        let policy_client = PolicyClient::for_local_endpoint_bootstrap(
            &format!("inproc://{policy_tag}"),
            metrics_signing_key.clone(),
            policy_signing_key.verifying_key(),
            None,
        )
        .expect("create policy client");

        let backend = Arc::new(
            DuckDbBackend::new(connection_string, Default::default(), None).expect("DuckDbBackend"),
        );
        let backend_for_tests = Arc::clone(&backend);
        let schema = hyprstream_metrics::metrics::get_metrics_schema();
        backend
            .create_table("metrics", &schema)
            .await
            .expect("create metrics table");
        let orchestrator = Arc::new(
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(QueryOrchestrator::new(
                    backend as Arc<dyn hyprstream_metrics::StorageBackend>,
                ))
            })
            .expect("QueryOrchestrator"),
        );

        let svc_tag = format!("test-metrics-{tag}");
        let service = MetricsService::new(
            orchestrator,
            TransportConfig::inproc(&svc_tag),
            metrics_signing_key.clone(),
            policy_client,
        );
        manager
            .spawn(Box::new(service))
            .await
            .expect("spawn metrics service");

        let client = MetricsClient::for_local_endpoint_bootstrap(
            &format!("inproc://{svc_tag}"),
            client_signing_key,
            metrics_signing_key.verifying_key(),
            None,
        )
        .expect("create metrics client");

        (client, manager, backend_for_tests)
    }

    async fn metrics_row_count(backend: &DuckDbBackend) -> i64 {
        let handle = backend
            .prepare_sql("SELECT COUNT(*) AS rows FROM metrics")
            .await
            .expect("prepare row count");
        let batch = backend.query_sql(&handle).await.expect("query row count");
        batch
            .column(0)
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
            .expect("row count type")
            .value(0)
    }

    async fn metrics_value_sum(backend: &DuckDbBackend) -> f64 {
        let handle = backend
            .prepare_sql("SELECT COALESCE(SUM(value_running_window_sum), 0) AS total FROM metrics")
            .await
            .expect("prepare value sum");
        let batch = backend.query_sql(&handle).await.expect("query value sum");
        batch
            .column(0)
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Float64Array>()
            .expect("value sum type")
            .value(0)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mediated_query_profile_has_structured_read_access_only() {
        let (policy_signing_key, _) = generate_signing_keypair();
        let (metrics_signing_key, _) = generate_signing_keypair();
        let (caller_signing_key, _) = generate_signing_keypair();
        let policy_manager = PolicyManager::new_in_memory()
            .await
            .expect("query-only policy manager");
        policy_manager
            .add_policy("service:metrics", "policy:Check", "check")
            .await
            .expect("policy check grant");
        // PolicyService evaluates the verified mediating Metrics identity. The
        // RPC caller below is a distinct signer; caller-bound grants are
        // intentionally unrepresented by the current API boundary.
        policy_manager
            .add_policy("service:metrics", "metrics:Query", "query")
            .await
            .expect("structured query grant");
        policy_manager
            .add_policy("service:metrics", "metrics:QueryStream", "query")
            .await
            .expect("structured query stream grant");

        let db_dir = tempfile::tempdir().expect("query-only database directory");
        let db_path = db_dir.path().join("metrics.duckdb");
        let (client, manager, backend) = start_metrics_service_with_access(
            "query-only-distinct-identities",
            db_path.to_string_lossy().into_owned(),
            policy_manager,
            policy_signing_key,
            metrics_signing_key,
            caller_signing_key,
            "service:policy",
            "service:metrics",
            "service:metrics-query",
        )
        .await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let seed = create_record_batch(&[MetricRecord {
            metric_id: "query-only.metric".to_owned(),
            timestamp: now,
            value_running_window_sum: 11.0,
            value_running_window_avg: 11.0,
            value_running_window_count: 1,
            ..Default::default()
        }])
        .expect("seed record batch");
        backend
            .insert_into_table("metrics", seed)
            .await
            .expect("seed backend directly");
        let before_tables = backend.list_tables().await.expect("list tables before");
        let before_rows = metrics_row_count(&backend).await;
        let before_sum = metrics_value_sum(&backend).await;
        assert_eq!(before_rows, 1);
        assert_eq!(before_sum, 11.0);

        let query = MetricQuery {
            sql: String::new(),
            metric_id: "query-only.metric".to_owned(),
            window_secs: 0,
            aggregation: AggregationFunc::Sum,
            group_by: vec![],
            limit_rows: 0,
            ephemeral_pubkey: vec![],
        };
        let rows = client.query(&query).await.expect("structured query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value, 11.0, "structured SUM should be 11.0");

        let (client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let stream = client
            .query_stream(
                &MetricQuery {
                    ephemeral_pubkey: client_pub.to_bytes().to_vec(),
                    ..query.clone()
                },
                [0u8; 32],
            )
            .await
            .expect("structured queryStream");
        assert!(!stream.stream_id.is_empty());
        // Exercise the server-side demand handshake instead of relying on the
        // scheduler to attach before a short query can publish and finish.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let (mac_key, enc_key, topic) =
            derive_client_stream_keys(&client_secret, &client_pub.to_bytes(), &stream.dh_public)
                .expect("derive stream keys");
        let origin = hyprstream_rpc::moq_stream::global_moq_origin()
            .expect("metrics test origin initialized");
        let broadcast = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin
                .consumer()
                .announced_broadcast(&stream.broadcast_path),
        )
        .await
        .expect("stream announcement timeout")
        .expect("stream broadcast announcement");
        let mut track = broadcast
            .subscribe_track(&Track::new(STREAM_TRACK))
            .expect("subscribe to query-only stream");
        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let mut payloads = Vec::new();
        for _ in 0..2 {
            let mut group =
                tokio::time::timeout(std::time::Duration::from_secs(5), track.next_group())
                    .await
                    .expect("stream group timeout")
                    .expect("stream group read")
                    .expect("stream ended before terminal frame");
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), group.read_frame())
                .await
                .expect("stream frame timeout")
                .expect("stream frame read")
                .expect("stream group had no frame");
            payloads.extend(
                hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &frame)
                    .expect("verify query-only stream frame"),
            );
        }
        let data = payloads
            .iter()
            .find_map(|payload| match payload {
                StreamPayload::Data(data) => Some(data.as_slice()),
                _ => None,
            })
            .expect("query-only stream data");
        let mut reader = hyprstream_metrics::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(data),
            None,
        )
        .expect("read Arrow query-only stream");
        let batch = reader
            .next()
            .expect("Arrow query-only stream batch")
            .expect("Arrow query-only stream batch decode");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Float64Array>()
            .expect("query-only stream value column");
        assert_eq!(values.value(0), 11.0, "streamed SUM should be 11.0");
        assert!(
            payloads
                .iter()
                .any(|payload| matches!(payload, StreamPayload::Complete(_))),
            "query-only stream must terminate after publishing its result"
        );
        let eof = tokio::time::timeout(std::time::Duration::from_secs(5), track.next_group())
            .await
            .expect("query-only stream EOF timeout")
            .expect("query-only stream should close cleanly");
        assert!(
            eof.is_none(),
            "query-only stream emitted data after Complete"
        );

        let denied_record = IngestRequest {
            records: vec![CMetricRecord {
                metric_id: "must-not-write".to_owned(),
                timestamp: now + 1,
                value_window_sum: 99.0,
                value_window_avg: 99.0,
                value_window_count: 1,
                labels: vec![],
                tenant_id: String::new(),
            }],
        };
        let error = client
            .ingest(&denied_record)
            .await
            .expect_err("mediated query-only profile must not ingest");
        assert!(error.to_string().contains("Unauthorized"));
        let error = client
            .create_view(&ViewSpec {
                name: "must_not_create".to_owned(),
                query: query.clone(),
            })
            .await
            .expect_err("mediated query-only profile must not create views");
        assert!(error.to_string().contains("Unauthorized"));
        let error = client
            .drop_view("must_not_drop")
            .await
            .expect_err("mediated query-only profile must not drop views");
        assert!(error.to_string().contains("Unauthorized"));

        let raw_cases = [
            "CREATE TABLE query_owned (value INTEGER)",
            "CREATE TABLE query_owned_ctas AS SELECT value_running_window_sum FROM metrics",
            "INSERT INTO metrics VALUES ('raw', 1, 2, 2, 1)",
            "SELECT 1; CREATE TABLE query_owned_again (value INTEGER)",
            "WITH rows AS (SELECT 1) SELECT * FROM rows",
            "SELECT * FROM read_csv('/synthetic/not-a-real-file')",
            "-- comment\nSELECT * FROM metrics",
            "   ",
            "SELECT * FROM metrics",
        ];
        for sql in raw_cases {
            let raw_query = MetricQuery {
                sql: sql.to_owned(),
                ..query.clone()
            };
            let error = client
                .query(&raw_query)
                .await
                .expect_err("raw SQL must be denied for mediated query-only profile");
            assert!(error.to_string().contains("raw SQL is not supported"));
            let mut stream_query = raw_query;
            stream_query.ephemeral_pubkey = client_pub.to_bytes().to_vec();
            let error = client
                .query_stream(&stream_query, [0u8; 32])
                .await
                .expect_err("raw SQL must be denied for queryStream");
            assert!(error.to_string().contains("raw SQL is not supported"));
        }

        let after_tables = backend.list_tables().await.expect("list tables after");
        let after_rows = metrics_row_count(&backend).await;
        let after_sum = metrics_value_sum(&backend).await;
        assert_eq!(
            before_tables, after_tables,
            "denials must not change schema"
        );
        assert_eq!(before_rows, after_rows, "denials must not change row count");
        assert_eq!(
            before_sum, after_sum,
            "denials must not change persisted values"
        );

        drop(client);
        drop(manager);
        drop(backend);
        let reopened = DuckDbBackend::new(
            db_path.to_string_lossy().into_owned(),
            Default::default(),
            None,
        )
        .expect("reopen query-only database");
        assert_eq!(
            before_tables,
            reopened.list_tables().await.expect("list reopened tables"),
            "denials must not change persisted schema"
        );
        assert_eq!(
            before_rows,
            metrics_row_count(&reopened).await,
            "denials must not change persisted row count"
        );
        assert_eq!(
            before_sum,
            metrics_value_sum(&reopened).await,
            "denials must not change persisted values"
        );
    }

    // ── health ────────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_health_fresh_db() {
        let (client, _mgr, _) = start_metrics_service("health-fresh").await;
        let info = client.health().await.expect("health on fresh DB");
        // Table exists (created in test helper) but is empty.
        assert!(info.ok, "expected ok=true — table exists even when empty");
        assert_eq!(info.row_count, 0, "expected 0 rows on fresh DB");
        assert_eq!(info.backend_type, "duckdb");
    }

    // ── ingest + health ───────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_ingest_and_health() {
        let (client, _mgr, _) = start_metrics_service("ingest-health").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let req = IngestRequest {
            records: vec![
                CMetricRecord {
                    metric_id: "cpu.usage".to_owned(),
                    timestamp: now,
                    value_window_sum: 42.0,
                    value_window_avg: 42.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                },
                CMetricRecord {
                    metric_id: "cpu.usage".to_owned(),
                    timestamp: now + 1000,
                    value_window_sum: 58.0,
                    value_window_avg: 58.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                },
            ],
        };

        client.ingest(&req).await.expect("ingest");

        let info = client.health().await.expect("health after ingest");
        assert!(info.ok, "expected ok=true after ingest");
        assert_eq!(info.row_count, 2, "expected 2 rows");
    }

    /// Schema v2: labels and tenant_id survive the ingest → storage → query
    /// round trip (group_by works on both new columns).
    #[tokio::test(flavor = "multi_thread")]
    async fn test_ingest_with_labels_and_tenant() {
        let (client, _mgr, _) = start_metrics_service("metrics-labels").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        client
            .ingest(&IngestRequest {
                records: vec![
                    CMetricRecord {
                        metric_id: "p99_latency_ms".to_owned(),
                        timestamp: now,
                        value_window_sum: 120.0,
                        value_window_avg: 120.0,
                        value_window_count: 1,
                        labels: vec![GroupEntry {
                            key: "primitive".to_owned(),
                            value: "choice".to_owned(),
                        }],
                        tenant_id: "tenant-a".to_owned(),
                    },
                    CMetricRecord {
                        metric_id: "p99_latency_ms".to_owned(),
                        timestamp: now + 1000,
                        value_window_sum: 95.0,
                        value_window_avg: 95.0,
                        value_window_count: 1,
                        labels: vec![],
                        tenant_id: "tenant-b".to_owned(),
                    },
                ],
            })
            .await
            .expect("ingest");

        let query = MetricQuery {
            sql: String::new(),
            metric_id: "p99_latency_ms".to_owned(),
            window_secs: 0,
            aggregation: AggregationFunc::Count,
            group_by: vec!["tenant_id".to_owned()],
            limit_rows: 0,
            ephemeral_pubkey: vec![],
        };
        let rows = client.query(&query).await.expect("tenant query");
        assert_eq!(rows.len(), 2, "expected one row per tenant");
        let mut tenants: Vec<String> = rows
            .iter()
            .map(|r| {
                assert_eq!(r.value as u64, 1, "COUNT per tenant should be 1");
                r.group_values
                    .iter()
                    .find(|g| g.key == "tenant_id")
                    .expect("tenant_id group value")
                    .value
                    .clone()
            })
            .collect();
        tenants.sort();
        assert_eq!(tenants, vec!["tenant-a".to_owned(), "tenant-b".to_owned()]);

        // Labels land as canonical JSON; the labeled record forms its own group.
        let rows = client
            .query(&MetricQuery {
                group_by: vec!["labels".to_owned()],
                ..query.clone()
            })
            .await
            .expect("labels query");
        let groups: Vec<String> = rows
            .iter()
            .flat_map(|r| {
                r.group_values
                    .iter()
                    .filter(|g| g.key == "labels")
                    .map(|g| g.value.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            groups
                .iter()
                .any(|g| g.contains("\"primitive\":\"choice\"")),
            "expected canonical label JSON group, got {groups:?}"
        );
    }

    // ── query ─────────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_count() {
        let (client, _mgr, _) = start_metrics_service("query-count").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "mem.rss".to_owned(),
                    timestamp: now,
                    value_window_sum: 1024.0,
                    value_window_avg: 1024.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                }],
            })
            .await
            .expect("ingest");

        let rows = client
            .query(&MetricQuery {
                sql: String::new(),
                metric_id: "mem.rss".to_owned(),
                window_secs: 0,
                aggregation: AggregationFunc::Count,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect("query count");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value as u64, 1, "COUNT should be 1");

        // RawSql with an empty wire field retains its historical structured
        // compatibility behavior; only non-empty caller SQL is rejected.
        let compatibility_rows = client
            .query(&MetricQuery {
                sql: String::new(),
                metric_id: "mem.rss".to_owned(),
                window_secs: 0,
                aggregation: AggregationFunc::RawSql,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect("empty RawSql compatibility query");
        assert_eq!(compatibility_rows.len(), 1);
        assert_eq!(compatibility_rows[0].value as u64, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_sum() {
        let (client, _mgr, _) = start_metrics_service("query-sum").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        client
            .ingest(&IngestRequest {
                records: vec![
                    CMetricRecord {
                        metric_id: "disk.io".to_owned(),
                        timestamp: now,
                        value_window_sum: 100.0,
                        value_window_avg: 100.0,
                        value_window_count: 1,
                        labels: vec![],
                        tenant_id: String::new(),
                    },
                    CMetricRecord {
                        metric_id: "disk.io".to_owned(),
                        timestamp: now + 500,
                        value_window_sum: 200.0,
                        value_window_avg: 200.0,
                        value_window_count: 1,
                        labels: vec![],
                        tenant_id: String::new(),
                    },
                ],
            })
            .await
            .expect("ingest");

        let rows = client
            .query(&MetricQuery {
                sql: String::new(),
                metric_id: "disk.io".to_owned(),
                window_secs: 0,
                aggregation: AggregationFunc::Sum,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect("query sum");

        assert_eq!(rows.len(), 1);
        assert!(
            (rows[0].value - 300.0).abs() < 1e-6,
            "SUM should be 300.0, got {}",
            rows[0].value
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_raw_sql_denied_before_storage() {
        let db_dir = tempfile::tempdir().expect("temporary metrics database directory");
        let db_path = db_dir.path().join("metrics.duckdb");
        let (client, manager, backend) = start_metrics_service_on_path(
            "query-raw-denied",
            db_path.to_string_lossy().into_owned(),
            true,
        )
        .await;
        let before = backend
            .list_tables()
            .await
            .expect("list tables before raw SQL");
        let before_rows = metrics_row_count(&backend).await;

        let cases = [
            "CREATE TABLE query_owned (value INTEGER)",
            "INSERT INTO metrics VALUES ('raw', 1, 2, 2, 1)",
            "SELECT 1; CREATE TABLE query_owned_again (value INTEGER)",
            "SELECT * FROM read_csv('/synthetic/not-a-real-file')",
            "WITH rows AS (SELECT 1) SELECT * FROM rows",
            "SELECT * FROM metrics",
            "-- comment\nSELECT * FROM metrics",
            "   ",
        ];
        for sql in cases {
            let query = MetricQuery {
                sql: sql.to_owned(),
                metric_id: String::new(),
                window_secs: 0,
                aggregation: AggregationFunc::RawSql,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            };
            let error = client
                .query(&query)
                .await
                .expect_err("raw SQL must be denied");
            assert!(
                error.to_string().contains("raw SQL is not supported"),
                "unexpected query denial for {sql:?}: {error}"
            );

            // The shared builder runs before stream context allocation, so a
            // rejected queryStream must leave no continuation to consume.
            let mut stream_query = query.clone();
            stream_query.ephemeral_pubkey = vec![0u8; 32];
            let error = client
                .query_stream(&stream_query, [0u8; 32])
                .await
                .expect_err("raw SQL must be denied for queryStream");
            assert!(
                error.to_string().contains("raw SQL is not supported"),
                "unexpected queryStream denial for {sql:?}: {error}"
            );
        }

        let after = backend
            .list_tables()
            .await
            .expect("list tables after raw SQL");
        let after_rows = metrics_row_count(&backend).await;
        assert_eq!(
            before, after,
            "raw SQL rejection must precede storage execution"
        );
        assert_eq!(
            before_rows, after_rows,
            "raw SQL rejection must prevent DML"
        );
        assert!(!after.iter().any(|name| name == "query_owned"));

        drop(client);
        drop(manager);
        drop(backend);
        let reopened = DuckDbBackend::new(
            db_path.to_string_lossy().into_owned(),
            Default::default(),
            None,
        )
        .expect("reopen metrics database");
        let reopened_tables = reopened
            .list_tables()
            .await
            .expect("list tables after reopen");
        let reopened_rows = metrics_row_count(&reopened).await;
        assert_eq!(
            before, reopened_tables,
            "raw SQL must not mutate persisted schema"
        );
        assert_eq!(
            before_rows, reopened_rows,
            "raw SQL must not mutate persisted rows"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_structured_query_requires_query_permission() {
        let (client, _manager, _) =
            start_metrics_service_on_path("query-permission-denied", ":memory:".to_owned(), false)
                .await;
        let error = client
            .query(&MetricQuery {
                sql: String::new(),
                metric_id: String::new(),
                window_secs: 0,
                aggregation: AggregationFunc::Count,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect_err("structured query must require query permission");
        assert!(
            error.to_string().contains("Unauthorized"),
            "unexpected insufficient-permission error: {error}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_stream_structured_result_is_prepared() {
        let (client, _mgr, _) = start_metrics_service("query-stream-structured").await;
        let (client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "stream.metric".to_owned(),
                    timestamp: now,
                    value_window_sum: 11.0,
                    value_window_avg: 11.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                }],
            })
            .await
            .expect("ingest");
        let stream = client
            .query_stream(
                &MetricQuery {
                    sql: String::new(),
                    metric_id: "stream.metric".to_owned(),
                    window_secs: 0,
                    aggregation: AggregationFunc::Sum,
                    group_by: vec![],
                    limit_rows: 0,
                    ephemeral_pubkey: client_pub.to_bytes().to_vec(),
                },
                [0u8; 32],
            )
            .await
            .expect("structured queryStream");
        assert!(
            !stream.stream_id.is_empty(),
            "queryStream must return a stream id"
        );

        let (mac_key, enc_key, topic) =
            derive_client_stream_keys(&client_secret, &client_pub.to_bytes(), &stream.dh_public)
                .expect("derive stream keys");
        let origin = hyprstream_rpc::moq_stream::global_moq_origin()
            .expect("metrics test origin initialized");
        let broadcast = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin
                .consumer()
                .announced_broadcast(&stream.broadcast_path),
        )
        .await
        .expect("stream announcement timeout")
        .expect("stream broadcast announcement");
        let mut track = broadcast
            .subscribe_track(&Track::new(STREAM_TRACK))
            .expect("subscribe to structured query stream");
        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let mut payloads = Vec::new();
        for _ in 0..2 {
            let mut group =
                tokio::time::timeout(std::time::Duration::from_secs(5), track.next_group())
                    .await
                    .expect("stream group timeout")
                    .expect("stream group read")
                    .expect("stream ended before terminal frame");
            let frame = tokio::time::timeout(std::time::Duration::from_secs(5), group.read_frame())
                .await
                .expect("stream frame timeout")
                .expect("stream frame read")
                .expect("stream group had no frame");
            payloads.extend(
                hyprstream_rpc::moq_stream::verify_moq_frame(&mut verifier, &topic, &frame)
                    .expect("verify stream frame"),
            );
        }
        let data = payloads
            .iter()
            .find_map(|payload| match payload {
                StreamPayload::Data(data) => Some(data.as_slice()),
                _ => None,
            })
            .expect("structured query stream data");
        let mut reader = hyprstream_metrics::arrow::ipc::reader::StreamReader::try_new(
            std::io::Cursor::new(data),
            None,
        )
        .expect("read Arrow query stream");
        let batch = reader
            .next()
            .expect("Arrow query stream batch")
            .expect("Arrow query stream batch decode");
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Float64Array>()
            .expect("structured stream value column");
        assert_eq!(values.value(0), 11.0, "streamed SUM should be 11.0");
        assert!(
            payloads
                .iter()
                .any(|payload| matches!(payload, StreamPayload::Complete(_))),
            "structured query stream must terminate after publishing its result"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_stream_no_subscriber_timeout_releases_admission_permit() {
        let (client, manager, backend) =
            start_metrics_service("query-stream-no-subscriber-timeout").await;
        let origin = hyprstream_rpc::moq_stream::global_moq_origin()
            .expect("metrics test origin initialized");
        let before_rows = metrics_row_count(&backend).await;

        // Fill the production per-service admission pool with real Metrics
        // continuations that have no consumer. They must remain in the
        // demand wait and time out before they can prepare or execute SQL.
        let mut blocked_streams = Vec::new();
        for index in 0..hyprstream_rpc::streaming::DEFAULT_MAX_CONCURRENT_STREAMS_PER_SERVICE {
            let (_client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
            let stream = client
                .query_stream(
                    &MetricQuery {
                        sql: String::new(),
                        metric_id: format!("no-subscriber-{index}"),
                        window_secs: 0,
                        aggregation: AggregationFunc::Count,
                        group_by: vec![],
                        limit_rows: 0,
                        ephemeral_pubkey: client_pub.to_bytes().to_vec(),
                    },
                    [0u8; 32],
                )
                .await
                .expect("blocked structured queryStream");
            blocked_streams.push(stream);
        }

        // This real Metrics continuation queues behind the full pool. Its
        // announcement is therefore a direct admission-release signal: it
        // cannot be created until a no-subscriber continuation reaches its
        // bounded timeout and drops its permit.
        let (_sentinel_secret, sentinel_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let started = std::time::Instant::now();
        let sentinel = client
            .query_stream(
                &MetricQuery {
                    sql: String::new(),
                    metric_id: "no-subscriber-sentinel".to_owned(),
                    window_secs: 0,
                    aggregation: AggregationFunc::Count,
                    group_by: vec![],
                    limit_rows: 0,
                    ephemeral_pubkey: sentinel_pub.to_bytes().to_vec(),
                },
                [0u8; 32],
            )
            .await
            .expect("sentinel structured queryStream");
        let announcement = tokio::time::timeout(
            std::time::Duration::from_secs(35),
            origin
                .consumer()
                .announced_broadcast(&sentinel.broadcast_path),
        )
        .await
        .expect("sentinel admission release timeout")
        .expect("sentinel stream broadcast announcement");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= std::time::Duration::from_secs(25),
            "sentinel bypassed the no-subscriber demand timeout: {elapsed:?}"
        );
        assert_eq!(before_rows, metrics_row_count(&backend).await);

        // Attach only after admission is released. This supplies demand to
        // the sentinel so its own continuation can publish a normal
        // structured response and finish cleanly.
        let mut track = announcement
            .subscribe_track(&Track::new(STREAM_TRACK))
            .expect("subscribe to sentinel stream");
        for _ in 0..2 {
            let mut group =
                tokio::time::timeout(std::time::Duration::from_secs(5), track.next_group())
                    .await
                    .expect("sentinel stream group timeout")
                    .expect("sentinel stream group read")
                    .expect("sentinel stream ended before terminal frame");
            tokio::time::timeout(std::time::Duration::from_secs(5), group.read_frame())
                .await
                .expect("sentinel stream frame timeout")
                .expect("sentinel stream frame read")
                .expect("sentinel stream group had no frame");
        }

        drop(track);
        drop(blocked_streams);
        drop(client);
        drop(manager);
        drop(backend);
    }

    // ── view management ───────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_create_list_drop_view() {
        let (client, _mgr, _) = start_metrics_service("views").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "cpu.usage".to_owned(),
                    timestamp: now,
                    value_window_sum: 75.0,
                    value_window_avg: 75.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                }],
            })
            .await
            .expect("ingest");

        // Create
        client
            .create_view(&ViewSpec {
                name: "cpu_sum_view".to_owned(),
                query: MetricQuery {
                    sql: String::new(),
                    metric_id: "cpu.usage".to_owned(),
                    window_secs: 0,
                    aggregation: AggregationFunc::Sum,
                    group_by: vec![],
                    limit_rows: 0,
                    ephemeral_pubkey: vec![],
                },
            })
            .await
            .expect("create_view");

        // List
        let views = client.list_views().await.expect("list_views");
        assert!(
            views.iter().any(|v| v.name == "cpu_sum_view"),
            "cpu_sum_view should appear in list"
        );

        // Drop
        client.drop_view("cpu_sum_view").await.expect("drop_view");

        let views_after = client.list_views().await.expect("list_views after drop");
        assert!(
            !views_after.iter().any(|v| v.name == "cpu_sum_view"),
            "cpu_sum_view should be gone after drop"
        );
    }

    // ── validation ────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_invalid_identifier_rejected() {
        let (client, _mgr, _) = start_metrics_service("validation").await;

        // SQL injection attempt via drop_view name
        let err = client.drop_view("view; DROP TABLE metrics; --").await;
        assert!(err.is_err(), "invalid identifier should be rejected");

        // SQL injection via view creation name
        let err2 = client
            .create_view(&ViewSpec {
                name: "bad name!".to_owned(),
                query: MetricQuery {
                    sql: String::new(),
                    metric_id: String::new(),
                    window_secs: 0,
                    aggregation: AggregationFunc::Count,
                    group_by: vec![],
                    limit_rows: 0,
                    ephemeral_pubkey: vec![],
                },
            })
            .await;
        assert!(err2.is_err(), "invalid view name should be rejected");
    }
    // ═══════════════════════════════════════════════════════════════════════
    // R6 decision-store verifier coverage.
    //
    // Real-boundary evidence:
    //  - positive tenant request through the ACTUAL signed-JWT verifier,
    //    generated dispatch, and the owned PolicyService (delegated bearer);
    //  - wrong-tenant cannot read (store boundary behind real policy);
    //  - a tenant-ONLY decision role is denied legacy global SQL/ingest —
    //    legacy node-global SQL remains a distinct, broader authority and is
    //    never implied by tenant decision grants.
    //
    // Narrow direct-handler tests (for_test contexts) stay scoped to
    // handler/store logic; they deliberately bypass RPC identity verification.
    // ═══════════════════════════════════════════════════════════════════════

    /// Tenant-bound JWT fixture result.
    struct DecisionStoreFixture {
        service: MetricsService,
        backend: Arc<DuckDbBackend>,
        /// Signing key of the in-proc metrics service (for client pinning).
        metrics_key: SigningKey,
        /// Signing key of the wire caller (envelope signer + token cnf).
        client_key: SigningKey,
        /// Verified tenant-a token (sub "alice").
        token_a: String,
        /// Verified tenant-b token (sub "bob").
        token_b: String,
        /// In-proc endpoint the metrics service listens on.
        endpoint: String,
    }

    fn decision_client(
        endpoint: &str,
        client_key: &SigningKey,
        metrics_key: &SigningKey,
        token: Option<String>,
    ) -> hyprstream_rpc_std::metrics_client::MetricsClient {
        hyprstream_rpc_std::metrics_client::MetricsClient::for_local_endpoint_bootstrap(
            endpoint,
            client_key.clone(),
            metrics_key.verifying_key(),
            token,
        )
        .unwrap()
    }

    /// Wire a metrics service with decision stores, real PolicyService, and
    /// ClusterKeySource JWT verification on BOTH hops; mint tenant-a/tenant-b
    /// tokens. NOT spawned — spawn manually for wire tests.
    async fn decision_store_fixture(
        tag: &str,
        policy_manager: PolicyManager,
    ) -> DecisionStoreFixture {
        let _ = init_global_moq_origin(MoqStreamOrigin::standalone().build());
        crate::mac::install_explicit_test_dispatch_pep();
        let _ = hyprstream_rpc::envelope::install_verify_config(
            hyprstream_rpc::envelope::EnvelopeVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        let _ = hyprstream_rpc::envelope::install_response_verify_config(
            hyprstream_rpc::envelope::ResponseVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        // RPC-path token verification fails closed without a process-global
        // revocation store; publish an isolated empty store when absent.
        if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
            let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
            ));
        }

        let (policy_signing_key, _) = generate_signing_keypair();
        let (metrics_signing_key, _) = generate_signing_keypair();
        let (client_signing_key, _) = generate_signing_keypair();
        let (token_signing_key, _) = generate_signing_keypair();
        for (key, subject) in [
            (&policy_signing_key, "service:policy"),
            (&metrics_signing_key, "service:metrics"),
            (&client_signing_key, "service:metrics-query"),
        ] {
            hyprstream_service::global_trust_store().insert(
                key.verifying_key(),
                hyprstream_service::Attestation {
                    scopes: std::collections::HashSet::new(),
                    subject: Some(subject.to_owned()),
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                },
            );
        }
        let issuer = format!("https://{tag}.metrics.test");
        let key_source = hyprstream_rpc::auth::ClusterKeySource::new(
            token_signing_key.verifying_key(),
            issuer.clone(),
        );

        let policy_manager = Arc::new(policy_manager);
        let git2db = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(tempfile::TempDir::new().unwrap().path())
                .await
                .unwrap(),
        ));
        let policy_svc = PolicyService::new(
            policy_manager,
            Arc::new(policy_signing_key.clone()),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc(format!("test-policy-{tag}")),
        )
        .with_jwt_key_source(Arc::new(key_source.clone()));
        let manager = InprocManager::new();
        manager.spawn(Box::new(policy_svc)).await.unwrap();

        let policy_client = PolicyClient::for_local_endpoint_bootstrap(
            &format!("inproc://test-policy-{tag}"),
            metrics_signing_key.clone(),
            policy_signing_key.verifying_key(),
            None,
        )
        .unwrap();

        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        hyprstream_metrics::create_schema_v2_tables(backend.as_ref())
            .await
            .unwrap();
        hyprstream_metrics::metrics::decision_outcome::create_decision_outcome_tables(
            backend.as_ref(),
        )
        .await
        .unwrap();
        let orchestrator = Arc::new(
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    QueryOrchestrator::new(
                        backend.clone() as Arc<dyn hyprstream_metrics::StorageBackend>
                    )
                    .await
                })
            })
            .unwrap(),
        );
        let service = MetricsService::new(
            orchestrator,
            TransportConfig::inproc(format!("test-metrics-{tag}")),
            metrics_signing_key.clone(),
            policy_client,
        )
        .with_jwt_key_source(Arc::new(key_source))
        .with_decision_stores(backend.clone());

        let now = chrono::Utc::now().timestamp();
        let client_vk_bytes = client_signing_key.verifying_key().as_bytes().to_owned();
        let mint = |sub: &str, tenant: &str| {
            hyprstream_rpc::auth::jwt::encode(
                &hyprstream_rpc::auth::Claims::new(sub.to_owned(), now, now + 300)
                    .with_issuer(issuer.clone())
                    .with_tenant(tenant.to_owned())
                    .with_cnf_jwk(&client_vk_bytes)
                    .with_client_id("hyprstream-oauth-client-1"),
                &token_signing_key,
            )
        };
        DecisionStoreFixture {
            service,
            backend,
            metrics_key: metrics_signing_key,
            client_key: client_signing_key,
            token_a: mint("alice", "tenant-a"),
            token_b: mint("bob", "tenant-b"),
            endpoint: format!("inproc://test-metrics-{tag}"),
        }
    }

    fn tenant_caller_ctx(tenant: &str, key: &SigningKey) -> EnvelopeContext {
        EnvelopeContext::for_test_authenticated_subject_in_tenant(
            hyprstream_rpc::envelope::Subject::new("service:metrics-query"),
            tenant,
            key.verifying_key(),
        )
    }

    fn decision_request(
        payload: Vec<u8>,
    ) -> hyprstream_rpc_std::metrics_client::DecisionRecordRequest {
        hyprstream_rpc_std::metrics_client::DecisionRecordRequest {
            schema_fingerprint: "jev1:route:v3".to_owned(),
            model_id: "clef-flash@8b2e5fd1".to_owned(),
            encoding_id: "clef-text-json-v1".to_owned(),
            quantization: "fp8-e4m3-dynamic".to_owned(),
            compute_profile: "cpu-f32".to_owned(),
            calib_version: String::new(),
            payload,
        }
    }

    fn outcome_request(
        decision_id: &str,
        row_id: &str,
    ) -> hyprstream_rpc_std::metrics_client::DecisionOutcomeRequest {
        hyprstream_rpc_std::metrics_client::DecisionOutcomeRequest {
            decision_id: decision_id.to_owned(),
            decision_row_id: row_id.to_owned(),
            question_id: "route".to_owned(),
            expected_schema_fingerprint: String::new(),
            expected_model_id: String::new(),
            expected_encoding_id: String::new(),
            expected_quantization: String::new(),
            expected_compute_profile: String::new(),
            expected_calib: ExpectedCalib::Any,
            expected_calib_version: String::new(),
            outcome_label: "beta".to_owned(),
            outcome_value: 0.0,
            has_outcome_value: false,
            outcome_source: "human-label".to_owned(),
            supersedes_outcome_id: String::new(),
        }
    }

    fn all_outcome_query() -> hyprstream_rpc_std::metrics_client::DecisionOutcomeQuery {
        hyprstream_rpc_std::metrics_client::DecisionOutcomeQuery {
            schema_fingerprint: String::new(),
            model_id: String::new(),
            question_id: String::new(),
            calib_filter: ExpectedCalib::Any,
            calib_version: String::new(),
            include_superseded: false,
            limit: 0,
            after_created_at: 0,
            after_outcome_id: String::new(),
        }
    }

    /// Minimal VALID decision batch: one float32 fixed-size-list question
    /// column with jev metadata + version-triple columns, `rows` input rows.
    fn minimal_decision_batch(rows: usize) -> RecordBatch {
        use hyprstream_metrics::arrow::array::{ArrayRef, FixedSizeListArray, StringArray};
        use hyprstream_metrics::arrow::datatypes::{DataType, Field, Float32Type, Schema};
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            hyprstream_metrics::metrics::decision_outcome::FIELD_META_KIND.to_owned(),
            "choice".to_owned(),
        );
        meta.insert(
            hyprstream_metrics::metrics::decision_outcome::FIELD_META_LABELS.to_owned(),
            "[\"alpha\",\"beta\"]".to_owned(),
        );
        let row_values = vec![Some(0.25_f32), Some(0.75)];
        let values: Vec<Option<Vec<Option<f32>>>> =
            (0..rows).map(|_| Some(row_values.clone())).collect();
        let route = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(values, 2);
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "route.probabilities",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            )
            .with_metadata(meta),
            Field::new("schema_version", DataType::Utf8, false),
            Field::new("model_version", DataType::Utf8, false),
            Field::new("calib_version", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(route) as ArrayRef,
                Arc::new(StringArray::from(vec!["v3"; rows])) as ArrayRef,
                Arc::new(StringArray::from(vec!["clef-flash@8b2e5fd1"; rows])) as ArrayRef,
                Arc::new(StringArray::from(vec![None::<&str>; rows])) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn minimal_decision_payload() -> Vec<u8> {
        hyprstream_metrics::metrics::decision_batch::encode_batch_ipc(&minimal_decision_batch(1))
            .unwrap()
    }

    /// COMPRESSION BOMB: a genuinely lz4-compressed decision payload whose
    /// first data buffer claims an 8-byte advertised prefix while the lz4
    /// stream actually expands to the full row data — the exact shape
    /// arrow-ipc's unbounded read_to_end would over-allocate on.
    fn compression_bomb_payload() -> Vec<u8> {
        use hyprstream_metrics::arrow::ipc::writer::IpcWriteOptions;
        use hyprstream_metrics::arrow::ipc::CompressionType;
        // Repetition ensures a genuine codec frame, not Arrow's -1/raw
        // representation for small incompressible buffers;256 rows stays
        // well inside the existing4096row/16MiB admission bounds.
        let batch = minimal_decision_batch(256);
        let options = IpcWriteOptions::default()
            .try_with_compression(Some(CompressionType::LZ4_FRAME))
            .unwrap();
        let mut buf = Vec::new();
        {
            let mut writer =
                hyprstream_metrics::arrow::ipc::writer::StreamWriter::try_new_with_options(
                    &mut buf,
                    &batch.schema(),
                    options,
                )
                .unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        // Locate the record-batch message, then the largest compressed
        // buffer's advertised prefix, and shrink the claim to 8 bytes.
        let bytes = buf;
        assert_eq!(
            hyprstream_metrics::metrics::decision_batch::decode_batch_ipc(&bytes).unwrap(),
            batch,
            "unmodified compression-bomb control must decode canonically"
        );
        // StreamWriter framing starts at byte 0 (no magic).
        let mut pos = 0usize;
        let (meta_start, meta_end, body_start) = loop {
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let meta_start = pos + 8;
            let meta_end = meta_start + size;
            let body_start = meta_end;
            let message =
                hyprstream_metrics::arrow::ipc::root_as_message(&bytes[meta_start..meta_end])
                    .unwrap();
            if message.header_as_record_batch().is_some() {
                break (meta_start, meta_end, body_start);
            }
            pos = body_start + message.bodyLength() as usize;
        };
        let message =
            hyprstream_metrics::arrow::ipc::root_as_message(&bytes[meta_start..meta_end]).unwrap();
        let rb = message.header_as_record_batch().unwrap();
        let buffers = rb.buffers().unwrap();
        let mut chosen: Option<(usize, i64)> = None;
        for i in 0..buffers.len() {
            let b = buffers.get(i);
            if b.length() < 12 {
                continue;
            }
            let prefix = body_start + usize::try_from(b.offset()).unwrap();
            let advertised = i64::from_le_bytes(bytes[prefix..prefix + 8].try_into().unwrap());
            if advertised > 8 && chosen.is_none_or(|(_, prior)| advertised > prior) {
                chosen = Some((prefix, advertised));
            }
        }
        let (prefix_pos, advertised) = chosen.expect("a genuine compressed buffer, not0/-1");
        assert!(advertised > 8, "negative must shrink actual decoded bytes");
        assert_eq!(
            &bytes[prefix_pos + 8..prefix_pos + 12],
            &[0x04, 0x22, 0x4D, 0x18],
            "selected positive prefix must precede a real LZ4 frame"
        );
        let mut bytes = bytes;
        bytes[prefix_pos..prefix_pos + 8].copy_from_slice(&8i64.to_le_bytes());
        bytes
    }

    /// NARROW scope: direct handler calls with a for_test tenant context.
    /// Proves handler/store logic (persistence, tenant stamping, exact-row
    /// linkage) — NOT RPC identity verification.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_store_handler_persists_under_verified_tenant() {
        let me = PolicyManager::permissive().await.unwrap();
        let fixture = decision_store_fixture("ds-direct", me).await;
        let ctx = tenant_caller_ctx("tenant-a", &fixture.client_key);

        let response = fixture
            .service
            .handle_record_decision(&ctx, 1, &decision_request(minimal_decision_payload()))
            .await
            .expect("record decision");
        let stored = match response {
            MetricsResponseVariant::RecordDecisionResult(r) => r,
            other => panic!("unexpected response {other:?}"),
        };
        assert_eq!(stored.row_ids.len(), 1);

        let handle = fixture
            .backend
            .prepare_sql("SELECT tenant_id, calib_version FROM decision_rows")
            .await
            .unwrap();
        let batch = fixture.backend.query_sql(&handle).await.unwrap();
        let tenants = batch
            .column_by_name("tenant_id")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::StringArray>()
            .unwrap();
        assert_eq!(tenants.value(0), "tenant-a");
        let calib = batch
            .column_by_name("calib_version")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::StringArray>()
            .unwrap();
        use hyprstream_metrics::arrow::array::Array as _;
        assert!(calib.is_null(0), "raw inference keeps calib_version NULL");

        let response = fixture
            .service
            .handle_record_outcome(
                &ctx,
                2,
                &outcome_request(&stored.decision_id, &stored.row_ids[0]),
            )
            .await
            .expect("record outcome");
        match response {
            MetricsResponseVariant::RecordOutcomeResult(r) => assert!(!r.outcome_id.is_empty()),
            other => panic!("unexpected response {other:?}"),
        }
    }

    /// Fail closed without a verified tenant: authorize (against a permissive
    /// manager) and the handler both deny.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_store_fails_closed_without_verified_tenant() {
        let me = PolicyManager::permissive().await.unwrap();
        let fixture = decision_store_fixture("ds-closed", me).await;
        let untenanted = EnvelopeContext::for_test_authenticated_subject(
            hyprstream_rpc::envelope::Subject::new("service:metrics-query"),
            fixture.client_key.verifying_key(),
        );
        let err = fixture
            .service
            .authorize(&untenanted, "metrics:RecordDecision", "write")
            .await
            .expect_err("missing verified tenant must fail closed");
        assert!(
            err.to_string().contains("no verified tenant domain"),
            "{err}"
        );
        let err = fixture
            .service
            .handle_record_decision(
                &untenanted,
                3,
                &decision_request(minimal_decision_payload()),
            )
            .await
            .expect_err("handler must fail closed without verified tenant");
        assert!(
            err.to_string().contains("no verified tenant domain"),
            "{err}"
        );
    }

    /// POSITIVE WIRE EVIDENCE: signed JWT through the generated dispatch and
    /// real verifier chain; PolicyService evaluates the delegated bearer.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_store_wire_positive_signed_jwt() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        for (res, op) in [
            ("metrics:RecordDecision", "write"),
            ("metrics:RecordOutcome", "write"),
            ("metrics:QueryOutcomes", "query"),
        ] {
            me.add_policy_with_domain("alice", "tenant-a", res, op, "allow")
                .await
                .unwrap();
        }
        let fixture = decision_store_fixture("ds-wire-pos", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );

        let stored = client
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect("wire record decision with verified tenant JWT");
        assert_eq!(stored.row_ids.len(), 1);

        let outcome = client
            .record_outcome(&outcome_request(&stored.decision_id, &stored.row_ids[0]))
            .await
            .expect("wire record outcome");
        assert!(!outcome.outcome_id.is_empty());

        let joined = client
            .query_outcomes(&all_outcome_query())
            .await
            .expect("wire outcome join");
        assert_eq!(joined.rows.len(), 1, "tenant-a sees its own outcome");
        assert_eq!(joined.rows[0].tenant_id, "tenant-a");
        assert_eq!(joined.rows[0].question_id, "route");
        assert_eq!(joined.rows[0].outcome_label, "beta");
        assert!(!joined.rows[0].has_outcome_value);
        assert!(joined.rows[0].calib_version.is_empty(), "raw row stays raw");

        // The durable row is stamped with the VERIFIED tenant from the JWT.
        let handle = fixture
            .backend
            .prepare_sql("SELECT tenant_id FROM decision_rows")
            .await
            .unwrap();
        let batch = fixture.backend.query_sql(&handle).await.unwrap();
        let tenants = batch
            .column_by_name("tenant_id")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::StringArray>()
            .unwrap();
        assert_eq!(tenants.value(0), "tenant-a");
    }

    /// WRONG-TENANT CANNOT READ: tenant-b's verified JWT yields an empty join
    /// over tenant-a rows, and cross-tenant outcome linkage is a not-found at
    /// the store boundary behind real policy.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_store_wire_wrong_tenant_cannot_read() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        for (res, op) in [
            ("metrics:RecordDecision", "write"),
            ("metrics:RecordOutcome", "write"),
            ("metrics:QueryOutcomes", "query"),
        ] {
            me.add_policy_with_domain("alice", "tenant-a", res, op, "allow")
                .await
                .unwrap();
            me.add_policy_with_domain("bob", "tenant-b", res, op, "allow")
                .await
                .unwrap();
        }
        let fixture = decision_store_fixture("ds-wire-x", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client_a = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );
        let client_b = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_b.clone()),
        );

        let stored = client_a
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect("tenant-a records");

        // tenant-b's join is empty: it cannot read tenant-a rows.
        let joined_b = client_b
            .query_outcomes(&all_outcome_query())
            .await
            .expect("tenant-b join authorizes in its own domain");
        assert_eq!(
            joined_b.rows.len(),
            0,
            "tenant-b must not see tenant-a rows"
        );

        // Cross-tenant outcome linkage: authorized in tenant-b, but the store
        // scopes by the verified tenant -> not-found, never a leak.
        let err = client_b
            .record_outcome(&outcome_request(&stored.decision_id, &stored.row_ids[0]))
            .await
            .expect_err("cross-tenant outcome must be denied");
        assert!(err.to_string().contains("not found in tenant"), "{err}");

        // tenant-b CAN persist and read its own rows.
        let own = client_b
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect("tenant-b records its own");
        let joined_b0 = client_b
            .query_outcomes(&all_outcome_query())
            .await
            .expect("tenant-b join");
        assert_eq!(joined_b0.rows.len(), 0, "no outcomes recorded yet for b");
        client_b
            .record_outcome(&outcome_request(&own.decision_id, &own.row_ids[0]))
            .await
            .expect("tenant-b links its own row");
        let joined_b1 = client_b
            .query_outcomes(&all_outcome_query())
            .await
            .expect("tenant-b join after own outcome");
        assert_eq!(joined_b1.rows.len(), 1);
        assert_eq!(joined_b1.rows[0].tenant_id, "tenant-b");
    }

    /// TENANT-ONLY DECISION ROLE: typed operations allowed, legacy global
    /// SQL/ingest/stream DENIED. Legacy node-global SQL is a distinct broader
    /// authority (Casbin domain "*"/broad roles can reach backend tables) and
    /// is never implied by tenant decision grants; Flight carries no tenant
    /// row predicates, so tenant roles are never granted flight:* either.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_tenant_only_decision_role_denied_legacy_sql() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        for (res, op) in [
            ("metrics:RecordDecision", "write"),
            ("metrics:RecordOutcome", "write"),
            ("metrics:QueryOutcomes", "query"),
        ] {
            me.add_policy_with_domain("alice", "tenant-a", res, op, "allow")
                .await
                .unwrap();
        }
        let fixture = decision_store_fixture("ds-role", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );

        // Typed surface allowed.
        client
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect("typed decision store allowed for the tenant role");

        // Legacy structured query (metrics:Query) denied.
        let err = client
            .query(&MetricQuery {
                sql: String::new(),
                metric_id: String::new(),
                window_secs: 0,
                aggregation: AggregationFunc::Count,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect_err("tenant-only role must not read legacy metrics SQL");
        assert!(err.to_string().contains("Unauthorized"), "{err}");

        // Legacy ingest denied.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let err = client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "x".to_owned(),
                    timestamp: now,
                    value_window_sum: 1.0,
                    value_window_avg: 1.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                }],
            })
            .await
            .expect_err("tenant-only role must not ingest legacy metrics");
        assert!(err.to_string().contains("Unauthorized"), "{err}");

        // Legacy streaming query denied at the same boundary.
        let (client_secret, client_pub) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let err = client
            .query_stream(
                &MetricQuery {
                    sql: String::new(),
                    metric_id: String::new(),
                    window_secs: 0,
                    aggregation: AggregationFunc::Count,
                    group_by: vec![],
                    limit_rows: 0,
                    ephemeral_pubkey: client_pub.to_bytes().to_vec(),
                },
                [0u8; 32],
            )
            .await
            .expect_err("tenant-only role must not stream legacy metrics SQL");
        assert!(err.to_string().contains("Unauthorized"), "{err}");
        let _ = client_secret;
    }

    /// Legacy METHOD-SPECIFIC grants (domain "*" on metrics:Query/Ingest)
    /// do not implicitly authorize the new decision resources, and tenant
    /// decision grants do not carry legacy SQL authority — resource scopes
    /// separate the surfaces. Explicit administrator wildcards naming a
    /// decision resource with domain "*" remain global authority by design
    /// (Casbin `p.dom == "*"` matches any request domain); they are never
    /// part of a tenant-only role.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_legacy_global_grant_separate_from_decision_store() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        // The classic legacy shape: global-domain metrics SQL/ingest.
        me.add_policy_with_domain("alice", "*", "metrics:Query", "query", "allow")
            .await
            .unwrap();
        me.add_policy_with_domain("alice", "*", "metrics:Ingest", "write", "allow")
            .await
            .unwrap();
        let fixture = decision_store_fixture("ds-sep", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "legacy.metric".to_owned(),
                    timestamp: now,
                    value_window_sum: 1.0,
                    value_window_avg: 1.0,
                    value_window_count: 1,
                    labels: vec![],
                    tenant_id: String::new(),
                }],
            })
            .await
            .expect("legacy global grant authorizes ingest unchanged");

        let err = client
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect_err("legacy global grant must not authorize the tenant store");
        assert!(err.to_string().contains("Unauthorized"), "{err}");

        let err = client
            .query_outcomes(&all_outcome_query())
            .await
            .expect_err("legacy global grant must not authorize outcome joins");
        assert!(err.to_string().contains("Unauthorized"), "{err}");
    }

    /// Wire fail-closed still holds with the JWT plumbing installed: a client
    /// with NO bearer has no verified tenant and is denied even under a
    /// permissive manager.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_store_wire_denies_missing_verified_tenant() {
        let me = PolicyManager::permissive().await.unwrap();
        let fixture = decision_store_fixture("ds-wire-nojwt", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            None,
        );

        let err = client
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .expect_err("wire request without verified tenant must be denied");
        assert!(
            err.to_string().contains("no verified tenant domain"),
            "{err}"
        );
    }
    /// recordDecision WIRE negative: a NaN distribution is rejected at the
    /// RPC boundary and inserts NOTHING (invalid input is never normalized
    /// into the store).
    #[tokio::test(flavor = "multi_thread")]
    async fn test_record_decision_wire_rejects_nan_with_zero_inserts() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        me.add_policy_with_domain(
            "alice",
            "tenant-a",
            "metrics:RecordDecision",
            "write",
            "allow",
        )
        .await
        .unwrap();
        let fixture = decision_store_fixture("ds-wire-nan", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );

        // Minimal payload with a NaN component.
        use hyprstream_metrics::arrow::array::{
            ArrayRef, FixedSizeListArray, RecordBatch, StringArray,
        };
        use hyprstream_metrics::arrow::datatypes::{DataType, Field, Float32Type, Schema};
        let mut meta = std::collections::HashMap::new();
        meta.insert(
            hyprstream_metrics::metrics::decision_outcome::FIELD_META_KIND.to_owned(),
            "choice".to_owned(),
        );
        meta.insert(
            hyprstream_metrics::metrics::decision_outcome::FIELD_META_LABELS.to_owned(),
            "[\"alpha\",\"beta\"]".to_owned(),
        );
        let route = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(f32::NAN), Some(0.4)])],
            2,
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(
                    "route.probabilities",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        2,
                    ),
                    true,
                )
                .with_metadata(meta),
                Field::new("schema_version", DataType::Utf8, false),
                Field::new("model_version", DataType::Utf8, false),
                Field::new("calib_version", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(route) as ArrayRef,
                Arc::new(StringArray::from(vec!["v3"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["clef-flash@8b2e5fd1"])) as ArrayRef,
                Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            ],
        )
        .unwrap();
        let payload =
            hyprstream_metrics::metrics::decision_batch::encode_batch_ipc(&batch).unwrap();

        let err = client
            .record_decision(&decision_request(payload))
            .await
            .expect_err("NaN distribution must be rejected at the wire");
        assert!(
            err.to_string().contains("not a finite probability"),
            "{err}"
        );

        // Zero inserts: the durable table is still empty.
        let handle = fixture
            .backend
            .prepare_sql("SELECT COUNT(*) AS n FROM decision_rows")
            .await
            .unwrap();
        let count = fixture.backend.query_sql(&handle).await.unwrap();
        let n = count
            .column_by_name("n")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(n, 0, "rejected recordDecision must insert nothing");
    }
    /// WIRE P1 regression: a compression-bomb recordDecision (tiny advertised
    /// prefix, LZ4 stream expanding far beyond it) is rejected by the bounded
    /// verifier and inserts NOTHING — Arrow's unbounded read_to_end decoder
    /// is never reached.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_record_decision_wire_rejects_expansion_bomb_with_zero_inserts() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        me.add_policy_with_domain(
            "alice",
            "tenant-a",
            "metrics:RecordDecision",
            "write",
            "allow",
        )
        .await
        .unwrap();
        let fixture = decision_store_fixture("ds-wire-bomb", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );

        let err = client
            .record_decision(&decision_request(compression_bomb_payload()))
            .await
            .expect_err("compression bomb must be rejected at the wire");
        assert!(
            err.to_string().contains("advertised") || err.to_string().contains("beyond"),
            "{err}"
        );

        // Zero inserts.
        let handle = fixture
            .backend
            .prepare_sql("SELECT COUNT(*) AS n FROM decision_rows")
            .await
            .unwrap();
        let count = fixture.backend.query_sql(&handle).await.unwrap();
        let n = count
            .column_by_name("n")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(n, 0, "rejected compression bomb must insert nothing");
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_record_decision_wire_rejects_dictionary_representation_with_zero_inserts() {
        use hyprstream_metrics::arrow::array::{
            ArrayRef, DictionaryArray, Int32Array, RecordBatch, UInt8Array,
        };
        use hyprstream_metrics::arrow::datatypes::{DataType, Field, Int32Type, Schema};
        use hyprstream_metrics::arrow::ipc::writer::{
            DictionaryHandling, IpcWriteOptions, StreamWriter,
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "extra",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::UInt8)),
            false,
        )]));
        const BASE_VALUES: usize = 192 * 1024;
        const EPOCHS: usize = 96;
        // Every interleaved record retains its distinct cumulative generation.
        let retained_values = EPOCHS * BASE_VALUES + EPOCHS * (EPOCHS - 1) / 2;
        assert_eq!(retained_values, 18_878_928);
        assert!(retained_values > MAX_PAYLOAD_BYTES);
        let mut payload = Vec::new();
        {
            let mut writer = StreamWriter::try_new_with_options(
                &mut payload,
                &schema,
                IpcWriteOptions::default().with_dictionary_handling(DictionaryHandling::Delta),
            )
            .unwrap();
            // Pinned Arrow56 writer measurement: 282696 encoded bytes, including
            // validity, padding and framing; 1 full dictionary, 95 deltas, 96 records.
            // Without admission retained values total ~18MiB, never GiB.
            for n in 0..EPOCHS {
                let values = Arc::new(UInt8Array::from(vec![0u8; BASE_VALUES + n])) as ArrayRef;
                let array = Arc::new(DictionaryArray::<Int32Type>::new(
                    Int32Array::from(vec![0]),
                    values,
                )) as ArrayRef;
                let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
                writer.write(&batch).unwrap();
            }
            writer.finish().unwrap();
        }
        assert!(
            payload.len() < 300 * 1024,
            "actual encoded fixture bytes: {}",
            payload.len()
        );
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        me.add_policy_with_domain(
            "alice",
            "tenant-a",
            "metrics:RecordDecision",
            "write",
            "allow",
        )
        .await
        .unwrap();
        let fixture = decision_store_fixture("ds-wire-representation", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );
        let error = client
            .record_decision(&decision_request(payload))
            .await
            .expect_err("representation amplification must reject before Arrow");
        assert!(
            error
                .to_string()
                .contains("IPC representation: materialized buffer budget exceeded"),
            "{error}"
        );
        let query = fixture
            .backend
            .prepare_sql("SELECT COUNT(*) AS n FROM decision_rows")
            .await
            .unwrap();
        let count = fixture.backend.query_sql(&query).await.unwrap();
        let n = count
            .column_by_name("n")
            .unwrap()
            .as_any()
            .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(
            n, 0,
            "predecode representation rejection must insert no durable rows"
        );
    }

    /// Typed generated FB fixtures only: no raw table-offset/header parser.
    /// A missing preflight would at most allocate small tens of MiB, never GiB.
    fn metadata_budget_wire_payload(case: &str) -> Vec<u8> {
        use flatbuffers::FlatBufferBuilder;
        use hyprstream_metrics::arrow::ipc::{
            self,
            gen::Message::{DictionaryBatchArgs, FieldNode, Message, MessageArgs, RecordBatchArgs},
        };
        fn append_message(out: &mut Vec<u8>, bytes: &[u8]) {
            use hyprstream_metrics::arrow::ipc::writer::{
                write_message, EncodedData, IpcWriteOptions,
            };
            write_message(
                out,
                EncodedData {
                    ipc_message: bytes.to_vec(),
                    arrow_data: Vec::new(),
                },
                &IpcWriteOptions::default(),
            )
            .expect("canonical IPC fixture framing");
        }
        let mut out = Vec::new();
        let mut b = FlatBufferBuilder::new();
        let null = ipc::Null::create(&mut b, &ipc::NullArgs {});
        let name = b.create_string(if case == "schema" { "" } else { "child" });
        let long_name = if case == "schema" {
            Some(b.create_string(&"n".repeat(16 * 1024)))
        } else {
            None
        };
        let child = ipc::Field::create(
            &mut b,
            &ipc::FieldArgs {
                name: long_name.or(Some(name)),
                nullable: false,
                type_type: ipc::Type::Null,
                type_: Some(null.as_union_value()),
                ..Default::default()
            },
        );
        let field = if case == "schema" {
            child
        } else {
            let children = b.create_vector(&[child]);
            let structure = ipc::Struct_::create(&mut b, &ipc::Struct_Args {});
            let index = ipc::Int::create(
                &mut b,
                &ipc::IntArgs {
                    bitWidth: 8,
                    is_signed: true,
                },
            );
            let dictionary = ipc::DictionaryEncoding::create(
                &mut b,
                &ipc::DictionaryEncodingArgs {
                    id: 1,
                    indexType: Some(index),
                    ..Default::default()
                },
            );
            let name = b.create_string("dictionary");
            ipc::Field::create(
                &mut b,
                &ipc::FieldArgs {
                    name: Some(name),
                    nullable: true,
                    type_type: ipc::Type::Struct_,
                    type_: Some(structure.as_union_value()),
                    dictionary: Some(dictionary),
                    children: Some(children),
                    ..Default::default()
                },
            )
        };
        let fields = b.create_vector(&vec![field; if case == "schema" { 1024 } else { 1 }]);
        let schema = ipc::Schema::create(
            &mut b,
            &ipc::SchemaArgs {
                fields: Some(fields),
                ..Default::default()
            },
        );
        let msg = Message::create(
            &mut b,
            &MessageArgs {
                version: ipc::MetadataVersion::V5,
                header_type: ipc::MessageHeader::Schema,
                header: Some(schema.as_union_value()),
                ..Default::default()
            },
        );
        b.finish(msg, None);
        append_message(&mut out, b.finished_data());
        if case != "schema" {
            let mut b = FlatBufferBuilder::new();
            let n = (hyprstream_metrics::metrics::decision_batch::MAX_PAYLOAD_BYTES as i64) * 8 + 1;
            let nodes = if case == "node" {
                vec![FieldNode::new(-1, 0)]
            } else {
                vec![FieldNode::new(n, 0), FieldNode::new(n, n)]
            };
            let nodes = b.create_vector(&nodes);
            let buffers = b.create_vector(&[ipc::Buffer::new(0, 0)]);
            let data = ipc::RecordBatch::create(
                &mut b,
                &RecordBatchArgs {
                    length: if case == "node" { 1 } else { n },
                    nodes: Some(nodes),
                    buffers: Some(buffers),
                    ..Default::default()
                },
            );
            let dictionary = ipc::DictionaryBatch::create(
                &mut b,
                &DictionaryBatchArgs {
                    id: 1,
                    data: Some(data),
                    isDelta: false,
                },
            );
            let msg = Message::create(
                &mut b,
                &MessageArgs {
                    version: ipc::MetadataVersion::V5,
                    header_type: ipc::MessageHeader::DictionaryBatch,
                    header: Some(dictionary.as_union_value()),
                    ..Default::default()
                },
            );
            b.finish(msg, None);
            append_message(&mut out, b.finished_data());
        }
        out
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn test_record_decision_wire_rejects_metadata_allocations_with_zero_inserts() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        me.add_policy_with_domain(
            "alice",
            "tenant-a",
            "metrics:RecordDecision",
            "write",
            "allow",
        )
        .await
        .unwrap();
        let fixture = decision_store_fixture("ds-wire-metadata", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );
        // This existing fixture uses real signed original-user JWT, generated
        // MetricsClient, real dispatch+Policy and actual durable DuckDB store.
        for (case, cause) in [
            ("schema", "owned schema"),
            ("bitmap", "logical bitmap"),
            ("node", "negative field node"),
        ] {
            let payload = metadata_budget_wire_payload(case);
            assert!(
                payload.len() < 128 * 1024,
                "tiny encoded allocation fixture"
            );
            let Err(err) = client.record_decision(&decision_request(payload)).await else {
                panic!("metadata allocation must reject");
            };
            assert!(err.to_string().contains(cause), "{case}: {err}");
            // A generic Arrow/column/schema failure does not satisfy this test:
            // it must be the BEFORE-decoder admission guard's specific cause.
            assert!(err.to_string().contains("IPC metadata"), "{err}");
            let query = fixture
                .backend
                .prepare_sql("SELECT COUNT(*) AS n FROM decision_rows")
                .await
                .unwrap();
            let count = fixture.backend.query_sql(&query).await.unwrap();
            let n = count
                .column_by_name("n")
                .unwrap()
                .as_any()
                .downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>()
                .unwrap()
                .value(0);
            assert_eq!(
                n, 0,
                "{case}: rejected metadata must insert no durable rows"
            );
        }
    }
    /// Actual generated signed-JWT dispatch with real owned PolicyService and
    /// DuckDB stores. Cursor fields position rows; they never confer tenancy.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_decision_outcome_wire_keyset_pages_and_controls() {
        let me = PolicyManager::new_in_memory().await.unwrap();
        me.add_policy("service:metrics", "policy:Check", "check")
            .await
            .unwrap();
        for (res, op) in [
            ("metrics:RecordDecision", "write"),
            ("metrics:RecordOutcome", "write"),
            ("metrics:QueryOutcomes", "query"),
        ] {
            for (subject, tenant) in [("alice", "tenant-a"), ("bob", "tenant-b")] {
                me.add_policy_with_domain(subject, tenant, res, op, "allow")
                    .await
                    .unwrap();
            }
        }
        let fixture = decision_store_fixture("ds-wire-pages", me).await;
        let manager = InprocManager::new();
        manager.spawn(Box::new(fixture.service)).await.unwrap();
        let client_a = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_a.clone()),
        );
        let client_b = decision_client(
            &fixture.endpoint,
            &fixture.client_key,
            &fixture.metrics_key,
            Some(fixture.token_b.clone()),
        );
        let stored = client_a
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .unwrap();
        let mut recorded = Vec::new();
        for _ in 0..3 {
            recorded.push(
                client_a
                    .record_outcome(&outcome_request(&stored.decision_id, &stored.row_ids[0]))
                    .await
                    .unwrap()
                    .outcome_id,
            );
        }
        let mut correction = outcome_request(&stored.decision_id, &stored.row_ids[0]);
        correction.supersedes_outcome_id = recorded[0].clone();
        let correction = client_a
            .record_outcome(&correction)
            .await
            .unwrap()
            .outcome_id;
        let foreign = client_b
            .record_decision(&decision_request(minimal_decision_payload()))
            .await
            .unwrap();
        client_b
            .record_outcome(&outcome_request(&foreign.decision_id, &foreign.row_ids[0]))
            .await
            .unwrap();
        let mut query = all_outcome_query();
        query.limit = 1;
        query.schema_fingerprint = "jev1:route:v3".to_owned();
        query.model_id = "clef-flash@8b2e5fd1".to_owned();
        query.question_id = "route".to_owned();
        query.calib_filter = ExpectedCalib::Raw;
        let mut seen = std::collections::HashSet::new();
        let mut previous = None;
        let mut first_cursor = None;
        loop {
            let page = client_a
                .query_outcomes(&query)
                .await
                .expect("real signed RPC page");
            assert_eq!(page.rows.len(), 1);
            let row = &page.rows[0];
            assert_eq!(row.tenant_id, "tenant-a");
            assert!(row.calib_version.is_empty());
            let key = (row.created_at, row.outcome_id.clone());
            if let Some(previous) = previous {
                assert!(previous < key);
            }
            previous = Some(key);
            assert!(seen.insert(row.outcome_id.clone()));
            if !page.has_more {
                assert_eq!(page.next_created_at, 0);
                assert!(page.next_outcome_id.is_empty());
                break;
            }
            assert_eq!(page.next_created_at, row.created_at);
            assert_eq!(page.next_outcome_id, row.outcome_id);
            first_cursor
                .get_or_insert_with(|| (page.next_created_at, page.next_outcome_id.clone()));
            query.after_created_at = page.next_created_at;
            query.after_outcome_id = page.next_outcome_id;
            assert!(seen.len() < 5);
        }
        assert_eq!(seen.len(), 3);
        assert!(
            !seen.contains(&recorded[0]),
            "correction predicate retained on every page"
        );
        assert!(seen.contains(&correction));
        let (after_created_at, after_outcome_id) = first_cursor.unwrap();
        query.after_created_at = after_created_at;
        query.after_outcome_id = after_outcome_id;
        // Reusing tenant-a's values does not authorize tenant-a's data.
        let foreign_page = client_b.query_outcomes(&query).await.unwrap();
        assert!(foreign_page
            .rows
            .iter()
            .all(|row| row.tenant_id == "tenant-b"));
        for filter in ["schema", "model", "question", "calib"] {
            let mut filtered = query.clone();
            match filter {
                "schema" => filtered.schema_fingerprint = "other-schema".to_owned(),
                "model" => filtered.model_id = "other-model".to_owned(),
                "question" => filtered.question_id = "other_question".to_owned(),
                _ => {
                    filtered.calib_filter = ExpectedCalib::Version;
                    filtered.calib_version = "cal-1".to_owned();
                }
            }
            let page = client_a.query_outcomes(&filtered).await.unwrap();
            assert!(page.rows.is_empty(), "every-page filter {filter}");
            assert!(!page.has_more);
            assert_eq!(page.next_created_at, 0);
            assert!(page.next_outcome_id.is_empty());
        }
        for (time, id) in [(1, ""), (-1, "outc-x"), (0, "bad' OR 1=1")] {
            let mut invalid = all_outcome_query();
            invalid.after_created_at = time;
            invalid.after_outcome_id = id.to_owned();
            assert!(
                client_a.query_outcomes(&invalid).await.is_err(),
                "invalid cursor must fail generated dispatch"
            );
        }
        let mut overlong = all_outcome_query();
        overlong.after_outcome_id = "x".repeat(65);
        assert!(client_a.query_outcomes(&overlong).await.is_err());
        let mut capped = all_outcome_query();
        capped.limit = u32::MAX;
        let page = client_a.query_outcomes(&capped).await.unwrap();
        assert_eq!(page.rows.len(), 3);
        assert!(!page.has_more);
        capped.include_superseded = true;
        assert_eq!(
            client_a.query_outcomes(&capped).await.unwrap().rows.len(),
            4
        );
    }
}
