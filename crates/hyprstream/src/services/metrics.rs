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
use hyprstream_metrics::metrics::{MetricRecord, create_record_batch, get_metrics_schema};
use hyprstream_metrics::query::QueryOrchestrator;
use hyprstream_metrics::storage::view::{AggregationSpec, ViewDefinition};
#[allow(unused_imports)]
use hyprstream_metrics::StorageBackend as _;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::streaming::StreamChannel;
use hyprstream_rpc::transport::TransportConfig;
use tracing::{error, trace};

use crate::services::{Continuation, EnvelopeContext, PolicyClient, ZmqService};
use crate::services::generated::policy_client::PolicyCheck;
use crate::services::generated::metrics_client::{
    MetricsHandler, MetricsResponseVariant,
    ErrorInfo,
    AggregateRow, GroupEntry,
    HealthInfo, ViewInfo,
    IngestRequest,
    MetricQuery,
    ViewSpec,
    AggregationFunc,
    StreamInfo,
    dispatch_metrics, serialize_response,
};

// ============================================================================
// MetricsInner
// ============================================================================

struct MetricsInner {
    orchestrator: Arc<QueryOrchestrator>,
    stream_channel: StreamChannel,
    // stream_endpoint is resolved lazily from EndpointRegistry on first query_stream call.
    // This avoids requiring the registry to be initialized at service construction time.
}

// ============================================================================
// MetricsService
// ============================================================================

/// ZMQ RPC service wrapping hyprstream-metrics.
pub struct MetricsService {
    inner: Arc<MetricsInner>,
    policy_client: PolicyClient,
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
    expected_audience: Option<String>,
    local_issuer_url: Option<String>,
    federation_key_source: Option<Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,
}

impl MetricsService {
    pub fn new(
        orchestrator: Arc<QueryOrchestrator>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
    ) -> Self {
        let stream_channel = StreamChannel::new(Arc::clone(&context), signing_key.clone());

        Self {
            inner: Arc::new(MetricsInner {
                orchestrator,
                stream_channel,
            }),
            policy_client,
            context,
            transport,
            signing_key,
            expected_audience: None,
            local_issuer_url: None,
            federation_key_source: None,
        }
    }

    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    pub fn with_local_issuer_url(mut self, url: String) -> Self {
        self.local_issuer_url = Some(url);
        self
    }

    pub fn with_federation_key_source(
        mut self,
        src: Arc<dyn hyprstream_rpc::auth::FederationKeySource>,
    ) -> Self {
        self.federation_key_source = Some(src);
        self
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
        if i == 0 { c.is_ascii_alphabetic() || c == '_' }
        else { c.is_ascii_alphanumeric() || c == '_' }
    });
    if !valid {
        anyhow::bail!("invalid identifier {:?}: only [a-zA-Z_][a-zA-Z0-9_]* allowed", name);
    }
    Ok(())
}

fn build_sql(q: &MetricQuery) -> Result<String> {
    if !q.sql.is_empty() {
        // Raw SQL: caller must have been granted query scope by the dispatcher.
        return Ok(q.sql.clone());
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
    // AggregateRow.timestamp will be 0 for structured aggregate queries (use raw SQL to project it).
    // NOTE: timestamp is stored and compared in milliseconds (Unix epoch ms).
    // The MetricRecord.timestamp field, ingest path, and window filter all use ms.
    let mut sql = format!(
        "SELECT {agg} AS value{group_select} FROM metrics"
    );

    if !q.metric_id.is_empty() {
        sql.push_str(&format!(
            " WHERE metric_id = '{}'",
            q.metric_id.replace('\'', "''")
        ));
    }

    if q.window_secs > 0 {
        let clause = if q.metric_id.is_empty() { "WHERE" } else { "AND" };
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
// MetricsHandler (generated dispatch trait)
// ============================================================================

#[async_trait(?Send)]
impl MetricsHandler for MetricsService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject().to_string();
        let allowed = self.policy_client
            .check(&PolicyCheck {
                subject: subject.clone(),
                domain: "*".to_owned(),
                resource: resource.to_owned(),
                operation: operation.to_owned(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Policy check error for {}: {}", subject, e))?;
        if allowed {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
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

        let rust_records: Vec<MetricRecord> = data.records
            .iter()
            .map(|r| MetricRecord {
                metric_id: r.metric_id.clone(),
                timestamp: r.timestamp,
                value_running_window_sum: r.value_window_sum,
                value_running_window_avg: r.value_window_avg,
                value_running_window_count: r.value_window_count,
            })
            .collect();

        let batch = create_record_batch(&rust_records)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        self.inner
            .orchestrator
            .storage()
            .insert_into_table("metrics", batch)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(MetricsResponseVariant::IngestResult)
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
            let value_col = batch
                .column_by_name("value")
                .and_then(|c| c.as_any().downcast_ref::<hyprstream_metrics::arrow::array::Float64Array>());
            let ts_col = batch
                .column_by_name("timestamp")
                .and_then(|c| c.as_any().downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>());

            for i in 0..n {
                let value = value_col.map(|c| c.value(i)).unwrap_or(0.0);
                let timestamp = ts_col.map(|c| c.value(i)).unwrap_or(0);

                let mut group_values: Vec<GroupEntry> = Vec::new();
                for col_name in &q.group_by {
                    let col = batch.column_by_name(col_name.as_str())
                        .ok_or_else(|| anyhow::anyhow!("group_by column {:?} not found in result", col_name))?;
                    let val = col.as_any()
                        .downcast_ref::<hyprstream_metrics::arrow::array::StringArray>()
                        .ok_or_else(|| anyhow::anyhow!(
                            "group_by column {:?} has type {:?}; only Utf8/String columns are \
                             supported for group_by (metrics schema: metric_id is Utf8; \
                             timestamp and value_running_window_count are Int64 — not groupable)",
                            col_name, col.data_type()
                        ))?
                        .value(i)
                        .to_owned();
                    group_values.push(GroupEntry {
                        key: col_name.clone(),
                        value: val,
                    });
                }

                rows.push(AggregateRow { value, group_values, timestamp });
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
            .prepare_stream(&q.ephemeral_pubkey, 600)
            .await?;

        let stream_info = StreamInfo {
            stream_id: stream_ctx.stream_id().to_owned(),
            endpoint: self.inner.stream_channel.stream_endpoint(),
            server_pubkey: *stream_ctx.server_pubkey(),
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

                            writer.finish().map_err(|e| anyhow::anyhow!("IPC finish: {e}"))?;
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
                        let v = c.as_any()
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
// ZmqService
// ============================================================================

#[async_trait(?Send)]
impl ZmqService for MetricsService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        trace!("Metrics request from {} (id={})", ctx.subject(), ctx.request_id);
        dispatch_metrics(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "metrics"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
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

    fn local_issuer_url(&self) -> Option<&str> {
        self.local_issuer_url.as_deref()
    }

    fn federation_key_source(
        &self,
    ) -> Option<Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
        self.federation_key_source.clone()
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
    use std::sync::Arc;

    use hyprstream_metrics::query::QueryOrchestrator;
    use hyprstream_metrics::storage::duckdb::DuckDbBackend;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::envelope::RequestIdentity;
    use hyprstream_rpc::transport::TransportConfig;
    use hyprstream_service::{InprocManager, ServiceManager};

    use crate::auth::PolicyManager;
    use crate::services::generated::metrics_client::{
        AggregationFunc, IngestRequest, MetricQuery, MetricRecord as CMetricRecord,
        MetricsClient, ViewSpec,
    };
    use crate::services::{PolicyClient, PolicyService};
    use crate::zmq::global_context;

    /// Spin up an in-memory MetricsService and return a typed client + InprocManager handle.
    async fn start_metrics_service(
        tag: &str,
    ) -> (MetricsClient, InprocManager) {
        let (signing_key, _vk) = generate_signing_keypair();
        let context = global_context();

        // Permissive policy service so authorize() always passes.
        let policy_tag = format!("test-policy-{tag}");
        let policy_manager = Arc::new(
            PolicyManager::permissive()
                .await
                .expect("permissive policy manager"),
        );
        let git2db = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(tempfile::TempDir::new().unwrap().path())
                .await
                .expect("git2db open"),
        ));
        let policy_svc = PolicyService::new(
            policy_manager,
            Arc::new(signing_key.clone()),
            crate::config::TokenConfig::default(),
            git2db,
            context.clone(),
            TransportConfig::inproc(&policy_tag),
        );
        let manager = InprocManager::new();
        manager
            .spawn(Box::new(policy_svc))
            .await
            .expect("spawn policy service");

        let policy_client = PolicyClient::with_endpoint(
            &format!("inproc://{policy_tag}"),
            signing_key.clone(),
            RequestIdentity::anonymous(),
        );

        // In-memory DuckDB backend + metrics table creation.
        let backend = Arc::new(
            DuckDbBackend::new(":memory:".to_owned(), Default::default(), None)
                .expect("DuckDbBackend"),
        );
        // Create the metrics table before handing backend to orchestrator.
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
            context.clone(),
            TransportConfig::inproc(&svc_tag),
            signing_key.clone(),
            policy_client,
        );
        manager
            .spawn(Box::new(service))
            .await
            .expect("spawn metrics service");

        let client = MetricsClient::with_endpoint(
            &format!("inproc://{svc_tag}"),
            signing_key,
            RequestIdentity::anonymous(),
        );

        (client, manager)
    }

    // ── health ────────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_health_fresh_db() {
        let (client, _mgr) = start_metrics_service("health-fresh").await;
        let info = client.health().await.expect("health on fresh DB");
        // Table exists (created in test helper) but is empty.
        assert!(info.ok, "expected ok=true — table exists even when empty");
        assert_eq!(info.row_count, 0, "expected 0 rows on fresh DB");
        assert_eq!(info.backend_type, "duckdb");
    }

    // ── ingest + health ───────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_ingest_and_health() {
        let (client, _mgr) = start_metrics_service("ingest-health").await;

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
                },
                CMetricRecord {
                    metric_id: "cpu.usage".to_owned(),
                    timestamp: now + 1000,
                    value_window_sum: 58.0,
                    value_window_avg: 58.0,
                    value_window_count: 1,
                },
            ],
        };

        client.ingest(&req).await.expect("ingest");

        let info = client.health().await.expect("health after ingest");
        assert!(info.ok, "expected ok=true after ingest");
        assert_eq!(info.row_count, 2, "expected 2 rows");
    }

    // ── query ─────────────────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_count() {
        let (client, _mgr) = start_metrics_service("query-count").await;

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
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_sum() {
        let (client, _mgr) = start_metrics_service("query-sum").await;

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
                    },
                    CMetricRecord {
                        metric_id: "disk.io".to_owned(),
                        timestamp: now + 500,
                        value_window_sum: 200.0,
                        value_window_avg: 200.0,
                        value_window_count: 1,
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
        assert!((rows[0].value - 300.0).abs() < 1e-6, "SUM should be 300.0, got {}", rows[0].value);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_query_raw_sql() {
        let (client, _mgr) = start_metrics_service("query-raw").await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        client
            .ingest(&IngestRequest {
                records: vec![CMetricRecord {
                    metric_id: "net.rx".to_owned(),
                    timestamp: now,
                    value_window_sum: 500.0,
                    value_window_avg: 500.0,
                    value_window_count: 1,
                }],
            })
            .await
            .expect("ingest");

        let rows = client
            .query(&MetricQuery {
                sql: "SELECT CAST(COUNT(*) AS DOUBLE) AS value FROM metrics".to_owned(),
                metric_id: String::new(),
                window_secs: 0,
                aggregation: AggregationFunc::RawSql,
                group_by: vec![],
                limit_rows: 0,
                ephemeral_pubkey: vec![],
            })
            .await
            .expect("query raw sql");

        assert_eq!(rows.len(), 1, "raw SQL COUNT should return one row");
        assert_eq!(rows[0].value as u64, 1, "raw SQL COUNT should be 1");
    }

    // ── view management ───────────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_create_list_drop_view() {
        let (client, _mgr) = start_metrics_service("views").await;

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
        let (client, _mgr) = start_metrics_service("validation").await;

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
}
