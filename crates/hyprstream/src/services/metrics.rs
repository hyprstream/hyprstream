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
use futures::StreamExt;
use hyprstream_metrics::aggregation::{AggregateFunction, GroupBy, TimeWindow};
use hyprstream_metrics::metrics::{MetricRecord, create_record_batch, get_metrics_schema};
use hyprstream_metrics::query::QueryOrchestrator;
use hyprstream_metrics::storage::view::{AggregationSpec, ViewDefinition};
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
    stream_endpoint: String,
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
        let stream_endpoint = stream_channel.stream_endpoint();

        Self {
            inner: Arc::new(MetricsInner {
                orchestrator,
                stream_channel,
                stream_endpoint,
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
        AggregationFunc::Min => "MIN(value_running_window_sum)",
        AggregationFunc::Max => "MAX(value_running_window_sum)",
    };

    let group_select = if q.group_by.is_empty() {
        String::new()
    } else {
        format!(", {}", q.group_by.join(", "))
    };

    let mut sql = format!(
        "SELECT {agg} AS value, timestamp{group_select} FROM metrics"
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
        if ctx.identity.is_local() {
            return Ok(());
        }
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
        let batches = self
            .inner
            .orchestrator
            .query_collect(&sql)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

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
                            "group_by column {:?} has non-String type {:?}; only String columns are supported",
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
        let sql = build_sql(q)?;

        let stream_ctx = self
            .inner
            .stream_channel
            .prepare_stream(&q.ephemeral_pubkey, 600)
            .await?;

        let stream_info = StreamInfo {
            stream_id: stream_ctx.stream_id().to_owned(),
            endpoint: self.inner.stream_endpoint.clone(),
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
                            let mut query_stream = inner
                                .orchestrator
                                .query(&sql)
                                .await
                                .map_err(|e| anyhow::anyhow!("{e}"))?;

                            // Accumulate all batches into one Arrow IPC stream so
                            // clients can open a single StreamReader over the result.
                            // Even zero-row results must emit a valid IPC stream
                            // (schema + EOS marker) so the client StreamReader initialises.
                            let metrics_schema = Arc::new(get_metrics_schema());
                            let mut buf = Vec::new();
                            let mut writer =
                                hyprstream_metrics::arrow::ipc::writer::StreamWriter::try_new(
                                    &mut buf,
                                    &metrics_schema,
                                )
                                .map_err(|e| anyhow::anyhow!("IPC writer: {e}"))?;

                            while let Some(batch_result) = query_stream.next().await {
                                let batch = batch_result.map_err(|e| anyhow::anyhow!("{e}"))?;
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
        if !q.sql.is_empty() {
            anyhow::bail!("create_view with raw SQL is not supported; use structured query fields");
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
        let result = self
            .inner
            .orchestrator
            .query_collect("SELECT COUNT(*) AS cnt FROM metrics")
            .await;

        let (ok, row_count) = match result {
            Ok(batches) => {
                let cnt = batches
                    .first()
                    .and_then(|b| b.column_by_name("cnt"))
                    .and_then(|c| c.as_any().downcast_ref::<hyprstream_metrics::arrow::array::Int64Array>())
                    .map(|c| c.value(0) as u64)
                    .unwrap_or(0);
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
