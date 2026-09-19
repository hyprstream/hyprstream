//! Producer-side write API for schema-v2 metric rows.
//!
//! `MetricsProducer` is the first real producer path into the metrics store:
//! it constructs labeled/tenanted [`MetricRecord`]s (merging configured
//! defaults) and writes them through any [`StorageBackend`]. The metrics
//! service's ingest handler is the wire-facing adapter onto the same records;
//! the serving path (P3.4) emits decision-quality rows the same way.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

use super::calibration::CalibrationStore;
use super::decision_batch::DecisionBatchStore;
use super::{METRICS_TABLE, MetricRecord, create_record_batch};
use crate::storage::StorageBackend;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::Status;

/// Current Unix epoch time in milliseconds (the `MetricRecord.timestamp` unit).
pub fn now_unix_ms() -> i64 {
    let ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    i64::try_from(ms).unwrap_or(i64::MAX)
}

/// Producer of schema-v2 metric rows over any [`StorageBackend`].
///
/// Default labels/tenant are merged into every emitted record; per-record
/// values win on conflict.
pub struct MetricsProducer<B: StorageBackend> {
    backend: Arc<B>,
    default_labels: BTreeMap<String, String>,
    default_tenant: Option<String>,
}

impl<B: StorageBackend> MetricsProducer<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            backend,
            default_labels: BTreeMap::new(),
            default_tenant: None,
        }
    }

    /// Add a default label applied to every emitted record.
    pub fn with_default_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_labels.insert(key.into(), value.into());
        self
    }

    /// Set a default tenant applied to every emitted record.
    pub fn with_default_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.default_tenant = Some(tenant.into());
        self
    }

    /// Create all schema-v2 tables (`metrics`, `calibration_params`,
    /// `decision_batches`) and migrate pre-v2 `metrics` tables in place.
    pub async fn init(&self) -> Result<(), Status> {
        super::create_schema_v2_tables(&*self.backend).await
    }

    /// Build a point-sample record (window of one: sum == avg == value) with
    /// this producer's default labels/tenant pre-applied.
    pub fn point(&self, metric_id: impl Into<String>, value: f64) -> MetricRecord {
        MetricRecord {
            metric_id: metric_id.into(),
            timestamp: now_unix_ms(),
            value_running_window_sum: value,
            value_running_window_avg: value,
            value_running_window_count: 1,
            labels: self.default_labels.clone(),
            tenant_id: self.default_tenant.clone(),
        }
    }

    /// Emit one record. Producer defaults fill any label/tenant the record
    /// does not set itself.
    pub async fn emit(&self, mut record: MetricRecord) -> Result<(), Status> {
        for (key, value) in &self.default_labels {
            record.labels.entry(key.clone()).or_insert_with(|| value.clone());
        }
        if record.tenant_id.is_none() {
            record.tenant_id.clone_from(&self.default_tenant);
        }
        if record.metric_id.is_empty() {
            return Err(Status::invalid_argument("metric_id must not be empty"));
        }
        let batch = create_record_batch(&[record])?;
        self.backend.insert_into_table(METRICS_TABLE, batch).await
    }

    /// Emit a batch of records (one insert), applying the same default-merge
    /// as [`emit`](Self::emit).
    pub async fn emit_all(
        &self,
        records: impl IntoIterator<Item = MetricRecord>,
    ) -> Result<(), Status> {
        let mut merged = Vec::new();
        for mut record in records {
            for (key, value) in &self.default_labels {
                record.labels.entry(key.clone()).or_insert_with(|| value.clone());
            }
            if record.tenant_id.is_none() {
                record.tenant_id.clone_from(&self.default_tenant);
            }
            if record.metric_id.is_empty() {
                return Err(Status::invalid_argument("metric_id must not be empty"));
            }
            merged.push(record);
        }
        if merged.is_empty() {
            return Ok(());
        }
        let batch = create_record_batch(&merged)?;
        self.backend.insert_into_table(METRICS_TABLE, batch).await
    }

    /// Calibration parameter writer/reader on the same backend (P2.2 re-fit).
    pub fn calibration(&self) -> CalibrationStore<B> {
        CalibrationStore::new(Arc::clone(&self.backend))
    }

    /// Arrow-native decision-batch store on the same backend.
    pub fn decision_batches(&self) -> DecisionBatchStore<B> {
        DecisionBatchStore::new(Arc::clone(&self.backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::{AggregateFunction, GroupBy};
    use crate::metrics::calibration::DecisionPrimitive;
    use crate::metrics::storage::{MetricsStorage, MetricsStorageImpl};
    use crate::storage::duckdb::DuckDbBackend;
    use duckdb::arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use duckdb::arrow::datatypes::{DataType, Field, Schema};

    #[tokio::test]
    async fn test_producer_emit_with_default_labels_and_tenant() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let producer = MetricsProducer::new(backend.clone())
            .with_default_label("service", "inference")
            .with_default_label("primitive", "choice")
            .with_default_tenant("tenant-a");
        producer.init().await?;

        // Defaults apply.
        producer.emit(producer.point("decision.latency_ms", 6.5)).await?;
        // Per-record values win over defaults.
        let mut record = producer.point("decision.latency_ms", 7.0);
        record.labels.insert("primitive".to_owned(), "score".to_owned());
        record.tenant_id = Some("tenant-b".to_owned());
        producer.emit(record).await?;
        // A bare producer emits NULL labels/tenant.
        let bare = MetricsProducer::new(backend.clone());
        bare.emit(bare.point("requests.total", 1.0)).await?;

        let handle = backend
            .prepare_sql("SELECT * FROM metrics ORDER BY timestamp ASC")
            .await?;
        let batch = backend.query_sql(&handle).await?;
        let rows = MetricRecord::try_from_record_batch(&batch)?;
        assert_eq!(rows.len(), 3);

        assert_eq!(rows[0].tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(
            rows[0].labels,
            BTreeMap::from([
                ("primitive".to_owned(), "choice".to_owned()),
                ("service".to_owned(), "inference".to_owned()),
            ])
        );

        assert_eq!(rows[1].tenant_id.as_deref(), Some("tenant-b"));
        assert_eq!(rows[1].labels.get("primitive").map(String::as_str), Some("score"));
        assert_eq!(
            rows[1].labels.get("service").map(String::as_str),
            Some("inference"),
            "unset keys still inherit defaults"
        );

        assert_eq!(rows[2].tenant_id, None);
        assert!(rows[2].labels.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_group_by_tenant_id() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        let producer = MetricsProducer::new(backend.clone());
        producer.init().await?;

        let mut a = producer.point("m", 1.0);
        a.tenant_id = Some("tenant-a".to_owned());
        let mut b = producer.point("m", 2.0);
        b.tenant_id = Some("tenant-b".to_owned());
        let global = producer.point("m", 4.0);
        producer.emit_all(vec![a, b, global]).await?;

        let storage = MetricsStorageImpl::new((*backend).clone());
        let results = storage
            .aggregate_metrics(
                AggregateFunction::Count,
                &GroupBy {
                    columns: vec!["tenant_id".to_owned()],
                    time_column: None,
                },
                0,
                None,
            )
            .await?;
        assert_eq!(results.len(), 3, "one group per tenant incl. the NULL group");
        let by_tenant: BTreeMap<String, f64> = results
            .into_iter()
            .map(|r| (r.group_values.get("tenant_id").cloned().unwrap_or_default(), r.value))
            .collect();
        assert_eq!(by_tenant.get("tenant-a"), Some(&1.0));
        assert_eq!(by_tenant.get("tenant-b"), Some(&1.0));
        Ok(())
    }

    #[tokio::test]
    async fn test_init_migrates_pre_v2_metrics_table() -> Result<(), Status> {
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);

        // Simulate a pre-v2 database: 5-column metrics table with a legacy row.
        backend
            .execute_ddl(
                "CREATE TABLE metrics (\
                 metric_id VARCHAR, timestamp BIGINT, \
                 value_running_window_sum DOUBLE PRECISION, \
                 value_running_window_avg DOUBLE PRECISION, \
                 value_running_window_count BIGINT)",
            )
            .await?;
        let v1_schema = Arc::new(Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]));
        let v1_batch = RecordBatch::try_new(
            v1_schema,
            vec![
                Arc::new(StringArray::from(vec!["legacy.metric"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![123])) as ArrayRef,
                Arc::new(Float64Array::from(vec![9.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![9.0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))?;
        backend.insert_into_table(METRICS_TABLE, v1_batch).await?;

        // Schema-v2 init adds the new columns and creates the new tables.
        let producer = MetricsProducer::new(backend.clone());
        producer.init().await?;

        // Legacy row still reads back, with v2 defaults.
        let storage = MetricsStorageImpl::new((*backend).clone());
        let rows = storage.query_metrics(0).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].metric_id, "legacy.metric");
        assert!(rows[0].labels.is_empty());
        assert_eq!(rows[0].tenant_id, None);

        // New v2 rows insert alongside.
        let mut record = producer.point("decision.ece", 0.03);
        record.tenant_id = Some("tenant-a".to_owned());
        producer.emit(record).await?;
        let rows = storage.query_metrics(0).await?;
        assert_eq!(rows.len(), 2);

        // The new tables exist and are usable.
        assert_eq!(
            producer
                .calibration()
                .history("s", "m", DecisionPrimitive::Noul)
                .await?
                .len(),
            0
        );
        assert_eq!(producer.decision_batches().list(None).await?.len(), 0);
        Ok(())
    }
}
