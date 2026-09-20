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
