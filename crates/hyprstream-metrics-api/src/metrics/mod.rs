// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

pub mod calibration;
pub mod decision_batch;
pub mod producer;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tonic::Status;

/// Primary metrics table name.
pub const METRICS_TABLE: &str = "metrics";

/// Version of the metrics table schema emitted by [`get_metrics_schema`].
/// v1: five running-window columns; v2: adds `labels` and `tenant_id`.
pub const METRICS_SCHEMA_VERSION: u32 = 2;

/// A single metric record with running window calculations.
///
/// Schema v2 adds `labels` and `tenant_id`; both default to empty/`None` so
/// v1 producers and serialized records keep working unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp in milliseconds
    pub timestamp: i64,
    /// Running sum within the window
    pub value_running_window_sum: f64,
    /// Running average within the window
    pub value_running_window_avg: f64,
    /// Running count within the window
    pub value_running_window_count: i64,
    /// Metric labels; stored as a JSON object in the nullable `labels` column
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    /// Tenant the row belongs to; `None` = node-global (no tenant)
    #[serde(default)]
    pub tenant_id: Option<String>,
}

/// Serialize labels to their canonical storage form: a JSON object with
/// sorted keys. Empty label sets map to `None` (SQL NULL on write).
pub fn encode_labels(labels: &BTreeMap<String, String>) -> Option<String> {
    if labels.is_empty() {
        None
    } else {
        serde_json::to_string(labels).ok()
    }
}

/// Parse the JSON storage form of a label map.
pub fn decode_labels(json: &str) -> Result<BTreeMap<String, String>, Status> {
    serde_json::from_str(json)
        .map_err(|e| Status::internal(format!("Invalid labels JSON: {e}")))
}

impl MetricRecord {
    pub fn try_from_record_batch(batch: &RecordBatch) -> Result<Vec<Self>, Status> {
        let metric_ids = batch
            .column_by_name("metric_id")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

        let sums = batch
            .column_by_name("value_running_window_sum")
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

        let avgs = batch
            .column_by_name("value_running_window_avg")
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

        let counts = batch
            .column_by_name("value_running_window_count")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

        // Schema v2 columns are optional so pre-v2 batches still decode.
        let labels_col = batch
            .column_by_name("labels")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>());
        let tenant_col = batch
            .column_by_name("tenant_id")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>());

        let mut metrics = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let labels = match labels_col {
                Some(col) if !col.is_null(i) && !col.value(i).is_empty() => {
                    decode_labels(col.value(i))?
                }
                _ => BTreeMap::new(),
            };
            let tenant_id = tenant_col.and_then(|col| {
                if col.is_null(i) || col.value(i).is_empty() {
                    None
                } else {
                    Some(col.value(i).to_owned())
                }
            });
            metrics.push(MetricRecord {
                metric_id: metric_ids.value(i).to_owned(),
                timestamp: timestamps.value(i),
                value_running_window_sum: sums.value(i),
                value_running_window_avg: avgs.value(i),
                value_running_window_count: counts.value(i),
                labels,
                tenant_id,
            });
        }

        Ok(metrics)
    }
}

/// Gets the schema for metric records in Arrow format (schema v2).
pub fn get_metrics_schema() -> Schema {
    Schema::new(vec![
        Field::new("metric_id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value_running_window_sum", DataType::Float64, false),
        Field::new("value_running_window_avg", DataType::Float64, false),
        Field::new("value_running_window_count", DataType::Int64, false),
        // v2: label map as canonical JSON (see encode_labels) and tenant key.
        // Nullable so pre-v2 rows and writers stay valid; DuckDB stores both
        // as VARCHAR, queryable with the built-in JSON functions.
        Field::new("labels", DataType::Utf8, true),
        Field::new("tenant_id", DataType::Utf8, true),
    ])
}

/// Creates a RecordBatch from a vector of MetricRecords.
pub fn create_record_batch(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
    let schema = get_metrics_schema();

    let metric_ids = StringArray::from_iter_values(metrics.iter().map(|m| m.metric_id.as_str()));
    let timestamps = Int64Array::from_iter_values(metrics.iter().map(|m| m.timestamp));
    let sums = Float64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_sum));
    let avgs = Float64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_avg));
    let counts = Int64Array::from_iter_values(metrics.iter().map(|m| m.value_running_window_count));
    let labels = StringArray::from_iter(metrics.iter().map(|m| encode_labels(&m.labels)));
    let tenant_ids = StringArray::from_iter(metrics.iter().map(|m| m.tenant_id.as_deref()));

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(metric_ids),
        Arc::new(timestamps),
        Arc::new(sums),
        Arc::new(avgs),
        Arc::new(counts),
        Arc::new(labels),
        Arc::new(tenant_ids),
    ];

    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|e| Status::internal(format!("Failed to create record batch: {e}")))
}

/// Encodes a RecordBatch into a vector of MetricRecords.
pub fn encode_record_batch(batch: &RecordBatch) -> Result<Vec<MetricRecord>, Status> {
    MetricRecord::try_from_record_batch(batch)
}

/// Creates all schema-v2 tables and migrates pre-v2 `metrics` tables in place.
///
/// `metrics` is created with the v2 schema; `calibration_params` and
/// `decision_batches` are created fresh. On databases initialized before v2,
/// the new `metrics` columns are added with `ALTER TABLE ... IF NOT EXISTS`
/// (a no-op on fresh v2 tables); existing rows read back with empty
/// labels/no tenant.
pub async fn create_schema_v2_tables(
    backend: &dyn crate::storage::StorageBackend,
) -> Result<(), Status> {
    backend.create_table(METRICS_TABLE, &get_metrics_schema()).await?;
    backend
        .create_table(
            calibration::CALIBRATION_TABLE,
            &calibration::calibration_params_schema(),
        )
        .await?;
    backend
        .create_table(
            decision_batch::DECISION_BATCH_TABLE,
            &decision_batch::decision_batch_schema(),
        )
        .await?;
    for column in ["labels", "tenant_id"] {
        backend
            .execute_ddl(&format!(
                "ALTER TABLE {METRICS_TABLE} ADD COLUMN IF NOT EXISTS {column} VARCHAR"
            ))
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled_record() -> MetricRecord {
        MetricRecord {
            metric_id: "decision.latency_ms".to_owned(),
            timestamp: 1_758_000_000_000,
            value_running_window_sum: 12.5,
            value_running_window_avg: 12.5,
            value_running_window_count: 1,
            labels: BTreeMap::from([
                ("head".to_owned(), "routing".to_owned()),
                ("primitive".to_owned(), "choice".to_owned()),
            ]),
            tenant_id: Some("tenant-a".to_owned()),
        }
    }

    #[test]
    fn test_metrics_schema_v2_columns() -> Result<(), Status> {
        let schema = get_metrics_schema();
        assert_eq!(schema.fields().len(), 7);
        let labels = schema
            .field_with_name("labels")
            .map_err(|e| Status::internal(e.to_string()))?;
        assert!(labels.is_nullable());
        assert_eq!(labels.data_type(), &DataType::Utf8);
        let tenant = schema
            .field_with_name("tenant_id")
            .map_err(|e| Status::internal(e.to_string()))?;
        assert!(tenant.is_nullable());
        assert_eq!(tenant.data_type(), &DataType::Utf8);
        Ok(())
    }

    #[test]
    fn test_record_batch_round_trip_with_labels_and_tenant() -> Result<(), Status> {
        let records = vec![
            labeled_record(),
            MetricRecord {
                metric_id: "requests.total".to_owned(),
                timestamp: 1_758_000_001_000,
                value_running_window_sum: 3.0,
                value_running_window_avg: 1.5,
                value_running_window_count: 2,
                labels: BTreeMap::new(),
                tenant_id: None,
            },
        ];
        let batch = create_record_batch(&records)?;
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 7);

        // Labels land as NULL (not empty string) when unset.
        let labels_col = batch
            .column_by_name("labels")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("labels column missing"))?;
        assert!(labels_col.is_null(1));
        assert!(!labels_col.is_null(0));
        assert_eq!(
            labels_col.value(0),
            r#"{"head":"routing","primitive":"choice"}"#
        );

        let decoded = MetricRecord::try_from_record_batch(&batch)?;
        assert_eq!(decoded, records);

        let decoded_via_encode = encode_record_batch(&batch)?;
        assert_eq!(decoded_via_encode, records);
        Ok(())
    }

    #[test]
    fn test_pre_v2_batch_decodes_with_defaults() -> Result<(), Status> {
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
                Arc::new(StringArray::from(vec!["cpu.usage"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![42])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            ],
        )
        .map_err(|e| Status::internal(e.to_string()))?;

        let decoded = MetricRecord::try_from_record_batch(&v1_batch)?;
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].labels.is_empty());
        assert_eq!(decoded[0].tenant_id, None);
        assert_eq!(decoded[0].metric_id, "cpu.usage");
        Ok(())
    }

    #[test]
    fn test_labels_json_canonical_order() {
        let labels = BTreeMap::from([
            ("zeta".to_owned(), "1".to_owned()),
            ("alpha".to_owned(), "2".to_owned()),
        ]);
        assert_eq!(
            encode_labels(&labels).as_deref(),
            Some(r#"{"alpha":"2","zeta":"1"}"#)
        );
        assert_eq!(encode_labels(&BTreeMap::new()), None);
    }

    #[test]
    fn test_default_record_is_unlabeled_and_untenanted() {
        let record = MetricRecord::default();
        assert!(record.labels.is_empty());
        assert!(record.tenant_id.is_none());
    }
}
