//! Engine-side integration tests for the api crate's stores.
//!
//! `CalibrationStore`, `DecisionBatchStore`, and `MetricsProducer` are defined
//! in the Apache-2.0 `hyprstream-metrics-api` crate over the `StorageBackend`
//! trait; these tests exercise them against the DuckDB engine backend — the
//! serving wiring — including the NULL round-trip and version-triple
//! regressions from review round 1.

use crate::aggregation::{AggregateFunction, GroupBy};
use crate::metrics::calibration::{
    CalibrationParams, CalibrationStore, ConformalParams, ConformalQuantile, DecisionPrimitive,
    DirichletParams, OrdinalInterval, VersionTriple, mint_calib_version,
};
use crate::metrics::decision_batch::DecisionBatchStore;
use crate::metrics::producer::MetricsProducer;
use crate::metrics::storage::{MetricsStorage, MetricsStorageImpl};
use crate::metrics::{METRICS_TABLE, MetricRecord};
use crate::storage::StorageBackend;
use crate::storage::duckdb::DuckDbBackend;
use duckdb::arrow::array::{
    ArrayRef, BooleanArray, Float32Builder, Float64Array, Int64Array, ListBuilder, RecordBatch,
    StringArray, UInt32Array,
};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use std::collections::BTreeMap;
use std::sync::Arc;
use tonic::Status;

fn params(primitive: DecisionPrimitive, calib_version: &str, fitted_at: i64) -> CalibrationParams {
    CalibrationParams {
        version: VersionTriple {
            schema_id: "jev1:q1".to_owned(),
            model_id: "qwen3.5-0.8b@main".to_owned(),
            calib_version: calib_version.to_owned(),
        },
        primitive,
        temperature: Some(1.25),
        null_bias: Some(vec![0.1, -0.2, 0.05]),
        dirichlet: None,
        conformal: None,
        cardinality: Some(3),
        calib_set_id: "calibset-2026-09".to_owned(),
        calib_size: Some(10_000),
        objective: Some("nll".to_owned()),
        fitted_at,
    }
}

#[tokio::test]
async fn test_insert_and_latest_round_trip() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = CalibrationStore::new(backend);
    store.init().await?;

    let mut older = params(DecisionPrimitive::Choice, "cal-v1", 1_000);
    older.dirichlet = Some(DirichletParams {
        weights: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        bias: vec![0.1, 0.0, -0.1],
    });
    older.conformal = Some(ConformalParams::Quantiles {
        quantiles: vec![
            ConformalQuantile { level: 0.8, quantile: 0.31 },
            ConformalQuantile { level: 0.9, quantile: 0.47 },
        ],
    });
    store.insert(&older).await?;

    let mut newer = params(DecisionPrimitive::Choice, "cal-v2", 2_000);
    newer.temperature = Some(0.9);
    store.insert(&newer).await?;

    let latest = store
        .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
        .await?
        .ok_or_else(|| Status::internal("latest row missing"))?;
    assert_eq!(latest.version.calib_version, "cal-v2");
    assert_eq!(latest.temperature, Some(0.9));

    let history = store
        .history("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
        .await?;
    assert_eq!(history.len(), 2);
    assert_eq!(history[1], older, "older row keeps its exact params");

    // Score rows are separate per primitive.
    let mut score = params(DecisionPrimitive::Score, "cal-v1", 1_500);
    score.conformal = Some(ConformalParams::OrdinalIntervals {
        intervals: vec![OrdinalInterval {
            level: 0.9,
            lower: 1.0,
            upper: 2.0,
        }],
    });
    store.insert(&score).await?;
    let score_latest = store
        .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Score)
        .await?
        .ok_or_else(|| Status::internal("score row missing"))?;
    assert_eq!(
        score_latest.conformal,
        Some(ConformalParams::OrdinalIntervals {
            intervals: vec![OrdinalInterval {
                level: 0.9,
                lower: 1.0,
                upper: 2.0,
            }],
        })
    );

    // Unknown (schema, model, primitive) reads as empty.
    assert!(store
        .latest("jev1:other", "qwen3.5-0.8b@main", DecisionPrimitive::Noul)
        .await?
        .is_none());
    Ok(())
}

/// Regression: nullable Int64/Float64 columns must persist as SQL NULL,
/// not the zeroed buffer slot of `array.value()` on a null.
#[tokio::test]
async fn test_none_numeric_params_round_trip_as_null() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = CalibrationStore::new(backend);
    store.init().await?;

    let mut row = params(DecisionPrimitive::Choice, "cal-v1", 1_000);
    row.temperature = None; // None = identity
    row.cardinality = None;
    row.calib_size = None;
    row.validate()?;
    store.insert(&row).await?;

    let back = store
        .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Choice)
        .await?
        .ok_or_else(|| Status::internal("row missing"))?;
    assert_eq!(back.temperature, None, "identity temperature must stay NULL");
    assert_eq!(back.cardinality, None);
    assert_eq!(back.calib_size, None);
    assert_eq!(back.null_bias, row.null_bias, "set fields still round-trip");
    // The store must not persist rows its own validation would reject.
    back.validate()
}

#[tokio::test]
async fn test_insert_rejects_duplicate_version_triple() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = CalibrationStore::new(backend);
    store.init().await?;

    let row = params(DecisionPrimitive::Noul, "cal-v1", 1_000);
    store.insert(&row).await?;
    let err = match store.insert(&row).await {
        Err(err) => err,
        Ok(()) => {
            return Err(Status::internal("duplicate version triple must be rejected"));
        }
    };
    assert_eq!(err.code(), tonic::Code::AlreadyExists);

    let mut refit = row.clone();
    refit.calib_size = Some(20_000);
    refit.version.calib_version = mint_calib_version();
    store.insert(&refit).await?;
    Ok(())
}

#[tokio::test]
async fn test_insert_rejects_invalid_params() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = CalibrationStore::new(backend);
    store.init().await?;

    let mut bad = params(DecisionPrimitive::Score, "cal-v1", 1_000);
    bad.temperature = Some(-1.0);
    assert!(store.insert(&bad).await.is_err());
    // Nothing was written.
    assert!(store
        .latest("jev1:q1", "qwen3.5-0.8b@main", DecisionPrimitive::Score)
        .await?
        .is_none());
    Ok(())
}
/// A batch shaped like a model-emitted decision result: per-question
/// probabilities, chosen option, abstention flag — exercising types beyond
/// the plain scalar columns (nested lists, bool, u32).
fn model_emitted_batch() -> Result<RecordBatch, Status> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("question_id", DataType::Utf8, false),
        Field::new("head", DataType::Utf8, false),
        Field::new("choice", DataType::UInt32, true),
        Field::new(
            "probs",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
        Field::new("abstained", DataType::Boolean, false),
    ]));
    let mut probs_builder = ListBuilder::new(Float32Builder::new());
    for probs in [vec![0.7f32, 0.2, 0.1], vec![0.4, 0.35, 0.25]] {
        let values = probs_builder.values();
        for p in probs {
            values.append_value(p);
        }
        probs_builder.append(true);
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["q-1", "q-2"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["routing", "routing"])) as ArrayRef,
            Arc::new(UInt32Array::from(vec![Some(0), None])) as ArrayRef,
            Arc::new(probs_builder.finish()) as ArrayRef,
            Arc::new(BooleanArray::from(vec![false, true])) as ArrayRef,
        ],
    )
    .map_err(|e| Status::internal(e.to_string()))
}

#[tokio::test]
async fn test_put_get_round_trip_preserves_batch() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = DecisionBatchStore::new(backend);
    store.init().await?;

    let batch = model_emitted_batch()?;
    let batch_id = store
        .put("jev1:routing:v3", "qwen3.5-0.8b@main", Some("cal-v2"), &batch)
        .await?;
    assert!(batch_id.starts_with("batch-"));

    let (meta, decoded) = store
        .get(&batch_id)
        .await?
        .ok_or_else(|| Status::internal("stored batch missing"))?;
    assert_eq!(meta.schema_fingerprint, "jev1:routing:v3");
    assert_eq!(meta.model_id, "qwen3.5-0.8b@main");
    assert_eq!(meta.calib_version.as_deref(), Some("cal-v2"));
    assert_eq!(meta.row_count, 2);
    assert_eq!(decoded, batch, "payload must round-trip byte-for-byte typed");
    Ok(())
}

#[tokio::test]
async fn test_fetch_by_schema_and_list() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = DecisionBatchStore::new(backend);
    store.init().await?;

    let batch = model_emitted_batch()?;
    store.put("jev1:routing:v3", "m1", None, &batch).await?;
    store.put("jev1:routing:v3", "m1", Some("cal-v1"), &batch).await?;
    store.put("jev1:eta:v1", "m1", None, &batch).await?;

    let routing = store.fetch_by_schema("jev1:routing:v3").await?;
    assert_eq!(routing.len(), 2);
    assert!(routing.iter().all(|(_, b)| b == &batch));

    let all = store.list(None).await?;
    assert_eq!(all.len(), 3);
    let eta_only = store.list(Some("jev1:eta:v1")).await?;
    assert_eq!(eta_only.len(), 1);
    assert_eq!(eta_only[0].calib_version, None);

    assert!(store.get("batch-does-not-exist").await?.is_none());
    Ok(())
}

#[tokio::test]
async fn test_put_rejects_empty_keys() -> Result<(), Status> {
    let backend = Arc::new(DuckDbBackend::new_in_memory()?);
    let store = DecisionBatchStore::new(backend);
    store.init().await?;
    let batch = model_emitted_batch()?;
    assert!(store.put("", "m1", None, &batch).await.is_err());
    assert!(store.put("fp", "", None, &batch).await.is_err());
    assert_eq!(store.list(None).await?.len(), 0);
    Ok(())
}
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
