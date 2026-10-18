//! Engine-side integration tests for the R6 durable decision/outcome path.
//!
//! The stores live in the Apache-2.0 `hyprstream-metrics-api` crate; these
//! tests exercise them against the REAL DuckDB serving backend — including
//! file-backed close/reopen persistence, tenant denial on reads/writes/joins,
//! exact-row outcome linkage against the stored Arrow schema, and the
//! append-only supersession semantics. Mock-only storage would not prove a
//! durable engine.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::metrics::decision_batch::encode_batch_ipc;
use crate::metrics::decision_outcome::{
    create_decision_outcome_tables, mint_decision_row_id, CalibExpectation, CalibFilter,
    DecisionOutcomeQuery, DecisionPins, DecisionRowQuery, DecisionRowStore, ExpectedPins,
    NewOutcome, OutcomeStore, TrustedTenant, DECISION_ROWS_TABLE, DECISION_ROW_ID_COLUMN,
    FIELD_META_KIND, FIELD_META_LABELS, MAX_DECISION_ROWS_PER_BATCH,
};
use crate::metrics::producer::MetricsProducer;
use crate::storage::duckdb::DuckDbBackend;
use duckdb::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use duckdb::arrow::datatypes::{DataType, Field, Float32Type, Schema};
use hyprstream_metrics_api::StorageBackend;
use std::collections::HashMap;
use std::sync::Arc;

const TENANT_A: &str = "tenant-a";
const TENANT_B: &str = "tenant-b";

fn tenant(name: &str) -> TrustedTenant {
    TrustedTenant::new(name).expect("valid tenant")
}

fn pins(calib: Option<&str>) -> DecisionPins {
    DecisionPins {
        schema_fingerprint: "jev1:route:v3".to_owned(),
        model_id: "clef-flash@8b2e5fd1".to_owned(),
        encoding_id: "clef-text-json-v1".to_owned(),
        quantization: "fp8-e4m3-dynamic".to_owned(),
        compute_profile: "cpu-f32".to_owned(),
        calib_version: calib.map(str::to_owned),
    }
}

/// Build a payload batch in the exact shape the jev Arrow emitter produces:
/// per-question `fixed_size_list<f32; N>` probability columns carrying
/// `jev.kind`/`jev.labels` field metadata, plus the version-triple columns.
/// The `calib_version` column is all-NULL (raw) or all-`cal-1` (calibrated),
/// so stored payloads agree with their declared pins.
fn decision_payload() -> RecordBatch {
    decision_payload_with_calib(None)
}

fn calibrated_payload() -> RecordBatch {
    decision_payload_with_calib(Some("cal-1"))
}

fn decision_payload_with_calib(calib: Option<&str>) -> RecordBatch {
    let mut choice_meta = HashMap::new();
    choice_meta.insert(FIELD_META_KIND.to_owned(), "choice".to_owned());
    choice_meta.insert(
        FIELD_META_LABELS.to_owned(),
        r#"["alpha","beta","gamma"]"#.to_owned(),
    );
    let mut score_meta = HashMap::new();
    score_meta.insert(FIELD_META_KIND.to_owned(), "score".to_owned());
    score_meta.insert(FIELD_META_LABELS.to_owned(), r#"["0","1","2"]"#.to_owned());

    let choice_values: Vec<Option<Vec<Option<f32>>>> = vec![
        Some(vec![Some(0.7), Some(0.2), Some(0.1)]),
        None, // abstained row: null distribution stays null
        Some(vec![Some(0.1), Some(0.6), Some(0.3)]),
    ];
    let route = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(choice_values, 3);
    let score_values: Vec<Option<Vec<Option<f32>>>> = vec![
        Some(vec![Some(0.2), Some(0.5), Some(0.3)]),
        Some(vec![Some(0.9), Some(0.05), Some(0.05)]),
        Some(vec![Some(0.3), Some(0.3), Some(0.4)]),
    ];
    let score = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(score_values, 3);

    let schema = Arc::new(Schema::new(vec![
        Field::new("route.probabilities", route.data_type().clone(), true)
            .with_metadata(choice_meta),
        Field::new("score_level.probabilities", score.data_type().clone(), true)
            .with_metadata(score_meta),
        Field::new("schema_version", DataType::Utf8, false),
        Field::new("model_version", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(route) as ArrayRef,
            Arc::new(score) as ArrayRef,
            Arc::new(StringArray::from(vec!["v3", "v3", "v3"])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "clef-flash@8b2e5fd1",
                "clef-flash@8b2e5fd1",
                "clef-flash@8b2e5fd1",
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![calib; 3])) as ArrayRef,
        ],
    )
    .expect("payload batch")
}

async fn stores_on_backend(
    backend: Arc<DuckDbBackend>,
) -> (DecisionRowStore<DuckDbBackend>, OutcomeStore<DuckDbBackend>) {
    create_decision_outcome_tables(backend.as_ref())
        .await
        .expect("create outcome tables");
    (
        DecisionRowStore::new(Arc::clone(&backend)),
        OutcomeStore::new(backend),
    )
}

async fn seed_two_decisions(
    rows: &DecisionRowStore<DuckDbBackend>,
) -> (String, Vec<String>, String, Vec<String>) {
    let raw = rows
        .put(&tenant(TENANT_A), &pins(None), &decision_payload())
        .await
        .expect("put raw decision");
    let calibrated = rows
        .put(
            &tenant(TENANT_A),
            &pins(Some("cal-1")),
            &calibrated_payload(),
        )
        .await
        .expect("put calibrated decision");
    (
        raw.decision_id,
        raw.row_ids,
        calibrated.decision_id,
        calibrated.row_ids,
    )
}

#[tokio::test]
async fn test_put_get_round_trip_stable_ids_and_lossless_payload() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;
    let (raw_id, raw_rows, calib_id, calib_rows) = seed_two_decisions(&rows).await;

    let meta = rows
        .get_decision(&tenant(TENANT_A), &raw_id)
        .await
        .expect("get")
        .expect("decision present");
    assert_eq!(meta.decision_id, raw_id);
    assert_eq!(meta.tenant_id, TENANT_A);
    assert_eq!(
        meta.pins,
        pins(None),
        "raw inference keeps calib_version None"
    );
    assert!(meta.pins.calib_version.is_none());
    assert_eq!(meta.row_count, 3);
    assert_eq!(meta.payload_format, "arrow-ipc-stream");

    // Typed distributions decode losslessly (Float32 exact values survive).
    let (dist_pins, dist) = rows
        .fetch_distribution(&tenant(TENANT_A), &raw_id, &raw_rows[0])
        .await
        .expect("fetch")
        .expect("row present");
    assert_eq!(dist_pins.schema_fingerprint, "jev1:route:v3");
    assert_eq!(dist.num_rows(), 1);
    let probs = dist
        .column_by_name("route.probabilities")
        .expect("probabilities column");
    let flat = probs
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("fixed size list");
    let row_dist = flat.value(0);
    let values = row_dist
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("f32 child");
    assert_eq!(values.value(0), 0.7_f32);
    assert_eq!(values.value(1), 0.2_f32);
    assert_eq!(values.value(2), 0.1_f32);
    // Stable ids repeat across independent reads (no regeneration).
    let again = rows
        .fetch_distribution(&tenant(TENANT_A), &calib_id, &calib_rows[2])
        .await
        .expect("fetch")
        .expect("row present");
    assert_eq!(again.1.num_rows(), 1);
}

#[tokio::test]
async fn test_file_backed_persistence_reload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("outcomes.duckdb");
    let (raw_id, raw_rows, _calib_id, _calib_rows) = {
        let backend = Arc::new(
            DuckDbBackend::new(
                db_path.to_string_lossy().into_owned(),
                Default::default(),
                None,
            )
            .expect("backend"),
        );
        let (rows, _outcomes) = stores_on_backend(backend).await;
        seed_two_decisions(&rows).await
        // backend dropped here -> DuckDB file closed
    };

    // Reopen from disk: decision rows, stable ids, pins and distributions
    // must survive the restart unchanged.
    let backend = Arc::new(
        DuckDbBackend::new(
            db_path.to_string_lossy().into_owned(),
            Default::default(),
            None,
        )
        .expect("backend reopen"),
    );
    let (rows, outcomes) = stores_on_backend(backend).await;

    let meta = rows
        .get_decision(&tenant(TENANT_A), &raw_id)
        .await
        .expect("get")
        .expect("decision survives reload");
    assert_eq!(meta.pins, pins(None));
    let (_, dist) = rows
        .fetch_distribution(&tenant(TENANT_A), &raw_id, &raw_rows[1])
        .await
        .expect("fetch")
        .expect("row survives reload");
    let probs = dist.column_by_name("route.probabilities").unwrap();
    let flat = probs.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert!(
        flat.is_null(0),
        "abstained distribution stays null across reload"
    );

    // Outcome linkage keeps working against the reloaded payload.
    let _recorded = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: raw_id.clone(),
                decision_row_id: raw_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("beta".to_owned()),
                outcome_value: None,
                outcome_source: "human-label".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect("record on reloaded payload");
    let joined = outcomes
        .join_outcomes(&tenant(TENANT_A), &DecisionOutcomeQuery::default())
        .await
        .expect("join");
    assert_eq!(joined.num_rows(), 1);
    assert_eq!(raw_id, meta.decision_id);
}

#[tokio::test]
async fn test_tenant_denial_reads_writes_and_joins() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, outcomes) = stores_on_backend(backend).await;
    let (raw_id, raw_rows, _calib_id, _calib_rows) = seed_two_decisions(&rows).await;

    // Reads from another tenant are not-found / empty — no existence leak.
    assert!(rows
        .get_decision(&tenant(TENANT_B), &raw_id)
        .await
        .expect("get")
        .is_none());
    assert!(rows
        .fetch_distribution(&tenant(TENANT_B), &raw_id, &raw_rows[0])
        .await
        .expect("fetch")
        .is_none());
    assert!(rows
        .fetch_meta(&tenant(TENANT_B), &DecisionRowQuery::default())
        .await
        .expect("meta")
        .is_empty());
    assert_eq!(
        outcomes
            .join_outcomes(&tenant(TENANT_B), &DecisionOutcomeQuery::default())
            .await
            .expect("join")
            .num_rows(),
        0
    );

    // Cross-tenant outcome writes are rejected.
    let err = outcomes
        .record(
            &tenant(TENANT_B),
            &NewOutcome {
                decision_id: raw_id.clone(),
                decision_row_id: raw_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("alpha".to_owned()),
                outcome_value: None,
                outcome_source: "ground-truth".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect_err("cross-tenant outcome must be denied");
    assert_eq!(err.code(), tonic::Code::NotFound);

    // Unknown decision in the RIGHT tenant is equally not-found.
    let err = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: "dec-missing".to_owned(),
                decision_row_id: raw_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("alpha".to_owned()),
                outcome_value: None,
                outcome_source: "ground-truth".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect_err("unknown decision");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn test_outcome_validation_against_stored_schema_and_pins() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, outcomes) = stores_on_backend(backend).await;
    let (raw_id, raw_rows, _calib_id, _calib_rows) = seed_two_decisions(&rows).await;

    let base = |decision_row: usize, question: &str, label: Option<&str>| NewOutcome {
        decision_id: raw_id.clone(),
        decision_row_id: raw_rows[decision_row].clone(),
        question_id: question.to_owned(),
        expected_pins: None,
        expected_calib: CalibExpectation::Any,
        outcome_label: label.map(str::to_owned),
        outcome_value: None,
        outcome_source: "human-label".to_owned(),
        supersedes_outcome_id: None,
    };

    // Valid choice outcome.
    outcomes
        .record(&tenant(TENANT_A), &base(0, "route", Some("gamma")))
        .await
        .expect("valid choice outcome");

    // Unknown question id (not in the stored Arrow schema).
    let err = outcomes
        .record(&tenant(TENANT_A), &base(0, "route_other", Some("alpha")))
        .await
        .expect_err("unknown question");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(err.message().contains("unknown question"));

    // Label outside the declared choice domain.
    let err = outcomes
        .record(&tenant(TENANT_A), &base(0, "route", Some("delta")))
        .await
        .expect_err("undeclared label");
    assert!(err.message().contains("not in the declared domain"));

    // Score: undeclared level index and out-of-domain value.
    let err = outcomes
        .record(&tenant(TENANT_A), &base(0, "score_level", Some("7")))
        .await
        .expect_err("undeclared score level");
    assert!(err.message().contains("not in the declared domain"));
    let mut bad_value = base(0, "score_level", None);
    bad_value.outcome_value = Some(9.0);
    let err = outcomes
        .record(&tenant(TENANT_A), &bad_value)
        .await
        .expect_err("score value outside domain");
    assert!(err.message().contains("outside the declared score domain"));
    // In-domain score value with no label is a valid minimal observation.
    let mut value_only = base(0, "score_level", None);
    value_only.outcome_value = Some(2.0);
    outcomes
        .record(&tenant(TENANT_A), &value_only)
        .await
        .expect("value-only score outcome");

    // Neither label nor value: no fictitious gold labels.
    let err = outcomes
        .record(&tenant(TENANT_A), &base(0, "route", None))
        .await
        .expect_err("empty outcome");
    assert!(err.message().contains("needs a realized label or value"));

    // Exact-row linkage: a real row id that belongs to a DIFFERENT row set.
    let err = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_row_id: mint_decision_row_id(),
                ..base(0, "route", Some("alpha"))
            },
        )
        .await
        .expect_err("foreign row id");
    assert!(err.message().contains("not a row of decision"));

    // R6B: complete pin assertions — each asserted field must match.
    let mut expected = ExpectedPins::default();
    expected.schema_fingerprint = Some("jev1:other:v1".to_owned());
    let mut wrong_schema = base(0, "route", Some("alpha"));
    wrong_schema.expected_pins = Some(expected);
    let err = outcomes
        .record(&tenant(TENANT_A), &wrong_schema)
        .await
        .expect_err("schema pin mismatch");
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);

    let mut expected = ExpectedPins::default();
    expected.quantization = Some("none".to_owned());
    let mut wrong_quant = base(0, "route", Some("alpha"));
    wrong_quant.expected_pins = Some(expected);
    let err = outcomes
        .record(&tenant(TENANT_A), &wrong_quant)
        .await
        .expect_err("quantization pin mismatch");
    assert!(err.message().contains("quantization"));

    // Calibration expectation: the raw decision rejects a version pin.
    let mut wants_calib = base(1, "route", Some("alpha"));
    wants_calib.expected_calib = CalibExpectation::Version("cal-1".to_owned());
    let err = outcomes
        .record(&tenant(TENANT_A), &wants_calib)
        .await
        .expect_err("raw row under version expectation");
    assert!(err.message().contains("calibration mismatch"));
    // ... and matches Raw.
    let mut wants_raw = base(0, "route", Some("alpha"));
    wants_raw.expected_calib = CalibExpectation::Raw;
    outcomes
        .record(&tenant(TENANT_A), &wants_raw)
        .await
        .expect("raw expectation matches raw row");
}

#[tokio::test]
async fn test_join_filters_calib_distinction_and_supersession() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, outcomes) = stores_on_backend(backend).await;
    let (raw_id, raw_rows, calib_id, calib_rows) = seed_two_decisions(&rows).await;

    let a = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: raw_id.clone(),
                decision_row_id: raw_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("alpha".to_owned()),
                outcome_value: None,
                outcome_source: "ground-truth".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect("raw outcome");
    let b = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: calib_id.clone(),
                decision_row_id: calib_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("beta".to_owned()),
                outcome_value: None,
                outcome_source: "ground-truth".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect("calibrated outcome");

    // The raw vs calibrated distinction stays authoritative in the join.
    let mut raw_query = DecisionOutcomeQuery::default();
    raw_query.calib = CalibFilter::RawOnly;
    let raw_join = outcomes
        .join_outcomes(&tenant(TENANT_A), &raw_query)
        .await
        .expect("raw join");
    assert_eq!(raw_join.num_rows(), 1);
    let calib_col = raw_join
        .column_by_name("calib_version")
        .expect("calib column");
    let arr = calib_col
        .as_any()
        .downcast_ref::<duckdb::arrow::array::StringArray>()
        .expect("utf8");
    assert!(arr.is_null(0), "raw rows carry NULL calib_version");

    let mut version_query = DecisionOutcomeQuery::default();
    version_query.calib = CalibFilter::Version("cal-1".to_owned());
    assert_eq!(
        outcomes
            .join_outcomes(&tenant(TENANT_A), &version_query)
            .await
            .expect("version join")
            .num_rows(),
        1
    );

    // Question filter.
    let mut score_query = DecisionOutcomeQuery::default();
    score_query.question_id = Some("score_level".to_owned());
    assert_eq!(
        outcomes
            .join_outcomes(&tenant(TENANT_A), &score_query)
            .await
            .expect("question join")
            .num_rows(),
        0,
        "no score outcomes recorded"
    );

    // Correction via supersedes: appended row wins, superseded row leaves
    // the default join but stays queryable explicitly (append-only audit).
    let correction = outcomes
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: raw_id.clone(),
                decision_row_id: raw_rows[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("gamma".to_owned()),
                outcome_value: None,
                outcome_source: "human-label".to_owned(),
                supersedes_outcome_id: Some(a.outcome_id.clone()),
            },
        )
        .await
        .expect("correction");
    let default_join = outcomes
        .join_outcomes(&tenant(TENANT_A), &DecisionOutcomeQuery::default())
        .await
        .expect("join");
    assert_eq!(
        default_join.num_rows(),
        2,
        "correction replaces, not duplicates"
    );
    let ids = default_join
        .column_by_name("outcome_id")
        .unwrap()
        .as_any()
        .downcast_ref::<duckdb::arrow::array::StringArray>()
        .unwrap();
    let visible: Vec<&str> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert!(!visible.contains(&a.outcome_id.as_str()));
    assert!(visible.contains(&b.outcome_id.as_str()));
    assert!(visible.contains(&correction.outcome_id.as_str()));

    let mut with_superseded = DecisionOutcomeQuery::default();
    with_superseded.include_superseded = true;
    assert_eq!(
        outcomes
            .join_outcomes(&tenant(TENANT_A), &with_superseded)
            .await
            .expect("join with superseded")
            .num_rows(),
        3
    );

    // Supersession must correct the SAME observation.
    let mut wrong_target = NewOutcome {
        decision_id: calib_id.clone(),
        decision_row_id: calib_rows[1].clone(),
        question_id: "route".to_owned(),
        expected_pins: None,
        expected_calib: CalibExpectation::Any,
        outcome_label: Some("alpha".to_owned()),
        outcome_value: None,
        outcome_source: "human-label".to_owned(),
        supersedes_outcome_id: Some(b.outcome_id.clone()),
    };
    wrong_target.decision_id = calib_id.clone();
    let err = outcomes
        .record(&tenant(TENANT_A), &wrong_target)
        .await
        .expect_err("supersede across different decision rows");
    assert!(err.message().contains("same decision row and question"));

    // Unknown supersede target is not-found, cross-tenant target invisible.
    let mut unknown_target = NewOutcome {
        supersedes_outcome_id: Some("outc-missing".to_owned()),
        ..wrong_target
    };
    unknown_target.supersedes_outcome_id = Some("outc-missing".to_owned());
    let err = outcomes
        .record(&tenant(TENANT_A), &unknown_target)
        .await
        .expect_err("unknown supersede target");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn test_caller_supplied_row_ids_preserved_and_minted_when_absent() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;

    // Payload already carrying the reserved column: ids preserved verbatim.
    let payload = encode_batch_ipc(&decision_payload()).expect("encode");
    let decoded = crate::metrics::decision_batch::decode_batch_ipc(&payload).expect("decode");
    let ids = vec!["row-1", "row-2", "row-3"];
    let mut fields: Vec<_> = decoded.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        DECISION_ROW_ID_COLUMN,
        DataType::Utf8,
        false,
    )));
    let mut columns: Vec<ArrayRef> = decoded.columns().to_vec();
    columns.push(Arc::new(StringArray::from(ids.clone())));
    let with_ids = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("batch");

    let stored = rows
        .put(&tenant(TENANT_A), &pins(None), &with_ids)
        .await
        .expect("put with caller ids");
    assert_eq!(
        stored.row_ids, ids,
        "caller-supplied ids preserved verbatim"
    );

    // Duplicate ids within one batch are rejected.
    let dup = vec!["row-1", "row-1", "row-3"];
    let mut columns2 = decoded.columns().to_vec();
    columns2.push(Arc::new(StringArray::from(dup)));
    let dup_batch = RecordBatch::try_new(
        Arc::new({
            let mut f: Vec<_> = decoded.schema().fields().iter().cloned().collect();
            f.push(Arc::new(Field::new(
                DECISION_ROW_ID_COLUMN,
                DataType::Utf8,
                false,
            )));
            Schema::new(f)
        }),
        columns2,
    )
    .expect("batch");
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &dup_batch)
        .await
        .expect_err("duplicate row ids");
    assert!(err.message().contains("unique"));

    // Oversized batch is rejected before any write.
    let big = vec![decision_payload(); MAX_DECISION_ROWS_PER_BATCH + 1];
    let oversized =
        duckdb::arrow::compute::concat_batches(&decision_payload().schema(), &big).expect("concat");
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &oversized)
        .await
        .expect_err("oversized batch");
    assert!(err.message().contains("exceeds"));
}

#[tokio::test]
async fn test_existing_schema_v2_tables_untouched_by_new_tables() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    crate::metrics::create_schema_v2_tables(backend.as_ref())
        .await
        .expect("schema v2");
    create_decision_outcome_tables(backend.as_ref())
        .await
        .expect("outcome tables");
    let tables = backend.list_tables().await.expect("list tables");
    for expected in [
        "metrics",
        "calibration_params",
        "decision_batches",
        DECISION_ROWS_TABLE,
        "decision_outcomes",
    ] {
        assert!(tables.iter().any(|t| t == expected), "missing {expected}");
    }
}

#[tokio::test]
async fn test_fetch_meta_filters_by_pins() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;
    seed_two_decisions(&rows).await;

    let mut raw_query = DecisionRowQuery::default();
    raw_query.calib = CalibFilter::RawOnly;
    let metas = rows
        .fetch_meta(&tenant(TENANT_A), &raw_query)
        .await
        .expect("meta");
    assert_eq!(metas.len(), 1);
    assert!(metas[0].pins.calib_version.is_none());

    let mut model_query = DecisionRowQuery::default();
    model_query.model_id = Some("other-model".to_owned());
    assert!(rows
        .fetch_meta(&tenant(TENANT_A), &model_query)
        .await
        .expect("meta")
        .is_empty());

    // Tenant isolation on listing.
    assert!(rows
        .fetch_meta(&tenant(TENANT_B), &DecisionRowQuery::default())
        .await
        .expect("meta")
        .is_empty());
}

#[tokio::test]
async fn test_put_admission_rejects_bad_payload_schemas() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;

    let question_field = |kind: &str, labels_json: &str, inner: DataType| {
        let mut meta = HashMap::new();
        meta.insert(FIELD_META_KIND.to_owned(), kind.to_owned());
        meta.insert(FIELD_META_LABELS.to_owned(), labels_json.to_owned());
        Arc::new(
            Field::new(
                "route.probabilities",
                DataType::FixedSizeList(Arc::new(Field::new("item", inner, true)), 3),
                true,
            )
            .with_metadata(meta),
        )
    };

    // Duplicate field names: name-keyed question lookup must stay unambiguous.
    let dup = Schema::new(vec![
        question_field(
            "choice",
            r#"["a","b","c"]"#,
            DataType::Float32,
        );
        2
    ]);
    let batch = RecordBatch::new_empty(Arc::new(dup));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("duplicate field names");
    assert!(err.message().contains("duplicate field name"));

    // Zero question fields.
    let no_questions = Schema::new(vec![Field::new("note", DataType::Utf8, false)]);
    let batch = RecordBatch::new_empty(Arc::new(no_questions));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("no question fields");
    assert!(err.message().contains("at least one"));

    // Non-float32 inner dtype.
    let f64_inner = Schema::new(vec![question_field(
        "choice",
        r#"["a","b","c"]"#,
        DataType::Float64,
    )]);
    let batch = RecordBatch::new_empty(Arc::new(f64_inner));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("float64 inner dtype");
    assert!(err.message().contains("float32"));

    // Non-canonical score labels ("00") rejected at admission.
    let bad_score = Schema::new(vec![question_field(
        "score",
        r#"["00","1","2"]"#,
        DataType::Float32,
    )]);
    let batch = RecordBatch::new_empty(Arc::new(bad_score));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("non-canonical score labels");
    assert!(err.message().contains("canonical levels"));

    // Noul labels must be exactly [false, true].
    let bad_noul = Schema::new(vec![question_field(
        "noul",
        r#"["true","false"]"#,
        DataType::Float32,
    )]);
    let batch = RecordBatch::new_empty(Arc::new(bad_noul));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("swapped noul labels");
    assert!(
        err.message().contains("canonical [false, true]"),
        "unexpected error: {err}"
    );

    // Choice with a single option and duplicate options rejected.
    for labels in [r#"["a"]"#, r#"["a","a","a"]"#] {
        let bad_choice = Schema::new(vec![question_field("choice", labels, DataType::Float32)]);
        let batch = RecordBatch::new_empty(Arc::new(bad_choice));
        let err = rows
            .put(&tenant(TENANT_A), &pins(None), &batch)
            .await
            .expect_err("bad choice domain");
        assert!(
            err.message().contains(">= 2 options") || err.message().contains("distinct"),
            "unexpected error: {err}"
        );
    }

    // Reserved profile-v2 kind rejected.
    let v2 = Schema::new(vec![question_field(
        "span",
        r#"["0","1"]"#,
        DataType::Float32,
    )]);
    let batch = RecordBatch::new_empty(Arc::new(v2));
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &batch)
        .await
        .expect_err("reserved v2 kind");
    assert!(err.message().contains("profile-v2"));
}

#[tokio::test]
async fn test_score_outcome_label_canonical_membership_and_exclusivity() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, outcomes) = stores_on_backend(backend).await;
    let (raw_id, raw_rows, _calib_id, _calib_rows) = seed_two_decisions(&rows).await;

    let base = |row: usize| NewOutcome {
        decision_id: raw_id.clone(),
        decision_row_id: raw_rows[row].clone(),
        question_id: "score_level".to_owned(),
        expected_pins: None,
        expected_calib: CalibExpectation::Any,
        outcome_label: None,
        outcome_value: None,
        outcome_source: "ground-truth".to_owned(),
        supersedes_outcome_id: None,
    };

    // Non-canonical level spellings are not declared members.
    for spelling in ["01", "+1", " 1", "1.0"] {
        let mut o = base(0);
        o.outcome_label = Some(spelling.to_owned());
        let err = outcomes
            .record(&tenant(TENANT_A), &o)
            .await
            .expect_err("non-canonical score label");
        assert!(
            err.message().contains("not in the declared domain"),
            "unexpected error for {spelling:?}: {err}"
        );
    }

    // Conflicting label+value is ambiguous and rejected (review followup).
    let mut both = base(0);
    both.outcome_label = Some("0".to_owned());
    both.outcome_value = Some(2.0);
    let err = outcomes
        .record(&tenant(TENANT_A), &both)
        .await
        .expect_err("score label+value conflict");
    assert!(err.message().contains("mutually exclusive"));

    // noul/choice reject the numeric channel entirely.
    let mut noul_value = base(0);
    noul_value.question_id = "route".to_owned();
    noul_value.outcome_label = None;
    noul_value.outcome_value = Some(1.0);
    let err = outcomes
        .record(&tenant(TENANT_A), &noul_value)
        .await
        .expect_err("value-only noul");
    assert!(err.message().contains("score-only"));
}

/// Build a minimal raw payload whose `route.probabilities` column carries the
/// given per-row distributions (jev metadata + version-triple columns).
fn payload_with_choice_values(values: Vec<Option<Vec<Option<f32>>>>) -> RecordBatch {
    let mut meta = HashMap::new();
    meta.insert(FIELD_META_KIND.to_owned(), "choice".to_owned());
    meta.insert(
        FIELD_META_LABELS.to_owned(),
        r#"["alpha","beta","gamma"]"#.to_owned(),
    );
    let route = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(values, 3);
    let schema = Arc::new(Schema::new(vec![
        Field::new("route.probabilities", route.data_type().clone(), true).with_metadata(meta),
        Field::new("schema_version", DataType::Utf8, false),
        Field::new("model_version", DataType::Utf8, false),
        Field::new("calib_version", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(route) as ArrayRef,
            Arc::new(StringArray::from(vec!["v3", "v3", "v3"])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "clef-flash@8b2e5fd1",
                "clef-flash@8b2e5fd1",
                "clef-flash@8b2e5fd1",
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![None::<&str>, None, None])) as ArrayRef,
        ],
    )
    .expect("payload batch")
}

#[tokio::test]
async fn test_put_rejects_invalid_distribution_values_with_zero_inserts() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend.clone()).await;

    let base: Vec<Option<Vec<Option<f32>>>> = vec![
        Some(vec![Some(0.7), Some(0.2), Some(0.1)]),
        None,
        Some(vec![Some(0.1), Some(0.6), Some(0.3)]),
    ];
    let count_stored = || async {
        let handle = backend
            .prepare_sql("SELECT COUNT(*) AS n FROM decision_rows")
            .await
            .expect("prepare count");
        let batch = backend.query_sql(&handle).await.expect("count");
        batch
            .column_by_name("n")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    };

    // NaN component.
    let mut values = base.clone();
    values[0].as_mut().unwrap()[1] = Some(f32::NAN);
    let err = rows
        .put(
            &tenant(TENANT_A),
            &pins(None),
            &payload_with_choice_values(values),
        )
        .await
        .expect_err("NaN component must be rejected");
    assert!(err.message().contains("not a finite probability"), "{err}");

    // Negative component (finite but out of [0, 1]).
    let mut values = base.clone();
    values[2].as_mut().unwrap()[0] = Some(-0.1);
    let err = rows
        .put(
            &tenant(TENANT_A),
            &pins(None),
            &payload_with_choice_values(values),
        )
        .await
        .expect_err("negative component must be rejected");
    assert!(err.message().contains("in [0, 1]"), "{err}");

    // Sum beyond the consumer tolerance (D5 reader bound 1e-2).
    let mut values = base.clone();
    values[0].as_mut().unwrap()[2] = Some(0.05);
    let err = rows
        .put(
            &tenant(TENANT_A),
            &pins(None),
            &payload_with_choice_values(values),
        )
        .await
        .expect_err("beyond-consumer sum must be rejected");
    assert!(err.message().contains("sums to"), "{err}");

    // Null component inside a PRESENT distribution (whole-list null is the
    // only abstention form).
    let mut values = base.clone();
    values[0].as_mut().unwrap()[1] = None;
    let err = rows
        .put(
            &tenant(TENANT_A),
            &pins(None),
            &payload_with_choice_values(values),
        )
        .await
        .expect_err("null component in present distribution");
    assert!(err.message().contains("null component"), "{err}");

    // Zero rows landed for every rejection above.
    assert_eq!(count_stored().await, 0, "rejections must insert nothing");

    // CAUSAL BAND CONTROL: a sum inside the consumer tolerance but outside
    // the tighter producer bound (0.99, deviation 1e-2 boundary) is accepted
    // UNCHANGED — no normalization at the reader boundary.
    let band: Vec<Option<Vec<Option<f32>>>> = vec![
        Some(vec![Some(0.80), Some(0.15), Some(0.04)]),
        None,
        Some(vec![Some(0.1), Some(0.6), Some(0.3)]),
    ];
    let stored = rows
        .put(
            &tenant(TENANT_A),
            &pins(None),
            &payload_with_choice_values(band),
        )
        .await
        .expect("within-consumer sum accepted unchanged");
    let (_, dist) = rows
        .fetch_distribution(&tenant(TENANT_A), &stored.decision_id, &stored.row_ids[0])
        .await
        .expect("fetch")
        .expect("row present");
    let probs = dist.column_by_name("route.probabilities").unwrap();
    let flat = probs.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    let row = flat.value(0);
    let values = row.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(values.value(0), 0.80_f32, "stored value is the caller's");
    assert_eq!(
        values.value(2),
        0.04_f32,
        "0.04 preserved exactly as supplied"
    );
}

/// Reserved provenance columns contradicting the declared pins are rejected:
/// model identity and raw/calibrated classification come from pins AND
/// payload together.
#[tokio::test]
async fn test_put_rejects_pin_provenance_mismatch() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;

    // Payload claiming a DIFFERENT model than the declared pin.
    let mut meta = HashMap::new();
    meta.insert(FIELD_META_KIND.to_owned(), "choice".to_owned());
    meta.insert(
        FIELD_META_LABELS.to_owned(),
        r#"["alpha","beta"]"#.to_owned(),
    );
    let wrong_model = || {
        let route = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(0.6), Some(0.4)])],
            2,
        );
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("route.probabilities", route.data_type().clone(), true)
                    .with_metadata(meta.clone()),
                Field::new("schema_version", DataType::Utf8, false),
                Field::new("model_version", DataType::Utf8, false),
                Field::new("calib_version", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(route) as ArrayRef,
                Arc::new(StringArray::from(vec!["v3"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["other-model@v1"])) as ArrayRef,
                Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            ],
        )
        .unwrap()
    };
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &wrong_model())
        .await
        .expect_err("payload model contradicts the declared pin");
    assert!(
        err.message().contains("contradicts the declared model pin"),
        "{err}"
    );

    // Calibrated payload (calib column filled) under a RAW pin.
    let err = rows
        .put(&tenant(TENANT_A), &pins(None), &calibrated_payload())
        .await
        .expect_err("calibrated payload under raw pin");
    assert!(err.message().contains("contradicts the RAW pin"), "{err}");

    // Raw payload (calib column null) under a CALIBRATED pin.
    let err = rows
        .put(&tenant(TENANT_A), &pins(Some("cal-1")), &decision_payload())
        .await
        .expect_err("raw payload under calibrated pin");
    assert!(
        err.message().contains("contradicts the calibrated pin"),
        "{err}"
    );

    // Calib column present but claiming a DIFFERENT version.
    let mut other_calib = calibrated_payload();
    // rebuild with cal-9 instead of cal-1
    let mut columns: Vec<ArrayRef> = other_calib.columns().to_vec();
    let calib_idx = other_calib
        .schema()
        .index_of("calib_version")
        .expect("calib_version column");
    columns[calib_idx] = Arc::new(StringArray::from(vec![
        Some("cal-9"),
        Some("cal-9"),
        Some("cal-9"),
    ]));
    other_calib = RecordBatch::try_new(other_calib.schema(), columns).unwrap();
    let err = rows
        .put(&tenant(TENANT_A), &pins(Some("cal-1")), &other_calib)
        .await
        .expect_err("different calibration version must be rejected");
    assert!(
        err.message()
            .contains("contradicts the declared calibration pin"),
        "{err}"
    );

    // Consistent pairs still store: raw payload/raw pin (covered above by
    // seed), calibrated payload/calibrated pin.
    let stored = rows
        .put(
            &tenant(TENANT_A),
            &pins(Some("cal-1")),
            &calibrated_payload(),
        )
        .await
        .expect("consistent calibrated pair");
    assert_eq!(stored.row_ids.len(), 3);
}

#[tokio::test]
async fn test_producer_init_creates_decision_tables_for_accessors() {
    // Lifecycle-faithful callers: producer.init() must leave the store usable
    // through the new accessors with NO separate table creation.
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let producer = MetricsProducer::new(backend);
    producer.init().await.expect("producer init");

    let stored = producer
        .decision_rows()
        .put(&tenant(TENANT_A), &pins(None), &decision_payload())
        .await
        .expect("put right after producer init");
    assert_eq!(stored.row_ids.len(), 3);
    let outcome = producer
        .outcomes()
        .record(
            &tenant(TENANT_A),
            &NewOutcome {
                decision_id: stored.decision_id.clone(),
                decision_row_id: stored.row_ids[0].clone(),
                question_id: "route".to_owned(),
                expected_pins: None,
                expected_calib: CalibExpectation::Any,
                outcome_label: Some("alpha".to_owned()),
                outcome_value: None,
                outcome_source: "ground-truth".to_owned(),
                supersedes_outcome_id: None,
            },
        )
        .await
        .expect("record outcome right after producer init");
    assert!(!outcome.outcome_id.is_empty());
}

#[tokio::test]
async fn test_fetch_meta_is_deterministic_oldest_first() {
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, _outcomes) = stores_on_backend(backend).await;
    let batch = decision_payload();
    let mut insertion = Vec::new();
    for _ in 0..4 {
        let stored = rows
            .put(&tenant(TENANT_A), &pins(None), &batch)
            .await
            .expect("put");
        insertion.push(stored.decision_id);
    }
    let metas = rows
        .fetch_meta(&tenant(TENANT_A), &DecisionRowQuery::default())
        .await
        .expect("meta");
    assert_eq!(metas.len(), 4);
    // Total deterministic order: (created_at ASC, decision_id ASC) with a
    // stable tie-break, identical across repeated reads.
    let ordered_ids: Vec<String> = {
        let mut v = metas.clone();
        v.sort_by_key(|m| (m.created_at, m.decision_id.clone()));
        v.into_iter().map(|m| m.decision_id).collect()
    };
    assert_eq!(
        metas
            .iter()
            .map(|m| m.decision_id.clone())
            .collect::<Vec<_>>(),
        ordered_ids,
        "fetch_meta must return rows in (created_at, decision_id) order"
    );
    assert_eq!(
        metas[0].decision_id, insertion[0],
        "monotonic clock keeps insertion order"
    );
    let again = rows
        .fetch_meta(&tenant(TENANT_A), &DecisionRowQuery::default())
        .await
        .expect("meta again");
    assert_eq!(metas, again, "repeat reads are identical");
}

/// Bulk seed uses the real backend and actual store-admitted immutable decision
/// rows. Direct insertion avoids 10k individually parsed payload writes; it is
/// not a substitute for the separate signed-RPC admission fixtures.
#[tokio::test]
async fn test_outcome_keyset_pages_cross_ten_thousand_with_ties_and_filters() {
    use crate::metrics::decision_outcome::OutcomeCursor;
    let backend = Arc::new(DuckDbBackend::new_in_memory().expect("backend"));
    let (rows, outcomes) = stores_on_backend(Arc::clone(&backend)).await;
    let (raw, raw_rows, calibrated, calibrated_rows) = seed_two_decisions(&rows).await;
    let foreign = rows
        .put(&tenant(TENANT_B), &pins(None), &decision_payload())
        .await
        .expect("foreign admitted decision");
    backend.execute_ddl(&format!(
        "INSERT INTO decision_outcomes (outcome_id, tenant_id, decision_id, decision_row_id, question_id, outcome_label, outcome_value, outcome_source, supersedes_outcome_id, created_at) \
         SELECT 'outc-' || lpad(CAST(i AS VARCHAR), 6, '0'), 'tenant-a', '{}', '{}', 'route', 'beta', NULL, 'bulk-fixture', NULL, CAST(floor(i / 7) AS BIGINT) FROM range(10037) AS seeded(i)",
        raw, raw_rows[0],
    )).await.expect("bulk actual DuckDB outcomes with tied ascending keys");
    for sql in [
        format!("INSERT INTO decision_outcomes VALUES ('outc-calibrated','tenant-a','{}','{}','route','beta',NULL,'fixture',NULL,0)", calibrated, calibrated_rows[0]),
        format!("INSERT INTO decision_outcomes VALUES ('outc-score','tenant-a','{}','{}','score_level',NULL,1,'fixture',NULL,0)", raw, raw_rows[0]),
        format!("INSERT INTO decision_outcomes VALUES ('outc-foreign','tenant-b','{}','{}','route','beta',NULL,'fixture',NULL,0)", foreign.decision_id, foreign.row_ids[0]),
        format!("INSERT INTO decision_outcomes VALUES ('outc-fixed','tenant-a','{}','{}','route','alpha',NULL,'correction','outc-010036',2000)", raw, raw_rows[0]),
    ] {
        backend.execute_ddl(&sql).await.expect("bounded control rows");
    }
    let query = DecisionOutcomeQuery {
        schema_fingerprint: Some(pins(None).schema_fingerprint),
        model_id: Some(pins(None).model_id),
        question_id: Some("route".to_owned()),
        calib: CalibFilter::RawOnly,
        include_superseded: false,
        limit: 1000,
    };
    let first_legacy = outcomes
        .join_outcomes(&tenant(TENANT_A), &query)
        .await
        .expect("legacy first page");
    assert_eq!(first_legacy.num_rows(), 1000);
    let mut cursor = None;
    let mut seen = std::collections::HashSet::new();
    let mut keys = Vec::new();
    let mut pages = 0;
    loop {
        let page = outcomes
            .join_outcomes_page(&tenant(TENANT_A), &query, cursor.as_ref())
            .await
            .expect("actual bounded page");
        assert!(page.rows.num_rows() <= 1000);
        let ids = page
            .rows
            .column_by_name("outcome_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let times = page
            .rows
            .column_by_name("created_at")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let tenants = page
            .rows
            .column_by_name("tenant_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..page.rows.num_rows() {
            assert!(
                seen.insert(ids.value(i).to_owned()),
                "no outcome repeats across tied boundary"
            );
            assert_eq!(tenants.value(i), TENANT_A);
            keys.push((times.value(i), ids.value(i).to_owned()));
        }
        if pages == 0 {
            assert_eq!(
                page.rows, first_legacy,
                "legacy shape and ASC first-page order remain unchanged"
            );
        }
        pages += 1;
        match page.next_cursor {
            Some(next) => {
                assert_eq!(
                    (next.created_at(), next.outcome_id()),
                    (
                        times.value(page.rows.num_rows() - 1),
                        ids.value(page.rows.num_rows() - 1)
                    )
                );
                cursor = Some(next);
            }
            None => break,
        }
        assert!(pages < 20, "finite continuation");
    }
    assert_eq!(pages, 11);
    assert_eq!(seen.len(), 10037);
    assert!(
        !seen.contains("outc-010036"),
        "superseded original excluded on every page"
    );
    assert!(seen.contains("outc-fixed"));
    assert!(
        keys.windows(2).all(|v| v[0] < v[1]),
        "strict ASC total order including ties"
    );
    for i in 0..10036 {
        assert!(
            seen.contains(&format!("outc-{i:06}")),
            "no matching outcome is unreachable"
        );
    }
    let final_key = keys.last().unwrap();
    let final_cursor = OutcomeCursor::new(final_key.0, final_key.1.clone()).unwrap();
    let empty = outcomes
        .join_outcomes_page(&tenant(TENANT_A), &query, Some(&final_cursor))
        .await
        .unwrap();
    assert_eq!(empty.rows.num_rows(), 0);
    assert!(empty.next_cursor.is_none());
    let mut cap_query = query.clone();
    cap_query.limit = u32::MAX;
    let capped = outcomes
        .join_outcomes_page(&tenant(TENANT_A), &cap_query, None)
        .await
        .unwrap();
    assert_eq!(capped.rows.num_rows(), 10000);
    assert!(capped.next_cursor.is_some());
    cap_query.limit = 0;
    assert_eq!(
        outcomes
            .join_outcomes_page(&tenant(TENANT_A), &cap_query, None)
            .await
            .unwrap()
            .rows
            .num_rows(),
        1000
    );
    let mut all_query = query.clone();
    all_query.include_superseded = true;
    all_query.limit = u32::MAX;
    let inclusive_first = outcomes
        .join_outcomes_page(&tenant(TENANT_A), &all_query, None)
        .await
        .unwrap();
    let inclusive_last = outcomes
        .join_outcomes_page(
            &tenant(TENANT_A),
            &all_query,
            inclusive_first.next_cursor.as_ref(),
        )
        .await
        .unwrap();
    assert_eq!(
        inclusive_first.rows.num_rows() + inclusive_last.rows.num_rows(),
        10038
    );
    assert!(inclusive_last.next_cursor.is_none());
    let start = OutcomeCursor::new(0, "outc-000000".to_owned()).unwrap();
    // A cursor copied across tenants can only position the caller's own rows.
    let foreign_page = outcomes
        .join_outcomes_page(&tenant(TENANT_B), &query, Some(&start))
        .await
        .unwrap();
    assert_eq!(foreign_page.rows.num_rows(), 1);
    let mut calibrated_query = query.clone();
    calibrated_query.calib = CalibFilter::Version("cal-1".to_owned());
    assert_eq!(
        outcomes
            .join_outcomes_page(&tenant(TENANT_A), &calibrated_query, Some(&start))
            .await
            .unwrap()
            .rows
            .num_rows(),
        1
    );
    let mut score_query = query.clone();
    score_query.question_id = Some("score_level".to_owned());
    assert_eq!(
        outcomes
            .join_outcomes_page(&tenant(TENANT_A), &score_query, Some(&start))
            .await
            .unwrap()
            .rows
            .num_rows(),
        1
    );
    for bad in ["schema", "model"] {
        let mut filtered = query.clone();
        if bad == "schema" {
            filtered.schema_fingerprint = Some("other-schema".to_owned());
        } else {
            filtered.model_id = Some("other-model".to_owned());
        }
        assert_eq!(
            outcomes
                .join_outcomes_page(&tenant(TENANT_A), &filtered, Some(&start))
                .await
                .unwrap()
                .rows
                .num_rows(),
            0
        );
    }
}
