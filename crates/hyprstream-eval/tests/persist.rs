//! Persistence via the P0.3 metrics API: a run's Arrow batch round-trips
//! through `DecisionBatchStore` over an in-memory `StorageBackend`, keyed by
//! the decision-schema fingerprint with the version triple intact.

#![allow(clippy::unwrap_used, clippy::result_large_err)]

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use async_trait::async_trait;
use hyprstream_eval::{BatchSink, EvalRow, EvalSet, Harness, TruthSubject};
use hyprstream_metrics_api::arrow::array::RecordBatch;
use hyprstream_metrics_api::arrow::compute;
use hyprstream_metrics_api::arrow::datatypes::Schema;
use hyprstream_metrics_api::storage::view::{ViewDefinition, ViewMetadata};
use hyprstream_metrics_api::storage::{Credentials, StorageBackend};
use tonic::Status;

/// Minimal in-memory backend: tables are concatenated row stores; `query_sql`
/// supports exactly the `SELECT ... FROM <table> [WHERE col = 'val']` shapes
/// the P0.3 stores issue, filtering rows by the named Utf8 column.
#[derive(Default)]
struct MemoryBackend {
    tables: Mutex<HashMap<String, (Arc<Schema>, Vec<RecordBatch>)>>,
}

impl MemoryBackend {
    fn err(message: impl Into<String>) -> Status {
        Status::internal(message.into())
    }

    fn filter(&self, table: &str, where_col: Option<(&str, &str)>) -> Result<RecordBatch, Status> {
        let tables = self.tables.lock();
        let (schema, batches) = tables
            .get(table)
            .ok_or_else(|| Self::err(format!("no such table: {table}")))?;
        let batch = if batches.is_empty() {
            RecordBatch::new_empty(schema.clone())
        } else {
            compute::concat_batches(schema, batches).map_err(|e| Self::err(e.to_string()))?
        };
        let Some((col_name, value)) = where_col else {
            return Ok(batch);
        };
        let col = batch
            .column_by_name(col_name)
            .and_then(|c| {
                c.as_any()
                    .downcast_ref::<hyprstream_metrics_api::arrow::array::StringArray>()
            })
            .ok_or_else(|| Self::err(format!("where column {col_name} not found")))?;
        let mask = hyprstream_metrics_api::arrow::array::BooleanArray::from(
            col.iter()
                .map(|v| Some(v == Some(value)))
                .collect::<Vec<_>>(),
        );
        compute::filter_record_batch(&batch, &mask).map_err(|e| Self::err(e.to_string()))
    }
}

fn parse_where(sql: &str) -> Option<(&str, &str)> {
    let pos = sql.find(" WHERE ")?;
    let clause = &sql[pos + 7..];
    let (col, rest) = clause.split_once(" = '")?;
    let value = rest.split('\'').next()?;
    Some((col.trim(), value))
}

#[async_trait]
impl StorageBackend for MemoryBackend {
    async fn init(&self) -> Result<(), Status> {
        Ok(())
    }

    async fn prepare_sql(&self, query: &str) -> Result<Vec<u8>, Status> {
        Ok(query.as_bytes().to_vec())
    }

    async fn query_sql(&self, statement_handle: &[u8]) -> Result<RecordBatch, Status> {
        let sql = std::str::from_utf8(statement_handle).map_err(|e| Self::err(e.to_string()))?;
        let from_pos = sql
            .find(" FROM ")
            .ok_or_else(|| Self::err("no FROM clause"))?;
        let table = sql[from_pos + 6..]
            .split_whitespace()
            .next()
            .ok_or_else(|| Self::err("no table name"))?
            .trim_end_matches(';');
        self.filter(table, parse_where(sql))
    }

    fn new_with_options(
        _connection_string: &str,
        _options: &HashMap<String, String>,
        _credentials: Option<&Credentials>,
    ) -> Result<Self, Status> {
        Ok(Self::default())
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        self.tables.lock().insert(
            table_name.to_owned(),
            (Arc::new(schema.clone()), Vec::new()),
        );
        Ok(())
    }

    async fn insert_into_table(&self, table_name: &str, batch: RecordBatch) -> Result<(), Status> {
        self.tables
            .lock()
            .get_mut(table_name)
            .ok_or_else(|| Self::err(format!("no such table: {table_name}")))?
            .1
            .push(batch);
        Ok(())
    }

    async fn create_view(&self, _name: &str, _definition: ViewDefinition) -> Result<(), Status> {
        Err(Status::unimplemented("memory backend: views"))
    }
    async fn get_view(&self, _name: &str) -> Result<ViewMetadata, Status> {
        Err(Status::unimplemented("memory backend: views"))
    }
    async fn list_views(&self) -> Result<Vec<String>, Status> {
        Ok(Vec::new())
    }
    async fn list_tables(&self) -> Result<Vec<String>, Status> {
        Ok(self.tables.lock().keys().cloned().collect())
    }
    async fn get_table_schema(&self, table_name: &str) -> Result<Arc<Schema>, Status> {
        self.tables
            .lock()
            .get(table_name)
            .map(|(schema, _)| schema.clone())
            .ok_or_else(|| Self::err(format!("no such table: {table_name}")))
    }
    async fn drop_view(&self, _name: &str) -> Result<(), Status> {
        Err(Status::unimplemented("memory backend: views"))
    }
    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        self.tables.lock().remove(table_name);
        Ok(())
    }
    async fn export_to_parquet(
        &self,
        _table_name: &str,
        _path: &std::path::Path,
    ) -> Result<(), Status> {
        Err(Status::unimplemented("memory backend: parquet"))
    }
    async fn import_from_parquet(
        &self,
        _table_name: &str,
        _path: &std::path::Path,
    ) -> Result<(), Status> {
        Err(Status::unimplemented("memory backend: parquet"))
    }
}

fn workflow_set() -> EvalSet {
    let set = hyprstream_decision::author::parse_yaml(
        r#"
questions:
  is_refund:
    type: noul
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
"#,
    )
    .unwrap();
    let truth = |pairs: &[(&str, usize)]| {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), *v))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    EvalSet {
        name: "refunds".to_owned(),
        schema_version: "refunds-1.0.0".to_owned(),
        questions: set.questions,
        rows: vec![
            EvalRow {
                id: "row-1".to_owned(),
                family: Some("refunds".to_owned()),
                stratum: Some("clean".to_owned()),
                group: None,
                state: hyprstream_decision::entry::Entry::Str(" crushed ".into()),
                truth: truth(&[("is_refund", 1), ("severity", 2)]),
            },
            EvalRow {
                id: "row-2".to_owned(),
                family: Some("refunds".to_owned()),
                stratum: Some("clean".to_owned()),
                group: None,
                state: hyprstream_decision::entry::Entry::Str(" scratched ".into()),
                truth: truth(&[("is_refund", 0), ("severity", 0)]),
            },
        ],
    }
}

#[tokio::test]
async fn run_persists_and_reads_back_through_p03_store() {
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend);
    sink.init().await.unwrap();

    let set = workflow_set();
    let subject = TruthSubject::new("truth-1")
        .with_truth("is_refund", 1)
        .with_truth("severity", 2);
    let output = Harness.run_set(&set, &subject).await.unwrap();
    let fingerprint = output.schema_fingerprint.clone().unwrap();

    let batch_id = sink.persist_run(&output).await.unwrap();
    let (meta, batch) = sink.store().get(&batch_id).await.unwrap().unwrap();
    assert_eq!(meta.schema_fingerprint, fingerprint);
    assert_eq!(meta.model_id, "truth-1");
    assert_eq!(meta.calib_version, None);
    assert_eq!(meta.row_count, 2);
    assert_eq!(batch.num_rows(), 2);
    // Version triple columns round-trip.
    let schema_col = batch
        .column_by_name("schema_version")
        .unwrap()
        .as_any()
        .downcast_ref::<hyprstream_metrics_api::arrow::array::StringArray>()
        .unwrap();
    assert_eq!(schema_col.value(0), "refunds-1.0.0");
    // Fingerprint-keyed fetch finds the batch.
    let listed = sink.store().fetch_by_schema(&fingerprint).await.unwrap();
    assert_eq!(listed.len(), 1);
}

#[tokio::test]
async fn item_runs_are_not_persistable_as_one_batch() {
    let items: Vec<hyprstream_eval::EvalItem> =
        hyprstream_bench::generate_all(&hyprstream_bench::BenchConfig {
            seeds_per_stratum: 1,
            seed_base: 0x06,
        })
        .iter()
        .take(3)
        .map(hyprstream_eval::EvalItem::from)
        .collect();
    let subject = TruthSubject::new("truth-1");
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend);
    sink.init().await.unwrap();
    assert!(sink.persist_run(&output).await.is_err());
}

#[tokio::test]
async fn empty_runs_cannot_persist_without_schema_version_metadata() {
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend.clone());
    sink.init().await.unwrap();
    let mut set = workflow_set();
    set.rows.clear();
    let output = Harness
        .run_set(&set, &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    assert_eq!(output.version.schema, "refunds-1.0.0");
    assert_eq!(output.batch.as_ref().unwrap().num_rows(), 0);
    assert!(matches!(
        sink.persist_run(&output).await,
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    assert!(
        backend.tables.lock()["decision_batches"].1.is_empty(),
        "rejected persistence must not insert an unrecoverable artifact"
    );
    // The supported nonempty path still persists and reads a complete triple.
    let output = Harness
        .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    let id = sink.persist_run(&output).await.unwrap();
    let (meta, batch) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(meta.model_id, output.version.model);
    assert_eq!(meta.calib_version, output.version.calib);
    let schemas = batch
        .column_by_name("schema_version")
        .unwrap()
        .as_any()
        .downcast_ref::<hyprstream_metrics_api::arrow::array::StringArray>()
        .unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert!(schemas
        .iter()
        .all(|value| value == Some(output.version.schema.as_str())));
}

fn replace_version_column(
    output: &mut hyprstream_eval::RunOutput,
    name: &str,
    values: Vec<Option<&str>>,
) {
    use hyprstream_metrics_api::arrow::array::StringArray;
    let batch = output.batch.as_ref().unwrap();
    let index = batch.schema().index_of(name).unwrap();
    let mut columns = batch.columns().to_vec();
    columns[index] = Arc::new(StringArray::from(values));
    output.batch = Some(RecordBatch::try_new(batch.schema(), columns).unwrap());
}

#[tokio::test]
async fn detached_or_embedded_provenance_forgery_is_rejected_before_backend_write() {
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend.clone());
    sink.init().await.unwrap();
    for mutation in [
        "fingerprint",
        "schema",
        "model",
        "calib",
        "embedded-second-model",
        "embedded-calib",
        "coherent-empty-calib",
        "labels",
    ] {
        let mut output = Harness
            .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
            .await
            .unwrap();
        match mutation {
            "fingerprint" => output.schema_fingerprint = Some("forged-schema".into()),
            "schema" => output.version.schema = "other-schema-version".into(),
            "model" => output.version.model = "other-model".into(),
            "calib" => output.version.calib = Some("other-calibration".into()),
            "embedded-second-model" => replace_version_column(
                &mut output,
                "model_version",
                vec![Some("truth-1"), Some("other-model")],
            ),
            "embedded-calib" => replace_version_column(
                &mut output,
                "calib_version",
                vec![None, Some("other-calibration")],
            ),
            "coherent-empty-calib" => {
                output.version.calib = Some(String::new());
                replace_version_column(&mut output, "calib_version", vec![Some(""), Some("")]);
            }
            "labels" => {
                let batch = output.batch.as_ref().unwrap();
                let mut fields = batch.schema().fields().to_vec();
                let mut metadata = fields[0].metadata().clone();
                metadata.insert("jev.labels".into(), "[\"false\",\"renamed\"]".into());
                fields[0] = Arc::new(fields[0].as_ref().clone().with_metadata(metadata));
                output.batch = Some(
                    RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())
                        .unwrap(),
                );
            }
            _ => unreachable!(),
        }
        assert!(
            matches!(
                sink.persist_run(&output).await,
                Err(hyprstream_eval::EvalError::InvalidInput(_))
            ),
            "mutation {mutation}"
        );
        assert!(
            backend.tables.lock()["decision_batches"].1.is_empty(),
            "mutation {mutation} reached the backend"
        );
    }
    // Absent calibration remains distinct from a present, empty version.
    let uncalibrated = Harness
        .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    let id = sink.persist_run(&uncalibrated).await.unwrap();
    let (meta, _) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(meta.calib_version, None);

    // A matching non-null calibration version remains valid on every payload row.
    let mut valid = Harness
        .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    valid.version.calib = Some("calibration-1".into());
    replace_version_column(
        &mut valid,
        "calib_version",
        vec![Some("calibration-1"), Some("calibration-1")],
    );
    let id = sink.persist_run(&valid).await.unwrap();
    let (meta, _) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(meta.calib_version.as_deref(), Some("calibration-1"));
    assert_eq!(meta.model_id, "truth-1");
}

/// Changes only data arrays, preserving the already-valid schema and provenance.
fn replace_answer_column(
    output: &mut hyprstream_eval::RunOutput,
    name: &str,
    replacement: hyprstream_metrics_api::arrow::array::ArrayRef,
) {
    let batch = output.batch.as_ref().unwrap();
    let index = batch.schema().index_of(name).unwrap();
    let mut columns = batch.columns().to_vec();
    columns[index] = replacement;
    output.batch = Some(RecordBatch::try_new(batch.schema(), columns).unwrap());
}

fn choice_probabilities(
    first: Option<Vec<Option<f32>>>,
) -> hyprstream_metrics_api::arrow::array::ArrayRef {
    use hyprstream_metrics_api::arrow::array::{FixedSizeListBuilder, Float32Builder};
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
    for row in [first, Some(vec![Some(0.4), Some(0.4), Some(0.2)]), None] {
        match row {
            Some(values) => {
                for value in values {
                    builder.values().append_option(value);
                }
                builder.append(true);
            }
            None => {
                for _ in 0..3 {
                    builder.values().append_value(f32::NAN);
                }
                builder.append(false);
            }
        }
    }
    Arc::new(builder.finish())
}

fn choice_conformal(
    first: Option<Vec<Option<&str>>>,
    last: Option<Vec<Option<&str>>>,
) -> hyprstream_metrics_api::arrow::array::ArrayRef {
    use hyprstream_metrics_api::arrow::array::{ListBuilder, StringBuilder};
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in [first, Some(vec![]), last] {
        match row {
            Some(values) => {
                for value in values {
                    builder.values().append_option(value);
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }
    Arc::new(builder.finish())
}

#[tokio::test]
async fn mutable_batch_row_values_are_validated_before_backend_write() {
    use hyprstream_decision::answer::{AnswerValue, QuestionAnswer};
    use hyprstream_decision::arrow::{AnswerRow, DecisionSchema};
    use hyprstream_metrics_api::arrow::array::StringArray;

    let questions = hyprstream_decision::author::parse_yaml(
        r#"
questions:
  is_refund:
    type: noul
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
  route:
    type: choice
    criteria: {zebra: null, alpha: null, omega: null}
"#,
    )
    .unwrap()
    .questions;
    let schema = DecisionSchema::new(questions).unwrap();
    let mut valid = Harness
        .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    let rows = (0..3)
        .map(|row| AnswerRow {
            answers: schema
                .questions()
                .iter()
                .map(|question| {
                    let answer = if row == 2 {
                        QuestionAnswer::abstained()
                    } else {
                        let probabilities = if row == 0 {
                            vec![0.1, 0.2, 0.7]
                        } else {
                            vec![0.4, 0.4, 0.2]
                        };
                        let value = match question.id.as_str() {
                            "is_refund" => AnswerValue::Noul {
                                p_true: if row == 0 { 0.7 } else { 0.5 },
                            },
                            "severity" => AnswerValue::Score { probabilities },
                            "route" => AnswerValue::Choice { probabilities },
                            _ => unreachable!(),
                        };
                        QuestionAnswer {
                            value: Some(value),
                            conformal_set: Some(if row == 0 {
                                // Preserve builder semantics: repeated members are allowed.
                                vec![question.labels()[0].clone(), question.labels()[0].clone()]
                            } else {
                                vec![]
                            }),
                        }
                    };
                    (question.id.clone(), answer)
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    valid.batch = Some(schema.build_batch(&valid.version, &rows).unwrap());
    valid.schema_fingerprint = Some(schema.fingerprint());
    let original = valid.batch.clone().unwrap();
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend.clone());
    sink.init().await.unwrap();
    for mutation in [
        "unknown-label",
        "wrong-argmax",
        "wrong-tie-break",
        "nan",
        "infinity",
        "negative",
        "above-one",
        "not-normalized",
        "null-component",
        "null-label",
        "null-probabilities-with-label",
        "unknown-conformal",
        "null-conformal-member",
        "abstained-with-conformal",
    ] {
        valid.batch = Some(original.clone());
        match mutation {
            "unknown-label" | "wrong-argmax" | "wrong-tie-break" | "null-label" => {
                let (first, second) = match mutation {
                    "unknown-label" => (Some("outside"), Some("zebra")),
                    "wrong-argmax" => (Some("zebra"), Some("zebra")),
                    "wrong-tie-break" => (Some("omega"), Some("alpha")),
                    "null-label" => (None, Some("zebra")),
                    _ => unreachable!(),
                };
                replace_answer_column(
                    &mut valid,
                    "route.label",
                    Arc::new(StringArray::from(vec![first, second, None])),
                );
            }
            "unknown-conformal" | "null-conformal-member" | "abstained-with-conformal" => {
                let first = match mutation {
                    "unknown-conformal" => Some(vec![Some("outside")]),
                    "null-conformal-member" => Some(vec![None]),
                    _ => Some(vec![Some("zebra")]),
                };
                let last = (mutation == "abstained-with-conformal").then(Vec::new);
                replace_answer_column(
                    &mut valid,
                    "route.conformal_set",
                    choice_conformal(first, last),
                );
            }
            "null-probabilities-with-label" => {
                replace_answer_column(
                    &mut valid,
                    "route.probabilities",
                    choice_probabilities(None),
                );
                // Isolate the label mismatch from the independent conformal invariant.
                replace_answer_column(
                    &mut valid,
                    "route.conformal_set",
                    choice_conformal(None, None),
                );
            }
            _ => {
                let first = match mutation {
                    "nan" => Some(vec![Some(f32::NAN), Some(0.2), Some(0.7)]),
                    "infinity" => Some(vec![Some(f32::INFINITY), Some(0.2), Some(0.7)]),
                    "negative" => Some(vec![Some(-0.1), Some(0.4), Some(0.7)]),
                    "above-one" => Some(vec![Some(1.1), Some(0.0), Some(0.0)]),
                    "not-normalized" => Some(vec![Some(0.1), Some(0.2), Some(0.6)]),
                    "null-component" => Some(vec![None, Some(0.2), Some(0.8)]),
                    _ => unreachable!(),
                };
                replace_answer_column(
                    &mut valid,
                    "route.probabilities",
                    choice_probabilities(first),
                );
            }
        }
        // Every adversary keeps all admission checks from the previous validator true.
        assert_eq!(valid.batch.as_ref().unwrap().schema(), original.schema());
        assert_eq!(
            valid.schema_fingerprint.as_deref(),
            Some(schema.fingerprint().as_str())
        );
        assert!(
            matches!(
                sink.persist_run(&valid).await,
                Err(hyprstream_eval::EvalError::InvalidInput(_))
            ),
            "mutation {mutation}"
        );
        assert!(
            backend.tables.lock()["decision_batches"].1.is_empty(),
            "mutation {mutation} reached backend"
        );
    }
    // Valid soft distributions, canonical ties, all three kinds, abstention,
    // repeated members and present-empty conformal sets survive unchanged.
    valid.batch = Some(original.clone());
    let id = sink.persist_run(&valid).await.unwrap();
    let (_, stored) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(stored, original);
    // Null-parent children are inactive; even NaN filler must not be interpreted.
    replace_answer_column(
        &mut valid,
        "route.probabilities",
        choice_probabilities(Some(vec![Some(0.1), Some(0.2), Some(0.7)])),
    );
    let id = sink.persist_run(&valid).await.unwrap();
    let (_, stored) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(stored.num_rows(), 3);
    // Consumers that omit the reserved conformal columns remain supported.
    let without_conformal = schema.without_conformal_set();
    valid.batch = Some(
        without_conformal
            .build_batch(&valid.version, &rows)
            .unwrap(),
    );
    valid.schema_fingerprint = Some(without_conformal.fingerprint());
    let expected = valid.batch.clone().unwrap();
    let id = sink.persist_run(&valid).await.unwrap();
    let (_, stored) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(stored, expected);
}

#[tokio::test]
async fn contradictory_model_provenance_is_rejected_by_scoring_and_persistence() {
    use hyprstream_eval::{score_run, ScoreConfig};
    let backend = Arc::new(MemoryBackend::default());
    let sink = BatchSink::new(backend.clone());
    sink.init().await.unwrap();
    for mutation in [
        "report-only",
        "version-and-batch",
        "empty-report",
        "both-empty",
    ] {
        let mut output = Harness
            .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
            .await
            .unwrap();
        match mutation {
            "report-only" => output.model_id = "different-report-model".into(),
            "version-and-batch" => {
                output.version.model = "different-stored-model".into();
                replace_version_column(
                    &mut output,
                    "model_version",
                    vec![
                        Some("different-stored-model"),
                        Some("different-stored-model"),
                    ],
                );
            }
            "empty-report" => output.model_id.clear(),
            "both-empty" => {
                output.model_id.clear();
                output.version.model.clear();
                replace_version_column(&mut output, "model_version", vec![Some(""), Some("")]);
            }
            _ => unreachable!(),
        }
        assert!(
            matches!(
                score_run(&output, &ScoreConfig::default()),
                Err(hyprstream_eval::EvalError::InvalidInput(_))
            ),
            "scoring mutation {mutation}"
        );
        assert!(
            matches!(
                sink.persist_run(&output).await,
                Err(hyprstream_eval::EvalError::InvalidInput(_))
            ),
            "persistence mutation {mutation}"
        );
        assert!(
            backend.tables.lock()["decision_batches"].1.is_empty(),
            "mutation {mutation} reached backend"
        );
    }
    let valid = Harness
        .run_set(&workflow_set(), &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    score_run(&valid, &ScoreConfig::default()).unwrap();
    let id = sink.persist_run(&valid).await.unwrap();
    let (meta, _) = sink.store().get(&id).await.unwrap().unwrap();
    assert_eq!(meta.model_id, valid.model_id);
}

#[tokio::test]
async fn wire_precision_winner_survives_scoring_and_persistence_for_all_kinds() {
    use hyprstream_eval::{score_run, ScoreConfig, Subject};
    struct PrecisionSubject(f64);
    #[async_trait]
    impl Subject for PrecisionSubject {
        fn model_id(&self) -> &str {
            "precision-1"
        }
        async fn decide(
            &self,
            set: &hyprstream_decision::QuestionSet,
            _state: &hyprstream_decision::Entry,
            _row: usize,
        ) -> Result<hyprstream_decision::arrow::AnswerRow, hyprstream_eval::EvalError> {
            hyprstream_eval::subject::parse_response_answers(
                set,
                &serde_json::json!({"answers": {
                    "binary": {"type": "noul", "noul": self.0},
                    "route": {"type": "choice", "probabilities": {"first": 1.0-self.0, "later": self.0}},
                    "rank": {"type": "score", "probabilities": {"0": 1.0-self.0, "1": self.0}}
                }}),
            )
        }
    }
    let questions = hyprstream_decision::author::parse_yaml("questions:\n  binary: {type: noul}\n  route: {type: choice, criteria: {first: null, later: null}}\n  rank: {type: score, criteria: [low, high]}\n").unwrap().questions;
    for (wire, winner) in [(0.50000001, 1), (0.49999999, 0), (0.5, 0)] {
        let set = EvalSet {
            name: "wire-precision".into(),
            schema_version: "v1".into(),
            questions: questions.clone(),
            rows: vec![EvalRow {
                id: "row".into(),
                family: None,
                stratum: None,
                group: None,
                state: hyprstream_decision::Entry::Null,
                truth: questions.iter().map(|q| (q.id.clone(), winner)).collect(),
            }],
        };
        let output = Harness
            .run_set(&set, &PrecisionSubject(wire))
            .await
            .unwrap();
        for obs in &output.observations {
            assert_eq!(obs.argmax_label_index(), Some(winner));
        }
        assert_eq!(
            score_run(&output, &ScoreConfig::default())
                .unwrap()
                .accuracy,
            Some(1.0)
        );
        let backend = Arc::new(MemoryBackend::default());
        let sink = BatchSink::new(backend);
        sink.init().await.unwrap();
        let id = sink.persist_run(&output).await.unwrap();
        let (_, stored) = sink.store().get(&id).await.unwrap().unwrap();
        for question in &set.questions {
            let labels = stored
                .column_by_name(&format!("{}.label", question.id))
                .unwrap()
                .as_any()
                .downcast_ref::<hyprstream_metrics_api::arrow::array::StringArray>()
                .unwrap();
            assert_eq!(labels.value(0), question.labels()[winner]);
        }
    }
}
