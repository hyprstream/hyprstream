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
    let fingerprint = output.schema_fingerprint().unwrap();

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
    let listed = sink.store().fetch_by_schema(fingerprint).await.unwrap();
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
    assert_eq!(output.version().schema, "refunds-1.0.0");
    assert_eq!(output.batch().unwrap().num_rows(), 0);
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
    assert_eq!(meta.model_id, output.version().model);
    assert_eq!(meta.calib_version, output.version().calib);
    let schemas = batch
        .column_by_name("schema_version")
        .unwrap()
        .as_any()
        .downcast_ref::<hyprstream_metrics_api::arrow::array::StringArray>()
        .unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert!(schemas
        .iter()
        .all(|value| value == Some(output.version().schema.as_str())));
}

#[tokio::test]
async fn eval_v2_roundtrips_none_and_empty_tags_through_arrow_ipc() {
    use hyprstream_metrics_api::arrow::array::StringArray;
    use std::io::Cursor;

    let mut set = workflow_set();
    set.rows[0].family = None;
    set.rows[0].stratum = Some(String::new());
    set.rows[0].group = Some(String::new());
    set.rows[1].family = Some(String::new());
    set.rows[1].stratum = None;
    set.rows[1].group = None;
    let output = Harness
        .run_set(&set, &TruthSubject::new("truth-1"))
        .await
        .unwrap();
    let mut bytes = Vec::new();
    {
        let mut writer = hyprstream_metrics_api::arrow::ipc::writer::StreamWriter::try_new(
            &mut bytes,
            &output.batch().unwrap().schema(),
        )
        .unwrap();
        writer.write(output.batch().unwrap()).unwrap();
        writer.finish().unwrap();
    }
    let mut reader =
        hyprstream_metrics_api::arrow::ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)
            .unwrap();
    let decoded = reader.next().unwrap().unwrap();
    assert_eq!(decoded, *output.batch().unwrap());
    for (name, expected) in [
        ("family", vec![None, Some("")]),
        ("stratum", vec![Some(""), None]),
        ("group", vec![Some(""), None]),
    ] {
        let values = decoded
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), expected);
    }
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
        ) -> Result<(hyprstream_decision::arrow::AnswerRow, String), hyprstream_eval::EvalError>
        {
            hyprstream_eval::subject::parse_response_answers(
                set,
                &serde_json::json!({"answers": {
                    "binary": {"type": "noul", "noul": self.0},
                    "route": {"type": "choice", "probabilities": {"first": 1.0-self.0, "later": self.0}},
                    "rank": {"type": "score", "probabilities": {"0": 1.0-self.0, "1": self.0}}
                }}),
            ).map(|answers| (answers, self.model_id().to_owned()))
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
        for obs in output.observations() {
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
