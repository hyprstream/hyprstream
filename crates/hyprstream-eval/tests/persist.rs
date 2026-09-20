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

    fn filter(
        &self,
        table: &str,
        where_col: Option<(&str, &str)>,
    ) -> Result<RecordBatch, Status> {
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
            col.iter().map(|v| Some(v == Some(value))).collect::<Vec<_>>(),
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
        let from_pos = sql.find(" FROM ").ok_or_else(|| Self::err("no FROM clause"))?;
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
        self.tables
            .lock()
            .insert(table_name.to_owned(), (Arc::new(schema.clone()), Vec::new()));
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
        Ok(self
            .tables
            .lock()
            .keys()
            .cloned()
            .collect())
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
        self.tables
            .lock()
            .remove(table_name);
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
