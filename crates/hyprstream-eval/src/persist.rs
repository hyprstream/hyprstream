//! Persistence: runs → P0.3 `decision_batches` rows via the Apache metrics
//! API. The harness links only [`hyprstream_metrics_api`]; the caller wires in
//! whatever [`StorageBackend`] the serving process provides (DuckDB in the
//! AGPL engine, an in-memory backend in tests) — the two-process serving
//! shape means this crate never sees an AGPL type.

use std::sync::Arc;

use hyprstream_decision::arrow::{
    DecisionSchema, CALIB_VERSION_COLUMN, FIELD_META_KIND, FIELD_META_LABELS, MODEL_VERSION_COLUMN,
    SCHEMA_VERSION_COLUMN,
};
use hyprstream_decision::spec::{ChoiceOption, QuestionBody, QuestionKind, QuestionSpec};
use hyprstream_metrics_api::arrow::array::{Array, RecordBatch, StringArray};
use hyprstream_metrics_api::{DecisionBatchStore, StorageBackend};

use crate::error::EvalError;
use crate::run::RunOutput;

/// A persistence sink over any storage backend.
pub struct BatchSink<B: StorageBackend> {
    store: DecisionBatchStore<B>,
}

impl<B: StorageBackend> BatchSink<B> {
    /// Wrap a backend; call [`BatchSink::init`] once before first use.
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            store: DecisionBatchStore::new(backend),
        }
    }

    /// Create the `decision_batches` table (idempotent).
    pub async fn init(&self) -> Result<(), EvalError> {
        self.store.init().await.map_err(EvalError::from)
    }

    /// The underlying store (reads, listings).
    pub fn store(&self) -> &DecisionBatchStore<B> {
        &self.store
    }

    /// Persist a run's Arrow batch. Item runs carry no batch (their questions
    /// are not a shared schema) — persist those per-set instead, or not at
    /// all; the observation stream is the report artifact there.
    ///
    /// Empty set runs remain usable as report artifacts, but cannot be persisted:
    /// the store has no schema-version metadata and an empty version column
    /// cannot carry that component of the version triple.
    ///
    /// The detached fingerprint and complete version triple must agree with
    /// the canonical Arrow payload before any backend write.
    ///
    /// Returns the minted batch id.
    pub async fn persist_run(&self, output: &RunOutput) -> Result<String, EvalError> {
        let batch = output.batch.as_ref().ok_or_else(|| {
            EvalError::InvalidInput(
                "run has no Arrow batch (loose item runs are not persistable as one batch)"
                    .to_owned(),
            )
        })?;
        if batch.num_rows() == 0 {
            return Err(EvalError::InvalidInput(
                "cannot persist an empty run: schema version would be lost".to_owned(),
            ));
        }
        let fingerprint = output
            .schema_fingerprint
            .as_deref()
            .ok_or_else(|| EvalError::InvalidInput("run has no schema fingerprint".to_owned()))?;
        validate_provenance(output, batch, fingerprint)?;
        self.store
            .put(
                fingerprint,
                &output.version.model,
                output.version.calib.as_deref(),
                batch,
            )
            .await
            .map_err(EvalError::from)
    }
}

/// RunOutput is public and mutable. Reconstruct the canonical emission schema
/// from the payload itself before trusting any detached persistence metadata.
fn validate_provenance(
    output: &RunOutput,
    batch: &RecordBatch,
    fingerprint: &str,
) -> Result<(), EvalError> {
    let invalid = |message: &str| EvalError::InvalidInput(message.to_owned());
    let arrow_schema = batch.schema();
    let mut questions = Vec::new();
    for field in arrow_schema.fields() {
        let Some(id) = field.name().strip_suffix(".probabilities") else {
            continue;
        };
        let kind = match field.metadata().get(FIELD_META_KIND).map(String::as_str) {
            Some("noul") => QuestionKind::Noul,
            Some("choice") => QuestionKind::Choice,
            Some("score") => QuestionKind::Score,
            _ => {
                return Err(invalid(
                    "batch has missing or invalid question kind metadata",
                ))
            }
        };
        let labels: Vec<String> = serde_json::from_str(
            field
                .metadata()
                .get(FIELD_META_LABELS)
                .ok_or_else(|| invalid("batch has no question labels metadata"))?,
        )
        .map_err(|_| invalid("batch has invalid question labels metadata"))?;
        let body = match kind {
            QuestionKind::Noul => QuestionBody::Noul { criteria: None },
            QuestionKind::Choice => QuestionBody::Choice {
                options: labels
                    .iter()
                    .map(|name| ChoiceOption {
                        name: name.clone(),
                        rubric: None,
                    })
                    .collect(),
            },
            QuestionKind::Score => QuestionBody::Score {
                levels: vec![None; labels.len()],
            },
            _ => return Err(invalid("batch question kind is not supported in v1")),
        };
        let question = QuestionSpec {
            id: id.to_owned(),
            kind,
            instructions: None,
            body,
        };
        crate::run::validate_question_profile(&question).map_err(EvalError::InvalidInput)?;
        if question.labels() != labels {
            return Err(invalid("batch question labels disagree with its kind"));
        }
        questions.push(question);
    }
    let schema = DecisionSchema::new(questions)?;
    let schema = if arrow_schema
        .fields()
        .iter()
        .any(|field| field.name().ends_with(".conformal_set"))
    {
        schema
    } else {
        schema.without_conformal_set()
    };
    if schema.arrow_schema() != *arrow_schema || schema.fingerprint() != fingerprint {
        return Err(invalid(
            "run schema fingerprint or Arrow layout differs from the batch",
        ));
    }
    if output.version.schema.is_empty()
        || output.version.model.is_empty()
        || output.version.calib.as_deref() == Some("")
    {
        return Err(invalid(
            "run requires nonempty schema, model, and present calibration versions",
        ));
    }
    for (name, expected) in [
        (SCHEMA_VERSION_COLUMN, Some(output.version.schema.as_str())),
        (MODEL_VERSION_COLUMN, Some(output.version.model.as_str())),
        (CALIB_VERSION_COLUMN, output.version.calib.as_deref()),
    ] {
        let values = batch
            .column_by_name(name)
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| invalid("batch has a missing or invalid version column"))?;
        for row in 0..batch.num_rows() {
            let actual = (!values.is_null(row)).then(|| values.value(row));
            if actual != expected {
                return Err(invalid(
                    "run version triple differs from an embedded batch row",
                ));
            }
        }
    }
    Ok(())
}
