//! Persistence: eval-v2 runs → P0.3 `decision_batches` rows via the Apache metrics
//! API. The harness links only [`hyprstream_metrics_api`]; the caller wires in
//! whatever [`StorageBackend`] the serving process provides (DuckDB in the
//! AGPL engine, an in-memory backend in tests) — the two-process serving
//! shape means this crate never sees an AGPL type.

use std::collections::HashSet;
use std::sync::Arc;

use hyprstream_decision::arrow::{
    DecisionSchema, CALIB_VERSION_COLUMN, MODEL_VERSION_COLUMN, SCHEMA_VERSION_COLUMN,
};
use hyprstream_decision::confidence::{self, PRODUCER_SUM_TOLERANCE};
use hyprstream_metrics_api::arrow::array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray,
};
use hyprstream_metrics_api::{DecisionBatchStore, StorageBackend};

use crate::error::EvalError;
use crate::eval_batch::EvalBatchSchema;
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
        output.validate_model_provenance()?;
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
pub(crate) fn validate_provenance(
    output: &RunOutput,
    batch: &RecordBatch,
    fingerprint: &str,
) -> Result<DecisionSchema, EvalError> {
    let invalid = |message: &str| EvalError::InvalidInput(message.to_owned());
    let eval_schema = EvalBatchSchema::from_arrow_schema(batch.schema().as_ref())?;
    let schema = eval_schema.decision().clone();
    if eval_schema.fingerprint() != fingerprint {
        return Err(invalid("eval-v2 fingerprint differs from the batch"));
    }
    if output.version.schema.is_empty()
        || output.version.model.is_empty()
        || output.version.calib.as_deref() == Some("")
    {
        return Err(invalid(
            "run requires nonempty schema, model, and present calibration versions",
        ));
    }
    validate_row_values(&schema, batch)?;
    eval_schema.validate_observations(output, batch)?;
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
    Ok(schema)
}

/// Inspect actual Arrow values: public RunOutput callers can replace columns
/// without changing the schema, fingerprint, or version triple.
fn validate_row_values(schema: &DecisionSchema, batch: &RecordBatch) -> Result<(), EvalError> {
    let invalid = |message: &str| EvalError::InvalidInput(message.to_owned());
    for question in schema.questions() {
        let labels = question.labels();
        let allowed: HashSet<&str> = labels.iter().map(String::as_str).collect();
        let probabilities = batch
            .column_by_name(&format!("{}.probabilities", question.id))
            .and_then(|column| column.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| invalid("batch has an invalid probability column"))?;
        let predicted = batch
            .column_by_name(&format!("{}.label", question.id))
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| invalid("batch has an invalid label column"))?;
        let conformal = batch
            .column_by_name(&format!("{}.conformal_set", question.id))
            .map(|column| {
                column
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| invalid("batch has an invalid conformal column"))
            })
            .transpose()?;
        for row in 0..batch.num_rows() {
            if probabilities.is_null(row) {
                if !predicted.is_null(row) || conformal.is_some_and(|sets| !sets.is_null(row)) {
                    return Err(invalid(
                        "abstained batch answer carries a label or conformal set",
                    ));
                }
                // Child slots of a null parent are not logical answer values.
                continue;
            }
            if predicted.is_null(row) {
                return Err(invalid("answered batch row has a null label"));
            }
            let values = probabilities.value(row);
            let values = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| invalid("batch has invalid probability values"))?;
            if values.null_count() != 0 {
                return Err(invalid(
                    "answered batch distribution contains a null component",
                ));
            }
            let distribution = values.values().as_ref();
            confidence::check_distribution(distribution, PRODUCER_SUM_TOLERANCE)
                .map_err(|_| invalid("batch contains an invalid probability distribution"))?;
            let winner = confidence::argmax_index(distribution)
                .ok_or_else(|| invalid("batch contains an empty probability distribution"))?;
            if predicted.value(row) != labels[winner] {
                return Err(invalid(
                    "batch label differs from the canonical distribution argmax",
                ));
            }
            if let Some(sets) = conformal.filter(|sets| !sets.is_null(row)) {
                let members = sets.value(row);
                let members = members
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| invalid("batch has invalid conformal values"))?;
                if members
                    .iter()
                    .any(|member| member.is_none_or(|label| !allowed.contains(label)))
                {
                    return Err(invalid(
                        "batch conformal set contains a null or unknown label",
                    ));
                }
            }
        }
    }
    Ok(())
}
