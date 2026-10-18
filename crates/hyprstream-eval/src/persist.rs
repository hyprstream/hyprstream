//! Persistence: runs → P0.3 `decision_batches` rows via the Apache metrics
//! API. The harness links only [`hyprstream_metrics_api`]; the caller wires in
//! whatever [`StorageBackend`] the serving process provides (DuckDB in the
//! AGPL engine, an in-memory backend in tests) — the two-process serving
//! shape means this crate never sees an AGPL type.

use std::sync::Arc;

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
    /// Returns the minted batch id.
    pub async fn persist_run(&self, output: &RunOutput) -> Result<String, EvalError> {
        let batch = output.batch.as_ref().ok_or_else(|| {
            EvalError::InvalidInput(
                "run has no Arrow batch (loose item runs are not persistable as one batch)"
                    .to_owned(),
            )
        })?;
        let fingerprint = output.schema_fingerprint.as_deref().ok_or_else(|| {
            EvalError::InvalidInput("run has no schema fingerprint".to_owned())
        })?;
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
