//! Metrics API: public types, schemas, and storage-abstract stores.
//!
//! This crate is the library half of the metrics substrate (Apache-2.0):
//! - Schema-v2 [`metrics::MetricRecord`] with labels and tenant id
//! - [`metrics::producer::MetricsProducer`], the producer-side write API
//! - [`metrics::calibration::CalibrationStore`]: versioned, append-only
//!   calibration parameter rows keyed by (primitive, schema, model, calib
//!   version)
//! - [`metrics::decision_batch::DecisionBatchStore`]: Arrow-native decision
//!   batches keyed by schema fingerprint
//! - The [`storage::StorageBackend`] trait and view/aggregation types the
//!   stores and the serving engine share.
//!
//! DuckDB/DataFusion engine implementations live in the `hyprstream-metrics`
//! crate behind these traits; the default serving shape is two-process, so
//! consumers link only this API crate.

pub mod aggregation;
pub mod metrics;
pub mod storage;

pub use aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
pub use metrics::calibration::{
    calibration_params_schema, mint_calib_version, CalibrationParams, CalibrationStore,
    ConformalParams, ConformalQuantile, DecisionPrimitive, DirichletParams, OrdinalInterval,
    VersionTriple, CALIBRATION_TABLE,
};
pub use metrics::decision_batch::{
    decision_batch_schema, mint_batch_id, DecisionBatchMeta, DecisionBatchStore,
    DECISION_BATCH_TABLE,
};
pub use metrics::decision_outcome::{
    create_decision_outcome_tables, mint_decision_id, mint_decision_row_id, mint_outcome_id,
    CalibExpectation, CalibFilter, DecisionOutcomeQuery, DecisionPins, DecisionRowMeta,
    DecisionRowQuery, DecisionRowStore, ExpectedPins, NewOutcome, OutcomeRecord, OutcomeStore,
    StoredDecision, TrustedTenant, DECISION_OUTCOMES_TABLE, DECISION_OUTCOME_SCHEMA_VERSION,
    DECISION_ROWS_TABLE,
};
pub use metrics::decision_outcome::{FIELD_META_KIND, FIELD_META_LABELS};
pub use metrics::producer::MetricsProducer;
pub use metrics::{create_schema_v2_tables, MetricRecord, METRICS_SCHEMA_VERSION, METRICS_TABLE};
pub use storage::StorageBackend;

// Re-export Arrow so API consumers name the exact types the stores exchange.
pub use arrow;
