//! Metrics engine: storage, aggregation, and query execution.
//!
//! This crate is the serving/engine half of the metrics substrate
//! (AGPL-3.0-only): DuckDB and DataFusion backends, the query planner and
//! orchestrator, metric aggregation execution, and checkpointing. The public
//! record types, schema, producer API, and the calibration / decision-batch
//! stores live in the Apache-2.0 `hyprstream-metrics-api` crate; everything
//! consumers need is re-exported here so existing `hyprstream_metrics::...`
//! paths keep working.

pub mod checkpoint;
pub mod config;
pub mod error;
pub mod metrics;
pub mod query;
pub mod storage;
pub mod utils;

pub use hyprstream_metrics_api::aggregation;
pub use aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
pub use checkpoint::{
    Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata, RecoveryManager,
    RecoveryStatus, RegistryClient, RegistryError,
};
pub use metrics::calibration::{
    CALIBRATION_TABLE, CalibrationParams, CalibrationStore, ConformalParams, ConformalQuantile,
    DecisionPrimitive, DirichletParams, OrdinalInterval, VersionTriple, calibration_params_schema,
    mint_calib_version,
};
pub use metrics::decision_batch::{
    DECISION_BATCH_TABLE, DecisionBatchMeta, DecisionBatchStore, decision_batch_schema,
    mint_batch_id,
};
pub use metrics::producer::MetricsProducer;
pub use metrics::{METRICS_SCHEMA_VERSION, METRICS_TABLE, create_schema_v2_tables};
pub use query::{
    CachedStatement, DataFusionExecutor, DataFusionPlanner, ExecutorConfig, Query,
    QueryOrchestrator,
};
pub use storage::StorageBackend;
pub use storage::context::{ContextRecord, ContextStore, SearchResult, context_schema, DEFAULT_EMBEDDING_DIM};

// Re-export Arrow types from DuckDB for consistency
pub use duckdb::arrow;
