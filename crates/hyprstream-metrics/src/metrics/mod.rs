//! Metric record types, stores, and metric-specific aggregation.
//!
//! The public record types, schema, producer API, and the calibration /
//! decision-batch stores live in the Apache-2.0 `hyprstream-metrics-api`
//! crate and are re-exported here unchanged; this module adds the
//! engine-side metric aggregation and storage execution.

// tonic::Status is the idiomatic gRPC error type - boxing would break API
#![allow(clippy::result_large_err)]

pub mod storage;

#[cfg(test)]
mod api_store_tests;

pub use hyprstream_metrics_api::metrics::*;
pub use storage::{MetricsStorage, MetricsStorageImpl};
