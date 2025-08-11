pub mod storage;

// Arrow dependencies removed - using simplified metrics for ML inference
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::sync::Arc;
use tonic::Status;

pub use storage::{VDBMetricsStorage, VDBMetricsStorageImpl};

/// A single metric record with running window calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp in seconds
    pub timestamp: i64,
    /// Metric value
    pub value: f64,
    /// Running sum within the window
    pub value_running_window_sum: f64,
    /// Running average within the window
    pub value_running_window_avg: f64,
    /// Running count within the window
    pub value_running_window_count: i64,
}

/// Simplified metric record for VDB-first architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp in seconds
    pub timestamp: i64,
    /// Metric value
    pub value: f64,
}

// Arrow RecordBatch conversion removed - simplified for ML inference focus

impl Default for MetricRecord {
    fn default() -> Self {
        Self {
            metric_id: String::new(),
            timestamp: 0,
            value: 0.0,
            value_running_window_sum: 0.0,
            value_running_window_avg: 0.0,
            value_running_window_count: 0,
        }
    }
}

impl From<SimpleMetricRecord> for MetricRecord {
    fn from(simple: SimpleMetricRecord) -> Self {
        Self {
            metric_id: simple.metric_id,
            timestamp: simple.timestamp,
            value: simple.value,
            value_running_window_sum: simple.value,
            value_running_window_avg: simple.value,
            value_running_window_count: 1,
        }
    }
}

// Arrow schema and RecordBatch functions removed - focused on ML inference metrics only
