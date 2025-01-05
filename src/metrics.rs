//! Metrics module for real-time aggregation and time-windowed calculations.
//!
//! This module provides the core functionality for:
//! - Dynamic metric calculations including running sums, averages, and counts
//! - Time-windowed aggregations for granular analysis
//! - Efficient state management for aggregate calculations
//! - Arrow-compatible data structures for high-performance data transport
//!
//! The metrics system is designed to support real-time analytics and ML/AI pipelines
//! with minimal latency and memory overhead.

use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use lazy_static::lazy_static;
use std::sync::Arc;
use tonic::Status;

/// Represents a single metric record with aggregated values over a time window.
///
/// Each record contains:
/// - A unique metric identifier
/// - Timestamp for the window
/// - Running calculations including sum, average, and count
///
/// This structure is optimized for real-time updates and efficient serialization
/// to Arrow format for transport.
#[derive(Debug, Clone)]
pub struct MetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp (seconds since epoch)
    pub timestamp: i64,
    /// Running sum of values within the current window
    pub value_running_window_sum: f64,
    /// Running average of values within the current window
    pub value_running_window_avg: f64,
    /// Count of values within the current window
    pub value_running_window_count: i64,
}

// Helper function to get the metrics schema - use lazy_static for zero-copy reuse
lazy_static! {
    /// Arrow schema for metric records, optimized for columnar storage and transport.
    ///
    /// The schema defines the structure for storing metrics in Arrow format with fields for:
    /// - metric_id (String): Unique identifier
    /// - timestamp (Int64): Unix timestamp
    /// - value_running_window_sum (Float64): Running sum
    /// - value_running_window_avg (Float64): Running average
    /// - value_running_window_count (Int64): Count of values
    pub static ref METRICS_SCHEMA: Schema = Schema::new(vec![
        Arc::new(Field::new("metric_id", DataType::Utf8, false)),
        Arc::new(Field::new("timestamp", DataType::Int64, false)),
        Arc::new(Field::new(
            "value_running_window_sum",
            DataType::Float64,
            false
        )),
        Arc::new(Field::new(
            "value_running_window_avg",
            DataType::Float64,
            false
        )),
        Arc::new(Field::new(
            "value_running_window_count",
            DataType::Int64,
            false
        )),
    ]);
}

/// Creates an Arrow RecordBatch from a vector of MetricRecords.
///
/// This function efficiently converts metric records into Arrow's columnar format
/// for high-performance transport and querying. The resulting RecordBatch uses
/// the standard metrics schema defined in `METRICS_SCHEMA`.
///
/// # Arguments
///
/// * `records` - Vector of MetricRecord instances to convert
///
/// # Returns
///
/// * `Result<RecordBatch, Status>` - Arrow RecordBatch on success, or error status
pub fn create_record_batch(records: Vec<MetricRecord>) -> Result<RecordBatch, Status> {
    // Pre-allocate vectors with known capacity
    let mut metric_ids = Vec::with_capacity(records.len());
    let mut timestamps = Vec::with_capacity(records.len());
    let mut sums = Vec::with_capacity(records.len());
    let mut avgs = Vec::with_capacity(records.len());
    let mut counts = Vec::with_capacity(records.len());

    // Single pass over records to fill all vectors
    for record in &records {
        metric_ids.push(record.metric_id.as_str());
        timestamps.push(record.timestamp);
        sums.push(record.value_running_window_sum);
        avgs.push(record.value_running_window_avg);
        counts.push(record.value_running_window_count);
    }

    RecordBatch::try_new(
        Arc::new(METRICS_SCHEMA.clone()),
        vec![
            Arc::new(StringArray::from(metric_ids)),
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Float64Array::from(sums)),
            Arc::new(Float64Array::from(avgs)),
            Arc::new(Int64Array::from(counts)),
        ],
    )
    .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))
}

/// Encodes an Arrow RecordBatch into IPC format for transport.
///
/// This function serializes both the schema and data of a RecordBatch into
/// Arrow's IPC format, suitable for transport over Arrow Flight. The encoding
/// is optimized for columnar data transport.
///
/// # Arguments
///
/// * `batch` - Reference to the RecordBatch to encode
///
/// # Returns
///
/// * `Result<(Vec<u8>, Vec<u8>), Status>` - Tuple of (schema buffer, data buffer) on success
pub fn encode_record_batch(batch: &RecordBatch) -> Result<(Vec<u8>, Vec<u8>), Status> {
    let mut schema_buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut schema_buffer, batch.schema().as_ref())
            .map_err(|e| Status::internal(format!("Failed to create schema writer: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("Failed to write schema: {}", e)))?;
    }

    let mut data_buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut data_buffer, batch.schema().as_ref())
            .map_err(|e| Status::internal(format!("Failed to create writer: {}", e)))?;
        writer
            .write(batch)
            .map_err(|e| Status::internal(format!("Failed to write batch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| Status::internal(format!("Failed to finish writing: {}", e)))?;
    }

    Ok((schema_buffer, data_buffer))
}
