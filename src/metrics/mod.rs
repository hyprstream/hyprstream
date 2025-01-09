pub mod aggregation;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::array::{Array, StructBuilder};
use arrow_schema::{DataType, Field, Schema};
use arrow_convert::{ArrowField, ArrowSerialize};
use arrow_convert::serialize::TryIntoArrow;
use std::sync::Arc;
use tonic::Status;

/// A single metric record with running window calculations.
#[derive(Debug, Clone)]
pub struct MetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp in seconds
    pub timestamp: i64,
    /// Running sum within the window
    pub value_running_window_sum: f64,
    /// Running average within the window
    pub value_running_window_avg: f64,
    /// Running count within the window
    pub value_running_window_count: i64,
}

impl ArrowField for MetricRecord {
    fn arrow_field() -> Field {
        Field::new("metric_record", DataType::Struct(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]), false)
    }
}

impl ArrowSerialize for MetricRecord {
    fn arrow_serialize(&self) -> Result<Box<dyn Array>, Status> {
        let schema = Self::arrow_field().data_type().clone();
        let mut builder = arrow::array::StructBuilder::from_fields(
            match schema {
                DataType::Struct(fields) => fields,
                _ => unreachable!(),
            },
            self.metric_id.len(),
        );

        builder.field_builder::<StringArray>(0).unwrap().append_value(&self.metric_id);
        builder.field_builder::<Int64Array>(1).unwrap().append_value(self.timestamp);
        builder.field_builder::<Float64Array>(2).unwrap().append_value(self.value_running_window_sum);
        builder.field_builder::<Float64Array>(3).unwrap().append_value(self.value_running_window_avg);
        builder.field_builder::<Int64Array>(4).unwrap().append_value(self.value_running_window_count);
        builder.append(true);

        Ok(Box::new(builder.finish()))
    }
}

impl MetricRecord {
    pub fn try_from_record_batch(batch: &RecordBatch) -> Result<Vec<Self>, Status> {
        let metric_ids = batch
            .column_by_name("metric_id")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Invalid metric_id column"))?;

        let timestamps = batch
            .column_by_name("timestamp")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Status::internal("Invalid timestamp column"))?;

        let sums = batch
            .column_by_name("value_running_window_sum")
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_sum column"))?;

        let avgs = batch
            .column_by_name("value_running_window_avg")
            .and_then(|col| col.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_avg column"))?;

        let counts = batch
            .column_by_name("value_running_window_count")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Status::internal("Invalid value_running_window_count column"))?;

        let mut metrics = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            metrics.push(MetricRecord {
                metric_id: metric_ids.value(i).to_string(),
                timestamp: timestamps.value(i),
                value_running_window_sum: sums.value(i),
                value_running_window_avg: avgs.value(i),
                value_running_window_count: counts.value(i),
            });
        }

        Ok(metrics)
    }
}

/// Gets the schema for metric records in Arrow format.
pub fn get_metrics_schema() -> Schema {
    Schema::new(vec![
        Field::new("metric_id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("value_running_window_sum", DataType::Float64, false),
        Field::new("value_running_window_avg", DataType::Float64, false),
        Field::new("value_running_window_count", DataType::Int64, false),
    ])
}

/// Creates a RecordBatch from a vector of MetricRecords.
pub fn create_record_batch(metrics: &[MetricRecord]) -> Result<RecordBatch, Status> {
    let schema = get_metrics_schema();
    
    let arrays: Result<Vec<ArrayRef>, _> = metrics.iter()
        .map(|m| m.arrow_serialize().map(|arr| Arc::new(arr) as ArrayRef))
        .collect();
    
    let arrays = arrays.map_err(|e| Status::internal(format!("Failed to serialize metrics: {}", e)))?;

    RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))
}

/// Encodes a RecordBatch into a vector of MetricRecords.
pub fn encode_record_batch(batch: &RecordBatch) -> Result<Vec<MetricRecord>, Status> {
    MetricRecord::try_from_record_batch(batch)
} 