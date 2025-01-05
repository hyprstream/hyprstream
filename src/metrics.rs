use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use lazy_static::lazy_static;
use std::sync::Arc;
use tonic::Status;

#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub metric_id: String,
    pub timestamp: i64,
    pub value_running_window_sum: f64,
    pub value_running_window_avg: f64,
    pub value_running_window_count: i64,
}

// Helper function to get the metrics schema - use lazy_static for zero-copy reuse
lazy_static! {
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
