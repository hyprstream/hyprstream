use std::sync::Arc;
use arrow_array::{Array, RecordBatch as ArrowRecordBatch, Int64Array, Float64Array, StringArray, ArrayRef};
use arrow_schema::DataType;
use arrow_convert::serialize::TryIntoArrow;
use rayon::prelude::*;
use tonic::Status;
use tracing;

#[derive(Debug)]
pub enum ZeroCopyError {
    IncompatibleLayout,
    MetadataError,
    Other(String),
}

impl From<ZeroCopyError> for Status {
    fn from(e: ZeroCopyError) -> Self {
        Status::internal(e.to_string())
    }
}

impl std::fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZeroCopyError::IncompatibleLayout => write!(f, "Incompatible memory layout"),
            ZeroCopyError::MetadataError => write!(f, "Missing or invalid metadata"),
            ZeroCopyError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

impl std::error::Error for ZeroCopyError {}

/// Trait that data sources can implement if they can produce Arrow-compatible buffers.
pub trait ZeroCopySource {
    /// Attempts to return an Arrow `RecordBatch` referencing the existing buffers.
    fn try_zero_copy(&self, parallel_threshold: usize) -> Result<ArrowRecordBatch, ZeroCopyError>;

    /// Returns the schema of the data source
    fn schema(&self) -> arrow_schema::Schema;

    /// Returns the number of rows in the data source
    fn num_rows(&self) -> usize;

    /// Returns a reference to the column at the given index
    fn column(&self, i: usize) -> &dyn Array;
}

impl ZeroCopySource for duckdb::arrow::array::RecordBatch {
    fn try_zero_copy(&self, parallel_threshold: usize) -> Result<ArrowRecordBatch, ZeroCopyError> {
        let arrays: Result<Vec<ArrayRef>, _> = if self.columns().len() >= parallel_threshold {
            tracing::debug!(
                "Using parallel processing for {} columns (threshold: {})",
                self.columns().len(),
                parallel_threshold
            );
            
            self.columns()
                .par_iter()
                .enumerate()
                .map(|(i, col)| {
                    tracing::trace!(column_index = i, "Processing column in parallel");
                    process_column(i, col)
                })
                .collect()
        } else {
            tracing::trace!(
                "Using sequential processing for {} columns (threshold: {})",
                self.columns().len(),
                parallel_threshold
            );
            
            self.columns()
                .iter()
                .enumerate()
                .map(|(i, col)| process_column(i, col))
                .collect()
        };

        match arrays {
            Ok(arrays) => {
                let schema = Arc::new(self.schema());
                ArrowRecordBatch::try_new(schema, arrays)
                    .map_err(|e| ZeroCopyError::Other(e.to_string()))
            },
            Err(e) => Err(e)
        }
    }

    fn schema(&self) -> arrow_schema::Schema {
        self.schema().clone()
    }

    fn num_rows(&self) -> usize {
        self.num_rows()
    }

    fn column(&self, i: usize) -> &dyn Array {
        self.column(i).as_ref()
    }
}