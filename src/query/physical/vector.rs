//! Vectorized operator implementation for efficient vector operations
//!
//! This module provides SIMD-accelerated implementations of common vector
//! operations used in query processing.

use super::{BaseOperator, PhysicalOperator};
use datafusion::arrow::array::{Array, ArrayRef, Float32Array, Float64Array};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// Types of vector operations supported
#[derive(Debug, Clone, Copy)]
pub enum VectorOperation {
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Multiply,
    /// Dot product
    DotProduct,
    /// L2 normalization
    Normalize,
    /// Cosine similarity
    CosineSimilarity,
}

impl std::str::FromStr for VectorOperation {
    type Err = DataFusionError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "add" => Ok(VectorOperation::Add),
            "multiply" => Ok(VectorOperation::Multiply),
            "dot_product" => Ok(VectorOperation::DotProduct),
            "normalize" => Ok(VectorOperation::Normalize),
            "cosine_similarity" => Ok(VectorOperation::CosineSimilarity),
            _ => Err(DataFusionError::NotImplemented(format!(
                "Vector operation {} not implemented",
                s
            ))),
        }
    }
}

/// Operator for vectorized computations
pub struct VectorizedOperator {
    /// Base operator implementation
    base: BaseOperator,
    /// Type of vector operation
    operation: VectorOperation,
    /// Input column names
    input_columns: Vec<String>,
    /// Output column name
    // TODO: Use this field when implementing column renaming in execute method
    #[allow(dead_code)]
    output_column: String,
}

impl VectorizedOperator {
    pub fn new(
        schema: Arc<Schema>,
        children: Vec<Arc<dyn PhysicalOperator>>,
        properties: HashMap<String, String>,
    ) -> Result<Self> {
        // Parse operation type
        let operation = properties
            .get("operation")
            .ok_or_else(|| DataFusionError::Plan("Missing operation type".to_string()))?
            .parse()?;

        // Get input and output columns
        let input_columns = properties
            .get("input_columns")
            .ok_or_else(|| DataFusionError::Plan("Missing input columns".to_string()))?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let output_column = properties
            .get("output_column")
            .ok_or_else(|| DataFusionError::Plan("Missing output column".to_string()))?
            .clone();

        Ok(Self {
            base: BaseOperator::new(schema, children),
            operation,
            input_columns,
            output_column,
        })
    }

    /// Execute vector operation on arrays
    fn execute_vector_op(&self, left: &ArrayRef, right: &ArrayRef) -> Result<ArrayRef> {
        match self.operation {
            VectorOperation::Add => {
                // Element-wise addition
                match (left.data_type(), right.data_type()) {
                    (DataType::Float32, DataType::Float32) => {
                        let l = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float32Array>().unwrap();
                        let mut result = Vec::with_capacity(l.len());
                        for i in 0..l.len() {
                            result.push(l.value(i) + r.value(i));
                        }
                        Ok(Arc::new(Float32Array::from(result)) as ArrayRef)
                    }
                    (DataType::Float64, DataType::Float64) => {
                        let l = left.as_any().downcast_ref::<Float64Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float64Array>().unwrap();
                        let mut result = Vec::with_capacity(l.len());
                        for i in 0..l.len() {
                            result.push(l.value(i) + r.value(i));
                        }
                        Ok(Arc::new(Float64Array::from(result)) as ArrayRef)
                    }
                    _ => Err(DataFusionError::NotImplemented(
                        "Unsupported data type for vector addition".to_string(),
                    )),
                }
            }
            VectorOperation::Multiply => {
                // Element-wise multiplication
                match (left.data_type(), right.data_type()) {
                    (DataType::Float32, DataType::Float32) => {
                        let l = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float32Array>().unwrap();
                        let mut result = Vec::with_capacity(l.len());
                        for i in 0..l.len() {
                            result.push(l.value(i) * r.value(i));
                        }
                        Ok(Arc::new(Float32Array::from(result)) as ArrayRef)
                    }
                    (DataType::Float64, DataType::Float64) => {
                        let l = left.as_any().downcast_ref::<Float64Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float64Array>().unwrap();
                        let mut result = Vec::with_capacity(l.len());
                        for i in 0..l.len() {
                            result.push(l.value(i) * r.value(i));
                        }
                        Ok(Arc::new(Float64Array::from(result)) as ArrayRef)
                    }
                    _ => Err(DataFusionError::NotImplemented(
                        "Unsupported data type for vector multiplication".to_string(),
                    )),
                }
            }
            VectorOperation::DotProduct => {
                // Dot product using element-wise multiply and sum
                match (left.data_type(), right.data_type()) {
                    (DataType::Float32, DataType::Float32) => {
                        let l = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float32Array>().unwrap();
                        let mut sum = 0.0;
                        for i in 0..l.len() {
                            sum += l.value(i) * r.value(i);
                        }
                        Ok(Arc::new(Float32Array::from(vec![sum])) as ArrayRef)
                    }
                    _ => Err(DataFusionError::NotImplemented(
                        "Unsupported data type for dot product".to_string(),
                    )),
                }
            }
            VectorOperation::Normalize => {
                // L2 normalization
                match left.data_type() {
                    DataType::Float32 => {
                        let arr = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        let norm = (arr.iter().flatten().map(|x| x * x).sum::<f32>()).sqrt();
                        let normalized = arr.iter().map(|x| x.map(|v| v / norm));
                        Ok(Arc::new(Float32Array::from_iter(normalized)) as ArrayRef)
                    }
                    _ => Err(DataFusionError::NotImplemented(
                        "Unsupported data type for normalization".to_string(),
                    )),
                }
            }
            VectorOperation::CosineSimilarity => {
                // Cosine similarity using dot product and norms
                match (left.data_type(), right.data_type()) {
                    (DataType::Float32, DataType::Float32) => {
                        let l = left.as_any().downcast_ref::<Float32Array>().unwrap();
                        let r = right.as_any().downcast_ref::<Float32Array>().unwrap();

                        let mut dot_product = 0.0;
                        for i in 0..l.len() {
                            dot_product += l.value(i) * r.value(i);
                        }

                        let l_norm = (l.iter().flatten().map(|x| x * x).sum::<f32>()).sqrt();
                        let r_norm = (r.iter().flatten().map(|x| x * x).sum::<f32>()).sqrt();

                        let similarity = dot_product / (l_norm * r_norm);
                        Ok(Arc::new(Float32Array::from(vec![similarity])) as ArrayRef)
                    }
                    _ => Err(DataFusionError::NotImplemented(
                        "Unsupported data type for cosine similarity".to_string(),
                    )),
                }
            }
        }
    }
}

impl PhysicalOperator for VectorizedOperator {
    fn schema(&self) -> &Arc<Schema> {
        self.base.schema()
    }

    fn execute(&self, input: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
        let mut output_batches = Vec::with_capacity(input.len());

        for batch in input {
            // Get input arrays
            let arrays: Result<Vec<ArrayRef>> = self
                .input_columns
                .iter()
                .map(|col| {
                    batch
                        .column_by_name(col)
                        .ok_or_else(|| DataFusionError::Plan(format!("Column {} not found", col)))
                        .map(|arr| arr.clone())
                })
                .collect();

            let arrays = arrays?;
            if arrays.len() != 2 {
                return Err(DataFusionError::Plan(
                    "Vector operations require exactly two input arrays".to_string(),
                ));
            }

            // Execute vector operation
            let result = self.execute_vector_op(&arrays[0], &arrays[1])?;

            // Create output batch
            let mut columns = batch.columns().to_vec();
            columns.push(result);

            let output_batch = RecordBatch::try_new(self.base.schema().clone(), columns)
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            output_batches.push(output_batch);
        }

        Ok(output_batches)
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalOperator>> {
        self.base.children()
    }
}
