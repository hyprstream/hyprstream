//! Physical operators for query execution
//!
//! This module provides implementations of physical operators that can be used
//! in query execution plans. These operators are optimized for performance and
//! include specialized implementations for vector operations.

mod vector;

pub use vector::VectorizedOperator;

use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use std::sync::Arc;

/// Trait for physical operators that can be executed
pub trait PhysicalOperator: Send + Sync {
    /// Get the schema of the output
    fn schema(&self) -> &Arc<Schema>;

    /// Execute the operator and produce output batches
    fn execute(&self, input: Vec<RecordBatch>) -> Result<Vec<RecordBatch>>;

    /// Get children operators
    fn children(&self) -> Vec<Arc<dyn PhysicalOperator>>;
}

/// Statistics about operator execution
#[derive(Debug, Default)]
pub struct OperatorMetrics {
    /// Number of input rows processed
    pub input_rows: usize,
    /// Number of output rows produced
    pub output_rows: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory used in bytes
    pub memory_used: usize,
}

/// Base implementation for physical operators
pub struct BaseOperator {
    /// Output schema
    schema: Arc<Schema>,
    /// Child operators
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// Execution metrics
    metrics: Arc<parking_lot::RwLock<OperatorMetrics>>,
}

impl BaseOperator {
    pub fn new(schema: Arc<Schema>, children: Vec<Arc<dyn PhysicalOperator>>) -> Self {
        Self {
            schema,
            children,
            metrics: Arc::new(parking_lot::RwLock::new(OperatorMetrics::default())),
        }
    }

    pub fn metrics(&self) -> Arc<parking_lot::RwLock<OperatorMetrics>> {
        self.metrics.clone()
    }
}

impl PhysicalOperator for BaseOperator {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    fn execute(&self, input: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
        // Base implementation just passes through input
        Ok(input)
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalOperator>> {
        self.children.clone()
    }
}

/// Factory for creating physical operators
pub struct OperatorFactory;

impl OperatorFactory {
    /// Create a new physical operator
    pub fn create(
        operator_type: &str,
        schema: Arc<Schema>,
        children: Vec<Arc<dyn PhysicalOperator>>,
        properties: std::collections::HashMap<String, String>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        match operator_type {
            "vector" => Ok(Arc::new(vector::VectorizedOperator::new(
                schema, children, properties,
            )?)),
            _ => Err(datafusion::error::DataFusionError::NotImplemented(format!(
                "Operator type {operator_type} not implemented"
            ))),
        }
    }
}
