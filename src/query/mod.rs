//! Query planning and execution infrastructure
//!
//! This module provides the core query processing capabilities including:
//! - Query planning and optimization
//! - Physical plan execution
//! - Vectorized operations
//! - Performance monitoring

mod executor;
mod physical;
mod planner;

pub use executor::{DataFusionExecutor, ExecutorConfig, QueryExecutor};
pub use physical::{OperatorMetrics, PhysicalOperator, VectorizedOperator};
pub use planner::{DataFusionPlanner, OptimizationHint, Query, QueryPlanner};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use std::sync::Arc;

/// High-level query engine that combines planning and execution
pub struct QueryEngine {
    /// Query planner
    planner: Arc<dyn QueryPlanner>,
    /// Query executor
    executor: Arc<dyn QueryExecutor>,
}

impl QueryEngine {
    /// Create a new query engine with default configuration
    pub fn new() -> Self {
        let planner = Arc::new(DataFusionPlanner::new());
        let executor = Arc::new(DataFusionExecutor::new(ExecutorConfig::default()));
        Self { planner, executor }
    }

    /// Create a new query engine with custom components
    pub fn with_components(
        planner: Arc<dyn QueryPlanner>,
        executor: Arc<dyn QueryExecutor>,
    ) -> Self {
        Self { planner, executor }
    }

    /// Execute a query and return results as a vector of record batches
    pub async fn execute_query(&self, query: &Query) -> Result<Vec<RecordBatch>> {
        // Plan the query
        let physical_plan = self.planner.plan_query(query).await?;

        // Execute the plan
        self.executor.execute_collect(physical_plan).await
    }

    /// Execute a query and return results as a stream
    pub async fn execute_query_stream(
        &self,
        query: &Query,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
        // Plan the query
        let physical_plan = self.planner.plan_query(query).await?;

        // Execute the plan
        self.executor.execute_stream(physical_plan).await
    }
}

/// Builder for configuring and creating a QueryEngine
pub struct QueryEngineBuilder {
    executor_config: Option<ExecutorConfig>,
    optimization_hints: Vec<OptimizationHint>,
}

impl QueryEngineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            executor_config: None,
            optimization_hints: Vec::new(),
        }
    }

    /// Set the executor configuration
    pub fn with_executor_config(mut self, config: ExecutorConfig) -> Self {
        self.executor_config = Some(config);
        self
    }

    /// Add an optimization hint
    pub fn with_optimization_hint(mut self, hint: OptimizationHint) -> Self {
        self.optimization_hints.push(hint);
        self
    }

    /// Build the query engine
    pub fn build(self) -> QueryEngine {
        let executor_config = self.executor_config.unwrap_or_default();

        // Create planner with optimization hints
        let planner = Arc::new(DataFusionPlanner::new());

        // Create executor with config
        let executor = Arc::new(DataFusionExecutor::new(executor_config));

        QueryEngine::with_components(planner, executor)
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Add missing imports
use futures::Stream;
use std::pin::Pin;
