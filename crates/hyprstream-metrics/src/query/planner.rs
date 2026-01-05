use crate::query::rules::view::ViewOptimizationRule;
use crate::storage::StorageBackend;
use datafusion::arrow::datatypes::Schema;
use crate::error::StatusWrapper;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimizer::OptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::datasource::memory::MemTable;
use std::sync::Arc;

/// Represents a query that can be planned and executed
#[derive(Debug, Clone)]
pub struct Query {
    /// SQL query string
    pub sql: String,
    /// Optional schema hint for better planning
    pub schema_hint: Option<Schema>,
    /// Query optimization hints
    pub hints: Vec<OptimizationHint>,
}

/// Optimization hints that can be passed to the query planner
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Prefer pushing predicates to storage
    PreferPredicatePushdown,
    /// Prefer parallel execution where possible
    PreferParallelExecution,
    /// Optimize for streaming execution
    OptimizeForStreaming,
    /// Optimize for vector operations
    OptimizeForVectorOps,
    /// Use views when possible
    PreferViews,
}

/// Core trait for query planning and optimization
#[async_trait::async_trait]
pub trait QueryPlanner: Send + Sync {
    /// Create a logical plan from a query
    async fn create_logical_plan(&self, query: &Query) -> Result<LogicalPlan>;

    /// Optimize a logical plan into a physical plan
    async fn optimize(&self, plan: LogicalPlan) -> Result<Arc<dyn ExecutionPlan>>;

    /// Create and optimize a physical plan directly from a query
    async fn plan_query(&self, query: &Query) -> Result<Arc<dyn ExecutionPlan>> {
        let logical_plan = self.create_logical_plan(query).await?;
        self.optimize(logical_plan).await
    }
}

/// Default query planner implementation using DataFusion
pub struct DataFusionPlanner {
    /// DataFusion context for planning and optimization
    ctx: SessionContext,
    /// Custom optimization rules
    optimizations: Vec<Box<dyn OptimizerRule + Send + Sync>>,
    /// Storage backend for view lookups
    storage: Arc<dyn StorageBackend>,
}

impl DataFusionPlanner {
    pub async fn new(storage: Arc<dyn StorageBackend>) -> Result<Self> {
        let ctx = SessionContext::new();
        let mut planner = Self {
            ctx,
            optimizations: Vec::new(),
            storage: storage.clone(),
        };

        // Add default optimization rules
        planner.add_default_rules();

        // Register tables from storage
        let tables = storage.list_tables().await.map_err(|e| Into::<DataFusionError>::into(StatusWrapper(e)))?;
        for table_name in tables {
            let schema = storage.get_table_schema(&table_name).await.map_err(|e| Into::<DataFusionError>::into(StatusWrapper(e)))?;
            // Use vec![vec![]] to provide one empty partition (required by MemTable)
            planner.ctx.register_table(
                &table_name,
                Arc::new(MemTable::try_new(schema, vec![vec![]])?),
            )?;
        }

        Ok(planner)
    }

    /// Add a custom optimization rule
    pub fn with_optimization(mut self, rule: Box<dyn OptimizerRule + Send + Sync>) -> Self {
        self.optimizations.push(rule);
        self
    }

    /// Add default optimization rules
    fn add_default_rules(&mut self) {
        // Add view optimization rule
        self.optimizations.push(Box::new(ViewOptimizationRule::new(
            self.storage.clone(),
        )));
    }
}

#[async_trait::async_trait]
impl QueryPlanner for DataFusionPlanner {
    async fn create_logical_plan(&self, query: &Query) -> Result<LogicalPlan> {
        // Parse SQL into logical plan
        let logical_plan = self.ctx.sql(&query.sql).await?.into_optimized_plan()?;

        // Apply schema hints if available
        if let Some(_schema) = &query.schema_hint {
            // TODO: Implement schema hint application
        }

        Ok(logical_plan)
    }

    async fn optimize(&self, plan: LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        // Get session state
        let state = self.ctx.state();

        // Apply custom optimization rules
        let mut optimized_plan = plan;
        for rule in &self.optimizations {
            optimized_plan = rule.rewrite(optimized_plan, &state)?.data;
        }

        // Convert to physical plan using session state
        state.create_physical_plan(&optimized_plan).await
    }
}

/// Context for optimization rules
#[derive(Debug)]
pub struct OptimizerContext {
    /// Statistics about the data
    pub statistics: Statistics,
    /// Available optimizations
    pub available_optimizations: Vec<OptimizationHint>,
}

impl OptimizerContext {
    pub fn new() -> Self {
        Self {
            statistics: Statistics::default(),
            available_optimizations: Vec::new(),
        }
    }
}

/// Statistics used for optimization decisions
#[derive(Debug, Default)]
pub struct Statistics {
    /// Estimated row count
    pub row_count: Option<usize>,
    /// Estimated total bytes
    pub total_bytes: Option<usize>,
    /// Column statistics
    pub column_statistics: HashMap<String, ColumnStatistics>,
}

/// Statistics for a single column
#[derive(Debug)]
pub struct ColumnStatistics {
    /// Number of distinct values
    pub distinct_count: Option<usize>,
    /// Number of null values
    pub null_count: Option<usize>,
    /// Minimum value
    pub min_value: Option<ScalarValue>,
    /// Maximum value
    pub max_value: Option<ScalarValue>,
}

// Add missing imports
use datafusion::scalar::ScalarValue;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::duckdb::DuckDbBackend;
    use duckdb::arrow::datatypes::{DataType, Field, Schema};

    #[tokio::test]
    async fn test_query_planning_with_views() -> Result<()> {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.unwrap();

        // Create test table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        backend.create_table("test_table", &schema).await.unwrap();

        // Create planner with backend
        let planner = DataFusionPlanner::new(backend).await?;

        // Create test query
        let query = Query {
            sql: "SELECT * FROM test_table".to_string(),
            schema_hint: None,
            hints: vec![OptimizationHint::PreferViews],
        };

        // Plan should succeed
        let plan = planner.create_logical_plan(&query).await?;
        assert!(plan.schema().fields().len() > 0);

        Ok(())
    }
}
