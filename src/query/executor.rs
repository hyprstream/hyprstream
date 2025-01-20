use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::context::{SessionConfig, SessionContext, TaskContext};
use datafusion::execution::runtime_env::{RuntimeConfig, RuntimeEnv};
use datafusion::physical_plan::ExecutionPlan;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;

/// Trait for executing physical query plans
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// Execute a physical plan and return the results as a stream of record batches
    async fn execute_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>>;

    /// Execute a physical plan and collect all results into a vector
    async fn execute_collect(&self, plan: Arc<dyn ExecutionPlan>) -> Result<Vec<RecordBatch>> {
        let mut stream = self.execute_stream(plan).await?;
        let mut results = Vec::new();
        while let Some(batch) = stream.next().await {
            results.push(batch?);
        }
        Ok(results)
    }
}

/// Default executor implementation using DataFusion
pub struct DataFusionExecutor {
    /// Runtime configuration
    config: ExecutorConfig,
    /// Session context
    ctx: SessionContext,
}

/// Configuration for the executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Memory limit per query in bytes
    pub memory_limit: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get(),
            batch_size: 8192,
            memory_limit: 1024 * 1024 * 1024, // 1GB
        }
    }
}

impl DataFusionExecutor {
    pub fn new(config: ExecutorConfig) -> Self {
        let ctx = SessionContext::new();
        Self { config, ctx }
    }

    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        self.config = config;
        self
    }

    fn create_runtime_env(&self) -> Result<Arc<RuntimeEnv>> {
        let config = RuntimeConfig::new();
        RuntimeEnv::new(config).map(Arc::new)
    }
}

#[async_trait::async_trait]
impl QueryExecutor for DataFusionExecutor {
    async fn execute_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
        use std::collections::HashMap;

        // Create runtime environment
        let runtime = self.create_runtime_env()?;

        // Create session config
        let session_config = SessionConfig::new().with_batch_size(self.config.batch_size);

        // Create task context with minimal configuration
        let task_ctx = Arc::new(TaskContext::new(
            Some("query_execution".to_string()),
            "task_1".to_string(),
            session_config,
            HashMap::new(), // scalar functions
            HashMap::new(), // aggregate functions
            HashMap::new(), // window functions
            runtime,
        ));

        // Execute the plan
        let stream = plan.execute(0, task_ctx)?;

        Ok(Box::pin(stream))
    }
}

/// Execution metrics for monitoring and optimization
#[derive(Debug, Default)]
pub struct ExecutionMetrics {
    /// Number of records processed
    pub records_processed: usize,
    /// Number of batches processed
    pub batches_processed: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

// Add missing imports
use futures::StreamExt;
