use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::context::{SessionConfig, SessionContext, TaskContext};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Trait for executing physical query plans
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// Execute a physical plan and return the results as a stream of record batches
    async fn execute_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>>;

    /// Execute a physical plan and collect all results into a vector
    #[instrument(skip(self, plan))]
    async fn execute_collect(&self, plan: Arc<dyn ExecutionPlan>) -> Result<Vec<RecordBatch>> {
        debug!("Starting batch collection");
        let mut stream = self.execute_stream(plan).await?;
        let mut results = Vec::new();
        let mut total_rows = 0;
        
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            total_rows += batch.num_rows();
            debug!(
                batch_rows = batch.num_rows(),
                total_rows = total_rows,
                "Collected batch"
            );
            results.push(batch);
        }

        info!(
            total_batches = results.len(),
            total_rows = total_rows,
            "Query execution completed"
        );
        Ok(results)
    }
}

/// Default executor implementation using DataFusion
pub struct DataFusionExecutor {
    /// Runtime configuration
    config: ExecutorConfig,
    /// Session context - used indirectly through task context in execute_stream
    #[allow(dead_code)]
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
    #[instrument(skip(config))]
    pub fn new(config: ExecutorConfig) -> Self {
        debug!(
            max_tasks = config.max_concurrent_tasks,
            batch_size = config.batch_size,
            memory_limit = config.memory_limit,
            "Creating new DataFusionExecutor"
        );
        let ctx = SessionContext::new();
        info!("Created new DataFusionExecutor");
        Self { config, ctx }
    }

    #[instrument(skip(self, config))]
    pub fn with_config(mut self, config: ExecutorConfig) -> Self {
        debug!(
            old_max_tasks = self.config.max_concurrent_tasks,
            new_max_tasks = config.max_concurrent_tasks,
            old_batch_size = self.config.batch_size,
            new_batch_size = config.batch_size,
            old_memory_limit = self.config.memory_limit,
            new_memory_limit = config.memory_limit,
            "Updating executor configuration"
        );
        self.config = config;
        info!("Updated executor configuration");
        self
    }

    #[instrument(skip(self))]
    fn create_runtime_env(&self) -> Arc<RuntimeEnv> {
        debug!("Creating runtime environment");
        // In DataFusion 50, get RuntimeEnv from the SessionContext
        let env = self.ctx.runtime_env();
        debug!("Created runtime environment");
        env
    }
}

#[async_trait::async_trait]
impl QueryExecutor for DataFusionExecutor {
    #[instrument(skip(self, plan), fields(plan_type = ?plan.as_ref()))]
    async fn execute_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
        use std::collections::HashMap;

        debug!(
            schema = ?plan.schema(),
            partitions = ?plan.as_ref().output_partitioning(),
            "Starting query execution"
        );

        // Create runtime environment
        let runtime = self.create_runtime_env();
        debug!(
            max_tasks = self.config.max_concurrent_tasks,
            batch_size = self.config.batch_size,
            memory_limit = self.config.memory_limit,
            "Created runtime environment"
        );

        // Create session config
        let session_config = SessionConfig::new().with_batch_size(self.config.batch_size);

        // Create task context with minimal configuration
        let task_ctx = Arc::new(TaskContext::new(
            Some("query_execution".to_owned()),
            "task_1".to_owned(),
            session_config,
            HashMap::new(), // scalar functions
            HashMap::new(), // aggregate functions
            HashMap::new(), // window functions
            runtime,
        ));
        debug!("Created task context");

        // Execute the plan
        let stream = plan.execute(0, task_ctx)?;
        info!("Query execution started successfully");

        // Wrap stream with logging
        let logged_stream = Box::pin(stream.inspect(|result| match result {
            Ok(batch) => debug!(
                rows = batch.num_rows(),
                columns = batch.num_columns(),
                "Processed record batch"
            ),
            Err(e) => warn!(error = ?e, "Error processing record batch"),
        }));

        Ok(logged_stream)
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
