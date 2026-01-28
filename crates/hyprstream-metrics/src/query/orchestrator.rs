//! Query orchestrator that coordinates DataFusion planning and execution.
//!
//! The orchestrator provides a unified interface for:
//! - Preparing SQL queries into physical plans
//! - Caching prepared statements for reuse
//! - Executing queries and streaming results

use crate::query::{DataFusionExecutor, DataFusionPlanner, ExecutorConfig, Query, QueryExecutor, QueryPlanner};
use crate::storage::StorageBackend;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::Result;
use datafusion::physical_plan::ExecutionPlan;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

/// A cached prepared statement with its physical plan
#[derive(Clone)]
pub struct CachedStatement {
    /// Unique handle for this statement
    pub handle: u64,
    /// Original SQL query
    pub sql: String,
    /// Compiled physical execution plan
    pub physical_plan: Arc<dyn ExecutionPlan>,
    /// Schema of the result set
    pub schema: SchemaRef,
}

impl std::fmt::Debug for CachedStatement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedStatement")
            .field("handle", &self.handle)
            .field("sql", &self.sql)
            .field("schema", &self.schema)
            .finish()
    }
}

/// Cache for prepared statements
struct StatementCache {
    /// Map from handle to cached statement
    by_handle: HashMap<u64, CachedStatement>,
    /// Map from SQL to handle for deduplication
    by_sql: HashMap<String, u64>,
    /// Maximum number of cached statements
    max_size: usize,
}

impl StatementCache {
    fn new(max_size: usize) -> Self {
        Self {
            by_handle: HashMap::new(),
            by_sql: HashMap::new(),
            max_size,
        }
    }

    fn get_by_handle(&self, handle: u64) -> Option<&CachedStatement> {
        self.by_handle.get(&handle)
    }

    fn get_by_sql(&self, sql: &str) -> Option<&CachedStatement> {
        self.by_sql.get(sql).and_then(|h| self.by_handle.get(h))
    }

    fn insert(&mut self, stmt: CachedStatement) {
        // Evict oldest entries if at capacity
        if self.by_handle.len() >= self.max_size {
            // Simple eviction: remove first entry (could be improved with LRU)
            if let Some((&handle, _)) = self.by_handle.iter().next() {
                if let Some(stmt) = self.by_handle.remove(&handle) {
                    self.by_sql.remove(&stmt.sql);
                }
            }
        }

        let handle = stmt.handle;
        let sql = stmt.sql.clone();
        self.by_handle.insert(handle, stmt);
        self.by_sql.insert(sql, handle);
    }

    fn remove(&mut self, handle: u64) -> Option<CachedStatement> {
        if let Some(stmt) = self.by_handle.remove(&handle) {
            self.by_sql.remove(&stmt.sql);
            Some(stmt)
        } else {
            None
        }
    }
}

/// Query orchestrator that coordinates planning and execution
pub struct QueryOrchestrator {
    /// DataFusion query planner
    planner: Arc<RwLock<DataFusionPlanner>>,
    /// Query executor
    executor: Arc<DataFusionExecutor>,
    /// Statement cache
    cache: RwLock<StatementCache>,
    /// Handle counter
    next_handle: AtomicU64,
    /// Storage backend reference
    storage: Arc<dyn StorageBackend>,
}

impl QueryOrchestrator {
    /// Create a new query orchestrator
    #[instrument(skip(storage))]
    pub async fn new(storage: Arc<dyn StorageBackend>) -> Result<Self> {
        info!("Creating QueryOrchestrator");

        // Create planner with storage backend
        let planner = DataFusionPlanner::new(Arc::clone(&storage)).await?;
        let executor = DataFusionExecutor::new(ExecutorConfig::default());

        debug!("QueryOrchestrator initialized with default executor config");

        Ok(Self {
            planner: Arc::new(RwLock::new(planner)),
            executor: Arc::new(executor),
            cache: RwLock::new(StatementCache::new(1000)),
            next_handle: AtomicU64::new(1),
            storage,
        })
    }

    /// Create a new query orchestrator with custom executor config
    pub async fn with_executor_config(
        storage: Arc<dyn StorageBackend>,
        config: ExecutorConfig,
    ) -> Result<Self> {
        let planner = DataFusionPlanner::new(Arc::clone(&storage)).await?;
        let executor = DataFusionExecutor::new(config);

        Ok(Self {
            planner: Arc::new(RwLock::new(planner)),
            executor: Arc::new(executor),
            cache: RwLock::new(StatementCache::new(1000)),
            next_handle: AtomicU64::new(1),
            storage,
        })
    }

    /// Prepare a SQL query and return a cached statement
    #[instrument(skip(self), fields(sql_len = sql.len()))]
    pub async fn prepare(&self, sql: &str) -> Result<CachedStatement> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(stmt) = cache.get_by_sql(sql) {
                debug!(handle = stmt.handle, "Cache hit for prepared statement");
                return Ok(stmt.clone());
            }
        }

        debug!("Cache miss, planning query");

        // Create query and plan it
        let query = Query {
            sql: sql.to_owned(),
            schema_hint: None,
            hints: vec![],
        };

        let planner = self.planner.read().await;
        let physical_plan = planner.plan_query(&query).await?;
        let schema = physical_plan.schema();

        // Generate handle and create cached statement
        let handle = self.next_handle.fetch_add(1, Ordering::SeqCst);
        let stmt = CachedStatement {
            handle,
            sql: sql.to_owned(),
            physical_plan,
            schema,
        };

        // Insert into cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(stmt.clone());
        }

        info!(handle = handle, "Prepared and cached new statement");
        Ok(stmt)
    }

    /// Get a prepared statement by handle
    pub async fn get_statement(&self, handle: u64) -> Option<CachedStatement> {
        let cache = self.cache.read().await;
        cache.get_by_handle(handle).cloned()
    }

    /// Execute a prepared statement and return results as a stream
    #[instrument(skip(self, stmt), fields(handle = stmt.handle))]
    pub async fn execute(
        &self,
        stmt: &CachedStatement,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
        debug!("Executing prepared statement");
        self.executor.execute_stream(Arc::clone(&stmt.physical_plan)).await
    }

    /// Execute a prepared statement and collect all results
    #[instrument(skip(self, stmt), fields(handle = stmt.handle))]
    pub async fn execute_collect(&self, stmt: &CachedStatement) -> Result<Vec<RecordBatch>> {
        debug!("Executing and collecting prepared statement results");
        self.executor
            .execute_collect(Arc::clone(&stmt.physical_plan))
            .await
    }

    /// Convenience method: prepare and execute in one call
    #[instrument(skip(self), fields(sql_len = sql.len()))]
    pub async fn query(
        &self,
        sql: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>> {
        let stmt = self.prepare(sql).await?;
        self.execute(&stmt).await
    }

    /// Convenience method: prepare, execute, and collect in one call
    #[instrument(skip(self), fields(sql_len = sql.len()))]
    pub async fn query_collect(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        let stmt = self.prepare(sql).await?;
        self.execute_collect(&stmt).await
    }

    /// Get the schema for a SQL query without executing it
    #[instrument(skip(self), fields(sql_len = sql.len()))]
    pub async fn get_schema(&self, sql: &str) -> Result<SchemaRef> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(stmt) = cache.get_by_sql(sql) {
                debug!(handle = stmt.handle, "Schema from cached statement");
                return Ok(Arc::clone(&stmt.schema));
            }
        }

        // Prepare to get schema
        let stmt = self.prepare(sql).await?;
        Ok(stmt.schema)
    }

    /// Close a prepared statement and remove from cache
    pub async fn close_statement(&self, handle: u64) -> bool {
        let mut cache = self.cache.write().await;
        cache.remove(handle).is_some()
    }

    /// Get the underlying storage backend
    pub fn storage(&self) -> &Arc<dyn StorageBackend> {
        &self.storage
    }

    /// Refresh table registrations (call after schema changes)
    #[instrument(skip(self))]
    pub async fn refresh_tables(&self) -> Result<()> {
        info!("Refreshing table registrations");

        // Re-create planner with updated storage
        let new_planner = DataFusionPlanner::new(Arc::clone(&self.storage)).await?;

        // Replace planner
        let mut planner = self.planner.write().await;
        *planner = new_planner;

        // Clear statement cache since plans may be invalid
        let mut cache = self.cache.write().await;
        cache.by_handle.clear();
        cache.by_sql.clear();

        info!("Table registrations refreshed, statement cache cleared");
        Ok(())
    }
}

impl std::fmt::Debug for QueryOrchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryOrchestrator")
            .field("next_handle", &self.next_handle)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::duckdb::DuckDbBackend;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    #[tokio::test]
    async fn test_orchestrator_prepare_and_execute() -> Result<()> {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.unwrap();

        // Create test table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        backend.create_table("test_table", &schema).await.unwrap();

        // Create orchestrator
        let orchestrator = QueryOrchestrator::new(backend).await?;

        // Prepare query
        let stmt = orchestrator.prepare("SELECT * FROM test_table").await?;
        assert!(stmt.handle > 0);
        assert_eq!(stmt.schema.fields().len(), 2);

        // Execute (should return empty since no data)
        let results = orchestrator.execute_collect(&stmt).await?;
        assert!(results.is_empty() || results[0].num_rows() == 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_orchestrator_statement_caching() -> Result<()> {
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));
        backend.create_table("cache_test", &schema).await.unwrap();

        let orchestrator = QueryOrchestrator::new(backend).await?;

        // First prepare
        let stmt1 = orchestrator.prepare("SELECT * FROM cache_test").await?;
        let handle1 = stmt1.handle;

        // Second prepare should return same handle (cache hit)
        let stmt2 = orchestrator.prepare("SELECT * FROM cache_test").await?;
        assert_eq!(stmt2.handle, handle1);

        // Different query should get different handle
        let stmt3 = orchestrator.prepare("SELECT id FROM cache_test").await?;
        assert_ne!(stmt3.handle, handle1);

        Ok(())
    }

    #[tokio::test]
    async fn test_orchestrator_close_statement() -> Result<()> {
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
        ]));
        backend.create_table("close_test", &schema).await.unwrap();

        let orchestrator = QueryOrchestrator::new(backend).await?;

        // Prepare
        let stmt = orchestrator.prepare("SELECT * FROM close_test").await?;
        let handle = stmt.handle;

        // Statement should be in cache
        assert!(orchestrator.get_statement(handle).await.is_some());

        // Close
        assert!(orchestrator.close_statement(handle).await);

        // Statement should be gone
        assert!(orchestrator.get_statement(handle).await.is_none());

        // Closing again should return false
        assert!(!orchestrator.close_statement(handle).await);

        Ok(())
    }
}
