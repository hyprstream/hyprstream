//! DataFusion TableProvider implementation for DuckDB storage.
//!
//! This module provides a bridge between DuckDB tables and DataFusion's
//! query planning system, allowing DataFusion to query DuckDB tables directly.

use crate::storage::duckdb::DuckDbBackend;
use crate::storage::StorageBackend;
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::datasource::TableType;
use datafusion::error::Result;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use std::any::Any;
use std::sync::Arc;
use tracing::{debug, instrument};

/// A DataFusion TableProvider that reads from a DuckDB table.
///
/// This provider enables DataFusion to plan and execute queries against
/// DuckDB tables using its standard query planning infrastructure.
pub struct DuckDBTableProvider {
    /// Reference to the DuckDB backend
    backend: Arc<DuckDbBackend>,
    /// Name of the table in DuckDB
    table_name: String,
    /// Arrow schema for the table
    schema: SchemaRef,
}

impl DuckDBTableProvider {
    /// Create a new DuckDBTableProvider for the specified table.
    #[instrument(skip(backend), fields(table = %table_name))]
    pub async fn new(
        backend: Arc<DuckDbBackend>,
        table_name: String,
    ) -> std::result::Result<Self, tonic::Status> {
        debug!("Creating DuckDBTableProvider");

        // Get schema from DuckDB
        let schema = backend.get_table_schema(&table_name).await?;

        debug!(
            fields = schema.fields().len(),
            "Retrieved schema for table"
        );

        Ok(Self {
            backend,
            table_name,
            schema,
        })
    }

    /// Create a provider with a known schema (avoids async schema lookup).
    pub fn with_schema(
        backend: Arc<DuckDbBackend>,
        table_name: String,
        schema: SchemaRef,
    ) -> Self {
        Self {
            backend,
            table_name,
            schema,
        }
    }
}

impl std::fmt::Debug for DuckDBTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDBTableProvider")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for DuckDBTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    #[instrument(skip(self, _state, projection, filters, limit), fields(table = %self.table_name))]
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        debug!(
            projection = ?projection,
            filters_count = filters.len(),
            limit = ?limit,
            "Creating scan plan for DuckDB table"
        );

        // Create a DuckDBExec plan that will execute the query
        let plan = DuckDBExec::new(
            self.backend.clone(),
            self.table_name.clone(),
            self.schema.clone(),
            projection.cloned(),
            filters.to_vec(),
            limit,
        );

        Ok(Arc::new(plan))
    }
}

/// Physical execution plan for scanning a DuckDB table.
pub struct DuckDBExec {
    backend: Arc<DuckDbBackend>,
    table_name: String,
    schema: SchemaRef,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    limit: Option<usize>,
}

impl std::fmt::Debug for DuckDBExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckDBExec")
            .field("table_name", &self.table_name)
            .field("schema", &self.schema)
            .field("projected_schema", &self.projected_schema)
            .field("projection", &self.projection)
            .field("filters", &self.filters)
            .field("limit", &self.limit)
            .finish()
    }
}

impl DuckDBExec {
    fn new(
        backend: Arc<DuckDbBackend>,
        table_name: String,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        filters: Vec<Expr>,
        limit: Option<usize>,
    ) -> Self {
        // Compute projected schema
        let projected_schema = if let Some(ref proj) = projection {
            let fields: Vec<_> = proj
                .iter()
                .filter_map(|i| schema.fields().get(*i).cloned())
                .collect();
            Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
        } else {
            schema.clone()
        };

        Self {
            backend,
            table_name,
            schema,
            projected_schema,
            projection,
            filters,
            limit,
        }
    }

    /// Build the SQL query for this scan
    fn build_sql(&self) -> String {
        // Build column list
        let columns = if let Some(ref proj) = self.projection {
            proj.iter()
                .filter_map(|i| self.schema.fields().get(*i))
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            "*".to_string()
        };

        let mut sql = format!("SELECT {} FROM {}", columns, self.table_name);

        // Add filters (simplified - full implementation would convert Expr to SQL)
        // For now, we'll push down simple filters
        if !self.filters.is_empty() {
            // TODO: Implement proper filter pushdown
            // This would require converting DataFusion Expr to SQL WHERE clauses
        }

        // Add limit
        if let Some(limit) = self.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        sql
    }
}

impl datafusion::physical_plan::DisplayAs for DuckDBExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose
            | datafusion::physical_plan::DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DuckDBExec: table={}, projection={:?}, limit={:?}",
                    self.table_name, self.projection, self.limit
                )
            }
        }
    }
}

impl datafusion::physical_plan::ExecutionPlan for DuckDBExec {
    fn name(&self) -> &str {
        "DuckDBExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        // Use a static reference for properties
        // This is a simplified implementation
        static PROPERTIES: std::sync::OnceLock<datafusion::physical_plan::PlanProperties> =
            std::sync::OnceLock::new();
        PROPERTIES.get_or_init(|| {
            use datafusion::physical_plan::Partitioning;
            use datafusion::physical_expr::EquivalenceProperties;
            use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};

            datafusion::physical_plan::PlanProperties::new(
                EquivalenceProperties::new(Arc::new(datafusion::arrow::datatypes::Schema::empty())),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            )
        })
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;
        use futures::StreamExt;

        let sql = self.build_sql();
        let backend = self.backend.clone();
        let schema = self.projected_schema.clone();

        // Create a stream that executes the query
        let stream = stream::once(async move {
            // Prepare and execute the query
            let handle = backend
                .prepare_sql(&sql)
                .await
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

            let batch = backend
                .query_sql(&handle)
                .await
                .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;

            Ok(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;

    #[tokio::test]
    async fn test_duckdb_table_provider() -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        backend.init().await?;

        // Create test table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        backend.create_table("test_provider", &schema).await?;

        // Create provider
        let provider = DuckDBTableProvider::new(backend, "test_provider".to_string()).await?;

        // Verify schema
        assert_eq!(provider.schema().fields().len(), 2);
        assert_eq!(provider.table_type(), TableType::Base);

        Ok(())
    }

    #[tokio::test]
    async fn test_duckdb_provider_with_datafusion() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory()?);
        backend.init().await?;

        // Create test table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, true),
        ]));
        backend.create_table("metrics", &schema).await?;

        // Create provider and register with DataFusion
        let provider = DuckDBTableProvider::new(backend, "metrics".to_string()).await?;
        let ctx = SessionContext::new();
        ctx.register_table("metrics", Arc::new(provider))?;

        // Query through DataFusion (should work even with empty table)
        let df = ctx.sql("SELECT * FROM metrics").await?;
        let results = df.collect().await?;

        // Empty table returns 0-1 batches
        assert!(results.is_empty() || results[0].num_rows() == 0);

        Ok(())
    }
}
