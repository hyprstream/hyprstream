use crate::storage::view::ViewMetadata;
use crate::storage::VDBSparseStorage;
use datafusion::error::Result;
use datafusion::logical_expr::{LogicalPlan, TableScan};
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion::sql::TableReference;
#[cfg(test)]
use {
    datafusion::datasource::{TableProvider, TableType},
    datafusion::logical_expr::TableSource,
    datafusion::physical_plan::{empty::EmptyExec, ExecutionPlan},
    datafusion::catalog::Session,
    async_trait::async_trait,
};

#[cfg(test)]
struct EmptyTableProvider {
    schema: arrow_schema::SchemaRef,
}

#[cfg(test)]
impl TableSource for EmptyTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }
}

#[async_trait]
#[cfg(test)]
impl TableProvider for EmptyTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(EmptyExec::new(TableProvider::schema(self))))
    }
}
use std::sync::Arc;

/// Optimization rule that rewrites queries to use views when beneficial  
/// Adapted for VDB-first architecture (mostly disabled for embeddings-only queries)
pub struct ViewOptimizationRule {
    vdb_storage: Arc<VDBSparseStorage>,
}

impl ViewOptimizationRule {
    pub fn new(vdb_storage: Arc<VDBSparseStorage>) -> Self {
        Self { vdb_storage }
    }

    /// Check if a view can be used for this query (disabled for VDB-first)
    #[allow(dead_code)]
    async fn find_matching_view(&self, _plan: &LogicalPlan) -> Result<Option<ViewMetadata>> {
        // VDB-first architecture focuses on embeddings, not traditional views
        // Return None to disable view optimization
        Ok(None)
    }

    /// Check if a view can be used to answer this query
    #[allow(dead_code)]
    fn can_use_view(&self, plan: &LogicalPlan, view: &ViewMetadata) -> Result<bool> {
        match plan {
            LogicalPlan::TableScan(scan) => {
                // Check if the query is accessing the view's source table
                if scan.table_name.to_string() == view.definition.source_table {
                    // TODO: Add more sophisticated matching logic here
                    // - Check if required columns are available
                    // - Check if aggregations match
                    // - Check if grouping is compatible
                    // - Check if time windows align
                    return Ok(true);
                }
            }
            // TODO: Add support for more complex query patterns
            _ => {}
        }

        Ok(false)
    }

    /// Rewrite the plan to use the view
    #[allow(dead_code)]
    fn rewrite_with_view(&self, plan: &LogicalPlan, view: &ViewMetadata) -> Result<LogicalPlan> {
        match plan {
            LogicalPlan::TableScan(scan) => {
                // Convert Arrow schema to DataFusion schema
                let arrow_schema = view.definition.schema.as_ref();
                let df_schema = datafusion::common::DFSchema::try_from_qualified_schema(
                    &view.name,
                    arrow_schema,
                )
                .map_err(|e| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "Failed to convert schema: {}",
                        e
                    ))
                })?;
                let df_schema = Arc::new(df_schema);

                // Create new table scan using the view
                let new_scan = LogicalPlan::TableScan(TableScan {
                    table_name: TableReference::from(view.name.clone()),
                    source: scan.source.clone(),
                    projection: Some(
                        view.definition
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(i, _)| i)
                            .collect(),
                    ),
                    projected_schema: df_schema,
                    filters: scan.filters.clone(),
                    fetch: scan.fetch,
                });

                Ok(new_scan)
            }
            // TODO: Add support for rewriting more complex plans
            _ => Ok(plan.clone()),
        }
    }
}

impl OptimizerRule for ViewOptimizationRule {
    fn name(&self) -> &str {
        "view_optimization"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(&self, plan: LogicalPlan, _config: &dyn OptimizerConfig) -> Result<Transformed<LogicalPlan>> {
        // VDB-first architecture doesn't use traditional views - it's embeddings-focused
        // Return the plan unchanged for now
        Ok(Transformed::no(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::view::ViewDefinition;
    use crate::storage::duckdb::DuckDbBackend;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_view_optimization() -> Result<()> {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.map_err(|e|
            datafusion::error::DataFusionError::Internal(format!("Failed to init backend: {}", e))
        )?;

        // Create source table
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));
        backend
            .create_table("test_source", &source_schema)
            .await
            .map_err(|e|
                datafusion::error::DataFusionError::Internal(format!("Failed to create table: {}", e))
            )?;

        // Create view
        let view_def = ViewDefinition::new(
            "test_source".to_string(),
            vec!["metric".to_string(), "value".to_string()],
            vec![],
            None,
            None,
            Arc::new(Schema::new(vec![
                Field::new("metric", DataType::Utf8, false),
                Field::new("value", DataType::Float64, false),
            ])),
        );
        backend
            .create_view("test_view", view_def)
            .await
            .map_err(|e|
                datafusion::error::DataFusionError::Internal(format!("Failed to create view: {}", e))
            )?;

        // Create optimizer rule
        let rule = ViewOptimizationRule::new(backend);

        // Create test plan
        let arrow_schema = source_schema.as_ref().clone();
        let df_schema = DFSchema::try_from(arrow_schema.clone())
            .map_err(|e|
                datafusion::error::DataFusionError::Internal(format!("Failed to convert schema: {}", e))
            )?;
        let plan = LogicalPlan::TableScan(TableScan {
            table_name: "test_source".into(),
            source: Arc::new(EmptyTableProvider {
                schema: Arc::new(df_schema.clone().into())
            }),
            projection: None,
            projected_schema: Arc::new(df_schema),
            filters: vec![],
            fetch: None,
        });

        // Apply optimization
        let transformed = rule.rewrite(plan.clone(), &datafusion::optimizer::OptimizerContext::new())?;

        // Verify plan was rewritten to use view
        match transformed.data {
            LogicalPlan::TableScan(scan) => {
                assert_eq!(scan.table_name.to_string(), "test_view");
            }
            _ => panic!("Expected TableScan"),
        }

        Ok(())
    }
}