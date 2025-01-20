use crate::storage::view::ViewMetadata;
use crate::storage::StorageBackend;
use datafusion::error::Result;
use datafusion::logical_expr::{LogicalPlan, TableScan};
use datafusion::optimizer::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::sql::TableReference;
use std::sync::Arc;

/// Optimization rule that rewrites queries to use views when beneficial
pub struct ViewOptimizationRule {
    storage: Arc<dyn StorageBackend>,
}

impl ViewOptimizationRule {
    pub fn new(storage: Arc<dyn StorageBackend>) -> Self {
        Self { storage }
    }

    /// Check if a view can be used for this query
    async fn find_matching_view(&self, plan: &LogicalPlan) -> Result<Option<ViewMetadata>> {
        // Get list of available views
        let views = self.storage.list_views().await.map_err(|e| {
            datafusion::error::DataFusionError::Internal(format!("Failed to list views: {}", e))
        })?;

        // For each view, check if it can be used for this query
        for view_name in views {
            let view = self.storage.get_view(&view_name).await.map_err(|e| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Failed to get view metadata: {}",
                    e
                ))
            })?;

            if self.can_use_view(plan, &view)? {
                return Ok(Some(view));
            }
        }

        Ok(None)
    }

    /// Check if a view can be used to answer this query
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

    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        // Use tokio runtime to run async code
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        // Check if we can use a view for this query
        if let Some(view) = rt.block_on(self.find_matching_view(plan))? {
            // Rewrite the plan to use the view
            Ok(Some(self.rewrite_with_view(plan, &view)?))
        } else {
            // No matching view found
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::{AggregateFunction, GroupBy};
    use crate::storage::view::{AggregationSpec, ViewDefinition};
    use crate::storage::DuckDbBackend;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::empty::EmptyTable;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_view_optimization() -> Result<()> {
        // Create test backend
        let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
        backend.init().await.unwrap();

        // Create source table
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
            Field::new("timestamp", DataType::Int64, false),
        ]));
        backend
            .create_table("test_source", &source_schema)
            .await
            .unwrap();

        // Create view
        let view_def = ViewDefinition::new(
            "test_source".to_string(),
            vec!["metric".to_string()],
            vec![AggregationSpec {
                column: "value".to_string(),
                function: AggregateFunction::Avg,
            }],
            Some(GroupBy {
                columns: vec!["metric".to_string()],
                time_column: None,
            }),
            None,
            Arc::new(Schema::new(vec![
                Field::new("metric", DataType::Utf8, false),
                Field::new("avg_value", DataType::Float64, false),
            ])),
        );
        backend
            .create_view("test_view", view_def)
            .await
            .unwrap();

        // Create optimizer rule
        let rule = ViewOptimizationRule::new(backend);

        // Create test plan
        let df_schema = datafusion::arrow::datatypes::Schema::try_from(source_schema.as_ref().clone())
            .unwrap();
        let plan = LogicalPlan::TableScan(TableScan {
            table_name: "test_source".into(),
            source: Arc::new(EmptyTable::new(Arc::new(df_schema))),
            projection: None,
            projected_schema: Arc::new(df_schema),
            filters: vec![],
            fetch: None,
        });

        // Apply optimization
        let optimized = rule.try_optimize(&plan, &datafusion::optimizer::OptimizerContext::new())?;

        // Verify plan was rewritten to use view
        match optimized {
            Some(LogicalPlan::TableScan(scan)) => {
                assert_eq!(scan.table_name.to_string(), "test_view");
            }
            _ => panic!("Expected TableScan"),
        }

        Ok(())
    }
}