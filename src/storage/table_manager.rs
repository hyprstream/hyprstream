use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use arrow_schema::Schema;
use tonic::Status;
use serde::{Serialize, Deserialize};
use crate::aggregation::{TimeWindow, AggregateFunction, GroupBy};
use crate::storage::{
    HyprAggregateFunction, HyprGroupBy, HyprTimeWindow
};

/// Configuration for an aggregation view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyprAggregationView {
    pub source_table: String,
    pub function: HyprAggregateFunction,
    pub group_by: HyprGroupBy,
    pub window: HyprTimeWindow,
    pub aggregate_columns: Vec<String>,
}

#[derive(Debug)]
pub struct HyprTableManager {
    tables: Arc<RwLock<HashMap<String, Schema>>>,
    views: Arc<RwLock<HashMap<String, HyprAggregationView>>>,
}

impl Clone for HyprTableManager {
    fn clone(&self) -> Self {
        Self {
            tables: self.tables.clone(),
            views: self.views.clone(),
        }
    }
}

impl Default for HyprTableManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HyprTableManager {
    pub fn new() -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            views: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_table(&self, name: String, schema: Schema) -> Result<(), Status> {
        let mut tables = self.tables.write().await;
        if tables.contains_key(&name) {
            return Err(Status::already_exists(format!("Table {} already exists", name)));
        }
        tables.insert(name, schema);
        Ok(())
    }

    pub async fn get_table_schema(&self, name: &str) -> Result<Schema, Status> {
        let tables = self.tables.read().await;
        tables.get(name)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("Table {} not found", name)))
    }

    pub async fn create_aggregation_view(
        &self,
        name: String,
        source_table: String,
        function: AggregateFunction,
        group_by: GroupBy,
        window: TimeWindow,
        aggregate_columns: Vec<String>,
    ) -> Result<(), Status> {
        // Verify source table exists
        {
            let tables = self.tables.read().await;
            if !tables.contains_key(&source_table) {
                return Err(Status::not_found(format!("Source table {} not found", source_table)));
            }

            // Verify aggregate columns exist in source table
            let schema = tables.get(&source_table).unwrap();
            for col in &aggregate_columns {
                if !schema.fields().iter().any(|f| f.name() == col) {
                    return Err(Status::invalid_argument(format!(
                        "Column {} not found in source table {}",
                        col, source_table
                    )));
                }
            }
        }

        let view = HyprAggregationView {
            source_table,
            function,
            group_by,
            window,
            aggregate_columns,
        };

        let mut views = self.views.write().await;
        if views.contains_key(&name) {
            return Err(Status::already_exists(format!("View {} already exists", name)));
        }
        views.insert(name, view);
        Ok(())
    }

    pub async fn get_aggregation_view(&self, name: &str) -> Result<HyprAggregationView, Status> {
        let views = self.views.read().await;
        views.get(name)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("View {} not found", name)))
    }

    pub async fn list_tables(&self) -> Vec<String> {
        let tables = self.tables.read().await;
        tables.keys().cloned().collect()
    }

    pub async fn list_aggregation_views(&self) -> Vec<String> {
        let views = self.views.read().await;
        views.keys().cloned().collect()
    }

    pub async fn drop_table(&self, name: &str) -> Result<(), Status> {
        let mut tables = self.tables.write().await;
        if tables.remove(name).is_none() {
            return Err(Status::not_found(format!("Table {} not found", name)));
        }
        Ok(())
    }

    pub async fn drop_aggregation_view(&self, name: &str) -> Result<(), Status> {
        let mut views = self.views.write().await;
        if views.remove(name).is_none() {
            return Err(Status::not_found(format!("View {} not found", name)));
        }
        Ok(())
    }
} 