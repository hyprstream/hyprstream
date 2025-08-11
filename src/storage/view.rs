use crate::aggregation::{AggregateFunction, GroupBy, TimeWindow};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::SystemTime;

/// Simplified schema representation for ML inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Int64,
    Float64,
    String,
    Timestamp,
}

impl Field {
    pub fn new(name: &str, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            nullable,
        }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
    
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }
    
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}

/// Specification for a view aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSpec {
    pub column: String,
    pub function: AggregateFunction,
}

/// Complete definition of a view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewDefinition {
    pub source_table: String,
    pub columns: Vec<String>,
    pub aggregations: Vec<AggregationSpec>,
    pub group_by: Option<GroupBy>,
    pub window: Option<TimeWindow>,
    pub dependencies: HashSet<String>,
    pub schema: Schema,
}

/// Metadata for a view stored in the backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewMetadata {
    pub name: String,
    pub definition: ViewDefinition,
    pub created_at: SystemTime,
}

impl ViewDefinition {
    /// Create a new view definition
    pub fn new(
        source_table: String,
        columns: Vec<String>,
        aggregations: Vec<AggregationSpec>,
        group_by: Option<GroupBy>,
        window: Option<TimeWindow>,
        schema: Schema,
    ) -> Self {
        // Calculate dependencies from source table and any referenced columns
        let mut dependencies = HashSet::new();
        dependencies.insert(source_table.clone());

        Self {
            source_table,
            columns,
            aggregations,
            group_by,
            window,
            dependencies,
            schema,
        }
    }

    /// Build the SQL definition for this view
    pub fn to_sql(&self) -> String {
        let mut sql = String::from("SELECT ");
        
        // Add columns
        let mut first = true;
        for col in &self.columns {
            if !first {
                sql.push_str(", ");
            }
            sql.push_str(col);
            first = false;
        }

        // Add aggregations
        for agg in &self.aggregations {
            if !first {
                sql.push_str(", ");
            }
            sql.push_str(&format!("{}({}) as {}_{}",
                agg.function,
                agg.column,
                agg.function.to_string().to_lowercase(),
                agg.column
            ));
            first = false;
        }

        // Add FROM clause
        sql.push_str(&format!(" FROM {}", self.source_table));

        // Add GROUP BY if present
        if let Some(group_by) = &self.group_by {
            if !group_by.columns.is_empty() {
                sql.push_str(" GROUP BY ");
                sql.push_str(&group_by.columns.join(", "));
            }
        }

        sql
    }
}

impl ViewMetadata {
    /// Create new view metadata
    pub fn new(name: String, definition: ViewDefinition) -> Self {
        Self {
            name,
            definition,
            created_at: SystemTime::now(),
        }
    }
}