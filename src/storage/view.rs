use crate::aggregation::{AggregateFunction, GroupBy, TimeWindow};
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::SystemTime;

/// Serializable schema representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableSchema {
    fields: Vec<SerializableField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableField {
    name: String,
    data_type: String,
    nullable: bool,
}

impl From<&Schema> for SerializableSchema {
    fn from(schema: &Schema) -> Self {
        SerializableSchema {
            fields: schema
                .fields()
                .iter()
                .map(|f| SerializableField {
                    name: f.name().clone(),
                    data_type: format!("{:?}", f.data_type()),
                    nullable: f.is_nullable(),
                })
                .collect(),
        }
    }
}

impl From<SerializableSchema> for Schema {
    fn from(schema: SerializableSchema) -> Self {
        let fields: Fields = schema
            .fields
            .into_iter()
            .map(|f| {
                Field::new(
                    &f.name,
                    // Parse data type from string representation
                    match f.data_type.as_str() {
                        "Int64" => DataType::Int64,
                        "Float64" => DataType::Float64,
                        "Utf8" => DataType::Utf8,
                        "Timestamp(Nanosecond, None)" => DataType::Timestamp(TimeUnit::Nanosecond, None),
                        _ => DataType::Utf8, // Default to string for unknown types
                    },
                    f.nullable,
                )
            })
            .collect::<Vec<Field>>()
            .into();

        Schema::new(fields)
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
    #[serde(serialize_with = "serialize_schema", deserialize_with = "deserialize_schema")]
    pub schema: Arc<Schema>,
}

fn serialize_schema<S>(schema: &Arc<Schema>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    SerializableSchema::from(schema.as_ref()).serialize(serializer)
}

fn deserialize_schema<'de, D>(deserializer: D) -> Result<Arc<Schema>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let serializable = SerializableSchema::deserialize(deserializer)?;
    Ok(Arc::new(Schema::from(serializable)))
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
        schema: Arc<Schema>,
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