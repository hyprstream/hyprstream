use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, StringArray, RecordBatch,
};
use arrow_schema::{Schema, Field, DataType};
use std::sync::Arc;
use tonic::Status;
use arrow_convert::{ArrowField, ArrowSerialize};
use arrow_convert::serialize::TryIntoArrow;

/// Helper functions for working with RecordBatches
pub trait RecordBatchExt {
    /// Get a column by name with proper error handling
    fn get_column(&self, name: &str) -> Result<&ArrayRef, Status>;
}

impl RecordBatchExt for RecordBatch {
    fn get_column(&self, name: &str) -> Result<&ArrayRef, Status> {
        self.column_by_name(name)
            .ok_or_else(|| Status::internal(format!("Column {} not found", name)))
    }
}

/// Create a schema builder with common field types
pub struct SchemaBuilder {
    fields: Vec<Field>,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn add_string(&mut self, name: &str, nullable: bool) -> &mut Self {
        self.fields.push(Field::new(name, DataType::Utf8, nullable));
        self
    }

    pub fn add_i64(&mut self, name: &str, nullable: bool) -> &mut Self {
        self.fields.push(Field::new(name, DataType::Int64, nullable));
        self
    }

    pub fn add_f64(&mut self, name: &str, nullable: bool) -> &mut Self {
        self.fields.push(Field::new(name, DataType::Float64, nullable));
        self
    }

    pub fn build(self) -> Schema {
        Schema::new(self.fields)
    }
} 