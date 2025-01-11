use arrow_schema::{Schema as ArrowSchema, Field as ArrowSchemaField, DataType as ArrowSchemaDataType};
use tonic::Status;
use arrow_convert::{ArrowField, ArrowSerialize};
use arrow_convert::serialize::TryIntoArrow;
use tracing;
use std::sync::Arc;
use arrow::array::{
    Array, ArrayBuilder, Int64Builder, Float64Builder, StringBuilder, ArrayRef, Int64Array, Float64Array, StringArray
};
use arrow::datatypes::{Schema, DataType, Field};
use arrow::record_batch::RecordBatch;
//use super::{HyprQueryResult, HyprRow};
use crate::storage::zerocopy::ZeroCopySource;

pub struct BatchConverter;

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

impl BatchConverter {
    pub fn convert_to_record_batch(
        source: Box<dyn ZeroCopySource>,
    ) -> Result<RecordBatch, Status> {
        // 1. Attempt zero-copy if the underlying type supports it
        match source.try_zero_copy() {
            Ok(rb) => Ok(rb),
            Err(e) => {
                tracing::debug!("Zero-copy attempt failed: {}, falling back to copy", e);
                
                // 2. Fall back to row-by-row building using Arrow arrays directly
                let schema = source.schema();
                let mut builders: Vec<Box<dyn ArrayBuilder>> = schema
                    .fields()
                    .iter()
                    .map(|field| Self::create_array_builder(field))
                    .collect();

                // Convert the data using Arrow arrays
                for row_idx in 0..source.num_rows() {
                    for (col_idx, builder) in builders.iter_mut().enumerate() {
                        let col = source.column(col_idx);
                        match col.data_type() {
                            DataType::Int64 => {
                                let array = col.as_any().downcast_ref::<Int64Array>()
                                    .ok_or_else(|| Status::internal("Invalid Int64 column"))?;
                                builder.as_any_mut().downcast_mut::<Int64Builder>()
                                    .ok_or_else(|| Status::internal("Invalid Int64 builder"))?
                                    .append_value(array.value(row_idx));
                            }
                            DataType::Float64 => {
                                let array = col.as_any().downcast_ref::<Float64Array>()
                                    .ok_or_else(|| Status::internal("Invalid Float64 column"))?;
                                builder.as_any_mut().downcast_mut::<Float64Builder>()
                                    .ok_or_else(|| Status::internal("Invalid Float64 builder"))?
                                    .append_value(array.value(row_idx));
                            }
                            DataType::Utf8 => {
                                let array = col.as_any().downcast_ref::<StringArray>()
                                    .ok_or_else(|| Status::internal("Invalid String column"))?;
                                builder.as_any_mut().downcast_mut::<StringBuilder>()
                                    .ok_or_else(|| Status::internal("Invalid String builder"))?
                                    .append_value(array.value(row_idx));
                            }
                            _ => return Err(Status::internal(format!("Unsupported column type: {:?}", col.data_type()))),
                        }
                    }
                }

                // Finish building arrays
                let arrays = builders
                    .into_iter()
                    .map(|mut builder| Arc::new(builder.finish()) as ArrayRef)
                    .collect();

                RecordBatch::try_new(Arc::new(schema), arrays)
                    .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))
            }
        }
    }

    fn create_array_builder(field: &Field) -> Box<dyn ArrayBuilder> {
        match field.data_type() {
            DataType::Int64 => Box::new(Int64Builder::new()),
            DataType::Float64 => Box::new(Float64Builder::new()),
            DataType::Utf8 => Box::new(StringBuilder::new()),
            _ => panic!("Unsupported column type"),
        }
    }
}