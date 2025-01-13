use tonic::Status;
use tracing;
use std::sync::Arc;
use arrow::array::{
    Array, ArrayBuilder, Int64Builder, Float64Builder, StringBuilder, ArrayRef, Int64Array, Float64Array, StringArray,
};
use arrow::buffer::Buffer as ArrowBuffer;
use arrow::datatypes::{Schema, DataType, Field};
use arrow::record_batch::RecordBatch;
use zerocopy::{FromZeroes, FromBytes};
use crate::storage::zerocopy::ZeroCopySource;

/// Safe memory layout for Arrow array data
#[repr(C, align(8))]
#[derive(FromZeroes, FromBytes)]
pub struct ArrowBufferHeader {
    len: u64,
    null_count: u64,
    offset: u64,
}

/// Zero-copy wrapper for Arrow array buffers
pub struct SafeArrayBuffer {
    header_bytes: Vec<u8>,
    data: Arc<ArrowBuffer>,
}

/// Helper for creating Arrow arrays from raw data
impl SafeArrayBuffer {
    pub fn new<T: arrow::datatypes::ArrowNativeType>(values: &[T], null_count: u64, offset: u64) -> Self {
        let header = ArrowBufferHeader {
            len: values.len() as u64,
            null_count,
            offset,
        };
        
        // Store header bytes
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const ArrowBufferHeader as *const u8,
                std::mem::size_of::<ArrowBufferHeader>(),
            ).to_vec()
        };
        
        // Create Arrow buffer from values
        let byte_len = values.len() * std::mem::size_of::<T>();
        let mut bytes = Vec::with_capacity(byte_len);
        unsafe {
            bytes.extend_from_slice(std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                byte_len,
            ));
        }
        let data = Arc::new(ArrowBuffer::from(bytes));
        
        Self {
            header_bytes,
            data,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn validate_layout(&self) -> Result<(), Status> {
        let header = unsafe {
            let ptr = self.header_bytes.as_ptr() as *const ArrowBufferHeader;
            if self.header_bytes.len() != std::mem::size_of::<ArrowBufferHeader>() {
                return Err(Status::internal("Invalid header layout"));
            }
            &*ptr
        };
            
        if header.len as usize != self.data.len() {
            return Err(Status::internal("Data length mismatch"));
        }
        
        Ok(())
    }
}

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
    fn append_to_builder<B>(
        builder: &mut B,
        array_len: usize,
        get_value: impl Fn(usize) -> f64, // example type
    ) -> Result<(), Status>
    where
        B: ArrayBuilder,
    {
        for i in 0..array_len {
            let value = get_value(i);
            if let Some(typed_builder) = builder.as_any_mut().downcast_mut::<Int64Builder>() {
                typed_builder.append_value(value as i64);
            } else {
                return Err(Status::internal("Unsupported builder type"));
            }
        }
        Ok(())
    }

    pub fn convert_to_record_batch(
        source: Box<dyn ZeroCopySource>,
    ) -> Result<RecordBatch, Status> {
        // 1. Attempt zero-copy with compile-time layout verification
        match source.try_zero_copy(4) {
            Ok(rb) => {
                // Create safe buffer wrappers for each column
                let safe_arrays: Result<Vec<ArrayRef>, Status> = rb.columns()
                    .iter()
                    .map(|col| Self::create_safe_array(col))
                    .collect();

                let arrays = safe_arrays?;
                Ok(RecordBatch::try_new(rb.schema(), arrays)
                    .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?)
            },
            Err(e) => {
                tracing::debug!("Zero-copy attempt failed: {}, falling back to copy", e);
                Self::fallback_copy(source)
            }
        }
    }

    fn create_safe_array(col: &ArrayRef) -> Result<ArrayRef, Status> {
        // Validate input array
        if !col.is_valid(0) {
            return Err(Status::internal("Invalid array data"));
        }

        match col.data_type() {
            DataType::Int64 => {
                let array = col.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| Status::internal("Invalid Int64 column"))?;
                
                let mut builder = Int64Builder::with_capacity(array.len());
                for i in 0..array.len() {
                    builder.append_value(array.value(i));
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
            DataType::Float64 => {
                let array = col.as_any().downcast_ref::<Float64Array>()
                    .ok_or_else(|| Status::internal("Invalid Float64 column"))?;
                
                let mut builder = Float64Builder::with_capacity(array.len());
                Self::append_to_builder(&mut builder, array.len(), |i| array.value(i))?;
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
            DataType::Utf8 => {
                let array = col.as_any().downcast_ref::<StringArray>()
                    .ok_or_else(|| Status::internal("Invalid String column"))?;
                
                let mut builder = StringBuilder::new();
                for i in 0..array.len() {
                    builder.append_value(array.value(i));
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
            _ => Err(Status::internal(format!("Unsupported column type: {:?}", col.data_type()))),
        }
    }

    fn fallback_copy(source: Box<dyn ZeroCopySource>) -> Result<RecordBatch, Status> {
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

    fn create_array_builder(field: &Field) -> Box<dyn ArrayBuilder> {
        match field.data_type() {
            DataType::Int64 => Box::new(Int64Builder::new()),
            DataType::Float64 => Box::new(Float64Builder::new()),
            DataType::Utf8 => Box::new(StringBuilder::new()),
            _ => panic!("Unsupported column type"),
        }
    }
}