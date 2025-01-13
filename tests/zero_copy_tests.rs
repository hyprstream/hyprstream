use arrow::array::{Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use std::sync::Arc;
use zerocopy::{AsBytes, FromBytes};
use crate::storage::arrow_utils::{SafeArrayBuffer, ArrowBufferHeader};
use crate::storage::zerocopy::{WeightHeader, ModelWeightArray};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_array_buffer() {
        let data = vec![1u8, 2, 3, 4, 5];
        let buffer = SafeArrayBuffer::new(data.clone(), 0, 0);
        
        // Verify data access through safe interface
        assert_eq!(buffer.as_bytes(), &[1, 2, 3, 4, 5]);
        
        // Verify header is properly initialized
        let header = buffer.header();
        assert_eq!(header.len, 5);
        assert_eq!(header.null_count, 0);
        assert_eq!(header.offset, 0);
        
        // Verify FromBytes trait implementation
        let header_bytes = header.as_bytes();
        let parsed_header = ArrowBufferHeader::read_from(header_bytes)
            .expect("Failed to parse header");
        assert_eq!(parsed_header.len, header.len);
    }

    #[test]
    fn test_arrow_batch_sharing() {
        // Create sample arrays with safe buffers
        let values = vec![1.0, 2.0, 3.0];
        let timestamps = vec![1000, 2000, 3000];
        
        let value_buffer = SafeArrayBuffer::new(
            values.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect(),
            0,
            0,
        );
        let ts_buffer = SafeArrayBuffer::new(
            timestamps.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect(),
            0,
            0,
        );
        
        // Create arrays using safe buffers
        let value_array = Float64Array::from(value_buffer.as_bytes());
        let ts_array = Int64Array::from(ts_buffer.as_bytes());
        
        // Create schema and record batch
        let schema = Schema::new(vec![
            Field::new("value", DataType::Float64, false),
            Field::new("timestamp", DataType::Int64, false),
        ]);
        
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(value_array), Arc::new(ts_array)]
        ).unwrap();
        
        // Verify data through safe interfaces
        let value_col = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let ts_col = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        
        assert_eq!(value_col.value(0), 1.0);
        assert_eq!(ts_col.value(0), 1000);
    }

    #[test]
    fn test_model_weight_array() {
        let data = vec![1.0f32, 2.0, 3.0];
        let weight_array = ModelWeightArray::new(
            data.iter().flat_map(|v| v.to_le_bytes().to_vec()).collect(),
            1, // f32 dtype
            0, // CPU device
        );
        
        // Verify header
        let header = weight_array.header();
        assert_eq!(header.len, 12); // 3 * 4 bytes
        assert_eq!(header.dtype, 1);
        assert_eq!(header.device, 0);
        
        // Verify data access
        let bytes = weight_array.as_bytes();
        let floats: Vec<f32> = bytes.chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compile_time_layout() {
        // Verify WeightHeader layout at compile time
        assert_eq!(std::mem::size_of::<WeightHeader>(), 16);
        assert_eq!(std::mem::align_of::<WeightHeader>(), 8);
        
        // Verify ArrowBufferHeader layout at compile time
        assert_eq!(std::mem::size_of::<ArrowBufferHeader>(), 24);
        assert_eq!(std::mem::align_of::<ArrowBufferHeader>(), 8);
    }
}
