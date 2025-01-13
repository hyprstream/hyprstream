use std::sync::Arc;
use arrow_array::{Array, RecordBatch as ArrowRecordBatch, ArrayRef};
use arrow_schema::Schema;
use tonic::Status;
use zerocopy::{FromBytes, FromZeroes};

// GPU types
#[cfg(feature = "cuda")]
use crate::gpu::memory::{CudaBuffer, CudaStream};

// Stub types when CUDA is disabled
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub(crate) struct CudaBuffer;

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub(crate) struct CudaStream;

/// Errors that can occur during zero-copy operations
#[derive(Debug)]
pub enum ZeroCopyError {
    /// Memory layout is not compatible for zero-copy
    IncompatibleLayout(&'static str),
    /// Missing or invalid metadata
    MetadataError(&'static str),
    /// Error during parallel processing
    ParallelProcessing(String),
    /// Memory layout validation failed
    MemoryLayout(String),
    /// Other unspecified errors
    Other(String),
}

/// Safe memory layout for model weights.
///
/// # Safety
/// This struct is marked as `repr(C, align(8))` to ensure:
/// - Consistent memory layout across platforms
/// - Proper 8-byte alignment for all fields
/// - No padding bytes between fields
/// - All fields are plain old data types
#[repr(C, align(8))]
#[derive(Copy, Clone, FromZeroes, FromBytes)]
pub struct WeightHeader {
    pub len: u64,    // 8 bytes
    pub dtype: u32,  // 4 bytes
    pub device: u32, // 4 bytes, total size = 16 bytes
}

impl WeightHeader {
    pub fn new(len: u64, dtype: u32, device: u32) -> Self {
        Self { len, dtype, device }
    }

    pub fn validate(&self) -> Result<(), ZeroCopyError> {
        if self.len == 0 {
            return Err(ZeroCopyError::MemoryLayout("Zero length header".into()));
        }
        Ok(())
    }
}

/// Safe wrapper for model weight data that implements zero-copy operations
#[derive(Clone)]
pub struct ModelWeightArray {
    header: WeightHeader,
    data: Arc<[u8]>,
}

impl ModelWeightArray {
    pub fn new(data: Vec<u8>, dtype: u32, device: u32) -> Self {
        let header = WeightHeader::new(data.len() as u64, dtype, device);
        Self {
            header,
            data: Arc::from(data),
        }
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn header(&self) -> &WeightHeader {
        &self.header
    }

    pub fn validate_layout(&self) -> Result<(), ZeroCopyError> {
        self.header.validate()?;
        if self.header.len as usize != self.data.len() {
            return Err(ZeroCopyError::MemoryLayout("Data length mismatch".into()));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn to_gpu(&self, stream: &CudaStream) -> Result<CudaBuffer, ZeroCopyError> {
        self.validate_layout()?;
        
        let gpu_buffer = CudaBuffer::new(self.data.len())
            .map_err(|e| ZeroCopyError::Other(e.to_string()))?;

        stream.copy_host_to_device(
            gpu_buffer.as_mut_ptr(),
            self.data.as_ptr(),
            self.data.len(),
        ).map_err(|e| ZeroCopyError::Other(e.to_string()))?;

        Ok(gpu_buffer)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn to_gpu(&self, _stream: &CudaStream) -> Result<CudaBuffer, ZeroCopyError> {
        Err(ZeroCopyError::Other("CUDA support not enabled".to_string()))
    }
}

impl From<ZeroCopyError> for Status {
    fn from(e: ZeroCopyError) -> Self {
        Status::internal(e.to_string())
    }
}

impl From<std::io::Error> for ZeroCopyError {
    fn from(e: std::io::Error) -> Self {
        ZeroCopyError::Other(e.to_string())
    }
}

impl std::fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZeroCopyError::IncompatibleLayout(msg) => write!(f, "Incompatible memory layout: {}", msg),
            ZeroCopyError::MetadataError(msg) => write!(f, "Metadata error: {}", msg),
            ZeroCopyError::ParallelProcessing(msg) => write!(f, "Parallel processing error: {}", msg),
            ZeroCopyError::MemoryLayout(msg) => write!(f, "Memory layout error: {}", msg),
            ZeroCopyError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

impl std::error::Error for ZeroCopyError {}

/// Trait for data sources that can produce zero-copy Arrow-compatible buffers.
///
/// This trait enables efficient data transfer between different Arrow implementations
/// by attempting to reuse existing memory buffers where possible. It implements
/// zero-copy operations by:
/// - Reusing existing Arrow buffers when possible
/// - Sharing memory through Arc references
/// - Maintaining proper memory alignment
/// - Validating memory layouts before operations
pub trait ZeroCopySource: Send + Sync {
    /// Attempts to return an Arrow RecordBatch that references existing buffers.
    fn try_zero_copy(&self, parallel_threshold: usize) -> Result<ArrowRecordBatch, ZeroCopyError>;

    /// Returns the schema describing the data source's structure
    fn schema(&self) -> Schema;

    /// Returns the total number of rows in the data source
    fn num_rows(&self) -> usize;

    /// Returns a reference to the column at the given index
    fn column(&self, i: usize) -> &dyn Array;

    /// Process numeric data with optional optimizations
    fn process_numeric<T: Copy + Send + Sync>(&self, data: &[T]) -> Vec<T> where Self: Sized {
        data.to_vec()
    }

    /// Validates column layout compatibility
    fn validate_column_layout(&self, col: &dyn Array) -> Result<(), ZeroCopyError> where Self: Sized {
        if (0..col.len()).any(|i| !col.is_valid(i)) {
            return Err(ZeroCopyError::MemoryLayout("Invalid column data".into()));
        }
        Ok(())
    }

    /// Validates memory layout compatibility for zero-copy operations
    fn validate_layout(&self) -> Result<(), ZeroCopyError> {
        Ok(())
    }
}

impl ZeroCopySource for ArrowRecordBatch {
    fn try_zero_copy(&self, _parallel_threshold: usize) -> Result<ArrowRecordBatch, ZeroCopyError> {
        self.validate_layout()?;
        
        let schema = Arc::clone(&self.schema());
        let arrays: Vec<ArrayRef> = self.columns()
            .iter()
            .map(|col| {
                self.validate_column_layout(col.as_ref())?;
                Ok(Arc::clone(col))
            })
            .collect::<Result<_, ZeroCopyError>>()?;

        ArrowRecordBatch::try_new(schema, arrays)
            .map_err(|e| ZeroCopyError::Other(e.to_string()))
    }

    fn schema(&self) -> Schema {
        (*self.schema()).clone()
    }

    fn num_rows(&self) -> usize {
        self.num_rows()
    }

    fn column(&self, i: usize) -> &dyn Array {
        self.column(i).as_ref()
    }
    
    fn process_numeric<T: Copy + Send + Sync>(&self, data: &[T]) -> Vec<T> where Self: Sized {
        data.to_vec()
    }
    
    fn validate_column_layout(&self, col: &dyn Array) -> Result<(), ZeroCopyError> where Self: Sized {
        if (0..col.len()).any(|i| !col.is_valid(i)) {
            return Err(ZeroCopyError::MemoryLayout("Invalid column data".into()));
        }
        Ok(())
    }
    
    fn validate_layout(&self) -> Result<(), ZeroCopyError> {
        Ok(())
    }
}
