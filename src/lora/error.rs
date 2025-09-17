//! Unified error types for LoRA module

use thiserror::Error;

/// Unified error type for all LoRA operations
#[derive(Error, Debug)]
pub enum LoRAError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    #[error("Invalid rank {rank}: must be between 1 and {max}")]
    InvalidRank { rank: usize, max: usize },
    
    #[error("Tensor operation failed: {0}")]
    TensorOp(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Sparse conversion error: sparsity ratio {0} out of range")]
    SparsityRange(f32),
    
    #[error("Backend not supported for this operation: {0}")]
    UnsupportedBackend(String),
    
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    
    #[error("Training not supported for backend: {0}")]
    TrainingNotSupported(String),
    
    #[error("Module not found: {0}")]
    ModuleNotFound(String),
    
    #[error("Weight format error: {0}")]
    WeightFormat(String),
    
    #[error("OpenVDB operation failed: {0}")]
    OpenVDB(String),
    
    #[error("Generic error: {0}")]
    Other(#[from] anyhow::Error),
}

impl LoRAError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        LoRAError::Config(msg.into())
    }
    
    /// Create a dimension mismatch error
    pub fn dimension_mismatch<S: Into<String>>(expected: S, actual: S) -> Self {
        LoRAError::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
    
    /// Create a tensor operation error
    pub fn tensor_op<S: Into<String>>(msg: S) -> Self {
        LoRAError::TensorOp(msg.into())
    }
}

/// Result type alias for LoRA operations
pub type LoRAResult<T> = Result<T, LoRAError>;

/// Convert from PyTorch tensor errors
impl From<tch::TchError> for LoRAError {
    fn from(err: tch::TchError) -> Self {
        LoRAError::TensorOp(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = LoRAError::config("Invalid configuration");
        assert!(matches!(err, LoRAError::Config(_)));
        
        let err = LoRAError::dimension_mismatch("[768, 768]", "[512, 768]");
        assert!(matches!(err, LoRAError::DimensionMismatch { .. }));
        
        let err = LoRAError::InvalidRank { rank: 1000, max: 768 };
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("768"));
    }
}