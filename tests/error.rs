use std::io;
use tokio::task;
use config;
use hyprstream::error::{Error, Result};

#[test]
fn test_error_creation() {
    let config_err = Error::ConfigError("invalid config".to_string());
    let runtime_err = Error::RuntimeError("runtime failure".to_string());
    let storage_err = Error::StorageError("storage error".to_string());
    let model_err = Error::ModelError("model error".to_string());
    let validation_err = Error::ValidationError("validation failed".to_string());

    assert!(matches!(config_err, Error::ConfigError(_)));
    assert!(matches!(runtime_err, Error::RuntimeError(_)));
    assert!(matches!(storage_err, Error::StorageError(_)));
    assert!(matches!(model_err, Error::ModelError(_)));
    assert!(matches!(validation_err, Error::ValidationError(_)));
}

#[test]
fn test_error_conversion() {
    // Test IO error conversion
    let io_err = io::Error::new(io::ErrorKind::Other, "io error");
    let converted: Error = io_err.into();
    assert!(matches!(converted, Error::RuntimeError(_)));

    // Test config error conversion
    let config_err = config::ConfigError::NotFound("key".to_string());
    let converted: Error = config_err.into();
    assert!(matches!(converted, Error::ConfigError(_)));
}

#[test]
fn test_error_messages() {
    let err = Error::ConfigError("test config error".to_string());
    assert_eq!(err.to_string(), "test config error");

    let err = Error::RuntimeError("test runtime error".to_string());
    assert_eq!(err.to_string(), "test runtime error");

    let err = Error::StorageError("test storage error".to_string());
    assert_eq!(err.to_string(), "test storage error");
}

#[test]
fn test_result_type() {
    fn returns_ok() -> Result<()> {
        Ok(())
    }

    fn returns_err() -> Result<()> {
        Err(Error::ValidationError("test error".to_string()))
    }

    assert!(returns_ok().is_ok());
    assert!(returns_err().is_err());
}
