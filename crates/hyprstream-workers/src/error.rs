//! Error types for hyprstream-workers

use thiserror::Error;

/// Main error type for worker operations
#[derive(Error, Debug)]
pub enum WorkerError {
    // ─────────────────────────────────────────────────────────────────────
    // Sandbox (Pod) Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Sandbox not found: {0}")]
    SandboxNotFound(String),

    #[error("Sandbox creation failed: {0}")]
    SandboxCreationFailed(String),

    #[error("Sandbox already exists: {0}")]
    SandboxAlreadyExists(String),

    #[error("Sandbox in invalid state for operation: {sandbox_id} is {state}, expected {expected}")]
    SandboxInvalidState {
        sandbox_id: String,
        state: String,
        expected: String,
    },

    #[error("Sandbox pool exhausted: max {max} sandboxes")]
    PoolExhausted { max: usize },

    #[error("Sandbox timeout: {operation} took longer than {timeout_secs}s")]
    SandboxTimeout { operation: String, timeout_secs: u64 },

    // ─────────────────────────────────────────────────────────────────────
    // VM Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("VM start failed: {0}")]
    VmStartFailed(String),

    #[error("VM stop failed: {0}")]
    VmStopFailed(String),

    #[error("Cloud-init generation failed: {0}")]
    CloudInitFailed(String),

    #[error("VirtioFS daemon failed: {0}")]
    VirtiofsFailed(String),

    // ─────────────────────────────────────────────────────────────────────
    // Container Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Container not found: {0}")]
    ContainerNotFound(String),

    #[error("Container creation failed: {0}")]
    ContainerCreationFailed(String),

    #[error("Container already exists: {0}")]
    ContainerAlreadyExists(String),

    #[error("Container in invalid state: {container_id} is {state}, expected {expected}")]
    ContainerInvalidState {
        container_id: String,
        state: String,
        expected: String,
    },

    #[error("Container exec failed: {0}")]
    ExecFailed(String),

    // ─────────────────────────────────────────────────────────────────────
    // Image Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Image not found: {0}")]
    ImageNotFound(String),

    #[error("Failed to pull image {image}: {reason}")]
    ImagePullFailed { image: String, reason: String },

    #[error("Failed to parse image reference {image}: {reason}")]
    ImageParseFailed { image: String, reason: String },

    #[error("Image invalid: {0}")]
    ImageInvalid(String),

    #[error("Registry authentication failed: {0}")]
    RegistryAuthFailed(String),

    #[error("RAFS store error: {0}")]
    RafsError(String),

    // ─────────────────────────────────────────────────────────────────────
    // Workflow Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Workflow not found: {0}")]
    WorkflowNotFound(String),

    #[error("Workflow parse error: {0}")]
    WorkflowParseError(String),

    #[error("Workflow run not found: {0}")]
    RunNotFound(String),

    #[error("Workflow dispatch failed: {0}")]
    DispatchFailed(String),

    #[error("Workflow step failed: job={job}, step={step}, error={error}")]
    StepFailed {
        job: String,
        step: String,
        error: String,
    },

    #[error("Workflow timeout: {workflow_id} exceeded {timeout_secs}s")]
    WorkflowTimeout {
        workflow_id: String,
        timeout_secs: u64,
    },

    // ─────────────────────────────────────────────────────────────────────
    // Event Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Event subscription failed: {0}")]
    SubscriptionFailed(String),

    #[error("Event publish failed: {0}")]
    PublishFailed(String),

    #[error("Invalid event topic: {0}")]
    InvalidTopic(String),

    // ─────────────────────────────────────────────────────────────────────
    // Policy Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("Unauthorized: {subject} cannot {operation} on {resource}")]
    Unauthorized {
        subject: String,
        operation: String,
        resource: String,
    },

    #[error("Policy check failed: {0}")]
    PolicyError(String),

    // ─────────────────────────────────────────────────────────────────────
    // Infrastructure Errors
    // ─────────────────────────────────────────────────────────────────────
    #[error("ZMQ error: {0}")]
    ZmqError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<anyhow::Error> for WorkerError {
    fn from(err: anyhow::Error) -> Self {
        WorkerError::Internal(err.to_string())
    }
}

impl From<serde_json::Error> for WorkerError {
    fn from(err: serde_json::Error) -> Self {
        WorkerError::SerializationError(err.to_string())
    }
}

impl From<serde_yaml::Error> for WorkerError {
    fn from(err: serde_yaml::Error) -> Self {
        WorkerError::SerializationError(err.to_string())
    }
}

impl From<capnp::Error> for WorkerError {
    fn from(err: capnp::Error) -> Self {
        WorkerError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for WorkerError {
    fn from(err: std::io::Error) -> Self {
        WorkerError::IoError(err.to_string())
    }
}

/// Result type alias for worker operations
pub type Result<T> = std::result::Result<T, WorkerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WorkerError::SandboxNotFound("abc123".to_string());
        assert_eq!(err.to_string(), "Sandbox not found: abc123");

        let err = WorkerError::Unauthorized {
            subject: "anonymous".to_string(),
            operation: "execute".to_string(),
            resource: "sandbox:*".to_string(),
        };
        assert!(err.to_string().contains("anonymous"));
        assert!(err.to_string().contains("execute"));
    }
}
