//! Error types for the RPC system.

use thiserror::Error;

/// Errors that can occur during envelope operations.
#[derive(Debug, Error)]
pub enum EnvelopeError {
    /// Signature verification failed.
    #[error("signature verification failed: {0}")]
    InvalidSignature(#[from] ed25519_dalek::SignatureError),

    /// Signer public key doesn't match expected key.
    #[error("signer pubkey mismatch: expected {expected}, got {actual}")]
    SignerMismatch { expected: String, actual: String },

    /// Invalid public key format.
    #[error("invalid public key: expected {expected} bytes, got {actual}")]
    InvalidPublicKey { expected: usize, actual: usize },

    /// Cap'n Proto serialization/deserialization failed.
    #[error("envelope serialization failed: {0}")]
    Serialization(#[from] capnp::Error),

    /// Cap'n Proto schema mismatch.
    #[error("envelope schema error: {0}")]
    Schema(#[from] capnp::NotInSchema),

    /// HMAC verification failed.
    #[error("HMAC verification failed")]
    InvalidHmac,

    /// Key exchange failed.
    #[error("key exchange failed: {0}")]
    KeyExchange(String),

    /// Missing required field.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// Replay attack detected (duplicate nonce or old timestamp).
    #[error("replay attack detected: {0}")]
    ReplayAttack(String),

    /// Invalid topic format (not valid hex).
    #[error("invalid topic format: expected valid hex string")]
    InvalidTopicFormat,
}

/// Result type alias for envelope operations.
pub type EnvelopeResult<T> = std::result::Result<T, EnvelopeError>;

/// Errors that can occur during RPC operations.
#[derive(Debug, Error)]
pub enum RpcError {
    /// Envelope error.
    #[error("envelope error: {0}")]
    Envelope(#[from] EnvelopeError),

    /// Process spawn failed.
    #[error("spawn failed: {0}")]
    SpawnFailed(String),

    /// Process stop failed.
    #[error("stop failed: {0}")]
    StopFailed(String),

    /// Invalid operation for this backend/mode.
    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    /// ZMQ error.
    #[error("zmq error: {0}")]
    Zmq(#[from] zmq::Error),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type alias for RPC operations.
pub type Result<T> = std::result::Result<T, RpcError>;
