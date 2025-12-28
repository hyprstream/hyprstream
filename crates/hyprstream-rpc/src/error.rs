//! Error types for the RPC envelope system.

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
}

/// Result type alias for envelope operations.
pub type EnvelopeResult<T> = Result<T, EnvelopeError>;
