//! Error type for the backbone conversion crate.

/// Errors from backbone construction, forward passes, and model integration.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A configuration value is inconsistent (dims not divisible, unknown layer
    /// type, mask policy applied to a layer that has none).
    #[error("invalid backbone config: {0}")]
    Config(String),

    /// A tensor shape did not match the contract (e.g. token ids not `[batch, seq]`).
    #[error("tensor shape mismatch: {0}")]
    Shape(String),

    /// A libtorch call failed.
    #[error(transparent)]
    Torch(#[from] tch::TchError),

    /// The shared scorer rejected the forward pass (batch/anchor-map mismatch).
    #[error(transparent)]
    Scorer(#[from] hyprstream_decision_head::Error),
}
