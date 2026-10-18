//! Native port of Cloudflare Clef's joint-schema inference contract.
//!
//! Source: `Cloudflare/clef-flash@8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`,
//! `joint_schema_model.py`, Apache-2.0. This is deliberately separate from the
//! legacy anchor scorer and its versioned serialization contract.
//!
//! [`ClefEncoder`] preserves upstream per-segment tokenization, question order,
//! IDs inside the prompt, sorted choice IDs, and the model's `[true,false]`
//! noul order. Call [`clef_label_permutation`] to return distributions in the
//! caller's canonical label order; do not reorder the input spans themselves.
//!
//! [`JointSchemaHead`] implements eval-mode inference over one unpadded record.
//! This equals the upstream batched head's per-record slice selected by the
//! attention-mask sum. Its input is the final normalized *backbone* token
//! tensor; the head then applies its separate learned hidden LayerNorm. The
//! gather callback provides original output-embedding rows for lexical spans,
//! avoiding a full vocabulary projection. Head attention is deliberately
//! noncausal: fields and options are jointly conditioned on the complete state.
//!
//! Tests using tiny mathematical fixtures establish local operation semantics.
//! Optional pinned-tokenizer and real-head smoke tests require external cached
//! artifacts; neither establishes upstream full-model numerical parity or
//! product quality. Keep those acceptance claims separate.
mod encoding;
mod head;
pub use encoding::*;
pub use head::*;

#[derive(Debug, thiserror::Error)]
pub enum ClefError {
    #[error("invalid Clef input: {0}")]
    Input(String),
    #[error("invalid Clef head: {0}")]
    Head(String),
    #[error("Clef tokenizer: {0}")]
    Tokenizer(String),
    #[error(transparent)]
    Torch(#[from] tch::TchError),
}
