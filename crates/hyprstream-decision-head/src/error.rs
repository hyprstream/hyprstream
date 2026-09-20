//! Error types for the encoder and scorer.

use thiserror::Error;

/// Errors from encoding a [`hyprstream_decision::QuestionSet`] or validating an
/// encoded request against its question specs.
#[derive(Debug, Error)]
pub enum Error {
    /// The tokenizer failed to encode the canonical serialization.
    #[error("tokenizer encode failed: {0}")]
    Tokenizer(String),

    /// The learned anchor token could not be assigned an id.
    #[error("anchor token {token:?} was added but has no id in the tokenizer")]
    AnchorTokenMissing {
        /// The anchor token string that failed to resolve.
        token: &'static str,
    },

    /// The serialized request exceeds the tokenizer-bound token budget
    /// ([`hyprstream_decision::serialize::MAX_SERIALIZED_SPEC_TOKENS`] by default).
    /// Above this budget the two-stage score-then-choice recipe is mandatory (P0.1a).
    #[error(
        "serialized request is {tokens} tokens, exceeding the budget of {budget}; \
         split the request or use the two-stage score-then-choice recipe"
    )]
    TokenBudgetExceeded {
        /// Encoded token count of the full request.
        tokens: usize,
        /// The enforced budget.
        budget: usize,
    },

    /// The number of anchor tokens in the encoding does not match the question arities.
    /// This means the tokenizer and the serialization contract disagree (e.g. the anchor
    /// token id belongs to a different bundle than the tokenizer in use) — fail loudly
    /// rather than misalign scorer readout positions.
    #[error(
        "anchor count mismatch: serialization produced {expected} anchor placeholders \
         but the encoding contains {found} anchor tokens"
    )]
    AnchorCountMismatch {
        /// Anchors implied by the question specs (1 per noul, 1 per choice option,
        /// 1 per score level).
        expected: usize,
        /// Anchor-token occurrences found in the token ids.
        found: usize,
    },

    /// A question's recorded anchor count does not match its spec arity when scoring.
    #[error(
        "question {question_id:?} ({kind}) expects {expected} anchor(s), got {found}"
    )]
    QuestionArityMismatch {
        /// The mismatched question id.
        question_id: String,
        /// The question kind.
        kind: hyprstream_decision::QuestionKind,
        /// Arity implied by the spec.
        expected: usize,
        /// Anchors actually recorded.
        found: usize,
    },

    /// The anchor-position batch passed to the scorer does not match the hidden-state
    /// batch dimension.
    #[error("hidden-state batch is {batch} sequences but {maps} anchor maps were given")]
    BatchMismatch {
        /// Batch size of the hidden-state tensor.
        batch: usize,
        /// Number of per-sequence anchor maps supplied.
        maps: usize,
    },
}
