//! # hyprstream-decision-head — question/criteria encoder + shared option-scorer
//!
//! System One program, DAG node **P1.1** (Wave 1). This crate owns the model-side
//! contract between the pinned jev-1 serialization ([`hyprstream-decision`], P0.1a)
//! and the converted backbone (P1.2): it turns a runtime-declared
//! [`QuestionSet`](hyprstream_decision::QuestionSet) into token ids plus an **anchor
//! map**, and it defines the **shared option-scorer** — the generalist head that reads
//! one hidden state per anchor position and produces a typed, per-question probability
//! distribution in a single forward pass.
//!
//! ## The generalist contract
//!
//! One shared scorer, one forward pass, no cross-question conditioning: every question
//! is scored **only** from its own anchor positions. Question ids never reach the model
//! (P0.1a serialization contract), options carry learned special anchor tokens
//! immediately after their rubric text (S6b1: adjacent-after placement, never a trailing
//! anchor section), and the scorer applies the *same* linear projection to every anchor
//! state. Position/option-ID priors are eliminated by construction (learned anchors, not
//! vocabulary letter tokens — Zheng et al.'s selection-bias mechanism); residual
//! order-correlation risk is owned data-side by P1.3's mandatory permutation
//! augmentation, and measured by P0.6's cyclic-permutation stratum.
//!
//! ## Readouts per primitive (S6b2-pinned)
//!
//! - **noul** — exactly one anchor (P0.1a contract); `P(true) = σ(logit)`, emitted as the
//!   2-wide `[P(false), P(true)]` distribution matching the Arrow label order.
//! - **choice** — softmax over the option anchor logits, in canonical authoring order.
//! - **score** — binned softmax over the per-level anchor logits, trained against
//!   teacher-average histograms (distribution targets; [`scorer::soft_target_kl`] is the
//!   distillation objective for P1.4), with an **expectation readout**
//!   ([`scorer::QuestionScore::expected_value`]) for the probability-weighted value
//!   `Σᵢ i·pᵢ`. The CORN runner-up and its flip conditions are recorded in the plan
//!   (S6b2); this crate implements the primary parameterization only.
//!
//! ## Tokenizer-bound budget + learned anchor id (deferred here from P0.1a)
//!
//! P0.1a froze the byte-level serialization and the
//! [`MAX_SERIALIZED_SPEC_TOKENS`](hyprstream_decision::serialize::MAX_SERIALIZED_SPEC_TOKENS)
//! budget constant; this node binds the budget to a real tokenizer
//! ([`encoder::DecisionEncoder::encode`] fails with
//! [`Error::TokenBudgetExceeded`](crate::Error::TokenBudgetExceeded)) and assigns the
//! learned special anchor token id ([`encoder::assign_anchor_token`]). The anchor id is
//! recorded per model bundle at conversion time (P1.2); encoding validates that every
//! anchor placeholder tokenized to exactly the assigned id and that the anchor count
//! matches the question arities, so a tokenizer/bundle skew fails loudly at encode time
//! instead of silently misaligning the scorer's readout positions.
//!
//! ## Framework choice
//!
//! tch-rs (the workspace's `hyprstream/tch-rs` fork) in-workspace, matching the
//! hyprstream idiom already used by `hyprstream-inference` and the choice P1.2's
//! conversion consumes: the scorer's `VarStore` checkpoints load directly into the
//! converted backbone, and the same fork provides the torch-native GDN path pinned for
//! ROCm/MI210 training (S6c).

pub mod encoder;
pub mod error;
pub mod scorer;

pub use encoder::{
    assign_anchor_token, DecisionEncoder, EncodedRequest, QuestionAnchors, ANCHOR_TOKEN,
};
pub use error::Error;
pub use scorer::{soft_target_kl, QuestionScore, SharedScorer};
