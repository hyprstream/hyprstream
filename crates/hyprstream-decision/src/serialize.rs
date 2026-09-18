//! The ONE canonical serialization of a decision request (pinned per S6b1).
//!
//! This module is a **frozen contract**: meaning-preserving format changes swing model
//! accuracy by up to 76 pp (Sclar et al., ICLR 2024), so the byte-level layout below does
//! not change without a new schema version. The question encoder (P1.1) trains on exactly
//! this form and the serving path emits exactly this form.
//!
//! ## Layout
//!
//! ```text
//! <|state|>\n{state}\n
//! <|question|>\n{instructions}\n            (instructions line omitted when absent)
//!     noul:   <|rubric|>\ntrue\n{true criteria}\n     (each rubric block only if present)
//!             <|rubric|>\nfalse\n{false criteria}\n
//!             <|anchor|>\n
//!     choice: <|option|>\n{name}\n{rubric}\n<|anchor|>\n   (rubric line omitted when null)
//!             ... one block per option, in canonical (authoring) order
//!     score:  <|level|>\n{rubric}\n<|anchor|>\n            (one per level; the level's
//!             ...                                                    NUMBER is never emitted)
//! ```
//!
//! - The **anchor placeholder comes immediately AFTER each option's rubric text**
//!   (`<|anchor|>` stands in for the learned special token whose id the tokenizer assigns
//!   in P1.1/P1.2). Never a trailing anchor section: the shared scorer reads the hidden
//!   state at each anchor position, and option-adjacent placement keeps every option's
//!   local readout structurally identical through the backbone's causal linear-attention
//!   layers (S6b1).
//! - Score levels carry no index and no neighbour context — each level is judged
//!   independently against the state (upstream behavioral contract, S6a §2.2).
//! - Question ids are caller-facing only and never appear in the serialization.
//! - Non-string entries (objects/arrays) serialize as compact canonical JSON
//!   ([`crate::entry::Entry::canonical_text`]).
//!
//! ## Token budget (acceptance criterion)
//!
//! A serialized request must fit [`MAX_SERIALIZED_SPEC_TOKENS`] at the 2k-token serving
//! context. When a choice question cannot fit its options within the budget — the
//! canonical case being cardinality above the 255-option profile limit — the two-stage
//! recipe (score candidates in parallel, then choose among the top scorers) is
//! **mandatory**, not optional (S6a §2.4: both stages are client-side patterns, not wire
//! features). Token counting is tokenizer-bound and lands with the P1.1 encoder; this
//! module exposes [`serialized_bytes`] as the pre-tokenizer proxy.

use crate::spec::{QuestionBody, QuestionSet, QuestionSpec};

/// Placeholder for the learned special anchor token. The concrete token (and its id) is
/// assigned when the tokenizer is extended in P1.1/P1.2; the serialization contract fixes
/// only the *position*: immediately after each option's rubric text.
pub const ANCHOR_PLACEHOLDER: &str = "<|anchor|>";

/// The serialized-spec token budget at the 2k-token serving context. Exceeding it makes
/// the two-stage >255 recipe mandatory. Enforced with the real tokenizer in P1.1; this
/// constant is the contract.
pub const MAX_SERIALIZED_SPEC_TOKENS: usize = 2048;

/// Serialize one question to its canonical block (without the state header).
pub fn serialize_question(question: &QuestionSpec) -> String {
    let mut out = String::new();
    out.push_str("<|question|>\n");
    if let Some(instructions) = &question.instructions {
        out.push_str(&instructions.canonical_text());
        out.push('\n');
    }
    match &question.body {
        QuestionBody::Noul { criteria } => {
            if let Some(criteria) = criteria {
                if let Some(on_true) = &criteria.on_true {
                    out.push_str("<|rubric|>\ntrue\n");
                    out.push_str(&on_true.canonical_text());
                    out.push('\n');
                }
                if let Some(on_false) = &criteria.on_false {
                    out.push_str("<|rubric|>\nfalse\n");
                    out.push_str(&on_false.canonical_text());
                    out.push('\n');
                }
            }
            out.push_str(ANCHOR_PLACEHOLDER);
            out.push('\n');
        }
        QuestionBody::Choice { options } => {
            for option in options {
                out.push_str("<|option|>\n");
                out.push_str(&option.name);
                out.push('\n');
                if let Some(rubric) = &option.rubric {
                    out.push_str(&rubric.canonical_text());
                    out.push('\n');
                }
                out.push_str(ANCHOR_PLACEHOLDER);
                out.push('\n');
            }
        }
        QuestionBody::Score { levels } => {
            for level in levels {
                out.push_str("<|level|>\n");
                if let Some(rubric) = level {
                    out.push_str(&rubric.canonical_text());
                    out.push('\n');
                }
                out.push_str(ANCHOR_PLACEHOLDER);
                out.push('\n');
            }
        }
    }
    out
}

/// Serialize a full request: the state header followed by every question block in
/// canonical (authoring) order.
pub fn serialize_request(set: &QuestionSet) -> String {
    let mut out = String::new();
    out.push_str("<|state|>\n");
    if let Some(state) = &set.state {
        out.push_str(&state.canonical_text());
        out.push('\n');
    }
    for question in &set.questions {
        out.push_str(&serialize_question(question));
    }
    out
}

/// Byte length of the canonical serialization — the pre-tokenizer proxy for the
/// [`MAX_SERIALIZED_SPEC_TOKENS`] budget (token counting is tokenizer-bound, P1.1).
pub fn serialized_bytes(set: &QuestionSet) -> usize {
    serialize_request(set).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::author::parse_yaml;

    #[test]
    fn anchor_follows_each_option_rubric_and_ids_never_appear() {
        let set = parse_yaml(
            r#"
state: "The box was crushed."
questions:
  tone:
    type: choice
    instructions: "Classify the review."
    criteria:
      positive: "The review expresses approval."
      negative: ~
"#,
        )
        .unwrap_or_else(|error| panic!("fixture parses: {error}"));
        let text = serialize_request(&set);
        assert_eq!(
            text,
            "<|state|>\nThe box was crushed.\n\
             <|question|>\nClassify the review.\n\
             <|option|>\npositive\nThe review expresses approval.\n<|anchor|>\n\
             <|option|>\nnegative\n<|anchor|>\n"
        );
        assert!(
            !text.contains("tone"),
            "question ids never reach the model: {text}"
        );
    }

    #[test]
    fn score_levels_carry_no_index() {
        let set = parse_yaml(
            r#"
questions:
  severity:
    type: score
    criteria: ["trivial", "critical"]
"#,
        )
        .unwrap_or_else(|error| panic!("fixture parses: {error}"));
        let text = serialize_request(&set);
        assert_eq!(
            text,
            "<|state|>\n<|question|>\n\
             <|level|>\ntrivial\n<|anchor|>\n\
             <|level|>\ncritical\n<|anchor|>\n"
        );
    }

    #[test]
    fn noul_emits_one_anchor_after_criteria() {
        let set = parse_yaml(
            r#"
questions:
  is_refund:
    type: noul
    instructions: "The customer wants money back."
    criteria:
      "true": "Explicit refund request"
"#,
        )
        .unwrap_or_else(|error| panic!("fixture parses: {error}"));
        let text = serialize_request(&set);
        assert_eq!(
            text,
            "<|state|>\n<|question|>\nThe customer wants money back.\n\
             <|rubric|>\ntrue\nExplicit refund request\n<|anchor|>\n"
        );
    }
}
