//! Tokenizer-bound linearization: the canonical jev-1 serialization (P0.1a) turned
//! into token ids plus a per-question anchor map, with the 2048-token spec budget
//! enforced against the real tokenizer and the learned anchor token id assigned.

use hyprstream_decision::serialize;
use hyprstream_decision::{QuestionBody, QuestionKind, QuestionSet};
use tokenizers::{AddedToken, Tokenizer};

use crate::Error;

/// The learned special anchor token string. Byte-identical to
/// [`serialize::ANCHOR_PLACEHOLDER`]: the placeholder in the frozen serialization *is*
/// the special token's text, so the tokenizer's added-token matcher maps every
/// placeholder occurrence to the anchor id.
pub const ANCHOR_TOKEN: &str = serialize::ANCHOR_PLACEHOLDER;

/// Assign the learned anchor token id in `tokenizer`, adding it as a special token when
/// absent. Idempotent: an existing entry (e.g. loaded from a converted bundle's
/// tokenizer) is reused, so the id is stable across encode sites that share the bundle.
///
/// The assigned id is part of the model bundle recorded at conversion time (P1.2); the
/// freshly extended backbone embedding row for this id is what "learned anchor" refers
/// to — the token starts untrained and is learned during distillation (P1.4).
pub fn assign_anchor_token(tokenizer: &mut Tokenizer) -> Result<u32, Error> {
    if let Some(id) = tokenizer.token_to_id(ANCHOR_TOKEN) {
        return Ok(id);
    }
    tokenizer.add_special_tokens(&[AddedToken::from(ANCHOR_TOKEN, true)]);
    tokenizer
        .token_to_id(ANCHOR_TOKEN)
        .ok_or(Error::AnchorTokenMissing {
            token: ANCHOR_TOKEN,
        })
}

/// Per-question anchor readout positions in one encoded request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionAnchors {
    /// Caller-facing question id (never serialized to the model; used to align the
    /// scorer's outputs with the request).
    pub question_id: String,
    /// The primitive type — selects the readout (sigmoid / softmax / binned softmax).
    pub kind: QuestionKind,
    /// Token positions of this question's anchor tokens, in canonical option/level
    /// order. Exactly 1 for noul; `options.len()` for choice; `levels.len()` for score.
    pub anchor_positions: Vec<u32>,
}

/// One tokenized decision request: the ids fed to the backbone and the anchor map the
/// shared scorer reads at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRequest {
    /// Token ids of the canonical serialization (state header + question blocks).
    pub input_ids: Vec<u32>,
    /// Per-question anchor maps, in canonical (authoring) question order.
    pub questions: Vec<QuestionAnchors>,
}

impl EncodedRequest {
    /// The encoded length in tokens — the quantity the spec budget constrains.
    pub fn token_count(&self) -> usize {
        self.input_ids.len()
    }
}

/// The question/criteria encoder. Wraps the bundle tokenizer with the assigned anchor
/// id and enforces the tokenizer-bound spec budget.
pub struct DecisionEncoder {
    tokenizer: Tokenizer,
    anchor_token_id: u32,
    budget: usize,
}

impl DecisionEncoder {
    /// Build an encoder over `tokenizer`, assigning the anchor token id if the
    /// tokenizer does not already carry it. The default budget is the contract
    /// constant [`serialize::MAX_SERIALIZED_SPEC_TOKENS`] (2048 at the 2k serving
    /// context).
    pub fn new(mut tokenizer: Tokenizer) -> Result<Self, Error> {
        let anchor_token_id = assign_anchor_token(&mut tokenizer)?;
        Ok(Self {
            tokenizer,
            anchor_token_id,
            budget: serialize::MAX_SERIALIZED_SPEC_TOKENS,
        })
    }

    /// Override the token budget (tests and future context sizes). The production
    /// contract is the default.
    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// The assigned learned anchor token id.
    pub fn anchor_token_id(&self) -> u32 {
        self.anchor_token_id
    }

    /// The wrapped tokenizer (bundle tokenizer + anchor token).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode a question set: canonical serialization → token ids → anchor map.
    ///
    /// Fails with [`Error::TokenBudgetExceeded`] over budget (the two-stage
    /// score-then-choice recipe becomes mandatory there), and with
    /// [`Error::AnchorCountMismatch`] if the encoding's anchor-token count disagrees
    /// with the question arities — tokenizer/bundle skew must fail loudly.
    pub fn encode(&self, set: &QuestionSet) -> Result<EncodedRequest, Error> {
        let text = serialize::serialize_request(set);
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        let ids = encoding.get_ids();
        if ids.len() > self.budget {
            return Err(Error::TokenBudgetExceeded {
                tokens: ids.len(),
                budget: self.budget,
            });
        }
        let anchor_positions: Vec<u32> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == self.anchor_token_id)
            .map(|(position, _)| position as u32)
            .collect();
        let expected: usize = set.questions.iter().map(anchor_arity).sum();
        if anchor_positions.len() != expected {
            return Err(Error::AnchorCountMismatch {
                expected,
                found: anchor_positions.len(),
            });
        }
        let mut cursor = 0;
        let questions = set
            .questions
            .iter()
            .map(|question| {
                let arity = anchor_arity(question);
                let positions = anchor_positions[cursor..cursor + arity].to_vec();
                cursor += arity;
                QuestionAnchors {
                    question_id: question.id.clone(),
                    kind: question.kind,
                    anchor_positions: positions,
                }
            })
            .collect();
        Ok(EncodedRequest {
            input_ids: ids.to_vec(),
            questions,
        })
    }
}

/// Anchors implied by one question's spec: 1 for noul (the single readout position
/// after the optional criteria blocks), 1 per option for choice, 1 per level for score.
fn anchor_arity(question: &hyprstream_decision::QuestionSpec) -> usize {
    match &question.body {
        QuestionBody::Noul { .. } => 1,
        QuestionBody::Choice { options } => options.len(),
        QuestionBody::Score { levels } => levels.len(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_decision::parse_yaml;

    /// Minimal in-memory WordLevel tokenizer; unknown words map to `[UNK]`. The anchor
    /// placeholder is not in the vocab — it arrives via `assign_anchor_token`.
    fn fixture_tokenizer() -> Tokenizer {
        const TOK_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(TOK_JSON.as_bytes()).expect("fixture tokenizer parses")
    }

    fn fixture_set() -> QuestionSet {
        parse_yaml(
            r#"
state: "The box was crushed."
questions:
  tone:
    type: choice
    criteria:
      positive: "The review expresses approval."
      negative: "The review expresses disapproval."
  severe:
    type: score
    criteria: ["trivial", "moderate", "critical"]
  is_refund:
    type: noul
"#,
        )
        .expect("fixture spec parses")
    }

    #[test]
    fn anchor_id_assignment_is_idempotent() {
        let mut tokenizer = fixture_tokenizer();
        let first = assign_anchor_token(&mut tokenizer).unwrap();
        let second = assign_anchor_token(&mut tokenizer).unwrap();
        assert_eq!(first, second, "re-assigning reuses the id");
        assert_eq!(tokenizer.token_to_id(ANCHOR_TOKEN), Some(first));
    }

    #[test]
    fn encode_maps_anchors_per_question_in_arity_order() {
        let encoder = DecisionEncoder::new(fixture_tokenizer()).unwrap();
        let encoded = encoder.encode(&fixture_set()).unwrap();
        assert_eq!(encoded.questions.len(), 3);

        let tone = &encoded.questions[0];
        assert_eq!(tone.question_id, "tone");
        assert_eq!(tone.kind, QuestionKind::Choice);
        assert_eq!(tone.anchor_positions.len(), 2);

        let severe = &encoded.questions[1];
        assert_eq!(severe.kind, QuestionKind::Score);
        assert_eq!(severe.anchor_positions.len(), 3);

        let is_refund = &encoded.questions[2];
        assert_eq!(is_refund.kind, QuestionKind::Noul);
        assert_eq!(is_refund.anchor_positions.len(), 1, "noul has exactly one anchor");

        let mut all: Vec<u32> = encoded
            .questions
            .iter()
            .flat_map(|question| question.anchor_positions.iter().copied())
            .collect();
        let total = all.len();
        all.dedup();
        assert_eq!(all.len(), total, "anchor positions are distinct");
        for window in all.windows(2) {
            assert!(window[0] < window[1], "anchors appear in serialization order");
        }
        for position in &all {
            assert_eq!(
                encoded.input_ids[*position as usize],
                encoder.anchor_token_id(),
                "every mapped position holds the anchor token"
            );
        }
    }

    #[test]
    fn budget_is_enforced_against_token_count() {
        let encoder = DecisionEncoder::new(fixture_tokenizer()).unwrap().with_budget(4);
        let error = encoder.encode(&fixture_set()).unwrap_err();
        match error {
            Error::TokenBudgetExceeded { tokens, budget } => {
                assert_eq!(budget, 4);
                assert!(tokens > 4);
            }
            other => panic!("expected TokenBudgetExceeded, got {other}"),
        }
    }

    #[test]
    fn anchor_mismatch_fails_loudly() {
        // A tokenizer whose anchor id belongs to a *different* token than the
        // serialization placeholder simulates bundle/tokenizer skew: the placeholders
        // tokenize as ordinary text and no anchor tokens are found.
        let encoder = DecisionEncoder::new(fixture_tokenizer()).unwrap();
        let skewed = DecisionEncoder {
            tokenizer: fixture_tokenizer(),
            anchor_token_id: encoder.anchor_token_id(),
            budget: serialize::MAX_SERIALIZED_SPEC_TOKENS,
        };
        let error = skewed.encode(&fixture_set()).unwrap_err();
        assert!(
            matches!(error, Error::AnchorCountMismatch { .. }),
            "expected AnchorCountMismatch, got {error}"
        );
    }
}
