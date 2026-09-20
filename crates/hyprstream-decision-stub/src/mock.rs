//! The deterministic mock decision model.
//!
//! Distributions are a pure function of (model id, canonical question serialization,
//! state, row index): blake3-drawn uniforms, normalized, with the largest component
//! corrected so the sum meets the producer tolerance (D5) exactly. Determinism is the
//! point — contract tests, golden vectors, and the wire-overhead floor all pin exact
//! bytes.
//!
//! Test hook: a question whose instructions contain the literal `[abstain]` is answered
//! with an abstention, so null-probability paths are exercisable on demand.

use hyprstream_decision::answer::{AnswerValue, QuestionAnswer};
use hyprstream_decision::arrow::AnswerRow;
use hyprstream_decision::serialize;
use hyprstream_decision::spec::{QuestionBody, QuestionSet};

/// The resolved model id the stub reports on every response. Alias resolution is part of
/// the serving contract: any requested alias resolves to this versioned id.
pub const STUB_MODEL_VERSION: &str = "jev-stub-1.0.0";

/// The alias listed by `GET /v1/models` (and accepted, like every string, on requests).
pub const STUB_MODEL_ALIAS: &str = "jev-stub-latest";

/// The deterministic mock.
#[derive(Debug, Clone, Default)]
pub struct MockDecisionModel;

/// Draw one uniform in (0, 1) from the hash of `key` and `index`.
fn draw(key: &[u8], index: u64) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key);
    hasher.update(&index.to_le_bytes());
    let output = hasher.finalize();
    let bits = u64::from_le_bytes(output.as_bytes()[..8].try_into().unwrap_or_else(|error| panic!("8 bytes: {error}")));
    // Map to (0, 1) exclusive: exact 0/1 components would pin confidence to 1.0 and hide
    // the derived-statistic paths the contract tests exercise.
    (bits as f64 + 0.5) / (u64::MAX as f64 + 1.0)
}

/// A normalized distribution of width `cardinality`, exact to producer tolerance (D5).
fn distribution(key: &[u8], cardinality: usize) -> Vec<f32> {
    debug_assert!(cardinality >= 1);
    let draws: Vec<f64> = (0..cardinality as u64).map(|i| draw(key, i)).collect();
    let total: f64 = draws.iter().sum();
    let mut probabilities: Vec<f32> = draws.iter().map(|d| (d / total) as f32).collect();
    // Correct the largest component so the f32 sum is exactly 1; the correction is far
    // below the 1e-6 producer tolerance.
    let sum: f32 = probabilities.iter().sum();
    if let Some((max_index, _)) = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
    {
        probabilities[max_index] += 1.0 - sum;
    }
    probabilities
}

impl MockDecisionModel {
    /// Answer every question in `set` for one state row (`row` is the batch row index,
    /// 0 for single-request facade calls).
    pub fn answer_row(&self, set: &QuestionSet, state_text: &str, row: usize) -> AnswerRow {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let abstain = question
                .instructions
                .as_ref()
                .is_some_and(|entry| entry.canonical_text().contains("[abstain]"));
            if abstain {
                answers.insert(question.id.clone(), QuestionAnswer::abstained());
                continue;
            }
            let key = format!(
                "{STUB_MODEL_VERSION}\x00{}\x00{state_text}\x00{row}",
                serialize::serialize_question(question)
            );
            let value = match &question.body {
                QuestionBody::Noul { .. } => AnswerValue::Noul {
                    p_true: draw(key.as_bytes(), 0) as f32,
                },
                QuestionBody::Choice { options } => AnswerValue::Choice {
                    probabilities: distribution(key.as_bytes(), options.len()),
                },
                QuestionBody::Score { levels } => AnswerValue::Score {
                    probabilities: distribution(key.as_bytes(), levels.len()),
                },
            };
            answers.insert(question.id.clone(), QuestionAnswer::answered(value));
        }
        AnswerRow { answers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_decision::author;
    use hyprstream_decision::confidence::{self, PRODUCER_SUM_TOLERANCE};

    fn fixture() -> QuestionSet {
        author::parse_yaml(
            r#"
state: "The box was crushed."
questions:
  is_refund:
    type: noul
    instructions: "The customer wants money back."
  tone:
    type: choice
    criteria:
      angry: "Hostile message"
      calm: ~
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
"#,
        )
        .unwrap_or_else(|error| panic!("fixture parses: {error}"))
    }

    #[test]
    fn distributions_are_deterministic_and_normalized() {
        let set = fixture();
        let mock = MockDecisionModel;
        let first = mock.answer_row(&set, "The box was crushed.", 0);
        let second = mock.answer_row(&set, "The box was crushed.", 0);
        assert_eq!(first, second, "the mock is a pure function");
        for (id, answer) in &first.answers {
            let value = answer.value.as_ref().unwrap_or_else(|| panic!("answered"));
            let question = set.question(id).unwrap_or_else(|| panic!("declared"));
            assert_eq!(value.probabilities().len(), question.cardinality());
            confidence::check_distribution(&value.probabilities(), PRODUCER_SUM_TOLERANCE)
                .unwrap_or_else(|error| panic!("producer tolerance: {error}"));
        }
    }

    #[test]
    fn state_and_row_change_the_answer() {
        let set = fixture();
        let mock = MockDecisionModel;
        let a = mock.answer_row(&set, "state A", 0);
        let b = mock.answer_row(&set, "state B", 0);
        let c = mock.answer_row(&set, "state A", 1);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn abstain_hook_abstains() {
        let set = author::parse_yaml(
            r#"
questions:
  q:
    type: noul
    instructions: "Judge this. [abstain]"
"#,
        )
        .unwrap_or_else(|error| panic!("parses: {error}"));
        let row = MockDecisionModel.answer_row(&set, "anything", 0);
        assert_eq!(
            row.answers.get("q"),
            Some(&QuestionAnswer::abstained()),
            "[abstain] in instructions yields an abstention"
        );
    }
}
