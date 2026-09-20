//! Rendering: IR answers → the jev-1 wire answer shapes.
//!
//! Raw distributions come from the mock; everything else on the wire is derived through
//! `hyprstream-decision`'s normative functions — argmax label (D6 tie-break), expected
//! score (`Σ i·pᵢ`), confidence-from-peakedness (S6a formulas), the legend echo. The stub
//! never hand-computes a derived statistic, so the facade cannot drift from the profile.
//!
//! Abstention on the facade (a superset concept — upstream has no abstain value) renders
//! the value positions as JSON `null`: `{"type": "noul", "noul": null}` etc. Stock SDKs
//! validate those fields as non-nullable and will reject an abstained answer — that is
//! honest: abstention is only contract-stable on the Arrow surface in v1.

use hyprstream_decision::answer::AnswerValue;
use hyprstream_decision::arrow::AnswerRow;
use hyprstream_decision::confidence;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::{QuestionBody, QuestionSet};
use serde_json::{Map, Value};

/// Convert an IR entry back to JSON (insertion order → sorted, matching serde_json's
/// canonical map order; duplicate keys, already rejected at validation, collapse
/// last-wins).
pub fn entry_to_json(entry: &Entry) -> Value {
    match entry {
        Entry::Null => Value::Null,
        Entry::Bool(value) => Value::Bool(*value),
        Entry::Number(value) => serde_json::Number::from_f64(*value)
            .map_or(Value::Null, Value::Number),
        Entry::Str(text) => Value::String(text.clone()),
        Entry::Seq(items) => Value::Array(items.iter().map(entry_to_json).collect()),
        Entry::Map(pairs) => Value::Object(
            pairs
                .iter()
                .map(|(key, value)| (key.clone(), entry_to_json(value)))
                .collect(),
        ),
    }
}

fn probability_map(labels: &[String], probabilities: &[f32]) -> Value {
    Value::Object(
        labels
            .iter()
            .zip(probabilities)
            .map(|(label, p)| {
                (
                    label.clone(),
                    serde_json::Number::from_f64(f64::from(*p)).map_or(Value::Null, Value::Number),
                )
            })
            .collect::<Map<String, Value>>(),
    )
}

fn render_one(question: &hyprstream_decision::spec::QuestionSpec, row: &AnswerRow) -> Value {
    let answer = row.answers.get(&question.id);
    let value = answer.and_then(|a| a.value.as_ref());
    let mut out = Map::new();
    out.insert(
        "type".to_owned(),
        Value::String(question.kind.as_str().to_owned()),
    );
    match (&question.body, value) {
        (QuestionBody::Noul { .. }, Some(AnswerValue::Noul { p_true })) => {
            out.insert(
                "noul".to_owned(),
                serde_json::Number::from_f64(f64::from(*p_true)).map_or(Value::Null, Value::Number),
            );
        }
        (QuestionBody::Noul { .. }, None) => {
            out.insert("noul".to_owned(), Value::Null);
        }
        (QuestionBody::Choice { .. }, Some(value @ AnswerValue::Choice { .. })) => {
            let labels = question.labels();
            let probabilities = value.probabilities();
            let argmax = confidence::argmax_index(&probabilities).unwrap_or(0);
            out.insert(
                "choice".to_owned(),
                Value::String(labels[argmax].clone()),
            );
            out.insert(
                "probabilities".to_owned(),
                probability_map(&labels, &probabilities),
            );
            out.insert(
                "confidence".to_owned(),
                serde_json::Number::from_f64(f64::from(confidence::choice_confidence(
                    &probabilities,
                )))
                .map_or(Value::Null, Value::Number),
            );
        }
        (QuestionBody::Choice { .. }, None) => {
            out.insert("choice".to_owned(), Value::Null);
            out.insert("probabilities".to_owned(), Value::Null);
            out.insert("confidence".to_owned(), Value::Null);
        }
        (QuestionBody::Score { levels }, Some(value @ AnswerValue::Score { .. })) => {
            let labels = question.labels();
            let probabilities = value.probabilities();
            out.insert(
                "score".to_owned(),
                serde_json::Number::from_f64(f64::from(confidence::expected_score(
                    &probabilities,
                )))
                .map_or(Value::Null, Value::Number),
            );
            out.insert(
                "legend".to_owned(),
                Value::Object(
                    levels
                        .iter()
                        .enumerate()
                        .map(|(index, level)| {
                            (
                                index.to_string(),
                                level.as_ref().map_or(Value::Null, entry_to_json),
                            )
                        })
                        .collect(),
                ),
            );
            out.insert(
                "probabilities".to_owned(),
                probability_map(&labels, &probabilities),
            );
            out.insert(
                "confidence".to_owned(),
                serde_json::Number::from_f64(f64::from(confidence::score_confidence(
                    &probabilities,
                )))
                .map_or(Value::Null, Value::Number),
            );
        }
        (QuestionBody::Score { .. }, None) => {
            out.insert("score".to_owned(), Value::Null);
            out.insert("legend".to_owned(), Value::Null);
            out.insert("probabilities".to_owned(), Value::Null);
            out.insert("confidence".to_owned(), Value::Null);
        }
        // Kind mismatches are unreachable: answers are produced from the same IR.
        _ => {
            out.insert("error".to_owned(), Value::String("kind mismatch".to_owned()));
        }
    }
    Value::Object(out)
}

/// Render the full `answers` map for one facade request, keyed by question id.
pub fn render_answers(set: &QuestionSet, row: &AnswerRow) -> Value {
    Value::Object(
        set.questions
            .iter()
            .map(|question| (question.id.clone(), render_one(question, row)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockDecisionModel;
    use hyprstream_decision::author;
    use hyprstream_decision::confidence::CONSUMER_SUM_TOLERANCE;

    #[test]
    fn rendered_answers_satisfy_consumer_tolerance_and_derivations() {
        let set = author::parse_yaml(
            r#"
state: "The box was crushed."
questions:
  is_refund:
    type: noul
  tone:
    type: choice
    criteria:
      angry: "Hostile"
      calm: ~
  severity:
    type: score
    criteria: ["cosmetic", "usable", "unusable"]
"#,
        )
        .unwrap_or_else(|error| panic!("parses: {error}"));
        let row = MockDecisionModel.answer_row(&set, "The box was crushed.", 0);
        let answers = render_answers(&set, &row);

        let noul = &answers["is_refund"];
        assert_eq!(noul["type"], "noul");
        assert!(noul["noul"].as_f64().unwrap_or_else(|| panic!("p")) >= 0.0);

        let choice = &answers["tone"];
        let probabilities = choice["probabilities"].as_object().unwrap_or_else(|| panic!("map"));
        let sum: f64 = probabilities.values().map(|v| v.as_f64().unwrap_or_else(|| panic!("p"))).sum();
        assert!((sum - 1.0).abs() <= f64::from(CONSUMER_SUM_TOLERANCE));
        let argmax = probabilities
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.as_f64()
                    .unwrap_or_else(|| panic!("p"))
                    .partial_cmp(&b.as_f64().unwrap_or_else(|| panic!("p")))
                    .unwrap_or_else(|| panic!("ordered"))
            })
            .map(|(label, _)| label);
        assert_eq!(choice["choice"].as_str(), argmax.map(String::as_str));

        let score = &answers["severity"];
        let levels = score["probabilities"].as_object().unwrap_or_else(|| panic!("map"));
        assert_eq!(levels.len(), 3);
        let expected: f64 = levels
            .iter()
            .map(|(level, p)| level.parse::<f64>().unwrap_or_else(|error| panic!("index: {error}")) * p.as_f64().unwrap_or_else(|| panic!("p")))
            .sum();
        let rendered = score["score"].as_f64().unwrap_or_else(|| panic!("score"));
        assert!(
            (expected - rendered).abs() < 1e-5,
            "score == Σ i·pᵢ: {rendered} vs {expected}"
        );
        assert_eq!(score["legend"]["0"], "cosmetic", "legend echoes criteria");
    }
}
