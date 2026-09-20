//! Subjects: the one seam every eval arm implements.
//!
//! A [`Subject`] answers jev-1 question sets. The harness is deliberately
//! blind to what sits behind the trait — the P0.7 stub, the TypeSafe adapter
//! fronting a frontier LLM, Jev itself, or our own model behind its serving
//! contract all present the same surface, which is exactly the portability
//! thesis of the program.
//!
//! Wire shape for [`HttpSubject`]: `POST {base}/v1/systemone` with
//! `{state, model, questions}`, answers parsed back out of the response
//! envelope. `questions` is a map of id → `{type, instructions?, criteria?}`
//! matching the P0.1a authoring profile (noul criteria as `{true,false}`,
//! choice/score criteria keyed by option name / level index).

use async_trait::async_trait;
use hyprstream_decision::answer::{AnswerValue, QuestionAnswer};
use hyprstream_decision::arrow::AnswerRow;
use hyprstream_decision::confidence::{self, CONSUMER_SUM_TOLERANCE};
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::{QuestionBody, QuestionSet, QuestionSpec};
use serde_json::{Map, Value};

use crate::error::EvalError;

/// One jev-1-shaped answer producer.
#[async_trait]
pub trait Subject: Send + Sync {
    /// The resolved model id recorded on runs and persisted batches. Alias
    /// resolution is the serving side's job; the id returned here is treated
    /// as already resolved.
    fn model_id(&self) -> &str;

    /// Answer every question in `set` for one state row. `row` is the batch
    /// row index, passed through so deterministic subjects can vary by row
    /// exactly the way the P0.7 mock does. Every declared question must be
    /// answered — abstain explicitly rather than omitting the key.
    async fn decide(
        &self,
        set: &QuestionSet,
        state: &Entry,
        row: usize,
    ) -> Result<AnswerRow, EvalError>;
}

/// Convert an IR entry to JSON (insertion order → sorted object keys, the
/// same canonical map order the P0.7 facade renders).
fn entry_to_json(entry: &Entry) -> Value {
    match entry {
        Entry::Null => Value::Null,
        Entry::Bool(value) => Value::Bool(*value),
        Entry::Number(value) => {
            serde_json::Number::from_f64(*value).map_or(Value::Null, Value::Number)
        }
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

fn question_to_json(question: &QuestionSpec) -> Value {
    let mut out = Map::new();
    out.insert(
        "type".to_owned(),
        Value::String(question.kind.as_str().to_owned()),
    );
    if let Some(instructions) = &question.instructions {
        out.insert("instructions".to_owned(), entry_to_json(instructions));
    }
    match &question.body {
        QuestionBody::Noul { criteria } => {
            if let Some(criteria) = criteria {
                let mut map = Map::new();
                if let Some(on_true) = &criteria.on_true {
                    map.insert("true".to_owned(), entry_to_json(on_true));
                }
                if let Some(on_false) = &criteria.on_false {
                    map.insert("false".to_owned(), entry_to_json(on_false));
                }
                out.insert("criteria".to_owned(), Value::Object(map));
            }
        }
        QuestionBody::Choice { options } => {
            let map: Map<String, Value> = options
                .iter()
                .map(|option| {
                    (
                        option.name.clone(),
                        option.rubric.as_ref().map_or(Value::Null, entry_to_json),
                    )
                })
                .collect();
            out.insert("criteria".to_owned(), Value::Object(map));
        }
        QuestionBody::Score { levels } => {
            out.insert(
                "criteria".to_owned(),
                Value::Array(
                    levels
                        .iter()
                        .map(|level| level.as_ref().map_or(Value::Null, entry_to_json))
                        .collect(),
                ),
            );
        }
    }
    Value::Object(out)
}

/// Serialize a request body in the jev-1 wire shape. Exposed for the
/// wire-floor bench, which pins request/response byte counts.
pub fn request_to_json(set: &QuestionSet, state: &Entry, model: &str) -> Value {
    let questions: Map<String, Value> = set
        .questions
        .iter()
        .map(|question| (question.id.clone(), question_to_json(question)))
        .collect();
    let mut body = Map::new();
    body.insert("state".to_owned(), entry_to_json(state));
    body.insert("model".to_owned(), Value::String(model.to_owned()));
    body.insert("questions".to_owned(), Value::Object(questions));
    Value::Object(body)
}

/// Parse one answer object from the response `answers` map back into the IR.
/// A JSON `null` value position (`{"type":"noul","noul":null}`, …) is an
/// abstention; anything structurally off is an [`EvalError::InvalidAnswer`].
fn parse_answer(question: &QuestionSpec, value: &Value) -> Result<QuestionAnswer, EvalError> {
    let invalid = |message: String| EvalError::InvalidAnswer {
        question_id: question.id.clone(),
        message,
    };
    let declared = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("answer object is missing `type`".to_owned()))?;
    if declared != question.kind.as_str() {
        return Err(invalid(format!(
            "answer type `{declared}` does not match question kind `{}`",
            question.kind
        )));
    }
    let labels = question.labels();
    let probabilities_from = |key: &str| -> Result<Option<Vec<f32>>, EvalError> {
        match value.get(key) {
            None => Err(invalid(format!("answer object is missing `{key}`"))),
            Some(Value::Null) => Ok(None),
            Some(Value::Object(map)) => {
                let mut probabilities = Vec::with_capacity(labels.len());
                for label in &labels {
                    let p = map
                        .get(label)
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid(format!("probabilities missing label `{label}`")))?;
                    probabilities.push(p as f32);
                }
                confidence::check_distribution(&probabilities, CONSUMER_SUM_TOLERANCE)
                    .map_err(|error| invalid(format!("distribution: {error}")))?;
                Ok(Some(probabilities))
            }
            Some(_) => Err(invalid(format!("`{key}` is neither null nor an object"))),
        }
    };
    match &question.body {
        QuestionBody::Noul { .. } => match value.get("noul") {
            None => Err(invalid("answer object is missing `noul`".to_owned())),
            Some(Value::Null) => Ok(QuestionAnswer::abstained()),
            Some(number) => {
                let p_true = number
                    .as_f64()
                    .ok_or_else(|| invalid("`noul` is not a number".to_owned()))? as f32;
                if !(0.0..=1.0).contains(&p_true) {
                    return Err(invalid(format!("`noul` out of range: {p_true}")));
                }
                Ok(QuestionAnswer::answered(AnswerValue::Noul { p_true }))
            }
        },
        QuestionBody::Choice { .. } => Ok(match probabilities_from("probabilities")? {
            None => QuestionAnswer::abstained(),
            Some(probabilities) => {
                QuestionAnswer::answered(AnswerValue::Choice { probabilities })
            }
        }),
        QuestionBody::Score { .. } => Ok(match probabilities_from("probabilities")? {
            None => QuestionAnswer::abstained(),
            Some(probabilities) => QuestionAnswer::answered(AnswerValue::Score { probabilities }),
        }),
    }
}

/// Parse the full response envelope into an [`AnswerRow`].
pub fn parse_response_answers(
    set: &QuestionSet,
    body: &Value,
) -> Result<AnswerRow, EvalError> {
    let answers = body
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| EvalError::Http("response envelope is missing `answers`".to_owned()))?;
    let mut row = std::collections::BTreeMap::new();
    for question in &set.questions {
        let value = answers
            .get(&question.id)
            .ok_or_else(|| EvalError::InvalidAnswer {
                question_id: question.id.clone(),
                message: "response has no answer for a declared question".to_owned(),
            })?;
        row.insert(question.id.clone(), parse_answer(question, value)?);
    }
    Ok(AnswerRow { answers: row })
}

/// A subject reached over the jev-1 JSON/HTTPS wire (`POST /v1/systemone`).
///
/// This is the reference client for every remote arm: the P0.7 stub, the
/// TypeSafe adapter fronting frontier LLMs, or Jev itself — the harness
/// speaks one wire and never adapts per provider.
#[derive(Debug, Clone)]
pub struct HttpSubject {
    base_url: String,
    model: String,
    token: String,
    client: reqwest::Client,
}

impl HttpSubject {
    /// A subject at `base_url` (e.g. the stub's loopback address), requesting
    /// `model`, authenticating with `Authorization: Bearer <token>` (any
    /// non-empty token satisfies the stub; real providers need theirs).
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            token: token.into(),
            client: reqwest::Client::new(),
        }
    }

    /// The full decide-endpoint URL.
    pub fn endpoint(&self) -> String {
        format!("{}/v1/systemone", self.base_url)
    }
}

#[async_trait]
impl Subject for HttpSubject {
    fn model_id(&self) -> &str {
        &self.model
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, EvalError> {
        let body = request_to_json(set, state, &self.model);
        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .map_err(|error| EvalError::Http(format!("request failed: {error}")))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| EvalError::Http(format!("reading body: {error}")))?;
        if !status.is_success() {
            return Err(EvalError::Http(format!(
                "status {}: {}",
                status.as_u16(),
                text
            )));
        }
        let parsed: Value = serde_json::from_str(&text)
            .map_err(|error| EvalError::Http(format!("response is not JSON: {error}")))?;
        parse_response_answers(set, &parsed)
    }
}

/// A deterministic in-process reference subject: distributions are a pure
/// function of (model id, canonical question serialization, state, row),
/// blake3-drawn and normalized to producer tolerance — the same construction
/// as the P0.7 mock, available without a server for offline runs and the
/// harness-baseline benches. It is a *fixture*, not a model.
#[derive(Debug, Clone)]
pub struct HashSubject {
    model_id: String,
}

impl HashSubject {
    /// A subject answering as `model_id`.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

fn draw(key: &[u8], index: u64) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key);
    hasher.update(&index.to_le_bytes());
    let output = hasher.finalize();
    let bits = u64::from_le_bytes(
        output.as_bytes()[..8]
            .try_into()
            .unwrap_or([0; 8]),
    );
    (bits as f64 + 0.5) / (u64::MAX as f64 + 1.0)
}

fn hash_distribution(key: &[u8], cardinality: usize) -> Vec<f32> {
    debug_assert!(cardinality >= 1);
    let draws: Vec<f64> = (0..cardinality as u64).map(|i| draw(key, i)).collect();
    let total: f64 = draws.iter().sum();
    let mut probabilities: Vec<f32> = draws.iter().map(|d| (d / total) as f32).collect();
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

#[async_trait]
impl Subject for HashSubject {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        state: &Entry,
        row: usize,
    ) -> Result<AnswerRow, EvalError> {
        let state_text = state.canonical_text();
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let key = format!(
                "{}\x00{}\x00{state_text}\x00{row}",
                self.model_id,
                hyprstream_decision::serialize::serialize_question(question)
            );
            let value = match &question.body {
                QuestionBody::Noul { .. } => AnswerValue::Noul {
                    p_true: draw(key.as_bytes(), 0) as f32,
                },
                QuestionBody::Choice { options } => AnswerValue::Choice {
                    probabilities: hash_distribution(key.as_bytes(), options.len()),
                },
                QuestionBody::Score { levels } => AnswerValue::Score {
                    probabilities: hash_distribution(key.as_bytes(), levels.len()),
                },
            };
            answers.insert(question.id.clone(), QuestionAnswer::answered(value));
        }
        Ok(AnswerRow { answers })
    }
}

/// A subject that answers one-hot at a supplied truth index per question —
/// the perfectly-calibrated, perfectly-accurate anchor arm. Truth is supplied
/// at construction as a map from question id (or item id, via the harness's
/// per-item sets) to the correct label index; questions without a truth entry
/// are answered uniformly rather than guessed.
#[derive(Debug, Clone, Default)]
pub struct TruthSubject {
    model_id: String,
    truth: std::collections::HashMap<String, usize>,
}

impl TruthSubject {
    /// A truth-answering subject with an empty truth table.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            truth: std::collections::HashMap::new(),
        }
    }

    /// Record the correct label index for a question id.
    pub fn with_truth(mut self, question_id: impl Into<String>, index: usize) -> Self {
        self.truth.insert(question_id.into(), index);
        self
    }
}

#[async_trait]
impl Subject for TruthSubject {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let cardinality = question.cardinality();
            let mut probabilities = vec![0.0f32; cardinality];
            match self.truth.get(&question.id) {
                Some(&index) if index < cardinality => probabilities[index] = 1.0,
                _ => probabilities.fill(1.0 / cardinality as f32),
            }
            let value = match &question.body {
                QuestionBody::Noul { .. } => AnswerValue::Noul {
                    p_true: probabilities[1],
                },
                QuestionBody::Choice { .. } => AnswerValue::Choice { probabilities },
                QuestionBody::Score { .. } => AnswerValue::Score { probabilities },
            };
            answers.insert(question.id.clone(), QuestionAnswer::answered(value));
        }
        Ok(AnswerRow { answers })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_decision::author;

    fn fixture() -> QuestionSet {
        author::parse_yaml(
            r#"
state: "The box was crushed."
questions:
  is_refund:
    type: noul
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
        .unwrap()
    }

    #[test]
    fn request_json_roundtrips_through_authoring() {
        let set = fixture();
        let body = request_to_json(&set, &Entry::Str("The box was crushed.".into()), "m");
        let text = serde_json::to_string(&body).unwrap();
        // The wire parse path is the stub's; here assert the shape survives
        // the P0.1a authoring layer unchanged (same questions, same order).
        let value: Value = serde_json::from_str(&text).unwrap();
        let questions = value["questions"].as_object().unwrap();
        assert_eq!(questions.len(), 3);
        assert_eq!(questions["tone"]["criteria"]["angry"], "Hostile message");
        assert_eq!(questions["severity"]["criteria"][1], "usable");
    }

    #[tokio::test]
    async fn hash_subject_is_deterministic_and_normalized() {
        let set = fixture();
        let subject = HashSubject::new("hash-1");
        let state = Entry::Str("s".into());
        let a = subject.decide(&set, &state, 0).await.unwrap();
        let b = subject.decide(&set, &state, 0).await.unwrap();
        assert_eq!(a, b);
        for (id, answer) in &a.answers {
            let value = answer.value.as_ref().unwrap();
            let question = set.question(id).unwrap();
            assert_eq!(value.probabilities().len(), question.cardinality());
            confidence::check_distribution(
                &value.probabilities(),
                confidence::PRODUCER_SUM_TOLERANCE,
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn truth_subject_is_one_hot_at_truth() {
        let set = fixture();
        let subject = TruthSubject::new("truth")
            .with_truth("is_refund", 1)
            .with_truth("tone", 0)
            .with_truth("severity", 2);
        let row = subject
            .decide(&set, &Entry::Null, 0)
            .await
            .unwrap();
        let p = row.answers["severity"].value.as_ref().unwrap().probabilities();
        assert_eq!(p, vec![0.0, 0.0, 1.0]);
        let noul = row.answers["is_refund"].value.as_ref().unwrap().probabilities();
        assert_eq!(noul, vec![0.0, 1.0]);
    }

    #[test]
    fn parse_answer_handles_abstention_and_rejects_garbage() {
        let set = fixture();
        let tone = set.question("tone").unwrap();
        let abstained = parse_answer(tone, &serde_json::json!({"type":"choice","choice":null,"probabilities":null,"confidence":null})).unwrap();
        assert_eq!(abstained, QuestionAnswer::abstained());
        let bad = parse_answer(tone, &serde_json::json!({"type":"noul","noul":0.5}));
        assert!(bad.is_err());
        let missing_label = parse_answer(
            tone,
            &serde_json::json!({"type":"choice","probabilities":{"angry":1.0}}),
        );
        assert!(missing_label.is_err());
    }
}
