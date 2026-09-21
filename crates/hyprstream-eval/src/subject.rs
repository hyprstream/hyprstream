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
use serde_json::Value;

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

    /// The **resolved** model id. Alias resolution is part of the serving
    /// contract: a server may answer an alias request with a different
    /// versioned id (`response.model`), and that resolved id — not the
    /// requested alias — is what runs and persisted batches must record.
    /// In-process subjects resolve to themselves (the default); HTTP subjects
    /// return the last resolved id once a response has been seen, falling
    /// back to the requested string before the first call.
    fn resolved_model_id(&self) -> String {
        self.model_id().to_owned()
    }
}

/// Ordered JSON for the request wire. serde_json's own `Map` is a BTreeMap in
/// this workspace (no `preserve_order` feature), and building the request
/// through it **re-sorts object keys** — destroying declared option order,
/// which is canonical (D6) and load-bearing: the cyclic-permutation stratum
/// measures exactly that order, and the answering side's distribution is a
/// function of it. `serialize_map`/`serialize_seq` emit entries in iteration
/// order, so this type preserves declared order end to end.
#[derive(Debug, Clone, PartialEq)]
pub enum WireJson {
    /// JSON null.
    Null,
    /// JSON bool.
    Bool(bool),
    /// JSON number (non-finite values serialize as null, matching serde_json).
    Number(f64),
    /// JSON string.
    Str(String),
    /// JSON array.
    Array(Vec<WireJson>),
    /// JSON object in **insertion order**.
    Object(Vec<(String, WireJson)>),
}

impl serde::Serialize for WireJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{SerializeMap, SerializeSeq};
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => match serde_json::Number::from_f64(*value) {
                Some(number) => number.serialize(serializer),
                None => serializer.serialize_unit(),
            },
            Self::Str(text) => serializer.serialize_str(text),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

/// Convert an IR entry to wire JSON, preserving map insertion order
/// recursively (the authoring profile's `Entry` order is meaningful).
fn entry_to_wire(entry: &Entry) -> WireJson {
    match entry {
        Entry::Null => WireJson::Null,
        Entry::Bool(value) => WireJson::Bool(*value),
        Entry::Number(value) => WireJson::Number(*value),
        Entry::Str(text) => WireJson::Str(text.clone()),
        Entry::Seq(items) => WireJson::Array(items.iter().map(entry_to_wire).collect()),
        Entry::Map(pairs) => WireJson::Object(
            pairs
                .iter()
                .map(|(key, value)| (key.clone(), entry_to_wire(value)))
                .collect(),
        ),
    }
}

fn question_to_wire(question: &QuestionSpec) -> WireJson {
    let mut out = vec![(
        "type".to_owned(),
        WireJson::Str(question.kind.as_str().to_owned()),
    )];
    if let Some(instructions) = &question.instructions {
        out.push(("instructions".to_owned(), entry_to_wire(instructions)));
    }
    match &question.body {
        QuestionBody::Noul { criteria } => {
            if let Some(criteria) = criteria {
                let mut map = Vec::new();
                if let Some(on_true) = &criteria.on_true {
                    map.push(("true".to_owned(), entry_to_wire(on_true)));
                }
                if let Some(on_false) = &criteria.on_false {
                    map.push(("false".to_owned(), entry_to_wire(on_false)));
                }
                out.push(("criteria".to_owned(), WireJson::Object(map)));
            }
        }
        QuestionBody::Choice { options } => {
            // Declared option order is canonical (D6) — emit in order.
            let map: Vec<(String, WireJson)> = options
                .iter()
                .map(|option| {
                    (
                        option.name.clone(),
                        option.rubric.as_ref().map_or(WireJson::Null, entry_to_wire),
                    )
                })
                .collect();
            out.push(("criteria".to_owned(), WireJson::Object(map)));
        }
        QuestionBody::Score { levels } => {
            out.push((
                "criteria".to_owned(),
                WireJson::Array(
                    levels
                        .iter()
                        .map(|level| level.as_ref().map_or(WireJson::Null, entry_to_wire))
                        .collect(),
                ),
            ));
        }
    }
    WireJson::Object(out)
}

/// Serialize a request body in the jev-1 wire shape, preserving declared
/// question and option order. Also the wire-floor bench's size input.
pub fn request_body(set: &QuestionSet, state: &Entry, model: &str) -> WireJson {
    let questions: Vec<(String, WireJson)> = set
        .questions
        .iter()
        .map(|question| (question.id.clone(), question_to_wire(question)))
        .collect();
    WireJson::Object(vec![
        ("state".to_owned(), entry_to_wire(state)),
        ("model".to_owned(), WireJson::Str(model.to_owned())),
        ("questions".to_owned(), WireJson::Object(questions)),
    ])
}

/// Renormalize a wire-accepted distribution to an exact f32 sum of 1
/// (divide through, then correct the largest component — the same approach
/// as the stub's mock). Wire distributions are accepted at consumer
/// tolerance; the producer-side IR and batch validators require the tighter
/// producer tolerance, so the boundary normalizes on the way in.
fn normalize_distribution(probabilities: &mut [f32]) {
    let sum: f32 = probabilities.iter().sum();
    if sum > 0.0 {
        for p in probabilities.iter_mut() {
            *p /= sum;
        }
    }
    let residual = 1.0 - probabilities.iter().sum::<f32>();
    if let Some(max) = probabilities.iter_mut().max_by(|a, b| a.total_cmp(b)) {
        *max += residual;
    }
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
                // The key set must be exactly the declared labels: extra keys
                // would be silently dropped from a distribution that still
                // sums within tolerance, hiding a schema mismatch.
                for key in map.keys() {
                    if !labels.contains(key) {
                        return Err(invalid(format!(
                            "probabilities carry undeclared label `{key}`"
                        )));
                    }
                }
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
                // The wire accepts consumer-tolerance sums (e.g. a rounded
                // [0.333, 0.333, 0.333]); renormalize so the IR meets the
                // producer tolerance the run/batch validators require.
                normalize_distribution(&mut probabilities);
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
    // Every response key must name a declared question: extra answers would
    // otherwise be silently discarded, hiding a response/request schema
    // mismatch behind a row that passes all per-question checks.
    for key in answers.keys() {
        if set.question(key).is_none() {
            return Err(EvalError::InvalidAnswer {
                question_id: key.clone(),
                message: "response answers an undeclared question".to_owned(),
            });
        }
    }
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
#[derive(Clone)]
pub struct HttpSubject {
    base_url: String,
    model: String,
    token: String,
    client: reqwest::Client,
    /// The server's resolved id (`response.model`), set on the first
    /// successful decide. `Arc` so clones observe the same resolution.
    resolved: std::sync::Arc<parking_lot::RwLock<Option<String>>>,
}

impl std::fmt::Debug for HttpSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the bearer token: Debug output lands in logs and panic
        // diagnostics, and the token is a provider credential.
        f.debug_struct("HttpSubject")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("token", &"<redacted>")
            .field("client", &self.client)
            .field("resolved", &self.resolved)
            .finish()
    }
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
            resolved: std::sync::Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// The full decide-endpoint URL.
    pub fn endpoint(&self) -> String {
        format!("{}/v1/systemone", self.base_url)
    }

    /// The requested model string (possibly an alias), as configured.
    pub fn requested_model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl Subject for HttpSubject {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn resolved_model_id(&self) -> String {
        self.resolved
            .read()
            .clone()
            .unwrap_or_else(|| self.model.clone())
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, EvalError> {
        let body = request_body(set, state, &self.model);
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
        if let Some(model) = parsed.get("model").and_then(Value::as_str) {
            *self.resolved.write() = Some(model.to_owned());
        }
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
        let body = request_body(&set, &Entry::Str("The box was crushed.".into()), "m");
        let text = serde_json::to_string(&body).unwrap();
        // The wire parse path is the stub's; here assert the shape survives
        // the P0.1a authoring layer unchanged (same questions, same order).
        let value: Value = serde_json::from_str(&text).unwrap();
        let questions = value["questions"].as_object().unwrap();
        assert_eq!(questions.len(), 3);
        assert_eq!(questions["tone"]["criteria"]["angry"], "Hostile message");
        assert_eq!(questions["severity"]["criteria"][1], "usable");
    }

    #[test]
    fn request_wire_preserves_declared_option_order() {
        // D6 / cyclic-permutation regression: serde_json's Map is a BTreeMap
        // here (no preserve_order), so anything built through it re-sorts —
        // the wire serializer must not.
        let set = author::parse_yaml(
            r#"
questions:
  pick:
    type: choice
    criteria: { zeta: "rubric z", alpha: ~, mid: ~ }
"#,
        )
        .unwrap();
        let text = serde_json::to_string(&request_body(&set, &Entry::Null, "m")).unwrap();
        let zeta = text.find("zeta").unwrap();
        let alpha = text.find("alpha").unwrap();
        let mid = text.find("mid").unwrap();
        assert!(zeta < alpha && alpha < mid, "declared order on the wire: {text}");
        // …and question order (is_refund before tone before severity) too.
        let body = serde_json::to_string(&request_body(
            &fixture(),
            &Entry::Str("s".into()),
            "m",
        ))
        .unwrap();
        assert!(
            body.find("is_refund").unwrap() < body.find("tone").unwrap()
                && body.find("tone").unwrap() < body.find("severity").unwrap()
        );
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
        // All declared labels present but an extra, undeclared one: the key
        // set must match exactly (a truncated distribution that still sums
        // within tolerance would hide a schema mismatch).
        let extra_label = parse_answer(
            tone,
            &serde_json::json!({"type":"choice","probabilities":{"angry":0.5,"calm":0.4,"bored":0.1}}),
        );
        assert!(extra_label.is_err());
    }

    #[test]
    fn response_answers_reject_undeclared_question_ids() {
        // A stale endpoint answering an extra, undeclared question would
        // otherwise have the key silently discarded.
        let set = fixture();
        let body = serde_json::json!({
            "answers": {
                "is_refund": {"type": "noul", "noul": 0.5},
                "tone": {"type": "choice", "probabilities": {"angry": 0.5, "calm": 0.5}},
                "severity": {"type": "score", "probabilities": {"0": 0.34, "1": 0.33, "2": 0.33}},
                "mystery": {"type": "noul", "noul": 0.5}
            }
        });
        let error = parse_response_answers(&set, &body).err().unwrap();
        assert!(
            matches!(error, EvalError::InvalidAnswer { .. }),
            "undeclared answer key must be InvalidAnswer, got {error:?}"
        );
    }

    #[test]
    fn consumer_tolerance_wire_distributions_are_normalized() {
        // A wire response rounded to consumer tolerance ([0.333, 0.333,
        // 0.333] sums to 0.999) is legal on the wire; the IR it enters is
        // validated at producer tolerance, so the boundary must renormalize.
        let set = fixture();
        let tone = set.question("tone").unwrap();
        let parsed = parse_answer(
            tone,
            &serde_json::json!({
                "type": "choice",
                "probabilities": {"angry": 0.499, "calm": 0.5}
            }),
        )
        .unwrap();
        let probabilities = parsed.value.as_ref().unwrap().probabilities();
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE)
            .unwrap();
        // Renormalization must not reorder or skew: both components ~1/2.
        assert!(probabilities.iter().all(|p| (*p - 0.5).abs() < 1e-3));
    }
}
