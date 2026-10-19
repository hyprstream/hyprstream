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
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

use crate::error::EvalError;

/// One jev-1-shaped answer producer.
#[async_trait]
pub trait Subject: Send + Sync {
    /// The resolved model id recorded on runs and persisted batches. Alias
    /// resolution is the serving side's job; the id returned here is treated
    /// as already resolved.
    fn model_id(&self) -> &str;

    /// Check subject-wide input constraints without I/O, model queries,
    /// mutable state, or other side effects. The default accepts inputs,
    /// including explicit null states supported by native/in-process
    /// subjects. Transport adapters may be stricter. Harness execution uses
    /// [`Subject::preflight_row`] so row-dependent subjects can check the
    /// exact request before any row is decided.
    fn preflight(&self, _set: &QuestionSet, _state: &Entry) -> Result<(), EvalError> {
        Ok(())
    }

    /// Check subject-specific constraints for one concrete request row.
    /// Subjects with row-dependent behavior can validate the same selection
    /// that `decide` will use. The default preserves compatibility for
    /// subjects whose preflight does not depend on the row index.
    fn preflight_row(
        &self,
        set: &QuestionSet,
        state: &Entry,
        _row: usize,
    ) -> Result<(), EvalError> {
        self.preflight(set, state)
    }

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

    /// Decide one row and return the model version bound to THAT response.
    /// Runs stamp one version on every row, so the version must come from
    /// the same exchange as the answers: reading `resolved_model_id` after
    /// `decide` races with another concurrent run sharing this subject (its
    /// response can overwrite the resolved slot in between). The default
    /// implementation composes the two and is correct for subjects whose
    /// resolved id never changes; `HttpSubject` overrides it to return the
    /// version from its own response.
    async fn decide_with_version(
        &self,
        set: &QuestionSet,
        state: &Entry,
        row: usize,
    ) -> Result<(AnswerRow, String), EvalError> {
        let answers = self.decide(set, state, row).await?;
        let resolved = self.resolved_model_id();
        Ok((answers, resolved))
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
    /// JSON number (non-finite values fail serialization).
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
                None => Err(serde::ser::Error::custom("non-finite Entry::Number")),
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

/// Renormalize an accepted distribution to producer tolerance, in stages:
/// Producer-valid vectors retain their values when the intended winner
/// survived narrowing. Otherwise divide through and
/// apply a uniform correction first, preserving ties where f32 permits it.
/// If rounding merges a later-index winner with an earlier runner-up, lift
/// that original D6 winner by one representable step. Residual shortfalls go
/// to the winner; overages are removed from another component. The winner
/// is never lowered to a runner-up level, so normalization cannot change
/// the predicted label. Very wide uniform vectors may require breaking a
/// tie to meet producer tolerance; their earliest-index winner is retained.
/// Also used by TruthSubject's uniform fallback and teacher averages.
pub(crate) fn normalize_distribution(probabilities: &mut [f32]) {
    if let Some(winner) = hyprstream_decision::confidence::argmax_index(probabilities) {
        normalize_distribution_with_winner(probabilities, winner);
    }
}

pub(crate) fn normalize_distribution_with_winner(probabilities: &mut [f32], winner: usize) {
    use hyprstream_decision::confidence::{check_distribution, PRODUCER_SUM_TOLERANCE};

    // The caller may have selected the winner before narrowing f64 values.
    if hyprstream_decision::confidence::argmax_index(probabilities) != Some(winner) {
        let max = probabilities.iter().copied().fold(0.0f32, f32::max);
        probabilities[winner] = max.next_up().min(1.0);
    }
    // Stage 0: a producer-valid input is already fine — normalizing it
    // further can push it out of tolerance AND below zero (see the doc
    // comment's 49-way example).
    if check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).is_ok() {
        return;
    }
    let sum: f32 = probabilities.iter().sum();
    if sum > 0.0 {
        for p in probabilities.iter_mut() {
            *p /= sum;
        }
    }
    if hyprstream_decision::confidence::argmax_index(probabilities) != Some(winner) {
        let max = probabilities.iter().copied().fold(0.0f32, f32::max);
        probabilities[winner] = max.next_up().min(1.0);
    }
    if check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).is_ok() {
        return;
    }
    // Order-preserving correction: adding the same share to every component
    // is monotone in f32, so ties stay ties and no leader can cross its
    // runner-up (a single-component correction by a large residual could
    // flip the argmax when two leading probabilities are closer than the
    // residual — observed on a consumer-valid 41-way vector). The share is
    // bounded below by `-min_component` so no component can go negative.
    let residual = 1.0 - probabilities.iter().sum::<f32>();
    #[allow(clippy::cast_precision_loss)]
    let mut share = residual / probabilities.len().max(1) as f32;
    if share < 0.0 {
        let min_component = probabilities.iter().copied().fold(f32::INFINITY, f32::min);
        share = share.max(-min_component);
    }
    if share != 0.0 {
        for p in probabilities.iter_mut() {
            *p += share;
        }
    }
    if hyprstream_decision::confidence::argmax_index(probabilities) != Some(winner) {
        let max = probabilities.iter().copied().fold(0.0f32, f32::max);
        probabilities[winner] = max.next_up().min(1.0);
    }
    if check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).is_ok() {
        return;
    }
    // Correct remaining f32 summation error without lowering the original
    // winner. Keep the number of full-vector passes fixed: correcting one
    // losing component at a time can take O(n) iterations for a sparse,
    // high-cardinality vector, and each distribution check is itself O(n).
    // Scaling the losing mass distributes a negative residual over all
    // non-winners, while a positive residual still belongs on the winner. A
    // fixed-precision search finds a scale whose *sequential f32 fold* meets
    // the contract; using an f64 mass estimate here can overcorrect when a
    // long f32 fold has accumulated rounding error. A few bounded outer
    // passes account for the final f32 rounding without making work depend
    // on question cardinality.
    for _ in 0..8 {
        if check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).is_ok() {
            return;
        }
        let residual = 1.0 - probabilities.iter().sum::<f32>();
        if residual > 0.0 {
            probabilities[winner] = (probabilities[winner] + residual).min(1.0);
        } else if residual < 0.0 {
            let mut low = 0.0f32;
            let mut high = 1.0f32;
            // Binary32 has 24 bits of precision; 32 iterations also cover
            // endpoint rounding while keeping this correction O(n).
            for _ in 0..32 {
                let scale = low + (high - low) * 0.5;
                if scale == low || scale == high {
                    break;
                }
                let mut sum = 0.0f32;
                for (index, probability) in probabilities.iter().enumerate() {
                    sum += if index == winner {
                        *probability
                    } else {
                        *probability * scale
                    };
                }
                if sum <= 1.0 + PRODUCER_SUM_TOLERANCE {
                    low = scale;
                } else {
                    high = scale;
                }
            }
            for (index, probability) in probabilities.iter_mut().enumerate() {
                if index != winner {
                    *probability *= low;
                }
            }
        } else {
            break;
        }
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
                // Equal cardinality plus the per-declared-label lookups below
                // establishes exact key equality without scanning all labels
                // for every response key (Score cardinality is unbounded).
                if map.len() != labels.len() {
                    return Err(invalid(format!(
                        "probabilities carry {} labels; expected {}",
                        map.len(),
                        labels.len()
                    )));
                }
                let mut probabilities = Vec::with_capacity(labels.len());
                let mut wire_winner: Option<(usize, f64)> = None;
                for (index, label) in labels.iter().enumerate() {
                    let p = map
                        .get(label)
                        .and_then(Value::as_f64)
                        .ok_or_else(|| invalid(format!("probabilities missing label `{label}`")))?;
                    if !(0.0..=1.0).contains(&p) {
                        return Err(invalid(format!(
                            "probability for `{label}` out of range: {p}"
                        )));
                    }
                    if wire_winner.is_none_or(|(_, best)| p > best) {
                        wire_winner = Some((index, p));
                    }
                    probabilities.push(p as f32);
                }
                confidence::check_distribution(&probabilities, CONSUMER_SUM_TOLERANCE)
                    .map_err(|error| invalid(format!("distribution: {error}")))?;
                // The wire accepts consumer-tolerance sums (e.g. a rounded
                // [0.333, 0.333, 0.333]); renormalize so the IR meets the
                // producer tolerance the run/batch validators require.
                let winner = wire_winner
                    .map(|(index, _)| index)
                    .ok_or_else(|| invalid("empty wire distribution".to_owned()))?;
                normalize_distribution_with_winner(&mut probabilities, winner);
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
                    .ok_or_else(|| invalid("`noul` is not a number".to_owned()))?;
                if !(0.0..=1.0).contains(&p_true) {
                    return Err(invalid(format!("`noul` out of range: {p_true}")));
                }
                let narrowed = p_true as f32;
                // Noul's implicit [1-p, p] must retain a true wire winner.
                let narrowed = if p_true > 0.5 && narrowed == 0.5 {
                    narrowed.next_up()
                } else {
                    narrowed
                };
                Ok(QuestionAnswer::answered(AnswerValue::Noul {
                    p_true: narrowed,
                }))
            }
        },
        QuestionBody::Choice { .. } => Ok(match probabilities_from("probabilities")? {
            None => QuestionAnswer::abstained(),
            Some(probabilities) => QuestionAnswer::answered(AnswerValue::Choice { probabilities }),
        }),
        QuestionBody::Score { .. } => Ok(match probabilities_from("probabilities")? {
            None => QuestionAnswer::abstained(),
            Some(probabilities) => QuestionAnswer::answered(AnswerValue::Score { probabilities }),
        }),
    }
}

/// Parse the full response envelope into an [`AnswerRow`].
pub fn parse_response_answers(set: &QuestionSet, body: &Value) -> Result<AnswerRow, EvalError> {
    // This public parser also accepts programmatic inputs without HTTP
    // preflight. Exact key-set admission requires unique IDs and labels.
    let ids = crate::run::question_ids(set).map_err(EvalError::InvalidInput)?;
    for question in &set.questions {
        crate::run::validate_question_profile(question).map_err(EvalError::InvalidInput)?;
    }
    let answers = body
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| EvalError::Http("response envelope is missing `answers`".to_owned()))?;
    // Every response key must name a declared question: extra answers would
    // otherwise be silently discarded, hiding a response/request schema
    // mismatch behind a row that passes all per-question checks.
    for key in answers.keys() {
        if !ids.contains(key.as_str()) {
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

/// Deserialize JSON while rejecting repeated object members at every depth.
/// Parsing into `serde_json::Value` directly silently keeps only one value
/// for duplicate keys, which can conceal conflicting model provenance,
/// answers, or probability labels before their respective validators run.
struct UniqueJsonValueSeed;

impl<'de> DeserializeSeed<'de> for UniqueJsonValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonValueVisitor)
    }
}

struct UniqueJsonValueVisitor;

impl<'de> Visitor<'de> for UniqueJsonValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value with unique object member names")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element_seed(UniqueJsonValueSeed)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object member `{key}`"
                )));
            }
            let value = map.next_value_seed(UniqueJsonValueSeed)?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

/// Decode and validate the complete HTTP response envelope. This is the same
/// client work used by [`HttpSubject`] and the wire-floor response benchmark.
pub fn decode_response(set: &QuestionSet, text: &str) -> Result<(AnswerRow, String), EvalError> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let parsed = UniqueJsonValueSeed
        .deserialize(&mut deserializer)
        .map_err(|error| EvalError::Http(format!("response is not JSON: {error}")))?;
    deserializer
        .end()
        .map_err(|error| EvalError::Http(format!("response is not JSON: {error}")))?;
    let Some(resolved) = parsed
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
    else {
        return Err(EvalError::Http(
            "response is missing the resolved `model` id the serving contract requires (absent, non-string, or empty)"
                .to_owned(),
        ));
    };
    let resolved = resolved.to_owned();
    let answers = parse_response_answers(set, &parsed)?;
    crate::run::validate_set_answer_row(set, &answers).map_err(|message| {
        EvalError::InvalidAnswer {
            question_id: "response".to_owned(),
            message,
        }
    })?;
    Ok((answers, resolved))
}

/// Operational bounds for remote evaluation, not inference latency targets.
#[derive(Debug, Clone, Copy)]
pub struct HttpBudget {
    /// Maximum connection-establishment time.
    pub connect_timeout: std::time::Duration,
    /// Whole HTTP query deadline, including response-body consumption.
    pub request_timeout: std::time::Duration,
    /// Maximum response-body bytes, for successful and error responses alike.
    pub max_response_bytes: usize,
}

impl Default for HttpBudget {
    fn default() -> Self {
        Self {
            connect_timeout: std::time::Duration::from_secs(10),
            request_timeout: std::time::Duration::from_secs(120),
            max_response_bytes: 16 * 1024 * 1024,
        }
    }
}

impl HttpBudget {
    fn validate(self) -> Result<(), &'static str> {
        let now = std::time::Instant::now();
        if self.connect_timeout.is_zero()
            || self.request_timeout.is_zero()
            || now.checked_add(self.connect_timeout).is_none()
            || now.checked_add(self.request_timeout).is_none()
            || self.max_response_bytes == 0
        {
            return Err(
                "HTTP budgets require positive, representable deadlines and a positive body cap",
            );
        }
        Ok(())
    }
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
    client: Result<reqwest::Client, &'static str>,
    budget: HttpBudget,
    /// The latest fully accepted response's resolved id. `Arc` so clones
    /// share the cache; version-bound results never depend on this slot.
    resolved: std::sync::Arc<parking_lot::RwLock<Option<String>>>,
}

impl std::fmt::Debug for HttpSubject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the bearer token: Debug output lands in logs and panic
        // diagnostics, and the token is a provider credential.
        f.debug_struct("HttpSubject")
            .field("base_url", &"<redacted>")
            .field("model", &self.model)
            .field("token", &"<redacted>")
            .field("client_ready", &self.client.is_ok())
            .field("budget", &self.budget)
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
        Self::new_with_budget(base_url, model, token, HttpBudget::default())
    }

    /// Configure finite evaluation safeguards. Invalid budgets or client
    /// initialization are rejected by pure preflight before any teacher call;
    /// construction never falls back to an unbounded client or panics.
    pub fn new_with_budget(
        base_url: impl Into<String>,
        model: impl Into<String>,
        token: impl Into<String>,
        budget: HttpBudget,
    ) -> Self {
        let client = budget.validate().and_then(|()| {
            reqwest::Client::builder()
                .connect_timeout(budget.connect_timeout)
                .timeout(budget.request_timeout)
                .build()
                .map_err(|_| "HTTP client initialization failed")
        });
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            token: token.into(),
            client,
            budget,
            resolved: std::sync::Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// The full decide-endpoint URL.
    pub fn endpoint(&self) -> String {
        format!("{}/v1/systemone", self.base_url)
    }

    // Shared with side-effect-free preflight so request construction cannot
    // fail deterministically after an earlier ensemble teacher was queried.
    fn request_builder(&self) -> Result<reqwest::RequestBuilder, EvalError> {
        let client = self
            .client
            .as_ref()
            .map_err(|message| EvalError::InvalidInput((*message).to_owned()))?;
        Ok(client
            .post(self.endpoint())
            .bearer_auth(&self.token)
            .header(reqwest::header::CONTENT_TYPE, "application/json"))
    }

    /// The requested model string (possibly an alias), as configured.
    pub fn requested_model(&self) -> &str {
        &self.model
    }
    /// POST one decide request, returning the parsed answers and the model
    /// version THIS response resolved to. The version travels with the
    /// answers so a concurrent run sharing this subject cannot overwrite
    /// the resolved slot in between. The serving contract requires
    /// `response.model` to identify the resolved version — a response
    /// without it is rejected, never attributed to the requested alias or
    /// a version cached from an unrelated earlier response.
    async fn post(
        &self,
        set: &QuestionSet,
        state: &Entry,
    ) -> Result<(AnswerRow, String), EvalError> {
        self.preflight(set, state)?;
        // Encode before opening a connection: serde_json otherwise turns a
        // directly-constructed non-finite IR number into JSON null.
        let body = serde_json::to_vec(&request_body(set, state, &self.model))
            .map_err(|error| EvalError::InvalidInput(format!("invalid request entry: {error}")))?;
        let mut response = self
            .request_builder()?
            .body(body)
            .send()
            .await
            .map_err(|_| EvalError::Http("HTTP request failed or timed out".to_owned()))?;
        let status = response.status();
        let limit = self.budget.max_response_bytes;
        if response
            .content_length()
            .is_some_and(|length| !usize::try_from(length).is_ok_and(|length| length <= limit))
        {
            return Err(EvalError::Http(
                "HTTP response exceeds the byte budget".to_owned(),
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| EvalError::Http("reading HTTP response failed or timed out".to_owned()))?
        {
            if chunk.len() > limit.saturating_sub(bytes.len()) {
                return Err(EvalError::Http(
                    "HTTP response exceeds the byte budget".to_owned(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            // Providers may echo credentials in their bodies. Keep only status.
            return Err(EvalError::Http(format!("status {}", status.as_u16())));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| EvalError::Http("HTTP response is not UTF-8".to_owned()))?;
        let (answers, resolved) = decode_response(set, text)?;
        *self.resolved.write() = Some(resolved.clone());
        Ok((answers, resolved))
    }
}

#[async_trait]
impl Subject for HttpSubject {
    fn preflight(&self, set: &QuestionSet, state: &Entry) -> Result<(), EvalError> {
        // Direct callers also require unique question IDs and declared labels
        // before any network call; do not rely on Harness/teacher validation.
        crate::run::question_ids(set).map_err(EvalError::InvalidInput)?;
        for question in &set.questions {
            crate::run::validate_question_profile(question).map_err(EvalError::InvalidInput)?;
        }
        self.budget
            .validate()
            .map_err(|message| EvalError::InvalidInput(message.to_owned()))?;
        if self.model.is_empty() {
            return Err(EvalError::InvalidInput(
                "the HTTP subject requires a non-empty model".to_owned(),
            ));
        }
        if self.token.trim().is_empty() {
            return Err(EvalError::InvalidInput(
                "the HTTP subject requires a non-empty bearer token".to_owned(),
            ));
        }
        if matches!(state, Entry::Null) {
            return Err(EvalError::InvalidInput(
                "the legacy /v1/systemone HTTP endpoint requires a non-null state".to_owned(),
            ));
        }
        let endpoint = reqwest::Url::parse(&self.endpoint()).map_err(|_| {
            EvalError::InvalidInput(
                "the HTTP subject requires an absolute HTTP(S) endpoint".to_owned(),
            )
        })?;
        if !matches!(endpoint.scheme(), "http" | "https") {
            return Err(EvalError::InvalidInput(
                "the HTTP subject requires an absolute HTTP(S) endpoint".to_owned(),
            ));
        }
        self.request_builder()?.build().map_err(|_| {
            // Request/headers/URL may contain credentials. Never include the
            // underlying error or Debug representation in this public error.
            EvalError::InvalidInput("invalid HTTP subject request configuration".to_owned())
        })?;
        Ok(())
    }

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
        Ok(self.post(set, state).await?.0)
    }

    async fn decide_with_version(
        &self,
        set: &QuestionSet,
        state: &Entry,
        _row: usize,
    ) -> Result<(AnswerRow, String), EvalError> {
        self.post(set, state).await
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
    let bits = u64::from_le_bytes(output.as_bytes()[..8].try_into().unwrap_or([0; 8]));
    (bits as f64 + 0.5) / (u64::MAX as f64 + 1.0)
}

fn hash_distribution(key: &[u8], cardinality: usize) -> Vec<f32> {
    debug_assert!(cardinality >= 1);
    let draws: Vec<f64> = (0..cardinality as u64).map(|i| draw(key, i)).collect();
    let total: f64 = draws.iter().sum();
    let mut probabilities: Vec<f32> = draws.iter().map(|d| (d / total) as f32).collect();
    // The same order-preserving normalization the wire and teacher paths
    // use: applying the whole residual to the maximum (the old approach)
    // can cross the runner-up when the negative residual exceeds the
    // max/runner-up gap, changing the deterministic baseline's predicted
    // label and corrupting accuracy/flip measurements.
    normalize_distribution(&mut probabilities);
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
/// are answered uniformly rather than guessed. Set runs can apply the same
/// question to rows with DIFFERENT truths: [`TruthSubject::with_row_truth`]
/// keys an entry by (row index, question id) and takes precedence over the
/// question-level entry for that row.
#[derive(Debug, Clone, Default)]
pub struct TruthSubject {
    model_id: String,
    truth: std::collections::HashMap<String, usize>,
    row_truth: std::collections::HashMap<(usize, String), usize>,
}

impl TruthSubject {
    /// A truth-answering subject with an empty truth table.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            truth: std::collections::HashMap::new(),
            row_truth: std::collections::HashMap::new(),
        }
    }

    fn validate_model_id(&self) -> Result<(), EvalError> {
        if self.model_id.is_empty() {
            return Err(EvalError::InvalidInput(
                "the truth subject requires a non-empty model id".to_owned(),
            ));
        }
        Ok(())
    }

    /// Record the correct label index for a question id.
    pub fn with_truth(mut self, question_id: impl Into<String>, index: usize) -> Self {
        self.truth.insert(question_id.into(), index);
        self
    }

    /// Record the correct label index for a question on ONE ROW of a set
    /// run (the `row` argument of `decide`). Without this, every row gets
    /// the question-level entry — the "perfectly accurate" anchor would
    /// score at chance on a set whose rows have different truths.
    pub fn with_row_truth(
        mut self,
        row: usize,
        question_id: impl Into<String>,
        index: usize,
    ) -> Self {
        self.row_truth.insert((row, question_id.into()), index);
        self
    }
}

#[async_trait]
impl Subject for TruthSubject {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn preflight(&self, set: &QuestionSet, _state: &Entry) -> Result<(), EvalError> {
        self.validate_model_id()?;
        for question in &set.questions {
            let cardinality = question.cardinality();
            if let Some(&index) = self.truth.get(&question.id) {
                validate_truth_index(&question.id, index, cardinality)?;
            }
        }
        Ok(())
    }

    fn preflight_row(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        row: usize,
    ) -> Result<(), EvalError> {
        self.validate_model_id()?;
        for question in &set.questions {
            let cardinality = question.cardinality();
            let configured = self
                .row_truth
                .get(&(row, question.id.clone()))
                .or_else(|| self.truth.get(&question.id));
            if let Some(&index) = configured {
                validate_truth_index(&question.id, index, cardinality)?;
            }
        }
        Ok(())
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        row: usize,
    ) -> Result<AnswerRow, EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let cardinality = question.cardinality();
            let mut probabilities = vec![0.0f32; cardinality];
            let configured = self
                .row_truth
                .get(&(row, question.id.clone()))
                .or_else(|| self.truth.get(&question.id));
            match configured {
                Some(&index) if index < cardinality => probabilities[index] = 1.0,
                // A configured-but-invalid truth is NOT the same as no
                // truth: silently degrading the documented perfectly
                // accurate anchor into a uniform subject corrupts
                // sanity-check results.
                Some(&index) => return Err(truth_index_error(&question.id, index, cardinality)),
                // Uniform fallback: the rounded f32 reciprocal can violate
                // the producer sum tolerance at high cardinalities (78
                // options sum to ~1.0000011), so renormalize. The D6-safe
                // correction keeps the argmax at the earliest index (no f32
                // vector of 78 exactly-equal components can meet the
                // tolerance, so one component carries the residual).
                None => {
                    probabilities.fill(1.0 / cardinality as f32);
                    normalize_distribution(&mut probabilities);
                }
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

fn validate_truth_index(
    question_id: &str,
    index: usize,
    cardinality: usize,
) -> Result<(), EvalError> {
    if index >= cardinality {
        Err(truth_index_error(question_id, index, cardinality))
    } else {
        Ok(())
    }
}

fn truth_index_error(question_id: &str, index: usize, cardinality: usize) -> EvalError {
    EvalError::InvalidInput(format!(
        "truth index {index} configured for question `{question_id}` is outside its cardinality {cardinality}"
    ))
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
        assert!(
            zeta < alpha && alpha < mid,
            "declared order on the wire: {text}"
        );
        // …and question order (is_refund before tone before severity) too.
        let body =
            serde_json::to_string(&request_body(&fixture(), &Entry::Str("s".into()), "m")).unwrap();
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
        let row = subject.decide(&set, &Entry::Null, 0).await.unwrap();
        let p = row.answers["severity"]
            .value
            .as_ref()
            .unwrap()
            .probabilities();
        assert_eq!(p, vec![0.0, 0.0, 1.0]);
        let noul = row.answers["is_refund"]
            .value
            .as_ref()
            .unwrap()
            .probabilities();
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
    fn wire_probability_keys_require_exact_declared_set() {
        let set = fixture();
        for id in ["tone", "severity"] {
            let question = set.question(id).unwrap();
            let labels = question.labels();
            let mut exact = serde_json::Map::new();
            // Opposite insertion order to the declared labels; the numeric
            // vector must still follow the question, not response ordering.
            for (index, label) in labels.iter().enumerate().rev() {
                exact.insert(
                    label.clone(),
                    serde_json::json!(if index == labels.len() - 1 { 1.0 } else { 0.0 }),
                );
            }
            let response = |probabilities: serde_json::Map<String, Value>| serde_json::json!({"type": question.kind.as_str(), "probabilities": probabilities});
            let parsed = parse_answer(question, &response(exact.clone())).unwrap();
            let probabilities = parsed.value.unwrap().probabilities();
            assert_eq!(probabilities.len(), labels.len());
            assert_eq!(probabilities[labels.len() - 1], 1.0);
            assert!(probabilities[..labels.len() - 1].iter().all(|p| *p == 0.0));

            let mut extra = exact.clone();
            extra.insert("undeclared".to_owned(), serde_json::json!(0.0));
            let mut missing = exact.clone();
            missing.remove(&labels[0]);
            let mut substituted = missing.clone();
            substituted.insert("undeclared".to_owned(), serde_json::json!(0.0));
            // All three retain a valid sum of one. The substitution also has
            // equal cardinality, so only exact key membership can reject it.
            for malformed in [extra, missing, substituted] {
                let error = parse_answer(question, &response(malformed)).unwrap_err();
                assert!(
                    matches!(error, EvalError::InvalidAnswer { question_id, .. } if question_id == id)
                );
            }
        }
    }

    #[test]
    fn wide_score_wire_preserves_numeric_level_order() {
        const LEVELS: usize = 4096;
        const WINNER: usize = 3073;
        let question = QuestionSpec {
            id: "wide_score".to_owned(),
            kind: hyprstream_decision::QuestionKind::Score,
            instructions: None,
            body: QuestionBody::Score {
                levels: vec![None; LEVELS],
            },
        };
        let wire: serde_json::Map<String, Value> = (0..LEVELS)
            .rev()
            .map(|level| {
                (
                    level.to_string(),
                    serde_json::json!(if level == WINNER { 1.0 } else { 0.0 }),
                )
            })
            .collect();
        let parsed = parse_answer(
            &question,
            &serde_json::json!({"type": "score", "probabilities": wire}),
        )
        .unwrap();
        let Some(AnswerValue::Score { probabilities }) = parsed.value else {
            panic!("valid score must remain answered")
        };
        assert_eq!(probabilities.len(), LEVELS);
        for (level, probability) in probabilities.iter().enumerate() {
            assert_eq!(
                probability.to_bits(),
                if level == WINNER {
                    1.0_f32.to_bits()
                } else {
                    0.0_f32.to_bits()
                }
            );
        }
    }

    #[test]
    fn wire_normalization_preserves_ties() {
        // Seven options rounded to 0.1429 each (sum 1.0003, legal at
        // consumer tolerance): normalization must not turn the tie into a
        // unique winner — D6 earliest-index argmax depends on it.
        let set = author::parse_yaml(
            r#"
questions:
  pick:
    type: choice
    criteria: { o1: ~, o2: ~, o3: ~, o4: ~, o5: ~, o6: ~, o7: ~ }
"#,
        )
        .unwrap();
        let question = set.question("pick").unwrap();
        let parsed = parse_answer(
            question,
            &serde_json::json!({
                "type": "choice",
                "probabilities": {"o1": 0.1429, "o2": 0.1429, "o3": 0.1429, "o4": 0.1429, "o5": 0.1429, "o6": 0.1429, "o7": 0.1429}
            }),
        )
        .unwrap();
        let probabilities = parsed.value.as_ref().unwrap().probabilities();
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        assert!(
            probabilities.windows(2).all(|w| w[0] == w[1]),
            "the tie must survive normalization: {probabilities:?}"
        );
        assert_eq!(
            hyprstream_decision::confidence::argmax_index(&probabilities),
            Some(0),
            "D6: earliest index on a tie"
        );
    }

    #[tokio::test]
    async fn truth_subject_uniform_fallback_is_producer_valid_at_high_cardinality() {
        // 78 options: the naive `1.0 / 78` f32 fill sums to ~1.0000011,
        // beyond producer tolerance — the fallback must renormalize.
        let options = (0..78)
            .map(|i| hyprstream_decision::spec::ChoiceOption {
                name: format!("o{i}"),
                rubric: None,
            })
            .collect();
        let question = QuestionSpec {
            id: "wide".into(),
            kind: hyprstream_decision::QuestionKind::Choice,
            instructions: None,
            body: QuestionBody::Choice { options },
        };
        let set = QuestionSet {
            state: None,
            questions: vec![question],
        };
        let subject = TruthSubject::new("truth"); // no truth configured
        let row = subject.decide(&set, &Entry::Null, 0).await.unwrap();
        let probabilities = row.answers["wide"].value.as_ref().unwrap().probabilities();
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        // No f32 vector of 78 exactly-equal components can meet the producer
        // tolerance (best achievable error ~1.07e-6), so the fallback is as
        // uniform as f32 allows: one component carries the residual, spread
        // stays within a few ulps, and the D6 argmax is still the earliest
        // index (the tie-break outcome).
        let min = probabilities.iter().copied().fold(f32::INFINITY, f32::min);
        let max = probabilities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max - min) / max < 1e-3,
            "near-uniform: min {min}, max {max}"
        );
        assert_eq!(
            hyprstream_decision::confidence::argmax_index(&probabilities),
            Some(0),
            "D6: earliest index"
        );
    }

    #[test]
    fn wire_normalization_preserves_a_near_tied_argmax() {
        // Consumer-valid 41-way response summing to 1.0045794 with two
        // leading probabilities 4e-8 apart: a single-component residual
        // correction (-1.1e-4) would cross the runner-up and flip the D6
        // argmax; the order-preserving correction must not.
        let options = (0..41)
            .map(|i| hyprstream_decision::spec::ChoiceOption {
                name: format!("o{i}"),
                rubric: None,
            })
            .collect();
        let question = QuestionSpec {
            id: "wide41".into(),
            kind: hyprstream_decision::QuestionKind::Choice,
            instructions: None,
            body: QuestionBody::Choice { options },
        };
        let rest = (1.0045794f64 - 0.02450203 - 0.02450199) / 39.0;
        let mut map = serde_json::Map::new();
        map.insert("o0".to_owned(), serde_json::json!(0.02450203));
        map.insert("o1".to_owned(), serde_json::json!(0.02450199));
        for i in 2..41 {
            map.insert(format!("o{i}"), serde_json::json!(rest));
        }
        let parsed = parse_answer(
            &question,
            &serde_json::json!({"type": "choice", "probabilities": map}),
        )
        .unwrap();
        let probabilities = parsed.value.as_ref().unwrap().probabilities();
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        assert!(
            probabilities[0] > probabilities[1],
            "the leader gap must survive normalization: {:?}",
            &probabilities[..2]
        );
        assert_eq!(
            hyprstream_decision::confidence::argmax_index(&probabilities),
            Some(0),
            "D6 argmax must not flip"
        );
    }

    #[test]
    fn sparse_high_cardinality_overage_is_corrected_without_dropping_levels() {
        const LEVELS: usize = 100_000;
        let mut probabilities = Vec::with_capacity(LEVELS * 2 + 1);
        probabilities.push(0.9f32);
        probabilities.extend(std::iter::repeat_n(0.1 / LEVELS as f32, LEVELS));
        probabilities.resize(LEVELS * 2 + 1, 0.0);

        // This is consumer-valid but rounds slightly over one after f32
        // normalization. Zero components clamp the uniform correction to
        // zero, which used to trigger a component-at-a-time full scan.
        confidence::check_distribution(&probabilities, confidence::CONSUMER_SUM_TOLERANCE).unwrap();
        normalize_distribution_with_winner(&mut probabilities, 0);

        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        assert_eq!(confidence::argmax_index(&probabilities), Some(0));
        assert!(probabilities[1..=LEVELS].iter().all(|p| *p > 0.0));
        assert!(probabilities[LEVELS + 1..].iter().all(|p| *p == 0.0));
    }

    #[tokio::test]
    async fn truth_subject_rejects_an_out_of_range_configured_truth() {
        // A configured-but-invalid truth index must NOT degrade into the
        // uniform fallback — that would silently turn the documented
        // perfectly accurate anchor into an uncertain subject.
        let set = fixture();
        let subject = TruthSubject::new("truth").with_truth("tone", 7);
        let error = subject.decide(&set, &Entry::Null, 0).await.err().unwrap();
        assert!(
            matches!(error, EvalError::InvalidInput(_)),
            "out-of-range configured truth must be InvalidInput, got {error:?}"
        );
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
    fn decode_response_rejects_duplicate_model_answer_and_probability_members() {
        let set = fixture();
        let cases = [
            (
                "model",
                r#"{"model":"first","model":"second","answers":{}}"#,
            ),
            (
                "tone",
                r#"{"model":"m","answers":{"tone":{"type":"choice","probabilities":{"angry":0.5,"calm":0.5}},"tone":{"type":"choice","probabilities":{"angry":0.5,"calm":0.5}}}}"#,
            ),
            (
                "angry",
                r#"{"model":"m","answers":{"tone":{"type":"choice","probabilities":{"angry":0.25,"angry":0.5,"calm":0.5}}}}"#,
            ),
        ];

        for (member, response) in cases {
            let error = decode_response(&set, response).unwrap_err();
            assert!(
                matches!(error, EvalError::Http(ref message) if message.contains(&format!("duplicate JSON object member `{member}`"))),
                "duplicate `{member}` must be rejected during JSON parsing, got {error:?}"
            );
        }

        // A duplicate in the answer tree is diagnosed before a missing model
        // can be accepted/rejected by the provenance extractor.
        let error = decode_response(
            &set,
            r#"{"answers":{"tone":{"type":"choice","probabilities":{"angry":0.5,"angry":0.5,"calm":0.5}}}}"#,
        )
        .unwrap_err();
        assert!(
            matches!(error, EvalError::Http(ref message) if message.contains("duplicate JSON object member `angry`")),
            "the whole response must be checked for duplicates before field extraction, got {error:?}"
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
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        // Renormalization must not reorder or skew: both components ~1/2.
        assert!(probabilities.iter().all(|p| (*p - 0.5).abs() < 1e-3));
    }

    #[test]
    fn producer_valid_wire_distribution_is_not_renormalized_negative() {
        // 48 options at 1/48 plus one exact zero: the f32 sequential sum is
        // 0.9999996 — already inside producer tolerance. Dividing through by
        // that sum (the old behavior) pushed the sum to 1.0000012, and the
        // uniform-share correction then drove the zero component NEGATIVE.
        let mut yaml = String::from("questions:\n  pick:\n    type: choice\n    criteria:\n");
        let mut wire = serde_json::Map::new();
        for i in 0..49 {
            let name = format!("o{i:02}");
            yaml.push_str(&format!("      {name}: ~\n"));
            wire.insert(
                name,
                if i == 0 {
                    serde_json::json!(0.0)
                } else {
                    serde_json::json!(1.0f64 / 48.0)
                },
            );
        }
        let set = author::parse_yaml(&yaml).unwrap();
        let question = set.question("pick").unwrap();
        let parsed = parse_answer(
            question,
            &serde_json::json!({"type": "choice", "probabilities": wire}),
        )
        .unwrap();
        let probabilities = parsed.value.as_ref().unwrap().probabilities();
        assert_eq!(probabilities.len(), 49);
        confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE).unwrap();
        assert!(
            probabilities.iter().all(|p| (0.0..=1.0).contains(p)),
            "no component may leave [0, 1]: {probabilities:?}"
        );
        // The early return leaves the already-valid vector untouched: the
        // zero stays exactly zero.
        assert_eq!(probabilities[0], 0.0);
    }

    #[test]
    // The literals are the review-thread evidence values; f32 rounds them
    // to the exact bit patterns at issue, so keep the full decimals.
    #[allow(clippy::excessive_precision)]
    fn normalization_overage_never_flips_the_argmax() {
        // Rotate the unique winner through every index: merging its level
        // with an earlier runner-up must not change D6's selected label.
        for winner in 0..59 {
            let mut probabilities = vec![0.0167830512f32; 59];
            probabilities[winner] = 0.0167830531f32;
            normalize_distribution(&mut probabilities);
            confidence::check_distribution(&probabilities, confidence::PRODUCER_SUM_TOLERANCE)
                .unwrap();
            assert!(probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
            assert_eq!(
                confidence::argmax_index(&probabilities),
                Some(winner),
                "normalization changed winner {winner}: {probabilities:?}"
            );
        }
    }

    #[test]
    fn hash_distribution_preserves_the_raw_argmax() {
        // The deterministic baseline's distributions must survive
        // normalization with their predicted label intact: applying the
        // whole residual to the maximum (the old approach) could cross the
        // runner-up, and max_by would pick the LAST of an exact tie.
        for cardinality in [2usize, 3, 17, 59, 78] {
            for seed in 0..200u64 {
                let key = format!("argmax-invariant\x00{cardinality}\x00{seed}");
                let draws: Vec<f64> = (0..cardinality as u64)
                    .map(|i| draw(key.as_bytes(), i))
                    .collect();
                let total: f64 = draws.iter().sum();
                let raw: Vec<f32> = draws.iter().map(|d| (d / total) as f32).collect();
                // Raw draws are continuous — ties have measure zero — so a
                // plain max identifies the intended winner.
                let raw_argmax = raw
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index);
                let normalized = hash_distribution(key.as_bytes(), cardinality);
                confidence::check_distribution(&normalized, confidence::PRODUCER_SUM_TOLERANCE)
                    .unwrap();
                assert_eq!(
                    confidence::argmax_index(&normalized),
                    raw_argmax,
                    "cardinality {cardinality} seed {seed}: normalization changed the label"
                );
            }
        }
    }
    #[tokio::test]
    async fn http_rejects_nonfinite_nested_entries_before_connecting() {
        let subject = HttpSubject::new("http://127.0.0.1:1", "model", "token");
        for location in 0..6 {
            for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.25] {
                let mut set = fixture();
                let mut state = Entry::Str("valid state".into());
                let entry = Entry::Seq(vec![Entry::Map(vec![(
                    "nested".into(),
                    Entry::Number(number),
                )])]);
                match location {
                    0 => state = entry,
                    1 => set.questions[0].instructions = Some(entry),
                    2 | 3 => {
                        set.questions[0].body = QuestionBody::Noul {
                            criteria: Some(hyprstream_decision::NoulCriteria {
                                on_false: (location == 2).then(|| entry.clone()),
                                on_true: (location == 3).then_some(entry),
                            }),
                        }
                    }
                    4 => {
                        if let QuestionBody::Choice { options } = &mut set.questions[1].body {
                            options[0].rubric = Some(entry);
                        }
                    }
                    _ => {
                        if let QuestionBody::Score { levels } = &mut set.questions[2].body {
                            levels[0] = Some(entry);
                        }
                    }
                }
                if number.is_finite() {
                    assert!(serde_json::to_vec(&request_body(&set, &state, "model")).is_ok());
                } else {
                    let error = subject.post(&set, &state).await.unwrap_err();
                    assert!(
                        matches!(error, EvalError::InvalidInput(_)),
                        "location {location} must fail input validation before HTTP: {error:?}"
                    );
                }
            }
        }
    }
    #[test]
    fn wire_probability_bounds_are_checked_before_f32_rounding() {
        let set = fixture();
        for probability in [-1e-50f64, 1.00000001, 0.0, 1.0] {
            let valid = (0.0..=1.0).contains(&probability);
            let remainder = (1.0 - probability).clamp(0.0, 1.0);
            let answers = [
                serde_json::json!({"type":"noul", "noul":probability}),
                serde_json::json!({"type":"choice", "probabilities":{"angry":probability,"calm":remainder}}),
                serde_json::json!({"type":"score", "probabilities":{"0":probability,"1":remainder,"2":0.0}}),
            ];
            for (question, answer) in set.questions.iter().zip(answers) {
                assert_eq!(
                    parse_answer(question, &answer).is_ok(),
                    valid,
                    "{} must check original {probability} before narrowing",
                    question.id
                );
            }
        }
    }
}
