use super::ClefError;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use tokenizers::Tokenizer;

pub const CLEF_ENCODING_VERSION: &str = "cloudflare-clef-8b2e5fd-1";
pub const CLEF_SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClefQuestionType {
    Noul,
    Choice,
    Score,
}
impl ClefQuestionType {
    pub fn type_id(self) -> i64 {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}
#[derive(Clone, Debug, Deserialize)]
pub struct ClefQuestion {
    #[serde(rename = "type")]
    pub kind: ClefQuestionType,
    #[serde(default)]
    pub instructions: Value,
    /// Noul: null or object overriding true/false descriptions; choice: object; score: array.
    #[serde(default)]
    pub criteria: Value,
}
#[derive(Clone, Debug)]
pub struct ClefRecord {
    pub record_id: String,
    pub state: Value,
    /// Preserves author question order; ids are included in the Clef prompt.
    pub questions: Vec<(String, ClefQuestion)>,
}
/// The text/JSON-only systemone request envelope. Deserializes the question map
/// directly, preserving insertion order and rejecting duplicate IDs. In particular,
/// do not first parse this through serde_json::Value (whose maps may be sorted).
/// This Deserialize implementation is JSON-specific. Bound request bytes before
/// parsing: RawValue capture and normalization copy the admitted body. Encoding v1
/// normalizes integer -0 to 0 (Python semantics), preserves floating -0.0, and
/// rejects integers outside i64/u64 rather than silently converting them to f64.
/// Typed constructors and any legacy f64 IR adapter are a distinct normalized
/// input profile, not evidence of original wire JSON numeric fidelity.
#[derive(Clone, Debug)]
pub struct ClefRequest {
    pub model: String,
    pub record: ClefRecord,
}
impl<'de> Deserialize<'de> for ClefRequest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Capture lexical numbers before serde_json::Value can round oversized
        // integers to f64 or reinterpret the Python integer spelling -0.
        let raw = Box::<serde_json::value::RawValue>::deserialize(deserializer)?;
        let normalized = normalize_wire_numbers(raw.get()).map_err(serde::de::Error::custom)?;
        let mut parser = serde_json::Deserializer::from_str(&normalized);
        let request = deserialize_request_body(&mut parser).map_err(serde::de::Error::custom)?;
        parser.end().map_err(serde::de::Error::custom)?;
        Ok(request)
    }
}
// Input was validated as JSON by RawValue. Only JSON numeric tokens outside
// strings are changed. Exponents identify floats and preserve signed zero.
fn normalize_wire_numbers(raw: &str) -> Result<String, &'static str> {
    let bytes = raw.as_bytes();
    let mut result = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            result.extend_from_slice(&bytes[start..i]);
        } else if bytes[i] == b'-' || bytes[i].is_ascii_digit() {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            let token = &raw[start..i];
            if !token.contains(['.', 'e', 'E']) {
                let valid = if token.starts_with('-') {
                    token.parse::<i64>().is_ok()
                } else {
                    token.parse::<u64>().is_ok()
                };
                if !valid {
                    return Err("integer outside supported i64/u64 range; refusing lossy Clef JSON conversion");
                }
                if token == "-0" {
                    result.push(b'0');
                } else {
                    result.extend_from_slice(token.as_bytes());
                }
            } else {
                result.extend_from_slice(token.as_bytes());
            }
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(result).map_err(|_| "invalid UTF-8 after Clef JSON number normalization")
}
fn deserialize_request_body<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<ClefRequest, D::Error> {
    struct RequestVisitor;
    impl<'de> serde::de::Visitor<'de> for RequestVisitor {
        type Value = ClefRequest;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a text/JSON Clef systemone request")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut model = None;
            let mut state = None;
            let mut questions = None;
            let mut id = None;
            let mut seen = std::collections::HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate request field {key}"
                    )));
                }
                match key.as_str() {
                    "model" => model = Some(map.next_value::<String>()?),
                    "state" => state = Some(map.next_value::<Value>()?),
                    "questions" => questions = Some(map.next_value::<OrderedQuestions>()?.0),
                    "id" => id = Some(map.next_value::<String>()?),
                    "images" | "videos" | "audio" | "media_kwargs" => {
                        return Err(serde::de::Error::custom(format!(
                            "unsupported Clef modality field {key}; text/JSON only"
                        )))
                    }
                    _ => {
                        let _ = map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
            }
            let model = model.ok_or_else(|| serde::de::Error::missing_field("model"))?;
            let state = state.ok_or_else(|| serde::de::Error::missing_field("state"))?;
            let questions =
                questions.ok_or_else(|| serde::de::Error::missing_field("questions"))?;
            if questions.is_empty() {
                return Err(serde::de::Error::custom("at least one question required"));
            }
            Ok(ClefRequest {
                model,
                record: ClefRecord {
                    record_id: id.unwrap_or_else(|| "unknown".into()),
                    state,
                    questions,
                },
            })
        }
    }
    deserializer.deserialize_map(RequestVisitor)
}
impl<'de> Deserialize<'de> for ClefRecord {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(ClefRequest::deserialize(deserializer)?.record)
    }
}
struct OrderedQuestions(Vec<(String, ClefQuestion)>);
impl<'de> Deserialize<'de> for OrderedQuestions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct QuestionsVisitor;
        impl<'de> serde::de::Visitor<'de> for QuestionsVisitor {
            type Value = OrderedQuestions;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an ordered question map")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut result = Vec::new();
                let mut seen = std::collections::HashSet::new();
                while let Some((id, q)) = map.next_entry::<String, ClefQuestion>()? {
                    if !seen.insert(id.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate question id {id}"
                        )));
                    }
                    result.push((id, q));
                }
                Ok(OrderedQuestions(result))
            }
        }
        deserializer.deserialize_map(QuestionsVisitor)
    }
}

#[derive(Clone, Debug)]
pub struct EncodedClefQuestion {
    pub question_id: String,
    pub kind: ClefQuestionType,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    /// Model order: noul true/false, choice sorted ids, score increasing indices.
    pub option_ids: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct EncodedClefRecord {
    pub record_id: String,
    pub input_ids: Vec<i64>,
    pub questions: Vec<EncodedClefQuestion>,
    /// State token count before either state cap or total-input truncation.
    pub state_tokens_before_truncation: usize,
    /// State tokens actually present in input_ids (excludes prompt and schema).
    pub state_tokens_admitted: usize,
    /// True exactly when at least one original state token was removed.
    pub state_truncated: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct ClefEncodingLimits {
    pub max_length: usize,
    pub max_state_tokens: Option<usize>,
}
impl Default for ClefEncodingLimits {
    fn default() -> Self {
        Self {
            max_length: 16384,
            max_state_tokens: None,
        }
    }
}

/// Python json.dumps(ensure_ascii=False, separators=(',', ':'), sort_keys=True).
/// String states/instructions are rendered without JSON quotes.
pub fn clef_render(value: &Value) -> String {
    fn json(v: &Value) -> String {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<_> = m.keys().collect();
                keys.sort();
                format!(
                    "{{{}}}",
                    keys.into_iter()
                        .map(|k| format!("{}:{}", Value::String(k.clone()), json(&m[k])))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Value::Array(a) => format!("[{}]", a.iter().map(json).collect::<Vec<_>>().join(",")),
            Value::Number(n) if n.is_f64() => python_float(n.as_f64().unwrap_or(0.0)),
            _ => v.to_string(),
        }
    }
    match value {
        Value::String(s) => s.clone(),
        _ => json(value),
    }
}
/// CPython's shortest round-trip digits use round-to-nearest, ties-to-even.
/// Rust Debug is not that contract (some shortest decimal ties round away).
/// Ryū supplies the correctly selected digits; presentation is separate because
/// its built-in fixed/scientific boundary differs from Python's repr/json.
fn python_float(value: f64) -> String {
    debug_assert!(value.is_finite()); // serde_json::Number admits finite f64 only.
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0.0 {
        return format!("{sign}0.0");
    }
    let mut buffer = ryu::Buffer::new();
    let shortest = buffer.format_finite(value.abs());
    let (mantissa, explicit_exponent) = match shortest.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().unwrap_or(0)),
        None => (shortest, 0),
    };
    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0, |(_, fractional)| fractional.len() as i32);
    let all_digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let significant = all_digits.trim_start_matches('0');
    let exponent = explicit_exponent - fractional_digits + significant.len() as i32 - 1;
    let digits = significant.trim_end_matches('0');
    if !(-4..16).contains(&exponent) {
        let (first, tail) = digits.split_at(1);
        let decimal = if tail.is_empty() {
            String::new()
        } else {
            format!(".{tail}")
        };
        format!("{sign}{first}{decimal}e{exponent:+03}")
    } else if exponent < 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else {
        let point = exponent as usize + 1;
        if point >= digits.len() {
            format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
        } else {
            let (whole, fractional) = digits.split_at(point);
            format!("{sign}{whole}.{fractional}")
        }
    }
}

fn options(q: &ClefQuestion) -> Result<Vec<(String, Value)>, ClefError> {
    match q.kind {
        ClefQuestionType::Noul => {
            if !q.criteria.is_null() && !q.criteria.is_object() {
                return Err(ClefError::Input(
                    "noul criteria must be null or an object".into(),
                ));
            }
            Ok([
                ("true", "The proposition is true or the answer is yes."),
                ("false", "The proposition is false or the answer is no."),
            ]
            .into_iter()
            .map(|(id, default)| {
                (
                    id.into(),
                    q.criteria
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| Value::String(default.into())),
                )
            })
            .collect())
        }
        ClefQuestionType::Choice => {
            let m = q
                .criteria
                .as_object()
                .ok_or_else(|| ClefError::Input("choice criteria must be an object".into()))?;
            let mut result: Vec<_> = m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            result.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(result)
        }
        ClefQuestionType::Score => Ok(q
            .criteria
            .as_array()
            .ok_or_else(|| ClefError::Input("score criteria must be an array".into()))?
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect()),
    }
}

pub struct ClefEncoder;
impl ClefEncoder {
    pub fn encode(
        tokenizer: &Tokenizer,
        record: &ClefRecord,
        limits: ClefEncodingLimits,
    ) -> Result<EncodedClefRecord, ClefError> {
        Self::encode_with(record, limits, |text| {
            tokenizer
                .encode(text, false)
                .map(|v| v.get_ids().iter().map(|&id| i64::from(id)).collect())
                .map_err(|e| ClefError::Tokenizer(e.to_string()))
        })
    }
    fn encode_with<F>(
        record: &ClefRecord,
        limits: ClefEncodingLimits,
        mut tokens: F,
    ) -> Result<EncodedClefRecord, ClefError>
    where
        F: FnMut(&str) -> Result<Vec<i64>, ClefError>,
    {
        if record.questions.is_empty() {
            return Err(ClefError::Input("at least one question required".into()));
        }
        let mut schema = tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (i, (id, q)) in record.questions.iter().enumerate() {
            if !seen.insert(id) {
                return Err(ClefError::Input("duplicate question id".into()));
            }
            schema.extend(tokens(&format!(
                "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
                i + 1,
                id,
                q.kind.name()
            ))?);
            let start = schema.len();
            let instruction = if q.instructions.is_null() || q.instructions.as_str() == Some("") {
                id.clone()
            } else {
                clef_render(&q.instructions)
            };
            schema.extend(tokens(&instruction)?);
            let end = schema.len();
            schema.extend(tokens("\nALLOWED OPTIONS:\n")?);
            let mut spans = Vec::new();
            let mut ids = Vec::new();
            let opts = options(q)?;
            if opts.is_empty() {
                return Err(ClefError::Input("criteria cannot be empty".into()));
            }
            for (j, (option_id, description)) in opts.into_iter().enumerate() {
                schema.extend(tokens(&format!("OPTION {}: ", j + 1))?);
                let a = schema.len();
                let mut semantics = serde_json::Map::new();
                semantics.insert("option_id".into(), Value::String(option_id.clone()));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                schema.extend(tokens(&clef_render(&Value::Object(semantics)))?);
                spans.push((a, schema.len()));
                ids.push(option_id);
                schema.extend(tokens("\n")?);
            }
            schema.extend(tokens("END FIELD\n")?);
            questions.push(EncodedClefQuestion {
                question_id: id.clone(),
                kind: q.kind,
                question_span: (start, end),
                option_spans: spans,
                option_ids: ids,
            });
        }
        let mut prefix = tokens(&format!(
            "<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        let suffix = tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > limits.max_length {
            return Err(ClefError::Input(format!(
                "schema requires {fixed} tokens before state; maximum is {}",
                limits.max_length
            )));
        }
        let mut state = tokens(&clef_render(&record.state))?;
        let state_tokens_before_truncation = state.len();
        state.truncate(
            limits
                .max_state_tokens
                .unwrap_or(usize::MAX)
                .min(limits.max_length - fixed),
        );
        let state_tokens_admitted = state.len();
        let state_truncated = state_tokens_admitted < state_tokens_before_truncation;
        let offset = prefix.len() + state_tokens_admitted;
        for q in &mut questions {
            q.question_span.0 += offset;
            q.question_span.1 += offset;
            for (a, b) in &mut q.option_spans {
                *a += offset;
                *b += offset;
            }
        }
        prefix.extend(state);
        prefix.extend(schema);
        prefix.extend(suffix);
        Ok(EncodedClefRecord {
            record_id: record.record_id.clone(),
            input_ids: prefix,
            questions,
            state_tokens_before_truncation,
            state_tokens_admitted,
            state_truncated,
        })
    }
}

/// Reorders by label, including model noul [true,false] -> contract [false,true].
pub fn clef_label_permutation(
    model_ids: &[String],
    canonical_ids: &[String],
) -> Result<Vec<i64>, ClefError> {
    if model_ids.len() != canonical_ids.len() {
        return Err(ClefError::Input("label cardinality mismatch".into()));
    }
    let mut seen = std::collections::HashSet::new();
    canonical_ids
        .iter()
        .map(|id| {
            if !seen.insert(id) {
                return Err(ClefError::Input("duplicate canonical label".into()));
            }
            let matches: Vec<_> = model_ids
                .iter()
                .enumerate()
                .filter(|(_, v)| *v == id)
                .collect();
            if matches.len() != 1 {
                Err(ClefError::Input(format!(
                    "missing/duplicate model label {id}"
                )))
            } else {
                Ok(matches[0].0 as i64)
            }
        })
        .collect()
}
#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    #[test]
    fn clef_wire_order_numeric_types_and_duplicate_ids() {
        let json = r#"{"model":"clef","state":{"integer":1,"float":1.0},"questions":{"z":{"type":"noul"},"a":{"type":"choice","criteria":{"z":null,"a":"first"}}}}"#;
        let request: ClefRequest = serde_json::from_str(json).expect("request");
        assert_eq!(request.model, "clef");
        assert_eq!(
            request
                .record
                .questions
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
        assert_eq!(
            clef_render(&request.record.state),
            r#"{"float":1.0,"integer":1}"#
        );
        assert!(serde_json::from_str::<ClefRequest>(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ClefRequest>(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}},"images":[]}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ClefRequest>(
            r#"{"model":"m","questions":{"q":{"type":"noul"}}}"#
        )
        .is_err());
    }
    #[test]
    fn clef_wire_numeric_domain_and_escaping() {
        let parse = |state: &str| {
            serde_json::from_str::<ClefRequest>(&format!(
                r#"{{"model":"m","state":{state},"questions":{{"q":{{"type":"noul"}}}}}}"#
            ))
        };
        let r = parse(r#"[-0,-0.0,-0e0,0,18446744073709551615,-9223372036854775808,1e20,1e-7]"#)
            .expect("numeric domain");
        assert_eq!(
            clef_render(&r.record.state),
            "[0,-0.0,-0.0,0,18446744073709551615,-9223372036854775808,1e+20,1e-07]"
        );
        assert!(parse("18446744073709551616").is_err());
        assert!(parse("-9223372036854775809").is_err());
        assert!(parse(r#"{"nested":[1,{"n":18446744073709551616}]}"#).is_err());
        let escaped = r#"{"text":"\"-0 18446744073709551616\\1e20","unicode":"数-0","actual":-0}"#;
        let r = parse(escaped).expect("quoted numeric text untouched");
        assert_eq!(
            r.record.state["text"].as_str(),
            Some("\"-0 18446744073709551616\\1e20")
        );
        assert_eq!(r.record.state["unicode"].as_str(), Some("数-0"));
        assert_eq!(r.record.state["actual"], serde_json::json!(0));
    }
    #[test]
    fn clef_encoding_spans_order_truncation() {
        let record = ClefRecord {
            record_id: "r".into(),
            state: Value::String("a state".into()),
            questions: vec![(
                "κ".into(),
                ClefQuestion {
                    kind: ClefQuestionType::Choice,
                    instructions: Value::Null,
                    criteria: serde_json::json!({"z":null,"a":{"β":2,"a":1}}),
                },
            )],
        };
        let encode = |r: &ClefRecord, l| {
            ClefEncoder::encode_with(r, l, |s| {
                Ok(s.as_bytes().iter().map(|&b| i64::from(b)).collect())
            })
        };
        let encoded = encode(&record, ClefEncodingLimits::default()).expect("encode");
        let q = &encoded.questions[0];
        assert_eq!(q.option_ids, ["a", "z"]);
        assert_eq!(
            &encoded.input_ids[q.question_span.0..q.question_span.1],
            "κ".as_bytes()
                .iter()
                .map(|&b| i64::from(b))
                .collect::<Vec<_>>()
        );
        let option = String::from_utf8(
            encoded.input_ids[q.option_spans[0].0..q.option_spans[0].1]
                .iter()
                .map(|&v| v as u8)
                .collect(),
        )
        .expect("utf8");
        assert_eq!(
            option,
            "{\"description\":{\"a\":1,\"β\":2},\"option_id\":\"a\"}"
        );
        let truncated = encode(
            &record,
            ClefEncodingLimits {
                max_length: encoded.input_ids.len() - 3,
                max_state_tokens: None,
            },
        )
        .expect("truncate");
        assert_eq!(truncated.input_ids.len(), encoded.input_ids.len() - 3);
        assert!(encode(
            &record,
            ClefEncodingLimits {
                max_length: 1,
                max_state_tokens: None
            }
        )
        .is_err());
    }
    #[test]
    fn clef_state_token_accounting_uses_admitted_encoding() {
        let state = "stateκ数";
        let calls = std::cell::Cell::new(0);
        let record = ClefRecord {
            record_id: "accounting".into(),
            state: Value::String(state.into()),
            questions: vec![(
                "q".into(),
                ClefQuestion {
                    kind: ClefQuestionType::Noul,
                    instructions: Value::Null,
                    criteria: Value::Null,
                },
            )],
        };
        let encode = |limits| {
            ClefEncoder::encode_with(&record, limits, |text| {
                if text == state {
                    calls.set(calls.get() + 1);
                }
                Ok(text.bytes().map(i64::from).collect())
            })
            .expect("encoding")
        };
        let full = encode(ClefEncodingLimits::default());
        assert_eq!(calls.get(), 1);
        assert_eq!(full.state_tokens_before_truncation, state.len());
        assert_eq!(full.state_tokens_admitted, state.len());
        assert!(!full.state_truncated);
        let capped = encode(ClefEncodingLimits {
            max_length: full.input_ids.len(),
            max_state_tokens: Some(4),
        });
        assert_eq!(calls.get(), 2);
        assert_eq!(capped.state_tokens_before_truncation, state.len());
        assert_eq!(capped.state_tokens_admitted, 4);
        assert!(capped.state_truncated);
        assert_eq!(
            full.input_ids.len() - capped.input_ids.len(),
            state.len() - 4
        );
        let budget = encode(ClefEncodingLimits {
            max_length: full.input_ids.len() - (state.len() - 3),
            max_state_tokens: None,
        });
        assert_eq!(calls.get(), 3);
        assert_eq!(budget.state_tokens_admitted, 3);
        assert!(budget.state_truncated);
        let both = encode(ClefEncodingLimits {
            max_length: budget.input_ids.len(),
            max_state_tokens: Some(2),
        });
        assert_eq!(calls.get(), 4);
        assert_eq!(both.state_tokens_admitted, 2);
        assert_eq!(both.state_tokens_before_truncation, state.len());
        let zero = encode(ClefEncodingLimits {
            max_length: full.input_ids.len(),
            max_state_tokens: Some(0),
        });
        assert_eq!(calls.get(), 5);
        assert_eq!(zero.state_tokens_admitted, 0);
        assert!(zero.state_truncated);
        assert_eq!(
            full.questions[0].question_span.0 - zero.questions[0].question_span.0,
            state.len()
        );
        let empty = ClefRecord {
            state: Value::String(String::new()),
            ..record
        };
        let empty = ClefEncoder::encode_with(&empty, ClefEncodingLimits::default(), |text| {
            Ok(text.bytes().map(i64::from).collect())
        })
        .expect("empty state");
        assert_eq!(empty.state_tokens_before_truncation, 0);
        assert_eq!(empty.state_tokens_admitted, 0);
        assert!(!empty.state_truncated);
    }
    #[test]
    #[ignore = "requires CLEF_TOKENIZER_JSON pointing to the pinned release tokenizer"]
    fn clef_pinned_tokenizer_contract() {
        let path = std::env::var("CLEF_TOKENIZER_JSON").expect("pinned tokenizer path");
        let tokenizer = Tokenizer::from_file(path).expect("release tokenizer");
        let record = ClefRecord {
            record_id: "real-tokenizer".into(),
            state: serde_json::json!({"z":null,"a":"Unicode κ 数"}),
            questions: vec![
                (
                    "z".into(),
                    ClefQuestion {
                        kind: ClefQuestionType::Choice,
                        instructions: Value::String("Select α".into()),
                        criteria: serde_json::json!({"z":null,"a":"first"}),
                    },
                ),
                (
                    "a".into(),
                    ClefQuestion {
                        kind: ClefQuestionType::Noul,
                        instructions: Value::Null,
                        criteria: Value::Null,
                    },
                ),
            ],
        };
        let encoded = ClefEncoder::encode(&tokenizer, &record, ClefEncodingLimits::default())
            .expect("encode");
        assert_eq!(
            encoded.input_ids[0],
            i64::from(tokenizer.token_to_id("<|im_start|>").expect("im_start"))
        );
        let q = &encoded.questions[0];
        let token_ids = |text: &str| {
            tokenizer
                .encode(text, false)
                .expect("tokens")
                .get_ids()
                .iter()
                .map(|&v| i64::from(v))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            encoded.input_ids[q.question_span.0..q.question_span.1],
            token_ids("Select α")
        );
        assert_eq!(
            encoded.input_ids[q.option_spans[0].0..q.option_spans[0].1],
            token_ids("{\"description\":\"first\",\"option_id\":\"a\"}")
        );
        assert_eq!(
            encoded.input_ids[q.option_spans[1].0..q.option_spans[1].1],
            token_ids("{\"option_id\":\"z\"}")
        );
        let last = &encoded.questions[1];
        assert_eq!(
            encoded.input_ids[last.question_span.0..last.question_span.1],
            token_ids("a")
        );
        assert_eq!(last.option_ids, ["true", "false"]);
        let suffix = token_ids(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        );
        assert!(encoded.input_ids.ends_with(&suffix));
    }
    #[test]
    fn clef_question_order_noul_defaults_and_score_indices() {
        let record = ClefRecord {
            record_id: "id".into(),
            state: Value::Null,
            questions: vec![
                (
                    "z".into(),
                    ClefQuestion {
                        kind: ClefQuestionType::Noul,
                        instructions: Value::String("".into()),
                        criteria: serde_json::json!({"false":null}),
                    },
                ),
                (
                    "a".into(),
                    ClefQuestion {
                        kind: ClefQuestionType::Score,
                        instructions: serde_json::json!({"query":"数"}),
                        criteria: Value::Array(vec![Value::Null; 11]),
                    },
                ),
            ],
        };
        let encoded = ClefEncoder::encode_with(&record, ClefEncodingLimits::default(), |s| {
            Ok(s.bytes().map(i64::from).collect())
        })
        .expect("encode");
        assert_eq!(
            encoded
                .questions
                .iter()
                .map(|q| q.question_id.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
        assert_eq!(encoded.questions[0].option_ids, ["true", "false"]);
        assert_eq!(encoded.questions[1].option_ids[10], "10");
        let (a, b) = encoded.questions[0].option_spans[1];
        let rendered =
            String::from_utf8(encoded.input_ids[a..b].iter().map(|&v| v as u8).collect())
                .expect("string");
        assert_eq!(rendered, "{\"option_id\":\"false\"}");
        let text =
            String::from_utf8(encoded.input_ids.iter().map(|&v| v as u8).collect()).expect("text");
        assert!(text.contains("STATE:\nnull\n\nSCHEMA FIELDS:"));
        assert!(text.contains("The proposition is true or the answer is yes."));
        assert!(text.ends_with("JOINT SCHEMA DECISIONS:"));
    }
    #[test]
    fn clef_json_float_spelling() {
        assert_eq!(
            clef_render(&serde_json::json!([
                1e-7, 1e16, 0.0001, 1.0, -0.0, 0.00001, 1e15
            ])),
            "[1e-07,1e+16,0.0001,1.0,-0.0,1e-05,1000000000000000.0]"
        );
        assert_eq!(clef_render(&Value::String("raw\ntext".into())), "raw\ntext");
    }
    const PYTHON_FLOAT_GOLDENS: &str =
        include_str!("../../tests/fixtures/python-json-float-v1/expected.tsv");
    #[test]
    fn clef_float_bits_match_cpython_shortest_corpus() {
        for line in PYTHON_FLOAT_GOLDENS.lines() {
            let (bits, expected) = line.split_once('\t').expect("fixture columns");
            let bits = u64::from_str_radix(bits, 16).expect("fixture bits");
            let value = f64::from_bits(bits);
            let number = serde_json::Number::from_f64(value).expect("finite fixture");
            assert_eq!(
                clef_render(&Value::Number(number)),
                expected,
                "bits {bits:016x}"
            );
        }
    }
    #[test]
    fn clef_float_wire_parse_bits_and_render_match_cpython_corpus() {
        let mut failures = Vec::new();
        let mut count = 0;
        for line in PYTHON_FLOAT_GOLDENS.lines() {
            let (bits, expected) = line.split_once('\t').expect("fixture columns");
            let bits = u64::from_str_radix(bits, 16).expect("fixture bits");
            let body = format!(
                r#"{{"model":"m","state":{expected},"questions":{{"q":{{"type":"noul"}}}}}}"#
            );
            let request = serde_json::from_str::<ClefRequest>(&body);
            let problem = match request {
                Ok(request) => {
                    let actual = request.record.state.as_f64().expect("float");
                    let rendered = clef_render(&request.record.state);
                    (actual.to_bits() != bits || rendered != expected).then(|| {
                        format!(
                            "{expected}: expected {bits:016x}, parsed {:016x}, rendered {rendered}",
                            actual.to_bits()
                        )
                    })
                }
                Err(error) => Some(format!("{expected}: {error}")),
            };
            if let Some(problem) = problem {
                count += 1;
                if failures.len() < 12 {
                    failures.push(problem);
                }
            }
        }
        assert_eq!(
            count, 0,
            "wire numeric parity failures (first12): {failures:?}"
        );
        for nonfinite in ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"] {
            let body = format!(
                r#"{{"model":"m","state":{nonfinite},"questions":{{"q":{{"type":"noul"}}}}}}"#
            );
            assert!(
                serde_json::from_str::<ClefRequest>(&body).is_err(),
                "{nonfinite}"
            );
        }
    }

    #[test]
    fn clef_noul_label_mapping() {
        assert_eq!(
            clef_label_permutation(
                &["true".into(), "false".into()],
                &["false".into(), "true".into()]
            )
            .expect("map"),
            [1, 0]
        );
        assert!(
            clef_label_permutation(&["x".into(), "x".into()], &["x".into(), "y".into()]).is_err()
        );
    }
}
