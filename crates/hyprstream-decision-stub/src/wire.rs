//! The jev-1 wire envelope: `POST /v1/systemone` request → validated IR.
//!
//! The request body is `{state, model, questions}`. Parsing goes through
//! [`hyprstream_decision::entry::Entry`] so insertion order survives (option order is
//! canonical, D6) and validation is delegated to the P0.1a authoring layer — the facade
//! enforces exactly the profile rules, nothing parallel.
//!
//! Error bodies mirror the FastAPI detail shape (S6a §6.1): 422 carries
//! `{"detail": [{"loc": [...], "msg", "type"}]}`; other error statuses carry
//! `{"detail": "<message>"}` (A10 — non-422 shapes are unspecified upstream).

use hyprstream_decision::author;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::error::SpecError;
use hyprstream_decision::spec::QuestionSet;
use serde::Serialize;

/// A parsed, validated `SystemOneRequest`.
#[derive(Debug, Clone)]
pub struct WireRequest {
    /// The shared state (required, non-null on the wire).
    pub state: Entry,
    /// The requested model string (alias or versioned id). Resolution happens at answer
    /// time — the response echoes the resolved id.
    pub model: String,
    /// The validated question set (state included, questions in canonical order).
    pub set: QuestionSet,
}

/// One FastAPI-style 422 detail item.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetailItem {
    /// Location of the offending element: `["body", <json-pointer>]`.
    pub loc: Vec<String>,
    /// Human-readable rule violation.
    pub msg: String,
    /// Machine-readable category (the P0.1a `SpecErrorKind` code, or `value_error`).
    #[serde(rename = "type")]
    pub error_type: String,
}

/// A facade error: an HTTP status plus a body that matches the documented error shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct WireError {
    /// HTTP status code (401, 422, …).
    pub status: u16,
    /// Serialized body. For 422 this is `{"detail": [DetailItem, …]}`; for anything else
    /// `{"detail": "<msg>"}` (the FastAPI non-validation shape).
    pub body: String,
}

impl WireError {
    /// A FastAPI-style validation error (HTTP 422).
    pub fn unprocessable(items: Vec<DetailItem>) -> Self {
        #[derive(Serialize)]
        struct DetailBody {
            detail: Vec<DetailItem>,
        }
        let body = serde_json::to_string(&DetailBody { detail: items })
            .unwrap_or_else(|_| r#"{"detail":[]}"#.to_owned());
        Self { status: 422, body }
    }

    /// A non-validation error with the `{"detail": "<msg>"}` shape.
    pub fn simple(status: u16, message: impl Into<String>) -> Self {
        #[derive(Serialize)]
        struct SimpleBody {
            detail: String,
        }
        let body = serde_json::to_string(&SimpleBody {
            detail: message.into(),
        })
        .unwrap_or_else(|_| r#"{"detail":"internal error"}"#.to_owned());
        Self { status, body }
    }

    /// One-item 422 at a body-level location.
    pub fn invalid(loc: Vec<String>, msg: impl Into<String>, error_type: &str) -> Self {
        Self::unprocessable(vec![DetailItem {
            loc,
            msg: msg.into(),
            error_type: error_type.to_owned(),
        }])
    }
}

fn body_loc(pointer: &str) -> Vec<String> {
    vec!["body".to_owned(), pointer.to_owned()]
}

impl From<SpecError> for WireError {
    fn from(error: SpecError) -> Self {
        WireError::invalid(
            body_loc(&error.path.to_string()),
            error.message.clone(),
            error.kind.code(),
        )
    }
}

/// Parse and validate a `POST /v1/systemone` request body.
///
/// Field rules: exactly `state` (required, non-null), `model` (required string),
/// `questions` (required map, ≥ 1 entry); unknown fields are rejected (parity with the
/// stock SDKs' `extra=forbid`). Question-level rules (D1–D6 cardinalities, reserved v2
/// types, duplicate keys) come from the P0.1a authoring layer. Bare booleans/numbers in
/// entry positions are accepted — see the crate docs for that profile decision.
pub fn parse_request(body: &[u8]) -> Result<WireRequest, WireError> {
    let root: Entry = serde_json::from_slice(body).map_err(|error| {
        WireError::invalid(
            vec!["body".to_owned()],
            format!("request body is not valid JSON: {error}"),
            "json_invalid",
        )
    })?;
    let Entry::Map(fields) = &root else {
        return Err(WireError::invalid(
            vec!["body".to_owned()],
            "request body must be a JSON object with `state`, `model`, and `questions`",
            "structure",
        ));
    };
    for (index, (key, _)) in fields.iter().enumerate() {
        if fields[..index].iter().any(|(earlier, _)| earlier == key) {
            return Err(WireError::invalid(
                body_loc(&format!("/{key}")),
                format!("duplicate key `{key}`"),
                "duplicate_key",
            ));
        }
    }

    let mut state = None;
    let mut model = None;
    let mut questions = None;
    for (key, value) in fields {
        match key.as_str() {
            "state" => state = Some(value),
            "model" => model = Some(value),
            "questions" => questions = Some(value),
            other => {
                return Err(WireError::invalid(
                    body_loc(&format!("/{other}")),
                    format!("unknown field `{other}`; the envelope accepts only `state`, `model`, and `questions`"),
                    "unknown_field",
                ));
            }
        }
    }

    let state = match state {
        None => {
            return Err(WireError::invalid(
                body_loc("/state"),
                "`state` is required",
                "missing_field",
            ));
        }
        Some(Entry::Null) => {
            return Err(WireError::invalid(
                body_loc("/state"),
                "`state` must not be null",
                "structure",
            ));
        }
        Some(entry) => entry.clone(),
    };
    let model = match model {
        Some(Entry::Str(name)) if !name.is_empty() => name.clone(),
        Some(_) => {
            return Err(WireError::invalid(
                body_loc("/model"),
                "`model` must be a non-empty string (alias or versioned id)",
                "structure",
            ));
        }
        None => {
            return Err(WireError::invalid(
                body_loc("/model"),
                "`model` is required",
                "missing_field",
            ));
        }
    };
    let questions = match questions {
        Some(entry @ Entry::Map(_)) => entry.clone(),
        Some(_) => {
            return Err(WireError::invalid(
                body_loc("/questions"),
                "`questions` must be a mapping of question id to spec",
                "structure",
            ));
        }
        None => {
            return Err(WireError::invalid(
                body_loc("/questions"),
                "`questions` is required",
                "missing_field",
            ));
        }
    };

    // Delegate every question-level rule to the P0.1a authoring layer by re-serializing
    // the (state, questions) pair as an authoring document. `Entry::canonical_text` of a
    // map is compact JSON, and the re-parse preserves insertion order exactly.
    let document = Entry::Map(vec![
        ("state".to_owned(), state.clone()),
        ("questions".to_owned(), questions),
    ]);
    let set = author::parse_json(&document.canonical_text())?;

    Ok(WireRequest { state, model, set })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // panicking is correct in unit tests
mod tests {
    use super::*;

    #[test]
    fn golden_request_parses() {
        let body = br#"{
            "state": "The refund arrived two weeks late and the box was crushed.",
            "model": "jev-latest",
            "questions": {
                "is_refund": {"type": "noul", "instructions": "The customer wants money back."},
                "tone": {"type": "choice", "criteria": {"angry": "Hostile message", "calm": null}},
                "severity": {"type": "score", "criteria": ["cosmetic", "unusable"]}
            }
        }"#;
        let request = parse_request(body).unwrap_or_else(|error| panic!("golden request parses: {error:?}"));
        assert_eq!(request.model, "jev-latest");
        assert_eq!(request.set.questions.len(), 3);
        assert_eq!(request.set.questions[1].id, "tone");
        assert_eq!(request.set.questions[1].labels(), vec!["angry", "calm"]);
    }

    #[test]
    fn bare_bool_and_number_entries_are_accepted() {
        // The P0.7 profile decision (crate docs): superset scalars pass through as IR
        // values, not strings, not errors.
        let body = br#"{
            "state": {"ticket": {"priority": 3, "paid": true}},
            "model": "jev-latest",
            "questions": {
                "q": {"type": "noul", "instructions": {"threshold": 0.75, "strict": false}}
            }
        }"#;
        let request = parse_request(body).unwrap_or_else(|error| panic!("superset scalars parse: {error:?}"));
        let question = &request.set.questions[0];
        assert_eq!(
            question.instructions.as_ref().map(hyprstream_decision::Entry::canonical_text),
            Some(r#"{"threshold":0.75,"strict":false}"#.to_owned()),
            "scalars stay scalars through the IR (no stringification)"
        );
    }

    #[test]
    fn unknown_top_level_field_is_422() {
        let error = parse_request(
            br#"{"state": "x", "model": "m", "questions": {"q": {"type": "noul"}}, "extra": 1}"#,
        )
        .unwrap_err();
        assert_eq!(error.status, 422);
        assert!(error.body.contains("unknown_field"), "{}", error.body);
    }

    #[test]
    fn reserved_and_unknown_types_are_422() {
        for (tag, code) in [("span", "reserved_question_type"), ("bogus", "unknown_question_type")] {
            let body = format!(
                r#"{{"state": "x", "model": "m", "questions": {{"q": {{"type": "{tag}"}}}}}}"#
            );
            let error = parse_request(body.as_bytes()).unwrap_err();
            assert_eq!(error.status, 422);
            assert!(error.body.contains(code), "{tag}: {}", error.body);
        }
    }

    #[test]
    fn profile_cardinalities_enforced() {
        // 256 options exceeds the pinned 255 limit (D2/A4).
        let options = (0..256)
            .map(|i| format!(r#""opt{i}": null"#))
            .collect::<Vec<_>>()
            .join(",");
        let body = format!(
            r#"{{"state": "x", "model": "m", "questions": {{"q": {{"type": "choice", "criteria": {{{options}}}}}}}}}"#
        );
        let error = parse_request(body.as_bytes()).unwrap_err();
        assert_eq!(error.status, 422);
        assert!(error.body.contains("cardinality"), "{}", error.body);
    }
}
