//! Structured, positioned errors for question-spec authoring.
//!
//! Every error carries a machine-readable [`SpecErrorKind`], a document path
//! ([`ErrorPath`], rendered JSON-pointer-style, e.g. `/questions/tone/criteria/2`), and
//! for text-syntax errors a 1-based line/column. Authors of question specs need no Rust,
//! capnp, or Arrow knowledge: the path points at the offending element of their YAML/JSON
//! document and the message says what rule was violated.

use std::fmt;

/// Machine-readable category of an authoring/validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecErrorKind {
    /// The document text is not valid YAML/JSON. Carries line/column when known.
    Syntax,
    /// A document element has the wrong shape (e.g. a sequence where a mapping was
    /// required).
    Structure,
    /// A required key is absent (e.g. `type`, or `criteria` on a choice/score question).
    MissingField,
    /// A key the profile does not define was present. Unknown fields are rejected
    /// (parity with the upstream SDKs' `extra=forbid` behavior).
    UnknownField,
    /// `type` names no known question primitive.
    UnknownQuestionType,
    /// `type` names a reserved v2 primitive (`span`, `derived`) that v1 does not
    /// implement. This is deliberately distinct from `UnknownQuestionType` so callers can
    /// report "recognized but reserved" instead of "typo".
    ReservedQuestionType,
    /// A cardinality rule was violated: choice needs 2–255 options (D2), score needs ≥ 2
    /// levels (D1), a request needs ≥ 1 question.
    Cardinality,
    /// A required string is empty (question id, option name).
    EmptyName,
    /// The same key appears twice in one mapping (question id or option name).
    DuplicateKey,
}

impl SpecErrorKind {
    /// Stable machine-readable code (snake_case), suitable for API error payloads.
    pub fn code(self) -> &'static str {
        match self {
            Self::Syntax => "syntax",
            Self::Structure => "structure",
            Self::MissingField => "missing_field",
            Self::UnknownField => "unknown_field",
            Self::UnknownQuestionType => "unknown_question_type",
            Self::ReservedQuestionType => "reserved_question_type",
            Self::Cardinality => "cardinality",
            Self::EmptyName => "empty_name",
            Self::DuplicateKey => "duplicate_key",
        }
    }
}

/// A position inside an authored document: a sequence of mapping keys and array indices.
///
/// Renders as a JSON pointer (`/questions/tone/criteria/2`). The empty path (`/`) refers
/// to the document root.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ErrorPath {
    segments: Vec<String>,
}

impl ErrorPath {
    /// The document root.
    pub fn root() -> Self {
        Self::default()
    }

    /// Append a mapping-key segment.
    pub fn key(&self, key: &str) -> Self {
        let mut segments = self.segments.clone();
        segments.push(key.to_owned());
        Self { segments }
    }

    /// Append an array-index segment.
    pub fn index(&self, index: usize) -> Self {
        let mut segments = self.segments.clone();
        segments.push(index.to_string());
        Self { segments }
    }
}

impl fmt::Display for ErrorPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.segments.is_empty() {
            return f.write_str("/");
        }
        for segment in &self.segments {
            // JSON-pointer escaping (RFC 6901) so keys containing '/' or '~' stay exact.
            f.write_str("/")?;
            f.write_str(&segment.replace('~', "~0").replace('/', "~1"))?;
        }
        Ok(())
    }
}

/// A single positioned authoring error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecError {
    /// Machine-readable category.
    pub kind: SpecErrorKind,
    /// Document position of the offending element.
    pub path: ErrorPath,
    /// 1-based line of a text-syntax error, when the parser reported one.
    pub line: Option<usize>,
    /// 1-based column of a text-syntax error, when the parser reported one.
    pub column: Option<usize>,
    /// Human-readable explanation of the violated rule.
    pub message: String,
}

impl SpecError {
    pub(crate) fn new(kind: SpecErrorKind, path: ErrorPath, message: impl Into<String>) -> Self {
        Self {
            kind,
            path,
            line: None,
            column: None,
            message: message.into(),
        }
    }

    pub(crate) fn syntax(
        message: impl Into<String>,
        line: Option<usize>,
        column: Option<usize>,
    ) -> Self {
        Self {
            kind: SpecErrorKind::Syntax,
            path: ErrorPath::root(),
            line,
            column,
            message: message.into(),
        }
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.kind.code(), self.path)?;
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(f, " (line {line}, column {column})")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for SpecError {}
