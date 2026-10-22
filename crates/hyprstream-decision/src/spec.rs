//! The jev-1 question-spec intermediate representation.
//!
//! The IR is what every downstream node consumes: P0.1b emits capnp from it, P1.1
//! linearizes it (via [`crate::serialize`]), P3.5 pins it in prepared statements. It is
//! authored through [`crate::author`] (YAML/JSON) or constructed directly in Rust.

use std::fmt;

use crate::entry::Entry;

/// Question type tags. `Span`/`Derived` are reserved for profile v2: recognized by the
/// authoring layer (rejected with [`crate::error::SpecErrorKind::ReservedQuestionType`])
/// so that a v2 document fails loudly instead of being misread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    /// Statement judged against the state; answered with P(true).
    Noul,
    /// One of 2–255 pre-enumerated options; answered with a probability per option.
    Choice,
    /// Ordered rubric levels; answered with a probability per level.
    Score,
    /// Reserved for profile v2 (span extraction).
    Span,
    /// Reserved for profile v2 (derived/computed questions).
    Derived,
}

impl QuestionKind {
    /// The wire tag (`"noul"`, `"choice"`, `"score"`, `"span"`, `"derived"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Span => "span",
            Self::Derived => "derived",
        }
    }

    /// Parse a wire tag. Unknown tags return `None`; reserved v2 tags parse successfully
    /// and are rejected by the authoring layer as `ReservedQuestionType`.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "noul" => Some(Self::Noul),
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "span" => Some(Self::Span),
            "derived" => Some(Self::Derived),
            _ => None,
        }
    }

    /// Whether this tag is usable in profile v1 (reserved v2 tags are not).
    pub fn is_v1(self) -> bool {
        matches!(self, Self::Noul | Self::Choice | Self::Score)
    }
}

impl fmt::Display for QuestionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Noul criteria: free-form descriptions of what the `true` and `false` outcomes mean.
/// Keys are literally `"true"`/`"false"` on the wire; both entries are optional.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoulCriteria {
    /// Description of the true outcome.
    pub on_true: Option<Entry>,
    /// Description of the false outcome.
    pub on_false: Option<Entry>,
}

/// One choice option: a name (its identity and its label) plus an optional rubric.
/// `None` rubric = undescribed option (D3's choice-side counterpart).
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceOption {
    /// Option name. Sent to the model; also the emitted label string.
    pub name: String,
    /// Rubric text (any entry type; structured entries are serialized as canonical JSON).
    pub rubric: Option<Entry>,
}

/// The per-kind payload of a question spec.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionBody {
    /// Noul: optional true/false criteria.
    Noul {
        /// Optional descriptions of the two outcomes.
        criteria: Option<NoulCriteria>,
    },
    /// Choice: 2–255 options (D2) in canonical authoring order.
    Choice {
        /// Options in insertion order — this order is canonical (D6).
        options: Vec<ChoiceOption>,
    },
    /// Score: ≥ 2 ordered levels (D1); index = level, 0-based. `None` = undescribed (D3).
    Score {
        /// Level rubrics in ascending level order. Level *numbers* are never sent to the
        /// model — each level is judged independently against the state.
        levels: Vec<Option<Entry>>,
    },
}

/// One runtime-declared question.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionSpec {
    /// Caller-chosen id. Opaque on the wire and **never serialized to the model**. For
    /// Arrow emission it must additionally be identifier-safe (see
    /// [`crate::arrow::DecisionSchema`]).
    pub id: String,
    /// The primitive type.
    pub kind: QuestionKind,
    /// Optional instructions (D4): the question text in practice.
    pub instructions: Option<Entry>,
    /// The per-kind payload.
    pub body: QuestionBody,
}

impl QuestionSpec {
    /// Number of labels in this question's distribution: 2 for noul
    /// (`[false, true]`), option count for choice, level count for score.
    pub fn cardinality(&self) -> usize {
        match &self.body {
            QuestionBody::Noul { .. } => 2,
            QuestionBody::Choice { options } => options.len(),
            QuestionBody::Score { levels } => levels.len(),
        }
    }

    /// The label list, in canonical order. Noul labels are `["false", "true"]` (index 1
    /// is the `noul` probability itself); choice labels are option names; score labels
    /// are the decimal level indices `"0".."n-1"`.
    pub fn labels(&self) -> Vec<String> {
        match &self.body {
            QuestionBody::Noul { .. } => vec!["false".to_owned(), "true".to_owned()],
            QuestionBody::Choice { options } => {
                options.iter().map(|option| option.name.clone()).collect()
            }
            QuestionBody::Score { levels } => {
                (0..levels.len()).map(|level| level.to_string()).collect()
            }
        }
    }
}

/// A parsed question-spec document: the questions map plus the optional shared `state`.
///
/// Insertion order is authoring order and is preserved end to end — it defines
/// serialization order and argmax tie-breaking (D6). Question ids are unique (the
/// authoring layer rejects duplicates with a positioned error).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuestionSet {
    /// Optional shared state the questions refer to.
    pub state: Option<Entry>,
    /// The questions, in authoring order.
    pub questions: Vec<QuestionSpec>,
}

impl QuestionSet {
    /// Look up a question by id.
    pub fn question(&self, id: &str) -> Option<&QuestionSpec> {
        self.questions.iter().find(|question| question.id == id)
    }
}
