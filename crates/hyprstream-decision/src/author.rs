//! Language-agnostic authoring: YAML/JSON question-spec documents → IR.
//!
//! The document format is the jev-1 questions map with an optional shared state, in our
//! own prose (clean-room, S6a §3):
//!
//! ```yaml
//! state: "The refund arrived two weeks late and the box was crushed."
//! questions:
//!   is_refund:                    # caller-chosen id — never sent to the model
//!     type: noul
//!     instructions: "The customer is asking for a refund."
//!     criteria:                   # optional; keys are literally "true" / "false"
//!       "true": "Explicitly requests money back"
//!       "false": ~                # null = undescribed
//!   tone:
//!     type: choice
//!     instructions: "Classify the tone of the review."
//!     criteria:                   # option name -> rubric (null = undescribed option)
//!       positive: "The review expresses approval."
//!       negative: "The review expresses disapproval."
//!       neutral: ~
//!   severity:
//!     type: score
//!     criteria:                   # ordered levels, 0-indexed; null = undescribed level
//!       - "Cosmetic issue only"
//!       - "Usable but damaged"
//!       - "Completely unusable"
//! ```
//!
//! Validation produces structured, positioned [`SpecError`]s — the first violation is
//! returned with a document path (and line/column for syntax errors). The enforced rules:
//!
//! - top level: a mapping with optional `state` and required `questions`; unknown keys
//!   rejected (D-parity with the upstream SDKs' `extra=forbid`).
//! - `questions`: ≥ 1 entry; ids non-empty and unique.
//! - every question: `type` required; `instructions` optional and nullable (D4); unknown
//!   fields rejected.
//! - `noul`: `criteria` optional; only keys `"true"`/`"false"` allowed.
//! - `choice`: `criteria` required; 2–255 options (D2); option names non-empty, unique.
//! - `score`: `criteria` required; ≥ 2 levels (D1); null levels allowed (D3).
//! - `span` / `derived` are recognized and rejected as reserved v2 types.
//!
//! There is intentionally **no limit on questions per document** (S6a A12 footnote: the
//! upstream profile publishes only token budgets). The serialized-size contract is the 2k
//! token budget documented on [`crate::serialize`].

use crate::entry::Entry;
use crate::error::{ErrorPath, SpecError, SpecErrorKind};
use crate::spec::{
    ChoiceOption, NoulCriteria, QuestionBody, QuestionKind, QuestionSet, QuestionSpec,
};

/// The profile-pinned maximum number of options on a choice question (S6a A4: the 255
/// limit is prose-documented upstream; this profile pins it as a hard validation).
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// Parse a YAML question-spec document into the IR.
pub fn parse_yaml(text: &str) -> Result<QuestionSet, SpecError> {
    let entry: Entry = serde_yaml::from_str(text).map_err(|error| {
        let (line, column) = error
            .location()
            .map(|location| (Some(location.line()), Some(location.column())))
            .unwrap_or((None, None));
        SpecError::syntax(error.to_string(), line, column)
    })?;
    build_question_set(&entry)
}

/// Parse a JSON question-spec document into the IR.
pub fn parse_json(text: &str) -> Result<QuestionSet, SpecError> {
    let entry: Entry = serde_json::from_str(text).map_err(|error| {
        SpecError::syntax(error.to_string(), Some(error.line()), Some(error.column()))
    })?;
    build_question_set(&entry)
}

fn build_question_set(root: &Entry) -> Result<QuestionSet, SpecError> {
    let root_path = ErrorPath::root();
    let map = expect_map(
        root,
        &root_path,
        "the document must be a mapping with a `questions` key",
    )?;

    reject_duplicates(map, &root_path)?;
    let mut state = None;
    let mut questions = None;
    for (key, value) in map {
        match key.as_str() {
            "state" => state = Some(value.clone()),
            "questions" => questions = Some(value),
            other => {
                return Err(SpecError::new(
                    SpecErrorKind::UnknownField,
                    root_path.key(other),
                    format!("unknown top-level field `{other}`; the document accepts only `state` and `questions`"),
                ));
            }
        }
    }

    let questions_value = questions.ok_or_else(|| {
        SpecError::new(
            SpecErrorKind::MissingField,
            root_path.key("questions"),
            "the document must declare a `questions` mapping",
        )
    })?;
    let questions_path = root_path.key("questions");
    let questions_map = expect_map(
        questions_value,
        &questions_path,
        "`questions` must be a mapping of question id to spec",
    )?;
    if questions_map.is_empty() {
        return Err(SpecError::new(
            SpecErrorKind::Cardinality,
            questions_path.clone(),
            "`questions` must declare at least one question",
        ));
    }
    reject_duplicates(questions_map, &questions_path)?;

    let mut parsed = Vec::with_capacity(questions_map.len());
    for (id, value) in questions_map {
        if id.is_empty() {
            return Err(SpecError::new(
                SpecErrorKind::EmptyName,
                questions_path.clone(),
                "question ids must be non-empty",
            ));
        }
        parsed.push(build_question(id, value, &questions_path.key(id))?);
    }

    Ok(QuestionSet {
        state,
        questions: parsed,
    })
}

fn build_question(id: &str, value: &Entry, path: &ErrorPath) -> Result<QuestionSpec, SpecError> {
    let map = expect_map(
        value,
        path,
        "a question spec must be a mapping with `type` and (for choice/score) `criteria`",
    )?;
    reject_duplicates(map, path)?;

    let mut type_tag = None;
    let mut instructions = None;
    let mut criteria = None;
    for (key, field) in map {
        match key.as_str() {
            "type" => type_tag = Some(field),
            "instructions" => instructions = Some(field),
            "criteria" => criteria = Some(field),
            other => {
                return Err(SpecError::new(
                    SpecErrorKind::UnknownField,
                    path.key(other),
                    format!(
                        "unknown field `{other}`; question specs accept only `type`, `instructions`, and `criteria`"
                    ),
                ));
            }
        }
    }

    let type_path = path.key("type");
    let type_value = type_tag.ok_or_else(|| {
        SpecError::new(
            SpecErrorKind::MissingField,
            type_path.clone(),
            "every question requires a `type` (`noul`, `choice`, or `score`)",
        )
    })?;
    let tag = match type_value {
        Entry::Str(tag) => tag.as_str(),
        _ => {
            return Err(SpecError::new(
                SpecErrorKind::Structure,
                type_path,
                "`type` must be a string (`noul`, `choice`, or `score`)",
            ));
        }
    };
    let kind = match QuestionKind::from_tag(tag) {
        Some(kind) if kind.is_v1() => kind,
        Some(reserved) => {
            return Err(SpecError::new(
                SpecErrorKind::ReservedQuestionType,
                type_path,
                format!(
                    "`{}` is reserved for profile v2 and is not implemented by schema v1",
                    reserved.as_str()
                ),
            ));
        }
        None => {
            return Err(SpecError::new(
                SpecErrorKind::UnknownQuestionType,
                type_path,
                format!("unknown question type `{tag}`; expected `noul`, `choice`, or `score`"),
            ));
        }
    };

    // D4: instructions optional and nullable on every primitive.
    let instructions = instructions.and_then(|entry| match entry {
        Entry::Null => None,
        other => Some(other.clone()),
    });

    let criteria_path = path.key("criteria");
    let body = match kind {
        QuestionKind::Noul => QuestionBody::Noul {
            criteria: build_noul_criteria(criteria, &criteria_path)?,
        },
        QuestionKind::Choice => QuestionBody::Choice {
            options: build_choice_options(criteria, &criteria_path)?,
        },
        QuestionKind::Score => QuestionBody::Score {
            levels: build_score_levels(criteria, &criteria_path)?,
        },
        QuestionKind::Span | QuestionKind::Derived => unreachable!("reserved kinds rejected above"),
    };

    Ok(QuestionSpec {
        id: id.to_owned(),
        kind,
        instructions,
        body,
    })
}

fn build_noul_criteria(
    criteria: Option<&Entry>,
    path: &ErrorPath,
) -> Result<Option<NoulCriteria>, SpecError> {
    let Some(entry) = criteria else {
        return Ok(None);
    };
    if matches!(entry, Entry::Null) {
        return Ok(None);
    }
    let map = expect_map(
        entry,
        path,
        "noul `criteria` must be a mapping with optional `true`/`false` keys",
    )?;
    reject_duplicates(map, path)?;
    let mut parsed = NoulCriteria::default();
    for (key, value) in map {
        let entry = match value {
            Entry::Null => None,
            other => Some(other.clone()),
        };
        match key.as_str() {
            "true" => parsed.on_true = entry,
            "false" => parsed.on_false = entry,
            other => {
                return Err(SpecError::new(
                    SpecErrorKind::UnknownField,
                    path.key(other),
                    format!("noul criteria accept only the keys `true` and `false`, got `{other}`"),
                ));
            }
        }
    }
    Ok(Some(parsed))
}

fn build_choice_options(
    criteria: Option<&Entry>,
    path: &ErrorPath,
) -> Result<Vec<ChoiceOption>, SpecError> {
    let entry = criteria.ok_or_else(|| {
        SpecError::new(
            SpecErrorKind::MissingField,
            path.clone(),
            "choice questions require a `criteria` mapping of option name to rubric",
        )
    })?;
    let map = expect_map(
        entry,
        path,
        "choice `criteria` must be a mapping of option name to rubric",
    )?;
    // D2: 2..=255 options.
    if map.len() < 2 {
        return Err(SpecError::new(
            SpecErrorKind::Cardinality,
            path.clone(),
            format!(
                "choice questions need at least 2 options, got {}",
                map.len()
            ),
        ));
    }
    if map.len() > MAX_CHOICE_OPTIONS {
        return Err(SpecError::new(
            SpecErrorKind::Cardinality,
            path.clone(),
            format!(
                "choice questions support at most {MAX_CHOICE_OPTIONS} options, got {}; \
                 above the limit, use the two-stage score-then-choice recipe",
                map.len()
            ),
        ));
    }
    reject_duplicates(map, path)?;
    let mut options = Vec::with_capacity(map.len());
    for (name, rubric) in map {
        if name.is_empty() {
            return Err(SpecError::new(
                SpecErrorKind::EmptyName,
                path.clone(),
                "choice option names must be non-empty",
            ));
        }
        options.push(ChoiceOption {
            name: name.clone(),
            rubric: match rubric {
                Entry::Null => None,
                other => Some(other.clone()),
            },
        });
    }
    Ok(options)
}

fn build_score_levels(
    criteria: Option<&Entry>,
    path: &ErrorPath,
) -> Result<Vec<Option<Entry>>, SpecError> {
    let entry = criteria.ok_or_else(|| {
        SpecError::new(
            SpecErrorKind::MissingField,
            path.clone(),
            "score questions require a `criteria` array of ordered level rubrics",
        )
    })?;
    let Entry::Seq(items) = entry else {
        return Err(SpecError::new(
            SpecErrorKind::Structure,
            path.clone(),
            "score `criteria` must be an array of ordered level rubrics (index = level, 0-based)",
        ));
    };
    // D1: >= 2 levels. (The docs' "up to 10" is a recommendation only, S6a A3.)
    if items.len() < 2 {
        return Err(SpecError::new(
            SpecErrorKind::Cardinality,
            path.clone(),
            format!(
                "score questions need at least 2 levels, got {}",
                items.len()
            ),
        ));
    }
    Ok(items
        .iter()
        .map(|item| match item {
            Entry::Null => None,
            other => Some(other.clone()),
        })
        .collect())
}

fn expect_map<'a>(
    entry: &'a Entry,
    path: &ErrorPath,
    message: &str,
) -> Result<&'a [(String, Entry)], SpecError> {
    match entry {
        Entry::Map(pairs) => Ok(pairs),
        _ => Err(SpecError::new(
            SpecErrorKind::Structure,
            path.clone(),
            message,
        )),
    }
}

fn reject_duplicates(map: &[(String, Entry)], path: &ErrorPath) -> Result<(), SpecError> {
    for (index, (key, _)) in map.iter().enumerate() {
        if map[..index].iter().any(|(earlier, _)| earlier == key) {
            return Err(SpecError::new(
                SpecErrorKind::DuplicateKey,
                path.key(key),
                format!("duplicate key `{key}`"),
            ));
        }
    }
    Ok(())
}
