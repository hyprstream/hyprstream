//! capnp emission for the jev-1 decision IR (System One P0.1b).
//!
//! Marshals [`hyprstream_decision`] types — [`QuestionSet`], [`AnswerRow`],
//! [`VersionTriple`] — into the `decision.capnp` wire types
//! ([`crate::decision_capnp`]) and back. The schema is the data contract for
//! the P3.1 inference-surface extension; the golden-vector conformance tests
//! (`tests/decision_golden.rs`) pin it against the P0.1a Arrow emission using
//! the docs' numeric examples (S6a).
//!
//! Contract pins (normative in hyprstream-decision; enforced on decode):
//! - `kind` tag must match the body union discriminant; span/derived (reserved
//!   v2 types) decode as [`DecodeError::ReservedQuestionType`].
//! - `OptEntry.none` (key absent) and `some(Entry.null)` (explicit null) are
//!   distinct, mirroring D3/D4.
//! - `null` = abstained; an abstained answer must not carry a conformal set.
//! - `conformalSet` is an explicit option wrapper: `none` (absent) and a
//!   present empty set stay distinct across the wire.
//! - Decode enforces the CONSUMER distribution tolerance (looser than the
//!   encoder's PRODUCER tolerance), mirroring the Arrow consumer surface.

use hyprstream_decision::confidence::{
    check_distribution, DistributionError, CONSUMER_SUM_TOLERANCE, PRODUCER_SUM_TOLERANCE,
};
use hyprstream_decision::{
    AnswerRow, AnswerValue, ChoiceOption, Entry, NoulCriteria, QuestionAnswer, QuestionBody,
    QuestionKind, QuestionSet, QuestionSpec, VersionTriple,
};

use crate::decision_capnp;

/// Errors decoding a `decision.capnp` message into the IR.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DecodeError {
    /// capnp-level structural failure (truncated message, bad pointer).
    #[error("capnp decode failed: {0}")]
    Capnp(String),
    /// The `kind` tag disagrees with the body union discriminant.
    #[error("kind tag {tag} does not match body union for question {question_id}")]
    KindBodyMismatch {
        /// The question carrying the mismatch.
        question_id: String,
        /// The conflicting tag.
        tag: String,
    },
    /// A reserved v2 question type (span/derived) appeared on the wire.
    #[error("reserved v2 question type {0} on the wire for question {1}")]
    ReservedQuestionType(&'static str, String),
    /// An abstained answer carried a conformal set (abstention already says
    /// "no commitment").
    #[error("abstained answer for question {0} carries a conformal set")]
    AbstainedWithConformalSet(String),
    /// A noul probability fell outside [0, 1].
    #[error("noul probability out of range for question {0}: {1}")]
    NoulProbabilityOutOfRange(String, f32),
    /// An empty question id (authoring rejects these; the wire must too).
    #[error("empty question id on the wire")]
    EmptyQuestionId,
    /// An empty choice option name (authoring rejects these).
    #[error("empty choice option name in question {0}")]
    EmptyChoiceOptionName(String),
    /// A repeated choice option name — labels must be unique for argmax and
    /// conformal-set output to stay well-defined (authoring rejects these).
    #[error("duplicate choice option name {1:?} in question {0}")]
    DuplicateChoiceOptionName(String, String),
    /// A question set with zero questions (authoring requires >= 1).
    #[error("empty question set on the wire")]
    EmptyQuestionSet,
    /// A choice/score distribution violating the consumer contract (finite
    /// components in [0, 1], sum within the D5 consumer tolerance). The
    /// decoder applies the CONSUMER tolerance — looser than the producer
    /// tolerance the encoder enforces — so nothing reaches the IR that the
    /// Arrow surface would reject downstream.
    #[error("invalid distribution on the wire for question {0}: {1}")]
    InvalidDistribution(String, DistributionError),
    /// An answer row with an empty question id — it cannot bind to any
    /// valid question (decoded question sets categorically reject empty
    /// ids).
    #[error("empty question id in an answer row on the wire")]
    EmptyAnswerQuestionId,
    /// An answer distribution whose cardinality cannot match any valid
    /// question spec (choice: 2-255 options per D2, score: >= 2 levels per
    /// D1) — malformed wire data must not enter the IR.
    #[error("answer distribution of {len} components cannot match a {kind} question ({question_id})")]
    AnswerCardinalityOutOfRange {
        /// The answer's question id.
        question_id: String,
        /// The invalid component count.
        len: usize,
        /// Which primitive the count cannot belong to.
        kind: &'static str,
    },
}

/// Errors encoding IR into a `decision.capnp` message.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EncodeError {
    /// An abstained answer carried a conformal set — the wire contract forbids
    /// it (abstention already says "no commitment") and this module's own
    /// decoder rejects such messages.
    #[error("abstained answer for question {0} carries a conformal set")]
    AbstainedWithConformalSet(String),
    /// A question set with zero questions (the decoder rejects these).
    #[error("empty question set cannot be encoded")]
    EmptyQuestionSet,
    /// An empty question id (the decoder rejects these; the IR is public, so
    /// encode mirrors the rule for directly-constructed specs).
    #[error("empty question id cannot be encoded")]
    EmptyQuestionId,
    /// An empty choice option name in question {0} (the decoder rejects these).
    #[error("empty choice option name in question {0}")]
    EmptyChoiceOptionName(String),
    /// A repeated choice option name {1:?} in question {0} — labels must be
    /// unique for argmax and conformal-set output (the decoder rejects these).
    #[error("duplicate choice option name {1:?} in question {0}")]
    DuplicateChoiceOptionName(String, String),
    /// A spec's `kind` disagrees with its body union. The decoder rejects
    /// the mismatched tag, and downstream Arrow emission consults `kind`,
    /// so the encoder refuses to silently rewrite it (exact round trip).
    #[error("kind tag {tag} does not match the body union of question {question_id}")]
    KindBodyMismatch {
        /// The question carrying the mismatch.
        question_id: String,
        /// The stale explicit kind tag.
        tag: String,
    },
    /// A repeated question id — answers bind by id, so duplicates are
    /// ambiguous and the Arrow schema rejects them.
    #[error("duplicate question id {0}")]
    DuplicateQuestionId(String),
    /// A choice question outside the D2 range of 2–255 options.
    #[error("choice question {question_id} has {len} options (D2 requires 2-255)")]
    ChoiceCardinalityOutOfRange {
        /// The offending question.
        question_id: String,
        /// The invalid option count.
        len: usize,
    },
    /// A score question with fewer than the D1 minimum of 2 levels.
    #[error("score question {question_id} has {len} levels (D1 requires >= 2)")]
    ScoreLevelCountInvalid {
        /// The offending question.
        question_id: String,
        /// The invalid level count.
        len: usize,
    },
    /// A non-finite `Entry::Number` anywhere in the set — the IR guarantees
    /// finite numbers, and non-finite values have no canonical JSON text.
    #[error("non-finite Entry::Number in the question set (question: {question:?})")]
    NonFiniteEntryNumber {
        /// The question containing the entry (None = the shared state).
        question: Option<String>,
    },
    /// A noul probability outside [0, 1] or not finite.
    #[error("noul probability out of range for question {0}: {1}")]
    NoulProbabilityOutOfRange(String, f32),
    /// A choice/score vector violating the producer distribution contract
    /// (finite components in [0, 1], sum within D5 producer tolerance).
    #[error("invalid distribution for question {0}: {1}")]
    InvalidDistribution(String, DistributionError),
    /// An answer distribution whose length cannot match any valid question
    /// spec (choice: 2-255 options per D2, score: >= 2 levels per D1) —
    /// the decoder rejects the same messages, so encode mirrors the rule.
    #[error("answer distribution of {len} components cannot match a {kind} question ({question_id})")]
    AnswerCardinalityOutOfRange {
        /// The answer's question id.
        question_id: String,
        /// The invalid component count.
        len: usize,
        /// Which primitive the count cannot belong to.
        kind: &'static str,
    },
    /// An answer row with an empty question id — the decoder rejects these,
    /// so a successful encode must stay decodable by this module.
    #[error("empty question id in an answer row cannot be encoded")]
    EmptyAnswerQuestionId,
}

impl From<capnp::Error> for DecodeError {
    fn from(error: capnp::Error) -> Self {
        Self::Capnp(error.to_string())
    }
}

impl From<capnp::NotInSchema> for DecodeError {
    fn from(error: capnp::NotInSchema) -> Self {
        Self::Capnp(error.to_string())
    }
}

impl From<std::str::Utf8Error> for DecodeError {
    fn from(error: std::str::Utf8Error) -> Self {
        Self::Capnp(error.to_string())
    }
}

// ============================================================================
// Entry
// ============================================================================

fn set_entry(mut builder: decision_capnp::entry::Builder<'_>, entry: &Entry) {
    match entry {
        Entry::Null => builder.set_null(()),
        Entry::Bool(value) => builder.set_bool(*value),
        Entry::Number(value) => builder.set_number(*value),
        Entry::Str(text) => builder.set_str(text),
        Entry::Seq(items) => {
            let mut list = builder.init_seq(items.len() as u32);
            for (index, item) in items.iter().enumerate() {
                set_entry(list.reborrow().get(index as u32), item);
            }
        }
        Entry::Map(pairs) => {
            let mut list = builder.init_map(pairs.len() as u32);
            for (index, (key, value)) in pairs.iter().enumerate() {
                let mut pair = list.reborrow().get(index as u32);
                pair.set_key(key);
                set_entry(pair.init_value(), value);
            }
        }
    }
}

fn get_entry(reader: decision_capnp::entry::Reader<'_>) -> Result<Entry, DecodeError> {
    Ok(match reader.which()? {
        decision_capnp::entry::Null(()) => Entry::Null,
        decision_capnp::entry::Bool(value) => Entry::Bool(value),
        decision_capnp::entry::Number(value) => Entry::Number(value),
        decision_capnp::entry::Str(text) => Entry::Str(text?.to_str()?.to_owned()),
        decision_capnp::entry::Seq(items) => {
            let items = items?;
            let mut out = Vec::with_capacity(items.len() as usize);
            for item in items.iter() {
                out.push(get_entry(item)?);
            }
            Entry::Seq(out)
        }
        decision_capnp::entry::Map(pairs) => {
            let pairs = pairs?;
            let mut out = Vec::with_capacity(pairs.len() as usize);
            for pair in pairs.iter() {
                out.push((
                    pair.get_key()?.to_str()?.to_owned(),
                    get_entry(pair.get_value()?)?,
                ));
            }
            Entry::Map(out)
        }
    })
}

fn set_opt_entry(mut builder: decision_capnp::opt_entry::Builder<'_>, entry: Option<&Entry>) {
    match entry {
        None => builder.set_none(()),
        Some(entry) => set_entry(builder.init_some(), entry),
    }
}

fn get_opt_entry(reader: decision_capnp::opt_entry::Reader<'_>) -> Result<Option<Entry>, DecodeError> {
    Ok(match reader.which()? {
        decision_capnp::opt_entry::None(()) => None,
        decision_capnp::opt_entry::Some(entry) => Some(get_entry(entry?)?),
    })
}

// ============================================================================
// Question specs
// ============================================================================

fn question_kind_tag(kind: QuestionKind) -> decision_capnp::QuestionKind {
    match kind {
        QuestionKind::Noul => decision_capnp::QuestionKind::Noul,
        QuestionKind::Choice => decision_capnp::QuestionKind::Choice,
        QuestionKind::Score => decision_capnp::QuestionKind::Score,
        QuestionKind::Span => decision_capnp::QuestionKind::Span,
        QuestionKind::Derived => decision_capnp::QuestionKind::Derived,
    }
}

fn set_noul_criteria(
    mut builder: decision_capnp::noul_criteria::Builder<'_>,
    criteria: &NoulCriteria,
) {
    set_opt_entry(builder.reborrow().init_on_true(), criteria.on_true.as_ref());
    set_opt_entry(builder.init_on_false(), criteria.on_false.as_ref());
}

fn set_question_spec(builder: &mut decision_capnp::question_spec::Builder<'_>, spec: &QuestionSpec) {
    builder.set_id(&spec.id);
    // The wire tag is derived from the body, never from the denormalized
    // `kind` field: a directly-constructed IR value could carry a conflicting
    // pair, and the decoder rejects kind/body mismatches.
    builder.set_kind(question_kind_tag(match &spec.body {
        QuestionBody::Noul { .. } => QuestionKind::Noul,
        QuestionBody::Choice { .. } => QuestionKind::Choice,
        QuestionBody::Score { .. } => QuestionKind::Score,
    }));
    set_opt_entry(builder.reborrow().init_instructions(), spec.instructions.as_ref());
    match &spec.body {
        QuestionBody::Noul { criteria } => {
            let mut body = builder.reborrow().init_body();
            match criteria {
                None => body.reborrow().init_noul().set_none(()),
                Some(criteria) => {
                    let mut opt = body.init_noul();
                    set_noul_criteria(opt.reborrow().init_some(), criteria);
                }
            }
        }
        QuestionBody::Choice { options } => {
            let body = builder.reborrow().init_body();
            let mut list = body.init_choice(options.len() as u32);
            for (index, option) in options.iter().enumerate() {
                let mut item = list.reborrow().get(index as u32);
                item.set_name(&option.name);
                set_opt_entry(item.init_rubric(), option.rubric.as_ref());
            }
        }
        QuestionBody::Score { levels } => {
            let body = builder.reborrow().init_body();
            let mut list = body.init_score(levels.len() as u32);
            for (index, level) in levels.iter().enumerate() {
                set_opt_entry(list.reborrow().get(index as u32), level.as_ref());
            }
        }
    }
}

fn get_question_spec(
    reader: decision_capnp::question_spec::Reader<'_>,
) -> Result<QuestionSpec, DecodeError> {
    let id = reader.get_id()?.to_str()?.to_owned();
    if id.is_empty() {
        return Err(DecodeError::EmptyQuestionId);
    }
    let kind = reader.get_kind()?;
    let instructions = get_opt_entry(reader.get_instructions()?)?;
    let body = match reader.get_body().which()? {
        decision_capnp::question_spec::body::Noul(criteria) => {
            if kind != decision_capnp::QuestionKind::Noul {
                return Err(DecodeError::KindBodyMismatch {
                    question_id: id,
                    tag: format!("{kind:?}"),
                });
            }
            let criteria = match criteria?.which()? {
                decision_capnp::opt_noul_criteria::None(()) => None,
                decision_capnp::opt_noul_criteria::Some(reader) => {
                    let reader = reader?;
                    Some(NoulCriteria {
                        on_true: get_opt_entry(reader.get_on_true()?)?,
                        on_false: get_opt_entry(reader.get_on_false()?)?,
                    })
                }
            };
            QuestionBody::Noul { criteria }
        }
        decision_capnp::question_spec::body::Choice(options) => {
            if kind != decision_capnp::QuestionKind::Choice {
                return Err(DecodeError::KindBodyMismatch {
                    question_id: id,
                    tag: format!("{kind:?}"),
                });
            }
            let options = options?;
            let mut out = Vec::with_capacity(options.len() as usize);
            for option in options.iter() {
                let name = option.get_name()?.to_str()?.to_owned();
                if name.is_empty() {
                    return Err(DecodeError::EmptyChoiceOptionName(id));
                }
                if out.iter().any(|existing: &ChoiceOption| existing.name == name) {
                    return Err(DecodeError::DuplicateChoiceOptionName(id, name));
                }
                out.push(ChoiceOption {
                    name,
                    rubric: get_opt_entry(option.get_rubric()?)?,
                });
            }
            QuestionBody::Choice { options: out }
        }
        decision_capnp::question_spec::body::Score(levels) => {
            if kind != decision_capnp::QuestionKind::Score {
                return Err(DecodeError::KindBodyMismatch {
                    question_id: id,
                    tag: format!("{kind:?}"),
                });
            }
            let levels = levels?;
            let mut out = Vec::with_capacity(levels.len() as usize);
            for level in levels.iter() {
                out.push(get_opt_entry(level)?);
            }
            QuestionBody::Score { levels: out }
        }
        decision_capnp::question_spec::body::Span(()) => {
            return Err(DecodeError::ReservedQuestionType("span", id));
        }
        decision_capnp::question_spec::body::Derived(()) => {
            return Err(DecodeError::ReservedQuestionType("derived", id));
        }
    };
    Ok(QuestionSpec {
        id,
        kind: match kind {
            decision_capnp::QuestionKind::Noul => QuestionKind::Noul,
            decision_capnp::QuestionKind::Choice => QuestionKind::Choice,
            decision_capnp::QuestionKind::Score => QuestionKind::Score,
            decision_capnp::QuestionKind::Span => QuestionKind::Span,
            decision_capnp::QuestionKind::Derived => QuestionKind::Derived,
        },
        instructions,
        body,
    })
}

/// Recursively reject non-finite `Entry::Number` values: the IR guarantees
/// finite numbers, and a non-finite value has no canonical JSON text, so it
/// cannot survive the canonical serialization the model-facing pipeline
/// depends on.
fn validate_entry_finite(
    entry: Option<&Entry>,
    question_id: Option<&str>,
) -> Result<(), EncodeError> {
    let entry = match entry {
        Some(entry) => entry,
        None => return Ok(()),
    };
    match entry {
        Entry::Number(value) if !value.is_finite() => {
            Err(EncodeError::NonFiniteEntryNumber {
                question: question_id.map(str::to_owned),
            })
        }
        Entry::Seq(items) => {
            for item in items {
                validate_entry_finite(Some(item), question_id)?;
            }
            Ok(())
        }
        Entry::Map(pairs) => {
            for (_, value) in pairs {
                validate_entry_finite(Some(value), question_id)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Encode a question set as a `QuestionSet` message.
pub fn question_set_to_message(
    set: &QuestionSet,
) -> Result<capnp::message::Builder<capnp::message::HeapAllocator>, EncodeError> {
    if set.questions.is_empty() {
        return Err(EncodeError::EmptyQuestionSet);
    }
    // Encode mirrors the decode-side authoring-layer rules (get_question_spec)
    // and the D1/D2 cardinality contract: the IR is public, so a
    // directly-constructed spec must not emit a message our own decoder
    // rejects — or one whose explicit `kind` disagrees with its body, which
    // would make the IR and the wire disagree after a round trip
    // (EmptyQuestionId / DuplicateQuestionId / KindBodyMismatch /
    // ChoiceCardinalityOutOfRange / ScoreLevelCountInvalid /
    // EmptyChoiceOptionName / DuplicateChoiceOptionName /
    // NonFiniteEntryNumber).
    validate_entry_finite(set.state.as_ref(), None)?;
    let mut seen_ids = std::collections::HashSet::with_capacity(set.questions.len());
    for spec in &set.questions {
        if spec.id.is_empty() {
            return Err(EncodeError::EmptyQuestionId);
        }
        if !seen_ids.insert(spec.id.as_str()) {
            return Err(EncodeError::DuplicateQuestionId(spec.id.clone()));
        }
        let derived_kind = match &spec.body {
            QuestionBody::Noul { .. } => QuestionKind::Noul,
            QuestionBody::Choice { .. } => QuestionKind::Choice,
            QuestionBody::Score { .. } => QuestionKind::Score,
        };
        if spec.kind != derived_kind {
            return Err(EncodeError::KindBodyMismatch {
                question_id: spec.id.clone(),
                tag: format!("{:?}", spec.kind),
            });
        }
        validate_entry_finite(spec.instructions.as_ref(), Some(&spec.id))?;
        match &spec.body {
            QuestionBody::Noul { criteria } => {
                if let Some(criteria) = criteria {
                    validate_entry_finite(criteria.on_true.as_ref(), Some(&spec.id))?;
                    validate_entry_finite(criteria.on_false.as_ref(), Some(&spec.id))?;
                }
            }
            QuestionBody::Choice { options } => {
                let count = options.len();
                if !(2..=255).contains(&count) {
                    return Err(EncodeError::ChoiceCardinalityOutOfRange {
                        question_id: spec.id.clone(),
                        len: count,
                    });
                }
                let mut seen_names = std::collections::HashSet::with_capacity(count);
                for option in options {
                    if option.name.is_empty() {
                        return Err(EncodeError::EmptyChoiceOptionName(spec.id.clone()));
                    }
                    if !seen_names.insert(option.name.as_str()) {
                        return Err(EncodeError::DuplicateChoiceOptionName(
                            spec.id.clone(),
                            option.name.clone(),
                        ));
                    }
                    validate_entry_finite(option.rubric.as_ref(), Some(&spec.id))?;
                }
            }
            QuestionBody::Score { levels } => {
                if levels.len() < 2 {
                    return Err(EncodeError::ScoreLevelCountInvalid {
                        question_id: spec.id.clone(),
                        len: levels.len(),
                    });
                }
                for level in levels {
                    validate_entry_finite(level.as_ref(), Some(&spec.id))?;
                }
            }
        }
    }
    let mut message = capnp::message::Builder::new_default();
    let mut root = message.init_root::<decision_capnp::question_set::Builder<'_>>();
    set_opt_entry(root.reborrow().init_state(), set.state.as_ref());
    let mut questions = root.init_questions(set.questions.len() as u32);
    for (index, spec) in set.questions.iter().enumerate() {
        let mut item = questions.reborrow().get(index as u32);
        set_question_spec(&mut item, spec);
    }
    Ok(message)
}

/// Decode a `QuestionSet` message back into the IR.
pub fn question_set_from_reader(
    reader: decision_capnp::question_set::Reader<'_>,
) -> Result<QuestionSet, DecodeError> {
    let state = get_opt_entry(reader.get_state()?)?;
    let questions_reader = reader.get_questions()?;
    let mut questions = Vec::with_capacity(questions_reader.len() as usize);
    for spec in questions_reader.iter() {
        questions.push(get_question_spec(spec)?);
    }
    if questions.is_empty() {
        return Err(DecodeError::EmptyQuestionSet);
    }
    Ok(QuestionSet { state, questions })
}

// ============================================================================
// Answers
// ============================================================================

fn set_answer_value(mut builder: decision_capnp::answer_value::Builder<'_>, value: Option<&AnswerValue>) {
    match value {
        None => builder.set_abstained(()),
        Some(AnswerValue::Noul { p_true }) => builder.set_noul(*p_true),
        Some(AnswerValue::Choice { probabilities }) | Some(AnswerValue::Score { probabilities }) => {
            let mut list = if matches!(value, Some(AnswerValue::Choice { .. })) {
                builder.init_choice(probabilities.len() as u32)
            } else {
                builder.init_score(probabilities.len() as u32)
            };
            for (index, &p) in probabilities.iter().enumerate() {
                list.set(index as u32, p);
            }
        }
    }
}

/// Encode a decision batch (version triple + rows) as a `DecisionBatch` message.
///
/// Answers are emitted keyed by question id (capnp has no map type); rows are
/// emitted in slice order, answers within a row in the row's `BTreeMap` order.
pub fn batch_to_message(
    version: &VersionTriple,
    rows: &[AnswerRow],
) -> Result<capnp::message::Builder<capnp::message::HeapAllocator>, EncodeError> {
    let mut message = capnp::message::Builder::new_default();
    let mut root = message.init_root::<decision_capnp::decision_batch::Builder<'_>>();
    {
        let mut triple = root.reborrow().init_version();
        triple.set_schema(&version.schema);
        triple.set_model(&version.model);
        match &version.calib {
            None => triple.reborrow().init_calib().set_none(()),
            Some(calib) => triple.init_calib().set_some(calib),
        }
    }
    let mut rows_builder = root.init_rows(rows.len() as u32);
    for (row_index, row) in rows.iter().enumerate() {
        let row_builder = rows_builder.reborrow().get(row_index as u32);
        let mut answers = row_builder.init_answers(row.answers.len() as u32);
        for (answer_index, (question_id, answer)) in row.answers.iter().enumerate() {
            if question_id.is_empty() {
                return Err(EncodeError::EmptyAnswerQuestionId);
            }
            if answer.value.is_none() && answer.conformal_set.is_some() {
                return Err(EncodeError::AbstainedWithConformalSet(question_id.clone()));
            }
            match &answer.value {
                Some(AnswerValue::Noul { p_true })
                    if !p_true.is_finite() || !(0.0..=1.0).contains(p_true) =>
                {
                    return Err(EncodeError::NoulProbabilityOutOfRange(
                        question_id.clone(),
                        *p_true,
                    ));
                }
                Some(AnswerValue::Choice { probabilities }) => {
                    // A choice answer must be able to match a valid spec
                    // (D2: 2-255 options); the decoder rejects the same
                    // messages, so a successful encode stays decodable.
                    if !(2..=255).contains(&probabilities.len()) {
                        return Err(EncodeError::AnswerCardinalityOutOfRange {
                            question_id: question_id.clone(),
                            len: probabilities.len(),
                            kind: "choice",
                        });
                    }
                    check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).map_err(|error| {
                        EncodeError::InvalidDistribution(question_id.clone(), error)
                    })?;
                }
                Some(AnswerValue::Score { probabilities }) => {
                    // A score answer must be able to match a valid spec
                    // (D1: >= 2 levels).
                    if probabilities.len() < 2 {
                        return Err(EncodeError::AnswerCardinalityOutOfRange {
                            question_id: question_id.clone(),
                            len: probabilities.len(),
                            kind: "score",
                        });
                    }
                    check_distribution(probabilities, PRODUCER_SUM_TOLERANCE).map_err(|error| {
                        EncodeError::InvalidDistribution(question_id.clone(), error)
                    })?;
                }
                _ => {}
            }
            let mut answer_builder = answers.reborrow().get(answer_index as u32);
            answer_builder.set_question_id(question_id);
            set_answer_value(
                answer_builder.reborrow().init_value(),
                answer.value.as_ref(),
            );
            match &answer.conformal_set {
                None => answer_builder.init_conformal_set().set_none(()),
                Some(set) => {
                    let mut members =
                        answer_builder.init_conformal_set().init_some(set.len() as u32);
                    for (member_index, member) in set.iter().enumerate() {
                        members.set(member_index as u32, member.as_str());
                    }
                }
            }
        }
    }
    Ok(message)
}

/// Decode a `DecisionBatch` message: the version triple plus one [`AnswerRow`]
/// per wire row, answers keyed by question id.
pub fn batch_from_reader(
    reader: decision_capnp::decision_batch::Reader<'_>,
) -> Result<(VersionTriple, Vec<AnswerRow>), DecodeError> {
    let triple = reader.get_version()?;
    let version = VersionTriple {
        schema: triple.get_schema()?.to_str()?.to_owned(),
        model: triple.get_model()?.to_str()?.to_owned(),
        calib: match triple.get_calib()?.which()? {
            crate::optional_capnp::option_text::None(()) => None,
            crate::optional_capnp::option_text::Some(calib) => {
                Some(calib?.to_str()?.to_owned())
            }
        },
    };
    let rows_reader = reader.get_rows()?;
    let mut rows = Vec::with_capacity(rows_reader.len() as usize);
    for row in rows_reader.iter() {
        let mut answers = std::collections::BTreeMap::new();
        for answer in row.get_answers()?.iter() {
            let question_id = answer.get_question_id()?.to_str()?.to_owned();
            if question_id.is_empty() {
                return Err(DecodeError::EmptyAnswerQuestionId);
            }
            let conformal_set = match answer.get_conformal_set()?.which()? {
                decision_capnp::opt_conformal_set::None(()) => None,
                decision_capnp::opt_conformal_set::Some(members) => {
                    let members = members?;
                    let mut set = Vec::with_capacity(members.len() as usize);
                    for member in members.iter() {
                        set.push(member?.to_str()?.to_owned());
                    }
                    Some(set)
                }
            };
            let value = match answer.get_value()?.which()? {
                decision_capnp::answer_value::Abstained(()) => {
                    if conformal_set.is_some() {
                        return Err(DecodeError::AbstainedWithConformalSet(question_id));
                    }
                    None
                }
                decision_capnp::answer_value::Noul(p_true) => {
                    if !(0.0..=1.0).contains(&p_true) {
                        return Err(DecodeError::NoulProbabilityOutOfRange(question_id, p_true));
                    }
                    Some(AnswerValue::Noul { p_true })
                }
                decision_capnp::answer_value::Choice(probabilities) => {
                    let probabilities: Vec<f32> = probabilities?.iter().collect();
                    // A choice answer must be able to match a valid spec
                    // (D2: 2-255 options); anything else is malformed wire
                    // data and must not enter the IR.
                    if !(2..=255).contains(&probabilities.len()) {
                        return Err(DecodeError::AnswerCardinalityOutOfRange {
                            question_id,
                            len: probabilities.len(),
                            kind: "choice",
                        });
                    }
                    check_distribution(&probabilities, CONSUMER_SUM_TOLERANCE).map_err(
                        |error| DecodeError::InvalidDistribution(question_id.clone(), error),
                    )?;
                    Some(AnswerValue::Choice { probabilities })
                }
                decision_capnp::answer_value::Score(probabilities) => {
                    let probabilities: Vec<f32> = probabilities?.iter().collect();
                    // A score answer must be able to match a valid spec
                    // (D1: >= 2 levels).
                    if probabilities.len() < 2 {
                        return Err(DecodeError::AnswerCardinalityOutOfRange {
                            question_id,
                            len: probabilities.len(),
                            kind: "score",
                        });
                    }
                    check_distribution(&probabilities, CONSUMER_SUM_TOLERANCE).map_err(
                        |error| DecodeError::InvalidDistribution(question_id.clone(), error),
                    )?;
                    Some(AnswerValue::Score { probabilities })
                }
            };
            answers.insert(question_id, QuestionAnswer { value, conformal_set });
        }
        rows.push(AnswerRow { answers });
    }
    Ok((version, rows))
}
