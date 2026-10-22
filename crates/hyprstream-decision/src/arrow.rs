//! Arrow emission: the columnar contract for decision batches.
//!
//! For a set of questions, [`DecisionSchema`] emits (in question authoring order, then
//! the version triple):
//!
//! | column | type | nullability | content |
//! |---|---|---|---|
//! | `{qid}.probabilities` | `fixed_size_list<f32; N>` | nullable | one probability per label, canonical order; null = abstained |
//! | `{qid}.label` | `utf8` | nullable | argmax label (D6 tie-break); **null = abstained** |
//! | `{qid}.conformal_set` | `list<utf8>` | nullable | conformal prediction set — **reserved in v1**, may be entirely unpopulated |
//! | `schema_version` | `utf8` | non-null | question-set schema version |
//! | `model_version` | `utf8` | non-null | resolved model id |
//! | `calib_version` | `utf8` | nullable | calibration-fit version; null = uncalibrated |
//!
//! N (cardinality) is 2 for noul (`["false", "true"]`), the option count for choice, the
//! level count for score.
//!
//! ## Contract rules
//!
//! - **Label lists live in Arrow *field* metadata** on the probabilities column
//!   (`jev.kind`, `jev.labels` as a JSON array) — never in schema metadata, which this
//!   crate always emits empty. Calibration parameters are likewise never schema content;
//!   they are rows in the P0.3 metrics tables.
//! - **Label-set evolution is schema evolution and fails loudly**:
//!   [`DecisionSchema::check_evolution`] rejects any change to an existing question's kind
//!   or label set — and any flip of conformal-set emission, which changes the emitted
//!   column set; the remedy is minting a new schema version (the `schema_version`
//!   triple column), which is what prepared statements pin (P3.5).
//! - Question ids must be identifier-safe (`[A-Za-z_][A-Za-z0-9_]*`) for the columnar
//!   surface so emitted columns compose unquoted in SQL (the ADBC path). Wire-level ids
//!   stay opaque; the constraint applies at schema construction, not at authoring.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use thiserror::Error;

use crate::answer::{AnswerValue, QuestionAnswer, VersionTriple};
use crate::confidence::{self, PRODUCER_SUM_TOLERANCE};
use crate::spec::{QuestionKind, QuestionSet, QuestionSpec};

/// Field-metadata key carrying the question kind (`noul` / `choice` / `score`).
pub const FIELD_META_KIND: &str = "jev.kind";
/// Field-metadata key carrying the label list as a JSON array in canonical order.
pub const FIELD_META_LABELS: &str = "jev.labels";

/// Column name of the question-set schema version (version triple).
pub const SCHEMA_VERSION_COLUMN: &str = "schema_version";
/// Column name of the resolved model id (version triple).
pub const MODEL_VERSION_COLUMN: &str = "model_version";
/// Column name of the calibration-fit version (version triple; null = uncalibrated).
pub const CALIB_VERSION_COLUMN: &str = "calib_version";

/// Why a set of questions cannot back an Arrow schema.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ArrowSchemaError {
    /// A schema needs at least one question.
    #[error("a decision schema needs at least one question")]
    Empty,
    /// A question id is not identifier-safe for the columnar surface.
    #[error(
        "question id `{0}` is not identifier-safe ([A-Za-z_][A-Za-z0-9_]*); \
         wire ids stay opaque, but the Arrow surface needs unquoted SQL composition"
    )]
    InvalidIdentifier(String),
    /// Two questions share an id.
    #[error("duplicate question id `{0}`")]
    DuplicateQuestionId(String),
    /// A question id collides with a reserved version-triple column.
    #[error("question id `{0}` collides with a reserved version-triple column name")]
    ReservedColumnName(String),
}

/// One changed question in a failed label-set evolution check.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelChange {
    /// The question whose contract changed.
    pub question_id: String,
    /// Kind in the prior schema.
    pub old_kind: QuestionKind,
    /// Kind in the new schema.
    pub new_kind: QuestionKind,
    /// Labels in the prior schema.
    pub old_labels: Vec<String>,
    /// Labels in the new schema.
    pub new_labels: Vec<String>,
}

/// Label-set evolution on an existing question — schema evolution fails loudly.
#[derive(Debug, Clone, PartialEq, Error)]
#[error(
    "label-set evolution requires a new schema version; changed questions: {}",
    .changes.iter().map(|change| change.question_id.as_str()).collect::<Vec<_>>().join(", ")
)]
pub struct LabelEvolutionError {
    /// Every question whose kind or label set changed.
    pub changes: Vec<LabelChange>,
}

/// Why a schema generation pair fails the evolution check.
///
/// Covers everything [`DecisionSchema::fingerprint`] treats as schema content, so the
/// two drift detectors can never disagree: an evolution the fingerprint would see is an
/// evolution this check rejects.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EvolutionError {
    /// An existing question's kind or label set changed.
    #[error(transparent)]
    LabelSetChanged(#[from] LabelEvolutionError),
    /// The conformal-set column flag flipped. The emitted Arrow schema gains or loses a
    /// column per question, breaking positional consumers and prepared statements, so
    /// this is schema evolution even though no question changed.
    #[error(
        "conformal-set emission flipped ({old_conformal_set} -> {new_conformal_set}); \
         the emitted column set changes, so this requires a new schema version"
    )]
    EmissionShapeChanged {
        /// Prior schema's flag.
        old_conformal_set: bool,
        /// New schema's flag.
        new_conformal_set: bool,
    },
}

/// Compatible differences between two schema generations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvolutionReport {
    /// Question ids present only in the new schema.
    pub added: Vec<String>,
    /// Question ids present only in the prior schema.
    pub removed: Vec<String>,
}

/// Why a row cannot be encoded into a batch under this schema.
#[derive(Debug, Error)]
pub enum BatchError {
    /// A row is missing an answer (even an abstention) for a declared question.
    #[error(
        "row {row} is missing an answer for question `{question_id}` (abstain explicitly instead)"
    )]
    MissingAnswer {
        /// Offending row index.
        row: usize,
        /// The unanswered question.
        question_id: String,
    },
    /// A row answers a question the schema does not declare.
    #[error("row {row} answers unknown question `{question_id}`")]
    UnknownQuestion {
        /// Offending row index.
        row: usize,
        /// The unknown question id.
        question_id: String,
    },
    /// The answer kind does not match the question kind.
    #[error("row {row} question `{question_id}`: answer kind {got} does not match question kind {expected}")]
    KindMismatch {
        /// Offending row index.
        row: usize,
        /// The question.
        question_id: String,
        /// The question's kind.
        expected: QuestionKind,
        /// The answer's kind.
        got: QuestionKind,
    },
    /// The probability vector length does not equal the question cardinality.
    #[error("row {row} question `{question_id}`: {got} probabilities for a cardinality-{expected} question")]
    CardinalityMismatch {
        /// Offending row index.
        row: usize,
        /// The question.
        question_id: String,
        /// The question's cardinality.
        expected: usize,
        /// The vector length supplied.
        got: usize,
    },
    /// The distribution violates the producer tolerance (D5).
    #[error("row {0} question `{1}`: {2}")]
    InvalidDistribution(usize, String, #[source] confidence::DistributionError),
    /// An abstained answer carries a conformal set.
    #[error(
        "row {row} question `{question_id}`: an abstained answer cannot carry a conformal set"
    )]
    AbstainedWithConformalSet {
        /// Offending row index.
        row: usize,
        /// The question.
        question_id: String,
    },
    /// A conformal set names a label outside the question's label set.
    #[error("row {row} question `{question_id}`: conformal set member `{label}` is not one of the question's labels")]
    UnknownConformalLabel {
        /// Offending row index.
        row: usize,
        /// The question.
        question_id: String,
        /// The offending label.
        label: String,
    },
    /// Arrow rejected the assembled batch (internal invariant violation).
    #[error("arrow batch assembly failed: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
}

/// One row of a decision batch: answers keyed by question id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnswerRow {
    /// Per-question answers. Every declared question must be present — abstain
    /// explicitly ([`QuestionAnswer::abstained`]) rather than omitting the key.
    pub answers: std::collections::BTreeMap<String, QuestionAnswer>,
}

/// The Arrow-emission view over a set of questions. Cheap to clone (the questions are
/// shared); the Arrow schema itself is rebuilt on demand so an emitted schema can never
/// drift from the IR.
#[derive(Debug, Clone)]
pub struct DecisionSchema {
    questions: Vec<QuestionSpec>,
    emit_conformal_set: bool,
}

impl DecisionSchema {
    /// Build from validated questions. The conformal-set column is emitted by default
    /// (reserved in v1); see [`DecisionSchema::without_conformal_set`].
    pub fn new(questions: Vec<QuestionSpec>) -> Result<Self, ArrowSchemaError> {
        if questions.is_empty() {
            return Err(ArrowSchemaError::Empty);
        }
        let mut seen: Vec<&str> = Vec::with_capacity(questions.len());
        for question in &questions {
            let id = question.id.as_str();
            if !is_identifier_safe(id) {
                return Err(ArrowSchemaError::InvalidIdentifier(question.id.clone()));
            }
            if matches!(
                id,
                SCHEMA_VERSION_COLUMN | MODEL_VERSION_COLUMN | CALIB_VERSION_COLUMN
            ) {
                return Err(ArrowSchemaError::ReservedColumnName(question.id.clone()));
            }
            if seen.contains(&id) {
                return Err(ArrowSchemaError::DuplicateQuestionId(question.id.clone()));
            }
            seen.push(id);
        }
        Ok(Self {
            questions,
            emit_conformal_set: true,
        })
    }

    /// Build from an authored question set (the state is not part of the columnar
    /// contract).
    pub fn from_question_set(set: &QuestionSet) -> Result<Self, ArrowSchemaError> {
        Self::new(set.questions.clone())
    }

    /// Omit the reserved conformal-set columns (v1 consumers that do not read them).
    pub fn without_conformal_set(mut self) -> Self {
        self.emit_conformal_set = false;
        self
    }

    /// The questions backing this schema, in canonical order.
    pub fn questions(&self) -> &[QuestionSpec] {
        &self.questions
    }

    /// The Arrow schema. Schema metadata is always empty by construction: labels live in
    /// field metadata, calibration parameters in P0.3 tables.
    pub fn arrow_schema(&self) -> Schema {
        let mut fields = Vec::with_capacity(self.questions.len() * 3 + 3);
        for question in &self.questions {
            let mut metadata = HashMap::new();
            metadata.insert(
                FIELD_META_KIND.to_owned(),
                question.kind.as_str().to_owned(),
            );
            metadata.insert(
                FIELD_META_LABELS.to_owned(),
                labels_metadata(&question.labels()),
            );
            // Inner field nullable=true matches the FixedSizeListBuilder default
            // (repo convention, hyprstream-metrics); individual components are never
            // null in practice — abstention nulls the whole list.
            let cardinality = i32::try_from(question.cardinality()).unwrap_or(i32::MAX);
            let probabilities = Field::new(
                format!("{}.probabilities", question.id),
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    cardinality,
                ),
                true,
            )
            .with_metadata(metadata);
            fields.push(probabilities);
            fields.push(Field::new(
                format!("{}.label", question.id),
                DataType::Utf8,
                true,
            ));
            if self.emit_conformal_set {
                fields.push(Field::new(
                    format!("{}.conformal_set", question.id),
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    true,
                ));
            }
        }
        fields.push(Field::new(SCHEMA_VERSION_COLUMN, DataType::Utf8, false));
        fields.push(Field::new(MODEL_VERSION_COLUMN, DataType::Utf8, false));
        fields.push(Field::new(CALIB_VERSION_COLUMN, DataType::Utf8, true));
        Schema::new(fields)
    }

    /// A stable content fingerprint over (id, kind, labels) per question plus the
    /// conformal-set flag. Registries persist this to detect drift; the hex string is
    /// safe to embed in P0.3 rows.
    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hyprstream-decision schema v1\x00");
        hasher.update(&[self.emit_conformal_set as u8]);
        for question in &self.questions {
            hasher.update(question.id.as_bytes());
            hasher.update(b"\x00");
            hasher.update(question.kind.as_str().as_bytes());
            hasher.update(b"\x00");
            for label in question.labels() {
                hasher.update(label.as_bytes());
                hasher.update(b"\x00");
            }
            hasher.update(b"\x01");
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Fail-loud schema evolution check. Adding or removing questions is compatible
    /// (reported); changing an existing question's kind or label set — or flipping
    /// conformal-set emission, which changes the emitted column set — is not: mint a new
    /// schema version instead. The rejected inputs are exactly the ones
    /// [`DecisionSchema::fingerprint`] distinguishes.
    pub fn check_evolution(old: &Self, new: &Self) -> Result<EvolutionReport, EvolutionError> {
        if old.emit_conformal_set != new.emit_conformal_set {
            return Err(EvolutionError::EmissionShapeChanged {
                old_conformal_set: old.emit_conformal_set,
                new_conformal_set: new.emit_conformal_set,
            });
        }
        let mut changes = Vec::new();
        for old_question in &old.questions {
            if let Some(new_question) = new.questions.iter().find(|q| q.id == old_question.id) {
                let old_labels = old_question.labels();
                let new_labels = new_question.labels();
                if old_question.kind != new_question.kind || old_labels != new_labels {
                    changes.push(LabelChange {
                        question_id: old_question.id.clone(),
                        old_kind: old_question.kind,
                        new_kind: new_question.kind,
                        old_labels,
                        new_labels,
                    });
                }
            }
        }
        if !changes.is_empty() {
            return Err(LabelEvolutionError { changes }.into());
        }
        let report = EvolutionReport {
            added: new
                .questions
                .iter()
                .filter(|q| !old.questions.iter().any(|old_q| old_q.id == q.id))
                .map(|q| q.id.clone())
                .collect(),
            removed: old
                .questions
                .iter()
                .filter(|q| !new.questions.iter().any(|new_q| new_q.id == q.id))
                .map(|q| q.id.clone())
                .collect(),
        };
        Ok(report)
    }

    /// Encode rows into a record batch under this schema. Every row must carry an answer
    /// (possibly abstained) for every declared question; distributions are validated
    /// against the producer tolerance (D5).
    pub fn build_batch(
        &self,
        version: &VersionTriple,
        rows: &[AnswerRow],
    ) -> Result<RecordBatch, BatchError> {
        let row_count = rows.len();
        for (row_index, row) in rows.iter().enumerate() {
            for key in row.answers.keys() {
                if !self.questions.iter().any(|question| &question.id == key) {
                    return Err(BatchError::UnknownQuestion {
                        row: row_index,
                        question_id: key.clone(),
                    });
                }
            }
        }

        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.questions.len() * 3 + 3);

        for question in &self.questions {
            let labels = question.labels();
            let cardinality = i32::try_from(labels.len()).unwrap_or(i32::MAX);
            let mut probabilities = FixedSizeListBuilder::new(Float32Builder::new(), cardinality);
            let mut label_builder = StringBuilder::new();
            let mut conformal_builder = ListBuilder::new(StringBuilder::new());

            for (row_index, row) in rows.iter().enumerate() {
                let answer =
                    row.answers
                        .get(&question.id)
                        .ok_or_else(|| BatchError::MissingAnswer {
                            row: row_index,
                            question_id: question.id.clone(),
                        })?;
                match (&answer.value, &answer.conformal_set) {
                    (None, Some(_)) => {
                        return Err(BatchError::AbstainedWithConformalSet {
                            row: row_index,
                            question_id: question.id.clone(),
                        });
                    }
                    (None, None) => {
                        // Abstention: null probabilities + null label.
                        for _ in 0..cardinality {
                            probabilities.values().append_value(0.0);
                        }
                        probabilities.append(false);
                        label_builder.append_null();
                        conformal_builder.append_null();
                    }
                    (Some(value), conformal_set) => {
                        let distribution = validated_distribution(question, value, row_index)?;
                        for &p in &distribution {
                            probabilities.values().append_value(p);
                        }
                        probabilities.append(true);
                        let argmax = confidence::argmax_index(&distribution).unwrap_or(0);
                        label_builder.append_value(&labels[argmax]);
                        match conformal_set {
                            None => conformal_builder.append_null(),
                            Some(set) => {
                                for member in set {
                                    if !labels.contains(member) {
                                        return Err(BatchError::UnknownConformalLabel {
                                            row: row_index,
                                            question_id: question.id.clone(),
                                            label: member.clone(),
                                        });
                                    }
                                    conformal_builder.values().append_value(member);
                                }
                                conformal_builder.append(true);
                            }
                        }
                    }
                }
            }

            columns.push(Arc::new(probabilities.finish()));
            columns.push(Arc::new(label_builder.finish()));
            if self.emit_conformal_set {
                columns.push(Arc::new(conformal_builder.finish()));
            }
        }

        columns.push(Arc::new(StringArray::from_iter(std::iter::repeat_n(
            Some(version.schema.as_str()),
            row_count,
        ))));
        columns.push(Arc::new(StringArray::from_iter(std::iter::repeat_n(
            Some(version.model.as_str()),
            row_count,
        ))));
        columns.push(Arc::new(StringArray::from_iter(std::iter::repeat_n(
            version.calib.as_deref(),
            row_count,
        ))));

        Ok(RecordBatch::try_new(
            Arc::new(self.arrow_schema()),
            columns,
        )?)
    }
}

/// Validate kind/cardinality/distribution and return the canonical-order distribution.
fn validated_distribution(
    question: &QuestionSpec,
    value: &AnswerValue,
    row: usize,
) -> Result<Vec<f32>, BatchError> {
    let answer_kind = match value {
        AnswerValue::Noul { .. } => QuestionKind::Noul,
        AnswerValue::Choice { .. } => QuestionKind::Choice,
        AnswerValue::Score { .. } => QuestionKind::Score,
    };
    if answer_kind != question.kind {
        return Err(BatchError::KindMismatch {
            row,
            question_id: question.id.clone(),
            expected: question.kind,
            got: answer_kind,
        });
    }
    let distribution = value.probabilities();
    let cardinality = question.cardinality();
    if distribution.len() != cardinality {
        return Err(BatchError::CardinalityMismatch {
            row,
            question_id: question.id.clone(),
            expected: cardinality,
            got: distribution.len(),
        });
    }
    confidence::check_distribution(&distribution, PRODUCER_SUM_TOLERANCE)
        .map_err(|error| BatchError::InvalidDistribution(row, question.id.clone(), error))?;
    Ok(distribution)
}

/// The `jev.labels` field-metadata value: a compact JSON array of label strings.
fn labels_metadata(labels: &[String]) -> String {
    let mut out = String::from("[");
    for (index, label) in labels.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(label).unwrap_or_else(|_| "\"\"".to_owned()));
    }
    out.push(']');
    out
}

fn is_identifier_safe(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}
