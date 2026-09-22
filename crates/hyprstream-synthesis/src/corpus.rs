//! The corpus: one JSONL row per (item, augmentation) with **raw per-teacher
//! probability vectors**, provenance, and the distributability flag.
//!
//! Rows are the training-time contract (P1.4): corrections fit by P0.5 are
//! applied to the raw vectors at consumption time, never baked in at
//! synthesis time.

use std::fmt::Write as _;

use hyprstream_decision::{
    ChoiceOption, Entry, NoulCriteria, QuestionBody, QuestionKind, QuestionSpec,
};
use serde::{Deserialize, Serialize};

use crate::item::SynthItem;
use crate::teacher::{Teacher, TosClass};

/// One teacher's raw answer, persisted verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherAnswer {
    /// Roster id.
    pub id: String,
    /// Resolved version pin.
    pub version: String,
    /// ToS class wire form (`open-weights` / `api-distillation-permitted` /
    /// `api-prohibited`).
    pub tos_class: String,
    /// Raw probability vector over the canonical labels.
    pub probs: Vec<f32>,
}

/// Row-level provenance: who made this row and against which frozen inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Generator crate + version.
    pub generator: String,
    /// Frozen benchmark release the firewalls consumed.
    pub benchmark_release: String,
    /// blake3 of the consumed manifest file.
    pub manifest_blake3: String,
}

/// One corpus row: a synthesized (state, question) pair plus the raw
/// per-teacher answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusRow {
    /// Item id (`syn1_<family>_<kind>_<seed>_r<rotation>_p<paraphrase>`).
    pub id: String,
    /// Synthetic family id.
    pub family: String,
    /// Augmentation group (all rotation/paraphrase variants of one base item).
    pub group: String,
    /// Question kind (`noul` / `choice` / `score`).
    pub kind: String,
    /// blake3 of the item's canonical bytes (dedup + firewall handle).
    pub blake3: String,
    /// Generation seed.
    pub seed: u64,
    /// Cyclic option rotation (choice; 0 otherwise).
    pub rotation: u32,
    /// Paraphrase variant (0 = base).
    pub paraphrase: u32,
    /// State, canonical text form.
    pub state: String,
    /// Instructions text, if any.
    pub instructions: Option<String>,
    /// Canonical labels.
    pub labels: Vec<String>,
    /// Rubric texts per label where the kind carries them (choice options,
    /// score levels); `None` = undescribed. Empty for noul.
    pub rubrics: Vec<Option<String>>,
    /// Noul criteria `(on_true, on_false)`; absent for other kinds.
    pub noul_criteria: Option<(Option<String>, Option<String>)>,
    /// Raw per-teacher answers.
    pub teachers: Vec<TeacherAnswer>,
    /// `false` if any answering teacher is encumbered: never publish, never
    /// ship in Apache/MIT artifacts.
    pub distributable: bool,
    /// Provenance.
    pub provenance: Provenance,
}

impl CorpusRow {
    /// Build a row from an item and the answering ensemble. The
    /// distributability flag is derived from the teachers' ToS classes.
    pub(crate) fn from_item(
        item: &SynthItem,
        teachers: &[&dyn Teacher],
        answers: Vec<Vec<f32>>,
        provenance: Provenance,
    ) -> Self {
        let (rubrics, noul_criteria) = match &item.question.body {
            QuestionBody::Choice { options } => (
                options
                    .iter()
                    .map(|option| option.rubric.as_ref().map(Entry::canonical_text))
                    .collect(),
                None,
            ),
            QuestionBody::Score { levels } => (
                levels
                    .iter()
                    .map(|level| level.as_ref().map(Entry::canonical_text))
                    .collect(),
                None,
            ),
            QuestionBody::Noul { criteria } => {
                let pair = criteria.as_ref().map(|c| {
                    (
                        c.on_true.as_ref().map(Entry::canonical_text),
                        c.on_false.as_ref().map(Entry::canonical_text),
                    )
                });
                (Vec::new(), pair)
            }
        };
        let teacher_answers: Vec<TeacherAnswer> = teachers
            .iter()
            .zip(answers)
            .map(|(teacher, probs)| {
                let pin = teacher.pin();
                TeacherAnswer {
                    id: pin.id.clone(),
                    version: pin.version.clone(),
                    tos_class: pin.tos_class.as_str().to_owned(),
                    probs,
                }
            })
            .collect();
        let distributable = teachers
            .iter()
            .all(|teacher| teacher.pin().tos_class.is_distributable());
        Self {
            id: item.id.clone(),
            family: item.family.as_str().to_owned(),
            group: item.group.clone(),
            kind: item.question.kind.as_str().to_owned(),
            blake3: item.hash(),
            seed: item.seed,
            rotation: item.rotation,
            paraphrase: item.paraphrase,
            state: item.state.canonical_text(),
            instructions: item
                .question
                .instructions
                .as_ref()
                .map(Entry::canonical_text),
            labels: item.question.labels(),
            rubrics,
            noul_criteria,
            teachers: teacher_answers,
            distributable,
            provenance,
        }
    }

    /// Reconstruct the jev-1 question spec this row was answered against.
    pub fn question_spec(&self) -> Option<QuestionSpec> {
        let kind = QuestionKind::from_tag(&self.kind)?;
        let body = match kind {
            QuestionKind::Noul => {
                let criteria = self
                    .noul_criteria
                    .clone()
                    .map(|(on_true, on_false)| NoulCriteria {
                        on_true: on_true.map(Entry::Str),
                        on_false: on_false.map(Entry::Str),
                    });
                QuestionBody::Noul { criteria }
            }
            QuestionKind::Choice => {
                let options = self
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(i, name)| ChoiceOption {
                        name: name.clone(),
                        rubric: self.rubrics.get(i).cloned().flatten().map(Entry::Str),
                    })
                    .collect();
                QuestionBody::Choice { options }
            }
            QuestionKind::Score => {
                let levels = self
                    .rubrics
                    .iter()
                    .map(|rubric| rubric.clone().map(Entry::Str))
                    .collect();
                QuestionBody::Score { levels }
            }
            _ => return None,
        };
        Some(QuestionSpec {
            id: self.id.clone(),
            kind,
            instructions: self.instructions.clone().map(Entry::Str),
            body,
        })
    }

    /// Cardinality of the question.
    pub fn cardinality(&self) -> usize {
        self.labels.len()
    }

    /// One JSON line.
    pub fn to_json_line(&self) -> String {
        match serde_json::to_string(self) {
            Ok(line) => line,
            Err(_) => unreachable!("corpus row serialization is infallible"),
        }
    }

    /// Parse one JSONL row.
    pub fn from_json_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

/// One label-histogram row: the JSON-compatible projection of the internal
/// `(kind, cardinality, label)` → count map (serde_json object keys must be
/// strings, so the tuple-keyed map cannot serialize as JSON).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelCount {
    /// Question kind (noul/choice/score).
    pub kind: String,
    /// Cardinality of the question.
    pub cardinality: usize,
    /// The label index.
    pub label: usize,
    /// Accepted rows carrying this label.
    pub count: usize,
}

impl LabelCount {
    /// Deterministically ordered rows from the internal tuple-keyed
    /// histogram (sorted by kind, then cardinality, then label).
    pub(crate) fn rows_from(
        histogram: &std::collections::HashMap<(String, usize, usize), usize>,
    ) -> Vec<Self> {
        let mut rows: Vec<Self> = histogram
            .iter()
            .map(|((kind, cardinality, label), count)| Self {
                kind: kind.clone(),
                cardinality: *cardinality,
                label: *label,
                count: *count,
            })
            .collect();
        rows.sort_by(|a, b| {
            (&a.kind, a.cardinality, a.label).cmp(&(&b.kind, b.cardinality, b.label))
        });
        rows
    }
}

/// Run statistics (also the audit trail for label control + dedup).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthStats {
    /// Items generated before any filtering.
    pub generated: usize,
    /// Rows accepted into the corpus.
    pub accepted: usize,
    /// Items dropped as exact canonical-hash duplicates.
    pub dedup_dropped: usize,
    /// Items deferred by label-distribution control.
    pub label_deferred: usize,
    /// The final label histogram as JSON-compatible rows (sorted by kind,
    /// cardinality, label) — the audit trail for label control + dedup.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub label_histogram: Vec<LabelCount>,
}

/// A synthesized corpus plus run statistics.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    /// Accepted rows, in generation order.
    pub rows: Vec<CorpusRow>,
    /// Run statistics from the pipeline that produced this corpus (empty for
    /// a parsed corpus).
    pub stats: SynthStats,
}

impl Corpus {
    /// Rows cleared for publication / permissive artifacts (distributable
    /// teachers only). Encumbered rows never appear here.
    pub fn publishable(&self) -> impl Iterator<Item = &CorpusRow> {
        self.rows.iter().filter(|row| row.distributable)
    }

    /// Serialize rows as JSONL. `publishable_only` drops encumbered rows.
    pub fn to_jsonl(&self, publishable_only: bool) -> String {
        let mut out = String::new();
        for row in &self.rows {
            if publishable_only && !row.distributable {
                continue;
            }
            let _ = writeln!(out, "{}", row.to_json_line());
        }
        out
    }

    /// Parse a JSONL corpus.
    pub fn from_jsonl(text: &str) -> Result<Self, serde_json::Error> {
        let mut rows = Vec::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            rows.push(CorpusRow::from_json_line(line)?);
        }
        Ok(Self {
            rows,
            stats: SynthStats::default(),
        })
    }
}

/// Whether a ToS class wire string parses back to a known class (used by
/// consumers validating rows).
pub fn parse_tos_class(wire: &str) -> Option<TosClass> {
    match wire {
        "open-weights" => Some(TosClass::OpenWeights),
        "api-distillation-permitted" => Some(TosClass::ApiDistillationPermitted),
        "api-prohibited" => Some(TosClass::ApiProhibited),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::gen;
    use crate::teacher::{HashTeacher, TeacherPin};
    use crate::SynthFamily;

    fn sample_row(tos: TosClass) -> CorpusRow {
        let item = gen::generate(SynthFamily::Approvals, 1, 11, 0);
        let pin = TeacherPin {
            id: "sim".to_owned(),
            version: "simulated".to_owned(),
            tos_class: tos,
        };
        let teacher = HashTeacher::new(pin, 1, 1.0);
        let teachers: Vec<&dyn Teacher> = vec![&teacher];
        let answers = vec![teacher.answer(&item)];
        CorpusRow::from_item(
            &item,
            &teachers,
            answers,
            Provenance {
                generator: "test".to_owned(),
                benchmark_release: hyprstream_bench::RELEASE.to_owned(),
                manifest_blake3: "0".repeat(64),
            },
        )
    }

    #[test]
    fn row_roundtrip_jsonl_and_spec_reconstruction() {
        let row = sample_row(TosClass::OpenWeights);
        let parsed = CorpusRow::from_json_line(&row.to_json_line()).unwrap();
        assert_eq!(row, parsed);
        let spec = row.question_spec().unwrap();
        assert_eq!(spec.id, row.id);
        assert_eq!(spec.labels(), row.labels);
    }

    #[test]
    fn encumbered_teacher_marks_row_undistributable() {
        assert!(sample_row(TosClass::OpenWeights).distributable);
        assert!(sample_row(TosClass::ApiDistillationPermitted).distributable);
        assert!(!sample_row(TosClass::ApiProhibited).distributable);
    }

    #[test]
    fn publishable_excludes_encumbered_rows() {
        let corpus = Corpus {
            rows: vec![
                sample_row(TosClass::OpenWeights),
                sample_row(TosClass::ApiProhibited),
            ],
            ..Default::default()
        };
        assert_eq!(corpus.publishable().count(), 1);
        assert_eq!(corpus.to_jsonl(true).lines().count(), 1);
        assert_eq!(corpus.to_jsonl(false).lines().count(), 2);
    }

    #[test]
    fn tos_class_wire_forms_roundtrip() {
        for class in [
            TosClass::OpenWeights,
            TosClass::ApiDistillationPermitted,
            TosClass::ApiProhibited,
        ] {
            assert_eq!(parse_tos_class(class.as_str()), Some(class));
        }
        assert_eq!(parse_tos_class("unknown"), None);
    }
}
