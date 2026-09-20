//! Teacher ensemble: roster, ToS provenance, raw per-teacher vectors,
//! corrected-ensemble stand-in (the average — corrections land in P0.5), and
//! argmax agreement against it.
//!
//! Per the program plan (P1.3/P0.5 contract): teacher identity is
//! configuration with provenance, **raw per-teacher probability vectors are
//! persisted** (corrections fit in P0.5 must be re-appliable at training
//! time), and every teacher carries a [`TosClass`] so encumbered rows are
//! never published and never enter Apache artifacts.

use hyprstream_decision::arrow::AnswerRow;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::QuestionSet;

use crate::error::EvalError;
use crate::subject::Subject;

/// Distribution-license class of a teacher's outputs, per the out-of-band
/// roster decision (plan Q4). Recorded per teacher at roster time; the
/// harness refuses to construct an ensemble with an unclassified teacher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TosClass {
    /// Outputs may be redistributed in public/Apache artifacts.
    Distributable,
    /// Outputs usable for training internally but never published.
    InternalOnly,
    /// Class not yet determined — treated as the most restrictive class for
    /// any publication decision.
    Unknown,
}

impl TosClass {
    /// Wire form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Distributable => "distributable",
            Self::InternalOnly => "internal-only",
            Self::Unknown => "unknown",
        }
    }
}

/// One teacher: a named subject plus its provenance.
pub struct Teacher {
    /// Stable roster id (recorded in DISCLOSURE / provenance rows).
    pub id: String,
    /// ToS class from the roster decision.
    pub tos_class: TosClass,
    /// The subject answering on this teacher's behalf.
    pub subject: Box<dyn Subject>,
}

/// One teacher's raw answer to one item, with provenance. This is the
/// persisted unit — corrections are never baked in here.
#[derive(Debug, Clone)]
pub struct TeacherAnswer {
    /// Roster id.
    pub teacher_id: String,
    /// ToS class at run time.
    pub tos_class: TosClass,
    /// Resolved model id the teacher's endpoint reported.
    pub model_id: String,
    /// The raw answer row (distributions as emitted).
    pub row: AnswerRow,
}

/// A roster of teachers queried in lockstep over the same items.
pub struct TeacherEnsemble {
    teachers: Vec<Teacher>,
}

/// The ensemble's answers to one item: every teacher's raw row, the plain
/// average distribution per question (the corrected-ensemble stand-in until
/// P0.5 fits corrections), and per-question argmax agreement.
#[derive(Debug, Clone)]
pub struct EnsembleOutput {
    /// Raw per-teacher answers, in roster order.
    pub teacher_answers: Vec<TeacherAnswer>,
    /// Per-question average distribution (question id → mean over teachers).
    pub average: std::collections::BTreeMap<String, Vec<f32>>,
    /// Per-question fraction of teachers whose argmax label matches the
    /// average distribution's argmax (D6 tie-breaks, earliest first).
    pub argmax_agreement: std::collections::BTreeMap<String, f64>,
}

impl TeacherEnsemble {
    /// Build an ensemble from a roster. A single-teacher roster is legal (the
    /// agreement metrics then degenerate to 1.0); an empty one is not.
    pub fn new(teachers: Vec<Teacher>) -> Result<Self, EvalError> {
        if teachers.is_empty() {
            return Err(EvalError::InvalidInput(
                "a teacher ensemble needs at least one teacher".to_owned(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for teacher in &teachers {
            if !ids.insert(teacher.id.clone()) {
                return Err(EvalError::InvalidInput(format!(
                    "duplicate teacher id `{}`",
                    teacher.id
                )));
            }
        }
        Ok(Self { teachers })
    }

    /// The roster, in order.
    pub fn teachers(&self) -> &[Teacher] {
        &self.teachers
    }

    /// Query every teacher for one item and combine the answers.
    pub async fn decide(
        &self,
        set: &QuestionSet,
        state: &Entry,
        row: usize,
        item_id: &str,
    ) -> Result<EnsembleOutput, EvalError> {
        let mut teacher_answers = Vec::with_capacity(self.teachers.len());
        for teacher in &self.teachers {
            let answer_row =
                teacher
                    .subject
                    .decide(set, state, row)
                    .await
                    .map_err(|error| EvalError::Subject {
                        model: teacher.subject.model_id().to_owned(),
                        item_id: item_id.to_owned(),
                        message: error.to_string(),
                    })?;
            teacher_answers.push(TeacherAnswer {
                teacher_id: teacher.id.clone(),
                tos_class: teacher.tos_class,
                model_id: teacher.subject.model_id().to_owned(),
                row: answer_row,
            });
        }
        Ok(combine(set, teacher_answers))
    }
}

/// Combine raw teacher answers into the average + agreement view.
fn combine(set: &QuestionSet, teacher_answers: Vec<TeacherAnswer>) -> EnsembleOutput {
    let mut average = std::collections::BTreeMap::new();
    let mut argmax_agreement = std::collections::BTreeMap::new();
    for question in &set.questions {
        let cardinality = question.cardinality();
        // Abstentions count as disagreements and do not contribute to the
        // average (no distribution to average); if every teacher abstains the
        // average is the uniform distribution and agreement is 0.
        let mut sum = vec![0f64; cardinality];
        let mut answered = 0usize;
        let mut argmaxes = Vec::with_capacity(teacher_answers.len());
        for answer in &teacher_answers {
            let value = answer
                .row
                .answers
                .get(&question.id)
                .and_then(|qa| qa.value.as_ref());
            match value {
                Some(value) => {
                    let probs = value.probabilities();
                    for (acc, p) in sum.iter_mut().zip(&probs) {
                        *acc += f64::from(*p);
                    }
                    answered += 1;
                    argmaxes.push(hyprstream_decision::confidence::argmax_index(&probs));
                }
                None => argmaxes.push(None),
            }
        }
        let mean: Vec<f32> = if answered > 0 {
            #[allow(clippy::cast_precision_loss)]
            sum.iter().map(|s| (s / answered as f64) as f32).collect()
        } else {
            let uniform = 1.0 / cardinality as f32;
            vec![uniform; cardinality]
        };
        let ensemble_argmax = hyprstream_decision::confidence::argmax_index(&mean);
        let matches = argmaxes
            .iter()
            .filter(|argmax| **argmax == ensemble_argmax)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let agreement = matches as f64 / teacher_answers.len() as f64;
        average.insert(question.id.clone(), mean);
        argmax_agreement.insert(question.id.clone(), agreement);
    }
    EnsembleOutput {
        teacher_answers,
        average,
        argmax_agreement,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::subject::TruthSubject;
    use hyprstream_decision::author;

    fn fixture() -> QuestionSet {
        author::parse_yaml(
            r#"
questions:
  tone:
    type: choice
    criteria: { angry: "Hostile", calm: ~ }
"#,
        )
        .unwrap()
    }

    fn teacher(id: &str, truth: usize) -> Teacher {
        Teacher {
            id: id.to_owned(),
            tos_class: TosClass::Distributable,
            subject: Box::new(TruthSubject::new(id).with_truth("tone", truth)),
        }
    }

    #[tokio::test]
    async fn agreement_tracks_argmax_consensus() {
        let set = fixture();
        let ensemble =
            TeacherEnsemble::new(vec![teacher("a", 0), teacher("b", 0), teacher("c", 1)]).unwrap();
        let out = ensemble
            .decide(&set, &Entry::Null, 0, "item")
            .await
            .unwrap();
        // Ensemble argmax is option 0 (two one-hot votes to one); teachers a
        // and b match, c does not.
        let agreement = out.argmax_agreement["tone"];
        assert!((agreement - 2.0 / 3.0).abs() < 1e-12);
        let avg = &out.average["tone"];
        assert!((avg[0] - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(out.teacher_answers.len(), 3);
        assert_eq!(out.teacher_answers[0].tos_class, TosClass::Distributable);
    }

    #[tokio::test]
    async fn empty_roster_is_rejected() {
        assert!(TeacherEnsemble::new(vec![]).is_err());
    }
}
