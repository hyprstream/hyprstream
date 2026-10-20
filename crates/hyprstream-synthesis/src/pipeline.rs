//! The synthesis pipeline: families × kinds × seeds × (mandatory) paraphrase
//! variants × (mandatory, choice-only) cyclic permutations → dedup →
//! firewall → teacher ensemble → label control → corpus rows.

use std::collections::{HashMap, HashSet};

use hyprstream_decision::QuestionKind;

use crate::corpus::{Corpus, CorpusRow, LabelCount, Provenance, SynthStats};
use crate::ensemble::{argmax_label, corrected_average, valid_probabilities};
use crate::family::SynthFamily;
use crate::firewall::{Firewall, FirewallViolation};
use crate::gen::{generate, item_seed, PARAPHRASES};
use crate::item::SynthItem;
use crate::labelctl::{LabelController, LabelPolicy};
use crate::teacher::Teacher;

/// Pipeline configuration. Deterministic for a fixed `(seed_base, roster)`:
/// same config + same teachers ⇒ bit-identical corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthConfig {
    /// Seed stream base (distinct tag space from the benchmark's).
    pub seed_base: u64,
    /// Base items per (family, kind) before augmentation expansion.
    pub base_items_per_family: u32,
    /// Paraphrase variants per base item, in `2..=PARAPHRASES` — paraphrase
    /// augmentation is **mandatory** (S6b1), so 1 is a config error.
    pub paraphrase_variants: u32,
    /// Label-distribution policy.
    pub label_policy: LabelPolicy,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            seed_base: 0x51A1,
            base_items_per_family: 16,
            paraphrase_variants: 2,
            label_policy: LabelPolicy::Unlimited,
        }
    }
}

/// Pipeline errors.
#[derive(Debug)]
pub enum SynthError {
    /// `paraphrase_variants` outside `2..=PARAPHRASES`.
    BadParaphraseVariants(u32),
    /// `base_items_per_family` of 0 (a zero-sized run would write an empty
    /// corpus while reporting success).
    ZeroBaseItems,
    /// Balanced label control requires a positive slack.
    ZeroLabelSlack,
    /// No teacher targets can be produced from an empty roster.
    EmptyRoster,
    /// Empty id/version cannot identify an auditable teacher.
    BadTeacherIdentity(String),
    /// Two roster teachers share an id.
    DuplicateTeacherId(String),
    /// A correction temperature referenced a teacher id outside the roster.
    UnknownCorrectionTeacher(String),
    /// A correction temperature was non-finite or non-positive.
    BadCorrectionTemperature(String),
    /// A generated item hit a firewall (fail-closed: the run aborts — a
    /// contaminated generator must be fixed, not filtered around).
    Firewall(FirewallViolation),
    /// A teacher returned a malformed distribution (wrong width or outside
    /// the producer sum tolerance).
    BadTeacherAnswer {
        /// Teacher roster id.
        teacher: String,
        /// Item id.
        item: String,
    },
}

impl std::fmt::Display for SynthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadParaphraseVariants(n) => write!(
                f,
                "paraphrase_variants must be in 2..={PARAPHRASES} (paraphrase augmentation is mandatory, S6b1); got {n}"
            ),
            Self::ZeroLabelSlack => f.write_str("balanced label_slack must be at least 1"),
            Self::EmptyRoster => f.write_str("teacher roster must not be empty"),
            Self::BadTeacherIdentity(id) => write!(f, "teacher identity must contain nonempty id/version: {id:?}"),
            Self::ZeroBaseItems => write!(
                f,
                "base_items_per_family must be at least 1 (zero-sized run)"
            ),
            Self::DuplicateTeacherId(id) => {
                write!(f, "duplicate teacher roster id {id}")
            }
            Self::UnknownCorrectionTeacher(id) => {
                write!(f, "correction temperature references non-roster teacher {id}")
            }
            Self::BadCorrectionTemperature(id) => {
                write!(
                    f,
                    "correction temperature for teacher {id} is not finite and positive"
                )
            }
            Self::Firewall(violation) => write!(f, "firewall violation: {violation}"),
            Self::BadTeacherAnswer { teacher, item } => write!(
                f,
                "teacher {teacher} returned a malformed distribution for item {item}"
            ),
        }
    }
}

impl std::error::Error for SynthError {}

/// The production per-item teacher gate, also exercised with a call counter.
/// Rechecks content immediately before inference, even if an earlier dedup
/// preflight already checked it; no future caller can bypass the firewall.
pub(crate) fn answer_checked_item(
    item: &crate::SynthItem,
    teachers: &[&dyn Teacher],
    firewall: &Firewall,
) -> Result<Vec<Vec<f32>>, SynthError> {
    firewall.check_item(item).map_err(SynthError::Firewall)?;
    let mut answers = Vec::with_capacity(teachers.len());
    for teacher in teachers {
        let probs = teacher.answer(item);
        if !valid_probabilities(&probs, item.question.cardinality()) {
            return Err(SynthError::BadTeacherAnswer {
                teacher: teacher.pin().id.clone(),
                item: item.id.clone(),
            });
        }
        answers.push(probs);
    }
    Ok(answers)
}

/// First-seen exact typed request content wins, independently of artifact
/// coordinates. Artifact hashes remain unchanged for provenance.
fn admit_unique_content(seen: &mut HashSet<String>, item: &SynthItem) -> Result<bool, SynthError> {
    let fingerprint = item.content_fingerprint().map_err(|error| {
        SynthError::Firewall(FirewallViolation::InvalidContent(error.to_string()))
    })?;
    Ok(seen.insert(fingerprint))
}

/// Run the pipeline. `teachers` is the roster order; every item is answered
/// by every teacher and all raw vectors are persisted. `temperatures` is the
/// fitted P0.5 correction map (roster id → temperature): label control must
/// balance against the **same corrected average that training later
/// distills**, so the map is applied to the balancing calculation as well —
/// pass an empty map only when no corrections have been fitted (e.g. the
/// simulated CLI dry-run).
pub fn run(
    config: &SynthConfig,
    teachers: &[&dyn Teacher],
    firewall: &Firewall,
    temperatures: &HashMap<String, f64>,
) -> Result<Corpus, SynthError> {
    if matches!(config.label_policy, LabelPolicy::Balanced { slack: 0 }) {
        return Err(SynthError::ZeroLabelSlack);
    }
    firewall
        .ensure_content_index()
        .map_err(SynthError::Firewall)?;
    if !(2..=PARAPHRASES).contains(&config.paraphrase_variants) {
        return Err(SynthError::BadParaphraseVariants(
            config.paraphrase_variants,
        ));
    }
    if config.base_items_per_family == 0 {
        return Err(SynthError::ZeroBaseItems);
    }
    // Roster ids key the persisted per-teacher vectors and the correction
    // map; a duplicate id would make rows and corrections indistinguishable.
    if teachers.is_empty() {
        return Err(SynthError::EmptyRoster);
    }
    let mut roster_ids = HashSet::new();
    for teacher in teachers {
        if teacher.pin().id.trim().is_empty() || teacher.pin().version.trim().is_empty() {
            return Err(SynthError::BadTeacherIdentity(teacher.pin().id.clone()));
        }
        if !roster_ids.insert(teacher.pin().id.as_str()) {
            return Err(SynthError::DuplicateTeacherId(teacher.pin().id.clone()));
        }
    }
    // Corrections must name roster teachers and be usable: a stray id or a
    // non-finite/non-positive temperature would otherwise surface mid-run (or
    // never, on a run whose rows all defer) instead of failing fast.
    for (id, temperature) in temperatures {
        if !roster_ids.contains(id.as_str()) {
            return Err(SynthError::UnknownCorrectionTeacher(id.clone()));
        }
        if !temperature.is_finite() || *temperature <= 0.0 {
            return Err(SynthError::BadCorrectionTemperature(id.clone()));
        }
    }
    let provenance = Provenance {
        generator: format!("hyprstream-synthesis {}", env!("CARGO_PKG_VERSION")),
        benchmark_release: firewall.release().to_owned(),
        manifest_blake3: firewall.manifest_blake3().to_owned(),
    };
    let mut corpus = Corpus::default();
    let mut stats = SynthStats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut labels = LabelController::default();

    for family in SynthFamily::ALL {
        for kind in 0..3u64 {
            for index in 0..config.base_items_per_family {
                let seed = item_seed(config.seed_base, family, kind, index);
                for paraphrase in 0..config.paraphrase_variants {
                    let base = generate(family, kind, seed, paraphrase);
                    // Mandatory cyclic permutation closure for choice items
                    // (S6b1): every rotation is emitted, linked by group.
                    let mut variants = vec![base.clone()];
                    if base.question.kind == QuestionKind::Choice {
                        for rotation in 1..base.question.cardinality() {
                            if let Some(rotated) = base.cyclic_permutation(rotation) {
                                variants.push(rotated);
                            }
                        }
                    }
                    for item in variants {
                        stats.generated += 1;
                        firewall.check_item(&item).map_err(SynthError::Firewall)?;
                        if !admit_unique_content(&mut seen, &item)? {
                            stats.dedup_dropped += 1;
                            continue;
                        }
                        let answers = answer_checked_item(&item, teachers, firewall)?;
                        let row =
                            CorpusRow::from_item(&item, teachers, answers, provenance.clone());
                        // The label histogram is recorded under every policy
                        // (the audit trail must not depend on whether control
                        // is on); only the deferral is policy-gated. The mean
                        // is the temperature-corrected average — the same
                        // distribution training distills — so a corpus reported
                        // as balanced is balanced under its actual training
                        // targets.
                        let mean = corrected_average(&row, temperatures).map_err(|_| {
                            SynthError::BadTeacherAnswer {
                                teacher: "ensemble".to_owned(),
                                item: item.id.clone(),
                            }
                        })?;
                        let label = argmax_label(&mean);
                        if !labels.accept(config.label_policy, &row.kind, row.cardinality(), label)
                        {
                            stats.label_deferred += 1;
                            continue;
                        }
                        corpus.rows.push(row);
                        stats.accepted += 1;
                    }
                }
            }
        }
    }
    stats.label_histogram = LabelCount::rows_from(labels.histogram());
    stats.correction_temperatures = temperatures
        .iter()
        .map(|(teacher, t)| (teacher.clone(), *t))
        .collect();
    corpus.stats = stats;
    Ok(corpus)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod content_dedup_tests {
    use super::*;
    use hyprstream_decision::{ChoiceOption, Entry, QuestionBody};

    fn item() -> SynthItem {
        SynthItem::new(
            SynthFamily::Triage,
            7,
            0,
            0,
            Entry::Map(vec![
                ("first".into(), Entry::Number(1.0)),
                ("second".into(), Entry::Number(2.0)),
            ]),
            Some("Choose the owner.".into()),
            QuestionBody::Choice {
                options: vec![
                    ChoiceOption {
                        name: "zebra".into(),
                        rubric: None,
                    },
                    ChoiceOption {
                        name: "alpha".into(),
                        rubric: Some(Entry::Str("Escalation".into())),
                    },
                ],
            },
        )
    }

    #[test]
    fn content_dedup_ignores_artifact_coordinates() {
        let original = item();
        let mut duplicate = original.clone();
        duplicate.id = "other_artifact".into();
        duplicate.question.id = "other_question".into();
        duplicate.group = "other_group".into();
        duplicate.seed += 1;
        duplicate.rotation = 1;
        duplicate.paraphrase = 1;
        assert_ne!(original.hash(), duplicate.hash());
        let mut seen = HashSet::new();
        assert!(admit_unique_content(&mut seen, &original).unwrap());
        assert!(!admit_unique_content(&mut seen, &duplicate).unwrap());
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn content_dedup_preserves_typed_request_distinctions() {
        let original = item();
        let mut seen = HashSet::new();
        assert!(admit_unique_content(&mut seen, &original).unwrap());
        for change in [
            "state-value",
            "state-order",
            "instructions",
            "absent-instructions",
            "null-instructions",
            "option-order",
            "rubric",
            "null-rubric",
        ] {
            let mut changed = original.clone();
            match change {
                "state-value" => changed.state = Entry::Str("different state".into()),
                "state-order" => {
                    let Entry::Map(entries) = &mut changed.state else {
                        unreachable!()
                    };
                    entries.reverse();
                }
                "instructions" => {
                    changed.question.instructions =
                        Some(Entry::Str("Reworded instructions.".into()));
                }
                "absent-instructions" => changed.question.instructions = None,
                "null-instructions" => changed.question.instructions = Some(Entry::Null),
                "option-order" | "rubric" | "null-rubric" => {
                    let QuestionBody::Choice { options } = &mut changed.question.body else {
                        unreachable!()
                    };
                    match change {
                        "option-order" => options.reverse(),
                        "rubric" => options[1].rubric = Some(Entry::Str("Changed rubric".into())),
                        "null-rubric" => options[0].rubric = Some(Entry::Null),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
            assert!(
                admit_unique_content(&mut seen, &changed).unwrap(),
                "{change}"
            );
        }
        let mut score = original;
        score.question.kind = QuestionKind::Score;
        score.question.body = QuestionBody::Score {
            levels: vec![None, Some(Entry::Str("High".into()))],
        };
        assert!(admit_unique_content(&mut seen, &score).unwrap());
        for change in ["null-level", "level-order", "level-rubric"] {
            let mut changed = score.clone();
            let QuestionBody::Score { levels } = &mut changed.question.body else {
                unreachable!()
            };
            match change {
                "null-level" => levels[0] = Some(Entry::Null),
                "level-order" => levels.reverse(),
                "level-rubric" => levels[1] = Some(Entry::Str("Changed level".into())),
                _ => unreachable!(),
            }
            assert!(
                admit_unique_content(&mut seen, &changed).unwrap(),
                "{change}"
            );
        }
    }

    #[test]
    fn content_dedup_fails_closed_on_invalid_typed_content() {
        let mut seen = HashSet::new();
        let mut invalid = item();
        invalid.state = Entry::Number(f64::NAN);
        assert!(matches!(
            admit_unique_content(&mut seen, &invalid),
            Err(SynthError::Firewall(FirewallViolation::InvalidContent(_)))
        ));
        let mut invalid = item();
        invalid.question.kind = QuestionKind::Noul;
        assert!(matches!(
            admit_unique_content(&mut seen, &invalid),
            Err(SynthError::Firewall(FirewallViolation::InvalidContent(_)))
        ));
        assert!(seen.is_empty());
    }
}
