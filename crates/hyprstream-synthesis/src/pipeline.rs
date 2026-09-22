//! The synthesis pipeline: families × kinds × seeds × (mandatory) paraphrase
//! variants × (mandatory, choice-only) cyclic permutations → dedup →
//! firewall → teacher ensemble → label control → corpus rows.

use std::collections::{HashMap, HashSet};

use hyprstream_decision::QuestionKind;

use crate::corpus::{Corpus, CorpusRow, Provenance, SynthStats};
use crate::ensemble::{argmax_label, corrected_average};
use crate::family::SynthFamily;
use crate::firewall::{Firewall, FirewallViolation};
use crate::gen::{generate, item_seed, PARAPHRASES};
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
    /// Two roster teachers share an id.
    DuplicateTeacherId(String),
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
            Self::ZeroBaseItems => write!(
                f,
                "base_items_per_family must be at least 1 (zero-sized run)"
            ),
            Self::DuplicateTeacherId(id) => {
                write!(f, "duplicate teacher roster id {id}")
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

/// Producer-sum tolerance, mirrored from the jev-1 contract (D5).
const PRODUCER_SUM_TOLERANCE: f32 = 1e-6;

/// Run the pipeline. `teachers` is the roster order; every item is answered
/// by every teacher and all raw vectors are persisted.
pub fn run(
    config: &SynthConfig,
    teachers: &[&dyn Teacher],
    firewall: &Firewall,
) -> Result<Corpus, SynthError> {
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
    let mut roster_ids = HashSet::new();
    for teacher in teachers {
        if !roster_ids.insert(teacher.pin().id.as_str()) {
            return Err(SynthError::DuplicateTeacherId(teacher.pin().id.clone()));
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
                        if !seen.insert(item.hash()) {
                            stats.dedup_dropped += 1;
                            continue;
                        }
                        let mut answers = Vec::with_capacity(teachers.len());
                        for teacher in teachers {
                            let probs = teacher.answer(&item);
                            if probs.len() != item.question.cardinality()
                                || (probs.iter().sum::<f32>() - 1.0).abs() > PRODUCER_SUM_TOLERANCE
                            {
                                return Err(SynthError::BadTeacherAnswer {
                                    teacher: teacher.pin().id.clone(),
                                    item: item.id.clone(),
                                });
                            }
                            answers.push(probs);
                        }
                        let row =
                            CorpusRow::from_item(&item, teachers, answers, provenance.clone());
                        // The label histogram is recorded under every policy
                        // (the audit trail must not depend on whether control
                        // is on); only the deferral is policy-gated.
                        let mean = corrected_average(&row, &HashMap::new()).map_err(|_| {
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
    stats.label_histogram = labels.histogram().clone();
    corpus.stats = stats;
    Ok(corpus)
}
