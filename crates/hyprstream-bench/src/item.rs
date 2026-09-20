//! One benchmark item: a jev-1 question spec, its state, and the verifiable
//! truth — plus the canonical byte form the frozen manifest hashes.

use std::fmt::Write as _;

use hyprstream_decision::{ChoiceOption, Entry, QuestionBody, QuestionKind, QuestionSpec};

use crate::family::{Family, Stratum};

/// One score-bearing, verifiable-outcome item.
///
/// `truth` is the index of the correct label in the question's canonical
/// label order (`["false","true"]` for noul, option order for choice, level
/// order for score). The truth procedure is mechanical — every item is
/// generated together with its answer, immune to teacher-pretraining
/// contamination by construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// Stable item id: `vob1-<family>-<stratum>-<kind>-<seed:016x>`.
    pub id: String,
    /// Question family.
    pub family: Family,
    /// Difficulty/robustness stratum.
    pub stratum: Stratum,
    /// Permutation group: all cyclic rotations of one choice item share this
    /// id (the base item's id). Base items are their own group.
    pub group: String,
    /// The raw seed this item was generated from (reproducibility handle).
    pub seed: u64,
    /// Shared state the question refers to.
    pub state: Entry,
    /// The jev-1 question spec (`question.id == item.id`).
    pub question: QuestionSpec,
    /// Index of the correct label in canonical label order.
    pub truth: usize,
}

impl Item {
    /// Build a noul item.
    pub fn noul(
        family: Family,
        stratum: Stratum,
        seed: u64,
        state: Entry,
        instructions: impl Into<String>,
        truth: bool,
    ) -> Self {
        let id = item_id(family, stratum, QuestionKind::Noul, seed);
        let question = QuestionSpec {
            id: id.clone(),
            kind: QuestionKind::Noul,
            instructions: Some(Entry::Str(instructions.into())),
            body: QuestionBody::Noul { criteria: None },
        };
        Self {
            group: id.clone(),
            id,
            family,
            stratum,
            seed,
            state,
            question,
            truth: usize::from(truth),
        }
    }

    /// Build a choice item. `correct` is the index of the correct option.
    pub fn choice(
        family: Family,
        stratum: Stratum,
        seed: u64,
        state: Entry,
        instructions: impl Into<String>,
        options: Vec<String>,
        correct: usize,
    ) -> Self {
        assert!(options.len() >= 2, "jev-1 choice requires >= 2 options");
        assert!(correct < options.len(), "truth index out of range");
        let id = item_id(family, stratum, QuestionKind::Choice, seed);
        let question = QuestionSpec {
            id: id.clone(),
            kind: QuestionKind::Choice,
            instructions: Some(Entry::Str(instructions.into())),
            body: QuestionBody::Choice {
                options: options
                    .into_iter()
                    .map(|name| ChoiceOption {
                        name,
                        rubric: None,
                    })
                    .collect(),
            },
        };
        Self {
            group: id.clone(),
            id,
            family,
            stratum,
            seed,
            state,
            question,
            truth: correct,
        }
    }

    /// Build a score item over `levels` ordered rubric texts.
    pub fn score(
        family: Family,
        stratum: Stratum,
        seed: u64,
        state: Entry,
        instructions: impl Into<String>,
        levels: usize,
        correct_level: usize,
    ) -> Self {
        assert!(levels >= 2, "jev-1 score requires >= 2 levels");
        assert!(correct_level < levels, "truth level out of range");
        let id = item_id(family, stratum, QuestionKind::Score, seed);
        let question = QuestionSpec {
            id: id.clone(),
            kind: QuestionKind::Score,
            instructions: Some(Entry::Str(instructions.into())),
            body: QuestionBody::Score {
                levels: vec![None; levels],
            },
        };
        Self {
            group: id.clone(),
            id,
            family,
            stratum,
            seed,
            state,
            question,
            truth: correct_level,
        }
    }

    /// The correct label string in canonical label order.
    pub fn truth_label(&self) -> String {
        self.question
            .labels()
            .get(self.truth)
            .cloned()
            .unwrap_or_default()
    }

    /// One-hot truth distribution over the canonical labels.
    pub fn truth_distribution(&self) -> Vec<f32> {
        let mut distribution = vec![0.0; self.question.cardinality()];
        distribution[self.truth] = 1.0;
        distribution
    }

    /// The cyclic permutation of a **choice** item (CircularEval stratum).
    ///
    /// Options rotate left by `rotation` positions; the truth index rotates
    /// with them so the same option stays correct. Returns `None` for
    /// non-choice questions (score levels are ordered — permutation would
    /// change the question — and noul has no options) and for rotation 0.
    pub fn cyclic_permutation(&self, rotation: usize) -> Option<Item> {
        let QuestionBody::Choice { options } = &self.question.body else {
            return None;
        };
        let n = options.len();
        let rotation = rotation % n;
        if rotation == 0 || !self.stratum.is_base() {
            return None;
        }
        let rotated: Vec<ChoiceOption> = (0..n)
            .map(|i| options[(i + rotation) % n].clone())
            .collect();
        let mut item = self.clone();
        item.stratum = Stratum::Permuted {
            rotation: rotation as u32,
        };
        item.id = format!("{}-p{rotation}", self.id);
        item.group = self.id.clone();
        item.truth = (self.truth + n - rotation) % n;
        item.question = QuestionSpec {
            id: item.id.clone(),
            kind: QuestionKind::Choice,
            instructions: self.question.instructions.clone(),
            body: QuestionBody::Choice { options: rotated },
        };
        Some(item)
    }

    /// Canonical byte form — the exact input the manifest hashes.
    ///
    /// `FROZEN (vob-item/1)`: length-prefixed fields, one per line, so the
    /// hash is independent of any serializer's key order or float formatting.
    /// Changing anything here invalidates every published item hash.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("vob-item/1\n");
        write_field(&mut out, "id", &self.id);
        write_field(&mut out, "family", self.family.as_str());
        write_field(&mut out, "stratum", &self.stratum.as_str());
        write_field(&mut out, "group", &self.group);
        write_field(&mut out, "seed", &format!("{:016x}", self.seed));
        write_field(&mut out, "state", &self.state.canonical_text());
        write_field(&mut out, "kind", self.question.kind.as_str());
        match &self.question.instructions {
            Some(entry) => write_field(&mut out, "instructions", &entry.canonical_text()),
            None => write_field(&mut out, "instructions", ""),
        }
        let labels = self.question.labels();
        write_field(&mut out, "cardinality", &labels.len().to_string());
        for (index, label) in labels.iter().enumerate() {
            write_field(&mut out, &format!("label.{index}"), label);
        }
        if let QuestionBody::Choice { options } = &self.question.body {
            for (index, option) in options.iter().enumerate() {
                let rubric = option
                    .rubric
                    .as_ref()
                    .map(Entry::canonical_text)
                    .unwrap_or_default();
                write_field(&mut out, &format!("rubric.{index}"), &rubric);
            }
        }
        write_field(&mut out, "truth", &self.truth.to_string());
        out.into_bytes()
    }

    /// blake3 hex digest of [`Self::canonical_bytes`] — the manifest hash.
    pub fn hash(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

/// The item-id shape. Ids are identifier-safe (they double as jev-1 question
/// ids, which flow into Arrow field names).
fn item_id(family: Family, stratum: Stratum, kind: QuestionKind, seed: u64) -> String {
    format!(
        "vob1-{}-{}-{}-{seed:016x}",
        family.as_str(),
        stratum.as_str(),
        kind.as_str()
    )
}

fn write_field(out: &mut String, name: &str, value: &str) {
    // Length prefix in bytes so values may contain any character (including
    // newlines and colons) without ambiguity.
    let _ = write!(out, "{}={}:{};", name, value.len(), value);
    out.push('\n');
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample_choice() -> Item {
        Item::choice(
            Family::Arith,
            Stratum::Clean,
            7,
            Entry::Str("A crate holds 4 apples.".into()),
            "How many apples?",
            vec!["3".into(), "4".into(), "5".into()],
            1,
        )
    }

    #[test]
    fn permutation_rotates_truth_with_options() {
        let item = sample_choice();
        for rotation in 1..3 {
            let rotated = item.cyclic_permutation(rotation).unwrap();
            assert_eq!(rotated.truth_label(), item.truth_label());
            assert_eq!(rotated.group, item.id);
            assert_ne!(rotated.question.labels(), item.question.labels());
        }
        assert!(item.cyclic_permutation(0).is_none());
        assert!(item.cyclic_permutation(3).is_none());
    }

    #[test]
    fn score_and_noul_are_not_permutable() {
        let score = Item::score(
            Family::Arith,
            Stratum::Clean,
            7,
            Entry::Null,
            "How large?",
            4,
            2,
        );
        assert!(score.cyclic_permutation(1).is_none());
        let noul = Item::noul(
            Family::Arith,
            Stratum::Clean,
            7,
            Entry::Null,
            "Is it 4?",
            true,
        );
        assert!(noul.cyclic_permutation(1).is_none());
    }

    #[test]
    fn hash_is_stable() {
        let item = sample_choice();
        // Regression lock: pinned golden digest (blake3 over canonical bytes).
        assert_eq!(
            item.hash(),
            "764f41e7ebd698307176671b7114369ff009d4659989bcaba8ef3e1858630344"
        );
    }
}
