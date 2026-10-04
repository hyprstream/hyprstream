//! One synthesized item: a jev-1 question spec plus its state, augmentation
//! coordinates, and the canonical byte form the dedup/firewall hash covers.

use std::fmt::Write as _;

use hyprstream_decision::{ChoiceOption, Entry, QuestionBody, QuestionKind, QuestionSpec};

use crate::family::SynthFamily;

/// One synthesized (state, question spec) pair at a specific augmentation
/// coordinate. Answers are not part of the item — they come from the teacher
/// ensemble and live in the corpus row.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthItem {
    /// Stable item id: `syn1_<family>_<kind>_<seed:016x>_r<rotation>_p<paraphrase>`.
    pub id: String,
    /// Synthetic workflow family.
    pub family: SynthFamily,
    /// Augmentation group: every (rotation, paraphrase) variant of one base
    /// item shares this id (the base item's id, `_r0_p0`).
    pub group: String,
    /// The raw seed this item was generated from (reproducibility handle).
    pub seed: u64,
    /// Cyclic option rotation applied (choice only; 0 otherwise).
    pub rotation: u32,
    /// Paraphrase variant index (0 = base phrasing).
    pub paraphrase: u32,
    /// Shared state the question refers to.
    pub state: Entry,
    /// The jev-1 question spec (`question.id == item.id`).
    pub question: QuestionSpec,
}

impl SynthItem {
    /// Build an item. `id`/`group` are derived from (family, kind, seed,
    /// rotation, paraphrase); the caller supplies the concrete texts.
    pub(crate) fn new(
        family: SynthFamily,
        seed: u64,
        rotation: u32,
        paraphrase: u32,
        state: Entry,
        instructions: Option<String>,
        body: QuestionBody,
    ) -> Self {
        let kind = match &body {
            QuestionBody::Noul { .. } => QuestionKind::Noul,
            QuestionBody::Choice { .. } => QuestionKind::Choice,
            QuestionBody::Score { .. } => QuestionKind::Score,
        };
        let base = base_id(family, kind, seed);
        let id = format!("{base}_r{rotation}_p{paraphrase}");
        let group = format!("{base}_r0_p0");
        let question = QuestionSpec {
            id: id.clone(),
            kind,
            instructions: instructions.map(Entry::Str),
            body,
        };
        Self {
            id,
            family,
            group,
            seed,
            rotation,
            paraphrase,
            state,
            question,
        }
    }

    /// The cyclic rotation of a **choice** item (mandatory S6b1 permutation
    /// augmentation). Options rotate left by `rotation` positions; the
    /// rubrics rotate with them. Returns `None` for non-choice questions
    /// (score levels are ordered — permutation would change the question —
    /// and noul has no options) and for rotation 0.
    pub fn cyclic_permutation(&self, rotation: usize) -> Option<SynthItem> {
        let QuestionBody::Choice { options } = &self.question.body else {
            return None;
        };
        let n = options.len();
        let rotation = rotation % n;
        if rotation == 0 {
            return None;
        }
        let rotated: Vec<ChoiceOption> = (0..n)
            .map(|i| options[(i + rotation) % n].clone())
            .collect();
        Some(SynthItem::new(
            self.family,
            self.seed,
            rotation as u32,
            self.paraphrase,
            self.state.clone(),
            self.question
                .instructions
                .as_ref()
                .map(Entry::canonical_text),
            QuestionBody::Choice { options: rotated },
        ))
    }

    /// Canonical byte form — the exact input the dedup and item-level
    /// contamination firewall hashes.
    ///
    /// `FROZEN (syn-item/1)`: length-prefixed fields, one per line, so the
    /// hash is independent of any serializer's key order. Note the distinct
    /// format tag: a synthesized item can never byte-collide with a `vob-item/1`
    /// benchmark item even if every text field matched — the item-level
    /// firewall is a *hash* check, so the formats must not overlap.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("syn-item/1\n");
        write_field(&mut out, "id", &self.id);
        write_field(&mut out, "family", self.family.as_str());
        write_field(&mut out, "group", &self.group);
        write_field(&mut out, "seed", &format!("{:016x}", self.seed));
        write_field(&mut out, "rotation", &self.rotation.to_string());
        write_field(&mut out, "paraphrase", &self.paraphrase.to_string());
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
        match &self.question.body {
            QuestionBody::Choice { options } => {
                for (index, option) in options.iter().enumerate() {
                    let rubric = option
                        .rubric
                        .as_ref()
                        .map(Entry::canonical_text)
                        .unwrap_or_default();
                    write_field(&mut out, &format!("rubric.{index}"), &rubric);
                }
            }
            QuestionBody::Score { levels } => {
                for (index, level) in levels.iter().enumerate() {
                    let rubric = level
                        .as_ref()
                        .map(Entry::canonical_text)
                        .unwrap_or_default();
                    write_field(&mut out, &format!("level.{index}"), &rubric);
                }
            }
            QuestionBody::Noul { criteria } => {
                let (on_true, on_false) = criteria
                    .as_ref()
                    .map(|c| (c.on_true.as_ref(), c.on_false.as_ref()))
                    .unwrap_or((None, None));
                let on_true = on_true.map(Entry::canonical_text).unwrap_or_default();
                let on_false = on_false.map(Entry::canonical_text).unwrap_or_default();
                write_field(&mut out, "criteria.true", &on_true);
                write_field(&mut out, "criteria.false", &on_false);
            }
        }
        out.into_bytes()
    }

    /// blake3 hex digest of [`Self::canonical_bytes`] — the dedup/firewall hash.
    pub fn hash(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

/// The base-item id shape (augmentation coordinates `_r0_p0` appended by
/// [`SynthItem::new`]). Ids are identifier-safe under the jev-1 Arrow
/// contract — `[A-Za-z_][A-Za-z0-9_]*`, the grammar
/// `hyprstream_decision::arrow::DecisionSchema` enforces for question ids
/// (they flow into Arrow field names, so no hyphens).
fn base_id(family: SynthFamily, kind: QuestionKind, seed: u64) -> String {
    format!("syn1_{}_{}_{seed:016x}", family.as_str(), kind.as_str())
}

fn write_field(out: &mut String, name: &str, value: &str) {
    // Length prefix in bytes so values may contain any character (including
    // newlines and colons) without ambiguity.
    let _ = write!(out, "{}={}:{};", name, value.len(), value);
    out.push('\n');
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_choice() -> SynthItem {
        SynthItem::new(
            SynthFamily::Triage,
            7,
            0,
            0,
            Entry::Str("Ticket: printer jammed.".into()),
            Some("Which team?".into()),
            QuestionBody::Choice {
                options: vec![
                    ChoiceOption {
                        name: "hardware".into(),
                        rubric: Some(Entry::Str("Physical device problems.".into())),
                    },
                    ChoiceOption {
                        name: "software".into(),
                        rubric: None,
                    },
                    ChoiceOption {
                        name: "billing".into(),
                        rubric: Some(Entry::Str("Invoices and refunds.".into())),
                    },
                ],
            },
        )
    }

    #[test]
    fn permutation_rotates_options_and_keeps_group() {
        let item = sample_choice();
        for rotation in 1..3 {
            let rotated = item.cyclic_permutation(rotation).unwrap();
            assert_eq!(rotated.group, item.group);
            assert_eq!(rotated.rotation, rotation as u32);
            let mut a = item.question.labels();
            let mut b = rotated.question.labels();
            a.sort();
            b.sort();
            assert_eq!(a, b);
            assert_ne!(rotated.question.labels(), item.question.labels());
        }
        assert!(item.cyclic_permutation(0).is_none());
        assert!(item.cyclic_permutation(3).is_none());
    }

    #[test]
    fn hash_is_stable_and_format_tagged() {
        let item = sample_choice();
        assert!(item.canonical_bytes().starts_with(b"syn-item/1\n"));
        // Regression lock: pinned golden digest (blake3 over canonical bytes).
        assert_eq!(
            item.hash(),
            "f5b751efcaa370fdca2fa1a7cd070228d3a2ee1cdfc0f31ed6df3836c29755f7"
        );
    }

    #[test]
    fn hash_changes_with_augmentation_coordinates() {
        let item = sample_choice();
        let rotated = item.cyclic_permutation(1).unwrap();
        assert_ne!(item.hash(), rotated.hash());
    }
}
