//! The teacher ensemble interface: teacher identity is **configuration with
//! provenance** (plan v1.4), authorized by humans out-of-band. This crate
//! ships the trait, the roster pin, and deterministic stand-in teachers for
//! tests and dry runs — real API/self-hosted adapters implement [`Teacher`]
//! downstream (P0.4 harness, P1.4 training).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::item::SynthItem;

/// Terms-of-service class of a teacher (S2 spike). This flag gates corpus
/// row distributability — and, through the v1.7.1 licensing policy, whether
/// the trained flagship artifact can be released under MIT at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TosClass {
    /// Self-hosted open weights: distillation unconstrained. Rows are
    /// distributable.
    OpenWeights,
    /// API teacher whose terms permit distillation (S2: e.g. DeepSeek).
    /// Rows are distributable; the pin records the terms reference.
    ApiDistillationPermitted,
    /// API teacher whose terms prohibit distillation (S2: OpenAI /
    /// Anthropic / Google). Rows are **encumbered**: never published, never
    /// in Apache/MIT artifacts. Such a teacher may only serve evaluation or
    /// internal iteration.
    ApiProhibited,
}

impl TosClass {
    /// Wire form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenWeights => "open-weights",
            Self::ApiDistillationPermitted => "api-distillation-permitted",
            Self::ApiProhibited => "api-prohibited",
        }
    }

    /// Whether rows answered by this teacher may be published / shipped in
    /// permissive-licensed artifacts.
    pub fn is_distributable(self) -> bool {
        !matches!(self, Self::ApiProhibited)
    }
}

/// One roster entry: the provenance pin recorded on every corpus row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeacherPin {
    /// Roster id (e.g. `qwen3-32b-selfhost-b`, `deepseek-v3-api`).
    pub id: String,
    /// Resolved model/snapshot version (never an alias — aliases resolve at
    /// roster authoring time).
    pub version: String,
    /// Terms-of-service class.
    pub tos_class: TosClass,
}

/// A teacher: answers a synthesized item with a raw probability vector over
/// the question's canonical labels. Implementations must be deterministic
/// for a fixed (item, pin) or record their nondeterminism in the pin's
/// version — the corpus contract is that rows are re-auditable.
pub trait Teacher {
    /// The roster pin recorded on every answer.
    fn pin(&self) -> &TeacherPin;
    /// Raw probability vector, one entry per canonical label
    /// (`question.cardinality()`), producer-sum within `1e-6` of 1.
    fn answer(&self, item: &SynthItem) -> Vec<f32>;
}

/// A deterministic stand-in teacher: pseudo-logits derived from blake3 over
/// the item's canonical bytes keyed by the teacher seed, sharpened/flattened
/// by a per-teacher temperature. Used by tests and the CLI dry-run path;
/// clearly pinned `simulated` in the roster so simulated rows can never be
/// mistaken for real teacher output.
#[derive(Debug, Clone)]
pub struct HashTeacher {
    pin: TeacherPin,
    seed: u64,
    temperature: f64,
}

impl HashTeacher {
    /// Build a stand-in teacher. `temperature` in `(0, 10]`; 1.0 is neutral.
    pub fn new(pin: TeacherPin, seed: u64, temperature: f64) -> Self {
        assert!(
            temperature > 0.0 && temperature <= 10.0,
            "temperature out of range"
        );
        Self {
            pin,
            seed,
            temperature,
        }
    }
}

impl Teacher for HashTeacher {
    fn pin(&self) -> &TeacherPin {
        &self.pin
    }

    fn answer(&self, item: &SynthItem) -> Vec<f32> {
        let n = item.question.cardinality();
        let base = item.canonical_bytes();
        let mut logits = Vec::with_capacity(n);
        for label in 0..n {
            let mut keyed = base.clone();
            keyed.extend_from_slice(format!("teacher/{:016x}/{label}", self.seed).as_bytes());
            let digest = blake3::hash(&keyed);
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&digest.as_bytes()[..8]);
            let unit = u64::from_le_bytes(bytes) as f64 / 18446744073709551616.0;
            logits.push((unit + 1e-9).ln() / self.temperature);
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| (e / sum) as f32).collect()
    }
}

/// A teacher roster: the out-of-band human-authorized list of teachers with
/// their ToS classes, mirrored into `DISCLOSURE.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roster {
    /// Roster entries, in authorization order.
    pub teachers: Vec<TeacherPin>,
}

impl Roster {
    /// Whether every rostered teacher is distributable (the whole-corpus
    /// publishability shortcut; per-row flags remain authoritative).
    pub fn all_distributable(&self) -> bool {
        self.teachers
            .iter()
            .all(|pin| pin.tos_class.is_distributable())
    }

    /// The roster rendered as a Markdown table for `DISCLOSURE.md`.
    pub fn to_markdown_table(&self) -> String {
        let mut out =
            String::from("| teacher | version | ToS class | distributable |\n|---|---|---|---|\n");
        for pin in &self.teachers {
            let _ = writeln!(
                out,
                "| `{}` | `{}` | `{}` | {} |",
                pin.id,
                pin.version,
                pin.tos_class.as_str(),
                pin.tos_class.is_distributable()
            );
        }
        out
    }

    /// Parse a roster from JSON (`{"teachers": [{id, version, tos_class}, ...]}`).
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Pins keyed by id, for ensemble lookups.
    pub fn pins_by_id(&self) -> BTreeMap<&str, &TeacherPin> {
        self.teachers
            .iter()
            .map(|pin| (pin.id.as_str(), pin))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::family::SynthFamily;
    use crate::gen;

    fn simulated_pin(id: &str, tos: TosClass) -> TeacherPin {
        TeacherPin {
            id: id.to_owned(),
            version: "simulated".to_owned(),
            tos_class: tos,
        }
    }

    #[test]
    fn hash_teacher_is_deterministic_and_normalized() {
        let item = gen::generate(SynthFamily::Triage, 1, 99, 0);
        let teacher = HashTeacher::new(simulated_pin("sim-a", TosClass::OpenWeights), 1, 1.0);
        let a = teacher.answer(&item);
        let b = teacher.answer(&item);
        assert_eq!(a, b);
        assert_eq!(a.len(), item.question.cardinality());
        assert!((a.iter().sum::<f32>() - 1.0).abs() <= 1e-6);
        assert!(a.iter().all(|p| *p > 0.0));
    }

    #[test]
    fn distinct_teachers_disagree() {
        let item = gen::generate(SynthFamily::Compliance, 2, 5, 0);
        let a =
            HashTeacher::new(simulated_pin("sim-a", TosClass::OpenWeights), 1, 1.0).answer(&item);
        let b =
            HashTeacher::new(simulated_pin("sim-b", TosClass::OpenWeights), 2, 1.0).answer(&item);
        assert_ne!(a, b);
    }

    #[test]
    fn roster_json_roundtrip_and_table() {
        let roster = Roster {
            teachers: vec![
                simulated_pin("sim-a", TosClass::OpenWeights),
                simulated_pin("sim-b", TosClass::ApiProhibited),
            ],
        };
        let text = serde_json::to_string(&roster).unwrap();
        assert_eq!(Roster::from_json(&text).unwrap(), roster);
        assert!(!roster.all_distributable());
        let table = roster.to_markdown_table();
        assert!(table.contains("api-prohibited"));
        assert!(table.contains("| `sim-b` |"));
    }
}
