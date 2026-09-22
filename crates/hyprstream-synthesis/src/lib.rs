//! # hyprstream-synthesis — teacher-agnostic synthetic corpus pipeline (System One, P1.3)
//!
//! Generates **(state, question spec, answer) triples** for the decision-model
//! program, synthetic-first and **teacher-agnostic**: question specs are
//! procedurally synthesized across workflow families with rubric/option
//! paraphrase diversity, cardinality variation, and primitive mixes (noul /
//! choice / score — that diversity is what teaches generalization), and the
//! answers come from a pluggable [`teacher::Teacher`] ensemble whose identity
//! is configuration with provenance, never baked in.
//!
//! ## What the pipeline pins
//!
//! - **Mandatory augmentations (S6b1)** — option-order permutation (full
//!   cyclic closure for choice items, CircularEval precedent) **and**
//!   rubric/instruction paraphrase variants are always on
//!   ([`pipeline::SynthConfig`] validates `paraphrase_variants >= 2`; the
//!   cyclic closure is not configurable). Option reordering alone swings
//!   accuracy 13–75 pp otherwise; the load-bearing debiasing is data-side
//!   and ~free since we own the synthesizer.
//! - **Raw per-teacher probability vectors persisted**
//!   ([`corpus::TeacherAnswer`]) — corrections are fit later (P0.5) and
//!   applied at training time (P1.4), so provenance alone is not enough: the
//!   raw vectors are the corpus. [`ensemble`] re-derives the corrected
//!   teacher-average at consumption time, with the named fallback (drop
//!   worst-agreement teachers, use ensemble spread).
//! - **Provenance + distributability flags** — every row carries the teacher
//!   roster pins (id, version, [`teacher::TosClass`]) it was answered by.
//!   Rows answered by an encumbered teacher are flagged
//!   `distributable = false`: never published, never in Apache/MIT artifacts
//!   ([`corpus::Corpus::publishable`]).
//! - **Contamination firewalls** ([`firewall`]) — the frozen **vob-1.1**
//!   benchmark manifest is a consumed artifact, not a promise: the loader
//!   verifies the pinned release id and blake3 digests (manifest file +
//!   DISCLOSURE text) and the pipeline rejects any synthesized item that
//!   hits a gate family or collides with a frozen item hash.
//! - **Label-distribution control + dedup** ([`labelctl`]) — optional
//!   balancing over ensemble-argmax labels, and canonical-hash dedup of
//!   emitted rows.
//!
//! ## No human labeling
//!
//! No row in this corpus is human-labeled. Truth, where it exists, lives in
//! the verifiable-outcome benchmark (`hyprstream-bench`); this corpus carries
//! teacher distributions only, and the pre-committed disclosure language
//! (`DISCLOSURE.md`) states plainly that calibration-to-teacher ≠
//! calibration-to-reality.

pub mod corpus;
pub mod ensemble;
pub mod family;
pub mod firewall;
pub mod gen;
pub mod item;
pub mod labelctl;
pub mod pipeline;
pub mod teacher;

pub use corpus::{Corpus, CorpusRow, LabelCount, Provenance, SynthStats, TeacherAnswer};
pub use family::SynthFamily;
pub use firewall::{Firewall, FirewallError, FirewallViolation};
pub use item::SynthItem;
pub use labelctl::LabelPolicy;
pub use pipeline::{run, SynthConfig, SynthError};
pub use teacher::{HashTeacher, Teacher, TeacherPin, TosClass};

/// The pinned splitmix64 stream, shared with the benchmark harness so the
/// two generators can never silently diverge in stream semantics.
pub use hyprstream_bench::rng::BenchRng;
