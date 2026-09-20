//! # hyprstream-bench — verifiable-outcome calibration benchmark (System One, P0.6)
//!
//! The program's **truth anchor**: score-bearing items with mechanically
//! verifiable outcomes, emitted as jev-1 question specs
//! ([`hyprstream_decision`]). Everything is **procedurally generated fresh**
//! from published generators and pinned seeds ([`gen`]) — no public corpora
//! in the gating strata, so the benchmark is immune to teacher-pretraining
//! contamination by construction.
//!
//! ## What the release pins
//!
//! - **Generators + seeds** ([`gen`], [`rng::BenchRng`] — a frozen splitmix64
//!   stream). Same config ⇒ bit-identical items, forever. The PRNG, the
//!   canonical item bytes ([`Item::canonical_bytes`]), and the family
//!   taxonomy are all part of the frozen contract.
//! - **Adversarial strata** ([`Stratum`]): `clean`, `nearmiss` (near-miss
//!   JSON, near-miss arithmetic, close distractors), reported per stratum by
//!   the P0.2 measurement protocol.
//! - **Cyclic-permutation stratum** (CircularEval precedent, S6b1): every
//!   generated *choice* item is additionally emitted under all `k−1` non-
//!   identity cyclic option rotations, linked by [`Item::group`], truth
//!   rotating with the options — so option-order robustness (flip rate, ECE
//!   drift) is measured at zero retraining cost. Score levels are **not**
//!   permuted: level order is the semantics there. Noul has no options.
//! - **Family-level holdout firewall** ([`Family::designation`],
//!   [`Manifest::gate_families`]): `temporal` and `syllogism` are designated
//!   **gate** families — never synthesized into training data; the P1.4
//!   leg-(e) zero-shot transfer gate measures the model on exactly these.
//! - **Frozen item manifest** ([`manifest`], `manifest/vob-1.0.manifest.json`):
//!   per-item blake3 hashes, frozen before any P1.3 synthesis run. P1.3
//!   consumes the manifest as its item-level contamination firewall and its
//!   family firewall list.
//!
//! ## Licensing
//!
//! Harness: Apache-2.0. Generated items: CC-BY-4.0 (the default public
//! release shape per the program licensing policy). See `DISCLOSURE.md` for
//! the pre-committed disclosure language that accompanies every published
//! result.

pub mod family;
pub mod gen;
pub mod item;
pub mod manifest;
pub mod rng;

pub use family::{Designation, Family, Stratum};
pub use gen::{generate_all, BenchConfig, BASE_STRATA, DEFAULT_SEEDS_PER_STRATUM, SEED_BASE};
pub use item::Item;
pub use manifest::{Manifest, ManifestItem, VerifyError, RELEASE};
