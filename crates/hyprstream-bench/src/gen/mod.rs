//! Per-family item generators. Every generator is a pure function of
//! `(seed, stratum)` over the pinned [`BenchRng`] stream — published
//! generators + seeds make every item reproducible and immune to
//! teacher-pretraining contamination.

pub mod arith;
pub mod jsoncheck;
pub mod syllogism;
pub mod temporal;
pub mod unitconv;

use hyprstream_decision::QuestionKind;

use crate::family::{Family, Stratum};
use crate::item::Item;
use crate::rng::BenchRng;

/// Benchmark generation configuration. The frozen release pins
/// [`DEFAULT_SEEDS_PER_STRATUM`] and [`SEED_BASE`]; regenerating with the same
/// values must reproduce the committed manifest bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchConfig {
    /// Base items per (family, base stratum). Each base item is one question
    /// kind — total base items = families × strata × 3 kinds × this count.
    pub seeds_per_stratum: u32,
    /// Seed stream base; per-item seeds are `SEED_BASE + family * K + ...`.
    pub seed_base: u64,
}

/// Seeds per (family, base stratum, kind) in the frozen vob-1.0 release.
pub const DEFAULT_SEEDS_PER_STRATUM: u32 = 64;

/// Seed base for the frozen vob-1.0 release.
pub const SEED_BASE: u64 = 0x06;

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            seeds_per_stratum: DEFAULT_SEEDS_PER_STRATUM,
            seed_base: SEED_BASE,
        }
    }
}

/// The base strata every family emits (permutation expansion is mechanical
/// and applied afterwards to choice items only).
pub const BASE_STRATA: [Stratum; 2] = [Stratum::Clean, Stratum::NearMiss];

/// Deterministic per-(family, stratum, kind, index) seed.
fn item_seed(config: &BenchConfig, family: Family, stratum: Stratum, kind: u64, index: u32) -> u64 {
    let family_tag = Family::ALL
        .iter()
        .position(|f| *f == family)
        .map_or(0, |position| position as u64);
    let stratum_tag = u64::from(stratum == Stratum::NearMiss);
    config.seed_base
        ^ (family_tag << 48)
        ^ (stratum_tag << 44)
        ^ (kind << 36)
        ^ u64::from(index)
}

/// Generate the full benchmark: all families × base strata × kinds × seeds,
/// plus the cyclic-permutation expansion of every choice item.
pub fn generate_all(config: &BenchConfig) -> Vec<Item> {
    let mut items = Vec::new();
    for family in Family::ALL {
        for stratum in BASE_STRATA {
            for index in 0..config.seeds_per_stratum {
                let base = generate_family(family, stratum, config, index);
                for item in base {
                    if item.question.kind == QuestionKind::Choice {
                        let n = item.question.cardinality();
                        for rotation in 1..n {
                            if let Some(permuted) = item.cyclic_permutation(rotation) {
                                items.push(permuted);
                            }
                        }
                    }
                    items.push(item);
                }
            }
        }
    }
    items
}

/// Generate one (kind-triple) of base items for a family at `index`.
fn generate_family(
    family: Family,
    stratum: Stratum,
    config: &BenchConfig,
    index: u32,
) -> Vec<Item> {
    let mut triple = Vec::with_capacity(3);
    for (kind_tag, build) in [
        (
            0u64,
            match family {
                Family::Arith => arith::noul as fn(u64, Stratum) -> Item,
                Family::JsonCheck => jsoncheck::noul,
                Family::UnitConv => unitconv::noul,
                Family::Temporal => temporal::noul,
                Family::Syllogism => syllogism::noul,
            },
        ),
        (
            1,
            match family {
                Family::Arith => arith::choice,
                Family::JsonCheck => jsoncheck::choice,
                Family::UnitConv => unitconv::choice,
                Family::Temporal => temporal::choice,
                Family::Syllogism => syllogism::choice,
            },
        ),
        (
            2,
            match family {
                Family::Arith => arith::score,
                Family::JsonCheck => jsoncheck::score,
                Family::UnitConv => unitconv::score,
                Family::Temporal => temporal::score,
                Family::Syllogism => syllogism::score,
            },
        ),
    ] {
        let seed = item_seed(config, family, stratum, kind_tag, index);
        triple.push(build(seed, stratum));
    }
    triple
}

/// Convenience: a keyed stream for one item.
pub(crate) fn stream(seed: u64) -> BenchRng {
    BenchRng::new(seed)
}
