//! End-to-end gates for the frozen release: determinism, the committed
//! manifest, permutation closure, and jev-1 contract conformance of every
//! generated item.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use hyprstream_bench::family::Family;
use hyprstream_bench::gen::{generate_all, BenchConfig};
use hyprstream_bench::manifest::{blake3_hex, build, Manifest};
use hyprstream_bench::{Stratum, RELEASE};
use hyprstream_decision::QuestionKind;

fn small_config() -> BenchConfig {
    BenchConfig {
        seeds_per_stratum: 3,
        seed_base: 0x06,
    }
}

#[test]
fn generation_is_bit_deterministic() {
    let config = small_config();
    let a = generate_all(&config);
    let b = generate_all(&config);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.canonical_bytes(), y.canonical_bytes());
    }
}

#[test]
fn every_item_conforms_to_the_jev1_contract() {
    for item in generate_all(&small_config()) {
        let cardinality = item.question.cardinality();
        assert!((2..=255).contains(&cardinality), "{}: cardinality", item.id);
        assert!(item.truth < cardinality, "{}: truth in range", item.id);
        assert_eq!(item.question.id, item.id, "{}: question id", item.id);
        // Identifier-safe ids (they flow into Arrow field names downstream).
        assert!(
            item.id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{}: identifier-safe",
            item.id
        );
        assert_eq!(
            item.truth_distribution().iter().sum::<f32>(),
            1.0,
            "{}: one-hot",
            item.id
        );
    }
}

#[test]
fn permutation_stratum_is_closed_and_score_is_never_permuted() {
    let items = generate_all(&small_config());
    let mut groups: std::collections::HashMap<&str, Vec<&hyprstream_bench::Item>> =
        std::collections::HashMap::new();
    for item in &items {
        groups.entry(item.group.as_str()).or_default().push(item);
    }
    for item in &items {
        if item.question.kind == QuestionKind::Choice && item.stratum.is_base() {
            let group = &groups[item.group.as_str()];
            let k = item.question.cardinality();
            assert_eq!(group.len(), k, "{}: group has all k rotations", item.id);
            // Every rotation carries the same correct label text.
            let label = item.truth_label();
            for rotated in group {
                assert_eq!(rotated.truth_label(), label, "{}: truth rotates", rotated.id);
                let mut a = item.question.labels();
                let mut b = rotated.question.labels();
                a.sort();
                b.sort();
                assert_eq!(a, b, "{}: same option multiset", rotated.id);
            }
        }
        if item.question.kind == QuestionKind::Score {
            assert!(
                item.stratum.is_base(),
                "{}: score items are never permuted (level order is semantics)",
                item.id
            );
        }
    }
}

#[test]
fn truth_labels_mix_within_strata() {
    // A degenerate generator (all-true, all-argmax-0) would be a broken
    // benchmark. Base strata must mix truth values.
    let items = generate_all(&small_config());
    for family in Family::ALL {
        for base in [Stratum::Clean, Stratum::NearMiss] {
            let truths: Vec<usize> = items
                .iter()
                .filter(|i| i.family == family && i.stratum == base)
                .map(|i| i.truth)
                .collect();
            assert!(
                truths.iter().any(|t| *t != truths[0]),
                "{family}/{base}: truth labels must mix"
            );
        }
    }
}

#[test]
fn committed_manifest_matches_regeneration() {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "manifest",
        &format!("{RELEASE}.manifest.json"),
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let manifest = Manifest::from_json(&text).unwrap();
    manifest.verify(&BenchConfig::default()).unwrap();

    // The disclosure text is pinned by digest inside the manifest.
    let disclosure = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("DISCLOSURE.md"),
    )
    .unwrap();
    assert_eq!(blake3_hex(&disclosure), manifest.disclosure.blake3);
}

#[test]
fn built_manifest_is_byte_stable() {
    let config = small_config();
    let items = generate_all(&config);
    let a = build(&config, &items, "0".repeat(64)).to_json();
    let b = build(&config, &items, "0".repeat(64)).to_json();
    assert_eq!(a, b);
}
