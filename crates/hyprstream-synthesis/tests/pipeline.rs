//! End-to-end pipeline gates: determinism, mandatory augmentation closure,
//! provenance + distributability, raw-vector persistence, label control, and
//! the firewalls against the committed frozen vob-1.1 manifest.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hyprstream_decision::QuestionKind;
use hyprstream_synthesis::corpus::Corpus;
use hyprstream_synthesis::ensemble::{
    argmax_label, corrected_average, drop_worst_teachers, spread,
};
use hyprstream_synthesis::firewall::{Firewall, EXPECTED_MANIFEST_BLAKE3};
use hyprstream_synthesis::labelctl::LabelPolicy;
use hyprstream_synthesis::pipeline::{run, SynthConfig, SynthError};
use hyprstream_synthesis::teacher::{HashTeacher, Teacher, TeacherPin, TosClass};

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hyprstream-bench/manifest/vob-1.1.manifest.json")
}

fn firewall() -> Firewall {
    Firewall::load(&manifest_path()).expect("committed manifest loads")
}

fn pin(id: &str, tos: TosClass) -> TeacherPin {
    TeacherPin {
        id: id.to_owned(),
        version: "simulated".to_owned(),
        tos_class: tos,
    }
}

fn ensemble(specs: &[(&str, TosClass)]) -> (Vec<HashTeacher>, Vec<TeacherPin>) {
    let teachers = specs
        .iter()
        .enumerate()
        .map(|(i, (id, tos))| HashTeacher::new(pin(id, *tos), 100 + i as u64, 1.0))
        .collect::<Vec<_>>();
    let pins = teachers.iter().map(|t| t.pin().clone()).collect();
    (teachers, pins)
}

fn small_config() -> SynthConfig {
    SynthConfig {
        seed_base: 0x51A1,
        base_items_per_family: 4,
        paraphrase_variants: 2,
        label_policy: LabelPolicy::Unlimited,
    }
}

#[test]
fn pipeline_is_bit_deterministic() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let a = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    let b = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    assert_eq!(a.to_jsonl(false), b.to_jsonl(false));
    assert_eq!(a.stats, b.stats);
}

#[test]
fn label_histogram_is_recorded_under_unlimited_policy() {
    // Regression (review): unlimited runs used to skip histogram recording
    // entirely — the audit trail must exist under every policy.
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let config = SynthConfig {
        label_policy: LabelPolicy::Unlimited,
        ..small_config()
    };
    let corpus = run(&config, &refs, &firewall(), &HashMap::new()).unwrap();
    assert!(!corpus.stats.label_histogram.is_empty());
    let total: usize = corpus.stats.label_histogram.values().sum();
    assert_eq!(total, corpus.stats.accepted);
}

#[test]
fn reconstructed_specs_pass_the_arrow_identifier_contract() {
    // Regression (review): synthesized question ids double as jev-1 question
    // ids and flow into Arrow field names, so DecisionSchema's identifier
    // grammar ([A-Za-z_][A-Za-z0-9_]*) must accept every one of them.
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let corpus = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    assert!(!corpus.rows.is_empty());
    for row in &corpus.rows {
        let spec = row.question_spec().unwrap();
        hyprstream_decision::arrow::DecisionSchema::new(vec![spec])
            .unwrap_or_else(|err| panic!("{} rejected by DecisionSchema: {err}", row.id));
    }
}

#[test]
fn cli_rejects_malformed_config_values() {
    // Regression (review rounds): malformed config must fail loud — unknown
    // or non-string label_policy never silently selects Unlimited, and
    // out-of-range integers never wrap into a different run configuration.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir();
    let roster_path = dir.join("p13-test-roster.json");
    let config_path = dir.join("p13-test-config.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let run_with = |config: &str| -> std::process::Output {
        std::fs::write(&config_path, config).unwrap();
        std::process::Command::new(bin)
            .args([
                "run",
                "--roster",
                roster_path.to_str().unwrap(),
                "--manifest",
                manifest_path().to_str().unwrap(),
                "--config",
                config_path.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    for bad in [
        r#"[]"#,
        r#"null"#,
        r#""balanced""#,
        r#"42"#,
        r#"{"label_policy":"sometimes","base_items_per_family":1}"#,
        r#"{"label_policy":false,"base_items_per_family":1}"#,
        r#"{"label_policy":42,"base_items_per_family":1}"#,
        r#"{"label_policy":null,"base_items_per_family":1}"#,
        r#"{"base_items_per_family":4294967296}"#,
        r#"{"paraphrase_variants":4294967298}"#,
        r#"{"base_items_per_family":false}"#,
        r#"{"paraphrase_variants":-1}"#,
        r#"{"seed_base":"0x51A1"}"#,
        r#"{"label_policy":"balanced","label_slack":-1}"#,
        r#"{"base_item_per_family":5}"#,
        r#"{"base_items_per_family":0}"#,
        r#"{"unknown_key":1}"#,
        r#"{"label_policy":"unlimited","label_slack":"high"}"#,
        r#"{"label_slack":false}"#,
    ] {
        let out = run_with(bad);
        assert_eq!(
            out.status.code(),
            Some(1),
            "config {bad} must be rejected, stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let good = run_with(r#"{"label_policy":"unlimited","base_items_per_family":1}"#);
    assert!(
        good.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&good.stderr)
    );
}

#[test]
fn mandatory_augmentation_closure() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let config = small_config();
    let corpus = run(&config, &refs, &firewall(), &HashMap::new()).unwrap();
    let mut groups: HashMap<&str, Vec<&hyprstream_synthesis::CorpusRow>> = HashMap::new();
    for row in &corpus.rows {
        groups.entry(row.group.as_str()).or_default().push(row);
    }
    let mut saw_choice = false;
    for row in &corpus.rows {
        // Paraphrase variants exist for every base item.
        let group = &groups[row.group.as_str()];
        let paraphrases: HashSet<u32> = group.iter().map(|r| r.paraphrase).collect();
        assert_eq!(
            paraphrases.len(),
            config.paraphrase_variants as usize,
            "{}: full paraphrase set",
            row.group
        );
        if row.kind == "choice" {
            saw_choice = true;
            let k = row.cardinality();
            // Full cyclic closure per paraphrase variant.
            for p in 0..config.paraphrase_variants {
                let rotations: HashSet<u32> = group
                    .iter()
                    .filter(|r| r.paraphrase == p)
                    .map(|r| r.rotation)
                    .collect();
                assert_eq!(
                    rotations.len(),
                    k,
                    "{}-p{p}: all k rotations present",
                    row.group
                );
            }
            // Rotation preserves the option multiset and rubric pairing.
            let base = group
                .iter()
                .find(|r| r.rotation == 0 && r.paraphrase == row.paraphrase)
                .unwrap();
            let rotated = group
                .iter()
                .find(|r| r.rotation == 1 && r.paraphrase == row.paraphrase)
                .unwrap();
            let mut a: Vec<_> = base.labels.iter().zip(&base.rubrics).collect();
            let mut b: Vec<_> = rotated.labels.iter().zip(&rotated.rubrics).collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "rotation preserves option/rubric pairs");
        }
    }
    assert!(saw_choice);
}

#[test]
fn raw_teacher_vectors_are_persisted_with_provenance() {
    let (teachers, _) = ensemble(&[
        ("sim-a", TosClass::OpenWeights),
        ("sim-b", TosClass::ApiDistillationPermitted),
    ]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let corpus = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    assert!(!corpus.rows.is_empty());
    for row in &corpus.rows {
        assert_eq!(row.teachers.len(), 2, "{}: every teacher answered", row.id);
        for answer in &row.teachers {
            assert_eq!(answer.probs.len(), row.cardinality());
            assert!((answer.probs.iter().sum::<f32>() - 1.0).abs() <= 1e-6);
            assert_eq!(answer.version, "simulated");
        }
        assert_eq!(row.provenance.benchmark_release, "vob-1.1");
        assert_eq!(row.provenance.manifest_blake3, EXPECTED_MANIFEST_BLAKE3);
        // Spec reconstruction works for every row.
        let spec = row.question_spec().unwrap();
        assert_eq!(spec.labels(), row.labels);
        assert_ne!(spec.kind, QuestionKind::Span);
    }
}

#[test]
fn distributability_flag_gates_publishable_export() {
    let (teachers, _) = ensemble(&[
        ("sim-open", TosClass::OpenWeights),
        ("sim-encumbered", TosClass::ApiProhibited),
    ]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let corpus = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    assert!(!corpus.rows.is_empty());
    assert!(
        corpus.rows.iter().all(|row| !row.distributable),
        "any encumbered teacher encumbers the row"
    );
    assert_eq!(corpus.publishable().count(), 0);
    assert_eq!(corpus.to_jsonl(true).lines().count(), 0);
    assert!(corpus.to_jsonl(false).lines().count() > 0);
    // Round-trip through JSONL preserves the flag.
    let parsed = Corpus::from_jsonl(&corpus.to_jsonl(false)).unwrap();
    assert!(parsed.rows.iter().all(|row| !row.distributable));
}

#[test]
fn label_control_balances_argmax_histogram() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let config = SynthConfig {
        label_policy: LabelPolicy::Balanced { slack: 1 },
        ..small_config()
    };
    let corpus = run(&config, &refs, &firewall(), &HashMap::new()).unwrap();
    // Within each (kind, cardinality) bucket, label counts differ by at most
    // slack + 1 (deferral can leave the minimum trailing by one).
    let mut buckets: HashMap<(String, usize), Vec<usize>> = HashMap::new();
    for ((kind, cardinality, _label), count) in &corpus.stats.label_histogram {
        buckets
            .entry((kind.clone(), *cardinality))
            .or_default()
            .push(*count);
    }
    for (bucket, counts) in &buckets {
        let max = counts.iter().max().copied().unwrap_or(0);
        let min = counts.iter().min().copied().unwrap_or(0);
        assert!(
            max <= min + 2,
            "{bucket:?}: histogram {counts:?} out of balance"
        );
    }
}

#[test]
fn firewall_rejects_gate_families_and_manifest_collisions() {
    let fw = firewall();
    assert!(fw.gate_families().contains("temporal"));
    assert!(fw.gate_families().contains("syllogism"));
    // Every accepted row passes both firewalls (re-check against the raw set).
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let corpus = run(&small_config(), &refs, &fw, &HashMap::new()).unwrap();
    for row in &corpus.rows {
        assert!(!fw.gate_families().contains(&row.family));
        assert!(!fw.is_contaminated_hash(&row.blake3));
    }
}

#[test]
fn zero_base_items_and_duplicate_teacher_ids_are_rejected() {
    // Regression (review): the public pipeline API must fail loud on
    // zero-sized runs and on duplicate roster ids, not just the CLI parser.
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let zero = SynthConfig {
        base_items_per_family: 0,
        ..small_config()
    };
    assert!(matches!(
        run(&zero, &refs, &firewall(), &HashMap::new()),
        Err(SynthError::ZeroBaseItems)
    ));

    let (dupes, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let dupe_refs: Vec<&dyn Teacher> = vec![&dupes[0], &dupes[0]];
    assert!(matches!(
        run(&small_config(), &dupe_refs, &firewall(), &HashMap::new()),
        Err(SynthError::DuplicateTeacherId(id)) if id == "sim-a"
    ));
}

#[test]
fn paraphrase_variants_below_two_is_a_config_error() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let config = SynthConfig {
        paraphrase_variants: 1,
        ..small_config()
    };
    assert!(matches!(
        run(&config, &refs, &firewall(), &HashMap::new()),
        Err(SynthError::BadParaphraseVariants(1))
    ));
}

#[test]
fn ensemble_fallbacks_work_on_pipeline_rows() {
    let (teachers, _) = ensemble(&[
        ("sim-a", TosClass::OpenWeights),
        ("sim-b", TosClass::OpenWeights),
        ("sim-c", TosClass::OpenWeights),
    ]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let corpus = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    let row = corpus.rows.first().unwrap();
    let mean = corrected_average(row, &HashMap::new()).unwrap();
    assert!((mean.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    let _ = argmax_label(&mean);
    assert_eq!(spread(row).len(), row.cardinality());
    let kept = drop_worst_teachers(row, 2);
    assert_eq!(kept.len(), 2);
    for id in &kept {
        assert!(row.teachers.iter().any(|answer| &answer.id == id));
    }
}

#[test]
fn balancing_uses_the_fitted_correction_map() {
    // Regression (review): label control used to balance the uncorrected
    // average even though training distills the temperature-corrected one —
    // with non-unit P0.5 temperatures a corpus could be reported as balanced
    // while skewed under its actual training targets. The fitted correction
    // map is now threaded into the balancing calculation.
    let (teachers, _) = ensemble(&[
        ("sim-a", TosClass::OpenWeights),
        ("sim-b", TosClass::OpenWeights),
        ("sim-c", TosClass::OpenWeights),
    ]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();

    // An identity map (all T = 1.0) must be exactly the empty map.
    let identity: HashMap<String, f64> = ["sim-a", "sim-b", "sim-c"]
        .iter()
        .map(|id| ((*id).to_owned(), 1.0))
        .collect();
    let plain = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    let neutral = run(&small_config(), &refs, &firewall(), &identity).unwrap();
    assert_eq!(plain.to_jsonl(false), neutral.to_jsonl(false));
    assert_eq!(plain.stats, neutral.stats);

    // Strongly sharpening one teacher changes the ensemble argmax, so the
    // recorded labels must follow the corrected average. The expected
    // histogram is rebuilt independently from the persisted raw vectors.
    let sharpened = HashMap::from([("sim-a".to_owned(), 0.05)]);
    let corrected = run(&small_config(), &refs, &firewall(), &sharpened).unwrap();
    assert_eq!(corrected.rows.len(), plain.rows.len());
    let mut expected: HashMap<(String, usize, usize), usize> = HashMap::new();
    for row in &corrected.rows {
        let mean = corrected_average(row, &sharpened).unwrap();
        *expected
            .entry((row.kind.clone(), row.cardinality(), argmax_label(&mean)))
            .or_default() += 1;
    }
    assert_eq!(corrected.stats.label_histogram, expected);
    assert_ne!(
        corrected.stats.label_histogram, plain.stats.label_histogram,
        "sharpening a teacher must flip at least one recorded label"
    );
}

#[test]
fn invalid_correction_maps_are_rejected_upfront() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let ghost = HashMap::from([("ghost".to_owned(), 1.0)]);
    assert!(matches!(
        run(&small_config(), &refs, &firewall(), &ghost),
        Err(SynthError::UnknownCorrectionTeacher(id)) if id == "ghost"
    ));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let map = HashMap::from([("sim-a".to_owned(), bad)]);
        assert!(
            matches!(
                run(&small_config(), &refs, &firewall(), &map),
                Err(SynthError::BadCorrectionTemperature(id)) if id == "sim-a"
            ),
            "temperature {bad} must be rejected"
        );
    }
}

#[test]
fn cli_rejects_unknown_arguments() {
    // Regression (review): argument handling used to search for known flags
    // and ignore the rest, so a typo'd flag silently reverted to defaults —
    // a misspelled --publishable-only would silently export encumbered rows.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir();
    let roster_path = dir.join("p13-test-roster.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let roster = roster_path.to_str().unwrap();
    let manifest = manifest_path().to_str().unwrap().to_owned();
    for bad_args in [
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--confg",
            "tuned.json",
        ],
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--publishable",
        ],
        vec!["run", "--roster", roster, "--manifest", &manifest, "stray"],
        vec!["run", "--roster"],
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--out",
            "--publishable-only",
        ],
        vec!["roster-md", "--roster", roster, "--verbse"],
    ] {
        let out = std::process::Command::new(bin)
            .args(&bad_args)
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "args {bad_args:?} must be rejected, stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // The correctly-spelled flags still pass validation.
    let good = std::process::Command::new(bin)
        .args([
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--publishable-only",
        ])
        .output()
        .unwrap();
    assert!(
        good.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&good.stderr)
    );
}

#[test]
fn cli_fails_closed_when_disclosure_path_is_unresolvable() {
    // Regression (review): supplying the pinned manifest as a single-
    // component relative path made disclosure_path_for return None, and the
    // loader silently skipped disclosure verification — the exact pinned
    // bytes would load even with DISCLOSURE.md absent or drifted. The
    // unresolvable branch now fails closed with DisclosureIo.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir().join(format!("p13-fw-bare-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(manifest_path(), dir.join("vob-1.1.manifest.json")).unwrap();
    let roster_path = dir.join("roster.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let out = std::process::Command::new(bin)
        .current_dir(&dir)
        .args([
            "run",
            "--roster",
            roster_path.to_str().unwrap(),
            "--manifest",
            "vob-1.1.manifest.json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("DISCLOSURE"),
        "stderr must name the disclosure failure: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
