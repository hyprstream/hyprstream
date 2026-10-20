//! End-to-end pipeline gates: determinism, mandatory augmentation closure,
//! provenance + distributability, raw-vector persistence, label control, and
//! the firewalls against the committed frozen vob-1.1 manifest.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};

use crate::corpus::{Corpus, LabelCount, SynthStats};
use crate::ensemble::{argmax_label, corrected_average, drop_worst_teachers, spread};
use crate::firewall::Firewall;
use crate::labelctl::LabelPolicy;
use crate::pipeline::{run, SynthConfig, SynthError};
use crate::teacher::{HashTeacher, Teacher, TeacherPin, TosClass};
use hyprstream_decision::QuestionKind;

fn firewall() -> Firewall {
    Firewall::tiny_fixture()
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
    let total: usize = corpus
        .stats
        .label_histogram
        .iter()
        .map(|row| row.count)
        .sum();
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
fn mandatory_augmentation_closure() {
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let config = small_config();
    let corpus = run(&config, &refs, &firewall(), &HashMap::new()).unwrap();
    let mut groups: HashMap<&str, Vec<&crate::CorpusRow>> = HashMap::new();
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
        assert_eq!(row.provenance.benchmark_release, firewall().release());
        assert_eq!(row.provenance.manifest_blake3, firewall().manifest_blake3());
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
    for row in &corpus.stats.label_histogram {
        buckets
            .entry((row.kind.clone(), row.cardinality))
            .or_default()
            .push(row.count);
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
    assert_eq!(spread(row).unwrap().len(), row.cardinality());
    let kept = drop_worst_teachers(row, 2).unwrap();
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
    // The selection must be identical, while the audit trail now records
    // which temperatures were applied (identity vs none) — compare the
    // selection fields, not the whole stats struct.
    assert_eq!(plain.rows, neutral.rows);
    assert_eq!(plain.stats.label_histogram, neutral.stats.label_histogram);
    assert_eq!(plain.stats.accepted, neutral.stats.accepted);
    assert_eq!(plain.stats.label_deferred, neutral.stats.label_deferred);
    assert!(plain.stats.correction_temperatures.is_empty());
    let expected_identity: std::collections::BTreeMap<String, f64> = identity
        .iter()
        .map(|(teacher, temperature)| (teacher.clone(), *temperature))
        .collect();
    assert_eq!(neutral.stats.correction_temperatures, expected_identity);

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
    let got: HashMap<(String, usize, usize), usize> = corrected
        .stats
        .label_histogram
        .iter()
        .map(|row| ((row.kind.clone(), row.cardinality, row.label), row.count))
        .collect();
    assert_eq!(got, expected);
    assert_ne!(
        corrected.stats.label_histogram, plain.stats.label_histogram,
        "sharpening a teacher must flip at least one recorded label"
    );
}

#[test]
fn synth_stats_audit_trail_serializes_to_json() {
    // Regression (review thread 2026-09-22): the label histogram was a
    // tuple-keyed map, which serde_json cannot serialize (object keys must
    // be strings) — the advertised SynthStats audit trail was unpersistable.
    // Rows keep the audit trail JSON-compatible and deterministically
    // ordered.
    let stats = SynthStats {
        generated: 3,
        accepted: 3,
        label_histogram: vec![
            LabelCount {
                kind: "choice".to_owned(),
                cardinality: 2,
                label: 0,
                count: 2,
            },
            LabelCount {
                kind: "noul".to_owned(),
                cardinality: 2,
                label: 1,
                count: 1,
            },
        ],
        correction_temperatures: std::collections::BTreeMap::from([
            ("sim-a".to_owned(), 0.8),
            ("sim-b".to_owned(), 1.2),
        ]),
        ..SynthStats::default()
    };
    let text = serde_json::to_string(&stats).expect("audit trail serializes");
    assert!(text.contains("\"label_histogram\""));
    assert!(text.contains("\"correction_temperatures\""));
    let back: SynthStats = serde_json::from_str(&text).expect("audit trail round-trips");
    assert_eq!(back, stats);
}

#[test]
fn applied_correction_temperatures_are_recorded() {
    // Regression (review thread 2026-09-22 20:58): corrections change the
    // argmax the label policy defers on, but were absent from the audit
    // trail — two runs with identical configs and different corrections
    // produced different corpora with identical recorded provenance. The
    // applied temperatures are now part of SynthStats, so the selection
    // decision is reproducible from the corpus alone.
    let (teachers, _) = ensemble(&[("sim-a", TosClass::OpenWeights)]);
    let refs: Vec<&dyn Teacher> = teachers.iter().map(|t| t as &dyn Teacher).collect();
    let sharpened = HashMap::from([("sim-a".to_owned(), 0.05)]);
    let corpus = run(&small_config(), &refs, &firewall(), &sharpened).unwrap();
    assert_eq!(
        corpus.stats.correction_temperatures.get("sim-a"),
        Some(&0.05),
        "applied temperatures recorded verbatim"
    );
    let uncorrected = run(&small_config(), &refs, &firewall(), &HashMap::new()).unwrap();
    assert!(uncorrected.stats.correction_temperatures.is_empty());
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

struct CountTeacher {
    pin: TeacherPin,
    calls: std::cell::Cell<usize>,
    answer: Vec<f32>,
}
impl Teacher for CountTeacher {
    fn pin(&self) -> &TeacherPin {
        &self.pin
    }
    fn answer(&self, _: &crate::SynthItem) -> Vec<f32> {
        self.calls.set(self.calls.get() + 1);
        self.answer.clone()
    }
}
#[test]
fn balanced_zero_slack_rejects_before_admission_or_teacher_calls() {
    let teacher = CountTeacher {
        pin: pin("counter", TosClass::OpenWeights),
        calls: std::cell::Cell::new(0),
        answer: vec![0.5, 0.5],
    };
    let item = Firewall::tiny_item();
    let missing = Firewall::tiny_unattested(std::slice::from_ref(&item));
    let mut config = small_config();
    config.label_policy = LabelPolicy::Balanced { slack: 0 };
    assert!(matches!(
        run(&config, &[&teacher], &missing, &HashMap::new()),
        Err(SynthError::ZeroLabelSlack)
    ));
    assert_eq!(teacher.calls.get(), 0);
    // Valid policies pass configuration validation and reach the independent
    // missing-index gate, still without generating items or querying a teacher.
    for policy in [
        LabelPolicy::Unlimited,
        LabelPolicy::Balanced { slack: 1 },
        LabelPolicy::Balanced { slack: usize::MAX },
    ] {
        config.label_policy = policy;
        assert!(matches!(
            run(&config, &[&teacher], &missing, &HashMap::new()),
            Err(SynthError::Firewall(
                crate::firewall::FirewallViolation::MissingContentIndex
            ))
        ));
        assert_eq!(teacher.calls.get(), 0);
    }
}

#[test]
fn admission_is_before_teachers_and_teacher_probabilities_are_strict() {
    let t = CountTeacher {
        pin: pin("counter", TosClass::OpenWeights),
        calls: std::cell::Cell::new(0),
        answer: vec![0.5, 0.5],
    };
    let item = Firewall::tiny_item();
    let missing = Firewall::tiny_unattested(std::slice::from_ref(&item));
    assert!(matches!(
        run(&small_config(), &[&t], &missing, &HashMap::new()),
        Err(SynthError::Firewall(
            crate::firewall::FirewallViolation::MissingContentIndex
        ))
    ));
    assert_eq!(t.calls.get(), 0);
    for json in [
        "corrupt",
        r#"{"version":"decision-content/1","items":[]}"#,
        r#"{"version":"unknown","items":[]}"#,
    ] {
        let result =
            Firewall::tiny_unattested(std::slice::from_ref(&item)).with_content_index_json(json);
        assert!(result.is_err());
        assert_eq!(t.calls.get(), 0);
    }
    assert!(matches!(
        run(&small_config(), &[], &firewall(), &HashMap::new()),
        Err(SynthError::EmptyRoster)
    ));
    for probs in [
        vec![f32::NAN, 1.0],
        vec![1.0, f32::INFINITY],
        vec![f32::NEG_INFINITY, 0.0],
        vec![-0.1, 1.1],
        vec![0.0, 0.0],
        vec![1.0],
    ] {
        let t = CountTeacher {
            pin: pin("counter", TosClass::OpenWeights),
            calls: std::cell::Cell::new(0),
            answer: probs,
        };
        assert!(matches!(
            run(&small_config(), &[&t], &firewall(), &HashMap::new()),
            Err(SynthError::BadTeacherAnswer { .. })
        ));
        assert_eq!(t.calls.get(), 1);
    }
}

#[test]
fn matching_content_aborts_before_first_teacher_answer() {
    let reference = Firewall::tiny_item();
    let copied = crate::item::SynthItem::new(
        crate::family::SynthFamily::Triage,
        999,
        0,
        1,
        reference.state.clone(),
        reference
            .question
            .instructions
            .as_ref()
            .map(hyprstream_decision::Entry::canonical_text),
        reference.question.body.clone(),
    );
    let firewall = Firewall::tiny_unattested(std::slice::from_ref(&reference))
        .with_attested_items(&[reference])
        .unwrap();
    let teacher = CountTeacher {
        pin: pin("counter", TosClass::OpenWeights),
        calls: std::cell::Cell::new(0),
        answer: vec![0.5, 0.5],
    };
    assert!(matches!(
        crate::pipeline::answer_checked_item(&copied, &[&teacher], &firewall),
        Err(SynthError::Firewall(
            crate::firewall::FirewallViolation::ContaminatedItem { .. }
        ))
    ));
    assert_eq!(teacher.calls.get(), 0);
}
