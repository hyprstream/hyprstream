//! Protocol, joint-coverage, bootstrap, and disclosure tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_calibration::disclosure::{disclosure_with, DISCLOSURE};
use hyprstream_calibration::protocol::{
    macro_average, FieldMetrics, GateReport, MacroMetrics, ShiftSplit, GATE_BIN_COUNT,
    GATE_BOOTSTRAP_RESAMPLES, GATE_CI_LEVEL,
};
use hyprstream_calibration::{bootstrap_ci, joint_coverage, MetricError, SplitMix64};

#[test]
fn protocol_constants_are_pinned() {
    assert_eq!(GATE_BIN_COUNT, 15);
    assert_eq!(GATE_BOOTSTRAP_RESAMPLES, 1000);
    assert!((GATE_CI_LEVEL - 0.95).abs() < f64::EPSILON);
}

#[test]
fn macro_rule_weights_fields_equally() {
    assert_eq!(macro_average(&[]), None);
    // Field counts do not matter here: the macro rule consumes per-field values only.
    let values = [0.02, 0.10];
    let m = macro_average(&values).unwrap();
    assert!((m - 0.06).abs() < 1e-12);
}

#[test]
fn bootstrap_is_seeded_and_deterministic() {
    let values: Vec<f64> = (0..200).map(|i| (i as f64 * 0.37).sin()).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (obs_a, ci_a) = bootstrap_ci(&values, mean, 500, 0.95, 42).unwrap();
    let (obs_b, ci_b) = bootstrap_ci(&values, mean, 500, 0.95, 42).unwrap();
    assert_eq!(ci_a, ci_b, "same seed must reproduce the same CI");
    assert_eq!(obs_a, obs_b);
    assert!(ci_a.low <= obs_a && obs_a <= ci_a.high);
    // A different seed almost surely changes the CI.
    let (_, ci_c) = bootstrap_ci(&values, mean, 500, 0.95, 43).unwrap();
    assert!(ci_a != ci_c);
}

#[test]
fn splitmix64_is_deterministic_and_varied() {
    let mut a = SplitMix64::new(7);
    let mut b = SplitMix64::new(7);
    let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
    let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
    assert_eq!(seq_a, seq_b);
    let unique: std::collections::HashSet<u64> = seq_a.iter().copied().collect();
    assert_eq!(unique.len(), 8);
}

#[test]
fn shift_split_rejects_family_overlap() {
    let split = ShiftSplit::new(["alpha", "beta"], ["gamma"]).unwrap();
    assert!(split.is_eval_family("gamma"));
    assert!(!split.is_eval_family("alpha"));
    let err = ShiftSplit::new(["alpha", "beta"], ["beta", "gamma"]);
    assert!(matches!(err, Err(MetricError::ShiftSplitOverlap { .. })));
}

#[test]
fn joint_coverage_reports_joint_and_independence_reference() {
    // 4 rows x 2 questions: marginals 3/4 each, joint 2/4, independence reference 9/16.
    let covered = vec![
        vec![true, true],
        vec![true, false],
        vec![false, true],
        vec![true, true],
    ];
    let jc = joint_coverage(&covered).unwrap();
    assert_eq!(jc.n, 4);
    assert!((jc.joint - 0.5).abs() < 1e-12);
    assert!((jc.marginals[0] - 0.75).abs() < 1e-12);
    assert!((jc.independence_reference - 0.5625).abs() < 1e-12);
    // Ragged matrix is rejected.
    assert!(matches!(
        joint_coverage(&[vec![true], vec![true, false]]),
        Err(MetricError::LengthMismatch { .. })
    ));
    assert!(joint_coverage(&[]).is_err());
}

#[test]
fn gate_report_applies_macro_rule_and_serializes() {
    let fields = vec![
        FieldMetrics {
            field: "q_choice".into(),
            family: "alpha".into(),
            n: 500,
            ece: 0.02,
            brier: 0.10,
            sce: None,
            ace: None,
            rps: None,
        },
        FieldMetrics {
            field: "q_score".into(),
            family: "beta".into(),
            n: 10,
            ece: 0.10,
            brier: 0.30,
            sce: Some(0.05),
            ace: Some(0.04),
            rps: Some(0.12),
        },
    ];
    let split = ShiftSplit::new(["alpha", "beta"], ["gamma"]).unwrap();
    let report =
        GateReport::assemble("subject-arm", "deterministic", split, fields, 1234).unwrap();
    // Macro means are field-equal: the n=10 field counts as much as the n=500 field.
    assert!((report.macro_ece.mean - 0.06).abs() < 1e-12);
    assert!((report.macro_brier.mean - 0.20).abs() < 1e-12);
    // Per-primitive aggregates cover only the fields that carry them.
    assert!((report.macro_sce.unwrap().mean - 0.05).abs() < 1e-12);
    assert!((report.macro_rps.unwrap().mean - 0.12).abs() < 1e-12);
    assert!(report.macro_ece.upper_bound() >= report.macro_ece.mean);
    // serde roundtrip: the report is a persistable artifact.
    let json = serde_json::to_string(&report).unwrap();
    let back: GateReport = serde_json::from_str(&json).unwrap();
    assert_eq!(report, back);
}

#[test]
fn macro_metrics_ci_covers_mean_on_stable_values() {
    let values: Vec<f64> = vec![0.03, 0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06];
    let m = MacroMetrics::from_fields(&values, 99).unwrap();
    assert!(m.ci.low <= m.mean && m.mean <= m.ci.high);
}

#[test]
fn disclosure_language_is_pre_committed() {
    // The five numbered sections must survive any edit verbatim in spirit; pin the
    // load-bearing phrases so a softening edit fails the build.
    for phrase in [
        "MECHANICALLY-CHECKABLE FAMILIES",
        "TEACHER-AGREEMENT SPLITS",
        "Calibration-to-teacher is NOT",
        "MARGINAL VS JOINT COVERAGE",
        "UNSEEN FAMILIES",
        "{subject}",
        "{telemetry_since}",
    ] {
        assert!(DISCLOSURE.contains(phrase), "disclosure lost {phrase:?}");
    }
    let rendered = disclosure_with("hyprstream-jev-0.1", "2026-09-20");
    assert!(!rendered.contains("{subject}"));
    assert!(rendered.contains("hyprstream-jev-0.1"));
    assert!(rendered.contains("2026-09-20"));
}
