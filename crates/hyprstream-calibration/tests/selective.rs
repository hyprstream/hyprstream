//! Selective-prediction curve tests.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_calibration::{accuracy_at_coverage, risk_coverage_curve, selective_auc};

#[test]
fn risk_coverage_perfect_ranker() {
    // Confidence perfectly separates correct from incorrect answers.
    let confs = [0.9, 0.8, 0.4, 0.3];
    let correct = [true, true, false, false];
    let curve = risk_coverage_curve(&confs, &correct).unwrap();
    assert_eq!(curve.len(), 4);
    // At 50% coverage only the correct answers are kept.
    let acc = accuracy_at_coverage(&curve, 0.5).unwrap();
    assert!((acc - 1.0).abs() < 1e-12);
    // Full coverage gives overall accuracy 0.5.
    let last = curve.last().unwrap();
    assert!((last.coverage - 1.0).abs() < 1e-12);
    assert!((last.accuracy - 0.5).abs() < 1e-12);
}

#[test]
fn selective_auc_bounds() {
    let confs = [0.9, 0.8, 0.4, 0.3];
    let correct = [true, true, false, false];
    let curve = risk_coverage_curve(&confs, &correct).unwrap();
    let auc = selective_auc(&curve);
    assert!(auc > 0.5 && auc <= 1.0, "auc {auc}");
    assert_eq!(selective_auc(&[]), 0.0);
}

#[test]
fn accuracy_at_coverage_step_semantics() {
    let confs = [0.9, 0.7, 0.2];
    let correct = [true, false, true];
    let curve = risk_coverage_curve(&confs, &correct).unwrap();
    // Below the smallest realized coverage there is no operating point.
    assert_eq!(accuracy_at_coverage(&curve, 0.1), None);
    // Between points the largest realized coverage <= target is used.
    let acc = accuracy_at_coverage(&curve, 0.5).unwrap();
    assert!((acc - 1.0).abs() < 1e-12); // coverage 1/3 point
}

#[test]
fn selective_validation() {
    assert!(risk_coverage_curve(&[0.5], &[]).is_err());
    assert!(risk_coverage_curve(&[], &[]).is_err());
}
