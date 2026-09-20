//! Known-value tests for the metric primitives.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_calibration::metrics::{
    ace, argmax, brier, ece, expected_value, nll, reliability_curve, rps, sce,
};
use hyprstream_calibration::MetricError;

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn brier_known_values() {
    // Perfect prediction.
    let p: &[&[f64]] = &[&[1.0, 0.0]];
    approx(brier(p, &[0]).unwrap(), 0.0);
    // Uniform binary, label 0: (0.5-1)^2 + (0.5-0)^2 = 0.5.
    let p: &[&[f64]] = &[&[0.5, 0.5]];
    approx(brier(p, &[0]).unwrap(), 0.5);
}

#[test]
fn rps_hand_computed() {
    // p = [0.25, 0.5, 0.25], y = 0:
    // F0 = 0.25 vs 1 -> 0.5625; F1 = 0.75 vs 1 -> 0.0625; sum 0.625; /(K-1)=2 -> 0.3125.
    let p: &[&[f64]] = &[&[0.25, 0.5, 0.25]];
    approx(rps(p, &[0]).unwrap(), 0.3125);
    // Width-2 RPS equals half the multiclass Brier score.
    let p: &[&[f64]] = &[&[0.7, 0.3]];
    let r = rps(p, &[1]).unwrap();
    let b = brier(p, &[1]).unwrap();
    approx(r, b / 2.0);
    // Perfect ordinal prediction scores 0.
    let p: &[&[f64]] = &[&[0.0, 1.0, 0.0]];
    approx(rps(p, &[1]).unwrap(), 0.0);
}

#[test]
fn nll_and_expected_value() {
    let p: &[&[f64]] = &[&[0.5, 0.5]];
    approx(nll(p, &[0]).unwrap(), std::f64::consts::LN_2);
    approx(expected_value(&[0.25, 0.5, 0.25]), 1.0);
    approx(expected_value(&[1.0, 0.0, 0.0]), 0.0);
}

#[test]
fn ece_zero_when_calibrated() {
    // All observations at confidence 0.7 with exactly 70% correct.
    let probs: Vec<&[f64]> = (0..100).map(|_| &[0.7, 0.3][..]).collect();
    let labels: Vec<usize> = (0..100).map(|i| usize::from(i >= 70)).collect();
    let e = ece(&probs, &labels, 15).unwrap();
    assert!(e.abs() < 1e-9, "calibrated set must have ECE 0, got {e}");
    // All observations at confidence 0.9 with 50% correct: ECE = 0.4.
    let probs: Vec<&[f64]> = (0..100).map(|_| &[0.9, 0.1][..]).collect();
    let labels: Vec<usize> = (0..100).map(|i| usize::from(i < 50)).collect();
    let e = ece(&probs, &labels, 15).unwrap();
    assert!((e - 0.4).abs() < 1e-9, "expected ECE 0.4, got {e}");
}

#[test]
fn reliability_curve_populates_bins() {
    // Two clusters of confidence, each perfectly calibrated.
    let mut owned: Vec<[f64; 2]> = Vec::new();
    let mut labels = Vec::new();
    for i in 0..200 {
        if i < 100 {
            owned.push([0.6, 0.4]);
            labels.push(usize::from(i >= 60));
        } else {
            owned.push([0.9, 0.1]);
            labels.push(usize::from(i >= 190));
        }
    }
    let probs: Vec<&[f64]> = owned.iter().map(|p| &p[..]).collect();
    let bins = reliability_curve(&probs, &labels, 2).unwrap();
    assert_eq!(bins.len(), 2);
    for b in &bins {
        assert!(
            (b.mean_confidence - b.mean_outcome).abs() < 1e-9,
            "bin {b:?} must be calibrated"
        );
    }
}

#[test]
fn sce_ace_zero_when_classwise_calibrated() {
    // Two classes; for each class the confidence equals the empirical frequency.
    let probs: Vec<&[f64]> = (0..100).map(|_| &[0.6, 0.4][..]).collect();
    let labels: Vec<usize> = (0..100).map(|i| usize::from(i >= 60)).collect();
    let s = sce(&probs, &labels, 5).unwrap();
    let a = ace(&probs, &labels, 5).unwrap();
    assert!(s.abs() < 1e-9, "SCE {s}");
    assert!(a.abs() < 1e-9, "ACE {a}");
}

#[test]
fn argmax_ties_break_earliest() {
    assert_eq!(argmax(&[0.5, 0.5, 0.0]), 0);
    assert_eq!(argmax(&[0.1, 0.5, 0.4]), 1);
}

#[test]
fn validation_errors() {
    // Probabilities must sum to 1 within consumer tolerance.
    let p: &[&[f64]] = &[&[0.9, 0.9]];
    assert!(matches!(
        brier(p, &[0]),
        Err(MetricError::InvalidDistribution { .. })
    ));
    // Label out of range.
    let p: &[&[f64]] = &[&[0.5, 0.5]];
    assert!(matches!(
        brier(p, &[2]),
        Err(MetricError::InvalidLabel { .. })
    ));
    // Negative probability.
    let p: &[&[f64]] = &[&[1.5, -0.5]];
    assert!(matches!(
        brier(p, &[0]),
        Err(MetricError::InvalidDistribution { .. })
    ));
    // Length mismatch.
    let p: &[&[f64]] = &[&[0.5, 0.5]];
    assert!(matches!(
        brier(p, &[]),
        Err(MetricError::LengthMismatch { .. })
    ));
    // Zero-probability realized label -> infinite NLL is an error, not an inf.
    let p: &[&[f64]] = &[&[1.0, 0.0]];
    assert!(matches!(
        nll(p, &[1]),
        Err(MetricError::InvalidDistribution { .. })
    ));
}
