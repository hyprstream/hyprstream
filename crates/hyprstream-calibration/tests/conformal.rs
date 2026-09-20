//! Conformal nonconformity scores: formula checks plus a seeded synthetic coverage study.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_calibration::{
    aps_nonconformity, aps_set, conformal_quantile, min_cps_interval, min_cps_nonconformity,
    ordinal_aps_interval, ordinal_set_rps, raps_nonconformity, raps_set, rps_nonconformity,
    SplitMix64,
};

#[test]
fn quantile_finite_sample_correction() {
    let scores: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    // alpha = 0.1: rank = ceil(11 * 0.9) = 10 -> the largest score.
    assert_eq!(conformal_quantile(&scores, 0.1).unwrap(), 10.0);
    // alpha = 0.05: rank = ceil(10.45) = 11 > n -> uninformative (+inf).
    assert_eq!(conformal_quantile(&scores, 0.05).unwrap(), f64::INFINITY);
    assert!(conformal_quantile(&[], 0.1).is_err());
    assert!(conformal_quantile(&scores, 1.5).is_err());
}

#[test]
fn aps_scores_and_sets() {
    let p = [0.5, 0.3, 0.2];
    // Label 0 is the mode: its APS score is its own mass.
    assert_eq!(aps_nonconformity(&p, 0), 0.5);
    // Label 2 is last in rank order: cumulative mass up to and including it is 1.0.
    assert_eq!(aps_nonconformity(&p, 2), 1.0);
    assert_eq!(aps_set(&p, 0.5), vec![0]);
    assert_eq!(aps_set(&p, 0.75), vec![0, 1]);
    // An infinite quantile yields the full set.
    assert_eq!(aps_set(&p, f64::INFINITY).len(), 3);
}

#[test]
fn raps_penalizes_deep_ranks() {
    let p = [0.25, 0.25, 0.25, 0.25];
    // Uniform: label 3 is rank 4 (ties break earliest-first).
    let aps = aps_nonconformity(&p, 3);
    let raps = raps_nonconformity(&p, 3, 2, 0.1);
    assert!((raps - (aps + 0.1 * 2.0)).abs() < 1e-12);
    let set = raps_set(&p, 0.6, 2, 0.1);
    // rank1: 0.25; rank2: 0.5 (no penalty yet); rank3: 0.75 + 0.1 -> stops at 3 labels.
    assert_eq!(set.len(), 3);
}

#[test]
fn ordinal_intervals_are_contiguous_and_mode_centered() {
    // Unimodal histogram over 7 levels, mode at 3.
    let p = [0.02, 0.08, 0.2, 0.4, 0.2, 0.08, 0.02];
    let (lo, hi) = ordinal_aps_interval(&p, 0.8);
    assert!(lo <= 3 && 3 <= hi);
    let mass: f64 = p[lo..=hi].iter().sum();
    assert!(mass >= 0.8);
    // RPS-CP: the interval contains the mode and is contiguous by construction.
    let (lo, hi) = ordinal_set_rps(&p, 0.05);
    assert!(lo <= 3 && 3 <= hi, "({lo}, {hi}) must contain mode 3");
    // min-CPS: monotone in qhat.
    let narrow = min_cps_interval(&p, 0.5);
    let wide = min_cps_interval(&p, 0.95);
    assert!(wide.1 - wide.0 >= narrow.1 - narrow.0);
    // Degenerate: zero-width-ish distribution collapses to the mode.
    let peaked = [0.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(ordinal_aps_interval(&peaked, 0.9), (2, 2));
    assert_eq!(ordinal_set_rps(&peaked, 0.0), (2, 2));
}

/// Uniform sample in [0, 1) from the test RNG.
fn uniform(rng: &mut SplitMix64) -> f64 {
    (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 // 2^53
}

/// Sample an index from a distribution.
fn sample(rng: &mut SplitMix64, probs: &[f64]) -> usize {
    let u = uniform(rng);
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if u < cdf {
            return i;
        }
    }
    probs.len() - 1
}

/// A Gaussian-ish histogram over `k` levels centered at `mode` with spread `sigma`.
fn gaussian_hist(k: usize, mode: f64, sigma: f64) -> Vec<f64> {
    let mut p: Vec<f64> = (0..k)
        .map(|i| (-((i as f64 - mode).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

#[test]
fn ordinal_conformal_marginal_coverage_on_synthetic() {
    // Seeded synthetic study: truthful histograms with sampled labels. Split-conformal
    // on well-specified probabilities must cover at ~1 - alpha marginally.
    let mut rng = SplitMix64::new(0xC0AF);
    let k = 8;
    let alpha = 0.1;
    let n_cal = 2000;
    let n_eval = 2000;

    let mut rps_scores = Vec::new();
    let mut cps_scores = Vec::new();
    let mut aps_scores = Vec::new();
    for _ in 0..n_cal {
        let mode = uniform(&mut rng) * (k - 1) as f64;
        let sigma = 0.5 + uniform(&mut rng) * 2.0;
        let p = gaussian_hist(k, mode, sigma);
        let y = sample(&mut rng, &p);
        rps_scores.push(rps_nonconformity(&p, y));
        cps_scores.push(min_cps_nonconformity(&p, y));
        aps_scores.push(aps_nonconformity(&p, y));
    }
    let q_rps = conformal_quantile(&rps_scores, alpha).unwrap();
    let q_cps = conformal_quantile(&cps_scores, alpha).unwrap();
    let q_aps = conformal_quantile(&aps_scores, alpha).unwrap();

    let mut cov_rps = 0usize;
    let mut cov_cps = 0usize;
    let mut cov_aps = 0usize;
    let mut width_cps = 0usize;
    let mut width_aps = 0usize;
    for _ in 0..n_eval {
        let mode = uniform(&mut rng) * (k - 1) as f64;
        let sigma = 0.5 + uniform(&mut rng) * 2.0;
        let p = gaussian_hist(k, mode, sigma);
        let y = sample(&mut rng, &p);
        let (lo, hi) = ordinal_set_rps(&p, q_rps);
        cov_rps += usize::from(lo <= y && y <= hi);
        let (lo, hi) = min_cps_interval(&p, q_cps);
        cov_cps += usize::from(lo <= y && y <= hi);
        let set = aps_set(&p, q_aps);
        cov_aps += usize::from(set.contains(&y));
        width_cps += hi - lo + 1;
        width_aps += set.len();
    }
    let denom = n_eval as f64;
    // Finite-sample slack: 1 - alpha - 0.04 with n=2000 is >30 standard errors safe.
    assert!(cov_rps as f64 / denom >= 0.86, "RPS-CP coverage {}", cov_rps);
    assert!(cov_cps as f64 / denom >= 0.86, "min-CPS coverage {}", cov_cps);
    assert!(cov_aps as f64 / denom >= 0.86, "APS coverage {}", cov_aps);
    // min-CPS should not blow up: mean width below full range.
    assert!(
        (width_cps as f64 / denom) < k as f64 - 1.0,
        "min-CPS mean width {}",
        width_cps as f64 / denom
    );
    let _ = width_aps;
}
