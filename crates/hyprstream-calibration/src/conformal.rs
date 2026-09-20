//! Conformal nonconformity scores and prediction sets.
//!
//! Split-conformal usage: compute a nonconformity score per calibration observation, take
//! the finite-sample-corrected quantile ([`conformal_quantile`]) at the target
//! miscoverage `α`, and construct prediction sets on new observations from the same
//! score. All scores are deterministic (no randomized tie-breaking), which makes them
//! slightly conservative and fully reproducible.
//!
//! - **Nominal** (`noul`/`choice`): [`aps_nonconformity`] / [`aps_set`] (Romano et al.
//!   2020) and [`raps_nonconformity`] / [`raps_set`] (Angelopoulos et al. 2021).
//! - **Ordinal** (`score`): probability-sorted sets are order-blind and can be
//!   non-contiguous over an ordered scale, which is semantically broken for a score
//!   (S6b2). This module provides the ordinal alternatives MAPIE does not ship:
//!   - [`rps_nonconformity`] / [`ordinal_set_rps`] — RPS-based nonconformity; sets are
//!     median-centered contiguous intervals by construction (Haas et al. 2026,
//!     arXiv:2606.24959);
//!   - [`ordinal_aps_interval`] — contiguous interval grown around the mode (Ordinal APS;
//!     Lu, Angelopoulos, Pomerantz 2022, arXiv:2207.02238);
//!   - [`min_cps_nonconformity`] / [`min_cps_interval`] — minimum-length contiguous
//!     sliding-window sets (min-CPS style; Zhang et al. 2025, arXiv:2511.16845).
//!
//! Ordinal sets are returned as inclusive index intervals `(lo, hi)` over levels and are
//! always contiguous; any score-defined label set is returned as its contiguous hull.

use crate::error::MetricError;
use crate::metrics::{argmax, rps_single};

/// Finite-sample-corrected split-conformal quantile of calibration nonconformity scores:
/// the `⌈(n+1)(1−α)⌉`-th smallest score. Returns [`f64::INFINITY`] when the rank exceeds
/// `n` (the prediction set is then "everything" — the guarantee is uninformative at
/// requested `α` for this calibration size).
///
/// # Errors
/// [`MetricError::EmptyInput`] on empty scores; [`MetricError::InvalidLevel`] for `α`
/// outside `(0, 1)`.
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64, MetricError> {
    if scores.is_empty() {
        return Err(MetricError::EmptyInput("conformal quantile"));
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(MetricError::InvalidLevel { value: alpha });
    }
    let mut sorted: Vec<f64> = scores.to_vec();
    sorted.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss)]
    let n = sorted.len() as f64;
    let rank = ((n + 1.0) * (1.0 - alpha)).ceil();
    #[allow(clippy::cast_sign_loss)]
    if rank > n {
        return Ok(f64::INFINITY);
    }
    let idx = (rank as usize).max(1) - 1;
    Ok(sorted[idx])
}

/// Class indices sorted by descending probability, earliest index first on ties (profile
/// decision D6).
fn rank_order(probs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| f64::total_cmp(&probs[b], &probs[a]));
    order
}

/// APS nonconformity: cumulative probability mass, in descending-probability order, up to
/// and including the realized label. Deterministic (the full label mass is included),
/// hence slightly conservative vs the randomized form.
#[must_use]
pub fn aps_nonconformity(probs: &[f64], label: usize) -> f64 {
    let order = rank_order(probs);
    let mut cumulative = 0.0;
    for &idx in &order {
        cumulative += probs[idx];
        if idx == label {
            break;
        }
    }
    cumulative
}

/// APS prediction set: labels in descending-probability order until the cumulative mass
/// reaches `qhat`; the crossing label is included.
#[must_use]
pub fn aps_set(probs: &[f64], qhat: f64) -> Vec<usize> {
    let order = rank_order(probs);
    let mut set = Vec::new();
    let mut cumulative = 0.0;
    for &idx in &order {
        set.push(idx);
        cumulative += probs[idx];
        if cumulative >= qhat {
            break;
        }
    }
    set
}

/// RAPS nonconformity: APS cumulative mass plus the regularization penalty
/// `lambda · max(rank(label) − k_reg, 0)` where `rank` is the 1-based rank in
/// descending-probability order (Angelopoulos et al. 2021). `k_reg` is the number of
/// penalty-free top ranks.
#[must_use]
pub fn raps_nonconformity(probs: &[f64], label: usize, k_reg: usize, lambda: f64) -> f64 {
    let order = rank_order(probs);
    let mut cumulative = 0.0;
    for (zero_rank, &idx) in order.iter().enumerate() {
        cumulative += probs[idx];
        if idx == label {
            let rank = zero_rank + 1;
            #[allow(clippy::cast_precision_loss)]
            let penalty = lambda * rank.saturating_sub(k_reg) as f64;
            return cumulative + penalty;
        }
    }
    cumulative
}

/// RAPS prediction set: labels in rank order while the penalized cumulative mass stays
/// below `qhat`; the crossing label is included.
#[must_use]
pub fn raps_set(probs: &[f64], qhat: f64, k_reg: usize, lambda: f64) -> Vec<usize> {
    let order = rank_order(probs);
    let mut set = Vec::new();
    let mut cumulative = 0.0;
    for (zero_rank, &idx) in order.iter().enumerate() {
        let rank = zero_rank + 1;
        #[allow(clippy::cast_precision_loss)]
        let penalty = lambda * rank.saturating_sub(k_reg) as f64;
        set.push(idx);
        cumulative += probs[idx] + penalty;
        if cumulative >= qhat {
            break;
        }
    }
    set
}

/// RPS-based ordinal nonconformity: the ranked probability score of the candidate label
/// against the predicted histogram (Haas et al. 2026). Cheap and model-agnostic; the
/// induced sets are median-centered and contiguous by construction.
#[must_use]
pub fn rps_nonconformity(probs: &[f64], label: usize) -> f64 {
    rps_single(probs, label)
}

/// RPS-CP prediction interval: the contiguous hull of `{k : rps_nonconformity(p, k) ≤
/// qhat}`. (The raw set is already contiguous for unimodal histograms; the hull
/// guarantees the interval contract for multimodal ones.) Falls back to the
/// lowest-nonconformity single level if no level clears `qhat`.
#[must_use]
pub fn ordinal_set_rps(probs: &[f64], qhat: f64) -> (usize, usize) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    let mut best = 0usize;
    let mut best_score = f64::INFINITY;
    for k in 0..probs.len() {
        let s = rps_single(probs, k);
        if s < best_score {
            best_score = s;
            best = k;
        }
        if s <= qhat {
            lo = lo.min(k);
            hi = hi.max(k);
        }
    }
    if lo > hi {
        (best, best)
    } else {
        (lo, hi)
    }
}

/// Ordinal APS interval (Lu, Angelopoulos, Pomerantz 2022): start at the modal level and
/// grow a contiguous interval, always extending toward the adjacent side with the larger
/// probability, until the enclosed mass reaches `qhat`.
#[must_use]
pub fn ordinal_aps_interval(probs: &[f64], qhat: f64) -> (usize, usize) {
    if probs.is_empty() {
        return (0, 0);
    }
    let mode = argmax(probs);
    let (mut lo, mut hi) = (mode, mode);
    let mut mass = probs[mode];
    while mass < qhat && (lo > 0 || hi + 1 < probs.len()) {
        let left = if lo > 0 { probs[lo - 1] } else { f64::NEG_INFINITY };
        let right = if hi + 1 < probs.len() { probs[hi + 1] } else { f64::NEG_INFINITY };
        if left >= right {
            lo -= 1;
            mass += probs[lo];
        } else {
            hi += 1;
            mass += probs[hi];
        }
    }
    (lo, hi)
}

/// The minimum-length contiguous window whose mass is at least `threshold`; ties break
/// toward higher mass, then toward the leftmost window. `O(K²)` via prefix sums.
fn min_length_window(probs: &[f64], threshold: f64) -> (usize, usize) {
    let k = probs.len();
    let mut prefix = vec![0.0f64; k + 1];
    for (i, &p) in probs.iter().enumerate() {
        prefix[i + 1] = prefix[i] + p;
    }
    let mass = |a: usize, b: usize| prefix[b + 1] - prefix[a];
    for len in 1..=k {
        let mut best: Option<(usize, usize)> = None;
        for a in 0..=(k - len) {
            let b = a + len - 1;
            if mass(a, b) < threshold {
                continue;
            }
            match best {
                None => best = Some((a, b)),
                Some((ba, bb)) => {
                    if mass(a, b) > mass(ba, bb) {
                        best = Some((a, b));
                    }
                }
            }
        }
        if let Some(w) = best {
            return w;
        }
    }
    (0, k.saturating_sub(1))
}

/// min-CPS-style nonconformity (Zhang et al. 2025): the smallest mass threshold `τ` at
/// which the minimum-length contiguous window with mass `≥ τ` contains the realized
/// label. Computed by enumerating candidate thresholds (distinct contiguous-window
/// masses); `O(K³)` worst case, intended for calibration-fitting batches, with the
/// serving-time set constructed by [`min_cps_interval`].
#[must_use]
pub fn min_cps_nonconformity(probs: &[f64], label: usize) -> f64 {
    let k = probs.len();
    let mut prefix = vec![0.0f64; k + 1];
    for (i, &p) in probs.iter().enumerate() {
        prefix[i + 1] = prefix[i] + p;
    }
    let mut masses: Vec<f64> = Vec::with_capacity(k * (k + 1) / 2);
    for a in 0..k {
        for b in a..k {
            masses.push(prefix[b + 1] - prefix[a]);
        }
    }
    masses.sort_by(f64::total_cmp);
    masses.dedup();
    for &tau in &masses {
        let (lo, hi) = min_length_window(probs, tau);
        if lo <= label && label <= hi {
            return tau;
        }
    }
    // The full-range window always contains the label, so this is unreachable for a
    // valid label; fall back to the total mass defensively.
    prefix[k]
}

/// min-CPS prediction interval: the contiguous hull of `{k : min_cps_nonconformity(p, k)
/// ≤ qhat}` — the minimum-length contiguous set with the calibrated coverage guarantee.
#[must_use]
pub fn min_cps_interval(probs: &[f64], qhat: f64) -> (usize, usize) {
    if probs.is_empty() {
        return (0, 0);
    }
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for k in 0..probs.len() {
        if min_cps_nonconformity(probs, k) <= qhat {
            lo = lo.min(k);
            hi = hi.max(k);
        }
    }
    if lo > hi { (0, probs.len() - 1) } else { (lo, hi) }
}

/// Validate that a distribution is usable for conformal scoring.
///
/// # Errors
/// [`MetricError::InvalidDistribution`] on empty, negative, or non-finite entries.
pub fn check_distribution(probs: &[f64], index: usize) -> Result<(), MetricError> {
    if probs.is_empty() {
        return Err(MetricError::InvalidDistribution {
            index,
            reason: "empty distribution".to_owned(),
        });
    }
    for &p in probs {
        if !p.is_finite() || p < 0.0 {
            return Err(MetricError::InvalidDistribution {
                index,
                reason: format!("non-finite or negative probability {p}"),
            });
        }
    }
    Ok(())
}
