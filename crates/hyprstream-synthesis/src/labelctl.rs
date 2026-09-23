//! Label-distribution control: keep the ensemble-argmax label histogram from
//! collapsing onto one answer per (kind, cardinality) bucket.
//!
//! Balancing is over the **corrected ensemble argmax**, not any single
//! teacher — the same distribution P1.4 distills toward.

use std::collections::HashMap;

/// Label-distribution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelPolicy {
    /// Accept every generated row (control off).
    Unlimited,
    /// Balance argmax labels per (kind, cardinality): accept a row only if
    /// its label bucket is within `slack` of the least-filled bucket.
    Balanced {
        /// How far a bucket may run ahead of the minimum before its rows are
        /// deferred. 1 = near-perfect balance.
        slack: usize,
    },
}

/// The running histogram, keyed by (kind, cardinality, label).
#[derive(Debug, Default)]
pub(crate) struct LabelController {
    counts: HashMap<(String, usize, usize), usize>,
}

impl LabelController {
    /// Decide whether to accept a row with the given ensemble-argmax label.
    /// Under `Balanced`, a row whose bucket is already `slack` ahead of the
    /// minimum is deferred (dropped from this run).
    pub(crate) fn accept(
        &mut self,
        policy: LabelPolicy,
        kind: &str,
        cardinality: usize,
        label: usize,
    ) -> bool {
        let LabelPolicy::Balanced { slack } = policy else {
            self.bump(kind, cardinality, label);
            return true;
        };
        let min = (0..cardinality)
            .map(|other| {
                *self
                    .counts
                    .get(&(kind.to_owned(), cardinality, other))
                    .unwrap_or(&0)
            })
            .min()
            .unwrap_or(0);
        let current = *self
            .counts
            .get(&(kind.to_owned(), cardinality, label))
            .unwrap_or(&0);
        // Saturating: an effectively unlimited slack (e.g. usize::MAX from a
        // tuned config) must not overflow the threshold — a debug panic or a
        // release wrap to zero would permanently defer the bucket.
        if current <= min.saturating_add(slack.saturating_sub(1)) {
            self.bump(kind, cardinality, label);
            true
        } else {
            false
        }
    }

    fn bump(&mut self, kind: &str, cardinality: usize, label: usize) {
        *self
            .counts
            .entry((kind.to_owned(), cardinality, label))
            .or_insert(0) += 1;
    }

    /// The current histogram (stats reporting).
    pub(crate) fn histogram(&self) -> &HashMap<(String, usize, usize), usize> {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_accepts_everything() {
        let mut controller = LabelController::default();
        for _ in 0..100 {
            assert!(controller.accept(LabelPolicy::Unlimited, "noul", 2, 0));
        }
    }

    #[test]
    fn balanced_keeps_buckets_within_slack() {
        let mut controller = LabelController::default();
        let policy = LabelPolicy::Balanced { slack: 1 };
        // First label-0 row accepted (bucket 0 ahead by 1 = slack).
        assert!(controller.accept(policy, "noul", 2, 0));
        // Second label-0 row deferred until label 1 catches up.
        assert!(!controller.accept(policy, "noul", 2, 0));
        assert!(controller.accept(policy, "noul", 2, 1));
        assert!(controller.accept(policy, "noul", 2, 0));
        assert_eq!(
            controller.histogram().get(&("noul".to_owned(), 2, 0)),
            Some(&2)
        );
        assert_eq!(
            controller.histogram().get(&("noul".to_owned(), 2, 1)),
            Some(&1)
        );
    }

    #[test]
    fn huge_slack_saturates_instead_of_overflowing() {
        // Regression (review): `min + slack - 1` with slack = usize::MAX
        // overflowed once every label had been accepted (debug panic;
        // release wrap to a zero threshold that permanently deferred rows).
        let mut controller = LabelController::default();
        let policy = LabelPolicy::Balanced { slack: usize::MAX };
        for _ in 0..2 {
            assert!(controller.accept(policy, "noul", 2, 0));
            assert!(controller.accept(policy, "noul", 2, 1));
        }
        // min is now 2; the threshold must saturate, not overflow.
        assert!(controller.accept(policy, "noul", 2, 0));
    }

    #[test]
    fn balancing_is_per_kind_and_cardinality() {
        let mut controller = LabelController::default();
        let policy = LabelPolicy::Balanced { slack: 1 };
        assert!(controller.accept(policy, "choice", 3, 0));
        // A different kind/cardinality bucket starts empty: accepted.
        assert!(controller.accept(policy, "choice", 4, 0));
        assert!(controller.accept(policy, "score", 3, 0));
    }
}
