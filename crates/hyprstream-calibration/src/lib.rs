//! # hyprstream-calibration — calibration metrics library (System One DAG node P0.2)
//!
//! Calibration is the product thesis of the System One program, and this crate owns its
//! *measurement* half: every number a gate (P0.5, P1.4, P2.3) or a published evaluation
//! reports is computed here, under one pinned protocol, so results are comparable across
//! arms, splits, and releases. Primitives, not opinions: these are pure functions over
//! probability vectors — no model code, no fitting (post-hoc parameter fitting is P2.2),
//! no serving policy.
//!
//! ## What is measured
//!
//! - **Nominal calibration** ([`metrics`]): top-label ECE, multiclass Brier score,
//!   negative log-likelihood, and reliability curves, all over the pinned binning
//!   ([`bins`]).
//! - **Ordinal calibration** ([`metrics`]): class-wise SCE/ACE and the ranked probability
//!   score (RPS) for the `score` primitive — top-label ECE is class-blind on ordinal
//!   tasks and cannot see the tail errors that corrupt the probability-weighted value
//!   (S6b2, ORCU arXiv:2410.15658). RPS is a proper scoring rule on the *cumulative*
//!   distribution.
//! - **Selective prediction** ([`selective`]): risk–coverage curves over
//!   confidence-thresholded answers; abstention (`null` answers in the jev-1 profile) is
//!   the data this operates on.
//! - **Conformal nonconformity** ([`conformal`]): vanilla APS/RAPS scores and prediction
//!   sets for `noul`/`choice`, and ordinal contiguous-interval nonconformity for `score`
//!   — RPS-based and min-CPS-style sliding window — because probability-sorted APS/RAPS
//!   sets are order-blind and can be non-contiguous over an ordered scale (S6b2). MAPIE
//!   ships no ordinal method; these are implemented in-house. Split-conformal quantiles
//!   use the standard finite-sample correction.
//! - **Cross-question / joint coverage** ([`joint`]): conformal coverage guarantees are
//!   *marginal* (per question); the fraction of batch rows in which *every* question is
//!   covered is reported separately, against the product-of-marginals independence
//!   reference.
//!
//! ## The gate measurement protocol ([`protocol`])
//!
//! Pinned so gate numerics are reproducible and comparable (consumed by P1.4 legs b/c/e
//! and P2.3):
//!
//! - **15 equal-mass bins** ([`protocol::GATE_BIN_COUNT`]) for ECE and reliability
//!   curves — equal-mass, not the classic equal-width, so bins are populated under
//!   peaked real-world confidence distributions.
//! - **Bootstrap percentile CIs** ([`bootstrap`]) at 95% over 1000 resamples
//!   ([`protocol::GATE_BOOTSTRAP_RESAMPLES`], [`protocol::GATE_CI_LEVEL`]); gate
//!   comparisons that pin a bound (e.g. P1.4 leg b's "≤ 2×") condition on the **upper CI
//!   bound**.
//! - **Per-field macro rule** ([`protocol::macro_average`]): metrics are computed per
//!   question field and averaged with equal field weights — never pooled over rows, so
//!   high-volume fields cannot dilute a miscalibrated rare field.
//! - **Shift-split definition** ([`protocol::ShiftSplit`]): the shift split is a
//!   held-out workflow *family* — evaluation families disjoint from every
//!   fitting/calibration family, enforced at construction.
//!
//! ## Disclosure ([`disclosure`])
//!
//! The pre-committed disclosure language that must accompany every published number is
//! pinned here as data ([`disclosure::DISCLOSURE`]): claims are scoped to
//! mechanically-checkable families plus teacher-agreement splits plus realized telemetry
//! from a stated date, and calibration-to-teacher is not calibration-to-reality.

pub mod bins;
pub mod bootstrap;
pub mod conformal;
pub mod disclosure;
pub mod error;
pub mod joint;
pub mod metrics;
pub mod protocol;
pub mod selective;

pub use bins::{Bin, Binning};
pub use bootstrap::{bootstrap_ci, ConfidenceInterval, SplitMix64};
pub use conformal::{
    aps_nonconformity, aps_set, conformal_quantile, min_cps_interval, min_cps_nonconformity,
    ordinal_aps_interval, ordinal_set_rps, raps_nonconformity, raps_set, rps_nonconformity,
};
pub use error::MetricError;
pub use joint::{joint_coverage, JointCoverage};
pub use metrics::{ace, brier, ece, expected_value, nll, reliability_curve, rps, sce};
pub use protocol::{
    FieldMetrics, GateReport, MacroMetrics, ShiftSplit, GATE_BIN_COUNT, GATE_BOOTSTRAP_RESAMPLES,
    GATE_CI_LEVEL,
};
pub use selective::{accuracy_at_coverage, risk_coverage_curve, selective_auc, RiskCoverage};
