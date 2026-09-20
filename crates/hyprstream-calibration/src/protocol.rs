//! The pinned gate measurement protocol.
//!
//! Every gate that consumes calibration numbers (P0.5 baselines, P1.4 legs b/c/e, P2.3)
//! measures them this way. The constants are the contract; [`GateReport`] is the
//! serializable artifact a gate run emits. Nothing here fits parameters — fitting is
//! P2.2; this module defines *measurement* only.
//!
//! ## The protocol, pinned
//!
//! 1. **Binning**: [`GATE_BIN_COUNT`] equal-mass bins for ECE and reliability curves
//!    ([`crate::bins::Binning::equal_mass`]). Equal-mass, not classic equal-width, so
//!    bins stay populated under peaked confidence distributions.
//! 2. **Uncertainty**: percentile bootstrap CIs at [`GATE_CI_LEVEL`] over
//!    [`GATE_BOOTSTRAP_RESAMPLES`] resamples, seeded and recorded. Gate comparisons that
//!    pin a bound (P1.4 leg b's "macro-ECE gap ≤ 2×") condition on the **upper CI
//!    bound**, never the point estimate.
//! 3. **Aggregation — per-field macro rule**: metrics are computed per question field
//!    and averaged with equal field weight ([`macro_average`]). Rows are never pooled
//!    across fields: a high-volume field must not dilute a miscalibrated rare one.
//! 4. **Shift split** ([`ShiftSplit`]): the shift evaluation split is a held-out
//!    workflow *family* — evaluation families disjoint from every family used for
//!    fitting or calibration. Item-level disjointness is not the shift split.
//! 5. **Per-primitive metrics** (P1.4 leg c): ECE for `noul`/`choice`; SCE/ACE + RPS for
//!    `score` (top-label ECE is class-blind on ordinal tasks — S6b2).
//! 6. **Joint coverage**: conformal coverage is marginal; cross-question joint coverage
//!    is reported separately ([`crate::joint`]).

use serde::{Deserialize, Serialize};

use crate::bootstrap::{bootstrap_ci, ConfidenceInterval};
use crate::error::MetricError;

/// Pinned bin count for ECE and reliability curves (equal-mass).
pub const GATE_BIN_COUNT: usize = 15;

/// Pinned bootstrap resample count for gate confidence intervals.
pub const GATE_BOOTSTRAP_RESAMPLES: usize = 1000;

/// Pinned confidence level for gate intervals.
pub const GATE_CI_LEVEL: f64 = 0.95;

/// Per-field macro aggregation: unweighted mean of per-field metric values.
/// Returns `None` for an empty field set (a report with no fields is malformed, not 0).
#[must_use]
pub fn macro_average(per_field: &[f64]) -> Option<f64> {
    if per_field.is_empty() {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = per_field.len() as f64;
    Some(per_field.iter().sum::<f64>() / n)
}

/// The shift-split definition: which workflow families may be used for
/// fitting/calibration and which are held out for evaluation.
///
/// The **shift split** (P1.4 leg d, and the zero-shot leg e at family level) holds out
/// whole workflow families: any family whose items were synthesized into training data,
/// used for distillation targets, or used to fit calibration parameters is a *fit*
/// family; the gate measures on the disjoint *evaluation* families. Construction enforces
/// disjointness — a family on both sides is a protocol violation, not a warning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShiftSplit {
    /// Families used for fitting (training, distillation, calibration-parameter fitting).
    pub fit_families: Vec<String>,
    /// Held-out families reserved for shift/zero-shot evaluation.
    pub eval_families: Vec<String>,
}

impl ShiftSplit {
    /// Construct a shift split, rejecting any family present on both sides.
    ///
    /// # Errors
    /// [`MetricError::ShiftSplitOverlap`] listing the overlapping families.
    pub fn new(
        fit_families: impl IntoIterator<Item = impl Into<String>>,
        eval_families: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, MetricError> {
        let fit: Vec<String> = fit_families.into_iter().map(Into::into).collect();
        let eval: Vec<String> = eval_families.into_iter().map(Into::into).collect();
        let mut overlap: Vec<String> = eval
            .iter()
            .filter(|f| fit.contains(f))
            .cloned()
            .collect();
        overlap.sort();
        overlap.dedup();
        if !overlap.is_empty() {
            return Err(MetricError::ShiftSplitOverlap { families: overlap });
        }
        Ok(Self {
            fit_families: fit,
            eval_families: eval,
        })
    }

    /// Which side of the split a family belongs to.
    #[must_use]
    pub fn is_eval_family(&self, family: &str) -> bool {
        self.eval_families.iter().any(|f| f == family)
    }
}

/// One question field's measured metrics (one row of the macro rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldMetrics {
    /// Question field identifier (question id / field name in the question set).
    pub field: String,
    /// Workflow family this field's rows were drawn from.
    pub family: String,
    /// Number of observations measured.
    pub n: usize,
    /// Top-label ECE (equal-mass, [`GATE_BIN_COUNT`] bins). Primary for noul/choice.
    pub ece: f64,
    /// Multiclass Brier score.
    pub brier: f64,
    /// Class-wise static calibration error. Present for score fields (P1.4 leg c).
    pub sce: Option<f64>,
    /// Class-wise adaptive calibration error. Present for score fields.
    pub ace: Option<f64>,
    /// Ranked probability score. Present for score fields.
    pub rps: Option<f64>,
}

/// Macro-averaged metric with its bootstrap CI, as reported by a gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MacroMetrics {
    /// Per-field macro mean (equal field weights).
    pub mean: f64,
    /// Bootstrap CI over the *field* values (fields are the resampling unit — resampling
    /// rows would understate field-level variance).
    pub ci: ConfidenceInterval,
}

impl MacroMetrics {
    /// Macro-average a metric over fields, with the protocol bootstrap CI.
    ///
    /// # Errors
    /// [`MetricError::EmptyInput`] if no field values are given; propagates bootstrap
    /// errors.
    pub fn from_fields(values: &[f64], seed: u64) -> Result<Self, MetricError> {
        let (mean, ci) = bootstrap_ci(
            values,
            |v| {
                #[allow(clippy::cast_precision_loss)]
                let n = v.len() as f64;
                v.iter().sum::<f64>() / n
            },
            GATE_BOOTSTRAP_RESAMPLES,
            GATE_CI_LEVEL,
            seed,
        )?;
        Ok(Self { mean, ci })
    }

    /// Upper CI bound — the value bound-style gate comparisons condition on.
    #[must_use]
    pub fn upper_bound(&self) -> f64 {
        self.ci.high
    }
}

/// A gate run's serializable measurement artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    /// Name of the arm/subject measured (e.g. a model version, a baseline arm).
    pub subject: String,
    /// The split this report measured (e.g. `"deterministic"`, `"shift"`, `"zero-shot"`).
    pub split: String,
    /// The shift split in force for this report (fit/eval family designation).
    pub shift_split: ShiftSplit,
    /// Per-field measurements (the macro rule's inputs).
    pub fields: Vec<FieldMetrics>,
    /// Macro-averaged ECE with CI.
    pub macro_ece: MacroMetrics,
    /// Macro-averaged Brier with CI.
    pub macro_brier: MacroMetrics,
    /// Macro-averaged SCE over score fields with CI, if any score fields were measured.
    pub macro_sce: Option<MacroMetrics>,
    /// Macro-averaged RPS over score fields with CI, if any score fields were measured.
    pub macro_rps: Option<MacroMetrics>,
    /// Seed used for all bootstrap CIs in this report.
    pub bootstrap_seed: u64,
}

impl GateReport {
    /// Assemble a gate report from per-field measurements, applying the protocol's macro
    /// rule and bootstrap CIs. SCE/ACE/RPS macro aggregates cover only the fields that
    /// carry them (score fields), keeping the per-primitive split of P1.4 leg c.
    ///
    /// # Errors
    /// [`MetricError::EmptyInput`] if `fields` is empty; propagates bootstrap errors.
    pub fn assemble(
        subject: impl Into<String>,
        split: impl Into<String>,
        shift_split: ShiftSplit,
        fields: Vec<FieldMetrics>,
        bootstrap_seed: u64,
    ) -> Result<Self, MetricError> {
        if fields.is_empty() {
            return Err(MetricError::EmptyInput("gate report"));
        }
        let macro_ece = MacroMetrics::from_fields(
            &fields.iter().map(|f| f.ece).collect::<Vec<_>>(),
            bootstrap_seed,
        )?;
        let macro_brier = MacroMetrics::from_fields(
            &fields.iter().map(|f| f.brier).collect::<Vec<_>>(),
            bootstrap_seed ^ 0x5EED_0001,
        )?;
        let score_metric = |pick: fn(&FieldMetrics) -> Option<f64>, salt: u64| {
            let values: Vec<f64> = fields.iter().filter_map(pick).collect();
            if values.is_empty() {
                return Ok(None);
            }
            MacroMetrics::from_fields(&values, bootstrap_seed ^ salt).map(Some)
        };
        let macro_sce = score_metric(|f| f.sce, 0x5EED_0002)?;
        let macro_rps = score_metric(|f| f.rps, 0x5EED_0003)?;
        Ok(Self {
            subject: subject.into(),
            split: split.into(),
            shift_split,
            fields,
            macro_ece,
            macro_brier,
            macro_sce,
            macro_rps,
            bootstrap_seed,
        })
    }
}
