//! Pre-committed disclosure language.
//!
//! The program's differentiator is falsifiability: every published calibration claim
//! ships with the same disclosed limits, committed here *before* the numbers exist so the
//! language cannot drift to fit results. Gates and reports (P0.5, P1.4, P2.3, P3.6)
//! embed [`DISCLOSURE`] verbatim; [`disclosure_with`] fills the two placeholders every
//! report must concrete-ize.

/// The pre-committed disclosure text, with `{telemetry_since}` and `{subject}`
/// placeholders for [`disclosure_with`].
///
/// Claims are scoped to (a) mechanically-checkable verifiable-outcome families, (b)
/// teacher-agreement splits, and (c) realized production telemetry from a stated date —
/// and calibration-to-teacher is explicitly not calibration-to-reality.
pub const DISCLOSURE: &str = "\
DISCLOSURE (pre-committed, System One program).

The calibration and agreement numbers in this report for {subject} are scoped as
follows, and no stronger claim is made:

1. MECHANICALLY-CHECKABLE FAMILIES. Absolute calibration anchors (ECE, Brier, SCE/ACE,
   RPS, conformal coverage) are measured on procedurally generated verifiable-outcome
   benchmark families whose ground truth is computed, not judged. They characterize
   behavior on those families and their stated difficulty strata only.

2. TEACHER-AGREEMENT SPLITS. Where ground truth is not mechanically checkable, numbers
   measure agreement with a corrected teacher ensemble. Calibration-to-teacher is NOT
   calibration-to-reality: a model can match its teachers' probabilities exactly and
   inherit their shared errors.

3. REALIZED TELEMETRY. Production-outcome calibration, where reported, covers realized
   traffic since {telemetry_since} only, under the version triple (schema, model,
   calibration) recorded with each batch. Past calibration does not guarantee future
   calibration under distribution shift; post-hoc calibration is re-fit online and each
   re-fit mints a new calibration version.

4. MARGINAL VS JOINT COVERAGE. Conformal coverage guarantees are marginal per question.
   Cross-question joint coverage is measured and reported separately; it is lower than
   the marginal guarantee whenever per-question errors correlate.

5. UNSEEN FAMILIES. Zero-shot transfer to never-synthesized question families is
   measured on the held-out gate families and published with the same protocol. Transfer
   is a measured result, not an asserted property.";

/// Render the disclosure with the two required substitutions: the subject (model/arm the
/// numbers describe) and the start date of the realized-telemetry window (ISO 8601, or
/// `"no production telemetry yet"` for pre-launch reports).
#[must_use]
pub fn disclosure_with(subject: &str, telemetry_since: &str) -> String {
    DISCLOSURE
        .replace("{subject}", subject)
        .replace("{telemetry_since}", telemetry_since)
}
