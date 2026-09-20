//! Answer-side IR: distributions, abstention, and the batch version triple.
//!
//! Raw probability distributions are the primary answer payload. Everything else —
//! labels, confidence, expected values — is derived ([`crate::confidence`]). A `null`
//! answer is an **abstention**: the model declined to commit, and consumers build
//! selective prediction on top of that value. Abstention is data, not an error.

/// The batch-level version triple carried by every emitted Arrow batch.
///
/// - `schema`: question-set schema version (see
///   [`crate::arrow::DecisionSchema::check_evolution`] — label-set evolution requires a
///   new schema version and fails loudly).
/// - `model`: resolved model id. Alias resolution is part of the serving contract: the
///   emitted value is the resolved versioned id, which may differ from the requested
///   alias.
/// - `calib`: calibration-fit version; `None` = uncalibrated (raw distribution). The
///   calibration *parameters* themselves are rows in the P0.3 metrics tables, never
///   schema content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionTriple {
    /// Question-set schema version.
    pub schema: String,
    /// Resolved model id.
    pub model: String,
    /// Calibration-fit version (`None` = uncalibrated).
    pub calib: Option<String>,
}

/// A typed answer payload: one probability distribution over the question's labels.
#[derive(Debug, Clone, PartialEq)]
pub enum AnswerValue {
    /// Noul: P(true). Emitted as the 2-wide distribution `[1 − p, p]` over labels
    /// `["false", "true"]`.
    Noul {
        /// Probability the statement is true.
        p_true: f32,
    },
    /// Choice: one probability per option, aligned with canonical option order.
    Choice {
        /// Per-option probabilities (length = option count; producer tolerance D5).
        probabilities: Vec<f32>,
    },
    /// Score: one probability per level, in ascending level order.
    Score {
        /// Per-level probabilities (length = level count; producer tolerance D5).
        probabilities: Vec<f32>,
    },
}

impl AnswerValue {
    /// The distribution over the question's labels, in canonical label order.
    pub fn probabilities(&self) -> Vec<f32> {
        match self {
            Self::Noul { p_true } => vec![1.0 - p_true, *p_true],
            Self::Choice { probabilities } | Self::Score { probabilities } => probabilities.clone(),
        }
    }
}

/// The answer to one question on one row.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionAnswer {
    /// `None` = **abstained** (emitted as null probabilities + null label).
    pub value: Option<AnswerValue>,
    /// Conformal prediction set (labels). **Reserved in schema v1** — the column is
    /// emitted but may be entirely unpopulated; `None` = no set on this row. An abstained
    /// answer must not carry a set (abstention already says "no commitment").
    pub conformal_set: Option<Vec<String>>,
}

impl QuestionAnswer {
    /// An abstained answer with no conformal set.
    pub fn abstained() -> Self {
        Self {
            value: None,
            conformal_set: None,
        }
    }

    /// An answered question with no conformal set.
    pub fn answered(value: AnswerValue) -> Self {
        Self {
            value: Some(value),
            conformal_set: None,
        }
    }
}
