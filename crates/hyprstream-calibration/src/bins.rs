//! Binning for calibration curves and error metrics.
//!
//! The gate protocol pins **15 equal-mass bins** ([`crate::protocol::GATE_BIN_COUNT`]) for
//! ECE and reliability curves. Equal-mass (equal-count) binning is used rather than the
//! classic equal-width ECE binning because real decision distributions are peaked: under
//! equal-width binning nearly all mass lands in one or two top bins and the curve says
//! nothing about the rest of the confidence range. Equal-width binning is retained for
//! the class-wise SCE, whose published definition is static (Nixon et al. 2019); ACE is
//! its equal-mass counterpart.

use serde::{Deserialize, Serialize};

use crate::error::MetricError;

/// A monotone set of bin edges over a confidence axis (`edges.len() == n_bins + 1`).
///
/// Construct via [`Binning::equal_mass`] (protocol default) or [`Binning::equal_width`]
/// (SCE's published static form). Degenerate duplicate edges are collapsed, so the
/// realized bin count may be smaller than requested when confidences tie heavily.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Binning {
    edges: Vec<f64>,
}

/// One populated bin of a reliability diagram.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bin {
    /// Bin index in the parent [`Binning`].
    pub index: usize,
    /// Inclusive lower edge.
    pub lower: f64,
    /// Inclusive upper edge.
    pub upper: f64,
    /// Number of observations in the bin.
    pub count: usize,
    /// Mean confidence within the bin.
    pub mean_confidence: f64,
    /// Mean outcome within the bin (empirical accuracy / event frequency).
    pub mean_outcome: f64,
}

impl Binning {
    /// Equal-mass binning: edges chosen so each bin holds (approximately) `n / n_bins`
    /// confidences. This is the protocol binning.
    ///
    /// # Errors
    /// [`MetricError::InvalidBinCount`] if `n_bins` is zero or exceeds the number of
    /// confidences; [`MetricError::EmptyInput`] on an empty slice.
    pub fn equal_mass(confidences: &[f64], n_bins: usize) -> Result<Self, MetricError> {
        if confidences.is_empty() {
            return Err(MetricError::EmptyInput("equal_mass binning"));
        }
        if n_bins == 0 || n_bins > confidences.len() {
            return Err(MetricError::InvalidBinCount {
                bins: n_bins,
                n: confidences.len(),
            });
        }
        let mut sorted: Vec<f64> = confidences.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let mut edges = Vec::with_capacity(n_bins + 1);
        for b in 0..=n_bins {
            // Rank-quantile edge: the (b / n_bins)-quantile of the sorted confidences.
            let pos = b * n / n_bins;
            let idx = pos.min(n - 1);
            edges.push(sorted[idx]);
        }
        edges[0] = 0.0_f64.min(edges[0]);
        edges[n_bins] = 1.0_f64.max(edges[n_bins]);
        edges.dedup();
        Ok(Self { edges })
    }

    /// Equal-width binning over `[0, 1]` — the published static form used by SCE.
    #[must_use]
    pub fn equal_width(n_bins: usize) -> Self {
        let n_bins = n_bins.max(1);
        #[allow(clippy::cast_precision_loss)]
        let width = 1.0 / n_bins as f64;
        let edges = (0..=n_bins).map(|b| b as f64 * width).collect();
        Self { edges }
    }

    /// Number of bins.
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.edges.len() - 1
    }

    /// Bin index for a confidence value (last bin is closed on both sides).
    #[must_use]
    pub fn assign(&self, confidence: f64) -> usize {
        let n_bins = self.n_bins();
        for (i, w) in self.edges.windows(2).enumerate() {
            if confidence >= w[0] && (confidence < w[1] || (i == n_bins - 1 && confidence <= w[1])) {
                return i;
            }
        }
        // Out-of-range confidences clamp to the nearest end bin.
        if confidence < self.edges[0] {
            0
        } else {
            n_bins - 1
        }
    }

    /// Populate the bins: mean confidence and mean outcome per non-empty bin.
    ///
    /// # Errors
    /// [`MetricError::LengthMismatch`] if the slices differ in length;
    /// [`MetricError::EmptyInput`] if both are empty.
    pub fn summarize(
        &self,
        confidences: &[f64],
        outcomes: &[bool],
    ) -> Result<Vec<Bin>, MetricError> {
        if confidences.len() != outcomes.len() {
            return Err(MetricError::LengthMismatch {
                left_name: "confidences",
                left: confidences.len(),
                right_name: "outcomes",
                right: outcomes.len(),
            });
        }
        if confidences.is_empty() {
            return Err(MetricError::EmptyInput("bin summarization"));
        }
        let n_bins = self.n_bins();
        let mut counts = vec![0usize; n_bins];
        let mut conf_sums = vec![0.0f64; n_bins];
        let mut out_sums = vec![0.0f64; n_bins];
        for (&c, &o) in confidences.iter().zip(outcomes) {
            let b = self.assign(c);
            counts[b] += 1;
            conf_sums[b] += c;
            out_sums[b] += f64::from(o);
        }
        let mut bins = Vec::with_capacity(n_bins);
        for (i, (&count, (&cs, &os))) in counts
            .iter()
            .zip(conf_sums.iter().zip(out_sums.iter()))
            .enumerate()
        {
            if count == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let n = count as f64;
            bins.push(Bin {
                index: i,
                lower: self.edges[i],
                upper: self.edges[i + 1],
                count,
                mean_confidence: cs / n,
                mean_outcome: os / n,
            });
        }
        Ok(bins)
    }
}
