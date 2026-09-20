//! Scoring: observations → the pinned measurement protocol's numbers.
//!
//! Fields are the macro rule's unit. For [`EvalSet`](crate::run::EvalSet)
//! runs a field is a declared question; for item runs (the bench shape, where
//! every item carries its own question id) fields are **family × kind ×
//! cardinality** groups — calibration metrics consume probability vectors and
//! truth indices, so items with different label sets still pool correctly,
//! while cardinality stays fixed inside a field so the ordinal metrics (SCE /
//! ACE / RPS) are well-defined.
//!
//! Everything numeric goes through [`hyprstream_calibration`]: ECE/Brier/NLL
//! under the 15 equal-mass bin gate protocol, SCE/ACE + RPS for the `score`
//! primitive, macro averages with bootstrap CIs, all assembled into the
//! protocol's [`GateReport`]. Abstained observations are excluded from
//! calibration (there is no distribution to score) and counted in the
//! abstention rate — selective prediction is downstream's job.

use std::collections::BTreeMap;

use hyprstream_calibration::metrics;
use hyprstream_calibration::protocol::{
    FieldMetrics, GateReport, ShiftSplit, GATE_BIN_COUNT,
};
use hyprstream_decision::spec::QuestionKind;

use crate::error::EvalError;
use crate::run::RunOutput;
use crate::teacher::EnsembleOutput;

/// Scoring knobs.
#[derive(Debug, Clone, Copy)]
pub struct ScoreConfig {
    /// Bootstrap seed for all CIs (gate reports pin theirs; a run report pins
    /// its own so numbers are reproducible).
    pub bootstrap_seed: u64,
    /// Fit families for the report's shift split (eval families are taken
    /// from the observations). Defaults to none — a pure-eval report.
    pub split_name: &'static str,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            bootstrap_seed: 0xE4A1,
            split_name: "eval",
        }
    }
}

/// Per-field metrics for one breakdown dimension (family or stratum).
#[derive(Debug, Clone, PartialEq)]
pub struct BreakdownMetrics {
    /// The dimension value (e.g. `arith`, `nearmiss`).
    pub key: String,
    /// Rows scored in this bucket.
    pub n: usize,
    /// Macro-averaged ECE over the bucket's fields (equal field weights).
    pub macro_ece: f64,
    /// Argmax accuracy over the bucket's scored rows.
    pub accuracy: f64,
    /// Abstention rate over the bucket's rows.
    pub abstention_rate: f64,
}

/// Option-order robustness over a permutation group (S6b1): the fraction of
/// group members whose subject argmax **label** differs from the base
/// member's. Labels are compared as canonical label indices mapped through
/// each member's own label list — for cyclic rotations of one choice item the
/// correct option keeps its name while its index rotates, so label comparison
/// is the semantics ("the same answer"), index comparison is not.
#[derive(Debug, Clone, PartialEq)]
pub struct FlipRate {
    /// Groups that contributed (≥ 2 answered members).
    pub groups: usize,
    /// Fraction of non-base members disagreeing with their base's argmax label.
    pub flip_rate: f64,
}

/// Teacher-ensemble agreement summary for a run.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleAgreement {
    /// Mean per-item argmax agreement (fraction of teachers matching the
    /// ensemble argmax), averaged over items and questions.
    pub mean_argmax_agreement: f64,
    /// Per-item agreement values (reporting/distribution input).
    pub per_item: Vec<(String, f64)>,
}

/// One item's score line (the smallest report unit).
#[derive(Debug, Clone, PartialEq)]
pub struct ItemScore {
    /// Item/row id.
    pub id: String,
    /// Question id.
    pub question_id: String,
    /// Whether the subject abstained.
    pub abstained: bool,
    /// Whether the argmax matched truth (`None` when unlabeled or abstained).
    pub correct: Option<bool>,
}

/// A scored run.
#[derive(Debug)]
pub struct ScoreReport {
    /// The gate-protocol report (per-field metrics + macro CIs). `None` when
    /// no labeled, answered observations exist.
    pub gate: Option<GateReport>,
    /// Per-family breakdown.
    pub by_family: Vec<BreakdownMetrics>,
    /// Per-stratum breakdown.
    pub by_stratum: Vec<BreakdownMetrics>,
    /// Option-order flip rate over permutation groups.
    pub flip_rate: FlipRate,
    /// Overall abstention rate.
    pub abstention_rate: f64,
    /// Argmax accuracy over labeled, answered observations.
    pub accuracy: Option<f64>,
    /// Per-item lines.
    pub items: Vec<ItemScore>,
}

fn argmax(probs: &[f64]) -> Option<usize> {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

/// Field key for an observation: the question id for declared questions whose
/// id is shared across rows (set runs), else the family × kind × cardinality
/// group (item runs). We cannot distinguish the two cases per-observation, so
/// the caller picks the mode; [`score_run`] uses the item-group mode when the
/// run has no batch (loose items) and the question-id mode otherwise.
fn field_key(obs: &crate::run::Observation, by_question: bool) -> String {
    if by_question {
        return obs.question_id.clone();
    }
    let family = obs.family.as_deref().unwrap_or("unknown");
    let cardinality = obs
        .probabilities
        .as_ref()
        .map_or(0, std::vec::Vec::len);
    format!("{family}/{}:{cardinality}", obs.kind)
}

fn field_metrics(
    field: &str,
    family: &str,
    kind: QuestionKind,
    probs: &[&[f64]],
    labels: &[usize],
) -> Result<FieldMetrics, EvalError> {
    let ordinal = kind == QuestionKind::Score;
    Ok(FieldMetrics {
        field: field.to_owned(),
        family: family.to_owned(),
        n: labels.len(),
        ece: metrics::ece(probs, labels, GATE_BIN_COUNT)?,
        brier: metrics::brier(probs, labels)?,
        sce: ordinal.then(|| metrics::sce(probs, labels, GATE_BIN_COUNT)).transpose()?,
        ace: ordinal.then(|| metrics::ace(probs, labels, GATE_BIN_COUNT)).transpose()?,
        rps: ordinal.then(|| metrics::rps(probs, labels)).transpose()?,
    })
}

/// Score a run's observations under the pinned protocol.
pub fn score_run(output: &RunOutput, config: &ScoreConfig) -> Result<ScoreReport, EvalError> {
    let by_question = output.batch.is_some();
    // Group labeled, answered observations into fields.
    let mut fields: BTreeMap<String, (QuestionKind, String, Vec<Vec<f64>>, Vec<usize>)> =
        BTreeMap::new();
    let mut items = Vec::with_capacity(output.observations.len());
    let mut abstained = 0usize;
    let mut correct = 0usize;
    let mut labeled_answered = 0usize;
    for obs in &output.observations {
        let (is_abstained, is_correct) = match (&obs.probabilities, obs.truth) {
            (Some(probs), Some(truth)) => {
                let probs64: Vec<f64> = probs.iter().map(|p| f64::from(*p)).collect();
                let hit = argmax(&probs64) == Some(truth);
                let key = field_key(obs, by_question);
                let family = obs.family.clone().unwrap_or_else(|| "unknown".to_owned());
                let entry = fields
                    .entry(key)
                    .or_insert_with(|| (obs.kind, family, Vec::new(), Vec::new()));
                entry.2.push(probs64);
                entry.3.push(truth);
                labeled_answered += 1;
                if hit {
                    correct += 1;
                }
                (false, Some(hit))
            }
            (Some(_), None) => (false, None),
            (None, _) => {
                abstained += 1;
                (true, None)
            }
        };
        items.push(ItemScore {
            id: obs.id.clone(),
            question_id: obs.question_id.clone(),
            abstained: is_abstained,
            correct: is_correct,
        });
    }

    let mut field_rows = Vec::with_capacity(fields.len());
    for (key, (kind, family, probs, labels)) in &fields {
        let refs: Vec<&[f64]> = probs.iter().map(std::vec::Vec::as_slice).collect();
        field_rows.push(field_metrics(key, family, *kind, &refs, labels)?);
    }

    let eval_families: Vec<String> = {
        let mut families: Vec<String> = output
            .observations
            .iter()
            .filter_map(|obs| obs.family.clone())
            .collect();
        families.sort();
        families.dedup();
        families
    };
    let gate = if field_rows.is_empty() {
        None
    } else {
        Some(GateReport::assemble(
            output.model_id.clone(),
            config.split_name,
            ShiftSplit::new(std::iter::empty::<String>(), eval_families)?,
            field_rows,
            config.bootstrap_seed,
        )?)
    };

    #[allow(clippy::cast_precision_loss)]
    let total = output.observations.len();
    let abstention_rate = if total == 0 {
        0.0
    } else {
        abstained as f64 / total as f64
    };
    #[allow(clippy::cast_precision_loss)]
    let accuracy = (labeled_answered > 0).then(|| correct as f64 / labeled_answered as f64);

    Ok(ScoreReport {
        gate,
        by_family: breakdown(&output.observations, |obs| obs.family.clone()),
        by_stratum: breakdown(&output.observations, |obs| obs.stratum.clone()),
        flip_rate: flip_rate_with_labels(&output.observations, no_labels),
        abstention_rate,
        accuracy,
        items,
    })
}

/// Macro-ECE + accuracy + abstention for one breakdown dimension.
fn breakdown(
    observations: &[crate::run::Observation],
    key_of: impl Fn(&crate::run::Observation) -> Option<String>,
) -> Vec<BreakdownMetrics> {
    let mut buckets: BTreeMap<String, Vec<&crate::run::Observation>> = BTreeMap::new();
    for obs in observations {
        if let Some(key) = key_of(obs) {
            buckets.entry(key).or_default().push(obs);
        }
    }
    buckets
        .into_iter()
        .map(|(key, bucket)| {
            // Fields inside the bucket follow the same family×kind×card rule.
            let mut fields: BTreeMap<String, (Vec<Vec<f64>>, Vec<usize>)> = BTreeMap::new();
            let mut correct = 0usize;
            let mut labeled = 0usize;
            let mut abstained = 0usize;
            for obs in &bucket {
                match (&obs.probabilities, obs.truth) {
                    (Some(probs), Some(truth)) => {
                        let probs64: Vec<f64> = probs.iter().map(|p| f64::from(*p)).collect();
                        if argmax(&probs64) == Some(truth) {
                            correct += 1;
                        }
                        labeled += 1;
                        let family = obs.family.as_deref().unwrap_or("unknown");
                        let field = format!("{family}/{}:{}", obs.kind, probs.len());
                        let entry = fields.entry(field).or_default();
                        entry.0.push(probs64);
                        entry.1.push(truth);
                    }
                    (Some(_), None) => {}
                    (None, _) => abstained += 1,
                }
            }
            let eces: Vec<f64> = fields
                .values()
                .filter_map(|(probs, labels)| {
                    let refs: Vec<&[f64]> = probs.iter().map(std::vec::Vec::as_slice).collect();
                    metrics::ece(&refs, labels, GATE_BIN_COUNT).ok()
                })
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let n = bucket.len();
            BreakdownMetrics {
                key,
                n,
                macro_ece: hyprstream_calibration::protocol::macro_average(&eces).unwrap_or(0.0),
                #[allow(clippy::cast_precision_loss)]
                accuracy: if labeled == 0 {
                    0.0
                } else {
                    correct as f64 / labeled as f64
                },
                #[allow(clippy::cast_precision_loss)]
                abstention_rate: if n == 0 {
                    0.0
                } else {
                    abstained as f64 / n as f64
                },
            }
        })
        .collect()
}

/// Option-order flip rate across permutation groups. Within a group the base
/// member is the one whose id equals the group id; every other member's
/// argmax **label** must equal the base's. Members without labels resolvable
/// through their question are skipped — the caller supplies each
/// observation's canonical labels via `labels_of`.
pub fn flip_rate_with_labels(
    observations: &[crate::run::Observation],
    labels_of: impl Fn(&str) -> Option<Vec<String>>,
) -> FlipRate {
    let mut groups: BTreeMap<&str, Vec<&crate::run::Observation>> = BTreeMap::new();
    for obs in observations {
        if let (Some(group), Some(_)) = (&obs.group, &obs.probabilities) {
            groups.entry(group.as_str()).or_default().push(obs);
        }
    }
    let label_of = |obs: &crate::run::Observation| -> Option<String> {
        let labels = labels_of(&obs.question_id)?;
        let index = obs.argmax_label_index()?;
        labels.get(index).cloned()
    };
    let mut used = 0usize;
    let mut flips = 0usize;
    let mut counted_groups = 0usize;
    for (base_id, members) in groups {
        if members.len() < 2 {
            continue;
        }
        let Some(base) = members.iter().find(|obs| obs.id == base_id) else {
            continue;
        };
        let Some(base_label) = label_of(base) else {
            continue;
        };
        let mut counted = false;
        for member in members.iter().filter(|member| member.id != base_id) {
            if let Some(label) = label_of(member) {
                used += 1;
                counted = true;
                if label != base_label {
                    flips += 1;
                }
            }
        }
        if counted {
            counted_groups += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    FlipRate {
        groups: counted_groups,
        flip_rate: if used == 0 {
            0.0
        } else {
            flips as f64 / used as f64
        },
    }
}

/// Flip rate for runs whose labels are id-addressable. `score_run` cannot
/// reconstruct labels from observations alone (indices rotate under
/// permutation, so index comparison would report spurious flips — S6b1), so
/// it leaves the flip rate empty; callers with items in hand must use
/// [`flip_rate_with_labels`] (or [`score_bench_run`], which does it for them).
fn no_labels(_: &str) -> Option<Vec<String>> {
    None
}

/// Score a bench-shaped run: [`score_run`] plus the exact flip rate computed
/// with each item's own labels.
pub fn score_bench_run(
    output: &RunOutput,
    items: &[crate::run::EvalItem],
    config: &ScoreConfig,
) -> Result<ScoreReport, EvalError> {
    let mut report = score_run(output, config)?;
    report.flip_rate = flip_rate_with_labels(&output.observations, |question_id| {
        items
            .iter()
            .find(|item| item.question.id == question_id)
            .map(|item| item.question.labels())
    });
    Ok(report)
}

/// Summarize teacher-ensemble agreement over per-item outputs.
pub fn agreement_summary(outputs: &[(String, EnsembleOutput)]) -> EnsembleAgreement {
    let per_item: Vec<(String, f64)> = outputs
        .iter()
        .map(|(id, out)| {
            let values: Vec<f64> = out.argmax_agreement.values().copied().collect();
            let mean = hyprstream_calibration::protocol::macro_average(&values).unwrap_or(0.0);
            (id.clone(), mean)
        })
        .collect();
    let mean_argmax_agreement =
        hyprstream_calibration::protocol::macro_average(
            &per_item.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
        )
        .unwrap_or(0.0);
    EnsembleAgreement {
        mean_argmax_agreement,
        per_item,
    }
}
