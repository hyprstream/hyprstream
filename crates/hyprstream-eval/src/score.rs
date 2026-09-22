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
#[derive(Debug, Clone)]
pub struct ScoreConfig {
    /// Bootstrap seed for all CIs (gate reports pin theirs; a run report pins
    /// its own so numbers are reproducible).
    pub bootstrap_seed: u64,
    /// The split name recorded on the gate report (e.g. `"deterministic"`,
    /// `"shift"`, `"zero-shot"`).
    pub split_name: &'static str,
    /// Fit families for the report's shift split: families whose items were
    /// synthesized into training data, used for distillation targets, or used
    /// to fit calibration parameters. Observed families NOT in this list are
    /// classified as held-out evaluation families, and observations tagged
    /// with a declared fit family are excluded from the gate's field rows
    /// entirely — they are training data, and silently macro-averaging them
    /// into a purported held-out gate would corrupt the measurement. Defaults
    /// to none — a pure-eval report.
    pub fit_families: Vec<String>,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            bootstrap_seed: 0xE4A1,
            split_name: "eval",
            fit_families: Vec::new(),
        }
    }
}

impl ScoreConfig {
    /// Declare the fit families (see [`ScoreConfig::fit_families`]).
    pub fn with_fit_families(mut self, families: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.fit_families = families.into_iter().map(Into::into).collect();
        self
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
    /// `None` when the bucket has no scored rows — an unmeasured calibration
    /// value is never reported as a perfect `0.0`.
    pub macro_ece: Option<f64>,
    /// Argmax accuracy over the bucket's scored rows. `None` when the bucket
    /// has no labeled, answered rows — unavailable, not zero accuracy.
    pub accuracy: Option<f64>,
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
    /// Option-order flip rate over permutation groups. `None` = **not
    /// computed** (labels were not available to the scorer — see
    /// [`flip_rate_with_labels`]); `Some` with `flip_rate == 0.0` = computed,
    /// no flips. The distinction matters: an uncomputed rate must never be
    /// misread as permutation-robustness.
    pub flip_rate: Option<FlipRate>,
    /// Overall abstention rate.
    pub abstention_rate: f64,
    /// Argmax accuracy over labeled, answered observations.
    pub accuracy: Option<f64>,
    /// Per-field negative log-likelihood, aligned with the gate report's
    /// field rows. (P0.2's `FieldMetrics` type has no NLL slot and is owned
    /// by the calibration crate, so NLL rides alongside the gate report here
    /// rather than inside it.) A field whose realized labels ever sit at
    /// zero probability reports `f64::INFINITY` — a legitimate measurement,
    /// not a scoring failure.
    pub nll_by_field: Vec<(String, f64)>,
    /// Per-item lines.
    pub items: Vec<ItemScore>,
}

/// Field key for an observation. Set runs key by question id (ids are shared
/// across rows), prefixing the family when the run spans more than one family
/// so families are never pooled into one field (the macro rule averages
/// fields with equal weight — pooling would dilute a miscalibrated rare
/// family). Item runs key by family × kind × cardinality, since every item
/// carries its own question id. The caller picks the mode; [`score_run`] uses
/// the item-group mode when the run has no batch (loose items) and the
/// question-id mode otherwise.
fn field_key(obs: &crate::run::Observation, by_question: bool, multi_family: bool) -> String {
    let family = obs.family.as_deref().unwrap_or("unknown");
    if by_question {
        if multi_family {
            return format!("{family}/{}", obs.question_id);
        }
        return obs.question_id.clone();
    }
    let cardinality = obs
        .probabilities
        .as_ref()
        .map_or(0, std::vec::Vec::len);
    format!("{family}/{}:{cardinality}", obs.kind)
}

/// Number of distinct family tags in the run (`None` family counts as one).
fn family_count(observations: &[crate::run::Observation]) -> usize {
    let mut families: Vec<&str> = observations
        .iter()
        .map(|obs| obs.family.as_deref().unwrap_or("unknown"))
        .collect();
    families.sort_unstable();
    families.dedup();
    families.len()
}

/// Per-field metrics plus the field's NLL (returned separately because
/// P0.2's `FieldMetrics` has no NLL slot — see [`ScoreReport::nll_by_field`]).
fn field_metrics(
    field: &str,
    family: &str,
    kind: QuestionKind,
    probs: &[&[f64]],
    labels: &[usize],
) -> Result<(FieldMetrics, f64), EvalError> {
    let ordinal = kind == QuestionKind::Score;
    // A valid distribution may assign zero probability to the realized label
    // (an imperfect one-hot classifier): NLL is then legitimately infinite,
    // not a scoring failure — the rest of the report must still assemble.
    let nll = metrics::nll(probs, labels).unwrap_or(f64::INFINITY);
    Ok((
        FieldMetrics {
            field: field.to_owned(),
            family: family.to_owned(),
            n: labels.len(),
            ece: metrics::ece(probs, labels, GATE_BIN_COUNT)?,
            brier: metrics::brier(probs, labels)?,
            sce: ordinal.then(|| metrics::sce(probs, labels, GATE_BIN_COUNT)).transpose()?,
            ace: ordinal.then(|| metrics::ace(probs, labels, GATE_BIN_COUNT)).transpose()?,
            rps: ordinal.then(|| metrics::rps(probs, labels)).transpose()?,
        },
        nll,
    ))
}

/// Score a run's observations under the pinned protocol.
pub fn score_run(output: &RunOutput, config: &ScoreConfig) -> Result<ScoreReport, EvalError> {
    let by_question = output.batch.is_some();
    let multi_family = family_count(&output.observations) > 1;
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
                let hit = metrics::argmax(&probs64) == truth;
                let key = field_key(obs, by_question, multi_family);
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
    let mut nll_by_field = Vec::with_capacity(fields.len());
    for (key, (kind, family, probs, labels)) in &fields {
        // Declared fit families are training data: their observations must
        // not enter the held-out gate's macro average (the split metadata
        // alone does not exclude them).
        if config.fit_families.contains(family) {
            continue;
        }
        let refs: Vec<&[f64]> = probs.iter().map(std::vec::Vec::as_slice).collect();
        let (metrics_row, nll) = field_metrics(key, family, *kind, &refs, labels)?;
        field_rows.push(metrics_row);
        nll_by_field.push((key.clone(), nll));
    }

    // Held-out eval families = observed families minus declared fit families.
    // Untagged observations score under the same "unknown" sentinel the field
    // construction uses, so the split always names every family in the gate.
    let eval_families: Vec<String> = {
        let mut families: Vec<String> = output
            .observations
            .iter()
            .map(|obs| obs.family.clone().unwrap_or_else(|| "unknown".to_owned()))
            .filter(|family| !config.fit_families.contains(family))
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
            ShiftSplit::new(config.fit_families.clone(), eval_families)?,
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
        by_family: breakdown(&output.observations, by_question, multi_family, |obs| {
            // Untagged rows score under the same "unknown" sentinel the gate
            // and shift split use — omitting them would leave the family
            // breakdown irreconcilable with the gate.
            Some(obs.family.clone().unwrap_or_else(|| "unknown".to_owned()))
        }),
        by_stratum: breakdown(&output.observations, by_question, multi_family, |obs| {
            obs.stratum.clone()
        }),
        flip_rate: None,
        abstention_rate,
        accuracy,
        nll_by_field,
        items,
    })
}

/// Macro-ECE + accuracy + abstention for one breakdown dimension. Field
/// keying inside a bucket follows the same rule as the top-level fields
/// (question id for set runs — two same-kind questions never merge into one
/// field — family × kind × cardinality for item runs).
fn breakdown(
    observations: &[crate::run::Observation],
    by_question: bool,
    multi_family: bool,
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
            let mut fields: BTreeMap<String, (Vec<Vec<f64>>, Vec<usize>)> = BTreeMap::new();
            let mut correct = 0usize;
            let mut labeled = 0usize;
            let mut abstained = 0usize;
            for obs in &bucket {
                match (&obs.probabilities, obs.truth) {
                    (Some(probs), Some(truth)) => {
                        let probs64: Vec<f64> = probs.iter().map(|p| f64::from(*p)).collect();
                        if metrics::argmax(&probs64) == truth {
                            correct += 1;
                        }
                        labeled += 1;
                        let field = field_key(obs, by_question, multi_family);
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
                macro_ece: hyprstream_calibration::protocol::macro_average(&eces),
                #[allow(clippy::cast_precision_loss)]
                accuracy: (labeled > 0).then(|| correct as f64 / labeled as f64),
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
/// argmax **label** must equal its base's. A multi-question `EvalSet` answers
/// EVERY question on the base row, so the group holds one base observation
/// per question — each member is then compared against the base observation
/// of its OWN question (comparing against another question's base label
/// would report spurious flips). A single-question (bench item) group has
/// exactly one base; cyclic rotations carry their own question ids and
/// label orders, and compare against that one base. Members without labels
/// resolvable through their question are skipped — the caller supplies each
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
        let bases: Vec<&crate::run::Observation> = members
            .iter()
            .filter(|obs| obs.id == base_id)
            .copied()
            .collect();
        if bases.is_empty() {
            continue;
        }
        let mut counted = false;
        for member in members.iter().filter(|member| member.id != base_id) {
            // The base for THIS member: with several base observations (a
            // multi-question set run) match the member's own question —
            // pooling across questions compares against another question's
            // label space and reports spurious flips. With exactly one base
            // (bench item groups) rotations carry their own question ids
            // and label orders, and compare against that base.
            let base = if let [base] = bases.as_slice() {
                *base
            } else {
                let Some(base) = bases
                    .iter()
                    .find(|base| base.question_id == member.question_id)
                else {
                    continue;
                };
                base
            };
            let Some(base_label) = label_of(base) else {
                continue;
            };
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

/// Score a bench-shaped run: [`score_run`] plus the exact flip rate computed
/// with each item's own labels. (`score_run` cannot reconstruct labels from
/// observations alone — indices rotate under permutation, so index comparison
/// would report spurious flips, S6b1 — and therefore reports `flip_rate:
/// None`, "not computed".)
pub fn score_bench_run(
    output: &RunOutput,
    items: &[crate::run::EvalItem],
    config: &ScoreConfig,
) -> Result<ScoreReport, EvalError> {
    let mut report = score_run(output, config)?;
    report.flip_rate = Some(flip_rate_with_labels(&output.observations, |question_id| {
        items
            .iter()
            .find(|item| item.question.id == question_id)
            .map(|item| item.question.labels())
    }));
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
