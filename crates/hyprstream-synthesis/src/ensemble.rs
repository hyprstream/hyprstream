//! Ensemble combination over the persisted **raw** teacher vectors.
//!
//! Corrections are fit by P0.5 (per-teacher temperatures on verifiable
//! families) and applied here, at consumption time — the corpus itself stays
//! raw. Named risk (plan P1.3): corrections fit on verifiable families may
//! not transfer to ambiguous judgments; the fallback is
//! [`drop_worst_teachers`] + [`spread`].

use std::collections::HashMap;

use crate::corpus::CorpusRow;

/// Errors from ensemble combination.
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleError {
    /// A correction referenced a teacher id not present on the row.
    UnknownTeacher(String),
    /// A row has no teacher answers.
    EmptyRoster,
    /// Teacher identities are empty or duplicated.
    BadRoster(String),
    /// A probability vector had the wrong length or was degenerate.
    BadVector {
        /// Teacher id.
        teacher: String,
    },
    /// A correction temperature was non-finite or non-positive.
    BadTemperature {
        /// Teacher id.
        teacher: String,
    },
}

impl std::fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRoster => f.write_str("teacher roster is empty"),
            Self::BadRoster(id) => write!(f, "empty or duplicate teacher identity {id:?}"),
            Self::UnknownTeacher(id) => write!(f, "no answer from teacher {id} on this row"),
            Self::BadVector { teacher } => {
                write!(f, "teacher {teacher} answer is missing or the wrong width")
            }
            Self::BadTemperature { teacher } => write!(
                f,
                "correction temperature for teacher {teacher} is not finite and positive"
            ),
        }
    }
}

impl std::error::Error for EnsembleError {}

/// The producer contract, shared by ingestion and imported-row consumers.
pub(crate) fn valid_probabilities(probs: &[f32], width: usize) -> bool {
    width >= 2
        && probs.len() == width
        && probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && (probs.iter().map(|p| f64::from(*p)).sum::<f64>() - 1.0).abs() <= 1e-6
}

fn validate_row(row: &CorpusRow) -> Result<(), EnsembleError> {
    if row.teachers.is_empty() {
        return Err(EnsembleError::EmptyRoster);
    }
    let mut ids = std::collections::HashSet::new();
    for answer in &row.teachers {
        if answer.id.trim().is_empty()
            || answer.version.trim().is_empty()
            || !ids.insert(&answer.id)
        {
            return Err(EnsembleError::BadRoster(answer.id.clone()));
        }
        if !valid_probabilities(&answer.probs, row.cardinality()) {
            return Err(EnsembleError::BadVector {
                teacher: answer.id.clone(),
            });
        }
    }
    Ok(())
}

/// Temperature-correct one raw vector: `softmax(ln(p) / T)`. `T = 1.0` is
/// the identity. Non-positive entries are floored before the log.
fn temper(probs: &[f32], temperature: f64) -> Vec<f64> {
    let logits: Vec<f64> = probs
        .iter()
        .map(|p| f64::from(*p).max(1e-12).ln() / temperature)
        .collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// The corrected teacher-average distribution for a row. `temperatures`
/// maps roster id → fitted temperature (P0.5); teachers without an entry
/// get `T = 1.0`.
pub fn corrected_average(
    row: &CorpusRow,
    temperatures: &HashMap<String, f64>,
) -> Result<Vec<f64>, EnsembleError> {
    validate_row(row)?;
    for (id, temperature) in temperatures {
        if !row.teachers.iter().any(|answer| &answer.id == id) {
            return Err(EnsembleError::UnknownTeacher(id.clone()));
        }
        // This is the training-time correction path: a zero/negative/NaN/
        // infinite temperature would produce NaN distributions or silently
        // invert the learned probabilities, so it fails loud here.
        if !temperature.is_finite() || *temperature <= 0.0 {
            return Err(EnsembleError::BadTemperature {
                teacher: id.clone(),
            });
        }
    }
    let n = row.cardinality();
    let mut sum = vec![0.0f64; n];
    for answer in &row.teachers {
        if answer.probs.len() != n {
            return Err(EnsembleError::BadVector {
                teacher: answer.id.clone(),
            });
        }
        let temperature = temperatures.get(&answer.id).copied().unwrap_or(1.0);
        for (slot, p) in sum.iter_mut().zip(temper(&answer.probs, temperature)) {
            *slot += p;
        }
    }
    let total = row.teachers.len() as f64;
    let result: Vec<f64> = sum.iter().map(|s| s / total).collect();
    if result
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (result.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err(EnsembleError::BadVector {
            teacher: "ensemble".into(),
        });
    }
    Ok(result)
}

/// The ensemble argmax label (index into canonical labels), ties broken
/// toward the earliest label (D6).
pub fn argmax_label(distribution: &[f64]) -> usize {
    distribution
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .map_or(0, |(index, _)| index)
}

/// Per-label standard deviation across teachers (uncorrected): the ensemble
/// spread used by the agreement metrics and the named fallback.
pub fn spread(row: &CorpusRow) -> Result<Vec<f64>, EnsembleError> {
    validate_row(row)?;
    let n = row.cardinality();
    let mut mean = vec![0.0f64; n];
    for answer in &row.teachers {
        for (slot, p) in mean.iter_mut().zip(&answer.probs) {
            *slot += f64::from(*p);
        }
    }
    let total = row.teachers.len().max(1) as f64;
    for slot in &mut mean {
        *slot /= total;
    }
    let mut var = vec![0.0f64; n];
    for answer in &row.teachers {
        for ((slot, p), m) in var.iter_mut().zip(&answer.probs).zip(&mean) {
            let d = f64::from(*p) - m;
            *slot += d * d;
        }
    }
    Ok(var.iter().map(|v| (v / total).sqrt()).collect())
}

/// Named fallback: drop the teachers whose raw vectors disagree most with
/// the ensemble mean, keeping `keep` of them. Returns the kept teacher ids
/// (deterministic: largest mean L1 distance dropped first, ties by id).
pub fn drop_worst_teachers(row: &CorpusRow, keep: usize) -> Result<Vec<String>, EnsembleError> {
    validate_row(row)?;
    let n = row.cardinality();
    let mut mean = vec![0.0f64; n];
    for answer in &row.teachers {
        for (slot, p) in mean.iter_mut().zip(&answer.probs) {
            *slot += f64::from(*p);
        }
    }
    let total = row.teachers.len().max(1) as f64;
    for slot in &mut mean {
        *slot /= total;
    }
    let mut ranked: Vec<(f64, &str)> = row
        .teachers
        .iter()
        .map(|answer| {
            let distance: f64 = answer
                .probs
                .iter()
                .zip(&mean)
                .map(|(p, m)| (f64::from(*p) - m).abs())
                .sum::<f64>()
                / n.max(1) as f64;
            (distance, answer.id.as_str())
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    Ok(ranked
        .into_iter()
        .take(keep)
        .map(|(_, id)| id.to_owned())
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::corpus::{Provenance, TeacherAnswer};

    fn row_with(probs: Vec<Vec<f32>>) -> CorpusRow {
        let n = probs.first().map_or(2, Vec::len);
        CorpusRow {
            id: "syn1_x_noul_0000000000000001_r0_p0".to_owned(),
            family: "triage".to_owned(),
            group: "g".to_owned(),
            kind: "noul".to_owned(),
            blake3: "0".repeat(64),
            seed: 1,
            rotation: 0,
            paraphrase: 0,
            state: String::new(),
            instructions: None,
            labels: (0..n).map(|i| i.to_string()).collect(),
            rubrics: Vec::new(),
            noul_criteria: None,
            teachers: probs
                .into_iter()
                .enumerate()
                .map(|(i, probs)| TeacherAnswer {
                    id: format!("t{i}"),
                    version: "simulated".to_owned(),
                    tos_class: "open-weights".to_owned(),
                    probs,
                })
                .collect(),
            distributable: true,
            provenance: Provenance {
                generator: "test".to_owned(),
                benchmark_release: hyprstream_bench::RELEASE.to_owned(),
                manifest_blake3: "0".repeat(64),
            },
        }
    }

    #[test]
    fn average_is_plain_mean_without_corrections() {
        let row = row_with(vec![vec![0.8, 0.2], vec![0.4, 0.6]]);
        let avg = corrected_average(&row, &HashMap::new()).unwrap();
        assert!((avg[0] - 0.6).abs() < 1e-6);
        assert!((avg[1] - 0.4).abs() < 1e-6);
        assert_eq!(argmax_label(&avg), 0);
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let row = row_with(vec![vec![0.7, 0.3]]);
        let mut hot = HashMap::new();
        hot.insert("t0".to_owned(), 2.0);
        let flattened = corrected_average(&row, &hot).unwrap();
        assert!(flattened[0] < 0.7 && flattened[0] > 0.5);
        let mut cold = HashMap::new();
        cold.insert("t0".to_owned(), 0.5);
        let sharpened = corrected_average(&row, &cold).unwrap();
        assert!(sharpened[0] > 0.7);
    }

    #[test]
    fn unknown_teacher_correction_fails_loudly() {
        let row = row_with(vec![vec![0.5, 0.5]]);
        let mut corrections = HashMap::new();
        corrections.insert("ghost".to_owned(), 1.0);
        assert!(matches!(
            corrected_average(&row, &corrections),
            Err(EnsembleError::UnknownTeacher(id)) if id == "ghost"
        ));
    }

    #[test]
    fn invalid_temperatures_fail_loudly() {
        let row = row_with(vec![vec![0.7, 0.3]]);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut corrections = HashMap::new();
            corrections.insert("t0".to_owned(), bad);
            assert!(
                matches!(
                    corrected_average(&row, &corrections),
                    Err(EnsembleError::BadTemperature { .. })
                ),
                "temperature {bad} must be rejected"
            );
        }
        // Valid values still pass.
        let mut corrections = HashMap::new();
        corrections.insert("t0".to_owned(), 0.5);
        assert!(corrected_average(&row, &corrections).is_ok());
    }

    #[test]
    fn spread_is_zero_for_unanimous_teachers() {
        let row = row_with(vec![vec![0.9, 0.1], vec![0.9, 0.1]]);
        let s = spread(&row).unwrap();
        assert!(s.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn drop_worst_keeps_closest_teachers() {
        let row = row_with(vec![vec![0.8, 0.2], vec![0.75, 0.25], vec![0.1, 0.9]]);
        let kept = drop_worst_teachers(&row, 2).unwrap();
        assert_eq!(kept, vec!["t1".to_owned(), "t0".to_owned()]);
    }

    #[test]
    fn argmax_breaks_ties_to_earliest_label() {
        assert_eq!(argmax_label(&[0.5, 0.5]), 0);
        assert_eq!(argmax_label(&[0.1, 0.9, 0.0]), 1);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod rejection_tests {
    use super::*;
    use crate::corpus::{Provenance, TeacherAnswer};
    fn row(probs: Vec<f32>) -> CorpusRow {
        CorpusRow {
            id: "x".into(),
            family: "triage".into(),
            group: "x".into(),
            kind: "noul".into(),
            blake3: "x".into(),
            seed: 0,
            rotation: 0,
            paraphrase: 0,
            state: "s".into(),
            instructions: None,
            labels: vec!["false".into(), "true".into()],
            rubrics: vec![],
            noul_criteria: None,
            teachers: vec![TeacherAnswer {
                id: "t".into(),
                version: "v".into(),
                tos_class: "open-weights".into(),
                probs,
            }],
            distributable: true,
            provenance: Provenance {
                generator: "fixture".into(),
                benchmark_release: "tiny".into(),
                manifest_blake3: "tiny".into(),
            },
        }
    }
    #[test]
    fn imported_rows_reject_every_invalid_vector_before_ensemble_math() {
        for probs in [
            vec![f32::NAN, 1.0],
            vec![1.0, f32::NAN],
            vec![f32::INFINITY, 0.0],
            vec![0.0, f32::NEG_INFINITY],
            vec![-0.1, 1.1],
            vec![1.1, -0.1],
            vec![0.0, 0.0],
            vec![0.2, 0.2],
            vec![1.0],
            vec![],
        ] {
            let row = row(probs);
            assert!(corrected_average(&row, &HashMap::new()).is_err());
            assert!(spread(&row).is_err());
            assert!(drop_worst_teachers(&row, 1).is_err());
        }
        for probs in [vec![0.0, 1.0], vec![1.0, 0.0], vec![0.25, 0.75]] {
            let row = row(probs);
            let p = corrected_average(&row, &HashMap::new()).unwrap();
            assert!(p.iter().all(|p| p.is_finite()));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }
    #[test]
    fn imported_rosters_reject_empty_duplicate_and_unversioned_answers() {
        let mut r = row(vec![0.5, 0.5]);
        r.teachers.clear();
        assert_eq!(
            corrected_average(&r, &HashMap::new()),
            Err(EnsembleError::EmptyRoster)
        );
        let mut r = row(vec![0.5, 0.5]);
        r.teachers.push(r.teachers[0].clone());
        assert!(corrected_average(&r, &HashMap::new()).is_err());
        for field in [0, 1] {
            let mut r = row(vec![0.5, 0.5]);
            if field == 0 {
                r.teachers[0].id = " ".into();
            } else {
                r.teachers[0].version.clear();
            }
            assert!(corrected_average(&r, &HashMap::new()).is_err());
        }
    }
}
