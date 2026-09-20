//! End-to-end harness tests over P0.6 bench items: a truth-answering subject
//! must score perfectly, a degenerate uniform subject must score at its
//! analytic baselines, and the permutation flip-rate machinery must separate
//! the two.

#![allow(clippy::unwrap_used)]

use hyprstream_bench::{generate_all, BenchConfig};
use hyprstream_decision::answer::{AnswerValue, QuestionAnswer};
use hyprstream_decision::arrow::AnswerRow;
use hyprstream_decision::entry::Entry;
use hyprstream_decision::spec::QuestionSet;
use hyprstream_eval::score::ScoreConfig;
use hyprstream_eval::{
    score_bench_run, score_run, EvalItem, Harness, Subject, TruthSubject,
};

fn small_items() -> Vec<hyprstream_bench::Item> {
    generate_all(&BenchConfig {
        seeds_per_stratum: 2,
        seed_base: 0x06,
    })
}

/// A subject that answers every question with the uniform distribution.
struct UniformSubject;

#[async_trait::async_trait]
impl Subject for UniformSubject {
    fn model_id(&self) -> &str {
        "uniform-1"
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let n = question.cardinality();
            let probabilities = vec![1.0 / n as f32; n];
            let value = match question.kind {
                hyprstream_decision::QuestionKind::Noul => AnswerValue::Noul { p_true: 0.5 },
                hyprstream_decision::QuestionKind::Choice => {
                    AnswerValue::Choice { probabilities }
                }
                hyprstream_decision::QuestionKind::Score => AnswerValue::Score { probabilities },
                other => panic!("reserved kind {other}"),
            };
            answers.insert(question.id.clone(), QuestionAnswer::answered(value));
        }
        Ok(AnswerRow { answers })
    }
}

fn truth_subject_for(items: &[EvalItem]) -> TruthSubject {
    let mut subject = TruthSubject::new("truth-1");
    for item in items {
        if let Some(truth) = item.truth {
            subject = subject.with_truth(item.question.id.clone(), truth);
        }
    }
    subject
}

#[tokio::test]
async fn truth_subject_scores_perfectly_on_bench_items() {
    let items: Vec<EvalItem> = small_items().iter().map(EvalItem::from).collect();
    assert!(items.len() > 50, "bench should expand, got {}", items.len());
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();

    assert_eq!(report.accuracy, Some(1.0));
    assert_eq!(report.abstention_rate, 0.0);
    let gate = report.gate.as_ref().unwrap();
    assert!(
        gate.macro_ece.mean < 0.05,
        "one-hot truth is (over)confident but never wrong: ECE ~ 0, got {}",
        gate.macro_ece.mean
    );
    // Score fields carry the ordinal metrics.
    assert!(gate.macro_rps.is_some());
    assert!(gate.macro_sce.is_some());
    assert!(gate.fields.iter().any(|f| f.rps.is_some()));
    // Per-family and per-stratum breakdowns cover every bucket.
    assert_eq!(report.by_family.len(), 5);
    assert!(report.by_stratum.iter().any(|b| b.key == "clean"));
    assert!(report.by_stratum.iter().any(|b| b.key == "nearmiss"));
    assert!(report.by_stratum.iter().any(|b| b.key.starts_with("perm-")));
    // Truth answers rotate with the options: computed, zero flips.
    let flip = report.flip_rate.as_ref().unwrap();
    assert!(flip.groups > 0);
    assert_eq!(flip.flip_rate, 0.0);
    for bucket in &report.by_family {
        assert_eq!(bucket.accuracy, 1.0, "family {}", bucket.key);
    }
}

#[tokio::test]
async fn uniform_subject_flips_and_miscalibrates() {
    let items: Vec<EvalItem> = small_items().iter().map(EvalItem::from).collect();
    let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();

    let gate = report.gate.as_ref().unwrap();
    // Uniform argmax is label 0 under the pinned D6 earliest-index tie-break
    // (score paths and the calibration internals agree on this); truth is
    // uniform-ish across labels, so accuracy is far below 1 and Brier is the
    // analytic uniform value.
    let accuracy = report.accuracy.unwrap();
    assert!(accuracy < 0.8, "uniform accuracy: {accuracy}");
    assert!(gate.macro_brier.mean > 0.3, "uniform Brier: {}", gate.macro_brier.mean);
    // Uniform argmax (D6 earliest-index) picks index 0 everywhere; rotations
    // move the correct label off index 0, so a label-comparing flip check
    // sees flips.
    let flip = report.flip_rate.as_ref().unwrap();
    assert!(flip.flip_rate > 0.5, "flip rate: {}", flip.flip_rate);
}

#[tokio::test]
async fn unlabeled_run_reports_no_gate() {
    let mut items: Vec<EvalItem> = small_items().iter().take(5).map(EvalItem::from).collect();
    for item in &mut items {
        item.truth = None;
    }
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_run(&output, &ScoreConfig::default()).unwrap();
    assert!(report.gate.is_none());
    assert!(report.accuracy.is_none());
    assert_eq!(report.items.len(), 5);
    assert!(report.items.iter().all(|line| line.correct.is_none()));
    // score_run has no label source: flip rate is "not computed", never a
    // silent zero.
    assert!(report.flip_rate.is_none());
}

#[tokio::test]
async fn argmax_ties_break_earliest_everywhere() {
    // D6 regression: a uniform noul [0.5, 0.5] with truth = 1 ("true") must
    // score as INCORRECT (argmax = index 0, "false") — and the report's
    // `correct` path must agree with the flip-label path
    // (`argmax_label_index`) and the calibration internals, all of which use
    // the pinned earliest-index tie-break.
    let item = EvalItem {
        id: "tie-1".into(),
        family: Some("ties".into()),
        stratum: None,
        group: None,
        state: hyprstream_decision::entry::Entry::Null,
        question: hyprstream_decision::QuestionSpec {
            id: "tie_q".into(),
            kind: hyprstream_decision::QuestionKind::Noul,
            instructions: None,
            body: hyprstream_decision::QuestionBody::Noul { criteria: None },
        },
        truth: Some(1),
    };
    let items = vec![item];
    let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
    let obs = &output.observations[0];
    assert_eq!(obs.probabilities.as_deref(), Some([0.5, 0.5].as_slice()));
    assert_eq!(obs.argmax_label_index(), Some(0), "flip-label path: D6 earliest");

    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    assert_eq!(report.items[0].correct, Some(false), "D6: argmax = index 0 on a tie");
    assert_eq!(report.accuracy, Some(0.0));
}

/// A subject that never answers the declared question (regression: the
/// harness must reject the row instead of scoring a silent gap).
struct SilentSubject;

#[async_trait::async_trait]
impl Subject for SilentSubject {
    fn model_id(&self) -> &str {
        "silent-1"
    }

    async fn decide(
        &self,
        _set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        Ok(AnswerRow {
            answers: std::collections::BTreeMap::new(),
        })
    }
}

/// A subject that answers a noul question with a Choice value (regression:
/// kind mismatches must be rejected at ingest).
struct WrongKindSubject;

#[async_trait::async_trait]
impl Subject for WrongKindSubject {
    fn model_id(&self) -> &str {
        "wrong-kind-1"
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            let n = question.cardinality();
            answers.insert(
                question.id.clone(),
                QuestionAnswer::answered(AnswerValue::Choice {
                    probabilities: vec![1.0 / n as f32; n],
                }),
            );
        }
        Ok(AnswerRow { answers })
    }
}

fn noul_item() -> EvalItem {
    EvalItem {
        id: "val-1".into(),
        family: Some("validation".into()),
        stratum: None,
        group: None,
        state: Entry::Null,
        question: hyprstream_decision::QuestionSpec {
            id: "val_q".into(),
            kind: hyprstream_decision::QuestionKind::Noul,
            instructions: None,
            body: hyprstream_decision::QuestionBody::Noul { criteria: None },
        },
        truth: Some(1),
    }
}

#[tokio::test]
async fn run_items_rejects_a_missing_answer() {
    let items = vec![noul_item()];
    let error = Harness.run_items(&items, &SilentSubject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidAnswer { .. }),
        "missing answer must be InvalidAnswer, got {error:?}"
    );
}

#[tokio::test]
async fn run_items_rejects_a_kind_mismatch() {
    let items = vec![noul_item()];
    let error = Harness
        .run_items(&items, &WrongKindSubject)
        .await
        .err()
        .unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidAnswer { .. }),
        "kind mismatch must be InvalidAnswer, got {error:?}"
    );
}

#[tokio::test]
async fn declared_fit_families_drive_the_shift_split() {
    let items: Vec<EvalItem> = small_items().iter().map(EvalItem::from).collect();
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(
        &output,
        &items,
        &ScoreConfig::default().with_fit_families(["arith"]),
    )
    .unwrap();
    let split = &report.gate.as_ref().unwrap().shift_split;
    assert_eq!(split.fit_families, vec!["arith".to_owned()]);
    assert!(
        !split.eval_families.iter().any(|f| f == "arith"),
        "a declared fit family must not also score as held-out: {:?}",
        split.eval_families
    );
    assert!(!split.eval_families.is_empty());
    // Fit-family observations are training data: none of their fields may
    // enter the held-out gate's macro average (or its NLL sidecar).
    let gate = report.gate.as_ref().unwrap();
    assert!(
        gate.fields.iter().all(|f| f.family != "arith"),
        "fit-family fields must be excluded from the gate: {:?}",
        gate.fields.iter().map(|f| &f.field).collect::<Vec<_>>()
    );
    assert_eq!(gate.fields.len(), report.nll_by_field.len());
}

#[tokio::test]
async fn nll_by_field_matches_the_analytic_values() {
    let items = vec![noul_item()];
    // Uniform [0.5, 0.5] over a 2-label field: NLL = -ln(0.5).
    let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    // Item runs key fields by family × kind × cardinality.
    let (key, nll) = report
        .nll_by_field
        .iter()
        .find(|(key, _)| key.starts_with("validation/"))
        .unwrap();
    assert_eq!(key, "validation/noul:2");
    assert!(
        (nll - std::f64::consts::LN_2).abs() < 1e-6,
        "uniform 2-label NLL must be ln 2, got {nll}"
    );
    // One-hot truth at the true label: NLL ~ 0.
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    let (_, nll) = report
        .nll_by_field
        .iter()
        .find(|(key, _)| key.starts_with("validation/"))
        .unwrap();
    assert!(nll.abs() < 1e-6, "truth NLL must be ~0, got {nll}");
}
