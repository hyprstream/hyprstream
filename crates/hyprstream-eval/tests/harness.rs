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
    // Truth answers rotate with the options: zero flips.
    assert!(report.flip_rate.groups > 0);
    assert_eq!(report.flip_rate.flip_rate, 0.0);
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
    // Uniform argmax is always label 0; truth is uniform-ish across labels, so
    // accuracy is far below 1 and Brier is the analytic uniform value.
    let accuracy = report.accuracy.unwrap();
    assert!(accuracy < 0.8, "uniform accuracy: {accuracy}");
    assert!(gate.macro_brier.mean > 0.3, "uniform Brier: {}", gate.macro_brier.mean);
    // Uniform argmax picks index 0 everywhere; rotations move the correct
    // label off index 0, so a label-comparing flip check sees flips.
    assert!(report.flip_rate.flip_rate > 0.5, "flip rate: {}", report.flip_rate.flip_rate);
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
}
