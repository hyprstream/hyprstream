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
        assert_eq!(bucket.accuracy, Some(1.0), "family {}", bucket.key);
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

#[tokio::test]
async fn untagged_observations_join_the_split_as_unknown() {
    // Observations with `family: None` score under the "unknown" sentinel;
    // the shift split must name that family too, or downstream
    // `is_eval_family` would reject data the gate actually measured.
    let mut item = noul_item();
    item.family = None;
    let items = vec![item];
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    let gate = report.gate.as_ref().unwrap();
    assert!(gate.fields.iter().all(|f| f.family == "unknown"));
    assert_eq!(gate.shift_split.eval_families, vec!["unknown".to_owned()]);
    // The family breakdown must reconcile with the gate: untagged rows
    // appear under the same sentinel, not vanish from by_family.
    assert!(
        report.by_family.iter().any(|bucket| bucket.key == "unknown"),
        "untagged rows must appear in by_family under `unknown`: {:?}",
        report.by_family.iter().map(|b| &b.key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn zero_probability_realized_label_scores_with_infinite_nll() {
    // A valid (producer-tolerance) one-hot distribution can put zero mass on
    // the realized label: NLL is legitimately infinite and must not abort
    // scoring of the otherwise-scoreable run.
    let items = vec![noul_item()]; // truth = 1
    let subject = TruthSubject::new("overconfident").with_truth("val_q", 0);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    let (_, nll) = report
        .nll_by_field
        .iter()
        .find(|(key, _)| key.starts_with("validation/"))
        .unwrap();
    assert!(nll.is_infinite(), "zero-probability realized label: {nll}");
    assert_eq!(report.accuracy, Some(0.0));
    assert!(report.gate.is_some(), "the gate must still assemble");
}

#[tokio::test]
async fn run_set_rejects_truth_for_undeclared_questions() {
    // A misspelled truth key would otherwise be silently ignored, scoring the
    // row as unlabeled and biasing every labeled denominator.
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![noul_item().question],
        rows: vec![hyprstream_eval::EvalRow {
            id: "row-1".into(),
            family: None,
            stratum: None,
            group: None,
            state: Entry::Null,
            truth: std::collections::BTreeMap::from([("val_q_typo".to_owned(), 1usize)]),
        }],
    };
    let subject = TruthSubject::new("t");
    let error = Harness.run_set(&set, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "undeclared truth key must be InvalidInput, got {error:?}"
    );
}

/// Counts decide calls, to prove input validation happens first.
struct CountingSubject(std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl Subject for CountingSubject {
    fn model_id(&self) -> &str {
        "counting"
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            answers.insert(
                question.id.clone(),
                QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.5 }),
            );
        }
        Ok(AnswerRow { answers })
    }
}

#[tokio::test]
async fn run_set_validates_the_schema_before_calling_the_subject() {
    // Duplicate question ids make DecisionSchema::new fail deterministically;
    // with an HTTP-backed model every row is a billed call, so the failure
    // must happen before the first decide.
    let question = noul_item().question;
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![question.clone(), question],
        rows: vec![hyprstream_eval::EvalRow {
            id: "row-1".into(),
            family: None,
            stratum: None,
            group: None,
            state: Entry::Null,
            truth: std::collections::BTreeMap::new(),
        }],
    };
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let result = Harness.run_set(&set, &subject).await;
    assert!(result.is_err(), "duplicate question ids must fail");
    assert_eq!(
        subject.0.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the subject must not be called for an invalid schema"
    );
}

#[tokio::test]
async fn run_set_rejects_out_of_cardinality_truth_before_calling_the_subject() {
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![noul_item().question],
        rows: vec![hyprstream_eval::EvalRow {
            id: "row-1".into(),
            family: None,
            stratum: None,
            group: None,
            state: Entry::Null,
            // noul cardinality is 2; index 2 is out of range.
            truth: std::collections::BTreeMap::from([("val_q".to_owned(), 2usize)]),
        }],
    };
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_set(&set, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "out-of-cardinality truth must be InvalidInput, got {error:?}"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn run_items_rejects_out_of_cardinality_truth_before_calling_the_subject() {
    let mut item = noul_item();
    item.truth = Some(2); // noul cardinality is 2
    let items = vec![item];
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_items(&items, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "out-of-cardinality truth must be InvalidInput, got {error:?}"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn run_items_preflights_every_truth_before_the_first_subject_call() {
    // A bad truth on item N must fail before item 1 is ever sent to the
    // (possibly billed) subject — not after N−1 calls.
    let mut bad = noul_item();
    bad.id = "val-2".into();
    bad.truth = Some(2); // noul cardinality is 2
    let items = vec![noul_item(), bad];
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_items(&items, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "out-of-cardinality truth must be InvalidInput, got {error:?}"
    );
    assert_eq!(
        subject.0.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "no item may reach the subject when a later item's truth is invalid"
    );
}

#[tokio::test]
async fn run_items_matches_answers_against_the_declared_kind() {
    // Programmatically inconsistent question: declared kind Noul, body
    // Choice. The jev-1 profile preflight rejects the question itself as
    // InvalidInput before the subject is ever called (and answers can
    // therefore never be scored under the wrong kind).
    let mut item = noul_item();
    item.question.body = hyprstream_decision::QuestionBody::Choice {
        options: vec![
            hyprstream_decision::spec::ChoiceOption {
                name: "a".into(),
                rubric: None,
            },
            hyprstream_decision::spec::ChoiceOption {
                name: "b".into(),
                rubric: None,
            },
        ],
    };
    let items = vec![item];
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_items(&items, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "kind/body disagreement must be InvalidInput, got {error:?}"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn run_items_rejects_a_profile_violating_question_before_the_subject() {
    // A one-option choice passes DecisionSchema's id checks but violates the
    // jev-1 choice cardinality rule (2–255); a jev-1 endpoint would reject
    // it, so the harness must fail before the first billed call.
    let mut item = noul_item();
    item.question.kind = hyprstream_decision::QuestionKind::Choice;
    item.question.body = hyprstream_decision::QuestionBody::Choice {
        options: vec![hyprstream_decision::spec::ChoiceOption {
            name: "only".into(),
            rubric: None,
        }],
    };
    item.truth = None;
    let items = vec![item];
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_items(&items, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "one-option choice must be InvalidInput, got {error:?}"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

/// Abstains on every question.
struct AbstainSubject;

#[async_trait::async_trait]
impl Subject for AbstainSubject {
    fn model_id(&self) -> &str {
        "abstain-1"
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            answers.insert(question.id.clone(), QuestionAnswer::abstained());
        }
        Ok(AnswerRow { answers })
    }
}

#[tokio::test]
async fn unmeasured_breakdown_metrics_are_absent_not_zero() {
    // A family whose rows are all abstained has no scored rows: macro-ECE
    // and accuracy must be absent, never a fake-perfect 0.0.
    let items: Vec<EvalItem> = small_items().iter().take(6).map(EvalItem::from).collect();
    let output = Harness.run_items(&items, &AbstainSubject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    assert!(report.gate.is_none(), "no scored rows, no gate");
    assert!(!report.by_family.is_empty());
    for bucket in &report.by_family {
        assert_eq!(bucket.macro_ece, None, "family {}", bucket.key);
        assert_eq!(bucket.accuracy, None, "family {}", bucket.key);
        assert_eq!(bucket.abstention_rate, 1.0, "family {}", bucket.key);
    }
    for bucket in &report.by_stratum {
        assert_eq!(bucket.macro_ece, None, "stratum {}", bucket.key);
        assert_eq!(bucket.accuracy, None, "stratum {}", bucket.key);
    }
}
