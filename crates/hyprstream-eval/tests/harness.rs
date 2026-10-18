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
use hyprstream_eval::{score_bench_run, score_run, EvalItem, Harness, Subject, TruthSubject};

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
                hyprstream_decision::QuestionKind::Choice => AnswerValue::Choice { probabilities },
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
    assert_eq!(report.abstention_rate, Some(0.0));
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
    assert!(
        gate.macro_brier.mean > 0.3,
        "uniform Brier: {}",
        gate.macro_brier.mean
    );
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
    assert_eq!(
        obs.argmax_label_index(),
        Some(0),
        "flip-label path: D6 earliest"
    );

    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    assert_eq!(
        report.items[0].correct,
        Some(false),
        "D6: argmax = index 0 on a tie"
    );
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
    let error = Harness
        .run_items(&items, &SilentSubject)
        .await
        .err()
        .unwrap();
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
async fn untagged_observations_have_a_distinct_split_identity() {
    // Observations with `family: None` score under the "~untagged" sentinel;
    // the shift split must name that family too, or downstream
    // `is_eval_family` would reject data the gate actually measured.
    let mut item = noul_item();
    item.family = None;
    let items = vec![item];
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    let gate = report.gate.as_ref().unwrap();
    assert!(gate.fields.iter().all(|f| f.family == "~untagged"));
    assert_eq!(gate.shift_split.eval_families, vec!["~untagged".to_owned()]);
    // The family breakdown must reconcile with the gate: untagged rows
    // appear under the same sentinel, not vanish from by_family.
    assert!(
        report
            .by_family
            .iter()
            .any(|bucket| bucket.key == "~untagged"),
        "untagged rows must appear in by_family under `~untagged`: {:?}",
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

#[tokio::test]
async fn run_items_rejects_an_empty_question_id_before_the_subject() {
    // The authoring layer requires nonempty ids; a programmatically built
    // empty id must fail the shared preflight, not reach a subject (an
    // HTTP subject would get a wire-level rejection instead — the
    // preflight must be consistent across subject implementations).
    let mut item = noul_item();
    item.question.id = String::new();
    let items = vec![item];
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_items(&items, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "empty question id must be InvalidInput, got {error:?}"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

/// Resolves to `v1` on the first decide and `v2` on every later one — a
/// rolling deploy landing mid-run.
struct VersionFlipSubject(std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl Subject for VersionFlipSubject {
    fn model_id(&self) -> &str {
        "version-flip"
    }

    fn resolved_model_id(&self) -> String {
        if self.0.load(std::sync::atomic::Ordering::SeqCst) <= 1 {
            "v1".to_owned()
        } else {
            "v2".to_owned()
        }
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
async fn run_items_errors_when_the_model_version_changes_mid_run() {
    // The version triple stamps ONE resolved id on every row; a subject that
    // re-resolves its alias to a new version mid-run (rolling deploy) must
    // fail loudly instead of mislabeling the earlier rows.
    let mut second = noul_item();
    second.id = "val-2".into();
    let items = vec![noul_item(), second];
    let error = Harness
        .run_items(
            &items,
            &VersionFlipSubject(std::sync::atomic::AtomicUsize::new(0)),
        )
        .await
        .err()
        .unwrap();
    match error {
        hyprstream_eval::EvalError::InvalidInput(message) => {
            assert!(
                message.contains("v1") && message.contains("v2"),
                "the error must name both versions, got: {message}"
            );
        }
        other => panic!("mid-run version change must be InvalidInput, got {other:?}"),
    }
}

#[tokio::test]
async fn run_set_errors_when_the_model_version_changes_mid_run() {
    let question = noul_item().question;
    let row = |id: &str| hyprstream_eval::EvalRow {
        id: id.into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Null,
        truth: std::collections::BTreeMap::new(),
    };
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![question],
        rows: vec![row("r1"), row("r2")],
    };
    let error = Harness
        .run_set(
            &set,
            &VersionFlipSubject(std::sync::atomic::AtomicUsize::new(0)),
        )
        .await
        .err()
        .unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidInput(_)),
        "mid-run version change must be InvalidInput, got {error:?}"
    );
}

#[test]
fn flip_rate_compares_each_question_against_its_own_base() {
    // A multi-question EvalSet answers EVERY question on every row, and rows
    // of one permutation group share the group id — so the base row (id ==
    // group id) contributes one base observation PER QUESTION. Each rotated
    // row's observation must be compared against its own question's base:
    // comparing q2's label against q1's base label (the old behavior) reports
    // a spurious flip rate. Here both questions are individually label-stable
    // across the rotation, so the flip rate is exactly 0.
    use hyprstream_eval::{flip_rate_with_labels, Observation};
    let obs = |id: &str, question_id: &str, probabilities: Vec<f32>| Observation {
        id: id.into(),
        question_id: question_id.into(),
        kind: hyprstream_decision::QuestionKind::Choice,
        family: None,
        stratum: None,
        group: Some("g".into()),
        probabilities: Some(probabilities),
        truth: None,
    };
    let observations = vec![
        // Base row: q1 argmax "a" (index 0), q2 argmax "y" (index 1).
        obs("g", "q1", vec![0.8, 0.1, 0.1]),
        obs("g", "q2", vec![0.1, 0.8, 0.1]),
        // Rotated row, probabilities in canonical label order: both
        // questions answer the same label as on the base row.
        obs("g-p1", "q1", vec![0.8, 0.1, 0.1]),
        obs("g-p1", "q2", vec![0.1, 0.8, 0.1]),
    ];
    let labels_of = |question_id: &str| -> Option<Vec<String>> {
        match question_id {
            "q1" => Some(vec!["a".into(), "b".into(), "c".into()]),
            "q2" => Some(vec!["x".into(), "y".into(), "z".into()]),
            _ => None,
        }
    };
    let flip = flip_rate_with_labels(&observations, labels_of);
    assert_eq!(flip.groups, 1);
    assert_eq!(flip.flip_rate, 0.0);
}

/// Answers row 0 correctly and omits the answer on every later row.
struct MalformedOnSecondRowSubject(std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl Subject for MalformedOnSecondRowSubject {
    fn model_id(&self) -> &str {
        "malformed-second"
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut answers = std::collections::BTreeMap::new();
        if call == 0 {
            for question in &set.questions {
                answers.insert(
                    question.id.clone(),
                    QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.5 }),
                );
            }
        }
        Ok(AnswerRow { answers })
    }
}

#[tokio::test]
async fn run_set_validates_each_row_before_requesting_the_next() {
    // A malformed answer on row 1 is a deterministic failure: with an
    // HTTP-backed subject every later row is a billed call, so the run must
    // stop there instead of validating only at build_batch after the loop.
    let question = noul_item().question;
    let row = |id: &str| hyprstream_eval::EvalRow {
        id: id.into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Null,
        truth: std::collections::BTreeMap::new(),
    };
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![question],
        rows: vec![row("r1"), row("r2"), row("r3")],
    };
    let subject = MalformedOnSecondRowSubject(std::sync::atomic::AtomicUsize::new(0));
    let error = Harness.run_set(&set, &subject).await.err().unwrap();
    assert!(
        matches!(error, hyprstream_eval::EvalError::InvalidAnswer { .. }),
        "malformed row must be InvalidAnswer, got {error:?}"
    );
    assert_eq!(
        subject.0.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "row 3 must not be requested after row 2 failed validation"
    );
}

/// Reports a stale global version but binds a fresh one to each response —
/// the shape an HTTP subject presents when a concurrent run overwrites the
/// shared resolved slot between a decide and a later resolved_model_id read.
struct FreshPerResponseSubject;

#[async_trait::async_trait]
impl Subject for FreshPerResponseSubject {
    fn model_id(&self) -> &str {
        "alias"
    }

    fn resolved_model_id(&self) -> String {
        "stale-global".to_owned()
    }

    async fn decide(
        &self,
        set: &QuestionSet,
        _state: &Entry,
        _row: usize,
    ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
        let mut answers = std::collections::BTreeMap::new();
        for question in &set.questions {
            answers.insert(
                question.id.clone(),
                QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.5 }),
            );
        }
        Ok(AnswerRow { answers })
    }

    async fn decide_with_version(
        &self,
        set: &QuestionSet,
        state: &Entry,
        row: usize,
    ) -> Result<(AnswerRow, String), hyprstream_eval::EvalError> {
        Ok((self.decide(set, state, row).await?, "fresh-v42".to_owned()))
    }
}

#[tokio::test]
async fn teacher_answers_use_the_version_bound_to_each_response() {
    // Teacher provenance is persisted raw: it must record the version from
    // the same exchange as the answers, never a later global read.
    let ensemble = hyprstream_eval::TeacherEnsemble::new(vec![hyprstream_eval::Teacher {
        id: "t1".into(),
        tos_class: hyprstream_eval::TosClass::InternalOnly,
        subject: Box::new(FreshPerResponseSubject),
    }])
    .unwrap();
    let set = QuestionSet {
        state: None,
        questions: vec![noul_item().question],
    };
    let output = ensemble
        .decide(&set, &Entry::Null, 0, "item-1")
        .await
        .unwrap();
    assert_eq!(output.teacher_answers.len(), 1);
    assert_eq!(output.teacher_answers[0].model_id, "fresh-v42");
}

#[test]
fn flip_rate_skips_members_whose_question_the_base_row_abstained_on() {
    // The base row answered q1 but abstained on q2 (abstentions never enter
    // a group), so q2 has NO base prediction: an answered rotated q2 member
    // must be skipped, not compared against q1's base (the old single-base
    // fallback recorded a spurious flip there). q1 is label-stable, so the
    // flip rate is exactly 0.
    use hyprstream_eval::{flip_rate_with_labels, Observation};
    let obs = |id: &str, question_id: &str, probabilities: Vec<f32>| Observation {
        id: id.into(),
        question_id: question_id.into(),
        kind: hyprstream_decision::QuestionKind::Choice,
        family: None,
        stratum: None,
        group: Some("g".into()),
        probabilities: Some(probabilities),
        truth: None,
    };
    let observations = vec![
        obs("g", "q1", vec![0.8, 0.1, 0.1]),
        // (no base observation for q2 — the base row abstained on it)
        obs("g-p1", "q1", vec![0.8, 0.1, 0.1]),
        obs("g-p1", "q2", vec![0.1, 0.8, 0.1]),
    ];
    let labels_of = |question_id: &str| -> Option<Vec<String>> {
        match question_id {
            "q1" => Some(vec!["a".into(), "b".into(), "c".into()]),
            "q2" => Some(vec!["x".into(), "y".into(), "z".into()]),
            _ => None,
        }
    };
    let flip = flip_rate_with_labels(&observations, labels_of);
    assert_eq!(flip.groups, 1);
    assert_eq!(flip.flip_rate, 0.0);
}

#[tokio::test]
async fn flip_rate_is_none_when_no_group_contributes() {
    // No permutation groups: the robustness result is UNMEASURED and must
    // stay None — a `Some(0.0)` would read as computed permutation-robustness.
    let mut items: Vec<EvalItem> = small_items().iter().take(5).map(EvalItem::from).collect();
    for item in &mut items {
        item.group = None;
    }
    let subject = truth_subject_for(&items);
    let output = Harness.run_items(&items, &subject).await.unwrap();
    let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    assert!(
        report.flip_rate.is_none(),
        "no permutation groups: unmeasured, never a fake 0.0"
    );
}

#[tokio::test]
async fn truth_subject_anchors_per_row_truths_in_set_runs() {
    // A set run applies the same question to rows with different truths;
    // the question-level truth map alone would emit the same one-hot for
    // every row (scoring at chance). with_row_truth keys the anchor by
    // (row, question), restoring the perfectly-accurate anchor.
    let question = noul_item().question; // id "val_q", cardinality 2
    let row = |id: &str, truth: usize| hyprstream_eval::EvalRow {
        id: id.into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Null,
        truth: std::collections::BTreeMap::from([("val_q".to_owned(), truth)]),
    };
    let set = hyprstream_eval::EvalSet {
        name: "t".into(),
        schema_version: "v1".into(),
        questions: vec![question],
        rows: vec![row("r1", 0), row("r2", 1)],
    };
    let subject = TruthSubject::new("truth-rows")
        .with_row_truth(0, "val_q", 0)
        .with_row_truth(1, "val_q", 1);
    let output = Harness.run_set(&set, &subject).await.unwrap();
    let report = score_run(&output, &ScoreConfig::default()).unwrap();
    assert_eq!(
        report.accuracy,
        Some(1.0),
        "row-keyed truth must anchor every row perfectly"
    );
}

#[test]
fn flip_rate_never_compares_across_questions_in_partial_set_groups() {
    // Partial set response: the base row answered q1 only, the rotated row
    // answered q2 only. Among ANSWERED observations the question ids are
    // unique — classifying the shape from answered-only observations (the
    // old behavior) misreads this as a bench group and compares the q2
    // member against the unrelated q1 base (a spurious flip). With the
    // abstentions included the group is set-shaped, and the q2 member has
    // no same-question base: nothing is measured.
    use hyprstream_eval::{flip_rate_with_labels, Observation};
    let obs = |id: &str, question_id: &str, probabilities: Option<Vec<f32>>| Observation {
        id: id.into(),
        question_id: question_id.into(),
        kind: hyprstream_decision::QuestionKind::Choice,
        family: None,
        stratum: None,
        group: Some("g".into()),
        probabilities,
        truth: None,
    };
    let observations = vec![
        obs("g", "q1", Some(vec![0.8, 0.1, 0.1])),
        obs("g", "q2", None),    // base row abstained on q2
        obs("g-p1", "q1", None), // rotated row abstained on q1
        obs("g-p1", "q2", Some(vec![0.1, 0.8, 0.1])),
    ];
    let labels_of = |question_id: &str| -> Option<Vec<String>> {
        match question_id {
            "q1" => Some(vec!["a".into(), "b".into(), "c".into()]),
            "q2" => Some(vec!["x".into(), "y".into(), "z".into()]),
            _ => None,
        }
    };
    let flip = flip_rate_with_labels(&observations, labels_of);
    assert_eq!(flip.groups, 0, "no same-question base/member pair exists");
    assert_eq!(flip.flip_rate, 0.0);
}

#[tokio::test]
async fn run_items_rejects_an_empty_resolved_model_id() {
    // Any subject (in-process ones included) can resolve to an empty id;
    // pinning it would produce a run with no model provenance that only
    // fails later at persistence — reject it at capture instead.
    let items = vec![noul_item()];
    let error = Harness
        .run_items(&items, &TruthSubject::new(""))
        .await
        .err()
        .unwrap();
    match error {
        hyprstream_eval::EvalError::InvalidInput(message) => {
            assert!(
                message.contains("empty model id"),
                "the error must name the problem: {message}"
            );
        }
        other => panic!("an empty resolved id must be InvalidInput, got {other:?}"),
    }
}

#[tokio::test]
async fn agreement_summary_is_unmeasured_without_samples() {
    // No ensemble outputs (every candidate filtered/failed): the mean is
    // absent — a 0.0 would read as measured total teacher disagreement.
    let empty = hyprstream_eval::agreement_summary(&[]);
    assert_eq!(empty.mean_argmax_agreement, None);
    assert!(empty.per_item.is_empty());
    // …and Some once any item produced an ensemble result.
    let ensemble = hyprstream_eval::TeacherEnsemble::new(vec![hyprstream_eval::Teacher {
        id: "t1".into(),
        tos_class: hyprstream_eval::TosClass::InternalOnly,
        subject: Box::new(FreshPerResponseSubject),
    }])
    .unwrap();
    let set = QuestionSet {
        state: None,
        questions: vec![noul_item().question],
    };
    let output = ensemble
        .decide(&set, &Entry::Null, 0, "item-1")
        .await
        .unwrap();
    let summary = hyprstream_eval::agreement_summary(&[("item-1".to_owned(), output)]);
    assert_eq!(summary.mean_argmax_agreement, Some(1.0));
}

#[tokio::test]
async fn empty_runs_validate_fallback_provenance_without_queries() {
    let set = hyprstream_eval::EvalSet {
        name: "empty".into(),
        schema_version: "v1".into(),
        questions: vec![noul_item().question],
        rows: vec![],
    };
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    assert_eq!(
        Harness.run_set(&set, &subject).await.unwrap().model_id,
        "counting"
    );
    assert_eq!(
        Harness.run_items(&[], &subject).await.unwrap().model_id,
        "counting"
    );
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(matches!(
        Harness.run_set(&set, &TruthSubject::new("")).await,
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    assert!(matches!(
        Harness.run_items(&[], &TruthSubject::new("")).await,
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
}

#[tokio::test]
async fn flip_rate_resolves_labels_for_each_item_with_a_shared_question_id() {
    for reversed in [false, true] {
        let question = |names: [&str; 2]| hyprstream_decision::QuestionSpec {
            id: "shared".into(),
            kind: hyprstream_decision::QuestionKind::Choice,
            instructions: None,
            body: hyprstream_decision::QuestionBody::Choice {
                options: names
                    .into_iter()
                    .map(|name| hyprstream_decision::ChoiceOption {
                        name: name.into(),
                        rubric: None,
                    })
                    .collect(),
            },
        };
        let item = |id: &str, names| EvalItem {
            id: id.into(),
            question: question(names),
            state: Entry::Null,
            family: None,
            stratum: None,
            group: Some("g".into()),
            truth: None,
        };
        let items = vec![
            item("g", ["A", "B"]),
            item("g-p1", if reversed { ["B", "A"] } else { ["A", "B"] }),
        ];
        let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
        let report = score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
        let flip = report.flip_rate.unwrap();
        assert_eq!(flip.groups, 1);
        assert_eq!(flip.flip_rate, if reversed { 1.0 } else { 0.0 });
    }
}

#[tokio::test]
async fn optional_family_identity_is_injective_and_fit_matching_uses_raw_tags() {
    let families = [None, Some("unknown"), Some("~untagged"), Some("~~tag")];
    let items: Vec<_> = families
        .into_iter()
        .enumerate()
        .map(|(i, family)| {
            let mut item = noul_item();
            item.id = format!("family-{i}");
            item.family = family.map(str::to_owned);
            item
        })
        .collect();
    let set = hyprstream_eval::EvalSet {
        name: "families".into(),
        schema_version: "v1".into(),
        questions: vec![items[0].question.clone()],
        rows: items
            .iter()
            .map(|item| hyprstream_eval::EvalRow {
                id: item.id.clone(),
                family: item.family.clone(),
                stratum: None,
                group: None,
                state: item.state.clone(),
                truth: std::collections::BTreeMap::from([("val_q".into(), 1)]),
            })
            .collect(),
    };
    let outputs = [
        Harness.run_items(&items, &UniformSubject).await.unwrap(),
        Harness.run_set(&set, &UniformSubject).await.unwrap(),
    ];
    for output in outputs {
        let report = score_run(&output, &ScoreConfig::default()).unwrap();
        let gate = report.gate.unwrap();
        let identities: std::collections::BTreeSet<_> =
            gate.fields.iter().map(|f| f.family.as_str()).collect();
        assert_eq!(
            identities,
            std::collections::BTreeSet::from(["~untagged", "unknown", "~~untagged", "~~~tag"])
        );
        assert_eq!(
            gate.fields.len(),
            4,
            "optional families must never pool samples"
        );
        assert_eq!(report.nll_by_field.len(), 4);
        assert_eq!(report.by_family.len(), 4);
        assert_eq!(
            gate.shift_split
                .eval_families
                .iter()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            identities
        );
        let report = score_run(
            &output,
            &ScoreConfig::default().with_fit_families(["unknown", "~untagged"]),
        )
        .unwrap();
        let gate = report.gate.unwrap();
        let identities: std::collections::BTreeSet<_> =
            gate.fields.iter().map(|f| f.family.as_str()).collect();
        assert_eq!(
            identities,
            std::collections::BTreeSet::from(["~untagged", "~~~tag"])
        );
        assert_eq!(gate.shift_split.fit_families, vec!["unknown", "~~untagged"]);
        assert_eq!(
            gate.shift_split
                .eval_families
                .iter()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            identities
        );
        assert_eq!(report.nll_by_field.len(), 2);
    }
}

#[tokio::test]
async fn duplicate_observation_identities_fail_before_subject_calls() {
    let item = noul_item();
    let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
    assert!(matches!(
        Harness
            .run_items(&[item.clone(), item.clone()], &subject)
            .await,
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    // The exported identity is (item id, question id): same row ID with
    // distinct questions remains valid, as it is in a multi-question set.
    let mut other = item.clone();
    other.question.id = "other".into();
    let valid = vec![item.clone(), other];
    let output = Harness.run_items(&valid, &subject).await.unwrap();
    score_bench_run(&output, &valid, &ScoreConfig::default()).unwrap();
    assert!(matches!(
        score_bench_run(&output, &[item.clone(), item], &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    let row = hyprstream_eval::EvalRow {
        id: "same-row".into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Null,
        truth: std::collections::BTreeMap::new(),
    };
    let set = hyprstream_eval::EvalSet {
        name: "duplicates".into(),
        schema_version: "v1".into(),
        questions: vec![noul_item().question],
        rows: vec![row.clone(), row],
    };
    assert!(matches!(
        Harness.run_set(&set, &subject).await,
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn nonfinite_request_entries_fail_before_any_subject_or_teacher_call() {
    use hyprstream_decision::spec::{ChoiceOption, NoulCriteria, QuestionBody, QuestionKind};
    use hyprstream_eval::teacher::{Teacher, TeacherEnsemble, TosClass};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountValid(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl Subject for CountValid {
        fn model_id(&self) -> &str {
            "finite-input-control"
        }
        async fn decide(
            &self,
            set: &QuestionSet,
            state: &Entry,
            row: usize,
        ) -> Result<AnswerRow, hyprstream_eval::EvalError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            UniformSubject.decide(set, state, row).await
        }
    }

    // Each request-bearing Entry position is nested through both Map and Seq.
    // Invalid item/row is second: no earlier request may escape preflight.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.25] {
        for position in 0..6 {
            let entry = Entry::Map(vec![(
                "nested".into(),
                Entry::Seq(vec![Entry::Number(value)]),
            )]);
            let mut item = noul_item();
            item.id = "second".into();
            match position {
                0 => item.state = entry,
                1 => item.question.instructions = Some(entry),
                2 | 3 => {
                    item.question.body = QuestionBody::Noul {
                        criteria: Some(NoulCriteria {
                            on_true: (position == 2).then(|| entry.clone()),
                            on_false: (position == 3).then_some(entry),
                        }),
                    }
                }
                4 => {
                    item.question.kind = QuestionKind::Choice;
                    item.question.body = QuestionBody::Choice {
                        options: vec![
                            ChoiceOption {
                                name: "a".into(),
                                rubric: None,
                            },
                            ChoiceOption {
                                name: "b".into(),
                                rubric: Some(entry),
                            },
                        ],
                    };
                }
                _ => {
                    item.question.kind = QuestionKind::Score;
                    item.question.body = QuestionBody::Score {
                        levels: vec![None, Some(entry)],
                    };
                }
            }
            let calls = Arc::new(AtomicUsize::new(0));
            let subject = CountValid(calls.clone());
            let items = vec![noul_item(), item.clone()];
            let result = Harness.run_items(&items, &subject).await;
            if value.is_finite() {
                assert_eq!(result.unwrap().observations.len(), 2);
                assert_eq!(calls.load(Ordering::SeqCst), 2);
            } else {
                assert!(matches!(
                    result,
                    Err(hyprstream_eval::EvalError::InvalidInput(_))
                ));
                assert_eq!(calls.load(Ordering::SeqCst), 0, "item position {position}");
            }
            calls.store(0, Ordering::SeqCst);
            let row = hyprstream_eval::EvalRow {
                id: "first".into(),
                family: None,
                stratum: None,
                group: None,
                state: Entry::Null,
                truth: std::collections::BTreeMap::new(),
            };
            let set = hyprstream_eval::EvalSet {
                name: "finite-input".into(),
                schema_version: "v1".into(),
                questions: vec![item.question.clone()],
                rows: vec![
                    row.clone(),
                    hyprstream_eval::EvalRow {
                        id: "second".into(),
                        state: item.state.clone(),
                        ..row
                    },
                ],
            };
            let result = Harness.run_set(&set, &subject).await;
            if value.is_finite() {
                assert_eq!(result.unwrap().observations.len(), 2);
                assert_eq!(calls.load(Ordering::SeqCst), 2);
            } else {
                assert!(matches!(
                    result,
                    Err(hyprstream_eval::EvalError::InvalidInput(_))
                ));
                assert_eq!(calls.load(Ordering::SeqCst), 0, "set position {position}");
            }
            calls.store(0, Ordering::SeqCst);
            let ensemble = TeacherEnsemble::new(vec![Teacher {
                id: "teacher".into(),
                tos_class: TosClass::Unknown,
                subject: Box::new(CountValid(calls.clone())),
            }])
            .unwrap();
            let result = ensemble
                .decide(&set.question_set(), &item.state, 0, "item")
                .await;
            if value.is_finite() {
                assert!(result.is_ok());
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            } else {
                assert!(matches!(
                    result,
                    Err(hyprstream_eval::EvalError::InvalidInput(_))
                ));
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    0,
                    "teacher position {position}"
                );
            }
        }
    }
}

#[tokio::test]
async fn empty_schema_versions_fail_preflight_even_for_empty_runs() {
    let item = noul_item();
    let row = hyprstream_eval::EvalRow {
        id: "row".into(),
        family: None,
        stratum: None,
        group: None,
        state: Entry::Null,
        truth: std::collections::BTreeMap::new(),
    };
    for rows in [vec![], vec![row]] {
        let subject = CountingSubject(std::sync::atomic::AtomicUsize::new(0));
        let mut set = hyprstream_eval::EvalSet {
            name: "versioned".into(),
            schema_version: String::new(),
            questions: vec![item.question.clone()],
            rows,
        };
        assert!(matches!(
            Harness.run_set(&set, &subject).await,
            Err(hyprstream_eval::EvalError::InvalidInput(_))
        ));
        assert_eq!(subject.0.load(std::sync::atomic::Ordering::SeqCst), 0);
        set.schema_version = "schema@revision".into();
        let output = Harness.run_set(&set, &subject).await.unwrap();
        assert_eq!(output.version.schema, "schema@revision");
        assert_eq!(
            subject.0.load(std::sync::atomic::Ordering::SeqCst),
            set.rows.len()
        );
    }
}

#[tokio::test]
async fn empty_runs_have_no_measured_abstention_rate() {
    let item = noul_item();
    let set = hyprstream_eval::EvalSet {
        name: "empty-rate".into(),
        schema_version: "v1".into(),
        questions: vec![item.question.clone()],
        rows: vec![],
    };
    let subject = TruthSubject::new("empty-rate-control");
    let output = Harness.run_set(&set, &subject).await.unwrap();
    let report = score_run(&output, &ScoreConfig::default()).unwrap();
    assert_eq!(report.abstention_rate, None);
    assert_eq!(report.accuracy, None);
    assert!(report.gate.is_none());
    assert!(report.by_family.is_empty());
    let output = Harness.run_items(&[], &subject).await.unwrap();
    let report = score_bench_run(&output, &[], &ScoreConfig::default()).unwrap();
    assert_eq!(report.abstention_rate, None);
    assert_eq!(report.accuracy, None);
    assert!(report.gate.is_none());
    // A real unanswered observation is measured, even with no ground truth.
    let mut output = Harness
        .run_items(std::slice::from_ref(&item), &subject)
        .await
        .unwrap();
    output.observations[0].probabilities = None;
    let report = score_bench_run(&output, &[item], &ScoreConfig::default()).unwrap();
    assert_eq!(report.abstention_rate, Some(1.0));
    assert_eq!(report.by_family[0].abstention_rate, 1.0);
}

#[tokio::test]
async fn bench_flip_rate_compares_all_rotations_with_mixed_question_id_reuse() {
    let mut question = hyprstream_decision::author::parse_yaml(
        "questions:\n  shared: {type: choice, criteria: {A: null, B: null}}\n",
    )
    .unwrap()
    .questions
    .remove(0);
    let item = |id: &str, question| EvalItem {
        id: id.into(),
        question,
        state: Entry::Null,
        family: None,
        stratum: None,
        group: Some("g".into()),
        truth: None,
    };
    let mut items = vec![item("g", question.clone()), item("g-p1", question.clone())];
    question.id = "distinct".into();
    if let hyprstream_decision::QuestionBody::Choice { options } = &mut question.body {
        options.reverse();
    }
    items.push(item("g-p2", question));
    let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
    let flip = score_bench_run(&output, &items, &ScoreConfig::default())
        .unwrap()
        .flip_rate
        .unwrap();
    assert_eq!(flip.groups, 1);
    assert_eq!(
        flip.flip_rate, 0.5,
        "the distinct-ID rotation must not be skipped"
    );
}

#[tokio::test]
async fn bench_scoring_requires_the_exact_item_observation_identity_set() {
    let question = hyprstream_decision::author::parse_yaml("questions:\n  shared: {type: noul}\n")
        .unwrap()
        .questions
        .remove(0);
    let items: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|id| EvalItem {
            id: id.into(),
            question: question.clone(),
            state: Entry::Null,
            family: None,
            stratum: None,
            group: None,
            truth: Some(0),
        })
        .collect();
    let output = Harness.run_items(&items, &UniformSubject).await.unwrap();
    assert_eq!(
        score_bench_run(&output, &items, &ScoreConfig::default())
            .unwrap()
            .items
            .len(),
        2
    );
    let partial = Harness
        .run_items(&items[..1], &UniformSubject)
        .await
        .unwrap();
    assert!(
        matches!(score_bench_run(&partial, &items, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(ref message)) if message.contains("no matching observation"))
    );
    let mut duplicate = Harness.run_items(&items, &UniformSubject).await.unwrap();
    duplicate
        .observations
        .push(duplicate.observations[0].clone());
    assert!(
        matches!(score_bench_run(&duplicate, &items, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(ref message)) if message.contains("duplicate observation identity"))
    );
    let mut substituted = Harness.run_items(&items, &UniformSubject).await.unwrap();
    substituted.observations[1].id = "outside".into();
    assert!(
        matches!(score_bench_run(&substituted, &items, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(ref message)) if message.contains("no matching eval item"))
    );
}

#[tokio::test]
async fn item_scoring_preserves_resolved_alias_identity_and_rejects_mutated_provenance() {
    let items = vec![noul_item()];
    let mut output = Harness
        .run_items(&items, &FreshPerResponseSubject)
        .await
        .unwrap();
    assert_eq!(output.model_id, "fresh-v42");
    assert_eq!(output.version.model, "fresh-v42");
    assert_ne!(output.model_id, FreshPerResponseSubject.model_id());
    assert!(output.batch.is_none());
    score_run(&output, &ScoreConfig::default()).unwrap();
    score_bench_run(&output, &items, &ScoreConfig::default()).unwrap();
    output.model_id = "alias".into();
    assert!(matches!(
        score_run(&output, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    assert!(matches!(
        score_bench_run(&output, &items, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    output.model_id = "fresh-v42".into();
    output.version.model = "another-resolved-version".into();
    assert!(matches!(
        score_run(&output, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
    output.model_id.clear();
    output.version.model.clear();
    assert!(matches!(
        score_bench_run(&output, &items, &ScoreConfig::default()),
        Err(hyprstream_eval::EvalError::InvalidInput(_))
    ));
}
