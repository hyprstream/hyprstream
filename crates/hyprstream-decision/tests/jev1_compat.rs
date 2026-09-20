//! jev-1 compatibility fixtures: the documented numeric examples from the S6a compat-spec
//! evidence (choice/score primitive pages, corroborated against the MIT adapter's
//! `confidence_metrics.py` semantics). These pin the normative confidence-from-peakedness
//! formulas, the expected-score derivation, the D5 sum tolerances, and the D6 tie-break.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_decision::confidence::{
    self, argmax_index, check_distribution, choice_confidence, expected_score, score_confidence,
    CONSUMER_SUM_TOLERANCE, PRODUCER_SUM_TOLERANCE,
};
use hyprstream_decision::{parse_yaml, DistributionError, QuestionBody, QuestionKind};

const EPSILON: f32 = 1e-6;

/// S6a §2.3 corroboration table (choice): distribution → normative confidence. The docs
/// display values are rounded; the exact formula values asserted here are S6a's
/// recomputation from the adapter source.
#[test]
fn choice_confidence_matches_documented_examples() {
    // primitives/choice "requested_resolution" — docs 0.16, formula exact.
    assert!((choice_confidence(&[0.1, 0.37, 0.24, 0.29]) - 0.16).abs() < EPSILON);
    // primitives/choice "tone" — docs 0.88, formula exact.
    assert!((choice_confidence(&[0.08, 0.92, 0.0]) - 0.88).abs() < EPSILON);
    // primitives/choice "shipping_issue" — formula 0.5375 (docs display 0.53).
    assert!((choice_confidence(&[0.63, 0.37, 0.0, 0.0, 0.0]) - 0.5375).abs() < EPSILON);
    // primitives/choice "department" — formula 0.400 (docs display 0.39).
    assert!((choice_confidence(&[0.02, 0.38, 0.6]) - 0.4).abs() < EPSILON);
    // Degenerate single option is certain by construction.
    assert_eq!(choice_confidence(&[1.0]), 1.0);
}

/// S6a §2.3 corroboration table (score): distribution → normative confidence.
#[test]
fn score_confidence_matches_documented_examples() {
    // primitives/score — formula 0.55 (docs display 0.54).
    assert!((score_confidence(&[0.0, 0.7, 0.3]) - 0.55).abs() < EPSILON);
    // primitives/score — formula 0.325 (docs display 0.33).
    assert!((score_confidence(&[0.0, 0.55, 0.45]) - 0.325).abs() < EPSILON);
    // primitives/score — docs 0.91, formula exact.
    assert!((score_confidence(&[0.0, 0.94, 0.06]) - 0.91).abs() < EPSILON);
    // primitives/score — concentrated at the last level: exactly 1.0.
    assert_eq!(score_confidence(&[0.0, 0.0, 0.0, 1.0]), 1.0);
    // Degenerate single level is certain by construction.
    assert_eq!(score_confidence(&[1.0]), 1.0);
}

/// S6a §2.2: the score value formula is explicit — `Σ i·pᵢ`, worked example
/// `0×0.0 + 1×0.70 + 2×0.30 = 1.30`.
#[test]
fn expected_score_matches_documented_worked_example() {
    assert!((expected_score(&[0.0, 0.7, 0.3]) - 1.3).abs() < EPSILON);
    assert!((expected_score(&[1.0, 0.0, 0.0]) - 0.0).abs() < EPSILON);
}

/// The uniform distribution has zero confidence by both formulas.
#[test]
fn uniform_distributions_have_zero_confidence() {
    assert!((choice_confidence(&[0.25; 4]) - 0.0).abs() < EPSILON);
    // Uniform score distribution: MAD_mode = MAD_uniform.
    assert!((score_confidence(&[0.25; 4]) - 0.0).abs() < EPSILON);
}

/// The generic dispatcher agrees with the per-primitive formulas; noul reuses the choice
/// formula over its 2-label distribution (superset convenience — upstream noul carries no
/// confidence field by design).
#[test]
fn confidence_dispatches_per_kind() {
    let p = [0.2, 0.8];
    assert_eq!(
        confidence::confidence(QuestionKind::Choice, &p),
        choice_confidence(&p)
    );
    assert_eq!(
        confidence::confidence(QuestionKind::Noul, &p),
        choice_confidence(&p)
    );
    assert_eq!(
        confidence::confidence(QuestionKind::Score, &p),
        score_confidence(&p)
    );
    assert_eq!(confidence::confidence(QuestionKind::Noul, &[0.5, 0.5]), 0.0);
}

/// D5: consumers accept |Σ−1| ≤ 1e-2; producers target ≤ 1e-6.
#[test]
fn distribution_tolerances_are_asymmetric() {
    let loose = [0.5, 0.505];
    assert!(check_distribution(&loose, CONSUMER_SUM_TOLERANCE).is_ok());
    assert!(matches!(
        check_distribution(&loose, PRODUCER_SUM_TOLERANCE),
        Err(DistributionError::SumOutOfTolerance { .. })
    ));

    assert!(check_distribution(&[0.1, 0.37, 0.24, 0.29], PRODUCER_SUM_TOLERANCE).is_ok());
    assert!(matches!(
        check_distribution(&[], CONSUMER_SUM_TOLERANCE),
        Err(DistributionError::Empty)
    ));
    assert!(matches!(
        check_distribution(&[0.5, f32::NAN], CONSUMER_SUM_TOLERANCE),
        Err(DistributionError::OutOfRange { .. })
    ));
    assert!(matches!(
        check_distribution(&[1.5, -0.5], CONSUMER_SUM_TOLERANCE),
        Err(DistributionError::OutOfRange { .. })
    ));
}

/// A9/D6: argmax ties break toward the earliest option in insertion order.
#[test]
fn argmax_ties_break_toward_earliest_option() {
    assert_eq!(argmax_index(&[0.5, 0.5]), Some(0));
    assert_eq!(argmax_index(&[0.1, 0.3, 0.3]), Some(1));
}

/// A jev-1-shaped request document round-trips through authoring with the profile's
/// pinned semantics (mixed primitives, null criteria entries, null instructions).
#[test]
fn jev1_shaped_document_round_trips() {
    let set = parse_yaml(
        r#"
state: "Customer: my order #4521 arrived with a cracked screen. I want my money back."
questions:
  requested_resolution:
    type: choice
    instructions: "What resolution did the customer ask for?"
    criteria:
      refund: "Full or partial refund."
      replacement: "A new unit."
      repair: "A repair of the existing unit."
      other: ~
  severity:
    type: score
    criteria: ["Cosmetic", "Functional damage", "Item unusable"]
  mentions_order_number:
    type: noul
    instructions: "The message mentions an order number."
"#,
    )
    .expect("jev-1-shaped document parses");

    let choice = set.question("requested_resolution").expect("choice");
    assert_eq!(choice.kind, QuestionKind::Choice);
    assert_eq!(choice.cardinality(), 4);
    assert_eq!(
        choice.labels(),
        vec!["refund", "replacement", "repair", "other"]
    );
    match &choice.body {
        QuestionBody::Choice { options } => {
            assert_eq!(options[3].rubric, None, "null rubric = undescribed option");
        }
        other => panic!("expected choice body, got {other:?}"),
    }

    let score = set.question("severity").expect("score");
    assert_eq!(score.labels(), vec!["0", "1", "2"]);

    let noul = set.question("mentions_order_number").expect("noul");
    assert_eq!(noul.cardinality(), 2);
    assert_eq!(noul.labels(), vec!["false", "true"]);
}
