//! Authoring validation: every profile rule produces a structured, positioned error.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_decision::{
    parse_json, parse_yaml, Entry, QuestionBody, QuestionKind, SpecErrorKind,
};

fn err_kind(doc: &str) -> hyprstream_decision::SpecError {
    parse_yaml(doc).expect_err("document must be rejected")
}

#[test]
fn full_document_parses_across_yaml_and_json() {
    let yaml = r#"
state: "The refund arrived two weeks late and the box was crushed."
questions:
  is_refund:
    type: noul
    instructions: "The customer is asking for a refund."
    criteria:
      "true": "Explicitly requests money back"
      "false": ~
  tone:
    type: choice
    instructions: "Classify the tone of the review."
    criteria:
      positive: "The review expresses approval."
      negative: "The review expresses disapproval."
      neutral: ~
  severity:
    type: score
    criteria:
      - "Cosmetic issue only"
      - "Usable but damaged"
      - "Completely unusable"
"#;
    let set = parse_yaml(yaml).expect("valid document parses");
    assert_eq!(set.questions.len(), 3);
    assert_eq!(
        set.state,
        Some(Entry::Str(
            "The refund arrived two weeks late and the box was crushed.".into()
        ))
    );

    let noul = set.question("is_refund").expect("noul present");
    assert_eq!(noul.kind, QuestionKind::Noul);
    match &noul.body {
        QuestionBody::Noul {
            criteria: Some(criteria),
        } => {
            assert_eq!(
                criteria.on_true,
                Some(Entry::Str("Explicitly requests money back".into()))
            );
            assert_eq!(criteria.on_false, None);
        }
        other => panic!("expected noul criteria, got {other:?}"),
    }

    let tone = set.question("tone").expect("choice present");
    assert_eq!(tone.labels(), vec!["positive", "negative", "neutral"]);
    assert_eq!(tone.cardinality(), 3);

    let severity = set.question("severity").expect("score present");
    assert_eq!(severity.labels(), vec!["0", "1", "2"]);

    // JSON authors the same IR.
    let json = r#"{
      "questions": {
        "tone": {
          "type": "choice",
          "instructions": "Classify the tone of the review.",
          "criteria": {
            "positive": "The review expresses approval.",
            "negative": "The review expresses disapproval.",
            "neutral": null
          }
        }
      }
    }"#;
    let json_set = parse_json(json).expect("valid JSON parses");
    assert_eq!(json_set.questions[0], *tone);
}

#[test]
fn yaml_boolean_criteria_keys_become_the_strings_true_and_false() {
    // YAML 1.1 parses unquoted `true:`/`false:` mapping keys as booleans; the noul
    // contract's keys are literally "true"/"false", so scalar keys are stringified.
    let set = parse_yaml(
        r#"
questions:
  is_refund:
    type: noul
    criteria:
      true: "Explicitly requests money back"
"#,
    )
    .expect("unquoted YAML bool keys parse");
    match &set.questions[0].body {
        QuestionBody::Noul {
            criteria: Some(criteria),
        } => {
            assert_eq!(
                criteria.on_true,
                Some(Entry::Str("Explicitly requests money back".into()))
            );
        }
        other => panic!("expected noul criteria, got {other:?}"),
    }
}

#[test]
fn reserved_v2_types_fail_loudly() {
    for tag in ["span", "derived"] {
        let error = err_kind(&format!("questions:\n  q1:\n    type: {tag}\n"));
        assert_eq!(error.kind, SpecErrorKind::ReservedQuestionType);
        assert_eq!(error.path.to_string(), "/questions/q1/type");
        assert!(error.message.contains("v2"), "message names v2: {error}");
    }
}

#[test]
fn unknown_question_type_is_distinguished_from_reserved() {
    let error = err_kind("questions:\n  q1:\n    type: choise\n");
    assert_eq!(error.kind, SpecErrorKind::UnknownQuestionType);
    assert_eq!(error.path.to_string(), "/questions/q1/type");
}

#[test]
fn choice_cardinality_rules() {
    let one = err_kind("questions:\n  q1:\n    type: choice\n    criteria:\n      only: \"x\"\n");
    assert_eq!(one.kind, SpecErrorKind::Cardinality);
    assert_eq!(one.path.to_string(), "/questions/q1/criteria");

    // 255 options is the pinned profile limit (parses)...
    let options_255: String = (0..255)
        .map(|i| format!("      opt_{i}: \"rubric {i}\"\n"))
        .collect();
    let doc = format!("questions:\n  q1:\n    type: choice\n    criteria:\n{options_255}");
    assert!(
        parse_yaml(&doc).is_ok(),
        "255 options is the limit, not an error"
    );

    // ...and 256 fails loudly, pointing at the two-stage recipe.
    let options_256: String = (0..256)
        .map(|i| format!("      opt_{i}: \"rubric {i}\"\n"))
        .collect();
    let error = err_kind(&format!(
        "questions:\n  q1:\n    type: choice\n    criteria:\n{options_256}"
    ));
    assert_eq!(error.kind, SpecErrorKind::Cardinality);
    assert!(
        error.message.contains("two-stage"),
        "message points at the recipe: {error}"
    );
}

#[test]
fn score_needs_at_least_two_levels() {
    let error = err_kind("questions:\n  q1:\n    type: score\n    criteria: [\"only\"]\n");
    assert_eq!(error.kind, SpecErrorKind::Cardinality);
    assert_eq!(error.path.to_string(), "/questions/q1/criteria");

    let two = parse_yaml("questions:\n  q1:\n    type: score\n    criteria: [\"lo\", ~]\n")
        .expect("two levels with a null level parses (D3)");
    match &two.questions[0].body {
        QuestionBody::Score { levels } => {
            assert_eq!(levels.len(), 2);
            assert_eq!(levels[1], None, "null level = undescribed level");
        }
        other => panic!("expected score body, got {other:?}"),
    }
}

#[test]
fn missing_and_misspelled_fields_are_positioned() {
    let missing_type = err_kind("questions:\n  q1:\n    instructions: \"hi\"\n");
    assert_eq!(missing_type.kind, SpecErrorKind::MissingField);
    assert_eq!(missing_type.path.to_string(), "/questions/q1/type");

    let missing_criteria = err_kind("questions:\n  q1:\n    type: choice\n");
    assert_eq!(missing_criteria.kind, SpecErrorKind::MissingField);
    assert_eq!(missing_criteria.path.to_string(), "/questions/q1/criteria");

    let unknown_field = err_kind("questions:\n  q1:\n    type: noul\n    rubrics: {}\n");
    assert_eq!(unknown_field.kind, SpecErrorKind::UnknownField);
    assert_eq!(unknown_field.path.to_string(), "/questions/q1/rubrics");

    let unknown_top = err_kind("question:\n  q1:\n    type: noul\n");
    assert_eq!(unknown_top.kind, SpecErrorKind::UnknownField);
    assert_eq!(unknown_top.path.to_string(), "/question");
}

#[test]
fn noul_criteria_accept_only_true_and_false_keys() {
    let error = err_kind("questions:\n  q1:\n    type: noul\n    criteria:\n      maybe: \"x\"\n");
    assert_eq!(error.kind, SpecErrorKind::UnknownField);
    assert_eq!(error.path.to_string(), "/questions/q1/criteria/maybe");
}

#[test]
fn duplicate_keys_are_positioned_errors_not_silent_drops() {
    let dup_option = err_kind(
        "questions:\n  q1:\n    type: choice\n    criteria:\n      a: \"first\"\n      b: \"x\"\n      a: \"second\"\n",
    );
    assert_eq!(dup_option.kind, SpecErrorKind::DuplicateKey);
    assert_eq!(dup_option.path.to_string(), "/questions/q1/criteria/a");

    let dup_question = err_kind("questions:\n  q1:\n    type: noul\n  q1:\n    type: noul\n");
    assert_eq!(dup_question.kind, SpecErrorKind::DuplicateKey);
    assert_eq!(dup_question.path.to_string(), "/questions/q1");
}

#[test]
fn empty_questions_map_is_a_cardinality_error() {
    let error = err_kind("questions: {}\n");
    assert_eq!(error.kind, SpecErrorKind::Cardinality);
    assert_eq!(error.path.to_string(), "/questions");
}

#[test]
fn syntax_errors_carry_line_and_column() {
    let error = err_kind("questions:\n  q1:\n    type: [unclosed\n");
    assert_eq!(error.kind, SpecErrorKind::Syntax);
    assert!(
        error.line.is_some(),
        "yaml syntax error carries a line: {error:?}"
    );

    let json_error = parse_json("{ not json").expect_err("invalid JSON rejected");
    assert_eq!(json_error.kind, SpecErrorKind::Syntax);
    assert!(json_error.line.is_some() && json_error.column.is_some());
}

#[test]
fn non_mapping_documents_and_questions_are_structure_errors() {
    let root = err_kind("- just\n- a\n- list\n");
    assert_eq!(root.kind, SpecErrorKind::Structure);
    assert_eq!(root.path.to_string(), "/");

    let question = err_kind("questions:\n  q1: \"a bare string\"\n");
    assert_eq!(question.kind, SpecErrorKind::Structure);
    assert_eq!(question.path.to_string(), "/questions/q1");

    let score_map =
        err_kind("questions:\n  q1:\n    type: score\n    criteria:\n      a: \"not a seq\"\n");
    assert_eq!(score_map.kind, SpecErrorKind::Structure);
}

#[test]
fn instructions_are_optional_and_nullable() {
    let set = parse_yaml(
        "questions:\n  q1:\n    type: noul\n  q2:\n    type: noul\n    instructions: ~\n",
    )
    .expect("absent and null instructions both parse (D4)");
    assert_eq!(set.questions[0].instructions, None);
    assert_eq!(set.questions[1].instructions, None);
}
