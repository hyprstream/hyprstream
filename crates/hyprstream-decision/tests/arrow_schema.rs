//! Arrow schema emission: snapshot contract, field-metadata placement, label-set
//! evolution, and batch encoding (including abstention and the version triple).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use arrow_array::{Array, FixedSizeListArray, Float32Array, ListArray, StringArray};
use arrow_schema::{DataType, Schema};
use hyprstream_decision::arrow::{DecisionSchema, FIELD_META_KIND, FIELD_META_LABELS};
use hyprstream_decision::{
    parse_yaml, AnswerRow, AnswerValue, ArrowSchemaError, BatchError, QuestionAnswer, QuestionSet,
    VersionTriple,
};

fn fixture_set() -> QuestionSet {
    parse_yaml(
        r#"
questions:
  is_refund:
    type: noul
    instructions: "The customer is asking for a refund."
  tone:
    type: choice
    criteria:
      positive: "The review expresses approval."
      negative: "The review expresses disapproval."
      neutral: ~
  severity:
    type: score
    criteria: ["Cosmetic issue only", "Usable but damaged", "Completely unusable"]
"#,
    )
    .expect("fixture parses")
}

fn version() -> VersionTriple {
    VersionTriple {
        schema: "qe-v1".into(),
        model: "jev-1.13.0".into(),
        calib: Some("calib-2026-09-18".into()),
    }
}

/// Compact, deterministic rendering for the snapshot assertions below: field order,
/// types, nullability, and sorted field metadata. (No snapshot crate — the workspace
/// keeps dev-deps minimal, and the render is the contract.)
fn render_schema(schema: &Schema) -> String {
    let mut out = String::new();
    for field in schema.fields() {
        out.push_str(&format!(
            "{}: {} nullable={}\n",
            field.name(),
            render_type(field.data_type()),
            field.is_nullable()
        ));
        let mut metadata: Vec<_> = field.metadata().iter().collect();
        metadata.sort();
        for (key, value) in metadata {
            out.push_str(&format!("  meta {key} = {value}\n"));
        }
    }
    out.push_str(&format!(
        "schema metadata entries: {}\n",
        schema.metadata().len()
    ));
    out
}

fn render_type(data_type: &DataType) -> String {
    match data_type {
        DataType::Float32 => "f32".to_owned(),
        DataType::Utf8 => "utf8".to_owned(),
        DataType::FixedSizeList(child, size) => {
            format!("fixed_size_list<{};{size}>", render_type(child.data_type()))
        }
        DataType::List(child) => format!("list<{}>", render_type(child.data_type())),
        other => format!("{other:?}"),
    }
}

#[test]
fn arrow_schema_snapshot() {
    let schema = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");
    let rendered = render_schema(&schema.arrow_schema());
    let expected = r#"is_refund.probabilities: fixed_size_list<f32;2> nullable=true
  meta jev.kind = noul
  meta jev.labels = ["false","true"]
is_refund.label: utf8 nullable=true
is_refund.conformal_set: list<utf8> nullable=true
tone.probabilities: fixed_size_list<f32;3> nullable=true
  meta jev.kind = choice
  meta jev.labels = ["positive","negative","neutral"]
tone.label: utf8 nullable=true
tone.conformal_set: list<utf8> nullable=true
severity.probabilities: fixed_size_list<f32;3> nullable=true
  meta jev.kind = score
  meta jev.labels = ["0","1","2"]
severity.label: utf8 nullable=true
severity.conformal_set: list<utf8> nullable=true
schema_version: utf8 nullable=false
model_version: utf8 nullable=false
calib_version: utf8 nullable=true
schema metadata entries: 0
"#;
    assert_eq!(rendered, expected);
}

#[test]
fn conformal_column_is_optional_but_reserved_by_default() {
    let without = DecisionSchema::from_question_set(&fixture_set())
        .expect("schema builds")
        .without_conformal_set();
    let rendered = render_schema(&without.arrow_schema());
    assert!(
        !rendered.contains("conformal_set"),
        "opt-out drops the reserved columns"
    );
    assert!(
        rendered.contains("schema_version"),
        "version triple always emitted"
    );
}

#[test]
fn labels_live_in_field_metadata_never_schema_metadata() {
    let schema = DecisionSchema::from_question_set(&fixture_set())
        .expect("schema builds")
        .arrow_schema();
    assert!(schema.metadata().is_empty(), "schema metadata stays empty");

    let probabilities = schema
        .field_with_name("tone.probabilities")
        .expect("field exists");
    assert_eq!(
        probabilities
            .metadata()
            .get(FIELD_META_KIND)
            .map(String::as_str),
        Some("choice")
    );
    assert_eq!(
        probabilities
            .metadata()
            .get(FIELD_META_LABELS)
            .map(String::as_str),
        Some(r#"["positive","negative","neutral"]"#)
    );

    let label_field = schema.field_with_name("tone.label").expect("label field");
    assert!(
        label_field.metadata().is_empty(),
        "label column carries no metadata"
    );
}

#[test]
fn label_set_evolution_fails_loudly() {
    let old = DecisionSchema::from_question_set(&fixture_set()).expect("old schema");
    let evolved_doc = r#"
questions:
  is_refund:
    type: noul
  tone:
    type: choice
    criteria:
      positive: "The review expresses approval."
      negative: "The review expresses disapproval."
      mixed: "Both approval and disapproval."
      neutral: ~
  severity:
    type: score
    criteria: ["Cosmetic issue only", "Usable but damaged", "Completely unusable"]
"#;
    let new = DecisionSchema::from_question_set(&parse_yaml(evolved_doc).expect("parses"))
        .expect("new schema builds");
    let error = DecisionSchema::check_evolution(&old, &new)
        .expect_err("changing the tone label set must fail loudly");
    let error = match error {
        hyprstream_decision::EvolutionError::LabelSetChanged(error) => error,
        other => panic!("label-set change must surface as LabelSetChanged, got {other:?}"),
    };
    assert_eq!(error.changes.len(), 1);
    assert_eq!(error.changes[0].question_id, "tone");
    assert_eq!(
        error.changes[0].new_labels,
        vec!["positive", "negative", "mixed", "neutral"]
    );
    assert!(error.to_string().contains("new schema version"));
    assert_ne!(
        old.fingerprint(),
        new.fingerprint(),
        "evolved label sets change the fingerprint"
    );
}

#[test]
fn conformal_emission_flip_fails_evolution_like_the_fingerprint() {
    use hyprstream_decision::EvolutionError;

    let old = DecisionSchema::from_question_set(&fixture_set()).expect("old schema");
    let new = DecisionSchema::from_question_set(&fixture_set())
        .expect("new schema")
        .without_conformal_set();
    let error = DecisionSchema::check_evolution(&old, &new)
        .expect_err("flipping conformal-set emission changes the emitted column set");
    assert_eq!(
        error,
        EvolutionError::EmissionShapeChanged {
            old_conformal_set: true,
            new_conformal_set: false,
        }
    );
    assert!(error.to_string().contains("new schema version"));
    assert_ne!(
        old.fingerprint(),
        new.fingerprint(),
        "the fingerprint already treated the flag as schema content; now the check agrees"
    );

    // The other direction (off -> on) is likewise schema evolution.
    let error = DecisionSchema::check_evolution(&new, &old)
        .expect_err("adding the reserved column is also a column-set change");
    assert!(matches!(
        error,
        EvolutionError::EmissionShapeChanged {
            old_conformal_set: false,
            new_conformal_set: true,
        }
    ));
}

#[test]
fn compatible_evolution_reports_added_and_removed_questions() {
    let old = DecisionSchema::from_question_set(&fixture_set()).expect("old schema");
    let same = DecisionSchema::from_question_set(&fixture_set()).expect("same schema");
    let report =
        DecisionSchema::check_evolution(&old, &same).expect("identical schemas are compatible");
    assert!(report.added.is_empty() && report.removed.is_empty());
    assert_eq!(
        old.fingerprint(),
        same.fingerprint(),
        "same labels, same fingerprint"
    );

    let evolved_doc = r#"
questions:
  is_refund:
    type: noul
  tone:
    type: choice
    criteria:
      positive: "The review expresses approval."
      negative: "The review expresses disapproval."
      neutral: ~
  queue:
    type: choice
    criteria:
      support: "Support queue."
      billing: "Billing queue."
"#;
    let new = DecisionSchema::from_question_set(&parse_yaml(evolved_doc).expect("parses"))
        .expect("new schema builds");
    let report = DecisionSchema::check_evolution(&old, &new).expect("add/remove is compatible");
    assert_eq!(report.added, vec!["queue"]);
    assert_eq!(report.removed, vec!["severity"]);
}

#[test]
fn schema_construction_validates_identifiers() {
    let bad_id = parse_yaml("questions:\n  \"not an id\":\n    type: noul\n")
        .expect("authoring accepts opaque ids");
    assert!(matches!(
        DecisionSchema::from_question_set(&bad_id),
        Err(ArrowSchemaError::InvalidIdentifier(_))
    ));

    let reserved = parse_yaml("questions:\n  schema_version:\n    type: noul\n").expect("parses");
    assert!(matches!(
        DecisionSchema::from_question_set(&reserved),
        Err(ArrowSchemaError::ReservedColumnName(_))
    ));
}

fn row(answers: Vec<(&str, QuestionAnswer)>) -> AnswerRow {
    AnswerRow {
        answers: answers
            .into_iter()
            .map(|(id, answer)| (id.to_owned(), answer))
            .collect(),
    }
}

#[test]
fn batch_encoding_with_abstention_and_conformal_set() {
    let schema = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");
    let rows = vec![
        row(vec![
            (
                "is_refund",
                QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
            ),
            (
                "tone",
                QuestionAnswer {
                    value: Some(AnswerValue::Choice {
                        probabilities: vec![0.1, 0.6, 0.3],
                    }),
                    conformal_set: Some(vec!["negative".into(), "neutral".into()]),
                },
            ),
            (
                "severity",
                QuestionAnswer::answered(AnswerValue::Score {
                    probabilities: vec![0.0, 0.7, 0.3],
                }),
            ),
        ]),
        row(vec![
            ("is_refund", QuestionAnswer::abstained()),
            ("tone", QuestionAnswer::abstained()),
            ("severity", QuestionAnswer::abstained()),
        ]),
    ];
    let batch = schema.build_batch(&version(), &rows).expect("batch builds");
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 12);

    // noul: [1 - p, p] over ["false", "true"]; label "true" for p = 0.7.
    let noul_probs = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("probabilities column");
    assert_eq!(noul_probs.null_count(), 1, "row 1 abstained");
    let first = noul_probs.value(0);
    let first = first
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("f32 values");
    assert_eq!(&first.values()[..], &[0.3_f32, 0.7_f32]);

    let noul_labels = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(noul_labels.value(0), "true");
    assert!(noul_labels.is_null(1), "abstention is a null label");

    // choice: argmax label + populated conformal set on row 0, nulls on row 1.
    let tone_labels = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(tone_labels.value(0), "negative");
    let tone_sets = batch
        .column(5)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("sets");
    assert!(tone_sets.is_null(1));
    let set0 = tone_sets.value(0);
    let set0 = set0
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("utf8 set");
    assert_eq!(
        set0.iter().flatten().collect::<Vec<_>>(),
        vec!["negative", "neutral"]
    );

    // severity: argmax of [0, 0.7, 0.3] is level "1".
    let severity_labels = batch
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(severity_labels.value(0), "1");

    // Version triple: repeated per row, calib populated here.
    let schema_col = batch
        .column(9)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("schema_version");
    assert_eq!(schema_col.value(0), "qe-v1");
    let model_col = batch
        .column(10)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("model_version");
    assert_eq!(model_col.value(1), "jev-1.13.0");
    let calib_col = batch
        .column(11)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("calib_version");
    assert_eq!(calib_col.value(0), "calib-2026-09-18");
    assert_eq!(calib_col.null_count(), 0);
}

#[test]
fn batch_version_triple_allows_uncalibrated() {
    let schema = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");
    let rows = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.5 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.5, 0.5, 0.0],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.5, 0.5, 0.0],
            }),
        ),
    ])];
    let uncalibrated = VersionTriple {
        schema: "qe-v1".into(),
        model: "jev-1.13.0".into(),
        calib: None,
    };
    let batch = schema
        .build_batch(&uncalibrated, &rows)
        .expect("batch builds");
    let calib_col = batch
        .column(11)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("calib_version");
    assert_eq!(
        calib_col.null_count(),
        1,
        "null calib_version = uncalibrated"
    );

    // D6: argmax ties break toward the earliest option.
    let tone_labels = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(tone_labels.value(0), "positive");
    // noul at exactly 0.5: ["false", "true"] order means the tie lands on "false".
    let noul_labels = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(noul_labels.value(0), "false");
}

#[test]
fn batch_validation_fails_loudly() {
    let schema = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");

    let missing = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &missing),
        Err(BatchError::MissingAnswer { row: 0, .. })
    ));

    let unknown = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
        ("ghost", QuestionAnswer::abstained()),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &unknown),
        Err(BatchError::UnknownQuestion { row: 0, .. })
    ));

    let wrong_kind = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.2, 0.8],
            }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &wrong_kind),
        Err(BatchError::KindMismatch { row: 0, .. })
    ));

    let wrong_cardinality = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.4, 0.6],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &wrong_cardinality),
        Err(BatchError::CardinalityMismatch { row: 0, .. })
    ));

    // Producer tolerance is 1e-6 (D5): 0.4 + 0.4 + 0.4 = 1.2 is far outside it.
    let bad_distribution = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.4, 0.4, 0.4],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &bad_distribution),
        Err(BatchError::InvalidDistribution(0, _, _))
    ));

    let abstained_with_set = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer {
                value: None,
                conformal_set: Some(vec!["neutral".into()]),
            },
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &abstained_with_set),
        Err(BatchError::AbstainedWithConformalSet { row: 0, .. })
    ));

    let unknown_label = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer {
                value: Some(AnswerValue::Choice {
                    probabilities: vec![0.2, 0.3, 0.5],
                }),
                conformal_set: Some(vec!["mixed".into()]),
            },
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.2, 0.3, 0.5],
            }),
        ),
    ])];
    assert!(matches!(
        schema.build_batch(&version(), &unknown_label),
        Err(BatchError::UnknownConformalLabel { row: 0, .. })
    ));
}

#[test]
fn batch_schema_matches_emitted_schema() {
    let schema = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");
    let batch = schema
        .build_batch(&version(), &[])
        .expect("an empty batch is valid");
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(
        batch.schema().as_ref().fields().len(),
        schema.arrow_schema().fields().len()
    );
    assert_eq!(batch.schema().as_ref(), &schema.arrow_schema());
}

#[test]
fn batch_without_conformal_set_encodes() {
    use hyprstream_decision::EvolutionError;

    let full = DecisionSchema::from_question_set(&fixture_set()).expect("schema builds");
    let schema = full.clone().without_conformal_set();
    let rows = vec![row(vec![
        (
            "is_refund",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.7 }),
        ),
        (
            "tone",
            QuestionAnswer::answered(AnswerValue::Choice {
                probabilities: vec![0.1, 0.6, 0.3],
            }),
        ),
        (
            "severity",
            QuestionAnswer::answered(AnswerValue::Score {
                probabilities: vec![0.0, 0.7, 0.3],
            }),
        ),
    ])];
    let batch = schema.build_batch(&version(), &rows).expect("batch builds");
    assert_eq!(batch.num_columns(), 9, "2 columns per question + version triple");
    assert_eq!(batch.schema().as_ref(), &schema.arrow_schema());
    // The flip is schema evolution, so a consumer diffing the two generations fails
    // loudly rather than silently reading positional columns.
    assert!(matches!(
        DecisionSchema::check_evolution(&full, &schema),
        Err(EvolutionError::EmissionShapeChanged { .. })
    ));
}

/// D5 producer tolerance at max cardinality: a 255-wide distribution whose f32 sum
/// drifts ~7.2e-7 from 1 must still pass the 1e-6 gate. The ramp p_i = (i+1)/32640
/// (Σ(i+1) = 32640) is deterministic and sits near the realistic worst case the
/// tolerance was sized for, so this pins the headroom.
#[test]
fn max_cardinality_distribution_within_producer_tolerance() {
    use hyprstream_decision::{ChoiceOption, QuestionBody, QuestionKind, QuestionSpec};

    let options: Vec<ChoiceOption> = (0..255)
        .map(|i| ChoiceOption {
            name: format!("opt_{i}"),
            rubric: None,
        })
        .collect();
    let question = QuestionSpec {
        id: "wide".to_owned(),
        kind: QuestionKind::Choice,
        instructions: None,
        body: QuestionBody::Choice { options },
    };
    let schema = DecisionSchema::new(vec![question]).expect("schema builds");

    let total = 255.0_f64 * 256.0 / 2.0; // 32640
    let probabilities: Vec<f32> = (0..255).map(|i| ((i + 1) as f64 / total) as f32).collect();
    let f32_sum = probabilities.iter().fold(0.0_f32, |acc, p| acc + p);
    let drift = (f32_sum - 1.0).abs();
    assert!(
        drift <= hyprstream_decision::confidence::PRODUCER_SUM_TOLERANCE,
        "ramp drift {drift:e} must stay within the 1e-6 producer tolerance"
    );
    assert!(drift > 1e-7, "the ramp should exercise real headroom, got {drift:e}");

    let rows = vec![row(vec![(
        "wide",
        QuestionAnswer::answered(AnswerValue::Choice { probabilities }),
    )])];
    let batch = schema.build_batch(&version(), &rows).expect("batch builds");
    let wide = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("probabilities column");
    assert_eq!(wide.value_length(), 255);
    let labels = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("labels");
    assert_eq!(labels.value(0), "opt_254", "argmax of the ramp is the last option");
}
