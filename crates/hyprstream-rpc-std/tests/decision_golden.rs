//! capnp ↔ Arrow golden-vector conformance across the jev-1 types (System One P0.1b).
//!
//! Golden vectors are the documented numeric examples from the S6a compat-spec
//! evidence (choice/score primitive pages, corroborated against the MIT
//! adapter's `confidence_metrics.py` semantics) — the same table pinned by
//! hyprstream-decision's `tests/jev1_compat.rs`. For every vector this test
//! proves that the two emission paths agree:
//!
//!   IR ──capnp──▶ decision.capnp message ──decode──▶ IR          (round-trip)
//!   IR ──Arrow──▶ RecordBatch (P0.1a contract)
//!
//! and that the capnp-decoded distribution matches the Arrow
//! `fixed_size_list<f32>` probabilities **bit-for-bit**, the derived argmax
//! label column, the field-metadata label list, the version-triple columns,
//! and the abstention/null semantics.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use arrow_array::{Array, FixedSizeListArray, Float32Array, StringArray};

use hyprstream_decision::confidence::{choice_confidence, expected_score, score_confidence};
use hyprstream_decision::{
    AnswerRow, AnswerValue, DecisionSchema, Entry, QuestionAnswer, QuestionSet, VersionTriple,
};
use hyprstream_rpc_std::decision;

const EPSILON: f32 = 1e-6;

/// One golden vector: an authored jev-1 document plus the answer distributions
/// from the docs' numeric examples.
struct GoldenVector {
    name: &'static str,
    yaml: &'static str,
    version: VersionTriple,
    rows: Vec<AnswerRow>,
}

fn row(pairs: Vec<(&str, QuestionAnswer)>) -> AnswerRow {
    AnswerRow {
        answers: pairs
            .into_iter()
            .map(|(id, answer)| (id.to_owned(), answer))
            .collect::<BTreeMap<_, _>>(),
    }
}

fn answered_choice(probabilities: &[f32]) -> QuestionAnswer {
    QuestionAnswer::answered(AnswerValue::Choice {
        probabilities: probabilities.to_vec(),
    })
}

fn answered_score(probabilities: &[f32]) -> QuestionAnswer {
    QuestionAnswer::answered(AnswerValue::Score {
        probabilities: probabilities.to_vec(),
    })
}

/// The golden-vector suite. Choice/score distributions are the docs' numeric
/// examples (S6a §2.3 corroboration table); noul exercises the third primitive
/// plus abstention.
fn golden_vectors() -> Vec<GoldenVector> {
    vec![
        // primitives/choice "requested_resolution" — docs confidence 0.16.
        GoldenVector {
            name: "choice/requested_resolution",
            yaml: r#"
state: "The user asked for the output at 1080p."
questions:
  requested_resolution:
    type: choice
    instructions: "What resolution did the user request?"
    criteria:
      720p: "The user requested 1280x720."
      1080p: "The user requested 1920x1080."
      1440p: "The user requested 2560x1440."
      4k: "The user requested 3840x2160."
"#,
            version: VersionTriple {
                schema: "golden-v1".into(),
                model: "decision-0.8b@w0".into(),
                calib: Some("calib-2026-09-20".into()),
            },
            rows: vec![
                row(vec![(
                    "requested_resolution",
                    answered_choice(&[0.1, 0.37, 0.24, 0.29]),
                )]),
                // Second row: abstained — null probabilities + null label.
                row(vec![("requested_resolution", QuestionAnswer::abstained())]),
            ],
        },
        // primitives/choice "tone" — docs confidence 0.88; uncalibrated batch
        // (null calib_version) plus an explicit Entry-null rubric (D3).
        GoldenVector {
            name: "choice/tone",
            yaml: r#"
questions:
  tone:
    type: choice
    criteria:
      formal: "Professional register."
      casual: "Conversational register."
      hostile: ~
"#,
            version: VersionTriple {
                schema: "golden-v1".into(),
                model: "decision-0.8b@w0".into(),
                calib: None,
            },
            rows: vec![row(vec![("tone", answered_choice(&[0.08, 0.92, 0.0]))])],
        },
        // primitives/score — docs worked example 0×0.0 + 1×0.70 + 2×0.30 = 1.30,
        // confidence 0.55. A null level exercises D3 on the score side.
        GoldenVector {
            name: "score/three_levels",
            yaml: r#"
questions:
  quality:
    type: score
    instructions: "Rate the overall quality."
    criteria:
      - "Unacceptable."
      - "Acceptable with minor issues."
      - ~
"#,
            version: VersionTriple {
                schema: "golden-v1".into(),
                model: "decision-0.8b@w0".into(),
                calib: Some("calib-2026-09-20".into()),
            },
            rows: vec![row(vec![("quality", answered_score(&[0.0, 0.7, 0.3]))])],
        },
        // noul: P(true) = 0.8, emitted as the 2-wide [0.2, 0.8] distribution.
        GoldenVector {
            name: "noul/basic",
            yaml: r#"
questions:
  is_refund:
    type: noul
    instructions: "Is the user requesting a refund?"
    criteria:
      "true": "The user explicitly asks for money back."
      "false": "No refund request is present."
"#,
            version: VersionTriple {
                schema: "golden-v1".into(),
                model: "decision-0.8b@w0".into(),
                calib: Some("calib-2026-09-20".into()),
            },
            rows: vec![row(vec![(
                "is_refund",
                QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.8 }),
            )])],
        },
    ]
}

/// Serialize + reparse a capnp message, proving the wire form (not just the
/// in-memory builder) round-trips.
fn wire_roundtrip(
    message: &capnp::message::Builder<capnp::message::HeapAllocator>,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, message).expect("serialize");
    bytes
}

/// Extract the f32 components of row `row` from a `{id}.probabilities`
/// fixed-size-list column. `None` = the whole list is null (abstention).
fn arrow_probabilities(batch: &arrow_array::RecordBatch, id: &str, row: usize) -> Option<Vec<f32>> {
    let column = batch
        .column_by_name(&format!("{id}.probabilities"))
        .unwrap_or_else(|| panic!("missing {id}.probabilities column"))
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("probabilities column is FixedSizeList");
    if column.is_null(row) {
        return None;
    }
    let values = column
        .value(row)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("probability components are Float32")
        .values()
        .to_vec();
    Some(values)
}

fn arrow_label(batch: &arrow_array::RecordBatch, id: &str, row: usize) -> Option<String> {
    let column = batch
        .column_by_name(&format!("{id}.label"))
        .unwrap_or_else(|| panic!("missing {id}.label column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("label column is Utf8");
    if column.is_null(row) {
        None
    } else {
        Some(column.value(row).to_owned())
    }
}

fn utf8_column(batch: &arrow_array::RecordBatch, name: &str, row: usize) -> Option<String> {
    let column = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("missing {name} column"))
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("version column is Utf8");
    if column.is_null(row) {
        None
    } else {
        Some(column.value(row).to_owned())
    }
}

/// The core conformance check for one golden vector.
fn check_golden_vector(vector: &GoldenVector) {
    let set: QuestionSet = hyprstream_decision::parse_yaml(vector.yaml)
        .unwrap_or_else(|error| panic!("{}: authored spec must parse: {error}", vector.name));

    // --- capnp question-set round-trip (spec emission) ---
    let bytes = wire_roundtrip(&decision::question_set_to_message(&set).expect("encode"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("capnp question-set message parses");
    let decoded_set = decision::question_set_from_reader(
        message
            .get_root::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
            .expect("question-set root"),
    )
    .unwrap_or_else(|error| panic!("{}: question-set decode: {error}", vector.name));
    assert_eq!(
        decoded_set, set,
        "{}: question-set capnp round-trip must be exact",
        vector.name
    );

    // --- capnp batch round-trip (answer emission) ---
    let bytes = wire_roundtrip(&decision::batch_to_message(&vector.version, &vector.rows).expect("encode"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("capnp batch message parses");
    let (decoded_version, decoded_rows) = decision::batch_from_reader(
        message
            .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
            .expect("batch root"),
    )
    .unwrap_or_else(|error| panic!("{}: batch decode: {error}", vector.name));
    assert_eq!(
        decoded_version, vector.version,
        "{}: version triple round-trip",
        vector.name
    );
    assert_eq!(
        decoded_rows, vector.rows,
        "{}: answer rows capnp round-trip must be exact",
        vector.name
    );

    // --- Arrow emission of the same rows (P0.1a contract) ---
    let schema = DecisionSchema::from_question_set(&set)
        .unwrap_or_else(|error| panic!("{}: arrow schema: {error}", vector.name));
    let batch = schema
        .build_batch(&vector.version, &vector.rows)
        .unwrap_or_else(|error| panic!("{}: arrow batch: {error}", vector.name));
    assert_eq!(batch.num_rows(), vector.rows.len());

    // Version-triple columns agree with the capnp-decoded triple.
    for (row_index, _) in vector.rows.iter().enumerate() {
        assert_eq!(
            utf8_column(&batch, "schema_version", row_index).as_deref(),
            Some(decoded_version.schema.as_str()),
            "{}: schema_version column",
            vector.name
        );
        assert_eq!(
            utf8_column(&batch, "model_version", row_index).as_deref(),
            Some(decoded_version.model.as_str()),
            "{}: model_version column",
            vector.name
        );
        assert_eq!(
            utf8_column(&batch, "calib_version", row_index),
            decoded_version.calib.clone(),
            "{}: calib_version column (null = uncalibrated)",
            vector.name
        );
    }

    // Per-question, per-row: capnp-decoded distribution ≡ Arrow probabilities
    // (bit-for-bit f32), and the abstention/null semantics agree.
    for question in &set.questions {
        let labels = question.labels();
        // Field metadata carries the label list (never schema metadata).
        let arrow_schema = batch.schema();
        let field = arrow_schema
            .field_with_name(&format!("{}.probabilities", question.id))
            .expect("probabilities field");
        assert_eq!(
            field.metadata().get("jev.kind").map(String::as_str),
            Some(question.kind.as_str()),
            "{}: jev.kind field metadata",
            vector.name
        );
        let meta_labels: Vec<String> =
            serde_json::from_str(field.metadata().get("jev.labels").expect("jev.labels"))
                .expect("jev.labels is a JSON array");
        assert_eq!(meta_labels, labels, "{}: jev.labels metadata", vector.name);

        for (row_index, (decoded_row, source_row)) in
            decoded_rows.iter().zip(vector.rows.iter()).enumerate()
        {
            let decoded_answer = &decoded_row.answers[&question.id];
            let arrow_probs = arrow_probabilities(&batch, &question.id, row_index);
            let arrow_label = arrow_label(&batch, &question.id, row_index);
            match (&decoded_answer.value, &source_row.answers[&question.id].value) {
                (None, None) => {
                    assert_eq!(
                        arrow_probs, None,
                        "{}: abstained row {row_index} has null probabilities",
                        vector.name
                    );
                    assert_eq!(
                        arrow_label, None,
                        "{}: abstained row {row_index} has null label",
                        vector.name
                    );
                }
                (Some(value), Some(_)) => {
                    let distribution = value.probabilities();
                    let arrow_probs = arrow_probs.unwrap_or_else(|| {
                        panic!("{}: answered row {row_index} must have probabilities", vector.name)
                    });
                    assert_eq!(
                        arrow_probs.len(),
                        distribution.len(),
                        "{}: cardinality agreement",
                        vector.name
                    );
                    // Bit-for-bit f32 equality between the two emission paths.
                    for (capnp_p, arrow_p) in distribution.iter().zip(arrow_probs.iter()) {
                        assert_eq!(
                            capnp_p.to_bits(),
                            arrow_p.to_bits(),
                            "{}: row {row_index} probability bits diverge",
                            vector.name
                        );
                    }
                    // Arrow label = argmax of the same distribution (D6 tie-break).
                    let argmax = hyprstream_decision::confidence::argmax_index(&distribution)
                        .expect("non-empty distribution");
                    assert_eq!(
                        arrow_label.as_deref(),
                        Some(labels[argmax].as_str()),
                        "{}: label column = argmax label",
                        vector.name
                    );
                }
                (decoded, source) => panic!(
                    "{}: capnp decode changed abstention: decoded {decoded:?} vs source {source:?}",
                    vector.name
                ),
            }
        }
    }
}

#[test]
fn golden_vectors_conform_across_capnp_and_arrow() {
    for vector in golden_vectors() {
        check_golden_vector(&vector);
    }
}

/// The docs' numeric confidence/expected-score values hold over the golden
/// distributions regardless of which emission path carried them (S6a §2.3
/// corroboration table — same numbers as hyprstream-decision's jev1_compat
/// tests, asserted here against the capnp-decoded payloads).
#[test]
fn golden_distributions_reproduce_documented_confidence_values() {
    let vectors = golden_vectors();
    let by_name: BTreeMap<_, _> = vectors.iter().map(|v| (v.name, v)).collect();

    let requested_resolution = by_name["choice/requested_resolution"].rows[0].answers
        ["requested_resolution"]
        .value
        .as_ref()
        .expect("answered")
        .probabilities();
    assert!((choice_confidence(&requested_resolution) - 0.16).abs() < EPSILON);

    let tone = by_name["choice/tone"].rows[0].answers["tone"]
        .value
        .as_ref()
        .expect("answered")
        .probabilities();
    assert!((choice_confidence(&tone) - 0.88).abs() < EPSILON);

    let quality = by_name["score/three_levels"].rows[0].answers["quality"]
        .value
        .as_ref()
        .expect("answered")
        .probabilities();
    assert!((score_confidence(&quality) - 0.55).abs() < EPSILON);
    assert!((expected_score(&quality) - 1.3).abs() < EPSILON);
}

/// Entry fidelity across the wire: structured entries (numbers, booleans,
/// nested maps/arrays, explicit null vs absent) survive the capnp round-trip
/// with ordering intact — the canonical serialization (P1.1) depends on it.
#[test]
fn structured_entries_roundtrip_exactly() {
    let yaml = r#"
state:
  request:
    resolution: [1920, 1080]
    hdr: true
    note: null
    bitrate: 12.5
questions:
  quality:
    type: score
    criteria: ["bad", "good"]
"#;
    let set = hyprstream_decision::parse_yaml(yaml).expect("parses");
    let bytes = wire_roundtrip(&decision::question_set_to_message(&set).expect("encode"));
    let message =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .expect("parses");
    let decoded = decision::question_set_from_reader(
        message
            .get_root::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
            .expect("root"),
    )
    .expect("decodes");
    assert_eq!(decoded, set);
    // Explicit null inside the state map is an Entry::Null, not an absent key.
    let Some(Entry::Map(state)) = decoded.state else {
        panic!("state is a map");
    };
    let Entry::Map(request) = &state[0].1 else {
        panic!("request is a map");
    };
    assert!(matches!(request[2], (ref key, Entry::Null) if key == "note"));
}

/// Decode-side guards: a kind tag that disagrees with the body union, and a
/// reserved v2 type, both fail loudly.
#[test]
fn decode_rejects_kind_body_mismatch_and_reserved_types() {
    use hyprstream_rpc_std::decision_capnp;

    let build_set = |kind: decision_capnp::QuestionKind,
                     body: &dyn Fn(
        decision_capnp::question_spec::body::Builder<'_>,
    )|
     -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut root = message.init_root::<decision_capnp::question_set::Builder<'_>>();
            root.reborrow().init_state().set_none(());
            let mut questions = root.reborrow().init_questions(1);
            let mut q = questions.reborrow().get(0);
            q.set_id("q");
            q.set_kind(kind);
            q.reborrow().init_instructions().set_none(());
            body(q.init_body());
        }
        wire_roundtrip(&message)
    };
    let decode = |bytes: &[u8]| {
        let message =
            capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
                .expect("parses");
        decision::question_set_from_reader(
            message
                .get_root::<decision_capnp::question_set::Reader<'_>>()
                .expect("root"),
        )
    };

    // kind=choice but body=noul → mismatch.
    let bytes = build_set(decision_capnp::QuestionKind::Choice, &|mut body| {
        body.reborrow().init_noul().set_none(());
    });
    let error = decode(&bytes).expect_err("mismatch rejected");
    assert!(matches!(
        error,
        decision::DecodeError::KindBodyMismatch { .. }
    ));

    // kind=span, body=span → reserved v2 type fails loudly.
    let bytes = build_set(decision_capnp::QuestionKind::Span, &|mut body| {
        body.reborrow().set_span(());
    });
    let error = decode(&bytes).expect_err("reserved type rejected");
    assert!(matches!(
        error,
        decision::DecodeError::ReservedQuestionType("span", _)
    ));
}

/// Wire-decode validation mirrors the authoring rules: empty ids, empty or
/// duplicated option names, and empty question sets are rejected (not deferred
/// to the Arrow surface).
#[test]
fn decode_rejects_invalid_wire_specs() {
    use hyprstream_rpc_std::decision_capnp;

    let build_and_decode = |build: &dyn Fn(
        decision_capnp::question_set::Builder<'_>,
    )|
     -> Result<QuestionSet, decision::DecodeError> {
        let mut message = capnp::message::Builder::new_default();
        build(message.init_root::<decision_capnp::question_set::Builder<'_>>());
        let bytes = wire_roundtrip(&message);
        let message = capnp::serialize::read_message(
            &mut &bytes[..],
            capnp::message::ReaderOptions::new(),
        )
        .expect("parses");
        decision::question_set_from_reader(
            message
                .get_root::<decision_capnp::question_set::Reader<'_>>()
                .expect("root"),
        )
    };
    let noul_question = |mut q: decision_capnp::question_spec::Builder<'_>, id: &str| {
        q.set_id(id);
        q.set_kind(decision_capnp::QuestionKind::Noul);
        q.reborrow().init_instructions().set_none(());
        q.init_body().init_noul().set_none(());
    };

    // Empty question id.
    let error = build_and_decode(&|mut root| {
        root.reborrow().init_state().set_none(());
        let mut questions = root.init_questions(1);
        noul_question(questions.reborrow().get(0), "");
    })
    .expect_err("empty id rejected");
    assert_eq!(error, decision::DecodeError::EmptyQuestionId);

    // Zero questions.
    let error = build_and_decode(&|mut root| {
        root.reborrow().init_state().set_none(());
        root.init_questions(0);
    })
    .expect_err("empty question set rejected");
    assert_eq!(error, decision::DecodeError::EmptyQuestionSet);

    let choice_question = |mut q: decision_capnp::question_spec::Builder<'_>,
                           names: &[&str]| {
        q.set_id("c");
        q.set_kind(decision_capnp::QuestionKind::Choice);
        q.reborrow().init_instructions().set_none(());
        let mut options = q.init_body().init_choice(names.len() as u32);
        for (index, name) in names.iter().enumerate() {
            let mut option = options.reborrow().get(index as u32);
            option.set_name(name);
            option.init_rubric().set_none(());
        }
    };

    // Empty option name.
    let error = build_and_decode(&|mut root| {
        root.reborrow().init_state().set_none(());
        let mut questions = root.init_questions(1);
        choice_question(questions.reborrow().get(0), &["a", ""]);
    })
    .expect_err("empty option name rejected");
    assert!(matches!(
        error,
        decision::DecodeError::EmptyChoiceOptionName(_)
    ));

    // Duplicate option name.
    let error = build_and_decode(&|mut root| {
        root.reborrow().init_state().set_none(());
        let mut questions = root.init_questions(1);
        choice_question(questions.reborrow().get(0), &["a", "b", "a"]);
    })
    .expect_err("duplicate option name rejected");
    assert!(matches!(
        error,
        decision::DecodeError::DuplicateChoiceOptionName(_, _)
    ));
}

/// The calib version survives a round-trip with the uncalibrated state
/// distinct from any string value (explicit OptionText wrapper — a null Text
/// pointer would collapse to "" in consumers without pointer-presence
/// tracking).
#[test]
fn calib_none_is_distinct_from_empty_string() {
    for calib in [None, Some(String::new()), Some("calib-1".to_owned())] {
        let version = VersionTriple {
            schema: "s".into(),
            model: "m".into(),
            calib: calib.clone(),
        };
        let rows = vec![row(vec![(
            "q",
            QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.5 }),
        )])];
        let bytes = wire_roundtrip(&decision::batch_to_message(&version, &rows).expect("encode"));
        let message = capnp::serialize::read_message(
            &mut &bytes[..],
            capnp::message::ReaderOptions::new(),
        )
        .expect("parses");
        let (decoded, _) = decision::batch_from_reader(
            message
                .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
                .expect("root"),
        )
        .expect("decodes");
        assert_eq!(decoded.calib, calib, "calib {calib:?} round-trips exactly");
    }
}

/// Encode-side guards mirror the decode-side ones: an abstained answer with a
/// conformal set is rejected instead of emitting a message the decoder would
/// refuse, and the wire kind tag is derived from the body even when a
/// directly-constructed spec carries a conflicting `kind`.
#[test]
fn encode_rejects_abstained_with_conformal_set_and_derives_kind_from_body() {
    let version = VersionTriple {
        schema: "s".into(),
        model: "m".into(),
        calib: None,
    };
    let mut invalid = row(vec![("q", QuestionAnswer::abstained())]);
    invalid.answers.get_mut("q").expect("answer").conformal_set = Some(vec!["x".into()]);
    let error = decision::batch_to_message(&version, &[invalid])
        .err()
        .expect("abstained with conformal set rejected at encode");
    assert_eq!(
        error,
        decision::EncodeError::AbstainedWithConformalSet("q".into())
    );

    // kind tag follows the body, not a stale denormalized field.
    let mut set: QuestionSet = hyprstream_decision::parse_yaml(
        r#"
questions:
  q:
    type: noul
"#,
    )
    .expect("parses");
    set.questions[0].kind = hyprstream_decision::QuestionKind::Score;
    let bytes = wire_roundtrip(&decision::question_set_to_message(&set).expect("encode"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("parses");
    let decoded = decision::question_set_from_reader(
        message
            .get_root::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
            .expect("root"),
    )
    .expect("derived tag keeps the message decodable");
    assert_eq!(decoded.questions[0].kind, hyprstream_decision::QuestionKind::Noul);
}

/// Encode-side distribution validation: NaN noul, out-of-range components,
/// and sums outside the D5 producer tolerance are rejected before emission;
/// an empty question set cannot be encoded into a message the decoder
/// rejects.
#[test]
fn encode_rejects_invalid_distributions_and_empty_sets() {
    let version = VersionTriple {
        schema: "s".into(),
        model: "m".into(),
        calib: None,
    };

    let nan_noul = row(vec![(
        "q",
        QuestionAnswer::answered(AnswerValue::Noul { p_true: f32::NAN }),
    )]);
    let error = decision::batch_to_message(&version, &[nan_noul])
        .err()
        .expect("NaN noul rejected");
    assert!(matches!(
        error,
        decision::EncodeError::NoulProbabilityOutOfRange(_, _)
    ));

    let bad_choice = row(vec![("q", answered_choice(&[0.9, 0.9]))]);
    let error = decision::batch_to_message(&version, &[bad_choice])
        .err()
        .expect("sum outside producer tolerance rejected");
    assert!(matches!(
        error,
        decision::EncodeError::InvalidDistribution(_, _)
    ));

    let negative = row(vec![("q", answered_score(&[1.1, -0.1]))]);
    let error = decision::batch_to_message(&version, &[negative])
        .err()
        .expect("out-of-range component rejected");
    assert!(matches!(
        error,
        decision::EncodeError::InvalidDistribution(_, _)
    ));

    let error = decision::question_set_to_message(&QuestionSet::default())
        .err()
        .expect("empty question set rejected at encode");
    assert_eq!(error, decision::EncodeError::EmptyQuestionSet);
}

/// Encode mirrors the decode-side authoring rules (review thread, 2026-09-21):
/// the IR is public, so a directly-constructed `QuestionSet` with an empty
/// question id, an empty choice option name, or duplicate option names is
/// rejected at encode — the same messages `question_set_from_reader` rejects
/// with EmptyQuestionId / EmptyChoiceOptionName / DuplicateChoiceOptionName.
#[test]
fn encode_rejects_empty_ids_and_bad_choice_option_names() {
    let noul = |id: &str| hyprstream_decision::QuestionSpec {
        id: id.to_owned(),
        kind: hyprstream_decision::QuestionKind::Noul,
        instructions: None,
        body: hyprstream_decision::QuestionBody::Noul { criteria: None },
    };
    let choice = |id: &str, names: &[&str]| hyprstream_decision::QuestionSpec {
        id: id.to_owned(),
        kind: hyprstream_decision::QuestionKind::Choice,
        instructions: None,
        body: hyprstream_decision::QuestionBody::Choice {
            options: names
                .iter()
                .map(|name| hyprstream_decision::ChoiceOption {
                    name: (*name).to_owned(),
                    rubric: None,
                })
                .collect(),
        },
    };

    let mut set = QuestionSet { state: None, questions: vec![noul("")] };
    let error = decision::question_set_to_message(&set)
        .err()
        .expect("empty question id rejected at encode");
    assert_eq!(error, decision::EncodeError::EmptyQuestionId);

    set.questions = vec![choice("q", &["", "b"])];
    let error = decision::question_set_to_message(&set)
        .err()
        .expect("empty choice option name rejected at encode");
    assert_eq!(error, decision::EncodeError::EmptyChoiceOptionName("q".into()));

    set.questions = vec![choice("q", &["a", "b", "a"])];
    let error = decision::question_set_to_message(&set)
        .err()
        .expect("duplicate choice option name rejected at encode");
    assert_eq!(
        error,
        decision::EncodeError::DuplicateChoiceOptionName("q".into(), "a".into())
    );

    // The well-formed counterpart still encodes and decodes cleanly.
    set.questions = vec![choice("q", &["a", "b"])];
    let bytes = wire_roundtrip(&decision::question_set_to_message(&set).expect("encodes"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("parses");
    let decoded = decision::question_set_from_reader(
        message
            .get_root::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
            .expect("root"),
    )
    .expect("decodes");
    assert_eq!(
        decoded.questions[0]
            .body,
        hyprstream_decision::QuestionBody::Choice {
            options: vec![
                hyprstream_decision::ChoiceOption { name: "a".into(), rubric: None },
                hyprstream_decision::ChoiceOption { name: "b".into(), rubric: None },
            ]
        },
    );
}

/// The conformal-set field is an explicit option wrapper (review thread,
/// 2026-09-21): `none` (absent) and a present empty set stay distinct across
/// the wire. A plain nullable List(Text) pointer reads an absent set as the
/// default empty list, so a raw-API consumer rebuild turned None into
/// Some(vec![]) — flipping the Arrow null semantics and failing abstained
/// rows with AbstainedWithConformalSet.
#[test]
fn conformal_set_none_and_present_empty_stay_distinct() {
    let version = VersionTriple {
        schema: "s".into(),
        model: "m".into(),
        calib: None,
    };

    // Absent set: decode gives None.
    let absent = row(vec![(
        "q",
        QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.75 }),
    )]);
    let bytes =
        wire_roundtrip(&decision::batch_to_message(&version, &[absent]).expect("encodes"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("parses");
    let reader = message
        .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .expect("root");
    let (_, rows) = decision::batch_from_reader(reader).expect("decodes");
    assert_eq!(rows[0].answers["q"].conformal_set, None);

    // Raw-API consumer rebuild that never touches the conformal-set field: the
    // explicit option union defaults to none, so presence cannot be gained
    // here (the nullable pointer read as an empty list and re-encoded
    // present-empty under the old schema).
    let mut rebuilt = capnp::message::Builder::new_default();
    {
        let mut root = rebuilt
            .init_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Builder<'_>>();
        root.set_version(reader.get_version().expect("version"))
            .expect("copies");
        let rows_builder = root.init_rows(1);
        let answers = rows_builder.get(0).init_answers(1);
        let mut answer = answers.get(0);
        answer.set_question_id("q");
        answer.init_value().set_noul(0.75);
    }
    let rebuilt_bytes = wire_roundtrip(&rebuilt);
    let message = capnp::serialize::read_message(
        &mut &rebuilt_bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("parses");
    let reader = message
        .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .expect("root");
    let (_, rows) = decision::batch_from_reader(reader).expect("decodes");
    assert_eq!(
        rows[0].answers["q"].conformal_set,
        None,
        "rebuild keeps the set absent"
    );

    // A present-but-empty set is preserved as Some(vec![]), not collapsed.
    let mut present = row(vec![(
        "q",
        QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.75 }),
    )]);
    present
        .answers
        .get_mut("q")
        .expect("answer")
        .conformal_set = Some(Vec::new());
    let bytes =
        wire_roundtrip(&decision::batch_to_message(&version, &[present]).expect("encodes"));
    let message = capnp::serialize::read_message(
        &mut &bytes[..],
        capnp::message::ReaderOptions::new(),
    )
    .expect("parses");
    let reader = message
        .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .expect("root");
    let (_, rows) = decision::batch_from_reader(reader).expect("decodes");
    assert_eq!(rows[0].answers["q"].conformal_set, Some(Vec::new()));
}
