//! R4 serving wire conformance: the `decide` request/response types round-trip
//! through Cap'n Proto in BOTH generated modules, the nested E2
//! `DecisionBatch` keeps calibration-null and typed distributions, and the
//! model/inference generated types alias ONE canonical decision type (no
//! duplicate struct drift).
//!
//! Requires the decision schema (E2) and the serving schema additions (R4).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use hyprstream_rpc::capnp::FromCapnp;
use hyprstream_rpc_std::decision;
use hyprstream_rpc_std::decision_client;
use hyprstream_rpc_std::inference_client;
use hyprstream_rpc_std::inference_client::{
    DecisionAppliedLimits, DecisionProvenance, DecisionRequest, DecisionResponse,
};

fn example_schema_id() -> String {
    format!("clef-raw-cloudflare-clef-8b2e5fd-1-q{}", "1".repeat(64))
}

fn example_batch() -> decision_client::DecisionBatch {
    use hyprstream_decision::{AnswerRow, AnswerValue, QuestionAnswer};
    let mut answers = std::collections::BTreeMap::new();
    answers.insert(
        "is_ok".to_owned(),
        QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.75 }),
    );
    answers.insert(
        "pick".to_owned(),
        QuestionAnswer::answered(AnswerValue::Choice {
            probabilities: vec![0.2, 0.8],
        }),
    );
    answers.insert(
        "grade".to_owned(),
        QuestionAnswer::answered(AnswerValue::Score {
            probabilities: vec![0.6, 0.4],
        }),
    );
    let row = AnswerRow { answers };
    let version = hyprstream_decision::VersionTriple {
        schema: example_schema_id(),
        model: "clef-flash-fp8@7b43c2d799e7b5e9".to_owned(),
        calib: None,
    };
    // E2 producer path enforces the distribution/abstention contract.
    let message = decision::batch_to_message(&version, &[row]).expect("encode");
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message).unwrap();
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .unwrap();
    let reader = read
        .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .unwrap();
    decision_client::DecisionBatch::read_from(reader).expect("read")
}

#[test]
fn decide_request_round_trips_through_wire() {
    let request = DecisionRequest {
        clef_json: r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}}}"#.to_owned(),
        max_state_tokens: Some(128),
        timeout_ms: Some(5_000),
    };
    let bytes = hyprstream_rpc::serialize_message(|message| {
        use hyprstream_rpc::capnp::ToCapnp;
        let mut builder = message
            .init_root::<hyprstream_rpc_std::inference_capnp::decision_request::Builder<'_>>();
        request.write_to(&mut builder);
    })
    .expect("serialize");
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .expect("capnp parse");
    let reader = read
        .get_root::<hyprstream_rpc_std::inference_capnp::decision_request::Reader<'_>>()
        .expect("root");
    let decoded = DecisionRequest::read_from(reader).expect("decode");
    assert_eq!(decoded.clef_json, request.clef_json);
    assert_eq!(decoded.max_state_tokens, Some(128));
    assert_eq!(decoded.timeout_ms, Some(5_000));
}

#[test]
fn decide_response_round_trips_batch_provenance_and_limits() {
    let batch = example_batch();
    let response = DecisionResponse {
        batch,
        provenance: DecisionProvenance {
            encoding_version: "cloudflare-clef-8b2e5fd-1".to_owned(),
            model_ref: "clef-flash-fp8:main".to_owned(),
            artifact_commit: "7b43c2d799e7b5e9fef96b6137840088d75af0d6".to_owned(),
            head_weights_sha256: "a".repeat(64),
            head_config_sha256: "b".repeat(64),
            tokenizer_sha256: "c".repeat(64),
            backbone_config_sha256: "d".repeat(64),
            quantization: "float-quantized-channel-w8-dynact".to_owned(),
            head_tensor_count: 122,
            compute_profile: "cpu-float-quantized-channel-w8-dynact".to_owned(),
        },
        limits: DecisionAppliedLimits {
            max_length: 16384,
            max_state_tokens: Some(8192),
            state_tokens_admitted: 42,
            sequence_tokens: 1000,
            state_truncated: true,
        },
    };
    let bytes = hyprstream_rpc::serialize_message(|message| {
        use hyprstream_rpc::capnp::ToCapnp;
        let mut builder = message
            .init_root::<hyprstream_rpc_std::inference_capnp::decision_response::Builder<'_>>();
        response.write_to(&mut builder);
    })
    .expect("serialize");
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .expect("capnp parse");
    let reader = read
        .get_root::<hyprstream_rpc_std::inference_capnp::decision_response::Reader<'_>>()
        .expect("root");
    let decoded = DecisionResponse::read_from(reader).expect("decode");

    // Provenance and limits survive exactly.
    let p = &decoded.provenance;
    assert_eq!(p.encoding_version, "cloudflare-clef-8b2e5fd-1");
    assert_eq!(p.artifact_commit.len(), 40);
    assert_eq!(p.head_tensor_count, 122);
    assert!(p.quantization.starts_with("float-quantized"));
    assert!(p.compute_profile.starts_with("cpu-"));
    assert!(decoded.limits.state_truncated);
    assert_eq!(decoded.limits.sequence_tokens, 1000);

    // The nested E2 batch keeps its identity and calibration-null.
    let b = &decoded.batch;
    assert_eq!(b.version.model, "clef-flash-fp8@7b43c2d799e7b5e9");
    assert!(
        b.version.calib.is_none(),
        "calibration must be explicit null"
    );
    assert_eq!(b.version.schema, example_schema_id());
    assert_eq!(b.rows.len(), 1);
    assert_eq!(b.rows[0].answers.len(), 3);
}

#[test]
fn model_and_inference_modules_share_one_canonical_decision_type() {
    // Compile-time alias identity: the model module's DecisionResponse IS the
    // inference module's (single canonical decision_client type via the
    // derive crate's SERVICE_MODULES aliasing).
    fn assert_same_type(_: &hyprstream_rpc_std::model_client::DecisionResponse) {}
    let response: DecisionResponse = DecisionResponse {
        batch: example_batch(),
        provenance: DecisionProvenance {
            encoding_version: "v".to_owned(),
            model_ref: "m".to_owned(),
            artifact_commit: "0".repeat(40),
            head_weights_sha256: String::new(),
            head_config_sha256: String::new(),
            tokenizer_sha256: String::new(),
            backbone_config_sha256: String::new(),
            quantization: String::new(),
            head_tensor_count: 0,
            compute_profile: "cpu-float32".to_owned(),
        },
        limits: DecisionAppliedLimits {
            max_length: 0,
            max_state_tokens: None,
            state_tokens_admitted: 0,
            sequence_tokens: 0,
            state_truncated: false,
        },
    };
    assert_same_type(&response);

    // And the nested batch type is the canonical decision_client type.
    fn assert_batch_same(_: &decision_client::DecisionBatch) {}
    assert_batch_same(&response.batch);
    let _ = inference_client::DecisionRequest::default();
}

// Exercise the generated production envelopes, rather than duplicating their
// serializers in this regression test.
mod model_server {
    hyprstream_rpc_derive::generate_rpc_server!(
        "model",
        types_crate = hyprstream_rpc_std,
        scope_handlers
    );
}
mod inference_server {
    hyprstream_rpc_derive::generate_rpc_server!("inference", types_crate = hyprstream_rpc_std);
}

fn exact_batch() -> decision_client::DecisionBatch {
    use decision_client::{AnswerValue, QuestionAnswer};
    let answers = vec![
        ("abstained", AnswerValue::Abstained),
        ("choice", AnswerValue::Choice(vec![0.125, 0.875])),
        ("noul", AnswerValue::Noul(0.75)),
        ("score", AnswerValue::Score(vec![0.25, 0.25, 0.5])),
    ]
    .into_iter()
    .map(|(id, value)| QuestionAnswer {
        question_id: id.to_owned(),
        value,
        conformal_set: None,
    })
    .collect();
    decision_client::DecisionBatch {
        version: decision_client::VersionTriple {
            schema: example_schema_id(),
            model: "fixture-model".to_owned(),
            calib: None,
        },
        rows: vec![decision_client::AnswerRow { answers }],
    }
}

fn assert_raw_batch(reader: hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>) {
    use hyprstream_rpc_std::decision_capnp::{answer_value, opt_conformal_set};
    let answers = reader.get_rows().unwrap().get(0).get_answers().unwrap();
    assert_eq!(answers.len(), 4);
    for answer in answers {
        assert!(matches!(
            answer.get_conformal_set().unwrap().which().unwrap(),
            opt_conformal_set::Which::None(())
        ));
        let id = answer.get_question_id().unwrap().to_str().unwrap();
        match (id, answer.get_value().unwrap().which().unwrap()) {
            ("abstained", answer_value::Which::Abstained(())) => {}
            ("noul", answer_value::Which::Noul(v)) => assert_eq!(v.to_bits(), 0.75_f32.to_bits()),
            ("choice", answer_value::Which::Choice(v)) => assert_eq!(
                v.unwrap().iter().map(f32::to_bits).collect::<Vec<_>>(),
                [0.125_f32.to_bits(), 0.875_f32.to_bits()]
            ),
            ("score", answer_value::Which::Score(v)) => assert_eq!(
                v.unwrap().iter().map(f32::to_bits).collect::<Vec<_>>(),
                [0.25_f32.to_bits(), 0.25_f32.to_bits(), 0.5_f32.to_bits()]
            ),
            _ => panic!("wrong answer union arm for {id}"),
        }
    }
}

fn exact_response() -> DecisionResponse {
    DecisionResponse {
        batch: exact_batch(),
        ..DecisionResponse::default()
    }
}

#[test]
fn handwritten_batch_decodes_exact_union_payloads() {
    use hyprstream_decision::{AnswerRow, AnswerValue, QuestionAnswer, VersionTriple};
    let row = AnswerRow {
        answers: [
            ("abstained".to_owned(), QuestionAnswer::abstained()),
            (
                "choice".to_owned(),
                QuestionAnswer::answered(AnswerValue::Choice {
                    probabilities: vec![0.125, 0.875],
                }),
            ),
            (
                "noul".to_owned(),
                QuestionAnswer::answered(AnswerValue::Noul { p_true: 0.75 }),
            ),
            (
                "score".to_owned(),
                QuestionAnswer::answered(AnswerValue::Score {
                    probabilities: vec![0.25, 0.25, 0.5],
                }),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let version = VersionTriple {
        schema: example_schema_id(),
        model: "fixture-model".to_owned(),
        calib: None,
    };
    let msg = decision::batch_to_message(&version, &[row]).unwrap();
    let reader = msg
        .get_root_as_reader::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .unwrap();
    assert_raw_batch(reader);
    assert_eq!(
        decision_client::DecisionBatch::read_from(reader).unwrap(),
        exact_batch()
    );
}

#[test]
fn worker_envelope_preserves_exact_union_payloads() {
    use inference_client::InferenceResponseVariant;
    let bytes = inference_server::serialize_response(
        71,
        &InferenceResponseVariant::DecideResult(exact_response()),
    )
    .unwrap();
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .unwrap();
    let root = read
        .get_root::<hyprstream_rpc_std::inference_capnp::inference_response::Reader<'_>>()
        .unwrap();
    assert_eq!(root.get_request_id(), 71);
    let hyprstream_rpc_std::inference_capnp::inference_response::Which::DecideResult(response) =
        root.which().unwrap()
    else {
        panic!("wrong worker response arm")
    };
    assert_raw_batch(response.unwrap().get_batch().unwrap());
    let InferenceResponseVariant::DecideResult(decoded) =
        InferenceResponseVariant::parse_response(&bytes).unwrap()
    else {
        panic!("wrong decoded worker response")
    };
    assert_eq!(decoded.batch, exact_batch());
}

#[test]
fn model_outer_infer_envelope_preserves_exact_union_payloads() {
    use hyprstream_rpc_std::model_client::{InferResponse, ModelResponseVariant};
    let bytes = model_server::serialize_response(
        72,
        &ModelResponseVariant::InferResult(InferResponse::Decide(exact_response())),
    )
    .unwrap();
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .unwrap();
    let root = read
        .get_root::<hyprstream_rpc_std::model_capnp::model_response::Reader<'_>>()
        .unwrap();
    assert_eq!(root.get_request_id(), 72);
    let hyprstream_rpc_std::model_capnp::model_response::Which::InferResult(infer) =
        root.which().unwrap()
    else {
        panic!("wrong model response arm")
    };
    let hyprstream_rpc_std::model_capnp::infer_response::Which::Decide(response) =
        infer.unwrap().which().unwrap()
    else {
        panic!("wrong model infer response arm")
    };
    assert_raw_batch(response.unwrap().get_batch().unwrap());
    let ModelResponseVariant::InferResult(InferResponse::Decide(decoded)) =
        ModelResponseVariant::parse_response(&bytes).unwrap()
    else {
        panic!("wrong decoded model response")
    };
    assert_eq!(decoded.batch, exact_batch());
}

#[test]
fn generated_question_set_preserves_mixed_union_lists_and_nested_entries() {
    use hyprstream_rpc::capnp::ToCapnp;
    let original = hyprstream_decision::author::parse_yaml(
        r#"
state:
  records:
    - ["first", null, true, 0.5]
    - {nested: "second"}
questions:
  pick:
    type: choice
    instructions: {nested: ["instruction", null]}
    criteria:
      z: {rubric: ["last", true]}
      a: null
  grade:
    type: score
    criteria:
      - {rubric: ["low", 0.25]}
      - null
      - "high"
"#,
    )
    .unwrap();
    let producer = decision::question_set_to_message(&original).unwrap();
    let reader = producer
        .get_root_as_reader::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
        .unwrap();
    // The independent E2 producer must already have the correct kind/body
    // agreement before generated decoding or writing is involved.
    assert_eq!(
        decision::question_set_from_reader(reader).unwrap(),
        original
    );
    let generated = decision_client::QuestionSet::read_from(reader).unwrap();
    let mut consumer = capnp::message::Builder::new_default();
    generated.write_to(
        &mut consumer.init_root::<hyprstream_rpc_std::decision_capnp::question_set::Builder<'_>>(),
    );
    let reader = consumer
        .get_root_as_reader::<hyprstream_rpc_std::decision_capnp::question_set::Reader<'_>>()
        .unwrap();
    assert_eq!(
        decision::question_set_from_reader(reader).unwrap_or_else(|error| {
            panic!("generated schema lost its body: {error}; generated={generated:?}")
        }),
        original
    );
}
