//! Offline official-source fixture comparison. Python is NEVER a serving path.
use super::{ClefHeadConfig, JointSchemaHead};
use crate::clef::{ClefEncoder, ClefEncodingLimits, ClefRequest};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use tch::{Kind, Tensor};

#[derive(Deserialize)]
struct Corpus {
    records: Vec<Record>,
}
#[derive(Deserialize)]
struct Record {
    id: String,
    request: ClefRequest,
    max_length: usize,
    max_state_tokens: Option<usize>,
    input_ids: Vec<i64>,
    questions: Vec<Question>,
    state_tokens_before_truncation: usize,
    state_tokens_admitted: usize,
    state_truncated: bool,
}
#[derive(Deserialize)]
struct Question {
    question_id: String,
    question_type: i64,
    question_span: (usize, usize),
    option_spans: Vec<(usize, usize)>,
    option_ids: Vec<String>,
}
#[derive(Deserialize)]
struct Expected {
    records: Vec<ExpectedRecord>,
}
#[derive(Deserialize)]
struct ExpectedRecord {
    id: String,
    questions: Vec<ExpectedQuestion>,
}
#[derive(Deserialize)]
struct ExpectedQuestion {
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

#[test]
#[ignore = "requires pinned official oracle tensors, head/tokenizer assets, and output directory"]
fn clef_official_encoder_head_parity() {
    tch::set_num_threads(2);
    let external =
        PathBuf::from(std::env::var("CLEF_OFFICIAL_ORACLE_DIR").expect("oracle directory"));
    let model =
        PathBuf::from(std::env::var("CLEF_HEAD_DIRECTORY").expect("original model directory"));
    let output = PathBuf::from(
        std::env::var("CLEF_NATIVE_ORACLE_OUTPUT_DIR").expect("native output directory"),
    );
    std::fs::create_dir_all(&output).expect("create output directory");
    let corpus: Corpus = serde_json::from_str(include_str!(
        "../../tests/fixtures/official-clef-oracle/records.json"
    ))
    .expect("ordered official corpus");
    let expected: Expected = serde_json::from_str(include_str!(
        "../../tests/fixtures/official-clef-oracle/expected.json"
    ))
    .expect("committed official outputs");
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/official-clef-oracle/manifest.json"
    ))
    .expect("fixture pins");
    let provenance: serde_json::Value = serde_json::from_slice(
        &std::fs::read(external.join("provenance.json")).expect("provenance"),
    )
    .expect("provenance JSON");
    assert_eq!(provenance["source_revision"], manifest["source_revision"]);
    assert_eq!(provenance["files"], manifest["files"]);
    assert_eq!(provenance["compute_dtype"], "float32");
    assert_eq!(provenance["mode"], "head");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).expect("official tokenizer");
    let config: ClefHeadConfig = serde_json::from_slice(
        &std::fs::read(model.join("joint_head_config.json")).expect("head config file"),
    )
    .expect("config");
    let weights = Tensor::read_safetensors(model.join("joint_head.safetensors"))
        .expect("original head")
        .into_iter()
        .map(|(name, value)| (name, value.to_kind(Kind::Float)))
        .collect();
    let head = JointSchemaHead::from_named_tensors(config, weights).expect("strict head");
    let inputs: HashMap<_, _> = Tensor::read_safetensors(external.join("tensors.safetensors"))
        .expect("oracle tensors")
        .into_iter()
        .collect();
    let row_ids = Vec::<i64>::try_from(&inputs["embedding_token_ids"]).expect("row IDs");
    let row_map: HashMap<_, _> = row_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as i64))
        .collect();
    let mut emitted = vec![
        (
            "embedding_token_ids".to_owned(),
            inputs["embedding_token_ids"].shallow_clone(),
        ),
        (
            "embedding_rows".to_owned(),
            inputs["embedding_rows"].shallow_clone(),
        ),
    ];
    let mut metrics = Vec::new();
    let mut encoded_records = Vec::new();
    let mut passed = true;
    assert_eq!(corpus.records.len(), expected.records.len());
    for (i, (reference, expected)) in corpus.records.iter().zip(&expected.records).enumerate() {
        assert_eq!(reference.id, expected.id);
        let encoded = ClefEncoder::encode(
            &tokenizer,
            &reference.request.record,
            ClefEncodingLimits {
                max_length: reference.max_length,
                max_state_tokens: reference.max_state_tokens,
            },
        )
        .expect("native encoding");
        assert_eq!(
            encoded.input_ids, reference.input_ids,
            "exact input IDs {}",
            reference.id
        );
        assert_eq!(encoded.record_id, reference.id);
        assert_eq!(encoded.questions.len(), reference.questions.len());
        assert_eq!(
            encoded.state_tokens_before_truncation,
            reference.state_tokens_before_truncation
        );
        assert_eq!(
            encoded.state_tokens_admitted,
            reference.state_tokens_admitted
        );
        assert_eq!(encoded.state_truncated, reference.state_truncated);
        for (native, q) in encoded.questions.iter().zip(&reference.questions) {
            assert_eq!(native.question_id, q.question_id);
            assert_eq!(native.kind.type_id(), q.question_type);
            assert_eq!(native.question_span, q.question_span);
            assert_eq!(native.option_spans, q.option_spans);
            assert_eq!(native.option_ids, q.option_ids);
        }
        let hidden = &inputs[&format!("r{i}.hidden")];
        let logits = tch::no_grad(|| {
            head.forward(hidden, &encoded, |ids| {
                let local: Vec<_> = ids
                    .iter()
                    .map(|id| *row_map.get(id).expect("original row coverage"))
                    .collect();
                Ok(inputs["embedding_rows"].index_select(0, &Tensor::from_slice(&local)))
            })
        })
        .expect("native joint head");
        emitted.push((format!("r{i}.hidden"), hidden.shallow_clone()));
        let normalized = tch::no_grad(|| head.norm(hidden, "hidden_norm"));
        let normalization_max_abs = (&normalized - &inputs[&format!("r{i}.normalized_hidden")])
            .abs()
            .max()
            .double_value(&[]);
        emitted.push((format!("r{i}.normalized_hidden"), normalized));
        assert_eq!(logits.len(), expected.questions.len());
        let mut question_metrics = Vec::new();
        for (j, (actual, expected)) in logits.iter().zip(&expected.questions).enumerate() {
            let probabilities = actual.softmax(-1, Kind::Float);
            let logit_error = (actual - Tensor::from_slice(&expected.logits))
                .abs()
                .max()
                .double_value(&[]);
            let probability_error = (&probabilities - Tensor::from_slice(&expected.probabilities))
                .abs()
                .max()
                .double_value(&[]);
            // Fixed by coordinator BEFORE comparing: no post-hoc tolerance changes.
            let within = logit_error <= 1e-4 && probability_error <= 1e-5;
            passed &= within;
            question_metrics.push(serde_json::json!({"index":j,"pass":within,"logit_max_absolute_error":logit_error,"probability_max_absolute_error":probability_error}));
            emitted.push((format!("r{i}.q{j}.logits"), actual.shallow_clone()));
            emitted.push((format!("r{i}.q{j}.probabilities"), probabilities));
        }
        metrics.push(serde_json::json!({"id":reference.id,"normalization_max_absolute_error":normalization_max_abs,"questions":question_metrics}));
        encoded_records.push(serde_json::json!({"id":encoded.record_id,"input_ids":encoded.input_ids,"questions":encoded.questions.iter().map(|q|serde_json::json!({"question_id":q.question_id,"question_type":q.kind.type_id(),"question_span":q.question_span,"option_spans":q.option_spans,"option_ids":q.option_ids})).collect::<Vec<_>>(),"state_tokens_before_truncation":encoded.state_tokens_before_truncation,"state_tokens_admitted":encoded.state_tokens_admitted,"state_truncated":encoded.state_truncated}));
    }
    Tensor::write_safetensors(&emitted, output.join("native-tensors.safetensors"))
        .expect("export native outputs");
    std::fs::write(
        output.join("native-records.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"records":encoded_records}))
            .expect("record JSON"),
    )
    .expect("write records");
    std::fs::write(output.join("metrics.json"),serde_json::to_vec_pretty(&serde_json::json!({"pass":passed,"logit_absolute_tolerance":1e-4,"probability_absolute_tolerance":1e-5,"records":metrics,"scope":"official head on synthetic hidden, not backbone or R5"})).expect("metric JSON")).expect("write metrics");
    assert!(passed,"official head parity failed; inspect saved per-question metrics before changing any tolerance");
}
