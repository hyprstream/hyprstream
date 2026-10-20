//! Service-facing typed decision serving glue (R4).
//!
//! Composition point between the authenticated InferenceService/ModelService
//! dispatch and the native Clef runtime in
//! [`crate::runtime::clef_decision`]. This module owns:
//!
//! - the SHARED cheap preflight contract used at BOTH layers (Model ingress,
//!   BEFORE `get_inference_client`/`load_model`, and the worker after its
//!   authorization PEP) so invalid media/duplicate/oversized/aggregate-
//!   blowing bodies can never allocate a backbone;
//! - server-controlled budget resolution (state-token cap, deadline clamp,
//!   sequence budget from the admitted engine context) — callers may lower,
//!   never raise;
//! - the pinned-artifact requirement for decision serving and the immutable
//!   provenance assembly (commit OID + content digests, never names/echoes);
//! - the versioned question-schema identity (a content hash over the
//!   canonical question set EXCLUDING state, so label/type/instruction
//!   changes change identity while state-only changes do not);
//! - the RESOLVED versioned model id for `DecisionBatch.version.model`
//!   (authorized ref @ pinned commit — never a bare branch alias);
//! - IR answer assembly and E2 `decision.capnp` marshalling
//!   (`batch_to_message`, producer-validated) into the generated
//!   `decision_client` owned types for the wire response. Choice
//!   distributions arrive in sorted canonical order and are remapped BY
//!   LABEL into the question spec's authored option order (E2 D6: answer
//!   vectors are unlabeled; consumers derive labels from the spec) in
//!   [`build_answer_row`].
//!
//! Wire-decoding note (accurate scope): the generated dispatch performs a
//! BOUNDED generic Cap'n Proto decode of the request body (including the
//! `clefJson` String) BEFORE authorize — existing behavior for every method,
//! capped by the generic traversal limits. Everything expensive and Clef-
//! specific (ordered JSON parse, per-question + aggregate admission,
//! tokenization, head/backbone work) happens only after the verified PEP and
//! the raw-byte bound in [`preflight_parsed`].

use crate::runtime::clef_decision::{
    self as clef, ClefDecisionAccounting, ClefDecisionError, ClefDecisionRuntime,
    ClefQuestionDistribution, DecisionCancel, DEFAULT_CLEF_MAX_LENGTH,
};
use hyprstream_decision::{AnswerRow, AnswerValue, QuestionAnswer, VersionTriple};
use hyprstream_decision_head::clef::{
    clef_label_permutation, clef_render, ClefQuestion, ClefQuestionType, ClefRequest,
    CLEF_ENCODING_VERSION,
};
use hyprstream_rpc_std::decision_client;
use hyprstream_rpc_std::inference_client::{
    DecisionAppliedLimits, DecisionProvenance, DecisionRequest, DecisionResponse,
};
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Hard server maximum for the per-request decision budget.
pub const MAX_DECISION_TIMEOUT_MS: u64 = 600_000;
/// Server default/cap for the caller's state-token budget (callers may lower).
pub use crate::runtime::clef_decision::DEFAULT_MAX_STATE_TOKENS;

/// Service-layer typed failures. Messages carry stable
/// `decision-*`/`clef-decision-*` prefixes; the adapter's ErrorInfo mapping
/// ([`error_code_for`]) and tests pin them.
#[derive(Debug, thiserror::Error)]
pub enum DecisionServeError {
    #[error("decision-not-pinned: {0}")]
    NotPinned(String),
    #[error("decision-model-mismatch: {0}")]
    ModelMismatch(String),
    #[error("decision-model-unbound: {0}")]
    ModelUnbound(String),
    #[error("decision-no-tokenizer: {0}")]
    NoTokenizer(String),
    #[error("decision-marshal: {0}")]
    Marshal(String),
    #[error(transparent)]
    Clef(#[from] ClefDecisionError),
}

impl DecisionServeError {
    /// Deterministic wire error code for the ErrorInfo mapping.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotPinned(_) => "DECISION_NOT_PINNED",
            Self::ModelMismatch(_) => "DECISION_MODEL_MISMATCH",
            Self::ModelUnbound(_) => "DECISION_MODEL_UNBOUND",
            Self::NoTokenizer(_) => "DECISION_NO_TOKENIZER",
            Self::Marshal(_) => "DECISION_MARSHAL",
            Self::Clef(ClefDecisionError::Cancelled(_)) => "DECISION_CANCELLED",
            Self::Clef(ClefDecisionError::Input(_)) => "DECISION_INPUT",
            Self::Clef(ClefDecisionError::Bundle(_)) => "DECISION_BUNDLE",
            Self::Clef(ClefDecisionError::Profile(_)) => "DECISION_PROFILE",
            Self::Clef(ClefDecisionError::Runtime(_)) => "DECISION_RUNTIME",
        }
    }
}

/// Wire error code for a dispatched error message. The generated dispatch
/// erases the error TYPE down to its display string before the adapter builds
/// ErrorInfo, so typed decision codes are recovered from the pinned message
/// prefixes here. Unknown prefixes keep the generic INTERNAL code.
pub fn error_code_for(message: &str) -> Option<&'static str> {
    if message.starts_with("decision-not-pinned") {
        Some("DECISION_NOT_PINNED")
    } else if message.starts_with("decision-model-mismatch") {
        Some("DECISION_MODEL_MISMATCH")
    } else if message.starts_with("decision-model-unbound") {
        Some("DECISION_MODEL_UNBOUND")
    } else if message.starts_with("decision-no-tokenizer") {
        Some("DECISION_NO_TOKENIZER")
    } else if message.starts_with("decision-marshal") {
        Some("DECISION_MARSHAL")
    } else if message.starts_with("clef-decision-cancelled") {
        Some("DECISION_CANCELLED")
    } else if message.starts_with("clef-decision-input") {
        Some("DECISION_INPUT")
    } else if message.starts_with("clef-decision-bundle") {
        Some("DECISION_BUNDLE")
    } else if message.starts_with("clef-decision-profile") {
        Some("DECISION_PROFILE")
    } else if message.starts_with("clef-decision-runtime") {
        Some("DECISION_RUNTIME")
    } else {
        None
    }
}

/// Shared cheap preflight over the raw wire request: byte bound, ordered
/// Clef JSON parse (duplicate fields/IDs, media, numeric domain), per-
/// question cardinalities, aggregate option/attention budgets, deadline
/// clamp. Returns the parsed request for reuse (model match, schema
/// identity, decide). Runs at Model ingress BEFORE any model load, and
/// again at the worker after its PEP. Tokenization and tensor work never
/// happen here.
pub fn preflight_parsed(data: &DecisionRequest) -> Result<ClefRequest, DecisionServeError> {
    let parsed = clef::preflight_body(&data.clef_json)?;
    if let Some(timeout) = data.timeout_ms {
        if u64::from(timeout) > MAX_DECISION_TIMEOUT_MS {
            return Err(DecisionServeError::Clef(ClefDecisionError::Input(format!(
                "timeoutMs {timeout} exceeds server maximum {MAX_DECISION_TIMEOUT_MS}"
            ))));
        }
    }
    Ok(parsed)
}

/// Effective state-token cap: caller value may only LOWER the server default.
pub fn resolve_state_cap(data: &DecisionRequest) -> Option<usize> {
    data.max_state_tokens
        .map(|v| v.min(DEFAULT_MAX_STATE_TOKENS) as usize)
        .or(Some(DEFAULT_MAX_STATE_TOKENS as usize))
}

/// Effective deadline: caller timeout clamped by the hard server maximum.
pub fn resolve_deadline(data: &DecisionRequest, now: Instant) -> Option<Instant> {
    let ms = data.timeout_ms.map_or(MAX_DECISION_TIMEOUT_MS, |ms| {
        u64::from(ms).min(MAX_DECISION_TIMEOUT_MS)
    });
    Some(now + Duration::from_millis(ms))
}

/// Return the remaining wire timeout for a request admitted at Model ingress.
/// The relative timeout sent to Inference must account for time spent loading
/// and routing the model, rather than starting the caller's full budget again.
pub fn remaining_timeout_ms(deadline: Instant, now: Instant) -> Result<u32, DecisionServeError> {
    let remaining_ms = deadline.saturating_duration_since(now).as_millis();
    if remaining_ms == 0 {
        return Err(DecisionServeError::Clef(ClefDecisionError::Cancelled(
            "decision deadline elapsed before inference dispatch".to_owned(),
        )));
    }
    Ok(u32::try_from(remaining_ms).unwrap_or(u32::MAX))
}

/// Effective total sequence budget: min(reference default, engine context).
pub fn resolve_max_length(engine_context_length: u32) -> usize {
    DEFAULT_CLEF_MAX_LENGTH.min(engine_context_length as usize)
}

/// The raw request `model` must equal the authorized model reference (there
/// is no validated alias mapping in this milestone). FAILS CLOSED when no
/// reference is bound: decision serving requires an explicit admitted
/// identity.
pub fn ensure_model_matches(
    request_model: &str,
    bound_model_ref: Option<&str>,
) -> Result<(), DecisionServeError> {
    let bound = bound_model_ref.ok_or_else(|| {
        DecisionServeError::ModelUnbound(
            "worker has no admitted model binding; refusing decision".into(),
        )
    })?;
    if bound.trim().is_empty() {
        return Err(DecisionServeError::ModelUnbound(
            "worker model binding is empty; refusing decision".into(),
        ));
    }
    if request_model != bound {
        return Err(DecisionServeError::ModelMismatch(format!(
            "request model {request_model:?} does not match admitted model {bound:?}"
        )));
    }
    Ok(())
}

/// Model-ingress pin gate: decision serving admits only refs that resolve to
/// an immutable identity — a commit ref, or the exact staging-pinned ref.
/// Plain branch/tag/revspec refs select mutable worktrees and are rejected
/// BEFORE the backbone loads (the worker re-requires the retained pinned
/// artifact).
pub fn ensure_pinned_for_decision(
    is_commit_ref: bool,
    staging_pin_admits: bool,
    model_ref: &str,
) -> Result<(), DecisionServeError> {
    if is_commit_ref || staging_pin_admits {
        Ok(())
    } else {
        Err(DecisionServeError::NotPinned(format!(
            "model {model_ref:?} resolves to a mutable ref; decision serving requires a pinned commit"
        )))
    }
}

/// Hash one question's canonical schema content with PER-FIELD length
/// framing. Every atomic unit (type tag, option count, option id, rendered
/// rubric, score slot) is framed independently — no delimiter joins exist,
/// so ids/rubrics containing '=' or NUL cannot collide across field splits
/// (S10: `a=b`+`c` vs `a`+`b=c`).
fn hash_question_schema(hasher: &mut Sha256, question: &ClefQuestion) {
    // ClefQuestionType::name is private upstream; tag locally (must stay in
    // sync with the wire tags noul/choice/score).
    let kind_tag = match question.kind {
        ClefQuestionType::Noul => "noul",
        ClefQuestionType::Choice => "choice",
        ClefQuestionType::Score => "score",
    };
    hash_field(hasher, kind_tag.as_bytes());
    // EVERY body arm first frames its ENTRY COUNT, so the exact number of
    // framed fields that follow is determined by the preceding count. A
    // trailing question's fields can never masquerade as an earlier
    // question's body entries (S10 structural collision).
    match question.kind {
        ClefQuestionType::Noul => {
            let overrides: Vec<(&str, &serde_json::Value)> = question
                .criteria
                .as_object()
                .map(|map| {
                    ["false", "true"]
                        .iter()
                        .filter_map(|id| map.get(*id).map(|v| (*id, v)))
                        .collect()
                })
                .unwrap_or_default();
            hash_field(hasher, &(overrides.len() as u64).to_le_bytes());
            for (id, v) in overrides {
                hash_field(hasher, id.as_bytes());
                hash_field(hasher, clef_render(v).as_bytes());
            }
        }
        ClefQuestionType::Choice => {
            let map = question.criteria.as_object();
            hash_field(hasher, &map.map_or(0, |m| m.len() as u64).to_le_bytes());
            if let Some(map) = map {
                let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                for (id, v) in entries {
                    hash_field(hasher, id.as_bytes());
                    hash_field(hasher, clef_render(v).as_bytes());
                }
            }
        }
        ClefQuestionType::Score => {
            let arr = question.criteria.as_array();
            hash_field(hasher, &arr.map_or(0, |a| a.len() as u64).to_le_bytes());
            if let Some(arr) = arr {
                for v in arr {
                    hash_field(hasher, clef_render(v).as_bytes());
                }
            }
        }
    }
}

/// Length-prefixed field framing for [`question_schema_identity`]: JSON
/// strings admit embedded NUL (U+0000), so bare delimiters are ambiguous
/// (id="a", instructions="b<NUL>c" vs id="a<NUL>b", instructions="c"). A
/// fixed-width little-endian length before each byte field is prefix-free
/// and collision-free.
fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Versioned question-schema identity: a content hash over the canonical
/// question set EXCLUDING state (author question order, ids, kinds,
/// instructions, and full option label/description sets). Changing any
/// question, type, option label set, or instructions changes the identity;
/// changing only the state does not. The encoding contract version is
/// reported separately in DecisionProvenance. The FULL digest is carried.
pub fn question_schema_identity(request: &ClefRequest) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, CLEF_ENCODING_VERSION.as_bytes());
    // The QUESTION COUNT is framed first: the identity stream is a
    // self-describing sequence (count, then per question: id, instructions,
    // type tag, body-entry count, entries), so no split of any byte string
    // across fields can reproduce another set's stream.
    hash_field(
        &mut hasher,
        &(request.record.questions.len() as u64).to_le_bytes(),
    );
    for (id, q) in &request.record.questions {
        hash_field(&mut hasher, id.as_bytes());
        hash_field(&mut hasher, clef_render(&q.instructions).as_bytes());
        hash_question_schema(&mut hasher, q);
    }
    let digest = hex::encode(hasher.finalize());
    format!("clef-raw-{CLEF_ENCODING_VERSION}-q{digest}")
}

/// Version triple for the decision batch. `model` is the CANONICAL RESOLVED
/// VERSIONED model id — resolved model NAME @ actual retained commit — never
/// a bare branch alias: a staging/branch alias binding still emits
/// name@commit, so eval/persistence key on the immutable identity. The
/// authorization reference stays in DecisionProvenance.modelRef. `calib` is
/// explicitly None (uncalibrated native distributions: no conformal or
/// confidence claims).
pub fn version_triple(
    request: &ClefRequest,
    model_ref: &str,
    artifact_commit: &str,
) -> VersionTriple {
    let name = model_ref.split(':').next().unwrap_or(model_ref);
    VersionTriple {
        schema: question_schema_identity(request),
        model: format!("{name}@{artifact_commit}"),
        calib: None,
    }
}

/// Assemble the IR answer row from canonical distributions. Every declared
/// question is answered (this runtime never abstains: raw native output).
///
/// ORDER CONTRACT (E2 D6): the runtime emits choice distributions in sorted
/// canonical order (the model's own prompt order), but E2 answer vectors are
/// UNLABELED and canonical in the question spec's AUTHORED option order —
/// consumers derive labels from the spec. Choice probabilities are remapped
/// BY LABEL into the authored order here, at the service boundary; noul
/// ([false,true]) and score (ascending array positions) authored orders
/// equal their canonical orders and pass through unchanged.
pub fn build_answer_row(
    distributions: &[ClefQuestionDistribution],
    request: &ClefRequest,
) -> Result<AnswerRow, DecisionServeError> {
    let mut answers = std::collections::BTreeMap::new();
    for (index, d) in distributions.iter().enumerate() {
        let (authored_id, question) = request.record.questions.get(index).ok_or_else(|| {
            DecisionServeError::Marshal(format!(
                "distribution {} ({}) has no matching authored question",
                index, d.question_id
            ))
        })?;
        if authored_id != &d.question_id {
            return Err(DecisionServeError::Marshal(format!(
                "question {}: distribution does not match authored question {authored_id} at index {index}",
                d.question_id
            )));
        }
        if question.kind != d.kind {
            return Err(DecisionServeError::Marshal(format!(
                "question {}: distribution kind {:?} != authored kind {:?}",
                d.question_id, d.kind, question.kind
            )));
        }
        if d.probabilities.len() != d.canonical_ids.len() {
            return Err(DecisionServeError::Clef(ClefDecisionError::Runtime(
                format!(
                    "question {}: distribution cardinality {} != {} canonical labels",
                    d.question_id,
                    d.probabilities.len(),
                    d.canonical_ids.len()
                ),
            )));
        }
        let value = match d.kind {
            // Canonical order is [false,true]: p(true) is component 1.
            ClefQuestionType::Noul => AnswerValue::Noul {
                p_true: d.probabilities[1],
            },
            ClefQuestionType::Choice => {
                // Remap sorted-canonical probabilities into the authored
                // criteria order by label (fail-closed permutation check).
                let authored = clef::authored_labels(question)?;
                let permutation = clef_label_permutation(&d.canonical_ids, &authored)
                    .map_err(ClefDecisionError::from)?;
                AnswerValue::Choice {
                    probabilities: permutation
                        .iter()
                        .map(|&i| d.probabilities[i as usize])
                        .collect(),
                }
            }
            ClefQuestionType::Score => AnswerValue::Score {
                probabilities: d.probabilities.clone(),
            },
        };
        answers.insert(
            d.question_id.clone(),
            QuestionAnswer::answered(value), // no conformal set, ever
        );
    }
    Ok(AnswerRow { answers })
}

/// Marshal through the E2 producer path (`batch_to_message` — enforces the
/// producer distribution/abstention/cardinality contract) and read it back
/// into the canonical generated owned type.
pub fn marshal_batch(
    version: &VersionTriple,
    rows: &[AnswerRow],
) -> Result<decision_client::DecisionBatch, DecisionServeError> {
    let message = hyprstream_rpc_std::decision::batch_to_message(version, rows)
        .map_err(|e| DecisionServeError::Marshal(format!("batch encode: {e}")))?;
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &message)
        .map_err(|e| DecisionServeError::Marshal(format!("batch serialize: {e}")))?;
    let read =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .map_err(|e| DecisionServeError::Marshal(format!("batch reread: {e}")))?;
    let reader = read
        .get_root::<hyprstream_rpc_std::decision_capnp::decision_batch::Reader<'_>>()
        .map_err(|e| DecisionServeError::Marshal(format!("batch root: {e}")))?;
    use hyprstream_rpc::FromCapnp as _;
    decision_client::DecisionBatch::read_from(reader)
        .map_err(|e| DecisionServeError::Marshal(format!("batch read: {e}")))
}

/// Assemble the wire response: typed batch + observed immutable provenance +
/// exact applied-limits accounting. `model_ref` is the ADMITTED worker model
/// reference (non-empty; the worker fails closed without a binding), and
/// `artifact_commit` the pinned artifact OID the batch version already
/// carries.
#[allow(clippy::too_many_arguments)]
pub fn build_response(
    runtime: &ClefDecisionRuntime,
    model_ref: &str,
    artifact_commit: &str,
    accounting: &ClefDecisionAccounting,
    observed_kind: crate::runtime::clef_decision::ComputeKind,
    batch: decision_client::DecisionBatch,
) -> DecisionResponse {
    let identity = &runtime.identity;
    DecisionResponse {
        batch,
        provenance: DecisionProvenance {
            encoding_version: runtime.encoding_version.to_owned(),
            // OBSERVED compute profile: derived from the actual successful
            // forward's hidden kind (validated == admitted head kind inside
            // decide_encoded) + device/path; artifact quantization stays in
            // the separate quantization field.
            compute_profile: crate::runtime::clef_decision::compute_profile_string(
                observed_kind,
                runtime.channel_fp8,
            ),
            model_ref: model_ref.to_owned(),
            artifact_commit: artifact_commit.to_owned(),
            head_weights_sha256: identity.head_weights_sha256.clone(),
            head_config_sha256: identity.head_config_sha256.clone(),
            tokenizer_sha256: identity.tokenizer_sha256.clone().unwrap_or_default(),
            backbone_config_sha256: identity.backbone_config_sha256.clone().unwrap_or_default(),
            quantization: identity.quantization.clone().unwrap_or_default(),
            head_tensor_count: identity.head_tensor_count,
        },
        limits: DecisionAppliedLimits {
            max_length: accounting.max_length as u32,
            max_state_tokens: accounting.max_state_tokens_effective.map(|v| v as u32),
            state_tokens_admitted: accounting.state_tokens_admitted as u32,
            sequence_tokens: accounting.sequence_tokens as u32,
            state_truncated: accounting.state_truncated,
        },
    }
}

/// Build the worker-side cooperative cancel checker: service drain flag plus
/// the resolved deadline.
pub fn worker_cancel(drain: Arc<AtomicBool>, deadline: Option<Instant>) -> DecisionCancel {
    DecisionCancel::new(drain, deadline)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // fixture assertions

    use super::*;
    use hyprstream_rpc_std::inference_client::DecisionRequest as WireDecisionRequest;

    #[test]
    fn remaining_timeout_ms_decreases_and_expires_at_ingress_deadline() {
        let start = Instant::now();
        let request = WireDecisionRequest {
            clef_json: String::new(),
            max_state_tokens: None,
            timeout_ms: Some(1_000),
        };
        let deadline = resolve_deadline(&request, start).expect("decision deadline is bounded");

        assert_eq!(remaining_timeout_ms(deadline, start).unwrap(), 1_000);
        assert_eq!(
            remaining_timeout_ms(deadline, start + Duration::from_millis(375)).unwrap(),
            625
        );
        let err = remaining_timeout_ms(deadline, deadline).unwrap_err();
        assert!(err.to_string().contains("clef-decision-cancelled"));

        let zero_timeout = WireDecisionRequest {
            clef_json: String::new(),
            max_state_tokens: None,
            timeout_ms: Some(0),
        };
        let zero_deadline = resolve_deadline(&zero_timeout, start).expect("bounded deadline");
        let err = remaining_timeout_ms(zero_deadline, start).unwrap_err();
        assert!(err.to_string().contains("clef-decision-cancelled"));
    }

    fn parsed(body: &str) -> ClefRequest {
        let wire = WireDecisionRequest {
            clef_json: body.to_owned(),
            max_state_tokens: None,
            timeout_ms: None,
        };
        preflight_parsed(&wire).expect("preflight")
    }

    #[test]
    fn schema_identity_excludes_state_and_orders_questions() {
        let base = parsed(
            r#"{"model":"m","state":{"a":1},"questions":{
                "z":{"type":"choice","criteria":{"b":"x","a":"y"}},
                "n":{"type":"noul","instructions":"Ask"},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        let identity = question_schema_identity(&base);

        // State-only change keeps identity.
        let state_changed = parsed(
            r#"{"model":"m","state":{"totally":"different"},"questions":{
                "z":{"type":"choice","criteria":{"b":"x","a":"y"}},
                "n":{"type":"noul","instructions":"Ask"},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        assert_eq!(identity, question_schema_identity(&state_changed));

        // Renaming an option changes identity.
        let renamed = parsed(
            r#"{"model":"m","state":{"a":1},"questions":{
                "z":{"type":"choice","criteria":{"b":"x","c":"y"}},
                "n":{"type":"noul","instructions":"Ask"},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        assert_ne!(identity, question_schema_identity(&renamed));

        // Changing instructions changes identity.
        let instructed = parsed(
            r#"{"model":"m","state":{"a":1},"questions":{
                "z":{"type":"choice","criteria":{"b":"x","a":"y"}},
                "n":{"type":"noul","instructions":"Other"},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        assert_ne!(identity, question_schema_identity(&instructed));

        // Reordering questions changes identity (order reaches the model).
        let reordered = parsed(
            r#"{"model":"m","state":{"a":1},"questions":{
                "n":{"type":"noul","instructions":"Ask"},
                "z":{"type":"choice","criteria":{"b":"x","a":"y"}},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        assert_ne!(identity, question_schema_identity(&reordered));

        // Choice label order in the AUTHOR body does not change identity
        // (canonical sorted rendering), but descriptions do.
        let reordered_criteria = parsed(
            r#"{"model":"m","state":{"a":1},"questions":{
                "z":{"type":"choice","criteria":{"a":"y","b":"x"}},
                "n":{"type":"noul","instructions":"Ask"},"q2":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        assert_eq!(identity, question_schema_identity(&reordered_criteria));
    }

    #[test]
    fn answer_row_aligns_choice_probabilities_with_authored_spec_order() {
        // Authored order is deliberately non-lexicographic: z, a, b. The
        // runtime distribution is in sorted canonical order (a, b, z — the
        // model/prompt parity order). E2 answer vectors are unlabeled, so
        // they must align with the AUTHORED spec order.
        let request = parsed(
            r#"{"model":"m","state":null,"questions":{
                "pick":{"type":"choice","criteria":{"z":"third","a":"first","b":"second"}}}}"#,
        );
        let distribution = |probabilities: Vec<f32>| ClefQuestionDistribution {
            question_id: "pick".into(),
            kind: ClefQuestionType::Choice,
            canonical_ids: vec!["a".into(), "b".into(), "z".into()],
            probabilities,
        };

        // Winner scan over the unlabeled vector. E2 D6 breaks argmax ties
        // toward the EARLIEST authored option, and `Iterator::max_by`
        // returns the LAST equal maximum — so the winner is computed with an
        // explicit first-max scan, the contract consumers implement.
        let first_argmax = |probs: &[f32]| -> usize {
            let mut best = 0;
            for (index, p) in probs.iter().enumerate() {
                if *p > probs[best] {
                    best = index;
                }
            }
            best
        };

        // Distinct probabilities with the maximum on "a": the sorted vector
        // [0.5, 0.3, 0.2] would put the winner at index 0, which a consumer
        // reading the authored spec (z, a, b) would attribute to "z".
        let row = build_answer_row(&[distribution(vec![0.5, 0.3, 0.2])], &request).unwrap();
        let value = row.answers["pick"].value.as_ref().unwrap();
        let AnswerValue::Choice { probabilities } = value else {
            panic!("choice answer expected")
        };
        // Authored positions: z=0 (third-authored), a=1 (first), b=2.
        assert_eq!(probabilities, &[0.2, 0.5, 0.3]);
        assert_eq!(
            first_argmax(probabilities),
            1,
            "winner must be authored option 'a' at index 1"
        );

        // Tie between "z" and "a": identical maxima stay attached to their
        // authored positions, so the earliest-authored winner is "z"
        // (index 0) — the sorted vector's index-0 maximum would misattribute
        // the tie-break to "a".
        let row = build_answer_row(&[distribution(vec![0.4, 0.2, 0.4])], &request).unwrap();
        let value = row.answers["pick"].value.as_ref().unwrap();
        let AnswerValue::Choice { probabilities } = value else {
            panic!("choice answer expected")
        };
        // Authored positions: z=0.4 at 0, a=0.4 at 1, b=0.2 at 2.
        assert_eq!(probabilities, &[0.4, 0.4, 0.2]);
        assert_eq!(
            first_argmax(probabilities),
            0,
            "tie-break winner must be earliest-authored 'z'"
        );

        // Noul and score authored orders equal their canonical orders and
        // pass through unchanged (p(true) still component 1 of [false,true]).
        let request = parsed(
            r#"{"model":"m","state":null,"questions":{
                "n":{"type":"noul"},"s":{"type":"score","criteria":["lo","hi"]}}}"#,
        );
        let distributions = vec![
            ClefQuestionDistribution {
                question_id: "n".into(),
                kind: ClefQuestionType::Noul,
                canonical_ids: vec!["false".into(), "true".into()],
                probabilities: vec![0.25, 0.75],
            },
            ClefQuestionDistribution {
                question_id: "s".into(),
                kind: ClefQuestionType::Score,
                canonical_ids: vec!["0".into(), "1".into()],
                probabilities: vec![0.6, 0.4],
            },
        ];
        let row = build_answer_row(&distributions, &request).unwrap();
        let noul = row.answers["n"].value.as_ref().unwrap();
        assert!(matches!(
            noul,
            AnswerValue::Noul { p_true } if (*p_true - 0.75).abs() < f32::EPSILON
        ));
        let score = row.answers["s"].value.as_ref().unwrap();
        assert!(matches!(
            score,
            AnswerValue::Score { probabilities } if *probabilities == vec![0.6, 0.4]
        ));

        // A drifted runtime distribution (cardinality mismatch) fails closed
        // instead of mis-ordering or truncating the vector.
        let err = build_answer_row(
            &[ClefQuestionDistribution {
                question_id: "pick".into(),
                kind: ClefQuestionType::Choice,
                canonical_ids: vec!["a".into(), "b".into(), "z".into()],
                probabilities: vec![1.0],
            }],
            &parsed(
                r#"{"model":"m","state":null,"questions":{
                    "pick":{"type":"choice","criteria":{"z":"third","a":"first","b":"second"}}}}"#,
            ),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("cardinality"), "{err}");
    }

    #[test]
    fn version_triple_carries_resolved_versioned_model_and_null_calib() {
        let base = parsed(r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}}}"#);
        // Branch alias binding resolves to name@commit, not the alias.
        let v = version_triple(&base, "clef:main", "abc123");
        assert_eq!(v.model, "clef@abc123");
        assert!(v.calib.is_none());
        assert!(v.schema.starts_with("clef-raw-"));
        assert!(v.schema.contains("-q"));
    }

    #[test]
    fn schema_identity_is_collision_free_against_embedded_nul() {
        // JSON escapes can carry U+0000 inside ids/instructions. Splitting
        // the same byte stream across fields must NOT collide under the
        // length-prefixed framing.
        let first = parsed(
            r#"{"model":"m","state":null,"questions":{"a":{"type":"noul","instructions":"b\u0000c"}}}"#,
        );
        let second = parsed(
            r#"{"model":"m","state":null,"questions":{"a\u0000b":{"type":"noul","instructions":"c"}}}"#,
        );
        assert_ne!(
            question_schema_identity(&first),
            question_schema_identity(&second)
        );
        assert_eq!(first.record.questions.len(), 1);
        assert_eq!(second.record.questions.len(), 1);
    }

    #[test]
    fn schema_identity_is_collision_free_against_delimiter_splits() {
        // S10: choice id "a=b" with rubric "c" vs choice id "a" with rubric
        // "b=c" — per-field framing must distinguish them.
        let first = parsed(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"choice","criteria":{"a=b":"c","x":"y"}}}}"#,
        );
        let second = parsed(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"choice","criteria":{"a":"b=c","x":"y"}}}}"#,
        );
        assert_ne!(
            question_schema_identity(&first),
            question_schema_identity(&second)
        );
        // S10 structural pair: A's noul OVERRIDE fields must not be
        // reproducible by B's trailing question fields.
        let a = parsed(
            r#"{"model":"m","state":null,"questions":{
                "q":{"type":"noul","criteria":{"false":"v"}},
                "noul":{"type":"noul","instructions":"false"}}}"#,
        );
        let b = parsed(
            r#"{"model":"m","state":null,"questions":{
                "q":{"type":"noul"},
                "false":{"type":"noul","instructions":"v","criteria":{"false":null}}}}"#,
        );
        assert_ne!(question_schema_identity(&a), question_schema_identity(&b));

        // Same split inside score rubrics and noul overrides is equally safe.
        let third = parsed(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"score","criteria":["a=b","c"]}}}"#,
        );
        let fourth = parsed(
            r#"{"model":"m","state":null,"questions":{"q":{"type":"score","criteria":["a","b=c"]}}}"#,
        );
        assert_ne!(
            question_schema_identity(&third),
            question_schema_identity(&fourth)
        );
    }

    #[test]
    fn model_match_fails_closed_unbound() {
        assert!(ensure_model_matches("m", Some("m")).is_ok());
        let err = ensure_model_matches("m", None).unwrap_err().to_string();
        assert!(err.starts_with("decision-model-unbound"), "{err}");
        // Empty/whitespace bindings fail closed too (T7).
        for empty in ["", "  "] {
            let err = ensure_model_matches("m", Some(empty))
                .unwrap_err()
                .to_string();
            assert!(err.starts_with("decision-model-unbound"), "{err}");
        }
        let err = ensure_model_matches("m", Some("other"))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("decision-model-mismatch"), "{err}");
    }

    #[test]
    fn ingress_pin_gate_admits_only_immutable_refs() {
        assert!(ensure_pinned_for_decision(true, false, "m@oid").is_ok());
        assert!(ensure_pinned_for_decision(false, true, "staged").is_ok());
        let err = ensure_pinned_for_decision(false, false, "m:main")
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("decision-not-pinned"), "{err}");
    }

    #[test]
    fn typed_error_codes_recover_from_message_prefixes() {
        assert_eq!(
            error_code_for("decision-not-pinned: x"),
            Some("DECISION_NOT_PINNED")
        );
        assert_eq!(
            error_code_for("clef-decision-input: y"),
            Some("DECISION_INPUT")
        );
        assert_eq!(
            error_code_for("clef-decision-cancelled: z"),
            Some("DECISION_CANCELLED")
        );
        assert_eq!(error_code_for("unrelated failure"), None);
    }

    #[test]
    fn budgets_clamp_caller_values() {
        let now = Instant::now();
        let wire = WireDecisionRequest {
            clef_json: r#"{"model":"m","state":null,"questions":{"q":{"type":"noul"}}}"#.into(),
            max_state_tokens: Some(4_000_000),
            timeout_ms: Some(9_000_000),
        };
        assert_eq!(
            resolve_state_cap(&wire),
            Some(DEFAULT_MAX_STATE_TOKENS as usize)
        );
        let deadline = resolve_deadline(&wire, now).unwrap();
        assert!(deadline <= now + Duration::from_millis(MAX_DECISION_TIMEOUT_MS));
        // Preflight rejects the over-cap deadline with a typed input error.
        let err = preflight_parsed(&wire).unwrap_err().to_string();
        assert!(err.contains("exceeds server maximum"), "{err}");
    }
}
