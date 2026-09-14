//! Real stream lifecycle regressions for the PR1522 → PR1520 integration.
//! Only the proposed draft is controlled; verification, emission, Drop, cache
//! selection, prefill and subsequent sampling all use production code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::runtime::architectures::qwen3_5::mtp_tests::whole_model_mtp;
use crate::runtime::kv_cache::{CacheOwner, KVCacheManager};
use crate::runtime::kv_compat::KvCompatDescriptor;
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

type SsmSnapshot = (Vec<Option<Tensor>>, Vec<Option<Tensor>>);
const PROMPT: &str = "tok1 tok7 tok12 tok3";
const EXTENDED: &str = "tok1 tok7 tok12 tok3 tok9";

fn engine(speculative: bool) -> TorchEngine {
    let mut engine = TorchEngine::new(RuntimeConfig {
        use_gpu: false,
        speculative_decoding: speculative,
        kv_quant_type: crate::runtime::KVQuantType::None,
        ..Default::default()
    })
    .unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = (0..48)
        .map(|i| (format!("tok{i}"), serde_json::json!(i)))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "tok0"}
    });
    let tokenizer_bytes = serde_json::to_vec(&tokenizer).unwrap();
    *engine.tokenizer.lock() = Some(Tokenizer::from_bytes(&tokenizer_bytes).unwrap());
    engine.tokenizer_vocab_size.store(48, Ordering::Relaxed);
    engine.eos_token_id.store(0, Ordering::Relaxed);
    *engine.var_store.lock() = Some(VarStore::new(Device::Cpu));
    engine.persistent_model = Some(Arc::new(Mutex::new(Box::new(whole_model_mtp()))));
    *engine.context_state.lock() = Some(ContextState {
        sequence_length: 0,
        context_window: 64,
        initialized: true,
    });
    engine.initialize_kv_registry(4, 64, crate::runtime::KVQuantType::None, None);
    *engine.active_cache_owner.lock() = Some(CacheOwner::session("mtp-lifecycle"));
    // Fixed deterministic fixture weights and tokenizer, identical for every
    // engine. Populate authoritative identity so reuse is actually exercised.
    let mut descriptor = KvCompatDescriptor::default();
    descriptor.weights.base_revision = "tiny-qwen-hybrid-mtp-fixture-v1".into();
    descriptor.set_tokenizer(48, Some("wordlevel-tok0-through-tok47-v1".into()));
    *engine.kv_compat.lock() = Some(descriptor);
    engine
}

fn request(prompt: &str, max_tokens: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..Default::default()
    }
}

fn poll(stream: &mut TextStream<'_>) -> Option<String> {
    let waker = futures::task::noop_waker();
    match Pin::new(stream).poll_next(&mut Context::from_waker(&waker)) {
        Poll::Ready(item) => item.map(|item| item.expect("real stream must succeed")),
        Poll::Pending => panic!("CPU stream has no asynchronous wait"),
    }
}

fn cache(engine: &TorchEngine) -> Arc<Mutex<KVCacheManager>> {
    engine
        .persistent_model
        .as_ref()
        .unwrap()
        .lock()
        .get_kv_cache()
        .unwrap()
}

fn snapshot(engine: &TorchEngine) -> SsmSnapshot {
    engine
        .snapshot_ssm_states()
        .expect("fixture must retain hybrid SSM states")
}

fn state_error(actual: &SsmSnapshot, expected: &SsmSnapshot) -> f64 {
    assert_eq!(actual.0.len(), expected.0.len());
    assert_eq!(actual.1.len(), expected.1.len());
    let mut error: f64 = 0.0;
    let mut recurrent_layers = 0;
    for (a, b) in actual
        .0
        .iter()
        .chain(&actual.1)
        .zip(expected.0.iter().chain(&expected.1))
    {
        match (a, b) {
            (Some(a), Some(b)) => {
                recurrent_layers += 1;
                assert_eq!(a.size(), b.size(), "SSM tensor shapes must match");
                let difference = (a - b).abs();
                assert_eq!(
                    difference.isfinite().all().int64_value(&[]),
                    1,
                    "SSM tensor differences must be finite"
                );
                error = error.max(difference.max().double_value(&[]));
            }
            (None, None) => {}
            _ => panic!("restored SSM occupancy differs from serial prefix"),
        }
    }
    assert_eq!(
        recurrent_layers, 6,
        "three real GDN layers, conv and recurrent state"
    );
    error
}

#[test]
fn mtp_stream_state_parity_rejects_nonfinite_and_shape_mismatch() {
    fn zeros() -> SsmSnapshot {
        let layers = || {
            (0..3)
                .map(|_| Some(Tensor::zeros([2, 2], (tch::Kind::Float, Device::Cpu))))
                .collect()
        };
        (layers(), layers())
    }

    let expected = zeros();
    assert_eq!(state_error(&zeros(), &expected), 0.0);
    for (value, shape, diagnostic) in [
        (f64::NAN, [2, 2], "SSM tensor differences must be finite"),
        (
            f64::INFINITY,
            [2, 2],
            "SSM tensor differences must be finite",
        ),
        (0.0, [1, 2], "SSM tensor shapes must match"),
    ] {
        let mut actual = zeros();
        actual.1[0] = Some(Tensor::full(shape, value, (tch::Kind::Float, Device::Cpu)));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state_error(&actual, &expected)
        }))
        .expect_err("invalid states must fail parity, including broadcastable shapes");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("guard assertion must report its diagnostic");
        assert!(message.contains(diagnostic), "unexpected guard: {message}");
    }
}

#[derive(Clone, Copy, Debug)]
enum Stop {
    MaxTokens,
    Eos,
    Drop,
}

fn accepted_round_session_reuse(stop: Stop) {
    let engine = engine(true);
    let oracle = self::engine(false);
    let budget = if matches!(stop, Stop::Drop) { 8 } else { 2 };
    let mut stream = TextStream::new(&engine, request(PROMPT, budget)).unwrap();
    let mut serial = TextStream::new(&oracle, request(PROMPT, 8)).unwrap();
    assert!(stream.speculative);
    assert!(!serial.speculative);
    assert_eq!(poll(&mut stream), poll(&mut serial));
    let prefix_state = snapshot(&oracle);
    assert!(state_error(&snapshot(&engine), &prefix_state) < 1e-7);

    // A deterministic oracle draft establishes an accepted production round;
    // it does not bypass or replace the two-token verifier under test.
    assert!(poll(&mut serial).is_some());
    stream.spec_pending_draft = Some(serial.last_generated.unwrap() as u32);
    assert!(poll(&mut stream).is_some());
    assert_eq!(stream.spec_accepted, 1, "fixture must exercise acceptance");
    assert_eq!(stream.spec_rejected, 0);
    assert_eq!(stream.tokens_generated, 2);
    assert_eq!(stream.spec_out_queue.len(), 1, "bonus remains unconsumed");
    assert_eq!(
        stream.kv_cache_position, 6,
        "verifier consumed prompt + two tokens"
    );
    assert!(
        state_error(&snapshot(&engine), &prefix_state) > 1e-5,
        "nonzero control: speculative decode must advance recurrent history"
    );

    match stop {
        Stop::MaxTokens => {
            assert!(poll(&mut stream).is_none());
            assert!(matches!(
                stream.finish_reason,
                Some(FinishReason::MaxTokens)
            ));
            assert_eq!(stream.spec_out_queue.len(), 1);
        }
        Stop::Eos => {
            // Arm EOS only after the first accepted token was emitted. This
            // also works when the tiny model repeats the same greedy token.
            let bonus = *stream.spec_out_queue.front().unwrap();
            assert_ne!(bonus, 0, "EOS sentinel must be representable");
            engine.eos_token_id.store(bonus, Ordering::Relaxed);
            stream.max_tokens = 8;
            assert!(poll(&mut stream).is_none());
            assert!(matches!(
                stream.finish_reason,
                Some(FinishReason::EndOfSequence)
            ));
            assert_eq!(stream.tokens_generated, 2, "EOS was not emitted");
        }
        Stop::Drop => {
            assert!(!stream.finished);
            assert!(
                stream.tokens_generated < stream.max_tokens,
                "explicit drop must cancel before budget exhaustion"
            );
        }
    }
    drop(stream); // actual production persistence, including explicit cancellation
    engine.eos_token_id.store(0, Ordering::Relaxed);
    let cached = cache(&engine);
    assert_eq!(cached.lock().cached_token_count(), 4);
    assert_eq!(cached.lock().prefix_match_len(&[1, 7, 12, 3, 9]), 4);

    let fresh = self::engine(false);
    let mut resumed = TextStream::new(&engine, request(EXTENDED, 3)).unwrap();
    let restored_prefix_len = resumed.prefill_start_pos;
    let restored_error = state_error(&snapshot(&engine), &prefix_state);
    let mut reference = TextStream::new(&fresh, request(EXTENDED, 3)).unwrap();
    // Both next sessions really prefill and emit. Check state as well as text:
    // weak recurrent contributions may leave argmax unchanged despite stale SSM.
    let resumed_first = poll(&mut resumed);
    let reference_first = poll(&mut reference);
    let prefill_error = state_error(&snapshot(&engine), &snapshot(&fresh));
    assert_eq!(
        resumed_first, reference_first,
        "{stop:?}: next-session token"
    );
    assert_eq!(
        restored_prefix_len, 4,
        "{stop:?}: retain valid prompt snapshot reuse"
    );
    assert!(restored_error < 1e-7 && prefill_error < 1e-5,
        "{stop:?}: restored prefix error={restored_error:e}; next-session prefill error={prefill_error:e}");
}

#[test]
fn mtp_stream_max_tokens_restores_serial_session_prefix() {
    accepted_round_session_reuse(Stop::MaxTokens);
}

#[test]
fn mtp_stream_eos_restores_serial_session_prefix() {
    accepted_round_session_reuse(Stop::Eos);
}

#[test]
fn mtp_stream_drop_restores_serial_session_prefix() {
    accepted_round_session_reuse(Stop::Drop);
}
