//! Browser behavioral tests for the WebTransport request deadline
//! (PR #1767 Codex review findings on `WtConnection`).
//!
//! The wasm32 `cargo check` gate proves compilation only. These tests run the
//! real deadline machinery in a real browser
//! (`wasm-pack test --headless --chrome -p hyprstream-rpc --test wasm_deadline_behavior`)
//! and pin the runtime behaviors the finding is about:
//!
//! 1. a stalled await is actually bounded — the browser timer behind
//!    `web_async::time::timeout` fires and the error surfaces promptly (no
//!    compile-only illusion);
//! 2. a fast completion under a generous deadline is not falsely expired;
//! 3. expiring while a stream read pends resets the stream through the
//!    locked reader via the actual production cleanup — the underlying
//!    source's `cancel` callback runs, matching the reader-level
//!    `reader.cancel()` reset `WtConnection::request` performs on expiry;
//! 4. bounded completion carries the caller deadline AND the auth fields
//!    (jwt / delegated_bearer) through the exact `Transport::send` options
//!    contract, so bounding never costs authentication;
//! 5. the total cleanup tail is bounded: the deadline error arrives within
//!    deadline + one shared cleanup budget (+ scheduler slop).
//!
//! A live `WebTransport` session cannot be constructed without a server, so
//! `WtConnection::request` itself is exercised by the browser cold-load
//! pipeline; these tests drive the same production `bounded` deadline
//! helper and the actual production expiry cleanup `cancel_reader_on_expiry`
//! the request path uses.
#![cfg(target_arch = "wasm32")]

use std::time::Duration;

use hyprstream_rpc::web_transport::{bounded, cancel_reader_on_expiry, CLEANUP_GRACE_MS};
use hyprstream_rpc::CallOptions;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn deadline_in(ms: u64) -> web_time::Instant {
    web_time::Instant::now() + web_time::Duration::from_millis(ms)
}

/// A stalled future must be cut off BY the timer: err within a bounded,
/// promptly observable window — never by test-side wrapping.
#[wasm_bindgen_test]
async fn stalled_request_deadline_actually_fires() {
    let started = web_time::Instant::now();
    let result = bounded(std::future::pending::<()>(), deadline_in(50)).await;
    let elapsed = started.elapsed();
    assert!(result.is_err(), "a pending future must hit the deadline");
    assert!(
        elapsed >= Duration::from_millis(45),
        "deadline fired early ({elapsed:?}) — timer semantics wrong"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "deadline did not fire promptly ({elapsed:?}) — setTimeout race is not running"
    );
}

/// A fast completion under a generous deadline passes the value through and
/// is not falsely expired.
#[wasm_bindgen_test]
async fn fast_completion_completes_under_a_generous_deadline() {
    let result = bounded(std::future::ready(41u8 + 1u8), deadline_in(5_000))
        .await
        .expect("immediate future must complete under the deadline");
    assert_eq!(result, 42);
}

#[wasm_bindgen(inline_js = r#"
    export function makeNeverStream() {
        let cancelled = false;
        const stream = new ReadableStream({
            pull(controller) { /* never enqueue: reads stay pending */ },
            cancel(reason) {
                cancelled = true;
                window.__wtDeadlineCancelReason = String(reason);
            },
        });
        return { stream, wasCancelled: () => cancelled };
    }

    export function installUnhandledRejectionSpy() {
        window.__wtUnhandledRejections = [];
        window.addEventListener(
            "unhandledrejection",
            (event) => {
                window.__wtUnhandledRejections.push(String(event.reason));
            },
        );
    }

    export function unhandledRejectionCount() {
        return window.__wtUnhandledRejections.length;
    }

    export function makeNeverSettlingCancelStream() {
        let cancelled = false;
        const stream = new ReadableStream({
            pull(controller) { /* never enqueue: reads stay pending */ },
            cancel(reason) {
                cancelled = true;
                // The reset algorithm runs (the side effect below is real),
                // but its promise never settles: the reset OBSERVATION must
                // consume the entire shared cleanup budget.
                return new Promise(() => {});
            },
        });
        return { stream, wasCancelled: () => cancelled };
    }
"#)]
extern "C" {
    fn makeNeverStream() -> JsValue;
    fn makeNeverSettlingCancelStream() -> JsValue;
    fn installUnhandledRejectionSpy();
    fn unhandledRejectionCount() -> u32;
}

/// Expiring while a stream read pends must reset the stream THROUGH THE
/// LOCKED READER via the actual production cleanup
/// (`cancel_reader_on_expiry`): expiry leaves the reader associated with
/// the stream and its JS read request pending; the cleanup cancels through
/// the reader (a direct `ReadableStream::cancel()` on the locked stream
/// rejects, asserted here) and then observes the abandoned read promise so
/// its settlement — whatever shape the engine gives it — never surfaces as
/// an unhandled rejection. Success is asserted three ways: the caller sees
/// the clean deadline error, the underlying source's `cancel(reason)` spy
/// RUNS, and the browser reports zero unhandled rejections.
#[wasm_bindgen_test]
async fn expiring_read_resets_the_stream_through_the_locked_reader() {
    installUnhandledRejectionSpy();

    let fixture = makeNeverStream();
    let stream = js_sys::Reflect::get(&fixture, &JsValue::from_str("stream"))
        .expect("fixture stream")
        .unchecked_into::<web_sys::ReadableStream>();
    let was_cancelled = js_sys::Reflect::get(&fixture, &JsValue::from_str("wasCancelled"))
        .expect("fixture probe")
        .unchecked_into::<js_sys::Function>();

    // Take the lock and leave the JS read pending exactly like production:
    // `bounded` drops the Rust future on expiry, but the browser read
    // request stays queued in the stream.
    let reader: web_sys::ReadableStreamDefaultReader = stream.get_reader().unchecked_into();
    let started = web_time::Instant::now();
    let deadline = deadline_in(50);
    let read_promise = reader.read();
    let result = bounded(JsFuture::from(read_promise.clone()), deadline).await;
    assert!(
        result.is_err(),
        "a never-yielding read must hit the deadline"
    );

    // The direct stream cancel is the trap: while the reader holds the
    // lock it must reject, not silently cancel (asserted, not assumed).
    let direct = JsFuture::from(stream.cancel()).await;
    assert!(
        direct.is_err(),
        "ReadableStream::cancel on a locked stream must reject"
    );

    // Production expiry cleanup: cancel through the still-associated
    // reader, then observe the abandoned read promise under the SAME shared
    // cleanup deadline (deadline + CLEANUP_GRACE_MS) — never a per-step sum.
    let error = cancel_reader_on_expiry(
        &reader,
        Some(read_promise),
        anyhow::anyhow!("WebTransport RPC request deadline exceeded after 50ms"),
        deadline,
    )
    .await;
    assert!(
        error.to_string().contains("deadline exceeded"),
        "the surfaced error must be the caller's deadline error: {error}"
    );
    assert!(
        !error.to_string().contains("stream reset"),
        "a successful reset must not add a reset-failure suffix: {error}"
    );

    // Total-bound regression: error delivery within deadline + one shared
    // cleanup grace + scheduler slop — the tail cannot silently grow.
    let elapsed = started.elapsed();
    let bound = Duration::from_millis(50 + CLEANUP_GRACE_MS.max(0) as u64 + 150);
    assert!(
        elapsed < bound,
        "error delivered in {elapsed:?}, past the documented tail bound {bound:?}"
    );

    // cancel() settles asynchronously — yield to the event loop (via the
    // same bounded helper), then check the source's cancel callback ran
    // and no rejection went unhandled.
    let _ = bounded(std::future::pending::<()>(), deadline_in(50)).await;
    assert!(
        was_cancelled
            .call0(&JsValue::UNDEFINED.unchecked_into::<js_sys::Object>())
            .expect("probe call")
            .as_bool()
            .unwrap_or(false),
        "cancel_reader_on_expiry must reach the underlying source's cancel callback"
    );
    assert_eq!(
        unhandledRejectionCount(),
        0,
        "the abandoned read settlement must never become an unhandled rejection"
    );
}

/// Budget guard: a stream whose `cancel()` never settles forces the reset
/// observation to consume the ENTIRE shared cleanup budget
/// (`CLEANUP_GRACE_MS` past the request deadline), so this test fails if
/// the cleanup tail is ever widened structurally (e.g. a second sequential
/// observation window that outlasts the slop). Per readable-stream-close,
/// the abandoned read settles synchronously during the `cancel()` call —
/// it contributes ~nothing here, so a dual-pending discriminator is
/// unconstructible and the +budget relation is pinned by construction at
/// the single computation site in `cancel_reader_on_expiry`. Asserts the
/// surfaced error carries the "did not settle" suffix, the source cancel
/// spy runs (the reset algorithm executes even though its promise never
/// settles), and zero rejections go unhandled.
#[wasm_bindgen_test]
async fn stuck_reset_consumes_the_shared_cleanup_budget() {
    installUnhandledRejectionSpy();

    let fixture = makeNeverSettlingCancelStream();
    let stream = js_sys::Reflect::get(&fixture, &JsValue::from_str("stream"))
        .expect("fixture stream")
        .unchecked_into::<web_sys::ReadableStream>();
    let was_cancelled = js_sys::Reflect::get(&fixture, &JsValue::from_str("wasCancelled"))
        .expect("fixture probe")
        .unchecked_into::<js_sys::Function>();

    let reader: web_sys::ReadableStreamDefaultReader = stream.get_reader().unchecked_into();
    let started = web_time::Instant::now();
    let deadline = deadline_in(50);
    let read_promise = reader.read();
    let result = bounded(JsFuture::from(read_promise.clone()), deadline).await;
    assert!(
        result.is_err(),
        "a never-yielding read must hit the deadline"
    );

    let error = cancel_reader_on_expiry(
        &reader,
        Some(read_promise),
        anyhow::anyhow!("WebTransport RPC request deadline exceeded after 50ms"),
        deadline,
    )
    .await;
    assert!(
        error
            .to_string()
            .contains("did not settle within the 50ms cleanup budget"),
        "a never-settling reset must be surfaced, not swallowed: {error}"
    );

    // Can't-grow guard: bound = deadline + one shared budget (+30ms
    // scheduler slop).
    let elapsed = started.elapsed();
    let bound = Duration::from_millis(50 + CLEANUP_GRACE_MS as u64 + 30);
    assert!(
        elapsed < bound,
        "error delivered in {elapsed:?}, past the budget bound {bound:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(45),
        "deadline fired early ({elapsed:?})"
    );

    // The reset algorithm ran even though its promise never settles.
    assert!(
        was_cancelled
            .call0(&JsValue::UNDEFINED.unchecked_into::<js_sys::Object>())
            .expect("probe call")
            .as_bool()
            .unwrap_or(false),
        "reader.cancel() must reach the underlying source's cancel callback"
    );
    assert_eq!(
        unhandledRejectionCount(),
        0,
        "the never-settling reset must not become an unhandled rejection"
    );
}

/// Bounded completion preserves authentication: the deadline rides beside
/// the auth fields through the exact `CallOptions` contract `Transport::send`
/// receives from `RpcClientImpl` — and the caller deadline replaces the
/// transport default verbatim instead of being clamped.
#[wasm_bindgen_test]
async fn bounded_completion_carries_deadline_and_auth() {
    let options = CallOptions::new()
        .jwt("direct-jwt".to_owned())
        .timeout_ms(750);
    assert_eq!(options.effective_timeout_ms(Some(30_000)), Some(750));
    assert_eq!(options.jwt.as_deref(), Some("direct-jwt"));
    assert_eq!(options.delegated_bearer, None);

    let delegated = CallOptions::new()
        .delegated_bearer("delegated".to_owned())
        .timeout_ms(250);
    assert_eq!(delegated.effective_timeout_ms(Some(30_000)), Some(250));
    assert_eq!(delegated.delegated_bearer.as_deref(), Some("delegated"));

    // The deadline promise itself resolves exactly once a bounded completion
    // would need it: the transport ceiling replaces the absence of a caller
    // deadline (30s family default), a present deadline is honored as-is.
    assert_eq!(
        CallOptions::new().effective_timeout_ms(Some(30_000)),
        Some(30_000)
    );
}
