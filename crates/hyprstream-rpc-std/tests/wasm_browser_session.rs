//! Real, browser-executed tests for `BrowserSession` — the shared
//! session/resolved-client API that constructs direct typed RPC clients
//! without `VfsShell`.
//!
//! Follows the established pattern from
//! `hyprstream-rpc/tests/wasm_browser_fetch.rs`: a JS-side
//! `fetch` override records every request and returns scripted responses, so
//! this exercises the real `wasm32` glue (`BrowserSession::establish` /
//! `renew` / `revoke`) against the unmodified production code path, not a
//! native reconstruction of it.
//!
//! `BrowserSession::client()` additionally dials a real WebTransport
//! connection via browser provisioning — not `fetch`-based, and not
//! mockable through this harness. The fail-closed gates on `client()`
//! (revoked / expired) are still fully covered here because they run
//! *before* any dial is attempted: a revoked or expired session rejects
//! `client()` without making any network call at all, which the mock call
//! count below confirms directly.
#![cfg(target_arch = "wasm32")]

use hyprstream_rpc_std::browser_session::BrowserSession;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// `JsError` implements neither `Debug` nor `Display` (it is a thin
/// wasm-bindgen wrapper around a JS `Error`), so `Result::expect_err`/
/// `{err}` don't work directly on it. Extract the real message via the
/// underlying JS `Error.message`, which `JsError::new(&message)` sets
/// verbatim — this is exactly the string `session_error`/`exchange_error`
/// constructed.
fn err_message(err: JsError) -> String {
    String::from(js_sys::Error::from(JsValue::from(err)).message())
}

#[wasm_bindgen(inline_js = "
    let __mockCalls = [];
    let __mockResponses = [];
    let __mockIdx = 0;

    export function installMockFetch(responsesJson) {
        __mockResponses = JSON.parse(responsesJson);
        __mockCalls = [];
        __mockIdx = 0;
        window.fetch = function (request) {
            return request.text().then(function (bodyText) {
                const headers = {};
                for (const pair of request.headers.entries()) {
                    headers[pair[0]] = pair[1];
                }
                __mockCalls.push({
                    url: request.url,
                    method: request.method,
                    headers: headers,
                    body: bodyText,
                });
                const r = __mockResponses[Math.min(__mockIdx, __mockResponses.length - 1)];
                __mockIdx += 1;
                const respHeaders = new Headers();
                for (const k in (r.headers || {})) {
                    respHeaders.set(k, r.headers[k]);
                }
                return new Response(r.body, { status: r.status, headers: respHeaders });
            });
        };
    }

    export function getMockCallsJson() {
        return JSON.stringify(__mockCalls);
    }
")]
#[allow(non_snake_case)]
extern "C" {
    fn installMockFetch(responses_json: &str);
    fn getMockCallsJson() -> String;
}

/// A real `js_sys::Function` sign callback backed by an actual Ed25519 key —
/// the same `sign_fn(Uint8Array) -> Promise<Uint8Array>` shape production
/// invokes for both DPoP proofs and RPC envelope signing.
fn make_sign_fn(
    signing_key: ed25519_dalek::SigningKey,
) -> (
    js_sys::Function,
    Closure<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>,
) {
    use ed25519_dalek::Signer as _;
    let closure = Closure::wrap(Box::new(move |input: js_sys::Uint8Array| -> js_sys::Promise {
        let bytes = input.to_vec();
        let signature = signing_key.sign(&bytes).to_bytes();
        js_sys::Promise::resolve(&JsValue::from(js_sys::Uint8Array::from(
            signature.as_slice(),
        )))
    }) as Box<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>);
    let function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    (function, closure)
}

/// A PQ arm callback is required by every resolved client this module
/// builds; its content is irrelevant to the tests here (none reach the
/// client-construction/WebTransport step), so a fixed dummy signature is
/// sufficient.
fn make_pq_sign_fn() -> (
    js_sys::Function,
    Closure<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>,
) {
    let closure = Closure::wrap(Box::new(move |_input: js_sys::Uint8Array| -> js_sys::Promise {
        js_sys::Promise::resolve(&JsValue::from(js_sys::Uint8Array::new_with_length(0)))
    }) as Box<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>);
    let function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    (function, closure)
}

fn dpop_bound_at_jwt(pubkey: &[u8; 32]) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    let jkt = hyprstream_rpc::wasm_token_exchange::ed25519_dpop_jkt(pubkey);
    let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"at+jwt"}"#);
    let payload = URL_SAFE_NO_PAD.encode(format!(
        r#"{{"sub":"did:plc:alice","exp":9999999999,"iat":1,"cnf":{{"jkt":"{jkt}"}}}}"#
    ));
    format!("{header}.{payload}.sig")
}

fn exchange_response_body(pubkey: &[u8; 32], expires_in: i64) -> String {
    serde_json::json!({
        "access_token": dpop_bound_at_jwt(pubkey),
        "token_type": "DPoP",
        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "expires_in": expires_in,
    })
    .to_string()
}

#[wasm_bindgen_test]
async fn establish_succeeds_with_sender_bound_token_and_captures_nonce() {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _c1) = make_sign_fn(signing_key);
    let (pq_sign_fn, _c2) = make_pq_sign_fn();

    let responses = serde_json::json!([{
        "status": 200,
        "body": exchange_response_body(&pubkey, 300),
        "headers": {"DPoP-Nonce": "n1", "content-type": "application/json"},
    }])
    .to_string();
    installMockFetch(&responses);

    let before = js_sys::Date::now();
    let session = BrowserSession::establish(
        "https://as.example.test",
        "subject-token".to_owned(),
        "urn:ietf:params:oauth:token-type:jwt".to_owned(),
        &pubkey,
        sign_fn,
        &[0u8; 32],
        pq_sign_fn,
        None,
    )
    .await
    .expect("establish must succeed against a valid sender-bound response");

    assert_eq!(session.subject_did(), "did:plc:alice");
    assert_eq!(session.exchange_nonce().as_deref(), Some("n1"));
    assert!(!session.is_revoked());
    assert!(
        session.expires_at() >= before + 300_000.0,
        "expiry must be ~300s in the future"
    );
}

#[wasm_bindgen_test]
async fn establish_rejects_non_sender_bound_response() {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _c1) = make_sign_fn(signing_key);
    let (pq_sign_fn, _c2) = make_pq_sign_fn();

    let bearer_body = serde_json::json!({
        "access_token": dpop_bound_at_jwt(&pubkey),
        "token_type": "Bearer",
        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "expires_in": 300,
    })
    .to_string();
    let responses = serde_json::json!([
        {"status": 200, "body": bearer_body, "headers": {"content-type": "application/json"}},
    ])
    .to_string();
    installMockFetch(&responses);

    let result = BrowserSession::establish(
        "https://as.example.test",
        "subject-token".to_owned(),
        "urn:ietf:params:oauth:token-type:jwt".to_owned(),
        &pubkey,
        sign_fn,
        &[0u8; 32],
        pq_sign_fn,
        None,
    )
    .await;
    let err = match result {
        Ok(_) => panic!("a Bearer (non-sender-bound) response must be rejected, not installed"),
        Err(e) => err_message(e),
    };
    assert!(
        err.contains("[sender_binding_missing]"),
        "unexpected error: {err}"
    );
}

#[wasm_bindgen_test]
async fn renew_chains_exchange_using_current_access_token_as_subject() {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _c1) = make_sign_fn(signing_key);
    let (pq_sign_fn, _c2) = make_pq_sign_fn();

    let first_token = dpop_bound_at_jwt(&pubkey);
    let first_body = serde_json::json!({
        "access_token": first_token,
        "token_type": "DPoP",
        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "expires_in": 300,
    })
    .to_string();
    let responses = serde_json::json!([
        {
            "status": 200,
            "body": first_body,
            "headers": {"content-type": "application/json"},
        },
        {
            "status": 200,
            "body": exchange_response_body(&pubkey, 600),
            "headers": {"content-type": "application/json"},
        },
    ])
    .to_string();
    installMockFetch(&responses);

    let session = BrowserSession::establish(
        "https://as.example.test",
        "subject-token".to_owned(),
        "urn:ietf:params:oauth:token-type:jwt".to_owned(),
        &pubkey,
        sign_fn,
        &[0u8; 32],
        pq_sign_fn,
        None,
    )
    .await
    .expect("establish must succeed");

    session.renew().await.expect("renew must succeed");

    let calls: Vec<serde_json::Value> =
        serde_json::from_str(&getMockCallsJson()).expect("mock calls must be valid JSON");
    assert_eq!(calls.len(), 2, "establish + one renew call");
    let renew_body = calls[1]["body"].as_str().unwrap();
    // JWT characters (base64url alphabet + `.` separators) are all in the
    // percent-encoder's unreserved set, so the token appears byte-for-byte.
    assert!(
        renew_body.contains(&format!("subject_token={first_token}")),
        "renew must present the prior access token as subject_token: {renew_body}"
    );
    assert!(
        renew_body.contains("subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Aaccess_token"),
        "renew must use the access_token subject_token_type (chained exchange): {renew_body}"
    );
}

#[wasm_bindgen_test]
async fn revoke_marks_session_unusable_even_when_transport_fails() {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x44; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _c1) = make_sign_fn(signing_key);
    let (pq_sign_fn, _c2) = make_pq_sign_fn();

    let responses = serde_json::json!([
        {
            "status": 200,
            "body": exchange_response_body(&pubkey, 300),
            "headers": {"content-type": "application/json"},
        },
        {
            "status": 500,
            "body": "internal error",
            "headers": {},
        },
    ])
    .to_string();
    installMockFetch(&responses);

    let session = BrowserSession::establish(
        "https://as.example.test",
        "subject-token".to_owned(),
        "urn:ietf:params:oauth:token-type:jwt".to_owned(),
        &pubkey,
        sign_fn,
        &[0u8; 32],
        pq_sign_fn,
        None,
    )
    .await
    .expect("establish must succeed");

    assert!(!session.is_revoked());
    let revoke_err = err_message(
        session
            .revoke()
            .await
            .expect_err("a transport failure on /oauth/revoke must surface as an error"),
    );
    assert!(revoke_err.contains("[revocation_transport_error]"));
    // The session is unusable regardless — the caller cannot tell "revoked"
    // from "server unreachable", and must not keep trusting a token it just
    // tried to kill.
    assert!(session.is_revoked());

    let client_result = session.client("https://registry.example.test", "registry").await;
    let client_err = match client_result {
        Ok(_) => panic!("client() must fail closed on a revoked session"),
        Err(e) => err_message(e),
    };
    assert!(
        client_err.contains("[session_revoked]"),
        "{client_err}"
    );
}

#[wasm_bindgen_test]
async fn expired_session_rejects_client_construction_without_any_network_call() {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x45; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _c1) = make_sign_fn(signing_key);
    let (pq_sign_fn, _c2) = make_pq_sign_fn();

    let responses = serde_json::json!([{
        "status": 200,
        // expires_in: 0 — clamped to "now", so the session is already
        // expired the instant establish() returns.
        "body": exchange_response_body(&pubkey, 0),
        "headers": {"content-type": "application/json"},
    }])
    .to_string();
    installMockFetch(&responses);

    let session = BrowserSession::establish(
        "https://as.example.test",
        "subject-token".to_owned(),
        "urn:ietf:params:oauth:token-type:jwt".to_owned(),
        &pubkey,
        sign_fn,
        &[0u8; 32],
        pq_sign_fn,
        None,
    )
    .await
    .expect("establish must succeed even though expires_in is 0");

    let expired_result = session.client("https://registry.example.test", "registry").await;
    let err = match expired_result {
        Ok(_) => panic!("client() must fail closed on an expired session"),
        Err(e) => err_message(e),
    };
    assert!(err.contains("[session_expired]"), "{err}");

    // The expiry gate runs before any dial: no provisioning/WebTransport
    // fetch was attempted beyond the one establish() call already made.
    let calls: Vec<serde_json::Value> =
        serde_json::from_str(&getMockCallsJson()).expect("mock calls must be valid JSON");
    assert_eq!(
        calls.len(),
        1,
        "client() on an expired session must not make any additional network call"
    );
}
