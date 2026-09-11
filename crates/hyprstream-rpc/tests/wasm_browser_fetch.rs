//! #1425 r2 P2: a real, browser-executed `fetch_exchange_token` test.
//!
//! Runs the actual `wasm32` fetch glue in a real browser
//! (`wasm-pack test --headless --chrome -p hyprstream-rpc --test wasm_browser_fetch`),
//! not a native reconstruction of the request shape. A JS-side `fetch`
//! override records every request (method/URL/headers/body) and returns
//! scripted responses, so this exercises: the real `js_sys::Function`
//! DPoP-sign callback invocation (Promise + Uint8Array marshaling), the
//! exact request body/headers `fetch_exchange_token` sends, the
//! `DPoP-Nonce` response header being read back, one `use_dpop_nonce`
//! retry, and the final success response (including the
//! `verify_sender_bound_token` / `resource` fixes) — all against the
//! unmodified production code path.
//!
//! This lives as a standalone integration test (rather than an inline
//! `#[cfg(test)]` module in the lib) because `cargo test --target
//! wasm32-unknown-unknown` compiles the crate's *entire* `--tests` unit —
//! including the hundreds of pre-existing native `#[tokio::test]`s scattered
//! through the lib's own source files, which are not wasm32-buildable. An
//! integration test file links against the lib built normally (no
//! `cfg(test)`), so none of those native-only unit tests are pulled in.
#![cfg(target_arch = "wasm32")]

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signer as _, SigningKey};
use hyprstream_rpc::wasm_token_exchange::{
    ed25519_dpop_jkt, fetch_exchange_token, verify_sender_bound_token, TokenType,
    ISSUED_TOKEN_TYPE,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

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
                    credentials: request.credentials,
                    cache: request.cache,
                    redirect: request.redirect,
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

/// Build a real `js_sys::Function` DPoP sign callback backed by an actual
/// Ed25519 key — the same `sign_fn(Uint8Array) -> Promise<Uint8Array>` shape
/// `build_ed25519_dpop_proof` calls in production. Returns the `Closure`
/// too; the caller must keep it alive until the callback is done being
/// invoked (dropping it earlier would invalidate the JS function and panic
/// on the next call).
fn make_sign_fn(
    signing_key: SigningKey,
) -> (js_sys::Function, Closure<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>) {
    let closure = Closure::wrap(Box::new(move |input: js_sys::Uint8Array| -> js_sys::Promise {
        let bytes = input.to_vec();
        let signature = signing_key.sign(&bytes).to_bytes();
        let array = js_sys::Uint8Array::from(signature.as_slice());
        js_sys::Promise::resolve(&JsValue::from(array))
    }) as Box<dyn FnMut(js_sys::Uint8Array) -> js_sys::Promise>);
    let function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
    (function, closure)
}

/// Build a minimal at+jwt access token JSON payload carrying `cnf.jkt`.
fn dpop_bound_at_jwt(pubkey: &[u8; 32]) -> String {
    let jkt = ed25519_dpop_jkt(pubkey);
    let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"at+jwt"}"#);
    let payload = URL_SAFE_NO_PAD.encode(format!(
        r#"{{"sub":"alice","exp":9999999999,"iat":1,"cnf":{{"jkt":"{jkt}"}}}}"#
    ));
    format!("{header}.{payload}.sig")
}

#[wasm_bindgen_test]
async fn browser_fetch_success_after_nonce_retry_exact_request_shape() {
    let signing_key = SigningKey::from_bytes(&[0x31; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _closure) = make_sign_fn(signing_key);

    let access_token = dpop_bound_at_jwt(&pubkey);
    let success_body = serde_json::json!({
        "access_token": access_token,
        "token_type": "DPoP",
        "issued_token_type": ISSUED_TOKEN_TYPE,
        "expires_in": 300,
    })
    .to_string();
    let responses = serde_json::json!([
        {
            "status": 400,
            "body": r#"{"error":"use_dpop_nonce","error_description":"nonce required"}"#,
            "headers": {"DPoP-Nonce": "server-nonce-1", "content-type": "application/json"},
        },
        {
            "status": 200,
            "body": success_body,
            "headers": {"DPoP-Nonce": "server-nonce-2", "content-type": "application/json"},
        },
    ])
    .to_string();
    installMockFetch(&responses);

    let result = fetch_exchange_token(
        "https://as.example.test",
        "subject-token-value",
        "urn:ietf:params:oauth:token-type:access_token",
        &pubkey,
        &sign_fn,
        None,
    )
    .await;

    let token = result.expect("bootstrap + nonce-retry exchange must succeed");
    assert_eq!(token.token_type, TokenType::Dpop);
    assert_eq!(token.nonce.as_deref(), Some("server-nonce-2"));
    // #1425 r2 P2: the returned token really is bound to THIS browser key.
    verify_sender_bound_token(&token, &pubkey)
        .expect("the token fetch_exchange_token returned must pass its own sender-bound check");

    let calls: Vec<serde_json::Value> =
        serde_json::from_str(&getMockCallsJson()).expect("mock calls must be valid JSON");
    assert_eq!(calls.len(), 2, "exactly one bootstrap + one nonce retry");

    for call in &calls {
        assert_eq!(call["method"], "POST");
        assert_eq!(call["url"], "https://as.example.test/oauth/token");
        assert_eq!(call["credentials"], "same-origin");
        assert_eq!(call["cache"], "no-store");
        assert_eq!(call["redirect"], "error");
        assert_eq!(
            call["headers"]["content-type"],
            "application/x-www-form-urlencoded"
        );
        assert_eq!(call["headers"]["accept"], "application/json");
        let body = call["body"].as_str().unwrap();
        assert!(body.contains("grant_type="), "{body}");
        assert!(body.contains("subject_token=subject-token-value"), "{body}");
        assert!(
            body.contains("client_id=hyprstream-browser-vfs"),
            "{body}"
        );
        // #1425 r2 P2: the real fetch always sends the canonical resource.
        assert!(
            body.contains("resource=https%3A%2F%2Fas.example.test"),
            "resource missing/wrong in request body: {body}"
        );
        // The DPoP proof is a real, freshly signed 3-segment JWT.
        let proof = call["headers"]["dpop"].as_str().expect("DPoP header present");
        assert_eq!(proof.split('.').count(), 3, "DPoP proof must be header.payload.signature");
    }

    // The retry proof must carry the nonce the first response returned.
    let first_proof = calls[0]["headers"]["dpop"].as_str().unwrap();
    let retry_proof = calls[1]["headers"]["dpop"].as_str().unwrap();
    assert_ne!(first_proof, retry_proof, "the retry must be a freshly signed proof");
    let retry_payload = retry_proof.split('.').nth(1).unwrap();
    let retry_json: serde_json::Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(retry_payload).unwrap()).unwrap();
    assert_eq!(retry_json["nonce"], "server-nonce-1");
    assert_eq!(retry_json["htm"], "POST");
    assert_eq!(retry_json["htu"], "https://as.example.test/oauth/token");
}

#[wasm_bindgen_test]
async fn browser_fetch_rejects_bearer_response() {
    // A misrouted/compatibility response that comes back as plain Bearer (no
    // sender binding) must be rejected by the browser sender-bound fetch,
    // not silently installed (#1425 r2 P2).
    let signing_key = SigningKey::from_bytes(&[0x32; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _closure) = make_sign_fn(signing_key);

    let bearer_body = serde_json::json!({
        "access_token": dpop_bound_at_jwt(&pubkey),
        "token_type": "Bearer",
        "issued_token_type": ISSUED_TOKEN_TYPE,
        "expires_in": 300,
    })
    .to_string();
    let responses = serde_json::json!([
        { "status": 200, "body": bearer_body, "headers": {"content-type": "application/json"} },
    ])
    .to_string();
    installMockFetch(&responses);

    let result = fetch_exchange_token(
        "https://as.example.test",
        "subject-token-value",
        "urn:ietf:params:oauth:token-type:access_token",
        &pubkey,
        &sign_fn,
        None,
    )
    .await;

    let err = result.expect_err("a Bearer response must be rejected, not installed");
    assert!(err.to_string().contains("DPoP"), "unexpected error: {err:#}");
}

#[wasm_bindgen_test]
async fn browser_fetch_rejects_foreign_key_binding() {
    // The response is `token_type: DPoP` but bound to a DIFFERENT key than
    // the one that produced the proof — a misrouted response must not be
    // mistaken for this browser's own credential.
    let signing_key = SigningKey::from_bytes(&[0x33; 32]);
    let pubkey: [u8; 32] = signing_key.verifying_key().to_bytes();
    let (sign_fn, _closure) = make_sign_fn(signing_key);
    let foreign_pubkey = [0x34u8; 32];

    let foreign_bound_body = serde_json::json!({
        "access_token": dpop_bound_at_jwt(&foreign_pubkey),
        "token_type": "DPoP",
        "issued_token_type": ISSUED_TOKEN_TYPE,
        "expires_in": 300,
    })
    .to_string();
    let responses = serde_json::json!([
        {
            "status": 200,
            "body": foreign_bound_body,
            "headers": {"content-type": "application/json"},
        },
    ])
    .to_string();
    installMockFetch(&responses);

    let result = fetch_exchange_token(
        "https://as.example.test",
        "subject-token-value",
        "urn:ietf:params:oauth:token-type:access_token",
        &pubkey,
        &sign_fn,
        None,
    )
    .await;

    let err = result.expect_err("a token bound to a foreign key must be rejected, not installed");
    assert!(err.to_string().contains("cnf.jkt"), "unexpected error: {err:#}");
}
