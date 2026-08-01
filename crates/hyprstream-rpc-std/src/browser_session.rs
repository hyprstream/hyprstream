//! Shared browser session / resolved-client API (hyprstream/hyprstream#1442).
//!
//! `BrowserSession` consumes a verified Hyprstream browser session
//! established from the PDS flow (RFC 8693 token-exchange + RFC 9449 DPoP
//! sender binding, #1425) and constructs direct typed Registry, Model, and
//! Policy RPC clients through the resolved browser-provisioning path
//! (`wasm_rpc_client::build_resolved_client`) — without requiring
//! `VfsShell`/`TclShell`.
//!
//! The exchange, nonce lifecycle, and sender-binding verification are
//! entirely delegated to `hyprstream_rpc::wasm_token_exchange`
//! (`fetch_exchange_token` / `verify_sender_bound_token`, #1425 r1/r2): this
//! module adds session *lifecycle* (renewal, revocation, expiry tracking)
//! and multi-service client construction on top of that already-verified
//! exchange, plus a stable error-code taxonomy for callers.
//!
//! `VfsShell::connect` builds its namespace on top of the same exchange
//! today; the client construction here is the shared primitive both paths
//! can use.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use js_sys::Function;
use wasm_bindgen::prelude::*;

use hyprstream_rpc::wasm_token_exchange::{
    classify_exchange_error_code, expires_at_ms, fetch_exchange_token, revoke_access_token,
    subject_from_access_token,
};

use crate::wasm_rpc_client::{build_resolved_client, WasmRpcClient};

/// RFC 8693 §2.1: presenting the current access token as the subject of a
/// fresh exchange (a "chained" exchange) to obtain a renewed one bound to
/// the same DPoP key. The #1444 P1 fix hardened exactly this path — the new
/// proof's key must match the *current* token's `cnf.jkt` before the
/// replay registry consumes it.
const REFRESH_SUBJECT_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

/// A verified Hyprstream browser session: a sender-bound (`cnf.jkt`) access
/// token plus the browser-held hybrid signing keys, sufficient to construct
/// direct typed RPC clients. The private signing material never enters this
/// struct or WASM memory — only public keys and the JS sign callbacks are
/// held, matching the existing `VfsShell`/`JsSigner` convention.
///
/// Nothing here is written to `localStorage`/`sessionStorage`/IndexedDB:
/// the access token and DPoP nonce live only in this struct's memory for its
/// lifetime. A page reload has no persisted credential to recover — the
/// caller re-establishes from a fresh PDS-issued subject token. Callers that
/// want to skip the DPoP nonce bootstrap round-trip on the next `establish`
/// may persist [`exchange_nonce`](BrowserSession::exchange_nonce) themselves
/// (it is not a credential — RFC 9449 §8 nonces carry no authority on their
/// own).
#[wasm_bindgen]
pub struct BrowserSession {
    exchange_endpoint: String,
    dpop_pubkey: [u8; 32],
    sign_fn: Function,
    ml_dsa65_pubkey: Vec<u8>,
    pq_sign_fn: Function,
    access_token: RefCell<String>,
    expires_at_ms: RefCell<f64>,
    nonce: RefCell<Option<String>>,
    revoked: RefCell<bool>,
    subject_did: String,
}

#[wasm_bindgen]
impl BrowserSession {
    /// Establish a session by exchanging `subject_token` (e.g. an ATProto
    /// service-auth JWT) for a hyprstream access token sender-bound to
    /// `signer_pubkey`/`sign_fn` — the same Ed25519 key used for RPC
    /// envelope signing (#1425 r1 P1#3: one browser-held key through
    /// exchange and every RPC request). `signer_ml_dsa65_pubkey`/`pq_sign_fn`
    /// are the hybrid PQ arm, required by every resolved client this session
    /// constructs.
    ///
    /// `exchange_nonce` is a previously-persisted DPoP nonce for this same
    /// key (`None` on first connect); pass it to skip the RFC 9449 §8
    /// bootstrap round-trip.
    ///
    /// Every failure mode fails closed: an invalid/missing proof, a
    /// misrouted or non-sender-bound response, or a wrong-shaped subject
    /// token all reject with a categorized error (see the module's error
    /// taxonomy) rather than installing a partially-trusted session.
    #[wasm_bindgen(js_name = "establish")]
    #[allow(clippy::too_many_arguments)]
    pub async fn establish(
        exchange_endpoint: &str,
        subject_token: String,
        subject_token_type: String,
        signer_pubkey: &[u8],
        sign_fn: Function,
        signer_ml_dsa65_pubkey: &[u8],
        pq_sign_fn: Function,
        exchange_nonce: Option<String>,
    ) -> Result<BrowserSession, JsError> {
        console_error_panic_hook::set_once();
        let dpop_pubkey: [u8; 32] = signer_pubkey.try_into().map_err(|_| {
            session_error(
                "invalid_signer_key",
                "signer_pubkey must be exactly 32 bytes",
            )
        })?;

        let exchanged = fetch_exchange_token(
            exchange_endpoint,
            &subject_token,
            &subject_token_type,
            &dpop_pubkey,
            &sign_fn,
            exchange_nonce.as_deref(),
        )
        .await
        .map_err(|e| exchange_error(&e))?;

        let subject = subject_from_access_token(&exchanged.access_token)
            .map_err(|e| session_error("invalid_token", &e.to_string()))?;
        let subject_did = subject.name().unwrap_or_default().to_owned();

        Ok(Self {
            exchange_endpoint: exchange_endpoint.to_owned(),
            dpop_pubkey,
            sign_fn,
            ml_dsa65_pubkey: signer_ml_dsa65_pubkey.to_vec(),
            pq_sign_fn,
            expires_at_ms: RefCell::new(expires_at_ms(js_sys::Date::now(), exchanged.expires_in)),
            nonce: RefCell::new(exchanged.nonce.clone()),
            revoked: RefCell::new(false),
            access_token: RefCell::new(exchanged.access_token),
            subject_did,
        })
    }

    /// Renew the session: chained-exchange the current access token for a
    /// fresh one bound to the same DPoP key. Fails closed on a revoked
    /// session or any exchange error — the prior token is left installed
    /// until the new one is verified sender-bound, so a failed renewal
    /// cannot downgrade an active session to an unbound or foreign-key one.
    #[wasm_bindgen(js_name = "renew")]
    pub async fn renew(&self) -> Result<(), JsError> {
        self.fail_if_revoked()?;
        let current = self.access_token.borrow().clone();
        let exchanged = fetch_exchange_token(
            &self.exchange_endpoint,
            &current,
            REFRESH_SUBJECT_TOKEN_TYPE,
            &self.dpop_pubkey,
            &self.sign_fn,
            self.nonce.borrow().as_deref(),
        )
        .await
        .map_err(|e| exchange_error(&e))?;
        *self.expires_at_ms.borrow_mut() = expires_at_ms(js_sys::Date::now(), exchanged.expires_in);
        *self.nonce.borrow_mut() = exchanged.nonce.clone();
        *self.access_token.borrow_mut() = exchanged.access_token;
        Ok(())
    }

    /// Revoke the session's current access token (RFC 7009,
    /// `POST /oauth/revoke`) and mark this session unusable. Every
    /// subsequent [`client`](BrowserSession::client)/[`renew`](BrowserSession::renew)
    /// call fails closed, regardless of whether the revoke request itself
    /// reached the server — a client that cannot confirm revocation must
    /// not keep trusting the token it just tried to kill.
    #[wasm_bindgen(js_name = "revoke")]
    pub async fn revoke(&self) -> Result<(), JsError> {
        let token = self.access_token.borrow().clone();
        let outcome = revoke_access_token(&self.exchange_endpoint, &token).await;
        *self.revoked.borrow_mut() = true;
        outcome.map_err(|e| session_error("revocation_transport_error", &e.to_string()))
    }

    /// Whether [`revoke`](BrowserSession::revoke) has been called on this session.
    #[wasm_bindgen(getter = isRevoked)]
    pub fn is_revoked(&self) -> bool {
        *self.revoked.borrow()
    }

    /// The verified subject's `did`, decoded from the exchanged token's
    /// `sub` claim (client-side bookkeeping only; the server re-verifies
    /// authority on every RPC — see `wasm_token_exchange`'s module note).
    #[wasm_bindgen(getter = subjectDid)]
    pub fn subject_did(&self) -> String {
        self.subject_did.clone()
    }

    /// Milliseconds since epoch (`Date.now()` domain) when the current
    /// access token expires. Callers should [`renew`](BrowserSession::renew)
    /// before this passes.
    #[wasm_bindgen(getter = expiresAt)]
    pub fn expires_at(&self) -> f64 {
        *self.expires_at_ms.borrow()
    }

    /// The server-issued DPoP nonce from the most recent exchange, if any.
    /// Not a credential — safe to persist and pass back as `exchange_nonce`
    /// to [`establish`](BrowserSession::establish) for the same DPoP key.
    #[wasm_bindgen(getter = exchangeNonce)]
    pub fn exchange_nonce(&self) -> Option<String> {
        self.nonce.borrow().clone()
    }

    /// Construct a direct typed `RpcClient` for `service_name`, resolved and
    /// pinned through browser provisioning at `origin`, presenting this
    /// session's current sender-bound access token. Returns the same
    /// `RpcClient` type generated TypeScript clients call `.call()` on —
    /// this does not instantiate `VfsShell`/`TclShell`.
    #[wasm_bindgen(js_name = "client")]
    pub async fn client(
        &self,
        origin: &str,
        service_name: &str,
    ) -> Result<WasmRpcClient, JsError> {
        self.fail_if_revoked()?;
        self.fail_if_expired()?;
        let inner = build_resolved_client(
            origin,
            service_name,
            &self.dpop_pubkey,
            self.sign_fn.clone(),
            &self.ml_dsa65_pubkey,
            self.pq_sign_fn.clone(),
            Some(self.access_token.borrow().clone()),
        )
        .await?;
        Ok(WasmRpcClient::from_resolved(inner))
    }

    /// Sugar for `client(origin, "registry")`.
    #[wasm_bindgen(js_name = "registryClient")]
    pub async fn registry_client(&self, origin: &str) -> Result<WasmRpcClient, JsError> {
        self.client(origin, "registry").await
    }

    /// Sugar for `client(origin, "model")`.
    #[wasm_bindgen(js_name = "modelClient")]
    pub async fn model_client(&self, origin: &str) -> Result<WasmRpcClient, JsError> {
        self.client(origin, "model").await
    }

    /// Sugar for `client(origin, "policy")`.
    #[wasm_bindgen(js_name = "policyClient")]
    pub async fn policy_client(&self, origin: &str) -> Result<WasmRpcClient, JsError> {
        self.client(origin, "policy").await
    }
}

impl BrowserSession {
    fn fail_if_revoked(&self) -> Result<(), JsError> {
        if *self.revoked.borrow() {
            return Err(session_error(
                "session_revoked",
                "this BrowserSession was revoked; establish a new one",
            ));
        }
        Ok(())
    }

    fn fail_if_expired(&self) -> Result<(), JsError> {
        if js_sys::Date::now() >= *self.expires_at_ms.borrow() {
            return Err(session_error(
                "session_expired",
                "access token expired; call renew() first",
            ));
        }
        Ok(())
    }
}

fn session_error(code: &str, message: &str) -> JsError {
    JsError::new(&format!("[{code}] {message}"))
}

/// Wrap a `fetch_exchange_token` failure as a categorized `JsError`, using
/// the pure, native-tested classifier in `wasm_token_exchange`
/// (`classify_exchange_error_code`). This function itself has no wasm32
/// logic of its own — it only exists to attach the `JsError` type, which
/// (like the rest of this module) is wasm32-only.
fn exchange_error(error: &anyhow::Error) -> JsError {
    let message = error.to_string();
    session_error(classify_exchange_error_code(&message), &message)
}
