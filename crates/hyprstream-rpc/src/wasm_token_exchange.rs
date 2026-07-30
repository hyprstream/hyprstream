//! Browser-side RFC 8693 token-exchange (#1314) for the wasm client.
//!
//! The browser `VfsShell` exchanges an atproto/external JWT for a short-lived
//! hyprstream at+jwt Bearer, then presents that Bearer as the default JWT on
//! every RPC. This module holds:
//!
//! - **Pure helpers** (`exchange_form_body`, `parse_exchange_response`,
//!   `subject_from_access_token`) — target-agnostic, unit-tested on native.
//! - **`fetch_exchange_token`** — the browser `fetch` glue, `wasm32`-only (it
//!   uses `web_sys`), mirroring `browser_provisioning::fetch_browser_provisioning`.
//!
//! # Authority note
//!
//! `subject_from_access_token` decodes the at+jwt `sub` **without** signature
//! verification. This is client-side bookkeeping only: it derives a display
//! `Subject` from the freshly-exchanged Bearer. The token's authority is never
//! trusted from this decode — the server re-verifies the bearer on every RPC
//! (`require_bearer_token` / `validate_oauth_access_token`). The exchange
//! response does not echo `sub`, and the mint stamps `sub = verified.sub`
//! (`token_exchange.rs`), so the client decodes it from the minted token rather
//! than extending the server response.

use anyhow::{anyhow, ensure, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};

/// The RFC 8693 token-exchange grant type.
pub const GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
/// Issued-token type the #1314 endpoint always mints (at+jwt / access_token).
pub const ISSUED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

/// The well-known public `client_id` the browser VfsShell presents at the
/// RFC 8693 token-exchange endpoint (#1425).
///
/// This is a **public client** identifier (RFC 9700 / OAuth 2.1 `none`
/// token-endpoint auth method): it names the browser client so the AS can
/// apply the public-client token-exchange contract, but it is **not a client
/// secret and not a proof of identity**. There is no corresponding secret —
/// the sender binding comes entirely from the RFC 9449 DPoP proof
/// (`cnf.jkt`), not from this identifier. The same constant is the
/// single source of truth the AS matches against (`hyprstream` imports it
/// from this crate) so the wire contract cannot drift between client and
/// server.
pub const BROWSER_PUBLIC_CLIENT_ID: &str = "hyprstream-browser-vfs";
/// Browser session exchange route on the hyprstream OAuth origin.
pub const SESSION_EXCHANGE_PATH: &str = "/api/session/exchange";
/// Server-derived viewer context route on the hyprstream OAuth origin.
pub const WHOAMI_PATH: &str = "/api/session/whoami";

/// Server-derived browser viewer authority.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionContext {
    pub did: Option<String>,
    pub kind: SessionKind,
    pub tenant: Option<String>,
    pub can_act_locally: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionKind {
    Local,
    Federated,
    Unauthenticated,
}

/// A successfully exchanged short-lived access token (#1314 / #1425).
///
/// `token_type` records whether the AS minted a sender-bound (`DPoP`,
/// RFC 9449) or bearer (`Bearer`) token. A `DPoP` token carries `cnf.jkt`
/// and **must** be presented as `Authorization: DPoP` with a fresh matching
/// proof on every resource request — the resource server rejects a
/// `cnf.jkt`-bound token presented as `Bearer` (RFC 9449 §7).
///
/// `Debug` is manual and redacts `access_token` so an accidental `{:?}` log
/// can't leak the credential.
#[derive(Clone, PartialEq, Eq)]
pub struct ExchangedToken {
    /// The at+jwt `access_token` — presented as the client's default JWT.
    pub access_token: String,
    /// Lifetime in seconds, as reported by the endpoint. Informational: Lane C
    /// uses a static default-JWT; a refresh path (`withTokenProvider`) is the
    /// future-work follow-on the recon notes.
    pub expires_in: i64,
    /// The OAuth `token_type` the AS returned — `"DPoP"` for a sender-bound
    /// token (the #1425 browser contract), `"Bearer"` otherwise. Drives
    /// whether downstream RPC requests must carry a DPoP proof.
    pub token_type: TokenType,
    /// The server-issued `DPoP-Nonce` from a successful response (#1425 r1
    /// P1#4). The caller persists this and supplies it on the next exchange
    /// / resource request to avoid a `use_dpop_nonce` round-trip. `None` when
    /// the AS did not return a nonce (e.g. a non-DPoP response).
    pub nonce: Option<String>,
}

/// The OAuth `token_type` of an [`ExchangedToken`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    /// `token_type: Bearer` — no key binding.
    Bearer,
    /// `token_type: DPoP` (RFC 9449) — sender-bound via `cnf.jkt`; every
    /// resource request needs a matching DPoP proof.
    Dpop,
}

impl TokenType {
    /// Parse the `token_type` field, case-insensitively. `None` (field
    /// omitted) is treated as `Bearer` for backward compatibility with the
    /// pre-#1424 endpoint shape.
    fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw.map(str::to_ascii_lowercase).as_deref() {
            None | Some("bearer") => Ok(TokenType::Bearer),
            Some("dpop") => Ok(TokenType::Dpop),
            Some(other) => Err(format!("unsupported token_type: {other}")),
        }
    }
}

impl std::fmt::Debug for ExchangedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExchangedToken")
            .field("access_token", &"<redacted>")
            .field("expires_in", &self.expires_in)
            .field("token_type", &self.token_type)
            .finish()
    }
}

#[derive(Deserialize)]
struct ExchangeResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    issued_token_type: Option<String>,
}

/// Build the `application/x-www-form-urlencoded` body for an RFC 8693 exchange.
///
/// #1425: the body now carries the required **public** `client_id`
/// ([`BROWSER_PUBLIC_CLIENT_ID`]) and is sent alongside a `DPoP` proof
/// header for sender-bound exchange. #1425 r2 P2: the body also carries the
/// RFC 8707 `resource` the caller is targeting — the browser always knows
/// this (it is the same origin `fetch_exchange_token` is calling), so the
/// contract is explicit rather than the server inferring a default. `grant_type`
/// and the `subject_token_type` constant are URL-safe; `subject_token`,
/// `client_id`, and `resource` are percent-encoded defensively.
pub fn exchange_form_body(
    subject_token: &str,
    subject_token_type: &str,
    client_id: &str,
    resource: &str,
) -> String {
    format!(
        "grant_type={GRANT_TYPE}\
         &subject_token={}\
         &subject_token_type={}\
         &client_id={}\
         &resource={}",
        percent_encode(subject_token),
        percent_encode(subject_token_type),
        percent_encode(client_id),
        percent_encode(resource),
    )
}

/// Minimal RFC 3986 unreserved percent-encoding.
fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{:02X}", byte));
        }
    }
    out
}

/// Parse the JSON body of a successful `POST /oauth/token` exchange.
///
/// Validates `access_token` is present and, when the endpoint sets them, that
/// `token_type` is a supported value (`Bearer` or, since #1425, `DPoP` for a
/// sender-bound token) and `issued_token_type` is the access-token type —
/// defending against a misrouted/shape-changed response being treated as a
/// valid credential. The parsed [`TokenType`] tells the caller whether
/// downstream resource requests must carry a DPoP proof.
pub fn parse_exchange_response(body: &str) -> Result<ExchangedToken> {
    let resp: ExchangeResponse = serde_json::from_str(body)
        .map_err(|e| anyhow!("token-exchange response is not valid JSON: {e}"))?;
    ensure!(
        !resp.access_token.is_empty(),
        "token-exchange response omitted access_token"
    );
    let token_type = TokenType::parse(resp.token_type.as_deref())
        .map_err(|e| anyhow!("token-exchange {e}"))?;
    if let Some(issued) = &resp.issued_token_type {
        ensure!(
            issued == ISSUED_TOKEN_TYPE,
            "token-exchange issued_token_type unexpected: {issued}"
        );
    }
    Ok(ExchangedToken {
        access_token: resp.access_token,
        expires_in: resp.expires_in.unwrap_or(0).max(0),
        token_type,
        nonce: None,
    })
}

/// Verify an [`ExchangedToken`] is actually sender-bound to `dpop_pubkey`
/// before the caller (the browser sender-bound exchange) trusts it (#1425 r2
/// P2).
///
/// [`parse_exchange_response`] treats an omitted or `Bearer` `token_type` as
/// backward-compatible (other, non-browser callers of this endpoint shape
/// still expect that). But `fetch_exchange_token` is *exclusively* the
/// sender-bound browser exchange: a response that comes back as `Bearer` (a
/// misrouted request, a downgraded/compatibility responder, or a proxy that
/// stripped the DPoP semantics) must be rejected here rather than silently
/// installed as if it were correctly bound — the caller only checks
/// `token_type` `once, at parse time`, and every downstream RPC assumes the
/// installed credential really is `cnf.jkt`-bound to this key.
///
/// This also independently confirms the token's `cnf.jkt` (decoded from the
/// unverified JWT payload — the server re-verifies the signature and binding
/// on every RPC; this is client-side defense in depth, not the trust
/// boundary) equals [`ed25519_dpop_jkt`] of the exact key that produced the
/// proof, so a compatibility/misrouted response bound to some *other* key
/// cannot be mistaken for a token this browser can actually use.
pub fn verify_sender_bound_token(token: &ExchangedToken, dpop_pubkey: &[u8; 32]) -> Result<()> {
    ensure!(
        token.token_type == TokenType::Dpop,
        "sender-bound browser exchange requires token_type: DPoP; got Bearer or an omitted token_type"
    );
    let claims = crate::auth::decode_unverified(&token.access_token)
        .map_err(|e| anyhow!("exchanged access_token is not a decodable JWT: {e}"))?;
    let actual_jkt = claims
        .cnf_jkt()
        .ok_or_else(|| anyhow!("exchanged token carries no cnf.jkt; sender binding is missing"))?;
    let expected_jkt = ed25519_dpop_jkt(dpop_pubkey);
    ensure!(
        actual_jkt == expected_jkt,
        "exchanged token cnf.jkt does not match the browser DPoP key"
    );
    Ok(())
}

/// Parse the exact session-exchange / whoami JSON response.
pub fn parse_session_context(body: &str) -> Result<SessionContext> {
    let context: SessionContext = serde_json::from_str(body)
        .map_err(|e| anyhow!("session context response is not valid JSON: {e}"))?;
    let valid = match context.kind {
        SessionKind::Local => {
            context.did.as_deref().is_some_and(|did| !did.is_empty())
                && context
                    .tenant
                    .as_deref()
                    .is_some_and(|tenant| !tenant.is_empty())
                && context.can_act_locally
        }
        SessionKind::Federated => {
            context.did.as_deref().is_some_and(|did| !did.is_empty())
                && context.tenant.is_none()
                && !context.can_act_locally
        }
        SessionKind::Unauthenticated => {
            context.did.is_none() && context.tenant.is_none() && !context.can_act_locally
        }
    };
    ensure!(valid, "session context contains inconsistent viewer authority");
    Ok(context)
}

/// Decode the `sub` claim from an exchanged at+jwt and build a `Subject`.
///
/// No signature verification (see the module authority note). Returns an error
/// if the token is not a decodable JWT or has no `sub`.
pub fn subject_from_access_token(token: &str) -> Result<crate::Subject> {
    let claims = crate::auth::decode_unverified(token)
        .map_err(|e| anyhow!("access_token is not a decodable JWT: {e}"))?;
    ensure!(!claims.sub.is_empty(), "access_token has no sub claim");
    Ok(crate::Subject::new(claims.sub))
}

// ============================================================================
// DPoP proof helpers (RFC 9449) — pure, native-testable (#1425 r1 P1 #3/#4)
// ============================================================================

/// Build the RFC 9449 DPoP JWT signing input (`header.payload`) for an
/// Ed25519 key. The caller signs `signing_input` with the private key and
/// assembles the proof via [`assemble_dpop_proof`].
///
/// Pure / native-testable: the wasm32 glue calls this, invokes the JS sign
/// callback, and assembles. Keeping the construction here means the htm/htu/
/// iat/jti/ath/nonce/alg/jwk shape is tested once, natively, rather than
/// only through wasm32-only glue.
pub fn ed25519_dpop_signing_input(
    pubkey: &[u8; 32],
    htm: &str,
    htu: &str,
    iat: i64,
    jti: &str,
    ath: Option<&str>,
    nonce: Option<&str>,
) -> Result<(String, String)> {
    let x = URL_SAFE_NO_PAD.encode(pubkey);
    let header = serde_json::json!({
        "typ": "dpop+jwt",
        "alg": "EdDSA",
        "jwk": {"kty": "OKP", "crv": "Ed25519", "x": x}
    });
    let mut payload = serde_json::json!({
        "jti": jti,
        "htm": htm,
        "htu": htu,
        "iat": iat,
    });
    if let Some(ath) = ath {
        payload["ath"] = serde_json::Value::String(ath.to_owned());
    }
    if let Some(nonce) = nonce {
        payload["nonce"] = serde_json::Value::String(nonce.to_owned());
    }
    let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?);
    let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload)?);
    let signing_input = format!("{header_b64}.{payload_b64}");
    Ok((signing_input, header_b64))
}

/// Assemble a complete DPoP proof JWT from the signing input and signature.
pub fn assemble_dpop_proof(signing_input: &str, signature: &[u8]) -> String {
    format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(signature))
}

/// Compute the RFC 7638 JWK thumbprint (`cnf.jkt`) for an Ed25519 public key.
pub fn ed25519_dpop_jkt(pubkey: &[u8; 32]) -> String {
    crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Ed25519 { x: pubkey })
}

/// Check whether an HTTP error response body is a `use_dpop_nonce` error
/// (RFC 9449 §8). The browser retries with a fresh proof carrying the
/// `DPoP-Nonce` response header.
pub fn is_use_dpop_nonce_response(status: u16, body: &str) -> bool {
    status == 400
        && serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(serde_json::Value::as_str)
                    .map(|s| s == "use_dpop_nonce")
            })
            .unwrap_or(false)
}

/// Compute the DPoP `ath` value: `base64url(SHA-256(access_token))`.
pub fn dpop_ath(access_token: &str) -> String {
    use sha2::{Digest, Sha256};
    URL_SAFE_NO_PAD.encode(Sha256::digest(access_token.as_bytes()))
}

/// Generate a fresh, random DPoP `jti` (RFC 9449 requires a unique value per
/// proof so the AS can detect replay). Pure / native-testable so JTI
/// freshness is a plain unit test rather than something only exercisable
/// inside the `wasm32`-only fetch glue (#1425 r1 P1#4).
pub fn generate_dpop_jti() -> String {
    use rand::RngCore as _;
    let mut jti_bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut jti_bytes);
    URL_SAFE_NO_PAD.encode(jti_bytes)
}

/// Outcome of one token-exchange fetch attempt, decided from the response's
/// status/body/headers (RFC 9449 §8 nonce lifecycle). Pure / native-testable:
/// the `wasm32` glue calls this after every `fetch` to decide whether to
/// return, retry with a fresh nonce, or fail (#1425 r1 P1#4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonceOutcome {
    /// The request succeeded; the caller should parse the body.
    Success,
    /// The AS requires a server nonce; retry once with this fresh value.
    RetryWithNonce(String),
    /// The AS requires a nonce but returned no `DPoP-Nonce` header to retry
    /// with, or the retry budget (one retry) is exhausted.
    Failed,
}

/// Decide the next action for a token-exchange fetch attempt.
///
/// `attempt` is 0 for the bootstrap request, 1 for the nonce retry. Only a
/// bootstrap attempt (`attempt == 0`) may be retried; a failure on attempt 1
/// is always [`NonceOutcome::Failed`] (RFC 9449 §8 bounds the retry to one
/// round-trip so a malicious/misbehaving AS cannot force an infinite loop).
pub fn decide_nonce_outcome(
    attempt: u8,
    response_ok: bool,
    status: u16,
    body: &str,
    fresh_nonce_header: Option<&str>,
) -> NonceOutcome {
    if response_ok {
        return NonceOutcome::Success;
    }
    if attempt == 0 && is_use_dpop_nonce_response(status, body) {
        if let Some(fresh) = fresh_nonce_header {
            return NonceOutcome::RetryWithNonce(fresh.to_owned());
        }
    }
    NonceOutcome::Failed
}

// ============================================================================
// Browser fetch glue (wasm32 only)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod fetch {
    use super::{
        assemble_dpop_proof, decide_nonce_outcome, ed25519_dpop_signing_input,
        exchange_form_body, generate_dpop_jti, parse_exchange_response, parse_session_context,
        verify_sender_bound_token, BROWSER_PUBLIC_CLIENT_ID, ExchangedToken, NonceOutcome,
        SessionContext, SESSION_EXCHANGE_PATH, WHOAMI_PATH,
    };
    use anyhow::{anyhow, ensure, Context, Result};
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_futures::JsFuture;

    /// Cap on a token-exchange response body. An OAuth token response is a few
    /// hundred bytes; anything larger signals a misconfigured/compromised
    /// endpoint trying to exhaust the browser tab (mirrors the spirit of
    /// `browser_provisioning::MAX_PROVISIONING_BYTES`).
    const MAX_EXCHANGE_RESPONSE_BYTES: usize = 64 * 1024;

    /// Validate `endpoint` is an absolute, credential-free HTTPS origin.
    fn parse_origin(endpoint: &str) -> Result<url::Url> {
        let parsed = url::Url::parse(endpoint).context("invalid exchange endpoint")?;
        ensure!(
            parsed.scheme() == "https"
                && parsed.host_str().is_some()
                && parsed.username().is_empty()
                && parsed.password().is_none()
                && parsed.fragment().is_none()
                && (parsed.path().is_empty() || parsed.path() == "/")
                && parsed.query().is_none(),
            "exchange endpoint must be an absolute credential-free HTTPS origin (no path/query)"
        );
        Ok(parsed)
    }

    /// Read a `Response` body fully into a UTF-8 string via its stream reader.
    ///
    /// Mirrors `browser_provisioning::fetch_browser_provisioning`'s reader loop
    /// (no `content-length` dependency).
    async fn read_body_text(response: &web_sys::Response) -> Result<String> {
        let body = response
            .body()
            .ok_or_else(|| anyhow!("token-exchange response omitted its body"))?;
        let reader: web_sys::ReadableStreamDefaultReader = body.get_reader().unchecked_into();
        let mut bytes = Vec::new();
        loop {
            let result = JsFuture::from(reader.read())
                .await
                .map_err(|e| anyhow!("token-exchange response read failed: {e:?}"))?;
            let done = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("done"))
                .map_err(|_| anyhow!("token-exchange chunk omitted done"))?
                .as_bool()
                .ok_or_else(|| anyhow!("token-exchange chunk had invalid done"))?;
            if done {
                break;
            }
            let value = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("value"))
                .map_err(|_| anyhow!("token-exchange chunk omitted value"))?;
            let chunk: js_sys::Uint8Array = value
                .dyn_into()
                .map_err(|_| anyhow!("token-exchange chunk was not a Uint8Array"))?;
            let chunk_bytes = chunk.to_vec();
            if bytes.len().saturating_add(chunk_bytes.len()) > MAX_EXCHANGE_RESPONSE_BYTES {
                return Err(anyhow!(
                    "token-exchange response exceeds {MAX_EXCHANGE_RESPONSE_BYTES} bytes"
                ));
            }
            bytes.extend_from_slice(&chunk_bytes);
        }
        String::from_utf8(bytes).context("token-exchange response was not valid UTF-8")
    }

    /// Build and sign an Ed25519 DPoP proof by calling the browser's sign
    /// callback (`dpop_sign_fn`). The callback receives the JWT signing input
    /// bytes and returns a 64-byte Ed25519 signature.
    async fn build_ed25519_dpop_proof(
        pubkey: &[u8; 32],
        sign_fn: &js_sys::Function,
        htm: &str,
        htu: &str,
        ath: Option<&str>,
        nonce: Option<&str>,
    ) -> Result<String> {
        let iat = chrono::Utc::now().timestamp();
        let jti = generate_dpop_jti();
        let (signing_input, _) = ed25519_dpop_signing_input(
            pubkey, htm, htu, iat, &jti, ath, nonce,
        )?;
        // Call the JS sign callback: sign_fn(Uint8Array) → Promise<Uint8Array(64)>.
        let input_array = js_sys::Uint8Array::from(signing_input.as_bytes());
        let result = sign_fn
            .call1(&wasm_bindgen::JsValue::UNDEFINED, &input_array)
            .map_err(|e| anyhow!("DPoP sign callback invocation failed: {e:?}"))?;
        let promise: js_sys::Promise = result
            .dyn_into()
            .map_err(|_| anyhow!("DPoP sign callback did not return a Promise"))?;
        let signature_value = JsFuture::from(promise)
            .await
            .map_err(|e| anyhow!("DPoP sign callback failed: {e:?}"))?;
        let signature_array: js_sys::Uint8Array = signature_value
            .dyn_into()
            .map_err(|_| anyhow!("DPoP sign callback returned non-Uint8Array"))?;
        Ok(assemble_dpop_proof(&signing_input, &signature_array.to_vec()))
    }

    /// POST an RFC 8693 token-exchange grant to `{exchange_endpoint}/oauth/token`
    /// and return the short-lived at+jwt access token.
    ///
    /// #1425 r1 P1#3/#4: the request carries the required public `client_id`
    /// ([`BROWSER_PUBLIC_CLIENT_ID`]) and a DPoP proof header generated from
    /// the browser-held Ed25519 key (`dpop_pubkey` + `dpop_sign_fn`). The AS
    /// mints a **sender-bound** (`token_type: DPoP`, `cnf.jkt`) token bound to
    /// this key.
    ///
    /// **Nonce lifecycle (RFC 9449 §8):** `nonce` is the server-issued nonce
    /// from a previous successful response (persisted by the caller). On a
    /// `use_dpop_nonce` error, this function extracts the fresh `DPoP-Nonce`
    /// response header, generates a new proof carrying it, and retries once.
    /// On success, the response's `DPoP-Nonce` is returned in
    /// [`ExchangedToken::nonce`] for the caller to persist.
    ///
    /// Every downstream resource request presenting the returned token must
    /// carry a fresh DPoP proof (with `ath`) from the same key — the resource
    /// server rejects a `cnf.jkt`-bound token presented as Bearer (RFC 9449 §7).
    pub async fn fetch_exchange_token(
        exchange_endpoint: &str,
        subject_token: &str,
        subject_token_type: &str,
        dpop_pubkey: &[u8; 32],
        dpop_sign_fn: &js_sys::Function,
        nonce: Option<&str>,
    ) -> Result<ExchangedToken> {
        let mut url = parse_origin(exchange_endpoint)?;
        // #1425 r2 P2: the canonical resource is this exact origin, no
        // trailing slash — computed from the validated input string (not the
        // re-serialized `url::Url`, which normalizes an empty path to `/` and
        // would otherwise silently drift from the server's
        // `canonical_issuer_origin` no-trailing-slash form).
        let resource = exchange_endpoint.trim_end_matches('/');
        url.set_path("/oauth/token");
        let token_endpoint_htu = url.as_str().to_owned();
        let body = exchange_form_body(
            subject_token,
            subject_token_type,
            BROWSER_PUBLIC_CLIENT_ID,
            resource,
        );

        let mut current_nonce = nonce.map(str::to_owned);

        // Up to two attempts: bootstrap → use_dpop_nonce retry.
        for attempt in 0..2u8 {
            let proof = build_ed25519_dpop_proof(
                dpop_pubkey,
                dpop_sign_fn,
                "POST",
                &token_endpoint_htu,
                None, // no ath at the token endpoint
                current_nonce.as_deref(),
            )
            .await?;

            let headers = web_sys::Headers::new()
                .map_err(|e| anyhow!("token-exchange header construction failed: {e:?}"))?;
            headers
                .set("content-type", "application/x-www-form-urlencoded")
                .map_err(|e| anyhow!("token-exchange content-type set failed: {e:?}"))?;
            headers
                .set("accept", "application/json")
                .map_err(|e| anyhow!("token-exchange accept set failed: {e:?}"))?;
            headers
                .set("dpop", &proof)
                .map_err(|e| anyhow!("token-exchange DPoP header set failed: {e:?}"))?;

            let init = web_sys::RequestInit::new();
            init.set_method("POST");
            init.set_body(&js_sys::JsString::from(body.as_str()));
            init.set_headers(headers.as_ref());
            init.set_cache(web_sys::RequestCache::NoStore);
            init.set_credentials(web_sys::RequestCredentials::SameOrigin);
            init.set_redirect(web_sys::RequestRedirect::Error);

            let request = web_sys::Request::new_with_str_and_init(url.as_str(), &init)
                .map_err(|e| anyhow!("token-exchange request construction failed: {e:?}"))?;
            let window =
                web_sys::window().ok_or_else(|| anyhow!("browser window unavailable"))?;
            let response_value = JsFuture::from(window.fetch_with_request(&request))
                .await
                .map_err(|e| anyhow!("token-exchange fetch failed: {e:?}"))?;
            let response: web_sys::Response = response_value
                .dyn_into()
                .map_err(|_| anyhow!("token-exchange fetch returned a non-Response"))?;

            let text = read_body_text(&response).await?;
            let status = response.status();
            let response_ok = response.ok();
            let fresh_nonce_header = response.headers().get("DPoP-Nonce").ok().flatten();

            // RFC 9449 §8 nonce lifecycle, decided by the same pure function
            // the native tests exercise directly (#1425 r1 P1#4).
            match decide_nonce_outcome(attempt, response_ok, status, &text, fresh_nonce_header.as_deref()) {
                NonceOutcome::Success => {}
                NonceOutcome::RetryWithNonce(fresh) => {
                    current_nonce = Some(fresh);
                    continue;
                }
                NonceOutcome::Failed => {
                    return Err(anyhow!(
                        "token-exchange endpoint returned HTTP {status}: {}",
                        text.trim()
                    ));
                }
            }

            // Success: parse body + carry the response nonce for the caller to persist.
            let mut token = parse_exchange_response(&text)?;
            token.nonce = fresh_nonce_header;
            // #1425 r2 P2: fail closed rather than installing a misrouted or
            // compatibility (Bearer / foreign-key) response as if it were
            // this browser's sender-bound credential.
            verify_sender_bound_token(&token, dpop_pubkey)?;
            return Ok(token);
        }
        Err(anyhow!(
            "token-exchange: exhausted nonce retries (this should not happen)"
        ))
    }

    /// Exchange a one-use ATProto service-auth JWT plus DPoP proof for the
    /// opaque HttpOnly hyprstream browser session cookie.
    pub async fn fetch_session_exchange(
        exchange_endpoint: &str,
        service_auth_jwt: &str,
        dpop_proof: &str,
    ) -> Result<SessionContext> {
        let mut url = parse_origin(exchange_endpoint)?;
        url.set_path(SESSION_EXCHANGE_PATH);

        let headers = web_sys::Headers::new()
            .map_err(|e| anyhow!("session-exchange header construction failed: {e:?}"))?;
        headers
            .set("authorization", &format!("Bearer {service_auth_jwt}"))
            .map_err(|e| anyhow!("session-exchange authorization set failed: {e:?}"))?;
        headers
            .set("dpop", dpop_proof)
            .map_err(|e| anyhow!("session-exchange DPoP set failed: {e:?}"))?;
        headers
            .set("accept", "application/json")
            .map_err(|e| anyhow!("session-exchange accept set failed: {e:?}"))?;

        let init = web_sys::RequestInit::new();
        init.set_method("POST");
        init.set_headers(headers.as_ref());
        init.set_cache(web_sys::RequestCache::NoStore);
        init.set_credentials(web_sys::RequestCredentials::Include);
        init.set_redirect(web_sys::RequestRedirect::Error);

        let request = web_sys::Request::new_with_str_and_init(url.as_str(), &init)
            .map_err(|e| anyhow!("session-exchange request construction failed: {e:?}"))?;
        session_context_fetch(request, "session-exchange").await
    }

    /// Fetch the current viewer authority using the opaque session cookie.
    pub async fn fetch_session_context(exchange_endpoint: &str) -> Result<SessionContext> {
        let mut url = parse_origin(exchange_endpoint)?;
        url.set_path(WHOAMI_PATH);

        let headers = web_sys::Headers::new()
            .map_err(|e| anyhow!("whoami header construction failed: {e:?}"))?;
        headers
            .set("accept", "application/json")
            .map_err(|e| anyhow!("whoami accept set failed: {e:?}"))?;
        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        init.set_headers(headers.as_ref());
        init.set_cache(web_sys::RequestCache::NoStore);
        init.set_credentials(web_sys::RequestCredentials::Include);
        init.set_redirect(web_sys::RequestRedirect::Error);

        let request = web_sys::Request::new_with_str_and_init(url.as_str(), &init)
            .map_err(|e| anyhow!("whoami request construction failed: {e:?}"))?;
        session_context_fetch(request, "whoami").await
    }

    async fn session_context_fetch(
        request: web_sys::Request,
        operation: &str,
    ) -> Result<SessionContext> {
        let window = web_sys::window().ok_or_else(|| anyhow!("browser window unavailable"))?;
        let response_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow!("{operation} fetch failed: {e:?}"))?;
        let response: web_sys::Response = response_value
            .dyn_into()
            .map_err(|_| anyhow!("{operation} fetch returned a non-Response"))?;
        let text = read_body_text(&response).await?;
        if !response.ok() {
            return Err(anyhow!(
                "{operation} endpoint returned HTTP {}: {}",
                response.status(),
                text.trim()
            ));
        }
        parse_session_context(&text)
    }
}

#[cfg(target_arch = "wasm32")]
pub use fetch::{fetch_exchange_token, fetch_session_context, fetch_session_exchange};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn jwt(payload_json: &str) -> String {
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"at+jwt"}"#);
        let payload = URL_SAFE_NO_PAD.encode(payload_json);
        format!("{header}.{payload}.c2ln")
    }

    #[test]
    fn form_body_round_trips_safe_constants() {
        // `:` is not RFC 3986 unreserved, so the caller-supplied URN is encoded;
        // the server's form parser decodes %3A back to ':'. A valid JWT
        // (base64url-no-pad + '.') is already unreserved, so it passes through.
        // #1425: the body also carries the public client_id. #1425 r2 P2: and
        // the canonical resource the browser is targeting.
        let body = exchange_form_body(
            "abc.def.ghi",
            "urn:ietf:params:oauth:token-type:jwt",
            BROWSER_PUBLIC_CLIENT_ID,
            "https://as.example",
        );
        assert_eq!(
            body,
            "grant_type=urn:ietf:params:oauth:grant-type:token-exchange\
             &subject_token=abc.def.ghi\
             &subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Ajwt\
             &client_id=hyprstream-browser-vfs\
             &resource=https%3A%2F%2Fas.example"
        );
    }

    #[test]
    fn form_body_percent_encodes_non_unreserved() {
        // base64url-no-pad never produces '+', but a tampered/odd token could carry it.
        let body = exchange_form_body("a+b c", "t/y", "c w", "r s");
        assert!(body.contains("subject_token=a%2Bb%20c"), "{body}");
        assert!(body.contains("subject_token_type=t%2Fy"), "{body}");
        assert!(body.contains("client_id=c%20w"), "{body}");
        assert!(body.contains("resource=r%20s"), "{body}");
    }

    // ── #1425 r2 P2: sender-bound response verification ───────────────────────

    fn dpop_bound_token(pubkey: &[u8; 32], token_type: TokenType) -> ExchangedToken {
        let jkt = ed25519_dpop_jkt(pubkey);
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"at+jwt"}"#);
        let payload = URL_SAFE_NO_PAD.encode(format!(
            r#"{{"sub":"alice","exp":9999999999,"iat":1,"cnf":{{"jkt":"{jkt}"}}}}"#
        ));
        ExchangedToken {
            access_token: format!("{header}.{payload}.sig"),
            expires_in: 300,
            token_type,
            nonce: None,
        }
    }

    #[test]
    fn verify_sender_bound_token_accepts_matching_dpop_token() {
        let pubkey = [0x71; 32];
        let token = dpop_bound_token(&pubkey, TokenType::Dpop);
        verify_sender_bound_token(&token, &pubkey).expect("matching DPoP token must be accepted");
    }

    #[test]
    fn verify_sender_bound_token_rejects_bearer() {
        let pubkey = [0x72; 32];
        let token = dpop_bound_token(&pubkey, TokenType::Bearer);
        let err = verify_sender_bound_token(&token, &pubkey)
            .expect_err("a Bearer response must be rejected by the sender-bound fetch");
        assert!(err.to_string().contains("DPoP"), "unexpected error: {err:#}");
    }

    #[test]
    fn verify_sender_bound_token_rejects_foreign_key_binding() {
        let pubkey = [0x73; 32];
        let attacker_pubkey = [0x74; 32];
        // Bound to a DIFFERENT key than the one that produced the proof.
        let token = dpop_bound_token(&attacker_pubkey, TokenType::Dpop);
        let err = verify_sender_bound_token(&token, &pubkey)
            .expect_err("a token bound to a foreign key must be rejected");
        assert!(err.to_string().contains("cnf.jkt"), "unexpected error: {err:#}");
    }

    #[test]
    fn verify_sender_bound_token_rejects_missing_cnf() {
        let pubkey = [0x75; 32];
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"EdDSA","typ":"at+jwt"}"#);
        let payload = URL_SAFE_NO_PAD.encode(r#"{"sub":"alice","exp":9999999999,"iat":1}"#);
        let token = ExchangedToken {
            access_token: format!("{header}.{payload}.sig"),
            expires_in: 300,
            token_type: TokenType::Dpop,
            nonce: None,
        };
        let err = verify_sender_bound_token(&token, &pubkey)
            .expect_err("a token with no cnf.jkt at all must be rejected");
        assert!(err.to_string().contains("cnf.jkt"), "unexpected error: {err:#}");
    }

    #[test]
    fn browser_public_client_id_is_stable() {
        // The AS matches this literal; a drift here breaks the wire contract.
        assert_eq!(BROWSER_PUBLIC_CLIENT_ID, "hyprstream-browser-vfs");
        assert!(!BROWSER_PUBLIC_CLIENT_ID.is_empty());
    }

    // ── #1425 r1: pure DPoP proof helpers (RFC 9449) ──────────────────────────

    #[test]
    fn ed25519_dpop_signing_input_has_correct_shape() {
        let pubkey = [0x42; 32];
        let (signing_input, _header) =
            ed25519_dpop_signing_input(&pubkey, "POST", "https://as.example/oauth/token", 1700000000, "jti-1", None, None)
                .unwrap();
        // Two dots → three segments (header.payload.???).
        assert_eq!(signing_input.matches('.').count(), 1);
        let (h, p) = signing_input.split_once('.').unwrap();
        let header: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(h).unwrap()).unwrap();
        let payload: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(p).unwrap()).unwrap();
        assert_eq!(header["typ"], "dpop+jwt");
        assert_eq!(header["alg"], "EdDSA");
        assert_eq!(header["jwk"]["kty"], "OKP");
        assert_eq!(header["jwk"]["crv"], "Ed25519");
        assert_eq!(payload["htm"], "POST");
        assert_eq!(payload["htu"], "https://as.example/oauth/token");
        assert_eq!(payload["iat"], 1700000000);
        assert_eq!(payload["jti"], "jti-1");
        assert!(payload.get("ath").is_none(), "ath must be absent when not provided");
        assert!(payload.get("nonce").is_none(), "nonce must be absent when not provided");
    }

    #[test]
    fn ed25519_dpop_signing_input_includes_ath_and_nonce() {
        let pubkey = [0x11; 32];
        let (signing_input, _) =
            ed25519_dpop_signing_input(&pubkey, "GET", "https://rpc.example/v1/models", 1, "j", Some("ath-hash"), Some("nonce-val"))
                .unwrap();
        let (_, p) = signing_input.split_once('.').unwrap();
        let payload: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(p).unwrap()).unwrap();
        assert_eq!(payload["ath"], "ath-hash");
        assert_eq!(payload["nonce"], "nonce-val");
    }

    #[test]
    fn assemble_dpop_proof_produces_three_segments() {
        let proof = assemble_dpop_proof("aaa.bbb", &[0xCD; 64]);
        let parts: Vec<&str> = proof.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "aaa");
        assert_eq!(parts[1], "bbb");
    }

    #[test]
    fn is_use_dpop_nonce_response_detects_400_nonce_error() {
        assert!(is_use_dpop_nonce_response(
            400,
            r#"{"error":"use_dpop_nonce","error_description":"nonce required"}"#
        ));
        // Not 400
        assert!(!is_use_dpop_nonce_response(
            401,
            r#"{"error":"use_dpop_nonce"}"#
        ));
        // 400 but different error
        assert!(!is_use_dpop_nonce_response(
            400,
            r#"{"error":"invalid_dpop_proof"}"#
        ));
        // Non-JSON
        assert!(!is_use_dpop_nonce_response(400, "not json"));
    }

    #[test]
    fn ed25519_dpop_jkt_matches_server_thumbprint() {
        let pubkey = [0x65; 32];
        let jkt = ed25519_dpop_jkt(&pubkey);
        // The server computes the same RFC 7638 thumbprint for an OKP Ed25519 key.
        let expected = crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Ed25519 {
            x: &pubkey,
        });
        assert_eq!(jkt, expected);
    }

    #[test]
    fn dpop_ath_is_base64url_sha256() {
        let ath = dpop_ath("hello.world.token");
        use sha2::{Digest, Sha256};
        assert_eq!(
            ath,
            URL_SAFE_NO_PAD.encode(Sha256::digest(b"hello.world.token"))
        );
    }

    // ── #1425 r1 P1#4: JTI freshness + nonce-lifecycle decision (native) ─────

    #[test]
    fn generate_dpop_jti_is_fresh_every_call() {
        let a = generate_dpop_jti();
        let b = generate_dpop_jti();
        assert_ne!(a, b, "two proofs must never reuse a jti");
        // 16 random bytes, base64url-no-pad-encoded.
        assert_eq!(URL_SAFE_NO_PAD.decode(&a).unwrap().len(), 16);
    }

    #[test]
    fn nonce_bootstrap_success_needs_no_retry() {
        // First-ever request for a key: the AS may accept without a nonce.
        let outcome = decide_nonce_outcome(0, true, 200, "{}", None);
        assert_eq!(outcome, NonceOutcome::Success);
    }

    #[test]
    fn nonce_required_failure_retries_with_fresh_nonce() {
        let body = r#"{"error":"use_dpop_nonce","error_description":"nonce required"}"#;
        let outcome = decide_nonce_outcome(0, false, 400, body, Some("fresh-nonce-1"));
        assert_eq!(
            outcome,
            NonceOutcome::RetryWithNonce("fresh-nonce-1".to_owned())
        );
    }

    #[test]
    fn nonce_retry_success_completes() {
        // Second attempt, now carrying the fresh nonce, succeeds.
        let outcome = decide_nonce_outcome(1, true, 200, "{}", None);
        assert_eq!(outcome, NonceOutcome::Success);
    }

    #[test]
    fn nonce_required_but_no_header_to_retry_with_fails() {
        // Server said use_dpop_nonce but (contrary to RFC 9449 §8) omitted the
        // DPoP-Nonce header — nothing to retry with, must fail closed.
        let body = r#"{"error":"use_dpop_nonce"}"#;
        let outcome = decide_nonce_outcome(0, false, 400, body, None);
        assert_eq!(outcome, NonceOutcome::Failed);
    }

    #[test]
    fn nonce_retry_budget_is_exactly_one_exhausted_on_second_failure() {
        // Even a legitimate use_dpop_nonce + fresh header on attempt 1 (the
        // retry itself) must NOT trigger a third attempt — bounds the loop.
        let body = r#"{"error":"use_dpop_nonce"}"#;
        let outcome = decide_nonce_outcome(1, false, 400, body, Some("another-nonce"));
        assert_eq!(outcome, NonceOutcome::Failed);
    }

    #[test]
    fn dpop_mismatch_error_is_not_a_nonce_retry() {
        // A key/proof mismatch (invalid_dpop_proof) must propagate as a hard
        // failure, never be mistaken for a retryable nonce error.
        let body = r#"{"error":"invalid_dpop_proof","error_description":"proof key mismatch"}"#;
        let outcome = decide_nonce_outcome(0, false, 400, body, Some("irrelevant-nonce"));
        assert_eq!(outcome, NonceOutcome::Failed);
    }

    #[test]
    fn nonce_expired_status_is_not_retried() {
        // A 401 (e.g. expired access/refresh token) is not a use_dpop_nonce
        // shape and must not be retried regardless of body content.
        let outcome = decide_nonce_outcome(0, false, 401, "{\"error\":\"use_dpop_nonce\"}", Some("n"));
        assert_eq!(outcome, NonceOutcome::Failed);
    }

    #[test]
    fn parse_full_response() {
        let body = r#"{"access_token":"aaa.bbb.ccc","issued_token_type":"urn:ietf:params:oauth:token-type:access_token","token_type":"Bearer","expires_in":300}"#;
        let t = parse_exchange_response(body).unwrap();
        assert_eq!(t.access_token, "aaa.bbb.ccc");
        assert_eq!(t.expires_in, 300);
        assert_eq!(t.token_type, TokenType::Bearer);
    }

    #[test]
    fn parse_tolerates_missing_extras() {
        let body = r#"{"access_token":"x.y.z"}"#;
        let t = parse_exchange_response(body).unwrap();
        assert_eq!(t.access_token, "x.y.z");
        assert_eq!(t.expires_in, 0);
        // Omitted token_type defaults to Bearer (pre-#1424 endpoint shape).
        assert_eq!(t.token_type, TokenType::Bearer);
    }

    #[test]
    fn parse_rejects_missing_access_token() {
        let body = r#"{"token_type":"Bearer","expires_in":300}"#;
        assert!(parse_exchange_response(body).is_err());
    }

    #[test]
    fn parse_rejects_unknown_token_type() {
        // #1425: only Bearer and DPoP are accepted; an unexpected token_type
        // must not be treated as a usable credential.
        let body = r#"{"access_token":"x","token_type":"N_A","expires_in":300}"#;
        assert!(parse_exchange_response(body).is_err());
    }

    #[test]
    fn parse_accepts_dpop_token_type() {
        // #1425: the sender-bound exchange returns token_type: DPoP. The
        // client must accept it (and downstream requests must carry a proof).
        let body = r#"{"access_token":"x.y.z","issued_token_type":"urn:ietf:params:oauth:token-type:access_token","token_type":"DPoP","expires_in":300}"#;
        let t = parse_exchange_response(body).unwrap();
        assert_eq!(t.access_token, "x.y.z");
        assert_eq!(t.token_type, TokenType::Dpop);
    }

    #[test]
    fn parse_rejects_wrong_issued_token_type() {
        let body = r#"{"access_token":"x","issued_token_type":"urn:ietf:params:oauth:token-type:id_token","token_type":"Bearer","expires_in":300}"#;
        assert!(parse_exchange_response(body).is_err());
    }

    #[test]
    fn parse_rejects_non_json() {
        assert!(parse_exchange_response("not json").is_err());
    }

    #[test]
    fn subject_decoded_from_sub_claim() {
        let token =
            jwt(r#"{"iss":"https://node.example","sub":"did:plc:abc","exp":9999999999,"iat":1}"#);
        let sub = subject_from_access_token(&token).unwrap();
        assert_eq!(sub.name(), Some("did:plc:abc"));
        assert!(!sub.is_anonymous());
    }

    #[test]
    fn subject_rejects_empty_sub() {
        let token = jwt(r#"{"iss":"x","sub":"","exp":1,"iat":1}"#);
        assert!(subject_from_access_token(&token).is_err());
    }

    #[test]
    fn subject_rejects_non_jwt() {
        assert!(subject_from_access_token("not-a-jwt").is_err());
    }

    #[test]
    fn exchanged_token_debug_redacts_access_token() {
        let t = ExchangedToken {
            access_token: "secret-bearer-value".to_owned(),
            expires_in: 300,
            token_type: TokenType::Bearer,
            nonce: None,
        };
        let rendered = format!("{t:?}");
        assert!(
            !rendered.contains("secret-bearer-value"),
            "ExchangedToken Debug leaked the access_token: {rendered}"
        );
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.contains("300"));
    }

    #[test]
    fn parses_local_session_context() {
        let context = parse_session_context(
            r#"{"did":"did:web:alice.example","kind":"local","tenant":"acme","canActLocally":true}"#,
        )
        .unwrap();
        assert_eq!(context.did.as_deref(), Some("did:web:alice.example"));
        assert_eq!(context.kind, SessionKind::Local);
        assert_eq!(context.tenant.as_deref(), Some("acme"));
        assert!(context.can_act_locally);
    }

    #[test]
    fn parses_unauthenticated_floor() {
        let context = parse_session_context(
            r#"{"did":null,"kind":"unauthenticated","tenant":null,"canActLocally":false}"#,
        )
        .unwrap();
        assert_eq!(context.kind, SessionKind::Unauthenticated);
        assert!(context.did.is_none());
        assert!(context.tenant.is_none());
        assert!(!context.can_act_locally);
    }

    #[test]
    fn rejects_inconsistent_session_authority() {
        assert!(parse_session_context(
            r#"{"did":"did:web:alice.example","kind":"federated","tenant":"client-asserted","canActLocally":true}"#,
        )
        .is_err());
    }
}
