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
/// header for sender-bound exchange. `grant_type` and the
/// `subject_token_type` constant are URL-safe; the `subject_token` (a JWT)
/// and `client_id` are percent-encoded defensively.
pub fn exchange_form_body(
    subject_token: &str,
    subject_token_type: &str,
    client_id: &str,
) -> String {
    format!(
        "grant_type={GRANT_TYPE}\
         &subject_token={}\
         &subject_token_type={}\
         &client_id={}",
        percent_encode(subject_token),
        percent_encode(subject_token_type),
        percent_encode(client_id),
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
    })
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
// Browser fetch glue (wasm32 only)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod fetch {
    use super::{
        exchange_form_body, parse_exchange_response, parse_session_context, ExchangedToken,
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

    /// POST an RFC 8693 token-exchange grant to `{exchange_endpoint}/oauth/token`
    /// and return the short-lived at+jwt access token.
    ///
    /// #1425: the request now carries the required **public** `client_id`
    /// ([`BROWSER_PUBLIC_CLIENT_ID`]) and a `DPoP` proof header so the AS
    /// mints a **sender-bound** (`token_type: DPoP`, `cnf.jkt`) token rather
    /// than a bearer. `dpop_proof` is the RFC 9449 proof JWT the caller built
    /// over `htm=POST, htu={origin}/oauth/token`; the AS verifies it (method,
    /// URI, `iat`, `jti`, server nonce) and binds the minted token to the
    /// proof key's `jkt`. The returned [`ExchangedToken::token_type`] is
    /// `DPoP`; every resource request presenting that token must likewise
    /// carry a fresh matching DPoP proof (RFC 9449 §7). CORS is open (#1316).
    pub async fn fetch_exchange_token(
        exchange_endpoint: &str,
        subject_token: &str,
        subject_token_type: &str,
        dpop_proof: &str,
    ) -> Result<ExchangedToken> {
        let mut url = parse_origin(exchange_endpoint)?;
        url.set_path("/oauth/token");

        let body = exchange_form_body(
            subject_token,
            subject_token_type,
            BROWSER_PUBLIC_CLIENT_ID,
        );

        let headers = web_sys::Headers::new()
            .map_err(|e| anyhow!("token-exchange header construction failed: {e:?}"))?;
        headers
            .set("content-type", "application/x-www-form-urlencoded")
            .map_err(|e| anyhow!("token-exchange content-type set failed: {e:?}"))?;
        headers
            .set("accept", "application/json")
            .map_err(|e| anyhow!("token-exchange accept set failed: {e:?}"))?;
        headers
            .set("dpop", dpop_proof)
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
        let window = web_sys::window().ok_or_else(|| anyhow!("browser window unavailable"))?;
        let response_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow!("token-exchange fetch failed: {e:?}"))?;
        let response: web_sys::Response = response_value
            .dyn_into()
            .map_err(|_| anyhow!("token-exchange fetch returned a non-Response"))?;

        let text = read_body_text(&response).await?;
        if !response.ok() {
            // The body is read first so a streamed OAuth error payload is surfaced.
            return Err(anyhow!(
                "token-exchange endpoint returned HTTP {}: {}",
                response.status(),
                text.trim()
            ));
        }
        parse_exchange_response(&text)
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
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

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
        // #1425: the body also carries the public client_id.
        let body = exchange_form_body(
            "abc.def.ghi",
            "urn:ietf:params:oauth:token-type:jwt",
            BROWSER_PUBLIC_CLIENT_ID,
        );
        assert_eq!(
            body,
            "grant_type=urn:ietf:params:oauth:grant-type:token-exchange\
             &subject_token=abc.def.ghi\
             &subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Ajwt\
             &client_id=hyprstream-browser-vfs"
        );
    }

    #[test]
    fn form_body_percent_encodes_non_unreserved() {
        // base64url-no-pad never produces '+', but a tampered/odd token could carry it.
        let body = exchange_form_body("a+b c", "t/y", "c w");
        assert!(body.contains("subject_token=a%2Bb%20c"), "{body}");
        assert!(body.contains("subject_token_type=t%2Fy"), "{body}");
        assert!(body.contains("client_id=c%20w"), "{body}");
    }

    #[test]
    fn browser_public_client_id_is_stable() {
        // The AS matches this literal; a drift here breaks the wire contract.
        assert_eq!(BROWSER_PUBLIC_CLIENT_ID, "hyprstream-browser-vfs");
        assert!(!BROWSER_PUBLIC_CLIENT_ID.is_empty());
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
