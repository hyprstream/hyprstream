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
use serde::Deserialize;

/// The RFC 8693 token-exchange grant type.
pub const GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
/// Issued-token type the #1314 endpoint always mints (at+jwt / access_token).
pub const ISSUED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

/// A successfully exchanged short-lived Bearer (at+jwt).
///
/// `Debug` is manual and redacts `access_token` so an accidental `{:?}` log
/// can't leak the Bearer.
#[derive(Clone, PartialEq, Eq)]
pub struct ExchangedToken {
    /// The at+jwt `access_token` — presented as the client's default JWT.
    pub access_token: String,
    /// Lifetime in seconds, as reported by the endpoint. Informational: Lane C
    /// uses a static default-JWT; a refresh path (`withTokenProvider`) is the
    /// future-work follow-on the recon notes.
    pub expires_in: i64,
}

impl std::fmt::Debug for ExchangedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExchangedToken")
            .field("access_token", &"<redacted>")
            .field("expires_in", &self.expires_in)
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
/// `grant_type` and the `subject_token_type` constant are URL-safe; the
/// `subject_token` (a JWT) is percent-encoded defensively.
pub fn exchange_form_body(subject_token: &str, subject_token_type: &str) -> String {
    format!(
        "grant_type={GRANT_TYPE}&subject_token={}&subject_token_type={}",
        percent_encode(subject_token),
        percent_encode(subject_token_type),
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
/// `token_type` is `Bearer` and `issued_token_type` is the access-token type —
/// defending against a misrouted/shape-changed response being treated as a
/// valid Bearer.
pub fn parse_exchange_response(body: &str) -> Result<ExchangedToken> {
    let resp: ExchangeResponse = serde_json::from_str(body)
        .map_err(|e| anyhow!("token-exchange response is not valid JSON: {e}"))?;
    ensure!(
        !resp.access_token.is_empty(),
        "token-exchange response omitted access_token"
    );
    if let Some(token_type) = &resp.token_type {
        ensure!(
            token_type.eq_ignore_ascii_case("Bearer"),
            "token-exchange token_type is not Bearer: {token_type}"
        );
    }
    if let Some(issued) = &resp.issued_token_type {
        ensure!(
            issued == ISSUED_TOKEN_TYPE,
            "token-exchange issued_token_type unexpected: {issued}"
        );
    }
    Ok(ExchangedToken {
        access_token: resp.access_token,
        expires_in: resp.expires_in.unwrap_or(0).max(0),
    })
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
    use super::{exchange_form_body, parse_exchange_response, ExchangedToken};
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
    /// and return the short-lived at+jwt Bearer.
    ///
    /// Form-encoded body, **no DPoP** — the generic exchange path (#1314) is
    /// bearer-only; only the UCAN-grant path needs DPoP. CORS is open (#1316).
    pub async fn fetch_exchange_token(
        exchange_endpoint: &str,
        subject_token: &str,
        subject_token_type: &str,
    ) -> Result<ExchangedToken> {
        let mut url = parse_origin(exchange_endpoint)?;
        url.set_path("/oauth/token");

        let body = exchange_form_body(subject_token, subject_token_type);

        let headers = web_sys::Headers::new()
            .map_err(|e| anyhow!("token-exchange header construction failed: {e:?}"))?;
        headers
            .set("content-type", "application/x-www-form-urlencoded")
            .map_err(|e| anyhow!("token-exchange content-type set failed: {e:?}"))?;
        headers
            .set("accept", "application/json")
            .map_err(|e| anyhow!("token-exchange accept set failed: {e:?}"))?;

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
}

#[cfg(target_arch = "wasm32")]
pub use fetch::fetch_exchange_token;

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
        let body = exchange_form_body("abc.def.ghi", "urn:ietf:params:oauth:token-type:jwt");
        assert_eq!(
            body,
            "grant_type=urn:ietf:params:oauth:grant-type:token-exchange\
             &subject_token=abc.def.ghi\
             &subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Ajwt"
        );
    }

    #[test]
    fn form_body_percent_encodes_non_unreserved() {
        // base64url-no-pad never produces '+', but a tampered/odd token could carry it.
        let body = exchange_form_body("a+b c", "t/y");
        assert!(body.contains("subject_token=a%2Bb%20c"), "{body}");
        assert!(body.contains("subject_token_type=t%2Fy"), "{body}");
    }

    #[test]
    fn parse_full_response() {
        let body = r#"{"access_token":"aaa.bbb.ccc","issued_token_type":"urn:ietf:params:oauth:token-type:access_token","token_type":"Bearer","expires_in":300}"#;
        let t = parse_exchange_response(body).unwrap();
        assert_eq!(t.access_token, "aaa.bbb.ccc");
        assert_eq!(t.expires_in, 300);
    }

    #[test]
    fn parse_tolerates_missing_extras() {
        let body = r#"{"access_token":"x.y.z"}"#;
        let t = parse_exchange_response(body).unwrap();
        assert_eq!(t.access_token, "x.y.z");
        assert_eq!(t.expires_in, 0);
    }

    #[test]
    fn parse_rejects_missing_access_token() {
        let body = r#"{"token_type":"Bearer","expires_in":300}"#;
        assert!(parse_exchange_response(body).is_err());
    }

    #[test]
    fn parse_rejects_non_bearer_token_type() {
        let body = r#"{"access_token":"x","token_type":"DPoP","expires_in":300}"#;
        assert!(parse_exchange_response(body).is_err());
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
        };
        let rendered = format!("{t:?}");
        assert!(
            !rendered.contains("secret-bearer-value"),
            "ExchangedToken Debug leaked the access_token: {rendered}"
        );
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.contains("300"));
    }
}
