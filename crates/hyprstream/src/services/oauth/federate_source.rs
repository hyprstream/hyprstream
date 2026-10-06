//! Fixed staging Federate source/possession verification.
//!
//! This verifies cryptographic evidence, NOT account authority, callback execution,
//! PKCE redemption, key freshness, or global replay consumption. Policy must
//! supply authority and durable replay admission. No evidence is a credential.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use ed25519_dalek::{Signature, VerifyingKey};
use hyprstream_rpc::crypto::pq::{ml_dsa_verify, ml_dsa_vk_from_bytes};
use jsonwebtoken::{crypto, Algorithm, DecodingKey};
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};
use subtle::ConstantTimeEq;
use tokio::sync::Mutex;

pub(crate) const ISSUER: &str = "https://login.federate.to";
pub(crate) const CLIENT: &str = hyprstream_rpc::auth::claims::FEDERATE_STAGING_CLIENT;
pub(crate) const WEBSITE: &str = "https://www.staging.lab.hyprstream.com";
pub(crate) const CALLBACK: &str = "https://www.staging.lab.hyprstream.com/federate/callback";
pub(crate) const HOST: &str = hyprstream_rpc::auth::claims::FEDERATE_STAGING_HOST;
const JWKS: &str = "https://login.federate.to/keys";
const TOKEN_ENDPOINT: &str = "https://discovery.staging.lab.hyprstream.com/oauth/token";
const SUITE: &str = "hs-cose-sign-ed25519-mldsa65-wns-v1";
const MAX_JWS: usize = 16 * 1024;
const MAX_JWKS: usize = 64 * 1024;
const MAX_TRANSCRIPT: usize = 8 * 1024;
const MAX_AUTHORITY_TEXT: usize = 256;
const MAX_SCOPE_BYTES: usize = 1024;
const KEY_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Error {
    #[error("Federate verifier disabled")]
    Disabled,
    #[error("invalid Federate evidence")]
    Invalid,
    #[error("Federate signing authority unavailable")]
    Unavailable,
}
type Result<T> = std::result::Result<T, Error>;
fn require(ok: bool) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Invalid)
    }
}
fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn equal(a: &[u8], b: &[u8]) -> bool {
    bool::from(a.ct_eq(b))
}
fn decode(value: &str, max: usize) -> Result<Vec<u8>> {
    require(value.len() <= max.saturating_mul(4).div_ceil(3))?;
    let bytes = B64.decode(value).map_err(|_| Error::Invalid)?;
    require(bytes.len() <= max && B64.encode(&bytes) == value)?;
    Ok(bytes)
}
fn fixed<const N: usize>(value: &str) -> Result<[u8; N]> {
    decode(value, N)?.try_into().map_err(|_| Error::Invalid)
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    let s = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(Error::Invalid)?;
    require(!s.is_empty() && s.len() <= 2048)?;
    Ok(s)
}
fn frame(domain: &str, fields: &[&[u8]]) -> Vec<u8> {
    let mut out = domain.as_bytes().to_vec();
    out.push(0);
    out.extend_from_slice(&(fields.len() as u32).to_be_bytes());
    for field in fields {
        out.extend_from_slice(&(field.len() as u32).to_be_bytes());
        out.extend_from_slice(field);
    }
    out
}

// Reject duplicate members at every nesting level, including unknown claims.
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("unique-key JSON")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Unique, M::Error> {
                let mut out = serde_json::Map::new();
                while let Some((k, v)) = m.next_entry::<String, Unique>()? {
                    if out.insert(k, v.0).is_some() {
                        return Err(de::Error::custom("duplicate member"));
                    }
                }
                Ok(Unique(Value::Object(out)))
            }
            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut s: S,
            ) -> std::result::Result<Unique, S::Error> {
                let mut out = Vec::new();
                while let Some(v) = s.next_element::<Unique>()? {
                    out.push(v.0);
                }
                Ok(Unique(Value::Array(out)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::String(v.into())))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Bool(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
        }
        d.deserialize_any(V)
    }
}
fn json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice::<Unique>(bytes)
        .map(|v| v.0)
        .map_err(|_| Error::Invalid)
}

/// Validated commitment inputs. The host cannot prove browser key freshness.
pub(crate) struct Commitment {
    random: [u8; 32],
    state: String,
    pkce: String,
    ed: [u8; 32],
    pq: Vec<u8>,
}
impl Commitment {
    pub(crate) fn new(random: &str, state: &str, pkce: &str, ed: &str, pq: &str) -> Result<Self> {
        let ed = fixed(ed)?;
        let vk = VerifyingKey::from_bytes(&ed).map_err(|_| Error::Invalid)?;
        require(!vk.is_weak())?;
        let pq = decode(pq, 1952)?;
        ml_dsa_vk_from_bytes(&pq).map_err(|_| Error::Invalid)?;
        fixed::<32>(state)?;
        fixed::<32>(pkce)?;
        Ok(Self {
            random: fixed(random)?,
            state: state.into(),
            pkce: pkce.into(),
            ed,
            pq,
        })
    }
    fn preimage(&self) -> Vec<u8> {
        frame(
            "hyprstream.federate.session-primary.nonce.v1",
            &[
                ISSUER.as_bytes(),
                CLIENT.as_bytes(),
                WEBSITE.as_bytes(),
                CALLBACK.as_bytes(),
                HOST.as_bytes(),
                HOST.as_bytes(),
                &self.random,
                &self.ed,
                &self.pq,
                &sha(self.state.as_bytes()),
                self.pkce.as_bytes(),
            ],
        )
    }
    fn nonce(&self) -> String {
        format!("hsn1.{}", B64.encode(sha(&self.preimage())))
    }
    fn kids(&self) -> [[u8; 32]; 2] {
        session_primary_kids(&self.ed, &self.pq)
    }
}

// Shared by possession verification and the disabled request-proof consumer.
pub(super) fn session_primary_kids(ed: &[u8; 32], pq: &[u8]) -> [[u8; 32]; 2] {
    [
        sha(&frame(
            "hyprstream.session-primary.kid.v1",
            &[b"Ed25519", ed],
        )),
        sha(&frame(
            "hyprstream.session-primary.kid.v1",
            &[b"ML-DSA-65", pq],
        )),
    ]
}

struct Keys {
    fetched: Instant,
    keys: BTreeMap<String, DecodingKey>,
}
fn parse_keys(bytes: &[u8]) -> Result<BTreeMap<String, DecodingKey>> {
    require(bytes.len() <= MAX_JWKS)?;
    let doc = json(bytes)?;
    let entries = doc
        .get("keys")
        .and_then(Value::as_array)
        .ok_or(Error::Invalid)?;
    require(!entries.is_empty() && entries.len() <= 16)?;
    let mut keys = BTreeMap::new();
    for entry in entries {
        let obj = entry.as_object().ok_or(Error::Invalid)?;
        require(
            obj.keys()
                .all(|k| ["kid", "kty", "use", "alg", "n", "e", "key_ops"].contains(&k.as_str())),
        )?;
        require(text(entry, "kty")? == "RSA")?;
        for (k, expected) in [("alg", "RS256"), ("use", "sig")] {
            if let Some(v) = entry.get(k) {
                require(v.as_str() == Some(expected))?;
            }
        }
        if let Some(ops) = entry.get("key_ops") {
            require(*ops == serde_json::json!(["verify"]))?;
        }
        let n = text(entry, "n")?;
        let e = text(entry, "e")?;
        let modulus = decode(n, 512)?;
        require(
            (256..=512).contains(&modulus.len())
                && modulus[0] >= 128
                && modulus.last().is_some_and(|b| b & 1 == 1),
        )?;
        let exponent = decode(e, 4)?;
        require(!exponent.is_empty() && exponent[0] != 0)?;
        let exponent = exponent.iter().fold(0u64, |n, b| (n << 8) | u64::from(*b));
        require(exponent >= 3 && exponent % 2 == 1)?;
        let key = DecodingKey::from_rsa_components(n, e).map_err(|_| Error::Invalid)?;
        require(keys.insert(text(entry, "kid")?.into(), key).is_none())?;
    }
    Ok(keys)
}

/// Disabled by default. No application constructs/enables this component yet.
#[derive(Default)]
pub(crate) struct Verifier {
    client: Option<reqwest::Client>,
    keys: Mutex<Option<Keys>>,
    #[cfg(test)]
    responses: Option<parking_lot::Mutex<Vec<Result<Vec<u8>>>>>,
}
impl Verifier {
    pub(crate) fn fixed_staging() -> Result<Self> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|_| Error::Unavailable)?;
        Ok(Self {
            client: Some(client),
            keys: Mutex::new(None),
            #[cfg(test)]
            responses: None,
        })
    }
    async fn fetch(&self, client: &reqwest::Client) -> Result<Vec<u8>> {
        #[cfg(test)]
        if let Some(responses) = &self.responses {
            let mut responses = responses.lock();
            return if responses.is_empty() {
                Err(Error::Unavailable)
            } else {
                responses.remove(0)
            };
        }
        let mut response = client
            .get(JWKS)
            .send()
            .await
            .map_err(|_| Error::Unavailable)?;
        if !response.status().is_success()
            || response
                .content_length()
                .is_some_and(|n| n > MAX_JWKS as u64)
        {
            return Err(Error::Unavailable);
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| Error::Unavailable)? {
            if bytes.len() + chunk.len() > MAX_JWKS {
                return Err(Error::Unavailable);
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
    async fn key(&self, kid: &str) -> Result<DecodingKey> {
        let client = self.client.as_ref().ok_or(Error::Disabled)?;
        // A single-flight cache; the deadline also bounds queue wait/body reads.
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut cache = self.keys.lock().await;
            if let Some(keys) = cache.as_ref().filter(|k| k.fetched.elapsed() < KEY_TTL) {
                if let Some(key) = keys.keys.get(kid) {
                    return Ok(key.clone());
                }
            }
            // One fixed-authority refresh per lookup; never follow JWS URLs.
            let bytes = self.fetch(client).await?;
            let keys = parse_keys(&bytes).map_err(|_| Error::Unavailable)?;
            let key = keys.get(kid).cloned().ok_or(Error::Invalid);
            *cache = Some(Keys {
                fetched: Instant::now(),
                keys,
            });
            key
        })
        .await
        .map_err(|_| Error::Unavailable)?
    }
    pub(crate) async fn source(
        &self,
        token: &str,
        code: &str,
        commitment: Commitment,
        now: u64,
        access_token: Option<&str>,
    ) -> Result<Source> {
        require(token.len() <= MAX_JWS && !token.is_empty())?;
        let parts: Vec<_> = token.split('.').collect();
        require(parts.len() == 3)?;
        let header = json(&decode(parts[0], 2048)?)?;
        require(
            header
                .as_object()
                .ok_or(Error::Invalid)?
                .keys()
                .all(|k| ["alg", "kid", "typ"].contains(&k.as_str())),
        )?;
        require(text(&header, "alg")? == "RS256")?;
        if let Some(typ) = header.get("typ") {
            require(typ.as_str() == Some("JWT"))?;
        }
        let claims = json(&decode(parts[1], MAX_JWS)?)?;
        let sig = decode(parts[2], 512)?;
        require((256..=512).contains(&sig.len()))?;
        let key = self.key(text(&header, "kid")?).await?;
        let signed = &token[..token.len() - parts[2].len() - 1];
        require(
            crypto::verify(parts[2], signed.as_bytes(), &key, Algorithm::RS256)
                .map_err(|_| Error::Invalid)?,
        )?;
        validate_claims(&claims, code, &commitment, now, access_token)?;
        Ok(Source {
            subject: text(&claims, "sub")?.into(),
            jti: text(&claims, "jti")?.into(),
            issued_at: number(&claims, "iat")?,
            expires_at: number(&claims, "exp")?,
            token_hash: sha(token.as_bytes()),
            code_hash: sha(code.as_bytes()),
            commitment,
        })
    }
}
fn number(v: &Value, k: &str) -> Result<u64> {
    v.get(k).and_then(Value::as_u64).ok_or(Error::Invalid)
}
fn times(iat: u64, exp: u64, now: u64) -> Result<()> {
    require(
        exp <= i64::MAX as u64
            && exp > iat
            && exp - iat <= 300
            && iat <= now.saturating_add(30)
            && now.saturating_sub(iat) <= 120
            && now < exp,
    )
}
fn hash_claim(claim: &str, input: &str) -> Result<()> {
    require(!input.is_empty() && input.len() <= MAX_JWS && input.is_ascii())?;
    require(equal(&fixed::<16>(claim)?, &sha(input.as_bytes())[..16]))
}
fn validate_claims(
    v: &Value,
    code: &str,
    c: &Commitment,
    now: u64,
    access: Option<&str>,
) -> Result<()> {
    require(text(v, "iss")? == ISSUER)?;
    require(
        v.get("aud")
            .is_some_and(|a| a.as_str() == Some(CLIENT) || *a == serde_json::json!([CLIENT])),
    )?;
    if let Some(azp) = v.get("azp") {
        require(azp.as_str() == Some(CLIENT))?;
    }
    text(v, "sub")?;
    text(v, "jti")?;
    let iat = number(v, "iat")?;
    let exp = number(v, "exp")?;
    times(iat, exp, now)?;
    if v.get("nbf").is_some() {
        let n = number(v, "nbf")?;
        require(n <= now.saturating_add(30) && n <= exp)?;
    }
    require(equal(text(v, "nonce")?.as_bytes(), c.nonce().as_bytes()))?;
    hash_claim(text(v, "c_hash")?, code)?;
    if v.get("at_hash").is_some() {
        let claim = text(v, "at_hash")?;
        fixed::<16>(claim)?;
        if let Some(access) = access {
            hash_claim(claim, access)?;
        }
    }
    Ok(())
}

/// Private fields, no Deserialize/Clone/Debug: not forgeable from a browser flag.
pub(crate) struct Source {
    subject: String,
    jti: String,
    issued_at: u64,
    expires_at: u64,
    token_hash: [u8; 32],
    code_hash: [u8; 32],
    commitment: Commitment,
}
impl Source {
    /// The nonce-committed proof keys must be pinned by Policy at prepare time.
    /// Only a verified `Source` exposes this pair to the authenticated adapter.
    #[allow(dead_code)] // Consumed by the B1b Policy adapter, not the disabled issuer.
    pub(crate) fn proof_public_keys(&self) -> (&[u8; 32], &[u8]) {
        (&self.commitment.ed, &self.commitment.pq)
    }

    #[cfg(feature = "postgres")]
    pub(crate) fn store_source(&self) -> hyprstream_session_store::Source {
        hyprstream_session_store::Source {
            issuer: ISSUER.into(),
            subject: self.subject.clone(),
            jti: self.jti.clone(),
            nonce: self.commitment.nonce(),
            token_hash: self.token_hash,
            issued_at: self.issued_at as i64,
            expires_at: self.expires_at as i64,
        }
    }
}
/// Supplied only by a future Policy integration; this verifier does NOT authorize it.
#[derive(Clone)]
pub(crate) struct AuthorityBinding {
    pub account: String,
    pub subject: String,
    pub tenant: String,
    pub requested: String,
    pub granted: String,
    pub revision: String,
}
pub(crate) fn scopes(s: &str) -> Result<()> {
    // Keep the browser and verifier scope grammar/limit identical. Scope
    // tokens are printable ASCII separated by exactly one ASCII space.
    require(!s.is_empty() && s.len() <= MAX_SCOPE_BYTES)?;
    let mut previous = "";
    for token in s.split(' ') {
        require(
            !token.is_empty()
                && token
                    .bytes()
                    .all(|b| b == 0x21 || (0x23..=0x5b).contains(&b) || (0x5d..=0x7e).contains(&b))
                && token > previous,
        )?;
        previous = token;
    }
    Ok(())
}
fn authority_text(s: &str) -> Result<()> {
    require(
        !s.is_empty()
            && s.len() <= MAX_AUTHORITY_TEXT
            && !s.bytes().any(|b| b <= 0x1f || b == 0x7f),
    )
}
pub(crate) struct Challenge {
    source: Source,
    binding: AuthorityBinding,
    id: [u8; 32],
    created: u64,
    expires: u64,
    transcript: Vec<u8>,
}
impl Source {
    pub(crate) fn challenge(
        self,
        binding: AuthorityBinding,
        id: [u8; 32],
        created: u64,
    ) -> Result<Challenge> {
        times(self.issued_at, self.expires_at, created)?;
        for s in [
            &binding.account,
            &binding.subject,
            &binding.tenant,
            &binding.revision,
        ] {
            authority_text(s)?;
        }
        require(binding.tenant != "*")?;
        scopes(&binding.requested)?;
        scopes(&binding.granted)?;
        require(
            binding
                .granted
                .split(' ')
                .all(|s| binding.requested.split(' ').any(|r| r == s)),
        )?;
        let expires = created
            .saturating_add(60)
            .min(self.expires_at)
            .min(self.issued_at.saturating_add(120));
        require(expires > created)?;
        let kids = self.commitment.kids();
        let transcript = frame(
            "hyprstream.session-primary.possession.v1",
            &[
                HOST.as_bytes(),
                TOKEN_ENDPOINT.as_bytes(),
                ISSUER.as_bytes(),
                CLIENT.as_bytes(),
                WEBSITE.as_bytes(),
                &id,
                &self.token_hash,
                self.commitment.nonce().as_bytes(),
                &self.code_hash,
                &sha(&self.commitment.preimage()),
                binding.account.as_bytes(),
                binding.subject.as_bytes(),
                binding.tenant.as_bytes(),
                HOST.as_bytes(),
                binding.requested.as_bytes(),
                binding.granted.as_bytes(),
                &created.to_be_bytes(),
                &expires.to_be_bytes(),
                SUITE.as_bytes(),
                &kids[0],
                &kids[1],
                binding.revision.as_bytes(),
            ],
        );
        // Match the A2 browser verifier's hard bound even if this encoding is
        // extended later; never hand back a transcript that the browser will
        // reject or truncate.
        require(transcript.len() <= MAX_TRANSCRIPT)?;
        Ok(Challenge {
            source: self,
            binding,
            id,
            created,
            expires,
            transcript,
        })
    }
}
/// Consuming a challenge prevents local reuse only. Durable replay is H1's job.
pub(crate) struct VerifiedPossession {
    source: Source,
    binding: AuthorityBinding,
    challenge_id: [u8; 32],
    created_at: u64,
    expires_at: u64,
}
impl Challenge {
    pub(crate) fn transcript(&self) -> &[u8] {
        &self.transcript
    }
    pub(crate) fn expires_at(&self) -> u64 {
        self.expires
    }
    pub(crate) fn challenge_id(&self) -> [u8; 32] {
        self.id
    }
    pub(crate) fn granted_scope(&self) -> &str {
        &self.binding.granted
    }
    pub(crate) fn expected_session(&self) -> (AuthorityBinding, [u8; 32], Vec<u8>) {
        (
            self.binding.clone(),
            self.source.commitment.ed,
            self.source.commitment.pq.clone(),
        )
    }
    pub(crate) fn verify(
        self,
        source: Source,
        id: &[u8; 32],
        ed_sig: &[u8],
        pq_sig: &[u8],
        now: u64,
    ) -> Result<VerifiedPossession> {
        require(
            now >= self.created
                && now < self.expires
                && equal(id, &self.id)
                && equal(&source.token_hash, &self.source.token_hash)
                && equal(&source.code_hash, &self.source.code_hash)
                && equal(
                    &source.commitment.preimage(),
                    &self.source.commitment.preimage(),
                ),
        )?;
        times(source.issued_at, source.expires_at, now)?;
        require(ed_sig.len() == 64 && pq_sig.len() == 3309)?;
        let c = &source.commitment;
        let vk = VerifyingKey::from_bytes(&c.ed).map_err(|_| Error::Invalid)?;
        vk.verify_strict(
            &self.transcript,
            &Signature::from_slice(ed_sig).map_err(|_| Error::Invalid)?,
        )
        .map_err(|_| Error::Invalid)?;
        ml_dsa_verify(
            &ml_dsa_vk_from_bytes(&c.pq).map_err(|_| Error::Invalid)?,
            &self.transcript,
            pq_sig,
        )
        .map_err(|_| Error::Invalid)?;
        Ok(VerifiedPossession {
            source,
            binding: self.binding,
            challenge_id: self.id,
            created_at: self.created,
            expires_at: self.expires,
        })
    }
}

impl VerifiedPossession {
    /// Consumed only by the authenticated Policy admission adapter. The
    /// verified keys/source are never reconstructed from the exchange form.
    #[cfg(feature = "postgres")]
    pub(crate) fn into_admission_parts(
        self,
    ) -> (
        hyprstream_session_store::Source,
        [u8; 32],
        Vec<u8>,
        AuthorityBinding,
        [u8; 32],
        u64,
        u64,
    ) {
        let source = self.source.store_source();
        (
            source,
            self.source.commitment.ed,
            self.source.commitment.pq,
            self.binding,
            self.challenge_id,
            self.created_at,
            self.expires_at,
        )
    }
}

#[cfg(test)]
mod tests;
