//! Fixed Federate challenge and token-exchange wire contract. The routes deny
//! while the authenticated Policy adapter and composite signer are absent.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine as _};
use hyprstream_rpc::auth::Scope;
#[cfg(feature = "postgres")]
use hyprstream_rpc::auth::{signer_suite::signer_suite_thumbprint, Claims, Cnf, CnfJwk};
#[cfg(feature = "postgres")]
use hyprstream_session_store::Session;
use serde::Deserialize;
#[cfg(feature = "postgres")]
use serde::Serialize;

use super::federate_source::{self, Commitment, Source, Verifier};

const PROFILE: &str = "federate-session-v1";
const ID_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:id_token";
const MAX_BODY: usize = 24 * 1024;
const MAX_SOURCE_TOKEN: usize = 16 * 1024;
const MAX_CODE: usize = 2048;

/// The exact A0 challenge envelope. `deny_unknown_fields` and serde's
/// duplicate-field rejection prevent browser-chosen authority from being
/// silently ignored or ambiguously interpreted.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChallengeRequest {
    profile: String,
    client_id: String,
    subject_token: String,
    subject_token_type: String,
    authorization_code: String,
    resource: String,
    scope: String,
    login_random: String,
    state: String,
    pkce_challenge: String,
    ed25519_public: String,
    ml_dsa65_public: String,
}

impl ChallengeRequest {
    pub(crate) fn parse(body: &[u8]) -> Result<Self, federate_source::Error> {
        if body.is_empty() || body.len() > MAX_BODY {
            return Err(federate_source::Error::Invalid);
        }
        let request: Self =
            serde_json::from_slice(body).map_err(|_| federate_source::Error::Invalid)?;
        if request.profile != PROFILE
            || request.client_id != federate_source::CLIENT
            || request.subject_token_type != ID_TOKEN_TYPE
            || request.resource != federate_source::HOST
            || request.subject_token.is_empty()
            || request.subject_token.len() > MAX_SOURCE_TOKEN
            || request.authorization_code.is_empty()
            || request.authorization_code.len() > MAX_CODE
            || !request.authorization_code.is_ascii()
        {
            return Err(federate_source::Error::Invalid);
        }
        federate_source::scopes(&request.scope)?;
        let requested: Vec<_> = request.scope.split(' ').collect();
        if requested.len() > 64
            || requested.iter().any(|text| {
                text.len() > 256
                    || Scope::parse(text).is_err()
                    || text
                        .split(':')
                        .any(|part| part.is_empty() || part.contains('*'))
            })
        {
            return Err(federate_source::Error::Invalid);
        }
        Ok(request)
    }

    /// Still only upstream identity evidence. The resulting Source must be
    /// mapped through Policy and paired with a challenge and both signatures.
    pub(crate) async fn verify_source(
        &self,
        verifier: &Verifier,
        now: u64,
    ) -> Result<Source, federate_source::Error> {
        let commitment = Commitment::new(
            &self.login_random,
            &self.state,
            &self.pkce_challenge,
            &self.ed25519_public,
            &self.ml_dsa65_public,
        )?;
        verifier
            .source(
                &self.subject_token,
                &self.authorization_code,
                commitment,
                now,
                None,
            )
            .await
    }

    pub(crate) fn requested_scope(&self) -> &str {
        &self.scope
    }

    pub(crate) fn subject_token(&self) -> &str {
        &self.subject_token
    }
}

const EXCHANGE_GRANT: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const ACCESS_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";

/// Exact public-client RFC 8693 form for this one staging profile. A generic
/// token-exchange request must never be reinterpreted as Federate authority.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExchangeRequest {
    grant_type: String,
    hs_profile: String,
    client_id: String,
    subject_token_type: String,
    subject_token: String,
    requested_token_type: String,
    resource: String,
    scope: String,
    challenge_id: String,
    ed25519_signature: String,
    ml_dsa65_signature: String,
}

impl ExchangeRequest {
    pub(crate) fn parse(body: &[u8]) -> Result<Self, federate_source::Error> {
        if body.is_empty() || body.len() > MAX_BODY {
            return Err(federate_source::Error::Invalid);
        }
        // serde_urlencoded does not reject duplicate form members for every
        // destination type. Count names before deserialization, including any
        // unknown field, so first/last-value ambiguity cannot enter this grant.
        let mut keys = std::collections::BTreeSet::new();
        for (key, _) in url::form_urlencoded::parse(body) {
            if !keys.insert(key.into_owned()) {
                return Err(federate_source::Error::Invalid);
            }
        }
        let request: Self =
            serde_urlencoded::from_bytes(body).map_err(|_| federate_source::Error::Invalid)?;
        if request.grant_type != EXCHANGE_GRANT
            || request.hs_profile != PROFILE
            || request.client_id != federate_source::CLIENT
            || request.subject_token_type != ID_TOKEN_TYPE
            || request.requested_token_type != ACCESS_TOKEN_TYPE
            || request.resource != federate_source::HOST
            || request.subject_token.is_empty()
            || request.subject_token.len() > MAX_SOURCE_TOKEN
        {
            return Err(federate_source::Error::Invalid);
        }
        federate_source::scopes(&request.scope)?;
        let requested: Vec<_> = request.scope.split(' ').collect();
        if requested.len() > 64
            || requested.iter().any(|text| {
                text.len() > 256
                    || Scope::parse(text).is_err()
                    || text
                        .split(':')
                        .any(|part| part.is_empty() || part.contains('*'))
            })
        {
            return Err(federate_source::Error::Invalid);
        }
        request.challenge_id()?;
        request.signatures()?;
        Ok(request)
    }

    pub(crate) fn challenge_id(&self) -> Result<[u8; 32], federate_source::Error> {
        let bytes = B64
            .decode(&self.challenge_id)
            .map_err(|_| federate_source::Error::Invalid)?;
        let id: [u8; 32] = bytes
            .try_into()
            .map_err(|_| federate_source::Error::Invalid)?;
        if B64.encode(id) != self.challenge_id {
            return Err(federate_source::Error::Invalid);
        }
        Ok(id)
    }

    pub(crate) fn signatures(&self) -> Result<(Vec<u8>, Vec<u8>), federate_source::Error> {
        let ed = B64
            .decode(&self.ed25519_signature)
            .map_err(|_| federate_source::Error::Invalid)?;
        let pq = B64
            .decode(&self.ml_dsa65_signature)
            .map_err(|_| federate_source::Error::Invalid)?;
        if ed.len() != 64
            || pq.len() != 3309
            || B64.encode(&ed) != self.ed25519_signature
            || B64.encode(&pq) != self.ml_dsa65_signature
        {
            return Err(federate_source::Error::Invalid);
        }
        Ok((ed, pq))
    }

    pub(crate) fn subject_token(&self) -> &str {
        &self.subject_token
    }

    pub(crate) fn requested_scope(&self) -> &str {
        &self.scope
    }
}

/// Only a committed Policy session can be passed to this constructor by the
/// host issuer. It derives every authority claim from the durable record.
#[cfg(feature = "postgres")]
pub(crate) fn committed_claims(
    session: &Session,
    now: i64,
) -> Result<Claims, federate_source::Error> {
    if session.host != federate_source::HOST
        || session.client_id != federate_source::CLIENT
        || session.resource != federate_source::HOST
        || session.sid.is_empty()
        || session.tenant.is_empty()
        || session.tenant == "*"
        || session.scopes.is_empty()
        || session.ed_public.iter().all(|byte| *byte == 0)
        || session.pq_public.len() != 1952
        || session.proof_epoch <= 0
        || session.expires_at <= now
        || session.expires_at > now.saturating_add(300)
    {
        return Err(federate_source::Error::Invalid);
    }
    let scope = session.scopes.join(" ");
    federate_source::scopes(&scope)?;
    if session.scopes.len() > 64
        || session.scopes.iter().any(|text| {
            text.len() > 256
                || Scope::parse(text).is_err()
                || text
                    .split(':')
                    .any(|part| part.is_empty() || part.contains('*'))
        })
    {
        return Err(federate_source::Error::Invalid);
    }
    let mut claims = Claims::new(session.subject.clone(), now, session.expires_at)
        .with_sid(session.sid.clone())
        .with_jti()
        .with_session_authority_generation(session.generation)
        .with_federate_signer_suite(B64.encode(signer_suite_thumbprint(
            hyprstream_session_store::SUITE,
            &[&session.ed_public, &session.pq_public],
        )));
    claims.iss = session.host.clone();
    claims.aud = Some(session.resource.clone());
    claims.client_id = Some(session.client_id.clone());
    claims.tenant = Some(session.tenant.clone());
    claims.scope = Some(scope);
    claims.cnf = Some(Cnf {
        jwk: Some(CnfJwk {
            kty: "OKP".into(),
            crv: "Ed25519".into(),
            x: B64.encode(session.ed_public),
        }),
        jkt: None,
        hs_signer_suite: None,
    });
    Ok(claims)
}

#[cfg(feature = "postgres")]
#[derive(Serialize)]
pub(crate) struct ExchangeResponse {
    access_token: String,
    issued_token_type: &'static str,
    token_type: &'static str,
    expires_in: i64,
    scope: String,
    session_id: String,
}

#[cfg(feature = "postgres")]
impl ExchangeResponse {
    pub(crate) fn committed(
        token: String,
        session: &Session,
        now: i64,
    ) -> Result<Self, federate_source::Error> {
        committed_claims(session, now)?;
        if token.is_empty() || token.len() > MAX_SOURCE_TOKEN * 2 {
            return Err(federate_source::Error::Invalid);
        }
        Ok(Self {
            access_token: token,
            issued_token_type: ACCESS_TOKEN_TYPE,
            token_type: "PoP",
            expires_in: session.expires_at - now,
            scope: session.scopes.join(" "),
            session_id: session.sid.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn exchange_form() -> String {
        url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", EXCHANGE_GRANT)
            .append_pair("hs_profile", PROFILE)
            .append_pair("client_id", federate_source::CLIENT)
            .append_pair("subject_token_type", ID_TOKEN_TYPE)
            .append_pair("subject_token", "header.claims.signature")
            .append_pair("requested_token_type", ACCESS_TOKEN_TYPE)
            .append_pair("resource", federate_source::HOST)
            .append_pair("scope", "query:registry:List")
            .append_pair("challenge_id", &B64.encode([7u8; 32]))
            .append_pair("ed25519_signature", &B64.encode([8u8; 64]))
            .append_pair("ml_dsa65_signature", &B64.encode(vec![9u8; 3309]))
            .finish()
    }

    #[test]
    fn exact_exchange_form_rejects_ambiguity_and_authority_fields() -> anyhow::Result<()> {
        let form = exchange_form();
        let parsed = ExchangeRequest::parse(form.as_bytes())?;
        assert_eq!(parsed.challenge_id()?, [7u8; 32]);
        assert_eq!(parsed.requested_scope(), "query:registry:List");
        for suffix in [
            "&scope=query%3Amodel%3AList",
            "&tenant=attacker",
            "&actor_token=attacker",
            "&hs_profile=generic",
            "&%68s_profile=federate-session-v1",
            "&resource=https%3A%2F%2Fattacker.invalid",
        ] {
            assert!(ExchangeRequest::parse(format!("{form}{suffix}").as_bytes()).is_err());
        }
        for (from, to) in [
            ("federate-session-v1", "other"),
            ("query%3Aregistry%3AList", "query%3Aregistry%3A%2A"),
            ("id_token", "access_token"),
        ] {
            assert!(ExchangeRequest::parse(form.replacen(from, to, 1).as_bytes()).is_err());
        }
        assert!(ExchangeRequest::parse(&vec![b'x'; MAX_BODY + 1]).is_err());
        assert!(ExchangeRequest::parse(
            form.replace("challenge_id=", "challenge_id=bad").as_bytes()
        )
        .is_err());
        assert!(ExchangeRequest::parse(
            form.replace("ed25519_signature=", "ed25519_signature=bad")
                .as_bytes()
        )
        .is_err());
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn committed_session() -> Session {
        Session {
            host: federate_source::HOST.into(),
            sid: "sid-1".into(),
            account_id: "account-1".into(),
            subject: "alice".into(),
            tenant: "tenant-1".into(),
            client_id: federate_source::CLIENT.into(),
            resource: federate_source::HOST.into(),
            scopes: vec!["query:registry:List".into()],
            grant_revision: "revision-1".into(),
            ed_public: [1; 32],
            pq_public: vec![2; 1952],
            generation: [3; 32],
            collision_inventory_id: [4; 32],
            proof_epoch: 1,
            expires_at: 1_800_000_120,
        }
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn committed_session_mints_only_exact_pop_claim_shape() -> anyhow::Result<()> {
        let session = committed_session();
        let claims = committed_claims(&session, 1_800_000_000)?;
        let value = serde_json::to_value(&claims)?;
        assert_eq!(value["hs_profile"], PROFILE);
        assert_eq!(value["iss"], federate_source::HOST);
        assert_eq!(value["aud"], federate_source::HOST);
        assert_eq!(value["sid"], session.sid);
        assert_eq!(value["scope"], "query:registry:List");
        assert_eq!(
            value["cnf"],
            json!({"jwk":{"kty":"OKP","crv":"Ed25519","x":B64.encode(session.ed_public)}})
        );
        assert_eq!(
            value["hs_signer_suite_v1"],
            B64.encode(signer_suite_thumbprint(
                hyprstream_session_store::SUITE,
                &[&session.ed_public, &session.pq_public]
            ))
        );
        let response = ExchangeResponse::committed("a.b.c".into(), &session, 1_800_000_000)?;
        let body = serde_json::to_value(response)?;
        assert_eq!(body["token_type"], "PoP");
        assert_eq!(body["issued_token_type"], ACCESS_TOKEN_TYPE);
        assert_eq!(body["expires_in"], 120);
        assert_eq!(body["session_id"], session.sid);
        let mut invalid = session.clone();
        invalid.proof_epoch = 0;
        assert!(committed_claims(&invalid, 1_800_000_000).is_err());
        invalid = session.clone();
        invalid.client_id = "other".into();
        assert!(committed_claims(&invalid, 1_800_000_000).is_err());
        invalid = session.clone();
        invalid.tenant = "*".into();
        assert!(committed_claims(&invalid, 1_800_000_000).is_err());
        Ok(())
    }

    fn request() -> Value {
        json!({
            "profile": PROFILE,
            "client_id": federate_source::CLIENT,
            "subject_token": "header.claims.signature",
            "subject_token_type": ID_TOKEN_TYPE,
            "authorization_code": "test-code",
            "resource": federate_source::HOST,
            "scope": "query:registry:List",
            "login_random": "a",
            "state": "b",
            "pkce_challenge": "c",
            "ed25519_public": "d",
            "ml_dsa65_public": "e"
        })
    }

    #[test]
    fn exact_profile_and_scope_envelope() -> anyhow::Result<()> {
        let valid = serde_json::to_vec(&request())?;
        assert_eq!(
            ChallengeRequest::parse(&valid)?.requested_scope(),
            "query:registry:List"
        );
        for (field, value) in [
            ("profile", "other"),
            ("client_id", "other"),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:access_token",
            ),
            ("resource", "https://other.example"),
            ("scope", "openid"),
            ("scope", "query:registry:List query:registry:List"),
            ("scope", "write:model:Load query:registry:List"),
            ("scope", "query:registry:List\\evil"),
        ] {
            let mut changed = request();
            changed[field] = value.into();
            assert!(ChallengeRequest::parse(&serde_json::to_vec(&changed)?).is_err());
        }
        Ok(())
    }

    #[test]
    fn malformed_and_ambiguous_json_denied() -> anyhow::Result<()> {
        assert!(ChallengeRequest::parse(b"").is_err());
        assert!(ChallengeRequest::parse(&vec![b' '; MAX_BODY + 1]).is_err());
        let valid = String::from_utf8(serde_json::to_vec(&request())?)?;
        let duplicate = valid.replacen("\"profile\":", "\"profile\":\"other\",\"profile\":", 1);
        assert!(ChallengeRequest::parse(duplicate.as_bytes()).is_err());
        let mut unknown = request();
        unknown["tenant"] = "attacker".into();
        assert!(ChallengeRequest::parse(&serde_json::to_vec(&unknown)?).is_err());
        let mut code = request();
        code["authorization_code"] = "nonascii-é".into();
        assert!(ChallengeRequest::parse(&serde_json::to_vec(&code)?).is_err());
        Ok(())
    }
}
