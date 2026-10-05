//! Source-only boundary for the fixed Federate challenge request.
//!
//! This module deliberately has no route, runtime factory, admission adapter, or
//! signer. Parsing a request and validating a Dex source are not authority to
//! issue a host token. Policy and the shared PostgreSQL admission must complete
//! before this profile can be exposed.

use hyprstream_rpc::auth::Scope;
use serde::Deserialize;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

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
