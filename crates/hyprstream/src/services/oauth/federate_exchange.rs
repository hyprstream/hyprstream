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

use super::federate_source::{self, ClientClass, Commitment, Source, Verifier};

const PROFILE: &str = "federate-session-v1";
const ID_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:id_token";
const MAX_BODY: usize = 24 * 1024;
const MAX_SOURCE_TOKEN: usize = 16 * 1024;
const MAX_CODE: usize = 2048;

/// Resolve a wire `client_id` against the classes this route accepts: the
/// hardcoded staging www pin plus the optional configured TUI class
/// (`None` = staging-only, exactly the previous behavior). Anything else
/// fails closed before any signature or claim work. Callers that hold a
/// [`Verifier`] must prefer `Verifier::resolve_client_class`, which applies
/// the same rule against the verifier's own configured class.
fn resolve_wire_class(
    client_id: &str,
    tui_client: Option<&ClientClass>,
) -> Result<ClientClass, federate_source::Error> {
    if client_id == federate_source::CLIENT {
        return Ok(ClientClass::staging());
    }
    match tui_client {
        Some(class) if class.client_id() == client_id => Ok(class.clone()),
        _ => Err(federate_source::Error::Invalid),
    }
}

#[cfg(all(test, feature = "postgres"))]
pub(crate) fn committed_claims(
    session: &Session,
    now: i64,
    issuer: &str,
    subject: &str,
) -> Result<Claims, federate_source::Error> {
    committed_claims_for_host(
        session,
        now,
        issuer,
        subject,
        &federate_source::FederateHost::new(federate_source::HOST)
            .map_err(|_| federate_source::Error::Invalid)?,
    )
}

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
    #[cfg(test)]
    pub(crate) fn parse(body: &[u8]) -> Result<Self, federate_source::Error> {
        Self::parse_for_host(
            body,
            &federate_source::FederateHost::new(federate_source::HOST)
                .map_err(|_| federate_source::Error::Invalid)?,
            None,
        )
    }
    pub(crate) fn parse_for_host(
        body: &[u8],
        host: &federate_source::FederateHost,
        tui_client: Option<&ClientClass>,
    ) -> Result<Self, federate_source::Error> {
        if body.is_empty() || body.len() > MAX_BODY {
            return Err(federate_source::Error::Invalid);
        }
        let request: Self =
            serde_json::from_slice(body).map_err(|_| federate_source::Error::Invalid)?;
        // The wire client_id selects the accepted class, and the profile
        // string must be exactly that class's profile (`federate-session-v1`
        // for www, `federate-tui-v1` for the configured TUI class). An
        // unconfigured/unknown client fails closed here.
        let class = resolve_wire_class(&request.client_id, tui_client)?;
        if request.profile != class.profile()
            || request.subject_token_type != ID_TOKEN_TYPE
            || request.resource != host.as_str()
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
        if self.resource != verifier.host()?.as_str() {
            return Err(federate_source::Error::Invalid);
        }
        // Build the class from the verifier's own canonical resolution —
        // never from raw request fields. `Verifier::source` re-resolves the
        // class from the commitment and rejects anything the verifier did
        // not configure (W1.1 acceptance check).
        let class = verifier.resolve_client_class(&self.client_id)?;
        let commitment = Commitment::for_class(
            verifier.host()?.clone(),
            &class,
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

    /// The class-resolved wire client id (validated against the accepted
    /// classes at parse time).
    pub(crate) fn client_id(&self) -> &str {
        &self.client_id
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
    #[cfg(test)]
    pub(crate) fn parse(body: &[u8]) -> Result<Self, federate_source::Error> {
        Self::parse_for_host(
            body,
            &federate_source::FederateHost::new(federate_source::HOST)
                .map_err(|_| federate_source::Error::Invalid)?,
            None,
        )
    }
    pub(crate) fn parse_for_host(
        body: &[u8],
        host: &federate_source::FederateHost,
        tui_client: Option<&ClientClass>,
    ) -> Result<Self, federate_source::Error> {
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
        // Per-class profile pin: the wire client_id selects the accepted
        // class and hs_profile must be exactly that class's profile string.
        let class = resolve_wire_class(&request.client_id, tui_client)?;
        if request.grant_type != EXCHANGE_GRANT
            || request.hs_profile != class.profile()
            || request.subject_token_type != ID_TOKEN_TYPE
            || request.requested_token_type != ACCESS_TOKEN_TYPE
            || request.resource != host.as_str()
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

    /// The class-resolved wire client id (validated against the accepted
    /// classes at parse time).
    pub(crate) fn client_id(&self) -> &str {
        &self.client_id
    }
}

/// Shared durable-record shape validation for both client classes. The
/// per-class mint (`committed_claims_for_host` / `committed_tui_claims_for_host`)
/// additionally pins the exact client and its marker set.
#[cfg(feature = "postgres")]
fn committed_session_shape(
    session: &Session,
    now: i64,
    source_issuer: &str,
    source_subject: &str,
    host: &federate_source::FederateHost,
) -> Result<(), federate_source::Error> {
    if session.host != host.as_str()
        || session.resource != host.as_str()
        || session.sid.is_empty()
        || session.tenant.is_empty()
        || session.tenant == "*"
        || session.scopes.is_empty()
        || session.ed_public.iter().all(|byte| *byte == 0)
        || session.pq_public.len() != 1952
        || session.proof_epoch <= 0
        || session.generation != session.collision_inventory_id
        || session.expires_at <= now
        || session.expires_at > now.saturating_add(300)
        || source_issuer != federate_source::ISSUER
        || source_subject.is_empty()
        || source_subject.len() > 256
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
    Ok(())
}

/// Only a committed Policy session can be passed to this constructor by the
/// host issuer. It derives every authority claim from the durable record.
/// The www staging mint: byte-for-byte the previous claim shape.
#[cfg(feature = "postgres")]
pub(crate) fn committed_claims_for_host(
    session: &Session,
    now: i64,
    source_issuer: &str,
    source_subject: &str,
    host: &federate_source::FederateHost,
) -> Result<Claims, federate_source::Error> {
    if session.client_id != federate_source::CLIENT {
        return Err(federate_source::Error::Invalid);
    }
    committed_session_shape(session, now, source_issuer, source_subject, host)?;
    let scope = session.scopes.join(" ");
    let mut claims = Claims::new(session.subject.clone(), now, session.expires_at)
        .with_sid(session.sid.clone())
        .with_jti()
        .with_federate_signer_suite(B64.encode(signer_suite_thumbprint(
            hyprstream_session_store::SUITE,
            &[&session.ed_public, &session.pq_public],
        )))
        .with_federate_signer_inventory(session.collision_inventory_id);
    claims.iss = session.host.clone();
    claims.aud = Some(session.resource.clone());
    claims.client_id = Some(session.client_id.clone());
    claims.tenant = Some(session.tenant.clone());
    claims.scope = Some(scope);
    claims.hs_signer_pq_key_v1 = Some(B64.encode(&session.pq_public));
    claims.hs_source_issuer_v1 = Some(source_issuer.to_owned());
    claims.hs_source_subject_v1 = Some(source_subject.to_owned());
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

/// The TUI-class mint (W1.3): same durable-record shape, but stamps the
/// TUI-consumer profile markers — namespaced `hs_tui_*` and DISTINCT from
/// the www v1 suite/inventory markers — so the credential takes NORMAL JWT
/// verification (never the svc.rs deferred federate branch) and carries NO
/// registry|model consumer authority: every granted scope must target the
/// `tui` service resource only.
#[cfg(feature = "postgres")]
pub(crate) fn committed_tui_claims_for_host(
    session: &Session,
    now: i64,
    source_issuer: &str,
    source_subject: &str,
    host: &federate_source::FederateHost,
    tui_client: &str,
) -> Result<Claims, federate_source::Error> {
    if session.client_id != tui_client {
        return Err(federate_source::Error::Invalid);
    }
    committed_session_shape(session, now, source_issuer, source_subject, host)?;
    // The TUI consumer profile cannot mint registry|model authority. This is
    // the mint-side floor under Policy's per-client scope ceiling: even a
    // Policy bug that granted a www scope to the TUI class stops here.
    if session.scopes.iter().any(|text| {
        Scope::parse(text)
            .map(|scope| scope.resource != "tui")
            .unwrap_or(true)
    }) {
        return Err(federate_source::Error::Invalid);
    }
    let scope = session.scopes.join(" ");
    let mut claims = Claims::new(session.subject.clone(), now, session.expires_at)
        .with_sid(session.sid.clone())
        .with_jti()
        .with_federate_tui_markers(
            B64.encode(&session.pq_public),
            source_issuer.to_owned(),
            source_subject.to_owned(),
        );
    claims.iss = session.host.clone();
    claims.aud = Some(session.resource.clone());
    claims.client_id = Some(session.client_id.clone());
    claims.tenant = Some(session.tenant.clone());
    claims.scope = Some(scope);
    // Holder binding stays the standard RFC 7800 `cnf.jwk` (the envelope
    // signer must match it at dispatch); no www signer-suite/inventory
    // digest is stamped for this class.
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

/// Dispatch a committed session to its per-class mint by the session's
/// Policy-bound `client_id`. An unconfigured TUI client (or any other
/// client) fails closed: the W1.1 default-deny invariant.
#[cfg(feature = "postgres")]
pub(crate) fn committed_claims_for_class(
    session: &Session,
    now: i64,
    source_issuer: &str,
    source_subject: &str,
    host: &federate_source::FederateHost,
    tui_client: Option<&str>,
) -> Result<Claims, federate_source::Error> {
    if session.client_id == federate_source::CLIENT {
        committed_claims_for_host(session, now, source_issuer, source_subject, host)
    } else {
        match tui_client {
            Some(client) if client == session.client_id => committed_tui_claims_for_host(
                session,
                now,
                source_issuer,
                source_subject,
                host,
                client,
            ),
            _ => Err(federate_source::Error::Invalid),
        }
    }
}

/// Attach only the exact subject's authority-enrolled MAC clearance. Policy's
/// action grants do not imply a clearance, and browser inputs cannot select it.
#[cfg(feature = "postgres")]
pub(crate) fn authorized_committed_claims_for_host(
    session: &Session,
    now: i64,
    source_issuer: &str,
    source_subject: &str,
    resolver: &dyn super::token_exchange::SubjectContextResolver,
    host: &federate_source::FederateHost,
    tui_client: Option<&str>,
) -> Result<Claims, federate_source::Error> {
    let claims = committed_claims_for_class(
        session,
        now,
        source_issuer,
        source_subject,
        host,
        tui_client,
    )?;
    let context = resolver
        .resolve(&claims.sub)
        .ok_or(federate_source::Error::Unavailable)?;
    Ok(claims.with_clearance(*context.clearance()))
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
    #[cfg(test)]
    pub(crate) fn committed(
        token: String,
        session: &Session,
        now: i64,
        issuer: &str,
        subject: &str,
    ) -> Result<Self, federate_source::Error> {
        Self::committed_for_host(
            token,
            session,
            now,
            issuer,
            subject,
            &federate_source::FederateHost::new(federate_source::HOST)
                .map_err(|_| federate_source::Error::Invalid)?,
            None,
        )
    }
    pub(crate) fn committed_for_host(
        token: String,
        session: &Session,
        now: i64,
        source_issuer: &str,
        source_subject: &str,
        host: &federate_source::FederateHost,
        tui_client: Option<&str>,
    ) -> Result<Self, federate_source::Error> {
        committed_claims_for_class(
            session,
            now,
            source_issuer,
            source_subject,
            host,
            tui_client,
        )?;
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
#[allow(clippy::unwrap_used)]
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
            collision_inventory_id: [3; 32],
            proof_epoch: 1,
            expires_at: 1_800_000_120,
        }
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn committed_session_mints_only_exact_pop_claim_shape() -> anyhow::Result<()> {
        let session = committed_session();
        let claims = committed_claims(
            &session,
            1_800_000_000,
            federate_source::ISSUER,
            "opaque-atproto-connector-subject",
        )?;
        let value = serde_json::to_value(&claims)?;
        assert_eq!(value["hs_profile"], PROFILE);
        assert_eq!(value["iss"], federate_source::HOST);
        assert_eq!(value["aud"], federate_source::HOST);
        assert_eq!(value["sid"], session.sid);
        assert_eq!(value["scope"], "query:registry:List");
        assert_eq!(value["hs_signer_pq_key_v1"], B64.encode(&session.pq_public));
        assert_eq!(
            value["hs_signer_inventory_v1"],
            B64.encode(session.collision_inventory_id)
        );
        assert!(value.get("hs_session_authority_generation").is_none());
        assert_eq!(value["hs_source_issuer_v1"], federate_source::ISSUER);
        assert_eq!(
            value["hs_source_subject_v1"],
            "opaque-atproto-connector-subject"
        );
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
        let response = ExchangeResponse::committed(
            "a.b.c".into(),
            &session,
            1_800_000_000,
            federate_source::ISSUER,
            "opaque-atproto-connector-subject",
        )?;
        let body = serde_json::to_value(response)?;
        assert_eq!(body["token_type"], "PoP");
        assert_eq!(body["issued_token_type"], ACCESS_TOKEN_TYPE);
        assert_eq!(body["expires_in"], 120);
        assert_eq!(body["session_id"], session.sid);
        let mut invalid = session.clone();
        invalid.proof_epoch = 0;
        assert!(
            committed_claims(&invalid, 1_800_000_000, federate_source::ISSUER, "subject").is_err()
        );
        invalid = session.clone();
        invalid.client_id = "other".into();
        assert!(
            committed_claims(&invalid, 1_800_000_000, federate_source::ISSUER, "subject").is_err()
        );
        invalid = session.clone();
        invalid.tenant = "*".into();
        assert!(
            committed_claims(&invalid, 1_800_000_000, federate_source::ISSUER, "subject").is_err()
        );
        assert!(committed_claims(
            &session,
            1_800_000_000,
            "https://attacker.invalid",
            "subject"
        )
        .is_err());
        assert!(committed_claims(&session, 1_800_000_000, federate_source::ISSUER, "").is_err());
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
    fn configured_resource_is_exact_for_challenge_and_exchange() -> anyhow::Result<()> {
        for local in ["https://vm-a.example", "https://vm-b.example:8443"] {
            let host = federate_source::FederateHost::new(local)?;
            let mut body = request();
            body["resource"] = local.into();
            assert!(
                ChallengeRequest::parse_for_host(&serde_json::to_vec(&body)?, &host, None).is_ok()
            );
            let form: String = url::form_urlencoded::Serializer::new(String::new())
                .extend_pairs(url::form_urlencoded::parse(exchange_form().as_bytes()).map(
                    |(k, v)| {
                        let v = if k == "resource" {
                            local.to_owned()
                        } else {
                            v.into_owned()
                        };
                        (k.into_owned(), v)
                    },
                ))
                .finish();
            assert!(ExchangeRequest::parse_for_host(form.as_bytes(), &host, None).is_ok());
            let other = federate_source::FederateHost::new("https://other.example")?;
            assert!(
                ChallengeRequest::parse_for_host(&serde_json::to_vec(&body)?, &other, None)
                    .is_err()
            );
            assert!(ExchangeRequest::parse_for_host(form.as_bytes(), &other, None).is_err());
        }
        Ok(())
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn configured_claims_reject_policy_record_from_another_host() -> anyhow::Result<()> {
        for local in ["https://vm-a.example", "https://vm-b.example:8443"] {
            let host = federate_source::FederateHost::new(local)?;
            let other = federate_source::FederateHost::new("https://other.example")?;
            let mut session = committed_session();
            session.host = local.into();
            session.resource = local.into();
            let claims = committed_claims_for_host(
                &session,
                1_800_000_000,
                federate_source::ISSUER,
                "source-sub",
                &host,
            )?;
            assert_eq!(claims.iss, local);
            assert_eq!(claims.aud.as_deref(), Some(local));
            assert!(committed_claims_for_host(
                &session,
                1_800_000_000,
                federate_source::ISSUER,
                "source-sub",
                &other
            )
            .is_err());
            session.resource = other.as_str().into();
            assert!(committed_claims_for_host(
                &session,
                1_800_000_000,
                federate_source::ISSUER,
                "source-sub",
                &host
            )
            .is_err());
        }
        Ok(())
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

    // ── W1.3: the TUI client class on the challenge/exchange wire ──────────

    const TUI_CLIENT_ID: &str = "hyprstream-tui-staging";
    const TUI_PROFILE_STR: &str = "federate-tui-v1";

    fn tui_class() -> federate_source::ClientClass {
        federate_source::ClientClass::new(TUI_CLIENT_ID, "hyprstream-tui-aud", None).unwrap()
    }

    #[test]
    fn w13_wire_accepts_the_tui_class_with_its_own_profile_string() -> anyhow::Result<()> {
        let host = federate_source::FederateHost::new("https://vm-tui.example")?;
        let mut body = request();
        body["client_id"] = TUI_CLIENT_ID.into();
        body["profile"] = TUI_PROFILE_STR.into();
        body["resource"] = "https://vm-tui.example".into();
        // Configured class + exact profile string: accepted.
        assert!(ChallengeRequest::parse_for_host(
            &serde_json::to_vec(&body)?,
            &host,
            Some(&tui_class())
        )
        .is_ok());
        let form: String = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(
                url::form_urlencoded::parse(exchange_form().as_bytes()).map(|(k, v)| {
                    let v = match k.as_ref() {
                        "client_id" => TUI_CLIENT_ID.to_owned(),
                        "hs_profile" => TUI_PROFILE_STR.to_owned(),
                        "resource" => "https://vm-tui.example".to_owned(),
                        _ => v.into_owned(),
                    };
                    (k.into_owned(), v)
                }),
            )
            .finish();
        assert!(
            ExchangeRequest::parse_for_host(form.as_bytes(), &host, Some(&tui_class())).is_ok()
        );
        // Substitution, both directions: the www profile on the TUI client
        // and the TUI profile on the www client are both denied.
        let mut swapped = body.clone();
        swapped["profile"] = PROFILE.into();
        assert!(ChallengeRequest::parse_for_host(
            &serde_json::to_vec(&swapped)?,
            &host,
            Some(&tui_class())
        )
        .is_err());
        let mut www_profile_on_www = request();
        www_profile_on_www["profile"] = TUI_PROFILE_STR.into();
        assert!(ChallengeRequest::parse_for_host(
            &serde_json::to_vec(&www_profile_on_www)?,
            &host,
            Some(&tui_class())
        )
        .is_err());
        // Default-deny: the TUI class is not accepted when unconfigured.
        assert!(
            ChallengeRequest::parse_for_host(&serde_json::to_vec(&body)?, &host, None).is_err()
        );
        assert!(ExchangeRequest::parse_for_host(form.as_bytes(), &host, None).is_err());
        Ok(())
    }

    #[cfg(feature = "postgres")]
    fn tui_session(host: &federate_source::FederateHost) -> Session {
        let ed = [31; 32];
        let pq = vec![32; 1952];
        Session {
            host: host.as_str().into(),
            sid: "tui-sid".into(),
            account_id: "tui-account".into(),
            subject: "did:plc:abcdefghijklmnopqrstuvwx".into(),
            tenant: "tenant-a".into(),
            client_id: TUI_CLIENT_ID.into(),
            resource: host.as_str().into(),
            scopes: vec!["connect:tui:session".into(), "query:tui:session".into()],
            grant_revision: "revision-1".into(),
            ed_public: ed,
            pq_public: pq,
            generation: [3; 32],
            collision_inventory_id: [3; 32],
            proof_epoch: 1,
            expires_at: 1_800_000_120,
        }
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn w13_tui_mint_stamps_distinct_markers_and_no_consumer_authority() -> anyhow::Result<()> {
        let host = federate_source::FederateHost::new("https://vm-tui.example")?;
        let now = 1_800_000_000i64;
        let session = tui_session(&host);
        let claims = committed_claims_for_class(
            &session,
            now,
            federate_source::ISSUER,
            "opaque-atproto-connector-subject",
            &host,
            Some(TUI_CLIENT_ID),
        )?;
        let value = serde_json::to_value(&claims)?;
        // The exact TUI marker set, and NONE of the www markers.
        assert_eq!(value["hs_tui_profile"], TUI_PROFILE_STR);
        assert_eq!(
            value["hs_tui_signer_pq_key_v1"],
            B64.encode(&session.pq_public)
        );
        assert_eq!(value["hs_tui_source_issuer_v1"], federate_source::ISSUER);
        assert_eq!(
            value["hs_tui_source_subject_v1"],
            "opaque-atproto-connector-subject"
        );
        for www_marker in [
            "hs_profile",
            "hs_signer_suite_v1",
            "hs_signer_pq_key_v1",
            "hs_signer_inventory_v1",
            "hs_source_issuer_v1",
            "hs_source_subject_v1",
            "hs_session_authority_generation",
        ] {
            assert!(
                value.get(www_marker).is_none(),
                "TUI mint stamped the www marker {www_marker}"
            );
        }
        // Normal-verification routing: no federate marker, not reserved.
        assert!(!claims.has_federate_profile_marker());
        assert!(!claims.is_reserved_federate_credential(host.as_str()));
        assert_eq!(value["client_id"], TUI_CLIENT_ID);
        assert_eq!(value["scope"], "connect:tui:session query:tui:session");
        assert_eq!(
            value["cnf"],
            json!({"jwk":{"kty":"OKP","crv":"Ed25519","x":B64.encode(session.ed_public)}})
        );

        // Registry|model consumer authority cannot ride the TUI mint, even
        // from a mutated Policy record.
        let mut smuggled = session.clone();
        smuggled.scopes = vec!["query:registry:List".into()];
        assert!(committed_tui_claims_for_host(
            &smuggled,
            now,
            federate_source::ISSUER,
            "subject",
            &host,
            TUI_CLIENT_ID
        )
        .is_err());
        let mut smuggled = session.clone();
        smuggled.scopes = vec!["infer:model:qwen2.5-0.5b-instruct:main".into()];
        assert!(committed_tui_claims_for_host(
            &smuggled,
            now,
            federate_source::ISSUER,
            "subject",
            &host,
            TUI_CLIENT_ID
        )
        .is_err());

        // Default-deny: the TUI session does not mint when the class is
        // unconfigured, and the www mint still rejects a TUI-client record.
        assert!(committed_claims_for_class(
            &session,
            now,
            federate_source::ISSUER,
            "subject",
            &host,
            None
        )
        .is_err());
        assert!(committed_claims_for_host(
            &session,
            now,
            federate_source::ISSUER,
            "subject",
            &host
        )
        .is_err());
        // The www mint is unchanged: its exact session still mints the www
        // marker set.
        let www = committed_session();
        let www_claims = committed_claims(&www, now, federate_source::ISSUER, "subject")?;
        let www_value = serde_json::to_value(&www_claims)?;
        assert_eq!(www_value["hs_profile"], PROFILE);
        assert!(www_value.get("hs_tui_profile").is_none());
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
