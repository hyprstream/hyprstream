//! CONNECT-time authentication for the `/moq` WebTransport plane (#1153).
//!
//! The `/moq` WebTransport CONNECT is an entry point onto a *shared* streaming
//! plane. This module authenticates it **before any stream is established**
//! (the accept loop verifies the credential and refuses the CONNECT — by
//! dropping the request, never completing the handshake — before
//! `Request::ok()` runs), then binds the accepted session to a tenant derived
//! from the *verified* identity.
//!
//! ## Credential accepted, and why
//!
//! The accepted credential is a **bearer JWT** carried in the HTTP/3
//! [`Authorization`] header (`Authorization: Bearer <jwt>`). It is
//! **verified, not merely parsed**: [`crate::auth::jwt::decode_with_key`]
//! checks the EdDSA signature (and rejects any non-`EdDSA` `alg`, defeating
//! algorithm-confusion), `exp`/`iat` validity, and the expected `aud`. A
//! forged signature, a stripped claim, or an expired token fails closed.
//!
//! This reuses the same verified-identity primitive the RPC envelope path uses
//! (`hyprstream_rpc::auth::jwt`/`Claims`) so the `/moq` plane and the RPC
//! plane agree on what "verified subject" means — the derivation can be
//! shared with the sibling Casbin-domain work (#1151) rather than invented a
//! second time.
//!
//! ## Why the tenant is NOT read from the token
//!
//! Per the #1128 spike and #1146 (T2.1 / PR #1147), JWT scope/claim fields are
//! not enforced at the verifier today — any tenant-shaped claim inside the
//! token is caller-supplied data and therefore unenforceable (the existing
//! `cap` field is the cautionary tale: present, unread). So the tenant is
//! **never** taken from the token. It is derived, server-side, from the
//! *verified subject* by [`MoqConnectAuthz::tenant_resolver`] — an
//! authoritative subject→tenant provisioning map the operator controls. The
//! token proves *who* the peer is; the resolver is the trust root for *which
//! tenant that subject belongs to*. See `VerifiedConnect`.
//!
//! ## What this guarantees today (read before assuming isolation)
//!
//! With a `MoqConnectAuthz` installed, an unauthenticated or tenant-less
//! `/moq` CONNECT is refused (fail-closed), and a connected peer can only
//! enumerate/subscribe to its own tenant's broadcasts (structural scoping via
//! [`crate::moq_authz::tenant_scoped_consumer`]). This is a **transport-plane,
//! metadata/announce isolation boundary** on the `/moq` consumer path. It is
//! NOT a cluster-wide tenant guarantee: frame content was already
//! AEAD-sealed + chained-HMAC'd at the source (the relay/consumer holds no
//! `enc_key`/`mac_key`), and other planes (RPC subject authorization, event
//! prefixes, VFS) have their own independent stories documented in the #1128
//! spike. Relay-mode (`with_moq_relay`) additionally refuses unauthenticated
//! CONNECTs but the relay still routes by track name — it must, to rendezvous
//! publisher and subscriber — so announce-name visibility through a relay is
//! inherent to relay mode and not something this boundary removes.

use std::sync::Arc;

use ed25519_dalek::VerifyingKey;

use crate::auth::jwt;
use crate::moq_authz::PeerIdentity;

/// HTTP/3 header carrying the bearer JWT.
pub const AUTHORIZATION_HEADER: &str = "authorization";
/// Scheme expected in the `Authorization` header value.
pub const BEARER_SCHEME: &str = "Bearer";

/// Server-side map from a **verified** subject to its tenant.
///
/// Authoritative server state (a subject→tenant provisioning table the
/// operator controls), never a caller-supplied field — see the module docs on
/// why a token claim is not used. The input is the verified `Claims::sub` from
/// a JWT whose signature has already been checked; returning `None` fails the
/// CONNECT closed.
pub type VerifiedSubjectTenantResolver = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// CONNECT-time authentication + tenant-binding config for the `/moq` plane.
///
/// When installed on a
/// [`crate::transport::quinn_transport::QuinnRpcServer`], every `/moq`
/// WebTransport CONNECT must present a bearer JWT verifiable with
/// `verify_key`; the verified subject is then mapped to a tenant by
/// `tenant_resolver`. [`MoqConnectAuthz::verify`] returns `None` on any
/// failure — missing header, wrong scheme, bad signature, expired/token-not-
/// yet-valid, wrong audience, or an authenticated subject with no tenant —
/// and the accept loop refuses the CONNECT (fail-closed). There is no
/// fall-through to an unscoped consumer: that is the #1145 pattern and the
/// single most likely way to get this wrong.
#[derive(Clone)]
pub struct MoqConnectAuthz {
    verify_key: VerifyingKey,
    expected_aud: Option<String>,
    tenant_resolver: VerifiedSubjectTenantResolver,
}

impl std::fmt::Debug for MoqConnectAuthz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoqConnectAuthz")
            .field("expected_aud", &self.expected_aud)
            .finish_non_exhaustive()
    }
}

impl MoqConnectAuthz {
    /// Build with the verifying key for the JWT issuer this `/moq` endpoint
    /// trusts and the authoritative subject→tenant resolver. The endpoint
    /// accepts any `aud` unless [`Self::with_expected_aud`] is set.
    pub fn new(
        verify_key: VerifyingKey,
        tenant_resolver: VerifiedSubjectTenantResolver,
    ) -> Self {
        Self {
            verify_key,
            expected_aud: None,
            tenant_resolver,
        }
    }

    /// Require the JWT `aud` to equal `aud` (trailing-slash tolerant, matches
    /// [`jwt::decode_with_key`]). A token with a different `aud` is rejected,
    /// and — unlike the codebase-wide lenient [`jwt::decode_with_key`] — a
    /// token with **no** `aud` is also rejected when this is set. On a
    /// dedicated `/moq` boundary that closes the token-substitution vector
    /// (a token minted for another service, which carries no `aud`, must not
    /// be admitted here) per BCP 225 / RFC 8725bis. If the operator does not
    /// set this, audience checking is off entirely (their choice).
    pub fn with_expected_aud(mut self, aud: impl Into<String>) -> Self {
        self.expected_aud = Some(aud.into());
        self
    }

    /// Verify the bearer credential in `headers` and resolve the tenant from
    /// the *verified* subject.
    ///
    /// Returns `None` — meaning the CONNECT MUST be refused — if:
    /// - no `Authorization: Bearer <jwt>` header is present,
    /// - the JWT signature, `alg`, `exp`, or `iat` check fails,
    /// - the `aud` check fails — including a **missing** `aud` when
    ///   [`Self::with_expected_aud`] is set (token substitution is not
    ///   admitted on a boundary that specified an audience),
    /// - or `tenant_resolver` returns `None` for the verified subject.
    ///
    /// Every branch is fail-closed; none falls back to an unscoped identity.
    pub fn verify(&self, headers: &http::HeaderMap) -> Option<VerifiedConnect> {
        let token = bearer_token(headers)?;
        let claims = jwt::decode_with_key(&token, &self.verify_key, self.expected_aud.as_deref()).ok()?;
        // [`jwt::decode_with_key`] is lenient on a *missing* `aud` (accepts
        // absent, rejects mismatch) to tolerate legacy federated issuers across
        // the codebase. This dedicated `/moq` boundary is stricter: when an
        // audience is configured, a token with no `aud` is a substitution
        // attempt from another service and must fail closed (#1153, BCP 225).
        if self.expected_aud.is_some() && claims.aud.is_none() {
            return None;
        }
        let tenant = (self.tenant_resolver)(&claims.sub)?;
        Some(VerifiedConnect {
            peer: PeerIdentity::authenticated(&claims.sub),
            tenant,
        })
    }
}

/// The outcome of a successful CONNECT-time verification: a peer whose
/// `subject` was verified by signature, and the tenant the server resolved
/// for that subject. The tenant is authoritative server-side state, not a
/// caller-supplied field.
#[derive(Debug, Clone)]
pub struct VerifiedConnect {
    /// The verified application subject.
    pub peer: PeerIdentity,
    /// Server-resolved tenant for the verified subject.
    pub tenant: String,
}

/// Extract the bearer token from the `Authorization` header.
///
/// Returns `None` if the header is absent, not valid UTF-8, not the `Bearer`
/// scheme (compared case-insensitively per RFC 7235), or carries an empty
/// token. Pure and independently testable.
pub fn bearer_token(headers: &http::HeaderMap) -> Option<String> {
    let raw = headers.get(AUTHORIZATION_HEADER)?.to_str().ok()?;
    let mut split = raw.splitn(2, ' ');
    let scheme = split.next()?;
    if !scheme.trim().eq_ignore_ascii_case(BEARER_SCHEME) {
        return None;
    }
    let token = split.next()?.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_owned())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::claims::Claims;
    use ed25519_dalek::SigningKey;

    fn mint(signing: &SigningKey, sub: &str, exp_delta_secs: i64) -> String {
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new(sub.to_owned(), now, now + exp_delta_secs);
        crate::auth::jwt::encode(&claims, signing)
    }

    fn resolver(mapping: &'static [(&'static str, &'static str)]) -> VerifiedSubjectTenantResolver {
        Arc::new(move |sub: &str| {
            mapping
                .iter()
                .find(|(s, _)| *s == sub)
                .map(|(_, t)| t.to_string())
        })
    }

    #[test]
    fn bearer_token_parses_well_formed_header() {
        let mut h = http::HeaderMap::new();
        h.insert("authorization", "Bearer abc.def.ghi".parse().unwrap());
        assert_eq!(bearer_token(&h).as_deref(), Some("abc.def.ghi"));
    }

    #[test]
    fn bearer_token_accepts_lowercase_scheme() {
        // RFC 7235: scheme is case-insensitive.
        let mut h = http::HeaderMap::new();
        h.insert("authorization", "bearer t.ok.en".parse().unwrap());
        assert_eq!(bearer_token(&h).as_deref(), Some("t.ok.en"));
    }

    #[test]
    fn bearer_token_rejects_missing_or_wrong_scheme() {
        let mut h = http::HeaderMap::new();
        assert!(bearer_token(&h).is_none()); // absent
        h.insert("authorization", "Basic dXNlcjpwYXNz".parse().unwrap());
        assert!(bearer_token(&h).is_none()); // wrong scheme
        h.insert("authorization", "Bearer   ".parse().unwrap());
        assert!(bearer_token(&h).is_none()); // empty token
    }

    #[test]
    fn verify_succeeds_for_known_subject_and_resolves_tenant() {
        let key = SigningKey::from_bytes(&[1u8; 32]);
        let vk = key.verifying_key();
        let authz = MoqConnectAuthz::new(vk, resolver(&[("did:key:alice", "alice")]));
        let tok = mint(&key, "did:key:alice", 60);
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        let vc = authz.verify(&h).expect("verified");
        assert_eq!(vc.peer.subject.as_deref(), Some("did:key:alice"));
        assert_eq!(vc.tenant, "alice");
        assert!(vc.peer.is_authenticated());
    }

    #[test]
    fn verify_fails_when_no_authorization_header() {
        let key = SigningKey::from_bytes(&[2u8; 32]);
        let vk = key.verifying_key();
        let authz = MoqConnectAuthz::new(vk, resolver(&[("s", "t")]));
        assert!(authz.verify(&http::HeaderMap::new()).is_none());
    }

    #[test]
    fn verify_fails_for_wrong_signature() {
        // Signed by a different key than the verifier expects.
        let issuer_key = SigningKey::from_bytes(&[3u8; 32]);
        let rogue_key = SigningKey::from_bytes(&[4u8; 32]);
        let authz = MoqConnectAuthz::new(issuer_key.verifying_key(), resolver(&[("s", "t")]));
        let tok = mint(&rogue_key, "s", 60); // signed by rogue
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(authz.verify(&h).is_none(), "forged signature must fail closed");
    }

    #[test]
    fn verify_fails_for_expired_token() {
        let key = SigningKey::from_bytes(&[5u8; 32]);
        let vk = key.verifying_key();
        let authz = MoqConnectAuthz::new(vk, resolver(&[("s", "t")]));
        // exp in the past.
        let now = chrono::Utc::now().timestamp();
        let claims = Claims::new("s".to_owned(), now - 120, now - 60);
        let tok = crate::auth::jwt::encode(&claims, &key);
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(authz.verify(&h).is_none());
    }

    #[test]
    fn verify_fails_when_subject_has_no_tenant() {
        // Verified signature, but the resolver doesn't know this subject →
        // fail closed (no wildcard tenant, no unscoped fallthrough).
        let key = SigningKey::from_bytes(&[6u8; 32]);
        let vk = key.verifying_key();
        let authz = MoqConnectAuthz::new(vk, resolver(&[("did:key:alice", "alice")]));
        let tok = mint(&key, "did:key:stranger", 60);
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(authz.verify(&h).is_none());
    }

    #[test]
    fn verify_enforces_expected_audience() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let vk = key.verifying_key();
        let authz =
            MoqConnectAuthz::new(vk, resolver(&[("s", "t")])).with_expected_aud("moq.example");
        // Mint a token with the wrong aud.
        let now = chrono::Utc::now().timestamp();
        let mut claims = Claims::new("s".to_owned(), now, now + 60);
        claims.aud = Some("other.example".to_owned());
        let tok = crate::auth::jwt::encode(&claims, &key);
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(authz.verify(&h).is_none(), "wrong aud must fail closed");
    }

    #[test]
    fn verify_rejects_missing_aud_when_audience_is_configured() {
        // BCP 225 / RFC 8725bis: on a boundary that specified an audience, a
        // token with NO aud is a substitution attempt from another service
        // and must fail closed — not be admitted by the lenient-on-absence
        // decode_with_key default. (`jwt::decode_with_key` accepts a missing
        // aud; `MoqConnectAuthz` must override that here.)
        let key = SigningKey::from_bytes(&[8u8; 32]);
        let vk = key.verifying_key();
        let authz =
            MoqConnectAuthz::new(vk, resolver(&[("s", "t")])).with_expected_aud("moq.example");
        // Token with no aud (the default from Claims::new).
        let tok = mint(&key, "s", 60);
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(
            authz.verify(&h).is_none(),
            "missing aud must fail closed when an audience is configured"
        );
    }

    #[test]
    fn verify_accepts_missing_aud_when_no_audience_configured() {
        // If the operator did not set expected_aud, audience checking is off
        // entirely — a token with no aud is admitted (subject to the other
        // checks). This is the operator's explicit choice.
        let key = SigningKey::from_bytes(&[9u8; 32]);
        let vk = key.verifying_key();
        let authz = MoqConnectAuthz::new(vk, resolver(&[("s", "t")]));
        let tok = mint(&key, "s", 60); // no aud
        let mut h = http::HeaderMap::new();
        h.insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert!(authz.verify(&h).is_some(), "no-aud token admitted when no audience configured");
    }
}
