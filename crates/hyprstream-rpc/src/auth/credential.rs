//! Issuer-scoped credential and session identity types.
//!
//! Per the v16 credential/session profile: credential identifiers are scoped
//! by issuer `(iss, jti/cti)` and session identifiers are scoped by issuer
//! `(iss, sid/workload_session_id)`. The credential and session namespaces
//! are disjoint by construction — they have separate types, separate stores,
//! and never share a key.
//!
//! CWT `cti` is a byte string. It is kept as bytes here and never stringified
//! into the JWT `jti` text namespace, which would create an ambiguous collision
//! domain. The [`CredentialValue`] enum preserves the encoding distinction.

use std::collections::HashMap;
use std::fmt;

// ── Credential identity ───────────────────────────────────────────────────

/// A credential identifier value — the JWT `jti` (text) or CWT `cti` (bytes).
///
/// The two variants are disjoint by type: a CWT `cti` byte string is never
/// stringified into the JWT `jti` text namespace. This prevents a CWT binary
/// `cti` that happens to decode as valid UTF-8 from colliding with a JWT text
/// `jti` from a different issuer's token.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CredentialValue {
    /// JWT `jti` claim (RFC 7519) — a case-sensitive text string.
    Jwt(String),
    /// CWT `cti` claim (RFC 8392) — a binary byte string, kept as raw bytes.
    Cwt(Vec<u8>),
}

impl CredentialValue {
    /// Construct a JWT `jti` credential value from a text string.
    pub fn jwt(s: impl Into<String>) -> Self {
        Self::Jwt(s.into())
    }

    /// Construct a CWT `cti` credential value from raw bytes.
    pub fn cwt(b: impl Into<Vec<u8>>) -> Self {
        Self::Cwt(b.into())
    }

    /// Whether this value is empty (zero-length string or bytes).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Jwt(s) => s.is_empty(),
            Self::Cwt(b) => b.is_empty(),
        }
    }
}

impl fmt::Display for CredentialValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jwt(s) => write!(f, "jti:{s}"),
            Self::Cwt(b) => {
                write!(f, "cti:{}", hex_compact(b))
            }
        }
    }
}

/// Issuer-scoped credential identifier: `(iss, jti/cti)`.
///
/// Verifier stores namespace credential IDs by issuer as defense in depth:
/// even if two issuers produce the same `jti`/`cti` value, only the
/// `(issuer, value)` pair identifies one credential.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CredentialId {
    /// The `iss` claim identifying the credential issuer.
    pub issuer: String,
    /// The `jti` (JWT) or `cti` (CWT) token-unique identifier.
    pub value: CredentialValue,
}

impl CredentialId {
    /// Construct a JWT credential ID from an issuer URL and `jti` string.
    pub fn jwt(issuer: impl Into<String>, jti: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            value: CredentialValue::jwt(jti),
        }
    }

    /// Construct a CWT credential ID from an issuer URL and raw `cti` bytes.
    pub fn cwt(issuer: impl Into<String>, cti: impl Into<Vec<u8>>) -> Self {
        Self {
            issuer: issuer.into(),
            value: CredentialValue::cwt(cti),
        }
    }

    /// Whether this credential ID is well-formed (non-empty issuer and value).
    ///
    /// Per v16 §3.1: empty issuer and malformed identifiers deny. Verification
    /// boundaries MUST treat an invalid credential ID as fail-closed.
    pub fn is_valid(&self) -> bool {
        !self.issuer.is_empty() && !self.value.is_empty()
    }
}

impl fmt::Display for CredentialId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.issuer, self.value)
    }
}

// ── Session identity ──────────────────────────────────────────────────────

/// A session identifier — disjoint OIDC `sid` vs workload session.
///
/// Per v16 §3.3: OIDC/user sessions carry the registered `sid` claim.
/// Workload credential families use a separately typed identifier and do
/// NOT overload OIDC `sid`. The two variants are disjoint types and wire
/// namespaces; one cannot be substituted for the other.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SessionIdentifier {
    /// OIDC user-session ID (the registered `sid` claim). Unique within
    /// an issuer; groups an interactive session whose access-token
    /// rotations have distinct credential IDs.
    OidcSid(String),
    /// Workload credential family session ID (`workload_session_id`).
    /// Used only when an issuer maintains a real revocable workload
    /// credential family (e.g. bootstrap/renewal producing several WITs).
    /// A standalone service credential with no such lifecycle omits it.
    WorkloadSessionId(String),
}

impl SessionIdentifier {
    /// Whether this identifier is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::OidcSid(s) | Self::WorkloadSessionId(s) => s.is_empty(),
        }
    }
}

impl fmt::Display for SessionIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OidcSid(s) => write!(f, "sid:{s}"),
            Self::WorkloadSessionId(s) => write!(f, "wlses:{s}"),
        }
    }
}

/// Issuer-scoped session key: `(iss, sid/workload_session_id)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// The `iss` claim identifying the session authority.
    pub issuer: String,
    /// The session identifier (OIDC `sid` or `workload_session_id`).
    pub id: SessionIdentifier,
}

impl SessionKey {
    /// Construct an OIDC session key.
    pub fn oidc(issuer: impl Into<String>, sid: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            id: SessionIdentifier::OidcSid(sid.into()),
        }
    }

    /// Construct a workload session key.
    pub fn workload(issuer: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            issuer: issuer.into(),
            id: SessionIdentifier::WorkloadSessionId(id.into()),
        }
    }

    /// Whether this session key is well-formed (non-empty issuer and identifier).
    pub fn is_valid(&self) -> bool {
        !self.issuer.is_empty() && !self.id.is_empty()
    }
}

// ── Session state ─────────────────────────────────────────────────────────

/// Whether a session is interactive (OIDC user login) or workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionKind {
    /// OIDC/user-agent interactive session.
    Interactive,
    /// Workload credential family (e.g. bootstrap/renewal WIT rotation).
    Workload,
}

/// Active-or-revoked session status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveOrRevoked {
    Active,
    Revoked,
}

/// Session state record per v16 §3.3.
///
/// The authority stores at least this record for every active session.
/// Revocation updates `status` to [`ActiveOrRevoked::Revoked`], which causes
/// every credential carrying that session ID to be rejected.
#[derive(Clone, Debug)]
pub struct SessionState {
    /// Subject identifier (`sub` claim).
    pub subject: String,
    /// Verified tenant/domain (required per v16 §3.3 — every session is
    /// tenant-bound).
    pub tenant: String,
    /// Interactive or workload session kind. MUST match the [`SessionKey`]'s
    /// identifier variant (OIDC sid → Interactive, workload → Workload).
    pub kind: SessionKind,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Expiry timestamp (Unix seconds).
    pub expires_at: i64,
    /// Active or revoked.
    pub status: ActiveOrRevoked,
    /// Clearance epoch at which the session's clearance was established.
    pub clearance_epoch: u64,
}

// ── Session registry trait + in-memory implementation ─────────────────────

/// Session registry: tracks active sessions and their revocation state.
///
/// Per v16 §3.3: revoking a `SessionKey` rejects every credential and handle
/// carrying that session ID. The trait is designed so a Valkey/Redis backend
/// (the identity-storage substrate) can drop in later with the same interface.
pub trait SessionRegistry: Send + Sync {
    /// Look up the state of a session, if known.
    fn session_state(&self, key: &SessionKey) -> Option<SessionState>;

    /// Register a new session. Returns `Err` if a session already exists at
    /// the given key — session identifiers are never reassigned (v16 §3.3).
    /// A revoked session CANNOT be reactivated by re-registration. Also
    /// rejects a key/state kind mismatch (OIDC sid must pair with Interactive,
    /// workload must pair with Workload).
    fn register_session(
        &self,
        key: SessionKey,
        state: SessionState,
    ) -> Result<(), SessionRegisterError>;

    /// Revoke a session: mark it revoked, then evict every credential
    /// derived from it.
    ///
    /// Ordering: the session's status is set to [`ActiveOrRevoked::Revoked`]
    /// BEFORE credential/handle eviction runs, so a concurrent verification
    /// checking session state fails the revocation check while cached
    /// authority is being flushed.
    fn revoke_session(&self, key: &SessionKey);

    /// Whether a session is currently revoked, expired, or unknown. Returns
    /// `true` if the session is revoked, expired, OR not found (fail-closed
    /// for unknown sessions).
    fn is_revoked(&self, key: &SessionKey) -> bool;
}

/// Error returned when a session already exists at the given key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionExists;

impl std::fmt::Display for SessionExists {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "session already exists (identifiers are never reassigned)")
    }
}

impl std::error::Error for SessionExists {}

/// Error returned when a session key's identifier variant disagrees with the
/// session state's kind (v16 §3.3 coherence requirement).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionKindMismatch;

impl std::fmt::Display for SessionKindMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "session key variant does not match session state kind"
        )
    }
}

impl std::error::Error for SessionKindMismatch {}

/// Error returned when a session registration record is malformed: an empty
/// issuer or session identifier in the key, or an empty subject/tenant in the
/// state (v16 §3.1 malformed-identifier denial, §3.3 record shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidSessionRecord;

impl std::fmt::Display for InvalidSessionRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "malformed session record (empty issuer, identifier, subject, or tenant)"
        )
    }
}

impl std::error::Error for InvalidSessionRecord {}

/// Error returned by [`SessionRegistry::register_session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionRegisterError {
    /// A session already exists at the given key.
    Exists(SessionExists),
    /// The key's identifier variant disagrees with the state's kind.
    KindMismatch(SessionKindMismatch),
    /// The key or state record is malformed (empty required field).
    InvalidRecord(InvalidSessionRecord),
}

impl std::fmt::Display for SessionRegisterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exists(e) => write!(f, "{e}"),
            Self::KindMismatch(e) => write!(f, "{e}"),
            Self::InvalidRecord(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SessionRegisterError {}

impl From<SessionExists> for SessionRegisterError {
    fn from(e: SessionExists) -> Self {
        Self::Exists(e)
    }
}

impl From<SessionKindMismatch> for SessionRegisterError {
    fn from(e: SessionKindMismatch) -> Self {
        Self::KindMismatch(e)
    }
}

impl From<InvalidSessionRecord> for SessionRegisterError {
    fn from(e: InvalidSessionRecord) -> Self {
        Self::InvalidRecord(e)
    }
}

/// Validate that a session key's identifier variant matches the session state
/// kind (OIDC sid → Interactive, workload → Workload).
fn validate_key_kind_coherence(
    key: &SessionKey,
    state: &SessionState,
) -> Result<(), SessionKindMismatch> {
    match (&key.id, state.kind) {
        (SessionIdentifier::OidcSid(_), SessionKind::Interactive)
        | (SessionIdentifier::WorkloadSessionId(_), SessionKind::Workload) => Ok(()),
        _ => Err(SessionKindMismatch),
    }
}

/// In-memory session registry with TTL cleanup.
///
/// This is the default process-local implementation. The trait allows a
/// Valkey/Redis-backed registry to drop in for multi-node deployments.
pub struct InMemorySessionRegistry {
    #[cfg(not(target_arch = "wasm32"))]
    sessions: parking_lot::RwLock<HashMap<SessionKey, SessionState>>,
    #[cfg(target_arch = "wasm32")]
    sessions: std::sync::RwLock<HashMap<SessionKey, SessionState>>,
}

impl Default for InMemorySessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemorySessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: Default::default(),
        }
    }
}

impl SessionRegistry for InMemorySessionRegistry {
    fn session_state(&self, key: &SessionKey) -> Option<SessionState> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.sessions.read().get(key).cloned()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.sessions
                .read()
                .expect("session registry lock poisoned")
                .get(key)
                .cloned()
        }
    }

    fn register_session(
        &self,
        key: SessionKey,
        state: SessionState,
    ) -> Result<(), SessionRegisterError> {
        // Reject malformed keys (empty issuer or session identifier).
        if !key.is_valid() {
            return Err(InvalidSessionRecord.into());
        }
        // Require nonempty subject and tenant (v16 §3.3 record shape).
        if state.subject.is_empty() || state.tenant.is_empty() {
            return Err(InvalidSessionRecord.into());
        }
        validate_key_kind_coherence(&key, &state)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut map = self.sessions.write();
            if map.contains_key(&key) {
                return Err(SessionExists.into());
            }
            map.insert(key, state);
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut map = self
                .sessions
                .write()
                .expect("session registry lock poisoned");
            if map.contains_key(&key) {
                return Err(SessionExists.into());
            }
            map.insert(key, state);
            Ok(())
        }
    }

    fn revoke_session(&self, key: &SessionKey) {
        // Phase 1 — PUBLISH: mark the session as revoked.
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(state) = self.sessions.write().get_mut(key) {
                state.status = ActiveOrRevoked::Revoked;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(state) = self
                .sessions
                .write()
                .expect("session registry lock poisoned")
                .get_mut(key)
            {
                state.status = ActiveOrRevoked::Revoked;
            }
        }

        // Phase 2 — EVICT: flush the verified-subject cache generation so
        // cached handles derived from credentials carrying this session are
        // invalidated. This is a conservative broad eviction: it flushes
        // ALL cached handles, not just those for this session. The
        // correctness guarantee holds (no stale authority survives), at
        // the cost of re-verifying unrelated subjects on their next access.
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::auth::mac::flush_verified_subject_cache_generation();
        }
    }

    fn is_revoked(&self, key: &SessionKey) -> bool {
        match self.session_state(key) {
            Some(state) => {
                state.status == ActiveOrRevoked::Revoked
                    || state.expires_at <= chrono::Utc::now().timestamp()
            }
            None => true, // fail-closed for unknown sessions
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Compact hex encoding for display (not for storage or comparison).
fn hex_compact(bytes: &[u8]) -> String {
    if bytes.len() <= 8 {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    } else {
        format!(
            "{}…{}",
            bytes[..4].iter().map(|b| format!("{b:02x}")).collect::<String>(),
            bytes[bytes.len() - 4..]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn jwt_and_cwt_values_never_collide() {
        // A CWT cti that happens to be valid UTF-8 must not equal a JWT jti
        // with the same bytes-as-string — they are different variants.
        let text = "abc-123";
        let jwt_val = CredentialValue::jwt(text);
        let cwt_val = CredentialValue::cwt(text.as_bytes());
        assert_ne!(jwt_val, cwt_val);
        assert_ne!(hash_of(&jwt_val), hash_of(&cwt_val));
    }

    #[test]
    fn credential_id_is_issuer_scoped() {
        let id_a = CredentialId::jwt("https://a.example", "token-1");
        let id_b = CredentialId::jwt("https://b.example", "token-1");
        assert_ne!(id_a, id_b, "same jti, different issuer → different ID");
    }

    #[test]
    fn session_key_oidc_and_workload_are_disjoint() {
        let oidc = SessionKey::oidc("https://a.example", "ses-1");
        let wl = SessionKey::workload("https://a.example", "ses-1");
        assert_ne!(oidc, wl, "OIDC sid and workload session must not collide");
        assert_ne!(hash_of(&oidc), hash_of(&wl));
    }

    /// Malformed registrations (empty issuer/identifier/subject/tenant) are
    /// rejected as [`InvalidSessionRecord`], NOT as a kind mismatch — a
    /// malformed record is not a kind disagreement.
    #[test]
    fn malformed_registration_is_invalid_record_not_kind_mismatch() {
        let reg = InMemorySessionRegistry::new();
        let state = SessionState {
            subject: "alice".to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };

        let bad_key = reg.register_session(SessionKey::oidc("", "ses-1"), state.clone());
        assert!(
            matches!(bad_key, Err(SessionRegisterError::InvalidRecord(_))),
            "empty issuer key → InvalidRecord, got {bad_key:?}"
        );

        let mut empty_subject = state.clone();
        empty_subject.subject = String::new();
        let bad_state = reg.register_session(SessionKey::oidc("https://a.example", "ses-1"), empty_subject);
        assert!(
            matches!(bad_state, Err(SessionRegisterError::InvalidRecord(_))),
            "empty subject → InvalidRecord, got {bad_state:?}"
        );

        // A genuine kind mismatch still reports KindMismatch.
        let mismatch = reg.register_session(SessionKey::workload("https://a.example", "ses-2"), state);
        assert!(
            matches!(mismatch, Err(SessionRegisterError::KindMismatch(_))),
            "workload key + interactive state → KindMismatch, got {mismatch:?}"
        );
    }

    fn hash_of<T: std::hash::Hash>(v: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }
}
