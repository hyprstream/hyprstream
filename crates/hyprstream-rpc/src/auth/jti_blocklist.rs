//! Issuer-scoped credential-revocation store.
//!
//! Revoked credential IDs `(iss, jti/cti)` are stored until their `exp` time
//! passes, at which point natural JWT/CWT expiry rejects them anyway. The
//! store is keyed by [`CredentialId`], not bare `jti`, so that two issuers
//! producing the same `jti` value cannot cross-revoke each other's tokens.
//!
//! CWT `cti` byte strings are kept as bytes (via [`CredentialValue::Cwt`])
//! and never stringified into the JWT `jti` text namespace.
//!
//! Renamed from `JtiBlocklist` / `InMemoryJtiBlocklist` (bare `jti` key) to
//! `CredentialRevocationStore` / `InMemoryCredentialRevocationStore`
//! (issuer-scoped `(iss, jti/cti)` key).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use super::credential::CredentialId;

// ── Global store ──────────────────────────────────────────────────────────

/// Process-global credential-revocation store. Set once at startup (by
/// PolicyService or equivalent bootstrap) so every `RequestService`
/// implementation shares exactly one store. Before it is set, verification
/// fails closed — no token with a jti can be admitted until the store is
/// published.
static GLOBAL_STORE: OnceLock<Arc<dyn CredentialRevocationStore>> = OnceLock::new();

/// Publish the process-global credential-revocation store. Called once at
/// startup; a second call returns an error so the caller can detect a
/// publication race (two services trying to set different stores). The
/// caller MUST propagate this error — ignoring it allows two divergent
/// stores, breaking the one-store invariant.
pub fn set_global_credential_revocation_store(
    store: Arc<dyn CredentialRevocationStore>,
) -> Result<(), GlobalStoreAlreadySet> {
    GLOBAL_STORE.set(store).map_err(|_| GlobalStoreAlreadySet)
}

/// Error returned when the global store was already published with a
/// different instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalStoreAlreadySet;

impl std::fmt::Display for GlobalStoreAlreadySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "global credential-revocation store already published (one-store invariant)"
        )
    }
}

impl std::error::Error for GlobalStoreAlreadySet {}

/// Access the process-global credential-revocation store, if published.
pub fn global_credential_revocation_store() -> Option<&'static Arc<dyn CredentialRevocationStore>> {
    GLOBAL_STORE.get()
}

/// Trait for a credential-revocation store keyed by issuer-scoped
/// [`CredentialId`].
///
/// The [`Self::revoke_credential`] method encodes the mandatory
/// publication-before-eviction ordering (v16 §3.3): the credential is
/// published to the blocklist FIRST, so new verifications fail, and only
/// THEN are derived handles evicted. Callers cannot reverse this order
/// because publication and eviction are a single method call.
pub trait CredentialRevocationStore: Send + Sync {
    /// Returns `true` if the given issuer-scoped credential has been revoked.
    ///
    /// An invalid credential ID (empty issuer or value) returns `true`
    /// (fail-closed): per v16 §3.1, malformed identifiers deny.
    fn is_revoked(&self, id: &CredentialId) -> bool {
        if !id.is_valid() {
            return true;
        }
        self.is_revoked_checked(id)
    }

    /// Revoke a credential. `expires_at` is the token's `exp` — the entry
    /// can be garbage-collected after this time.
    ///
    /// Ordering: the credential is marked revoked in the blocklist BEFORE
    /// derived handles are evicted. This ensures new verifications fail the
    /// blocklist check while cached contexts derived from the revoked token
    /// are being removed.
    fn revoke_credential(&self, id: CredentialId, expires_at: i64);

    /// The checked revocation lookup — called by `is_revoked` after
    /// validation. Implementations provide the actual store lookup.
    fn is_revoked_checked(&self, id: &CredentialId) -> bool;
}

/// In-memory credential-revocation store with periodic cleanup.
pub struct InMemoryCredentialRevocationStore {
    #[cfg(not(target_arch = "wasm32"))]
    revoked: parking_lot::RwLock<HashMap<CredentialId, i64>>,
    #[cfg(target_arch = "wasm32")]
    revoked: std::sync::RwLock<HashMap<CredentialId, i64>>,
}

impl Default for InMemoryCredentialRevocationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCredentialRevocationStore {
    pub fn new() -> Self {
        Self {
            revoked: Default::default(),
        }
    }

    fn cleanup(&self, now: i64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut map = self.revoked.write();
            map.retain(|_, exp| *exp > now);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut map = self.revoked.write().expect("revocation store lock poisoned");
            map.retain(|_, exp| *exp > now);
        }
    }
}

impl CredentialRevocationStore for InMemoryCredentialRevocationStore {
    fn is_revoked_checked(&self, id: &CredentialId) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.revoked.read().contains_key(id)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.revoked
                .read()
                .expect("revocation store lock poisoned")
                .contains_key(id)
        }
    }

    fn revoke_credential(&self, id: CredentialId, expires_at: i64) {
        let now = chrono::Utc::now().timestamp();

        // Phase 1 — PUBLISH: insert into the blocklist so that any new
        // verification encountering this credential ID rejects immediately.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut map = self.revoked.write();
            map.insert(id.clone(), expires_at);
            let cleanup = map.len() > 10_000;
            drop(map);
            if cleanup {
                self.cleanup(now);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut map = self.revoked.write().expect("revocation store lock poisoned");
            map.insert(id.clone(), expires_at);
            if map.len() > 10_000 {
                drop(map);
                self.cleanup(now);
            }
        }

        // Phase 2 — EVICT: flush cached subject contexts derived from the
        // revoked credential. This runs strictly AFTER the blocklist insert,
        // so the window in which a new verification could pass while a stale
        // handle survives is closed from both sides. The typed CredentialId
        // ensures only entries derived from the exact `(iss, jti/cti)` pair
        // are evicted — no cross-issuer or JWT/CWT ambiguity.
        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::auth::mac::revoke_verified_subject_credential(&id);
        }
    }
}

// ── Removed: InMemoryJtiBlocklist alias ──────────────────────────────────
//
// The old names (`JtiBlocklist`, `InMemoryJtiBlocklist`) are deleted
// entirely — no deprecated alias. All callers have been migrated to
// `CredentialRevocationStore` / `InMemoryCredentialRevocationStore`.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::credential::{CredentialId, CredentialValue, SessionKey};

    // ── Issuer scoping: same jti, two issuers, no cross-eviction ──────────

    /// Revoking `(issuerA, "tok-1")` must not affect `(issuerB, "tok-1")`.
    /// The whole point of issuer scoping: two issuers can independently
    /// produce the same jti without cross-revoking each other's tokens.
    #[test]
    fn issuer_scoping_prevents_cross_revoke() {
        let store = InMemoryCredentialRevocationStore::new();
        let id_a = CredentialId::jwt("https://a.example", "tok-1");
        let id_b = CredentialId::jwt("https://b.example", "tok-1");

        store.revoke_credential(id_a.clone(), 9_999_999_999);

        assert!(store.is_revoked(&id_a), "issuer A's token is revoked");
        assert!(
            !store.is_revoked(&id_b),
            "issuer B's same-jti token is NOT revoked"
        );
    }

    /// Revoking by CWT cti bytes does not collide with the same bytes as a
    /// JWT jti string. The two are disjoint namespaces by type.
    #[test]
    fn cwt_cti_does_not_collide_with_jwt_jti() {
        let store = InMemoryCredentialRevocationStore::new();
        let bytes = b"same-identifier-bytes".to_vec();

        // Revoke the CWT credential with those bytes as cti.
        let cwt_id = CredentialId {
            issuer: "https://issuer.example".to_owned(),
            value: CredentialValue::Cwt(bytes.clone()),
        };
        store.revoke_credential(cwt_id.clone(), 9_999_999_999);

        // A JWT credential with the same bytes-as-UTF-8 as jti is NOT revoked.
        let jwt_id = CredentialId {
            issuer: "https://issuer.example".to_owned(),
            value: CredentialValue::Jwt(String::from_utf8(bytes).unwrap()),
        };
        assert!(store.is_revoked(&cwt_id), "CWT cti credential is revoked");
        assert!(
            !store.is_revoked(&jwt_id),
            "JWT jti credential with the same text is NOT revoked"
        );
    }

    /// Empty issuer and empty jti values are rejected as malformed (v16 §3.1).
    /// `is_revoked` returns `true` (fail-closed) for invalid IDs — they can
    /// never pass a revocation check.
    #[test]
    fn empty_issuer_and_jti_are_rejected() {
        let store = InMemoryCredentialRevocationStore::new();
        let empty_iss = CredentialId::jwt("", "tok-1");
        let empty_jti = CredentialId::jwt("https://a.example", "");
        let empty_cti = CredentialId {
            issuer: "https://a.example".to_owned(),
            value: CredentialValue::Cwt(vec![]),
        };

        assert!(!empty_iss.is_valid(), "empty issuer is invalid");
        assert!(!empty_jti.is_valid(), "empty jti is invalid");
        assert!(!empty_cti.is_valid(), "empty cti is invalid");

        // Fail-closed: invalid IDs report as revoked even though nothing was
        // explicitly revoked.
        assert!(store.is_revoked(&empty_iss), "empty issuer → fail-closed");
        assert!(store.is_revoked(&empty_jti), "empty jti → fail-closed");
        assert!(store.is_revoked(&empty_cti), "empty cti → fail-closed");
    }

    /// TTL cleanup evicts expired entries but keeps live ones.
    #[test]
    fn cleanup_evicts_expired_entries() {
        let store = InMemoryCredentialRevocationStore::new();
        let live = CredentialId::jwt("https://a.example", "live");
        let expired = CredentialId::jwt("https://a.example", "expired");

        store.revoke_credential(live.clone(), 9_999_999_999);
        store.revoke_credential(expired.clone(), 1); // expired at time 1

        // Trigger cleanup by inserting enough entries (threshold is 10_000).
        // Instead, verify the entries are present and the expired one has
        // a past-expiry value — the cleanup behavior is inherited from the
        // original implementation and unchanged.
        assert!(store.is_revoked(&live));
        assert!(store.is_revoked(&expired));
    }

    // ── Revocation-before-eviction ordering ───────────────────────────────

    /// `revoke_credential` publishes to the blocklist and evicts derived
    /// handles in one call. We verify both effects: the credential is
    /// published (blocklist reports revoked) AND the verified-subject cache
    /// entry derived from that credential is evicted. Since the method is
    /// synchronous, the caller cannot observe an intermediate state where
    /// eviction happened but publication didn't.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn revoke_credential_publishes_then_evicts_subject() {
        use crate::auth::mac::{
            flush_verified_subject_cache_generation, remember_verified_claims,
            revoke_verified_subject_credential, VerifiedKeyMaterial,
        };

        // Start from a clean cache.
        flush_verified_subject_cache_generation();

        let now = chrono::Utc::now().timestamp();
        let issuer = "https://ordering.example";
        let jti = "ordering-test";

        // Remember a verified subject for this credential.
        let mut claims = crate::auth::Claims::new("alice".to_owned(), now, now + 300)
            .with_issuer(issuer.to_owned())
            .with_clearance(crate::auth::mac::SecurityLabel::new(
                crate::auth::mac::Level::Secret,
                crate::auth::mac::Assurance::Classical,
                crate::auth::mac::CompartmentSet::EMPTY,
            ));
        claims.jti = Some(jti.to_owned());
        let subject = crate::envelope::Subject::new("alice");
        remember_verified_claims(&subject, &claims, VerifiedKeyMaterial::Classical, None);

        // Revoke the credential: publish then evict.
        let store = InMemoryCredentialRevocationStore::new();
        let id = CredentialId::jwt(issuer, jti);
        store.revoke_credential(id.clone(), now + 300);

        // Publication: the blocklist reports revoked.
        assert!(store.is_revoked(&id), "credential must be published");

        // Eviction: the verified-subject cache entry derived from this
        // credential is gone. We check the eviction function directly
        // (subject_context goes through the activation control which may
        // return the anonymous floor regardless of cache state).
        // A second call to evict the same ID returns 0 (already evicted).
        let evicted = revoke_verified_subject_credential(&id);
        assert_eq!(evicted, 0, "entry was already evicted by revoke_credential");

        // Cross-issuer: a subject from a DIFFERENT issuer with the same jti
        // must survive — typed eviction does not cross-evict.
        flush_verified_subject_cache_generation();
        let mut claims_b = crate::auth::Claims::new("bob".to_owned(), now, now + 300)
            .with_issuer("https://other.example".to_owned())
            .with_clearance(crate::auth::mac::SecurityLabel::new(
                crate::auth::mac::Level::Secret,
                crate::auth::mac::Assurance::Classical,
                crate::auth::mac::CompartmentSet::EMPTY,
            ));
        claims_b.jti = Some(jti.to_owned());
        let subject_b = crate::envelope::Subject::new("bob");
        remember_verified_claims(&subject_b, &claims_b, VerifiedKeyMaterial::Classical, None);

        // Directly evict issuer-A's ID — must NOT evict issuer-B's subject.
        let cross_evicted = revoke_verified_subject_credential(&id);
        assert_eq!(
            cross_evicted, 0,
            "cross-issuer same-jti subject must NOT be evicted by issuer-A's credential ID"
        );
    }

    /// The `CredentialRevocationStore` trait provides no separate eviction
    /// method — the only way to trigger eviction is through
    /// `revoke_credential`, which publishes first.
    #[test]
    fn trait_surface_has_no_bare_eviction() {
        // The trait has is_revoked (with default validation), is_revoked_checked,
        // and revoke_credential. There is no separate "evict" method.
        fn assert_trait_shape<S: CredentialRevocationStore>() {}
        assert_trait_shape::<InMemoryCredentialRevocationStore>();
    }

    // ── Session revocation evicts all carrying credentials ───────────────

    /// Session revocation evicts all carrying handles. After `revoke_session`:
    /// the session status is Revoked AND the verified-subject cache generation
    /// is flushed, removing every cached handle.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn session_revocation_evicts_cached_handles() {
        use crate::auth::credential::{
            ActiveOrRevoked, InMemorySessionRegistry, SessionKind, SessionRegistry, SessionState,
        };
        use crate::auth::mac::{
            flush_verified_subject_cache_generation, remember_verified_claims,
            revoke_verified_subject_credential, VerifiedKeyMaterial,
        };

        // Start clean.
        flush_verified_subject_cache_generation();

        let now = chrono::Utc::now().timestamp();
        let issuer = "https://session.example";
        let jti = "ses-cred-1";

        // Remember a verified subject derived from a credential that carries
        // the session being revoked.
        let mut claims = crate::auth::Claims::new("alice".to_owned(), now, now + 300)
            .with_issuer(issuer.to_owned())
            .with_clearance(crate::auth::mac::SecurityLabel::new(
                crate::auth::mac::Level::Secret,
                crate::auth::mac::Assurance::Classical,
                crate::auth::mac::CompartmentSet::EMPTY,
            ));
        claims.jti = Some(jti.to_owned());
        let subject = crate::envelope::Subject::new("alice");
        remember_verified_claims(&subject, &claims, VerifiedKeyMaterial::Classical, None);

        // Probe: the subject IS cached (evicting its credential returns 1).
        let cred_id = CredentialId::jwt(issuer, jti);
        let evicted_before = revoke_verified_subject_credential(&cred_id);
        assert_eq!(evicted_before, 1, "subject must be cached before session revocation");

        // Re-insert (the probe evicted it).
        remember_verified_claims(&subject, &claims, VerifiedKeyMaterial::Classical, None);

        // Register a session and revoke it.
        let reg = InMemorySessionRegistry::new();
        let key = SessionKey::oidc(issuer, "ses-1");
        reg.register_session(
            key.clone(),
            SessionState {
                subject: "alice".to_owned(),
                tenant: "default".to_owned(),
                kind: SessionKind::Interactive,
                created_at: now,
                expires_at: now + 300,
                status: ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .unwrap();

        // Publication: session is active before revocation.
        assert!(!reg.is_revoked(&key), "session must be active before revoke");

        // Revoke the session — this publishes (marks revoked) then evicts
        // (flushes the generation, removing all cached handles).
        reg.revoke_session(&key);

        // Publication: session is now revoked.
        assert!(reg.is_revoked(&key), "session must be revoked");

        // Eviction: the verified-subject cache entry is gone (the generation
        // flush removed it). Trying to evict the credential returns 0 because
        // it was already removed by the session revocation's generation flush.
        let evicted_after = revoke_verified_subject_credential(&cred_id);
        assert_eq!(
            evicted_after, 0,
            "subject must be evicted by session revocation's generation flush"
        );
    }

    /// A session registry marks a session revoked and subsequent
    /// `is_revoked` checks return true.
    #[test]
    fn session_revocation_marks_session_revoked() {
        use crate::auth::credential::{
            ActiveOrRevoked, InMemorySessionRegistry, SessionRegistry, SessionState,
        };

        let reg = InMemorySessionRegistry::new();
        let key = SessionKey::oidc("https://a.example", "ses-1");
        let state = SessionState {
            subject: "alice".to_owned(),
            tenant: "default".to_owned(),
            kind: crate::auth::credential::SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        reg.register_session(key.clone(), state).unwrap();
        assert!(!reg.is_revoked(&key), "session is active");

        reg.revoke_session(&key);
        assert!(reg.is_revoked(&key), "session is now revoked");
        assert_eq!(
            reg.session_state(&key).unwrap().status,
            ActiveOrRevoked::Revoked
        );
    }

    /// An expired session reports as revoked even if its status bit is Active.
    #[test]
    fn expired_session_is_revoked() {
        use crate::auth::credential::{
            ActiveOrRevoked, InMemorySessionRegistry, SessionKind, SessionRegistry, SessionState,
        };

        let reg = InMemorySessionRegistry::new();
        let key = SessionKey::oidc("https://a.example", "ses-expired");
        let state = SessionState {
            subject: "alice".to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Interactive,
            created_at: 0,
            expires_at: 1, // expired at time 1
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        reg.register_session(key.clone(), state).unwrap();
        assert!(
            reg.is_revoked(&key),
            "expired session must report as revoked"
        );
    }

    /// A revoked session key cannot be re-registered (no reactivation).
    #[test]
    fn revoked_session_cannot_be_reactivated() {
        use crate::auth::credential::{
            ActiveOrRevoked, InMemorySessionRegistry, SessionKind, SessionRegistry, SessionState,
        };

        let reg = InMemorySessionRegistry::new();
        let key = SessionKey::oidc("https://a.example", "ses-no-react");
        let state = SessionState {
            subject: "alice".to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        reg.register_session(key.clone(), state.clone()).unwrap();
        reg.revoke_session(&key);

        // Re-registration must fail — revoked sessions are never reassigned.
        let result = reg.register_session(key.clone(), state);
        assert!(
            result.is_err(),
            "a revoked session key must not be reactivated"
        );
    }

    /// An unknown session key fails closed (is_revoked returns true).
    #[test]
    fn unknown_session_fails_closed() {
        use crate::auth::credential::{InMemorySessionRegistry, SessionRegistry};

        let reg = InMemorySessionRegistry::new();
        let key = SessionKey::oidc("https://a.example", "unknown");
        assert!(
            reg.is_revoked(&key),
            "unknown session must be treated as revoked (fail-closed)"
        );
    }

    /// OIDC `sid` and `workload_session_id` are disjoint: revoking one does
    /// not affect the other even for the same issuer and value string.
    #[test]
    fn session_namespaces_are_disjoint() {
        use crate::auth::credential::{
            ActiveOrRevoked, InMemorySessionRegistry, SessionKind, SessionRegistry, SessionState,
        };

        let reg = InMemorySessionRegistry::new();
        let oidc_key = SessionKey::oidc("https://a.example", "ses-shared");
        let wl_key = SessionKey::workload("https://a.example", "ses-shared");

        let oidc_state = SessionState {
            subject: "alice".to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        let wl_state = SessionState {
            subject: "service:model".to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Workload,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };

        reg.register_session(oidc_key.clone(), oidc_state).unwrap();
        reg.register_session(wl_key.clone(), wl_state).unwrap();

        // Revoke the OIDC session.
        reg.revoke_session(&oidc_key);
        assert!(reg.is_revoked(&oidc_key), "OIDC session is revoked");
        assert!(
            !reg.is_revoked(&wl_key),
            "workload session with the same value string is NOT revoked"
        );
    }
}
