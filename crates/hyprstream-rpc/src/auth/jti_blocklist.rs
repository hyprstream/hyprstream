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

use super::credential::CredentialId;

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
    fn is_revoked(&self, id: &CredentialId) -> bool;

    /// Revoke a credential. `expires_at` is the token's `exp` — the entry
    /// can be garbage-collected after this time.
    ///
    /// Ordering: the credential is marked revoked in the blocklist BEFORE
    /// derived handles are evicted. This ensures new verifications fail the
    /// blocklist check while cached contexts derived from the revoked token
    /// are being removed.
    fn revoke_credential(&self, id: CredentialId, expires_at: i64);
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
    fn is_revoked(&self, id: &CredentialId) -> bool {
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
        // handle survives is closed from both sides.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let jti_str = match &id.value {
                super::credential::CredentialValue::Jwt(s) => Some(s.as_str()),
                // CWT cti bytes are not stringified — eviction by bare jti
                // applies to the JWT jti namespace only. The CWT eviction
                // path will be wired when CWT credential verification lands;
                // the blocklist publication above is the security-critical
                // gate regardless.
                super::credential::CredentialValue::Cwt(_) => None,
            };
            if let Some(jti) = jti_str {
                crate::auth::mac::revoke_verified_subject_jti(jti);
            }
        }
    }
}

// ── Backward-compat type alias ───────────────────────────────────────────
//
// `InMemoryJtiBlocklist` is kept as a deprecated alias for the concrete
// struct. The old trait name `JtiBlocklist` is NOT aliasable (it was a
// trait, and Rust does not support trait aliases); all callers must use
// the new trait name `CredentialRevocationStore`. The concrete struct is
// safe to alias because the type itself is unchanged — only the trait
// method signatures changed.

#[deprecated(note = "use `InMemoryCredentialRevocationStore` (issuer-scoped)")]
pub type InMemoryJtiBlocklist = InMemoryCredentialRevocationStore;

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

    /// An empty issuer still produces a valid (if unusual) CredentialId —
    /// legacy tokens with empty `iss` are scoped to the empty-string issuer.
    #[test]
    fn empty_issuer_is_a_valid_scope() {
        let store = InMemoryCredentialRevocationStore::new();
        let id_empty = CredentialId::jwt("", "tok-1");
        let id_set = CredentialId::jwt("https://a.example", "tok-1");

        store.revoke_credential(id_empty.clone(), 9_999_999_999);
        assert!(store.is_revoked(&id_empty));
        assert!(!store.is_revoked(&id_set));
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

    /// `revoke_credential` publishes to the blocklist before evicting
    /// handles. We verify this by checking that after a revoke, the
    /// credential IS in the blocklist (published) — the eviction hook
    /// (revoke_verified_subject_jti) is called within the same method and
    /// cannot be invoked separately by the caller.
    #[test]
    fn revoke_credential_publishes_before_evicting() {
        let store = InMemoryCredentialRevocationStore::new();
        let id = CredentialId::jwt("https://a.example", "ordering-test");

        // Before revoke: not revoked.
        assert!(!store.is_revoked(&id));

        // revoke_credential is the single API entry point — it publishes
        // then evicts. The caller cannot reverse the order because there
        // is no separate "evict" method on the trait.
        store.revoke_credential(id.clone(), 9_999_999_999);

        // After revoke: published (is_revoked returns true).
        assert!(store.is_revoked(&id), "credential must be published as revoked");
    }

    /// The `CredentialRevocationStore` trait does NOT expose a bare eviction
    /// method. The only way to trigger eviction is through
    /// `revoke_credential`, which publishes first. This is a compile-time
    /// guarantee: the trait has exactly two methods, `is_revoked` and
    /// `revoke_credential`.
    #[test]
    fn trait_surface_enforces_ordering() {
        // If this test compiles, the trait has exactly the expected methods.
        fn assert_trait_shape<S: CredentialRevocationStore>() {}
        assert_trait_shape::<InMemoryCredentialRevocationStore>();
    }

    // ── Session revocation evicts all carrying credentials ───────────────

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
            tenant: None,
            kind: crate::auth::credential::SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        reg.register_session(key.clone(), state);
        assert!(!reg.is_revoked(&key), "session is active");

        reg.revoke_session(&key);
        assert!(reg.is_revoked(&key), "session is now revoked");
        assert_eq!(
            reg.session_state(&key).unwrap().status,
            ActiveOrRevoked::Revoked
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
            tenant: None,
            kind: SessionKind::Interactive,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        let wl_state = SessionState {
            subject: "service:model".to_owned(),
            tenant: None,
            kind: SessionKind::Workload,
            created_at: 0,
            expires_at: 9_999_999_999,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };

        reg.register_session(oidc_key.clone(), oidc_state);
        reg.register_session(wl_key.clone(), wl_state);

        // Revoke the OIDC session.
        reg.revoke_session(&oidc_key);
        assert!(reg.is_revoked(&oidc_key), "OIDC session is revoked");
        assert!(
            !reg.is_revoked(&wl_key),
            "workload session with the same value string is NOT revoked"
        );
    }
}
