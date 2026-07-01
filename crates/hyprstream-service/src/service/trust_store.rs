//! Key-centric trust store for service identity verification.
//!
//! The trust store holds verified Ed25519 pubkeys and their attestations.
//! Keys ARE identity — service names are authorization scopes.
//!
//! # Bootstrap
//!
//! PolicyService's verifying key is the root trust anchor, loaded from:
//! - Inproc: generated in memory during `generate_independent_service_keys`
//! - IPC: loaded from `ca-pubkey` credential file
//!
//! # Runtime
//!
//! Services register their pubkeys via `registerServiceKey` RPC.
//! Clients verify peer keys via `is_authorized(key, scope)`.
//!
//! # Multi-instance
//!
//! Multiple services with the same scope (e.g., two "model" instances) each
//! have their own key. The trust store holds all of them. A client connecting
//! to any endpoint verifies the response signer is authorized for the
//! requested scope.

use dashmap::DashMap;
use ed25519_dalek::VerifyingKey;
use std::collections::HashSet;
use std::sync::Arc;
use tracing;

/// An attestation binding a key to authorization scopes.
#[derive(Clone, Debug)]
pub struct Attestation {
    /// Authorization scopes this key is valid for (e.g., "model", "policy").
    pub scopes: HashSet<String>,
    /// Explicit subject identity (e.g., "alice", "https://node-b:alice").
    /// `None` for service keys (derived as "service:{scope}").
    pub subject: Option<String>,
    /// CA-signed JWT attesting this key.
    pub jwt: Option<String>,
    /// Unix timestamp when this attestation expires.
    pub expires_at: i64,
    /// Root/CA key that issued this attestation (for chain revocation).
    pub attested_by: Option<[u8; 32]>,
}

/// Key-centric trust store. Thread-safe, lock-free reads.
///
/// Keyed by `VerifyingKey` (Ed25519 pubkey). Each entry records what
/// scopes that key is authorized for and when the attestation expires.
#[derive(Clone)]
pub struct TrustStore {
    inner: Arc<DashMap<VerifyingKey, Attestation>>,
}

impl TrustStore {
    /// Create an empty trust store.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Insert a verified attestation for a key.
    ///
    /// If the key already exists, the attestation is replaced if the new one
    /// expires later. Otherwise the existing entry is kept (prevents downgrade).
    /// Logs a warning on equal-expiry discard.
    pub fn insert(&self, key: VerifyingKey, attestation: Attestation) {
        let subject_debug = attestation.subject.as_deref().unwrap_or("<scope-derived>").to_owned();
        let scopes_debug: Vec<String> = attestation.scopes.iter().cloned().collect();
        let expires_debug = attestation.expires_at;
        self.inner
            .entry(key)
            .and_modify(|existing| {
                if attestation.expires_at > existing.expires_at {
                    tracing::info!(
                        key = ?key.to_bytes()[..4],
                        subject = %subject_debug,
                        scopes = ?scopes_debug,
                        expires_at = expires_debug,
                        "Trust store: updating attestation (later expiry)"
                    );
                    *existing = attestation.clone();
                } else if attestation.expires_at == existing.expires_at {
                    tracing::warn!(
                        key = ?key.to_bytes()[..4],
                        new_subject = %subject_debug,
                        existing_subject = ?existing.subject,
                        "Trust store: equal-expiry attestation discarded (first-writer wins)"
                    );
                }
            })
            .or_insert_with(|| {
                tracing::info!(
                    key = ?key.to_bytes()[..4],
                    subject = %subject_debug,
                    scopes = ?scopes_debug,
                    expires_at = expires_debug,
                    "Trust store: new attestation inserted"
                );
                attestation
            });
    }

    /// Check if a key is authorized for the given scope.
    ///
    /// Returns `true` if:
    /// 1. The key has an entry in the trust store
    /// 2. The scope is in the entry's scope set
    /// 3. The attestation has not expired
    pub fn is_authorized(&self, key: &VerifyingKey, scope: &str) -> bool {
        match self.inner.get(key) {
            Some(att) => {
                let now = chrono::Utc::now().timestamp();
                (att.expires_at == 0 || att.expires_at > now)
                    && att.scopes.contains(scope)
            }
            None => false,
        }
    }

    /// Get all keys authorized for a given scope.
    ///
    /// Returns an empty Vec if no keys match. Useful for finding
    /// candidate endpoints when multiple instances exist.
    pub fn keys_for_scope(&self, scope: &str) -> Vec<VerifyingKey> {
        let now = chrono::Utc::now().timestamp();
        self.inner
            .iter()
            .filter(|entry| {
                let att = entry.value();
                (att.expires_at == 0 || att.expires_at > now)
                    && att.scopes.contains(scope)
            })
            .map(|entry| *entry.key())
            .collect()
    }

    /// Get the attestation for a specific key, if present and not expired.
    pub fn get(&self, key: &VerifyingKey) -> Option<Attestation> {
        self.inner.get(key).and_then(|att| {
            let now = chrono::Utc::now().timestamp();
            if att.expires_at == 0 || att.expires_at > now {
                Some(att.clone())
            } else {
                None
            }
        })
    }

    /// Remove a specific key from the trust store.
    pub fn remove(&self, key: &VerifyingKey) {
        self.inner.remove(key);
    }

    /// Resolve a single key for a given scope.
    ///
    /// Returns the first valid (non-expired) key found for this scope.
    /// For multi-instance deployments, use `keys_for_scope()` instead.
    /// Returns `None` if no keys are authorized for this scope.
    pub fn resolve_one(&self, scope: &str) -> Option<VerifyingKey> {
        self.keys_for_scope(scope).into_iter().next()
    }

    /// Resolve a signer key to an authorization subject.
    ///
    /// For user keys: returns `attestation.subject` (e.g., "alice").
    /// For service keys: returns `"service:{first_scope}"` (e.g., "service:model").
    /// Returns `None` if the key is not in the trust store or is expired.
    pub fn resolve_subject(&self, signer_pubkey: &[u8; 32]) -> Option<hyprstream_rpc::envelope::Subject> {
        let vk = VerifyingKey::from_bytes(signer_pubkey).ok()?;
        let att = self.get(&vk)?;
        // User keys: explicit subject
        if let Some(ref subject) = att.subject {
            return Some(hyprstream_rpc::envelope::Subject::new(subject.clone()));
        }
        // Service keys: derive from scope
        att.scopes.iter().next().map(|scope| {
            hyprstream_rpc::envelope::Subject::new(format!("service:{scope}"))
        })
    }

    /// Remove all entries whose attestation was issued by the given root key.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_root(&self, root_key: &VerifyingKey) -> usize {
        let root_bytes = root_key.to_bytes();
        let before = self.inner.len();
        self.inner.retain(|_, att| {
            att.attested_by.as_ref() != Some(&root_bytes)
        });
        let removed = before - self.inner.len();
        if removed > 0 {
            tracing::info!(
                root = ?root_bytes[..8],
                removed,
                "Invalidated keys attested by revoked root"
            );
        }
        removed
    }

    /// Number of entries in the trust store (including expired).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the trust store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for TrustStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter exposing the global trust store as the `hyprstream-rpc`
/// signer-key → subject resolver (#446).
///
/// `hyprstream-rpc`'s dispatch core cannot reference the trust store directly
/// (the dependency points the other way), so it defines a
/// [`hyprstream_rpc::auth::KeySubjectResolver`] seam. This adapter wires the
/// authoritative trust store (populated fail-closed under #441) into that seam
/// so a signed service-to-service IPC caller resolves as its `service:<name>`
/// identity instead of `anonymous`. Unregistered keys resolve to `None`
/// (fail-closed); resolution never fabricates an identity.
struct TrustStoreSubjectResolver;

impl hyprstream_rpc::auth::KeySubjectResolver for TrustStoreSubjectResolver {
    fn resolve_subject(
        &self,
        signer_pubkey: &[u8; 32],
    ) -> Option<hyprstream_rpc::envelope::Subject> {
        global_trust_store().resolve_subject(signer_pubkey)
    }
}

/// Global trust store singleton.
///
/// Set once during startup (before any service threads spawn), readable
/// from any thread thereafter. This is the "CA bundle" — the set of
/// keys the process trusts.
///
/// On first access the trust store also installs itself as the process-global
/// `hyprstream-rpc` key→subject resolver (#446), so every dispatched request
/// (across all service crates) maps a verified signer key to its authoritative
/// `service:<name>`/user subject via this store. The trust store is always
/// touched during startup, so this guarantees the resolver is wired before any
/// request is served.
static TRUST_STORE: once_cell::sync::Lazy<TrustStore> = once_cell::sync::Lazy::new(|| {
    hyprstream_rpc::auth::set_global_key_subject_resolver(std::sync::Arc::new(
        TrustStoreSubjectResolver,
    ));
    TrustStore::new()
});

/// Get the global trust store.
pub fn global_trust_store() -> &'static TrustStore {
    &TRUST_STORE
}


#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn random_key() -> (SigningKey, VerifyingKey) {
        let sk = SigningKey::generate(&mut rand::rngs::OsRng);
        let vk = sk.verifying_key();
        (sk, vk)
    }

    fn make_attestation(scopes: &[&str], expires_at: i64) -> Attestation {
        Attestation {
            scopes: scopes.iter().map(std::string::ToString::to_string).collect(),
            subject: None,
            jwt: None,
            expires_at,
            attested_by: None,
        }
    }

    #[test]
    fn insert_and_lookup() {
        let store = TrustStore::new();
        let (_, key) = random_key();
        let now = chrono::Utc::now().timestamp();
        let expires = now + 3600;

        store.insert(key, make_attestation(&["model"], expires));

        assert!(store.is_authorized(&key, "model"));
        assert!(!store.is_authorized(&key, "policy"));
    }

    /// #441 invariant: authoritative service-key resolution.
    /// A registered service resolves to *exactly* the registered key; an
    /// unregistered service resolves to `None` (which `handle_resolve_service_key`
    /// turns into a "service key '<name>' not registered" error — never a
    /// CA-derived guess).
    #[test]
    fn resolve_one_returns_registered_key_or_none() {
        let store = TrustStore::new();
        let (_, registered) = random_key();
        store.insert(registered, make_attestation(&["model"], 0));

        // Registered → exactly the registered key.
        assert_eq!(store.resolve_one("model"), Some(registered));

        // Unregistered scope → None (handler converts this to a clear error;
        // it must NOT derive a fallback key).
        assert_eq!(store.resolve_one("worker"), None);
    }

    #[test]
    fn expired_attestation_is_not_authorized() {
        let store = TrustStore::new();
        let (_, key) = random_key();
        let past = chrono::Utc::now().timestamp() - 3600;

        store.insert(key, make_attestation(&["model"], past));

        assert!(!store.is_authorized(&key, "model"));
    }

    #[test]
    fn zero_expiry_means_never_expires() {
        let store = TrustStore::new();
        let (_, key) = random_key();

        store.insert(key, make_attestation(&["model"], 0));

        assert!(store.is_authorized(&key, "model"));
    }

    #[test]
    fn multiple_keys_same_scope() {
        let store = TrustStore::new();
        let (_, key_a) = random_key();
        let (_, key_b) = random_key();
        let now = chrono::Utc::now().timestamp();
        let expires = now + 3600;

        store.insert(key_a, make_attestation(&["model"], expires));
        store.insert(key_b, make_attestation(&["model"], expires));

        let keys = store.keys_for_scope("model");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&key_a));
        assert!(keys.contains(&key_b));
    }

    #[test]
    fn no_downgrade_on_reinsert() {
        let store = TrustStore::new();
        let (_, key) = random_key();
        let now = chrono::Utc::now().timestamp();

        store.insert(key, make_attestation(&["model"], now + 7200));
        store.insert(key, make_attestation(&["model"], now + 3600)); // earlier expiry

        let att = match store.get(&key) {
            Some(a) => a,
            None => panic!("key should be present after insert"),
        };
        assert_eq!(att.expires_at, now + 7200); // kept the later expiry
    }

    #[test]
    fn impersonation_rejected() {
        let store = TrustStore::new();
        let (_, model_key) = random_key();
        let (_, policy_key) = random_key();
        let now = chrono::Utc::now().timestamp();
        let expires = now + 3600;

        store.insert(model_key, make_attestation(&["model"], expires));
        store.insert(policy_key, make_attestation(&["policy"], expires));

        // model key is NOT authorized for policy scope
        assert!(!store.is_authorized(&model_key, "policy"));
        // policy key is NOT authorized for model scope
        assert!(!store.is_authorized(&policy_key, "model"));
    }

    #[test]
    fn key_with_multiple_scopes() {
        let store = TrustStore::new();
        let (_, key) = random_key();
        let now = chrono::Utc::now().timestamp();
        let expires = now + 3600;

        store.insert(key, make_attestation(&["model", "inference"], expires));

        assert!(store.is_authorized(&key, "model"));
        assert!(store.is_authorized(&key, "inference"));
        assert!(!store.is_authorized(&key, "policy"));
    }

    #[test]
    fn remove_key() {
        let store = TrustStore::new();
        let (_, key) = random_key();
        let now = chrono::Utc::now().timestamp();
        let expires = now + 3600;

        store.insert(key, make_attestation(&["model"], expires));
        assert!(store.is_authorized(&key, "model"));

        store.remove(&key);
        assert!(!store.is_authorized(&key, "model"));
    }

    /// #446: accessing the global trust store installs it as the `hyprstream-rpc`
    /// signer-key → subject resolver, and a registered service key resolves to
    /// its authoritative `service:<scope>` subject through that seam (so a signed
    /// service-to-service IPC caller is no longer anonymous). An unregistered key
    /// resolves to `None` (fail-closed → anonymous).
    #[test]
    fn global_resolver_resolves_registered_service_key() {
        // Touching the global store installs the global key→subject resolver.
        let store = global_trust_store();
        let (_, service_key) = random_key();
        let (_, unregistered) = random_key();
        store.insert(service_key, make_attestation(&["discovery"], 0));

        // The rpc-layer global resolver routes through this same trust store.
        let registered = hyprstream_rpc::auth::key_subject_resolver::resolve_subject(
            &service_key.to_bytes(),
        );
        assert_eq!(
            registered.map(|s| s.to_string()),
            Some("service:discovery".to_owned()),
            "registered service key must resolve as service:discovery, not anonymous"
        );

        // Fail-closed: an unregistered key resolves to None → anonymous.
        assert!(
            hyprstream_rpc::auth::key_subject_resolver::resolve_subject(&unregistered.to_bytes())
                .is_none(),
            "unregistered key must NOT resolve to any identity"
        );

        // Cleanup so we don't leak into other tests sharing the global store.
        store.remove(&service_key);
    }
}
