//! Shared user CRUD logic for SCIM HTTP and ZMQ RPC transports.
//!
//! Both transports call into `UserService` — no duplicated CRUD logic.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;

use crate::auth::{UserFilter, UserProfile, UserStore, PubkeyEntry, decode_pubkey_base64};

/// Shared user information type (SCIM-informed).
#[derive(Debug, Clone)]
pub struct UserInfo {
    pub username: String,
    pub sub: String,
    pub pubkey_base64: String,
    pub name: Option<String>,
    pub email: Option<String>,
    pub email_verified: bool,
    pub active: bool,
    pub external_id: Option<String>,
    pub pubkeys: Vec<PubkeyInfo>,
}

/// Pubkey info for API responses.
#[derive(Debug, Clone)]
pub struct PubkeyInfo {
    pub fingerprint: String,
    pub pubkey_base64: String,
    pub label: Option<String>,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub algorithm: crate::auth::KeyAlgorithm,
    /// Bound ML-DSA-65 verifying key bytes for a hybrid record (#439); `None`
    /// for classical Ed25519.
    pub pq_pubkey: Option<Vec<u8>>,
}

impl From<&PubkeyEntry> for PubkeyInfo {
    fn from(entry: &PubkeyEntry) -> Self {
        Self {
            fingerprint: entry.fingerprint.clone(),
            pubkey_base64: STANDARD.encode(entry.pubkey.as_bytes()),
            label: entry.label.clone(),
            created_at: entry.created_at,
            last_used_at: entry.last_used_at,
            algorithm: entry.algorithm,
            pq_pubkey: entry.pq_pubkey.clone(),
        }
    }
}

/// Paginated user list result.
#[derive(Debug, Clone)]
pub struct UserList {
    pub users: Vec<UserInfo>,
    pub total_results: usize,
}

/// Fields to update on a user.
#[derive(Debug, Clone, Default)]
pub struct UserUpdate {
    pub name: Option<Option<String>>,
    pub email: Option<Option<String>>,
    pub email_verified: Option<bool>,
    pub external_id: Option<Option<String>>,
    /// The account's mapped atproto DID (#1113 r5 / #1124 seam).
    pub atproto_did: Option<Option<String>>,
}

/// Shared user CRUD service used by both SCIM HTTP and ZMQ RPC transports.
pub struct UserService {
    store: Arc<dyn UserStore>,
}

impl UserService {
    pub fn new(store: Arc<dyn UserStore>) -> Self {
        Self { store }
    }

    /// Register a new user. If `pubkey_base64` is non-empty, adds it as the first key.
    pub async fn register(&self, username: &str, pubkey_base64: &str) -> Result<UserInfo> {
        self.store.register(username).await?;

        if !pubkey_base64.is_empty() {
            let pubkey = decode_pubkey_base64(pubkey_base64)?;
            self.store.add_pubkey(username, pubkey, None).await?;
        }

        self.get(username)
            .await?
            .ok_or_else(|| anyhow!("User '{}' not found after registration", username))
    }

    /// Get a user by username.
    pub async fn get(&self, username: &str) -> Result<Option<UserInfo>> {
        let profile = self.store.get_profile(username).await?;
        let pubkeys = self.store.list_pubkeys(username).await.unwrap_or_default();

        match profile {
            Some(profile) => {
                // For backward compat, use first pubkey as the primary
                let primary_pubkey = pubkeys.first()
                    .map(|pk| STANDARD.encode(pk.pubkey.as_bytes()))
                    .unwrap_or_default();

                Ok(Some(UserInfo {
                    username: username.to_owned(),
                    sub: profile.sub.unwrap_or_default(),
                    pubkey_base64: primary_pubkey,
                    name: profile.name,
                    email: profile.email,
                    email_verified: profile.email_verified.unwrap_or(false),
                    active: profile.active.unwrap_or(true),
                    external_id: profile.external_id,
                    pubkeys: pubkeys.iter().map(PubkeyInfo::from).collect(),
                }))
            }
            None => Ok(None),
        }
    }

    /// List/search users with SCIM-aligned filtering, sorting, and pagination.
    pub async fn list(&self, filter: &UserFilter) -> Result<UserList> {
        let results = self.store.search(filter).await?;

        // Total count is before pagination (we need a separate count without pagination).
        // For simplicity, search() already applies pagination, so we re-run without
        // pagination to get total. This is fine for in-memory stores with small user counts.
        let total_filter = UserFilter {
            count: None,
            start_index: None,
            ..filter.clone()
        };
        let total_results = self.store.search(&total_filter).await?.len();

        let users = results
            .into_iter()
            .map(|(username, profile)| {
                UserInfo {
                    username,
                    sub: profile.sub.unwrap_or_default(),
                    pubkey_base64: String::new(), // Omitted in list for size; use get() for full info
                    name: profile.name,
                    email: profile.email,
                    email_verified: profile.email_verified.unwrap_or(false),
                    active: profile.active.unwrap_or(true),
                    external_id: profile.external_id,
                    pubkeys: vec![], // Omitted in list
                }
            })
            .collect();

        Ok(UserList {
            users,
            total_results,
        })
    }

    /// Update a user's profile fields.
    pub async fn update(&self, username: &str, update: UserUpdate) -> Result<UserInfo> {
        let existing = self.store
            .get_profile(username).await?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let merged = UserProfile {
            sub: existing.sub,
            name: update.name.unwrap_or(existing.name),
            email: update.email.unwrap_or(existing.email),
            email_verified: Some(update.email_verified.unwrap_or_else(|| existing.email_verified.unwrap_or(false))),
            active: existing.active,
            external_id: update.external_id.unwrap_or(existing.external_id),
            atproto_did: update.atproto_did.unwrap_or(existing.atproto_did),
        };
        self.store.set_profile(username, merged).await?;

        self.get(username)
            .await?
            .ok_or_else(|| anyhow!("User '{}' not found after update", username))
    }

    /// Suspend a user (set active = false).
    pub async fn suspend(&self, username: &str) -> Result<()> {
        self.store.set_active(username, false).await
    }

    /// Resume a suspended user (set active = true).
    pub async fn resume(&self, username: &str) -> Result<()> {
        self.store.set_active(username, true).await
    }

    /// Permanently remove a user.
    pub async fn remove(&self, username: &str) -> Result<bool> {
        self.store.remove(username).await
    }

    /// Add a public key to a user. Returns the new PubkeyInfo.
    pub async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<PubkeyInfo> {
        let fingerprint = self.store.add_pubkey(username, pubkey, label).await?;
        let entries = self.store.list_pubkeys(username).await?;
        entries
            .iter()
            .find(|e| e.fingerprint == fingerprint)
            .map(PubkeyInfo::from)
            .ok_or_else(|| anyhow!("pubkey not found after insert"))
    }

    /// Remove a public key by fingerprint. Returns true if removed.
    pub async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        self.store.remove_pubkey(username, fingerprint).await
    }

    /// List all public keys for a user.
    pub async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyInfo>> {
        let entries = self.store.list_pubkeys(username).await?;
        Ok(entries.iter().map(PubkeyInfo::from).collect())
    }

    /// Get the underlying store for direct access (e.g., by OAuth handlers).
    pub fn store(&self) -> Arc<dyn UserStore> {
        Arc::clone(&self.store)
    }
}
