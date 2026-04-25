//! Shared user CRUD logic for SCIM HTTP and ZMQ RPC transports.
//!
//! Both transports call into `UserService` — no duplicated CRUD logic.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use tokio::sync::RwLock;

use crate::auth::{UserFilter, UserProfile, UserStore};

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
}

/// Shared user CRUD service used by both SCIM HTTP and ZMQ RPC transports.
pub struct UserService {
    store: Arc<RwLock<Box<dyn UserStore>>>,
}

impl UserService {
    pub fn new(store: Box<dyn UserStore>) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
        }
    }

    /// Register a new user with their Ed25519 public key.
    pub async fn register(&self, username: &str, pubkey_base64: &str) -> Result<UserInfo> {
        let raw = STANDARD
            .decode(pubkey_base64)
            .map_err(|e| anyhow!("Invalid base64 for public key: {e}"))?;
        let bytes: [u8; 32] = raw
            .try_into()
            .map_err(|_| anyhow!("Public key must be 32 bytes (Ed25519)"))?;
        let pubkey = VerifyingKey::from_bytes(&bytes)
            .map_err(|e| anyhow!("Invalid Ed25519 public key: {e}"))?;

        let mut store = self.store.write().await;
        store.register(username, pubkey)?;
        drop(store);

        self.get(username)
            .await?
            .ok_or_else(|| anyhow!("User '{}' not found after registration", username))
    }

    /// Get a user by username.
    pub async fn get(&self, username: &str) -> Result<Option<UserInfo>> {
        let store = self.store.read().await;
        let profile = store.get_profile(username)?;
        let pubkey = store.get_pubkey(username)?;
        drop(store);

        match (profile, pubkey) {
            (Some(profile), Some(pubkey)) => Ok(Some(UserInfo {
                username: username.to_owned(),
                sub: profile.sub.unwrap_or_default(),
                pubkey_base64: STANDARD.encode(pubkey.as_bytes()),
                name: profile.name,
                email: profile.email,
                email_verified: profile.email_verified.unwrap_or(false),
                active: profile.active.unwrap_or(true),
                external_id: profile.external_id,
            })),
            _ => Ok(None),
        }
    }

    /// List/search users with SCIM-aligned filtering, sorting, and pagination.
    pub async fn list(&self, filter: &UserFilter) -> Result<UserList> {
        let store = self.store.read().await;
        let results = store.search(filter);
        drop(store);

        // Total count is before pagination (we need a separate count without pagination).
        // For simplicity, search() already applies pagination, so we re-run without
        // pagination to get total. This is fine for in-memory stores with small user counts.
        let total_filter = UserFilter {
            count: None,
            start_index: None,
            ..filter.clone()
        };
        let store = self.store.read().await;
        let total_results = store.search(&total_filter).len();
        drop(store);

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
        {
            let mut store = self.store.write().await;
            let existing = store
                .get_profile(username)?
                .ok_or_else(|| anyhow!("User '{}' not found", username))?;

            let merged = UserProfile {
                sub: existing.sub,
                name: update.name.unwrap_or(existing.name),
                email: update.email.unwrap_or(existing.email),
                email_verified: Some(update.email_verified.unwrap_or_else(|| existing.email_verified.unwrap_or(false))),
                active: existing.active,
                external_id: update.external_id.unwrap_or(existing.external_id),
            };
            store.set_profile(username, merged)?;
        }

        self.get(username)
            .await?
            .ok_or_else(|| anyhow!("User '{}' not found after update", username))
    }

    /// Suspend a user (set active = false).
    pub async fn suspend(&self, username: &str) -> Result<()> {
        let mut store = self.store.write().await;
        store.set_active(username, false)
    }

    /// Resume a suspended user (set active = true).
    pub async fn resume(&self, username: &str) -> Result<()> {
        let mut store = self.store.write().await;
        store.set_active(username, true)
    }

    /// Permanently remove a user.
    pub async fn remove(&self, username: &str) -> Result<bool> {
        let mut store = self.store.write().await;
        store.remove(username)
    }

    /// Get the underlying store for direct access (e.g., by OAuth handlers).
    pub fn store(&self) -> Arc<RwLock<Box<dyn UserStore>>> {
        Arc::clone(&self.store)
    }
}
