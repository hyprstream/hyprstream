//! User credential store abstraction.
//!
//! `UserStore` is the trait. `LocalKeyStore` is the legacy implementation
//! backed by an age-encrypted TOML file. `RocksDbUserStore` is the new
//! implementation with atomic updates and multi-pubkey support.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// User profile data (OIDC standard claims + SCIM-informed fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    /// Stable subject identifier (UUID). Generated at registration, never changes.
    #[serde(default)]
    pub sub: Option<String>,
    /// Display name (SCIM: displayName / OIDC: name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Email address (SCIM: emails[0].value / OIDC: email).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Whether the email is verified (OIDC: email_verified).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email_verified: Option<bool>,
    /// Whether the user is active (SCIM: active). None defaults to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    /// External identity ID from upstream IdP (SCIM: externalId).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
}

/// A pubkey entry associated with a user (like GitHub SSH keys).
#[derive(Debug, Clone)]
pub struct PubkeyEntry {
    /// Base64url SHA-256 fingerprint of the pubkey bytes.
    pub fingerprint: String,
    /// The actual Ed25519 public key.
    pub pubkey: VerifyingKey,
    /// User-provided label (e.g., "laptop", "work").
    pub label: Option<String>,
    /// Unix timestamp when the key was added.
    pub created_at: i64,
    /// Unix timestamp when the key was last used for auth, or None if never.
    pub last_used_at: Option<i64>,
}

/// Filter parameters for user search (SCIM-aligned).
#[derive(Debug, Clone, Default)]
pub struct UserFilter {
    /// SCIM filter expression: `userName eq "alice"`, `active pr`, etc.
    /// Minimum viable: `eq` and `pr` on userName, id, externalId, active.
    pub filter: Option<String>,
    /// If true, only return active users.
    pub active_only: Option<bool>,
    /// SCIM: max results per page (default 100).
    pub count: Option<usize>,
    /// SCIM: 1-indexed start position (converted to 0-indexed internally).
    pub start_index: Option<usize>,
    /// SCIM: attribute name to sort by (e.g., "userName", "id").
    pub sort_by: Option<String>,
    /// SCIM: sort order ("ascending" or "descending"). Default: ascending.
    pub sort_order: Option<String>,
}

/// Abstraction over user credential stores.
///
/// Supports profile CRUD and multi-pubkey management (like GitHub SSH keys).
#[async_trait]
pub trait UserStore: Send + Sync {
    // ─── Profile CRUD ────────────────────────────────────────────────────────

    /// Get a user's profile (OIDC claims).
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>>;

    /// Register a new user. Returns the generated subject UUID.
    async fn register(&self, username: &str) -> Result<String>;

    /// Update a user's profile fields (merge semantics).
    async fn set_profile(&self, username: &str, profile: UserProfile) -> Result<()>;

    /// Remove a user and all their pubkeys.
    async fn remove(&self, username: &str) -> Result<bool>;

    /// List all registered usernames.
    async fn list_users(&self) -> Vec<String>;

    /// Search users with SCIM-aligned filtering, sorting, and pagination.
    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>>;

    /// Set a user's active status.
    async fn set_active(&self, username: &str, active: bool) -> Result<()>;

    // ─── Pubkey Management ───────────────────────────────────────────────────

    /// List all pubkeys for a user.
    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>>;

    /// Add a pubkey to a user. Returns the fingerprint.
    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<String>;

    /// Remove a pubkey by fingerprint.
    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool>;

    /// Reverse lookup: find username by pubkey fingerprint (for auth).
    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>>;

    /// Update last_used_at timestamp for a pubkey.
    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()>;
}

/// Compute the fingerprint of a pubkey (base64url SHA-256).
pub fn pubkey_fingerprint(pubkey: &VerifyingKey) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(pubkey.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

/// Per-user entry in the TOML file (rich format).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserEntry {
    /// Base64-standard-encoded 32-byte Ed25519 public key.
    pubkey: String,
    /// Stable subject UUID.
    #[serde(default)]
    sub: Option<String>,
    /// Display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// Email address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    email: Option<String>,
    /// Whether the email is verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    email_verified: Option<bool>,
    /// Whether the user is active. None defaults to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active: Option<bool>,
    /// External identity ID from upstream IdP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    external_id: Option<String>,
}

/// Serialized users file.
///
/// Supports two formats for backward compatibility:
/// - Old: `users: HashMap<String, String>` (username → base64 pubkey)
/// - New: `users: HashMap<String, UserEntry>` (username → structured entry)
///
/// On load, old format is auto-detected and migrated. UUID subs are generated
/// for entries that don't have one.
#[derive(Debug, Serialize, Deserialize, Default)]
struct UsersFile {
    /// username -> user entry (rich format)
    users: HashMap<String, UserEntry>,
}

/// Credential store backed by an age-encrypted TOML file.
/// The age decryption key is stored in the OS keyring.
///
/// DEPRECATED: Use `RocksDbUserStore` for new code. This implementation
/// does not support multi-pubkey and uses interior mutability for async compat.
pub struct LocalKeyStore {
    path: PathBuf,
    /// In-memory state (loaded at startup, written on mutation)
    data: parking_lot::RwLock<UsersFile>,
    /// age identity for decrypting/re-encrypting the file
    identity: age::x25519::Identity,
}

impl LocalKeyStore {
    /// Load (or initialize) the credential store.
    /// Creates the file if it doesn't exist.
    pub fn load(credentials_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(credentials_dir)?;
        let path = credentials_dir.join("users.toml.age");

        let identity = Self::load_or_generate_identity(&path)?;

        let data = if path.exists() {
            Self::decrypt_and_parse(&path, &identity)?
        } else {
            UsersFile::default()
        };

        Ok(Self { path, data: parking_lot::RwLock::new(data), identity })
    }

    /// Load the age identity from the configured secrets directory, or generate on first run.
    ///
    /// `store_path` is the path of the `users.toml.age` file. When it already exists
    /// on disk, a missing key is a hard error — generating a new key would produce a
    /// "NoMatchingKeys" decryption failure.
    ///
    /// Recovery: set `HYPRSTREAM__OAUTH__CREDENTIAL_STORE_KEY=<age-secret-key-...>`
    /// in the environment to inject the original key, or delete the credential store
    /// file to start fresh.
    fn load_or_generate_identity(store_path: &Path) -> Result<age::x25519::Identity> {
        if let Some(identity) = crate::config::HyprConfig::credential_store_key_bypass()? {
            return Ok(identity);
        }
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir();
        crate::auth::identity_store::load_or_generate_credential_store_key(&secrets_dir, store_path)
    }

    fn decrypt_and_parse(path: &Path, identity: &age::x25519::Identity) -> Result<UsersFile> {
        let ciphertext = std::fs::read(path)?;
        let decryptor = age::Decryptor::new_buffered(&ciphertext[..])
            .map_err(|e| anyhow!("Failed to init decryptor: {:?}", e))?;
        let mut plaintext = vec![];
        let mut reader = decryptor
            .decrypt(std::iter::once(identity as &dyn age::Identity))
            .map_err(|e| anyhow!("Decryption failed: {:?}", e))?;
        reader.read_to_end(&mut plaintext)?;
        let text = std::str::from_utf8(&plaintext)?;

        // Try new format first (structured UserEntry), fall back to old format
        // (plain string pubkey) for backward compatibility.
        if let Ok(file) = toml::from_str::<UsersFile>(text) {
            return Ok(file);
        }

        // Old format: users is HashMap<String, String> where value is base64 pubkey.
        #[derive(Deserialize)]
        struct OldUsersFile {
            users: HashMap<String, String>,
        }
        let old: OldUsersFile = toml::from_str(text).context("Failed to parse users.toml")?;
        let users = old.users.into_iter().map(|(username, pubkey)| {
            let entry = UserEntry {
                pubkey,
                sub: Some(uuid::Uuid::new_v4().to_string()),
                name: None,
                email: None,
                email_verified: None,
                active: None,
                external_id: None,
            };
            (username, entry)
        }).collect();
        Ok(UsersFile { users })
    }

    fn encrypt_and_write(&self, data: &UsersFile) -> Result<()> {
        let plaintext =
            toml::to_string_pretty(data).context("Failed to serialize users.toml")?;
        let recipient = self.identity.to_public();
        let encryptor =
            age::Encryptor::with_recipients(std::iter::once(&recipient as &dyn age::Recipient))
                .map_err(|e| anyhow!("Failed to create encryptor: {:?}", e))?;
        let mut ciphertext = vec![];
        let mut writer = encryptor
            .wrap_output(&mut ciphertext)
            .map_err(|e| anyhow!("Failed to init encryption: {:?}", e))?;
        writer.write_all(plaintext.as_bytes())?;
        writer
            .finish()
            .map_err(|e| anyhow!("Failed to finalize encryption: {:?}", e))?;
        std::fs::write(&self.path, ciphertext)?;
        Ok(())
    }

    /// Get a user's single pubkey (legacy method for backward compatibility).
    pub fn get_pubkey(&self, username: &str) -> Result<Option<VerifyingKey>> {
        let data = self.data.read();
        match data.users.get(username) {
            None => Ok(None),
            Some(entry) => {
                let raw = STANDARD.decode(&entry.pubkey)?;
                let bytes: [u8; 32] = raw
                    .try_into()
                    .map_err(|_| anyhow!("Stored pubkey for {} is not 32 bytes", username))?;
                Ok(Some(VerifyingKey::from_bytes(&bytes)?))
            }
        }
    }
}

#[async_trait]
impl UserStore for LocalKeyStore {
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        let data = self.data.read();
        match data.users.get(username) {
            None => Ok(None),
            Some(entry) => Ok(Some(UserProfile {
                sub: entry.sub.clone(),
                name: entry.name.clone(),
                email: entry.email.clone(),
                email_verified: entry.email_verified,
                active: entry.active,
                external_id: entry.external_id.clone(),
            })),
        }
    }

    async fn register(&self, username: &str) -> Result<String> {
        if username.contains(':') {
            anyhow::bail!("Username '{}' must not contain ':'", username);
        }
        let sub = uuid::Uuid::new_v4().to_string();
        {
            let mut data = self.data.write();
            if data.users.contains_key(username) {
                tracing::warn!(
                    "Overwriting existing entry for user '{}' in credential store",
                    username
                );
            }
            let entry = UserEntry {
                pubkey: String::new(), // No pubkey on register in new API
                sub: Some(sub.clone()),
                name: None,
                email: None,
                email_verified: None,
                active: None,
                external_id: None,
            };
            data.users.insert(username.to_owned(), entry);
            self.encrypt_and_write(&data)?;
        }
        Ok(sub)
    }

    async fn set_profile(&self, username: &str, profile: UserProfile) -> Result<()> {
        let mut data = self.data.write();
        let entry = data.users.get_mut(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;
        if let Some(sub) = profile.sub {
            entry.sub = Some(sub);
        }
        if profile.name.is_some() {
            entry.name = profile.name;
        }
        if profile.email.is_some() {
            entry.email = profile.email;
        }
        if profile.email_verified.is_some() {
            entry.email_verified = profile.email_verified;
        }
        if profile.active.is_some() {
            entry.active = profile.active;
        }
        if profile.external_id.is_some() {
            entry.external_id = profile.external_id;
        }
        self.encrypt_and_write(&data)
    }

    async fn remove(&self, username: &str) -> Result<bool> {
        let mut data = self.data.write();
        let removed = data.users.remove(username).is_some();
        if removed {
            self.encrypt_and_write(&data)?;
        }
        Ok(removed)
    }

    async fn list_users(&self) -> Vec<String> {
        self.data.read().users.keys().cloned().collect()
    }

    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        let data = self.data.read();
        let mut results: Vec<(String, UserProfile)> = data.users.iter()
            .filter_map(|(username, entry)| {
                let profile = UserProfile {
                    sub: entry.sub.clone(),
                    name: entry.name.clone(),
                    email: entry.email.clone(),
                    email_verified: entry.email_verified,
                    active: entry.active,
                    external_id: entry.external_id.clone(),
                };

                // Active-only filter
                if filter.active_only == Some(true) && profile.active == Some(false) {
                    return None;
                }

                // SCIM filter expression
                if let Some(ref expr) = filter.filter {
                    if !matches_filter(expr, username, &entry.sub, &entry.external_id, entry.active) {
                        return None;
                    }
                }

                Some((username.clone(), profile))
            })
            .collect();

        // Sorting
        if let Some(ref sort_by) = filter.sort_by {
            let descending = filter.sort_order.as_deref() == Some("descending");
            results.sort_by(|a, b| {
                let cmp = match sort_by.as_str() {
                    "userName" => a.0.cmp(&b.0),
                    "id" | "sub" => a.1.sub.cmp(&b.1.sub),
                    "active" => a.1.active.cmp(&b.1.active),
                    "displayName" | "name" => a.1.name.cmp(&b.1.name),
                    "externalId" => a.1.external_id.cmp(&b.1.external_id),
                    _ => std::cmp::Ordering::Equal,
                };
                if descending { cmp.reverse() } else { cmp }
            });
        }

        // Pagination
        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);

        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let mut data = self.data.write();
        let entry = data.users.get_mut(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;
        entry.active = Some(active);
        self.encrypt_and_write(&data)
    }

    // ─── Pubkey Management (limited support for legacy store) ────────────────

    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>> {
        let data = self.data.read();
        let entry = data.users.get(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        if entry.pubkey.is_empty() {
            return Ok(vec![]);
        }

        let raw = STANDARD.decode(&entry.pubkey)?;
        let bytes: [u8; 32] = raw
            .try_into()
            .map_err(|_| anyhow!("Stored pubkey for {} is not 32 bytes", username))?;
        let pubkey = VerifyingKey::from_bytes(&bytes)?;
        let fingerprint = pubkey_fingerprint(&pubkey);

        Ok(vec![PubkeyEntry {
            fingerprint,
            pubkey,
            label: Some("legacy".to_owned()),
            created_at: 0,
            last_used_at: None,
        }])
    }

    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        _label: Option<String>,
    ) -> Result<String> {
        let mut data = self.data.write();
        let entry = data.users.get_mut(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        if !entry.pubkey.is_empty() {
            anyhow::bail!("LocalKeyStore only supports one pubkey per user. Use RocksDbUserStore for multi-pubkey.");
        }

        let b64 = STANDARD.encode(pubkey.as_bytes());
        entry.pubkey = b64;
        self.encrypt_and_write(&data)?;

        Ok(pubkey_fingerprint(&pubkey))
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let mut data = self.data.write();
        let entry = data.users.get_mut(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        if entry.pubkey.is_empty() {
            return Ok(false);
        }

        // Verify fingerprint matches
        let raw = STANDARD.decode(&entry.pubkey)?;
        let bytes: [u8; 32] = raw
            .try_into()
            .map_err(|_| anyhow!("Stored pubkey is not 32 bytes"))?;
        let pubkey = VerifyingKey::from_bytes(&bytes)?;
        let stored_fp = pubkey_fingerprint(&pubkey);

        if stored_fp != fingerprint {
            return Ok(false);
        }

        entry.pubkey = String::new();
        self.encrypt_and_write(&data)?;
        Ok(true)
    }

    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>> {
        let data = self.data.read();
        for (username, entry) in &data.users {
            if entry.pubkey.is_empty() {
                continue;
            }
            let raw = match STANDARD.decode(&entry.pubkey) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let bytes: [u8; 32] = match raw.try_into() {
                Ok(b) => b,
                Err(_) => continue,
            };
            let pubkey = match VerifyingKey::from_bytes(&bytes) {
                Ok(pk) => pk,
                Err(_) => continue,
            };
            if pubkey_fingerprint(&pubkey) == fingerprint {
                return Ok(Some(username.clone()));
            }
        }
        Ok(None)
    }

    async fn touch_pubkey(&self, username: &str, _fingerprint: &str) -> Result<()> {
        // LocalKeyStore doesn't track last_used_at
        let data = self.data.read();
        if !data.users.contains_key(username) {
            anyhow::bail!("User '{}' not found", username);
        }
        Ok(())
    }
}

/// Evaluate a simple SCIM filter expression against a user entry.
///
/// Supports:
/// - `userName eq "value"` — exact match on username
/// - `id eq "value"` or `sub eq "value"` — exact match on subject UUID
/// - `externalId eq "value"` — exact match on external ID
/// - `active eq true/false` — match on active status
/// - `userName pr` — presence (non-empty/non-None)
/// - `active pr` — presence check
pub(crate) fn matches_filter(
    expr: &str,
    username: &str,
    sub: &Option<String>,
    external_id: &Option<String>,
    active: Option<bool>,
) -> bool {
    let expr = expr.trim();

    // Presence operator: `attribute pr`
    if let Some(attr) = expr.strip_suffix(" pr") {
        let attr = attr.trim();
        return match attr {
            "userName" => !username.is_empty(),
            "id" | "sub" => sub.is_some(),
            "externalId" => external_id.is_some(),
            "active" => active.is_some(),
            "displayName" | "name" | "email" => true, // always "present" even if None conceptually
            _ => false,
        };
    }

    // Equality operator: `attribute eq "value"` or `attribute eq true/false`
    if let Some(rest) = expr.strip_prefix("userName eq ") {
        return username == unquote(rest);
    }
    if let Some(rest) = expr.strip_prefix("id eq ") {
        return sub.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("sub eq ") {
        return sub.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("externalId eq ") {
        return external_id.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("active eq ") {
        let val = rest.trim();
        let expected = val == "true";
        return active.unwrap_or(true) == expected;
    }

    tracing::warn!(%expr, "Unsupported SCIM filter expression");
    false
}

/// Strip surrounding double-quotes from a SCIM filter value.
fn unquote(s: &str) -> &str {
    s.trim().trim_matches('"')
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use tempfile::TempDir;

    // Tests bypass the keyring by constructing LocalKeyStore directly with a generated identity.
    fn make_store_with_identity(dir: &Path, identity: age::x25519::Identity) -> LocalKeyStore {
        LocalKeyStore {
            path: dir.join("users.toml.age"),
            data: parking_lot::RwLock::new(UsersFile::default()),
            identity,
        }
    }

    #[tokio::test]
    async fn test_register_and_get_pubkey() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("alice").await?;
        store.add_pubkey("alice", pubkey, None).await?;
        let retrieved = store
            .get_pubkey("alice")?
            .ok_or_else(|| anyhow!("alice not found"))?;
        assert_eq!(retrieved.as_bytes(), pubkey.as_bytes());
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_user() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("bob").await?;
        assert!(store.remove("bob").await?);
        assert!(store.get_pubkey("bob")?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_encrypt_decrypt_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity.clone());
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("carol").await?;
        store.add_pubkey("carol", pubkey, None).await?;

        // Re-load from disk using the same identity
        let loaded_data =
            LocalKeyStore::decrypt_and_parse(&dir.path().join("users.toml.age"), &identity)?;
        let entry = loaded_data
            .users
            .get("carol")
            .ok_or_else(|| anyhow!("carol not found in loaded data"))?;
        let raw = STANDARD.decode(&entry.pubkey)?;
        let bytes: [u8; 32] = raw
            .try_into()
            .map_err(|_| anyhow!("wrong key length after roundtrip"))?;
        let loaded_key = VerifyingKey::from_bytes(&bytes)?;
        assert_eq!(loaded_key.as_bytes(), pubkey.as_bytes());
        // Verify UUID sub was generated
        assert!(entry.sub.is_some(), "sub should be generated on register");
        Ok(())
    }

    #[tokio::test]
    async fn test_list_users() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("alice").await?;
        store.register("bob").await?;
        let mut users = store.list_users().await;
        users.sort();
        assert_eq!(users, vec!["alice", "bob"]);
        Ok(())
    }

    #[tokio::test]
    async fn test_register_rejects_colon_in_username() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        let result = store.register("bad:user").await;
        assert!(result.is_err(), "register should reject usernames with ':'");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must not contain"), "error message should mention colon restriction");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_filter_eq() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("alice").await?;
        store.register("bob").await?;

        let results = store.search(&UserFilter {
            filter: Some(r#"userName eq "alice""#.to_owned()),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_filter_pr() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("alice").await?;

        let results = store.search(&UserFilter {
            filter: Some("userName pr".to_owned()),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");

        let results = store.search(&UserFilter {
            filter: Some("active pr".to_owned()),
            ..Default::default()
        }).await?;
        assert!(results.is_empty(), "active is None by default, pr should not match");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_pagination() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        for name in &["alice", "bob", "carol"] {
            store.register(name).await?;
        }

        // Page 1: start_index=1, count=2
        let results = store.search(&UserFilter {
            start_index: Some(1),
            count: Some(2),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 2);

        // Page 2: start_index=3, count=2
        let results = store.search(&UserFilter {
            start_index: Some(3),
            count: Some(2),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_search_sorting() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("carol").await?;
        store.register("alice").await?;
        store.register("bob").await?;

        let results = store.search(&UserFilter {
            sort_by: Some("userName".to_owned()),
            sort_order: Some("ascending".to_owned()),
            ..Default::default()
        }).await?;
        assert_eq!(results[0].0, "alice");
        assert_eq!(results[1].0, "bob");
        assert_eq!(results[2].0, "carol");

        let results = store.search(&UserFilter {
            sort_by: Some("userName".to_owned()),
            sort_order: Some("descending".to_owned()),
            ..Default::default()
        }).await?;
        assert_eq!(results[0].0, "carol");
        assert_eq!(results[1].0, "bob");
        assert_eq!(results[2].0, "alice");
        Ok(())
    }

    #[tokio::test]
    async fn test_set_active() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity);
        store.register("alice").await?;

        // Default: active is None (treated as true)
        let profile = store.get_profile("alice").await?.unwrap();
        assert!(profile.active.is_none());

        // Suspend
        store.set_active("alice", false).await?;
        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(profile.active, Some(false));

        // Active-only search excludes suspended user
        let results = store.search(&UserFilter {
            active_only: Some(true),
            ..Default::default()
        }).await?;
        assert!(results.is_empty());

        // Resume
        store.set_active("alice", true).await?;
        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(profile.active, Some(true));
        Ok(())
    }

    #[tokio::test]
    async fn test_new_fields_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let store = make_store_with_identity(dir.path(), identity.clone());
        store.register("alice").await?;

        // Set profile with new fields
        store.set_profile("alice", UserProfile {
            sub: None,
            name: Some("Alice Smith".to_owned()),
            email: Some("alice@example.com".to_owned()),
            email_verified: Some(true),
            active: Some(true),
            external_id: Some("ext-123".to_owned()),
        }).await?;

        // Reload from disk
        let loaded = LocalKeyStore::decrypt_and_parse(
            &dir.path().join("users.toml.age"),
            &identity,
        )?;
        let entry = &loaded.users["alice"];
        assert_eq!(entry.name.as_deref(), Some("Alice Smith"));
        assert_eq!(entry.external_id.as_deref(), Some("ext-123"));
        assert_eq!(entry.active, Some(true));
        Ok(())
    }
}
