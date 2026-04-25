//! User credential store abstraction.
//!
//! `UserStore` is the trait. `LocalKeyStore` is the implementation
//! backed by an age-encrypted TOML file, with the decryption key
//! persisted in the secrets directory.

use anyhow::{anyhow, Context, Result};
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
pub trait UserStore: Send + Sync {
    /// Look up a user's Ed25519 public key by username.
    fn get_pubkey(&self, username: &str) -> Result<Option<VerifyingKey>>;
    /// Get a user's profile (OIDC claims).
    fn get_profile(&self, username: &str) -> Result<Option<UserProfile>>;
    /// Register a user with their Ed25519 public key.
    fn register(&mut self, username: &str, pubkey: VerifyingKey) -> Result<()>;
    /// Update a user's profile.
    fn set_profile(&mut self, username: &str, profile: UserProfile) -> Result<()>;
    /// Remove a user.
    fn remove(&mut self, username: &str) -> Result<bool>;
    /// List all registered usernames.
    fn list_users(&self) -> Vec<String>;
    /// Search users with SCIM-aligned filtering, sorting, and pagination.
    fn search(&self, filter: &UserFilter) -> Vec<(String, UserProfile)>;
    /// Set a user's active status.
    fn set_active(&mut self, username: &str, active: bool) -> Result<()>;
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
pub struct LocalKeyStore {
    path: PathBuf,
    /// In-memory state (loaded at startup, written on mutation)
    data: UsersFile,
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

        Ok(Self { path, data, identity })
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

    fn encrypt_and_write(&self) -> Result<()> {
        let plaintext =
            toml::to_string_pretty(&self.data).context("Failed to serialize users.toml")?;
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
}

impl UserStore for LocalKeyStore {
    fn get_pubkey(&self, username: &str) -> Result<Option<VerifyingKey>> {
        match self.data.users.get(username) {
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

    fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        match self.data.users.get(username) {
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

    fn register(&mut self, username: &str, pubkey: VerifyingKey) -> Result<()> {
        if username.contains(':') {
            anyhow::bail!("Username '{}' must not contain ':'", username);
        }
        let b64 = STANDARD.encode(pubkey.as_bytes());
        if self.data.users.contains_key(username) {
            tracing::warn!(
                "Overwriting existing public key for user '{}' in credential store",
                username
            );
        }
        let entry = UserEntry {
            pubkey: b64,
            sub: Some(uuid::Uuid::new_v4().to_string()),
            name: None,
            email: None,
            email_verified: None,
            active: None,
            external_id: None,
        };
        self.data.users.insert(username.to_owned(), entry);
        self.encrypt_and_write()
    }

    fn set_profile(&mut self, username: &str, profile: UserProfile) -> Result<()> {
        let entry = self.data.users.get_mut(username)
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
        self.encrypt_and_write()
    }

    fn remove(&mut self, username: &str) -> Result<bool> {
        let removed = self.data.users.remove(username).is_some();
        if removed {
            self.encrypt_and_write()?;
        }
        Ok(removed)
    }

    fn list_users(&self) -> Vec<String> {
        self.data.users.keys().cloned().collect()
    }

    fn search(&self, filter: &UserFilter) -> Vec<(String, UserProfile)> {
        let mut results: Vec<(String, UserProfile)> = self.data.users.iter()
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

        results.into_iter().skip(start).take(count).collect()
    }

    fn set_active(&mut self, username: &str, active: bool) -> Result<()> {
        let entry = self.data.users.get_mut(username)
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;
        entry.active = Some(active);
        self.encrypt_and_write()
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
fn matches_filter(
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
            data: UsersFile::default(),
            identity,
        }
    }

    #[test]
    fn test_register_and_get_pubkey() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("alice", pubkey)?;
        let retrieved = store
            .get_pubkey("alice")?
            .ok_or_else(|| anyhow!("alice not found"))?;
        assert_eq!(retrieved.as_bytes(), pubkey.as_bytes());
        Ok(())
    }

    #[test]
    fn test_remove_user() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("bob", key)?;
        assert!(store.remove("bob")?);
        assert!(store.get_pubkey("bob")?.is_none());
        Ok(())
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity.clone());
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("carol", pubkey)?;

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

    #[test]
    fn test_list_users() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key1)?;
        store.register("bob", key2)?;
        let mut users = store.list_users();
        users.sort();
        assert_eq!(users, vec!["alice", "bob"]);
        Ok(())
    }

    #[test]
    fn test_register_rejects_colon_in_username() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let result = store.register("bad:user", key);
        assert!(result.is_err(), "register should reject usernames with ':'");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must not contain"), "error message should mention colon restriction");
        Ok(())
    }

    #[test]
    fn test_search_filter_eq() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key1)?;
        store.register("bob", key2)?;

        let results = store.search(&UserFilter {
            filter: Some(r#"userName eq "alice""#.to_owned()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");
        Ok(())
    }

    #[test]
    fn test_search_filter_pr() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key)?;

        let results = store.search(&UserFilter {
            filter: Some("userName pr".to_owned()),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");

        let results = store.search(&UserFilter {
            filter: Some("active pr".to_owned()),
            ..Default::default()
        });
        assert!(results.is_empty(), "active is None by default, pr should not match");
        Ok(())
    }

    #[test]
    fn test_search_pagination() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        for name in &["alice", "bob", "carol"] {
            let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
            store.register(name, key)?;
        }

        // Page 1: start_index=1, count=2
        let results = store.search(&UserFilter {
            start_index: Some(1),
            count: Some(2),
            ..Default::default()
        });
        assert_eq!(results.len(), 2);

        // Page 2: start_index=3, count=2
        let results = store.search(&UserFilter {
            start_index: Some(3),
            count: Some(2),
            ..Default::default()
        });
        assert_eq!(results.len(), 1);
        Ok(())
    }

    #[test]
    fn test_search_sorting() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key3 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("carol", key1)?;
        store.register("alice", key2)?;
        store.register("bob", key3)?;

        let results = store.search(&UserFilter {
            sort_by: Some("userName".to_owned()),
            sort_order: Some("ascending".to_owned()),
            ..Default::default()
        });
        assert_eq!(results[0].0, "alice");
        assert_eq!(results[1].0, "bob");
        assert_eq!(results[2].0, "carol");

        let results = store.search(&UserFilter {
            sort_by: Some("userName".to_owned()),
            sort_order: Some("descending".to_owned()),
            ..Default::default()
        });
        assert_eq!(results[0].0, "carol");
        assert_eq!(results[1].0, "bob");
        assert_eq!(results[2].0, "alice");
        Ok(())
    }

    #[test]
    fn test_set_active() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key)?;

        // Default: active is None (treated as true)
        let profile = store.get_profile("alice")?.unwrap();
        assert!(profile.active.is_none());

        // Suspend
        store.set_active("alice", false)?;
        let profile = store.get_profile("alice")?.unwrap();
        assert_eq!(profile.active, Some(false));

        // Active-only search excludes suspended user
        let results = store.search(&UserFilter {
            active_only: Some(true),
            ..Default::default()
        });
        assert!(results.is_empty());

        // Resume
        store.set_active("alice", true)?;
        let profile = store.get_profile("alice")?.unwrap();
        assert_eq!(profile.active, Some(true));
        Ok(())
    }

    #[test]
    fn test_new_fields_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity.clone());
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key)?;

        // Set profile with new fields
        store.set_profile("alice", UserProfile {
            sub: None,
            name: Some("Alice Smith".to_owned()),
            email: Some("alice@example.com".to_owned()),
            email_verified: Some(true),
            active: Some(true),
            external_id: Some("ext-123".to_owned()),
        })?;

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
