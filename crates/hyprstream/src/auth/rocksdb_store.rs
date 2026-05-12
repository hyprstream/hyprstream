//! RocksDB-backed user store with Cap'n Proto serialization.
//!
//! Replaces `LocalKeyStore` (age-encrypted TOML, full rewrite per mutation)
//! with RocksDB for atomic single-key updates and concurrent access.
//!
//! Key scheme:
//! - `user:{username}` → Cap'n Proto `UserInfo` message (packed)
//! - `pubkey:{fingerprint}` → username (reverse lookup for auth)
//!
//! The database directory is `credentials_dir/users.db/`.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use capnp::message::{Builder, ReaderOptions};
use ed25519_dalek::VerifyingKey;
use std::io::Cursor;
use std::path::Path;

use super::user_store::{pubkey_fingerprint, matches_filter, DeviceRecord, DeviceStore, PubkeyEntry, UserFilter, UserProfile, UserStore};

const USER_PREFIX: &[u8] = b"user:";
const PUBKEY_PREFIX: &[u8] = b"pubkey:";

fn user_key(username: &str) -> Vec<u8> {
    let mut key = USER_PREFIX.to_vec();
    key.extend_from_slice(username.as_bytes());
    key
}

fn pubkey_key(fingerprint: &str) -> Vec<u8> {
    let mut key = PUBKEY_PREFIX.to_vec();
    key.extend_from_slice(fingerprint.as_bytes());
    key
}

fn strip_user_prefix(key: &[u8]) -> Option<&str> {
    key.strip_prefix(USER_PREFIX).and_then(|s| std::str::from_utf8(s).ok())
}

/// Helper text: reads a capnp Text field, returning None if not set or empty.
fn text_or_none(reader: capnp::Result<capnp::text::Reader<'_>>) -> Option<String> {
    reader.ok().filter(|t| !t.is_empty()).and_then(|t| t.to_string().ok())
}

/// Internal representation of a pubkey entry for serialization.
#[derive(Debug, Clone)]
struct StoredPubkey {
    fingerprint: String,
    pubkey_base64: String,
    label: Option<String>,
    created_at: i64,
    last_used_at: i64, // 0 means never used
}

pub struct RocksDbUserStore {
    db: rocksdb::DB,
}

impl RocksDbUserStore {
    /// Open (or create) the RocksDB user store at the given directory.
    pub fn open(credentials_dir: &Path) -> Result<Self> {
        let db_path = credentials_dir.join("users.db");
        std::fs::create_dir_all(&db_path)?;

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);

        let db = rocksdb::DB::open(&opts, &db_path)
            .with_context(|| format!("Failed to open RocksDB at {:?}", db_path))?;

        Ok(Self { db })
    }

    /// Open the RocksDB user store in read-only mode.
    ///
    /// Does not acquire the write lock, so this succeeds even when the server
    /// is running. All mutation methods will return `Err` at the RocksDB level.
    pub fn open_readonly(credentials_dir: &Path) -> Result<Self> {
        let db_path = credentials_dir.join("users.db");

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(false);

        let db = rocksdb::DB::open_for_read_only(&opts, &db_path, false)
            .with_context(|| format!("Failed to open RocksDB (read-only) at {:?}", db_path))?;

        Ok(Self { db })
    }

    /// Returns true if this store was opened under the write lock.
    ///
    /// Currently always true — read-only stores have no flag, but callers can
    /// track this themselves by which constructor they called.
    pub fn is_writable(&self) -> bool {
        true
    }

    /// Bitfield flags for optional Bool fields (capnp Bool has no has_* method).
    const FLAG_EMAIL_VERIFIED: u8 = 0x01;
    const FLAG_ACTIVE: u8 = 0x02;

    /// Serialize a `UserProfile` + pubkeys into Cap'n Proto `UserInfo` bytes with a 1-byte
    /// presence prefix for optional Bool fields.
    fn serialize_profile(sub: &str, profile: &UserProfile, pubkeys: &[StoredPubkey]) -> Result<Vec<u8>> {
        let mut flags: u8 = 0;
        if profile.email_verified.is_some() {
            flags |= Self::FLAG_EMAIL_VERIFIED;
        }
        if profile.active.is_some() {
            flags |= Self::FLAG_ACTIVE;
        }

        let mut message = Builder::new_default();
        {
            let mut ui = message.init_root::<crate::oauth_capnp::user_info::Builder>();
            ui.set_sub(sub);
            if let Some(ref name) = profile.name {
                ui.set_name(name);
            }
            if let Some(ref email) = profile.email {
                ui.set_email(email);
            }
            ui.set_email_verified(profile.email_verified.unwrap_or(false));
            ui.set_active(profile.active.unwrap_or(false));
            if let Some(ref eid) = profile.external_id {
                ui.set_external_id(eid);
            }

            // Serialize pubkeys list
            let mut pk_list = ui.init_pubkeys(pubkeys.len() as u32);
            for (i, pk) in pubkeys.iter().enumerate() {
                let mut entry = pk_list.reborrow().get(i as u32);
                entry.set_fingerprint(&pk.fingerprint);
                entry.set_pubkey_base64(&pk.pubkey_base64);
                if let Some(ref label) = pk.label {
                    entry.set_label(label);
                }
                entry.set_created_at(pk.created_at);
                entry.set_last_used_at(pk.last_used_at);
            }
        }
        let mut bytes = vec![flags];
        capnp::serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Deserialize Cap'n Proto `UserInfo` bytes (with presence prefix) into sub + UserProfile + pubkeys.
    fn deserialize_profile(bytes: &[u8]) -> Result<(String, UserProfile, Vec<StoredPubkey>)> {
        if bytes.is_empty() {
            anyhow::bail!("empty profile data");
        }
        let flags = bytes[0];
        let cursor = Cursor::new(&bytes[1..]);
        let reader = capnp::serialize::read_message(cursor, ReaderOptions::new())?;
        let ui = reader.get_root::<crate::oauth_capnp::user_info::Reader>()?;

        let sub = ui.get_sub()?.to_string()?;
        let profile = UserProfile {
            sub: Some(sub.clone()),
            name: text_or_none(ui.get_name()),
            email: text_or_none(ui.get_email()),
            email_verified: if flags & Self::FLAG_EMAIL_VERIFIED != 0 {
                Some(ui.get_email_verified())
            } else {
                None
            },
            active: if flags & Self::FLAG_ACTIVE != 0 {
                Some(ui.get_active())
            } else {
                None
            },
            external_id: text_or_none(ui.get_external_id()),
        };

        // Deserialize pubkeys list
        let mut pubkeys = Vec::new();
        if ui.has_pubkeys() {
            for pk in ui.get_pubkeys()? {
                pubkeys.push(StoredPubkey {
                    fingerprint: pk.get_fingerprint()?.to_string()?,
                    pubkey_base64: pk.get_pubkey_base64()?.to_string()?,
                    label: text_or_none(pk.get_label()),
                    created_at: pk.get_created_at(),
                    last_used_at: pk.get_last_used_at(),
                });
            }
        }

        Ok((sub, profile, pubkeys))
    }

    fn get_raw(&self, username: &str) -> Result<Option<(String, UserProfile, Vec<StoredPubkey>)>> {
        let key = user_key(username);
        match self.db.get(&key)? {
            Some(bytes) => {
                let tuple = Self::deserialize_profile(&bytes)
                    .with_context(|| format!("Failed to deserialize profile for '{}'", username))?;
                Ok(Some(tuple))
            }
            None => Ok(None),
        }
    }

    /// Store the user record and update pubkey reverse indexes.
    fn put_user(&self, username: &str, sub: &str, profile: &UserProfile, pubkeys: &[StoredPubkey]) -> Result<()> {
        let bytes = Self::serialize_profile(sub, profile, pubkeys)?;
        let key = user_key(username);
        self.db.put(&key, bytes)?;
        Ok(())
    }

    /// Add a reverse index entry: pubkey fingerprint → username.
    fn put_pubkey_index(&self, fingerprint: &str, username: &str) -> Result<()> {
        let key = pubkey_key(fingerprint);
        self.db.put(&key, username.as_bytes())?;
        Ok(())
    }

    /// Remove a reverse index entry.
    fn delete_pubkey_index(&self, fingerprint: &str) -> Result<()> {
        let key = pubkey_key(fingerprint);
        self.db.delete(&key)?;
        Ok(())
    }

    /// Lookup username by pubkey fingerprint.
    fn get_pubkey_index(&self, fingerprint: &str) -> Result<Option<String>> {
        let key = pubkey_key(fingerprint);
        match self.db.get(&key)? {
            Some(bytes) => Ok(Some(String::from_utf8(bytes.clone())?)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl UserStore for RocksDbUserStore {
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        Ok(self.get_raw(username)?.map(|(_, p, _)| p))
    }

    async fn register(&self, username: &str) -> Result<String> {
        // Allow exactly one colon for OIDC namespaced subjects (e.g. "google:abc123").
        // The RocksDB key is stored as b"user:" + username, so "user:google:abc" is
        // still unambiguous under prefix_iterator(b"user:").
        let colon_count = username.matches(':').count();
        if colon_count > 1 || username.starts_with(':') || username.ends_with(':') {
            anyhow::bail!(
                "Username '{}' must not contain more than one ':', and must not start or end with ':'",
                username
            );
        }
        if self.get_raw(username)?.is_some() {
            tracing::warn!(
                "Overwriting existing entry for user '{}' in credential store",
                username
            );
        }
        let sub = uuid::Uuid::new_v4().to_string();
        let profile = UserProfile::default();
        self.put_user(username, &sub, &profile, &[])?;
        Ok(sub)
    }

    async fn set_profile(&self, username: &str, update: UserProfile) -> Result<()> {
        let (mut sub, mut profile, pubkeys) = self.get_raw(username)
            .with_context(|| format!("User '{}' not found", username))?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        if let Some(s) = update.sub {
            sub = s;
        }
        if update.name.is_some() {
            profile.name = update.name;
        }
        if update.email.is_some() {
            profile.email = update.email;
        }
        if update.email_verified.is_some() {
            profile.email_verified = update.email_verified;
        }
        if update.active.is_some() {
            profile.active = update.active;
        }
        if update.external_id.is_some() {
            profile.external_id = update.external_id;
        }

        self.put_user(username, &sub, &profile, &pubkeys)?;
        Ok(())
    }

    async fn remove(&self, username: &str) -> Result<bool> {
        let key = user_key(username);
        match self.get_raw(username)? {
            Some((_, _, pubkeys)) => {
                // Remove all pubkey reverse indexes
                for pk in &pubkeys {
                    self.delete_pubkey_index(&pk.fingerprint)?;
                }
                self.db.delete(&key)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    async fn list_users(&self) -> Vec<String> {
        let mut users = Vec::new();
        let iter = self.db.prefix_iterator(USER_PREFIX);
        for item in iter {
            let (key, _) = match item {
                Ok(kv) => kv,
                Err(_) => continue,
            };
            if let Some(username) = strip_user_prefix(&key) {
                users.push(username.to_owned());
            }
        }
        users
    }

    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        let mut results: Vec<(String, UserProfile)> = Vec::new();
        let iter = self.db.prefix_iterator(USER_PREFIX);

        for item in iter {
            let (key, value) = match item {
                Ok(kv) => kv,
                Err(_) => continue,
            };
            let username = match strip_user_prefix(&key) {
                Some(u) => u.to_owned(),
                None => continue,
            };

            let (_sub, profile, _pubkeys) = match Self::deserialize_profile(&value) {
                Ok(p) => p,
                Err(_) => continue,
            };

            if filter.active_only == Some(true) && profile.active == Some(false) {
                continue;
            }

            if let Some(ref expr) = filter.filter {
                if !matches_filter(expr, &username, &profile.sub, &profile.external_id, profile.active) {
                    continue;
                }
            }

            results.push((username, profile));
        }

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

        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);

        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let (sub, mut profile, pubkeys) = self.get_raw(username)
            .with_context(|| format!("User '{}' not found", username))?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;
        profile.active = Some(active);
        self.put_user(username, &sub, &profile, &pubkeys)?;
        Ok(())
    }

    // ─── Pubkey Management ───────────────────────────────────────────────────

    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let (_, _, stored) = self.get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let mut entries = Vec::with_capacity(stored.len());
        for sp in stored {
            let pubkey_bytes = URL_SAFE_NO_PAD.decode(&sp.pubkey_base64)
                .with_context(|| format!("Invalid base64 for pubkey {}", sp.fingerprint))?;
            let pubkey_arr: [u8; 32] = pubkey_bytes.try_into()
                .map_err(|_| anyhow!("Pubkey {} is not 32 bytes", sp.fingerprint))?;
            let pubkey = VerifyingKey::from_bytes(&pubkey_arr)?;

            entries.push(PubkeyEntry {
                fingerprint: sp.fingerprint,
                pubkey,
                label: sp.label,
                created_at: sp.created_at,
                last_used_at: if sp.last_used_at == 0 { None } else { Some(sp.last_used_at) },
            });
        }
        Ok(entries)
    }

    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<String> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let (sub, profile, mut pubkeys) = self.get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let fingerprint = pubkey_fingerprint(&pubkey);

        // Check if this fingerprint already exists for this user
        if pubkeys.iter().any(|pk| pk.fingerprint == fingerprint) {
            anyhow::bail!("Pubkey with fingerprint {} already exists for user '{}'", fingerprint, username);
        }

        // Check if fingerprint is already associated with another user
        if let Some(existing_user) = self.get_pubkey_index(&fingerprint)? {
            if existing_user != username {
                anyhow::bail!("Pubkey already associated with user '{}'", existing_user);
            }
        }

        let now = chrono::Utc::now().timestamp();
        pubkeys.push(StoredPubkey {
            fingerprint: fingerprint.clone(),
            pubkey_base64: URL_SAFE_NO_PAD.encode(pubkey.as_bytes()),
            label,
            created_at: now,
            last_used_at: 0,
        });

        self.put_user(username, &sub, &profile, &pubkeys)?;
        self.put_pubkey_index(&fingerprint, username)?;

        Ok(fingerprint)
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let (sub, profile, mut pubkeys) = self.get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let original_len = pubkeys.len();
        pubkeys.retain(|pk| pk.fingerprint != fingerprint);

        if pubkeys.len() == original_len {
            return Ok(false);
        }

        self.put_user(username, &sub, &profile, &pubkeys)?;
        self.delete_pubkey_index(fingerprint)?;

        Ok(true)
    }

    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>> {
        self.get_pubkey_index(fingerprint)
    }

    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()> {
        let (sub, profile, mut pubkeys) = self.get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let now = chrono::Utc::now().timestamp();
        let mut found = false;
        for pk in &mut pubkeys {
            if pk.fingerprint == fingerprint {
                pk.last_used_at = now;
                found = true;
                break;
            }
        }

        if !found {
            anyhow::bail!("Pubkey {} not found for user '{}'", fingerprint, username);
        }

        self.put_user(username, &sub, &profile, &pubkeys)?;
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DeviceStore — anonymous device identity
// ────────────────────────────────────────────────────────────────────────────

const DEVICE_PREFIX: &[u8] = b"device:";

fn device_key(fingerprint: &str) -> Vec<u8> {
    let mut key = DEVICE_PREFIX.to_vec();
    key.extend_from_slice(fingerprint.as_bytes());
    key
}

#[async_trait]
impl DeviceStore for RocksDbUserStore {
    async fn enroll_device(&self, record: DeviceRecord) -> anyhow::Result<()> {
        let key = device_key(&record.fingerprint);
        // Preserve existing user_sub and label on re-enrollment.
        let existing: Option<DeviceRecord> = self.db.get(&key)?
            .and_then(|v| serde_json::from_slice(&v).ok());
        let to_store = if let Some(existing) = existing {
            DeviceRecord {
                user_sub: existing.user_sub.or(record.user_sub),
                label: existing.label.or(record.label),
                enrolled_at: record.enrolled_at,
                ..existing
            }
        } else {
            record
        };
        let bytes = serde_json::to_vec(&to_store)?;
        self.db.put(&key, &bytes)?;
        Ok(())
    }

    async fn link_device_user(&self, fingerprint: &str, user_sub: &str) -> anyhow::Result<()> {
        let key = device_key(fingerprint);
        let Some(bytes) = self.db.get(&key)? else {
            anyhow::bail!("device {} not found", fingerprint);
        };
        let mut record: DeviceRecord = serde_json::from_slice(&bytes)?;
        record.user_sub = Some(user_sub.to_owned());
        self.db.put(&key, serde_json::to_vec(&record)?)?;
        Ok(())
    }

    async fn get_device(&self, fingerprint: &str) -> anyhow::Result<Option<DeviceRecord>> {
        let key = device_key(fingerprint);
        Ok(self.db.get(&key)?.and_then(|v| serde_json::from_slice(&v).ok()))
    }

    async fn touch_device(&self, fingerprint: &str) -> anyhow::Result<()> {
        let key = device_key(fingerprint);
        let Some(bytes) = self.db.get(&key)? else { return Ok(()); };
        let mut record: DeviceRecord = serde_json::from_slice(&bytes)?;
        record.last_seen_at = Some(chrono::Utc::now().timestamp());
        self.db.put(&key, serde_json::to_vec(&record)?)?;
        Ok(())
    }

    async fn revoke_device(&self, fingerprint: &str) -> anyhow::Result<bool> {
        let key = device_key(fingerprint);
        let exists = self.db.get(&key)?.is_some();
        if exists {
            self.db.delete(&key)?;
        }
        Ok(exists)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use tempfile::TempDir;

    fn make_store(dir: &Path) -> RocksDbUserStore {
        RocksDbUserStore::open(dir).expect("Failed to open RocksDB store")
    }

    #[tokio::test]
    async fn test_register_and_get_profile() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        let sub = store.register("alice").await?;
        assert!(!sub.is_empty(), "sub should be returned on register");
        let profile = store.get_profile("alice").await?
            .ok_or_else(|| anyhow!("alice not found"))?;
        assert_eq!(profile.sub.as_deref(), Some(sub.as_str()));
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_user() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("bob").await?;
        assert!(store.remove("bob").await?);
        assert!(store.get_profile("bob").await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_list_users() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;
        store.register("bob").await?;
        let mut users = store.list_users().await;
        users.sort();
        assert_eq!(users, vec!["alice", "bob"]);
        Ok(())
    }

    #[tokio::test]
    async fn test_register_allows_namespaced_oidc_subject() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        // Single colon (provider:external_id) is allowed for federated OIDC identities.
        store.register("google:abc123").await?;
        let profile = store.get_profile("google:abc123").await?;
        assert!(profile.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_register_rejects_multiple_colons() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        let result = store.register("a:b:c").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must not contain more than one"));
        Ok(())
    }

    #[tokio::test]
    async fn test_register_rejects_leading_trailing_colon() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        assert!(store.register(":bad").await.is_err());
        assert!(store.register("bad:").await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_search_filter_eq() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
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
        let store = make_store(dir.path());
        store.register("alice").await?;

        let results = store.search(&UserFilter {
            filter: Some("userName pr".to_owned()),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 1);

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
        let store = make_store(dir.path());
        for name in &["alice", "bob", "carol"] {
            store.register(name).await?;
        }

        let results = store.search(&UserFilter {
            start_index: Some(1),
            count: Some(2),
            ..Default::default()
        }).await?;
        assert_eq!(results.len(), 2);

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
        let store = make_store(dir.path());
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
        let store = make_store(dir.path());
        store.register("alice").await?;

        let profile = store.get_profile("alice").await?.unwrap();
        assert!(profile.active.is_none());

        store.set_active("alice", false).await?;
        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(profile.active, Some(false));

        let results = store.search(&UserFilter {
            active_only: Some(true),
            ..Default::default()
        }).await?;
        assert!(results.is_empty());

        store.set_active("alice", true).await?;
        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(profile.active, Some(true));
        Ok(())
    }

    #[tokio::test]
    async fn test_new_fields_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        {
            let store = make_store(dir.path());
            store.register("alice").await?;
            store.set_profile("alice", UserProfile {
                sub: None,
                name: Some("Alice Smith".to_owned()),
                email: Some("alice@example.com".to_owned()),
                email_verified: Some(true),
                active: Some(true),
                external_id: Some("ext-123".to_owned()),
            }).await?;
        }
        // Open a fresh store instance to verify persistence
        let store2 = RocksDbUserStore::open(dir.path())?;
        let profile = store2.get_profile("alice").await?.unwrap();
        assert_eq!(profile.name.as_deref(), Some("Alice Smith"));
        assert_eq!(profile.external_id.as_deref(), Some("ext-123"));
        assert_eq!(profile.active, Some(true));
        assert_eq!(profile.email_verified, Some(true));
        Ok(())
    }

    #[tokio::test]
    async fn test_persistence_across_reopens() -> Result<()> {
        let dir = TempDir::new()?;
        {
            let store = make_store(dir.path());
            store.register("alice").await?;
            store.register("bob").await?;
        }
        let store2 = RocksDbUserStore::open(dir.path())?;
        let mut users = store2.list_users().await;
        users.sort();
        assert_eq!(users, vec!["alice", "bob"]);

        let profile = store2.get_profile("alice").await?.unwrap();
        assert!(profile.sub.is_some());
        Ok(())
    }

    // ─── Pubkey Management Tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_add_and_list_pubkeys() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();

        let fingerprint = store.add_pubkey("alice", pubkey, Some("laptop".to_owned())).await?;
        assert!(!fingerprint.is_empty());

        let keys = store.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fingerprint);
        assert_eq!(keys[0].label.as_deref(), Some("laptop"));
        assert_eq!(keys[0].pubkey.as_bytes(), pubkey.as_bytes());
        assert!(keys[0].last_used_at.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_pubkeys_per_user() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();

        let fp1 = store.add_pubkey("alice", key1, Some("laptop".to_owned())).await?;
        let fp2 = store.add_pubkey("alice", key2, Some("phone".to_owned())).await?;

        let keys = store.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 2);

        let fingerprints: Vec<_> = keys.iter().map(|k| k.fingerprint.as_str()).collect();
        assert!(fingerprints.contains(&fp1.as_str()));
        assert!(fingerprints.contains(&fp2.as_str()));
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_pubkey() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let fingerprint = store.add_pubkey("alice", key, None).await?;

        assert!(store.remove_pubkey("alice", &fingerprint).await?);
        assert!(!store.remove_pubkey("alice", &fingerprint).await?); // Already removed

        let keys = store.list_pubkeys("alice").await?;
        assert!(keys.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_get_pubkey_user_reverse_lookup() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;
        store.register("bob").await?;

        let alice_key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let bob_key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();

        let alice_fp = store.add_pubkey("alice", alice_key, None).await?;
        let bob_fp = store.add_pubkey("bob", bob_key, None).await?;

        assert_eq!(store.get_pubkey_user(&alice_fp).await?, Some("alice".to_owned()));
        assert_eq!(store.get_pubkey_user(&bob_fp).await?, Some("bob".to_owned()));
        assert_eq!(store.get_pubkey_user("nonexistent").await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_touch_pubkey() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let fingerprint = store.add_pubkey("alice", key, None).await?;

        // Initially last_used_at is None
        let keys = store.list_pubkeys("alice").await?;
        assert!(keys[0].last_used_at.is_none());

        // Touch updates last_used_at
        store.touch_pubkey("alice", &fingerprint).await?;
        let keys = store.list_pubkeys("alice").await?;
        assert!(keys[0].last_used_at.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_remove_user_cleans_pubkey_indexes() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let fingerprint = store.add_pubkey("alice", key, None).await?;

        // Verify reverse lookup works
        assert_eq!(store.get_pubkey_user(&fingerprint).await?, Some("alice".to_owned()));

        // Remove user
        store.remove("alice").await?;

        // Reverse lookup should now return None
        assert_eq!(store.get_pubkey_user(&fingerprint).await?, None);
        Ok(())
    }

    #[tokio::test]
    async fn test_duplicate_pubkey_rejected() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.add_pubkey("alice", key, Some("first".to_owned())).await?;

        // Adding same key again should fail
        let result = store.add_pubkey("alice", key, Some("second".to_owned())).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
        Ok(())
    }

    #[tokio::test]
    async fn test_pubkey_cross_user_rejected() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;
        store.register("bob").await?;

        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.add_pubkey("alice", key, None).await?;

        // Adding same key to different user should fail
        let result = store.add_pubkey("bob", key, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already associated"));
        Ok(())
    }

    #[tokio::test]
    async fn test_pubkeys_persist_across_reopens() -> Result<()> {
        let dir = TempDir::new()?;
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let fingerprint;
        {
            let store = make_store(dir.path());
            store.register("alice").await?;
            fingerprint = store.add_pubkey("alice", key, Some("laptop".to_owned())).await?;
        }

        // Reopen and verify
        let store2 = RocksDbUserStore::open(dir.path())?;
        let keys = store2.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fingerprint);
        assert_eq!(keys[0].label.as_deref(), Some("laptop"));

        // Reverse lookup should also work
        assert_eq!(store2.get_pubkey_user(&fingerprint).await?, Some("alice".to_owned()));
        Ok(())
    }
}
