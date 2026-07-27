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
use parking_lot::Mutex;
use std::io::Cursor;
use std::path::Path;

use super::user_store::{
    matches_filter, pubkey_fingerprint, AccountKeyCustody, DeviceRecord, DeviceStore,
    ExternalIdentityBinding, HostedAccountProvisionError, HostedAccountProvisioning, KeyAlgorithm,
    PubkeyEntry, UserFilter, UserProfile, UserProfilePatch, UserStore,
};

const USER_PREFIX: &[u8] = b"user:";
const PUBKEY_PREFIX: &[u8] = b"pubkey:";
const ACCOUNT_AUTH_PREFIX: &[u8] = b"account-auth:";

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

fn account_auth_key(username: &str) -> Vec<u8> {
    let mut key = ACCOUNT_AUTH_PREFIX.to_vec();
    key.extend_from_slice(username.as_bytes());
    key
}

fn strip_user_prefix(key: &[u8]) -> Option<&str> {
    key.strip_prefix(USER_PREFIX)
        .and_then(|s| std::str::from_utf8(s).ok())
}

/// Helper text: reads a capnp Text field, returning None if not set or empty.
fn text_or_none(reader: capnp::Result<capnp::text::Reader<'_>>) -> Option<String> {
    reader
        .ok()
        .filter(|t| !t.is_empty())
        .and_then(|t| t.to_string().ok())
}

/// Internal representation of a pubkey entry for serialization.
#[derive(Debug, Clone)]
struct StoredPubkey {
    fingerprint: String,
    pubkey_base64: String,
    label: Option<String>,
    created_at: i64,
    last_used_at: i64, // 0 means never used
    /// Algorithm tag (#439). Defaults to Ed25519 for pre-#439 records.
    algorithm: KeyAlgorithm,
    /// Bound ML-DSA-65 verifying key bytes for a hybrid record (#439); `None`
    /// for classical Ed25519. Invariant: `algorithm.is_hybrid() ⇔ Some`.
    pq_pubkey: Option<Vec<u8>>,
}

/// Hosted-account authentication metadata is intentionally a separate record
/// from the legacy Cap'n Proto user object. This is the persisted seam for
/// custody flavor and 0..N external identities without widening control-plane
/// RPC schema. Old accounts have no sidecar and deserialize to the defaults.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct StoredAccountAuth {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    key_custody: Option<AccountKeyCustody>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    external_identities: Vec<ExternalIdentityBinding>,
}

pub struct RocksDbUserStore {
    db: rocksdb::DB,
    /// RocksDB permits only one process to hold the writer lock. This mutex
    /// additionally serializes conditional user/key inserts inside that
    /// process so the checks and write batch form one provisioning operation.
    provisioning_lock: Mutex<()>,
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

        Ok(Self {
            db,
            provisioning_lock: Mutex::new(()),
        })
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

        Ok(Self {
            db,
            provisioning_lock: Mutex::new(()),
        })
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
    fn serialize_profile(
        sub: &str,
        profile: &UserProfile,
        pubkeys: &[StoredPubkey],
    ) -> Result<Vec<u8>> {
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
            if let Some(ref did) = profile.atproto_did {
                ui.set_atproto_did(did);
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
                entry.set_algorithm(pk.algorithm.as_str());
                // Enforce the hybrid⇔pq_pubkey invariant on the write path too,
                // so a malformed StoredPubkey can never be persisted.
                match (pk.algorithm.is_hybrid(), &pk.pq_pubkey) {
                    (true, Some(vk)) => entry.set_pq_pubkey(vk),
                    (false, None) => {}
                    (true, None) => anyhow::bail!(
                        "hybrid pubkey {} has no ML-DSA-65 key material (refusing to persist)",
                        pk.fingerprint
                    ),
                    (false, Some(_)) => anyhow::bail!(
                        "classical pubkey {} carries ML-DSA-65 key material (refusing to persist)",
                        pk.fingerprint
                    ),
                }
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
            atproto_did: text_or_none(ui.get_atproto_did()),
            key_custody: None,
            external_identities: Vec::new(),
        };

        // Deserialize pubkeys list
        let mut pubkeys = Vec::new();
        if ui.has_pubkeys() {
            for pk in ui.get_pubkeys()? {
                let fingerprint = pk.get_fingerprint()?.to_string()?;
                // Pre-#439 records have no algorithm tag → default Ed25519.
                // A present-but-unknown tag must error rather than be misread as
                // Ed25519 (would silently downgrade a PQ-hybrid key written by a
                // newer build).
                let algorithm = match text_or_none(pk.get_algorithm()) {
                    Some(s) => KeyAlgorithm::parse(&s)?,
                    None => KeyAlgorithm::default(),
                };
                // `pqPubkey` absent/empty ⇒ None. Enforce the hybrid⇔pq_pubkey
                // invariant: a Hybrid record with no PQ bytes is a fail-closed
                // read error, never a silent downgrade to Ed25519.
                let pq_pubkey = match pk.get_pq_pubkey() {
                    Ok(bytes) if !bytes.is_empty() => Some(bytes.to_vec()),
                    _ => None,
                };
                match (algorithm.is_hybrid(), &pq_pubkey) {
                    (true, Some(_)) | (false, None) => {}
                    (true, None) => anyhow::bail!(
                        "hybrid pubkey record {fingerprint} is missing its ML-DSA-65 \
                         key material (refusing to read — fail closed)"
                    ),
                    (false, Some(_)) => anyhow::bail!(
                        "classical pubkey record {fingerprint} carries unexpected \
                         ML-DSA-65 key material (refusing to read)"
                    ),
                }
                pubkeys.push(StoredPubkey {
                    fingerprint,
                    pubkey_base64: pk.get_pubkey_base64()?.to_string()?,
                    label: text_or_none(pk.get_label()),
                    created_at: pk.get_created_at(),
                    last_used_at: pk.get_last_used_at(),
                    algorithm,
                    pq_pubkey,
                });
            }
        }

        Ok((sub, profile, pubkeys))
    }

    fn get_raw(&self, username: &str) -> Result<Option<(String, UserProfile, Vec<StoredPubkey>)>> {
        let key = user_key(username);
        match self.db.get(&key)? {
            Some(bytes) => {
                let (sub, mut profile, pubkeys) = Self::deserialize_profile(&bytes)
                    .with_context(|| format!("Failed to deserialize profile for '{}'", username))?;
                if let Some(metadata) = self.db.get(account_auth_key(username))? {
                    let stored: StoredAccountAuth = serde_json::from_slice(&metadata)
                        .with_context(|| {
                            format!(
                                "Failed to deserialize hosted-account auth metadata for '{}'",
                                username
                            )
                        })?;
                    profile.key_custody = stored.key_custody;
                    profile.external_identities = stored.external_identities;
                }
                Ok(Some((sub, profile, pubkeys)))
            }
            None => Ok(None),
        }
    }

    /// Store the user record and update pubkey reverse indexes.
    fn put_user(
        &self,
        username: &str,
        sub: &str,
        profile: &UserProfile,
        pubkeys: &[StoredPubkey],
    ) -> Result<()> {
        let bytes = Self::serialize_profile(sub, profile, pubkeys)?;
        let key = user_key(username);
        let metadata = serde_json::to_vec(&StoredAccountAuth {
            key_custody: profile.key_custody,
            external_identities: profile.external_identities.clone(),
        })?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put(key, bytes);
        batch.put(account_auth_key(username), metadata);
        self.db.write(batch)?;
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
        let _guard = self.provisioning_lock.lock();
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

    async fn provision_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        pubkey: VerifyingKey,
        custody: AccountKeyCustody,
    ) -> std::result::Result<HostedAccountProvisioning, HostedAccountProvisionError> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let _guard = self.provisioning_lock.lock();
        let fingerprint = pubkey_fingerprint(&pubkey);
        let usable_binding = |owner: &str,
                              raw: &(String, UserProfile, Vec<StoredPubkey>),
                              required_fingerprint: Option<&str>|
         -> std::result::Result<bool, HostedAccountProvisionError> {
            let (sub, profile, keys) = raw;
            if sub.is_empty()
                || profile.sub.as_deref() != Some(sub)
                || profile.active != Some(true)
                || !profile
                    .atproto_did
                    .as_deref()
                    .is_some_and(|did| did.starts_with("did:web:"))
                || profile.key_custody.is_none()
                || keys.is_empty()
            {
                return Ok(false);
            }
            for key in keys {
                if required_fingerprint.is_some_and(|required| key.fingerprint != required) {
                    continue;
                }
                let valid_key = URL_SAFE_NO_PAD
                    .decode(&key.pubkey_base64)
                    .ok()
                    .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
                    .and_then(|bytes| VerifyingKey::from_bytes(&bytes).ok())
                    .is_some_and(|key_bytes| pubkey_fingerprint(&key_bytes) == key.fingerprint);
                if !valid_key {
                    continue;
                }
                if self
                    .get_pubkey_index(&key.fingerprint)
                    .map_err(HostedAccountProvisionError::Backend)?
                    .as_deref()
                    == Some(owner)
                {
                    return Ok(true);
                }
            }
            Ok(false)
        };

        if let Some(raw) = self
            .get_raw(username)
            .map_err(HostedAccountProvisionError::Backend)?
        {
            let (_, profile, keys) = &raw;
            let exact_key = keys.iter().any(|key| {
                key.fingerprint == fingerprint
                    && key.algorithm == KeyAlgorithm::Ed25519
                    && key.pq_pubkey.is_none()
                    && URL_SAFE_NO_PAD
                        .decode(&key.pubkey_base64)
                        .is_ok_and(|bytes| bytes.as_slice() == pubkey.as_bytes())
            });
            let exact = profile.sub.as_deref() == Some(raw.0.as_str())
                && profile.active.is_some()
                && profile.atproto_did.as_deref() == Some(atproto_did)
                && profile.key_custody == Some(custody)
                && profile.external_identities.is_empty()
                && exact_key
                && self
                    .get_pubkey_index(&fingerprint)
                    .map_err(HostedAccountProvisionError::Backend)?
                    .as_deref()
                    == Some(username);
            if exact {
                return Ok(HostedAccountProvisioning {
                    sub: raw.0,
                    fingerprint,
                    resumed: true,
                });
            }
            if usable_binding(username, &raw, None)? {
                return Err(HostedAccountProvisionError::AccountAlreadyExists);
            }
            return Err(HostedAccountProvisionError::Backend(anyhow!(
                "existing hosted-account username has no usable credential binding"
            )));
        }

        if let Some(owner) = self
            .get_pubkey_index(&fingerprint)
            .map_err(HostedAccountProvisionError::Backend)?
        {
            let owner_raw = self
                .get_raw(&owner)
                .map_err(HostedAccountProvisionError::Backend)?
                .ok_or_else(|| {
                    HostedAccountProvisionError::Backend(anyhow!(
                        "hosted-account key owner profile is missing"
                    ))
                })?;
            if usable_binding(&owner, &owner_raw, Some(&fingerprint))? {
                return Err(HostedAccountProvisionError::KeyAlreadyBound);
            }
            return Err(HostedAccountProvisionError::Backend(anyhow!(
                "existing hosted-account key owner has no usable credential binding"
            )));
        }

        let sub = uuid::Uuid::new_v4().to_string();
        let profile = UserProfile {
            sub: Some(sub.clone()),
            // The OAuth authorize transaction activates this binding only
            // after durable PDS genesis publication succeeds.
            active: Some(false),
            atproto_did: Some(atproto_did.to_owned()),
            key_custody: Some(custody),
            external_identities: Vec::new(),
            ..Default::default()
        };
        let stored_key = StoredPubkey {
            fingerprint: fingerprint.clone(),
            pubkey_base64: URL_SAFE_NO_PAD.encode(pubkey.as_bytes()),
            label: Some("aegis-vault".to_owned()),
            created_at: chrono::Utc::now().timestamp(),
            last_used_at: 0,
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: None,
        };
        let profile_bytes = Self::serialize_profile(&sub, &profile, &[stored_key])
            .map_err(HostedAccountProvisionError::Backend)?;
        let metadata_bytes = serde_json::to_vec(&StoredAccountAuth {
            key_custody: profile.key_custody,
            external_identities: profile.external_identities.clone(),
        })
        .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put(user_key(username), profile_bytes);
        batch.put(account_auth_key(username), metadata_bytes);
        batch.put(pubkey_key(&fingerprint), username.as_bytes());
        self.db
            .write(batch)
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;

        Ok(HostedAccountProvisioning {
            sub,
            fingerprint,
            resumed: false,
        })
    }

    async fn activate_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        fingerprint: &str,
        custody: AccountKeyCustody,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        let _guard = self.provisioning_lock.lock();
        let (sub, mut profile, pubkeys) = self
            .get_raw(username)
            .map_err(HostedAccountProvisionError::Backend)?
            .ok_or_else(|| {
                HostedAccountProvisionError::Backend(anyhow!(
                    "staged hosted-account profile is missing"
                ))
            })?;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        let key_matches = pubkeys.iter().any(|key| {
            key.fingerprint == fingerprint
                && URL_SAFE_NO_PAD
                    .decode(&key.pubkey_base64)
                    .ok()
                    .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
                    .and_then(|bytes| VerifyingKey::from_bytes(&bytes).ok())
                    .is_some_and(|key_bytes| pubkey_fingerprint(&key_bytes) == fingerprint)
        });
        let reverse_matches = self
            .get_pubkey_index(fingerprint)
            .map_err(HostedAccountProvisionError::Backend)?
            .as_deref()
            == Some(username);
        if profile.sub.as_deref() != Some(sub.as_str())
            || profile.atproto_did.as_deref() != Some(atproto_did)
            || profile.key_custody != Some(custody)
            || !key_matches
            || !reverse_matches
        {
            return Err(HostedAccountProvisionError::Backend(anyhow!(
                "staged hosted-account binding changed before activation"
            )));
        }
        if profile.active == Some(true) {
            return Ok(());
        }
        if profile.active != Some(false) {
            return Err(HostedAccountProvisionError::Backend(anyhow!(
                "staged hosted-account binding has an invalid activation state"
            )));
        }
        profile.active = Some(true);
        let profile_bytes = Self::serialize_profile(&sub, &profile, &pubkeys)
            .map_err(HostedAccountProvisionError::Backend)?;
        self.db
            .put(user_key(username), profile_bytes)
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))
    }

    async fn set_profile(&self, username: &str, update: UserProfilePatch) -> Result<()> {
        let (mut sub, mut profile, pubkeys) = self
            .get_raw(username)
            .with_context(|| format!("User '{}' not found", username))?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        if let Some(Some(s)) = update.sub {
            sub = s;
        }
        if let Some(value) = update.name {
            profile.name = value;
        }
        if let Some(value) = update.email {
            profile.email = value;
        }
        if let Some(value) = update.email_verified {
            profile.email_verified = value;
        }
        if let Some(value) = update.active {
            profile.active = value;
        }
        if let Some(value) = update.external_id {
            profile.external_id = value;
        }
        if let Some(value) = update.atproto_did {
            profile.atproto_did = value;
        }
        if let Some(value) = update.key_custody {
            profile.key_custody = value;
        }
        if let Some(value) = update.external_identities {
            profile.external_identities = value;
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
                let mut batch = rocksdb::WriteBatch::default();
                batch.delete(&key);
                batch.delete(account_auth_key(username));
                self.db.write(batch)?;
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
                if !matches_filter(
                    expr,
                    &username,
                    &profile.sub,
                    &profile.external_id,
                    profile.active,
                ) {
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
                if descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }

        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);

        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let (sub, mut profile, pubkeys) = self
            .get_raw(username)
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

        let (_, _, stored) = self
            .get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let mut entries = Vec::with_capacity(stored.len());
        for sp in stored {
            let pubkey_bytes = URL_SAFE_NO_PAD
                .decode(&sp.pubkey_base64)
                .with_context(|| format!("Invalid base64 for pubkey {}", sp.fingerprint))?;
            let pubkey_arr: [u8; 32] = pubkey_bytes
                .try_into()
                .map_err(|_| anyhow!("Pubkey {} is not 32 bytes", sp.fingerprint))?;
            let pubkey = VerifyingKey::from_bytes(&pubkey_arr)?;

            entries.push(PubkeyEntry {
                fingerprint: sp.fingerprint,
                pubkey,
                label: sp.label,
                created_at: sp.created_at,
                last_used_at: if sp.last_used_at == 0 {
                    None
                } else {
                    Some(sp.last_used_at)
                },
                algorithm: sp.algorithm,
                pq_pubkey: sp.pq_pubkey,
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

        let _guard = self.provisioning_lock.lock();
        let (sub, profile, mut pubkeys) = self
            .get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        let fingerprint = pubkey_fingerprint(&pubkey);

        // Check if this fingerprint already exists for this user
        if pubkeys.iter().any(|pk| pk.fingerprint == fingerprint) {
            anyhow::bail!(
                "Pubkey with fingerprint {} already exists for user '{}'",
                fingerprint,
                username
            );
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
            // Classical binding: no PQ component (#439).
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: None,
        });

        self.put_user(username, &sub, &profile, &pubkeys)?;
        self.put_pubkey_index(&fingerprint, username)?;

        Ok(fingerprint)
    }

    async fn add_pubkey_hybrid(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        ml_dsa_vk: Vec<u8>,
        label: Option<String>,
    ) -> Result<String> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let _guard = self.provisioning_lock.lock();
        if ml_dsa_vk.is_empty() {
            anyhow::bail!("add_pubkey_hybrid: empty ML-DSA-65 verifying key");
        }

        let (sub, profile, mut pubkeys) = self
            .get_raw(username)?
            .ok_or_else(|| anyhow!("User '{}' not found", username))?;

        // Fingerprint is the Ed25519 anchor's (kid) — the PQ vk does not change it.
        let fingerprint = pubkey_fingerprint(&pubkey);

        // Reject a fingerprint bound to a *different* user.
        if let Some(existing_user) = self.get_pubkey_index(&fingerprint)? {
            if existing_user != username {
                anyhow::bail!("Pubkey already associated with user '{}'", existing_user);
            }
        }

        let pubkey_base64 = URL_SAFE_NO_PAD.encode(pubkey.as_bytes());
        if let Some(existing) = pubkeys.iter_mut().find(|pk| pk.fingerprint == fingerprint) {
            // In-place upgrade path: an existing record for the same anchor is
            // lifted Ed25519 → Hybrid. Hybrid → Hybrid is idempotent (re-bind).
            // There is no Hybrid → Ed25519 transition here (this method only
            // ever sets Hybrid), so the forbidden downgrade cannot occur.
            existing.algorithm = KeyAlgorithm::HybridEd25519MlDsa65;
            existing.pq_pubkey = Some(ml_dsa_vk);
            if label.is_some() {
                existing.label = label;
            }
        } else {
            let now = chrono::Utc::now().timestamp();
            pubkeys.push(StoredPubkey {
                fingerprint: fingerprint.clone(),
                pubkey_base64,
                label,
                created_at: now,
                last_used_at: 0,
                algorithm: KeyAlgorithm::HybridEd25519MlDsa65,
                pq_pubkey: Some(ml_dsa_vk),
            });
        }

        self.put_user(username, &sub, &profile, &pubkeys)?;
        self.put_pubkey_index(&fingerprint, username)?;

        Ok(fingerprint)
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let (sub, profile, mut pubkeys) = self
            .get_raw(username)?
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
        let (sub, profile, mut pubkeys) = self
            .get_raw(username)?
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
        let existing: Option<DeviceRecord> = self
            .db
            .get(&key)?
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
        Ok(self
            .db
            .get(&key)?
            .and_then(|v| serde_json::from_slice(&v).ok()))
    }

    async fn touch_device(&self, fingerprint: &str) -> anyhow::Result<()> {
        let key = device_key(fingerprint);
        let Some(bytes) = self.db.get(&key)? else {
            return Ok(());
        };
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
        let profile = store
            .get_profile("alice")
            .await?
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must not contain more than one"));
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

        let results = store
            .search(&UserFilter {
                filter: Some(r#"userName eq "alice""#.to_owned()),
                ..Default::default()
            })
            .await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "alice");
        Ok(())
    }

    #[tokio::test]
    async fn test_search_filter_pr() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let results = store
            .search(&UserFilter {
                filter: Some("userName pr".to_owned()),
                ..Default::default()
            })
            .await?;
        assert_eq!(results.len(), 1);

        let results = store
            .search(&UserFilter {
                filter: Some("active pr".to_owned()),
                ..Default::default()
            })
            .await?;
        assert!(
            results.is_empty(),
            "active is None by default, pr should not match"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_search_pagination() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        for name in &["alice", "bob", "carol"] {
            store.register(name).await?;
        }

        let results = store
            .search(&UserFilter {
                start_index: Some(1),
                count: Some(2),
                ..Default::default()
            })
            .await?;
        assert_eq!(results.len(), 2);

        let results = store
            .search(&UserFilter {
                start_index: Some(3),
                count: Some(2),
                ..Default::default()
            })
            .await?;
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

        let results = store
            .search(&UserFilter {
                sort_by: Some("userName".to_owned()),
                sort_order: Some("ascending".to_owned()),
                ..Default::default()
            })
            .await?;
        assert_eq!(results[0].0, "alice");
        assert_eq!(results[1].0, "bob");
        assert_eq!(results[2].0, "carol");

        let results = store
            .search(&UserFilter {
                sort_by: Some("userName".to_owned()),
                sort_order: Some("descending".to_owned()),
                ..Default::default()
            })
            .await?;
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

        let results = store
            .search(&UserFilter {
                active_only: Some(true),
                ..Default::default()
            })
            .await?;
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
            store
                .set_profile(
                    "alice",
                    UserProfile {
                        sub: None,
                        name: Some("Alice Smith".to_owned()),
                        email: Some("alice@example.com".to_owned()),
                        email_verified: Some(true),
                        active: Some(true),
                        external_id: Some("ext-123".to_owned()),
                        atproto_did: Some("did:plc:abcdefghijklmnqrstuvwx2p".to_owned()),
                        key_custody: Some(AccountKeyCustody::SelfCustody),
                        external_identities: vec![ExternalIdentityBinding {
                            issuer: "https://issuer.example".to_owned(),
                            subject: "alice-123".to_owned(),
                        }],
                    }
                    .into(),
                )
                .await?;
        }
        // Open a fresh store instance to verify persistence
        let store2 = RocksDbUserStore::open(dir.path())?;
        let profile = store2.get_profile("alice").await?.unwrap();
        assert_eq!(profile.name.as_deref(), Some("Alice Smith"));
        assert_eq!(profile.external_id.as_deref(), Some("ext-123"));
        assert_eq!(profile.active, Some(true));
        assert_eq!(profile.email_verified, Some(true));
        assert_eq!(
            profile.atproto_did.as_deref(),
            Some("did:plc:abcdefghijklmnqrstuvwx2p")
        );
        assert_eq!(profile.key_custody, Some(AccountKeyCustody::SelfCustody));
        assert_eq!(
            profile.external_identities,
            vec![ExternalIdentityBinding {
                issuer: "https://issuer.example".to_owned(),
                subject: "alice-123".to_owned(),
            }]
        );
        Ok(())
    }

    #[tokio::test]
    async fn hosted_account_provisioning_is_atomic_and_preserves_future_idp_seam() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        let alice_key = ed25519_dalek::SigningKey::from_bytes(&[0x71; 32]).verifying_key();
        let provisioned = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.accounts.example.test",
                alice_key,
                AccountKeyCustody::SelfCustody,
            )
            .await?;
        assert_eq!(provisioned.fingerprint, pubkey_fingerprint(&alice_key));
        assert!(!provisioned.resumed);
        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(
            profile.atproto_did.as_deref(),
            Some("did:web:alice.accounts.example.test")
        );
        assert_eq!(profile.key_custody, Some(AccountKeyCustody::SelfCustody));
        assert_eq!(profile.active, Some(false));
        assert!(profile.external_identities.is_empty());
        assert_eq!(
            store.get_pubkey_user(&provisioned.fingerprint).await?,
            Some("alice".to_owned())
        );
        let resumed = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.accounts.example.test",
                alice_key,
                AccountKeyCustody::SelfCustody,
            )
            .await?;
        assert!(resumed.resumed);
        assert_eq!(resumed.sub, provisioned.sub);
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.accounts.example.test",
                &provisioned.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await?;
        assert_eq!(
            store.get_profile("alice").await?.unwrap().active,
            Some(true)
        );
        assert!(matches!(
            store
                .provision_hosted_account(
                    "alice",
                    "did:web:alice.accounts.example.test",
                    ed25519_dalek::SigningKey::from_bytes(&[0x72; 32]).verifying_key(),
                    AccountKeyCustody::SelfCustody,
                )
                .await,
            Err(HostedAccountProvisionError::AccountAlreadyExists)
        ));
        assert!(matches!(
            store
                .provision_hosted_account(
                    "bob",
                    "did:web:bob.accounts.example.test",
                    alice_key,
                    AccountKeyCustody::SelfCustody,
                )
                .await,
            Err(HostedAccountProvisionError::KeyAlreadyBound)
        ));

        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    external_identities: Some(vec![ExternalIdentityBinding {
                        issuer: "https://issuer.example".to_owned(),
                        subject: "external-alice".to_owned(),
                    }]),
                    ..Default::default()
                },
            )
            .await?;
        assert_eq!(
            store
                .get_external_identity_user("https://issuer.example", "external-alice")
                .await?,
            Some("alice".to_owned())
        );
        assert_eq!(
            store.list_external_identities("alice").await?,
            vec![ExternalIdentityBinding {
                issuer: "https://issuer.example".to_owned(),
                subject: "external-alice".to_owned(),
            }]
        );
        Ok(())
    }

    #[tokio::test]
    async fn corrupt_exact_key_index_never_returns_trusted_key_conflict() -> Result<()> {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;

        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        let alice_key = SigningKey::from_bytes(&[0x74; 32]).verifying_key();
        let alice = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.accounts.example.test",
                alice_key,
                AccountKeyCustody::SelfCustody,
            )
            .await?;
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.accounts.example.test",
                &alice.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await?;

        let conflicting_key = SigningKey::from_bytes(&[0x75; 32]).verifying_key();
        let conflicting_fingerprint = pubkey_fingerprint(&conflicting_key);
        store
            .db
            .put(pubkey_key(&conflicting_fingerprint), b"alice")?;

        let attempt = || {
            store.provision_hosted_account(
                "bob",
                "did:web:bob.accounts.example.test",
                conflicting_key,
                AccountKeyCustody::SelfCustody,
            )
        };
        assert!(matches!(
            attempt().await,
            Err(HostedAccountProvisionError::Backend(_))
        ));

        let (sub, profile, mut keys) = store.get_raw("alice")?.unwrap();
        keys.push(StoredPubkey {
            fingerprint: conflicting_fingerprint,
            // The index claims the conflicting fingerprint, but these are
            // Alice's other key bytes. Recomputing the fingerprint must catch
            // the mismatch instead of trusting any usable owner key.
            pubkey_base64: URL_SAFE_NO_PAD.encode(alice_key.as_bytes()),
            label: Some("corrupt-mismatch".to_owned()),
            created_at: chrono::Utc::now().timestamp(),
            last_used_at: 0,
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: None,
        });
        store.put_user("alice", &sub, &profile, &keys)?;
        assert!(matches!(
            attempt().await,
            Err(HostedAccountProvisionError::Backend(_))
        ));

        keys.last_mut().unwrap().pubkey_base64 = "not-base64".to_owned();
        store.put_user("alice", &sub, &profile, &keys)?;
        assert!(matches!(
            attempt().await,
            Err(HostedAccountProvisionError::Backend(_))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_same_key_different_handles_never_reserves_two_accounts() -> Result<()> {
        let dir = TempDir::new()?;
        let store = std::sync::Arc::new(make_store(dir.path()));
        let key = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]).verifying_key();
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(3));

        let attempt = |handle: &'static str| {
            let store = store.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                store
                    .provision_hosted_account(
                        handle,
                        &format!("did:web:{handle}.accounts.example.test"),
                        key,
                        AccountKeyCustody::SelfCustody,
                    )
                    .await
                    .map(|provisioned| (handle, provisioned))
            })
        };
        let alice = attempt("alice");
        let bob = attempt("bob");
        barrier.wait().await;
        let results = [alice.await?, bob.await?];
        let winner = results
            .iter()
            .find_map(|result| result.as_ref().ok())
            .expect("one transaction must win");
        assert_eq!(
            results.iter().filter(|result| result.is_ok()).count(),
            1,
            "the same vault key must never stage two handles"
        );
        assert!(results
            .iter()
            .any(|result| matches!(result, Err(HostedAccountProvisionError::Backend(_)))));
        let loser = if winner.0 == "alice" { "bob" } else { "alice" };
        assert!(store.get_profile(winner.0).await?.is_some());
        assert!(store.get_profile(loser).await?.is_none());

        store
            .activate_hosted_account(
                winner.0,
                &format!("did:web:{}.accounts.example.test", winner.0),
                &winner.1.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await?;
        assert!(matches!(
            store
                .provision_hosted_account(
                    loser,
                    &format!("did:web:{loser}.accounts.example.test"),
                    key,
                    AccountKeyCustody::SelfCustody,
                )
                .await,
            Err(HostedAccountProvisionError::KeyAlreadyBound)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn test_atproto_did_can_be_explicitly_cleared() -> Result<()> {
        let dir = TempDir::new()?;
        let store = std::sync::Arc::new(make_store(dir.path()));
        store.register("alice").await?;
        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    atproto_did: Some(Some("did:plc:abcdefghijklmnqrstuvwx2p".to_owned())),
                    ..Default::default()
                },
            )
            .await?;

        let service = crate::services::oauth::user_service::UserService::new(store.clone());
        service
            .update(
                "alice",
                crate::services::oauth::user_service::UserUpdate {
                    atproto_did: Some(None),
                    ..Default::default()
                },
            )
            .await?;

        let profile = store.get_profile("alice").await?.unwrap();
        assert_eq!(profile.atproto_did, None);
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

        let fingerprint = store
            .add_pubkey("alice", pubkey, Some("laptop".to_owned()))
            .await?;
        assert!(!fingerprint.is_empty());

        let keys = store.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fingerprint);
        assert_eq!(keys[0].label.as_deref(), Some("laptop"));
        assert_eq!(keys[0].pubkey.as_bytes(), pubkey.as_bytes());
        assert!(keys[0].last_used_at.is_none());
        // #439: every stored pubkey carries an algorithm tag (Ed25519 today).
        assert_eq!(keys[0].algorithm, KeyAlgorithm::Ed25519);
        Ok(())
    }

    /// #439: the algorithm tag survives a RocksDB roundtrip (serialize →
    /// close → reopen → deserialize), so widening the store to PQ later is
    /// additive rather than a re-migration of existing records.
    #[tokio::test]
    async fn test_pubkey_algorithm_tag_persists_across_reopen() -> Result<()> {
        let dir = TempDir::new()?;
        {
            let store = make_store(dir.path());
            store.register("alice").await?;
            let pubkey = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
            store.add_pubkey("alice", pubkey, None).await?;
            // `store` is dropped here, releasing the RocksDB write lock so the
            // reopen below succeeds in the same process.
        }

        // Reopen the same store (exercises capnp serialize + deserialize path).
        let store2 = RocksDbUserStore::open(dir.path())?;
        let keys = store2.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].algorithm, KeyAlgorithm::Ed25519);
        assert_eq!(keys[0].algorithm.as_str(), "ed25519");
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_pubkeys_per_user() -> Result<()> {
        let dir = TempDir::new()?;
        let store = make_store(dir.path());
        store.register("alice").await?;

        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();

        let fp1 = store
            .add_pubkey("alice", key1, Some("laptop".to_owned()))
            .await?;
        let fp2 = store
            .add_pubkey("alice", key2, Some("phone".to_owned()))
            .await?;

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

        assert_eq!(
            store.get_pubkey_user(&alice_fp).await?,
            Some("alice".to_owned())
        );
        assert_eq!(
            store.get_pubkey_user(&bob_fp).await?,
            Some("bob".to_owned())
        );
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
        assert_eq!(
            store.get_pubkey_user(&fingerprint).await?,
            Some("alice".to_owned())
        );

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
        store
            .add_pubkey("alice", key, Some("first".to_owned()))
            .await?;

        // Adding same key again should fail
        let result = store
            .add_pubkey("alice", key, Some("second".to_owned()))
            .await;
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already associated"));
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
            fingerprint = store
                .add_pubkey("alice", key, Some("laptop".to_owned()))
                .await?;
        }

        // Reopen and verify
        let store2 = RocksDbUserStore::open(dir.path())?;
        let keys = store2.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fingerprint);
        assert_eq!(keys[0].label.as_deref(), Some("laptop"));

        // Reverse lookup should also work
        assert_eq!(
            store2.get_pubkey_user(&fingerprint).await?,
            Some("alice".to_owned())
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_hybrid_pubkey_round_trips_pq_material_across_reopen() -> Result<()> {
        let dir = TempDir::new()?;
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        // A stand-in ML-DSA-65 vk (bytes are opaque to the store).
        let pq_vk = vec![0xABu8; 1952];
        let fingerprint;
        {
            let store = make_store(dir.path());
            store.register("alice").await?;
            fingerprint = store
                .add_pubkey_hybrid("alice", key, pq_vk.clone(), Some("laptop".to_owned()))
                .await?;
        }

        // Reopen (exercises capnp serialize + deserialize of pqPubkey).
        let store2 = RocksDbUserStore::open(dir.path())?;
        let keys = store2.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fingerprint);
        assert_eq!(keys[0].algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        assert_eq!(keys[0].pq_pubkey.as_ref(), Some(&pq_vk));
        Ok(())
    }

    #[tokio::test]
    async fn test_hybrid_upgrade_in_place_preserves_fingerprint() -> Result<()> {
        let dir = TempDir::new()?;
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let store = make_store(dir.path());
        store.register("alice").await?;

        // Classical first, then upgrade the SAME anchor to hybrid.
        let fp1 = store.add_pubkey("alice", key, None).await?;
        let fp2 = store
            .add_pubkey_hybrid("alice", key, vec![0x11u8; 1952], None)
            .await?;
        assert_eq!(
            fp1, fp2,
            "hybrid upgrade keeps the Ed25519 anchor fingerprint"
        );

        let keys = store.list_pubkeys("alice").await?;
        assert_eq!(keys.len(), 1, "in-place upgrade, not a second key");
        assert_eq!(keys[0].algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        assert!(keys[0].pq_pubkey.is_some());
        Ok(())
    }

    #[test]
    fn test_serialize_rejects_invariant_violations() {
        // Hybrid tag with no PQ material must not be persistable.
        let bad_hybrid = StoredPubkey {
            fingerprint: "SHA256:x".to_owned(),
            pubkey_base64: "AAAA".to_owned(),
            label: None,
            created_at: 0,
            last_used_at: 0,
            algorithm: KeyAlgorithm::HybridEd25519MlDsa65,
            pq_pubkey: None,
        };
        let profile = UserProfile::default();
        assert!(
            RocksDbUserStore::serialize_profile("sub", &profile, &[bad_hybrid]).is_err(),
            "hybrid record with no PQ key must fail to serialize"
        );

        // Classical tag carrying PQ material is equally invalid.
        let bad_classical = StoredPubkey {
            fingerprint: "SHA256:y".to_owned(),
            pubkey_base64: "AAAA".to_owned(),
            label: None,
            created_at: 0,
            last_used_at: 0,
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: Some(vec![1, 2, 3]),
        };
        assert!(
            RocksDbUserStore::serialize_profile("sub", &profile, &[bad_classical]).is_err(),
            "classical record carrying PQ key material must fail to serialize"
        );
    }

    #[test]
    fn test_deserialize_rejects_unknown_algorithm_tag() {
        // Hand-build a UserInfo capnp record whose pubkey carries a future,
        // unknown algorithm tag; the read path must error (never silently
        // downgrade to Ed25519).
        let mut message = Builder::new_default();
        {
            let mut ui = message.init_root::<crate::oauth_capnp::user_info::Builder>();
            ui.set_sub("sub");
            let mut pk_list = ui.init_pubkeys(1);
            let mut e = pk_list.reborrow().get(0);
            e.set_fingerprint("SHA256:z");
            e.set_pubkey_base64("AAAA");
            e.set_created_at(0);
            e.set_last_used_at(0);
            e.set_algorithm("ml-dsa-99-future");
        }
        let mut bytes = vec![0u8]; // presence-flags prefix
        capnp::serialize::write_message(&mut bytes, &message).unwrap();

        let err = RocksDbUserStore::deserialize_profile(&bytes)
            .expect_err("unknown algorithm tag must fail the read");
        assert!(
            err.to_string()
                .to_lowercase()
                .contains("unknown key algorithm"),
            "error must name the unknown tag: {err}"
        );
    }
}
