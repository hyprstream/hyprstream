//! Valkey-backed user store.
//!
//! Requires the `valkey` feature. Uses fred (async Redis/Valkey client).
//!
//! Key schema:
//!   hs:users                  SET of all usernames
//!   hs:user:{u}               JSON UserProfile
//!   hs:user:{u}:keys          SET of fingerprints
//!   hs:key:{fp}               JSON pubkey data (base64, label, timestamps)
//!   hs:keyowner:{fp}          username string (fingerprint → user reverse index)
//!   hs:hosted-staged-binding:{u}:{fp} immutable expected hosted-binding bytes
//!   hs:hosted-binding:{u}:{fp} activated hosted-binding integrity marker
//!   hs:idx:sub:{sub}          username (sub → username reverse index)
//!   hs:idx:extid:{extid}      username (externalId → username reverse index)

#![cfg(feature = "valkey")]

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use fred::prelude::*;
use serde::{Deserialize, Serialize};

use super::user_store::{
    matches_filter, pubkey_fingerprint, AccountKeyCustody, HostedAccountProvisionError,
    HostedAccountProvisioning, PubkeyEntry, ScimFilter, UserFilter, UserProfile, UserProfilePatch,
    UserStore,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredKey {
    pubkey_base64: String,
    label: Option<String>,
    created_at: i64,
    last_used_at: Option<i64>,
    /// Algorithm tag (#439). Defaults to Ed25519 for pre-#439 records.
    #[serde(default)]
    algorithm: crate::auth::KeyAlgorithm,
    /// Standard-base64 ML-DSA-65 verifying key for a hybrid record (#439);
    /// `None`/absent for classical Ed25519. Invariant: present ⇔ hybrid tag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pq_pubkey_base64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostedBindingMarker {
    fingerprint: String,
    did: String,
    custody: String,
    pubkey_base64: String,
}

fn validate_stored_key_record(
    fingerprint: &str,
    key_json: &str,
) -> Result<(StoredKey, VerifyingKey, Option<Vec<u8>>)> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    let key: StoredKey = serde_json::from_str(key_json)?;
    let key_bytes: [u8; 32] = STANDARD
        .decode(&key.pubkey_base64)?
        .try_into()
        .map_err(|_| anyhow!("bad key length"))?;
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    if pubkey_fingerprint(&verifying_key) != fingerprint {
        anyhow::bail!("stored key does not match fingerprint {fingerprint}");
    }

    // Keep hosted-account activation/conflict classification and ordinary
    // authentication on one complete-record acceptance policy. A metadata-
    // corrupt record must never become a recovery-triggering trusted 409.
    let pq_pubkey = match key.pq_pubkey_base64.as_deref() {
        Some(b64) if !b64.is_empty() => Some(
            STANDARD
                .decode(b64)
                .with_context(|| format!("invalid base64 ML-DSA-65 key for {fingerprint}"))?,
        ),
        _ => None,
    };
    match (key.algorithm.is_hybrid(), &pq_pubkey) {
        (true, Some(_)) | (false, None) => {}
        (true, None) => anyhow::bail!(
            "hybrid pubkey record {fingerprint} is missing its ML-DSA-65 key \
             material (refusing to read — fail closed)"
        ),
        (false, Some(_)) => anyhow::bail!(
            "classical pubkey record {fingerprint} carries unexpected ML-DSA-65 \
             key material (refusing to read)"
        ),
    }

    Ok((key, verifying_key, pq_pubkey))
}

fn validated_stored_key_base64(
    fingerprint: &str,
    key_json: Option<&str>,
) -> std::result::Result<String, HostedAccountProvisionError> {
    let key_json = key_json.ok_or_else(|| {
        HostedAccountProvisionError::Backend(anyhow!("stored hosted-account key is missing"))
    })?;
    let (key, _, _) = validate_stored_key_record(fingerprint, key_json).map_err(|error| {
        HostedAccountProvisionError::Backend(
            error.context("stored hosted-account key is not authentication-usable"),
        )
    })?;
    Ok(key.pubkey_base64)
}

fn validate_key_conflict_snapshot(
    fingerprint: &str,
    owner: Option<&str>,
    owner_has_fingerprint: bool,
    profile_json: Option<&str>,
    key_json: Option<&str>,
    marker_json: Option<&str>,
) -> std::result::Result<(), HostedAccountProvisionError> {
    let invalid = || {
        HostedAccountProvisionError::Backend(anyhow!(
            "existing hosted-account key binding is incomplete or inconsistent"
        ))
    };
    owner
        .filter(|owner| !owner.is_empty())
        .ok_or_else(invalid)?;
    if !owner_has_fingerprint {
        return Err(invalid());
    }
    let profile: UserProfile =
        serde_json::from_str(profile_json.ok_or_else(invalid)?).map_err(|_| invalid())?;
    let did = profile
        .atproto_did
        .as_deref()
        .filter(|did| did.starts_with("did:web:"))
        .ok_or_else(invalid)?;
    let custody = profile
        .key_custody
        .filter(|_| profile.active == Some(true))
        .ok_or_else(invalid)?;
    let key_base64 = validated_stored_key_base64(fingerprint, key_json)?;
    let marker: HostedBindingMarker =
        serde_json::from_str(marker_json.ok_or_else(invalid)?).map_err(|_| invalid())?;
    if marker.fingerprint != fingerprint
        || marker.did != did
        || marker.custody != custody.as_str()
        || marker.pubkey_base64 != key_base64
    {
        return Err(invalid());
    }
    Ok(())
}

pub struct ValkeyUserStore {
    pool: RedisPool,
}

impl ValkeyUserStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let config = RedisConfig::from_url(url).context("invalid Valkey URL")?;
        let pool = Builder::from_config(config).build_pool(8)?;
        pool.connect();
        pool.wait_for_connect().await?;
        Ok(Self { pool })
    }

    async fn get_profile_raw(&self, username: &str) -> Result<Option<UserProfile>> {
        let key = format!("hs:user:{username}");
        let val: Option<String> = self.pool.get(&key).await?;
        match val {
            None => Ok(None),
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
        }
    }

    async fn validate_key_conflict(
        &self,
        fingerprint: &str,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        let owner: Option<String> = self
            .pool
            .get(format!("hs:keyowner:{fingerprint}"))
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let profile_json: Option<String> = match owner.as_deref() {
            Some(owner) => self
                .pool
                .get(format!("hs:user:{owner}"))
                .await
                .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?,
            None => None,
        };
        let key_json: Option<String> = self
            .pool
            .get(format!("hs:key:{fingerprint}"))
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let marker_json: Option<String> = match owner.as_deref() {
            Some(owner) => self
                .pool
                .get(format!("hs:hosted-binding:{owner}:{fingerprint}"))
                .await
                .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?,
            None => None,
        };
        let owner_has_fingerprint = match owner.as_deref() {
            Some(owner) => self
                .pool
                .sismember(format!("hs:user:{owner}:keys"), fingerprint)
                .await
                .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?,
            None => false,
        };
        validate_key_conflict_snapshot(
            fingerprint,
            owner.as_deref(),
            owner_has_fingerprint,
            profile_json.as_deref(),
            key_json.as_deref(),
            marker_json.as_deref(),
        )
    }

    async fn validate_account_conflict(
        &self,
        username: &str,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        // list_pubkeys applies the same complete-record policy as activation
        // and trusted key conflicts, and fails the whole account read if any
        // algorithm/PQ metadata is corrupt.
        let keys = self
            .list_pubkeys(username)
            .await
            .map_err(HostedAccountProvisionError::Backend)?;
        for key in keys {
            if self.validate_key_conflict(&key.fingerprint).await.is_ok() {
                return Ok(());
            }
        }
        Err(HostedAccountProvisionError::Backend(anyhow!(
            "existing hosted account has no complete active credential binding"
        )))
    }

    async fn validated_staged_public_key(
        &self,
        username: &str,
        atproto_did: &str,
        fingerprint: &str,
        custody: AccountKeyCustody,
    ) -> std::result::Result<String, HostedAccountProvisionError> {
        let key_json: Option<String> = self
            .pool
            .get(format!("hs:key:{fingerprint}"))
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let public_key = validated_stored_key_base64(fingerprint, key_json.as_deref())?;
        let marker_json: Option<String> = self
            .pool
            .get(format!("hs:hosted-staged-binding:{username}:{fingerprint}"))
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let marker: HostedBindingMarker =
            serde_json::from_str(marker_json.as_deref().ok_or_else(|| {
                HostedAccountProvisionError::Backend(anyhow!(
                    "staged hosted-account key binding marker is missing"
                ))
            })?)
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        if marker.fingerprint != fingerprint
            || marker.did != atproto_did
            || marker.custody != custody.as_str()
            || marker.pubkey_base64 != public_key
        {
            return Err(HostedAccountProvisionError::Backend(anyhow!(
                "staged hosted-account key binding marker is inconsistent"
            )));
        }
        Ok(public_key)
    }
}

#[async_trait]
impl UserStore for ValkeyUserStore {
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        self.get_profile_raw(username).await
    }

    async fn register(&self, username: &str) -> Result<String> {
        let sub = uuid::Uuid::new_v4().to_string();
        let profile = UserProfile {
            sub: Some(sub.clone()),
            active: Some(true),
            ..Default::default()
        };
        let json = serde_json::to_string(&profile)?;
        self.pool
            .set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false)
            .await?;
        self.pool.sadd::<i64, _, _>("hs:users", username).await?;
        self.pool
            .set::<(), _, _>(format!("hs:idx:sub:{sub}"), username, None, None, false)
            .await?;
        Ok(sub)
    }

    async fn provision_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        pubkey: VerifyingKey,
        custody: AccountKeyCustody,
    ) -> std::result::Result<HostedAccountProvisioning, HostedAccountProvisionError> {
        use base64::Engine;

        let sub = uuid::Uuid::new_v4().to_string();
        let fingerprint = pubkey_fingerprint(&pubkey);
        let profile = UserProfile {
            sub: Some(sub.clone()),
            active: Some(false),
            atproto_did: Some(atproto_did.to_owned()),
            key_custody: Some(custody),
            external_identities: Vec::new(),
            ..Default::default()
        };
        let profile_json = serde_json::to_string(&profile)
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let stored_key = StoredKey {
            pubkey_base64: base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes()),
            label: Some("aegis-vault".to_owned()),
            created_at: chrono::Utc::now().timestamp(),
            last_used_at: None,
            algorithm: crate::auth::KeyAlgorithm::Ed25519,
            pq_pubkey_base64: None,
        };
        let key_json = serde_json::to_string(&stored_key)
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        let marker_json = serde_json::to_string(&HostedBindingMarker {
            fingerprint: fingerprint.clone(),
            did: atproto_did.to_owned(),
            custody: custody.as_str().to_owned(),
            pubkey_base64: base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes()),
        })
        .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;

        // All conflict checks and writes run inside one Valkey script. A
        // conflict status is emitted only after the existing hosted binding
        // has passed profile/key/reverse-index integrity checks.
        // Status: 0 = inserted, 1 = usable username conflict,
        // 2 = usable key conflict, 3 = exact resumable binding,
        // 4 = partial/corrupt state.
        const PROVISION_HOSTED_ACCOUNT: &str = r#"
local function usable(owner, profile, fingerprint)
  if not profile or profile == false then return false end
  local ok, decoded = pcall(cjson.decode, profile)
  if not ok or decoded.active ~= true
      or type(decoded.sub) ~= 'string' or decoded.sub == ''
      or type(decoded.atproto_did) ~= 'string'
      or string.sub(decoded.atproto_did, 1, 8) ~= 'did:web:'
      or type(decoded.key_custody) ~= 'string' then
    return false
  end
  local key_json = redis.call('GET', 'hs:key:' .. fingerprint)
  if redis.call('SISMEMBER', 'hs:user:' .. owner .. ':keys', fingerprint) ~= 1
      or redis.call('GET', 'hs:keyowner:' .. fingerprint) ~= owner
      or not key_json then
    return false
  end
  local key_ok, key = pcall(cjson.decode, key_json)
  if not key_ok or type(key.pubkey_base64) ~= 'string'
      or key.pubkey_base64 == '' then return false end
  local marker_json = redis.call(
      'GET', 'hs:hosted-binding:' .. owner .. ':' .. fingerprint)
  if not marker_json then return false end
  local marker_ok, marker = pcall(cjson.decode, marker_json)
  if not marker_ok
      or marker.fingerprint ~= fingerprint
      or marker.did ~= decoded.atproto_did
      or marker.custody ~= decoded.key_custody
      or marker.pubkey_base64 ~= key.pubkey_base64 then
    return false
  end
  return true
end

local existing_profile = redis.call('GET', KEYS[1])
if existing_profile then
  local ok, decoded = pcall(cjson.decode, existing_profile)
  local exact = ok
      and (decoded.active == true or decoded.active == false)
      and type(decoded.sub) == 'string' and decoded.sub ~= ''
      and decoded.atproto_did == ARGV[5]
      and decoded.key_custody == ARGV[6]
      and (decoded.external_identities == nil
          or (type(decoded.external_identities) == 'table'
              and next(decoded.external_identities) == nil))
      and redis.call('GET', KEYS[5]) == ARGV[2]
      and redis.call('SISMEMBER', KEYS[6], ARGV[4]) == 1
  local existing_key = redis.call('GET', KEYS[4])
  local existing_marker = redis.call('GET', KEYS[7])
  if exact and existing_key and existing_marker then
    local key_ok, key_decoded = pcall(cjson.decode, existing_key)
    local marker_ok, marker = pcall(cjson.decode, existing_marker)
    exact = key_ok and marker_ok
        and key_decoded.pubkey_base64 == ARGV[7]
        and marker.fingerprint == ARGV[4]
        and marker.did == ARGV[5]
        and marker.custody == ARGV[6]
        and marker.pubkey_base64 == ARGV[7]
  else
    exact = false
  end
  if exact and (decoded.active == false
      or usable(ARGV[2], existing_profile, ARGV[4])) then return 3 end
  if exact then return 4 end

  local fingerprints = redis.call('SMEMBERS', KEYS[6])
  for _, existing_fingerprint in ipairs(fingerprints) do
    if usable(ARGV[2], existing_profile, existing_fingerprint) then return 1 end
  end
  return 4
end

local existing_owner = redis.call('GET', KEYS[5])
if existing_owner then
  if usable(existing_owner, redis.call('GET', 'hs:user:' .. existing_owner), ARGV[4]) then
    return 2
  end
  return 4
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], ARGV[2])
redis.call('SET', KEYS[3], ARGV[2])
redis.call('SET', KEYS[4], ARGV[3])
redis.call('SADD', KEYS[6], ARGV[4])
redis.call('SET', KEYS[5], ARGV[2])
redis.call('SET', KEYS[7], ARGV[8])
return 0
"#;
        let status: i64 = self
            .pool
            .eval(
                PROVISION_HOSTED_ACCOUNT,
                vec![
                    format!("hs:user:{username}"),
                    "hs:users".to_owned(),
                    format!("hs:idx:sub:{sub}"),
                    format!("hs:key:{fingerprint}"),
                    format!("hs:keyowner:{fingerprint}"),
                    format!("hs:user:{username}:keys"),
                    format!("hs:hosted-staged-binding:{username}:{fingerprint}"),
                ],
                vec![
                    profile_json,
                    username.to_owned(),
                    key_json,
                    fingerprint.clone(),
                    atproto_did.to_owned(),
                    custody.as_str().to_owned(),
                    base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes()),
                    marker_json,
                ],
            )
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        match status {
            0 => Ok(HostedAccountProvisioning {
                sub,
                fingerprint,
                resumed: false,
            }),
            1 => {
                self.validate_account_conflict(username).await?;
                Err(HostedAccountProvisionError::AccountAlreadyExists)
            }
            2 => {
                self.validate_key_conflict(&fingerprint).await?;
                Err(HostedAccountProvisionError::KeyAlreadyBound)
            }
            3 => {
                let existing = self
                    .get_profile_raw(username)
                    .await
                    .map_err(HostedAccountProvisionError::Backend)?;
                let existing_sub = existing
                    .filter(|profile| {
                        profile.active.is_some()
                            && profile.atproto_did.as_deref() == Some(atproto_did)
                            && profile.key_custody == Some(custody)
                            && profile.external_identities.is_empty()
                    })
                    .and_then(|profile| profile.sub)
                    .filter(|sub| !sub.is_empty())
                    .ok_or_else(|| {
                        HostedAccountProvisionError::Backend(anyhow!(
                            "resumed hosted-account profile has no subject"
                        ))
                    })?;
                Ok(HostedAccountProvisioning {
                    sub: existing_sub,
                    fingerprint,
                    resumed: true,
                })
            }
            4 => Err(HostedAccountProvisionError::Backend(anyhow!(
                "existing hosted-account state is incomplete or inconsistent"
            ))),
            other => Err(HostedAccountProvisionError::Backend(anyhow!(
                "unexpected hosted-account provisioning status {other}"
            ))),
        }
    }

    async fn activate_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        fingerprint: &str,
        custody: AccountKeyCustody,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        let expected_public_key = self
            .validated_staged_public_key(username, atproto_did, fingerprint, custody)
            .await?;
        const ACTIVATE_HOSTED_ACCOUNT: &str = r#"
local profile_json = redis.call('GET', KEYS[1])
if not profile_json then return 1 end
local ok, profile = pcall(cjson.decode, profile_json)
local key_json = redis.call('GET', KEYS[4])
local key_ok, key = pcall(cjson.decode, key_json or '')
local marker_json = redis.call('GET', KEYS[5])
local marker_ok, marker = pcall(cjson.decode, marker_json or '')
if not ok
    or not key_ok
    or not marker_ok
    or (profile.active ~= false and profile.active ~= true)
    or profile.atproto_did ~= ARGV[1]
    or profile.key_custody ~= ARGV[4]
    or type(key.pubkey_base64) ~= 'string' or key.pubkey_base64 == ''
    or marker.fingerprint ~= ARGV[3]
    or marker.did ~= ARGV[1]
    or marker.custody ~= ARGV[4]
    or marker.pubkey_base64 ~= key.pubkey_base64
    or marker.pubkey_base64 ~= ARGV[5]
    or redis.call('GET', KEYS[2]) ~= ARGV[2]
    or redis.call('SISMEMBER', KEYS[3], ARGV[3]) ~= 1
    or not key_json
    or not marker_json then
  return 1
end
profile.active = true
redis.call('SET', KEYS[1], cjson.encode(profile))
redis.call('SET', KEYS[6], marker_json)
return 0
"#;
        let status: i64 = self
            .pool
            .eval(
                ACTIVATE_HOSTED_ACCOUNT,
                vec![
                    format!("hs:user:{username}"),
                    format!("hs:keyowner:{fingerprint}"),
                    format!("hs:user:{username}:keys"),
                    format!("hs:key:{fingerprint}"),
                    format!("hs:hosted-staged-binding:{username}:{fingerprint}"),
                    format!("hs:hosted-binding:{username}:{fingerprint}"),
                ],
                vec![
                    atproto_did.to_owned(),
                    username.to_owned(),
                    fingerprint.to_owned(),
                    custody.as_str().to_owned(),
                    expected_public_key,
                ],
            )
            .await
            .map_err(|error| HostedAccountProvisionError::Backend(error.into()))?;
        match status {
            0 => Ok(()),
            _ => Err(HostedAccountProvisionError::Backend(anyhow!(
                "staged hosted-account binding changed before activation"
            ))),
        }
    }

    async fn set_profile(&self, username: &str, new: UserProfilePatch) -> Result<()> {
        let existing = self.get_profile_raw(username).await?.unwrap_or_default();

        // Remove stale externalId index if it changed.
        if let Some(ref old_extid) = existing.external_id {
            if new
                .external_id
                .as_ref()
                .is_some_and(|value| value.as_deref() != Some(old_extid))
            {
                let _: i64 = self
                    .pool
                    .del(format!("hs:idx:extid:{old_extid}"))
                    .await
                    .unwrap_or(0);
            }
        }

        // Apply tri-state patch fields: omitted keeps existing, explicit None clears.
        let merged = UserProfile {
            sub: new.sub.unwrap_or(existing.sub),
            name: new.name.unwrap_or(existing.name),
            email: new.email.unwrap_or(existing.email),
            email_verified: new.email_verified.unwrap_or(existing.email_verified),
            active: new.active.unwrap_or(existing.active),
            external_id: new.external_id.unwrap_or(existing.external_id),
            atproto_did: new.atproto_did.unwrap_or(existing.atproto_did),
            key_custody: new.key_custody.unwrap_or(existing.key_custody),
            external_identities: new
                .external_identities
                .unwrap_or(existing.external_identities),
        };

        // Write new externalId index.
        if let Some(ref extid) = merged.external_id {
            self.pool
                .set::<(), _, _>(format!("hs:idx:extid:{extid}"), username, None, None, false)
                .await?;
        }

        let json = serde_json::to_string(&merged)?;
        self.pool
            .set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false)
            .await?;
        Ok(())
    }

    async fn remove(&self, username: &str) -> Result<bool> {
        let existing = self.get_profile_raw(username).await?;
        let Some(profile) = existing else {
            return Ok(false);
        };

        // Delete reverse indexes.
        if let Some(ref sub) = profile.sub {
            let _: i64 = self
                .pool
                .del(format!("hs:idx:sub:{sub}"))
                .await
                .unwrap_or(0);
        }
        if let Some(ref extid) = profile.external_id {
            let _: i64 = self
                .pool
                .del(format!("hs:idx:extid:{extid}"))
                .await
                .unwrap_or(0);
        }

        // Delete pubkeys.
        let fps: Vec<String> = self
            .pool
            .smembers(format!("hs:user:{username}:keys"))
            .await
            .unwrap_or_default();
        for fp in &fps {
            let _: i64 = self.pool.del(format!("hs:key:{fp}")).await.unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:keyowner:{fp}"))
                .await
                .unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:hosted-binding:{username}:{fp}"))
                .await
                .unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:hosted-staged-binding:{username}:{fp}"))
                .await
                .unwrap_or(0);
        }
        let _: i64 = self
            .pool
            .del(format!("hs:user:{username}:keys"))
            .await
            .unwrap_or(0);
        let _: i64 = self
            .pool
            .del(format!("hs:user:{username}"))
            .await
            .unwrap_or(0);
        let _: i64 = self.pool.srem("hs:users", username).await.unwrap_or(0);
        Ok(true)
    }

    async fn list_users(&self) -> Vec<String> {
        self.pool
            .smembers::<Vec<String>, _>("hs:users")
            .await
            .unwrap_or_default()
    }

    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        let scim_filter = filter.filter.as_deref().map(ScimFilter::parse);

        // Fast-path: point lookups for known eq filters.
        if let Some(ref sf) = scim_filter {
            match sf {
                ScimFilter::UserNameEq(name) => {
                    return match self.get_profile_raw(name).await? {
                        Some(p) => Ok(vec![(name.clone(), p)]),
                        None => Ok(vec![]),
                    };
                }
                ScimFilter::IdEq(sub) => {
                    let username: Option<String> =
                        self.pool.get(format!("hs:idx:sub:{sub}")).await?;
                    if let Some(u) = username {
                        if let Some(p) = self.get_profile_raw(&u).await? {
                            return Ok(vec![(u, p)]);
                        }
                    }
                    return Ok(vec![]);
                }
                ScimFilter::ExternalIdEq(extid) => {
                    let username: Option<String> =
                        self.pool.get(format!("hs:idx:extid:{extid}")).await?;
                    if let Some(u) = username {
                        if let Some(p) = self.get_profile_raw(&u).await? {
                            return Ok(vec![(u, p)]);
                        }
                    }
                    return Ok(vec![]);
                }
                _ => {} // fall through to full scan
            }
        }

        // Full scan: SMEMBERS + in-memory filter + sort + paginate.
        let all_usernames: Vec<String> = self.pool.smembers("hs:users").await?;
        let mut results: Vec<(String, UserProfile)> = Vec::new();
        for username in all_usernames {
            let Some(profile) = self.get_profile_raw(&username).await? else {
                continue;
            };

            // Apply active_only shortcut.
            if filter.active_only == Some(true) && profile.active == Some(false) {
                continue;
            }

            // Apply SCIM filter expression.
            if let Some(ref sf) = scim_filter {
                let pass = match sf {
                    ScimFilter::ActiveEq(b) => profile.active.unwrap_or(true) == *b,
                    ScimFilter::Presence(attr) => matches_filter(
                        &format!("{attr} pr"),
                        &username,
                        &profile.sub,
                        &profile.external_id,
                        profile.active,
                    ),
                    ScimFilter::Unrecognised(expr) => matches_filter(
                        expr,
                        &username,
                        &profile.sub,
                        &profile.external_id,
                        profile.active,
                    ),
                    _ => true, // point-lookup cases already handled above
                };
                if !pass {
                    continue;
                }
            }
            results.push((username, profile));
        }

        // Sort.
        if let Some(ref sort_by) = filter.sort_by {
            let descending = filter.sort_order.as_deref() == Some("descending");
            results.sort_by(|(a_name, a_prof), (b_name, b_prof)| {
                let ord = match sort_by.as_str() {
                    "id" | "sub" => a_prof.sub.cmp(&b_prof.sub),
                    _ => a_name.cmp(b_name),
                };
                if descending {
                    ord.reverse()
                } else {
                    ord
                }
            });
        }

        // Paginate.
        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);
        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let mut profile = self
            .get_profile_raw(username)
            .await?
            .ok_or_else(|| anyhow!("User '{username}' not found"))?;
        profile.active = Some(active);
        let json = serde_json::to_string(&profile)?;
        self.pool
            .set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false)
            .await?;
        Ok(())
    }

    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>> {
        let fps: Vec<String> = self
            .pool
            .smembers(format!("hs:user:{username}:keys"))
            .await?;
        let mut entries = Vec::new();
        for fp in fps {
            let val: Option<String> = self.pool.get(format!("hs:key:{fp}")).await?;
            if let Some(s) = val {
                let (stored, pubkey, pq_pubkey) = validate_stored_key_record(&fp, &s)?;
                entries.push(PubkeyEntry {
                    fingerprint: fp,
                    pubkey,
                    label: stored.label,
                    created_at: stored.created_at,
                    last_used_at: stored.last_used_at,
                    algorithm: stored.algorithm,
                    pq_pubkey,
                });
            }
        }
        Ok(entries)
    }

    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<String> {
        use base64::Engine;
        let fp = pubkey_fingerprint(&pubkey);
        let pubkey_base64 = base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes());
        let now = chrono::Utc::now().timestamp();
        let stored = StoredKey {
            pubkey_base64,
            label,
            created_at: now,
            last_used_at: None,
            algorithm: crate::auth::KeyAlgorithm::Ed25519,
            pq_pubkey_base64: None,
        };
        let json = serde_json::to_string(&stored)?;
        const ADD_CLASSICAL_KEY: &str = r#"
local owner = redis.call('GET', KEYS[1])
if owner and owner ~= ARGV[1] then return 1 end
if redis.call('SISMEMBER', KEYS[3], ARGV[3]) == 1 then return 2 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('SADD', KEYS[3], ARGV[3])
redis.call('SET', KEYS[1], ARGV[1])
return 0
"#;
        let status: i64 = self
            .pool
            .eval(
                ADD_CLASSICAL_KEY,
                vec![
                    format!("hs:keyowner:{fp}"),
                    format!("hs:key:{fp}"),
                    format!("hs:user:{username}:keys"),
                ],
                vec![username.to_owned(), json, fp.clone()],
            )
            .await?;
        match status {
            0 => Ok(fp),
            1 => anyhow::bail!("Pubkey already associated with another user"),
            2 => anyhow::bail!("Pubkey with fingerprint {fp} already exists for user '{username}'"),
            other => anyhow::bail!("unexpected add-pubkey status {other}"),
        }
    }

    async fn add_pubkey_hybrid(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        ml_dsa_vk: Vec<u8>,
        label: Option<String>,
    ) -> Result<String> {
        use base64::Engine;
        if ml_dsa_vk.is_empty() {
            anyhow::bail!("add_pubkey_hybrid: empty ML-DSA-65 verifying key");
        }
        // Fingerprint is the Ed25519 anchor's (kid) — the PQ vk does not change it.
        let fp = pubkey_fingerprint(&pubkey);
        // Reject a fingerprint already owned by a *different* user (matches the
        // RocksDB backend): without this, a cross-user re-bind would overwrite
        // hs:keyowner and leave the key in both users' sets.
        if let Some(existing_user) = self.get_pubkey_user(&fp).await? {
            if existing_user != username {
                anyhow::bail!("Pubkey already associated with user '{existing_user}'");
            }
        }
        let pubkey_base64 = base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes());
        let pq_pubkey_base64 = Some(base64::engine::general_purpose::STANDARD.encode(&ml_dsa_vk));

        // In-place upgrade (Ed25519 → Hybrid) or idempotent re-bind: preserve
        // the original created_at/last_used_at if a record already exists.
        let (created_at, last_used_at, existing_label) = match self
            .pool
            .get::<Option<String>, _>(format!("hs:key:{fp}"))
            .await?
        {
            Some(s) => {
                let prev: StoredKey = serde_json::from_str(&s)?;
                (prev.created_at, prev.last_used_at, prev.label)
            }
            None => (chrono::Utc::now().timestamp(), None, None),
        };

        let stored = StoredKey {
            pubkey_base64,
            label: label.or(existing_label),
            created_at,
            last_used_at,
            algorithm: crate::auth::KeyAlgorithm::HybridEd25519MlDsa65,
            pq_pubkey_base64,
        };
        let json = serde_json::to_string(&stored)?;
        const UPSERT_HYBRID_KEY: &str = r#"
local owner = redis.call('GET', KEYS[1])
if owner and owner ~= ARGV[1] then return 1 end
redis.call('SET', KEYS[2], ARGV[2])
redis.call('SADD', KEYS[3], ARGV[3])
redis.call('SET', KEYS[1], ARGV[1])
return 0
"#;
        let status: i64 = self
            .pool
            .eval(
                UPSERT_HYBRID_KEY,
                vec![
                    format!("hs:keyowner:{fp}"),
                    format!("hs:key:{fp}"),
                    format!("hs:user:{username}:keys"),
                ],
                vec![username.to_owned(), json, fp.clone()],
            )
            .await?;
        match status {
            0 => Ok(fp),
            1 => anyhow::bail!("Pubkey already associated with another user"),
            other => anyhow::bail!("unexpected hybrid add-pubkey status {other}"),
        }
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let removed: i64 = self
            .pool
            .srem(format!("hs:user:{username}:keys"), fingerprint)
            .await?;
        if removed > 0 {
            let _: i64 = self
                .pool
                .del(format!("hs:key:{fingerprint}"))
                .await
                .unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:keyowner:{fingerprint}"))
                .await
                .unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:hosted-binding:{username}:{fingerprint}"))
                .await
                .unwrap_or(0);
            let _: i64 = self
                .pool
                .del(format!("hs:hosted-staged-binding:{username}:{fingerprint}"))
                .await
                .unwrap_or(0);
        }
        Ok(removed > 0)
    }

    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>> {
        Ok(self.pool.get(format!("hs:keyowner:{fingerprint}")).await?)
    }

    async fn touch_pubkey(&self, _username: &str, fingerprint: &str) -> Result<()> {
        let val: Option<String> = self.pool.get(format!("hs:key:{fingerprint}")).await?;
        if let Some(s) = val {
            let mut stored: StoredKey = serde_json::from_str(&s)?;
            stored.last_used_at = Some(chrono::Utc::now().timestamp());
            let json = serde_json::to_string(&stored)?;
            self.pool
                .set::<(), _, _>(format!("hs:key:{fingerprint}"), json, None, None, false)
                .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupt_key_conflict_snapshot_never_becomes_trusted_conflict() -> Result<()> {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x76; 32]).verifying_key();
        let fingerprint = pubkey_fingerprint(&key);
        let profile_json = serde_json::to_string(&UserProfile {
            sub: Some("alice-sub".to_owned()),
            active: Some(true),
            atproto_did: Some("did:web:alice.accounts.example.test".to_owned()),
            key_custody: Some(AccountKeyCustody::SelfCustody),
            ..Default::default()
        })?;
        let key_base64 = STANDARD.encode(key.as_bytes());
        let key_json = serde_json::to_string(&StoredKey {
            pubkey_base64: key_base64.clone(),
            label: Some("aegis-vault".to_owned()),
            created_at: 1,
            last_used_at: None,
            algorithm: crate::auth::KeyAlgorithm::Ed25519,
            pq_pubkey_base64: None,
        })?;
        let marker_json = serde_json::to_string(&HostedBindingMarker {
            fingerprint: fingerprint.clone(),
            did: "did:web:alice.accounts.example.test".to_owned(),
            custody: "self_custody".to_owned(),
            pubkey_base64: key_base64,
        })?;
        assert!(validate_key_conflict_snapshot(
            &fingerprint,
            Some("alice"),
            true,
            Some(&profile_json),
            Some(&key_json),
            Some(&marker_json),
        )
        .is_ok());

        for corrupt in [
            validate_key_conflict_snapshot(
                &fingerprint,
                Some("alice"),
                false,
                Some(&profile_json),
                Some(&key_json),
                Some(&marker_json),
            ),
            validate_key_conflict_snapshot(
                &fingerprint,
                Some("alice"),
                true,
                Some(&profile_json),
                None,
                Some(&marker_json),
            ),
        ] {
            assert!(matches!(
                corrupt,
                Err(HostedAccountProvisionError::Backend(_))
            ));
        }

        let malformed_key_json = serde_json::to_string(&StoredKey {
            pubkey_base64: "not-base64".to_owned(),
            label: None,
            created_at: 1,
            last_used_at: None,
            algorithm: crate::auth::KeyAlgorithm::Ed25519,
            pq_pubkey_base64: None,
        })?;
        assert!(matches!(
            validate_key_conflict_snapshot(
                &fingerprint,
                Some("alice"),
                true,
                Some(&profile_json),
                Some(&malformed_key_json),
                Some(&marker_json),
            ),
            Err(HostedAccountProvisionError::Backend(_))
        ));

        let different_key = ed25519_dalek::SigningKey::from_bytes(&[0x77; 32]).verifying_key();
        let mismatch_base64 = STANDARD.encode(different_key.as_bytes());
        let mismatch_key_json = serde_json::to_string(&StoredKey {
            pubkey_base64: mismatch_base64.clone(),
            label: None,
            created_at: 1,
            last_used_at: None,
            algorithm: crate::auth::KeyAlgorithm::Ed25519,
            pq_pubkey_base64: None,
        })?;
        let mismatch_marker_json = serde_json::to_string(&HostedBindingMarker {
            fingerprint: fingerprint.clone(),
            did: "did:web:alice.accounts.example.test".to_owned(),
            custody: "self_custody".to_owned(),
            pubkey_base64: mismatch_base64,
        })?;
        assert!(matches!(
            validate_key_conflict_snapshot(
                &fingerprint,
                Some("alice"),
                true,
                Some(&profile_json),
                Some(&mismatch_key_json),
                Some(&mismatch_marker_json),
            ),
            Err(HostedAccountProvisionError::Backend(_))
        ));
        Ok(())
    }

    #[test]
    fn corrupt_algorithm_pq_metadata_never_becomes_trusted_key_conflict() -> Result<()> {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x78; 32]).verifying_key();
        let fingerprint = pubkey_fingerprint(&key);
        let key_base64 = STANDARD.encode(key.as_bytes());
        let profile_json = serde_json::to_string(&UserProfile {
            sub: Some("alice-sub".to_owned()),
            active: Some(true),
            atproto_did: Some("did:web:alice.accounts.example.test".to_owned()),
            key_custody: Some(AccountKeyCustody::SelfCustody),
            ..Default::default()
        })?;
        let marker_json = serde_json::to_string(&HostedBindingMarker {
            fingerprint: fingerprint.clone(),
            did: "did:web:alice.accounts.example.test".to_owned(),
            custody: "self_custody".to_owned(),
            pubkey_base64: key_base64.clone(),
        })?;
        let conflict_result = |algorithm: crate::auth::KeyAlgorithm,
                               pq_pubkey_base64: Option<String>| {
            let key_json = serde_json::to_string(&StoredKey {
                pubkey_base64: key_base64.clone(),
                label: Some("aegis-vault".to_owned()),
                created_at: 1,
                last_used_at: None,
                algorithm,
                pq_pubkey_base64,
            })
            .unwrap();
            validate_key_conflict_snapshot(
                &fingerprint,
                Some("alice"),
                true,
                Some(&profile_json),
                Some(&key_json),
                Some(&marker_json),
            )
        };

        // Hybrid tag without its mandatory ML-DSA-65 component.
        assert!(matches!(
            conflict_result(crate::auth::KeyAlgorithm::HybridEd25519MlDsa65, None),
            Err(HostedAccountProvisionError::Backend(_))
        ));

        // Classical tag carrying PQ material is inconsistent and must not be
        // treated as an authentication-usable hosted binding.
        assert!(matches!(
            conflict_result(
                crate::auth::KeyAlgorithm::Ed25519,
                Some(STANDARD.encode(vec![0x5a; 1952])),
            ),
            Err(HostedAccountProvisionError::Backend(_))
        ));

        // A hybrid record with malformed PQ encoding cannot be read by the
        // ordinary authentication path, so it cannot justify a trusted 409.
        assert!(matches!(
            conflict_result(
                crate::auth::KeyAlgorithm::HybridEd25519MlDsa65,
                Some("not-base64".to_owned()),
            ),
            Err(HostedAccountProvisionError::Backend(_))
        ));

        // Preserve the positive side of the invariant: a complete hybrid
        // record with the same Ed25519 anchor remains accepted.
        assert!(conflict_result(
            crate::auth::KeyAlgorithm::HybridEd25519MlDsa65,
            Some(STANDARD.encode(vec![0x5a; 1952])),
        )
        .is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn list_pubkeys_propagates_smembers_read_error() -> Result<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // A tiny RESP server accepts Fred's connection setup, then returns a
        // synthetic Valkey error for SMEMBERS. This directly exercises the
        // backend seam and would regress to Ok(empty) if list_pubkeys swallowed
        // the read error again.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut buffer = [0_u8; 1024];
            loop {
                let read = socket.read(&mut buffer).await?;
                if read == 0 {
                    break;
                }
                let command = String::from_utf8_lossy(&buffer[..read]);
                let response = if command.contains("SMEMBERS") {
                    "-ERR synthetic SMEMBERS read failure\r\n"
                } else {
                    "+OK\r\n"
                };
                socket.write_all(response.as_bytes()).await?;
            }
            Ok::<(), std::io::Error>(())
        });

        let config = RedisConfig::from_url(&format!("redis://{address}"))?;
        let pool = Builder::from_config(config).build_pool(1)?;
        let _connection = pool.connect();
        pool.wait_for_connect().await?;
        let store = ValkeyUserStore { pool };

        assert!(store.list_pubkeys("alice").await.is_err());
        server.abort();
        Ok(())
    }
}
