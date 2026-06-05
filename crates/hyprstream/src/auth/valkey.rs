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
//!   hs:idx:sub:{sub}          username (sub → username reverse index)
//!   hs:idx:extid:{extid}      username (externalId → username reverse index)

#![cfg(feature = "valkey")]

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use fred::prelude::*;
use serde::{Deserialize, Serialize};

use super::user_store::{
    matches_filter, pubkey_fingerprint, PubkeyEntry, ScimFilter, UserFilter, UserProfile, UserStore,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredKey {
    pubkey_base64: String,
    label: Option<String>,
    created_at: i64,
    last_used_at: Option<i64>,
}

pub struct ValkeyUserStore {
    pool: RedisPool,
}

impl ValkeyUserStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let config = RedisConfig::from_url(url)
            .context("invalid Valkey URL")?;
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
        self.pool.set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false).await?;
        self.pool.sadd::<i64, _, _>("hs:users", username).await?;
        self.pool.set::<(), _, _>(format!("hs:idx:sub:{sub}"), username, None, None, false).await?;
        Ok(sub)
    }

    async fn set_profile(&self, username: &str, new: UserProfile) -> Result<()> {
        let existing = self.get_profile_raw(username).await?.unwrap_or_default();

        // Remove stale externalId index if it changed.
        if let Some(ref old_extid) = existing.external_id {
            if new.external_id.as_deref() != Some(old_extid.as_str()) {
                let _: i64 = self.pool.del(format!("hs:idx:extid:{old_extid}")).await.unwrap_or(0);
            }
        }

        // Merge: new fields overwrite existing; None in `new` keeps existing value.
        let merged = UserProfile {
            sub: new.sub.or(existing.sub.clone()),
            name: new.name.or(existing.name),
            email: new.email.or(existing.email),
            email_verified: new.email_verified.or(existing.email_verified),
            active: new.active.or(existing.active),
            external_id: new.external_id.or(existing.external_id),
        };

        // Write new externalId index.
        if let Some(ref extid) = merged.external_id {
            self.pool.set::<(), _, _>(format!("hs:idx:extid:{extid}"), username, None, None, false).await?;
        }

        let json = serde_json::to_string(&merged)?;
        self.pool.set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false).await?;
        Ok(())
    }

    async fn remove(&self, username: &str) -> Result<bool> {
        let existing = self.get_profile_raw(username).await?;
        let Some(profile) = existing else { return Ok(false) };

        // Delete reverse indexes.
        if let Some(ref sub) = profile.sub {
            let _: i64 = self.pool.del(format!("hs:idx:sub:{sub}")).await.unwrap_or(0);
        }
        if let Some(ref extid) = profile.external_id {
            let _: i64 = self.pool.del(format!("hs:idx:extid:{extid}")).await.unwrap_or(0);
        }

        // Delete pubkeys.
        let fps: Vec<String> = self.pool.smembers(format!("hs:user:{username}:keys")).await.unwrap_or_default();
        for fp in &fps {
            let _: i64 = self.pool.del(format!("hs:key:{fp}")).await.unwrap_or(0);
            let _: i64 = self.pool.del(format!("hs:keyowner:{fp}")).await.unwrap_or(0);
        }
        let _: i64 = self.pool.del(format!("hs:user:{username}:keys")).await.unwrap_or(0);
        let _: i64 = self.pool.del(format!("hs:user:{username}")).await.unwrap_or(0);
        let _: i64 = self.pool.srem("hs:users", username).await.unwrap_or(0);
        Ok(true)
    }

    async fn list_users(&self) -> Vec<String> {
        self.pool.smembers::<Vec<String>, _>("hs:users").await.unwrap_or_default()
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
                    let username: Option<String> = self.pool.get(format!("hs:idx:sub:{sub}")).await?;
                    if let Some(u) = username {
                        if let Some(p) = self.get_profile_raw(&u).await? {
                            return Ok(vec![(u, p)]);
                        }
                    }
                    return Ok(vec![]);
                }
                ScimFilter::ExternalIdEq(extid) => {
                    let username: Option<String> = self.pool.get(format!("hs:idx:extid:{extid}")).await?;
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
            let Some(profile) = self.get_profile_raw(&username).await? else { continue };

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
                if !pass { continue; }
            }
            results.push((username, profile));
        }

        // Sort.
        if let Some(ref sort_by) = filter.sort_by {
            let descending = filter.sort_order.as_deref() == Some("descending");
            results.sort_by(|(a_name, a_prof), (b_name, b_prof)| {
                let ord = match sort_by.as_str() {
                    "userName" => a_name.cmp(b_name),
                    "id" | "sub" => a_prof.sub.cmp(&b_prof.sub),
                    _ => a_name.cmp(b_name),
                };
                if descending { ord.reverse() } else { ord }
            });
        }

        // Paginate.
        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);
        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let mut profile = self.get_profile_raw(username).await?
            .ok_or_else(|| anyhow!("User '{username}' not found"))?;
        profile.active = Some(active);
        let json = serde_json::to_string(&profile)?;
        self.pool.set::<(), _, _>(format!("hs:user:{username}"), json, None, None, false).await?;
        Ok(())
    }

    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>> {
        let fps: Vec<String> = self.pool
            .smembers(format!("hs:user:{username}:keys"))
            .await
            .unwrap_or_default();
        let mut entries = Vec::new();
        for fp in fps {
            let val: Option<String> = self.pool.get(format!("hs:key:{fp}")).await?;
            if let Some(s) = val {
                let stored: StoredKey = serde_json::from_str(&s)?;
                use base64::Engine;
                let raw = base64::engine::general_purpose::STANDARD.decode(&stored.pubkey_base64)?;
                let key_bytes: [u8; 32] = raw.try_into().map_err(|_| anyhow!("bad key length"))?;
                let pubkey = VerifyingKey::from_bytes(&key_bytes)?;
                entries.push(PubkeyEntry {
                    fingerprint: fp,
                    pubkey,
                    label: stored.label,
                    created_at: stored.created_at,
                    last_used_at: stored.last_used_at,
                });
            }
        }
        Ok(entries)
    }

    async fn add_pubkey(&self, username: &str, pubkey: VerifyingKey, label: Option<String>) -> Result<String> {
        use base64::Engine;
        let fp = pubkey_fingerprint(&pubkey);
        let pubkey_base64 = base64::engine::general_purpose::STANDARD.encode(pubkey.as_bytes());
        let now = chrono::Utc::now().timestamp();
        let stored = StoredKey { pubkey_base64, label, created_at: now, last_used_at: None };
        let json = serde_json::to_string(&stored)?;
        self.pool.set::<(), _, _>(format!("hs:key:{fp}"), json, None, None, false).await?;
        self.pool.sadd::<i64, _, _>(format!("hs:user:{username}:keys"), &fp).await?;
        self.pool.set::<(), _, _>(format!("hs:keyowner:{fp}"), username, None, None, false).await?;
        Ok(fp)
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let removed: i64 = self.pool.srem(format!("hs:user:{username}:keys"), fingerprint).await?;
        if removed > 0 {
            let _: i64 = self.pool.del(format!("hs:key:{fingerprint}")).await.unwrap_or(0);
            let _: i64 = self.pool.del(format!("hs:keyowner:{fingerprint}")).await.unwrap_or(0);
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
            self.pool.set::<(), _, _>(format!("hs:key:{fingerprint}"), json, None, None, false).await?;
        }
        Ok(())
    }
}
