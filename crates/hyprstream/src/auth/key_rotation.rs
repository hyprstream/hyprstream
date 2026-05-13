//! JWT signing key rotation — multi-slot drain/active/lead lifecycle.
//!
//! Keys live in three ordered slots:
//!   lead   — pre-published (nbf in the future); clients see it in JWKS but no tokens use it yet
//!   active — current issuance key
//!   drain  — old active, still valid for token verification until its exp passes
//!
//! The background task checks every 6 h:
//!   1. lead.nbf <= now  → promote lead → active, old active → drain
//!   2. drain exp + drain_days * 86400 < now → remove drain
//!   3. lead is None and active.exp - now < lead_days * 86400 → generate new lead, persist

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, MissedTickBehavior};
use tracing::{info, warn};

use crate::config::OAuthConfig;

// ── Key slot ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct KeySlot {
    pub key: Arc<SigningKey>,
    /// Unix timestamp at which this key may be used for issuance (not_before).
    pub nbf: i64,
    /// Unix timestamp at which this key expires.
    pub exp: i64,
}

impl KeySlot {
    pub fn new(key: SigningKey, nbf: i64, exp: i64) -> Self {
        Self { key: Arc::new(key), nbf, exp }
    }

    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub fn kid(&self) -> String {
        crate::services::oauth::jwks::compute_kid(&self.verifying_key_bytes())
    }
}

// ── Slot container ───────────────────────────────────────────────────────────

#[derive(Default)]
pub struct KeySlots {
    pub drain: Option<KeySlot>,
    pub active: Option<KeySlot>,
    pub lead: Option<KeySlot>,
}

impl KeySlots {
    /// All non-None slots (drain, active, lead) in that order.
    pub fn all(&self) -> Vec<&KeySlot> {
        [&self.drain, &self.active, &self.lead]
            .into_iter()
            .flatten()
            .collect()
    }
}

// ── SigningKeyStore ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SigningKeyStore(pub Arc<RwLock<KeySlots>>);

impl SigningKeyStore {
    pub fn new(slots: KeySlots) -> Self {
        Self(Arc::new(RwLock::new(slots)))
    }

    pub async fn active_key(&self) -> Option<Arc<SigningKey>> {
        let slots = self.0.read().await;
        slots.active.as_ref().map(|s| Arc::clone(&s.key))
    }

    pub async fn active_verifying_key_bytes(&self) -> Option<[u8; 32]> {
        let slots = self.0.read().await;
        slots.active.as_ref().map(KeySlot::verifying_key_bytes)
    }

    pub async fn all_slots_snapshot(&self) -> Vec<KeySlot> {
        let slots = self.0.read().await;
        slots.all().into_iter().cloned().collect()
    }

    /// Find a verifying key by kid across all slots (for token verification).
    pub async fn verifying_key_for_kid(&self, kid: &str) -> Option<ed25519_dalek::VerifyingKey> {
        let slots = self.0.read().await;
        for slot in slots.all() {
            if slot.kid() == kid {
                return Some(slot.key.verifying_key());
            }
        }
        None
    }
}

// ── Persistence ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct SlotMeta {
    nbf: i64,
    exp: i64,
}

fn slot_paths(secrets_dir: &Path, name: &str) -> (PathBuf, PathBuf) {
    (
        secrets_dir.join(format!("jwt-signing-key.{name}")),
        secrets_dir.join(format!("jwt-signing-key.{name}.meta")),
    )
}

fn load_slot(secrets_dir: &Path, name: &str) -> Option<KeySlot> {
    let (key_path, meta_path) = slot_paths(secrets_dir, name);
    let seed = std::fs::read(&key_path).ok()?;
    if seed.len() != 32 {
        warn!("JWT key slot '{name}': unexpected seed length {}", seed.len());
        return None;
    }
    let meta_bytes = std::fs::read(&meta_path).ok()?;
    let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(&seed);
    Some(KeySlot::new(SigningKey::from_bytes(&seed_arr), meta.nbf, meta.exp))
}

fn persist_slot(secrets_dir: &Path, name: &str, slot: &KeySlot) -> anyhow::Result<()> {
    let (key_path, meta_path) = slot_paths(secrets_dir, name);
    std::fs::write(&key_path, slot.key.to_bytes())?;
    let meta = SlotMeta { nbf: slot.nbf, exp: slot.exp };
    std::fs::write(&meta_path, serde_json::to_vec(&meta)?)?;
    Ok(())
}

fn delete_slot(secrets_dir: &Path, name: &str) {
    let (key_path, meta_path) = slot_paths(secrets_dir, name);
    let _ = std::fs::remove_file(&key_path);
    let _ = std::fs::remove_file(&meta_path);
}

fn generate_slot(nbf: i64, exp: i64) -> KeySlot {
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    KeySlot::new(key, nbf, exp)
}

// ── Load or initialize slots at startup ────────────────────────────────────

/// Load all three JWT key slots from `secrets_dir`.
///
/// If no active slot exists, generate one immediately (first boot).
/// Slot files: `jwt-signing-key.{active,drain,lead}` + `.meta` JSON.
pub fn load_or_init_key_store(secrets_dir: &Path, config: &OAuthConfig) -> SigningKeyStore {
    let now = chrono::Utc::now().timestamp();
    let active_secs = i64::from(config.jwt_key_active_days) * 86400;
    let lead_secs = i64::from(config.jwt_key_lead_days) * 86400;

    let drain = load_slot(secrets_dir, "drain");
    let mut active = load_slot(secrets_dir, "active");
    let lead = load_slot(secrets_dir, "lead");

    if active.is_none() {
        info!("No active JWT signing key found — generating on first boot");
        let slot = generate_slot(now, now + active_secs);
        if let Err(e) = persist_slot(secrets_dir, "active", &slot) {
            warn!("Could not persist active JWT key: {e}");
        } else {
            info!("Active JWT key generated (kid={})", slot.kid());
        }
        active = Some(slot);
    }

    // If we have active but no lead and active is close to expiry, generate lead now.
    let should_gen_lead = lead.is_none() && active.as_ref()
        .is_some_and(|a| a.exp - now < lead_secs);
    if should_gen_lead {
        let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
        let lead_exp = lead_nbf + active_secs;
        let slot = generate_slot(lead_nbf, lead_exp);
        if let Err(e) = persist_slot(secrets_dir, "lead", &slot) {
            warn!("Could not persist lead JWT key: {e}");
        } else {
            info!("Lead JWT key pre-generated at startup (kid={})", slot.kid());
        }
    }

    // Re-load lead if we just generated it
    let lead = if should_gen_lead { load_slot(secrets_dir, "lead") } else { lead };

    SigningKeyStore::new(KeySlots { drain, active, lead })
}

// ── Rotation logic ──────────────────────────────────────────────────────────

pub async fn rotate_jwt_keys(
    config: &OAuthConfig,
    secrets_dir: &Path,
    store: &SigningKeyStore,
    now: i64,
) {
    let mut slots = store.0.write().await;

    let active_secs = i64::from(config.jwt_key_active_days) * 86400;
    let lead_secs = i64::from(config.jwt_key_lead_days) * 86400;
    let drain_secs = i64::from(config.jwt_key_drain_days) * 86400;

    // 1. Promote lead → active if lead.nbf has passed.
    if let Some(new_lead) = slots.lead.take().filter(|l| l.nbf <= now) {
        let old_active = slots.active.take();

        info!("Promoting lead JWT key (kid={}) to active", new_lead.kid());

        // Old active → drain (evict old drain first)
        if let Some(prev_drain) = slots.drain.take() {
            info!("Evicting drain JWT key (kid={})", prev_drain.kid());
            delete_slot(secrets_dir, "drain");
        }
        if let Some(old_active_slot) = old_active {
            if let Err(e) = persist_slot(secrets_dir, "drain", &old_active_slot) {
                warn!("Could not persist drain JWT key: {e}");
            }
            slots.drain = Some(old_active_slot);
        }

        if let Err(e) = persist_slot(secrets_dir, "active", &new_lead) {
            warn!("Could not persist active JWT key: {e}");
        }
        delete_slot(secrets_dir, "lead");
        slots.active = Some(new_lead);
    }

    // 2. Remove drain if drain window has closed.
    if slots.drain.as_ref().is_some_and(|d| now >= d.exp + drain_secs) {
        let kid = slots.drain.as_ref().map(KeySlot::kid).unwrap_or_default();
        info!("Removing expired drain JWT key (kid={kid})");
        delete_slot(secrets_dir, "drain");
        slots.drain = None;
    }

    // 3. Generate lead if active is approaching expiry and lead is absent.
    if slots.lead.is_none() {
        if let Some(active) = &slots.active {
            if active.exp - now < lead_secs {
                let lead_nbf = active.exp - lead_secs;
                let lead_exp = lead_nbf + active_secs;
                let new_lead = generate_slot(lead_nbf, lead_exp);
                info!("Generated new lead JWT key (kid={}, nbf={})", new_lead.kid(), new_lead.nbf);
                if let Err(e) = persist_slot(secrets_dir, "lead", &new_lead) {
                    warn!("Could not persist lead JWT key: {e}");
                }
                slots.lead = Some(new_lead);
            }
        }
    }
}

// ── Background task ─────────────────────────────────────────────────────────

pub fn spawn_rotation_task(
    config: Arc<OAuthConfig>,
    secrets_dir: PathBuf,
    store: Arc<SigningKeyStore>,
) {
    tokio::task::spawn_local(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(6 * 3600));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // Skip the first tick (fires immediately on creation)
        interval.tick().await;
        loop {
            interval.tick().await;
            let now = chrono::Utc::now().timestamp();
            rotate_jwt_keys(&config, &secrets_dir, &store, now).await;
        }
    });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config() -> OAuthConfig {
        OAuthConfig {
            jwt_key_active_days: 14,
            jwt_key_lead_days: 7,
            jwt_key_drain_days: 30,
            ..OAuthConfig::default()
        }
    }

    #[test]
    fn load_or_init_creates_active_key_on_first_boot() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store = load_or_init_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let vk = rt.block_on(store.active_verifying_key_bytes());
        assert!(vk.is_some(), "active key should be present after first boot");
    }

    #[test]
    fn persist_and_reload_slot() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        {
            let store = load_or_init_key_store(dir.path(), &config);
            let rt = tokio::runtime::Runtime::new().unwrap();
            let vk1 = rt.block_on(store.active_verifying_key_bytes()).unwrap();
            // Second load should return same key
            let store2 = load_or_init_key_store(dir.path(), &config);
            let vk2 = rt.block_on(store2.active_verifying_key_bytes()).unwrap();
            assert_eq!(vk1, vk2, "persisted key must reload identically");
        }
    }

    #[tokio::test]
    async fn rotate_promotes_lead_to_active() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Build a store with a lead whose nbf is in the past.
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 1,
        );
        let lead_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1, // nbf already passed
            now + 14 * 86400,
        );
        persist_slot(dir.path(), "active", &active_slot).unwrap();
        persist_slot(dir.path(), "lead", &lead_slot).unwrap();
        let lead_vk = lead_slot.verifying_key_bytes();

        let store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: Some(lead_slot),
        });

        rotate_jwt_keys(&config, dir.path(), &store, now).await;

        let new_active_vk = store.active_verifying_key_bytes().await.unwrap();
        assert_eq!(new_active_vk, lead_vk, "lead must become active after promotion");

        // Old active must now be drain
        let slots = store.0.read().await;
        assert!(slots.drain.is_some(), "old active must become drain");
        assert!(slots.lead.is_none(), "lead slot must be cleared after promotion");
    }

    #[tokio::test]
    async fn drain_is_removed_after_drain_window() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let drain_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 60 * 86400,
            now - 31 * 86400, // exp + 30 drain days < now
        );
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 86400,
            now + 13 * 86400,
        );
        persist_slot(dir.path(), "drain", &drain_slot).unwrap();
        persist_slot(dir.path(), "active", &active_slot).unwrap();

        let store = SigningKeyStore::new(KeySlots {
            drain: Some(drain_slot),
            active: Some(active_slot),
            lead: None,
        });

        rotate_jwt_keys(&config, dir.path(), &store, now).await;

        let slots = store.0.read().await;
        assert!(slots.drain.is_none(), "drain must be removed after drain window closes");
    }

    #[tokio::test]
    async fn lead_is_generated_when_active_near_expiry() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Active expires in 6 days — less than lead_days (7)
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 8 * 86400,
            now + 6 * 86400,
        );
        persist_slot(dir.path(), "active", &active_slot).unwrap();

        let store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: None,
        });

        rotate_jwt_keys(&config, dir.path(), &store, now).await;

        let slots = store.0.read().await;
        assert!(slots.lead.is_some(), "lead must be generated when active is within lead window");
    }
}
