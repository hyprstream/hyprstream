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
use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use crate::config::OAuthConfig;

/// Global shared ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
///
/// Populated at startup from the ML-DSA rotation store's current slots.
/// Updated by the rotation task after each promotion. Services read from this
/// via their `JwtKeySource` implementation.
// Cross-crate `Arc<std::sync::RwLock<..>>` contract with `JwtKeySource` and the
// service factory; intentionally `std::sync::RwLock`, not `parking_lot`.
#[allow(clippy::disallowed_types)]
static ML_DSA_VERIFYING_KEYS: std::sync::OnceLock<std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>> = std::sync::OnceLock::new();

/// Get or initialize the global ML-DSA verifying keys Arc.
#[allow(clippy::disallowed_types)]
pub fn global_ml_dsa_verifying_keys() -> std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>> {
    ML_DSA_VERIFYING_KEYS.get_or_init(|| {
        std::sync::Arc::new(std::sync::RwLock::new(Vec::new()))
    }).clone()
}

/// Live, shared handle to the node's published Ed25519 rotation-slot verifying
/// keys (drain/active/lead). This is the SAME Ed25519 key set the `/oauth/jwks`
/// endpoint publishes — the keys that sign rotation-issued at+JWTs (WIT, S6
/// grant / token-exchange, and any issuance path using the active slot). It is
/// public-key material only; no signing keys are ever exposed through it.
///
/// Consumers (e.g. the shared HTTP validator `verify_token_claims`) hold a clone
/// and verify locally-issued tokens against these keys in addition to the CA key.
// Cross-service `Arc<std::sync::RwLock<..>>` contract mirroring the ML-DSA
// verifying-key list above; intentionally `std::sync::RwLock`, not `parking_lot`.
#[allow(clippy::disallowed_types)]
pub type PublishedEd25519Keys = std::sync::Arc<std::sync::RwLock<Vec<ed25519_dalek::VerifyingKey>>>;

/// Global shared Ed25519 published verifying keys (rotation slots).
///
/// Populated at OAuth service init from the signing-key store's current slots
/// and refreshed by the rotation task after each promotion/eviction. The CA key
/// is NOT included here — callers already hold it separately (`verifying_key`).
static ED25519_VERIFYING_KEYS: std::sync::OnceLock<PublishedEd25519Keys> = std::sync::OnceLock::new();

/// Get or initialize the global published-Ed25519 verifying-key handle.
///
/// Returns a live, shared `Arc` — mutations by the rotation task are visible to
/// every holder. Starts empty until the OAuth service populates it.
#[allow(clippy::disallowed_types)]
pub fn global_ed25519_verifying_keys() -> PublishedEd25519Keys {
    ED25519_VERIFYING_KEYS
        .get_or_init(|| std::sync::Arc::new(std::sync::RwLock::new(Vec::new())))
        .clone()
}

/// Refresh the global published-Ed25519 verifying keys from a signing-key store
/// snapshot (all current drain/active/lead slots).
///
/// Called at OAuth init and after every rotation so the shared HTTP validator
/// keeps accepting tokens signed by any currently-published slot.
pub async fn refresh_ed25519_verifying_keys(store: &SigningKeyStore) {
    let vks: Vec<ed25519_dalek::VerifyingKey> = store
        .all_slots_snapshot()
        .await
        .iter()
        .map(|slot| slot.key.verifying_key())
        .collect();
    let shared = global_ed25519_verifying_keys();
    let _ = shared.write().map(|mut guard| *guard = vks);
}

/// Global ML-DSA signing key store singleton.
///
/// Ensures all services (PolicyService, OAuthService, rotation task) share
/// the same store instance — rotation applies universally.
static ML_DSA_SIGNING_STORE: std::sync::OnceLock<Arc<MlDsaSigningKeyStore>> = std::sync::OnceLock::new();

/// Get or initialize the global ML-DSA signing key store.
///
/// First call initializes from disk; subsequent calls return the same Arc.
pub fn global_ml_dsa_key_store(secrets_dir: &Path, config: &OAuthConfig) -> Arc<MlDsaSigningKeyStore> {
    ML_DSA_SIGNING_STORE.get_or_init(|| {
        Arc::new(load_or_init_ml_dsa_key_store(secrets_dir, config))
    }).clone()
}

/// Global ES256 signing key store singleton.
static ES256_SIGNING_STORE: std::sync::OnceLock<Arc<Es256SigningKeyStore>> = std::sync::OnceLock::new();

/// Get or initialize the global ES256 signing key store.
pub fn global_es256_key_store(secrets_dir: &Path, config: &OAuthConfig) -> Arc<Es256SigningKeyStore> {
    ES256_SIGNING_STORE.get_or_init(|| {
        Arc::new(load_or_init_es256_key_store(secrets_dir, config))
    }).clone()
}

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
    // Atomic write + 0600 perms (#179) — replaces bare std::fs::write which
    // used 0644 (world-readable) and was non-atomic (torn file on crash).
    super::identity_store::write_secret(secrets_dir, &format!("jwt-signing-key.{name}"), &slot.key.to_bytes())?;
    let meta = SlotMeta { nbf: slot.nbf, exp: slot.exp };
    super::identity_store::write_secret(secrets_dir, &format!("jwt-signing-key.{name}.meta"), &serde_json::to_vec(&meta)?)?;
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
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();

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

    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();
    let drain_secs = config.drain_secs();

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

/// Additional stores to rotate alongside the primary Ed25519 store.
pub struct RotationStores {
    pub es256: Option<Arc<Es256SigningKeyStore>>,
    pub ml_dsa: Option<Arc<MlDsaSigningKeyStore>>,
}

pub fn spawn_rotation_task(
    config: Arc<OAuthConfig>,
    secrets_dir: PathBuf,
    store: Arc<SigningKeyStore>,
    extra: RotationStores,
) {
    tokio::task::spawn_local(async move {
        let mut interval = tokio::time::interval(config.rotation_check_interval());
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // Skip the first tick (fires immediately on creation)
        interval.tick().await;
        loop {
            interval.tick().await;
            let now = chrono::Utc::now().timestamp();
            rotate_jwt_keys(&config, &secrets_dir, &store, now).await;
            // Keep the shared HTTP validator's published Ed25519 key set current
            // after every rotation (drain evicted / lead promoted → active).
            refresh_ed25519_verifying_keys(&store).await;
            if let Some(ref es256) = extra.es256 {
                rotate_es256_keys(&config, &secrets_dir, es256, now).await;
            }
            if let Some(ref ml_dsa) = extra.ml_dsa {
                rotate_ml_dsa_keys(&config, &secrets_dir, ml_dsa, now).await;
                // Refresh the global verifying key snapshot.
                let vks: Vec<_> = ml_dsa.all_slots_snapshot().await.iter()
                    .map(ml_dsa_rotation::MlDsaKeySlot::verifying_key)
                    .collect();
                let shared = global_ml_dsa_verifying_keys();
                let _ = shared.write().map(|mut guard| *guard = vks);
            }
        }
    });
}

// ════════════════════════════════════════════════════════════════════════════════
// ES256 (P-256) Key Rotation Store
// ════════════════════════════════════════════════════════════════════════════════

use p256::ecdsa::SigningKey as Es256SigningKey;

#[derive(Clone)]
pub struct Es256KeySlot {
    pub key: Arc<Es256SigningKey>,
    pub nbf: i64,
    pub exp: i64,
}

impl Es256KeySlot {
    pub fn new(key: Es256SigningKey, nbf: i64, exp: i64) -> Self {
        Self { key: Arc::new(key), nbf, exp }
    }

    pub fn kid(&self) -> String {
        crate::auth::jwt::es256_kid(&self.key)
    }
}

#[derive(Default)]
pub struct Es256KeySlots {
    pub drain: Option<Es256KeySlot>,
    pub active: Option<Es256KeySlot>,
    pub lead: Option<Es256KeySlot>,
}

impl Es256KeySlots {
    pub fn all(&self) -> Vec<&Es256KeySlot> {
        [&self.drain, &self.active, &self.lead]
            .into_iter()
            .flatten()
            .collect()
    }
}

#[derive(Clone)]
pub struct Es256SigningKeyStore(pub Arc<RwLock<Es256KeySlots>>);

impl Es256SigningKeyStore {
    pub fn new(slots: Es256KeySlots) -> Self {
        Self(Arc::new(RwLock::new(slots)))
    }

    pub async fn active_key(&self) -> Option<Arc<Es256SigningKey>> {
        self.0.read().await.active.as_ref().map(|s| Arc::clone(&s.key))
    }

    pub async fn all_slots_snapshot(&self) -> Vec<Es256KeySlot> {
        self.0.read().await.all().into_iter().cloned().collect()
    }
}

// ── ES256 persistence ──────────────────────────────────────────────────────

fn es256_slot_paths(secrets_dir: &Path, name: &str) -> (PathBuf, PathBuf) {
    (
        secrets_dir.join(format!("es256-signing-key.{name}")),
        secrets_dir.join(format!("es256-signing-key.{name}.meta")),
    )
}

fn load_es256_slot(secrets_dir: &Path, name: &str) -> Option<Es256KeySlot> {
    let (key_path, meta_path) = es256_slot_paths(secrets_dir, name);
    let seed = std::fs::read(&key_path).ok()?;
    if seed.len() != 32 {
        warn!("ES256 key slot '{name}': unexpected seed length {}", seed.len());
        return None;
    }
    let meta_bytes = std::fs::read(&meta_path).ok()?;
    let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
    let key = Es256SigningKey::from_bytes(seed.as_slice().into()).ok()?;
    Some(Es256KeySlot::new(key, meta.nbf, meta.exp))
}

fn persist_es256_slot(secrets_dir: &Path, name: &str, slot: &Es256KeySlot) -> anyhow::Result<()> {
    // Atomic write + 0600 perms (#179).
    super::identity_store::write_secret(secrets_dir, &format!("es256-signing-key.{name}"), &slot.key.to_bytes())?;
    let meta = SlotMeta { nbf: slot.nbf, exp: slot.exp };
    super::identity_store::write_secret(secrets_dir, &format!("es256-signing-key.{name}.meta"), &serde_json::to_vec(&meta)?)?;
    Ok(())
}

fn delete_es256_slot(secrets_dir: &Path, name: &str) {
    let (key_path, meta_path) = es256_slot_paths(secrets_dir, name);
    let _ = std::fs::remove_file(&key_path);
    let _ = std::fs::remove_file(&meta_path);
}

fn generate_es256_slot(nbf: i64, exp: i64) -> Es256KeySlot {
    let key = Es256SigningKey::random(&mut rand::rngs::OsRng);
    Es256KeySlot::new(key, nbf, exp)
}

pub fn load_or_init_es256_key_store(secrets_dir: &Path, config: &OAuthConfig) -> Es256SigningKeyStore {
    let now = chrono::Utc::now().timestamp();
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();

    let drain = load_es256_slot(secrets_dir, "drain");
    let mut active = load_es256_slot(secrets_dir, "active");
    let lead = load_es256_slot(secrets_dir, "lead");

    if active.is_none() {
        info!("No active ES256 signing key found — generating on first boot");
        let slot = generate_es256_slot(now, now + active_secs);
        if let Err(e) = persist_es256_slot(secrets_dir, "active", &slot) {
            warn!("Could not persist active ES256 key: {e}");
        } else {
            info!("Active ES256 key generated (kid={})", slot.kid());
        }
        active = Some(slot);
    }

    let should_gen_lead = lead.is_none() && active.as_ref()
        .is_some_and(|a| a.exp - now < lead_secs);
    if should_gen_lead {
        let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
        let lead_exp = lead_nbf + active_secs;
        let slot = generate_es256_slot(lead_nbf, lead_exp);
        if let Err(e) = persist_es256_slot(secrets_dir, "lead", &slot) {
            warn!("Could not persist lead ES256 key: {e}");
        }
    }

    let lead = if should_gen_lead { load_es256_slot(secrets_dir, "lead") } else { lead };
    Es256SigningKeyStore::new(Es256KeySlots { drain, active, lead })
}

pub async fn rotate_es256_keys(
    config: &OAuthConfig,
    secrets_dir: &Path,
    store: &Es256SigningKeyStore,
    now: i64,
) {
    let mut slots = store.0.write().await;
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();
    let drain_secs = config.drain_secs();

    // Phase 1: promote lead → active if lead.nbf <= now
    if let Some(ref lead) = slots.lead {
        if lead.nbf <= now {
            if let Some(old_active) = slots.active.take() {
                delete_es256_slot(secrets_dir, "drain");
                if let Err(e) = persist_es256_slot(secrets_dir, "drain", &old_active) {
                    warn!("ES256: failed to persist drain slot: {e}");
                }
                slots.drain = Some(old_active);
            }
            if let Some(new_active) = slots.lead.take() {
                if let Err(e) = persist_es256_slot(secrets_dir, "active", &new_active) {
                    warn!("ES256: failed to persist promoted active: {e}");
                }
                delete_es256_slot(secrets_dir, "lead");
                slots.active = Some(new_active);
                info!("ES256: promoted lead → active");
            }
        }
    }

    // Phase 2: remove expired drain
    if let Some(ref drain) = slots.drain {
        if now >= drain.exp + drain_secs {
            delete_es256_slot(secrets_dir, "drain");
            slots.drain = None;
            info!("ES256: removed expired drain slot");
        }
    }

    // Phase 3: generate lead if active is near expiry
    if let Some(ref active) = slots.active {
        if slots.lead.is_none() && active.exp - now < lead_secs {
            let lead_nbf = active.exp - lead_secs;
            let lead_exp = lead_nbf + active_secs;
            let new_lead = generate_es256_slot(lead_nbf, lead_exp);
            if let Err(e) = persist_es256_slot(secrets_dir, "lead", &new_lead) {
                warn!("ES256: failed to persist new lead: {e}");
            } else {
                info!("ES256: generated new lead key (kid={})", new_lead.kid());
            }
            slots.lead = Some(new_lead);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// ML-DSA-65 Key Rotation Store (always compiled; runtime CryptoPolicy)
// ════════════════════════════════════════════════════════════════════════════════

mod ml_dsa_rotation {
    use super::*;
    use hyprstream_rpc::crypto::pq::{
        MlDsaSigningKey, ml_dsa_generate_keypair, ml_dsa_sk_to_seed, ml_dsa_sk_from_seed,
    };

    #[derive(Clone)]
    pub struct MlDsaKeySlot {
        pub key: Arc<MlDsaSigningKey>,
        pub nbf: i64,
        pub exp: i64,
    }

    impl MlDsaKeySlot {
        pub fn new(key: MlDsaSigningKey, nbf: i64, exp: i64) -> Self {
            Self { key: Arc::new(key), nbf, exp }
        }

        pub fn verifying_key(&self) -> hyprstream_rpc::crypto::pq::MlDsaVerifyingKey {
            ml_dsa::Keypair::verifying_key(&*self.key).clone()
        }
    }

    #[derive(Default)]
    pub struct MlDsaKeySlots {
        pub drain: Option<MlDsaKeySlot>,
        pub active: Option<MlDsaKeySlot>,
        pub lead: Option<MlDsaKeySlot>,
    }

    impl MlDsaKeySlots {
        pub fn all(&self) -> Vec<&MlDsaKeySlot> {
            [&self.drain, &self.active, &self.lead]
                .into_iter()
                .flatten()
                .collect()
        }
    }

    #[derive(Clone)]
    pub struct MlDsaSigningKeyStore(pub Arc<RwLock<MlDsaKeySlots>>);

    impl MlDsaSigningKeyStore {
        pub fn new(slots: MlDsaKeySlots) -> Self {
            Self(Arc::new(RwLock::new(slots)))
        }

        pub async fn active_key(&self) -> Option<Arc<MlDsaSigningKey>> {
            self.0.read().await.active.as_ref().map(|s| Arc::clone(&s.key))
        }

        pub async fn all_slots_snapshot(&self) -> Vec<MlDsaKeySlot> {
            self.0.read().await.all().into_iter().cloned().collect()
        }
    }

    // ── ML-DSA persistence ─────────────────────────────────────────────────

    fn ml_dsa_slot_paths(secrets_dir: &Path, name: &str) -> (PathBuf, PathBuf) {
        (
            secrets_dir.join(format!("ml-dsa-signing-key.{name}")),
            secrets_dir.join(format!("ml-dsa-signing-key.{name}.meta")),
        )
    }

    pub(super) fn load_ml_dsa_slot(secrets_dir: &Path, name: &str) -> Option<MlDsaKeySlot> {
        let (key_path, meta_path) = ml_dsa_slot_paths(secrets_dir, name);
        let seed_bytes = std::fs::read(&key_path).ok()?;
        if seed_bytes.len() != 32 {
            warn!("ML-DSA key slot '{name}': unexpected seed length {}", seed_bytes.len());
            return None;
        }
        let meta_bytes = std::fs::read(&meta_path).ok()?;
        let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_bytes);
        let key = ml_dsa_sk_from_seed(&seed);
        Some(MlDsaKeySlot::new(key, meta.nbf, meta.exp))
    }

    pub(super) fn persist_ml_dsa_slot(secrets_dir: &Path, name: &str, slot: &MlDsaKeySlot) -> anyhow::Result<()> {
        // Atomic write + 0600 perms (#179).
        let seed = ml_dsa_sk_to_seed(&slot.key);
        super::super::identity_store::write_secret(secrets_dir, &format!("ml-dsa-signing-key.{name}"), &seed)?;
        let meta = SlotMeta { nbf: slot.nbf, exp: slot.exp };
        super::super::identity_store::write_secret(secrets_dir, &format!("ml-dsa-signing-key.{name}.meta"), &serde_json::to_vec(&meta)?)?;
        Ok(())
    }

    fn delete_ml_dsa_slot(secrets_dir: &Path, name: &str) {
        let (key_path, meta_path) = ml_dsa_slot_paths(secrets_dir, name);
        let _ = std::fs::remove_file(&key_path);
        let _ = std::fs::remove_file(&meta_path);
    }

    pub(super) fn generate_ml_dsa_slot(nbf: i64, exp: i64) -> MlDsaKeySlot {
        let (key, _vk) = ml_dsa_generate_keypair();
        MlDsaKeySlot::new(key, nbf, exp)
    }

    pub fn load_or_init_ml_dsa_key_store(secrets_dir: &Path, config: &OAuthConfig) -> MlDsaSigningKeyStore {
        let now = chrono::Utc::now().timestamp();
        let active_secs = config.active_secs();
        let lead_secs = config.lead_secs();

        let drain = load_ml_dsa_slot(secrets_dir, "drain");
        let mut active = load_ml_dsa_slot(secrets_dir, "active");
        let lead = load_ml_dsa_slot(secrets_dir, "lead");

        if active.is_none() {
            info!("No active ML-DSA-65 signing key found — generating on first boot");
            let slot = generate_ml_dsa_slot(now, now + active_secs);
            if let Err(e) = persist_ml_dsa_slot(secrets_dir, "active", &slot) {
                warn!("Could not persist active ML-DSA key: {e}");
            } else {
                info!("Active ML-DSA-65 key generated");
            }
            active = Some(slot);
        }

        let should_gen_lead = lead.is_none() && active.as_ref()
            .is_some_and(|a| a.exp - now < lead_secs);
        if should_gen_lead {
            let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
            let lead_exp = lead_nbf + active_secs;
            let slot = generate_ml_dsa_slot(lead_nbf, lead_exp);
            if let Err(e) = persist_ml_dsa_slot(secrets_dir, "lead", &slot) {
                warn!("Could not persist lead ML-DSA key: {e}");
            }
        }

        let lead = if should_gen_lead { load_ml_dsa_slot(secrets_dir, "lead") } else { lead };
        MlDsaSigningKeyStore::new(MlDsaKeySlots { drain, active, lead })
    }

    pub async fn rotate_ml_dsa_keys(
        config: &OAuthConfig,
        secrets_dir: &Path,
        store: &MlDsaSigningKeyStore,
        now: i64,
    ) {
        let mut slots = store.0.write().await;
        let active_secs = config.active_secs();
        let lead_secs = config.lead_secs();
        let drain_secs = config.drain_secs();

        // Phase 1: promote lead → active
        let lead_ready = slots.lead.as_ref().is_some_and(|lead| lead.nbf <= now);
        if lead_ready {
            if let Some(new_active) = slots.lead.take() {
                if let Some(old_active) = slots.active.take() {
                    delete_ml_dsa_slot(secrets_dir, "drain");
                    if let Err(e) = persist_ml_dsa_slot(secrets_dir, "drain", &old_active) {
                        warn!("ML-DSA: failed to persist drain slot: {e}");
                    }
                    slots.drain = Some(old_active);
                }
                if let Err(e) = persist_ml_dsa_slot(secrets_dir, "active", &new_active) {
                    warn!("ML-DSA: failed to persist promoted active: {e}");
                }
                delete_ml_dsa_slot(secrets_dir, "lead");
                slots.active = Some(new_active);
                info!("ML-DSA: promoted lead → active");
            }
        }

        // Phase 2: remove expired drain
        if let Some(ref drain) = slots.drain {
            if now >= drain.exp + drain_secs {
                delete_ml_dsa_slot(secrets_dir, "drain");
                slots.drain = None;
                info!("ML-DSA: removed expired drain slot");
            }
        }

        // Phase 3: generate lead if active is near expiry
        if let Some(ref active) = slots.active {
            if slots.lead.is_none() && active.exp - now < lead_secs {
                let lead_nbf = active.exp - lead_secs;
                let lead_exp = lead_nbf + active_secs;
                let new_lead = generate_ml_dsa_slot(lead_nbf, lead_exp);
                if let Err(e) = persist_ml_dsa_slot(secrets_dir, "lead", &new_lead) {
                    warn!("ML-DSA: failed to persist new lead: {e}");
                } else {
                    info!("ML-DSA: generated new lead key");
                }
                slots.lead = Some(new_lead);
            }
        }
    }
}

pub use ml_dsa_rotation::{MlDsaKeySlot, MlDsaKeySlots, MlDsaSigningKeyStore, load_or_init_ml_dsa_key_store, rotate_ml_dsa_keys};

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

    // ── ES256 store tests ──────────────────────────────────────────────────

    #[test]
    fn es256_load_or_init_creates_active_key() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store = load_or_init_es256_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let key = rt.block_on(store.active_key());
        assert!(key.is_some());
    }

    #[test]
    fn es256_persist_and_reload() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store1 = load_or_init_es256_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let kid1 = {
            let slots = rt.block_on(store1.all_slots_snapshot());
            slots[0].kid()
        };
        let store2 = load_or_init_es256_key_store(dir.path(), &config);
        let kid2 = {
            let slots = rt.block_on(store2.all_slots_snapshot());
            slots[0].kid()
        };
        assert_eq!(kid1, kid2, "ES256 key must survive reload from disk");
    }

    #[tokio::test]
    async fn es256_rotate_promotes_lead() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = generate_es256_slot(now - 14 * 86400, now + 1);
        let lead = generate_es256_slot(now - 1, now + 14 * 86400);
        persist_es256_slot(dir.path(), "active", &active).unwrap();
        persist_es256_slot(dir.path(), "lead", &lead).unwrap();
        let lead_kid = lead.kid();

        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });

        rotate_es256_keys(&config, dir.path(), &store, now).await;

        let slots = store.0.read().await;
        assert_eq!(slots.active.as_ref().unwrap().kid(), lead_kid);
        assert!(slots.drain.is_some());
        assert!(slots.lead.is_none());
    }

    // ── ML-DSA store tests ─────────────────────────────────────────────────

    #[test]
    fn ml_dsa_load_or_init_creates_active_key() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let key = rt.block_on(store.active_key());
        assert!(key.is_some());
    }

    #[test]
    fn ml_dsa_persist_and_reload() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store1 = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let vk1 = {
            let key = rt.block_on(store1.active_key()).unwrap();
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&ml_dsa::Keypair::verifying_key(&*key).clone())
        };
        let store2 = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let vk2 = {
            let key = rt.block_on(store2.active_key()).unwrap();
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&ml_dsa::Keypair::verifying_key(&*key).clone())
        };
        assert_eq!(vk1, vk2, "ML-DSA key must survive reload from disk");
    }

    #[tokio::test]
    async fn ml_dsa_rotate_promotes_lead() {
        use super::ml_dsa_rotation::*;
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = generate_ml_dsa_slot(now - 14 * 86400, now + 1);
        let lead = generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        persist_ml_dsa_slot(dir.path(), "active", &active).unwrap();
        persist_ml_dsa_slot(dir.path(), "lead", &lead).unwrap();
        let lead_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &ml_dsa::Keypair::verifying_key(&*lead.key).clone(),
        );

        let store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });

        rotate_ml_dsa_keys(&config, dir.path(), &store, now).await;

        let slots = store.0.read().await;
        let active_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
            &ml_dsa::Keypair::verifying_key(&*slots.active.as_ref().unwrap().key).clone(),
        );
        assert_eq!(active_vk, lead_vk, "lead must become active after promotion");
        assert!(slots.drain.is_some());
        assert!(slots.lead.is_none());
    }
}
