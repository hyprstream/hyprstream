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

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::config::OAuthConfig;

/// Global shared ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
///
/// Populated at startup from the ML-DSA rotation store's current slots.
/// Updated by the rotation task after each promotion. Services read from this
/// via their `JwtKeySource` implementation.
// Cross-crate `Arc<std::sync::RwLock<..>>` contract with `JwtKeySource` and the
// service factory; intentionally `std::sync::RwLock`, not `parking_lot`.
#[allow(clippy::disallowed_types)]
static ML_DSA_VERIFYING_KEYS: std::sync::OnceLock<
    std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>>,
> = std::sync::OnceLock::new();

/// Get or initialize the global ML-DSA verifying keys Arc.
#[allow(clippy::disallowed_types)]
pub fn global_ml_dsa_verifying_keys(
) -> std::sync::Arc<std::sync::RwLock<Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>>> {
    ML_DSA_VERIFYING_KEYS
        .get_or_init(|| std::sync::Arc::new(std::sync::RwLock::new(Vec::new())))
        .clone()
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
static ED25519_VERIFYING_KEYS: std::sync::OnceLock<PublishedEd25519Keys> =
    std::sync::OnceLock::new();

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

/// Refresh the published ML-DSA component-key snapshot.
pub async fn refresh_ml_dsa_verifying_keys(store: &Arc<MlDsaSigningKeyStore>) {
    let vks: Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey> = store
        .all_slots_snapshot()
        .await
        .iter()
        .map(ml_dsa_rotation::MlDsaKeySlot::verifying_key)
        .collect();
    let shared = global_ml_dsa_verifying_keys();
    let _ = shared.write().map(|mut guard| *guard = vks);
}

#[derive(Clone, Serialize, Deserialize)]
struct CompositeLedger {
    version: u64,
    #[serde(default)]
    component_digest: String,
    pairs: Vec<CompositeLedgerPair>,
}

#[derive(Clone, Serialize, Deserialize)]
struct CompositeCommit {
    version: u64,
    component_digest: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct CompositeAcknowledgement {
    version: u64,
    component_digest: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct CompositeLedgerPair {
    kid: String,
    ml_dsa_public: String,
    ed25519_public: String,
    role: String,
    state: String,
    not_before: i64,
    expires_at: i64,
}

fn composite_ledger_path(secrets_dir: &Path) -> PathBuf {
    secrets_dir.join("jwt-composite-pairs.json")
}

fn composite_committed_path(dir: &Path) -> PathBuf {
    dir.join("jwt-composite-pairs.committed")
}

/// An existing composite commit marker means this store already has signing
/// authority. Missing/unreadable rotation slots must not be mistaken for a
/// first boot: generating replacement Ed/PQ keys would mutate the component
/// state while the committed ledger still names the previous pair. Treat any
/// marker entry (including a malformed or unreadable one) as committed so the
/// normal restore path fails closed without creating keys.
fn has_committed_composite_marker(dir: &Path) -> bool {
    !matches!(
        std::fs::symlink_metadata(composite_committed_path(dir)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
    )
}
fn composite_committed_ledger_path(dir: &Path, commit: &CompositeCommit) -> PathBuf {
    dir.join(format!(
        "jwt-composite-pairs.committed-{}-{}.json",
        commit.version, commit.component_digest
    ))
}
fn composite_ledger_lock_path(dir: &Path) -> PathBuf {
    dir.join("jwt-composite-pairs.lock")
}
fn composite_writer_lock_path(dir: &Path) -> PathBuf {
    dir.join("jwt-composite-pairs.writer.lock")
}
fn composite_subscribers_dir(dir: &Path) -> PathBuf {
    dir.join("jwt-composite-subscribers")
}

static COMPOSITE_SUBSCRIPTION_STARTED: AtomicBool = AtomicBool::new(false);
static COMPOSITE_PUBLISHING: AtomicBool = AtomicBool::new(false);
static COMPOSITE_CA_SIGNING_KEY: std::sync::OnceLock<parking_lot::RwLock<Option<Arc<SigningKey>>>> =
    std::sync::OnceLock::new();
fn composite_ca_signing_key() -> &'static parking_lot::RwLock<Option<Arc<SigningKey>>> {
    COMPOSITE_CA_SIGNING_KEY.get_or_init(|| parking_lot::RwLock::new(None))
}
fn configure_composite_authority(dir: &Path) {
    hyprstream_rpc::auth::global_composite_key_set().configure_authority(
        composite_ledger_path(dir),
        composite_committed_path(dir),
        dir.join("jwt-composite-pairs.committed"),
        composite_ledger_lock_path(dir),
    );
}

/// Load the immutable ledger selected by the commit marker. The fallback is
/// only for ledgers written before immutable committed snapshots were added,
/// and is safe only when the mutable ledger still exactly matches the marker.
fn read_committed_composite_ledger(
    dir: &Path,
) -> anyhow::Result<(CompositeCommit, CompositeLedger)> {
    let commit: CompositeCommit =
        serde_json::from_slice(&std::fs::read(composite_committed_path(dir))?)?;
    let ledger = read_composite_ledger_selected_by_commit(dir, &commit)?;
    Ok((commit, ledger))
}

fn read_composite_ledger_selected_by_commit(
    dir: &Path,
    commit: &CompositeCommit,
) -> anyhow::Result<CompositeLedger> {
    let immutable = composite_committed_ledger_path(dir, commit);
    let ledger: CompositeLedger = match std::fs::read(&immutable) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::symlink_metadata(&immutable) {
                Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
                    serde_json::from_slice(&std::fs::read(composite_ledger_path(dir))?)?
                }
                Ok(_) => return Err(error.into()),
                Err(metadata_error) => return Err(metadata_error.into()),
            }
        }
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(
        ledger.version == commit.version
            && !ledger.component_digest.is_empty()
            && ledger.component_digest == commit.component_digest,
        "committed composite ledger does not match its commit marker"
    );
    Ok(ledger)
}

/// Load the marker-selected authority while the caller holds the exclusive
/// ledger lock. Only an absent marker means first bootstrap. A legacy marker
/// whose matching generation still lives in the mutable ledger is migrated to
/// an immutable snapshot before the lock can be released and a publisher can
/// replace the mutable file.
fn load_or_migrate_committed_composite_ledger(
    dir: &Path,
) -> anyhow::Result<Option<CompositeLedger>> {
    let marker_path = composite_committed_path(dir);
    let marker_bytes = match std::fs::read(&marker_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::symlink_metadata(&marker_path) {
                Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(None);
                }
                Ok(_) => return Err(error.into()),
                Err(metadata_error) => return Err(metadata_error.into()),
            }
        }
        Err(error) => return Err(error.into()),
    };
    let commit: CompositeCommit = serde_json::from_slice(&marker_bytes)?;
    let immutable = composite_committed_ledger_path(dir, &commit);
    let immutable_missing = match std::fs::symlink_metadata(&immutable) {
        Ok(_) => false,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
        Err(error) => return Err(error.into()),
    };
    let ledger = read_composite_ledger_selected_by_commit(dir, &commit)?;
    if immutable_missing {
        let name = immutable
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("invalid committed composite ledger path"))?;
        super::identity_store::write_secret(dir, name, &serde_json::to_vec(&ledger)?)?;
    }
    Ok(Some(ledger))
}

fn digest_component_slot(
    digest: &mut Sha256,
    family: &[u8],
    name: &[u8],
    public: Option<&[u8]>,
    validity: Option<(i64, i64)>,
) {
    digest.update((family.len() as u64).to_be_bytes());
    digest.update(family);
    digest.update((name.len() as u64).to_be_bytes());
    digest.update(name);
    match (public, validity) {
        (Some(public), Some((nbf, exp))) => {
            digest.update([1]);
            digest.update((public.len() as u64).to_be_bytes());
            digest.update(public);
            digest.update(nbf.to_be_bytes());
            digest.update(exp.to_be_bytes());
        }
        _ => digest.update([0]),
    }
}

fn component_state_digest_from_slots(
    ed: &KeySlots,
    pq: &ml_dsa_rotation::MlDsaKeySlots,
    ca_key: ed25519_dalek::VerifyingKey,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"hyprstream-composite-component-state-v1");
    let ed_drain = ed.drain.as_ref().map(KeySlot::verifying_key_bytes);
    let ed_active = ed.active.as_ref().map(KeySlot::verifying_key_bytes);
    let ed_lead = ed.lead.as_ref().map(KeySlot::verifying_key_bytes);
    digest_component_slot(
        &mut digest,
        b"ed25519",
        b"drain",
        ed_drain.as_ref().map(<[u8; 32]>::as_slice),
        ed.drain.as_ref().map(|slot| (slot.nbf, slot.exp)),
    );
    digest_component_slot(
        &mut digest,
        b"ed25519",
        b"active",
        ed_active.as_ref().map(<[u8; 32]>::as_slice),
        ed.active.as_ref().map(|slot| (slot.nbf, slot.exp)),
    );
    digest_component_slot(
        &mut digest,
        b"ed25519",
        b"lead",
        ed_lead.as_ref().map(<[u8; 32]>::as_slice),
        ed.lead.as_ref().map(|slot| (slot.nbf, slot.exp)),
    );
    for (name, slot) in [
        (b"drain".as_slice(), pq.drain.as_ref()),
        (b"active".as_slice(), pq.active.as_ref()),
        (b"lead".as_slice(), pq.lead.as_ref()),
    ] {
        let public =
            slot.map(|slot| hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&slot.verifying_key()));
        digest_component_slot(
            &mut digest,
            b"ml-dsa-65",
            name,
            public.as_deref(),
            slot.map(|slot| (slot.nbf, slot.exp)),
        );
    }
    digest_component_slot(
        &mut digest,
        b"ed25519",
        b"policy-ca",
        Some(&ca_key.to_bytes()),
        Some((i64::MIN, i64::MAX)),
    );
    URL_SAFE_NO_PAD.encode(digest.finalize())
}

/// Cheaply detect a component-key change before deciding that a rejected
/// proposal is still stale. Slot persistence uses atomic file replacement, so
/// identity plus modification time detects a new key without repeatedly
/// deserializing private ML-DSA material on the verifier hot path.
fn fingerprint_component_slot_files(
    secrets_dir: &Path,
    fingerprint: &mut Sha256,
) -> anyhow::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let state_dir = rotation_state_dir(secrets_dir);
    for dir in [secrets_dir, state_dir.as_path()] {
        for family in ["jwt-signing-key", "ml-dsa-signing-key"] {
            for slot in ["drain", "active", "lead"] {
                for suffix in ["", ".meta"] {
                    let path = dir.join(format!("{family}.{slot}{suffix}"));
                    match std::fs::metadata(&path) {
                        Ok(metadata) => {
                            fingerprint.update([1]);
                            fingerprint.update(metadata.dev().to_be_bytes());
                            fingerprint.update(metadata.ino().to_be_bytes());
                            fingerprint.update(metadata.len().to_be_bytes());
                            fingerprint.update(metadata.mtime().to_be_bytes());
                            fingerprint.update(metadata.mtime_nsec().to_be_bytes());
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            fingerprint.update([0]);
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
    }
    Ok(())
}

async fn component_state_digest(
    ed_store: &SigningKeyStore,
    pq_store: &MlDsaSigningKeyStore,
    ca_key: ed25519_dalek::VerifyingKey,
) -> String {
    let ed = ed_store.0.read().await;
    let pq = pq_store.0.read().await;
    component_state_digest_from_slots(&ed, &pq, ca_key)
}

/// Restore the public exact-pair ledger in verifier-only service processes.
pub async fn restore_composite_verifying_key_set(
    secrets_dir: &Path,
    _ed_store: &SigningKeyStore,
    _ml_dsa_store: &MlDsaSigningKeyStore,
    _ca_key: ed25519_dalek::VerifyingKey,
) -> anyhow::Result<()> {
    use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};

    configure_composite_authority(secrets_dir);
    #[cfg(not(test))]
    start_composite_authority_subscription(secrets_dir.to_path_buf(), _ca_key)?;
    use nix::fcntl::{flock, FlockArg};
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(composite_ledger_lock_path(secrets_dir))?;
    flock(lock.as_raw_fd(), FlockArg::LockShared)?;
    let (_, ledger) = read_committed_composite_ledger(secrets_dir)?;
    let now = chrono::Utc::now().timestamp();
    let mut pairs = Vec::new();
    for record in ledger.pairs {
        if record.expires_at <= now {
            continue;
        }
        let pq_bytes = URL_SAFE_NO_PAD.decode(&record.ml_dsa_public)?;
        let pq = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_bytes)?;
        let ed_bytes: [u8; 32] = URL_SAFE_NO_PAD
            .decode(&record.ed25519_public)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid committed Ed25519 key length"))?;
        let ed = ed25519_dalek::VerifyingKey::from_bytes(&ed_bytes)?;
        pairs.push(CompositeKeyPair::verifying(
            record.kid,
            pq,
            ed,
            if record.role == "policy" {
                CompositePairRole::Policy
            } else {
                CompositePairRole::OAuth
            },
            if record.state == "active" {
                CompositePairState::Active
            } else {
                CompositePairState::Drain
            },
            record.not_before,
            record.expires_at,
        ));
    }
    let key_set = hyprstream_rpc::auth::global_composite_key_set();
    if ledger.version > key_set.snapshot().version() {
        key_set.publish(ledger.version, ledger.component_digest, pairs)?;
    }
    Ok(())
}

fn start_composite_authority_subscription(
    secrets_dir: PathBuf,
    ca_key: ed25519_dalek::VerifyingKey,
) -> anyhow::Result<()> {
    if COMPOSITE_SUBSCRIPTION_STARTED.swap(true, Ordering::AcqRel) {
        return Ok(());
    }
    configure_composite_authority(&secrets_dir);
    let subscribers = composite_subscribers_dir(&secrets_dir);
    std::fs::create_dir_all(&subscribers)?;
    let name = format!("{}-{}", std::process::id(), uuid::Uuid::new_v4());
    super::identity_store::write_secret(&subscribers, &name, b"0")?;
    std::thread::Builder::new()
        .name("composite-authority".into())
        .spawn(move || {
            use nix::fcntl::{flock, FlockArg};
            use std::os::fd::AsRawFd;
            let mut last_error: Option<String> = None;
            let mut rejected_pending: Option<([u8; 32], std::time::Instant)> = None;
            loop {
                if COMPOSITE_PUBLISHING.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                let load = (|| -> anyhow::Result<()> {
                    let lock = std::fs::OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .read(true)
                        .write(true)
                        .open(composite_ledger_lock_path(&secrets_dir))?;
                    flock(lock.as_raw_fd(), FlockArg::LockShared)?;
                    let pending_bytes = std::fs::read(composite_ledger_path(&secrets_dir))?;
                    let marker_bytes = std::fs::read(composite_committed_path(&secrets_dir))?;
                    let mut fingerprint = Sha256::new();
                    fingerprint.update(&pending_bytes);
                    fingerprint.update(&marker_bytes);
                    fingerprint_component_slot_files(&secrets_dir, &mut fingerprint)?;
                    let fingerprint: [u8; 32] = fingerprint.finalize().into();
                    if rejected_pending.as_ref().is_some_and(|(previous, at)| {
                        *previous == fingerprint
                            && at.elapsed() < std::time::Duration::from_secs(60)
                    }) {
                        // Neither the proposal nor its commit marker changed.
                        // Retain the already-published authority without
                        // repeatedly deserializing private signing keys.
                        return Ok(());
                    }
                    let pending: CompositeLedger = serde_json::from_slice(&pending_bytes)?;
                    let ed = load_or_init_key_store(&secrets_dir, &OAuthConfig::default());
                    let pq = load_or_init_ml_dsa_key_store(&secrets_dir, &OAuthConfig::default());
                    // Acknowledgement means only that the proposed generation is
                    // completely loadable. It must not become live authority until
                    // the matching commit marker is durable.
                    let local_digest = {
                        let ed_slots = ed.0.blocking_read();
                        let pq_slots = pq.0.blocking_read();
                        component_state_digest_from_slots(&ed_slots, &pq_slots, ca_key)
                    };
                    if pending.component_digest != local_digest {
                        rejected_pending = Some((fingerprint, std::time::Instant::now()));
                        anyhow::bail!(
                            "pending composite ledger does not match local component authority"
                        );
                    }
                    rejected_pending = None;
                    let _staged = ledger_pairs_from_local_keys(&pending, &ed, &pq, ca_key, true)?;
                    super::identity_store::write_secret(
                        &subscribers,
                        &name,
                        &serde_json::to_vec(&CompositeAcknowledgement {
                            version: pending.version,
                            component_digest: pending.component_digest,
                        })?,
                    )?;
                    let (_, committed) = read_committed_composite_ledger(&secrets_dir)?;
                    let key_set = hyprstream_rpc::auth::global_composite_key_set();
                    if committed.version > key_set.snapshot().version()
                        || (committed.version == key_set.snapshot().version()
                            && committed.component_digest != key_set.snapshot().component_digest())
                    {
                        key_set.publish(
                            committed.version,
                            committed.component_digest.clone(),
                            ledger_pairs_from_local_keys(&committed, &ed, &pq, ca_key, false)?,
                        )?;
                    }
                    Ok(())
                })();
                match load {
                    Ok(()) => {
                        last_error = None;
                        std::thread::sleep(std::time::Duration::from_millis(25));
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        if last_error.as_deref() != Some(&message) {
                            tracing::warn!("composite authority reload failed closed: {message}");
                            last_error = Some(message);
                        }
                        // A stale or interrupted pending generation cannot become
                        // live authority. Retrying its expensive key-store load
                        // every 25 ms can exhaust a small verifier host before a
                        // publisher has a chance to replace the pending ledger.
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }
        })?;
    Ok(())
}

fn ledger_pairs_from_local_keys(
    ledger: &CompositeLedger,
    ed_store: &SigningKeyStore,
    pq_store: &MlDsaSigningKeyStore,
    ca_key: ed25519_dalek::VerifyingKey,
    require_local_pairs: bool,
) -> anyhow::Result<Vec<hyprstream_rpc::auth::CompositeKeyPair>> {
    use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};
    let ed_slots = ed_store.0.blocking_read();
    let pq_slots = pq_store.0.blocking_read();
    let ca_signing = composite_ca_signing_key().read().clone();
    let now = chrono::Utc::now().timestamp();
    ledger
        .pairs
        .iter()
        .filter(|r| r.expires_at > now)
        .map(|r| {
            let pq_bytes = URL_SAFE_NO_PAD.decode(&r.ml_dsa_public)?;
            let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_bytes)?;
            let ed_bytes: [u8; 32] = URL_SAFE_NO_PAD
                .decode(&r.ed25519_public)?
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid Ed25519 key length"))?;
            let ed_vk = ed25519_dalek::VerifyingKey::from_bytes(&ed_bytes)?;
            let role = if r.role == "policy" {
                CompositePairRole::Policy
            } else {
                CompositePairRole::OAuth
            };
            let state = if r.state == "active" {
                CompositePairState::Active
            } else {
                CompositePairState::Drain
            };
            let pq_signing = pq_slots
                .all()
                .into_iter()
                .find(|s| {
                    hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&s.verifying_key()) == pq_bytes
                })
                .map(|s| Arc::clone(&s.key));
            let ed_signing = if ca_key == ed_vk {
                ca_signing.clone()
            } else {
                ed_slots
                    .all()
                    .into_iter()
                    .find(|s| s.key.verifying_key() == ed_vk)
                    .map(|s| Arc::clone(&s.key))
            };
            if require_local_pairs {
                anyhow::ensure!(
                    pq_signing.is_some(),
                    "composite PQ key is absent from the pending component generation"
                );
                anyhow::ensure!(
                    ca_key == ed_vk || ed_signing.is_some(),
                    "composite Ed25519 key is absent from the pending component generation"
                );
            }
            Ok(match (pq_signing, ed_signing) {
                (Some(pq), Some(ed)) => CompositeKeyPair::signing(
                    r.kid.clone(),
                    pq,
                    ed,
                    role,
                    state,
                    r.not_before,
                    r.expires_at,
                ),
                _ => CompositeKeyPair::verifying(
                    r.kid.clone(),
                    pq_vk,
                    ed_vk,
                    role,
                    state,
                    r.not_before,
                    r.expires_at,
                ),
            })
        })
        .collect()
}

/// Restore exact persisted associations and atomically publish the complete
/// signing/verifying/JWKS authority. Component stores are used only to resolve
/// identities recorded in the ledger; they are never cross-paired.
pub async fn initialize_composite_key_set(
    secrets_dir: &Path,
    ed_store: &SigningKeyStore,
    ml_dsa_store: &MlDsaSigningKeyStore,
    ca_key: Arc<SigningKey>,
    drain_secs: i64,
) -> anyhow::Result<()> {
    publish_composite_key_set(
        secrets_dir,
        ed_store,
        ml_dsa_store,
        ca_key,
        drain_secs,
        true,
        None,
    )
    .await
}

/// Publish the post-rotation lifecycle as one atomic exact-pair snapshot.
pub async fn refresh_composite_key_set(
    secrets_dir: &Path,
    ed_store: &SigningKeyStore,
    ml_dsa_store: &MlDsaSigningKeyStore,
    ca_key: Arc<SigningKey>,
    drain_secs: i64,
    expected_component_digest: &str,
) -> anyhow::Result<()> {
    publish_composite_key_set(
        secrets_dir,
        ed_store,
        ml_dsa_store,
        ca_key,
        drain_secs,
        false,
        Some(expected_component_digest.to_owned()),
    )
    .await
}

async fn publish_composite_key_set(
    secrets_dir: &Path,
    ed_store: &SigningKeyStore,
    ml_dsa_store: &MlDsaSigningKeyStore,
    ca_key: Arc<SigningKey>,
    drain_secs: i64,
    restore: bool,
    expected_component_digest: Option<String>,
) -> anyhow::Result<()> {
    use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};
    use nix::fcntl::{flock, FlockArg};
    use std::os::fd::AsRawFd;
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            COMPOSITE_PUBLISHING.store(false, Ordering::Release);
        }
    }
    configure_composite_authority(secrets_dir);
    *composite_ca_signing_key().write() = Some(Arc::clone(&ca_key));
    let writer = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(composite_writer_lock_path(secrets_dir))?;
    flock(writer.as_raw_fd(), FlockArg::LockExclusive)?;
    COMPOSITE_PUBLISHING.store(true, Ordering::Release);
    let _guard = Guard;

    let now = chrono::Utc::now().timestamp();
    let ed_component_slots = ed_store.0.read().await.clone();
    let pq_component_slots = ml_dsa_store.0.read().await.clone();
    let ed_slots: Vec<_> = ed_component_slots.all().into_iter().cloned().collect();
    let pq_slots: Vec<_> = pq_component_slots.all().into_iter().cloned().collect();
    let proposed_component_digest = component_state_digest_from_slots(
        &ed_component_slots,
        &pq_component_slots,
        ca_key.verifying_key(),
    );
    let key_set = hyprstream_rpc::auth::global_composite_key_set();
    let current = key_set.snapshot();
    let ledger_lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(composite_ledger_lock_path(secrets_dir))?;
    flock(ledger_lock.as_raw_fd(), FlockArg::LockExclusive)?;
    let persisted = load_or_migrate_committed_composite_ledger(secrets_dir)?;
    let restoring_committed = restore && persisted.is_some();
    let persisted_version = persisted.as_ref().map_or(0, |ledger| ledger.version);
    let persisted_digest = persisted
        .as_ref()
        .map(|ledger| ledger.component_digest.clone());
    if persisted.is_none() {
        // First bootstrap: every required active signing component must be
        // loadable before any pending ledger, immutable snapshot, or commit
        // marker is created. Publishing from an incomplete local signer set
        // (a failed first-boot persistence, for example) would commit a
        // generation with missing OAuth/Policy active pairs — and the commit
        // marker then suppresses first-boot key generation on every restart,
        // so a transient failure would require manual state recovery. With an
        // existing commit marker this branch is skipped and the committed
        // fail-closed behavior is unchanged.
        let ed_active = ed_component_slots.active.is_some();
        let pq_active = pq_component_slots.active.is_some();
        anyhow::ensure!(
            ed_active && pq_active,
            "refusing to initialize composite authority from an incomplete \
             local signer set (ed25519_active={} ml_dsa_active={}): resolve \
             first-boot persistence and retry",
            ed_active,
            pq_active
        );
    }
    match (&persisted, expected_component_digest.as_deref()) {
        (Some(ledger), Some(expected)) => anyhow::ensure!(
            !ledger.component_digest.is_empty() && ledger.component_digest == expected,
            "stale composite component authority: expected {expected}, authoritative digest is {}",
            ledger.component_digest
        ),
        (Some(ledger), None) => anyhow::ensure!(
            restore && !ledger.component_digest.is_empty(),
            "persisted composite authority is not restorable"
        ),
        (None, Some(_)) => {
            anyhow::bail!("stale composite component authority: expected generation has no ledger")
        }
        (None, None) => anyhow::ensure!(restore, "composite authority is not initialized"),
    }
    let mut pairs = Vec::new();

    for record in persisted.into_iter().flat_map(|ledger| ledger.pairs) {
        if record.expires_at <= now {
            continue;
        }
        let pq_bytes = URL_SAFE_NO_PAD.decode(&record.ml_dsa_public)?;
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_bytes)?;
        let pq_signing = pq_slots.iter().find(|slot| {
            URL_SAFE_NO_PAD.encode(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
                &slot.verifying_key(),
            )) == record.ml_dsa_public
        });
        let ed_bytes: [u8; 32] = URL_SAFE_NO_PAD
            .decode(&record.ed25519_public)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid persisted Ed25519 key length"))?;
        let ed_vk = ed25519_dalek::VerifyingKey::from_bytes(&ed_bytes)?;
        let policy_ca_ed_matched =
            URL_SAFE_NO_PAD.encode(ca_key.verifying_key().to_bytes()) == record.ed25519_public;
        let ed_signing = if policy_ca_ed_matched {
            Some(Arc::clone(&ca_key))
        } else {
            ed_slots
                .iter()
                .find(|slot| {
                    URL_SAFE_NO_PAD.encode(slot.key.verifying_key().to_bytes())
                        == record.ed25519_public
                })
                .map(|slot| Arc::clone(&slot.key))
        };
        let role = if record.role == "policy" {
            CompositePairRole::Policy
        } else {
            CompositePairRole::OAuth
        };
        let state = if restoring_committed && record.state == "active" {
            CompositePairState::Active
        } else {
            CompositePairState::Drain
        };
        // A refresh carries forward only committed pairs whose private
        // components remain in the local drain/active/lead authority. Restore
        // may retain verification-only drains from the immutable ledger.
        if !restoring_committed && (pq_signing.is_none() || ed_signing.is_none()) {
            continue;
        }
        if restoring_committed && state == CompositePairState::Active {
            anyhow::ensure!(
                pq_signing.is_some() && ed_signing.is_some(),
                "committed active composite signing pair is unavailable after restart: \
                 kid={:?} role={:?} ed_component_available={} pq_component_available={} \
                 policy_ca_ed_matched={}",
                record.kid,
                record.role,
                ed_signing.is_some(),
                pq_signing.is_some(),
                policy_ca_ed_matched,
            );
        }
        pairs.push(match (pq_signing, ed_signing) {
            (Some(pq), Some(ed)) => CompositeKeyPair::signing(
                record.kid,
                Arc::clone(&pq.key),
                ed,
                role,
                state,
                record.not_before,
                record.expires_at,
            ),
            _ => CompositeKeyPair::verifying(
                record.kid,
                pq_vk,
                ed_vk,
                role,
                state,
                record.not_before,
                record.expires_at,
            ),
        });
    }

    if restoring_committed {
        anyhow::ensure!(
            [CompositePairRole::OAuth, CompositePairRole::Policy]
                .into_iter()
                .all(|role| pairs.iter().any(|pair| {
                    pair.role() == role
                        && pair.state() == CompositePairState::Active
                        && pair.signing_keys().is_some()
                })),
            "committed active composite signing authority is incomplete"
        );
        if persisted_version > current.version() {
            let digest = persisted_digest.ok_or_else(|| {
                anyhow::anyhow!("committed composite digest disappeared during restore")
            })?;
            key_set.publish(persisted_version, digest, pairs)?;
        }
        #[cfg(not(test))]
        start_composite_authority_subscription(secrets_dir.to_path_buf(), ca_key.verifying_key())?;
        return Ok(());
    }

    let active_pq_key = ml_dsa_store.active_key().await;
    let active_pq = active_pq_key.as_ref().and_then(|active| {
        let active_vk = ml_dsa::Keypair::verifying_key(&**active);
        pq_slots
            .iter()
            .find(|slot| {
                hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&active_vk)
                    == hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&slot.verifying_key())
            })
            .cloned()
    });
    let active_ed = ed_store.active_key().await;
    if let Some(pq) = active_pq {
        if let Some(ed) = active_ed {
            upsert_active_pair(
                &mut pairs,
                Arc::clone(&pq.key),
                ed,
                CompositePairRole::OAuth,
                pq.nbf,
                pq.exp + drain_secs,
            );
        }
        upsert_active_pair(
            &mut pairs,
            pq.key,
            Arc::clone(&ca_key),
            CompositePairRole::Policy,
            pq.nbf,
            pq.exp + drain_secs,
        );
    }

    let version = current.version().max(persisted_version).saturating_add(1);
    let ledger = CompositeLedger {
        version,
        component_digest: proposed_component_digest.clone(),
        pairs: pairs.iter().map(ledger_record).collect(),
    };
    super::identity_store::write_secret(
        secrets_dir,
        "jwt-composite-pairs.json",
        &serde_json::to_vec(&ledger)?,
    )?;
    #[cfg(test)]
    if let Some(marker) = std::env::var_os("HYPRSTREAM_COMPOSITE_STAGE_MARKER") {
        super::identity_store::write_secret(
            Path::new(&marker).parent().ok_or_else(|| {
                anyhow::anyhow!("composite stage marker must have a parent directory")
            })?,
            Path::new(&marker)
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow::anyhow!("invalid composite stage marker"))?,
            b"1",
        )?;
    }
    #[cfg(test)]
    if std::env::var_os("HYPRSTREAM_COMPOSITE_CRASH_AFTER_STAGE").is_some() {
        std::process::exit(86);
    }
    drop(ledger_lock);
    wait_for_composite_acknowledgements(secrets_dir, version, &proposed_component_digest)?;
    let ledger_lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(composite_ledger_lock_path(secrets_dir))?;
    flock(ledger_lock.as_raw_fd(), FlockArg::LockExclusive)?;
    let pending: CompositeLedger =
        serde_json::from_slice(&std::fs::read(composite_ledger_path(secrets_dir))?)?;
    anyhow::ensure!(
        pending.version == version && pending.component_digest == proposed_component_digest,
        "composite publication changed before commit"
    );
    let commit = CompositeCommit {
        version,
        component_digest: proposed_component_digest.clone(),
    };
    // Keep every committed generation immutable. A restart racing a later
    // pending publication can therefore restore the marker-selected snapshot
    // instead of either installing pending authority or losing the old one.
    let committed_name = composite_committed_ledger_path(secrets_dir, &commit)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid committed composite ledger path"))?
        .to_owned();
    super::identity_store::write_secret(
        secrets_dir,
        &committed_name,
        &serde_json::to_vec(&pending)?,
    )?;
    super::identity_store::write_secret(
        secrets_dir,
        "jwt-composite-pairs.committed",
        &serde_json::to_vec(&commit)?,
    )?;
    if version > key_set.snapshot().version() {
        key_set.publish(version, proposed_component_digest, pairs)?;
    }
    #[cfg(not(test))]
    start_composite_authority_subscription(secrets_dir.to_path_buf(), ca_key.verifying_key())?;
    Ok(())
}

fn wait_for_composite_acknowledgements(
    dir: &Path,
    version: u64,
    component_digest: &str,
) -> anyhow::Result<()> {
    let subscribers = composite_subscribers_dir(dir);
    std::fs::create_dir_all(&subscribers)?;
    let own = format!("{}-", std::process::id());
    let required: Vec<PathBuf> = std::fs::read_dir(&subscribers)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            if name.starts_with(&own) {
                return false;
            }
            let pid = name
                .split_once('-')
                .and_then(|(p, _)| p.parse::<u32>().ok());
            if pid.is_some_and(|p| !composite_subscriber_alive(p)) {
                let _ = std::fs::remove_file(path);
                return false;
            }
            true
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if required.iter().all(|p| {
            std::fs::read(p)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<CompositeAcknowledgement>(&bytes).ok())
                .is_some_and(|ack| {
                    ack.version == version && ack.component_digest == component_digest
                })
        }) {
            return Ok(());
        }
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "composite generation {version} was not acknowledged by all live service processes"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn composite_subscriber_alive(pid: u32) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    !matches!(kill(Pid::from_raw(pid as i32), None), Err(Errno::ESRCH))
}

fn upsert_active_pair(
    pairs: &mut Vec<hyprstream_rpc::auth::CompositeKeyPair>,
    ml_dsa: Arc<hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
    ed25519: Arc<SigningKey>,
    role: hyprstream_rpc::auth::CompositePairRole,
    not_before: i64,
    expires_at: i64,
) {
    let ml_vk = ml_dsa::Keypair::verifying_key(&*ml_dsa).clone();
    let kid = crate::auth::jwt::composite_kid(&ml_vk, &ed25519.verifying_key());
    pairs.retain(|pair| {
        pair.kid() != kid
            && !(pair.role() == role
                && pair.state() == hyprstream_rpc::auth::CompositePairState::Active)
    });
    pairs.push(hyprstream_rpc::auth::CompositeKeyPair::signing(
        kid,
        ml_dsa,
        ed25519,
        role,
        hyprstream_rpc::auth::CompositePairState::Active,
        not_before,
        expires_at,
    ));
}

fn ledger_record(pair: &hyprstream_rpc::auth::CompositeKeyPair) -> CompositeLedgerPair {
    CompositeLedgerPair {
        kid: pair.kid().to_owned(),
        ml_dsa_public: URL_SAFE_NO_PAD
            .encode(hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(pair.ml_dsa())),
        ed25519_public: URL_SAFE_NO_PAD.encode(pair.ed25519().to_bytes()),
        role: if pair.role() == hyprstream_rpc::auth::CompositePairRole::Policy {
            "policy"
        } else {
            "oauth"
        }
        .to_owned(),
        state: if pair.state() == hyprstream_rpc::auth::CompositePairState::Active {
            "active"
        } else {
            "drain"
        }
        .to_owned(),
        not_before: pair.not_before(),
        expires_at: pair.expires_at(),
    }
}

/// Global ML-DSA signing key store singleton.
///
/// Ensures all services (PolicyService, OAuthService, rotation task) share
/// the same store instance — rotation applies universally.
static ML_DSA_SIGNING_STORE: std::sync::OnceLock<Arc<MlDsaSigningKeyStore>> =
    std::sync::OnceLock::new();
static ED25519_SIGNING_STORE: std::sync::OnceLock<Arc<SigningKeyStore>> =
    std::sync::OnceLock::new();

/// Get or initialize the process-wide Ed25519 rotation store.
pub fn global_ed25519_key_store(secrets_dir: &Path, config: &OAuthConfig) -> Arc<SigningKeyStore> {
    ED25519_SIGNING_STORE
        .get_or_init(|| Arc::new(load_or_init_key_store(secrets_dir, config)))
        .clone()
}

/// Get or initialize the global ML-DSA signing key store.
///
/// First call initializes from disk; subsequent calls return the same Arc.
pub fn global_ml_dsa_key_store(
    secrets_dir: &Path,
    config: &OAuthConfig,
) -> Arc<MlDsaSigningKeyStore> {
    ML_DSA_SIGNING_STORE
        .get_or_init(|| Arc::new(load_or_init_ml_dsa_key_store(secrets_dir, config)))
        .clone()
}

/// Global ES256 signing key store singleton.
static ES256_SIGNING_STORE: std::sync::OnceLock<Arc<Es256SigningKeyStore>> =
    std::sync::OnceLock::new();

/// Get or initialize the global ES256 signing key store.
pub fn global_es256_key_store(
    secrets_dir: &Path,
    config: &OAuthConfig,
) -> Arc<Es256SigningKeyStore> {
    let store = ES256_SIGNING_STORE
        .get_or_init(|| Arc::new(load_or_init_es256_key_store(secrets_dir, config)))
        .clone();
    store.bind_secrets_dir(rotation_state_dir(secrets_dir));
    store
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
        Self {
            key: Arc::new(key),
            nbf,
            exp,
        }
    }

    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }

    pub fn kid(&self) -> String {
        crate::services::oauth::jwks::compute_kid(&self.verifying_key_bytes())
    }
}

// ── Slot container ───────────────────────────────────────────────────────────

#[derive(Clone, Default)]
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

    /// Snapshot the named drain/active/lead roles under one read guard.
    pub async fn slots_snapshot(&self) -> KeySlots {
        self.0.read().await.clone()
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

/// Resolve the directory JWT rotation *state* is persisted to.
///
/// Rotation slots are runtime-mutated state, not provisioned credentials, so
/// they need a writable home (#803). When `secrets_dir` is writable (bare
/// metal, dev, non-systemd) the slots live there, as before. When it is
/// read-only — the systemd `$CREDENTIALS_DIRECTORY` ramfs — slots fall back to
/// the unit's `$STATE_DIRECTORY` (`StateDirectory=`), or to the XDG state dir
/// when not running under systemd. Loading always checks the state dir first
/// and falls back to `secrets_dir`, so credentials provisioned into the
/// credstore still bootstrap the first boot.
pub fn rotation_state_dir(secrets_dir: &Path) -> PathBuf {
    if super::identity_store::is_writable(secrets_dir) {
        return secrets_dir.to_path_buf();
    }
    // systemd: `StateDirectory=` sets $STATE_DIRECTORY to a writable,
    // service-private dir (colon-separated list when multiple are declared;
    // hyprstream units declare exactly one).
    let systemd_state = std::env::var("STATE_DIRECTORY").ok().and_then(|v| {
        let first = v.split(':').next().unwrap_or_default();
        if first.is_empty() {
            None
        } else {
            Some(PathBuf::from(first))
        }
    });
    let base = match systemd_state {
        // Namespace by instance to match the socket/config isolation.
        Some(dir) => match std::env::var("HYPRSTREAM_INSTANCE") {
            Ok(inst) if !inst.is_empty() => Some(dir.join("instances").join(inst)),
            _ => Some(dir),
        },
        // Not systemd: XDG state home (already instance-namespaced via the
        // StoragePaths prefix).
        None => crate::storage::paths::StoragePaths::new()
            .and_then(|s| s.state_dir())
            .ok(),
    };
    match base {
        Some(dir) => dir.join("credentials"),
        None => {
            error!(
                "secrets dir '{}' is read-only and no writable state directory \
                 could be resolved; JWT rotation state will NOT survive restart \
                 (all issued tokens are invalidated on every restart)",
                secrets_dir.display()
            );
            secrets_dir.to_path_buf()
        }
    }
}

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
        warn!(
            "JWT key slot '{name}': unexpected seed length {}",
            seed.len()
        );
        return None;
    }
    let meta_bytes = std::fs::read(&meta_path).ok()?;
    let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
    let mut seed_arr = [0u8; 32];
    seed_arr.copy_from_slice(&seed);
    Some(KeySlot::new(
        SigningKey::from_bytes(&seed_arr),
        meta.nbf,
        meta.exp,
    ))
}

fn persist_slot(secrets_dir: &Path, name: &str, slot: &KeySlot) -> anyhow::Result<()> {
    // Atomic write + 0600 perms (#179) — replaces bare std::fs::write which
    // used 0644 (world-readable) and was non-atomic (torn file on crash).
    super::identity_store::write_secret(
        secrets_dir,
        &format!("jwt-signing-key.{name}"),
        &slot.key.to_bytes(),
    )?;
    let meta = SlotMeta {
        nbf: slot.nbf,
        exp: slot.exp,
    };
    super::identity_store::write_secret(
        secrets_dir,
        &format!("jwt-signing-key.{name}.meta"),
        &serde_json::to_vec(&meta)?,
    )?;
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
/// If no active slot exists, generate one immediately (first boot) — but only
/// activate it when it persists; a persistence failure leaves the store
/// without an active key rather than returning a process-only signer.
/// Slot files: `jwt-signing-key.{active,drain,lead}` + `.meta` JSON.
pub fn load_or_init_key_store(secrets_dir: &Path, config: &OAuthConfig) -> SigningKeyStore {
    let now = chrono::Utc::now().timestamp();
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();

    // Slots are state: read/write them via the writable state dir (#803),
    // falling back to the (possibly read-only, provisioned) secrets dir.
    let state_dir = rotation_state_dir(secrets_dir);
    let has_committed_authority = has_committed_composite_marker(secrets_dir);
    let drain = load_slot(&state_dir, "drain").or_else(|| load_slot(secrets_dir, "drain"));
    let mut active = load_slot(&state_dir, "active").or_else(|| load_slot(secrets_dir, "active"));
    let lead = load_slot(&state_dir, "lead").or_else(|| load_slot(secrets_dir, "lead"));

    if active.is_none() {
        if has_committed_authority {
            error!(
                "Committed composite authority exists but no active JWT signing slot is \
                 available; refusing implicit key generation"
            );
        } else {
            info!("No active JWT signing key found — generating on first boot");
            let slot = generate_slot(now, now + active_secs);
            // Disk-before-memory: the generated key becomes active only after
            // its slot persisted. An unpersisted key is process-only — it can
            // never be recovered after restart — so on failure the store stays
            // without an active key instead of serving one.
            match persist_slot(&state_dir, "active", &slot) {
                Ok(()) => {
                    info!("Active JWT key generated (kid={})", slot.kid());
                    active = Some(slot);
                }
                Err(e) => error!(
                    "Could not persist active JWT key to '{}': {e}. Refusing to \
                     activate the unpersisted key; no active signer is available \
                     until persistence succeeds on a restart.",
                    state_dir.display()
                ),
            }
        }
    }

    // If we have active but no lead and active is close to expiry, generate lead now.
    let should_gen_lead = !has_committed_authority
        && lead.is_none()
        && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
    if should_gen_lead {
        let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
        let lead_exp = lead_nbf + active_secs;
        let slot = generate_slot(lead_nbf, lead_exp);
        if let Err(e) = persist_slot(&state_dir, "lead", &slot) {
            error!(
                "Could not persist lead JWT key to '{}': {e}. Rotation state will \
                 not survive restart.",
                state_dir.display()
            );
        } else {
            info!("Lead JWT key pre-generated at startup (kid={})", slot.kid());
        }
    }

    // Re-load lead if we just generated it
    let lead = if should_gen_lead {
        load_slot(&state_dir, "lead")
    } else {
        lead
    };

    SigningKeyStore::new(KeySlots {
        drain,
        active,
        lead,
    })
}

/// Load the root-DID identity rotation set, seeding the first active slot from
/// the already-provisioned node identity key.
///
/// `secrets_dir` is a dedicated subdirectory, so these durable slot files never
/// collide with the JWT rotation set. Seeding from `initial_key` makes the first
/// deployment a compatibility-preserving transition rather than a flag day.
pub fn load_or_init_root_identity_key_store(
    secrets_dir: &Path,
    config: &OAuthConfig,
    initial_key: SigningKey,
) -> SigningKeyStore {
    let now = chrono::Utc::now().timestamp();
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();

    let drain = load_slot(secrets_dir, "drain");
    let mut active = load_slot(secrets_dir, "active");
    let lead = load_slot(secrets_dir, "lead");

    // Documented follow-up (outside this repair): activation below is not
    // gated on persistence success, unlike `load_or_init_key_store`. Recovery
    // differs — the seed is the provisioned node identity key, reloadable
    // from its own store — but the rotation slot can stay absent on disk
    // until the next successful persist.
    if active.is_none() {
        let slot = KeySlot::new(initial_key, now, now + active_secs);
        if let Err(error) = persist_slot(secrets_dir, "active", &slot) {
            warn!("Could not persist root-DID active identity key: {error}");
        } else {
            info!("Seeded root-DID active identity key (kid={})", slot.kid());
        }
        active = Some(slot);
    }

    let should_generate_lead = lead.is_none()
        && active
            .as_ref()
            .is_some_and(|slot| slot.exp - now < lead_secs);
    if should_generate_lead {
        let lead_nbf = active.as_ref().map_or(now, |slot| slot.exp - lead_secs);
        let slot = generate_slot(lead_nbf, lead_nbf + active_secs);
        if let Err(error) = persist_slot(secrets_dir, "lead", &slot) {
            warn!("Could not persist root-DID lead identity key: {error}");
        }
    }
    let lead = if should_generate_lead {
        load_slot(secrets_dir, "lead")
    } else {
        lead
    };

    SigningKeyStore::new(KeySlots {
        drain,
        active,
        lead,
    })
}

// ── Rotation logic ──────────────────────────────────────────────────────────

/// Returns `true` when every signer slot this cycle touched finished fully
/// restored (or promoted). `false` means a promotion rollback failed: the
/// durable state is only partially restored, the old active signer stays
/// recoverable in its drain-slot copy, and callers must not treat the cycle
/// as a clean success. A later tick retries the queued lead.
pub async fn rotate_jwt_keys(
    config: &OAuthConfig,
    secrets_dir: &Path,
    store: &SigningKeyStore,
    now: i64,
) -> bool {
    let state_dir = rotation_state_dir(secrets_dir);
    rotate_jwt_keys_in_state_dir(config, &state_dir, store, now).await
}

/// Rotation against an explicit state directory. Private test seam: passing a
/// state dir that cannot hold files injects deterministic persistence
/// failures without relying on permission bits.
async fn rotate_jwt_keys_in_state_dir(
    config: &OAuthConfig,
    state_dir: &Path,
    store: &SigningKeyStore,
    now: i64,
) -> bool {
    let mut slots = store.0.write().await;

    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();
    let drain_secs = config.drain_secs();

    // 1. Promote lead → active if lead.nbf has passed.
    //
    // Disk-before-memory: every persistence operation required to retain the
    // old active signer (the durable drain write) and to durably install the
    // new active signer must succeed before any in-memory slot changes. On a
    // failure the previous active stays active in memory, the lead stays
    // queued for the next tick, and no promotion is reported. The gate is
    // successful completion of `persist_slot` (temp-file + rename); that is
    // not proven power-loss durability, and key bytes and metadata remain two
    // separate files (documented residuals of this repair).
    if let Some(new_lead) = slots.lead.as_ref().filter(|l| l.nbf <= now).cloned() {
        let old_active = slots.active.clone();
        let old_drain = slots.drain.clone();

        info!("Promoting lead JWT key (kid={}) to active", new_lead.kid());

        // Old active → drain, persisted first so a process that reloads
        // between the writes can still verify the previous generation. The
        // previous drain is not evicted up front: the atomic write replaces
        // it, and it is restored if that write fails.
        if let Some(ref old) = old_active {
            if let Err(e) = persist_slot(state_dir, "drain", old) {
                // A failed drain write can already have replaced the seed
                // before its metadata write failed, so restoring (or
                // removing) the previous drain is part of the rollback and
                // its outcome must be confirmed before the cycle may claim
                // fully restored state.
                let drain_restored = if let Some(ref prev) = old_drain {
                    persist_slot(state_dir, "drain", prev)
                } else {
                    delete_slot(state_dir, "drain");
                    // Error-preserving removal proof: only a definite
                    // NotFound for both files counts as removed; any other
                    // metadata error leaves restoration unconfirmed.
                    let (drain_key, drain_meta) = slot_paths(state_dir, "drain");
                    let removed = [&drain_key, &drain_meta].iter().all(|path| {
                        matches!(
                            std::fs::symlink_metadata(path),
                            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                        )
                    });
                    if !removed {
                        Err(anyhow::anyhow!(
                            "partial replacement drain slot still present after removal"
                        ))
                    } else {
                        Ok(())
                    }
                };
                if let Err(drain_error) = drain_restored {
                    error!(
                        "Could not restore previous drain JWT key to '{}': \
                         {drain_error}. The durable drain slot is not fully \
                         restored; retaining the old active in memory and \
                         retrying next tick.",
                        state_dir.display()
                    );
                    return false;
                }
                error!(
                    "Could not persist drain JWT key to '{}': {e}. Retaining the \
                     old active in memory; promotion will be retried next tick.",
                    state_dir.display()
                );
                return true;
            }
        }

        if let Err(e) = persist_slot(state_dir, "active", &new_lead) {
            // Roll back the active slot first, and only after that restore is
            // confirmed touch the drain slot: the drain copy written above is
            // the only durable copy of the old active signer, so a failing
            // rollback must leave it in place for restart recovery instead of
            // deleting or overwriting it with the previous drain.
            let restored = match old_active {
                Some(ref old) => persist_slot(state_dir, "active", old),
                None => {
                    delete_slot(state_dir, "active");
                    Ok(())
                }
            };
            if let Err(restore_error) = restored {
                error!(
                    "Could not restore active JWT key to '{}': {restore_error}. The \
                     failed promotion left the old active signer only in its drain \
                     slot; that recoverable copy is kept untouched and promotion \
                     will be retried next tick.",
                    state_dir.display()
                );
                return false;
            }
            if old_active.is_some() {
                if let Some(ref prev) = old_drain {
                    if let Err(drain_error) = persist_slot(state_dir, "drain", prev) {
                        error!(
                            "Could not restore previous drain JWT key to '{}': \
                             {drain_error}. The old active signer is restored in the \
                             active slot, but the durable drain slot may hold a \
                             partial write; promotion will be retried next tick.",
                            state_dir.display()
                        );
                        return false;
                    }
                } else {
                    delete_slot(state_dir, "drain");
                }
            }
            error!(
                "Could not persist active JWT key to '{}': {e}. Retaining the old \
                 active in memory; promotion will be retried next tick.",
                state_dir.display()
            );
            return true;
        }

        // All persistence succeeded — swap the in-memory slots and clear the
        // persisted lead slot.
        if let Some(ref prev) = old_drain {
            info!(
                "Evicting previous drain JWT key (kid={}) during promotion",
                prev.kid()
            );
        }
        delete_slot(state_dir, "lead");
        if let Some(old) = old_active {
            slots.drain = Some(old);
        }
        slots.active = Some(new_lead);
        slots.lead = None;
    }

    // 2. Remove drain if drain window has closed.
    if slots
        .drain
        .as_ref()
        .is_some_and(|d| now >= d.exp + drain_secs)
    {
        let kid = slots.drain.as_ref().map(KeySlot::kid).unwrap_or_default();
        info!("Removing expired drain JWT key (kid={kid})");
        delete_slot(state_dir, "drain");
        slots.drain = None;
    }

    // 3. Generate lead if active is approaching expiry and lead is absent.
    if slots.lead.is_none() {
        if let Some(active) = &slots.active {
            if active.exp - now < lead_secs {
                let lead_nbf = active.exp - lead_secs;
                let lead_exp = lead_nbf + active_secs;
                let new_lead = generate_slot(lead_nbf, lead_exp);
                info!(
                    "Generated new lead JWT key (kid={}, nbf={})",
                    new_lead.kid(),
                    new_lead.nbf
                );
                if let Err(e) = persist_slot(state_dir, "lead", &new_lead) {
                    error!(
                        "Could not persist lead JWT key to '{}': {e}. Rotation state \
                         will not survive restart.",
                        state_dir.display()
                    );
                }
                slots.lead = Some(new_lead);
            }
        }
    }
    true
}

/// Rotate only the root-DID Ed25519 identity set.
///
/// The same proven lead → active → bounded-drain state machine is reused, but
/// the dedicated directory keeps its authority and persistence separate from
/// JWT signing keys.
pub fn spawn_root_identity_rotation_task(
    config: Arc<OAuthConfig>,
    secrets_dir: PathBuf,
    store: Arc<SigningKeyStore>,
) {
    tokio::task::spawn_local(async move {
        let mut interval = tokio::time::interval(
            config
                .rotation_check_interval()
                .min(std::time::Duration::from_secs(60)),
        );
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            interval.tick().await;
            rotate_jwt_keys(
                &config,
                &secrets_dir,
                &store,
                chrono::Utc::now().timestamp(),
            )
            .await;
        }
    });
}

// ── Background task ─────────────────────────────────────────────────────────

/// Additional stores to rotate alongside the primary Ed25519 store.
pub struct RotationStores {
    pub es256: Option<Arc<Es256SigningKeyStore>>,
    pub ml_dsa: Option<Arc<MlDsaSigningKeyStore>>,
    pub composite_ca_key: Arc<SigningKey>,
}

/// The recovery transaction recorded when a rotation cycle defers composite
/// publication: `committed_digest` names the marker-selected committed
/// generation the cycle deferred against, and `states` contains only
/// component digests computed from this process's in-memory stores. A fresh
/// reload from mutable slot files is never certified as part of the failed
/// cycle. A later tick may anchor publication only when its tick-start memory
/// digest is in `states`, and the committed digest is still checked against
/// the ledger. After restart, a matching reload may continue; a different
/// reload remains untrusted and cannot advance the ledger. This prefers
/// explicit repair over guessing which writer produced divergent disk state.
#[derive(Clone, Serialize, Deserialize)]
struct DeferredPredecessorRecord {
    committed_digest: String,
    states: Vec<String>,
}

static DEFERRED_PREDECESSOR_FALLBACK: parking_lot::RwLock<Option<DeferredPredecessorRecord>> =
    parking_lot::RwLock::new(None);

fn deferred_predecessor_path(secrets_dir: &Path) -> PathBuf {
    rotation_state_dir(secrets_dir).join("jwt-composite-deferred-predecessor")
}

/// Cross-process serialization for one rotation cycle: component slot
/// mutation, committed-predecessor validation, recovery-state capture and
/// record lifecycle all happen while this lock is held, so no competing
/// writer can interleave a component advance between them. The composite
/// publisher's own ledger flock is acquired later, inside publication, so
/// the lock order is cycle lock -> publisher lock and never the reverse.
fn cycle_lock_path(secrets_dir: &Path) -> PathBuf {
    rotation_state_dir(secrets_dir).join("jwt-composite-rotation-cycle.lock")
}

/// Read the recovery record. The in-process register is authoritative when
/// present (it is set whenever the durable write could not be persisted);
/// otherwise the durable file copy is parsed. A missing, unreadable, or
/// malformed record reads as `None` — never as supersession.
fn read_deferred_predecessor(secrets_dir: &Path) -> Option<DeferredPredecessorRecord> {
    if let Some(record) = DEFERRED_PREDECESSOR_FALLBACK.read().clone() {
        return Some(record);
    }
    let value = std::fs::read_to_string(deferred_predecessor_path(secrets_dir)).ok()?;
    serde_json::from_str::<DeferredPredecessorRecord>(value.trim()).ok()
}

/// Persist the durable copy of the record; returns whether it was written.
/// The in-process register retains the record until it is cleared or
/// successfully advanced, so a failed durable write never loses the recovery
/// transaction on its own.
fn write_deferred_predecessor(secrets_dir: &Path, record: &DeferredPredecessorRecord) -> bool {
    let state_dir = rotation_state_dir(secrets_dir);
    match serde_json::to_vec(record) {
        Ok(bytes) => super::identity_store::write_secret(
            &state_dir,
            "jwt-composite-deferred-predecessor",
            &bytes,
        )
        .is_ok(),
        Err(error) => {
            warn!("could not serialize the deferred predecessor record: {error}");
            false
        }
    }
}

fn clear_deferred_predecessor(secrets_dir: &Path) {
    *DEFERRED_PREDECESSOR_FALLBACK.write() = None;
    let _ = std::fs::remove_file(deferred_predecessor_path(secrets_dir));
}

/// One background rotation tick: rotate every algorithm store, refresh the
/// published verifier snapshots, and — only when every rotation finished
/// fully restored — republish the composite authority. A cycle that ended
/// not fully restored (a failed promotion rollback) must not publish: the
/// durable slots diverge from what memory would propose, so a staged pending
/// ledger could never be acknowledged and would only strand a divergent
/// proposal next to the retained prior authority.
///
/// When a cycle defers, the cycle records a recovery transaction: the
/// committed component digest it deferred against (validated to match the
/// marker-selected committed ledger) plus the certified component states it
/// left behind (post-rotation memory and the durable disk state). A later
/// all-restored tick anchors publication on the recorded committed digest
/// only when its own tick-start memory digest is one of the certified
/// states — proving its stores continue exactly that recorded transition;
/// any other record is ignored, and stores that match nothing keep the
/// pre-existing stale-authority rejection. Every anchored cycle advances
/// the certified states to its own post-rotation outcome, so recovery
/// survives chained deferrals, failed recovery publications, and restarts.
/// The record is durable (survives restart), cleared on successful
/// publication, retained on validation I/O errors, and discarded only after
/// a verified supersession, followed by one retry anchored on current
/// memory. Recovery therefore survives chained deferrals and failed
/// recovery publications (each certified cycle re-certifies its own
/// outcome); a restart recovers while the on-disk state still digests to a
/// certified state and fail-closes (retaining prior authority) after a
/// foreign writer mutates the shared slot files. Returns `true` when every
/// rotation finished fully restored; a lock-contended tick skips without
/// rotating and also returns `true` (the competing cycle owns the
/// transition; this tick changes nothing).
async fn run_rotation_cycle(
    config: &OAuthConfig,
    secrets_dir: &Path,
    store: &SigningKeyStore,
    extra: &RotationStores,
    now: i64,
) -> bool {
    // Serialize the whole cycle against competing writers: slot mutation,
    // predecessor validation, recovery-state capture and record lifecycle
    // must not interleave with another process's cycle, or a captured state
    // could certify a transition it did not produce. A contending writer
    // skips its tick entirely and retries on the next interval. The
    // publisher's ledger flock is acquired later, inside publication, so the
    // lock order is cycle lock -> publisher lock and never the reverse.
    let cycle_lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cycle_lock_path(secrets_dir))
        .map_err(|error| {
            warn!("could not open the rotation cycle lock: {error}");
        });
    let cycle_lock = match cycle_lock {
        Ok(file) => file,
        Err(()) => return true,
    };
    use nix::fcntl::{flock, FlockArg};
    use std::os::fd::AsRawFd as _;
    if let Err(error) = flock(cycle_lock.as_raw_fd(), FlockArg::LockExclusiveNonblock) {
        warn!(
            "another rotation cycle holds the cycle lock; skipping this tick \
             ({error})"
        );
        return true;
    }

    let pre_rotation_digest = if let Some(ref ml_dsa) = extra.ml_dsa {
        Some(component_state_digest(store, ml_dsa, extra.composite_ca_key.verifying_key()).await)
    } else {
        None
    };
    let mut all_restored = rotate_jwt_keys(config, secrets_dir, store, now).await;
    if !all_restored {
        warn!(
            "Ed25519 rotation ended not fully restored: a failed promotion \
             rollback left the signer slots only partially restored; the old \
             active signer remains active in memory and publication is \
             deferred until a retry fully restores it"
        );
    }
    // Keep the shared HTTP validator's published Ed25519 key set current after
    // every rotation (drain evicted / lead promoted → active). Memory is
    // unchanged on a not-restored cycle, so the previously published verifier
    // state is retained there by construction.
    refresh_ed25519_verifying_keys(store).await;
    if let Some(ref es256) = extra.es256 {
        rotate_es256_keys(config, secrets_dir, es256, now).await;
        // C4 (#1170): re-seal the op-log head after the ES256
        // promotion so cross-process readers (the registry's
        // `PdsPublisher` under `--ipc`) observe the new active
        // `#atproto` generation. Dedicated head-signing key + shared
        // state dir; async because the ES256 store is a tokio RwLock
        // (F1). C2 (#1168) seam — see `auth::op_log`.
        if let Err(error) = super::op_log::advance_sealed_head(secrets_dir, es256).await {
            error!(
                "failed to re-seal op-log head after ES256 rotation; \
                 cross-process readers will retain the prior generation: {error}"
            );
        }
    }
    if let (Some(ref ml_dsa), Some(pre_rotation)) = (&extra.ml_dsa, pre_rotation_digest.as_deref())
    {
        let pq_restored = rotate_ml_dsa_keys(config, secrets_dir, ml_dsa, now).await;
        all_restored &= pq_restored;
        if !pq_restored {
            warn!(
                "ML-DSA rotation ended not fully restored: a failed promotion \
                 rollback left the signer slots only partially restored; the old \
                 active signer remains active in memory and publication is \
                 deferred until a retry fully restores it"
            );
        }
        refresh_ml_dsa_verifying_keys(ml_dsa).await;
        let post_memory =
            component_state_digest(store, ml_dsa, extra.composite_ca_key.verifying_key()).await;
        // The recovery record applies only when this tick's component memory
        // is one of the certified states of the recorded deferred transition;
        // a record describing some other transition is ignored (never used,
        // never treated as supersession).
        let retained = read_deferred_predecessor(secrets_dir);
        let certified = retained
            .as_ref()
            .map(|record| record.states.iter().any(|state| state == pre_rotation))
            .unwrap_or(false);
        let anchor = match retained.as_ref() {
            Some(record) if certified => {
                info!(
                    "anchoring deferred composite publication on the retained \
                     committed predecessor"
                );
                Some(record.committed_digest.clone())
            }
            _ => None,
        };
        // Post-rotation component state: the cycle's in-memory outcome and the
        // durable slot state (what a restarted process would reload). Anchored
        // cycles certify this outcome as the transition's next state.
        if all_restored {
            let expected = anchor.as_deref().unwrap_or(pre_rotation);
            match refresh_composite_key_set(
                secrets_dir,
                store,
                ml_dsa,
                Arc::clone(&extra.composite_ca_key),
                config.drain_secs(),
                expected,
            )
            .await
            {
                Ok(()) => {
                    if retained.is_some() {
                        clear_deferred_predecessor(secrets_dir);
                    }
                }
                Err(error) => {
                    // Distinguish verified supersession from unavailable
                    // authority: only a successfully read ledger whose digest
                    // differs proves the recorded recovery transaction was
                    // superseded by a competing publication. A read failure
                    // retains the record and the prior authority. An anchored
                    // (certified) cycle that failed to publish advances the
                    // certified states to this tick's post-rotation outcome,
                    // so the next tick can still anchor and publish.
                    let superseded = retained.as_ref().map(|record| {
                        read_committed_composite_ledger(secrets_dir)
                            .map(|(_, ledger)| ledger.component_digest != record.committed_digest)
                            .unwrap_or(false)
                    });
                    match (retained.as_ref(), superseded) {
                        (Some(_record), Some(true)) => {
                            // Verified supersession: the recorded committed
                            // predecessor no longer matches the ledger.
                            clear_deferred_predecessor(secrets_dir);
                            warn!(
                                "deferred composite predecessor superseded by a \
                                 competing publication; retrying once anchored on \
                                 current component memory"
                            );
                            if let Err(retry_error) = refresh_composite_key_set(
                                secrets_dir,
                                store,
                                ml_dsa,
                                Arc::clone(&extra.composite_ca_key),
                                config.drain_secs(),
                                pre_rotation,
                            )
                            .await
                            {
                                warn!(
                                    "composite key-set publication failed; retaining \
                                     prior authority: {retry_error}"
                                );
                            }
                        }
                        (Some(record), _) if certified => {
                            // Anchored (certified) cycle whose publication
                            // failed: advance the certified states to this
                            // tick's own post-rotation memory outcome, keeping
                            // the committed predecessor, so the next tick can
                            // still anchor and publish the recovery. Only the
                            // memory outcome is certified here: the durable
                            // slot files this tick did not rewrite cannot be
                            // proven to continue the transition (another
                            // writer may have replaced them), so the disk
                            // digest is never added by an advance.
                            let advanced = DeferredPredecessorRecord {
                                committed_digest: record.committed_digest.clone(),
                                states: vec![post_memory.clone()],
                            };
                            if write_deferred_predecessor(secrets_dir, &advanced) {
                                *DEFERRED_PREDECESSOR_FALLBACK.write() = None;
                            } else {
                                *DEFERRED_PREDECESSOR_FALLBACK.write() = Some(advanced);
                            }
                            warn!(
                                "composite key-set publication failed; the certified \
                                 recovery state advanced with this tick's rotations \
                                 and publication will be retried"
                            );
                        }
                        _ => {
                            // An uncertified record must never acquire this
                            // tick's outcome: the stale-authority rejection
                            // stands, the retained record is left untouched
                            // for its own writer, and the prior authority is
                            // kept.
                            warn!(
                                "composite key-set publication failed; retaining prior \
                                 authority: {error}"
                            );
                        }
                    }
                }
            }
        } else {
            // Seed only from the state computed in this process. The
            // pre-rotation digest must still match the committed ledger; an
            // already-drifted store keeps the stale-authority rejection. Do
            // not reload and certify mutable slot files: another writer may
            // have changed them before this record was created. After a
            // restart, a disk digest not in this record cannot authorize a
            // retry; explicit repair is safer than guessing.
            //
            // A chained certified deferral (record present, memory still
            // certified) advances the certified states to this tick's own
            // memory outcome only when a family durably advanced; when the
            // failure left the transition unchanged, the record is kept
            // as-is. Durable slot files a tick did not rewrite cannot be
            // proven to continue the transition (another writer may have
            // replaced them between cycles), so their digest is never
            // certified by an advance — restart recovery after such an
            // advance fail-closes instead of authorizing a foreign state.
            // A failed record write retains the record in memory, and the
            // durable copy is retried on later ticks.
            match retained.as_ref() {
                None => {
                    if let Ok((_, ledger)) = read_committed_composite_ledger(secrets_dir) {
                        if ledger.component_digest == pre_rotation {
                            let record = DeferredPredecessorRecord {
                                committed_digest: pre_rotation.to_owned(),
                                states: vec![post_memory.clone()],
                            };
                            if write_deferred_predecessor(secrets_dir, &record) {
                                info!(
                                    "composite publication deferred; committed predecessor \
                                     and certified deferred states recorded for recovery"
                                );
                            } else {
                                *DEFERRED_PREDECESSOR_FALLBACK.write() = Some(record);
                                warn!(
                                    "could not persist the deferred predecessor record; \
                                     retaining it in memory for this process"
                                );
                            }
                        }
                    }
                }
                Some(record) if certified => {
                    let states = if post_memory == *pre_rotation {
                        // Transition unchanged (the same failure repeated):
                        // the recorded states already describe it.
                        record.states.clone()
                    } else {
                        // A family durably advanced in this cycle: certify
                        // only the memory outcome.
                        vec![post_memory.clone()]
                    };
                    let advanced = DeferredPredecessorRecord {
                        committed_digest: record.committed_digest.clone(),
                        states,
                    };
                    if write_deferred_predecessor(secrets_dir, &advanced) {
                        *DEFERRED_PREDECESSOR_FALLBACK.write() = None;
                    } else {
                        *DEFERRED_PREDECESSOR_FALLBACK.write() = Some(advanced);
                    }
                }
                Some(_) => {
                    // Uncertified record: it is not this tick's transition, so
                    // it is left untouched. The durable copy of the in-process
                    // register is refreshed (the register holds solely
                    // certified-at-write-time records); a durable file copy is
                    // left as-is.
                    if !deferred_predecessor_path(secrets_dir).exists() {
                        if let Some(record) = DEFERRED_PREDECESSOR_FALLBACK.read().clone() {
                            write_deferred_predecessor(secrets_dir, &record);
                        }
                    }
                }
            }
            warn!(
                "rotation cycle ended not fully restored; deferring composite \
                 key-set publication so the cycle cannot publish as successfully \
                 restored"
            );
        }
    }
    all_restored
}

pub fn spawn_rotation_task(
    config: Arc<OAuthConfig>,
    secrets_dir: PathBuf,
    store: Arc<SigningKeyStore>,
    extra: RotationStores,
) {
    tokio::task::spawn_local(async move {
        // Bounded overlap is an expiry contract, not merely six-hour
        // housekeeping. Do not leave an expired method served for a full
        // default rotation interval.
        let mut interval = tokio::time::interval(
            config
                .rotation_check_interval()
                .min(std::time::Duration::from_secs(60)),
        );
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // Skip the first tick (fires immediately on creation)
        interval.tick().await;
        loop {
            interval.tick().await;
            let now = chrono::Utc::now().timestamp();
            run_rotation_cycle(&config, &secrets_dir, &store, &extra, now).await;
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
        Self {
            key: Arc::new(key),
            nbf,
            exp,
        }
    }

    pub fn kid(&self) -> String {
        crate::auth::jwt::es256_kid(&self.key)
    }
}

#[derive(Default, Clone)]
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

type Es256PromotionHook = dyn Fn(Arc<Es256SigningKey>) -> anyhow::Result<()> + Send + Sync;

#[derive(Clone)]
pub struct Es256SigningKeyStore(
    pub Arc<parking_lot::RwLock<Es256KeySlots>>,
    Arc<parking_lot::RwLock<Option<Arc<Es256PromotionHook>>>>,
    Arc<parking_lot::RwLock<Option<PathBuf>>>,
);

impl Es256SigningKeyStore {
    pub fn new(slots: Es256KeySlots) -> Self {
        Self(
            Arc::new(parking_lot::RwLock::new(slots)),
            Arc::new(parking_lot::RwLock::new(None)),
            Arc::new(parking_lot::RwLock::new(None)),
        )
    }

    /// Bind this process-local cache to the durable slot directory.  Each
    /// process refreshes before it signs or resolves, so `--ipc` workers see a
    /// promotion made by the OAuth process rather than retaining a stale key.
    pub fn bind_secrets_dir(&self, secrets_dir: PathBuf) {
        *self.2.write() = Some(secrets_dir);
    }

    pub fn refresh_from_disk(&self) {
        let Some(secrets_dir) = self.2.read().clone() else {
            return;
        };
        let active = load_es256_slot(&secrets_dir, "active");
        // Never replace a usable in-memory authority with a partially written
        // or unavailable filesystem snapshot.
        let Some(active) = active else {
            warn!("ES256: durable active slot unavailable; retaining cached authority");
            return;
        };
        *self.0.write() = Es256KeySlots {
            drain: load_es256_slot(&secrets_dir, "drain"),
            active: Some(active),
            lead: load_es256_slot(&secrets_dir, "lead"),
        };
    }

    /// Install the in-process action run while a lead → active transition is
    /// still hidden behind the store write guard. The publisher stores a weak
    /// self-reference in this hook, so the shared key store does not keep the
    /// service alive. Cross-`--ipc` promotion delivery remains tracked by
    /// #1123.
    pub(crate) fn set_promotion_hook(&self, hook: Arc<Es256PromotionHook>) {
        *self.1.write() = Some(hook);
    }

    fn notify_promotion(&self, key: Arc<Es256SigningKey>) -> anyhow::Result<()> {
        let hook = self.1.read().clone();
        if let Some(hook) = hook {
            hook(key)?;
        }
        Ok(())
    }

    /// The current active ES256 signing key — the single `#atproto` key the
    /// DID document publishes and the repo head is signed with (#918
    /// re-sign-on-rotation). Synchronous (parking_lot lock) so the publisher
    /// can resolve the LIVE key at sign time without an async runtime.
    pub fn active_key(&self) -> Option<Arc<Es256SigningKey>> {
        self.refresh_from_disk();
        self.0.read().active.as_ref().map(|s| Arc::clone(&s.key))
    }

    /// Take active, drain and lead from one read guard so a DID document never
    /// mixes two rotation generations.
    pub fn slots_snapshot(&self) -> Es256KeySlots {
        self.refresh_from_disk();
        self.0.read().clone()
    }

    /// The active slot (key + `nbf`/`exp` bounds), if present.
    pub fn active_slot(&self) -> Option<Es256KeySlot> {
        self.0.read().active.clone()
    }

    /// Snapshot the bounded verification-only drain slot, if any.
    pub fn drain_slot(&self) -> Option<Es256KeySlot> {
        self.0.read().drain.clone()
    }

    /// Snapshot the bounded verification-only lead slot, if any. New commits
    /// are always signed by [`Es256SigningKeyStore::active_key`].
    pub fn lead_slot(&self) -> Option<Es256KeySlot> {
        self.0.read().lead.clone()
    }

    pub fn all_slots_snapshot(&self) -> Vec<Es256KeySlot> {
        self.0.read().all().into_iter().cloned().collect()
    }
}

/// Resolve an ES256 signing key by kid from the persisted slot files in
/// `secrets_dir` (scans drain/active/lead). Used by the sealed op-log head
/// reader ([`super::op_log`]) to materialize the active generation's signing
/// key once the head has authenticated *which* kid is active. Reading from
/// disk — not the in-memory `OnceLock` — is what makes a cross-process reader
/// observe a slot promoted by another process.
pub(crate) fn es256_signing_key_for_kid(secrets_dir: &Path, kid: &str) -> Option<Es256SigningKey> {
    let state_dir = rotation_state_dir(secrets_dir);
    for name in ["active", "drain", "lead"] {
        if let Some(slot) =
            load_es256_slot(&state_dir, name).or_else(|| load_es256_slot(secrets_dir, name))
        {
            if slot.kid() == kid {
                return Some((*slot.key).clone());
            }
        }
    }
    None
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
        warn!(
            "ES256 key slot '{name}': unexpected seed length {}",
            seed.len()
        );
        return None;
    }
    let meta_bytes = std::fs::read(&meta_path).ok()?;
    let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
    let key = Es256SigningKey::from_bytes(seed.as_slice().into()).ok()?;
    Some(Es256KeySlot::new(key, meta.nbf, meta.exp))
}

pub(crate) fn persist_es256_slot(
    secrets_dir: &Path,
    name: &str,
    slot: &Es256KeySlot,
) -> anyhow::Result<()> {
    // Atomic write + 0600 perms (#179).
    super::identity_store::write_secret(
        secrets_dir,
        &format!("es256-signing-key.{name}"),
        &slot.key.to_bytes(),
    )?;
    let meta = SlotMeta {
        nbf: slot.nbf,
        exp: slot.exp,
    };
    super::identity_store::write_secret(
        secrets_dir,
        &format!("es256-signing-key.{name}.meta"),
        &serde_json::to_vec(&meta)?,
    )?;
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

pub fn load_or_init_es256_key_store(
    secrets_dir: &Path,
    config: &OAuthConfig,
) -> Es256SigningKeyStore {
    match super::op_log::resolve_oplog_state_dir(secrets_dir) {
        Ok(state_dir) => {
            load_or_init_es256_key_store_with_oplog_state_dir(secrets_dir, config, &state_dir)
        }
        Err(error) => {
            error!(
                "Could not resolve durable ES256 operation-log authority: {error}; refusing implicit key generation"
            );
            load_or_init_es256_key_store_with_authority(secrets_dir, config, true)
        }
    }
}

fn load_or_init_es256_key_store_with_oplog_state_dir(
    secrets_dir: &Path,
    config: &OAuthConfig,
    oplog_state_dir: &Path,
) -> Es256SigningKeyStore {
    let has_durable_authority = match has_es256_oplog_authority(oplog_state_dir) {
        Ok(present) => present,
        Err(error) => {
            error!(
                "Could not inspect durable ES256 operation-log authority: {error}; refusing implicit key generation"
            );
            true
        }
    };
    load_or_init_es256_key_store_with_authority(secrets_dir, config, has_durable_authority)
}

fn has_es256_oplog_authority(state_dir: &Path) -> std::io::Result<bool> {
    use std::io::ErrorKind;

    for name in [
        super::op_log::SEALED_HEAD_FILENAME,
        super::op_log::SEALED_HEAD_MAX_SEQ_FILENAME,
        super::op_log::HEAD_VERIFYING_KEY_FILENAME,
    ] {
        match std::fs::symlink_metadata(state_dir.join(name)) {
            Ok(_) => return Ok(true),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}

fn load_or_init_es256_key_store_with_authority(
    secrets_dir: &Path,
    config: &OAuthConfig,
    has_durable_authority: bool,
) -> Es256SigningKeyStore {
    let now = chrono::Utc::now().timestamp();
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();

    let state_dir = rotation_state_dir(secrets_dir);
    let mut drain =
        load_es256_slot(&state_dir, "drain").or_else(|| load_es256_slot(secrets_dir, "drain"));
    let mut active =
        load_es256_slot(&state_dir, "active").or_else(|| load_es256_slot(secrets_dir, "active"));
    let mut lead =
        load_es256_slot(&state_dir, "lead").or_else(|| load_es256_slot(secrets_dir, "lead"));

    // Do not serve an already-expired bounded method after restart. This is
    // the same exclusive boundary the verifier and runtime cleanup use.
    if drain.as_ref().is_some_and(|slot| now >= slot.exp) {
        delete_es256_slot(&state_dir, "drain");
        drain = None;
    }
    if lead.as_ref().is_some_and(|slot| now >= slot.exp) {
        delete_es256_slot(&state_dir, "lead");
        lead = None;
    }

    if active.is_none() {
        if has_durable_authority {
            error!(
                "Durable ES256 operation-log authority exists but no active signing slot is \
                 available; refusing implicit key generation"
            );
        } else {
            info!("No active ES256 signing key found — generating on first boot");
            let slot = generate_es256_slot(now, now + active_secs);
            match persist_es256_slot(&state_dir, "active", &slot) {
                Ok(()) => {
                    info!("Active ES256 key generated (kid={})", slot.kid());
                    active = Some(slot);
                }
                Err(error) => error!(
                    "Could not persist active ES256 key to '{}': {error}. Refusing \
                     to activate the unpersisted key; no active signer is available \
                     until persistence succeeds on a restart.",
                    state_dir.display()
                ),
            }
        }
    }

    let should_gen_lead =
        lead.is_none() && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
    if should_gen_lead {
        let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
        let lead_exp = lead_nbf + active_secs;
        let slot = generate_es256_slot(lead_nbf, lead_exp);
        if let Err(e) = persist_es256_slot(&state_dir, "lead", &slot) {
            error!(
                "Could not persist lead ES256 key to '{}': {e}. Rotation state will \
                 not survive restart.",
                state_dir.display()
            );
        }
    }

    let lead = if should_gen_lead {
        load_es256_slot(&state_dir, "lead")
    } else {
        lead
    };
    Es256SigningKeyStore::new(Es256KeySlots {
        drain,
        active,
        lead,
    })
}

/// Returns `true` if the active key was promoted (lead → active). An installed
/// in-process promotion hook is invoked before the candidate becomes visible.
///
/// Promotion is recoverable with respect to the live key: the store write
/// guard prevents readers from observing the candidate key until persistence
/// and the persisted repo-head re-sign both succeed. Deployments without an
/// in-process publisher may omit the hook and rely on the bounded overlap
/// slots used by `--ipc` workers.
pub async fn rotate_es256_keys(
    config: &OAuthConfig,
    secrets_dir: &Path,
    store: &Es256SigningKeyStore,
    now: i64,
) -> bool {
    let state_dir = rotation_state_dir(secrets_dir);
    rotate_es256_keys_in_state_dir(config, &state_dir, store, now).await
}

async fn rotate_es256_keys_in_state_dir(
    config: &OAuthConfig,
    state_dir: &Path,
    store: &Es256SigningKeyStore,
    now: i64,
) -> bool {
    let mut slots = store.0.write();
    let active_secs = config.active_secs();
    let lead_secs = config.lead_secs();
    let drain_secs = config.drain_secs();
    let mut promoted = false;

    // Phase 1: promote lead → active if lead.nbf <= now
    if let Some(new_active) = slots.lead.as_ref().filter(|lead| lead.nbf <= now).cloned() {
        let old_active = slots.active.clone();
        let old_drain = slots.drain.clone();
        let new_drain = old_active.as_ref().map(|old_active| {
            // The old active key becomes verification-only for one bounded
            // drain interval. Store the actual publication expiry in the slot
            // itself so the DID document and cleanup use one boundary.
            Es256KeySlot::new(
                old_active.key.as_ref().clone(),
                old_active.nbf,
                now.saturating_add(drain_secs),
            )
        });

        // Persist drain before active. A process that reloads between the two
        // writes can therefore still verify K-signed commits.
        if let Some(ref drain) = new_drain {
            if let Err(error) = persist_es256_slot(state_dir, "drain", drain) {
                if let Some(ref old_drain) = old_drain {
                    let _ = persist_es256_slot(state_dir, "drain", old_drain);
                } else {
                    delete_es256_slot(state_dir, "drain");
                }
                error!(
                    "ES256: failed to persist drain slot to '{}'; retaining old active: {error}",
                    state_dir.display()
                );
                return false;
            }
        }

        if let Err(error) = persist_es256_slot(state_dir, "active", &new_active) {
            if let Some(ref old_active) = old_active {
                let _ = persist_es256_slot(state_dir, "active", old_active);
            } else {
                delete_es256_slot(state_dir, "active");
            }
            if let Some(ref old_drain) = old_drain {
                let _ = persist_es256_slot(state_dir, "drain", old_drain);
            } else {
                delete_es256_slot(state_dir, "drain");
            }
            error!(
                "ES256: failed to persist promoted active to '{}'; retaining old active: {error}",
                state_dir.display()
            );
            return false;
        }

        if let Err(error) = store.notify_promotion(Arc::clone(&new_active.key)) {
            // A hook may have committed the candidate head before returning an
            // error. Re-sign with the old key defensively, then restore the
            // durable slot generation so the queued lead can be retried.
            if let Some(ref old_active) = old_active {
                if let Err(rollback_error) = store.notify_promotion(Arc::clone(&old_active.key)) {
                    warn!(
                        "ES256: failed to restore PDS head after promotion failure: {rollback_error}"
                    );
                }
                if let Err(rollback_error) = persist_es256_slot(state_dir, "active", old_active) {
                    warn!(
                        "ES256: failed to restore persisted active after promotion failure: {rollback_error}"
                    );
                }
            } else {
                delete_es256_slot(state_dir, "active");
            }
            if let Some(ref old_drain) = old_drain {
                if let Err(rollback_error) = persist_es256_slot(state_dir, "drain", old_drain) {
                    warn!(
                        "ES256: failed to restore persisted drain after promotion failure: {rollback_error}"
                    );
                }
            } else {
                delete_es256_slot(state_dir, "drain");
            }
            warn!(
                "ES256: failed to re-sign PDS head; retaining old active and retrying next tick: {error}"
            );
            return false;
        }

        delete_es256_slot(state_dir, "lead");
        slots.drain = new_drain.or(old_drain);
        slots.active = Some(new_active);
        slots.lead = None;
        promoted = true;
        info!("ES256: promoted lead → active");
    }

    // Phase 2: remove expired drain
    if let Some(ref drain) = slots.drain {
        if now >= drain.exp {
            delete_es256_slot(state_dir, "drain");
            slots.drain = None;
            info!("ES256: removed expired drain slot");
        }
    }

    // A lead is publication-only and must not survive its own bounded window
    // if an administrator supplied inconsistent slot files.
    if slots.lead.as_ref().is_some_and(|lead| now >= lead.exp) {
        delete_es256_slot(state_dir, "lead");
        slots.lead = None;
        info!("ES256: removed expired lead slot");
    }

    // Phase 3: generate lead if active is near expiry
    if let Some(ref active) = slots.active {
        if slots.lead.is_none() && active.exp - now < lead_secs {
            let lead_nbf = active.exp - lead_secs;
            let lead_exp = lead_nbf + active_secs;
            let new_lead = generate_es256_slot(lead_nbf, lead_exp);
            if let Err(e) = persist_es256_slot(state_dir, "lead", &new_lead) {
                error!(
                    "ES256: failed to persist new lead to '{}': {e}. Rotation state \
                     will not survive restart.",
                    state_dir.display()
                );
            } else {
                info!("ES256: generated new lead key (kid={})", new_lead.kid());
            }
            slots.lead = Some(new_lead);
        }
    }
    promoted
}

// ════════════════════════════════════════════════════════════════════════════════
// ML-DSA-65 Key Rotation Store (always compiled; runtime CryptoPolicy)
// ════════════════════════════════════════════════════════════════════════════════

mod ml_dsa_rotation {
    use super::*;
    use hyprstream_rpc::crypto::pq::{
        ml_dsa_generate_keypair, ml_dsa_sk_from_seed, ml_dsa_sk_to_seed, MlDsaSigningKey,
    };

    #[derive(Clone)]
    pub struct MlDsaKeySlot {
        pub key: Arc<MlDsaSigningKey>,
        pub nbf: i64,
        pub exp: i64,
    }

    impl MlDsaKeySlot {
        pub fn new(key: MlDsaSigningKey, nbf: i64, exp: i64) -> Self {
            Self {
                key: Arc::new(key),
                nbf,
                exp,
            }
        }

        pub fn verifying_key(&self) -> hyprstream_rpc::crypto::pq::MlDsaVerifyingKey {
            ml_dsa::Keypair::verifying_key(&*self.key).clone()
        }
    }

    #[derive(Clone, Default)]
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
            self.0
                .read()
                .await
                .active
                .as_ref()
                .map(|s| Arc::clone(&s.key))
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
            warn!(
                "ML-DSA key slot '{name}': unexpected seed length {}",
                seed_bytes.len()
            );
            return None;
        }
        let meta_bytes = std::fs::read(&meta_path).ok()?;
        let meta: SlotMeta = serde_json::from_slice(&meta_bytes).ok()?;
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&seed_bytes);
        let key = ml_dsa_sk_from_seed(&seed);
        Some(MlDsaKeySlot::new(key, meta.nbf, meta.exp))
    }

    pub(super) fn persist_ml_dsa_slot(
        secrets_dir: &Path,
        name: &str,
        slot: &MlDsaKeySlot,
    ) -> anyhow::Result<()> {
        // Atomic write + 0600 perms (#179).
        let seed = ml_dsa_sk_to_seed(&slot.key);
        super::super::identity_store::write_secret(
            secrets_dir,
            &format!("ml-dsa-signing-key.{name}"),
            &seed,
        )?;
        let meta = SlotMeta {
            nbf: slot.nbf,
            exp: slot.exp,
        };
        super::super::identity_store::write_secret(
            secrets_dir,
            &format!("ml-dsa-signing-key.{name}.meta"),
            &serde_json::to_vec(&meta)?,
        )?;
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

    pub fn load_or_init_ml_dsa_key_store(
        secrets_dir: &Path,
        config: &OAuthConfig,
    ) -> MlDsaSigningKeyStore {
        let now = chrono::Utc::now().timestamp();
        let active_secs = config.active_secs();
        let lead_secs = config.lead_secs();

        let state_dir = rotation_state_dir(secrets_dir);
        let has_committed_authority = has_committed_composite_marker(secrets_dir);
        let drain = load_ml_dsa_slot(&state_dir, "drain")
            .or_else(|| load_ml_dsa_slot(secrets_dir, "drain"));
        let mut active = load_ml_dsa_slot(&state_dir, "active")
            .or_else(|| load_ml_dsa_slot(secrets_dir, "active"));
        let lead =
            load_ml_dsa_slot(&state_dir, "lead").or_else(|| load_ml_dsa_slot(secrets_dir, "lead"));

        if active.is_none() {
            if has_committed_authority {
                error!(
                    "Committed composite authority exists but no active ML-DSA signing slot is \
                     available; refusing implicit key generation"
                );
            } else {
                info!("No active ML-DSA-65 signing key found — generating on first boot");
                let slot = generate_ml_dsa_slot(now, now + active_secs);
                // Disk-before-memory, matching `load_or_init_key_store`: the
                // generated key activates only after its slot persisted.
                match persist_ml_dsa_slot(&state_dir, "active", &slot) {
                    Ok(()) => {
                        info!("Active ML-DSA-65 key generated");
                        active = Some(slot);
                    }
                    Err(e) => error!(
                        "Could not persist active ML-DSA key to '{}': {e}. Refusing \
                         to activate the unpersisted key; no active signer is \
                         available until persistence succeeds on a restart.",
                        state_dir.display()
                    ),
                }
            }
        }

        let should_gen_lead = !has_committed_authority
            && lead.is_none()
            && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
        if should_gen_lead {
            let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
            let lead_exp = lead_nbf + active_secs;
            let slot = generate_ml_dsa_slot(lead_nbf, lead_exp);
            if let Err(e) = persist_ml_dsa_slot(&state_dir, "lead", &slot) {
                error!(
                    "Could not persist lead ML-DSA key to '{}': {e}. Rotation state \
                     will not survive restart.",
                    state_dir.display()
                );
            }
        }

        let lead = if should_gen_lead {
            load_ml_dsa_slot(&state_dir, "lead")
        } else {
            lead
        };
        MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain,
            active,
            lead,
        })
    }

    /// Returns `true` when every signer slot this cycle touched finished fully
    /// restored (or promoted). `false` means a promotion rollback failed: the
    /// durable state is only partially restored, the old active signer stays
    /// recoverable in its drain-slot copy, and callers must not treat the
    /// cycle as a clean success. A later tick retries the queued lead.
    pub async fn rotate_ml_dsa_keys(
        config: &OAuthConfig,
        secrets_dir: &Path,
        store: &MlDsaSigningKeyStore,
        now: i64,
    ) -> bool {
        let state_dir = rotation_state_dir(secrets_dir);
        rotate_ml_dsa_keys_in_state_dir(config, &state_dir, store, now).await
    }

    /// Rotation against an explicit state directory. Private test seam: a
    /// state dir that cannot hold files injects deterministic persistence
    /// failures without relying on permission bits.
    pub(super) async fn rotate_ml_dsa_keys_in_state_dir(
        config: &OAuthConfig,
        state_dir: &Path,
        store: &MlDsaSigningKeyStore,
        now: i64,
    ) -> bool {
        let mut slots = store.0.write().await;
        let active_secs = config.active_secs();
        let lead_secs = config.lead_secs();
        let drain_secs = config.drain_secs();

        // Phase 1: promote lead → active. Same disk-before-memory contract as
        // the Ed25519 rotation: the durable drain write (retaining the old
        // active) and the durable active write (installing the new signer)
        // must both succeed before any in-memory slot changes. On a failure
        // the old active stays active in memory, the lead stays queued, the
        // pre-promotion durable state is restored best-effort, and no
        // promotion is reported. The gate is successful completion of
        // `persist_ml_dsa_slot`; not proven power-loss durability (residual).
        if let Some(new_active) = slots.lead.as_ref().filter(|lead| lead.nbf <= now).cloned() {
            let old_active = slots.active.clone();
            let old_drain = slots.drain.clone();

            info!("ML-DSA: promoting lead to active");

            if let Some(ref old) = old_active {
                if let Err(e) = persist_ml_dsa_slot(state_dir, "drain", old) {
                    // Mirror of the Ed25519 rollback: the restore (or removal)
                    // of the previous drain is part of the rollback and its
                    // outcome must be confirmed.
                    let drain_restored = if let Some(ref prev) = old_drain {
                        persist_ml_dsa_slot(state_dir, "drain", prev)
                    } else {
                        delete_ml_dsa_slot(state_dir, "drain");
                        // Error-preserving removal proof, mirroring Ed25519.
                        let (drain_key, drain_meta) = ml_dsa_slot_paths(state_dir, "drain");
                        let removed = [&drain_key, &drain_meta].iter().all(|path| {
                            matches!(
                                std::fs::symlink_metadata(path),
                                Err(error) if error.kind() == std::io::ErrorKind::NotFound
                            )
                        });
                        if !removed {
                            Err(anyhow::anyhow!(
                                "partial replacement drain slot still present after removal"
                            ))
                        } else {
                            Ok(())
                        }
                    };
                    if let Err(drain_error) = drain_restored {
                        error!(
                            "ML-DSA: could not restore previous drain slot to '{}': \
                             {drain_error}. The durable drain slot is not fully \
                             restored; retaining the old active in memory and \
                             retrying next tick.",
                            state_dir.display()
                        );
                        return false;
                    }
                    error!(
                        "ML-DSA: failed to persist drain slot to '{}': {e}. Retaining \
                         the old active in memory; promotion will be retried next \
                         tick.",
                        state_dir.display()
                    );
                    return true;
                }
            }

            if let Err(e) = persist_ml_dsa_slot(state_dir, "active", &new_active) {
                // Roll back the active slot first, and only after that restore
                // is confirmed touch the drain slot: the drain copy written
                // above is the only durable copy of the old active signer, so
                // a failing rollback must leave it in place for restart
                // recovery instead of deleting or overwriting it with the
                // previous drain.
                let restored = match old_active {
                    Some(ref old) => persist_ml_dsa_slot(state_dir, "active", old),
                    None => {
                        delete_ml_dsa_slot(state_dir, "active");
                        Ok(())
                    }
                };
                if let Err(restore_error) = restored {
                    error!(
                        "ML-DSA: could not restore active key to '{}': {restore_error}. \
                         The failed promotion left the old active signer only in its \
                         drain slot; that recoverable copy is kept untouched and \
                         promotion will be retried next tick.",
                        state_dir.display()
                    );
                    return false;
                }
                if old_active.is_some() {
                    if let Some(ref prev) = old_drain {
                        if let Err(drain_error) = persist_ml_dsa_slot(state_dir, "drain", prev) {
                            error!(
                                "ML-DSA: could not restore previous drain slot to '{}': \
                                 {drain_error}. The old active signer is restored in the \
                                 active slot, but the durable drain slot may hold a \
                                 partial write; promotion will be retried next tick.",
                                state_dir.display()
                            );
                            return false;
                        }
                    } else {
                        delete_ml_dsa_slot(state_dir, "drain");
                    }
                }
                error!(
                    "ML-DSA: failed to persist promoted active to '{}': {e}. Retaining \
                     the old active in memory; promotion will be retried next tick.",
                    state_dir.display()
                );
                return true;
            }

            delete_ml_dsa_slot(state_dir, "lead");
            if let Some(old) = old_active {
                slots.drain = Some(old);
            }
            slots.active = Some(new_active);
            slots.lead = None;
            info!("ML-DSA: promoted lead → active");
        }

        // Phase 2: remove expired drain
        if let Some(ref drain) = slots.drain {
            if now >= drain.exp + drain_secs {
                delete_ml_dsa_slot(state_dir, "drain");
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
                if let Err(e) = persist_ml_dsa_slot(state_dir, "lead", &new_lead) {
                    error!(
                        "ML-DSA: failed to persist new lead to '{}': {e}. Rotation state \
                         will not survive restart.",
                        state_dir.display()
                    );
                } else {
                    info!("ML-DSA: generated new lead key");
                }
                slots.lead = Some(new_lead);
            }
        }
        true
    }
}

pub use ml_dsa_rotation::{
    load_or_init_ml_dsa_key_store, rotate_ml_dsa_keys, MlDsaKeySlot, MlDsaKeySlots,
    MlDsaSigningKeyStore,
};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::TempDir;

    #[tokio::test]
    async fn committed_active_pair_failure_reports_safe_component_availability() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_DIAGNOSTIC_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::committed_active_pair_failure_reports_safe_component_availability",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let now = chrono::Utc::now().timestamp();
        let ca = Arc::new(SigningKey::from_bytes(&[0x61; 32]));
        let ed = SigningKeyStore::new(KeySlots {
            active: Some(KeySlot::new(
                SigningKey::from_bytes(&[0x62; 32]),
                now - 60,
                now + 3600,
            )),
            ..KeySlots::default()
        });
        let pq = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            active: Some(ml_dsa_rotation::generate_ml_dsa_slot(now - 60, now + 3600)),
            ..MlDsaKeySlots::default()
        });
        initialize_composite_key_set(dir.path(), &ed, &pq, Arc::clone(&ca), 300)
            .await
            .unwrap();
        let commit: CompositeCommit =
            serde_json::from_slice(&std::fs::read(composite_committed_path(dir.path())).unwrap())
                .unwrap();
        let immutable = composite_committed_ledger_path(dir.path(), &commit);
        let mut ledger: CompositeLedger =
            serde_json::from_slice(&std::fs::read(&immutable).unwrap()).unwrap();
        let oauth = ledger
            .pairs
            .iter()
            .find(|pair| pair.role == "oauth" && pair.state == "active")
            .unwrap();

        let missing_ed = SigningKeyStore::new(KeySlots::default());
        let error =
            initialize_composite_key_set(dir.path(), &missing_ed, &pq, Arc::clone(&ca), 300)
                .await
                .expect_err("missing committed OAuth Ed component must fail closed");
        assert_eq!(
            error.to_string(),
            format!(
                "committed active composite signing pair is unavailable after restart: \
                 kid={:?} role={:?} ed_component_available=false pq_component_available=true \
                 policy_ca_ed_matched=false",
                oauth.kid, oauth.role,
            )
        );
        assert!(!error.to_string().contains(&oauth.ed25519_public));
        assert!(!error.to_string().contains(&oauth.ml_dsa_public));

        // Put Policy first in this synthetic committed snapshot so the same
        // restore check reports its CA match when the PQ component is absent.
        ledger.pairs.sort_by_key(|pair| pair.role != "policy");
        std::fs::write(&immutable, serde_json::to_vec(&ledger).unwrap()).unwrap();
        let policy = ledger
            .pairs
            .iter()
            .find(|pair| pair.role == "policy" && pair.state == "active")
            .unwrap();
        let missing_pq = MlDsaSigningKeyStore::new(MlDsaKeySlots::default());
        let error = initialize_composite_key_set(dir.path(), &ed, &missing_pq, ca, 300)
            .await
            .expect_err("missing committed Policy PQ component must fail closed");
        assert_eq!(
            error.to_string(),
            format!(
                "committed active composite signing pair is unavailable after restart: \
                 kid={:?} role={:?} ed_component_available=true pq_component_available=false \
                 policy_ca_ed_matched=true",
                policy.kid, policy.role,
            )
        );
        assert!(!error.to_string().contains(&policy.ed25519_public));
        assert!(!error.to_string().contains(&policy.ml_dsa_public));
    }

    #[test]
    fn rejected_composite_proposal_rechecks_replaced_component_slot() {
        let dir = TempDir::new().unwrap();
        let fingerprint = || {
            let mut digest = Sha256::new();
            fingerprint_component_slot_files(dir.path(), &mut digest).unwrap();
            digest.finalize().to_vec()
        };
        let missing = fingerprint();
        super::super::identity_store::write_secret(
            dir.path(),
            "jwt-signing-key.active",
            b"first-component",
        )
        .unwrap();
        let first = fingerprint();
        assert_ne!(missing, first);
        super::super::identity_store::write_secret(
            dir.path(),
            "jwt-signing-key.active",
            b"second-component",
        )
        .unwrap();
        assert_ne!(first, fingerprint());
    }

    struct PermitFixtureAccountReads;

    impl hyprstream_pds_service::AccountRecordReadAuthorizer for PermitFixtureAccountReads {
        fn check_read(
            &self,
            _subject: &hyprstream_rpc::Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&hyprstream_rpc::auth::mac::SecurityContext>,
            _object_id: &str,
        ) -> hyprstream_rpc::auth::mac::MacDecision {
            hyprstream_rpc::auth::mac::MacDecision::Permit
        }
    }

    fn hosted_account_store(
        label: &str,
        zone: &str,
    ) -> anyhow::Result<Arc<hyprstream_pds_service::AccountRecordStore>> {
        let ed = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let (pq, pq_vk) = hyprstream_crypto::pq::ml_dsa_generate_keypair();
        let hybrid = hyprstream_pds::did_op::HybridRotationKey::new(
            ed.verifying_key().to_bytes(),
            hyprstream_crypto::pq::ml_dsa_vk_bytes(&pq_vk),
        )?;
        let rotations = hyprstream_pds::did_op::GenesisRotationKeys::new(
            hyprstream_pds::did_op::UserRotationKey::new(hybrid),
            hyprstream_pds::did_op::RecoveryKeyEnrollment::Declined,
            hyprstream_pds::did_op::HostKeyEnrollment::Absent,
        )?;
        let name =
            hyprstream_pds::AllocatedAccountName::new(label, format!("did:web:{label}.{zone}"))?;
        let mint = hyprstream_pds::HostedAccountMint::begin(name, rotations)?;
        let document = mint.seal_did_document(&format!("https://{zone}"))?;
        let pending =
            mint.prepare_genesis(document, hyprstream_pds::did_op::GenesisRepoHead::EmptyRepo)?;
        let signature = hyprstream_pds::did_op::sign_genesis(pending.unsigned_genesis(), &ed, &pq)?;
        let record = pending.seal(signature)?.record_bytes().to_vec();
        let root = hyprstream_vfs::SyntheticNode::dir().with_child(
            zone,
            hyprstream_vfs::SyntheticNode::dir().with_child(
                "accounts",
                hyprstream_vfs::SyntheticNode::dir().with_child(
                    label,
                    hyprstream_vfs::SyntheticNode::dir().with_child(
                        "account-record.cbor",
                        hyprstream_vfs::SyntheticNode::file(record),
                    ),
                ),
            ),
        );
        Ok(Arc::new(hyprstream_pds_service::AccountRecordStore::new(
            Arc::new(hyprstream_vfs::SyntheticMount::new(root)),
            Arc::new(PermitFixtureAccountReads),
        )))
    }

    fn test_config() -> OAuthConfig {
        OAuthConfig {
            jwt_key_active_days: 14,
            jwt_key_lead_days: 7,
            jwt_key_drain_days: 30,
            ..OAuthConfig::default()
        }
    }

    struct ProductionAuthorityVerifier {
        transport: hyprstream_rpc::transport::TransportConfig,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        key_source: Arc<hyprstream_rpc::auth::ClusterKeySource>,
        accepted: Arc<std::sync::atomic::AtomicBool>,
        error: Arc<parking_lot::Mutex<Option<String>>>,
    }

    #[async_trait::async_trait(?Send)]
    impl crate::services::RequestService for ProductionAuthorityVerifier {
        fn decode_request_body(
            &self,
            signed_body: &[u8],
        ) -> anyhow::Result<hyprstream_rpc::service::DecodedRequestBody> {
            Ok(hyprstream_rpc::service::DecodedRequestBody::opaque(
                signed_body.to_vec(),
            ))
        }

        async fn handle_request(
            &self,
            _ctx: &crate::services::EnvelopeContext,
            _body: &hyprstream_rpc::service::DecodedRequestBody,
        ) -> anyhow::Result<(Vec<u8>, Option<crate::services::Continuation>)> {
            self.accepted.store(true, Ordering::Release);
            Ok((Vec::new(), None))
        }

        fn name(&self) -> &str {
            "multiprocess-authority-verifier"
        }

        fn transport(&self) -> &hyprstream_rpc::transport::TransportConfig {
            &self.transport
        }

        fn signing_key(&self) -> hyprstream_rpc::prelude::SigningKey {
            self.signing_key.clone()
        }

        fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
            Some(self.key_source.clone())
        }

        fn expected_audience(&self) -> Option<&str> {
            Some("multiprocess")
        }

        fn require_cnf_binding(&self) -> bool {
            false
        }

        fn jwt_verify_policy(&self) -> hyprstream_rpc::crypto::CryptoPolicy {
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid
        }

        fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
            *self.error.lock() = Some(error.to_owned());
            Vec::new()
        }
    }

    async fn verify_through_rpc_endpoint(dir: &Path, token: String) -> anyhow::Result<()> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
        let rpc = RpcClientImpl::new(
            LocalSigner::new(SigningKey::from_bytes(&[0x75; 32])),
            LazyUdsTransport::new(dir.join("authority-rpc.sock")),
            Some(SigningKey::from_bytes(&[0x74; 32]).verifying_key()),
        )
        .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical)
        .with_default_jwt(token);
        rpc.call(b"verify-composite-authority".to_vec()).await?;
        Ok(())
    }

    fn policy_client_for_socket(
        dir: &Path,
    ) -> anyhow::Result<hyprstream_rpc_std::policy_client::PolicyClient> {
        policy_client_for_named_socket(dir, "authority-policy.sock")
    }

    fn policy_client_for_named_socket(
        dir: &Path,
        socket: &str,
    ) -> anyhow::Result<hyprstream_rpc_std::policy_client::PolicyClient> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
        let rpc = RpcClientImpl::new(
            LocalSigner::new(SigningKey::from_bytes(&[0x76; 32])),
            LazyUdsTransport::new(dir.join(socket)),
            Some(SigningKey::from_bytes(&[0x73; 32]).verifying_key()),
        )
        .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical);
        Ok(hyprstream_rpc_std::policy_client::PolicyClient::new(
            Arc::new(rpc),
        ))
    }

    fn authority_process_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("HYPRSTREAM_COMPOSITE_PRODUCTION_TEST_DIR")?;
        // Each authority role runs in a fresh process, so the dispatch fixture
        // must be installed here rather than in the parent orchestrator.
        crate::mac::install_explicit_test_dispatch_pep();
        Some(PathBuf::from(dir))
    }

    fn authority_ca_key() -> Arc<SigningKey> {
        Arc::new(SigningKey::from_bytes(&[0x5a; 32]))
    }

    fn install_hybrid_request_anchor(client_key: &SigningKey) -> anyhow::Result<()> {
        let mut store = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
        let pq_key = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(client_key);
        store.bind(
            client_key.verifying_key().to_bytes(),
            &ml_dsa::Keypair::verifying_key(&pq_key),
        );
        hyprstream_rpc::envelope::install_verify_config(
            hyprstream_rpc::envelope::EnvelopeVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
                pq_store: Some(Arc::new(store)),
            },
        )
    }

    async fn install_signing_authority(
        dir: &Path,
        subscribe: bool,
    ) -> anyhow::Result<(Arc<SigningKeyStore>, Arc<MlDsaSigningKeyStore>)> {
        let config = test_config();
        let ca = authority_ca_key();
        let ed = Arc::new(load_or_init_key_store(dir, &config));
        let pq = Arc::new(load_or_init_ml_dsa_key_store(dir, &config));
        *composite_ca_signing_key().write() = Some(Arc::clone(&ca));
        configure_composite_authority(dir);
        initialize_composite_key_set(dir, &ed, &pq, ca, 300).await?;
        if subscribe {
            start_composite_authority_subscription(
                dir.to_path_buf(),
                authority_ca_key().verifying_key(),
            )?;
        }
        Ok((ed, pq))
    }

    async fn install_verifying_authority(dir: &Path) -> anyhow::Result<()> {
        let config = test_config();
        let ca = authority_ca_key();
        let ed = load_or_init_key_store(dir, &config);
        let pq = load_or_init_ml_dsa_key_store(dir, &config);
        restore_composite_verifying_key_set(dir, &ed, &pq, ca.verifying_key()).await?;
        start_composite_authority_subscription(dir.to_path_buf(), ca.verifying_key())?;
        Ok(())
    }

    const PRODUCTION_AUTHORITY_BARRIER_TIMEOUT: std::time::Duration =
        std::time::Duration::from_secs(120);

    fn wait_path(path: &Path) {
        let deadline = std::time::Instant::now() + PRODUCTION_AUTHORITY_BARRIER_TIMEOUT;
        while !path.exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "timeout: {}",
                path.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    fn wait_for_target_authority(dir: &Path) -> CompositeCommit {
        let target_path = dir.join("target-authority.json");
        wait_path(&target_path);
        let target: CompositeCommit =
            serde_json::from_slice(&std::fs::read(target_path).unwrap()).unwrap();
        let deadline = std::time::Instant::now() + PRODUCTION_AUTHORITY_BARRIER_TIMEOUT;
        loop {
            let snapshot = hyprstream_rpc::auth::global_composite_key_set().snapshot();
            if snapshot.version() == target.version
                && snapshot.component_digest() == target.component_digest
            {
                return target;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "process did not converge to {} {}",
                target.version,
                target.component_digest
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    async fn verify_returned_tokens(dir: &Path, process: &str) -> anyhow::Result<()> {
        wait_path(&dir.join("tokens-ready"));
        for token_file in ["oauth-token", "policy-token"] {
            verify_through_rpc_endpoint(dir, std::fs::read_to_string(dir.join(token_file))?)
                .await?;
        }
        std::fs::write(dir.join(format!("verified-{process}")), b"1")?;
        Ok(())
    }

    fn wait_for_done(dir: &Path) {
        wait_path(&dir.join("done"));
    }

    /// Publish the completed target snapshot in one rename-visible operation.
    ///
    /// The child readers use file existence as their readiness signal, so a
    /// direct write can expose an empty file between create and completion.
    /// Keep this helper test-only: it models the completed fixture handoff
    /// without changing the production authority or rollback paths.
    fn publish_target_authority(dir: &Path, target: &CompositeCommit) {
        let bytes = serde_json::to_vec(target).unwrap();
        let mut temporary = tempfile::NamedTempFile::new_in(dir).unwrap();
        temporary.write_all(&bytes).unwrap();
        temporary.flush().unwrap();
        temporary
            .persist(dir.join("target-authority.json"))
            .unwrap();
    }

    async fn mutate_authority_for_failure(
        ed: &SigningKeyStore,
        pq: &MlDsaSigningKeyStore,
        persist_dir: Option<&Path>,
    ) {
        let now = chrono::Utc::now().timestamp();
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now,
            now + 1200,
        );
        let mut ed_slots = ed.0.write().await;
        ed_slots.drain = ed_slots.active.take();
        ed_slots.active = Some(new_ed.clone());
        if let Some(dir) = persist_dir {
            if let Some(drain) = &ed_slots.drain {
                persist_slot(dir, "drain", drain).unwrap();
            }
            persist_slot(dir, "active", &new_ed).unwrap();
        }
        drop(ed_slots);

        let (pq_key, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let new_pq = MlDsaKeySlot::new(pq_key, now, now + 1200);
        let mut pq_slots = pq.0.write().await;
        pq_slots.drain = pq_slots.active.take();
        pq_slots.active = Some(new_pq.clone());
        if let Some(dir) = persist_dir {
            if let Some(drain) = &pq_slots.drain {
                ml_dsa_rotation::persist_ml_dsa_slot(dir, "drain", drain).unwrap();
            }
            ml_dsa_rotation::persist_ml_dsa_slot(dir, "active", &new_pq).unwrap();
        }
    }

    /// Fixed classical primary key + base64url for the multiprocess issuance
    /// requests (they mint user at+jwt, which v16 binds to an authoritative
    /// primary suite).
    fn test_primary_key() -> [u8; 32] {
        [0x66; 32]
    }
    fn test_user_pub_key_b64() -> String {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        URL_SAFE_NO_PAD.encode(test_primary_key())
    }
    /// A permissive isolated primary resolver (any subject → the fixed classical
    /// key) — the fixture equivalent of WS-C's enrollment store for these
    /// multiprocess authority tests.
    fn test_permissive_primary_resolver(
    ) -> Arc<dyn crate::services::policy::PrimaryEnrollmentResolver> {
        struct R;
        impl crate::services::policy::PrimaryEnrollmentResolver for R {
            fn primary_group(
                &self,
                _principal: &str,
                _tenant: &str,
            ) -> Option<crate::services::policy::PrimaryGroup> {
                Some(crate::services::policy::PrimaryGroup {
                    suite_id: hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519.to_owned(),
                    ordered_component_keys: vec![test_primary_key().to_vec()],
                })
            }
        }
        Arc::new(R)
    }

    #[test]
    fn composite_oauth_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_signing_authority(&dir, true).await?;
                wait_path(&dir.join("authority-policy.sock"));
                // Mirror production `init_process_authority_stores`: a non-policy
                // process (this OAuth child) publishes RPC-client authority
                // stores that DELEGATE to the canonical policy-owned authority
                // over the policy socket — never private in-memory stores, which
                // the separate policy process (where `issue_token` actually runs)
                // could not see. A fresh sid registered here therefore crosses
                // RPC into the canonical registry that `handle_issue_token`
                // validates against, and issued-token verification checks the
                // same canonical revocation store. Fail-closed is preserved: an
                // unreachable authority answers not-active / revoked. Isolated to
                // this re-exec'd single-test subprocess — the main-binary
                // invocation of this fn early-returns above before reaching here.
                let _ = hyprstream_rpc::auth::set_global_session_registry(Arc::new(
                    crate::services::revocation::PolicyAuthoritySessionRegistry::new(
                        policy_client_for_socket(&dir)?,
                    ),
                ));
                let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                    crate::services::revocation::PolicyAuthorityRevocationStore::new(
                        policy_client_for_socket(&dir)?,
                    ),
                ));
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
                let address = listener.local_addr()?;
                let mut config = test_config();
                config.external_url = Some(format!("http://{address}"));
                config.token_ttl_seconds = 60;
                let unused: Arc<dyn hyprstream_rpc::transport::rpc_session::IrohRequestProcessor> =
                    Arc::new(hyprstream_rpc::transport::rpc_session::from_fn(|_| async {
                        Ok(bytes::Bytes::new())
                    }));
                hyprstream_rpc::dial::register_inproc("multiprocess-unused-discovery", &unused);
                let discovery = hyprstream_rpc_std::discovery_client::DiscoveryClient::for_local_endpoint_bootstrap(
                    "inproc://multiprocess-unused-discovery",
                    SigningKey::from_bytes(&[0x76; 32]),
                    SigningKey::from_bytes(&[0x77; 32]).verifying_key(),
                    None,
                )?;
                let user_store = Arc::new(crate::auth::RocksDbUserStore::open(
                    &dir.join("oauth-users"),
                )?);
                crate::auth::UserStore::register(user_store.as_ref(), "multiprocess-oauth").await?;
                crate::auth::UserStore::set_profile(
                    user_store.as_ref(),
                    "multiprocess-oauth",
                    crate::auth::UserProfilePatch {
                        atproto_did: Some(Some(
                            "did:web:multiprocess-oauth.example.test".to_owned(),
                        )),
                        ..Default::default()
                    },
                )
                .await?;
                let hosted_store = hosted_account_store("multiprocess-oauth", "example.test")?;
                // The production account store warms its signed hosted-DID
                // index during startup.  This isolated multi-process fixture
                // constructs the store directly, so warm the same index before
                // the OAuth child accepts its first authorization-code exchange.
                hosted_store
                    .refresh_hosted_did_index(&hyprstream_rpc::Subject::new(
                        hyprstream_pds_service::OAUTH_ACCOUNT_RESOLVER_SUBJECT,
                    ))
                    .await?;
                let state = Arc::new(
                    crate::services::oauth::state::OAuthState::new(
                        &config,
                        policy_client_for_socket(&dir)?,
                        discovery,
                        authority_ca_key().verifying_key().to_bytes(),
                    )
                    .with_user_store(crate::auth::ProductionUserStore::for_test(user_store))
                    .with_hosted_account_zone(crate::account::AccountZone::new(
                        "example.test",
                    )?)
                    .with_hosted_account_store(hosted_store),
                );
                let verifier = "multiprocess-pkce-verifier";
                let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
                for code in [
                    "multiprocess-code",
                    "post-failure-code",
                    "timeout-code",
                    "post-crash-code",
                ] {
                    state.pending_codes.write().await.insert(
                        code.to_owned(),
                        crate::services::oauth::state::PendingAuthCode {
                            code: code.to_owned(),
                            client_id: "multiprocess-client".to_owned(),
                            redirect_uri: "https://client.test/callback".to_owned(),
                            code_challenge: challenge.clone(),
                            scopes: vec!["read".to_owned()],
                            resource: Some("multiprocess".to_owned()),
                            oidc_nonce: None,
                            created_at: std::time::Instant::now(),
                            expires_at: std::time::Instant::now()
                                + std::time::Duration::from_secs(60),
                            username: "multiprocess-oauth".to_owned(),
                            verifying_key: None,
                            dpop_jkt: None,
                            client_assertion_jkt: None,
                        },
                    );
                }
                let app = crate::services::oauth::create_app(
                    state,
                    &crate::config::CorsConfig::default(),
                );
                let server = tokio::spawn(async move { axum::serve(listener, app).await });
                std::fs::write(dir.join("oauth-http-url"), format!("http://{address}"))?;
                std::fs::write(dir.join("ready-oauth"), b"1")?;
                wait_for_target_authority(&dir);
                std::fs::write(dir.join("converged-oauth"), b"1")?;
                verify_returned_tokens(&dir, "oauth").await?;
                wait_for_done(&dir);
                server.abort();
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_policy_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_signing_authority(&dir, true).await?;
                // Canonical policy-owned authority stores (v16 §3.3): the policy
                // process owns the single session registry and credential
                // revocation store. Every other process delegates to it over RPC
                // (see the OAuth child's PolicyAuthority* stores), so a sid the
                // OAuth child registers crosses RPC into THIS registry, which
                // `handle_issue_token` then validates against. Published before
                // the PolicyService serves. Isolated to this re-exec'd
                // single-test subprocess.
                let _ = hyprstream_rpc::auth::set_global_session_registry(Arc::new(
                    hyprstream_rpc::auth::InMemorySessionRegistry::new(),
                ));
                let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                    hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
                ));
                let db = dir.join(format!("policy-db-{}", std::process::id()));
                std::fs::create_dir_all(&db)?;
                let policy_manager = Arc::new(
                    crate::auth::PolicyManager::new(&db.join(".registry/policies")).await?,
                );
                let git2db = Arc::new(tokio::sync::RwLock::new(git2db::Git2DB::open(&db).await?));
                let client_key = SigningKey::from_bytes(&[0x76; 32]);
                install_hybrid_request_anchor(&client_key)?;
                hyprstream_service::global_trust_store().insert(
                    client_key.verifying_key(),
                    hyprstream_service::Attestation {
                        scopes: std::iter::once("oauth".to_owned()).collect(),
                        subject: None,
                        jwt: None,
                        expires_at: chrono::Utc::now().timestamp() + 300,
                        attested_by: None,
                    },
                );
                let service = crate::services::PolicyService::new(
                    policy_manager,
                    Arc::new(SigningKey::from_bytes(&[0x73; 32])),
                    crate::config::TokenConfig::default(),
                    git2db,
                    hyprstream_rpc::transport::TransportConfig::ipc(
                        dir.join("authority-policy.sock"),
                    ),
                )
                .with_default_audience("multiprocess".to_owned())
                // Production-equivalent key source: signing resolves the
                // composite authority through the configured JwtKeySource (the
                // `ClusterKeySource` ledger defaults to the process-global
                // authority this subprocess configures).
                .with_jwt_key_source(Arc::new(hyprstream_rpc::auth::ClusterKeySource::new(
                    SigningKey::from_bytes(&[0x73; 32]).verifying_key(),
                    "multiprocess".to_owned(),
                )))
                .with_primary_enrollment_resolver(test_permissive_primary_resolver());
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join("authority-policy.sock"));
                std::fs::write(dir.join("ready-policy"), b"1")?;
                wait_for_target_authority(&dir);
                std::fs::write(dir.join("converged-policy"), b"1")?;
                let token = policy_client_for_socket(&dir)?
                    .issue_token(&hyprstream_rpc_std::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("multiprocess-policy".to_owned()),
                        user_pub_key: Some(test_user_pub_key_b64()),
                        dpop_jkt: None,
                        issuer: None,
                        tenant: None,
                        require_clearance: false,
                        session_id: None,
                        issuance_profile:
                            hyprstream_rpc_std::policy_client::IssueTokenProfile::Rfc8693,
                        client_id: Some("hyprstream-oauth-client-1".to_owned()),
                    })
                    .await?
                    .token;
                std::fs::write(dir.join("policy-token"), token)?;
                verify_returned_tokens(&dir, "policy").await?;
                wait_for_done(&dir);
                shutdown.notify_waiters();
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_jwks_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_verifying_authority(&dir).await?;
                std::fs::write(dir.join("ready-jwks"), b"1")?;
                wait_for_target_authority(&dir);
                std::fs::write(dir.join("converged-jwks"), b"1")?;
                wait_path(&dir.join("jwks-response.json"));
                verify_returned_tokens(&dir, "jwks").await?;
                anyhow::Ok(())
            })
            .unwrap();
        wait_for_done(&dir);
    }

    #[test]
    fn composite_rpc_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_verifying_authority(&dir).await?;
                install_hybrid_request_anchor(&SigningKey::from_bytes(&[0x75; 32]))?;
                let accepted = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let service = ProductionAuthorityVerifier {
                    transport: hyprstream_rpc::transport::TransportConfig::ipc(
                        dir.join("authority-rpc.sock"),
                    ),
                    signing_key: SigningKey::from_bytes(&[0x74; 32]),
                    key_source: Arc::new(hyprstream_rpc::auth::ClusterKeySource::new(
                        authority_ca_key().verifying_key(),
                        "https://multiprocess.test".to_owned(),
                    )),
                    accepted,
                    error: Arc::new(parking_lot::Mutex::new(None)),
                };
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join("authority-rpc.sock"));
                std::fs::write(dir.join("ready-rpc"), b"1")?;
                wait_for_target_authority(&dir);
                std::fs::write(dir.join("converged-rpc"), b"1")?;
                verify_returned_tokens(&dir, "rpc").await?;
                wait_for_done(&dir);
                shutdown.notify_waiters();
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_stale_policy_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_signing_authority(&dir, false).await?;
                std::fs::write(dir.join("ready-stale-policy"), b"1")?;
                wait_path(&dir.join("stale-attempt-gate"));
                let db = dir.join(format!("stale-policy-db-{}", std::process::id()));
                std::fs::create_dir_all(&db)?;
                let policy_manager = Arc::new(
                    crate::auth::PolicyManager::new(&db.join(".registry/policies")).await?,
                );
                let git2db = Arc::new(tokio::sync::RwLock::new(git2db::Git2DB::open(&db).await?));
                let client_key = SigningKey::from_bytes(&[0x76; 32]);
                install_hybrid_request_anchor(&client_key)?;
                hyprstream_service::global_trust_store().insert(
                    client_key.verifying_key(),
                    hyprstream_service::Attestation {
                        scopes: std::iter::once("oauth".to_owned()).collect(),
                        subject: None,
                        jwt: None,
                        expires_at: chrono::Utc::now().timestamp() + 300,
                        attested_by: None,
                    },
                );
                let service = crate::services::PolicyService::new(
                    policy_manager,
                    Arc::new(SigningKey::from_bytes(&[0x73; 32])),
                    crate::config::TokenConfig::default(),
                    git2db,
                    hyprstream_rpc::transport::TransportConfig::ipc(dir.join("stale-policy.sock")),
                )
                .with_default_audience("multiprocess".to_owned())
                .with_jwt_key_source(Arc::new(hyprstream_rpc::auth::ClusterKeySource::new(
                    SigningKey::from_bytes(&[0x73; 32]).verifying_key(),
                    "multiprocess".to_owned(),
                )))
                .with_primary_enrollment_resolver(test_permissive_primary_resolver());
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join("stale-policy.sock"));
                let result = policy_client_for_named_socket(&dir, "stale-policy.sock")?
                    .issue_token(&hyprstream_rpc_std::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("stale-policy".to_owned()),
                        user_pub_key: Some(test_user_pub_key_b64()),
                        dpop_jkt: None,
                        issuer: None,
                        tenant: None,
                        require_clearance: false,
                        session_id: None,
                        issuance_profile:
                            hyprstream_rpc_std::policy_client::IssueTokenProfile::Rfc8693,
                        client_id: Some("hyprstream-oauth-client-1".to_owned()),
                    })
                    .await;
                anyhow::ensure!(result.is_err(), "stale PolicyService minted a token");
                shutdown.notify_waiters();
                std::fs::write(dir.join("stale-policy-refused"), b"1")?;
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_stale_writer_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                let (ed, pq) = install_signing_authority(&dir, false).await?;
                let expected = hyprstream_rpc::auth::global_composite_key_set()
                    .snapshot()
                    .component_digest()
                    .to_owned();
                std::fs::write(dir.join("ready-stale-writer"), b"1")?;
                wait_path(&dir.join("stale-attempt-gate"));
                let result =
                    refresh_composite_key_set(&dir, &ed, &pq, authority_ca_key(), 300, &expected)
                        .await;
                let error = result.expect_err("stale writer publication unexpectedly succeeded");
                anyhow::ensure!(
                    error
                        .to_string()
                        .contains("stale composite component authority"),
                    "unexpected stale writer error: {error:#}"
                );
                std::fs::write(dir.join("stale-writer-refused"), error.to_string())?;
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_crashing_writer_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                let config = test_config();
                let ed = load_or_init_key_store(&dir, &config);
                let pq = load_or_init_ml_dsa_key_store(&dir, &config);
                mutate_authority_for_failure(&ed, &pq, Some(&dir)).await;
                let expected = std::fs::read_to_string(dir.join("crash-expected-digest"))?;
                refresh_composite_key_set(&dir, &ed, &pq, authority_ca_key(), 300, &expected).await
            })
            .unwrap();
        panic!("crash mutation returned instead of exiting after stage");
    }

    #[test]
    fn composite_timing_out_writer_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                let config = test_config();
                let ed = load_or_init_key_store(&dir, &config);
                let pq = load_or_init_ml_dsa_key_store(&dir, &config);
                mutate_authority_for_failure(&ed, &pq, None).await;
                let expected = std::fs::read_to_string(dir.join("timeout-expected-digest"))?;
                let error =
                    refresh_composite_key_set(&dir, &ed, &pq, authority_ca_key(), 300, &expected)
                        .await
                        .expect_err("unacknowledged composite generation committed");
                anyhow::ensure!(
                    error.to_string().contains("was not acknowledged"),
                    "unexpected acknowledgement-timeout error: {error:#}"
                );
                std::fs::write(dir.join("timeout-writer-refused"), error.to_string())?;
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_restart_rpc_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_verifying_authority(&dir).await?;
                verify_through_rpc_endpoint(
                    &dir,
                    std::fs::read_to_string(dir.join("old-oauth-token"))?,
                )
                .await?;
                std::fs::write(dir.join("restart-drain-verified"), b"1")?;
                anyhow::Ok(())
            })
            .unwrap();
    }

    #[test]
    fn composite_restart_policy_production_process() {
        let Some(dir) = authority_process_dir() else {
            return;
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(async {
                install_signing_authority(&dir, false).await?;
                let db = dir.join(format!("restart-policy-db-{}", std::process::id()));
                std::fs::create_dir_all(&db)?;
                let policy_manager = Arc::new(
                    crate::auth::PolicyManager::new(&db.join(".registry/policies")).await?,
                );
                let git2db = Arc::new(tokio::sync::RwLock::new(git2db::Git2DB::open(&db).await?));
                let client_key = SigningKey::from_bytes(&[0x76; 32]);
                install_hybrid_request_anchor(&client_key)?;
                hyprstream_service::global_trust_store().insert(
                    client_key.verifying_key(),
                    hyprstream_service::Attestation {
                        scopes: std::iter::once("oauth".to_owned()).collect(),
                        subject: None,
                        jwt: None,
                        expires_at: chrono::Utc::now().timestamp() + 300,
                        attested_by: None,
                    },
                );
                let socket = "restart-policy.sock";
                let service = crate::services::PolicyService::new(
                    policy_manager,
                    Arc::new(SigningKey::from_bytes(&[0x73; 32])),
                    crate::config::TokenConfig::default(),
                    git2db,
                    hyprstream_rpc::transport::TransportConfig::ipc(dir.join(socket)),
                )
                .with_default_audience("multiprocess".to_owned())
                .with_jwt_key_source(Arc::new(hyprstream_rpc::auth::ClusterKeySource::new(
                    SigningKey::from_bytes(&[0x73; 32]).verifying_key(),
                    "multiprocess".to_owned(),
                )))
                .with_primary_enrollment_resolver(test_permissive_primary_resolver());
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join(socket));
                let token = policy_client_for_named_socket(&dir, socket)?
                    .issue_token(&hyprstream_rpc_std::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("restart-policy".to_owned()),
                        user_pub_key: Some(test_user_pub_key_b64()),
                        dpop_jkt: None,
                        issuer: None,
                        tenant: None,
                        require_clearance: false,
                        session_id: None,
                        issuance_profile:
                            hyprstream_rpc_std::policy_client::IssueTokenProfile::Rfc8693,
                        client_id: Some("hyprstream-oauth-client-1".to_owned()),
                    })
                    .await?
                    .token;
                verify_through_rpc_endpoint(&dir, token).await?;
                std::fs::write(dir.join("restart-policy-minted"), b"1")?;
                shutdown.notify_waiters();
                anyhow::Ok(())
            })
            .unwrap();
    }

    fn spawn_production_authority_process(dir: &Path, test_name: &str) -> std::process::Child {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test_name, "--nocapture"])
            .env("HYPRSTREAM_COMPOSITE_PRODUCTION_TEST_DIR", dir);
        if test_name.ends_with("composite_crashing_writer_process") {
            command.env("HYPRSTREAM_COMPOSITE_CRASH_AFTER_STAGE", "1");
        }
        if test_name.ends_with("composite_timing_out_writer_process") {
            command.env(
                "HYPRSTREAM_COMPOSITE_STAGE_MARKER",
                dir.join("timeout-staged"),
            );
        }
        command.spawn().unwrap()
    }

    #[test]
    fn composite_authority_production_paths_reject_semantic_rollback() {
        const ORCHESTRATOR_DIR: &str = "HYPRSTREAM_COMPOSITE_ORCHESTRATOR_TEST_DIR";
        if std::env::var_os(ORCHESTRATOR_DIR).is_none() {
            // UDS socket paths are limited to ~108 bytes; keep the parent
            // short and unique even when the worktree has a long absolute path.
            let dir = TempDir::new().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::composite_authority_production_paths_reject_semantic_rollback",
                    "--nocapture",
                ])
                .env(ORCHESTRATOR_DIR, dir.path())
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        struct AuthorityTestDir(PathBuf);
        impl AuthorityTestDir {
            fn path(&self) -> &Path {
                &self.0
            }
        }
        let dir = AuthorityTestDir(PathBuf::from(std::env::var_os(ORCHESTRATOR_DIR).unwrap()));
        let config = test_config();
        let now = chrono::Utc::now().timestamp();
        let old_ed = KeySlot::new(SigningKey::from_bytes(&[0x11; 32]), now - 10, now + 600);
        let (old_pq_key, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let old_pq = MlDsaKeySlot::new(old_pq_key, now - 10, now + 600);
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &old_pq).unwrap();
        let ca = authority_ca_key();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let old_ed_store = load_or_init_key_store(dir.path(), &config);
        let old_pq_store = load_or_init_ml_dsa_key_store(dir.path(), &config);
        runtime
            .block_on(initialize_composite_key_set(
                dir.path(),
                &old_ed_store,
                &old_pq_store,
                Arc::clone(&ca),
                300,
            ))
            .unwrap();
        let old_digest = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .component_digest()
            .to_owned();

        let process_tests = [
            (
                "oauth",
                "auth::key_rotation::tests::composite_oauth_production_process",
            ),
            (
                "policy",
                "auth::key_rotation::tests::composite_policy_production_process",
            ),
            (
                "jwks",
                "auth::key_rotation::tests::composite_jwks_production_process",
            ),
            (
                "rpc",
                "auth::key_rotation::tests::composite_rpc_production_process",
            ),
        ];
        let mut services: Vec<_> = process_tests
            .iter()
            .map(|(_, test)| spawn_production_authority_process(dir.path(), test))
            .collect();
        let mut stale_policy = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_stale_policy_production_process",
        );
        let mut stale_writer = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_stale_writer_process",
        );
        for process in [
            "oauth",
            "policy",
            "jwks",
            "rpc",
            "stale-policy",
            "stale-writer",
        ] {
            wait_path(&dir.path().join(format!("ready-{process}")));
        }

        // Capture a pre-rotation token through the booted OAuth HTTP endpoint;
        // it is used later to prove the committed drain remains usable.
        let oauth_url = std::fs::read_to_string(dir.path().join("oauth-http-url")).unwrap();
        let old_token = runtime
            .block_on(async {
                policy_client_for_socket(dir.path())?
                    .issue_token(&hyprstream_rpc_std::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("pre-rotation-policy".to_owned()),
                        user_pub_key: Some(test_user_pub_key_b64()),
                        dpop_jkt: None,
                        issuer: None,
                        tenant: None,
                        require_clearance: false,
                        session_id: None,
                        issuance_profile:
                            hyprstream_rpc_std::policy_client::IssueTokenProfile::Rfc8693,
                        client_id: Some("hyprstream-oauth-client-1".to_owned()),
                    })
                    .await?;
                let client = reqwest::Client::new();
                let authorization_code_form = [
                    ("grant_type", "authorization_code"),
                    ("client_id", "multiprocess-client"),
                    ("code", "multiprocess-code"),
                    ("redirect_uri", "https://client.test/callback"),
                    ("code_verifier", "multiprocess-pkce-verifier"),
                ];
                let response = client
                    .post(format!("{oauth_url}/oauth/token"))
                    .form(&authorization_code_form)
                    .send()
                    .await?;
                if !response.status().is_success() {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    anyhow::bail!("pre-rotation OAuth endpoint returned {status}: {body}");
                }
                let body: serde_json::Value = response.json().await?;
                let token = body
                    .get("access_token")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        anyhow::anyhow!("pre-rotation OAuth response omitted access_token")
                    })?;
                let replay = client
                    .post(format!("{oauth_url}/oauth/token"))
                    .form(&authorization_code_form)
                    .send()
                    .await?;
                anyhow::ensure!(
                    replay.status() == reqwest::StatusCode::BAD_REQUEST,
                    "authorization-code replay returned {}, expected 400",
                    replay.status()
                );
                let replay_body: serde_json::Value = replay.json().await?;
                anyhow::ensure!(
                    replay_body.get("error").and_then(serde_json::Value::as_str)
                        == Some("invalid_grant"),
                    "authorization-code replay was not rejected as invalid_grant: {replay_body}"
                );
                anyhow::Ok(token)
            })
            .unwrap();
        std::fs::write(dir.path().join("old-oauth-token"), old_token).unwrap();

        persist_slot(dir.path(), "drain", &old_ed).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "drain", &old_pq).unwrap();
        let new_ed = KeySlot::new(SigningKey::from_bytes(&[0x22; 32]), now, now + 1200);
        let (new_pq_key, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let new_pq = MlDsaKeySlot::new(new_pq_key, now, now + 1200);
        persist_slot(dir.path(), "active", &new_ed).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &new_pq).unwrap();
        let new_ed_store = load_or_init_key_store(dir.path(), &config);
        let new_pq_store = load_or_init_ml_dsa_key_store(dir.path(), &config);
        runtime
            .block_on(refresh_composite_key_set(
                dir.path(),
                &new_ed_store,
                &new_pq_store,
                Arc::clone(&ca),
                300,
                &old_digest,
            ))
            .unwrap();
        let committed: CompositeCommit =
            serde_json::from_slice(&std::fs::read(composite_committed_path(dir.path())).unwrap())
                .unwrap();
        let ledger_after_rotation: CompositeLedger =
            serde_json::from_slice(&std::fs::read(composite_ledger_path(dir.path())).unwrap())
                .unwrap();
        assert_eq!(committed.version, ledger_after_rotation.version);
        assert_eq!(
            committed.component_digest,
            ledger_after_rotation.component_digest
        );
        assert_ne!(committed.component_digest, old_digest);

        // Recreate a round-three committed-B layout (marker + matching mutable
        // ledger, no immutable B). The production initializer must migrate B
        // while holding the ledger lock, before either failure writer can
        // replace the mutable file with pending C.
        let committed_immutable = composite_committed_ledger_path(dir.path(), &committed);
        std::fs::remove_file(&committed_immutable).unwrap();
        runtime
            .block_on(initialize_composite_key_set(
                dir.path(),
                &new_ed_store,
                &new_pq_store,
                Arc::clone(&ca),
                300,
            ))
            .unwrap();
        let migrated: CompositeLedger =
            serde_json::from_slice(&std::fs::read(&committed_immutable).unwrap()).unwrap();
        assert_eq!(migrated.version, committed.version);
        assert_eq!(migrated.component_digest, committed.component_digest);
        assert_eq!(
            std::fs::read(composite_ledger_path(dir.path())).unwrap(),
            serde_json::to_vec(&ledger_after_rotation).unwrap(),
            "legacy migration changed mutable B before pending C was staged"
        );
        publish_target_authority(dir.path(), &committed);
        for process in ["oauth", "policy", "jwks", "rpc"] {
            wait_path(&dir.path().join(format!("converged-{process}")));
        }

        let oauth_url = std::fs::read_to_string(dir.path().join("oauth-http-url")).unwrap();
        let (oauth_token, jwks) = runtime
            .block_on(async {
                let client = reqwest::Client::new();
                let response = client
                    .post(format!("{oauth_url}/oauth/token"))
                    .form(&[
                        ("grant_type", "authorization_code"),
                        ("client_id", "multiprocess-client"),
                        ("code", "post-failure-code"),
                        ("redirect_uri", "https://client.test/callback"),
                        ("code_verifier", "multiprocess-pkce-verifier"),
                    ])
                    .send()
                    .await?;
                anyhow::ensure!(
                    response.status().is_success(),
                    "ordinary OAuth token endpoint returned {}: {}",
                    response.status(),
                    response.text().await?
                );
                let body: serde_json::Value = response.json().await?;
                let token = body
                    .get("access_token")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("OAuth response omitted access_token"))?
                    .to_owned();
                let jwks_response = client.get(format!("{oauth_url}/oauth/jwks")).send().await?;
                anyhow::ensure!(
                    jwks_response.status().is_success(),
                    "HTTP JWKS endpoint returned {}",
                    jwks_response.status()
                );
                let jwks = jwks_response.json::<serde_json::Value>().await?;
                anyhow::Ok((token, jwks))
            })
            .unwrap();
        std::fs::write(dir.path().join("oauth-token"), oauth_token).unwrap();
        std::fs::write(
            dir.path().join("jwks-response.json"),
            serde_json::to_vec(&jwks).unwrap(),
        )
        .unwrap();

        wait_path(&dir.path().join("oauth-token"));
        wait_path(&dir.path().join("policy-token"));
        std::fs::write(dir.path().join("tokens-ready"), b"1").unwrap();
        for process in ["oauth", "policy", "jwks", "rpc"] {
            wait_path(&dir.path().join(format!("verified-{process}")));
        }
        wait_path(&dir.path().join("jwks-response.json"));
        let jwks: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("jwks-response.json")).unwrap())
                .unwrap();
        assert_eq!(jwks["composite_version"].as_u64(), Some(committed.version));
        assert_eq!(
            jwks["composite_component_digest"].as_str(),
            Some(committed.component_digest.as_str())
        );
        let active_kids: Vec<&str> = ledger_after_rotation
            .pairs
            .iter()
            .filter(|pair| pair.state == "active")
            .map(|pair| pair.kid.as_str())
            .collect();
        for kid in active_kids {
            assert!(jwks["keys"]
                .as_array()
                .unwrap()
                .iter()
                .any(|key| { key.get("kid").and_then(serde_json::Value::as_str) == Some(kid) }));
        }

        // Hold a writer at the acknowledgement barrier with an in-memory-only
        // component generation. Live services must keep serving and minting
        // exclusively from the previous committed snapshot throughout timeout.
        std::fs::write(
            dir.path().join("timeout-expected-digest"),
            &committed.component_digest,
        )
        .unwrap();
        let mut timing_out_writer = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_timing_out_writer_process",
        );
        wait_path(&dir.path().join("timeout-staged"));
        let (timeout_oauth_token, timeout_policy_token, timeout_jwks) = runtime
            .block_on(async {
                let client = reqwest::Client::new();
                let response = client
                    .post(format!("{oauth_url}/oauth/token"))
                    .form(&[
                        ("grant_type", "authorization_code"),
                        ("client_id", "multiprocess-client"),
                        ("code", "timeout-code"),
                        ("redirect_uri", "https://client.test/callback"),
                        ("code_verifier", "multiprocess-pkce-verifier"),
                    ])
                    .send()
                    .await?;
                anyhow::ensure!(
                    response.status().is_success(),
                    "OAuth mint failed while publication was pending: {}",
                    response.text().await?
                );
                let oauth = response.json::<serde_json::Value>().await?["access_token"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("pending-timeout OAuth token missing"))?
                    .to_owned();
                let policy = policy_client_for_socket(dir.path())?
                    .issue_token(&hyprstream_rpc_std::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("timeout-policy".to_owned()),
                        user_pub_key: Some(test_user_pub_key_b64()),
                        dpop_jkt: None,
                        issuer: None,
                        tenant: None,
                        require_clearance: false,
                        session_id: None,
                        issuance_profile:
                            hyprstream_rpc_std::policy_client::IssueTokenProfile::Rfc8693,
                        client_id: Some("hyprstream-oauth-client-1".to_owned()),
                    })
                    .await?
                    .token;
                let jwks = client
                    .get(format!("{oauth_url}/oauth/jwks"))
                    .send()
                    .await?
                    .json::<serde_json::Value>()
                    .await?;
                verify_through_rpc_endpoint(dir.path(), oauth.clone()).await?;
                verify_through_rpc_endpoint(dir.path(), policy.clone()).await?;
                anyhow::Ok((oauth, policy, jwks))
            })
            .unwrap();
        assert_eq!(
            timeout_jwks["composite_version"].as_u64(),
            Some(committed.version)
        );
        assert_eq!(
            timeout_jwks["composite_component_digest"].as_str(),
            Some(committed.component_digest.as_str())
        );
        wait_path(&dir.path().join("timeout-writer-refused"));
        assert!(timing_out_writer.wait().unwrap().success());
        let committed_after_timeout: CompositeCommit =
            serde_json::from_slice(&std::fs::read(composite_committed_path(dir.path())).unwrap())
                .unwrap();
        assert_eq!(committed_after_timeout.version, committed.version);
        assert_eq!(
            committed_after_timeout.component_digest,
            committed.component_digest
        );

        // A distinct writer now dies immediately after persisting and staging
        // another generation. Pending-only pairs must never enter JWKS,
        // verification, or mint authority, and restart must select the marker.
        std::fs::write(
            dir.path().join("crash-expected-digest"),
            &committed.component_digest,
        )
        .unwrap();
        let mut crashing_writer = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_crashing_writer_process",
        );
        assert_eq!(crashing_writer.wait().unwrap().code(), Some(86));
        let pending_after_crash: CompositeLedger =
            serde_json::from_slice(&std::fs::read(composite_ledger_path(dir.path())).unwrap())
                .unwrap();
        assert_ne!(
            pending_after_crash.component_digest,
            committed.component_digest
        );
        let (post_crash_token, post_crash_jwks) = runtime
            .block_on(async {
                let client = reqwest::Client::new();
                let response = client
                    .post(format!("{oauth_url}/oauth/token"))
                    .form(&[
                        ("grant_type", "authorization_code"),
                        ("client_id", "multiprocess-client"),
                        ("code", "post-crash-code"),
                        ("redirect_uri", "https://client.test/callback"),
                        ("code_verifier", "multiprocess-pkce-verifier"),
                    ])
                    .send()
                    .await?;
                anyhow::ensure!(
                    response.status().is_success(),
                    "OAuth mint failed after writer crash: {}",
                    response.text().await?
                );
                let token = response.json::<serde_json::Value>().await?["access_token"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("post-crash OAuth token missing"))?
                    .to_owned();
                let jwks = client
                    .get(format!("{oauth_url}/oauth/jwks"))
                    .send()
                    .await?
                    .json::<serde_json::Value>()
                    .await?;
                verify_through_rpc_endpoint(dir.path(), token.clone()).await?;
                anyhow::Ok((token, jwks))
            })
            .unwrap();
        assert_eq!(
            post_crash_jwks["composite_version"].as_u64(),
            Some(committed.version)
        );
        assert_eq!(
            post_crash_jwks["composite_component_digest"].as_str(),
            Some(committed.component_digest.as_str())
        );
        for pending_only in pending_after_crash.pairs.iter().filter(|candidate| {
            !ledger_after_rotation
                .pairs
                .iter()
                .any(|committed_pair| committed_pair.kid == candidate.kid)
        }) {
            assert!(!post_crash_jwks["keys"]
                .as_array()
                .unwrap()
                .iter()
                .any(|key| key["kid"].as_str() == Some(pending_only.kid.as_str())));
        }
        let _ = (timeout_oauth_token, timeout_policy_token, post_crash_token);

        let mut restart = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_restart_rpc_production_process",
        );
        wait_path(&dir.path().join("restart-drain-verified"));
        assert!(restart.wait().unwrap().success());

        let mut restart_policy = spawn_production_authority_process(
            dir.path(),
            "auth::key_rotation::tests::composite_restart_policy_production_process",
        );
        wait_path(&dir.path().join("restart-policy-minted"));
        assert!(restart_policy.wait().unwrap().success());

        std::fs::write(dir.path().join("stale-attempt-gate"), b"1").unwrap();
        wait_path(&dir.path().join("stale-policy-refused"));
        wait_path(&dir.path().join("stale-writer-refused"));
        assert!(stale_policy.wait().unwrap().success());
        assert!(stale_writer.wait().unwrap().success());
        let ledger_after_stale: CompositeLedger =
            serde_json::from_slice(&std::fs::read(composite_ledger_path(dir.path())).unwrap())
                .unwrap();
        assert_eq!(ledger_after_stale.version, pending_after_crash.version);
        assert_eq!(
            ledger_after_stale.component_digest,
            pending_after_crash.component_digest
        );
        assert_eq!(
            ledger_after_stale.pairs.len(),
            pending_after_crash.pairs.len()
        );
        for pair in &pending_after_crash.pairs {
            assert!(ledger_after_stale.pairs.iter().any(|candidate| {
                candidate.kid == pair.kid
                    && candidate.role == pair.role
                    && candidate.state == pair.state
            }));
        }
        let committed_after_stale: CompositeCommit =
            serde_json::from_slice(&std::fs::read(composite_committed_path(dir.path())).unwrap())
                .unwrap();
        assert_eq!(committed_after_stale.version, committed.version);
        assert_eq!(
            committed_after_stale.component_digest,
            committed.component_digest
        );
        runtime
            .block_on(async {
                for token_file in ["oauth-token", "policy-token"] {
                    verify_through_rpc_endpoint(
                        dir.path(),
                        std::fs::read_to_string(dir.path().join(token_file))?,
                    )
                    .await?;
                }
                anyhow::Ok(())
            })
            .unwrap();

        std::fs::write(dir.path().join("done"), b"1").unwrap();
        for service in &mut services {
            assert!(service.wait().unwrap().success());
        }
    }

    #[test]
    fn load_or_init_creates_active_key_on_first_boot() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store = load_or_init_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let vk = rt.block_on(store.active_verifying_key_bytes());
        assert!(
            vk.is_some(),
            "active key should be present after first boot"
        );
    }

    #[test]
    fn committed_marker_with_missing_slots_does_not_bootstrap_replacement_keys() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let marker = composite_committed_path(dir.path());
        // The guard is intentionally based on marker presence, not successful
        // parsing: malformed committed metadata is not permission to mint a
        // different authority.
        std::fs::write(&marker, b"committed but unreadable").unwrap();

        let ed = load_or_init_key_store(dir.path(), &config);
        let pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(ed.active_verifying_key_bytes()).is_none());
        assert!(rt.block_on(pq.active_key()).is_none());
        assert_eq!(std::fs::read(&marker).unwrap(), b"committed but unreadable");
        for name in [
            "jwt-signing-key.active",
            "jwt-signing-key.active.meta",
            "jwt-signing-key.lead",
            "jwt-signing-key.lead.meta",
            "ml-dsa-signing-key.active",
            "ml-dsa-signing-key.active.meta",
            "ml-dsa-signing-key.lead",
            "ml-dsa-signing-key.lead.meta",
        ] {
            assert!(
                !dir.path().join(name).exists(),
                "committed store unexpectedly created {name}"
            );
        }
    }

    #[tokio::test]
    async fn committed_marker_does_not_generate_missing_lead_slots() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();
        let marker = composite_committed_path(dir.path());
        std::fs::write(&marker, b"committed").unwrap();
        persist_slot(
            dir.path(),
            "active",
            &KeySlot::new(SigningKey::from_bytes(&[0x71; 32]), now - 60, now + 1),
        )
        .unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(
            dir.path(),
            "active",
            &ml_dsa_rotation::generate_ml_dsa_slot(now - 60, now + 1),
        )
        .unwrap();

        let ed = load_or_init_key_store(dir.path(), &config);
        let pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
        assert!(ed.0.read().await.active.is_some());
        assert!(ed.0.read().await.lead.is_none());
        assert!(pq.0.read().await.active.is_some());
        assert!(pq.0.read().await.lead.is_none());
        for name in [
            "jwt-signing-key.lead",
            "jwt-signing-key.lead.meta",
            "ml-dsa-signing-key.lead",
            "ml-dsa-signing-key.lead.meta",
        ] {
            assert!(
                !dir.path().join(name).exists(),
                "committed store unexpectedly created {name}"
            );
        }
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
        assert_eq!(
            new_active_vk, lead_vk,
            "lead must become active after promotion"
        );

        // Old active must now be drain
        let slots = store.0.read().await;
        assert!(slots.drain.is_some(), "old active must become drain");
        assert!(
            slots.lead.is_none(),
            "lead slot must be cleared after promotion"
        );
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
        assert!(
            slots.drain.is_none(),
            "drain must be removed after drain window closes"
        );
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
        assert!(
            slots.lead.is_some(),
            "lead must be generated when active is within lead window"
        );
    }

    // ── Disk-before-memory promotion ───────────────────────────────────────
    //
    // Operator invariant under repair: a generated or promoted signing key
    // becomes active in memory only after its slot persistence reports
    // success. Failure injection is a filesystem fixture — a directory
    // planted at the slot path, or a state dir that is a regular file — which
    // stays deterministic under root, unlike permission-bit tricks.

    #[tokio::test]
    async fn first_boot_persistence_failure_leaves_no_active_ed25519_key() {
        // The production composite boundary mutates the process-global
        // key-set version/snapshot; a mutex cannot reset that state between
        // tests. Re-execute this test in its own process (fresh globals),
        // mirroring the other composite-global tests in this module.
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_FIRST_BOOT_ED_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::first_boot_persistence_failure_leaves_no_active_ed25519_key",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let dir = TempDir::new().unwrap();
        let config = test_config();
        // write_secret persists via rename onto this path; renaming a file
        // onto a directory fails for every user, including root.
        std::fs::create_dir(dir.path().join("jwt-signing-key.active")).unwrap();

        let store = load_or_init_key_store(dir.path(), &config);
        assert!(
            store.active_verifying_key_bytes().await.is_none(),
            "an unpersisted generated key must not be exposed as active"
        );
        assert!(
            !dir.path().join("jwt-signing-key.active.meta").exists(),
            "no slot metadata may exist when the key bytes failed to persist"
        );

        // Production composite-initialization boundary with a healthy ML-DSA
        // counterpart: an incomplete first-boot store must fail the
        // initialization instead of committing partial authority.
        let pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let error = initialize_composite_key_set(
            dir.path(),
            &store,
            &pq,
            Arc::new(SigningKey::from_bytes(&[0x5A; 32])),
            30,
        )
        .await
        .expect_err("incomplete first-boot store must not create composite authority");
        assert!(
            error.to_string().contains("incomplete local signer set"),
            "unexpected initialization error: {error:#}"
        );
        assert!(
            !dir.path().join("jwt-composite-pairs.json").exists(),
            "no pending ledger may be created from an incomplete store"
        );
        assert!(
            !dir.path().join("jwt-composite-pairs.committed").exists(),
            "no commit marker may be created from an incomplete store"
        );
        let snapshots = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("jwt-composite-pairs.committed-")
            })
            .count();
        assert_eq!(
            snapshots, 0,
            "no immutable committed snapshot may be created from an incomplete store"
        );

        // Remove the injected failure and recover with fresh stores through
        // the same production boundary.
        std::fs::remove_dir(dir.path().join("jwt-signing-key.active")).unwrap();
        let fresh_ed = load_or_init_key_store(dir.path(), &config);
        let fresh_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
        initialize_composite_key_set(
            dir.path(),
            &fresh_ed,
            &fresh_pq,
            Arc::new(SigningKey::from_bytes(&[0x5A; 32])),
            30,
        )
        .await
        .expect("normal initialization must succeed once persistence recovers");
        assert!(dir.path().join("jwt-composite-pairs.json").exists());
        assert!(dir.path().join("jwt-composite-pairs.committed").exists());
    }

    #[tokio::test]
    async fn ed25519_rotation_keeps_active_when_drain_persist_fails() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

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
        let drain_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 45 * 86400,
            now + 5 * 86400,
        );
        let active_vk = active_slot.verifying_key_bytes();
        let lead_vk = lead_slot.verifying_key_bytes();
        let drain_vk = drain_slot.verifying_key_bytes();

        let store = SigningKeyStore::new(KeySlots {
            drain: Some(drain_slot),
            active: Some(active_slot),
            lead: Some(lead_slot),
        });

        // Deterministic seam: this state dir is a regular file, so the first
        // promotion write (old active → drain) fails.
        let broken_state = dir.path().join("not-a-directory");
        std::fs::write(&broken_state, b"file").unwrap();
        rotate_jwt_keys_in_state_dir(&config, &broken_state, &store, now).await;

        let slots = store.0.read().await;
        assert_eq!(
            slots.active.as_ref().map(KeySlot::verifying_key_bytes),
            Some(active_vk),
            "the old active must remain the in-memory active signer"
        );
        assert_eq!(
            slots.lead.as_ref().map(KeySlot::verifying_key_bytes),
            Some(lead_vk),
            "the queued lead must stay in memory for the next tick"
        );
        assert_eq!(
            slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
            Some(drain_vk),
            "a failed promotion must not evict the in-memory drain"
        );
        drop(slots);

        // Persistence recovers: the queued lead promotes and the promoted
        // active is on disk before it is served from memory.
        rotate_jwt_keys_in_state_dir(&config, dir.path(), &store, now).await;
        assert_eq!(
            store.active_verifying_key_bytes().await,
            Some(lead_vk),
            "the queued lead must promote once persistence succeeds"
        );
        let persisted = load_slot(dir.path(), "active").expect("promoted active must be persisted");
        assert_eq!(
            persisted.verifying_key_bytes(),
            lead_vk,
            "memory promotion must follow the durable active write"
        );
        let slots = store.0.read().await;
        assert_eq!(
            slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
            Some(active_vk),
            "the old active becomes drain only after its persist succeeded"
        );
        assert!(slots.lead.is_none());
        assert!(!dir.path().join("jwt-signing-key.lead").exists());
    }

    #[tokio::test]
    async fn ed25519_rotation_keeps_active_when_active_persist_fails() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 1,
        );
        let lead_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let active_vk = active_slot.verifying_key_bytes();
        let lead_vk = lead_slot.verifying_key_bytes();
        persist_slot(dir.path(), "lead", &lead_slot).unwrap();

        let store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: Some(lead_slot),
        });

        // The drain write succeeds (old active → drain) but the promoted
        // active write fails before its seed lands: its slot path is a
        // directory. The rollback restore fails the same way, so the cycle
        // reports not-restored and keeps the durable drain copy of the old
        // active signer for restart recovery.
        std::fs::create_dir(dir.path().join("jwt-signing-key.active")).unwrap();
        let restored = rotate_jwt_keys_in_state_dir(&config, dir.path(), &store, now).await;
        assert!(
            !restored,
            "a failed rollback must report the stores as not fully restored"
        );

        assert_eq!(
            store.active_verifying_key_bytes().await,
            Some(active_vk),
            "the failed new key must not be exposed as active in memory"
        );
        {
            let slots = store.0.read().await;
            assert!(slots.lead.is_some(), "the lead stays queued for retry");
            assert!(
                slots.drain.is_none(),
                "in-memory drain must not change before every write succeeded"
            );
        }
        assert_eq!(
            load_slot(dir.path(), "drain")
                .expect("the drain copy of the old active must survive a failed rollback")
                .verifying_key_bytes(),
            active_vk,
            "a failed rollback must not erase the recoverable old active signer"
        );
        assert!(
            dir.path().join("jwt-signing-key.active").is_dir()
                && load_slot(dir.path(), "active").is_none(),
            "the active slot must not be durably replaced by the failed key; until the retry lands, the old active survives in memory and in drain"
        );

        // Remove the obstruction; the queued lead promotes on the next tick.
        std::fs::remove_dir(dir.path().join("jwt-signing-key.active")).unwrap();
        rotate_jwt_keys_in_state_dir(&config, dir.path(), &store, now).await;
        assert_eq!(store.active_verifying_key_bytes().await, Some(lead_vk));
        assert_eq!(
            load_slot(dir.path(), "active")
                .expect("promoted active must be persisted")
                .verifying_key_bytes(),
            lead_vk,
            "memory promotion must be preceded by the durable active write"
        );
        let slots = store.0.read().await;
        assert_eq!(
            slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
            Some(active_vk)
        );
        assert!(slots.lead.is_none());
    }

    #[tokio::test]
    async fn ed25519_failed_rollback_preserves_recoverable_active_in_drain() {
        // The production composite boundary mutates the process-global
        // key-set version/snapshot; a mutex cannot reset that state between
        // tests. Re-execute this test in its own process (fresh globals),
        // mirroring the other composite-global tests in this module.
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_ROLLBACK_RECOVERY_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_failed_rollback_preserves_recoverable_active_in_drain",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        for prev_drain_present in [false, true] {
            let dir = TempDir::new().unwrap();
            let config = test_config();
            let now = chrono::Utc::now().timestamp();

            let active_slot = KeySlot::new(
                SigningKey::generate(&mut rand::rngs::OsRng),
                now - 14 * 86400,
                now + 6 * 3600,
            );
            let lead_slot = KeySlot::new(
                SigningKey::generate(&mut rand::rngs::OsRng),
                now - 1, // nbf already passed
                now + 14 * 86400,
            );
            let active_seed = active_slot.key.to_bytes();
            let active_vk = active_slot.verifying_key_bytes();
            let lead_vk = lead_slot.verifying_key_bytes();
            persist_slot(dir.path(), "active", &active_slot).unwrap();
            persist_slot(dir.path(), "lead", &lead_slot).unwrap();
            let prev_drain = if prev_drain_present {
                let drain = KeySlot::new(
                    SigningKey::generate(&mut rand::rngs::OsRng),
                    now - 60 * 86400,
                    now - 45 * 86400,
                );
                persist_slot(dir.path(), "drain", &drain).unwrap();
                Some(drain)
            } else {
                None
            };

            let store = SigningKeyStore::new(KeySlots {
                drain: prev_drain.clone(),
                active: Some(active_slot),
                lead: Some(lead_slot),
            });

            // Commit real authority through the production writer: the
            // committed OAuth active pair names exactly this Ed25519 signer.
            let pq_store = load_or_init_ml_dsa_key_store(dir.path(), &config);
            let ca_key = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
            initialize_composite_key_set(dir.path(), &store, &pq_store, Arc::clone(&ca_key), 30)
                .await
                .expect("healthy starting authority must initialize composite");
            let (commit_before, ledger_before) =
                read_committed_composite_ledger(dir.path()).unwrap();
            let oauth_active_ed = ledger_before
                .pairs
                .iter()
                .find(|record| record.role == "oauth" && record.state == "active")
                .expect("initialized authority must carry an active OAuth pair");
            assert_eq!(
                oauth_active_ed.ed25519_public,
                URL_SAFE_NO_PAD.encode(active_vk),
                "prior_drain={prev_drain_present}: the committed active pair must \
                 name the old active Ed25519 signer"
            );

            // Deterministic partial write: the promotion's seed rename onto
            // `jwt-signing-key.active` succeeds but every metadata persistence
            // at `jwt-signing-key.active.meta` fails (directory obstruction).
            // The promoted active fails on its metadata write, and the
            // rollback of the old active fails the same way — after its seed
            // has been restored.
            let meta_path = dir.path().join("jwt-signing-key.active.meta");
            std::fs::remove_file(&meta_path).unwrap();
            std::fs::create_dir(&meta_path).unwrap();

            let restored = rotate_jwt_keys_in_state_dir(&config, dir.path(), &store, now).await;
            assert!(
                !restored,
                "prior_drain={prev_drain_present}: a failed rollback must report \
                 the stores as not fully restored"
            );

            // Memory is unchanged: old active still active, lead queued.
            let slots = store.0.read().await;
            assert_eq!(
                slots.active.as_ref().map(KeySlot::verifying_key_bytes),
                Some(active_vk),
                "prior_drain={prev_drain_present}: the old active must remain the \
                 in-memory active signer"
            );
            assert_eq!(
                slots.lead.as_ref().map(KeySlot::verifying_key_bytes),
                Some(lead_vk),
                "prior_drain={prev_drain_present}: the lead must stay queued"
            );
            assert_eq!(
                slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
                prev_drain.as_ref().map(KeySlot::verifying_key_bytes),
                "prior_drain={prev_drain_present}: memory drain must not change \
                 before the rollback is confirmed"
            );
            drop(slots);

            // The drain slot keeps the only complete durable copy of the old
            // active signer — not deleted, not overwritten with the previous
            // drain generation.
            let recovered = load_slot(dir.path(), "drain").expect(
                "prior_drain={prev_drain_present}: drain copy of the old active \
                 must survive a failed rollback",
            );
            assert_eq!(
                recovered.verifying_key_bytes(),
                active_vk,
                "prior_drain={prev_drain_present}: failed rollback must not erase \
                 the recoverable old active signer"
            );
            assert_eq!(
                std::fs::read(dir.path().join("jwt-signing-key.active")).unwrap(),
                active_seed,
                "prior_drain={prev_drain_present}: the rollback must have restored \
                 the old active seed before failing on metadata"
            );
            assert!(
                load_slot(dir.path(), "active").is_none(),
                "prior_drain={prev_drain_present}: the active slot is incomplete \
                 without its metadata and must not load"
            );

            // A fresh reload — as after a restart, with the real commit marker
            // present so no implicit generation may run — still recovers the
            // signer required by committed authority from the drain slot,
            // before any retry is attempted and without reusing the in-memory
            // store.
            let fresh = load_or_init_key_store(dir.path(), &config);
            {
                let fresh_slots = fresh.0.read().await;
                assert_eq!(
                    fresh_slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
                    Some(active_vk),
                    "prior_drain={prev_drain_present}: fresh reload must still have \
                     the committed old active signer recoverable in drain"
                );
                assert!(
                    fresh_slots.active.is_none(),
                    "prior_drain={prev_drain_present}: the torn active slot must not \
                     load as a usable signer"
                );
            }

            // The production restore resolver — the same initialization the
            // oauth/policy services run at startup — must rebuild the
            // committed active pair from the recovered slots. The only
            // surviving Ed25519 copy of the committed signer is the drain
            // slot, so this fails (with the #1713-style restart error) if a
            // rollback erased it.
            let fresh_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
            initialize_composite_key_set(dir.path(), &fresh, &fresh_pq, Arc::clone(&ca_key), 30)
                .await
                .expect(
                    "prior_drain={prev_drain_present}: committed authority must restore \
                 from the recovered drain slot after a failed rollback",
                );
            let (commit_after, ledger_after) = read_committed_composite_ledger(dir.path()).unwrap();
            assert_eq!(
                commit_after.version, commit_before.version,
                "prior_drain={prev_drain_present}: recovery must not need a new \
                 committed generation"
            );
            assert_eq!(
                commit_after.component_digest,
                commit_before.component_digest
            );
            let oauth_active_ed_after = ledger_after
                .pairs
                .iter()
                .find(|record| record.role == "oauth" && record.state == "active")
                .expect("restored authority must still carry an active OAuth pair");
            assert_eq!(
                oauth_active_ed_after.ed25519_public,
                URL_SAFE_NO_PAD.encode(active_vk),
                "prior_drain={prev_drain_present}: restored committed authority \
                 must still name the recovered old active signer"
            );

            // Only now remove the injected failure: a retry from the fresh
            // store promotes the queued lead while the recovered old active
            // survives in drain.
            std::fs::remove_dir(&meta_path).unwrap();
            let retried = rotate_jwt_keys_in_state_dir(&config, dir.path(), &fresh, now).await;
            assert!(
                retried,
                "prior_drain={prev_drain_present}: the retry must fully restore and \
                 promote"
            );
            assert_eq!(
                fresh.active_verifying_key_bytes().await,
                Some(lead_vk),
                "prior_drain={prev_drain_present}: the queued lead promotes after \
                 recovery"
            );
            let slots = fresh.0.read().await;
            assert_eq!(
                slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
                Some(active_vk),
                "prior_drain={prev_drain_present}: the recovered old active must \
                 survive the retry in drain"
            );
            assert!(slots.lead.is_none());
        }
    }

    // ── ES256 store tests ──────────────────────────────────────────────────

    #[test]
    fn es256_load_or_init_creates_active_key() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store = load_or_init_es256_key_store(dir.path(), &config);
        let key = store.active_key();
        assert!(key.is_some());
    }

    #[test]
    fn es256_missing_active_with_durable_oplog_authority_does_not_regenerate() {
        for marker in [
            super::super::op_log::SEALED_HEAD_FILENAME,
            super::super::op_log::SEALED_HEAD_MAX_SEQ_FILENAME,
            super::super::op_log::HEAD_VERIFYING_KEY_FILENAME,
        ] {
            let root = TempDir::new().unwrap();
            let secrets_dir = root.path().join("credentials");
            let oplog_dir = root.path().join("oplog-state");
            std::fs::create_dir(&secrets_dir).unwrap();
            std::fs::create_dir(&oplog_dir).unwrap();
            std::fs::write(oplog_dir.join(marker), b"existing authority marker").unwrap();

            let store = load_or_init_es256_key_store_with_oplog_state_dir(
                &secrets_dir,
                &test_config(),
                &oplog_dir,
            );
            assert!(store.active_key().is_none(), "marker={marker}");
            assert!(
                load_es256_slot(&secrets_dir, "active").is_none(),
                "marker={marker}"
            );
            assert!(
                load_es256_slot(&secrets_dir, "drain").is_none(),
                "marker={marker}"
            );
            assert!(
                load_es256_slot(&secrets_dir, "lead").is_none(),
                "marker={marker}"
            );
        }
    }

    #[tokio::test]
    async fn es256_failed_first_boot_persistence_can_recover_without_published_marker() {
        let root = TempDir::new().unwrap();
        let secrets_dir = root.path().join("credentials");
        let oplog_dir = root.path().join("oplog-state");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::create_dir(&oplog_dir).unwrap();

        // The seed write succeeds, but making the metadata destination a
        // directory forces the second half of the active-slot write to fail.
        let blocked_metadata = secrets_dir.join("es256-signing-key.active.meta");
        std::fs::create_dir(&blocked_metadata).unwrap();
        let failed_boot = load_or_init_es256_key_store_with_oplog_state_dir(
            &secrets_dir,
            &test_config(),
            &oplog_dir,
        );
        assert!(failed_boot.active_key().is_none());

        assert!(super::super::op_log::advance_sealed_head_with_state_dir(
            &secrets_dir,
            &oplog_dir,
            &failed_boot,
        )
        .await
        .is_err());
        assert!(!oplog_dir
            .join(super::super::op_log::HEAD_VERIFYING_KEY_FILENAME)
            .exists());
        assert!(!oplog_dir
            .join(super::super::op_log::SEALED_HEAD_FILENAME)
            .exists());

        // After the transient obstruction is repaired, first boot can
        // complete normally without deleting any operation-log marker.
        std::fs::remove_dir(&blocked_metadata).unwrap();
        let recovered = load_or_init_es256_key_store_with_oplog_state_dir(
            &secrets_dir,
            &test_config(),
            &oplog_dir,
        );
        assert!(recovered.active_key().is_some());
        super::super::op_log::advance_sealed_head_with_state_dir(
            &secrets_dir,
            &oplog_dir,
            &recovered,
        )
        .await
        .unwrap();
        assert!(oplog_dir
            .join(super::super::op_log::HEAD_VERIFYING_KEY_FILENAME)
            .exists());
        assert!(oplog_dir
            .join(super::super::op_log::SEALED_HEAD_FILENAME)
            .exists());
    }

    #[tokio::test]
    async fn advance_oplog_does_not_reload_unpromoted_es256_disk_candidate() {
        let root = TempDir::new().unwrap();
        let secrets_dir = root.path().join("credentials");
        let oplog_dir = root.path().join("oplog-state");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::create_dir(&oplog_dir).unwrap();

        let now = chrono::Utc::now().timestamp();
        let retained = generate_es256_slot(now - 120, now + 86_400);
        let rejected_candidate = generate_es256_slot(now, now + 172_800);
        let retained_kid = retained.kid();
        persist_es256_slot(&secrets_dir, "active", &rejected_candidate).unwrap();
        persist_es256_slot(&secrets_dir, "drain", &retained).unwrap();

        // Simulate a failed rotation rollback: the candidate remains in the
        // durable active path, but the store correctly retains the old key.
        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(retained),
            lead: Some(rejected_candidate),
        });
        store.bind_secrets_dir(secrets_dir.clone());
        assert_eq!(store.active_slot().unwrap().kid(), retained_kid);

        super::super::op_log::advance_sealed_head_with_state_dir(&secrets_dir, &oplog_dir, &store)
            .await
            .unwrap();

        assert_eq!(
            store.active_slot().unwrap().kid(),
            retained_kid,
            "op-log advancement must not reload an unpromoted disk candidate"
        );
        let head: super::super::op_log::SealedOpLogHead = serde_json::from_slice(
            &std::fs::read(oplog_dir.join(super::super::op_log::SEALED_HEAD_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(head.active_kid, retained_kid);
    }

    #[test]
    fn es256_persist_and_reload() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let store1 = load_or_init_es256_key_store(dir.path(), &config);
        let kid1 = {
            let slots = store1.all_slots_snapshot();
            slots[0].kid()
        };
        let store2 = load_or_init_es256_key_store(dir.path(), &config);
        let kid2 = {
            let slots = store2.all_slots_snapshot();
            slots[0].kid()
        };
        assert_eq!(kid1, kid2, "ES256 key must survive reload from disk");
    }

    #[test]
    fn es256_startup_removes_expired_bounded_slots() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();
        let active = generate_es256_slot(now - 60, now + 86_400);
        let expired_drain = generate_es256_slot(now - 120, now);
        persist_es256_slot(dir.path(), "active", &active).unwrap();
        persist_es256_slot(dir.path(), "drain", &expired_drain).unwrap();

        let store = load_or_init_es256_key_store(dir.path(), &config);
        assert!(store.drain_slot().is_none());
        assert!(load_es256_slot(dir.path(), "drain").is_none());
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
        store.set_promotion_hook(Arc::new(|_| Ok(())));

        rotate_es256_keys(&config, dir.path(), &store, now).await;

        let slots = store.0.read();
        assert_eq!(slots.active.as_ref().unwrap().kid(), lead_kid);
        assert!(slots.drain.is_some());
        assert!(slots.lead.is_none());
    }

    #[tokio::test]
    async fn es256_drain_is_removed_at_its_published_expiry() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();
        let active = generate_es256_slot(now - 14 * 86400, now + 1);
        let lead = generate_es256_slot(now - 1, now + 14 * 86400);
        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });
        store.set_promotion_hook(Arc::new(|_| Ok(())));

        assert!(rotate_es256_keys(&config, dir.path(), &store, now).await);
        let published_expiry = store.drain_slot().expect("drain after promotion").exp;
        assert!(
            rotate_es256_keys(&config, dir.path(), &store, published_expiry).await
                || store.active_slot().is_some(),
            "cleanup tick remains operational"
        );
        assert!(
            store.drain_slot().is_none(),
            "the drain slot must leave the publication source at its expiry"
        );
    }

    #[tokio::test]
    async fn es256_rotate_without_hook_publishes_drain_for_ipc_workers() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = generate_es256_slot(now - 14 * 86400, now + 1);
        let active_kid = active.kid();
        let lead = generate_es256_slot(now - 1, now + 14 * 86400);
        let lead_kid = lead.kid();
        persist_es256_slot(dir.path(), "active", &active).unwrap();
        persist_es256_slot(dir.path(), "lead", &lead).unwrap();
        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });

        assert!(rotate_es256_keys(&config, dir.path(), &store, now).await);
        let slots = store.0.read();
        assert_eq!(slots.active.as_ref().unwrap().kid(), lead_kid);
        assert!(slots.lead.is_none());
        assert_eq!(slots.drain.as_ref().unwrap().kid(), active_kid);
        assert_eq!(
            load_es256_slot(dir.path(), "active").unwrap().kid(),
            lead_kid
        );
    }

    #[tokio::test]
    async fn es256_rotation_retries_after_promotion_hook_failure() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = generate_es256_slot(now - 14 * 86400, now + 1);
        let active_kid = active.kid();
        let lead = generate_es256_slot(now - 1, now + 14 * 86400);
        let lead_kid = lead.kid();
        persist_es256_slot(dir.path(), "active", &active).unwrap();
        persist_es256_slot(dir.path(), "lead", &lead).unwrap();

        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });
        let fail_once = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let injected_failure = Arc::clone(&fail_once);
        store.set_promotion_hook(Arc::new(move |_| {
            if injected_failure.swap(false, Ordering::AcqRel) {
                anyhow::bail!("injected head re-sign failure");
            }
            Ok(())
        }));

        assert!(
            !rotate_es256_keys(&config, dir.path(), &store, now).await,
            "a failed promotion hook must leave the candidate queued"
        );
        {
            let slots = store.0.read();
            assert_eq!(slots.active.as_ref().unwrap().kid(), active_kid);
            assert_eq!(slots.lead.as_ref().unwrap().kid(), lead_kid);
            assert!(slots.drain.is_none());
        }
        assert_eq!(
            load_es256_slot(dir.path(), "active").unwrap().kid(),
            active_kid,
            "failed promotion must restore the durable active key"
        );

        assert!(rotate_es256_keys(&config, dir.path(), &store, now).await);
        let slots = store.0.read();
        assert_eq!(slots.active.as_ref().unwrap().kid(), lead_kid);
        assert!(slots.lead.is_none());
    }

    #[tokio::test]
    async fn es256_rotate_retries_after_active_persistence_failure() {
        let dir = TempDir::new().unwrap();
        let invalid_secrets_dir = dir.path().join("not-a-directory");
        std::fs::write(&invalid_secrets_dir, b"file").unwrap();
        let valid_secrets_dir = dir.path().join("secrets");
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = generate_es256_slot(now - 14 * 86400, now + 1);
        let active_kid = active.kid();
        let lead = generate_es256_slot(now - 1, now + 14 * 86400);
        let lead_kid = lead.kid();
        let store = Es256SigningKeyStore::new(Es256KeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });
        assert!(
            !rotate_es256_keys_in_state_dir(&config, &invalid_secrets_dir, &store, now).await,
            "failed active persistence must not report promotion"
        );
        {
            let slots = store.0.read();
            assert_eq!(slots.active.as_ref().unwrap().kid(), active_kid);
            assert_eq!(slots.lead.as_ref().unwrap().kid(), lead_kid);
        }
        assert!(
            rotate_es256_keys(&config, &valid_secrets_dir, &store, now).await,
            "queued lead must be retried after persistence recovers"
        );
        assert_eq!(store.active_slot().unwrap().kid(), lead_kid);
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
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
                &ml_dsa::Keypair::verifying_key(&*key).clone(),
            )
        };
        let store2 = load_or_init_ml_dsa_key_store(dir.path(), &config);
        let vk2 = {
            let key = rt.block_on(store2.active_key()).unwrap();
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(
                &ml_dsa::Keypair::verifying_key(&*key).clone(),
            )
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
        assert_eq!(
            active_vk, lead_vk,
            "lead must become active after promotion"
        );
        assert!(slots.drain.is_some());
        assert!(slots.lead.is_none());
    }

    fn ml_dsa_slot_vk_bytes(slot: &ml_dsa_rotation::MlDsaKeySlot) -> Vec<u8> {
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&slot.verifying_key())
    }

    fn ml_dsa_key_vk_bytes(key: &hyprstream_rpc::crypto::pq::MlDsaSigningKey) -> Vec<u8> {
        hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&ml_dsa::Keypair::verifying_key(key).clone())
    }

    #[tokio::test]
    async fn ml_dsa_first_boot_persistence_failure_leaves_no_active_key() {
        // The production composite boundary mutates the process-global
        // key-set version/snapshot; a mutex cannot reset that state between
        // tests. Re-execute this test in its own process (fresh globals),
        // mirroring the other composite-global tests in this module.
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_FIRST_BOOT_ML_DSA_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ml_dsa_first_boot_persistence_failure_leaves_no_active_key",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let dir = TempDir::new().unwrap();
        let config = test_config();
        // write_secret persists via rename onto this path; renaming a file
        // onto a directory fails for every user, including root.
        std::fs::create_dir(dir.path().join("ml-dsa-signing-key.active")).unwrap();

        let store = load_or_init_ml_dsa_key_store(dir.path(), &config);
        assert!(
            store.active_key().await.is_none(),
            "an unpersisted generated ML-DSA key must not be exposed as active"
        );
        assert!(
            !dir.path().join("ml-dsa-signing-key.active.meta").exists(),
            "no slot metadata may exist when the key bytes failed to persist"
        );

        // Production composite-initialization boundary with a healthy Ed25519
        // counterpart: the symmetric first-boot failure must fail the
        // initialization instead of committing authority without an active
        // ML-DSA component.
        let ed = load_or_init_key_store(dir.path(), &config);
        let error = initialize_composite_key_set(
            dir.path(),
            &ed,
            &store,
            Arc::new(SigningKey::from_bytes(&[0x5A; 32])),
            30,
        )
        .await
        .expect_err("incomplete first-boot store must not create composite authority");
        assert!(
            error.to_string().contains("incomplete local signer set"),
            "unexpected initialization error: {error:#}"
        );
        assert!(
            !dir.path().join("jwt-composite-pairs.json").exists(),
            "no pending ledger may be created from an incomplete store"
        );
        assert!(
            !dir.path().join("jwt-composite-pairs.committed").exists(),
            "no commit marker may be created from an incomplete store"
        );
        let snapshots = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("jwt-composite-pairs.committed-")
            })
            .count();
        assert_eq!(
            snapshots, 0,
            "no immutable committed snapshot may be created from an incomplete store"
        );

        // Remove the injected failure and recover with fresh stores through
        // the same production boundary.
        std::fs::remove_dir(dir.path().join("ml-dsa-signing-key.active")).unwrap();
        let fresh_ed = load_or_init_key_store(dir.path(), &config);
        let fresh_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
        initialize_composite_key_set(
            dir.path(),
            &fresh_ed,
            &fresh_pq,
            Arc::new(SigningKey::from_bytes(&[0x5A; 32])),
            30,
        )
        .await
        .expect("normal initialization must succeed once persistence recovers");
        assert!(dir.path().join("jwt-composite-pairs.json").exists());
        assert!(dir.path().join("jwt-composite-pairs.committed").exists());
    }

    #[tokio::test]
    async fn ml_dsa_rotation_keeps_active_when_drain_persist_fails() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = ml_dsa_rotation::generate_ml_dsa_slot(now - 14 * 86400, now + 1);
        let lead = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        let drain = ml_dsa_rotation::generate_ml_dsa_slot(now - 45 * 86400, now + 5 * 86400);
        let active_vk = ml_dsa_slot_vk_bytes(&active);
        let lead_vk = ml_dsa_slot_vk_bytes(&lead);
        let drain_vk = ml_dsa_slot_vk_bytes(&drain);

        let store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: Some(drain),
            active: Some(active),
            lead: Some(lead),
        });

        // Deterministic seam: this state dir is a regular file, so the first
        // promotion write (old active → drain) fails.
        let broken_state = dir.path().join("not-a-directory");
        std::fs::write(&broken_state, b"file").unwrap();
        ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, &broken_state, &store, now).await;

        let slots = store.0.read().await;
        assert_eq!(
            slots.active.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(active_vk.clone()),
            "the old active must remain the in-memory active signer"
        );
        assert_eq!(
            slots.lead.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(lead_vk.clone()),
            "the queued lead must stay in memory for the next tick"
        );
        assert_eq!(
            slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(drain_vk.clone()),
            "a failed promotion must not evict the in-memory drain"
        );
        drop(slots);

        // Persistence recovers: the queued lead promotes and the promoted
        // active is on disk before it is served from memory.
        ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, dir.path(), &store, now).await;
        let persisted = ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "active")
            .expect("promoted active must be persisted");
        assert_eq!(
            ml_dsa_slot_vk_bytes(&persisted),
            lead_vk,
            "memory promotion must follow the durable active write"
        );
        let slots = store.0.read().await;
        assert_eq!(
            slots.active.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(lead_vk),
            "the queued lead must promote once persistence succeeds"
        );
        assert_eq!(
            slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(active_vk),
            "the old active becomes drain only after its persist succeeded"
        );
        assert!(slots.lead.is_none());
        assert!(!dir.path().join("ml-dsa-signing-key.lead").exists());
    }

    #[tokio::test]
    async fn ml_dsa_rotation_keeps_active_when_active_persist_fails() {
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let active = ml_dsa_rotation::generate_ml_dsa_slot(now - 14 * 86400, now + 1);
        let lead = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        let active_vk = ml_dsa_slot_vk_bytes(&active);
        let lead_vk = ml_dsa_slot_vk_bytes(&lead);
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "lead", &lead).unwrap();

        let store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(active),
            lead: Some(lead),
        });

        // The drain write succeeds (old active → drain) but the promoted
        // active write fails: its slot path is a directory. The old active
        // lives only in memory here, which is exactly what the rollback and
        // the retry rely on.
        std::fs::create_dir(dir.path().join("ml-dsa-signing-key.active")).unwrap();
        let restored =
            ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, dir.path(), &store, now)
                .await;
        assert!(
            !restored,
            "a failed rollback must report the stores as not fully restored"
        );

        assert_eq!(
            store.active_key().await.as_deref().map(ml_dsa_key_vk_bytes),
            Some(active_vk.clone()),
            "the failed new key must not be exposed as active in memory"
        );
        {
            let slots = store.0.read().await;
            assert!(slots.lead.is_some(), "the lead stays queued for retry");
            assert!(
                slots.drain.is_none(),
                "in-memory drain must not change before every write succeeded"
            );
        }
        assert_eq!(
            ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "drain")
                .map(|slot| ml_dsa_slot_vk_bytes(&slot)),
            Some(active_vk.clone()),
            "a failed rollback must not erase the drain copy of the old active \
             signer"
        );
        assert!(
            dir.path().join("ml-dsa-signing-key.active").is_dir()
                && ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "active").is_none(),
            "the active slot must not be durably replaced by the failed key; \
             until the retry lands, the old active survives in memory and in \
             drain"
        );

        // Remove the obstruction; the queued lead promotes on the next tick.
        std::fs::remove_dir(dir.path().join("ml-dsa-signing-key.active")).unwrap();
        ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, dir.path(), &store, now).await;
        assert_eq!(
            store.active_key().await.as_deref().map(ml_dsa_key_vk_bytes),
            Some(lead_vk.clone()),
            "the queued lead must promote once persistence succeeds"
        );
        let persisted = ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "active")
            .expect("promoted active must be persisted");
        assert_eq!(
            ml_dsa_key_vk_bytes(persisted.key.as_ref()),
            lead_vk,
            "memory promotion must be preceded by the durable active write"
        );
        let slots = store.0.read().await;
        assert_eq!(
            slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
            Some(active_vk),
            "the old active becomes drain only after its persist succeeded"
        );
        assert!(slots.lead.is_none());
    }

    #[tokio::test]
    async fn ml_dsa_failed_rollback_preserves_recoverable_active_in_drain() {
        // The production composite boundary mutates the process-global
        // key-set version/snapshot; a mutex cannot reset that state between
        // tests. Re-execute this test in its own process (fresh globals),
        // mirroring the other composite-global tests in this module.
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ML_DSA_ROLLBACK_RECOVERY_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ml_dsa_failed_rollback_preserves_recoverable_active_in_drain",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        for prev_drain_present in [false, true] {
            let dir = TempDir::new().unwrap();
            let config = test_config();
            let now = chrono::Utc::now().timestamp();

            // Generous validity margins: this test may run on a slow or
            // suspended machine, and restore resolution skips expired
            // committed records.
            let active = ml_dsa_rotation::generate_ml_dsa_slot(now - 14 * 86400, now + 6 * 3600);
            let lead = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
            let active_vk = ml_dsa_slot_vk_bytes(&active);
            let lead_vk = ml_dsa_slot_vk_bytes(&lead);
            ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &active).unwrap();
            ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "lead", &lead).unwrap();
            let prev_drain = if prev_drain_present {
                let drain =
                    ml_dsa_rotation::generate_ml_dsa_slot(now - 60 * 86400, now - 45 * 86400);
                ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "drain", &drain).unwrap();
                Some(drain)
            } else {
                None
            };

            let store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
                drain: prev_drain.clone(),
                active: Some(active),
                lead: Some(lead),
            });

            // Commit real authority through the production writer: the
            // committed OAuth active pair names exactly this ML-DSA signer.
            let ed_store = load_or_init_key_store(dir.path(), &config);
            let ca_key = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
            initialize_composite_key_set(dir.path(), &ed_store, &store, Arc::clone(&ca_key), 30)
                .await
                .expect("healthy starting authority must initialize composite");
            let (commit_before, ledger_before) =
                read_committed_composite_ledger(dir.path()).unwrap();
            let oauth_active_pq = ledger_before
                .pairs
                .iter()
                .find(|record| record.role == "oauth" && record.state == "active")
                .expect("initialized authority must carry an active OAuth pair");
            assert_eq!(
                oauth_active_pq.ml_dsa_public,
                URL_SAFE_NO_PAD.encode(&active_vk),
                "prior_drain={prev_drain_present}: the committed active pair must \
                 name the old active ML-DSA signer"
            );

            // Deterministic partial write, mirroring the Ed25519 scenario: the
            // promotion's seed rename succeeds but every metadata persistence
            // at `ml-dsa-signing-key.active.meta` fails, so the promoted
            // active and its rollback both fail after the seed write.
            let meta_path = dir.path().join("ml-dsa-signing-key.active.meta");
            std::fs::remove_file(&meta_path).unwrap();
            std::fs::create_dir(&meta_path).unwrap();

            let restored =
                ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, dir.path(), &store, now)
                    .await;
            assert!(
                !restored,
                "prior_drain={prev_drain_present}: a failed rollback must report \
                 the stores as not fully restored"
            );

            let slots = store.0.read().await;
            assert_eq!(
                slots.active.as_ref().map(ml_dsa_slot_vk_bytes),
                Some(active_vk.clone()),
                "prior_drain={prev_drain_present}: the old active must remain the \
                 in-memory active signer"
            );
            assert_eq!(
                slots.lead.as_ref().map(ml_dsa_slot_vk_bytes),
                Some(lead_vk.clone()),
                "prior_drain={prev_drain_present}: the lead must stay queued"
            );
            assert_eq!(
                slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
                prev_drain.as_ref().map(ml_dsa_slot_vk_bytes),
                "prior_drain={prev_drain_present}: memory drain must not change \
                 before the rollback is confirmed"
            );
            drop(slots);

            let recovered = ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "drain").expect(
                "prior_drain={prev_drain_present}: drain copy of the old active \
                 must survive a failed rollback",
            );
            assert_eq!(
                ml_dsa_slot_vk_bytes(&recovered),
                active_vk,
                "prior_drain={prev_drain_present}: failed rollback must not erase \
                 the recoverable old active signer"
            );
            assert!(
                ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "active").is_none(),
                "prior_drain={prev_drain_present}: the active slot is incomplete \
                 without its metadata and must not load"
            );

            // Fresh reload with the real commit marker present: the old active
            // signer stays recoverable from drain, before any retry.
            let fresh = load_or_init_ml_dsa_key_store(dir.path(), &config);
            {
                let fresh_slots = fresh.0.read().await;
                assert_eq!(
                    fresh_slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
                    Some(active_vk.clone()),
                    "prior_drain={prev_drain_present}: fresh reload must still have \
                     the committed old active signer recoverable in drain"
                );
                assert!(
                    fresh_slots.active.is_none(),
                    "prior_drain={prev_drain_present}: the torn active slot must not \
                     load as a usable signer"
                );
            }

            // The production restore resolver must rebuild the committed
            // active pair from the recovered slots: the only surviving ML-DSA
            // copy of the committed signer is the drain slot, so this fails
            // (with the #1713-style restart error) if a rollback erased it.
            let fresh_ed = load_or_init_key_store(dir.path(), &config);
            initialize_composite_key_set(dir.path(), &fresh_ed, &fresh, Arc::clone(&ca_key), 30)
                .await
                .expect(
                    "prior_drain={prev_drain_present}: committed authority must restore \
                 from the recovered drain slot after a failed rollback",
                );
            let (commit_after, ledger_after) = read_committed_composite_ledger(dir.path()).unwrap();
            assert_eq!(
                commit_after.version, commit_before.version,
                "prior_drain={prev_drain_present}: recovery must not need a new \
                 committed generation"
            );
            assert_eq!(
                commit_after.component_digest,
                commit_before.component_digest
            );
            let oauth_active_pq_after = ledger_after
                .pairs
                .iter()
                .find(|record| record.role == "oauth" && record.state == "active")
                .expect("restored authority must still carry an active OAuth pair");
            assert_eq!(
                oauth_active_pq_after.ml_dsa_public,
                URL_SAFE_NO_PAD.encode(&active_vk),
                "prior_drain={prev_drain_present}: restored committed authority \
                 must still name the recovered old active signer"
            );

            // Remove the injected failure and retry from the fresh store.
            std::fs::remove_dir(&meta_path).unwrap();
            let retried =
                ml_dsa_rotation::rotate_ml_dsa_keys_in_state_dir(&config, dir.path(), &fresh, now)
                    .await;
            assert!(
                retried,
                "prior_drain={prev_drain_present}: the retry must fully restore and \
                 promote"
            );
            assert_eq!(
                fresh.active_key().await.as_deref().map(ml_dsa_key_vk_bytes),
                Some(lead_vk),
                "prior_drain={prev_drain_present}: the queued lead promotes after \
                 recovery"
            );
            let slots = fresh.0.read().await;
            assert_eq!(
                slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
                Some(active_vk),
                "prior_drain={prev_drain_present}: the recovered old active must \
                 survive the retry in drain"
            );
            assert!(slots.lead.is_none());
        }
    }

    #[tokio::test]
    async fn failed_rollback_defers_composite_publication() {
        // The production composite boundary mutates the process-global
        // key-set version/snapshot; a mutex cannot reset that state between
        // tests. Re-execute this test in its own process (fresh globals),
        // mirroring the other composite-global tests in this module.
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_PUBLICATION_GATE_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::failed_rollback_defers_composite_publication",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Healthy ML-DSA store; the Ed25519 promotion below fails its rollback
        // via a directory obstruction at the active slot metadata path.
        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 1,
        );
        let lead_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        persist_slot(dir.path(), "active", &active_slot).unwrap();
        persist_slot(dir.path(), "lead", &lead_slot).unwrap();
        let ed_store = Arc::new(SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: Some(lead_slot),
        }));

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("a fully available signer set must initialize composite authority");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
        let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .version();

        let meta_path = dir.path().join("jwt-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();

        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };
        run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;

        // The cycle must not publish as successfully restored: committed
        // authority, the pending ledger, and the live composite snapshot all
        // retain the prior generation.
        let (commit_after, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(
            commit_after.version, commit_before.version,
            "a not-fully-restored cycle must not advance the commit marker"
        );
        assert_eq!(
            commit_after.component_digest, commit_before.component_digest,
            "a not-fully-restored cycle must not change the committed digest"
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "a not-fully-restored cycle must not restate the pending ledger"
        );
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before,
            "a not-fully-restored cycle must not publish to the live key set"
        );
        // The committed predecessor is durably recorded for the recovery tick.
        assert_eq!(
            read_deferred_predecessor(dir.path()).map(|record| record.committed_digest.clone()),
            Some(commit_before.component_digest.clone()),
            "a deferring cycle must retain the committed predecessor digest"
        );
    }

    #[tokio::test]
    async fn ed25519_restart_mismatch_fails_closed() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_RESTART_MISMATCH_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_restart_mismatch_fails_closed",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Healthy ML-DSA store with its own ready lead; Ed25519 store whose
        // promotion will fail its rollback in the first cycle.
        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let new_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        let pq_lead = ml_dsa_rotation::MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: pq_store.active_key().await.map(|key| {
                ml_dsa_rotation::MlDsaKeySlot::new((*key).clone(), now - 3600, now + 14 * 86400)
            }),
            lead: Some(new_pq.clone()),
        });
        let old_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 6 * 3600,
        );
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        persist_slot(dir.path(), "lead", &new_ed).unwrap();
        let ed_store = Arc::new(SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(old_ed.clone()),
            lead: Some(new_ed.clone()),
        }));

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        // Initialize with exactly the stores cycle 1 consumes, so the
        // committed digest matches the cycle's pre-rotation component state
        // (queued leads included) and the predecessor seeding validates.
        initialize_composite_key_set(dir.path(), &ed_store, &pq_lead, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
        let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .version();

        // Cycle 1: the ML-DSA lead promotes while the Ed25519 promotion fails
        // its rollback. Publication is deferred and the committed predecessor
        // digest is retained for the recovery tick.
        let meta_path = dir.path().join("jwt-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();
        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::new(pq_lead)),
            composite_ca_key: Arc::clone(&ca),
        };
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(
            !all_restored,
            "the mixed cycle must report not fully restored"
        );
        let (commit_deferred, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_deferred.version, commit_before.version);
        assert_eq!(
            commit_deferred.component_digest,
            commit_before.component_digest
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "a deferred cycle must not restate the pending ledger"
        );
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before,
            "a deferred cycle must not publish to the live key set"
        );
        assert_eq!(
            read_deferred_predecessor(dir.path()).map(|record| record.committed_digest.clone()),
            Some(commit_before.component_digest.clone()),
            "the committed predecessor digest must be retained across the \
             deferred cycle"
        );

        // Clear the write obstruction and reload. The slot state may not
        // match what this process certified in memory, so restart must not
        // infer authority from the files. Even after the marker becomes
        // readable again, the ledger digest gate must keep this state from
        // being published.
        std::fs::remove_dir(&meta_path).unwrap();
        let fresh_ed = Arc::new(load_or_init_key_store(dir.path(), &config));
        let fresh_pq = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let extra_fresh = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&fresh_pq)),
            composite_ca_key: Arc::clone(&ca),
        };
        let initial_record = read_deferred_predecessor(dir.path()).unwrap();
        let reloaded_digest =
            component_state_digest(&fresh_ed, &fresh_pq, ca.verifying_key()).await;
        assert!(
            !initial_record.states.contains(&reloaded_digest),
            "a restart reload not certified by this process must remain untrusted"
        );
        let marker_path = dir.path().join("jwt-composite-pairs.committed");
        let marker_bytes = std::fs::read(&marker_path).unwrap();
        std::fs::write(&marker_path, b"corrupt").unwrap();
        // The rotation succeeds; the publication fails on the corrupt
        // committed marker and must not commit anything.
        let _failed_cycle =
            run_rotation_cycle(&config, dir.path(), &fresh_ed, &extra_fresh, now).await;
        // Ledger access restored: the committed generation is untouched, the
        // record is retained, and its certified states advanced to the
        // recovery tick's own post-rotation outcome.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let (commit_failed, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(
            commit_failed.version, commit_before.version,
            "a failed recovery publication must not advance the committed \
             generation"
        );
        let record_after_failure = read_deferred_predecessor(dir.path())
            .expect("a failed recovery publication must retain the record");
        assert_eq!(
            record_after_failure.committed_digest, commit_before.component_digest,
            "the retained record must keep the committed predecessor"
        );
        assert_eq!(
            record_after_failure.states, initial_record.states,
            "an untrusted restart state must not be added to the recovery record"
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "the failed recovery publication must not restate the pending ledger"
        );

        // A readable marker does not make the mismatching restart state
        // trustworthy. The attempt remains fail-closed and preserves the
        // last committed authority for operator recovery.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let recovered = run_rotation_cycle(&config, dir.path(), &fresh_ed, &extra_fresh, now).await;
        assert!(
            recovered,
            "the tick after a failed recovery publication must publish"
        );
        let (commit_after, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_after.version, commit_before.version);
        assert_eq!(commit_after.component_digest, commit_before.component_digest);
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before
        );
        assert!(read_deferred_predecessor(dir.path()).is_some());
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before
        );
    }

    #[tokio::test]
    async fn ml_dsa_restart_mismatch_fails_closed() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ML_DSA_RESTART_MISMATCH_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ml_dsa_restart_mismatch_fails_closed",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Healthy Ed25519 store with its own ready lead; ML-DSA store whose
        // promotion will fail its rollback in the first cycle.
        let ed_store = Arc::new(load_or_init_key_store(dir.path(), &config));
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let ed_lead = SigningKeyStore::new(KeySlots {
            drain: None,
            active: ed_store
                .active_key()
                .await
                .map(|key| KeySlot::new((*key).clone(), now - 3600, now + 14 * 86400)),
            lead: Some(new_ed.clone()),
        });
        let old_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 14 * 86400, now + 6 * 3600);
        let new_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &old_pq).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "lead", &new_pq).unwrap();
        let pq_store = Arc::new(MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(old_pq.clone()),
            lead: Some(new_pq.clone()),
        }));

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        // Initialize with exactly the stores cycle 1 consumes, so the
        // committed digest matches the cycle's pre-rotation component state
        // (queued leads included) and the predecessor seeding validates.
        initialize_composite_key_set(dir.path(), &ed_lead, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
        let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .version();

        // Cycle 1: the Ed25519 lead promotes while the ML-DSA promotion fails
        // its rollback. Publication is deferred.
        let meta_path = dir.path().join("ml-dsa-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();
        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_lead, &extra, now).await;
        assert!(
            !all_restored,
            "the mixed cycle must report not fully restored"
        );
        let (commit_deferred, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_deferred.version, commit_before.version);
        assert_eq!(
            commit_deferred.component_digest,
            commit_before.component_digest
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "a deferred cycle must not restate the pending ledger"
        );
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before,
            "a deferred cycle must not publish to the live key set"
        );
        assert_eq!(
            read_deferred_predecessor(dir.path()).map(|record| record.committed_digest.clone()),
            Some(commit_before.component_digest.clone()),
            "the committed predecessor digest must be retained across the \
             deferred cycle"
        );

        // Clear the write obstruction and reload. If the loaded state does
        // not match what this process certified, restart must not infer
        // authority from the slot files. A readable marker later does not
        // make that mismatching state trusted.
        std::fs::remove_dir(&meta_path).unwrap();
        let fresh_ed = Arc::new(load_or_init_key_store(dir.path(), &config));
        let fresh_pq = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let extra_fresh = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&fresh_pq)),
            composite_ca_key: Arc::clone(&ca),
        };
        let initial_record = read_deferred_predecessor(dir.path()).unwrap();
        let reloaded_digest =
            component_state_digest(&fresh_ed, &fresh_pq, ca.verifying_key()).await;
        assert!(
            !initial_record.states.contains(&reloaded_digest),
            "a restart reload not certified by this process must remain untrusted"
        );
        let marker_path = dir.path().join("jwt-composite-pairs.committed");
        let marker_bytes = std::fs::read(&marker_path).unwrap();
        std::fs::write(&marker_path, b"corrupt").unwrap();
        // The rotation succeeds; the publication fails on the corrupt
        // committed marker and must not commit anything.
        let _failed_cycle =
            run_rotation_cycle(&config, dir.path(), &fresh_ed, &extra_fresh, now).await;
        // Ledger access restored: the committed generation is untouched, the
        // record is retained, and its certified states advanced to the
        // recovery tick's own post-rotation outcome.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let (commit_failed, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(
            commit_failed.version, commit_before.version,
            "a failed recovery publication must not advance the committed \
             generation"
        );
        let record_after_failure = read_deferred_predecessor(dir.path())
            .expect("a failed recovery publication must retain the record");
        assert_eq!(
            record_after_failure.committed_digest, commit_before.component_digest,
            "the retained record must keep the committed predecessor"
        );
        assert_eq!(
            record_after_failure.states, initial_record.states,
            "an untrusted restart state must not be added to the recovery record"
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "the failed recovery publication must not restate the pending ledger"
        );

        // A readable marker does not make the mismatching restart state
        // trustworthy. The retry preserves the committed authority and waits
        // for explicit repair.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let recovered = run_rotation_cycle(&config, dir.path(), &fresh_ed, &extra_fresh, now).await;
        assert!(recovered, "the rotation operation itself should remain usable");
        let (commit_after, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_after.version, commit_before.version);
        assert_eq!(commit_after.component_digest, commit_before.component_digest);
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before
        );
        assert!(read_deferred_predecessor(dir.path()).is_some());
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before,
            "the restart mismatch must not publish new signer authority"
        );
    }

    #[tokio::test]
    async fn ed25519_failed_drain_rollback_defers_publication() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_DRAIN_ROLLBACK_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_failed_drain_rollback_defers_publication",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        for prev_drain_present in [false, true] {
            let dir = TempDir::new().unwrap();
            let config = test_config();
            let now = chrono::Utc::now().timestamp();

            let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
            let active_slot = KeySlot::new(
                SigningKey::generate(&mut rand::rngs::OsRng),
                now - 14 * 86400,
                now + 6 * 3600,
            );
            let lead_slot = KeySlot::new(
                SigningKey::generate(&mut rand::rngs::OsRng),
                now - 1,
                now + 14 * 86400,
            );
            let active_vk = active_slot.verifying_key_bytes();
            let lead_vk = lead_slot.verifying_key_bytes();
            let prev_drain = if prev_drain_present {
                let drain = KeySlot::new(
                    SigningKey::generate(&mut rand::rngs::OsRng),
                    now - 60 * 86400,
                    now - 45 * 86400,
                );
                persist_slot(dir.path(), "drain", &drain).unwrap();
                Some(drain)
            } else {
                None
            };
            persist_slot(dir.path(), "active", &active_slot).unwrap();
            persist_slot(dir.path(), "lead", &lead_slot).unwrap();

            let ed_store = SigningKeyStore::new(KeySlots {
                drain: prev_drain.clone(),
                active: Some(active_slot),
                lead: Some(lead_slot),
            });

            let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
            initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
                .await
                .expect("healthy starting authority must initialize composite");
            let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
            let pending_before =
                std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
            let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version();

            // Deterministic partial drain write: the old-active-to-drain seed
            // rename succeeds but the drain metadata persistence fails, and
            // the previous-drain restoration (prev present) or the partial
            // replacement removal (prev absent) fails on the same metadata
            // path, so restoration cannot be confirmed.
            let meta_path = dir.path().join("jwt-signing-key.drain.meta");
            if meta_path.exists() {
                std::fs::remove_file(&meta_path).unwrap();
            }
            std::fs::create_dir(&meta_path).unwrap();

            let extra = RotationStores {
                es256: None,
                ml_dsa: Some(Arc::clone(&pq_store)),
                composite_ca_key: Arc::clone(&ca),
            };
            let all_restored =
                run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
            assert!(
                !all_restored,
                "prior_drain={prev_drain_present}: an unconfirmed previous-drain \
                 restoration must report the stores as not fully restored"
            );

            let slots = ed_store.0.read().await;
            assert_eq!(
                slots.active.as_ref().map(KeySlot::verifying_key_bytes),
                Some(active_vk),
                "prior_drain={prev_drain_present}: the old active must remain the \
                 in-memory active signer"
            );
            assert_eq!(
                slots.lead.as_ref().map(KeySlot::verifying_key_bytes),
                Some(lead_vk),
                "prior_drain={prev_drain_present}: the lead must stay queued"
            );
            assert_eq!(
                slots.drain.as_ref().map(KeySlot::verifying_key_bytes),
                prev_drain.as_ref().map(KeySlot::verifying_key_bytes),
                "prior_drain={prev_drain_present}: memory drain must not change"
            );
            drop(slots);

            assert!(
                load_slot(dir.path(), "drain").is_none(),
                "prior_drain={prev_drain_present}: the torn drain slot must not load"
            );

            // The failed cleanup/restore must defer publication exactly like
            // the active-rollback failure: committed, pending, and live
            // authority all retain the prior generation.
            let (commit_after, _) = read_committed_composite_ledger(dir.path()).unwrap();
            assert_eq!(commit_after.version, commit_before.version);
            assert_eq!(
                commit_after.component_digest,
                commit_before.component_digest
            );
            assert_eq!(
                std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
                pending_before,
                "a not-fully-restored cycle must not restate the pending ledger"
            );
            assert_eq!(
                hyprstream_rpc::auth::global_composite_key_set()
                    .snapshot()
                    .version(),
                snapshot_version_before,
                "a not-fully-restored cycle must not publish to the live key set"
            );
            assert_eq!(
                read_deferred_predecessor(dir.path()).map(|record| record.committed_digest.clone()),
                Some(commit_before.component_digest.clone()),
                "a deferring cycle must retain the committed predecessor digest"
            );
        }
    }

    #[tokio::test]
    async fn ml_dsa_failed_drain_rollback_defers_publication() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ML_DSA_DRAIN_ROLLBACK_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ml_dsa_failed_drain_rollback_defers_publication",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        for prev_drain_present in [false, true] {
            let dir = TempDir::new().unwrap();
            let config = test_config();
            let now = chrono::Utc::now().timestamp();

            let ed_store = Arc::new(load_or_init_key_store(dir.path(), &config));
            let active = ml_dsa_rotation::generate_ml_dsa_slot(now - 14 * 86400, now + 6 * 3600);
            let lead = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
            let active_vk = ml_dsa_slot_vk_bytes(&active);
            let lead_vk = ml_dsa_slot_vk_bytes(&lead);
            let prev_drain = if prev_drain_present {
                let drain =
                    ml_dsa_rotation::generate_ml_dsa_slot(now - 60 * 86400, now - 45 * 86400);
                ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "drain", &drain).unwrap();
                Some(drain)
            } else {
                None
            };
            ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &active).unwrap();
            ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "lead", &lead).unwrap();

            let pq_store = Arc::new(MlDsaSigningKeyStore::new(MlDsaKeySlots {
                drain: prev_drain.clone(),
                active: Some(active),
                lead: Some(lead),
            }));

            let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
            initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
                .await
                .expect("healthy starting authority must initialize composite");
            let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
            let pending_before =
                std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
            let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version();

            // Deterministic partial drain write, mirroring the Ed25519 case.
            let meta_path = dir.path().join("ml-dsa-signing-key.drain.meta");
            if meta_path.exists() {
                std::fs::remove_file(&meta_path).unwrap();
            }
            std::fs::create_dir(&meta_path).unwrap();

            let extra = RotationStores {
                es256: None,
                ml_dsa: Some(Arc::clone(&pq_store)),
                composite_ca_key: Arc::clone(&ca),
            };
            let all_restored =
                run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
            assert!(
                !all_restored,
                "prior_drain={prev_drain_present}: an unconfirmed previous-drain \
                 restoration must report the stores as not fully restored"
            );

            let slots = pq_store.0.read().await;
            assert_eq!(
                slots.active.as_ref().map(ml_dsa_slot_vk_bytes),
                Some(active_vk.clone()),
                "prior_drain={prev_drain_present}: the old active must remain the \
                 in-memory active signer"
            );
            assert_eq!(
                slots.lead.as_ref().map(ml_dsa_slot_vk_bytes),
                Some(lead_vk.clone()),
                "prior_drain={prev_drain_present}: the lead must stay queued"
            );
            assert_eq!(
                slots.drain.as_ref().map(ml_dsa_slot_vk_bytes),
                prev_drain.as_ref().map(ml_dsa_slot_vk_bytes),
                "prior_drain={prev_drain_present}: memory drain must not change"
            );
            drop(slots);

            assert!(
                ml_dsa_rotation::load_ml_dsa_slot(dir.path(), "drain").is_none(),
                "prior_drain={prev_drain_present}: the torn drain slot must not load"
            );

            let (commit_after, _) = read_committed_composite_ledger(dir.path()).unwrap();
            assert_eq!(commit_after.version, commit_before.version);
            assert_eq!(
                commit_after.component_digest,
                commit_before.component_digest
            );
            assert_eq!(
                std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
                pending_before,
                "a not-fully-restored cycle must not restate the pending ledger"
            );
            assert_eq!(
                hyprstream_rpc::auth::global_composite_key_set()
                    .snapshot()
                    .version(),
                snapshot_version_before,
                "a not-fully-restored cycle must not publish to the live key set"
            );
            assert_eq!(
                read_deferred_predecessor(dir.path()).map(|record| record.committed_digest.clone()),
                Some(commit_before.component_digest.clone()),
                "a deferring cycle must retain the committed predecessor digest"
            );
        }
    }

    #[tokio::test]
    async fn ed25519_stale_anchor_does_not_authorize_publication() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_STALE_ANCHOR_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_stale_anchor_does_not_authorize_publication",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        // The active slot sits outside the lead-generation window so this
        // no-op cycle does not publish a newly generated lead over the
        // foreign-record assertion below.
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 8 * 86400,
        );
        persist_slot(dir.path(), "active", &active_slot).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: None,
        });

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, ledger_before) = read_committed_composite_ledger(dir.path()).unwrap();
        let oauth_before = ledger_before
            .pairs
            .iter()
            .find(|record| record.role == "oauth" && record.state == "active")
            .expect("initialized authority must carry an active OAuth pair")
            .clone();

        // Another writer's recovery record: its certified deferred state is a
        // foreign stale component state, and its committed digest is the
        // current one. The writer of this test process holds stores that are
        // the committed generation itself, so the continuity check must
        // reject the record (its deferred state is not this process's
        // tick-start memory) and no stale publication may be staged.
        let foreign_ed = load_or_init_key_store(&dir.path().join("foreign-ed"), &config);
        let foreign_pq = load_or_init_ml_dsa_key_store(&dir.path().join("foreign-pq"), &config);
        let foreign_state =
            component_state_digest(&foreign_ed, &foreign_pq, ca.verifying_key()).await;
        let record = DeferredPredecessorRecord {
            committed_digest: commit_before.component_digest.clone(),
            states: vec![foreign_state],
        };
        assert!(
            write_deferred_predecessor(dir.path(), &record),
            "the test fixture must be able to record a foreign anchor"
        );

        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(all_restored);

        // No stale publication: the committed generation still names this
        // process's active OAuth pair, and the foreign component state never
        // entered the ledger.
        let (commit_after, ledger_after) = read_committed_composite_ledger(dir.path()).unwrap();
        let oauth_active_after = ledger_after
            .pairs
            .iter()
            .find(|record| record.role == "oauth" && record.state == "active")
            .expect("authority must still carry an active OAuth pair");
        assert_eq!(
            oauth_active_after.ed25519_public, oauth_before.ed25519_public,
            "a foreign certified state must not replace the committed OAuth \
             signer"
        );
        assert_ne!(
            oauth_active_after.ed25519_public,
            URL_SAFE_NO_PAD.encode(foreign_ed.active_verifying_key_bytes().await.unwrap()),
            "the foreign stale component state must never be published"
        );
        assert_eq!(
            commit_after.component_digest, commit_before.component_digest,
            "an ignored record must not change the committed digest"
        );
        assert!(
            read_deferred_predecessor(dir.path()).is_none(),
            "the successful publication clears the ignored record"
        );
    }

    #[tokio::test]
    async fn ed25519_stale_writer_ticks_do_not_certify_or_publish() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_STALE_WRITER_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_stale_writer_ticks_do_not_certify_or_publish",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Writer A: healthy ML-DSA store; Ed25519 store whose promotion fails
        // its rollback in cycle 1 (mixed cycle), seeding a real deferred
        // record.
        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let old_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 6 * 3600,
        );
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let new_ed_vk = new_ed.verifying_key_bytes();
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        persist_slot(dir.path(), "lead", &new_ed).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(old_ed),
            lead: Some(new_ed),
        });

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
        let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .version();

        // Simulate a stale writer having changed the shared ML-DSA slot
        // files before this process records its first deferred cycle. The
        // current process's in-memory store still has the committed key; the
        // reload digest below must not be certified from disk.
        let foreign_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 30 * 86400, now + 8 * 86400);
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &foreign_pq).unwrap();

        let meta_path = dir.path().join("jwt-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();
        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(
            !all_restored,
            "the mixed cycle must defer and record its certified states"
        );
        let seeded = read_deferred_predecessor(dir.path())
            .expect("the deferring cycle must retain a recovery record");
        assert_eq!(
            seeded.committed_digest, commit_before.component_digest,
            "the record must anchor on the current committed generation"
        );
        assert_eq!(
            seeded.states.len(),
            1,
            "only this process's memory state is recorded"
        );
        let foreign_reload = {
            let reloaded_ed = load_or_init_key_store(dir.path(), &config);
            let reloaded_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
            component_state_digest(&reloaded_ed, &reloaded_pq, ca.verifying_key()).await
        };
        assert_ne!(
            foreign_reload, seeded.states[0],
            "the mutable slot state predated this record and is not trusted"
        );
        assert!(
            !seeded.states.contains(&foreign_reload),
            "a foreign reload must never be certified into the first record"
        );
        let certified_states = seeded.states.clone();

        // Stale writer B: stores from a different (non-member) component
        // generation, no due work, so both of its ticks are all-restored
        // no-ops whose tick-start digest is not certified.
        let stale_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 30 * 86400,
            now + 8 * 86400,
        );
        let stale_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 30 * 86400, now + 8 * 86400);
        let stale_ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(stale_ed),
            lead: None,
        });
        let stale_pq_store = Arc::new(MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(stale_pq),
            lead: None,
        }));
        let stale_extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&stale_pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };

        // Two all-restored stale-writer ticks: each supplies its own (stale)
        // expectation, fails the digest gate against the committed ledger,
        // and must neither publish nor certify its outcome into the record.
        for tick in 1..=2 {
            let all_restored =
                run_rotation_cycle(&config, dir.path(), &stale_ed_store, &stale_extra, now).await;
            assert!(
                all_restored,
                "stale tick {tick}: the no-op writer is restored"
            );
            let (commit_after_tick, _ledger_after_tick) =
                read_committed_composite_ledger(dir.path()).unwrap();
            assert_eq!(
                commit_after_tick.version, commit_before.version,
                "stale tick {tick}: a rejected stale writer must not advance the \
                 committed generation"
            );
            assert_eq!(
                commit_after_tick.component_digest, commit_before.component_digest,
                "stale tick {tick}: the committed digest must not change"
            );
            assert_eq!(
                std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
                pending_before,
                "stale tick {tick}: the pending ledger must be untouched"
            );
            assert_eq!(
                hyprstream_rpc::auth::global_composite_key_set()
                    .snapshot()
                    .version(),
                snapshot_version_before,
                "stale tick {tick}: a rejected stale writer must not publish to \
                 the live key set"
            );
            let record = read_deferred_predecessor(dir.path())
                .expect("stale tick {tick}: the recovery record must survive");
            assert_eq!(
                record.committed_digest, seeded.committed_digest,
                "stale tick {tick}: the recorded committed predecessor must be \
                 preserved"
            );
            assert_eq!(
                record.states, certified_states,
                "stale tick {tick}: the stale writer's outcome must never be \
                 certified into the record"
            );
        }

        // The legitimate writer still recovers: its certified state anchors
        // publication of the intended generation, and the record is cleared.
        std::fs::remove_dir(&meta_path).unwrap();
        let recovered = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(recovered, "the certified writer must recover and publish");
        let (commit_recovered, ledger_recovered) =
            read_committed_composite_ledger(dir.path()).unwrap();
        assert!(
            commit_recovered.version > commit_before.version,
            "the certified recovery must publish a new committed generation"
        );
        let oauth_active = ledger_recovered
            .pairs
            .iter()
            .find(|record| record.role == "oauth" && record.state == "active")
            .expect("the published generation must carry an active OAuth pair");
        assert_eq!(
            oauth_active.ed25519_public,
            URL_SAFE_NO_PAD.encode(new_ed_vk),
            "the published generation must carry the recovered Ed25519 signer"
        );
        assert!(read_deferred_predecessor(dir.path()).is_none());
    }

    #[tokio::test]
    async fn ed25519_foreign_disk_state_never_certified() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_FOREIGN_DISK_STATE_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_foreign_disk_state_never_certified",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Writer A: healthy ML-DSA store with a ready lead (it promotes in
        // the mixed cycle); Ed25519 store whose promotion fails its rollback.
        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let new_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        let pq_lead = Arc::new(MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: pq_store.active_key().await.map(|key| {
                ml_dsa_rotation::MlDsaKeySlot::new((*key).clone(), now - 3600, now + 14 * 86400)
            }),
            lead: Some(new_pq.clone()),
        }));
        let old_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 6 * 3600,
        );
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let new_ed_vk = new_ed.verifying_key_bytes();
        let new_pq_vk = ml_dsa_slot_vk_bytes(&new_pq);
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        persist_slot(dir.path(), "lead", &new_ed).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(old_ed),
            lead: Some(new_ed),
        });

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        // Initialize with exactly the stores cycle 1 consumes (queued leads
        // included), so the committed digest matches cycle 1's pre-rotation
        // component state and the predecessor seeding validates.
        initialize_composite_key_set(dir.path(), &ed_store, &pq_lead, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();
        let snapshot_version_before = hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .version();

        // Cycle 1: the mixed cycle defers and records the committed
        // predecessor plus its certified states.
        let meta_path = dir.path().join("jwt-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();
        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_lead)),
            composite_ca_key: Arc::clone(&ca),
        };
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(!all_restored, "the mixed cycle must defer");
        let seeded = read_deferred_predecessor(dir.path())
            .expect("the deferring cycle must retain a recovery record");
        assert_eq!(
            seeded.committed_digest, commit_before.component_digest,
            "the record must anchor on the current committed generation"
        );
        assert_eq!(
            seeded.states.len(),
            1,
            "only process-computed state recorded"
        );

        // The transient Ed25519 obstruction clears before writer B runs.
        std::fs::remove_dir(&meta_path).unwrap();

        // Writer B: the same pre-promotion Ed25519 slots as the committed
        // generation, but a different stale ML-DSA slot set with its own
        // ready lead. B promotes both families (successful due promotions),
        // writes its outcome to the shared disk, and its publication is
        // rejected by the digest gate (its captured predecessor differs).
        let new_ed_b = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let new_pq_b = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        let new_pq_b_vk = ml_dsa_slot_vk_bytes(&new_pq_b);
        let ed_b = SigningKeyStore::new(KeySlots {
            drain: None,
            active: ed_store
                .active_key()
                .await
                .map(|key| KeySlot::new((*key).clone(), now - 3600, now + 14 * 86400)),
            lead: Some(new_ed_b.clone()),
        });
        let pq_b = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(ml_dsa_rotation::generate_ml_dsa_slot(
                now - 3600,
                now + 14 * 86400,
            )),
            lead: Some(new_pq_b.clone()),
        });
        let extra_b = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::new(pq_b)),
            composite_ca_key: Arc::clone(&ca),
        };
        let b_restored = run_rotation_cycle(&config, dir.path(), &ed_b, &extra_b, now).await;
        assert!(b_restored, "writer B's own rotations must succeed");

        // Writer A's certified recovery tick: its memory is certified, but
        // its ML-DSA rotation has no due work and leaves writer B's disk
        // slots untouched. The publication is forced to fail (corrupt
        // committed marker), and the advance must certify only writer A's
        // own memory outcome — never writer B's durable outcome.
        let marker_path = dir.path().join("jwt-composite-pairs.committed");
        let marker_bytes = std::fs::read(&marker_path).unwrap();
        std::fs::write(&marker_path, b"corrupt").unwrap();
        let all_restored = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(
            all_restored,
            "writer A's rotation is restored; only its publication fails"
        );
        let record = read_deferred_predecessor(dir.path())
            .expect("the recovery record must survive the failed publication");
        assert_eq!(
            record.committed_digest, commit_before.component_digest,
            "the record must keep the committed predecessor"
        );
        // Writer A's memory outcome includes the ML-DSA store the cycle
        // rotates (pq_lead), which advanced in the mixed cycle.
        let post_memory = component_state_digest(&ed_store, &pq_lead, ca.verifying_key()).await;
        assert_eq!(
            record.states,
            vec![post_memory.clone()],
            "the advance must certify only writer A's own memory outcome"
        );
        let foreign_reload = {
            let reloaded_ed = load_or_init_key_store(dir.path(), &config);
            let reloaded_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
            component_state_digest(&reloaded_ed, &reloaded_pq, ca.verifying_key()).await
        };
        assert_ne!(
            foreign_reload, post_memory,
            "writer B's disk slots must still be on disk for this regression"
        );
        assert!(
            !record.states.contains(&foreign_reload),
            "the durable state left by the rejected writer must never be \
             certified into the record"
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "the failed publication must not restate the pending ledger"
        );
        assert_eq!(
            hyprstream_rpc::auth::global_composite_key_set()
                .snapshot()
                .version(),
            snapshot_version_before,
            "the failed publication must not publish to the live key set"
        );

        // Ledger access restored: the committed generation is untouched by
        // the failed publication.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let (commit_recovered_check, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(
            commit_recovered_check.version, commit_before.version,
            "the failed publication must not advance the committed generation"
        );

        // Retry writer B: its tick-start digest is still not certified, so it
        // can neither publish nor advance the record; committed and live
        // authority remain the retained generation.
        let b_retry = run_rotation_cycle(&config, dir.path(), &ed_b, &extra_b, now).await;
        assert!(b_retry, "writer B's no-op tick is restored");
        let (commit_b_retry, ledger_b_retry) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_b_retry.version, commit_before.version);
        let oauth_b = ledger_b_retry
            .pairs
            .iter()
            .find(|record| record.role == "oauth" && record.state == "active")
            .expect("authority must still carry an active OAuth pair");
        assert_ne!(
            oauth_b.ml_dsa_public,
            URL_SAFE_NO_PAD.encode(&new_pq_b_vk),
            "writer B's stale ML-DSA outcome must never become committed \
             authority"
        );
        assert_eq!(
            read_deferred_predecessor(dir.path())
                .expect("the record must survive writer B's retry")
                .states,
            vec![post_memory.clone()],
            "writer B's retry must not certify its outcome into the record"
        );

        // Ledger access restored: writer A's next tick anchors on the
        // certified memory state and publishes the recovered generation.
        std::fs::write(&marker_path, &marker_bytes).unwrap();
        let recovered = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(
            recovered,
            "writer A's certified tick must recover and publish"
        );
        let (commit_recovered, ledger_recovered) =
            read_committed_composite_ledger(dir.path()).unwrap();
        assert!(
            commit_recovered.version > commit_before.version,
            "the certified recovery must publish a new committed generation"
        );
        let oauth_recovered = ledger_recovered
            .pairs
            .iter()
            .find(|record| record.role == "oauth" && record.state == "active")
            .expect("the published generation must carry an active OAuth pair");
        assert_eq!(
            oauth_recovered.ed25519_public,
            URL_SAFE_NO_PAD.encode(new_ed_vk),
            "the published generation must carry the recovered Ed25519 signer"
        );
        assert_eq!(
            oauth_recovered.ml_dsa_public,
            URL_SAFE_NO_PAD.encode(&new_pq_vk),
            "the published generation must carry the promoted ML-DSA signer"
        );
        assert!(read_deferred_predecessor(dir.path()).is_none());
    }

    #[tokio::test]
    async fn ed25519_chained_deferral_never_certifies_reload() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_ED_CHAINED_DEFER_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::ed25519_chained_deferral_never_certifies_reload",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        // Bootstrap the ML-DSA active slot on disk (the store handle itself
        // is rebuilt below with its ready lead).
        load_or_init_ml_dsa_key_store(dir.path(), &config);
        let old_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 6 * 3600,
        );
        let new_ed = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 1,
            now + 14 * 86400,
        );
        let old_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 3600, now + 14 * 86400);
        let new_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        persist_slot(dir.path(), "lead", &new_ed).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &old_pq).unwrap();
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "lead", &new_pq).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(old_ed),
            lead: Some(new_ed),
        });
        let pq_store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(old_pq),
            lead: Some(new_pq),
        });

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();

        let meta_path = dir.path().join("jwt-signing-key.active.meta");
        std::fs::remove_file(&meta_path).unwrap();
        std::fs::create_dir(&meta_path).unwrap();
        let pq_store = Arc::new(pq_store);
        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };

        // Cycle 1: mixed cycle (Ed25519 rollback fails, ML-DSA promotes)
        // defers and seeds {committed, [memory]}.
        let deferred1 = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(!deferred1, "cycle 1 must defer");
        let seeded =
            read_deferred_predecessor(dir.path()).expect("cycle 1 must retain a recovery record");
        assert_eq!(seeded.states.len(), 1, "seed records process memory only");
        let seeded_mem_digest =
            component_state_digest(&ed_store, &pq_store, ca.verifying_key()).await;
        assert_eq!(
            seeded.states,
            vec![seeded_mem_digest.clone()],
            "the seed must match only this process's computed memory"
        );

        // Cycle 2: consecutive not-fully-restored certified cycle. The
        // transition is unchanged, so the record must be kept as-is; nothing
        // new enters the certified states.
        let deferred2 = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(!deferred2, "cycle 2 must defer again (same obstruction)");
        let advanced = read_deferred_predecessor(dir.path())
            .expect("the record must survive the chained deferral");
        assert_eq!(
            advanced.committed_digest, commit_before.component_digest,
            "the chained deferral must keep the committed predecessor"
        );
        assert_eq!(
            advanced.states, seeded.states,
            "an unchanged chained deferral must keep the record as-is"
        );

        // Writer B's contamination analogue: a foreign ML-DSA active slot
        // written to the shared state dir between certified cycles (the
        // Ed25519 side stays torn under its own obstruction, as in the
        // captured-transition state the record describes).
        let foreign_pq = ml_dsa_rotation::generate_ml_dsa_slot(now - 1, now + 14 * 86400);
        ml_dsa_rotation::persist_ml_dsa_slot(dir.path(), "active", &foreign_pq).unwrap();
        let foreign_reload = {
            let reloaded_ed = load_or_init_key_store(dir.path(), &config);
            let reloaded_pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
            component_state_digest(&reloaded_ed, &reloaded_pq, ca.verifying_key()).await
        };
        assert_ne!(
            foreign_reload, seeded_mem_digest,
            "fixture: the foreign disk state must differ from the certified \
             memory outcome"
        );

        // Cycle 3: the failure persists (the meta obstruction is still in
        // place). The tick is certified (memory unchanged), its rotations
        // durably advance a family only if due work exists - here none - and
        // the chained advance must never certify the foreign durable state.
        let deferred3 = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(!deferred3, "cycle 3 must defer again (same obstruction)");
        let after_foreign = read_deferred_predecessor(dir.path())
            .expect("the record must survive the third deferral");
        assert_eq!(
            after_foreign.committed_digest, commit_before.component_digest,
            "the committed predecessor must be preserved"
        );
        assert!(
            !after_foreign.states.contains(&foreign_reload),
            "the foreign durable state left by another writer must never be \
             certified into the record"
        );
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "deferred cycles must not touch the pending ledger"
        );
    }

    #[tokio::test]
    async fn rotation_cycle_lock_serializes_writers() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_CYCLE_LOCK_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::rotation_cycle_lock_serializes_writers",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let now = chrono::Utc::now().timestamp();

        let pq_store = Arc::new(load_or_init_ml_dsa_key_store(dir.path(), &config));
        let active_slot = KeySlot::new(
            SigningKey::generate(&mut rand::rngs::OsRng),
            now - 14 * 86400,
            now + 6 * 3600,
        );
        persist_slot(dir.path(), "active", &active_slot).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(active_slot),
            lead: None,
        });

        let ca: Arc<SigningKey> = Arc::new(SigningKey::from_bytes(&[0x5A; 32]));
        initialize_composite_key_set(dir.path(), &ed_store, &pq_store, Arc::clone(&ca), 30)
            .await
            .expect("healthy starting authority must initialize composite");
        let (commit_before, _) = read_committed_composite_ledger(dir.path()).unwrap();
        let pending_before = std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap();

        // A competing writer holding the cycle lock makes this tick skip
        // entirely: no rotation, no record lifecycle, no publication.
        let cycle_lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(cycle_lock_path(dir.path()))
            .unwrap();
        use nix::fcntl::{flock, FlockArg};
        use std::os::fd::AsRawFd as _;
        flock(cycle_lock.as_raw_fd(), FlockArg::LockExclusiveNonblock).unwrap();

        let extra = RotationStores {
            es256: None,
            ml_dsa: Some(Arc::clone(&pq_store)),
            composite_ca_key: Arc::clone(&ca),
        };
        let skipped = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(
            skipped,
            "a lock-contended cycle must skip without reporting not-restored"
        );
        let (commit_skipped, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(commit_skipped.version, commit_before.version);
        assert_eq!(
            std::fs::read(dir.path().join("jwt-composite-pairs.json")).unwrap(),
            pending_before,
            "a skipped cycle must not touch the pending ledger"
        );

        // Lock released: the next tick runs normally and may publish.
        drop(cycle_lock);
        let ran = run_rotation_cycle(&config, dir.path(), &ed_store, &extra, now).await;
        assert!(ran, "an uncontended cycle must run and stay restored");
        let (commit_ran, _) = read_committed_composite_ledger(dir.path()).unwrap();
        assert!(
            commit_ran.version >= commit_before.version,
            "an uncontended cycle must not regress the committed authority"
        );
    }

    #[tokio::test]
    async fn committed_marker_failures_never_reinitialize_from_mutable_authority() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_FAIL_CLOSED_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::committed_marker_failures_never_reinitialize_from_mutable_authority",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        enum ImmutableFailure {
            Missing,
            Corrupt,
            Mismatched,
            Unavailable,
        }

        for failure in [
            ImmutableFailure::Missing,
            ImmutableFailure::Corrupt,
            ImmutableFailure::Mismatched,
            ImmutableFailure::Unavailable,
        ] {
            let dir = TempDir::new().unwrap();
            let config = test_config();
            let ca = Arc::new(SigningKey::from_bytes(&[0x6a; 32]));
            let ed = load_or_init_key_store(dir.path(), &config);
            let pq = load_or_init_ml_dsa_key_store(dir.path(), &config);
            initialize_composite_key_set(dir.path(), &ed, &pq, Arc::clone(&ca), 300)
                .await
                .unwrap();

            let marker_bytes = std::fs::read(composite_committed_path(dir.path())).unwrap();
            let commit: CompositeCommit = serde_json::from_slice(&marker_bytes).unwrap();
            let immutable = composite_committed_ledger_path(dir.path(), &commit);
            let mut pending: CompositeLedger =
                serde_json::from_slice(&std::fs::read(composite_ledger_path(dir.path())).unwrap())
                    .unwrap();
            pending.version = pending.version.saturating_add(1);
            pending.component_digest = "staged-component-C".to_owned();
            let pending_bytes = serde_json::to_vec(&pending).unwrap();
            std::fs::write(composite_ledger_path(dir.path()), &pending_bytes).unwrap();

            match failure {
                ImmutableFailure::Missing => std::fs::remove_file(&immutable).unwrap(),
                ImmutableFailure::Corrupt => std::fs::write(&immutable, b"{").unwrap(),
                ImmutableFailure::Mismatched => {
                    std::fs::write(&immutable, &pending_bytes).unwrap();
                }
                ImmutableFailure::Unavailable => {
                    std::fs::remove_file(&immutable).unwrap();
                    std::fs::create_dir(&immutable).unwrap();
                }
            }

            let error = initialize_composite_key_set(dir.path(), &ed, &pq, Arc::clone(&ca), 300)
                .await
                .expect_err("marker-selected authority failure must fail closed");
            assert!(!error.to_string().is_empty());
            assert_eq!(
                std::fs::read(composite_committed_path(dir.path())).unwrap(),
                marker_bytes,
                "failure rewrote the committed marker"
            );
            assert_eq!(
                std::fs::read(composite_ledger_path(dir.path())).unwrap(),
                pending_bytes,
                "failure published over staged mutable C"
            );
            assert!(
                !composite_committed_ledger_path(
                    dir.path(),
                    &CompositeCommit {
                        version: pending.version,
                        component_digest: pending.component_digest.clone(),
                    },
                )
                .exists(),
                "failure materialized staged C as committed authority"
            );
        }
    }

    #[tokio::test]
    async fn first_bootstrap_and_locked_legacy_migration_are_distinct() {
        const ISOLATED: &str = "HYPRSTREAM_COMPOSITE_LEGACY_MIGRATION_TEST";
        if std::env::var_os(ISOLATED).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "auth::key_rotation::tests::first_bootstrap_and_locked_legacy_migration_are_distinct",
                    "--nocapture",
                ])
                .env(ISOLATED, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let dir = TempDir::new().unwrap();
        let config = test_config();
        let ca = Arc::new(SigningKey::from_bytes(&[0x6b; 32]));
        let ed = load_or_init_key_store(dir.path(), &config);
        let pq = load_or_init_ml_dsa_key_store(dir.path(), &config);

        assert!(!composite_committed_path(dir.path()).exists());
        initialize_composite_key_set(dir.path(), &ed, &pq, Arc::clone(&ca), 300)
            .await
            .unwrap();
        let commit: CompositeCommit =
            serde_json::from_slice(&std::fs::read(composite_committed_path(dir.path())).unwrap())
                .unwrap();
        let immutable = composite_committed_ledger_path(dir.path(), &commit);
        let committed_bytes = std::fs::read(&immutable).unwrap();

        // Recreate the round-three layout, then restart through the production
        // initializer. It must durably pin B before mutable C can be staged.
        std::fs::remove_file(&immutable).unwrap();
        initialize_composite_key_set(dir.path(), &ed, &pq, ca, 300)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&immutable).unwrap(), committed_bytes);

        let mut pending: CompositeLedger = serde_json::from_slice(&committed_bytes).unwrap();
        pending.version = pending.version.saturating_add(1);
        pending.component_digest = "staged-component-C".to_owned();
        std::fs::write(
            composite_ledger_path(dir.path()),
            serde_json::to_vec(&pending).unwrap(),
        )
        .unwrap();
        let (_, selected) = read_committed_composite_ledger(dir.path()).unwrap();
        assert_eq!(selected.version, commit.version);
        assert_eq!(selected.component_digest, commit.component_digest);
    }

    #[tokio::test]
    async fn composite_ledger_survives_joint_rotation_restart_and_drain_expiry() {
        use super::ml_dsa_rotation::*;

        let dir = TempDir::new().unwrap();
        let now = chrono::Utc::now().timestamp();
        let drain_secs = 600;
        let ca = Arc::new(SigningKey::from_bytes(&[91; 32]));
        let old_ed = KeySlot::new(SigningKey::from_bytes(&[92; 32]), now - 60, now + 60);
        let old_pq = generate_ml_dsa_slot(now - 60, now + 60);
        persist_slot(dir.path(), "active", &old_ed).unwrap();
        persist_ml_dsa_slot(dir.path(), "active", &old_pq).unwrap();
        let ed_store = SigningKeyStore::new(KeySlots {
            drain: None,
            active: Some(old_ed.clone()),
            lead: None,
        });
        let pq_store = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: None,
            active: Some(old_pq.clone()),
            lead: None,
        });
        initialize_composite_key_set(
            dir.path(),
            &ed_store,
            &pq_store,
            Arc::clone(&ca),
            drain_secs,
        )
        .await
        .unwrap();
        let initial = hyprstream_rpc::auth::global_composite_key_set().snapshot();
        let initial_component_digest = initial.component_digest().to_owned();
        let old_pair = initial
            .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::OAuth)
            .unwrap();
        let old_kid = old_pair.kid().to_owned();
        let (old_pq_signing, old_ed_signing) = old_pair.signing_keys().unwrap();
        let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 300)
            .with_issuer("https://local".to_owned())
            .with_audience(Some("https://resource".to_owned()));
        let token = crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
            &claims,
            &old_pq_signing,
            &old_ed_signing,
        );

        // Deterministic joint promotion: old exact pair becomes drain and a
        // new exact pair becomes active in one publication.
        // Keep the active generation outside the configured lead window so a
        // restart reloads this exact persisted component generation.
        let new_ed = KeySlot::new(SigningKey::from_bytes(&[93; 32]), now, now + 15 * 86_400);
        let new_pq = generate_ml_dsa_slot(now, now + 15 * 86_400);
        persist_slot(dir.path(), "drain", &old_ed).unwrap();
        persist_slot(dir.path(), "active", &new_ed).unwrap();
        persist_ml_dsa_slot(dir.path(), "drain", &old_pq).unwrap();
        persist_ml_dsa_slot(dir.path(), "active", &new_pq).unwrap();
        let rotated_ed = SigningKeyStore::new(KeySlots {
            drain: Some(old_ed),
            active: Some(new_ed),
            lead: None,
        });
        let rotated_pq = MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain: Some(old_pq),
            active: Some(new_pq),
            lead: None,
        });
        refresh_composite_key_set(
            dir.path(),
            &rotated_ed,
            &rotated_pq,
            Arc::clone(&ca),
            drain_secs,
            &initial_component_digest,
        )
        .await
        .unwrap();
        let rotated = hyprstream_rpc::auth::global_composite_key_set().snapshot();
        let retained = rotated
            .pair(&old_kid)
            .expect("old exact pair retained as drain");
        let dispatch = hyprstream_rpc::auth::parse_composite_dispatch(&token, &["at+jwt"]).unwrap();
        hyprstream_rpc::auth::jwt::decode_composite(
            &token,
            retained.ml_dsa(),
            retained.ed25519(),
            Some("https://resource"),
            &dispatch,
        )
        .unwrap();

        // Restart from component files plus the exact persisted association.
        let reloaded_ed = load_or_init_key_store(dir.path(), &test_config());
        let reloaded_pq = load_or_init_ml_dsa_key_store(dir.path(), &test_config());
        initialize_composite_key_set(
            dir.path(),
            &reloaded_ed,
            &reloaded_pq,
            Arc::clone(&ca),
            drain_secs,
        )
        .await
        .unwrap();
        let restarted = hyprstream_rpc::auth::global_composite_key_set().snapshot();
        assert!(restarted.pair(&old_kid).is_some());
        let jwks_kids: Vec<_> = restarted
            .pairs()
            .iter()
            .map(|pair| {
                crate::auth::jwt::composite_jwk(pair.ml_dsa(), pair.ed25519())["kid"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert!(jwks_kids.contains(&old_kid));

        // Removing the expired drain from component authority requires a new
        // committed publication. A mutable-ledger edit alone is pending input
        // and must never revoke the marker-selected live snapshot.
        reloaded_ed.0.write().await.drain = None;
        reloaded_pq.0.write().await.drain = None;
        refresh_composite_key_set(
            dir.path(),
            &reloaded_ed,
            &reloaded_pq,
            ca,
            drain_secs,
            restarted.component_digest(),
        )
        .await
        .unwrap();
        assert!(hyprstream_rpc::auth::global_composite_key_set()
            .snapshot()
            .pair(&old_kid)
            .is_none());
    }
}
