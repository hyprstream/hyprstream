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
                Err(metadata_error)
                    if metadata_error.kind() == std::io::ErrorKind::NotFound =>
                {
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
                Err(metadata_error)
                    if metadata_error.kind() == std::io::ErrorKind::NotFound =>
                {
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
                    let pending: CompositeLedger = serde_json::from_slice(&std::fs::read(
                        composite_ledger_path(&secrets_dir),
                    )?)?;
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
                    anyhow::ensure!(
                        pending.component_digest == local_digest,
                        "pending composite ledger does not match local component authority"
                    );
                    let _staged =
                        ledger_pairs_from_local_keys(&pending, &ed, &pq, ca_key, true)?;
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
                if let Err(error) = load {
                    tracing::warn!("composite authority reload failed closed: {error:#}");
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
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
        let ed_signing =
            if URL_SAFE_NO_PAD.encode(ca_key.verifying_key().to_bytes()) == record.ed25519_public {
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
                "committed active composite signing pair is unavailable after restart"
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
            key_set.publish(
                persisted_version,
                digest,
                pairs,
            )?;
        }
        #[cfg(not(test))]
        start_composite_authority_subscription(
            secrets_dir.to_path_buf(),
            ca_key.verifying_key(),
        )?;
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
    ES256_SIGNING_STORE
        .get_or_init(|| Arc::new(load_or_init_es256_key_store(secrets_dir, config)))
        .clone()
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
    let should_gen_lead =
        lead.is_none() && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
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
    let lead = if should_gen_lead {
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
    if slots
        .drain
        .as_ref()
        .is_some_and(|d| now >= d.exp + drain_secs)
    {
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
                info!(
                    "Generated new lead JWT key (kid={}, nbf={})",
                    new_lead.kid(),
                    new_lead.nbf
                );
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
    pub composite_ca_key: Arc<SigningKey>,
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
            let expected_component_digest = if let Some(ref ml_dsa) = extra.ml_dsa {
                Some(
                    component_state_digest(&store, ml_dsa, extra.composite_ca_key.verifying_key())
                        .await,
                )
            } else {
                None
            };
            rotate_jwt_keys(&config, &secrets_dir, &store, now).await;
            // Keep the shared HTTP validator's published Ed25519 key set current
            // after every rotation (drain evicted / lead promoted → active).
            refresh_ed25519_verifying_keys(&store).await;
            if let Some(ref es256) = extra.es256 {
                rotate_es256_keys(&config, &secrets_dir, es256, now).await;
            }
            if let (Some(ref ml_dsa), Some(expected_component_digest)) =
                (&extra.ml_dsa, expected_component_digest.as_deref())
            {
                rotate_ml_dsa_keys(&config, &secrets_dir, ml_dsa, now).await;
                refresh_ml_dsa_verifying_keys(ml_dsa).await;
                if let Err(error) = refresh_composite_key_set(
                    &secrets_dir,
                    &store,
                    ml_dsa,
                    Arc::clone(&extra.composite_ca_key),
                    config.drain_secs(),
                    expected_component_digest,
                )
                .await
                {
                    warn!(
                        "composite key-set publication failed; retaining prior authority: {error}"
                    );
                }
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
        self.0
            .read()
            .await
            .active
            .as_ref()
            .map(|s| Arc::clone(&s.key))
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

fn persist_es256_slot(secrets_dir: &Path, name: &str, slot: &Es256KeySlot) -> anyhow::Result<()> {
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

    let should_gen_lead =
        lead.is_none() && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
    if should_gen_lead {
        let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
        let lead_exp = lead_nbf + active_secs;
        let slot = generate_es256_slot(lead_nbf, lead_exp);
        if let Err(e) = persist_es256_slot(secrets_dir, "lead", &slot) {
            warn!("Could not persist lead ES256 key: {e}");
        }
    }

    let lead = if should_gen_lead {
        load_es256_slot(secrets_dir, "lead")
    } else {
        lead
    };
    Es256SigningKeyStore::new(Es256KeySlots {
        drain,
        active,
        lead,
    })
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

        let should_gen_lead =
            lead.is_none() && active.as_ref().is_some_and(|a| a.exp - now < lead_secs);
        if should_gen_lead {
            let lead_nbf = active.as_ref().map(|a| a.exp).unwrap_or(now) - lead_secs;
            let lead_exp = lead_nbf + active_secs;
            let slot = generate_ml_dsa_slot(lead_nbf, lead_exp);
            if let Err(e) = persist_ml_dsa_slot(secrets_dir, "lead", &slot) {
                warn!("Could not persist lead ML-DSA key: {e}");
            }
        }

        let lead = if should_gen_lead {
            load_ml_dsa_slot(secrets_dir, "lead")
        } else {
            lead
        };
        MlDsaSigningKeyStore::new(MlDsaKeySlots {
            drain,
            active,
            lead,
        })
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

pub use ml_dsa_rotation::{
    load_or_init_ml_dsa_key_store, rotate_ml_dsa_keys, MlDsaKeySlot, MlDsaKeySlots,
    MlDsaSigningKeyStore,
};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    struct ProductionAuthorityVerifier {
        transport: hyprstream_rpc::transport::TransportConfig,
        signing_key: hyprstream_rpc::prelude::SigningKey,
        key_source: Arc<hyprstream_rpc::auth::ClusterKeySource>,
        accepted: Arc<std::sync::atomic::AtomicBool>,
        error: Arc<parking_lot::Mutex<Option<String>>>,
    }

    #[async_trait::async_trait(?Send)]
    impl crate::services::RequestService for ProductionAuthorityVerifier {
        async fn handle_request(
            &self,
            _ctx: &crate::services::EnvelopeContext,
            _payload: &[u8],
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

        fn pq_signing_key(&self) -> Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey> {
            None
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
    ) -> anyhow::Result<crate::services::generated::policy_client::PolicyClient> {
        policy_client_for_named_socket(dir, "authority-policy.sock")
    }

    fn policy_client_for_named_socket(
        dir: &Path,
        socket: &str,
    ) -> anyhow::Result<crate::services::generated::policy_client::PolicyClient> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
        let rpc = RpcClientImpl::new(
            LocalSigner::new(SigningKey::from_bytes(&[0x76; 32])),
            LazyUdsTransport::new(dir.join(socket)),
            Some(SigningKey::from_bytes(&[0x73; 32]).verifying_key()),
        )
        .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical);
        Ok(crate::services::generated::policy_client::PolicyClient::new(Arc::new(rpc)))
    }

    fn authority_process_dir() -> Option<PathBuf> {
        std::env::var_os("HYPRSTREAM_COMPOSITE_PRODUCTION_TEST_DIR").map(PathBuf::from)
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
            start_composite_authority_subscription(dir.to_path_buf(), authority_ca_key().verifying_key())?;
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

    fn wait_path(path: &Path) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
                let discovery = crate::services::DiscoveryClient::for_local_endpoint_bootstrap(
                    "inproc://multiprocess-unused-discovery",
                    SigningKey::from_bytes(&[0x76; 32]),
                    SigningKey::from_bytes(&[0x77; 32]).verifying_key(),
                    None,
                )?;
                let state = Arc::new(crate::services::oauth::state::OAuthState::new(
                    &config,
                    policy_client_for_socket(&dir)?,
                    discovery,
                    authority_ca_key().verifying_key().to_bytes(),
                ));
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
                        expires_at: std::time::Instant::now() + std::time::Duration::from_secs(60),
                        username: "multiprocess-oauth".to_owned(),
                        verifying_key: None,
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
                .with_default_audience("multiprocess".to_owned());
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
                    .issue_token(&crate::services::generated::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("multiprocess-policy".to_owned()),
                        user_pub_key: None,
                        dpop_jkt: None,
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
                    hyprstream_rpc::transport::TransportConfig::ipc(
                        dir.join("stale-policy.sock"),
                    ),
                )
                .with_default_audience("multiprocess".to_owned());
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join("stale-policy.sock"));
                let result = policy_client_for_named_socket(&dir, "stale-policy.sock")?
                    .issue_token(&crate::services::generated::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("stale-policy".to_owned()),
                        user_pub_key: None,
                        dpop_jkt: None,
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
        runtime.block_on(async {
            let config = test_config();
            let ed = load_or_init_key_store(&dir, &config);
            let pq = load_or_init_ml_dsa_key_store(&dir, &config);
            mutate_authority_for_failure(&ed, &pq, Some(&dir)).await;
            let expected = std::fs::read_to_string(dir.join("crash-expected-digest"))?;
            refresh_composite_key_set(
                &dir,
                &ed,
                &pq,
                authority_ca_key(),
                300,
                &expected,
            )
            .await
        }).unwrap();
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
                let error = refresh_composite_key_set(
                    &dir,
                    &ed,
                    &pq,
                    authority_ca_key(),
                    300,
                    &expected,
                )
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
                .with_default_audience("multiprocess".to_owned());
                let shutdown = Arc::new(tokio::sync::Notify::new());
                let shutdown_server = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    use hyprstream_rpc::service::Spawnable as _;
                    Box::new(service).run(shutdown_server, None).unwrap();
                });
                wait_path(&dir.join(socket));
                let token = policy_client_for_named_socket(&dir, socket)?
                    .issue_token(&crate::services::generated::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("restart-policy".to_owned()),
                        user_pub_key: None,
                        dpop_jkt: None,
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
            // short even when the worktree has a long absolute path.
            let test_tmp = std::env::temp_dir().join("hs-composite-tests");
            std::fs::create_dir_all(&test_tmp).unwrap();
            let dir = TempDir::new_in(test_tmp).unwrap();
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
        let old_token = runtime.block_on(async {
            policy_client_for_socket(dir.path())?
                .issue_token(&crate::services::generated::policy_client::IssueToken {
                    requested_scopes: Some(vec!["read".to_owned()]),
                    ttl: Some(60),
                    audience: Some("multiprocess".to_owned()),
                    subject: Some("pre-rotation-policy".to_owned()),
                    user_pub_key: None,
                    dpop_jkt: None,
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
                .ok_or_else(|| anyhow::anyhow!("pre-rotation OAuth response omitted access_token"))?;
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
        }).unwrap();
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
        std::fs::write(
            dir.path().join("target-authority.json"),
            serde_json::to_vec(&committed).unwrap(),
        )
        .unwrap();
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
                    .issue_token(&crate::services::generated::policy_client::IssueToken {
                        requested_scopes: Some(vec!["read".to_owned()]),
                        ttl: Some(60),
                        audience: Some("multiprocess".to_owned()),
                        subject: Some("timeout-policy".to_owned()),
                        user_pub_key: None,
                        dpop_jkt: None,
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
        let committed_after_timeout: CompositeCommit = serde_json::from_slice(
            &std::fs::read(composite_committed_path(dir.path())).unwrap(),
        )
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
        assert_ne!(pending_after_crash.component_digest, committed.component_digest);
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
            initialize_composite_key_set(
                dir.path(),
                &ed,
                &pq,
                Arc::clone(&ca),
                300,
            )
            .await
            .unwrap();

            let marker_bytes = std::fs::read(composite_committed_path(dir.path())).unwrap();
            let commit: CompositeCommit = serde_json::from_slice(&marker_bytes).unwrap();
            let immutable = composite_committed_ledger_path(dir.path(), &commit);
            let mut pending: CompositeLedger = serde_json::from_slice(
                &std::fs::read(composite_ledger_path(dir.path())).unwrap(),
            )
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

            let error = initialize_composite_key_set(
                dir.path(),
                &ed,
                &pq,
                Arc::clone(&ca),
                300,
            )
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
        let commit: CompositeCommit = serde_json::from_slice(
            &std::fs::read(composite_committed_path(dir.path())).unwrap(),
        )
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

        let mut pending: CompositeLedger =
            serde_json::from_slice(&committed_bytes).unwrap();
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
