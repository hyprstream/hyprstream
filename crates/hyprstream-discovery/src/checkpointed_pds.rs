//! Concrete reader for daemon-authenticated, monotonic-checkpointed accepted
//! `did:at9p` state. This module is private so production authority cannot be
//! implemented, constructed, or installed by a downstream crate.

use anyhow::{bail, Context as _, Result};
use hyprstream_pds::at9p::h512;
use hyprstream_pds::at9p_duplicity::{AcceptedAt9pState, Watermark};
use hyprstream_pds::at9p_gate::DID_AT9P_PREFIX;
use std::path::Path;
#[cfg(feature = "rocksdb")]
use std::path::PathBuf;
use std::sync::Arc;

const AT9P_STATE_MAGIC: &[u8; 8] = b"AT9PST02";
const AT9P_STATE_HEADER_LEN: usize = 8 + 1 + 8 + 64 + 1 + 4;
const AT9P_ACCEPTANCE_MAGIC: &[u8; 8] = b"AT9PAC01";
const ED25519_KEY_LEN: usize = 32;
const ED25519_SIGNATURE_LEN: usize = 64;
const AT9P_ACCEPTANCE_PREFIX_LEN: usize =
    AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN + ED25519_SIGNATURE_LEN;
const AT9P_ACCEPTANCE_KEY_AAD: &[u8] = b"hyprstream-at9p-acceptance-key/1";
const AT9P_ACCEPTANCE_STATE_AAD: &[u8] = b"hyprstream-at9p-accepted-state/1";
const AT9P_CHECKPOINT_MAGIC: &[u8; 8] = b"AT9PCK01";
const AT9P_CHECKPOINT_PAYLOAD_LEN: usize = 8 + 8 + 64 + 1 + 64;
const AT9P_CHECKPOINT_LEN: usize = AT9P_CHECKPOINT_PAYLOAD_LEN + ED25519_SIGNATURE_LEN;
const AT9P_CHECKPOINT_AAD: &[u8] = b"hyprstream-at9p-monotonic-checkpoint/1";

#[cfg(feature = "rocksdb")]
pub(super) struct CheckpointedPdsAcceptedStateSource {
    path: PathBuf,
    acceptance_identity: Arc<dyn AcceptanceVerifier>,
    network_bootstrap: bool,
}

trait AcceptanceVerifier: Send + Sync {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()>;
}

impl AcceptanceVerifier for crate::service::RegistryDeploymentVerifier {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()> {
        self.verify_strict(message, signature)
    }
}

#[cfg(test)]
impl AcceptanceVerifier for ed25519_dalek::VerifyingKey {
    fn verify_strict(&self, message: &[u8], signature: &ed25519_dalek::Signature) -> Result<()> {
        self.verify_strict(message, signature).map_err(Into::into)
    }
}

#[cfg(feature = "rocksdb")]
impl CheckpointedPdsAcceptedStateSource {
    pub(super) fn open(
        path: &Path,
        acceptance_identity: crate::service::RegistryDeploymentVerifier,
    ) -> Result<Self> {
        // The monotonic checkpoint history is security state. A missing
        // database is indistinguishable from deletion of that history, so it
        // must never be recreated at resolver bootstrap. First boot is an
        // explicit deployment-provisioning step before services may start.
        let _probe = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)
            .with_context(|| format!("failed to open checkpointed PDS store at {path:?}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            acceptance_identity: Arc::new(acceptance_identity),
            network_bootstrap: false,
        })
    }

    #[cfg(test)]
    pub(super) fn open_test(
        path: &Path,
        acceptance_identity: ed25519_dalek::VerifyingKey,
    ) -> Result<Self> {
        let _probe = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)
            .with_context(|| format!("failed to open checkpointed PDS store at {path:?}"))?;
        Ok(Self { path: path.to_path_buf(), acceptance_identity: Arc::new(acceptance_identity), network_bootstrap: false })
    }

    pub(super) fn with_network_bootstrap(mut self, required: bool) -> Self {
        self.network_bootstrap = required;
        self
    }

    fn bootstrap_states(&self) -> Result<Vec<AcceptedAt9pState>> {
        let db = rocksdb::DB::open_for_read_only(&readonly_opts(), &self.path, false)?;
        let mut states = Vec::new();
        for entry in db.prefix_iterator(b"at9p-state\0") {
            let (key, _) = entry?;
            let Some(subject) = key.strip_prefix(b"at9p-state\0") else { break; };
            let subject = std::str::from_utf8(subject)?;
            if let Some(state) = load_at9p_state_from_db(&db, subject, self.acceptance_identity.as_ref())? {
                states.push(state);
            }
        }
        Ok(states)
    }

    pub(super) fn accepted_state(&self, did: &str) -> Result<Option<AcceptedAt9pState>> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow::anyhow!("identifier is not did:at9p: {did:?}"))?;
        let db = rocksdb::DB::open_for_read_only(&readonly_opts(), &self.path, false)?;
        let state = load_at9p_state_from_db(&db, subject, self.acceptance_identity.as_ref())?;
        if let Some(state) = &state {
            anyhow::ensure!(state.did == did, "accepted at9p state DID mismatch");
        }
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL (RDS) backend — the same verified envelopes in the shared `pds_kv`
// BYTEA shell. Compiled only with the `postgres` feature.
// ---------------------------------------------------------------------------

/// Accepted-state authority over the networked RDS Postgres store, selected
/// when the deployment's `[rds]` binding is configured. Uses the same
/// `hyprstream_pds::pgsql_kv::PgKv` connection/TLS/pool layer as the app
/// record store and the same [`verify_state_pair`] verifier as the RocksDB
/// reader — no divergent verification.
#[cfg(feature = "postgres")]
pub(super) struct CheckpointedPgAcceptedStateSource {
    kv: hyprstream_pds::pgsql_kv::PgKv,
    acceptance_identity: Arc<dyn AcceptanceVerifier>,
    network_bootstrap: bool,
}

#[cfg(feature = "postgres")]
impl CheckpointedPgAcceptedStateSource {
    /// READ-ONLY connect to the provisioned records store. FATAL on any error
    /// (including an unprovisioned store): a configured PostgreSQL authority
    /// never falls back to the local RocksDB store, and resolver startup never
    /// recreates security history.
    pub(super) fn connect(
        records: &hyprstream_pds::rds::RdsConfig,
        acceptance_identity: crate::service::RegistryDeploymentVerifier,
    ) -> Result<Self> {
        let kv = records
            .connect_kv_readonly()
            .context("failed to open checkpointed PDS Postgres store")?;
        Ok(Self {
            kv,
            acceptance_identity: Arc::new(acceptance_identity),
            network_bootstrap: false,
        })
    }

    #[cfg(test)]
    pub(super) fn connect_test(
        kv: hyprstream_pds::pgsql_kv::PgKv,
        acceptance_identity: ed25519_dalek::VerifyingKey,
    ) -> Self {
        Self {
            kv,
            acceptance_identity: Arc::new(acceptance_identity),
            network_bootstrap: false,
        }
    }

    pub(super) fn with_network_bootstrap(mut self, required: bool) -> Self {
        self.network_bootstrap = required;
        self
    }

    /// The snapshot pair read for one subject — the Postgres counterpart of
    /// `load_at9p_state_from_db`'s `rocksdb::DB::snapshot()` (one read-only
    /// REPEATABLE READ transaction inside `get_batch`).
    fn load_state(&self, subject: &str) -> Result<Option<AcceptedAt9pState>> {
        let mut values = self
            .kv
            .get_batch(&[state_key(subject), checkpoint_key(subject)])
            .context("checkpointed PDS Postgres read failed")?;
        let checkpoint = values.pop().flatten();
        let envelope = values.pop().flatten();
        verify_state_pair(subject, envelope, checkpoint, self.acceptance_identity.as_ref())
    }

    fn bootstrap_states(&self) -> Result<Vec<AcceptedAt9pState>> {
        let state_prefix = b"at9p-state\0".to_vec();
        let state_upper = hyprstream_pds::pgsql_kv::prefix_upper_bound(&state_prefix)
            .ok_or_else(|| anyhow::anyhow!("at9p state prefix has no upper bound"))?;
        let checkpoint_prefix = b"at9p-checkpoint\0".to_vec();
        let checkpoint_upper = hyprstream_pds::pgsql_kv::prefix_upper_bound(&checkpoint_prefix)
            .ok_or_else(|| anyhow::anyhow!("at9p checkpoint prefix has no upper bound"))?;
        let snap = self
            .kv
            .read_snapshot(&[(state_prefix, state_upper), (checkpoint_prefix, checkpoint_upper)], &[])
            .context("checkpointed PDS Postgres snapshot read failed")?;
        let mut ranges = snap.ranges.into_iter();
        let (states, checkpoints) = match (ranges.next(), ranges.next()) {
            (Some(states), Some(checkpoints)) => (states, checkpoints),
            _ => bail!("RDS read_snapshot returned fewer ranges than requested"),
        };
        let checkpoints: std::collections::BTreeMap<Vec<u8>, Vec<u8>> =
            checkpoints.into_iter().collect();
        let mut out = Vec::new();
        for (key, envelope) in states {
            let Some(subject) = key.strip_prefix(b"at9p-state\0".as_slice()) else { continue; };
            let subject = std::str::from_utf8(subject)?;
            let checkpoint = checkpoints.get(checkpoint_key(subject).as_slice()).cloned();
            if let Some(state) =
                verify_state_pair(subject, Some(envelope), checkpoint, self.acceptance_identity.as_ref())?
            {
                out.push(state);
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "postgres")]
impl super::service::AcceptedStateSource for CheckpointedPgAcceptedStateSource {
    fn accepted_state(&self, did: &str) -> Result<Option<AcceptedAt9pState>> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow::anyhow!("identifier is not did:at9p: {did:?}"))?;
        let state = self.load_state(subject)?;
        if let Some(state) = &state {
            anyhow::ensure!(state.did == did, "accepted at9p state DID mismatch");
        }
        Ok(state)
    }

    fn accepted_states(&self) -> Result<Vec<AcceptedAt9pState>> {
        self.bootstrap_states()
    }

    fn bootstrap_endpoints(&self, service_name: &str) -> Result<Option<Vec<crate::state_store::AnnouncedEndpoint>>> {
        if !self.network_bootstrap || !matches!(service_name, "discovery" | "policy") {
            return Ok(None);
        }
        let key = hyprstream_service::global_trust_store().resolve_one(service_name)
            .ok_or_else(|| anyhow::anyhow!("trusted bootstrap {service_name} response key is missing"))?;
        super::service::project_bootstrap_endpoint(&self.bootstrap_states()?, service_name, &key)
            .map(|endpoint| Some(vec![endpoint]))
    }
}

/// Select the deployment's accepted-state authority backend from the resolved
/// records-role binding. A configured PostgreSQL binding selects the networked
/// store and FAILS CLOSED on any error — there is never a silent local
/// fallback (a local store that missed the accepted-state history would
/// silently deny every deployment identity). Unconfigured keeps the local
/// RocksDB checkpoint store.
#[cfg_attr(not(feature = "postgres"), allow(unused_variables))]
pub(super) fn open_deployment_accepted_state_source(
    store_path: &Path,
    identity: crate::service::RegistryDeploymentVerifier,
    records: &hyprstream_pds::rds::RdsConfig,
    network_bootstrap: bool,
) -> Result<Arc<dyn super::service::AcceptedStateSource>> {
    if records.is_configured() {
        #[cfg(feature = "postgres")]
        {
            return Ok(Arc::new(
                CheckpointedPgAcceptedStateSource::connect(records, identity)?
                    .with_network_bootstrap(network_bootstrap),
            ));
        }
        #[cfg(not(feature = "postgres"))]
        bail!(
            "records RDS Postgres is configured but this binary lacks the discovery `postgres` feature;              refusing to silently fall back to the local store"
        );
    }
    #[cfg(feature = "rocksdb")]
    {
        Ok(Arc::new(
            CheckpointedPdsAcceptedStateSource::open(store_path, identity)?
                .with_network_bootstrap(network_bootstrap),
        ))
    }
    #[cfg(not(feature = "rocksdb"))]
    {
        // Fail closed: without `rocksdb` there is no local concrete authority.
        let _ = store_path;
        bail!("checkpointed PDS accepted-state authority requires the `rocksdb` feature")
    }
}

/// RocksDB key for the first-boot provisioning marker.
///
/// Written by [`initialize_deployment_store`] (and any other path that creates
/// a fresh empty store) and deleted by the registry service **in the same
/// synchronous `WriteBatch` as the first accepted-state commit** — so the
/// lifecycle transition is atomic with provisioning completion, not a
/// best-effort filesystem side-effect. Its presence alongside an empty store
/// is the ONLY lifecycle evidence that distinguishes a genuine first boot
/// (store freshly provisioned, no accepted state written yet) from a
/// steady-state store that lost its data. The QUIC startup gate reads it via
/// [`PdsRecordStore::first_boot_pending`] in the app crate.
#[cfg(feature = "rocksdb")]
pub const FIRST_BOOT_KEY: &[u8] = b"first-boot-pending";

/// Explicitly create the empty checkpoint store for a newly provisioned
/// deployment. Resolver startup never calls this: a missing store there is
/// treated as lost security history and fails closed.
///
/// Also writes the [`FIRST_BOOT_KEY`] so the QUIC startup gate can recognize
/// this as a genuine first boot (defer-eligible) rather than data loss. The
/// registry deletes the key (in the same RocksDB batch as the first
/// accepted-state commit) once provisioning completes.
#[cfg(feature = "rocksdb")]
pub(crate) fn initialize_deployment_store() -> Result<()> {
    let path = hyprstream_service::deployment_data_dir()?.join("pds-store");
    anyhow::ensure!(
        !path.exists(),
        "refusing to initialize existing checkpointed PDS store at {}",
        path.display()
    );
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    let db = rocksdb::DB::open(&opts, &path)
        .with_context(|| format!("failed to initialize checkpointed PDS store at {path:?}"))?;
    // Write the first-boot marker as a durable RocksDB key (not a filesystem
    // side-effect) so its deletion can be atomic with the first accepted-state
    // commit. A failed put propagates — provisioning must not appear successful
    // if the marker wasn't written.
    db.put(FIRST_BOOT_KEY, b"")
        .with_context(|| format!("failed to write first-boot marker in store at {path:?}"))?;
    Ok(())
}

#[cfg(feature = "rocksdb")]
impl super::service::AcceptedStateSource for CheckpointedPdsAcceptedStateSource {
    fn accepted_state(&self, did: &str) -> Result<Option<AcceptedAt9pState>> {
        self.accepted_state(did)
    }

    /// Every verified accepted state in the store — the roster universe the
    /// derived-admission uniqueness rule (#1652) enumerates. This is the
    /// deployment's provisioned identities AND any foreign record admitted
    /// through the public ingest RPC; provenance is not distinguishable in
    /// the store, so membership alone is never deployment authorization.
    fn accepted_states(&self) -> Result<Vec<AcceptedAt9pState>> {
        self.bootstrap_states()
    }

    fn bootstrap_endpoints(&self, service_name: &str) -> Result<Option<Vec<crate::state_store::AnnouncedEndpoint>>> {
        if !self.network_bootstrap || !matches!(service_name, "discovery" | "policy") {
            return Ok(None);
        }
        let key = hyprstream_service::global_trust_store().resolve_one(service_name)
            .ok_or_else(|| anyhow::anyhow!("trusted bootstrap {service_name} response key is missing"))?;
        super::service::project_bootstrap_endpoint(&self.bootstrap_states()?, service_name, &key)
            .map(|endpoint| Some(vec![endpoint]))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct At9pCheckpoint {
    watermark: Watermark,
    envelope_digest: [u8; 64],
}

#[cfg(feature = "rocksdb")]
fn readonly_opts() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(false);
    opts
}

fn state_key(subject: &str) -> Vec<u8> {
    format!("at9p-state\0{subject}").into_bytes()
}

fn checkpoint_key(subject: &str) -> Vec<u8> {
    format!("at9p-checkpoint\0{subject}").into_bytes()
}

fn acceptance_message(domain: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(domain.len() + payload.len());
    message.extend_from_slice(domain);
    message.extend_from_slice(payload);
    message
}

fn checkpoint_message(subject: &str, payload: &[u8]) -> Result<Vec<u8>> {
    let len = u32::try_from(subject.len()).context("at9p subject exceeds u32")?;
    let mut out = Vec::with_capacity(AT9P_CHECKPOINT_AAD.len() + 4 + subject.len() + payload.len());
    out.extend_from_slice(AT9P_CHECKPOINT_AAD);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(subject.as_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

#[cfg(feature = "rocksdb")]
fn load_at9p_state_from_db(
    db: &rocksdb::DB,
    subject: &str,
    identity: &dyn AcceptanceVerifier,
) -> Result<Option<AcceptedAt9pState>> {
    let snapshot = db.snapshot();
    let envelope = snapshot.get(state_key(subject))?;
    let checkpoint = snapshot.get(checkpoint_key(subject))?;
    verify_state_pair(subject, envelope, checkpoint, identity)
}

/// The backend-independent half of accepted-state reads: both stores hand the
/// state envelope and monotonic checkpoint read from ONE snapshot to this
/// verifier. There is exactly one verification path; a new backend never
/// re-implements it.
fn verify_state_pair(
    subject: &str,
    envelope: Option<Vec<u8>>,
    checkpoint: Option<Vec<u8>>,
    identity: &dyn AcceptanceVerifier,
) -> Result<Option<AcceptedAt9pState>> {
    match (envelope, checkpoint) {
        (None, None) => Ok(None),
        (Some(_), None) => bail!("accepted at9p state exists without its monotonic checkpoint"),
        (None, Some(_)) => bail!("accepted at9p checkpoint exists without its state envelope"),
        (Some(envelope), Some(checkpoint)) => {
            let state = decode_state(subject, &envelope, identity)?;
            let checkpoint = decode_checkpoint(subject, &checkpoint, identity)?;
            anyhow::ensure!(
                checkpoint.watermark == state.watermark(),
                "accepted at9p checkpoint/body watermark mismatch"
            );
            anyhow::ensure!(
                checkpoint.envelope_digest == h512(&envelope),
                "accepted at9p checkpoint/state envelope mismatch"
            );
            Ok(Some(state))
        }
    }
}

fn decode_checkpoint(
    subject: &str,
    bytes: &[u8],
    identity: &dyn AcceptanceVerifier,
) -> Result<At9pCheckpoint> {
    anyhow::ensure!(
        bytes.len() == AT9P_CHECKPOINT_LEN,
        "accepted at9p checkpoint length mismatch"
    );
    let payload = bytes
        .get(..AT9P_CHECKPOINT_PAYLOAD_LEN)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p checkpoint payload missing"))?;
    anyhow::ensure!(
        payload.starts_with(AT9P_CHECKPOINT_MAGIC),
        "accepted at9p checkpoint has bad version"
    );
    let signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(AT9P_CHECKPOINT_PAYLOAD_LEN..)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p checkpoint signature missing"))?
            .try_into()?,
    );
    identity
        .verify_strict(&checkpoint_message(subject, payload)?, &signature)
        .context("accepted at9p monotonic checkpoint signature rejected")?;
    let epoch = u64::from_be_bytes(
        payload
            .get(8..16)
            .ok_or_else(|| anyhow::anyhow!("checkpoint epoch missing"))?
            .try_into()?,
    );
    let record_digest = payload
        .get(16..80)
        .ok_or_else(|| anyhow::anyhow!("checkpoint digest missing"))?
        .try_into()?;
    let terminal = match payload.get(80) {
        Some(0) => false,
        Some(1) => true,
        _ => bail!("invalid checkpoint terminal flag"),
    };
    Ok(At9pCheckpoint {
        watermark: Watermark {
            epoch,
            record_digest,
            terminal,
        },
        envelope_digest: payload
            .get(81..145)
            .ok_or_else(|| anyhow::anyhow!("checkpoint envelope digest missing"))?
            .try_into()?,
    })
}

fn decode_state(
    subject: &str,
    bytes: &[u8],
    acceptance_identity: &dyn AcceptanceVerifier,
) -> Result<AcceptedAt9pState> {
    anyhow::ensure!(
        bytes.len() >= AT9P_ACCEPTANCE_PREFIX_LEN + AT9P_STATE_HEADER_LEN + ED25519_SIGNATURE_LEN,
        "accepted at9p envelope is truncated"
    );
    anyhow::ensure!(
        bytes.starts_with(AT9P_ACCEPTANCE_MAGIC),
        "accepted at9p envelope has bad version"
    );
    let audit_key_bytes: [u8; ED25519_KEY_LEN] = bytes
        .get(AT9P_ACCEPTANCE_MAGIC.len()..AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing audit key"))?
        .try_into()?;
    let key_signature_start = AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN;
    let key_signature_end = key_signature_start + ED25519_SIGNATURE_LEN;
    let key_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(key_signature_start..key_signature_end)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing key certificate"))?
            .try_into()?,
    );
    acceptance_identity
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_key_bytes),
            &key_signature,
        )
        .context("accepted at9p audit-key certificate rejected")?;
    let audit_key = ed25519_dalek::VerifyingKey::from_bytes(&audit_key_bytes)
        .context("accepted at9p envelope has malformed audit key")?;
    let state_signature_start = bytes.len() - ED25519_SIGNATURE_LEN;
    let body = bytes
        .get(AT9P_ACCEPTANCE_PREFIX_LEN..state_signature_start)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing state body"))?;
    let state_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(state_signature_start..)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p envelope missing state signature"))?
            .try_into()?,
    );
    audit_key
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, body),
            &state_signature,
        )
        .context("accepted at9p daemon acceptance signature rejected")?;
    decode_state_body(subject, body)
}

fn decode_state_body(subject: &str, bytes: &[u8]) -> Result<AcceptedAt9pState> {
    anyhow::ensure!(
        bytes.len() >= AT9P_STATE_HEADER_LEN,
        "accepted at9p state is truncated"
    );
    anyhow::ensure!(
        bytes.starts_with(AT9P_STATE_MAGIC),
        "accepted at9p state has bad version"
    );
    let kind = *bytes
        .get(8)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing kind"))?;
    let epoch = u64::from_be_bytes(
        bytes
            .get(9..17)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing epoch"))?
            .try_into()?,
    );
    let digest: [u8; 64] = bytes
        .get(17..81)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing digest"))?
        .try_into()?;
    let terminal = match bytes.get(81) {
        Some(0) => false,
        Some(1) => true,
        _ => bail!("accepted at9p state has invalid terminal flag"),
    };
    let head_len = u32::from_be_bytes(
        bytes
            .get(82..86)
            .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing length"))?
            .try_into()?,
    ) as usize;
    anyhow::ensure!(
        bytes.len() == AT9P_STATE_HEADER_LEN + head_len,
        "accepted at9p state length mismatch"
    );
    let head = bytes
        .get(AT9P_STATE_HEADER_LEN..)
        .ok_or_else(|| anyhow::anyhow!("accepted at9p state missing head"))?;
    let state = match kind {
        0 => AcceptedAt9pState::from_persisted_genesis(head)?,
        1 => AcceptedAt9pState::from_persisted_update(head)?,
        _ => bail!("accepted at9p state has unknown head kind {kind}"),
    };
    anyhow::ensure!(
        state.subject_cid512 == subject,
        "accepted at9p state subject mismatch"
    );
    anyhow::ensure!(
        state.epoch == epoch,
        "accepted at9p state epoch/body mismatch"
    );
    anyhow::ensure!(
        state.head_digest == digest,
        "accepted at9p state digest/body mismatch"
    );
    anyhow::ensure!(
        state.terminal == terminal,
        "accepted at9p state terminal/body mismatch"
    );
    Ok(state)
}

#[cfg(all(test, feature = "rocksdb"))]
pub(super) fn write_test_state(path: &Path, state: &AcceptedAt9pState, identity: &ed25519_dalek::SigningKey) -> Result<()> {
    let audit = ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]);
    let (envelope, checkpoint) = tests::encode_fixture(state, identity, &audit);
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    let db = rocksdb::DB::open(&options, path)?;
    let mut batch = rocksdb::WriteBatch::default();
    batch.put(state_key(&state.subject_cid512), envelope);
    batch.put(checkpoint_key(&state.subject_cid512), checkpoint);
    db.write(batch)?;
    Ok(())
}

/// Remove one accepted state from the test store — the store-wipe half of a
/// simulated service-identity re-initialization (#1652 churn tests: the
/// successor state is then written by [`write_test_state`]).
#[cfg(all(test, feature = "rocksdb"))]
pub(super) fn remove_test_state(path: &Path, state: &AcceptedAt9pState) -> Result<()> {
    let mut options = rocksdb::Options::default();
    options.create_if_missing(true);
    let db = rocksdb::DB::open(&options, path)?;
    let mut batch = rocksdb::WriteBatch::default();
    batch.delete(state_key(&state.subject_cid512));
    batch.delete(checkpoint_key(&state.subject_cid512));
    db.write(batch)?;
    Ok(())
}

#[cfg(all(test, feature = "rocksdb"))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer as _;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType,
        Transport as At9pTransport,
    };
    use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};

    pub(super) fn accepted_state() -> AcceptedAt9pState {
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let (pq_signing, pq_verifying) = ml_dsa_generate_keypair();
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_verifying),
        )
        .expect("hybrid keys");
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://checkpointed-reach")
            .expect("service endpoint");
        let service = ServiceEntry::new("#checkpointed", ServiceType::NinePExport, endpoint)
            .expect("service entry");
        let capsule = sign_capsule(
            CapsuleBody::new(vec![keys], vec![service]).expect("capsule body"),
            &signing,
            &pq_signing,
        )
        .expect("signed capsule");
        AcceptedAt9pState::from_persisted_genesis(&capsule.to_dag_cbor().expect("capsule bytes"))
            .expect("accepted genesis")
    }

    fn encode_body(state: &AcceptedAt9pState) -> Vec<u8> {
        let (kind, head) = match state.head() {
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Genesis(capsule) => {
                (0, capsule.to_dag_cbor().expect("genesis bytes"))
            }
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Update(update) => {
                (1, update.to_dag_cbor().expect("update bytes"))
            }
        };
        let mut out = Vec::new();
        out.extend_from_slice(AT9P_STATE_MAGIC);
        out.push(kind);
        out.extend_from_slice(&state.epoch.to_be_bytes());
        out.extend_from_slice(&state.head_digest);
        out.push(u8::from(state.terminal));
        out.extend_from_slice(
            &u32::try_from(head.len())
                .expect("head length")
                .to_be_bytes(),
        );
        out.extend_from_slice(&head);
        out
    }

    #[test]
    fn update_fixture_encodes_and_decodes_update_kind() {
        let genesis = accepted_state();
        let signing = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let (pq_signing, pq_verifying) = ml_dsa_generate_keypair();
        let keys = HybridKeyPair::new(
            signing.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_verifying),
        )
        .expect("hybrid keys");
        let endpoint = ServiceEndpoint::new(At9pTransport::Iroh, "iroh://updated-reach")
            .expect("service endpoint");
        let service = ServiceEntry::new("#updated", ServiceType::NinePExport, endpoint)
            .expect("service entry");
        let update = sign_update_record(
            genesis.subject_cid512.clone(),
            1,
            genesis.head_digest,
            CapsuleBody::new(vec![keys], vec![service]).expect("capsule body"),
            "2099-01-01T00:00:00Z".to_owned(),
            &signing,
            &pq_signing,
        )
        .expect("signed update");
        let state = AcceptedAt9pState::from_persisted_update(
            &update.to_dag_cbor().expect("update bytes"),
        )
        .expect("accepted update");
        assert_eq!(encode_body(&state)[AT9P_STATE_MAGIC.len()], 1);

        let identity = ed25519_dalek::SigningKey::from_bytes(&[0x45; 32]);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x46; 32]);
        let (envelope, _) = encode_fixture(&state, &identity, &audit);
        let decoded = decode_state(&state.subject_cid512, &envelope, &identity.verifying_key())
            .expect("decode update fixture");
        assert!(matches!(
            decoded.head(),
            hyprstream_pds::at9p_duplicity::AcceptedAt9pHead::Update(_)
        ));
    }

    pub(super) fn encode_fixture(
        state: &AcceptedAt9pState,
        identity: &ed25519_dalek::SigningKey,
        audit: &ed25519_dalek::SigningKey,
    ) -> (Vec<u8>, Vec<u8>) {
        let body = encode_body(state);
        let audit_key = audit.verifying_key().to_bytes();
        let key_signature = identity.sign(&acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_key));
        let state_signature = audit.sign(&acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, &body));
        let mut envelope = Vec::new();
        envelope.extend_from_slice(AT9P_ACCEPTANCE_MAGIC);
        envelope.extend_from_slice(&audit_key);
        envelope.extend_from_slice(&key_signature.to_bytes());
        envelope.extend_from_slice(&body);
        envelope.extend_from_slice(&state_signature.to_bytes());

        let mut payload = Vec::new();
        payload.extend_from_slice(AT9P_CHECKPOINT_MAGIC);
        payload.extend_from_slice(&state.epoch.to_be_bytes());
        payload.extend_from_slice(&state.head_digest);
        payload.push(u8::from(state.terminal));
        payload.extend_from_slice(&h512(&envelope));
        let checkpoint_signature = identity.sign(
            &checkpoint_message(&state.subject_cid512, &payload).expect("checkpoint message"),
        );
        let mut checkpoint = payload;
        checkpoint.extend_from_slice(&checkpoint_signature.to_bytes());
        (envelope, checkpoint)
    }

    #[test]
    fn concrete_checkpointed_pds_reader_accepts_only_matching_daemon_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let identity = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x43; 32]);
        let state = accepted_state();
        let (envelope, checkpoint) = encode_fixture(&state, &identity, &audit);
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let db = rocksdb::DB::open(&options, dir.path()).expect("open fixture store");
        db.put(state_key(&state.subject_cid512), envelope)
            .expect("write accepted state");
        db.put(checkpoint_key(&state.subject_cid512), checkpoint)
            .expect("write checkpoint");
        drop(db);

        let source = CheckpointedPdsAcceptedStateSource::open_test(dir.path(), identity.verifying_key())
            .expect("open checkpointed source");
        let recovered = source
            .accepted_state(&state.did)
            .expect("read checkpointed state")
            .expect("state present");
        assert_eq!(recovered.watermark(), state.watermark());

        let wrong_identity = ed25519_dalek::SigningKey::from_bytes(&[0x44; 32]);
        let rejected =
            CheckpointedPdsAcceptedStateSource::open_test(dir.path(), wrong_identity.verifying_key())
                .expect("open same store")
                .accepted_state(&state.did);
        assert!(
            rejected.is_err(),
            "caller-selected checkpoint identity accepted"
        );
    }
}

// ---------------------------------------------------------------------------
// Postgres authority tests: the resolver-side reader against a real scratch
// database (HYPRSTREAM_POSTGRES_TEST_URL_FILE), plus the fail-closed selection
// contract that needs no database at all.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "postgres", feature = "rocksdb"))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod pg_tests {
    use super::tests::{accepted_state, encode_fixture};
    use super::*;
    use crate::service::AcceptedStateSource as _;
    use hyprstream_pds::pgsql_kv::PgKv;

    /// Read the test database URL from a FILE, never from a direct env var
    /// (metal v1.1 acceptance check 1 — the password never transits the
    /// process environment).
    fn test_url() -> Option<String> {
        let path = std::env::var_os("HYPRSTREAM_POSTGRES_TEST_URL_FILE")?;
        let url = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read test URL file {}: {e}", path.to_string_lossy()));
        let url = url.trim();
        assert!(!url.is_empty(), "test URL file is empty");
        Some(url.to_owned())
    }

    macro_rules! require_db {
        () => {{
            let Some(url) = test_url() else {
                eprintln!(
                    "skipping checkpointed PDS Postgres test: \
                     HYPRSTREAM_POSTGRES_TEST_URL_FILE unset"
                );
                return;
            };
            url
        }};
    }

    fn deployment_verifier(seed: u8) -> (ed25519_dalek::SigningKey, crate::service::RegistryDeploymentVerifier) {
        let ed = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
        let pq = hyprstream_crypto::pq::ml_dsa_sk_from_seed(&[seed; 32]);
        let verifier = crate::service::RegistryDeploymentVerifier::for_test_deployment_root(&ed, &pq)
            .expect("test deployment verifier");
        (ed, verifier)
    }

    /// Run test-side DDL against the scratch server over a direct NoTls
    /// connection (database lifecycle for the unprovisioned gate).
    fn with_direct_connection(url: &str, ddl: impl Fn(tokio_postgres::Client) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("ddl runtime");
        rt.block_on(async move {
            let config: tokio_postgres::Config = url.parse().expect("ddl URL parses");
            let (client, connection) = config
                .connect(tokio_postgres::NoTls)
                .await
                .expect("ddl connect");
            let handle = tokio::spawn(async move {
                if let Err(e) = connection.await {
                    eprintln!("ddl connection error: {e}");
                }
            });
            ddl(client).await;
            handle.await.expect("ddl connection join");
        });
    }

    /// The repro's causal chain: accepted state written to PostgreSQL by the
    /// writer path is served read-only by the resolver authority — same
    /// envelopes, same checkpoint pairing, same verifier.
    #[test]
    fn live_pg_authority_serves_writer_committed_state() {
        let url = require_db!();
        let (identity, _) = deployment_verifier(0x61);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x62; 32]);
        let state = accepted_state();
        let (envelope, checkpoint) = encode_fixture(&state, &identity, &audit);

        // Writer posture: migrates and commits the pair.
        let writer = PgKv::connect_test(&url, "test-cell").expect("writer connect");
        writer.put(&state_key(&state.subject_cid512), &envelope).expect("write state");
        writer.put(&checkpoint_key(&state.subject_cid512), &checkpoint).expect("write checkpoint");
        drop(writer);

        // Resolver posture: read-only, no migration.
        let reader = CheckpointedPgAcceptedStateSource::connect_test(
            PgKv::connect_test_readonly(&url).expect("readonly connect"),
            identity.verifying_key(),
        );
        let recovered = reader
            .accepted_state(&state.did)
            .expect("read accepted state")
            .expect("state present");
        assert_eq!(recovered.watermark(), state.watermark());
        let roster = reader.accepted_states().expect("roster read");
        assert!(
            roster.iter().any(|s| s.subject_cid512 == state.subject_cid512),
            "roster must include the writer-committed state"
        );

        // The verifier is binding: a caller-selected wrong identity rejects.
        let wrong = ed25519_dalek::SigningKey::from_bytes(&[0x63; 32]);
        let rejected = CheckpointedPgAcceptedStateSource::connect_test(
            PgKv::connect_test_readonly(&url).expect("readonly connect"),
            wrong.verifying_key(),
        )
        .accepted_state(&state.did);
        assert!(rejected.is_err(), "caller-selected checkpoint identity accepted");
    }

    /// A state envelope without its monotonic checkpoint is torn security
    /// state — the reader must refuse it, exactly like the RocksDB reader.
    #[test]
    fn live_pg_authority_rejects_torn_pair() {
        let url = require_db!();
        let (identity, _) = deployment_verifier(0x64);
        let audit = ed25519_dalek::SigningKey::from_bytes(&[0x65; 32]);
        let state = accepted_state();
        let (envelope, _) = encode_fixture(&state, &identity, &audit);

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let subject = format!("torn-{}-{nanos}", std::process::id());
        let writer = PgKv::connect_test(&url, "test-cell").expect("writer connect");
        writer.put(&state_key(&subject), &envelope).expect("write lone state");

        let reader = CheckpointedPgAcceptedStateSource::connect_test(
            PgKv::connect_test_readonly(&url).expect("readonly connect"),
            identity.verifying_key(),
        );
        let error = match reader.accepted_state(&format!("did:at9p:{subject}")) {
            Ok(_) => panic!("torn pair must be rejected"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("without its monotonic checkpoint"),
            "unexpected error: {error}"
        );

        // Isolation: the lone fixture row lives under the roster-enumerated
        // `at9p-state\0` prefix on a shared scratch database, and serial
        // execution alone does not remove persisted fixtures — a later roster
        // read would trip on the torn subject. Remove it explicitly.
        with_direct_connection(&url, |client| {
            let key = state_key(&subject);
            Box::pin(async move {
                client
                    .execute("DELETE FROM pds_kv WHERE key = $1", &[&key])
                    .await
                    .expect("cleanup delete");
            })
        });
    }

    /// Resolver startup never recreates history: an unprovisioned database
    /// (no pds_kv at all) fails the read-only connect closed.
    #[test]
    fn live_pg_authority_fails_closed_when_unprovisioned() {
        let url = require_db!();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let scratch = format!("pg_unprovisioned_{}_{nanos}", std::process::id());
        // A genuinely empty scratch database on the same server: the shared
        // test database keeps its provisioned pds_kv.
        with_direct_connection(&url, |client| {
            let scratch = scratch.clone();
            Box::pin(async move {
                client
                    .execute(&format!("CREATE DATABASE {scratch}"), &[])
                    .await
                    .expect("create unprovisioned scratch database");
            })
        });
        let (base, query) = match url.split_once('?') {
            Some((base, query)) => (base, format!("?{query}")),
            None => (url.as_str(), String::new()),
        };
        let (head, _) = base.rsplit_once('/').expect("test URL has a database path");
        let scoped = format!("{head}/{scratch}{query}");
        let error = match PgKv::connect_test_readonly(&scoped) {
            Ok(_) => panic!("unprovisioned read-only connect unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("not provisioned"),
            "unexpected error: {error:#}"
        );
        with_direct_connection(&url, |client| {
            let scratch = scratch.clone();
            Box::pin(async move {
                client
                    .execute(&format!("DROP DATABASE {scratch}"), &[])
                    .await
                    .expect("drop unprovisioned scratch database");
            })
        });
    }

    /// Fail-closed selection: a configured RDS binding that cannot be read
    /// must error WITHOUT creating or consulting the local RocksDB store.
    #[test]
    fn configured_postgres_selection_never_falls_back_to_local() {
        let (_ed, verifier) = deployment_verifier(0x66);
        let dir = tempfile::tempdir().expect("tempdir");
        // url_file unreadable → the binding is configured but broken.
        let records = hyprstream_pds::rds::RdsConfig {
            url_file: Some(dir.path().join("missing-records-url")),
            root_cert_file: Some(dir.path().join("missing-rds-ca.pem")),
            cell_id: "test-cell".to_owned(),
        };
        assert!(records.is_configured());
        let local_store = dir.path().join("pds-store");
        let error = match open_deployment_accepted_state_source(
            &local_store,
            verifier,
            &records,
            false,
        ) {
            Ok(_) => panic!("a broken configured PostgreSQL binding unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("RDS url_file"),
            "unexpected error: {error:#}"
        );
        assert!(
            !local_store.exists(),
            "the failed configured-Postgres path created a local store"
        );
    }
}
