//! Discovery service — re-exports from `hyprstream-discovery`.
//!
//! The DiscoveryService implementation has moved to the `hyprstream-discovery` crate.
//! This module provides a `PolicyAuthProvider` that wraps `PolicyClient` to implement
//! the `AuthorizationProvider` trait expected by DiscoveryService.

pub use hyprstream_discovery::{AuthorizationProvider, DiscoveryService};

use crate::services::generated::policy_client::PolicyCheck;
use async_trait::async_trait;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context as _, Result as AnyResult};
use hyprstream_discovery::{RecordCarData, RecordResolver};
use hyprstream_pds::at9p::h512;
use hyprstream_pds::at9p_duplicity::{
    AcceptedAt9pHead, AcceptedAt9pState, ConditionalAdvance, DuplicityGuard, WalDuplicityAlarmSink,
    Watermark, WatermarkStore,
};
use hyprstream_pds::at9p_gate::{verify_did_at9p, DID_AT9P_PREFIX};
use hyprstream_pds::car::build_record_proof_car;
use hyprstream_pds::commit::{Commit, UnsignedCommit};
use hyprstream_pds::ledger::{AllocationRecord, CheckpointRecord, ReceiptRecord};
use hyprstream_pds::mst::Node;
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::tid::Tid;
use sha2::Digest as _;

// Re-export the generated discovery client types from our local generated module
// (hyprstream still needs its own discovery_client for the DiscoveryClient type)

/// Policy-based authorization provider wrapping PolicyClient.
///
/// Bridges the `AuthorizationProvider` trait from `hyprstream-discovery`
/// to the `PolicyClient` generated in this crate.
pub struct PolicyAuthProvider {
    client: crate::services::PolicyClient,
}

impl PolicyAuthProvider {
    /// Create a new policy-based authorization provider.
    pub fn new(client: crate::services::PolicyClient) -> Self {
        Self { client }
    }
}

#[async_trait(?Send)]
impl AuthorizationProvider for PolicyAuthProvider {
    async fn check(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: &str,
    ) -> anyhow::Result<bool> {
        self.client
            .check(&PolicyCheck {
                subject: subject.to_owned(),
                domain: domain.to_owned(),
                resource: resource.to_owned(),
                operation: operation.to_owned(),
            })
            .await
    }
}

// ============================================================================
// PdsRecordStore — durable, keyless-read atproto record store (#910a)
// ============================================================================

/// One account's repo, loaded from the durable store: its records (keyed by
/// `(collection, tid)`) plus the **signed commit** that was persisted at write
/// time.
///
/// The MST is a deterministic, keyless function of the record set
/// (`Node::from_records`), so it is *not* stored — it is rebuilt on demand. The
/// only thing that must be persisted (a private key signed it) is the commit
/// itself. Reads therefore never touch a signing key: they rebuild the MST,
/// load this already-signed commit, and assemble a proof CAR. This mirrors how
/// atproto works — the commit is signed **once** at write time and served
/// verbatim thereafter.
struct RepoState {
    /// `<collection>/<rkey>` is the MST key; here we key by `(collection, tid)`.
    /// One repo spans *many* collections (#910b) — all of them live in this one
    /// map, and the MST is built over the full multi-collection key space.
    records: BTreeMap<(String, Tid), ModelRecord>,
    /// The signed commit persisted by the writer — served verbatim on reads,
    /// never re-signed.
    commit: Commit,
}

// ── RocksDB key scheme ───────────────────────────────────────────────────────
//
// DIDs and collection NSIDs never contain NUL bytes, so `\0` is a safe,
// unambiguous field separator for compound keys (same idea as the delimited
// keys `RocksDbUserStore` uses, just with a compound suffix here).
//
//   rk\0{did}\0{collection}\0{tid}   -> canonical DAG-CBOR record bytes
//   commit\0{did}                    -> signed-commit DAG-CBOR bytes
//
// The record's rkey (a TID) is derived deterministically from the repo id
// (`PdsRecordStore::tid_for_repo`), so it is never minted or persisted. The
// `#atproto` signing key is held ONLY by the writer (`PdsPublisher`); it is
// never stored in this DB and never held by the resolver (#910a). The commit is
// signed once at write time and persisted under `commit\0{did}`, so reads are a
// keyless walk (rebuild the MST, load the signed commit, serve a proof).

fn record_prefix(did: &str) -> Vec<u8> {
    format!("rk\0{did}\0").into_bytes()
}

fn record_key(did: &str, collection: &str, tid: Tid) -> Vec<u8> {
    format!("rk\0{did}\0{collection}\0{}", tid.encode()).into_bytes()
}

fn commit_key(did: &str) -> Vec<u8> {
    format!("commit\0{did}").into_bytes()
}

fn at9p_state_key(subject_cid512: &str) -> Vec<u8> {
    format!("at9p-state\0{subject_cid512}").into_bytes()
}
fn at9p_checkpoint_key(subject_cid512: &str) -> Vec<u8> {
    format!("at9p-checkpoint\0{subject_cid512}").into_bytes()
}

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

/// Split a full record key back into `(collection, tid)`, given the `did` it
/// was built for.
fn parse_record_key(key: &[u8], did: &str) -> AnyResult<(String, Tid)> {
    let prefix = record_prefix(did);
    let rest = key
        .strip_prefix(prefix.as_slice())
        .ok_or_else(|| anyhow!("record key missing expected prefix for {did:?}"))?;
    let rest = std::str::from_utf8(rest).context("record key suffix is not UTF-8")?;
    let (collection, tid_str) = rest
        .split_once('\0')
        .ok_or_else(|| anyhow!("record key missing collection/tid separator"))?;
    let tid = Tid::parse(tid_str)?;
    Ok((collection.to_owned(), tid))
}

/// Durable, RocksDB-backed PDS record store keyed by repo DID.
///
/// This is the per-account record store the DiscoveryService resolves against,
/// and that the registry service's register/commit paths publish into (#910a).
/// Persistence follows the same pattern as `RocksDbUserStore`
/// (`auth/rocksdb_store.rs`): a single `rocksdb::DB` at a well-known directory,
/// opened read-write by the *one* process that publishes, and read-only by
/// resolvers.
///
/// **The store holds no signing key.** The writer (`PdsPublisher`) signs the
/// commit once and hands this store the signed bytes; the resolver only ever
/// reads them back. This is the security fix at the heart of #910a: a read path
/// that needed a private key to re-sign a commit on every `getRecord` was the
/// root of the key-exposure problem — atproto never re-signs on read.
///
/// RocksDB allows exactly one read-write handle per directory; on a single
/// node the registry service (the publisher) is the sole writer, and the
/// discovery service (the resolver) opens the same directory read-only —
/// mirroring `RocksDbUserStore::open` / `open_readonly`. A read-only handle
/// snapshots the manifest at open time, so [`PdsRecordStore::open_readonly`]
/// callers that need a live view reopen per read (see [`RecordBacking::ReadOnly`]).
pub struct PdsRecordStore {
    backing: RecordBacking,
    /// Trusted deployment identity that certifies the purpose-derived key on
    /// daemon-authenticated accepted-state envelopes. This is verification
    /// material only; the resolver still holds no signing key.
    at9p_acceptance_identity: Option<ed25519_dalek::VerifyingKey>,
    at9p_advance_lock: parking_lot::Mutex<()>,
}

enum RecordBacking {
    /// The sole writer: one long-lived, read-write RocksDB handle.
    ReadWrite(rocksdb::DB),
    /// A resolver: no handle is held between calls — each read opens a fresh
    /// `open_for_read_only` handle so writes from the publisher process become
    /// visible without a restart.
    ReadOnly(PathBuf),
}

impl PdsRecordStore {
    /// Open (or create) the durable store at `path` for read-write access.
    /// Exactly one process should hold this — the publisher (registry
    /// service). See the type-level docs for the single-writer rationale.
    pub fn open(path: &Path) -> AnyResult<Self> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("failed to create PDS store dir {path:?}"))?;
        harden_store_dir(path);
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, path)
            .with_context(|| format!("failed to open PDS record store (rw) at {path:?}"))?;
        Ok(Self {
            backing: RecordBacking::ReadWrite(db),
            at9p_acceptance_identity: None,
            at9p_advance_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Open the store read-only at `path`, for resolvers that never write
    /// (the discovery service). Verifies the directory opens successfully now
    /// (surfacing a missing/corrupt store at startup) but does not hold the
    /// handle — see the type-level docs. The resolver holds **no signing key**.
    pub fn open_readonly(path: &Path) -> AnyResult<Self> {
        let opts = readonly_opts();
        let _probe = rocksdb::DB::open_for_read_only(&opts, path, false)
            .with_context(|| format!("failed to open PDS record store (ro) at {path:?}"))?;
        Ok(Self {
            backing: RecordBacking::ReadOnly(path.to_path_buf()),
            at9p_acceptance_identity: None,
            at9p_advance_lock: parking_lot::Mutex::new(()),
        })
    }

    /// Pin the deployment identity that certifies accepted-state envelopes.
    pub fn with_at9p_acceptance_identity(mut self, identity: ed25519_dalek::VerifyingKey) -> Self {
        self.at9p_acceptance_identity = Some(identity);
        self
    }

    /// Atomically persist a record and the repo's freshly-signed commit.
    ///
    /// The record and the commit that covers it advance together in one
    /// `WriteBatch`, so a reader never observes a record whose MST root the
    /// stored commit does not sign. Fails if this store was opened read-only.
    fn put_record_and_commit(
        &self,
        did: &str,
        collection: &str,
        tid: Tid,
        record: &ModelRecord,
        commit: &Commit,
    ) -> AnyResult<()> {
        let RecordBacking::ReadWrite(db) = &self.backing else {
            bail!("PdsRecordStore::put_record_and_commit called on a read-only store");
        };
        let mut batch = rocksdb::WriteBatch::default();
        batch.put(record_key(did, collection, tid), record.to_dag_cbor());
        batch.put(commit_key(did), commit.to_dag_cbor());
        db.write(batch).context("PDS record+commit write failed")?;
        Ok(())
    }

    fn load_at9p_state(
        &self,
        subject_cid512: &str,
        acceptance_identity: &ed25519_dalek::VerifyingKey,
    ) -> AnyResult<Option<AcceptedAt9pState>> {
        match &self.backing {
            RecordBacking::ReadWrite(db) => {
                load_at9p_state_from_db(db, subject_cid512, acceptance_identity)
            }
            RecordBacking::ReadOnly(path) => {
                let db = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)?;
                load_at9p_state_from_db(&db, subject_cid512, acceptance_identity)
            }
        }
    }

    /// Typed accepted-current state read path for resolver/admission consumers.
    /// Read-only store instances reopen RocksDB for a live view, just like PDS
    /// record resolution. `now` enforces update expiry when supplied.
    pub fn accepted_at9p_state(
        &self,
        did: &str,
        now: Option<&str>,
    ) -> AnyResult<Option<AcceptedAt9pState>> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow!("identifier is not did:at9p: {did:?}"))?;
        let acceptance_identity = self
            .at9p_acceptance_identity
            .as_ref()
            .ok_or_else(|| anyhow!("accepted-state verification identity is not configured"))?;
        let Some(state) = self.load_at9p_state(subject, acceptance_identity)? else {
            return Ok(None);
        };
        anyhow::ensure!(state.did == did, "accepted at9p state DID mismatch");
        if let Some(now) = now {
            state.ensure_fresh(now)?;
        }
        Ok(Some(state))
    }

    /// Enumerate every checkpoint-verified accepted state. This is the
    /// production startup boundary used to build native announcements; a
    /// missing checkpoint half or any verification failure aborts the read.
    pub fn accepted_at9p_states(&self) -> AnyResult<Vec<AcceptedAt9pState>> {
        let acceptance_identity = self
            .at9p_acceptance_identity
            .as_ref()
            .ok_or_else(|| anyhow!("accepted-state verification identity is not configured"))?;
        let read = |db: &rocksdb::DB| -> AnyResult<Vec<AcceptedAt9pState>> {
            let prefix = b"at9p-state\0";
            let mut states = Vec::new();
            for item in db.iterator(rocksdb::IteratorMode::Start) {
                let (key, _) = item?;
                let Some(subject) = key.strip_prefix(prefix) else {
                    continue;
                };
                let subject =
                    std::str::from_utf8(subject).context("accepted-state key is not UTF-8")?;
                let state = load_at9p_state_from_db(db, subject, acceptance_identity)?.ok_or_else(
                    || anyhow!("accepted-state key disappeared during snapshot read"),
                )?;
                states.push(state);
            }
            states.sort_by(|a, b| a.did.cmp(&b.did));
            Ok(states)
        };
        match &self.backing {
            RecordBacking::ReadWrite(db) => read(db),
            RecordBacking::ReadOnly(path) => {
                let db = rocksdb::DB::open_for_read_only(&readonly_opts(), path, false)?;
                read(&db)
            }
        }
    }

    fn conditional_advance_at9p_state(
        &self,
        expected: Option<Watermark>,
        state: &AcceptedAt9pState,
        acceptance_identity: &ed25519_dalek::SigningKey,
        audit_key: &ed25519_dalek::SigningKey,
    ) -> AnyResult<ConditionalAdvance> {
        let RecordBacking::ReadWrite(db) = &self.backing else {
            bail!("accepted did:at9p state write attempted on a read-only PDS store");
        };
        let _advance = self.at9p_advance_lock.lock();
        let current = load_at9p_state_from_db(
            db,
            &state.subject_cid512,
            &acceptance_identity.verifying_key(),
        )?;
        if current.as_ref().map(AcceptedAt9pState::watermark) != expected {
            return current.map_or_else(
                || bail!("conditional accepted-state advance expected a durable head, but none exists"),
                |current| Ok(ConditionalAdvance::Conflict(Box::new(current))),
            );
        }
        let encoded = encode_at9p_state(state, acceptance_identity, audit_key)?;
        let checkpoint = encode_at9p_checkpoint(state, &encoded, acceptance_identity)?;
        let mut batch = rocksdb::WriteBatch::default();
        batch.put(at9p_state_key(&state.subject_cid512), encoded);
        batch.put(at9p_checkpoint_key(&state.subject_cid512), checkpoint);
        let mut options = rocksdb::WriteOptions::default();
        options.set_sync(true);
        db.write_opt(batch, &options)
            .context("synchronous accepted did:at9p state+checkpoint commit failed")?;
        Ok(ConditionalAdvance::Committed)
    }

    /// The stable record-key TID for `repo_id`, derived **deterministically**
    /// from the repo id (#910a M1).
    ///
    /// The rkey must be a pure function of `repo_id`: two distinct repos must
    /// never alias onto one `(collection, tid)` key (which would silently and
    /// permanently clobber one repo's record with the other's), and the same
    /// repo must always resolve to the same key so a re-publish is a pointer
    /// advance, not a new record. A wall-clock `Tid::now()` fails both (same
    /// microsecond → identical TID; and it needs a persisted mint with a
    /// get-then-put race). Instead derive the TID's 53-bit micros field and
    /// 10-bit clock id from `SHA-256(repo_id)` — a hash, not a timestamp:
    /// these records are not time-ordered, so 63 bits of collision resistance
    /// is exactly what we want, with no mint, no persistence, and no race.
    pub fn tid_for_repo(repo_id: &str) -> Tid {
        // SHA-256 output is always 32 bytes, so these indexes never panic and
        // need no fallible `try_into`.
        let d = sha2::Sha256::digest(repo_id.as_bytes());
        let micros = u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
        let clock_id = u16::from_be_bytes([d[8], d[9]]);
        Tid::from_micros(micros, clock_id)
    }

    /// Load the full record set + signed commit for `did`, or `None` if the
    /// account has never published (no records, or no persisted commit).
    fn load_repo(&self, did: &str) -> AnyResult<Option<RepoState>> {
        match &self.backing {
            RecordBacking::ReadWrite(db) => Self::load_repo_from(db, did),
            RecordBacking::ReadOnly(path) => {
                let opts = readonly_opts();
                let db = rocksdb::DB::open_for_read_only(&opts, path, false)
                    .with_context(|| format!("failed to reopen PDS record store at {path:?}"))?;
                Self::load_repo_from(&db, did)
            }
        }
    }

    fn load_repo_from(db: &rocksdb::DB, did: &str) -> AnyResult<Option<RepoState>> {
        let prefix = record_prefix(did);
        let mut records = BTreeMap::new();
        // `prefix_iterator` without a configured `prefix_extractor` degrades to
        // a full forward scan from `prefix` — bound it manually.
        for item in db.prefix_iterator(&prefix) {
            let (key, value) = item.context("PDS store record scan failed")?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            let (collection, tid) = parse_record_key(&key, did)?;
            let record = ModelRecord::from_dag_cbor(&value)
                .with_context(|| format!("corrupt PDS record for {did}/{collection}/{tid}"))?;
            records.insert((collection, tid), record);
        }
        // "Never published" == no records for this DID.
        if records.is_empty() {
            return Ok(None);
        }
        // A published repo always has a persisted signed commit (records and
        // the commit that signs their MST root advance together atomically —
        // see `put_record_and_commit`). If the commit is absent the store is
        // inconsistent (e.g. a legacy pre-#910a-rework write); we cannot serve
        // a keyless proof without it, so report "not published" rather than
        // re-sign (the resolver holds no key by design).
        let Some(commit_bytes) = db
            .get(commit_key(did))
            .context("PDS store commit read failed")?
        else {
            tracing::warn!(
                did,
                "PDS repo has records but no signed commit — refusing to serve (resolver holds no key)"
            );
            return Ok(None);
        };
        let commit = Commit::from_dag_cbor(&commit_bytes)
            .with_context(|| format!("corrupt signed commit for {did}"))?;
        Ok(Some(RepoState { records, commit }))
    }

    /// Build the MST over a repo's record set. The MST is a deterministic,
    /// keyless function of the record keys/CIDs, so this reproduces exactly the
    /// tree whose root the persisted commit signs.
    ///
    /// All of a repo's records — across **all** collections — share ONE MST,
    /// keyed by `<collection>/<rkey>` exactly as atproto specifies (#910b).
    /// One repo, one tree, one signed root: adding a record in any collection
    /// changes the single root the commit signs.
    fn build_tree(records: &BTreeMap<(String, Tid), ModelRecord>) -> Node {
        let keyed: BTreeMap<String, hyprstream_pds::cid::Cid> = records
            .iter()
            .map(|((collection, tid), rec)| (format!("{collection}/{}", tid.encode()), rec.cid()))
            .collect();
        Node::from_keyed_records(&keyed)
    }
}

fn acceptance_message(domain: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(domain.len() + payload.len());
    message.extend_from_slice(domain);
    message.extend_from_slice(payload);
    message
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct At9pCheckpoint {
    watermark: Watermark,
    envelope_digest: [u8; 64],
}

fn checkpoint_message(subject: &str, payload: &[u8]) -> AnyResult<Vec<u8>> {
    let len = u32::try_from(subject.len()).context("at9p subject exceeds u32")?;
    let mut out = Vec::with_capacity(AT9P_CHECKPOINT_AAD.len() + 4 + subject.len() + payload.len());
    out.extend_from_slice(AT9P_CHECKPOINT_AAD);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(subject.as_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

fn encode_at9p_checkpoint(
    state: &AcceptedAt9pState,
    envelope: &[u8],
    identity: &ed25519_dalek::SigningKey,
) -> AnyResult<Vec<u8>> {
    use ed25519_dalek::Signer as _;
    let mut payload = Vec::with_capacity(AT9P_CHECKPOINT_LEN);
    payload.extend_from_slice(AT9P_CHECKPOINT_MAGIC);
    payload.extend_from_slice(&state.epoch.to_be_bytes());
    payload.extend_from_slice(&state.head_digest);
    payload.push(u8::from(state.terminal));
    payload.extend_from_slice(&h512(envelope));
    let sig = identity.sign(&checkpoint_message(&state.subject_cid512, &payload)?);
    payload.extend_from_slice(&sig.to_bytes());
    Ok(payload)
}

fn decode_at9p_checkpoint(
    subject: &str,
    bytes: &[u8],
    identity: &ed25519_dalek::VerifyingKey,
) -> AnyResult<At9pCheckpoint> {
    anyhow::ensure!(
        bytes.len() == AT9P_CHECKPOINT_LEN,
        "accepted at9p checkpoint length mismatch"
    );
    let payload = bytes
        .get(..AT9P_CHECKPOINT_PAYLOAD_LEN)
        .ok_or_else(|| anyhow!("accepted at9p checkpoint payload missing"))?;
    anyhow::ensure!(
        payload.starts_with(AT9P_CHECKPOINT_MAGIC),
        "accepted at9p checkpoint has bad version"
    );
    let sig = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(AT9P_CHECKPOINT_PAYLOAD_LEN..)
            .ok_or_else(|| anyhow!("accepted at9p checkpoint signature missing"))?
            .try_into()?,
    );
    identity
        .verify_strict(&checkpoint_message(subject, payload)?, &sig)
        .context("accepted at9p monotonic checkpoint signature rejected")?;
    let epoch = u64::from_be_bytes(
        payload
            .get(8..16)
            .ok_or_else(|| anyhow!("checkpoint epoch missing"))?
            .try_into()?,
    );
    let record_digest = payload
        .get(16..80)
        .ok_or_else(|| anyhow!("checkpoint digest missing"))?
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
            .ok_or_else(|| anyhow!("checkpoint envelope digest missing"))?
            .try_into()?,
    })
}

fn load_at9p_state_from_db(
    db: &rocksdb::DB,
    subject: &str,
    identity: &ed25519_dalek::VerifyingKey,
) -> AnyResult<Option<AcceptedAt9pState>> {
    let snapshot = db.snapshot();
    let envelope = snapshot.get(at9p_state_key(subject))?;
    let checkpoint = snapshot.get(at9p_checkpoint_key(subject))?;
    match (envelope, checkpoint) {
        (None, None) => Ok(None),
        (Some(_), None) => bail!("accepted at9p state exists without its monotonic checkpoint"),
        (None, Some(_)) => bail!("accepted at9p checkpoint exists without its state envelope"),
        (Some(envelope), Some(checkpoint)) => {
            let state = decode_at9p_state(subject, &envelope, identity)?;
            let checkpoint = decode_at9p_checkpoint(subject, &checkpoint, identity)?;
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

fn encode_at9p_state_body(state: &AcceptedAt9pState) -> AnyResult<Vec<u8>> {
    let (kind, head) = match state.head() {
        AcceptedAt9pHead::Genesis(capsule) => (0u8, capsule.to_dag_cbor()?),
        AcceptedAt9pHead::Update(record) => (1u8, record.to_dag_cbor()?),
    };
    let head_len = u32::try_from(head.len()).context("accepted at9p head exceeds u32")?;
    let mut out = Vec::with_capacity(AT9P_STATE_HEADER_LEN + head.len());
    out.extend_from_slice(AT9P_STATE_MAGIC);
    out.push(kind);
    out.extend_from_slice(&state.epoch.to_be_bytes());
    out.extend_from_slice(&state.head_digest);
    out.push(u8::from(state.terminal));
    out.extend_from_slice(&head_len.to_be_bytes());
    out.extend_from_slice(&head);
    Ok(out)
}

/// Wrap one complete state value in a daemon-authenticated envelope. The
/// stable deployment identity certifies the purpose-derived audit key, and the
/// audit key signs the exact state bytes. A different internally valid at9p
/// fork therefore cannot be substituted by someone who can edit RocksDB.
fn encode_at9p_state(
    state: &AcceptedAt9pState,
    acceptance_identity: &ed25519_dalek::SigningKey,
    audit_key: &ed25519_dalek::SigningKey,
) -> AnyResult<Vec<u8>> {
    use ed25519_dalek::Signer as _;

    let body = encode_at9p_state_body(state)?;
    let audit_vk = audit_key.verifying_key().to_bytes();
    let key_signature =
        acceptance_identity.sign(&acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_vk));
    let state_signature = audit_key.sign(&acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, &body));
    let mut out =
        Vec::with_capacity(AT9P_ACCEPTANCE_PREFIX_LEN + body.len() + ED25519_SIGNATURE_LEN);
    out.extend_from_slice(AT9P_ACCEPTANCE_MAGIC);
    out.extend_from_slice(&audit_vk);
    out.extend_from_slice(&key_signature.to_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&state_signature.to_bytes());
    Ok(out)
}

fn decode_at9p_state(
    subject_cid512: &str,
    bytes: &[u8],
    acceptance_identity: &ed25519_dalek::VerifyingKey,
) -> AnyResult<AcceptedAt9pState> {
    anyhow::ensure!(
        bytes.len() >= AT9P_ACCEPTANCE_PREFIX_LEN + AT9P_STATE_HEADER_LEN + ED25519_SIGNATURE_LEN,
        "accepted at9p envelope is truncated"
    );
    anyhow::ensure!(
        bytes.starts_with(AT9P_ACCEPTANCE_MAGIC),
        "accepted at9p envelope has bad version"
    );
    let audit_vk_bytes: [u8; ED25519_KEY_LEN] = bytes
        .get(AT9P_ACCEPTANCE_MAGIC.len()..AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN)
        .ok_or_else(|| anyhow!("accepted at9p envelope missing audit key"))?
        .try_into()?;
    let key_sig_start = AT9P_ACCEPTANCE_MAGIC.len() + ED25519_KEY_LEN;
    let key_sig_end = key_sig_start + ED25519_SIGNATURE_LEN;
    let key_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(key_sig_start..key_sig_end)
            .ok_or_else(|| anyhow!("accepted at9p envelope missing key certificate"))?
            .try_into()?,
    );
    acceptance_identity
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_KEY_AAD, &audit_vk_bytes),
            &key_signature,
        )
        .context("accepted at9p audit-key certificate rejected")?;
    let audit_vk = ed25519_dalek::VerifyingKey::from_bytes(&audit_vk_bytes)
        .context("accepted at9p envelope has malformed audit key")?;
    let state_sig_start = bytes.len() - ED25519_SIGNATURE_LEN;
    let body = bytes
        .get(AT9P_ACCEPTANCE_PREFIX_LEN..state_sig_start)
        .ok_or_else(|| anyhow!("accepted at9p envelope missing state body"))?;
    let state_signature = ed25519_dalek::Signature::from_bytes(
        bytes
            .get(state_sig_start..)
            .ok_or_else(|| anyhow!("accepted at9p envelope missing state signature"))?
            .try_into()?,
    );
    audit_vk
        .verify_strict(
            &acceptance_message(AT9P_ACCEPTANCE_STATE_AAD, body),
            &state_signature,
        )
        .context("accepted at9p daemon acceptance signature rejected")?;
    decode_at9p_state_body(subject_cid512, body)
}

fn decode_at9p_state_body(subject_cid512: &str, bytes: &[u8]) -> AnyResult<AcceptedAt9pState> {
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
        .ok_or_else(|| anyhow!("accepted at9p state missing kind"))?;
    let epoch = u64::from_be_bytes(
        bytes
            .get(9..17)
            .ok_or_else(|| anyhow!("accepted at9p state missing epoch"))?
            .try_into()?,
    );
    let digest: [u8; 64] = bytes
        .get(17..81)
        .ok_or_else(|| anyhow!("accepted at9p state missing digest"))?
        .try_into()?;
    let terminal = match bytes
        .get(81)
        .ok_or_else(|| anyhow!("accepted at9p state missing terminal flag"))?
    {
        0 => false,
        1 => true,
        _ => bail!("accepted at9p state has invalid terminal flag"),
    };
    let head_len = u32::from_be_bytes(
        bytes
            .get(82..86)
            .ok_or_else(|| anyhow!("accepted at9p state missing length"))?
            .try_into()?,
    ) as usize;
    anyhow::ensure!(
        bytes.len() == AT9P_STATE_HEADER_LEN + head_len,
        "accepted at9p state length mismatch"
    );
    let head = bytes
        .get(AT9P_STATE_HEADER_LEN..)
        .ok_or_else(|| anyhow!("accepted at9p state missing head"))?;
    let state = match kind {
        0 => AcceptedAt9pState::from_persisted_genesis(head)?,
        1 => AcceptedAt9pState::from_persisted_update(head)?,
        _ => bail!("accepted at9p state has unknown head kind {kind}"),
    };
    anyhow::ensure!(
        state.subject_cid512 == subject_cid512,
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

#[derive(Clone)]
struct RocksAt9pStateStore {
    store: Arc<PdsRecordStore>,
    acceptance_identity: ed25519_dalek::SigningKey,
    audit_key: ed25519_dalek::SigningKey,
}

impl WatermarkStore for RocksAt9pStateStore {
    fn get(&self, subject_cid512: &str) -> AnyResult<Option<AcceptedAt9pState>> {
        self.store
            .load_at9p_state(subject_cid512, &self.acceptance_identity.verifying_key())
    }

    fn conditional_advance(
        &self,
        expected: Option<Watermark>,
        state: &AcceptedAt9pState,
    ) -> AnyResult<ConditionalAdvance> {
        self.store.conditional_advance_at9p_state(
            expected,
            state,
            &self.acceptance_identity,
            &self.audit_key,
        )
    }
}

/// Sole production ownership boundary for accepted current `did:at9p` state.
/// It GATE-seeds genesis, reloads predecessors exclusively from durable state,
/// calls the duplicity guard for every successor, and atomically commits the
/// complete accepted head. It does not dial, select reach, or admit carriers.
pub struct At9pStateIngest {
    guard: DuplicityGuard<RocksAt9pStateStore, WalDuplicityAlarmSink>,
}

impl At9pStateIngest {
    pub fn open(
        store: Arc<PdsRecordStore>,
        alarm_path: &Path,
        acceptance_identity: ed25519_dalek::SigningKey,
        audit_ed_sk: ed25519_dalek::SigningKey,
        pq_sk: hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    ) -> AnyResult<Self> {
        let guard = DuplicityGuard::with_durable_alarm(
            RocksAt9pStateStore {
                store,
                acceptance_identity,
                audit_key: audit_ed_sk.clone(),
            },
            alarm_path,
            audit_ed_sk,
            pq_sk,
        )?;
        Ok(Self { guard })
    }

    pub fn ingest_genesis(&self, did: &str, capsule_bytes: &[u8]) -> AnyResult<AcceptedAt9pState> {
        let verified = verify_did_at9p(did, capsule_bytes)
            .map_err(|e| anyhow!("did:at9p genesis ingest rejected: {e}"))?;
        self.guard.seed_genesis(&verified)?;
        self.accepted_state(did, None)
    }

    pub fn ingest_successor(
        &self,
        did: &str,
        update_bytes: &[u8],
        now: &str,
    ) -> AnyResult<AcceptedAt9pState> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow!("identifier is not did:at9p: {did:?}"))?;
        let candidate = hyprstream_pds::at9p::UpdateRecord::from_dag_cbor(update_bytes)?;
        anyhow::ensure!(
            candidate.subject_cid512 == subject,
            "update subject does not match ingest DID"
        );
        self.guard.admit_successor(&candidate, now)?;
        self.accepted_state(did, Some(now))
    }

    /// Read the complete accepted snapshot. Supplying `now` enforces successor
    /// expiry before the state is exposed to a consumer.
    pub fn accepted_state(&self, did: &str, now: Option<&str>) -> AnyResult<AcceptedAt9pState> {
        let subject = did
            .strip_prefix(DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow!("identifier is not did:at9p: {did:?}"))?;
        let state = self
            .guard
            .accepted_state(subject)?
            .ok_or_else(|| anyhow!("no accepted did:at9p state for {did}"))?;
        anyhow::ensure!(state.did == did, "accepted at9p state DID mismatch");
        if let Some(now) = now {
            state.ensure_fresh(now)?;
        }
        Ok(state)
    }
}

fn readonly_opts() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(false);
    opts
}

/// Restrict the record-store directory to owner-only (0700) on unix.
///
/// Defence in depth: the record *values* (and the signed commit) are what a
/// resolver serves as authentic PDS state — a store another local user could
/// write would let them inject records/commits the node then serves. On
/// non-unix this is a no-op (the store path is expected to be an OS-managed
/// private dir there).
fn harden_store_dir(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Best-effort: this is defence-in-depth (the private key never lands in
        // this DB, and the path never resolves to /tmp — H2). If the chmod
        // fails (e.g. the dir is owned by a different uid in a split-uid setup),
        // warn rather than hard-fail startup.
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)) {
            tracing::warn!("could not set 0700 on PDS store dir {path:?}: {e}");
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

// ============================================================================
// PdsPublisher — the register/commit write side of #910a
// ============================================================================

/// Publishes/advances a repo's `ai.hyprstream.model` PDS record on this node,
/// over a read-write [`PdsRecordStore`]. Constructed once by the registry
/// service (the sole writer) and called from `handle_register` /
/// `handle_commit_with_author`.
///
/// This is the **only** holder of the `#atproto` P-256 private key: it signs
/// the repo's commit once, here, at write time, and persists the signed bytes.
/// The resolver never sees the key.
pub struct PdsPublisher {
    store: Arc<PdsRecordStore>,
    at9p_state: Option<At9pStateIngest>,
    /// This node's own `did:key` identity — the `repo` at-uri authority for
    /// every record this node publishes (single-node, self-hosted PDS; a
    /// per-account DID scheme is out of scope for #910a).
    did: String,
    /// This node's `#atproto` commit-signing key (P-256/ES256), sourced from
    /// the shared `Es256SigningKeyStore` — the *same* key `did_document.rs`
    /// publishes as the `#atproto` verification method. Held only here (the
    /// writer); resolvers hold no key. Used to sign each commit exactly once.
    signing_key: p256::ecdsa::SigningKey,
    /// Serializes the load→rebuild→sign→persist critical section (#910b).
    ///
    /// A publish is a read-modify-write over the WHOLE repo (one MST, one
    /// signed root across all collections): two concurrent publishes that both
    /// load the same prior state would each sign a commit covering only their
    /// own record, and whichever lands last silently drops the other's from
    /// the signed root — the store would then fail the read-side
    /// root-consistency check. Holding this lock across the full critical
    /// section makes each commit cover every prior write.
    publish_lock: parking_lot::Mutex<()>,
}

impl PdsPublisher {
    pub fn new(
        store: Arc<PdsRecordStore>,
        did: String,
        signing_key: p256::ecdsa::SigningKey,
    ) -> Self {
        Self {
            store,
            at9p_state: None,
            did,
            signing_key,
            publish_lock: parking_lot::Mutex::new(()),
        }
    }

    /// Install the daemon-owned accepted-state ingest boundary. Production
    /// construction always calls this; the optional form keeps isolated PDS
    /// pointer-store tests free of unrelated audit-key setup.
    pub fn with_at9p_state_ingest(mut self, ingest: At9pStateIngest) -> Self {
        self.at9p_state = Some(ingest);
        self
    }

    pub fn ingest_at9p_genesis(
        &self,
        did: &str,
        capsule_bytes: &[u8],
    ) -> AnyResult<AcceptedAt9pState> {
        self.at9p_state
            .as_ref()
            .ok_or_else(|| anyhow!("accepted did:at9p ingest is not configured"))?
            .ingest_genesis(did, capsule_bytes)
    }

    pub fn ingest_at9p_successor(
        &self,
        did: &str,
        update_bytes: &[u8],
        now: &str,
    ) -> AnyResult<AcceptedAt9pState> {
        self.at9p_state
            .as_ref()
            .ok_or_else(|| anyhow!("accepted did:at9p ingest is not configured"))?
            .ingest_successor(did, update_bytes, now)
    }

    pub fn accepted_at9p_state(
        &self,
        did: &str,
        now: Option<&str>,
    ) -> AnyResult<AcceptedAt9pState> {
        self.at9p_state
            .as_ref()
            .ok_or_else(|| anyhow!("accepted did:at9p ingest is not configured"))?
            .accepted_state(did, now)
    }

    /// Publish (or advance) the `ai.hyprstream.model` record for `repo_id`,
    /// pointing it at `current_oid_hex` (a git OID, hex-encoded).
    ///
    /// Collection-parameterized publication lives in
    /// [`Self::publish_record`]; this is the `ai.hyprstream.model` wrapper the
    /// registry service calls — behavior unchanged from #910a.
    ///
    /// Best-effort by design: the caller (registry service) should log and
    /// continue on error rather than fail the git operation that triggered
    /// this — the local git commit is the durable source of truth; a missed
    /// PDS publish is caught up by the next successful one (same
    /// eventual-consistency contract `git::promote` documents).
    pub fn publish(&self, repo_id: &str, current_oid_hex: &str) -> AnyResult<()> {
        self.publish_record(COLLECTION_NSID, repo_id, current_oid_hex)
    }

    /// Publish (or advance) a record in `collection` for `record_id`, pointing
    /// it at `current_oid_hex` (a git OID, hex-encoded). This is the
    /// collection-parameterized publication path (#910b): the repo's ONE MST
    /// spans every collection, and the ONE commit signed here covers them all.
    ///
    /// The record's rkey is derived deterministically from `record_id`
    /// (`PdsRecordStore::tid_for_repo`) and reused on every call, so this
    /// always *replaces* the existing record's value rather than creating a
    /// new one. (rkeys are scoped per collection in atproto, so the same
    /// `record_id` in two collections is two distinct records at
    /// `<collection-a>/<rkey>` and `<collection-b>/<rkey>`.) The repo's MST is
    /// rebuilt over its full multi-collection record set, a new commit is
    /// signed **once** here (rev strictly advances; `prev` points at the
    /// previous commit's CID — the atproto pointer advance), and the record
    /// and signed commit are persisted together.
    pub fn publish_record(
        &self,
        collection: &str,
        record_id: &str,
        current_oid_hex: &str,
    ) -> AnyResult<()> {
        let current_oid_cid = git_oid_to_cid_string(current_oid_hex)?;
        self.publish_pointer(collection, record_id, current_oid_cid)
    }

    /// Publish (or advance) a holder-controlled `ai.hyprstream.ledger.allocation`
    /// record — the held entitlement (#924).
    ///
    /// The allocation's canonical DAG-CBOR bytes are content-addressed and the
    /// resulting CID becomes the pointer this repo's MST carries under the
    /// `ai.hyprstream.ledger.allocation` collection; the holder's signed PDS
    /// commit is the only signature the record needs (it lives in the holder's
    /// own repo). The rkey is derived deterministically from the record's grant
    /// CID, so re-publishing the same grant advances the same record.
    pub fn publish_ledger_allocation(&self, rec: &AllocationRecord) -> AnyResult<()> {
        self.publish_pointer(
            hyprstream_pds::ledger::allocation::COLLECTION_NSID,
            &rec.grant,
            hyprstream_pds::cid::Cid::from_dag_cbor(&rec.to_dag_cbor()).encode(),
        )
    }

    /// Publish (or advance) a holder-controlled `ai.hyprstream.ledger.receipt`
    /// record — a dual-signed usage record (#924).
    ///
    /// The rkey is derived from the receipt's `transferId`, giving at-least-once
    /// (idempotent) receipt delivery: re-publishing the same transfer advances
    /// the same record rather than creating a duplicate.
    pub fn publish_ledger_receipt(&self, rec: &ReceiptRecord) -> AnyResult<()> {
        self.publish_pointer(
            hyprstream_pds::ledger::receipt::COLLECTION_NSID,
            &rec.transfer_id,
            hyprstream_pds::cid::Cid::from_dag_cbor(&rec.to_dag_cbor()).encode(),
        )
    }

    /// Publish (or advance) a holder-controlled `ai.hyprstream.ledger.checkpoint`
    /// record — a signed ledger head (#924).
    ///
    /// `ledger_id` is the stable key of the ledger whose head this checkpoint
    /// commits, so successive checkpoints advance the one head pointer.
    pub fn publish_ledger_checkpoint(
        &self,
        ledger_id: &str,
        rec: &CheckpointRecord,
    ) -> AnyResult<()> {
        self.publish_pointer(
            hyprstream_pds::ledger::checkpoint::COLLECTION_NSID,
            ledger_id,
            hyprstream_pds::cid::Cid::from_dag_cbor(&rec.to_dag_cbor()).encode(),
        )
    }

    /// The shared publish critical section: write (or advance) the pointer record
    /// in `collection` for `record_id`, pointing it at the already-formed
    /// `current_oid_cid` (`format: "cid"` string). Every collection — the git
    /// model pointer and the ledger records alike — shares the repo's ONE MST and
    /// the ONE commit signed here (#910b).
    fn publish_pointer(
        &self,
        collection: &str,
        record_id: &str,
        current_oid_cid: String,
    ) -> AnyResult<()> {
        // The collection NSID becomes both a `\0`-delimited RocksDB key field
        // and the `<collection>/<rkey>` MST key prefix — reject anything that
        // would be ambiguous in either encoding.
        if collection.is_empty() || collection.contains(['\0', '/']) {
            bail!("invalid collection NSID {collection:?}");
        }
        let tid = PdsRecordStore::tid_for_repo(record_id);
        let record = ModelRecord::new(
            format!("at://{}", self.did),
            current_oid_cid,
            atproto_datetime_now(),
        )?;

        // ── critical section: load → rebuild → sign → persist ──────────────
        // Serialized so every signed commit covers ALL prior writes (see
        // `publish_lock`). A panicked publish is benign — the store write is
        // a single atomic WriteBatch, so the store stays consistent and the
        // next publisher proceeds (parking_lot has no poisoning).
        let _guard = self.publish_lock.lock();

        // Load the existing repo state (records + previous signed commit) so we
        // can rebuild the full MST and advance the commit pointer.
        let existing = self.store.load_repo(&self.did)?;
        let mut records = existing
            .as_ref()
            .map(|r| r.records.clone())
            .unwrap_or_default();
        records.insert((collection.to_owned(), tid), record.clone());

        // Rebuild the MST (deterministic, keyless) over the full
        // multi-collection record set and sign the commit ONCE over its root.
        let tree = PdsRecordStore::build_tree(&records);
        let root = tree.root_cid();
        let (rev, prev) = match existing.as_ref().map(|r| &r.commit) {
            Some(prev_commit) => (next_rev(prev_commit.rev), Some(prev_commit.cid())),
            None => (Tid::now(), None),
        };
        let unsigned = UnsignedCommit::new(self.did.clone(), root, rev, prev);
        let commit = Commit::sign(&unsigned, &self.signing_key);

        self.store
            .put_record_and_commit(&self.did, collection, tid, &record, &commit)
    }
}

/// The next commit `rev`: the current wall-clock TID, or one past the previous
/// rev if the clock has not advanced past it.
///
/// atproto requires `rev` to strictly increase across a repo's commits; two
/// publishes within the same microsecond (or a clock that went backwards) must
/// still advance.
fn next_rev(prev: Tid) -> Tid {
    let now = Tid::now();
    if now > prev {
        now
    } else {
        Tid::from_raw(prev.to_raw().saturating_add(1))
    }
}

/// Encode a git OID (hex string, SHA-1 or SHA-256) as a `format: "cid"`
/// string for `ModelRecord::currentOid`.
///
/// TODO(#395): this is a placeholder encoding (a CIDv1 `raw` codec over the
/// OID's raw bytes, i.e. a *hash of the OID*, not an identity-preserving
/// encoding of it) — good enough to be deterministic and format-valid today,
/// but the real git-OID↔CID grammar belongs to the CID encoder work tracked
/// there. Once that lands, swap this call for the real encoder; nothing else
/// in the publish path needs to change.
fn git_oid_to_cid_string(oid_hex: &str) -> AnyResult<String> {
    let oid_bytes =
        hex::decode(oid_hex).with_context(|| format!("git OID {oid_hex:?} is not valid hex"))?;
    Ok(hyprstream_pds::cid::Cid::from_raw(&oid_bytes).encode())
}

/// The current UTC time in the atproto `datetime` lexicon format
/// (ISO-8601, millisecond precision, trailing `Z`).
fn atproto_datetime_now() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

/// `RecordResolver` over a [`PdsRecordStore`]. Builds CAR proofs via the
/// reused `hyprstream_pds::car::build_record_proof_car` — no CAR/MST/commit
/// code is reinvented here, and **no signing happens on the read path**: the
/// commit served in the proof is the one the writer signed at publish time.
pub struct PdsRecordResolver {
    store: Arc<PdsRecordStore>,
}

impl PdsRecordResolver {
    pub fn new(store: Arc<PdsRecordStore>) -> Self {
        Self { store }
    }

    pub fn accepted_at9p_state(
        &self,
        did: &str,
        now: &str,
    ) -> AnyResult<Option<AcceptedAt9pState>> {
        self.store.accepted_at9p_state(did, Some(now))
    }
}

/// Rebuild the repo's MST (keyless) and assemble a proof CAR for one record,
/// served against the STORED signed commit — no re-signing.
///
/// **Fail-closed root-consistency check (#910b):** the rebuilt tree's root must
/// equal the root the persisted commit signs (`commit.data`). Records and their
/// covering commit are written atomically, so a mismatch means the store is
/// corrupt or was tampered with — serving would hand out a record the signed
/// commit does not actually cover (the proof could not verify anyway, but we
/// refuse loudly rather than emit a broken CAR). This is an error, not
/// NOT_FOUND: an inconsistent store must never look like a merely-absent record.
fn proof_car(
    repo: &RepoState,
    did: &str,
    collection: &str,
    tid: Tid,
    record: &ModelRecord,
) -> AnyResult<Vec<u8>> {
    let tree = PdsRecordStore::build_tree(&repo.records);
    let rebuilt_root = tree.root_cid();
    if rebuilt_root != repo.commit.data {
        bail!(
            "PDS store inconsistent for {did}: rebuilt MST root {rebuilt_root} \
             != signed commit root {} — refusing to serve",
            repo.commit.data
        );
    }
    let proof = tree
        .proof(collection, &tid)
        .ok_or_else(|| anyhow!("record present but no MST proof — store inconsistent"))?;
    let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
    Ok(build_record_proof_car(
        &repo.commit,
        &proof,
        &node_blocks,
        record,
    ))
}

#[async_trait(?Send)]
impl RecordResolver for PdsRecordResolver {
    async fn resolve_record(
        &self,
        did: &str,
        collection: &str,
        rkey: &str,
    ) -> anyhow::Result<Option<RecordCarData>> {
        let tid = match Tid::parse(rkey) {
            Ok(t) => t,
            // A malformed rkey simply names no record — report NOT_FOUND, not an
            // error, so a bad address looks the same as an absent one.
            Err(_) => return Ok(None),
        };

        let Some(repo) = self.store.load_repo(did)? else {
            return Ok(None);
        };
        let key = (collection.to_owned(), tid);
        let Some(record) = repo.records.get(&key).cloned() else {
            return Ok(None);
        };

        // Rebuild the MST (keyless) and serve the STORED signed commit — no
        // re-signing. `proof_car` fail-closes if the rebuilt root does not
        // match the root the persisted commit signs.
        let car = proof_car(&repo, did, collection, tid, &record)?;

        let uri = format!("at://{did}/{collection}/{rkey}");
        Ok(Some(RecordCarData { uri, car }))
    }

    async fn resolve_repo(&self, did: &str) -> anyhow::Result<Option<RecordCarData>> {
        let Some(repo) = self.store.load_repo(did)? else {
            return Ok(None);
        };

        // Full-repo CAR: a self-contained helper for the WHOLE repo (commit +
        // all MST nodes + all record blocks) does not exist in the PDS `car`
        // module yet (only `build_record_proof_car` for a single record). Rather
        // than reinvent the CAR/MST assembly here, return the proof CAR for ONE
        // record (the first, deterministically) as the repo-root proof. This is
        // documented as a deferral: getRepo currently returns a single-record
        // proof rooted at the same signed commit; a true all-records CAR is a
        // follow-up (needs a `build_repo_car` helper in hyprstream-pds).
        let Some(((collection, tid), record)) = repo.records.iter().next() else {
            return Ok(None);
        };
        let collection = collection.clone();
        let tid = *tid;
        let record = record.clone();

        let car = proof_car(&repo, did, &collection, tid, &record)?;

        let uri = format!("at://{did}/{collection}/{}", tid.encode());
        Ok(Some(RecordCarData { uri, car }))
    }
}

impl hyprstream_discovery::AcceptedStateSource for PdsRecordResolver {
    fn accepted_state(&self, did: &str) -> AnyResult<Option<AcceptedAt9pState>> {
        self.store.accepted_at9p_state(did, None)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod pds_store_tests {
    use super::*;
    use hyprstream_pds::car::{parse_car_v1, verify_record_proof};
    use p256::ecdsa::{SigningKey, VerifyingKey};

    /// A 40-hex-char (SHA-1-shaped) git OID for the publish path.
    const SAMPLE_OID: &str = "1111111111111111111111111111111111111111";
    const SAMPLE_OID_2: &str = "2222222222222222222222222222222222222222";
    const DID: &str = "did:key:zTestNode";

    /// Extract and decode the signed commit from a proof CAR (its single root).
    fn commit_from_car(car: &[u8]) -> Commit {
        let (roots, blocks) = parse_car_v1(car).expect("parse CAR");
        let root = roots.first().copied().expect("CAR has a root commit");
        let (_cid, bytes) = blocks
            .iter()
            .find(|(cid, _)| *cid == root)
            .expect("commit block present");
        Commit::from_dag_cbor(bytes).expect("decode commit")
    }

    /// Minimal single-threaded block_on so these tests don't need a full
    /// `#[tokio::test]` runtime just to drive one `async_trait(?Send)` call.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build current-thread runtime")
            .block_on(fut)
    }

    #[test]
    fn keyless_read_serves_writer_signed_commit() {
        // The writer signs once; a resolver constructed with NO key returns a
        // proof whose commit verifies against the published P-256 key.
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        publisher.publish("repo-a", SAMPLE_OID).expect("publish");

        // Resolver holds no key at all.
        let resolver = PdsRecordResolver::new(Arc::clone(&store));
        let tid = PdsRecordStore::tid_for_repo("repo-a");
        let got = block_on(resolver.resolve_record(DID, COLLECTION_NSID, &tid.encode()))
            .expect("resolve ok")
            .expect("record present");

        let commit = commit_from_car(&got.car);
        commit
            .verify(&vk)
            .expect("writer-signed commit must verify against the published #atproto key");
    }

    #[test]
    fn publish_read_full_proof_verifies() {
        // End-to-end: publish → getRecord → the FULL record proof (commit sig +
        // MST path + record CID) verifies keyless against the published key.
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        publisher.publish("repo-a", SAMPLE_OID).expect("publish");

        // Reconstruct the proof the same way the resolver does (store holds no
        // key), and run the offline verifier against the published key.
        let repo = store
            .load_repo(DID)
            .expect("load ok")
            .expect("repo present");
        let tid = PdsRecordStore::tid_for_repo("repo-a");
        let record = repo
            .records
            .get(&(COLLECTION_NSID.to_owned(), tid))
            .cloned()
            .expect("record present");
        let tree = PdsRecordStore::build_tree(&repo.records);
        let proof = tree.proof(COLLECTION_NSID, &tid).expect("proof");
        verify_record_proof(&repo.commit, &vk, &proof, &record)
            .expect("full keyless proof must verify");
    }

    #[test]
    fn pointer_advance_bumps_rev_and_links_prev() {
        // A re-publish must advance the commit: a new rev and prev = the old
        // commit's CID.
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);

        publisher.publish("repo-a", SAMPLE_OID).expect("publish 1");
        let first = store.load_repo(DID).expect("load").expect("repo").commit;
        assert!(first.prev.is_none(), "genesis commit has no prev");

        publisher
            .publish("repo-a", SAMPLE_OID_2)
            .expect("publish 2");
        let second = store.load_repo(DID).expect("load").expect("repo").commit;
        assert_eq!(
            second.prev,
            Some(first.cid()),
            "the advanced commit must link the previous commit"
        );
        assert!(second.rev > first.rev, "rev must strictly advance");
    }

    #[test]
    fn readonly_store_survives_restart_and_sees_writer_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        {
            let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
            let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
            publisher.publish("repo-b", SAMPLE_OID).expect("publish");
            // Writer handle dropped here — simulates a process restart.
        }

        let ro = Arc::new(PdsRecordStore::open_readonly(dir.path()).expect("open ro"));
        let resolver = PdsRecordResolver::new(Arc::clone(&ro));
        let got = block_on(resolver.resolve_repo(DID))
            .expect("resolve_repo ok")
            .expect("repo present after reopen");
        assert!(got.uri.starts_with(&format!("at://{DID}/")));
        // The signed commit survived the reopen and still verifies.
        commit_from_car(&got.car)
            .verify(&vk)
            .expect("commit persisted across restart must verify");
    }

    #[test]
    fn readonly_store_rejects_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Create the directory with a real DB first so open_readonly succeeds.
        drop(PdsRecordStore::open(dir.path()).expect("open rw"));

        let ro = PdsRecordStore::open_readonly(dir.path()).expect("open ro");
        let record = ModelRecord::new(
            "at://did:key:zX",
            format!("bafyreiexample{SAMPLE_OID}"),
            "2026-06-23T12:34:56.789Z",
        )
        .expect("record");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let tree = PdsRecordStore::build_tree(&BTreeMap::new());
        let unsigned = UnsignedCommit::new("did:key:zX", tree.root_cid(), Tid::now(), None);
        let commit = Commit::sign(&unsigned, &sk);
        let err = ro
            .put_record_and_commit("did:key:zX", COLLECTION_NSID, Tid::now(), &record, &commit)
            .unwrap_err();
        assert!(err.to_string().contains("read-only"));
    }

    /// A second collection for multi-collection tests (#910b) — the G3 (#908)
    /// name collection this slice exists to unblock.
    const NAME_COLLECTION: &str = "ai.hyprstream.name";

    /// Resolve `(collection, record_id)` via a keyless resolver and verify the
    /// full record proof offline against `vk`: commit signature + MST path from
    /// the signed root + record CID.
    fn resolve_and_verify(
        store: &Arc<PdsRecordStore>,
        vk: &VerifyingKey,
        collection: &str,
        record_id: &str,
    ) {
        let tid = PdsRecordStore::tid_for_repo(record_id);
        let resolver = PdsRecordResolver::new(Arc::clone(store));
        let got = block_on(resolver.resolve_record(DID, collection, &tid.encode()))
            .expect("resolve ok")
            .expect("record present");
        assert_eq!(got.uri, format!("at://{DID}/{collection}/{}", tid.encode()));
        commit_from_car(&got.car)
            .verify(vk)
            .expect("commit verifies");

        // Full offline proof, reconstructed the way the resolver does it.
        let repo = store
            .load_repo(DID)
            .expect("load ok")
            .expect("repo present");
        let record = repo
            .records
            .get(&(collection.to_owned(), tid))
            .cloned()
            .expect("record present in store");
        let tree = PdsRecordStore::build_tree(&repo.records);
        assert_eq!(
            tree.root_cid(),
            repo.commit.data,
            "rebuilt multi-collection root must match the signed commit root"
        );
        let proof = tree.proof(collection, &tid).expect("proof");
        verify_record_proof(&repo.commit, vk, &proof, &record)
            .expect("full keyless proof must verify");
    }

    #[test]
    fn multi_collection_round_trip() {
        // Records in two collections under ONE DID, covered by ONE signed
        // commit — each independently resolvable with a valid proof (#910b).
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        publisher
            .publish("repo-a", SAMPLE_OID)
            .expect("publish model");
        publisher
            .publish_record(NAME_COLLECTION, "name-a", SAMPLE_OID_2)
            .expect("publish name");

        // One repo, one commit: both records live under the same signed root.
        let repo = store.load_repo(DID).expect("load").expect("repo");
        assert_eq!(repo.records.len(), 2, "both collections in one repo");

        resolve_and_verify(&store, &vk, COLLECTION_NSID, "repo-a");
        resolve_and_verify(&store, &vk, NAME_COLLECTION, "name-a");
    }

    #[test]
    fn ledger_multi_collection_publish_resolves() {
        // #924: an allocation + a receipt written under ONE holder DID as two new
        // ledger collections, covered by ONE signed commit and each independently
        // resolvable via the #910b path. The receipt binds a *pairwise* spender
        // id (not a root DID), exercising the pseudonymity constraint end-to-end.
        use hyprstream_pds::ledger::{
            allocation as alloc_ns, receipt as receipt_ns, AllocationRecord, GrantClass,
            ReceiptRecord, Unit,
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);

        let alloc = AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            Unit {
                code: "compute-second".into(),
                issuer: "did:web:issuer.example.com".into(),
            },
            1_000,
            7,
            "did:web:issuer.example.com",
            "did:key:zHolderSubject",
            GrantClass::Prepaid,
        )
        .expect("valid allocation");
        let receipt = ReceiptRecord::new(
            alloc.cid().encode(),
            "pairwise:cell42:8f3ad9c0e1",
            "did:key:zHostNode",
            "transfer-01HXYZ",
            25,
        )
        .expect("valid receipt");

        publisher
            .publish_ledger_allocation(&alloc)
            .expect("publish allocation");
        publisher
            .publish_ledger_receipt(&receipt)
            .expect("publish receipt");

        // One repo, one signed commit, two ledger collections.
        let repo = store.load_repo(DID).expect("load").expect("repo");
        assert_eq!(repo.records.len(), 2, "allocation + receipt in one repo");

        // Each ledger collection resolves with a valid keyless proof.
        resolve_and_verify(&store, &vk, alloc_ns::COLLECTION_NSID, &alloc.grant);
        resolve_and_verify(
            &store,
            &vk,
            receipt_ns::COLLECTION_NSID,
            &receipt.transfer_id,
        );

        // The published pointer's currentOid is the ledger record's content CID —
        // the record is content-addressed, so its bytes are recoverable/verifiable.
        let alloc_tid = PdsRecordStore::tid_for_repo(&alloc.grant);
        let ptr = repo
            .records
            .get(&(alloc_ns::COLLECTION_NSID.to_owned(), alloc_tid))
            .cloned()
            .expect("allocation pointer present");
        assert_eq!(
            ptr.current_oid,
            alloc.cid().encode(),
            "pointer must address the allocation record's content CID"
        );
    }

    #[test]
    fn cross_collection_isolation() {
        // Writing collection B must not corrupt proofs for collection A: after
        // the B write, A still resolves with a valid proof under the (new)
        // signed root, and A's record bytes are untouched.
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        publisher
            .publish("repo-a", SAMPLE_OID)
            .expect("publish model");
        let tid_a = PdsRecordStore::tid_for_repo("repo-a");
        let record_a_before = store
            .load_repo(DID)
            .expect("load")
            .expect("repo")
            .records
            .get(&(COLLECTION_NSID.to_owned(), tid_a))
            .cloned()
            .expect("record a");

        publisher
            .publish_record(NAME_COLLECTION, "name-a", SAMPLE_OID_2)
            .expect("publish name");

        // A's record bytes are unchanged and its proof verifies under the
        // advanced commit.
        let repo = store.load_repo(DID).expect("load").expect("repo");
        let record_a_after = repo
            .records
            .get(&(COLLECTION_NSID.to_owned(), tid_a))
            .cloned()
            .expect("record a still present");
        assert_eq!(
            record_a_before, record_a_after,
            "B write must not touch A's record"
        );
        resolve_and_verify(&store, &vk, COLLECTION_NSID, "repo-a");
        resolve_and_verify(&store, &vk, NAME_COLLECTION, "name-a");

        // The name record is NOT addressable under the model collection: same
        // rkey, different collection → a different MST key.
        let tid_name = PdsRecordStore::tid_for_repo("name-a");
        let resolver = PdsRecordResolver::new(Arc::clone(&store));
        let cross = block_on(resolver.resolve_record(DID, COLLECTION_NSID, &tid_name.encode()))
            .expect("resolve ok");
        assert!(
            cross.is_none(),
            "collections must not alias onto each other"
        );
    }

    #[test]
    fn concurrent_publishes_serialize_and_stay_consistent() {
        // Concurrent writers — including writers to DIFFERENT collections of
        // the same DID — must serialize on the publish lock so the final
        // commit covers every record, and every record's proof verifies
        // against that single signed root (#910b).
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let vk: VerifyingKey = *sk.verifying_key();

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = Arc::new(PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk));

        const WRITERS_PER_COLLECTION: usize = 4;
        let mut handles = Vec::new();
        for i in 0..WRITERS_PER_COLLECTION {
            for collection in [COLLECTION_NSID, NAME_COLLECTION] {
                let publisher = Arc::clone(&publisher);
                handles.push(std::thread::spawn(move || {
                    publisher
                        .publish_record(collection, &format!("rec-{i}"), SAMPLE_OID)
                        .expect("concurrent publish");
                }));
            }
        }
        for h in handles {
            h.join().expect("writer thread");
        }

        let repo = store.load_repo(DID).expect("load").expect("repo");
        assert_eq!(
            repo.records.len(),
            WRITERS_PER_COLLECTION * 2,
            "every concurrent write across both collections must survive"
        );
        // The stored commit signs the FULL record set (no lost update).
        let tree = PdsRecordStore::build_tree(&repo.records);
        assert_eq!(
            tree.root_cid(),
            repo.commit.data,
            "final signed commit must cover all concurrent writes"
        );
        // And every record still proves against that one commit.
        for i in 0..WRITERS_PER_COLLECTION {
            resolve_and_verify(&store, &vk, COLLECTION_NSID, &format!("rec-{i}"));
            resolve_and_verify(&store, &vk, NAME_COLLECTION, &format!("rec-{i}"));
        }
    }

    #[test]
    fn root_consistency_check_fails_closed() {
        // A record the stored commit does NOT cover must make reads fail
        // loudly (Err), never serve a broken proof and never report a clean
        // NOT_FOUND.
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);

        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        publisher.publish("repo-a", SAMPLE_OID).expect("publish");

        // Corrupt the store: write a second record while re-persisting the
        // STALE commit (whose root covers only the first record).
        let stale_commit = store.load_repo(DID).expect("load").expect("repo").commit;
        let rogue = ModelRecord::new(
            format!("at://{DID}"),
            "bafyreiexampleoid00000000000000000000",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("record");
        store
            .put_record_and_commit(
                DID,
                NAME_COLLECTION,
                PdsRecordStore::tid_for_repo("rogue"),
                &rogue,
                &stale_commit,
            )
            .expect("raw write");

        let resolver = PdsRecordResolver::new(Arc::clone(&store));
        let tid_a = PdsRecordStore::tid_for_repo("repo-a");
        let err = block_on(resolver.resolve_record(DID, COLLECTION_NSID, &tid_a.encode()))
            .expect_err("inconsistent store must fail closed");
        assert!(
            err.to_string().contains("inconsistent"),
            "error must name the inconsistency: {err}"
        );
        // resolve_repo fails closed the same way.
        assert!(block_on(resolver.resolve_repo(DID)).is_err());
    }

    #[test]
    fn publish_record_rejects_malformed_collection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = SigningKey::random(&mut rand::rngs::OsRng);
        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let publisher = PdsPublisher::new(Arc::clone(&store), DID.to_owned(), sk);
        for bad in ["", "with/slash", "with\0nul"] {
            assert!(
                publisher.publish_record(bad, "repo-a", SAMPLE_OID).is_err(),
                "collection {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn tid_for_repo_is_deterministic_and_distinct() {
        let first = PdsRecordStore::tid_for_repo("repo-a");
        let second = PdsRecordStore::tid_for_repo("repo-a");
        assert_eq!(first, second, "the same repo must always get the same rkey");
        let other = PdsRecordStore::tid_for_repo("repo-b");
        assert_ne!(first, other, "distinct repos must not alias onto one rkey");
    }

    struct At9pSigner {
        ed: ed25519_dalek::SigningKey,
        pq: hyprstream_rpc::crypto::pq::MlDsaSigningKey,
        pair: hyprstream_pds::at9p::HybridKeyPair,
    }

    fn at9p_signer(tag: u8) -> At9pSigner {
        use hyprstream_rpc::crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
        let mut seed = [0u8; 32];
        seed[0] = tag;
        seed[31] = tag.wrapping_add(17);
        let ed = ed25519_dalek::SigningKey::from_bytes(&seed);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let pair = hyprstream_pds::at9p::HybridKeyPair::new(
            ed.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .expect("hybrid pair");
        At9pSigner { ed, pq, pair }
    }

    fn at9p_body(
        current: &At9pSigner,
        next: &[&At9pSigner],
        service_tag: &str,
    ) -> hyprstream_pds::at9p::CapsuleBody {
        use hyprstream_pds::at9p::{
            CapsuleBody, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
        };
        let endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://{service_tag}"))
            .expect("endpoint");
        let service =
            ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).expect("service");
        let mut body =
            CapsuleBody::new(vec![current.pair.clone()], vec![service]).expect("capsule body");
        body.next_key_commitments = next
            .iter()
            .map(|signer| signer.pair.commitment_digest())
            .collect();
        body
    }

    fn audit_keys() -> (
        ed25519_dalek::SigningKey,
        hyprstream_rpc::crypto::pq::MlDsaSigningKey,
    ) {
        (
            ed25519_dalek::SigningKey::from_bytes(&[91u8; 32]),
            hyprstream_rpc::crypto::pq::ml_dsa_sk_from_seed(&[92u8; 32]),
        )
    }

    fn production_at9p_publisher(store: Arc<PdsRecordStore>, alarm: &Path) -> PdsPublisher {
        let (audit_ed, audit_pq) = audit_keys();
        let acceptance_identity = ed25519_dalek::SigningKey::from_bytes(&[90u8; 32]);
        let ingest = At9pStateIngest::open(
            Arc::clone(&store),
            alarm,
            acceptance_identity,
            audit_ed,
            audit_pq,
        )
        .expect("open accepted-state owner");
        PdsPublisher::new(
            store,
            DID.to_owned(),
            SigningKey::random(&mut rand::rngs::OsRng),
        )
        .with_at9p_state_ingest(ingest)
    }

    #[test]
    fn independent_production_guards_share_rocks_cas() {
        use hyprstream_pds::at9p_duplicity::{Admission, DuplicityGuard};
        use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
        use std::sync::Barrier;

        const NOW: &str = "2026-07-16T12:00:00Z";
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("pds");
        let store = Arc::new(PdsRecordStore::open(&db_path).expect("open"));
        let (g, n1, n2) = (at9p_signer(41), at9p_signer(42), at9p_signer(43));
        let genesis =
            sign_capsule(at9p_body(&g, &[&n1], "genesis"), &g.ed, &g.pq).expect("genesis");
        let genesis_bytes = genesis.to_dag_cbor().expect("genesis bytes");
        let cid = genesis.cid512().expect("cid");
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let verified = verify_did_at9p(&did, &genesis_bytes).expect("verified genesis");
        let acceptance_identity = ed25519_dalek::SigningKey::from_bytes(&[90u8; 32]);
        let make_guard = |alarm: &str| {
            let (audit_ed, audit_pq) = audit_keys();
            DuplicityGuard::with_durable_alarm(
                RocksAt9pStateStore {
                    store: Arc::clone(&store),
                    acceptance_identity: acceptance_identity.clone(),
                    audit_key: audit_ed.clone(),
                },
                dir.path().join(alarm),
                audit_ed,
                audit_pq,
            )
            .expect("guard")
        };
        let guard_a = Arc::new(make_guard("alarm-a"));
        let guard_b = Arc::new(make_guard("alarm-b"));
        guard_a.seed_genesis(&verified).expect("seed");
        let candidate_a = sign_update_record(
            cid.clone(),
            1,
            h512(&genesis_bytes),
            at9p_body(&n1, &[&n2], "winner-a"),
            "2099-01-01T00:00:00Z".to_owned(),
            &n1.ed,
            &n1.pq,
        )
        .expect("candidate a");
        let candidate_b = sign_update_record(
            cid,
            1,
            h512(&genesis_bytes),
            at9p_body(&n1, &[&n2], "winner-b"),
            "2099-01-01T00:00:00Z".to_owned(),
            &n1.ed,
            &n1.pq,
        )
        .expect("candidate b");
        let barrier = Arc::new(Barrier::new(3));
        let run = |guard: Arc<DuplicityGuard<_, _>>,
                   candidate: hyprstream_pds::at9p::UpdateRecord| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                guard.admit_successor(&candidate, NOW)
            })
        };
        let thread_a = run(guard_a, candidate_a);
        let thread_b = run(guard_b, candidate_b);
        barrier.wait();
        let results = [
            thread_a.join().expect("thread a"),
            thread_b.join().expect("thread b"),
        ];
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Ok(Admission::Advanced(_))))
                .count(),
            1,
            "the RocksDB CAS must allow exactly one independently locked guard to advance"
        );
        assert_eq!(
            store
                .load_at9p_state(
                    &did[DID_AT9P_PREFIX.len()..],
                    &acceptance_identity.verifying_key()
                )
                .expect("load")
                .expect("state")
                .epoch,
            1
        );
    }

    #[test]
    fn production_at9p_ingest_rotation_forks_expiry_terminal_and_restart() {
        use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};

        const NOW: &str = "2026-07-16T12:00:00Z";
        const FUTURE: &str = "2099-01-01T00:00:00Z";
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("pds");
        let alarm_path = dir.path().join("at9p-alarm.wal");
        let (genesis_signer, n1, n2, n3) = (
            at9p_signer(1),
            at9p_signer(2),
            at9p_signer(3),
            at9p_signer(4),
        );
        let genesis = sign_capsule(
            at9p_body(&genesis_signer, &[&n1], "genesis"),
            &genesis_signer.ed,
            &genesis_signer.pq,
        )
        .expect("signed genesis");
        let genesis_bytes = genesis.to_dag_cbor().expect("genesis bytes");
        let did = format!("{DID_AT9P_PREFIX}{}", genesis.cid512().expect("cid"));

        let u1 = sign_update_record(
            genesis.cid512().expect("cid"),
            1,
            hyprstream_pds::at9p::h512(&genesis_bytes),
            at9p_body(&n1, &[&n2], "rotated"),
            FUTURE.to_owned(),
            &n1.ed,
            &n1.pq,
        )
        .expect("u1");

        {
            let store = Arc::new(PdsRecordStore::open(&db_path).expect("open rw"));
            let publisher = production_at9p_publisher(store, &alarm_path);
            let seeded = publisher
                .ingest_at9p_genesis(&did, &genesis_bytes)
                .expect("GATE seed");
            assert_eq!(seeded.epoch, 0);
            assert_eq!(seeded.current.subject_keys[0], genesis_signer.pair);

            let accepted = publisher
                .ingest_at9p_successor(&did, &u1.to_dag_cbor().expect("u1 bytes"), NOW)
                .expect("accept rotation");
            assert_eq!(accepted.epoch, 1);
            assert_eq!(accepted.current.subject_keys[0], n1.pair);
            assert_ne!(accepted.current.subject_keys[0], genesis_signer.pair);
            assert_eq!(
                accepted.current.services[0].endpoint.address,
                "iroh://rotated"
            );

            // Same-epoch, separately valid fork: fail closed and durably alarm.
            let fork = sign_update_record(
                genesis.cid512().expect("cid"),
                1,
                hyprstream_pds::at9p::h512(&genesis_bytes),
                at9p_body(&n1, &[&n2], "same-epoch-fork"),
                FUTURE.to_owned(),
                &n1.ed,
                &n1.pq,
            )
            .expect("fork");
            assert!(publisher
                .ingest_at9p_successor(&did, &fork.to_dag_cbor().expect("fork bytes"), NOW)
                .is_err());
            let alarm_after_same = std::fs::metadata(&alarm_path).expect("alarm WAL").len();

            // A higher record whose link is not the durable head is never chosen.
            let off_head = sign_update_record(
                genesis.cid512().expect("cid"),
                2,
                [44u8; 64],
                at9p_body(&n2, &[&n3], "off-head"),
                FUTURE.to_owned(),
                &n2.ed,
                &n2.pq,
            )
            .expect("off-head");
            assert!(publisher
                .ingest_at9p_successor(&did, &off_head.to_dag_cbor().expect("bytes"), NOW)
                .is_err());
            assert!(std::fs::metadata(&alarm_path).expect("alarm WAL").len() > alarm_after_same);

            // Expiry rejects without changing the accepted head.
            let expired = sign_update_record(
                genesis.cid512().expect("cid"),
                2,
                accepted.head_digest,
                at9p_body(&n2, &[&n3], "expired"),
                "2026-01-01T00:00:00Z".to_owned(),
                &n2.ed,
                &n2.pq,
            )
            .expect("expired");
            assert!(publisher
                .ingest_at9p_successor(&did, &expired.to_dag_cbor().expect("bytes"), NOW)
                .is_err());
            assert_eq!(
                publisher
                    .accepted_at9p_state(&did, Some(NOW))
                    .expect("state")
                    .epoch,
                1
            );

            // A valid final rotation persists terminal state; later successors
            // fail closed before any genesis/key fallback.
            let terminal = sign_update_record(
                genesis.cid512().expect("cid"),
                2,
                accepted.head_digest,
                at9p_body(&n2, &[], "terminal"),
                FUTURE.to_owned(),
                &n2.ed,
                &n2.pq,
            )
            .expect("terminal");
            let terminal_state = publisher
                .ingest_at9p_successor(&did, &terminal.to_dag_cbor().expect("bytes"), NOW)
                .expect("accept terminal");
            assert!(terminal_state.terminal);
            let after_terminal = sign_update_record(
                genesis.cid512().expect("cid"),
                3,
                terminal_state.head_digest,
                at9p_body(&n3, &[&n3], "forbidden"),
                FUTURE.to_owned(),
                &n3.ed,
                &n3.pq,
            )
            .expect("syntactically genuine successor");
            assert!(publisher
                .ingest_at9p_successor(&did, &after_terminal.to_dag_cbor().expect("bytes"), NOW)
                .is_err());
        }

        // Reopen verifies the signed alarm WAL and recovers the complete atomic
        // state. Re-seed and rollback remain refused after restart.
        let store = Arc::new(PdsRecordStore::open(&db_path).expect("reopen rw"));
        let publisher = production_at9p_publisher(store, &alarm_path);
        let recovered = publisher
            .accepted_at9p_state(&did, Some(NOW))
            .expect("recover state");
        assert_eq!(recovered.epoch, 2);
        assert!(recovered.terminal);
        assert_eq!(recovered.current.subject_keys[0], n2.pair);
        assert!(publisher.ingest_at9p_genesis(&did, &genesis_bytes).is_err());
        assert!(publisher
            .ingest_at9p_successor(&did, &u1.to_dag_cbor().expect("u1 bytes"), NOW)
            .is_err());
    }

    #[test]
    fn accepted_at9p_read_rejects_watermark_body_mutation() {
        use hyprstream_pds::at9p_sign::sign_capsule;
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let signer = at9p_signer(11);
        let capsule = sign_capsule(at9p_body(&signer, &[], "immutable"), &signer.ed, &signer.pq)
            .expect("capsule");
        let bytes = capsule.to_dag_cbor().expect("bytes");
        let cid = capsule.cid512().expect("cid");
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let publisher = production_at9p_publisher(Arc::clone(&store), &dir.path().join("alarm"));
        publisher.ingest_at9p_genesis(&did, &bytes).expect("seed");

        let RecordBacking::ReadWrite(db) = &store.backing else {
            panic!("writer")
        };
        let key = at9p_state_key(&cid);
        let mut encoded = db.get(&key).expect("get").expect("stored state");
        encoded[AT9P_ACCEPTANCE_PREFIX_LEN + 16] ^= 1;
        // Mutate persisted epoch without changing the retained head body or
        // recomputing the daemon acceptance signature.
        db.put(&key, encoded).expect("inject mutation");
        assert!(
            publisher.accepted_at9p_state(&did, None).is_err(),
            "a partial watermark/body representation must fail closed"
        );
    }

    #[test]
    fn accepted_checkpoint_rejects_old_envelope_and_reverse_mismatch() {
        use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("pds");
        let alarm = dir.path().join("alarm");
        let store = Arc::new(PdsRecordStore::open(&db_path).expect("open"));
        let (g, n1, n2) = (at9p_signer(31), at9p_signer(32), at9p_signer(33));
        let genesis =
            sign_capsule(at9p_body(&g, &[&n1], "genesis"), &g.ed, &g.pq).expect("genesis");
        let genesis_bytes = genesis.to_dag_cbor().expect("bytes");
        let cid = genesis.cid512().expect("cid");
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let publisher = production_at9p_publisher(Arc::clone(&store), &alarm);
        publisher
            .ingest_at9p_genesis(&did, &genesis_bytes)
            .expect("seed");
        let RecordBacking::ReadWrite(db) = &store.backing else {
            panic!("writer")
        };
        let state_key = at9p_state_key(&cid);
        let checkpoint_key = at9p_checkpoint_key(&cid);
        let old_state = db.get(&state_key).unwrap().unwrap();
        let old_checkpoint = db.get(&checkpoint_key).unwrap().unwrap();
        let update = sign_update_record(
            cid.clone(),
            1,
            h512(&genesis_bytes),
            at9p_body(&n1, &[&n2], "rotated"),
            "2099-01-01T00:00:00Z".to_owned(),
            &n1.ed,
            &n1.pq,
        )
        .expect("update");
        publisher
            .ingest_at9p_successor(&did, &update.to_dag_cbor().unwrap(), "2026-07-16T12:00:00Z")
            .expect("advance");
        let new_state = db.get(&state_key).unwrap().unwrap();
        let new_checkpoint = db.get(&checkpoint_key).unwrap().unwrap();
        db.delete(&state_key).expect("simulate state-half missing");
        assert!(publisher.accepted_at9p_state(&did, None).is_err());
        db.put(&state_key, &new_state).expect("restore state");
        db.delete(&checkpoint_key)
            .expect("simulate checkpoint-half missing");
        assert!(publisher.accepted_at9p_state(&did, None).is_err());
        db.put(&checkpoint_key, &new_checkpoint)
            .expect("restore checkpoint");
        db.put(&state_key, old_state).expect("old envelope replay");
        assert!(publisher.accepted_at9p_state(&did, None).is_err());
        db.put(&state_key, new_state).expect("restore body");
        db.put(&checkpoint_key, old_checkpoint)
            .expect("old checkpoint replay");
        assert!(publisher.accepted_at9p_state(&did, None).is_err());
        db.put(&checkpoint_key, new_checkpoint)
            .expect("restore checkpoint");
        assert_eq!(publisher.accepted_at9p_state(&did, None).unwrap().epoch, 1);
        drop(publisher);
        drop(store);
        let reopened = Arc::new(PdsRecordStore::open(&db_path).expect("reopen"));
        assert_eq!(
            production_at9p_publisher(reopened, &alarm)
                .accepted_at9p_state(&did, None)
                .unwrap()
                .epoch,
            1
        );
        let acceptance_identity = ed25519_dalek::SigningKey::from_bytes(&[90u8; 32]);
        let readonly = PdsRecordStore::open_readonly(&db_path)
            .expect("open real production read boundary")
            .with_at9p_acceptance_identity(acceptance_identity.verifying_key());
        let states = readonly
            .accepted_at9p_states()
            .expect("enumerate checkpoint-verified accepted states");
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].did, did);
        assert_eq!(states[0].epoch, 1);
    }

    #[test]
    fn accepted_at9p_read_rejects_attacker_selected_signed_fork() {
        use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};

        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(PdsRecordStore::open(dir.path()).expect("open rw"));
        let (genesis_signer, n1, n2) = (at9p_signer(21), at9p_signer(22), at9p_signer(23));
        let genesis = sign_capsule(
            at9p_body(&genesis_signer, &[&n1], "genesis"),
            &genesis_signer.ed,
            &genesis_signer.pq,
        )
        .expect("genesis");
        let genesis_bytes = genesis.to_dag_cbor().expect("genesis bytes");
        let cid = genesis.cid512().expect("cid");
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let publisher = production_at9p_publisher(Arc::clone(&store), &dir.path().join("alarm"));
        publisher
            .ingest_at9p_genesis(&did, &genesis_bytes)
            .expect("seed");

        // This update is internally canonical, correctly linked, and signed by
        // the identity's committed key. It was never selected by this daemon.
        let fork = sign_update_record(
            cid.clone(),
            1,
            hyprstream_pds::at9p::h512(&genesis_bytes),
            at9p_body(&n1, &[&n2], "attacker-selected-fork"),
            "2099-01-01T00:00:00Z".to_owned(),
            &n1.ed,
            &n1.pq,
        )
        .expect("signed fork");
        let fork_state =
            AcceptedAt9pState::from_persisted_update(&fork.to_dag_cbor().expect("fork bytes"))
                .expect("internally valid fork state");

        // A process that can rewrite RocksDB but lacks the stable deployment
        // identity can only certify the substituted value with its own key.
        let attacker_identity = ed25519_dalek::SigningKey::from_bytes(&[70u8; 32]);
        let attacker_audit = ed25519_dalek::SigningKey::from_bytes(&[71u8; 32]);
        let forged = encode_at9p_state(&fork_state, &attacker_identity, &attacker_audit)
            .expect("forge self-consistent envelope");
        let RecordBacking::ReadWrite(db) = &store.backing else {
            panic!("writer")
        };
        db.put(at9p_state_key(&cid), forged)
            .expect("inject attacker-selected fork");
        assert!(
            publisher.accepted_at9p_state(&did, None).is_err(),
            "an identity-valid fork without daemon acceptance must fail closed"
        );
    }
}
