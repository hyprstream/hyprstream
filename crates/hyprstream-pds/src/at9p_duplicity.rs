//! B2 (#886): KERI-style **duplicity detection** — the `duplicity-check` layer
//! that wraps B1's [`validate_successor`] (#885) — and B3 (#887): the
//! empty/exhausted `next_key_commitments` **commitment-lifecycle semantics**.
//!
//! B1 answers "is this candidate an authorized successor of *this one*
//! predecessor?" purely and locally. That is necessary but **not sufficient**
//! for `did:at9p` trust: an attacker holding stolen pre-committed keys, or an
//! equivocating identity holder, can mint a *second, individually-valid*
//! successor that diverges from the honest chain (design #879 §7.2, threats
//! A7/A12). B1 accepts each fork in isolation; only a client that remembers what
//! it already accepted can tell they are a fork.
//!
//! # The watermark (R6 — the #879 BLOCKING-finding remediation)
//!
//! This layer persists, **per DID**, a high-watermark of exactly the pair design
//! #879 §7.2/R6 mandates:
//!
//! > `(epoch, H512(accepted record))` — the epoch **and** the accepted record
//! > digest, never the bare epoch.
//!
//! A bare-epoch watermark is the exact bug the BLOCKING review found: an
//! attacker's `epoch N+1` exceeds a naive watermark `N` while silently replacing
//! the honest key state, and "accept the max epoch" would *actively select the
//! attacker's fork*. Carrying the digest makes divergence detectable and makes
//! max-epoch selection impossible by construction:
//!
//! * a candidate that advances the chain **must chain through the persisted
//!   watermark digest** (its `predecessor` must be the watermark head) — a
//!   higher-epoch record that chains through a *different* head is a fork, not
//!   progress ([`DuplicityKind::HigherEpochFork`]);
//! * a candidate at `epoch == watermark.epoch` with a **different** digest is a
//!   same-epoch fork ([`DuplicityKind::SameEpochFork`]);
//! * any B1-valid candidate at `epoch < watermark.epoch` is a branch reaching an
//!   already-passed epoch ([`DuplicityKind::BelowWatermarkFork`]) — we hold only
//!   the high-watermark, not full history, so a valid record below it is never
//!   preferred and never silently dropped: it fails closed.
//!
//! Every one of those is **duplicity: hard fail-closed for that DID plus an
//! operator alarm** — never a mere "lower-epoch reject", never a fork selection.
//! The watermark advances **only** on a non-divergent, honestly-anchored
//! successor.
//!
//! # B3 (#887): empty / exhausted commitment semantics
//!
//! `next_key_commitments` governs whether an identity can *ever* rotate again:
//!
//! * **Empty at genesis = declared-immutable identity.** Legal and useful (design
//!   §7.1): the chain is frozen at genesis by construction and no update-record is
//!   ever a valid successor. B1 already enforces this ([`SuccessorError::
//!   ImmutableIdentity`]); here it is formalized as a **terminal watermark** —
//!   [`Watermark::terminal`] is set the moment such a state is recorded, and this
//!   layer refuses to admit any successor against a terminal watermark without
//!   even re-running B1.
//! * **A rotation to empty commitments = exhausted / frozen-forever.** A holder
//!   MAY publish a final rotation whose new capsule commits to *no* further keys.
//!   That final rotation is accepted (it is a valid successor of its predecessor);
//!   the resulting state is terminal, and the identity is frozen at that epoch
//!   forever. This is a **feature, not an error** (design §7.1): there is no
//!   recovery path, because a recovery path is a backdoor.
//!
//! Both collapse onto the same terminal-watermark mechanism: an identity whose
//! accepted head pre-commits to nothing admits no successor, ever.
//!
//! # B4 (#888): durable alarm routing
//!
//! This layer only *raises* the duplicity signal, as a typed [`DuplicityAlarm`]
//! delivered to a pluggable [`DuplicityAlarmSink`]. [`WalDuplicityAlarmSink`]
//! provides the production route: an fsync-on-write, hash-chained journal with
//! a durable head checkpoint, plus a structured operator alarm through tracing
//! / OTel. Detection and response remain decoupled; callers that intentionally
//! do not need durable routing may still select [`NoopAlarmSink`] explicitly.
//!
//! # Persistence seam (#886)
//!
//! The watermark store is a trait ([`WatermarkStore`]) with an in-memory impl
//! ([`InMemoryWatermarkStore`]) for tests and single-process use. A durable
//! backend (RocksDB/Valkey) plugs in behind the same trait — **intentionally not
//! wired here** (#886 scopes the seam, not the storage engine).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::at9p::{h512, UpdateRecord, H512_LEN};
use crate::at9p_chain::{validate_successor, ChainState, SuccessorError};

/// The persistent per-DID duplicity high-watermark: design #879 §7.2/R6's
/// `(epoch, H512(accepted record))` pair, plus the terminal flag that carries
/// B3 (#887) commitment-lifecycle state.
///
/// Storing the digest (not just the epoch) is the whole point of the
/// remediation: it is what lets [`DuplicityGuard`] tell an honest continuation
/// from a divergent fork and makes max-epoch fork selection impossible.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Watermark {
    /// Highest accepted epoch for this DID.
    pub epoch: u64,
    /// `H512` over the canonical bytes of the record accepted at [`Watermark::
    /// epoch`] — the digest half of the R6 pair. A successor advancing the chain
    /// must echo this in its `prev_record_digest`; a divergent digest at
    /// `epoch <= this.epoch` is duplicity.
    pub record_digest: [u8; H512_LEN],
    /// `true` iff the accepted head pre-commits to **no** next keys — a
    /// declared-immutable (empty at genesis) or exhausted (rotated-to-empty)
    /// identity (B3, #887). A terminal watermark admits no successor, ever.
    pub terminal: bool,
}

impl Watermark {
    /// Project the watermark a freshly-accepted [`ChainState`] represents. The
    /// state is terminal iff it pre-commits to nothing (B3).
    fn from_state(state: &ChainState) -> Self {
        Self {
            epoch: state.epoch,
            record_digest: state.record_digest,
            terminal: state.next_key_commitments.is_empty(),
        }
    }
}

/// Persistence seam for the per-DID duplicity watermark (#886).
///
/// Keyed by `subject_cid512` (the genesis cid that names the identity). The
/// in-memory [`InMemoryWatermarkStore`] is provided; a durable backend
/// (RocksDB/Valkey) implements the same three methods — **not wired here** by
/// design.
///
/// Implementations must be safe for concurrent use (`Send + Sync`). The trait
/// makes **no** atomicity guarantee across `get`/`put`: a `get` followed by a
/// `put` are two independent calls, and concurrent callers can interleave
/// between them. Atomicity of the read-modify-write in [`DuplicityGuard::
/// admit_successor`] is the **guard's** responsibility — it holds a per-DID
/// admission lock across the whole classification+advance so concurrent
/// admissions of the same DID serialize. The store owns only its own interior
/// mutability (each individual call is sound under concurrency); it does NOT
/// need to provide compare-and-swap.
pub trait WatermarkStore: Send + Sync {
    /// The current watermark for `subject_cid512`, or `None` if this DID has not
    /// been seeded yet (first contact).
    fn get(&self, subject_cid512: &str) -> anyhow::Result<Option<Watermark>>;

    /// Persist `watermark` as the new high-watermark for `subject_cid512`,
    /// replacing any prior value.
    fn put(&self, subject_cid512: &str, watermark: Watermark) -> anyhow::Result<()>;
}

/// In-memory [`WatermarkStore`] — a `Mutex<HashMap<..>>`. Suitable for tests and
/// single-process use; **not** durable across restarts. The durable seam is
/// intentionally left for #886 follow-up (no RocksDB wired here).
#[derive(Default)]
pub struct InMemoryWatermarkStore {
    marks: Mutex<HashMap<String, Watermark>>,
}

impl InMemoryWatermarkStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WatermarkStore for InMemoryWatermarkStore {
    fn get(&self, subject_cid512: &str) -> anyhow::Result<Option<Watermark>> {
        Ok(self.marks.lock().get(subject_cid512).copied())
    }

    fn put(&self, subject_cid512: &str, watermark: Watermark) -> anyhow::Result<()> {
        self.marks
            .lock()
            .insert(subject_cid512.to_owned(), watermark);
        Ok(())
    }
}

/// Which of the three fork shapes tripped duplicity detection. B4 (#888) can use
/// this to grade the operator response; all three are equally fail-closed here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DuplicityKind {
    /// A B1-valid record at `epoch == watermark.epoch` with a **different**
    /// record digest than the one accepted at that epoch — a classic equivocation
    /// (two valid rotations at the same epoch).
    SameEpochFork,
    /// A B1-valid record at `epoch < watermark.epoch`: a branch reaching an epoch
    /// the client has already passed. We hold only the high-watermark, so this is
    /// never preferred over it and never silently dropped — it fails closed.
    BelowWatermarkFork,
    /// A B1-valid record at `epoch > watermark.epoch` that chains through a
    /// predecessor other than the watermark head — a higher-epoch record on the
    /// attacker's fork rather than the client's accepted history (design #879
    /// §7.2, the "never select max epoch" case).
    HigherEpochFork,
}

/// A typed duplicity signal handed to the [`DuplicityAlarmSink`]. Carries both
/// sides of the divergence (the persisted watermark and the divergent record) so
/// the routing layer (B4, #888) has everything it needs to audit or notify
/// without re-deriving state.
#[derive(Clone, Debug)]
pub struct DuplicityAlarm {
    /// The self-certifying DID subject whose chain forked.
    pub subject_cid512: String,
    /// Which fork shape was detected.
    pub kind: DuplicityKind,
    /// Epoch of the persisted, honestly-accepted watermark.
    pub watermark_epoch: u64,
    /// Digest of the persisted, honestly-accepted watermark record.
    pub watermark_digest: [u8; H512_LEN],
    /// Epoch of the divergent (but individually B1-valid) record.
    pub divergent_epoch: u64,
    /// Digest of the divergent (but individually B1-valid) record.
    pub divergent_digest: [u8; H512_LEN],
}

/// Sink for [`DuplicityAlarm`]s (B4/#888 seam). Detection raises; **routing** the
/// signal to the S7 audit store and/or an operator notification path is B4's
/// job. The guard has already committed to rejecting before this is called;
/// durable implementations may synchronously fsync so returning from admission
/// means the alarm is crash-durable.
pub trait DuplicityAlarmSink: Send + Sync {
    /// Handle a raised duplicity alarm. Called exactly once per detected fork,
    /// after the guard has committed to failing closed.
    fn raise(&self, alarm: &DuplicityAlarm);
}

const ALARM_WAL_MAGIC: &[u8; 8] = b"AT9PALM1";
const ALARM_FIXED_PAYLOAD_LEN: usize = 1 + 8 + H512_LEN + 8 + H512_LEN;

/// Durable B4 duplicity-alarm sink.
///
/// Each alarm is appended to a length-delimited WAL and linked to the previous
/// entry by BLAKE3 (`entry_hash = BLAKE3(prev_hash || payload)`). `raise`
/// returns only after `sync_all`, so a successfully recorded alarm survives a
/// crash. Opening an existing WAL verifies every link and refuses a truncated
/// or modified journal. The same call emits a structured `ERROR` event on the
/// `hyprstream.at9p.duplicity` target, which reaches stderr and OTLP through the
/// process tracing subscriber and is therefore visible to operators.
pub struct WalDuplicityAlarmSink {
    path: PathBuf,
    checkpoint_path: PathBuf,
    state: Mutex<AlarmWalState>,
}

struct AlarmWalState {
    file: File,
    head: [u8; 32],
}

impl WalDuplicityAlarmSink {
    /// Open (or create) an alarm WAL, verifying its complete hash chain before
    /// accepting new records. A corrupt journal fails closed at startup.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let checkpoint_path = path.with_extension("checkpoint");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let head = verify_alarm_wal(&bytes)?;
        if checkpoint_path.exists() {
            let checkpoint = std::fs::read(&checkpoint_path)?;
            anyhow::ensure!(
                checkpoint.as_slice() == head,
                "duplicity alarm WAL does not match its durable checkpoint"
            );
        }
        if bytes.is_empty() {
            file.write_all(ALARM_WAL_MAGIC)?;
            file.sync_all()?;
        }
        write_alarm_checkpoint(&checkpoint_path, &head)?;
        Ok(Self {
            path,
            checkpoint_path,
            state: Mutex::new(AlarmWalState { file, head }),
        })
    }

    /// Location of the durable journal (operator diagnostics).
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn record(&self, alarm: &DuplicityAlarm) -> anyhow::Result<()> {
        let payload = encode_alarm(alarm)?;
        let mut state = self.state.lock();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&state.head);
        hasher.update(&payload);
        let entry_hash = *hasher.finalize().as_bytes();
        let previous = state.head;

        state
            .file
            .write_all(&(payload.len() as u32).to_be_bytes())?;
        state.file.write_all(&previous)?;
        state.file.write_all(&payload)?;
        state.file.write_all(&entry_hash)?;
        state.file.sync_all()?;
        write_alarm_checkpoint(&self.checkpoint_path, &entry_hash)?;
        state.head = entry_hash;
        Ok(())
    }
}

fn write_alarm_checkpoint(path: &Path, head: &[u8; 32]) -> anyhow::Result<()> {
    let temporary = path.with_extension("checkpoint.tmp");
    let mut file = File::create(&temporary)?;
    file.write_all(head)?;
    file.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

impl DuplicityAlarmSink for WalDuplicityAlarmSink {
    fn raise(&self, alarm: &DuplicityAlarm) {
        let result = self.record(alarm);
        tracing::error!(
            target: "hyprstream.at9p.duplicity",
            subject_cid512 = %alarm.subject_cid512,
            kind = ?alarm.kind,
            watermark_epoch = alarm.watermark_epoch,
            watermark_digest = %hex_digest(&alarm.watermark_digest),
            divergent_epoch = alarm.divergent_epoch,
            divergent_digest = %hex_digest(&alarm.divergent_digest),
            durable = result.is_ok(),
            error = result.as_ref().err().map(ToString::to_string).as_deref().unwrap_or(""),
            "did:at9p duplicity detected; identity remains fail-closed"
        );
    }
}

fn encode_alarm(alarm: &DuplicityAlarm) -> anyhow::Result<Vec<u8>> {
    let subject = alarm.subject_cid512.as_bytes();
    anyhow::ensure!(
        subject.len() <= u16::MAX as usize,
        "at9p subject exceeds WAL limit"
    );
    let mut out = Vec::with_capacity(2 + subject.len() + ALARM_FIXED_PAYLOAD_LEN);
    out.extend_from_slice(&(subject.len() as u16).to_be_bytes());
    out.extend_from_slice(subject);
    out.push(match alarm.kind {
        DuplicityKind::SameEpochFork => 0,
        DuplicityKind::BelowWatermarkFork => 1,
        DuplicityKind::HigherEpochFork => 2,
    });
    out.extend_from_slice(&alarm.watermark_epoch.to_be_bytes());
    out.extend_from_slice(&alarm.watermark_digest);
    out.extend_from_slice(&alarm.divergent_epoch.to_be_bytes());
    out.extend_from_slice(&alarm.divergent_digest);
    Ok(out)
}

fn verify_alarm_wal(bytes: &[u8]) -> anyhow::Result<[u8; 32]> {
    if bytes.is_empty() {
        return Ok([0; 32]);
    }
    anyhow::ensure!(
        bytes.starts_with(ALARM_WAL_MAGIC),
        "invalid duplicity alarm WAL header"
    );
    let mut cursor = ALARM_WAL_MAGIC.len();
    let mut head = [0; 32];
    while cursor < bytes.len() {
        anyhow::ensure!(
            bytes.len() - cursor >= 4 + 32,
            "truncated duplicity alarm WAL entry"
        );
        let len = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into()?) as usize;
        cursor += 4;
        let previous: [u8; 32] = bytes[cursor..cursor + 32].try_into()?;
        cursor += 32;
        anyhow::ensure!(previous == head, "duplicity alarm WAL hash-chain break");
        anyhow::ensure!(
            bytes.len() - cursor >= len + 32,
            "truncated duplicity alarm WAL payload"
        );
        let payload = &bytes[cursor..cursor + len];
        cursor += len;
        let stored: [u8; 32] = bytes[cursor..cursor + 32].try_into()?;
        cursor += 32;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&head);
        hasher.update(payload);
        let expected = *hasher.finalize().as_bytes();
        anyhow::ensure!(
            stored == expected,
            "duplicity alarm WAL entry hash mismatch"
        );
        head = stored;
    }
    Ok(head)
}

fn hex_digest(digest: &[u8; H512_LEN]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(H512_LEN * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

/// Explicit no-op sink for tests or deployments that route alarms elsewhere.
/// Detection still fails closed, but this sink provides no durable/operator
/// signal; production callers should use [`WalDuplicityAlarmSink`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopAlarmSink;

impl DuplicityAlarmSink for NoopAlarmSink {
    fn raise(&self, _alarm: &DuplicityAlarm) {}
}

/// Outcome of admitting a candidate successor through the duplicity guard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Admission {
    /// Honest forward progress: the watermark advanced to `state`, a new highest
    /// epoch that chains through the previous watermark head.
    Advanced(ChainState),
    /// The candidate is byte-identical (same epoch, same digest) to the record
    /// already at the watermark — idempotent re-presentation, watermark
    /// unchanged, no alarm.
    AlreadyKnown,
    /// Advanced to a **terminal** state (B3, #887): the accepted head pre-commits
    /// to no further keys, so the identity is frozen at `state.epoch` forever. No
    /// successor will ever be admitted after this.
    Frozen(ChainState),
}

/// Why an admission failed. Every variant fails closed.
#[derive(Debug)]
#[non_exhaustive]
pub enum AdmissionError {
    /// B1's [`validate_successor`] rejected the candidate against the supplied
    /// predecessor — it is not an authorized successor at all (propagated
    /// unchanged so callers can match the specific gate).
    Successor(SuccessorError),
    /// Duplicity: the candidate is individually B1-valid but diverges from the
    /// persisted watermark. Fail-closed; the [`DuplicityAlarm`] has already been
    /// raised on the sink. Boxed to keep the happy-path `Result` small.
    Duplicity(Box<DuplicityAlarm>),
    /// The identity's watermark is terminal (declared-immutable or exhausted, B3
    /// #887): no successor is admissible, by design. B1 is not even consulted.
    FrozenIdentity {
        /// The terminal epoch the identity is frozen at.
        epoch: u64,
    },
    /// The guard was asked to admit a successor for a DID it has never seen (no
    /// seeded watermark). Callers must [`DuplicityGuard::seed_genesis`] from a
    /// GATE-verified genesis capsule first; admitting against an unknown baseline
    /// is refused fail-closed rather than trusting an unanchored predecessor.
    NotSeeded {
        /// The DID whose watermark is missing.
        subject_cid512: String,
    },
    /// The watermark for this DID already exists and must not be re-seeded
    /// (anti-rollback, FIX in #959): `seed_genesis` refuses to overwrite an
    /// already-advanced (`epoch > 0`) or different-genesis anchor so a stray /
    /// hostile re-seed cannot roll the anchor backward.
    AlreadySeeded {
        /// The DID whose watermark already exists.
        subject_cid512: String,
        /// Epoch of the existing watermark that refused the re-seed.
        existing_epoch: u64,
    },
    /// The underlying [`WatermarkStore`] failed (I/O, poisoned lock). Fail-closed:
    /// a watermark that cannot be read or written is treated as blocking, never
    /// bypassed.
    Store(anyhow::Error),
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Successor(err) => write!(f, "duplicity guard: successor-check failed: {err}"),
            Self::Duplicity(alarm) => write!(
                f,
                "duplicity guard: DUPLICITY ({:?}) — divergent record at epoch {} against watermark epoch {}",
                alarm.kind, alarm.divergent_epoch, alarm.watermark_epoch
            ),
            Self::FrozenIdentity { epoch } => write!(
                f,
                "duplicity guard: identity is frozen at epoch {epoch} (declared-immutable or exhausted); no successor is admissible"
            ),
            Self::NotSeeded { subject_cid512 } => write!(
                f,
                "duplicity guard: no watermark seeded for {subject_cid512:?}; seed the genesis state first"
            ),
            Self::AlreadySeeded {
                subject_cid512,
                existing_epoch,
            } => write!(
                f,
                "duplicity guard: watermark for {subject_cid512:?} already exists at epoch {existing_epoch}; refusing re-seed (anti-rollback)"
            ),
            Self::Store(err) => write!(f, "duplicity guard: watermark store error: {err}"),
        }
    }
}

impl std::error::Error for AdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Successor(err) => Some(err),
            Self::Store(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// The duplicity-detecting wrapper around B1 (#886, R6) with B3 (#887)
/// commitment-lifecycle semantics folded in.
///
/// Holds the [`WatermarkStore`] (persistence seam) and the [`DuplicityAlarmSink`]
/// (B4/#888 routing seam). Usage:
///
/// 1. Verify a genesis capsule externally (GATE 1/2), build its [`ChainState`]
///    ([`ChainState::genesis`]), and [`seed_genesis`](Self::seed_genesis) it.
/// 2. For each candidate update-record, resolve its predecessor state and call
///    [`admit_successor`](Self::admit_successor). B1 runs first; the watermark
///    reconciliation runs second and can only ever *reject* a B1-valid record
///    (duplicity) or *advance* on honest progress — never select a fork.
pub struct DuplicityGuard<S: WatermarkStore, A: DuplicityAlarmSink> {
    store: S,
    alarm: A,
    /// Per-DID admission locks. Holding one of these across the whole
    /// `admit_successor`/`seed_genesis` read-modify-write is what makes the R6
    /// watermark advance atomic (see FIX in #959): without it, two concurrent
    /// divergent-but-B1-valid epoch-`(N+1)` successors of the same head could
    /// BOTH observe the head, BOTH pass B1, and BOTH `put` — defeating
    /// duplicity detection. The `WatermarkStore` trait offers no atomicity of
    /// its own, so the guard owns it here.
    admit_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl<S: WatermarkStore> DuplicityGuard<S, NoopAlarmSink> {
    /// Build a guard whose alarm routing is explicitly disabled. Production
    /// callers should use [`with_alarm`](Self::with_alarm) with
    /// [`WalDuplicityAlarmSink`].
    pub fn new(store: S) -> Self {
        Self {
            store,
            alarm: NoopAlarmSink,
            admit_locks: Mutex::new(HashMap::new()),
        }
    }
}

impl<S: WatermarkStore, A: DuplicityAlarmSink> DuplicityGuard<S, A> {
    /// Build a guard with an explicit alarm sink (the B4/#888 routing seam).
    pub fn with_alarm(store: S, alarm: A) -> Self {
        Self {
            store,
            alarm,
            admit_locks: Mutex::new(HashMap::new()),
        }
    }

    /// Run `f` while holding the per-DID admission lock, so concurrent
    /// `admit_successor` / `seed_genesis` calls for the same DID serialize
    /// their read-modify-write. The outer map lock is released as soon as the
    /// per-DID [`Arc<Mutex<()>>`] is cloned out; only the inner lock is held
    /// for the duration of `f`. (Closure-scoped rather than returning a guard
    /// because the guard borrows the inner `Mutex`, which the `Arc` owns —
    /// keeping both in this frame for `f`'s lifetime is the only sound way to
    /// hand the lock across the whole RMW.)
    fn with_admission_lock<R>(&self, subject_cid512: &str, f: impl FnOnce() -> R) -> R {
        let lock = {
            let mut map = self.admit_locks.lock();
            map.entry(subject_cid512.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = lock.lock();
        f()
    }

    /// Seed the watermark for a DID from its **GATE-verified** genesis
    /// [`ChainState`] (epoch 0).
    ///
    /// **Anti-rollback (FIX in #959):** refuses to overwrite a watermark that
    /// is already-advanced (`epoch > 0`) or that anchors a *different* genesis
    /// digest — a stray or hostile re-seed must never move the anti-rollback
    /// anchor backward. The only overwrite permitted is an idempotent re-seed
    /// of the exact same genesis (epoch 0, identical `record_digest`), which is
    /// a no-op success. The per-DID admission lock is held across the
    /// read-then-conditional-`put` so this check is race-free against a
    /// concurrent `admit_successor`.
    ///
    /// If the genesis pre-commits to no next keys, the seeded watermark is
    /// **terminal** immediately — a declared-immutable identity (B3, #887).
    /// Callers must seed only from a genesis they have themselves verified
    /// (`H512(bytes) == cid512` + self-signature).
    pub fn seed_genesis(&self, genesis: &ChainState) -> Result<(), AdmissionError> {
        let subject = &genesis.subject_cid512;
        self.with_admission_lock(subject, || {
            if let Some(existing) = self.store.get(subject).map_err(AdmissionError::Store)? {
                // Idempotent re-seed of the exact same genesis is a no-op success.
                if existing.epoch == 0 && existing.record_digest == genesis.record_digest {
                    return Ok(());
                }
                // Anything else — an already-advanced watermark, or a different
                // genesis — is the anti-rollback anchor; refuse rather than let a
                // re-seed clobber it.
                return Err(AdmissionError::AlreadySeeded {
                    subject_cid512: subject.clone(),
                    existing_epoch: existing.epoch,
                });
            }
            self.store
                .put(subject, Watermark::from_state(genesis))
                .map_err(AdmissionError::Store)
        })
    }

    /// The current persisted watermark for a DID, or `None` if unseeded.
    pub fn watermark(&self, subject_cid512: &str) -> Result<Option<Watermark>, AdmissionError> {
        self.store
            .get(subject_cid512)
            .map_err(AdmissionError::Store)
    }

    /// Admit `candidate` as a successor of `predecessor` as of `now`, enforcing
    /// B1 **and** duplicity detection (R6) **and** B3 terminal semantics,
    /// fail-closed throughout.
    ///
    /// # Safety contract — `predecessor`
    ///
    /// The caller MUST supply only a `predecessor` it has itself GATE/B1-validated
    /// or reconstructed from a record whose `H512` it verified against the
    /// watermark. The guard trusts `predecessor.record_digest` and
    /// `predecessor.next_key_commitments` (the latter is what B1's
    /// signer-committed gate keys off) and does NOT re-verify that the two are
    /// mutually consistent — a forged predecessor could otherwise authorize an
    /// attacker's pre-rotated key. **Caller-supplied unverified `ChainState` is
    /// forbidden as an authoritative input** (the CLAUDE.md MAC rule: labels /
    /// clearance / chain state are never a plaintext caller parameter treated as
    /// authoritative PDP input). The full reconstruct-from-digest invariant is a
    /// separate follow-up; until then the contract above is the boundary.
    ///
    /// # Concurrency (FIX in #959)
    ///
    /// The whole classify-then-advance read-modify-write is serialized per-DID
    /// by a held admission lock, so two concurrent divergent epoch-`(N+1)`
    /// successors of the same head result in **exactly one**
    /// [`Admission::Advanced`] (the winner, which advances the watermark) and the
    /// other(s) classified as a fork ([`DuplicityKind::SameEpochFork`] once the
    /// watermark has moved to epoch `N+1`) — never two `Advanced`.
    ///
    /// # Order of checks (each fails closed before the next)
    ///
    /// 1. **Terminal watermark (B3)** — if the DID's accepted head pre-commits to
    ///    nothing, no successor is admissible; reject without consulting B1.
    /// 2. **B1 successor-check** — [`validate_successor`]; a candidate that is not
    ///    an authorized successor of `predecessor` is rejected here.
    /// 3. **Watermark reconciliation (R6)** — compare the now-B1-valid candidate
    ///    against the persisted `(epoch, digest)` watermark:
    ///    * `epoch > watermark` **and** `predecessor` *is* the watermark head →
    ///      honest advance (watermark moves; `Frozen` if the new head is
    ///      terminal);
    ///    * `epoch > watermark` but `predecessor` is **not** the watermark head →
    ///      [`DuplicityKind::HigherEpochFork`];
    ///    * `epoch == watermark`, same digest → [`Admission::AlreadyKnown`];
    ///    * `epoch == watermark`, different digest →
    ///      [`DuplicityKind::SameEpochFork`];
    ///    * `epoch < watermark` → [`DuplicityKind::BelowWatermarkFork`].
    ///
    /// A duplicity verdict raises the alarm on the sink and returns
    /// [`AdmissionError::Duplicity`]; the watermark is left **unchanged** (a fork
    /// never moves the honest watermark).
    pub fn admit_successor(
        &self,
        predecessor: &ChainState,
        candidate: &UpdateRecord,
        now: &str,
    ) -> Result<Admission, AdmissionError> {
        let subject = &candidate.subject_cid512;
        // Hold the per-DID admission lock across the ENTIRE read-modify-write so
        // concurrent admissions of this DID serialize (FIX in #959, R6 property).
        self.with_admission_lock(subject, || {
            // A DID we've never seeded has no anchored baseline; refuse rather than
            // trust an unanchored predecessor (fail-closed first contact).
            let watermark = self
                .store
                .get(subject)
                .map_err(AdmissionError::Store)?
                .ok_or_else(|| AdmissionError::NotSeeded {
                    subject_cid512: subject.clone(),
                })?;

            // (1) B3: a terminal identity admits no successor, ever. Checked before
            // B1 so a frozen DID short-circuits without spending a signature verify —
            // and so the reason surfaced is "frozen", not an incidental B1 gate.
            if watermark.terminal {
                return Err(AdmissionError::FrozenIdentity {
                    epoch: watermark.epoch,
                });
            }

            // (2) B1: is the candidate an authorized successor of THIS predecessor?
            let state = validate_successor(predecessor, candidate, now)
                .map_err(AdmissionError::Successor)?;

            // (3) R6 watermark reconciliation. `state` is now individually valid; the
            // only question left is whether it diverges from our accepted history.
            if state.epoch > watermark.epoch {
                // Forward progress is honest ONLY if it chains through the watermark
                // head. B1 already proved `candidate.prev_record_digest ==
                // predecessor.record_digest`; requiring the predecessor to BE the
                // watermark head therefore proves the candidate chains through our
                // accepted history. A higher-epoch record anchored anywhere else is a
                // fork — this is precisely the "never select max epoch" case.
                if predecessor.record_digest != watermark.record_digest {
                    return Err(self.raise_duplicity(
                        subject,
                        DuplicityKind::HigherEpochFork,
                        &watermark,
                        state.epoch,
                        state.record_digest,
                    ));
                }
                let terminal = state.next_key_commitments.is_empty();
                self.store
                    .put(subject, Watermark::from_state(&state))
                    .map_err(AdmissionError::Store)?;
                return Ok(if terminal {
                    Admission::Frozen(state)
                } else {
                    Admission::Advanced(state)
                });
            }

            if state.epoch == watermark.epoch {
                return if state.record_digest == watermark.record_digest {
                    // Byte-identical re-presentation of the record we already hold.
                    Ok(Admission::AlreadyKnown)
                } else {
                    // Two individually-valid records at the same epoch → equivocation.
                    Err(self.raise_duplicity(
                        subject,
                        DuplicityKind::SameEpochFork,
                        &watermark,
                        state.epoch,
                        state.record_digest,
                    ))
                };
            }

            // state.epoch < watermark.epoch: a valid record below the high-watermark.
            // We keep only the watermark, not full history, so we cannot prove this is
            // the same record we accepted at that epoch — and per R6 a client that has
            // accepted epoch N never accepts < N. Fail closed as a fork; never prefer.
            Err(self.raise_duplicity(
                subject,
                DuplicityKind::BelowWatermarkFork,
                &watermark,
                state.epoch,
                state.record_digest,
            ))
        })
    }

    /// Build the [`DuplicityAlarm`], deliver it to the sink, and package the
    /// fail-closed error. The watermark is deliberately NOT mutated.
    fn raise_duplicity(
        &self,
        subject_cid512: &str,
        kind: DuplicityKind,
        watermark: &Watermark,
        divergent_epoch: u64,
        divergent_digest: [u8; H512_LEN],
    ) -> AdmissionError {
        let alarm = DuplicityAlarm {
            subject_cid512: subject_cid512.to_owned(),
            kind,
            watermark_epoch: watermark.epoch,
            watermark_digest: watermark.record_digest,
            divergent_epoch,
            divergent_digest,
        };
        self.alarm.raise(&alarm);
        AdmissionError::Duplicity(Box::new(alarm))
    }
}

/// Recompute a record's `H512` digest — the value stored as the watermark's
/// `record_digest`. Exposed so callers reconstructing a watermark from a
/// persisted record (e.g. on restart) key it the same way the guard does.
pub fn record_digest(record: &UpdateRecord) -> [u8; H512_LEN] {
    h512(&record.to_dag_cbor())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::at9p::{
        Capsule, CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use crate::at9p_sign::{sign_capsule, sign_update_record};
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use rand::rngs::OsRng;

    const NOW: &str = "2026-07-09T00:00:00Z";
    const FUTURE: &str = "2099-01-01T00:00:00Z";

    /// A hybrid signer plus its public keypair (the pre-rotation commitment
    /// preimage). Mirrors the B1 test harness.
    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer() -> Signer {
        let ed_sk = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let keypair = HybridKeyPair::new(
            ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        Signer {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    fn service_entries() -> Vec<ServiceEntry> {
        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://node0").unwrap();
        vec![ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap()]
    }

    /// Capsule body whose primary subject key is `subject`, pre-committing to
    /// each key in `next` (empty `next` = declared-immutable / exhausted).
    fn body_committing_to(subject: &Signer, next: &[&Signer]) -> CapsuleBody {
        let mut body = CapsuleBody::new(vec![subject.keypair.clone()], service_entries()).unwrap();
        body.next_key_commitments = next.iter().map(|s| s.keypair.commitment_digest()).collect();
        body
    }

    fn genesis(g: &Signer, next: &[&Signer]) -> (Capsule, ChainState) {
        let body = body_committing_to(g, next);
        let capsule = sign_capsule(body, &g.ed_sk, &g.pq_sk).unwrap();
        let state = ChainState::genesis(&capsule).unwrap();
        (capsule, state)
    }

    /// Update-record signed by `signer_key`, revealing it as the new primary
    /// subject key and pre-committing to `next`. `salt` perturbs the label hints
    /// so two otherwise-identical rotations differ (divergent digest) — the
    /// equivocation primitive.
    fn update_salted(
        subject_cid512: &str,
        epoch: u64,
        prev_digest: [u8; H512_LEN],
        signer_key: &Signer,
        next: &[&Signer],
        expires_at: &str,
        salt: Option<&str>,
    ) -> UpdateRecord {
        let mut body = body_committing_to(signer_key, next);
        if let Some(s) = salt {
            body.label_hints = Some(vec![s.to_owned()]);
        }
        sign_update_record(
            subject_cid512.to_owned(),
            epoch,
            prev_digest,
            body,
            expires_at.to_owned(),
            &signer_key.ed_sk,
            &signer_key.pq_sk,
        )
        .unwrap()
    }

    fn update(
        subject_cid512: &str,
        epoch: u64,
        prev_digest: [u8; H512_LEN],
        signer_key: &Signer,
        next: &[&Signer],
    ) -> UpdateRecord {
        update_salted(
            subject_cid512,
            epoch,
            prev_digest,
            signer_key,
            next,
            FUTURE,
            None,
        )
    }

    /// A recording sink so tests can assert an alarm fired (and inspect its kind).
    #[derive(Default)]
    struct RecordingSink {
        alarms: Mutex<Vec<DuplicityAlarm>>,
    }
    impl DuplicityAlarmSink for RecordingSink {
        fn raise(&self, alarm: &DuplicityAlarm) {
            self.alarms.lock().push(alarm.clone());
        }
    }

    fn sample_alarm() -> DuplicityAlarm {
        DuplicityAlarm {
            subject_cid512: "bafyat9ptestsubject".to_owned(),
            kind: DuplicityKind::SameEpochFork,
            watermark_epoch: 7,
            watermark_digest: [0x11; H512_LEN],
            divergent_epoch: 7,
            divergent_digest: [0x22; H512_LEN],
        }
    }

    #[test]
    fn wal_alarm_is_durable_and_reopens_with_chain_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicity.wal");
        let sink = WalDuplicityAlarmSink::open(&path).unwrap();
        sink.raise(&sample_alarm());
        let first_len = std::fs::metadata(&path).unwrap().len();
        assert!(first_len > ALARM_WAL_MAGIC.len() as u64);
        drop(sink);

        let reopened = WalDuplicityAlarmSink::open(&path).unwrap();
        reopened.raise(&DuplicityAlarm {
            kind: DuplicityKind::HigherEpochFork,
            divergent_epoch: 9,
            ..sample_alarm()
        });
        assert!(std::fs::metadata(path).unwrap().len() > first_len);
    }

    #[test]
    fn wal_alarm_refuses_tampered_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicity.wal");
        let sink = WalDuplicityAlarmSink::open(&path).unwrap();
        sink.raise(&sample_alarm());
        drop(sink);

        let mut bytes = std::fs::read(&path).unwrap();
        let payload_offset = ALARM_WAL_MAGIC.len() + 4 + 32;
        bytes[payload_offset + 3] ^= 0x80;
        std::fs::write(&path, bytes).unwrap();
        assert!(WalDuplicityAlarmSink::open(path).is_err());
    }

    #[test]
    fn wal_alarm_checkpoint_detects_tail_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duplicity.wal");
        let sink = WalDuplicityAlarmSink::open(&path).unwrap();
        sink.raise(&sample_alarm());
        drop(sink);

        std::fs::write(&path, ALARM_WAL_MAGIC).unwrap();
        assert!(WalDuplicityAlarmSink::open(path).is_err());
    }
    impl RecordingSink {
        fn count(&self) -> usize {
            self.alarms.lock().len()
        }
        fn last_kind(&self) -> Option<DuplicityKind> {
            self.alarms.lock().last().map(|a| a.kind)
        }
    }

    fn guard() -> DuplicityGuard<InMemoryWatermarkStore, RecordingSink> {
        DuplicityGuard::with_alarm(InMemoryWatermarkStore::new(), RecordingSink::default())
    }

    // ---- honest succession advances the watermark -------------------------

    #[test]
    fn watermark_advances_on_honest_succession() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();
        // Watermark starts at genesis (epoch 0, non-terminal).
        let wm0 = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert_eq!(wm0.epoch, 0);
        assert!(!wm0.terminal);

        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2]);
        let s1 = match guard.admit_successor(&s0, &r1, NOW).unwrap() {
            Admission::Advanced(s) => s,
            other => panic!("expected Advanced, got {other:?}"),
        };
        assert_eq!(s1.epoch, 1);

        // Watermark advanced to epoch 1 with r1's digest.
        let wm1 = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert_eq!(wm1.epoch, 1);
        assert_eq!(wm1.record_digest, record_digest(&r1));
        // No alarms on a clean chain.
        assert_eq!(guard.store_alarm_count(), 0);

        // Extend once more: predecessor is now the watermark head s1.
        let (n3,) = (signer(),);
        let r2 = update(&s1.subject_cid512, 2, s1.record_digest, &n2, &[&n3]);
        let s2 = match guard.admit_successor(&s1, &r2, NOW).unwrap() {
            Admission::Advanced(s) => s,
            other => panic!("expected Advanced, got {other:?}"),
        };
        assert_eq!(s2.epoch, 2);
        assert_eq!(guard.store_alarm_count(), 0);
    }

    #[test]
    fn idempotent_representation_is_already_known() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2]);
        guard.admit_successor(&s0, &r1, NOW).unwrap();
        // Present the SAME record again: idempotent, no alarm, watermark unchanged.
        let outcome = guard.admit_successor(&s0, &r1, NOW).unwrap();
        assert_eq!(outcome, Admission::AlreadyKnown);
        assert_eq!(guard.store_alarm_count(), 0);
    }

    // ---- B2 core: same-epoch divergent fork = duplicity -------------------

    #[test]
    fn same_epoch_divergent_fork_is_duplicity() {
        let (g, n1, n2a, n2b) = (signer(), signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // Honest r1 at epoch 1, accepted → watermark (1, digest_a).
        let r1a = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n2a],
            FUTURE,
            Some("honest"),
        );
        guard.admit_successor(&s0, &r1a, NOW).unwrap();

        // Divergent r1' at epoch 1: ALSO signed by the committed key n1, ALSO
        // chaining from genesis — individually B1-valid — but a different body
        // (different salt + next key) ⇒ different digest. This is the stolen-
        // next-key / equivocating-holder fork.
        let r1b = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n2b],
            FUTURE,
            Some("evil"),
        );
        assert_ne!(record_digest(&r1a), record_digest(&r1b));

        let err = guard.admit_successor(&s0, &r1b, NOW).unwrap_err();
        match err {
            AdmissionError::Duplicity(alarm) => {
                assert_eq!(alarm.kind, DuplicityKind::SameEpochFork);
                assert_eq!(alarm.watermark_epoch, 1);
                assert_eq!(alarm.watermark_digest, record_digest(&r1a));
                assert_eq!(alarm.divergent_digest, record_digest(&r1b));
            }
            other => panic!("expected Duplicity, got {other:?}"),
        }
        // The alarm fired exactly once and the honest watermark did NOT move.
        assert_eq!(guard.store_alarm_count(), 1);
        assert_eq!(guard.last_alarm_kind(), Some(DuplicityKind::SameEpochFork));
        let wm = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert_eq!(wm.record_digest, record_digest(&r1a));
    }

    // ---- B2 core: no max-epoch selection ----------------------------------
    //
    // Honest chain reaches epoch 2; a valid fork appears at epoch 1 (below the
    // watermark). It must be REJECTED, never preferred, never allowed to rewrite
    // history — and it must alarm.
    #[test]
    fn lower_epoch_fork_below_watermark_is_rejected_not_preferred() {
        let (g, n1, n2, n1_forknext) = (signer(), signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // Honest: genesis → r1 (epoch 1) → r2 (epoch 2). Watermark at epoch 2.
        let r1 = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n2],
            FUTURE,
            Some("honest-1"),
        );
        let s1 = match guard.admit_successor(&s0, &r1, NOW).unwrap() {
            Admission::Advanced(s) => s,
            other => panic!("{other:?}"),
        };
        let (n3,) = (signer(),);
        let r2 = update(&s1.subject_cid512, 2, s1.record_digest, &n2, &[&n3]);
        guard.admit_successor(&s1, &r2, NOW).unwrap();
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            2
        );

        // A divergent BUT B1-VALID fork at epoch 1: same committed signer n1, same
        // genesis linkage, different body ⇒ a genuine lower-epoch fork. Naive
        // "max epoch" logic would keep epoch 2 (fine) but must still ALARM that an
        // equivocating lower branch exists; naive "prefer/select" logic must never
        // adopt it. We reject + alarm.
        let fork1 = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n1_forknext],
            FUTURE,
            Some("fork"),
        );
        let err = guard.admit_successor(&s0, &fork1, NOW).unwrap_err();
        match err {
            AdmissionError::Duplicity(alarm) => {
                assert_eq!(alarm.kind, DuplicityKind::BelowWatermarkFork);
                assert_eq!(alarm.watermark_epoch, 2);
                assert_eq!(alarm.divergent_epoch, 1);
            }
            other => panic!("expected Duplicity(BelowWatermarkFork), got {other:?}"),
        }
        // Watermark still at the honest epoch 2 — the fork did not rewrite it.
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            2
        );
        assert_eq!(guard.store_alarm_count(), 1);
    }

    // ---- B2 core: higher-epoch record on a different fork -----------------
    //
    // A record at epoch > watermark that chains through a predecessor other than
    // the watermark head is the max-epoch-selection trap: it looks like progress.
    // It must be rejected as a fork.
    #[test]
    fn higher_epoch_off_watermark_head_is_duplicity() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // Honest r1 accepted → watermark at epoch 1 (digest_a).
        let r1a = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n2],
            FUTURE,
            Some("honest"),
        );
        let _s1a = guard.admit_successor(&s0, &r1a, NOW).unwrap();

        // Attacker builds a DIVERGENT epoch-1 record r1b (a fork head we never
        // accepted), then a "successor" r2b at epoch 2 chaining through r1b. r2b
        // is individually B1-valid against r1b's state, and epoch 2 > watermark 1
        // — the classic "just take the higher epoch" bait.
        let (n2b, n3) = (signer(), signer());
        let r1b = update_salted(
            &s0.subject_cid512,
            1,
            s0.record_digest,
            &n1,
            &[&n2b],
            FUTURE,
            Some("evil"),
        );
        let s1b = ChainState::from_validated_update(&r1b);
        let r2b = update(&s1b.subject_cid512, 2, s1b.record_digest, &n2b, &[&n3]);

        // Present r2b with its own (fork) predecessor s1b. B1 passes; the guard
        // catches that s1b is NOT our watermark head.
        let err = guard.admit_successor(&s1b, &r2b, NOW).unwrap_err();
        match err {
            AdmissionError::Duplicity(alarm) => {
                assert_eq!(alarm.kind, DuplicityKind::HigherEpochFork);
                assert_eq!(alarm.watermark_epoch, 1);
                assert_eq!(alarm.divergent_epoch, 2);
            }
            other => panic!("expected Duplicity(HigherEpochFork), got {other:?}"),
        }
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            1
        );
        assert_eq!(guard.store_alarm_count(), 1);
    }

    // ---- B3: declared-immutable identity has a terminal watermark ---------

    #[test]
    fn immutable_identity_has_terminal_watermark() {
        let (g, n1) = (signer(), signer());
        // Genesis pre-committing to NOTHING = declared-immutable.
        let (_cap, s0) = genesis(&g, &[]);
        assert!(s0.next_key_commitments.is_empty());
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // The seeded watermark is terminal at epoch 0.
        let wm = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert!(wm.terminal);
        assert_eq!(wm.epoch, 0);

        // Any attempted successor is refused as frozen — B1 is not even reached.
        let rec = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n1]);
        let err = guard.admit_successor(&s0, &rec, NOW).unwrap_err();
        assert!(
            matches!(err, AdmissionError::FrozenIdentity { epoch: 0 }),
            "expected FrozenIdentity, got {err:?}"
        );
        assert_eq!(guard.store_alarm_count(), 0);
    }

    // ---- B3: exhausted / frozen-forever -----------------------------------
    //
    // A holder publishes a final rotation committing to no further keys. The
    // rotation is ACCEPTED (valid successor), the resulting state is terminal, and
    // every later successor is denied — no recovery path, by design.
    #[test]
    fn exhausted_rotation_accepts_then_freezes() {
        let (g, n1, later) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // Final rotation r1: signed by committed n1, pre-committing to EMPTY.
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[]);
        let s1 = match guard.admit_successor(&s0, &r1, NOW).unwrap() {
            Admission::Frozen(s) => s,
            other => panic!("expected Frozen (exhausted rotation accepted), got {other:?}"),
        };
        assert_eq!(s1.epoch, 1);
        assert!(s1.next_key_commitments.is_empty());

        // Watermark is now terminal at epoch 1.
        let wm = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert!(wm.terminal);
        assert_eq!(wm.epoch, 1);

        // Any further rotation is denied — frozen forever, no recovery.
        let r2 = update(&s1.subject_cid512, 2, s1.record_digest, &later, &[&later]);
        let err = guard.admit_successor(&s1, &r2, NOW).unwrap_err();
        assert!(
            matches!(err, AdmissionError::FrozenIdentity { epoch: 1 }),
            "expected FrozenIdentity at epoch 1, got {err:?}"
        );
        assert_eq!(guard.store_alarm_count(), 0);
    }

    // ---- fail-closed plumbing ---------------------------------------------

    #[test]
    fn unseeded_did_is_refused() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        // No seed_genesis.
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2]);
        let err = guard.admit_successor(&s0, &r1, NOW).unwrap_err();
        assert!(
            matches!(err, AdmissionError::NotSeeded { .. }),
            "expected NotSeeded, got {err:?}"
        );
    }

    #[test]
    fn b1_rejection_propagates_without_alarm() {
        let (g, n1, evil, n2) = (signer(), signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();
        // Signed by an uncommitted key → B1 SignerNotCommitted, NOT duplicity.
        let rec = update(&s0.subject_cid512, 1, s0.record_digest, &evil, &[&n2]);
        let err = guard.admit_successor(&s0, &rec, NOW).unwrap_err();
        assert!(
            matches!(
                err,
                AdmissionError::Successor(SuccessorError::SignerNotCommitted)
            ),
            "expected Successor(SignerNotCommitted), got {err:?}"
        );
        // A B1 failure is not a fork — no alarm.
        assert_eq!(guard.store_alarm_count(), 0);
    }

    // ---- FIX (#959): atomic per-DID admission under concurrency -----------
    //
    // Two+ threads present divergent epoch-2 successors of the SAME head. Each
    // is individually B1-valid; without the per-DID admission lock both could
    // read the head, both pass, both put → two Advanced. With the lock, exactly
    // one Advanced and the rest are duplicity; the watermark reflects only the
    // winner.
    #[test]
    fn concurrent_divergent_successors_of_same_head_yield_exactly_one_advance() {
        const N: usize = 4;
        let (g, n1, n2) = (signer(), signer(), signer());
        let nexts: Vec<Signer> = (0..N).map(|_| signer()).collect();
        let next_refs: Vec<&Signer> = nexts.iter().collect();
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        guard.seed_genesis(&s0).unwrap();

        // Honest r1 (epoch 1) accepted → watermark at (1, digest_r1).
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2]);
        let s1 = match guard.admit_successor(&s0, &r1, NOW).unwrap() {
            Admission::Advanced(s) => s,
            other => panic!("expected Advanced, got {other:?}"),
        };
        assert_eq!(guard.store_alarm_count(), 0);

        // N divergent epoch-2 successors of s1: each signed by the key committed
        // at epoch 1 (n2), each chaining from s1, but a distinct body (distinct
        // next key + salt) ⇒ a distinct digest.
        let records: Vec<UpdateRecord> = (0..N)
            .map(|i| {
                update_salted(
                    &s1.subject_cid512,
                    2,
                    s1.record_digest,
                    &n2,
                    &[next_refs[i]],
                    FUTURE,
                    Some(&format!("fork-{i}")),
                )
            })
            .collect();
        let digests: Vec<_> = records.iter().map(record_digest).collect();
        let mut deduped = digests.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), N, "records must all diverge");

        let outcomes: Mutex<Vec<Result<Admission, AdmissionError>>> = Mutex::new(Vec::new());
        let guard = &guard;
        let s1 = &s1;
        std::thread::scope(|scope| {
            for rec in &records {
                let outcomes = &outcomes;
                scope.spawn(move || {
                    outcomes.lock().push(guard.admit_successor(s1, rec, NOW));
                });
            }
        });

        let outcomes = outcomes.into_inner();
        assert_eq!(outcomes.len(), N, "every admission must complete");
        let advanced = outcomes
            .iter()
            .filter(|o| matches!(o, Ok(Admission::Advanced(_))))
            .count();
        assert_eq!(advanced, 1, "exactly one Advanced; got {outcomes:?}");
        let forks = outcomes
            .iter()
            .filter(|o| matches!(o, Err(AdmissionError::Duplicity(_))))
            .count();
        assert_eq!(forks, N - 1);
        // Every fork verdict is a same-epoch fork (the watermark moved to epoch
        // 2 under the winner, so each loser sees its own epoch-2 digest differ).
        let all_same_epoch = outcomes.iter().all(|o| match o {
            Err(AdmissionError::Duplicity(a)) => a.kind == DuplicityKind::SameEpochFork,
            Ok(Admission::Advanced(_)) => true,
            _ => false,
        });
        assert!(all_same_epoch, "non-fork outcome: {outcomes:?}");

        // The watermark reflects ONLY the winner: epoch 2, exactly one of the
        // candidate digests, and exactly one alarm was raised.
        let wm = guard.watermark(&s0.subject_cid512).unwrap().unwrap();
        assert_eq!(wm.epoch, 2);
        assert!(
            digests.contains(&wm.record_digest),
            "watermark digest must be one of the candidates"
        );
        assert_eq!(guard.store_alarm_count(), N - 1);
    }

    // ---- FIX (#959): seed_genesis is rollback-safe ------------------------
    #[test]
    fn seed_genesis_refuses_to_roll_back_after_advance() {
        let (g, n1, n2) = (signer(), signer(), signer());
        let (_cap, s0) = genesis(&g, &[&n1]);
        let guard = guard();
        // First seed and an idempotent re-seed of the exact same genesis are OK.
        guard.seed_genesis(&s0).unwrap();
        guard.seed_genesis(&s0).unwrap();
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            0
        );

        // Advance the chain to epoch 1.
        let r1 = update(&s0.subject_cid512, 1, s0.record_digest, &n1, &[&n2]);
        guard.admit_successor(&s0, &r1, NOW).unwrap();
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            1
        );

        // A stray / hostile re-seed must NOT roll the anchor back to genesis.
        let err = guard.seed_genesis(&s0).unwrap_err();
        assert!(
            matches!(
                err,
                AdmissionError::AlreadySeeded {
                    existing_epoch: 1,
                    ..
                }
            ),
            "expected AlreadySeeded, got {err:?}"
        );
        // Watermark unchanged at the honest epoch 1.
        assert_eq!(
            guard.watermark(&s0.subject_cid512).unwrap().unwrap().epoch,
            1
        );
        assert_eq!(guard.store_alarm_count(), 0);
    }

    // Small accessors so the guard's sink can be inspected in-test.
    impl DuplicityGuard<InMemoryWatermarkStore, RecordingSink> {
        fn store_alarm_count(&self) -> usize {
            self.alarm.count()
        }
        fn last_alarm_kind(&self) -> Option<DuplicityKind> {
            self.alarm.last_kind()
        }
    }
}
