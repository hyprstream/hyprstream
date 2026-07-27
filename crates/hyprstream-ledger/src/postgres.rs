//! `PostgresLedger` — the production durable backend (PAY-01 #1389).
//!
//! Single-writer double-entry ledger backed by dedicated Postgres (PAY-00 #6).
//! Shares the same [`crate::engine`] as [`crate::MemLedger`], so "same ops ⇒
//! same outcomes" holds by construction.
//!
//! ## Architecture: in-memory mirror + transactional DB writes
//!
//! `PostgresLedger` holds an in-memory mirror of the full working set
//! (accounts, pending, head, clock) — exactly like `MemLedger`. Every op:
//! 1. Stages against the mirror via `engine::stage()` (pure, no I/O).
//! 2. Writes journal + deltas + outcome to Postgres in one DB transaction.
//!    The DB transaction performs the idempotency check against
//!    `ledger_outcomes` (DB is the source of truth — never the mirror).
//! 3. On DB success, updates the mirror.
//!
//! ## Single writer per cell (the property the mirror depends on)
//!
//! Staging computes **absolute** account counters from the mirror. Those
//! counters only mean anything relative to the state they were derived from,
//! so persisting deltas staged against a stale mirror is precisely a lost
//! update — the later writer's absolute values overwrite the earlier writer's
//! committed ones. Serializing the *database transactions* does not fix this,
//! because the staleness happened before the transaction began.
//!
//! `PostgresLedger` therefore enforces one writer per cell, and proves it
//! rather than assuming it. Three mechanisms, in order of when they act:
//!
//! 1. **Admission — the writer lease.** [`PostgresLedger::connect`] claims a
//!    durable lease row (`ledger_meta.writer_lease`) carrying a monotonic
//!    fencing `epoch`, an owner tag, and a heartbeat. A live lease held by
//!    another instance makes `connect` fail with
//!    [`LedgerError::WriterLeaseHeld`]; a second writer never gets built.
//! 2. **Fencing — the epoch check.** Every commit re-reads the lease *inside*
//!    its transaction and refuses unless the epoch still matches the one this
//!    instance acquired ([`LedgerError::WriterLeaseLost`]). Safety here does
//!    not depend on connection liveness, process liveness, or clock skew: a
//!    partitioned or paused instance whose lease was taken over cannot commit,
//!    because the epoch it must match has already moved. The heartbeat TTL
//!    only decides how quickly takeover becomes *available*; it is never what
//!    makes takeover *safe*.
//! 3. **Proof — the head compare-and-set.** The same transaction asserts the
//!    authoritative head still equals the head the mirror staged against, and
//!    aborts with [`LedgerError::MirrorStale`] otherwise. Under correct
//!    single-writer operation this can never fire; it exists so that "the
//!    staged deltas describe the state that produced this head" is a checked
//!    invariant rather than a comment. Any violation poisons the instance.
//!
//! ## Restart: the mirror is rebuilt from the committed journal
//!
//! The materialized tables are a projection, never the source of truth. On
//! startup the mirror is rebuilt by **replaying the committed journal** through
//! the same [`crate::engine`] that produced it, which additionally re-derives
//! each entry's result and requires it to match what was recorded. The rebuild
//! checks sequence contiguity and the hash chain, requires the recomputed tail
//! to equal `ledger_meta.head`, and finally requires the materialized tables to
//! equal the replayed state. Any disagreement is fatal, not repaired silently.
//!
//! ## Runtime: dedicated background thread
//!
//! The `LedgerBackend` trait is sync (PLAN DECISION 8). `PostgresLedger` owns
//! a dedicated `std::thread` with a `deadpool-postgres` pool. Sync methods
//! send closures and block on the reply — runtime-independent, deadlock-free.

#![cfg(feature = "postgres")]

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::backend::LedgerBackend;
use crate::engine::{self, Op};
use crate::errors::LedgerError;
use crate::journal::{
    balances_root, pending_root, ChainHead, CheckpointContent, CheckpointSigner, JournalEntry,
    OutboxItem, OutboxKind, OutboxSeq, SignedCheckpoint, TickReport,
};
use crate::types::{
    Account, AccountId, AccountSpec, BalanceView, Did, IssueTransfer, Outcome, PendingReservation,
    PendingState, Transfer, TransferId, TransferResult,
};

pub use self::config::PostgresConfig;
pub const LEDGER_SCHEMA: &str = include_str!("../sql/ledger_schema.sql");
/// Idempotent upgrade of a database created by an earlier build (R4 finding 4).
pub const LEDGER_MIGRATE: &str = include_str!("../sql/ledger_migrate.sql");

/// The schema version this build writes and can serve.
const SCHEMA_VERSION: u64 = 2;

/// `ledger_meta` key holding the single-writer fencing lease.
const META_WRITER_LEASE: &str = "writer_lease";
/// `ledger_meta` key holding the schema version.
const META_SCHEMA_VERSION: &str = "schema_version";

/// How long a lease heartbeat stays valid. A lease whose heartbeat is older
/// than this is considered abandoned and may be taken over.
///
/// This bound governs **availability only** — how soon a crashed writer's cell
/// can be served again. It is deliberately not part of the safety argument:
/// takeover is made safe by bumping the fencing epoch, which the previous
/// holder must match inside its own commit transaction. A writer that is
/// merely slow, paused, or partitioned can therefore never corrupt state, it
/// can only lose its lease and fail closed.
const LEASE_TTL_SECS: u64 = 30;

/// The shape persisted in both `ledger_journal.result_cbor` and
/// `ledger_outcomes.result_cbor`. Always the full `Result` — readers must
/// decode this exact type (F2).
type PersistedResult = Result<TransferResult, LedgerError>;

mod config {
    #[derive(Debug, Clone)]
    pub struct PostgresConfig {
        pub url: String,
        pub pool_size: usize,
    }

    impl Default for PostgresConfig {
        fn default() -> Self {
            Self {
                url: "postgres://localhost/hyprstream_ledger".to_owned(),
                pool_size: 4,
            }
        }
    }
}

/// A job sent to the bg thread.
type DbJob =
    Box<dyn FnOnce(&deadpool_postgres::Pool) -> Result<Vec<u8>, LedgerError> + Send + 'static>;

enum BgCmd {
    Job(DbJob, mpsc::SyncSender<Result<Vec<u8>, LedgerError>>),
    Shutdown,
}

/// Production durable Postgres ledger backend.
///
/// In-memory mirror (rehydrated from DB on startup) + a dedicated bg thread
/// with a Postgres connection pool.
pub struct PostgresLedger {
    tx: mpsc::Sender<BgCmd>,
    _thread: Option<JoinHandle<()>>,
    ledger_id: Did,
    /// The fencing epoch this instance acquired. Every commit requires the
    /// database lease to still carry exactly this epoch.
    lease_epoch: u64,
    /// This instance's lease owner tag, used to distinguish "still mine" from
    /// "reclaimed by someone else who happened to reach the same epoch".
    lease_owner: [u8; 16],
    /// Set once an integrity invariant has been violated. Terminal: every
    /// subsequent mutating op fails closed with [`LedgerError::Poisoned`].
    poisoned: Option<String>,
    accounts: BTreeMap<AccountId, Account>,
    pending: BTreeMap<TransferId, PendingReservation>,
    outcomes: BTreeMap<TransferId, Outcome>,
    head: ChainHead,
    clock: u64,
    last_checkpoint: Option<SignedCheckpoint>,
    /// Verifies issuance authorizations. Defaults to
    /// [`DenyAllMintVerifier`](crate::mint::DenyAllMintVerifier) so a ledger
    /// nobody explicitly granted an issuance authority cannot mint.
    mint_verifier: Box<dyn crate::mint::MintVerifier>,
}

/// The durable single-writer lease (`ledger_meta.writer_lease`).
///
/// Wire layout is fixed at 32 bytes: `epoch(8) || owner(16) || heartbeat(8)`,
/// all integers big-endian. A `heartbeat` of 0 means the lease was released
/// cleanly on shutdown and may be claimed immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriterLease {
    epoch: u64,
    owner: [u8; 16],
    heartbeat: u64,
}

impl WriterLease {
    fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(32);
        v.extend_from_slice(&self.epoch.to_be_bytes());
        v.extend_from_slice(&self.owner);
        v.extend_from_slice(&self.heartbeat.to_be_bytes());
        v
    }

    fn decode(b: &[u8]) -> Result<Self, LedgerError> {
        if b.len() != 32 {
            return Err(LedgerError::CorruptRow(format!(
                "writer_lease expected 32 bytes, got {}",
                b.len()
            )));
        }
        let mut epoch = [0u8; 8];
        epoch.copy_from_slice(&b[..8]);
        let mut owner = [0u8; 16];
        owner.copy_from_slice(&b[8..24]);
        let mut hb = [0u8; 8];
        hb.copy_from_slice(&b[24..32]);
        Ok(WriterLease {
            epoch: u64::from_be_bytes(epoch),
            owner,
            heartbeat: u64::from_be_bytes(hb),
        })
    }

    /// Whether this lease still looks live to a would-be claimant at `now`.
    ///
    /// A heartbeat in the future (clock skew between writers) is treated as
    /// live — erring toward refusing admission rather than toward admitting a
    /// second writer.
    fn is_live_at(&self, now: u64) -> bool {
        self.heartbeat != 0 && now < self.heartbeat.saturating_add(LEASE_TTL_SECS)
    }
}

/// Wall-clock seconds, used only for lease heartbeats (never for ledger
/// semantics, which run on the logical commit clock).
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Derive a per-instance owner tag. Uniqueness only needs to be good enough to
/// distinguish concurrent instances; the fencing epoch, not this tag, is what
/// makes takeover safe.
fn new_owner_tag(ledger_id: &Did) -> [u8; 16] {
    let mut h = blake3::Hasher::new();
    h.update(b"hs-ledger-writer-owner-v1");
    h.update(ledger_id.0.as_bytes());
    h.update(&std::process::id().to_be_bytes());
    h.update(
        &std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    let mut out = [0u8; 16];
    h.finalize_xof().fill(&mut out);
    out
}

/// Per-cell advisory lock key. Distinct cells must not serialize against each
/// other, so the key is derived from the ledger id rather than a global
/// constant string.
fn advisory_key(ledger_id: &Did) -> i64 {
    let mut h = blake3::Hasher::new();
    h.update(b"hs-ledger-writer-lock-v1");
    h.update(ledger_id.0.as_bytes());
    let mut out = [0u8; 8];
    h.finalize_xof().fill(&mut out);
    i64::from_be_bytes(out)
}

impl std::fmt::Debug for PostgresLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresLedger")
            .field("ledger_id", &self.ledger_id)
            .field("head", &self.head)
            .field("accounts", &self.accounts.len())
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl PostgresLedger {
    /// Connect, run migrations, rehydrate the mirror, verify the chain.
    pub fn connect(config: PostgresConfig, ledger_id: Did) -> Result<Self, LedgerError> {
        let (tx, rx) = mpsc::channel::<BgCmd>();
        let thread = std::thread::Builder::new()
            .name("postgres-ledger-bg".to_owned())
            .spawn(move || bg_main(rx, &config))
            .map_err(|e| LedgerError::Internal(format!("pg thread spawn: {e}")))?;

        let lease_owner = new_owner_tag(&ledger_id);
        let mut ledger = Self {
            tx,
            _thread: Some(thread),
            ledger_id,
            lease_epoch: 0,
            lease_owner,
            poisoned: None,
            accounts: BTreeMap::new(),
            pending: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            head: ChainHead::default(),
            clock: 0,
            last_checkpoint: None,
            mint_verifier: Box::new(crate::mint::DenyAllMintVerifier),
        };
        ledger.run_migrations()?;
        // Claim the writer lease BEFORE reading any authoritative state, so
        // the state this instance rebuilds from is state no one else may
        // concurrently be advancing.
        ledger.acquire_writer_lease()?;
        ledger.rebuild_from_journal()?;
        Ok(ledger)
    }

    /// Install the issuance authority for this ledger.
    ///
    /// Without this, [`LedgerBackend::credit`] can never succeed: the default
    /// verifier refuses every authorization, so an un-configured ledger is
    /// mint-disabled rather than mint-open.
    #[must_use]
    pub fn with_mint_verifier(mut self, verifier: Box<dyn crate::mint::MintVerifier>) -> Self {
        self.mint_verifier = verifier;
        self
    }

    /// Claim the single-writer lease for this cell, or refuse to start.
    ///
    /// Runs as one serialized transaction: take the per-cell advisory lock,
    /// read the lease row `FOR UPDATE`, refuse if it is still live, otherwise
    /// install a lease at `epoch + 1`. The epoch is **monotonic across
    /// takeovers**, which is what lets a previous holder detect that it has
    /// been fenced out.
    fn acquire_writer_lease(&mut self) -> Result<(), LedgerError> {
        let key = advisory_key(&self.ledger_id);
        let owner = self.lease_owner;
        let now = now_unix();

        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let mut client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let tx = client
                    .transaction()
                    .await
                    .map_err(|e| ie(format!("lease begin: {e}")))?;
                tx.execute("SELECT pg_advisory_xact_lock($1)", &[&key])
                    .await
                    .map_err(|e| ie(format!("lease lock: {e}")))?;

                let row = tx
                    .query_opt(
                        "SELECT value FROM ledger_meta WHERE key = $1 FOR UPDATE",
                        &[&META_WRITER_LEASE],
                    )
                    .await
                    .map_err(|e| ie(format!("lease select: {e}")))?;

                let existing = match row {
                    Some(r) => {
                        let v: Vec<u8> = r.get(0);
                        Some(WriterLease::decode(&v)?)
                    }
                    None => None,
                };

                if let Some(prev) = existing {
                    if prev.is_live_at(now) && prev.owner != owner {
                        // Someone else is actively writing this cell.
                        let mut buf = Vec::new();
                        cbor(&(false, prev.epoch), &mut buf)?;
                        return Ok(buf);
                    }
                }

                let next = WriterLease {
                    epoch: existing.map_or(1, |p| p.epoch.saturating_add(1)),
                    owner,
                    heartbeat: now,
                };
                tx.execute(
                    "INSERT INTO ledger_meta (key, value) VALUES ($1, $2) \
                     ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    &[&META_WRITER_LEASE, &next.encode().as_slice()],
                )
                .await
                .map_err(|e| ie(format!("lease upsert: {e}")))?;
                tx.commit()
                    .await
                    .map_err(|e| ie(format!("lease commit: {e}")))?;

                let mut buf = Vec::new();
                cbor(&(true, next.epoch), &mut buf)?;
                Ok(buf)
            })
        })?;

        let (acquired, epoch): (bool, u64) = ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("lease decode: {e}")))?;
        if !acquired {
            return Err(LedgerError::WriterLeaseHeld { epoch });
        }
        self.lease_epoch = epoch;
        tracing::info!(
            ledger_id = %self.ledger_id.0,
            epoch,
            "PostgresLedger acquired the single-writer lease"
        );
        Ok(())
    }

    /// Mark this instance unusable for writes. Terminal and fail-closed: a
    /// mirror that has been proven wrong is never silently repaired mid-flight.
    fn poison(&mut self, why: String) {
        if self.poisoned.is_none() {
            tracing::error!(ledger_id = %self.ledger_id.0, "PostgresLedger poisoned: {why}");
            self.poisoned = Some(why);
        }
    }

    fn poison_check(&self) -> Result<(), LedgerError> {
        match &self.poisoned {
            Some(why) => Err(LedgerError::Poisoned(why.clone())),
            None => Ok(()),
        }
    }

    /// Verify the committed journal end-to-end, without mutating the mirror.
    ///
    /// Checks, for every entry: sequence **contiguity** (no gaps — a gap means
    /// entries are missing, which a per-entry hash check alone would not
    /// notice), chain linkage (`prev_hash` equals the prior `head_hash`), and
    /// hash integrity (the recomputed entry hash equals the stored one).
    /// Finally requires the recomputed tail to equal `ledger_meta.head`, so a
    /// truncated journal or a head pointer that ran ahead of the journal is
    /// caught rather than silently accepted. All decode paths fail closed.
    pub fn verify_chain(&self) -> Result<(), LedgerError> {
        let mut expected_seq = 1u64;
        let mut expected_prev = [0u8; 32];
        let mut from = 1u64;
        loop {
            let rows = self.load_journal_batch(from, JOURNAL_BATCH)?;
            if rows.is_empty() {
                break;
            }
            for row in &rows {
                let entry = row.verify(expected_seq, expected_prev)?;
                expected_prev = row.head_hash;
                expected_seq = entry.seq.saturating_add(1);
            }
            from = expected_seq;
        }

        let tail = ChainHead {
            seq: expected_seq.saturating_sub(1),
            head_hash: expected_prev,
        };
        let (meta_head, _) = self.load_meta()?;
        if tail != meta_head {
            return Err(LedgerError::Internal(format!(
                "journal tail (seq {}) disagrees with ledger_meta.head (seq {})",
                tail.seq, meta_head.seq
            )));
        }
        Ok(())
    }

    /// Rebuild the mirror strictly from the committed journal, then prove the
    /// materialized tables agree with it.
    ///
    /// The materialized `ledger_accounts` / `ledger_pending` / `ledger_outcomes`
    /// tables are a projection for fast reads; the journal is the ledger. So
    /// startup replays the journal through the same [`crate::engine`] that
    /// produced it and *derives* the state, rather than trusting the
    /// projection. Because the replay re-runs the state machine, it also
    /// re-derives each entry's result and requires it to equal what was
    /// recorded — a determinism check the projection could never provide.
    fn rebuild_from_journal(&mut self) -> Result<(), LedgerError> {
        self.accounts.clear();
        self.pending.clear();
        self.outcomes.clear();
        self.head = ChainHead::default();
        self.clock = 0;

        let replayed = self.replay_journal_from(1)?;

        let (meta_head, meta_clock) = self.load_meta()?;
        if self.head != meta_head {
            return Err(LedgerError::Internal(format!(
                "replayed journal head (seq {}) disagrees with ledger_meta.head (seq {})",
                self.head.seq, meta_head.seq
            )));
        }
        // The persisted clock only ever moves forward (it is advanced by
        // `tick`), so it may legitimately lead the last journal entry's ts.
        self.clock = self.clock.max(meta_clock);

        self.assert_materialized_matches_journal()?;
        self.last_checkpoint = self.load_last_checkpoint()?;

        tracing::info!(
            ledger_id = %self.ledger_id.0,
            "PostgresLedger rebuilt from journal: {replayed} entries, {} accounts, \
             {} pending, {} outcomes, head seq={}, checkpoint={}",
            self.accounts.len(),
            self.pending.len(),
            self.outcomes.len(),
            self.head.seq,
            self.last_checkpoint.is_some()
        );
        Ok(())
    }

    /// Replay committed journal entries from `from_seq` into the mirror,
    /// returning how many were applied.
    ///
    /// Used both for the full startup rebuild (`from_seq = 1`) and to adopt
    /// entries this instance discovers it committed but had not yet applied
    /// (the ambiguous-commit path), which is why it starts from the current
    /// mirror head rather than always from the beginning.
    fn replay_journal_from(&mut self, from_seq: u64) -> Result<u64, LedgerError> {
        let mut expected_seq = from_seq;
        let mut expected_prev = if from_seq <= 1 {
            [0u8; 32]
        } else {
            self.head.head_hash
        };
        let mut applied = 0u64;

        loop {
            let rows = self.load_journal_batch(expected_seq, JOURNAL_BATCH)?;
            if rows.is_empty() {
                break;
            }
            for row in rows {
                let entry = row.verify(expected_seq, expected_prev)?;

                // Deadlines and on-touch expiry are judged against the logical
                // commit clock, so replay must reinstate the clock the original
                // commit ran under or it would re-derive different results.
                self.clock = entry.ts;
                let staged = engine::stage(self, &entry.op);
                if staged.result != entry.result {
                    return Err(LedgerError::Internal(format!(
                        "journal replay diverged at seq {}: recomputed result does not match \
                         the recorded result",
                        entry.seq
                    )));
                }
                for delta in staged.deltas {
                    match delta {
                        engine::Delta::Account(a) => {
                            self.accounts.insert(a.id, a);
                        }
                        engine::Delta::Pending(r) => {
                            self.pending.insert(r.transfer.id, r);
                        }
                    }
                }
                if let Some(id) = entry.op.idempotency_id() {
                    self.outcomes.insert(
                        id,
                        Outcome {
                            result: entry.result.clone(),
                            seq: entry.seq,
                        },
                    );
                }
                self.head = ChainHead {
                    seq: entry.seq,
                    head_hash: row.head_hash,
                };
                expected_prev = row.head_hash;
                expected_seq = entry.seq.saturating_add(1);
                applied += 1;
            }
        }
        Ok(applied)
    }

    /// Require the materialized projection to equal the journal-derived state.
    ///
    /// This is the check that turns "the tables are maintained in the same
    /// transaction as the journal" from an implementation claim into a
    /// verified startup property. A mismatch means some write path diverged
    /// from the state machine, so it is fatal rather than repaired.
    fn assert_materialized_matches_journal(&self) -> Result<(), LedgerError> {
        let (accounts, pending, outcomes) = self.load_materialized()?;

        if accounts != self.accounts {
            return Err(LedgerError::Internal(format!(
                "materialized accounts disagree with the journal ({} materialized vs {} replayed)",
                accounts.len(),
                self.accounts.len()
            )));
        }
        if pending != self.pending {
            return Err(LedgerError::Internal(format!(
                "materialized pending reservations disagree with the journal \
                 ({} materialized vs {} replayed)",
                pending.len(),
                self.pending.len()
            )));
        }
        if outcomes != self.outcomes {
            return Err(LedgerError::Internal(format!(
                "materialized outcomes disagree with the journal ({} materialized vs {} replayed)",
                outcomes.len(),
                self.outcomes.len()
            )));
        }
        Ok(())
    }

    /// Load one ordered batch of journal rows starting at `from_seq`.
    fn load_journal_batch(
        &self,
        from_seq: u64,
        max: usize,
    ) -> Result<Vec<JournalRow>, LedgerError> {
        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT seq, prev_hash, ts, op_cbor, result_cbor, head_hash \
                         FROM ledger_journal WHERE seq >= $1 ORDER BY seq LIMIT $2",
                        &[&(from_seq as i64), &(max as i64)],
                    )
                    .await
                    .map_err(|e| ie(format!("journal select: {e}")))?;
                let mut recs: Vec<JournalRow> = Vec::with_capacity(rows.len());
                for r in &rows {
                    let seq_i: i64 = r.get(0);
                    let prev_b: &[u8] = r.get(1);
                    let ts_i: i64 = r.get(2);
                    let op_b: Vec<u8> = r.get(3);
                    let res_b: Vec<u8> = r.get(4);
                    let head_b: &[u8] = r.get(5);
                    recs.push(JournalRow {
                        seq: seq_i as u64,
                        prev_hash: array32(prev_b, "prev_hash")?,
                        ts: ts_i as u64,
                        op_cbor: op_b,
                        result_cbor: res_b,
                        head_hash: array32(head_b, "head_hash")?,
                    });
                }
                let mut buf = Vec::new();
                cbor(&recs, &mut buf)?;
                Ok(buf)
            })
        })?;
        ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("journal batch decode: {e}")))
    }

    /// Load `ledger_meta`'s head pointer and logical clock.
    fn load_meta(&self) -> Result<(ChainHead, u64), LedgerError> {
        let bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let head_row = client
                    .query_opt("SELECT value FROM ledger_meta WHERE key = 'head'", &[])
                    .await
                    .map_err(|e| ie(format!("head: {e}")))?;
                let clock_row = client
                    .query_opt("SELECT value FROM ledger_meta WHERE key = 'clock'", &[])
                    .await
                    .map_err(|e| ie(format!("clock: {e}")))?;
                let head_val: Option<Vec<u8>> = head_row.map(|r| r.get(0));
                let clock_val: Option<Vec<u8>> = clock_row.map(|r| r.get(0));
                let mut buf = Vec::new();
                cbor(&(head_val, clock_val), &mut buf)?;
                Ok(buf)
            })
        })?;
        let (head_val, clock_val): (Option<Vec<u8>>, Option<Vec<u8>>) =
            ciborium::from_reader(bytes.as_slice())
                .map_err(|e| LedgerError::Internal(format!("meta decode: {e}")))?;

        let head = match head_val {
            // Absent on a brand-new database: the genesis head.
            None => ChainHead::default(),
            Some(v) => decode_head(&v)?,
        };
        let clock = match clock_val {
            None => 0,
            Some(v) => {
                if v.len() != 8 {
                    return Err(LedgerError::CorruptRow(format!(
                        "clock expected 8 bytes, got {}",
                        v.len()
                    )));
                }
                let mut a = [0u8; 8];
                a.copy_from_slice(&v);
                u64::from_be_bytes(a)
            }
        };
        Ok((head, clock))
    }

    /// Load the materialized projection tables.
    #[allow(clippy::type_complexity)]
    fn load_materialized(
        &self,
    ) -> Result<
        (
            BTreeMap<AccountId, Account>,
            BTreeMap<TransferId, PendingReservation>,
            BTreeMap<TransferId, Outcome>,
        ),
        LedgerError,
    > {
        let acc_bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT id, unit_issuer, unit_resource, purpose_cbor, \
                         debits_pending, debits_posted, credits_pending, credits_posted, flags \
                         FROM ledger_accounts",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("accts: {e}")))?;
                let accounts: Vec<Account> =
                    rows.iter().map(row_to_account).collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&accounts, &mut buf)?;
                Ok(buf)
            })
        })?;
        let accounts: Vec<Account> = ciborium::from_reader(acc_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("accts decode: {e}")))?;

        let pen_bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT transfer_id, transfer_cbor, deadline, state FROM ledger_pending",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("pending: {e}")))?;
                let items: Vec<PendingReservation> =
                    rows.iter().map(row_to_pending).collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&items, &mut buf)?;
                Ok(buf)
            })
        })?;
        let items: Vec<PendingReservation> = ciborium::from_reader(pen_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("pending decode: {e}")))?;

        let out_bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT transfer_id, result_cbor, seq FROM ledger_outcomes",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("outcomes: {e}")))?;
                let pairs: Vec<(Vec<u8>, Vec<u8>, i64)> = rows
                    .iter()
                    .map(|r| {
                        let tid: Vec<u8> = r.get(0);
                        let rcbor: Vec<u8> = r.get(1);
                        let seq: i64 = r.get(2);
                        Ok((tid, rcbor, seq))
                    })
                    .collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&pairs, &mut buf)?;
                Ok(buf)
            })
        })?;
        let pairs: Vec<(Vec<u8>, Vec<u8>, i64)> = ciborium::from_reader(out_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("outcomes decode: {e}")))?;

        let mut outcomes = BTreeMap::new();
        for (tid_b, rcbor, seq_i) in pairs {
            let tid = try_u128(&tid_b, "ledger_outcomes.transfer_id")?;
            let persisted: PersistedResult = ciborium::from_reader(rcbor.as_slice())
                .map_err(|e| LedgerError::Internal(format!("outcome decode: {e}")))?;
            outcomes.insert(
                TransferId(tid),
                Outcome {
                    result: persisted,
                    seq: seq_i as u64,
                },
            );
        }

        Ok((
            accounts.into_iter().map(|a| (a.id, a)).collect(),
            items.into_iter().map(|r| (r.transfer.id, r)).collect(),
            outcomes,
        ))
    }

    /// Load the most recent signed checkpoint (F5: so `prev_checkpoint_hash`
    /// chains correctly after a restart).
    fn load_last_checkpoint(&self) -> Result<Option<SignedCheckpoint>, LedgerError> {
        let cp_bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let row = client
                    .query_opt(
                        "SELECT ledger_id, journal_seq, head_hash, balances_root, \
                         pending_root, ts, prev_checkpoint_hash, sig \
                         FROM ledger_checkpoints ORDER BY seq DESC LIMIT 1",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("cp load: {e}")))?;
                let pair: Option<SignedCheckpoint> = match row {
                    Some(r) => {
                        let ledger_id: String = r.get(0);
                        let jseq: i64 = r.get(1);
                        let head_b: &[u8] = r.get(2);
                        let bal_b: &[u8] = r.get(3);
                        let pen_b: &[u8] = r.get(4);
                        let ts: i64 = r.get(5);
                        let prev_b: &[u8] = r.get(6);
                        let sig: Vec<u8> = r.get(7);
                        Some(SignedCheckpoint {
                            ledger_id: Did(ledger_id),
                            seq: jseq as u64,
                            head_hash: array32(head_b, "head_hash")?,
                            balances_root: array32(bal_b, "balances_root")?,
                            pending_root: array32(pen_b, "pending_root")?,
                            ts: ts as u64,
                            prev_checkpoint_hash: array32(prev_b, "prev_checkpoint_hash")?,
                            sig,
                        })
                    }
                    None => None,
                };
                let mut buf = Vec::new();
                cbor(&pair, &mut buf)?;
                Ok(buf)
            })
        })?;
        ciborium::from_reader(cp_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("cp decode: {e}")))
    }

    // ─── Call the bg thread ────────────────────────────────────────────────

    fn call_db<F>(&self, job: F) -> Result<Vec<u8>, LedgerError>
    where
        F: FnOnce(&deadpool_postgres::Pool) -> Result<Vec<u8>, LedgerError> + Send + 'static,
    {
        let (reply_tx, reply_rx) = mpsc::sync_channel(0);
        self.tx
            .send(BgCmd::Job(Box::new(job), reply_tx))
            .map_err(|_| LedgerError::Internal("bg thread exited".into()))?;
        reply_rx
            .recv()
            .map_err(|_| LedgerError::Internal("bg dropped reply".into()))?
    }

    // ─── Migrations + rehydration ──────────────────────────────────────────

    /// Create the schema if absent, upgrade an existing database to the strict
    /// 128-bit representation, then gate on the schema version.
    ///
    /// Ordering matters: `ledger_schema.sql` is `CREATE TABLE IF NOT EXISTS`
    /// only, so on an existing database it is a no-op and cannot fix column
    /// types — `ledger_migrate.sql` is what actually converts legacy BIGINT
    /// columns and installs the length constraints.
    fn run_migrations(&self) -> Result<(), LedgerError> {
        let schema = LEDGER_SCHEMA.to_owned();
        let migrate = LEDGER_MIGRATE.to_owned();
        self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                client
                    .batch_execute(&schema)
                    .await
                    .map_err(|e| ie(format!("migrate schema: {e}")))?;
                client
                    .batch_execute(&migrate)
                    .await
                    .map_err(|e| ie(format!("migrate upgrade: {e}")))?;
                Ok(Vec::new())
            })
        })?;
        self.gate_schema_version()
    }

    /// Refuse to serve a database whose schema this build does not understand.
    ///
    /// A *newer* version is fatal: a forward-migrated database may carry
    /// representations this build would misread, and misreading money is worse
    /// than not starting. An older or absent version is stamped to current —
    /// `run_migrations` has just brought the catalog forward.
    fn gate_schema_version(&self) -> Result<(), LedgerError> {
        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let row = client
                    .query_opt(
                        "SELECT value FROM ledger_meta WHERE key = $1",
                        &[&META_SCHEMA_VERSION],
                    )
                    .await
                    .map_err(|e| ie(format!("schema version select: {e}")))?;
                let found: Option<u64> = match row {
                    Some(r) => {
                        let v: Vec<u8> = r.get(0);
                        if v.len() != 8 {
                            return Err(LedgerError::CorruptRow(format!(
                                "schema_version expected 8 bytes, got {}",
                                v.len()
                            )));
                        }
                        let mut a = [0u8; 8];
                        a.copy_from_slice(&v);
                        Some(u64::from_be_bytes(a))
                    }
                    None => None,
                };
                if let Some(v) = found {
                    if v > SCHEMA_VERSION {
                        return Err(LedgerError::SchemaIncompatible(format!(
                            "database is at schema version {v}, this build serves {SCHEMA_VERSION}"
                        )));
                    }
                }
                client
                    .execute(
                        "INSERT INTO ledger_meta (key, value) VALUES ($1, $2) \
                         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                        &[
                            &META_SCHEMA_VERSION,
                            &SCHEMA_VERSION.to_be_bytes().as_slice(),
                        ],
                    )
                    .await
                    .map_err(|e| ie(format!("schema version stamp: {e}")))?;
                let mut buf = Vec::new();
                cbor(&found, &mut buf)?;
                Ok(buf)
            })
        })?;
        let found: Option<u64> = ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("schema version decode: {e}")))?;
        if found != Some(SCHEMA_VERSION) {
            tracing::info!("ledger schema migrated: {:?} -> {SCHEMA_VERSION}", found);
        }
        Ok(())
    }

    // ─── The commit loop ───────────────────────────────────────────────────

    fn commit(&mut self, op: Op) -> Outcome {
        // R2-F1 / R2-F2: the DB transaction is the source of truth for both
        // idempotency AND the chain head. The mirror is a performance cache;
        // the journal seq/prev_hash are derived from the DB head inside the
        // locked transaction — never from the mirror.

        // 1. Stage against the mirror (pure). We need `staged.result` for the
        //    journal entry and `staged.deltas` for the mirror update on the
        //    commit path. The JournalEntry itself is NOT built here — the DB
        //    head may differ from the mirror head (another writer committed
        //    first), and the entry's `seq`/`prev_hash` MUST chain off the DB.
        if let Err(e) = self.poison_check() {
            return Outcome {
                result: Err(e),
                seq: self.head.seq,
            };
        }

        let staged = engine::stage(self, &op);

        let clock = self.clock;
        let deltas = staged.deltas.clone();
        let staged_result = staged.result.clone();
        let idempotency_id = op.idempotency_id();

        // The head the mirror staged against. The commit transaction requires
        // the authoritative head to still be exactly this, which is what makes
        // "these absolute counters describe the current state" checked rather
        // than assumed.
        let mirror_head = self.head;
        let lease_epoch = self.lease_epoch;
        let lease_owner = self.lease_owner;
        let lock_key = advisory_key(&self.ledger_id);
        let heartbeat = now_unix();

        // 2. Write to DB transactionally. The DB transaction (a) acquires the
        //    writer advisory lock, (b) locks + reads the authoritative head
        //    row, (c) checks idempotency, (d) on commit builds the
        //    JournalEntry from the DB head, inserts everything, and advances
        //    the head. The CBOR wire format returned to the caller is a
        //    4-tuple: `(result_cbor, seq, is_committed, new_head_hash)`.
        //    `is_committed=false` covers both replay and ambiguous-commit
        //    reconciliation — in neither case does the caller apply deltas or
        //    advance its mirror head.
        let db_result: Result<Vec<u8>, LedgerError> = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let mut client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let tx = client
                    .transaction()
                    .await
                    .map_err(|e| ie(format!("begin: {e}")))?;

                // Ordering boundary. Keyed per cell so distinct ledgers never
                // serialize against each other. Auto-releases on commit or
                // rollback.
                tx.execute("SELECT pg_advisory_xact_lock($1)", &[&lock_key])
                    .await
                    .map_err(|e| ie(format!("advisory lock: {e}")))?;

                // ── Fencing: are we still the writer? ───────────────────────
                //
                // Read inside the ordering boundary, so the answer cannot be
                // invalidated between the check and the write. If another
                // instance took the lease over, its epoch is strictly greater
                // and this commit is refused. This is what makes correctness
                // independent of connection liveness and wall-clock skew.
                let lease_row = tx
                    .query_opt(
                        "SELECT value FROM ledger_meta WHERE key = $1 FOR UPDATE",
                        &[&META_WRITER_LEASE],
                    )
                    .await
                    .map_err(|e| ie(format!("lease check: {e}")))?;
                let current = match lease_row {
                    Some(r) => {
                        let v: Vec<u8> = r.get(0);
                        WriterLease::decode(&v)?
                    }
                    None => {
                        // The lease we installed at startup is gone; treat that
                        // as having been fenced out rather than re-claiming it.
                        return Err(LedgerError::WriterLeaseLost {
                            expected: lease_epoch,
                            found: 0,
                        });
                    }
                };
                if current.epoch != lease_epoch || current.owner != lease_owner {
                    return Err(LedgerError::WriterLeaseLost {
                        expected: lease_epoch,
                        found: current.epoch,
                    });
                }

                // Lock + read the authoritative head row.
                let head_row = tx
                    .query_opt(
                        "SELECT value FROM ledger_meta WHERE key = 'head' FOR UPDATE",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("head lock: {e}")))?;
                let db_head = match head_row {
                    Some(r) => {
                        let val: Vec<u8> = r.get(0);
                        decode_head(&val)?
                    }
                    None => ChainHead::default(),
                };

                // ── Proof: the mirror described this exact head ─────────────
                //
                // Staging produced absolute counters from `mirror_head`. If the
                // authoritative head has moved, those counters describe a state
                // that no longer exists and writing them would silently discard
                // whatever advanced the head. Refuse instead.
                if db_head != mirror_head {
                    return Err(LedgerError::MirrorStale {
                        mirror_seq: mirror_head.seq,
                        db_seq: db_head.seq,
                    });
                }
                let (db_seq, db_prev_hash) = (db_head.seq, db_head.head_hash);

                // R2-F1: DB-level idempotency check inside the locked
                // transaction. If the transfer_id already has an outcome, this
                // is a replay — return the stored result verbatim and do NOT
                // insert anything. The caller learns `is_committed=false` and
                // skips both delta application and head advancement.
                if let Some(id) = idempotency_id {
                    let id_bytes = u128_to_bytes(id.0);
                    let row = tx
                        .query_opt(
                            "SELECT result_cbor, seq FROM ledger_outcomes \
                             WHERE transfer_id = $1",
                            &[&id_bytes.as_slice()],
                        )
                        .await
                        .map_err(|e| ie(format!("idem select: {e}")))?;
                    if let Some(row) = row {
                        let result_cbor: Vec<u8> = row.get(0);
                        let seq_i: i64 = row.get(1);
                        // Replay — rollback the empty transaction and surface
                        // the stored outcome with is_committed=false. The
                        // new_head_hash slot is unused on replay, so zero it.
                        let _ = tx.rollback().await;
                        let mut buf = Vec::new();
                        cbor(&(result_cbor, seq_i as u64, false, [0u8; 32]), &mut buf)?;
                        return Ok(buf);
                    }
                }

                // Build the JournalEntry from the DB head — seq/prev_hash
                // chain off the authoritative DB state, not the mirror.
                let seq = db_seq + 1;
                let prev_hash = db_prev_hash;
                let entry = JournalEntry {
                    seq,
                    prev_hash,
                    ts: clock,
                    op: op.clone(),
                    result: staged_result.clone(),
                };
                let new_head_hash = entry.hash().map_err(|e| ie(format!("entry hash: {e}")))?;

                // Serialize op + persisted result for persistence.
                // F2: PersistedResult = Result<TransferResult, LedgerError>,
                // encoded for both journal and outcomes tables.
                let mut op_cbor_buf = Vec::new();
                cbor(&op, &mut op_cbor_buf)?;
                let persisted: PersistedResult = staged_result.clone();
                let mut result_cbor_buf = Vec::new();
                cbor(&persisted, &mut result_cbor_buf)?;

                // Journal entry (no result_ok column — schema uses
                // result_cbor only).
                tx.execute(
                    "INSERT INTO ledger_journal \
                     (seq, prev_hash, ts, op_cbor, result_cbor, head_hash) \
                     VALUES ($1,$2,$3,$4,$5,$6)",
                    &[
                        &(seq as i64),
                        &prev_hash.as_slice(),
                        &(clock as i64),
                        &op_cbor_buf.as_slice(),
                        &result_cbor_buf.as_slice(),
                        &new_head_hash.as_slice(),
                    ],
                )
                .await
                .map_err(|e| ie(format!("journal: {e}")))?;

                // Deltas — upserts with BYTEA(16) IDs (F1).
                for delta in &deltas {
                    match delta {
                        engine::Delta::Account(a) => {
                            let mut purpose_cbor = Vec::new();
                            cbor(&a.purpose, &mut purpose_cbor)?;
                            let id_bytes = u128_to_bytes(a.id.0);
                            let dp = u128_to_bytes(a.debits_pending);
                            let dpo = u128_to_bytes(a.debits_posted);
                            let cp = u128_to_bytes(a.credits_pending);
                            let cpo = u128_to_bytes(a.credits_posted);
                            tx.execute(
                                "INSERT INTO ledger_accounts \
                                 (id, unit_issuer, unit_resource, purpose_cbor, \
                                  debits_pending, debits_posted, credits_pending, \
                                  credits_posted, flags) \
                                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) \
                                 ON CONFLICT (id) DO UPDATE SET \
                                  unit_issuer=EXCLUDED.unit_issuer, \
                                  unit_resource=EXCLUDED.unit_resource, \
                                  purpose_cbor=EXCLUDED.purpose_cbor, \
                                  debits_pending=EXCLUDED.debits_pending, \
                                  debits_posted=EXCLUDED.debits_posted, \
                                  credits_pending=EXCLUDED.credits_pending, \
                                  credits_posted=EXCLUDED.credits_posted, \
                                  flags=EXCLUDED.flags",
                                &[
                                    &id_bytes.as_slice(),
                                    &a.unit.issuer.as_str(),
                                    &a.unit.resource_class.as_str(),
                                    &purpose_cbor.as_slice(),
                                    &dp.as_slice(),
                                    &dpo.as_slice(),
                                    &cp.as_slice(),
                                    &cpo.as_slice(),
                                    &(a.flags.bits() as i32),
                                ],
                            )
                            .await
                            .map_err(|e| ie(format!("account: {e}")))?;
                        }
                        engine::Delta::Pending(r) => {
                            let mut transfer_cbor = Vec::new();
                            cbor(&r.transfer, &mut transfer_cbor)?;
                            let tid_bytes = u128_to_bytes(r.transfer.id.0);
                            tx.execute(
                                "INSERT INTO ledger_pending \
                                 (transfer_id, transfer_cbor, deadline, state) \
                                 VALUES ($1,$2,$3,$4) \
                                 ON CONFLICT (transfer_id) DO UPDATE SET \
                                  transfer_cbor=EXCLUDED.transfer_cbor, \
                                  deadline=EXCLUDED.deadline, \
                                  state=EXCLUDED.state",
                                &[
                                    &tid_bytes.as_slice(),
                                    &transfer_cbor.as_slice(),
                                    &(r.deadline as i64),
                                    &(r.state as i16),
                                ],
                            )
                            .await
                            .map_err(|e| ie(format!("pending: {e}")))?;
                        }
                    }
                }

                // Outcome (if idempotent).
                if let Some(id) = idempotency_id {
                    let id_bytes = u128_to_bytes(id.0);
                    tx.execute(
                        "INSERT INTO ledger_outcomes \
                         (transfer_id, result_cbor, seq) VALUES ($1,$2,$3)",
                        &[
                            &id_bytes.as_slice(),
                            &result_cbor_buf.as_slice(),
                            &(seq as i64),
                        ],
                    )
                    .await
                    .map_err(|e| ie(format!("outcome: {e}")))?;
                }

                // Outbox (settled value only).
                if matches!(
                    staged_result,
                    Ok(TransferResult::Issued) | Ok(TransferResult::Applied { .. })
                ) {
                    // transfer_id is nullable for checkpoint rows; for
                    // receipts it is always the op's idempotency id. A
                    // settled transfer is by definition idempotent.
                    let tid_opt: Option<Vec<u8>> = idempotency_id.map(|id| u128_to_bytes(id.0));
                    tx.execute(
                        "INSERT INTO ledger_outbox (kind, transfer_id, journal_seq) \
                         VALUES (0, $1, $2)",
                        &[&tid_opt, &(seq as i64)],
                    )
                    .await
                    .map_err(|e| ie(format!("outbox: {e}")))?;
                }

                // Head pointer.
                let mut head_val = Vec::with_capacity(40);
                head_val.extend_from_slice(&seq.to_be_bytes());
                head_val.extend_from_slice(&new_head_hash);
                tx.execute(
                    "INSERT INTO ledger_meta (key, value) VALUES ('head', $1) \
                     ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                    &[&head_val.as_slice()],
                )
                .await
                .map_err(|e| ie(format!("head: {e}")))?;

                // Refresh the lease heartbeat in the same transaction. An
                // actively committing writer therefore never looks abandoned,
                // and the refresh is atomic with the work it authorizes.
                let renewed = WriterLease {
                    epoch: lease_epoch,
                    owner: lease_owner,
                    heartbeat,
                };
                tx.execute(
                    "UPDATE ledger_meta SET value = $2 WHERE key = $1",
                    &[&META_WRITER_LEASE, &renewed.encode().as_slice()],
                )
                .await
                .map_err(|e| ie(format!("lease renew: {e}")))?;

                tx.commit().await.map_err(|e| ie(format!("commit: {e}")))?;

                // R2-F1: tagged wire reply — is_committed=true with the fresh
                // head hash so the caller can advance its mirror.
                let mut buf = Vec::new();
                cbor(&(result_cbor_buf, seq, true, new_head_hash), &mut buf)?;
                Ok(buf)
            })
        });

        // 3. Decode the 4-tuple wire reply. On error, attempt reconciliation:
        //    the tx may have committed but the reply was lost (ambiguous
        //    commit). A reconciled result has is_committed=false — the DB
        //    already has it, the caller did not commit this time.
        let (result_cbor_bytes, committed_seq, is_committed, new_head_hash): (
            Vec<u8>,
            u64,
            bool,
            [u8; 32],
        ) = match db_result {
            Ok(data) => match ciborium::from_reader(data.as_slice()) {
                Ok(v) => v,
                Err(e) => {
                    return Outcome {
                        result: Err(LedgerError::Internal(format!("commit decode: {e}"))),
                        seq: self.head.seq,
                    };
                }
            },
            Err(e) => {
                // An integrity violation is terminal: this instance's mirror
                // is no longer trustworthy, so it must stop writing entirely
                // rather than continue and hope.
                if matches!(
                    e,
                    LedgerError::WriterLeaseLost { .. }
                        | LedgerError::MirrorStale { .. }
                        | LedgerError::Poisoned(_)
                ) {
                    self.poison(e.to_string());
                    return Outcome {
                        result: Err(e),
                        seq: self.head.seq,
                    };
                }

                // Ambiguous commit — reconcile from DB if this op was
                // idempotent. If it was not idempotent, or reconciliation
                // finds nothing, the error was real.
                if let Some(id) = idempotency_id {
                    match self.reconcile_outcome(id) {
                        Ok(Some((cbor_vec, seq))) => {
                            // The transaction actually committed server-side;
                            // only the reply was lost. The mirror is now behind
                            // the journal, so adopt the committed state by
                            // replaying the entries this instance never applied.
                            // Reporting success while continuing to stage from
                            // pre-commit state is exactly the divergence this
                            // path used to introduce.
                            if let Err(adopt_err) = self.catch_up_from_journal() {
                                self.poison(format!(
                                    "failed to adopt an ambiguously-committed outcome: {adopt_err}"
                                ));
                                return Outcome {
                                    result: Err(adopt_err),
                                    seq: self.head.seq,
                                };
                            }
                            // `is_committed = false`: the deltas staged on THIS
                            // call must not be applied — the replay above
                            // already applied the authoritative ones.
                            (cbor_vec, seq, false, [0u8; 32])
                        }
                        Ok(None) => {
                            return Outcome {
                                result: Err(e),
                                seq: self.head.seq,
                            };
                        }
                        Err(recon_err) => {
                            return Outcome {
                                result: Err(recon_err),
                                seq: self.head.seq,
                            };
                        }
                    }
                } else {
                    return Outcome {
                        result: Err(e),
                        seq: self.head.seq,
                    };
                }
            }
        };

        // 4. Decode the persisted result.
        let persisted: PersistedResult = match ciborium::from_reader(result_cbor_bytes.as_slice()) {
            Ok(v) => v,
            Err(e) => {
                return Outcome {
                    result: Err(LedgerError::Internal(format!("result decode: {e}"))),
                    seq: self.head.seq,
                };
            }
        };

        let outcome = Outcome {
            result: persisted,
            seq: committed_seq,
        };

        // 5. R2-F1: ONLY apply deltas + advance the mirror head when this
        //    call actually committed. Replays and reconciled ambiguous
        //    commits (is_committed=false) leave the mirror untouched — the
        //    deltas were already applied by the original commit, and applying
        //    them again would double-state staged mutations.
        if is_committed {
            for delta in staged.deltas {
                match delta {
                    engine::Delta::Account(a) => {
                        self.accounts.insert(a.id, a);
                    }
                    engine::Delta::Pending(r) => {
                        self.pending.insert(r.transfer.id, r);
                    }
                }
            }
            self.head = ChainHead {
                seq: committed_seq,
                head_hash: new_head_hash,
            };
        }
        // Always mirror the outcomes cache for idempotent ops — both commit
        // and replay paths need this so a future in-process replay hits the
        // cache. The cache value is the persisted outcome (verbatim on
        // replay, freshly-computed on commit).
        if let Some(id) = idempotency_id {
            self.outcomes.insert(id, outcome.clone());
        }
        outcome
    }

    /// Refresh the lease heartbeat, verifying we still hold it.
    ///
    /// Conditional on epoch **and** owner, so a lease that was taken over is
    /// never resurrected by a stale writer's heartbeat: the `UPDATE` simply
    /// matches no row, and this instance poisons itself instead.
    fn renew_lease(&mut self) -> Result<(), LedgerError> {
        let epoch = self.lease_epoch;
        let owner = self.lease_owner;
        let key = advisory_key(&self.ledger_id);
        let now = now_unix();
        let renewed = WriterLease {
            epoch,
            owner,
            heartbeat: now,
        }
        .encode();

        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let mut client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let tx = client
                    .transaction()
                    .await
                    .map_err(|e| ie(format!("renew begin: {e}")))?;
                tx.execute("SELECT pg_advisory_xact_lock($1)", &[&key])
                    .await
                    .map_err(|e| ie(format!("renew lock: {e}")))?;
                let row = tx
                    .query_opt(
                        "SELECT value FROM ledger_meta WHERE key = $1 FOR UPDATE",
                        &[&META_WRITER_LEASE],
                    )
                    .await
                    .map_err(|e| ie(format!("renew select: {e}")))?;
                let found = match row {
                    Some(r) => {
                        let v: Vec<u8> = r.get(0);
                        WriterLease::decode(&v)?
                    }
                    None => {
                        let mut buf = Vec::new();
                        cbor(&(false, 0u64), &mut buf)?;
                        return Ok(buf);
                    }
                };
                if found.epoch != epoch || found.owner != owner {
                    let mut buf = Vec::new();
                    cbor(&(false, found.epoch), &mut buf)?;
                    return Ok(buf);
                }
                tx.execute(
                    "UPDATE ledger_meta SET value = $2 WHERE key = $1",
                    &[&META_WRITER_LEASE, &renewed.as_slice()],
                )
                .await
                .map_err(|e| ie(format!("renew update: {e}")))?;
                tx.commit()
                    .await
                    .map_err(|e| ie(format!("renew commit: {e}")))?;
                let mut buf = Vec::new();
                cbor(&(true, epoch), &mut buf)?;
                Ok(buf)
            })
        })?;

        let (held, found): (bool, u64) = ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("renew decode: {e}")))?;
        if !held {
            let err = LedgerError::WriterLeaseLost {
                expected: epoch,
                found,
            };
            self.poison(err.to_string());
            return Err(err);
        }
        Ok(())
    }

    /// Adopt committed journal entries this instance has not applied yet.
    ///
    /// Replays forward from the current mirror head and then requires the
    /// result to equal `ledger_meta.head`, so the mirror ends up exactly at the
    /// authoritative state rather than merely closer to it.
    fn catch_up_from_journal(&mut self) -> Result<(), LedgerError> {
        let from = self.head.seq.saturating_add(1);
        self.replay_journal_from(from)?;
        let (meta_head, meta_clock) = self.load_meta()?;
        if self.head != meta_head {
            return Err(LedgerError::Internal(format!(
                "catch-up left the mirror at seq {} but ledger_meta.head is seq {}",
                self.head.seq, meta_head.seq
            )));
        }
        self.clock = self.clock.max(meta_clock);
        Ok(())
    }

    /// Reconcile an ambiguous commit: check whether the given transfer id has
    /// a stored outcome in the DB. Returns `Ok(Some((cbor, seq)))` if the op
    /// was actually committed, or `Ok(None)` if it was not (the error was
    /// real). Used by [`commit`](Self::commit) when the reply from the bg
    /// thread was lost (F4).
    fn reconcile_outcome(&self, id: TransferId) -> Result<Option<(Vec<u8>, u64)>, LedgerError> {
        let id_bytes = u128_to_bytes(id.0);
        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let row = client
                    .query_opt(
                        "SELECT result_cbor, seq FROM ledger_outcomes \
                         WHERE transfer_id = $1",
                        &[&id_bytes.as_slice()],
                    )
                    .await
                    .map_err(|e| ie(format!("reconcile: {e}")))?;
                let pair: Option<(Vec<u8>, u64)> = match row {
                    Some(r) => {
                        let cbor: Vec<u8> = r.get(0);
                        let seq: i64 = r.get(1);
                        Some((cbor, seq as u64))
                    }
                    None => None,
                };
                let mut buf = Vec::new();
                cbor(&pair, &mut buf)?;
                Ok(buf)
            })
        })?;
        let result: Option<(Vec<u8>, u64)> = ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("reconcile decode: {e}")))?;
        Ok(result)
    }
}

impl engine::StateView for PostgresLedger {
    fn account(&self, id: AccountId) -> Option<&Account> {
        self.accounts.get(&id)
    }
    fn pending(&self, id: TransferId) -> Option<&PendingReservation> {
        self.pending.get(&id)
    }
    fn now(&self) -> u64 {
        self.clock
    }
}

impl Drop for PostgresLedger {
    fn drop(&mut self) {
        // Release the lease on a clean shutdown so a restart can take over
        // immediately instead of waiting out the heartbeat TTL. The epoch is
        // left intact (it only ever moves forward), so this is a liveness
        // courtesy and never weakens fencing: a crash simply skips it and the
        // TTL handles the takeover instead.
        if self.poisoned.is_none() && self.lease_epoch != 0 {
            let epoch = self.lease_epoch;
            let owner = self.lease_owner;
            let released = WriterLease {
                epoch,
                owner,
                heartbeat: 0,
            }
            .encode();
            let _ = self.call_db(move |pool| {
                let rt = rt_new()?;
                rt.block_on(async {
                    let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                    // Conditional on still being the holder: encoded epoch and
                    // owner are a prefix of the stored value, so a lease that
                    // was taken over does not match and is left alone.
                    let mut prefix = Vec::with_capacity(24);
                    prefix.extend_from_slice(&epoch.to_be_bytes());
                    prefix.extend_from_slice(&owner);
                    client
                        .execute(
                            "UPDATE ledger_meta SET value = $2 \
                             WHERE key = $1 AND substring(value from 1 for 24) = $3",
                            &[&META_WRITER_LEASE, &released.as_slice(), &prefix.as_slice()],
                        )
                        .await
                        .map_err(|e| ie(format!("lease release: {e}")))?;
                    Ok(Vec::new())
                })
            });
        }
        let _ = self.tx.send(BgCmd::Shutdown);
        if let Some(h) = self._thread.take() {
            let _ = h.join();
        }
    }
}

impl LedgerBackend for PostgresLedger {
    fn open_account(&mut self, spec: AccountSpec) -> Result<Account, LedgerError> {
        let id = spec.account_id()?;
        if let Some(existing) = self.accounts.get(&id) {
            if existing.unit != spec.unit {
                return Err(LedgerError::AccountUnitConflict { id });
            }
            return Ok(existing.clone());
        }
        let outcome = self.commit(Op::OpenAccount(Box::new(spec)));
        match outcome.result {
            Ok(_) => self
                .accounts
                .get(&id)
                .cloned()
                .ok_or_else(|| LedgerError::Internal("account absent after open".into())),
            Err(e) => Err(e),
        }
    }

    fn credit(&mut self, cap: crate::mint::MintCapability<'_>) -> Outcome {
        self.commit(Op::Credit(cap.transfer().clone()))
    }

    fn authorize_mint<'a>(
        &self,
        t: &'a IssueTransfer,
        sig: &[u8],
    ) -> Result<crate::mint::MintCapability<'a>, LedgerError> {
        crate::mint::authorize(self.mint_verifier.as_ref(), t, sig)
    }

    fn debit(&mut self, t: Transfer) -> Outcome {
        self.commit(Op::Debit(t))
    }

    fn reserve(&mut self, t: Transfer, timeout_s: u32) -> Outcome {
        self.commit(Op::Reserve {
            transfer: t,
            timeout_s,
        })
    }

    fn post(&mut self, id: TransferId, pending: TransferId, amount: Option<u128>) -> Outcome {
        self.commit(Op::Post {
            id,
            pending,
            amount,
        })
    }

    fn void(&mut self, id: TransferId, pending: TransferId) -> Outcome {
        self.commit(Op::Void { id, pending })
    }

    fn balance(&self, account: AccountId) -> Result<BalanceView, LedgerError> {
        self.accounts
            .get(&account)
            .map(|a| a.view(self.head.seq))
            .ok_or(LedgerError::UnknownAccount(account))
    }

    fn checkpoint(
        &mut self,
        signer: &dyn CheckpointSigner,
    ) -> Result<SignedCheckpoint, LedgerError> {
        let bal = balances_root(self.accounts.values())?;
        let pen = pending_root(self.pending.values())?;
        // F5: prev_checkpoint_hash comes from the rehydrated last_checkpoint,
        // not a default zero hash.
        let prev_cp = match &self.last_checkpoint {
            Some(cp) => cp.digest()?,
            None => [0u8; 32],
        };
        let content = CheckpointContent {
            ledger_id: signer.ledger_id(),
            seq: self.head.seq,
            head_hash: self.head.head_hash,
            balances_root: bal,
            pending_root: pen,
            ts: self.clock,
            prev_checkpoint_hash: prev_cp,
        };
        let sig = signer.sign(&content.signing_input()?)?;
        let cp = SignedCheckpoint {
            ledger_id: signer.ledger_id().clone(),
            seq: self.head.seq,
            head_hash: self.head.head_hash,
            balances_root: bal,
            pending_root: pen,
            ts: self.clock,
            prev_checkpoint_hash: prev_cp,
            sig: sig.clone(),
        };
        let digest = cp.digest()?;

        // F5: checkpoint insert + outbox insert in ONE DB transaction (was
        // two separate autocommit statements).
        let cp_clone = cp.clone();
        let digest_clone = digest;
        self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let mut client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let tx = client
                    .transaction()
                    .await
                    .map_err(|e| ie(format!("cp begin: {e}")))?;
                tx.execute(
                    "INSERT INTO ledger_checkpoints \
                     (ledger_id, journal_seq, head_hash, balances_root, pending_root, \
                     ts, prev_checkpoint_hash, sig, digest) \
                     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
                    &[
                        &cp_clone.ledger_id.as_str(),
                        &(cp_clone.seq as i64),
                        &cp_clone.head_hash.as_slice(),
                        &cp_clone.balances_root.as_slice(),
                        &cp_clone.pending_root.as_slice(),
                        &(cp_clone.ts as i64),
                        &cp_clone.prev_checkpoint_hash.as_slice(),
                        &cp_clone.sig.as_slice(),
                        &digest_clone.as_slice(),
                    ],
                )
                .await
                .map_err(|e| ie(format!("cp: {e}")))?;
                tx.execute(
                    "INSERT INTO ledger_outbox (kind, transfer_id, journal_seq) \
                     VALUES (1, NULL, $1)",
                    &[&(cp_clone.seq as i64)],
                )
                .await
                .map_err(|e| ie(format!("cp outbox: {e}")))?;
                tx.commit()
                    .await
                    .map_err(|e| ie(format!("cp commit: {e}")))?;
                Ok(Vec::new())
            })
        })?;

        self.last_checkpoint = Some(cp.clone());
        Ok(cp)
    }

    fn tick(&mut self, _signer: &dyn CheckpointSigner) -> Result<TickReport, LedgerError> {
        self.poison_check()?;
        // Keep the writer lease live even across idle periods, so an otherwise
        // healthy writer is not treated as abandoned merely for having nothing
        // to commit.
        self.renew_lease()?;

        let now = now_unix().max(self.clock);
        self.clock = self.clock.max(now);

        let due: Vec<TransferId> = self
            .pending
            .iter()
            .filter(|(_, r)| r.state == PendingState::Pending && self.clock >= r.deadline)
            .map(|(id, _)| *id)
            .collect();
        let mut expired = 0;
        for id in due {
            if self.commit(Op::Expire { pending: id }).is_ok() {
                expired += 1;
            }
        }

        let clock_val = self.clock.to_be_bytes().to_vec();
        self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                client
                    .execute(
                        "INSERT INTO ledger_meta (key, value) VALUES ('clock', $1) \
                         ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                        &[&clock_val.as_slice()],
                    )
                    .await
                    .map_err(|e| ie(format!("clock: {e}")))?;
                Ok(Vec::new())
            })
        })?;

        Ok(TickReport {
            expired,
            checkpointed: None,
        })
    }

    fn outbox_peek(&self, max: usize) -> Result<Vec<OutboxItem>, LedgerError> {
        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT seq, kind, transfer_id, journal_seq FROM ledger_outbox \
                         WHERE emitted = FALSE ORDER BY seq LIMIT $1",
                        &[&(max as i64)],
                    )
                    .await
                    .map_err(|e| ie(format!("peek: {e}")))?;
                let items: Vec<OutboxItem> = rows
                    .iter()
                    .map(|r| {
                        let seq: i64 = r.get(0);
                        let kind: i16 = r.get(1);
                        let tid_b: Option<Vec<u8>> = r.get(2);
                        let jseq: i64 = r.get(3);
                        Ok(OutboxItem {
                            seq: OutboxSeq(seq as u64),
                            kind: if kind == 0 {
                                OutboxKind::Receipt
                            } else {
                                OutboxKind::Checkpoint
                            },
                            transfer_id: match tid_b.as_deref() {
                                Some(b) => {
                                    Some(TransferId(try_u128(b, "ledger_outbox.transfer_id")?))
                                }
                                None => None,
                            },
                            journal_seq: jseq as u64,
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&items, &mut buf)?;
                Ok(buf)
            })
        })?;
        ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("peek decode: {e}")))
    }

    fn outbox_ack(&mut self, up_to: OutboxSeq) -> Result<(), LedgerError> {
        self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                client
                    .execute(
                        "UPDATE ledger_outbox SET emitted = TRUE WHERE seq <= $1",
                        &[&(up_to.0 as i64)],
                    )
                    .await
                    .map_err(|e| ie(format!("ack: {e}")))?;
                Ok(Vec::new())
            })
        })?;
        Ok(())
    }

    fn journal_range(&self, from_seq: u64, max: usize) -> Result<Vec<JournalEntry>, LedgerError> {
        let bytes = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT seq, prev_hash, ts, op_cbor, result_cbor \
                         FROM ledger_journal WHERE seq >= $1 ORDER BY seq LIMIT $2",
                        &[&(from_seq as i64), &(max as i64)],
                    )
                    .await
                    .map_err(|e| ie(format!("jrange: {e}")))?;
                let entries: Vec<JournalEntry> = rows
                    .iter()
                    .map(row_to_journal_entry)
                    .collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&entries, &mut buf)?;
                Ok(buf)
            })
        })?;
        ciborium::from_reader(bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("jrange decode: {e}")))
    }

    fn head(&self) -> ChainHead {
        self.head
    }
}

// ─── Background thread ──────────────────────────────────────────────────────

fn bg_main(rx: mpsc::Receiver<BgCmd>, config: &PostgresConfig) {
    let mut cfg = deadpool_postgres::Config::new();
    cfg.url = Some(config.url.clone());
    let pool = match cfg.create_pool(
        Some(deadpool_postgres::Runtime::Tokio1),
        tokio_postgres::NoTls,
    ) {
        Ok(pool) => pool,
        Err(e) => {
            tracing::error!("postgres-ledger bg thread: pool creation failed: {e}");
            return;
        }
    };

    tracing::info!("postgres-ledger bg thread started");
    while let Ok(cmd) = rx.recv() {
        match cmd {
            BgCmd::Job(job, reply) => {
                let _ = reply.send(job(&pool));
            }
            BgCmd::Shutdown => break,
        }
    }
    tracing::info!("postgres-ledger bg thread exiting");
}

// ─── Row helpers ────────────────────────────────────────────────────────────

/// Decode a 32-byte hash column (fail-closed on wrong length).
fn array32(b: &[u8], col: &str) -> Result<[u8; 32], LedgerError> {
    if b.len() != 32 {
        return Err(LedgerError::Internal(format!(
            "{col} expected 32 bytes, got {}",
            b.len()
        )));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(b);
    Ok(arr)
}

fn row_to_account(row: &tokio_postgres::Row) -> Result<Account, LedgerError> {
    // F1: all 128-bit ID/amount columns are BYTEA(16).
    let id_b: &[u8] = row.get(0);
    let issuer: &str = row.get(1);
    let resource: &str = row.get(2);
    let purpose_cbor: &[u8] = row.get(3);
    let dp: &[u8] = row.get(4);
    let dpo: &[u8] = row.get(5);
    let cp: &[u8] = row.get(6);
    let cpo: &[u8] = row.get(7);
    let flags: i32 = row.get(8);
    let purpose: crate::types::Purpose =
        ciborium::from_reader(purpose_cbor).map_err(|e| ie(format!("purpose: {e}")))?;
    Ok(Account {
        id: AccountId(try_u128(id_b, "ledger_accounts.id")?),
        unit: crate::types::UnitId {
            issuer: Did(issuer.to_owned()),
            resource_class: resource.to_owned(),
        },
        purpose,
        debits_pending: try_u128(dp, "ledger_accounts.debits_pending")?,
        debits_posted: try_u128(dpo, "ledger_accounts.debits_posted")?,
        credits_pending: try_u128(cp, "ledger_accounts.credits_pending")?,
        credits_posted: try_u128(cpo, "ledger_accounts.credits_posted")?,
        flags: crate::types::AccountFlags::from_bits_truncate(flags as u32),
    })
}

fn row_to_pending(row: &tokio_postgres::Row) -> Result<PendingReservation, LedgerError> {
    // Column 0 is transfer_id BYTEA(16) but we do not need it — the transfer
    // is fully encoded in transfer_cbor (column 1).
    let transfer_cbor: &[u8] = row.get(1);
    let deadline: i64 = row.get(2);
    let state: i16 = row.get(3);
    let transfer: Transfer = ciborium::from_reader(transfer_cbor)
        .map_err(|e| LedgerError::Internal(format!("xfer: {e}")))?;
    let ps = match state {
        0 => PendingState::Pending,
        1 => PendingState::Posted,
        2 => PendingState::Voided,
        _ => PendingState::Expired,
    };
    Ok(PendingReservation {
        transfer,
        deadline: deadline as u64,
        state: ps,
    })
}

fn row_to_journal_entry(row: &tokio_postgres::Row) -> Result<JournalEntry, LedgerError> {
    // F2: decode result_cbor as PersistedResult (the full Result). No
    // result_ok column in the schema.
    let seq: i64 = row.get(0);
    let prev_bytes: &[u8] = row.get(1);
    let ts: i64 = row.get(2);
    let op_cbor: &[u8] = row.get(3);
    let result_cbor: &[u8] = row.get(4);
    let prev = array32(prev_bytes, "prev_hash")?;
    let op: Op = ciborium::from_reader(op_cbor).map_err(|e| ie(format!("op: {e}")))?;
    let result: PersistedResult =
        ciborium::from_reader(result_cbor).map_err(|e| ie(format!("result: {e}")))?;
    Ok(JournalEntry {
        seq: seq as u64,
        prev_hash: prev,
        ts: ts as u64,
        op,
        result,
    })
}

// ─── Inline helpers (kept tiny to minimize boilerplate) ─────────────────────

fn rt_new() -> Result<tokio::runtime::Runtime, LedgerError> {
    tokio::runtime::Runtime::new().map_err(|e| LedgerError::Internal(format!("rt: {e}")))
}

fn cbor<T: serde::Serialize>(val: &T, buf: &mut Vec<u8>) -> Result<(), LedgerError> {
    ciborium::into_writer(val, buf).map_err(|e| LedgerError::Internal(format!("cbor: {e}")))
}

fn ie(msg: impl std::fmt::Display) -> LedgerError {
    LedgerError::Internal(msg.to_string())
}

/// Encode u128 as 16-byte big-endian Vec<u8> for BYTEA columns (F1).
fn u128_to_bytes(v: u128) -> Vec<u8> {
    v.to_be_bytes().to_vec()
}

/// Decode a 128-bit BYTEA column, requiring **exactly** 16 bytes.
///
/// Deliberately strict in both directions. A longer slice is not truncated:
/// truncation is how two distinct ids silently become the same id, which on a
/// money path means one transfer's outcome answering for another's. A shorter
/// slice is not zero-extended: mapping every malformed row to id 0 would
/// collapse them onto a single phantom account. Both are rejected as
/// [`LedgerError::CorruptRow`] so a legacy or damaged row stops the ledger
/// instead of quietly aliasing.
fn try_u128(b: &[u8], col: &str) -> Result<u128, LedgerError> {
    if b.len() != 16 {
        return Err(LedgerError::CorruptRow(format!(
            "{col} expected exactly 16 bytes, got {}",
            b.len()
        )));
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(b);
    Ok(u128::from_be_bytes(arr))
}

/// Decode the 40-byte `ledger_meta.head` value: `seq(8) || head_hash(32)`.
fn decode_head(b: &[u8]) -> Result<ChainHead, LedgerError> {
    if b.len() != 40 {
        return Err(LedgerError::CorruptRow(format!(
            "ledger_meta.head expected exactly 40 bytes, got {}",
            b.len()
        )));
    }
    let mut seq = [0u8; 8];
    seq.copy_from_slice(&b[..8]);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&b[8..40]);
    Ok(ChainHead {
        seq: u64::from_be_bytes(seq),
        head_hash: hash,
    })
}

/// How many journal rows to load per round-trip during replay/verification.
const JOURNAL_BATCH: usize = 1024;

/// A journal row exactly as stored, before its payloads are decoded.
///
/// Kept CBOR-serializable so it can cross the background-thread boundary in one
/// hop; `verify` is what turns it into a checked [`JournalEntry`].
#[derive(serde::Serialize, serde::Deserialize)]
struct JournalRow {
    seq: u64,
    prev_hash: [u8; 32],
    ts: u64,
    op_cbor: Vec<u8>,
    result_cbor: Vec<u8>,
    head_hash: [u8; 32],
}

impl JournalRow {
    /// Check this row against the expected position in the chain and return the
    /// decoded entry.
    ///
    /// Three independent checks: **contiguity** (the sequence is exactly the
    /// one expected — a per-entry hash check cannot notice a missing entry),
    /// **linkage** (`prev_hash` equals the previous entry's `head_hash`), and
    /// **integrity** (the entry re-hashes to the stored `head_hash`).
    fn verify(
        &self,
        expected_seq: u64,
        expected_prev: [u8; 32],
    ) -> Result<JournalEntry, LedgerError> {
        if self.seq != expected_seq {
            return Err(LedgerError::Internal(format!(
                "journal sequence not contiguous: expected seq {expected_seq}, found {}",
                self.seq
            )));
        }
        if self.prev_hash != expected_prev {
            return Err(LedgerError::Internal(format!(
                "journal chain broken at seq {}: prev_hash does not match the prior entry",
                self.seq
            )));
        }
        let op: Op = ciborium::from_reader(self.op_cbor.as_slice()).map_err(|e| {
            LedgerError::Internal(format!("journal op decode seq {}: {e}", self.seq))
        })?;
        let result: PersistedResult =
            ciborium::from_reader(self.result_cbor.as_slice()).map_err(|e| {
                LedgerError::Internal(format!("journal result decode seq {}: {e}", self.seq))
            })?;
        let entry = JournalEntry {
            seq: self.seq,
            prev_hash: self.prev_hash,
            ts: self.ts,
            op,
            result,
        };
        let recomputed = entry.hash().map_err(|e| {
            LedgerError::Internal(format!("journal hash recompute seq {}: {e}", self.seq))
        })?;
        if recomputed != self.head_hash {
            return Err(LedgerError::Internal(format!(
                "journal head_hash mismatch at seq {}",
                self.seq
            )));
        }
        Ok(entry)
    }
}
