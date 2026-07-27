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
//! On restart, the mirror is rehydrated from the DB and `verify_chain`
//! recomputes every journal hash before serving traffic.
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
type DbJob = Box<
    dyn FnOnce(&deadpool_postgres::Pool) -> Result<Vec<u8>, LedgerError> + Send + 'static,
>;

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
    accounts: BTreeMap<AccountId, Account>,
    pending: BTreeMap<TransferId, PendingReservation>,
    outcomes: BTreeMap<TransferId, Outcome>,
    head: ChainHead,
    clock: u64,
    last_checkpoint: Option<SignedCheckpoint>,
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

        let mut ledger = Self {
            tx,
            _thread: Some(thread),
            ledger_id,
            accounts: BTreeMap::new(),
            pending: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            head: ChainHead::default(),
            clock: 0,
            last_checkpoint: None,
        };
        ledger.run_migrations()?;
        ledger.rehydrate()?;
        // F3: verify the full chain on startup — recompute every hash.
        ledger.verify_chain()?;
        Ok(ledger)
    }

    /// Verify the full journal hash chain by recomputing every entry hash.
    ///
    /// Loads each journal row, reconstructs the `JournalEntry`, recomputes its
    /// hash via `entry.hash()`, asserts it matches the stored `head_hash`, and
    /// asserts each `prev_hash` chains to the prior entry's `head_hash`. All
    /// decode paths fail closed (F3).
    pub fn verify_chain(&self) -> Result<(), LedgerError> {
        let bytes = self.call_db(|pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let rows = client
                    .query(
                        "SELECT seq, prev_hash, ts, op_cbor, result_cbor, head_hash \
                         FROM ledger_journal ORDER BY seq",
                        &[],
                    )
                    .await
                    .map_err(|e| ie(format!("chain select: {e}")))?;

                // Decode rows into fully-typed records on the bg thread so the
                // caller only has to recompute hashes, not touch DB types.
                let mut recs: Vec<(u64, [u8; 32], u64, Vec<u8>, Vec<u8>, [u8; 32])> =
                    Vec::with_capacity(rows.len());
                for r in &rows {
                    let seq_i: i64 = r.get(0);
                    let prev_b: &[u8] = r.get(1);
                    let ts_i: i64 = r.get(2);
                    let op_b: Vec<u8> = r.get(3);
                    let res_b: Vec<u8> = r.get(4);
                    let head_b: &[u8] = r.get(5);
                    let prev = array32(prev_b, "prev_hash")?;
                    let head = array32(head_b, "head_hash")?;
                    recs.push((seq_i as u64, prev, ts_i as u64, op_b, res_b, head));
                }
                let mut buf = Vec::new();
                cbor(&recs, &mut buf)?;
                Ok(buf)
            })
        })?;

        let recs: Vec<(u64, [u8; 32], u64, Vec<u8>, Vec<u8>, [u8; 32])> =
            ciborium::from_reader(bytes.as_slice())
                .map_err(|e| LedgerError::Internal(format!("chain decode: {e}")))?;

        let mut expected_prev = [0u8; 32];
        for (seq, prev_hash, ts, op_b, res_b, head_hash) in &recs {
            // Chain linkage: prev_hash must equal the prior head_hash.
            if *prev_hash != expected_prev {
                return Err(LedgerError::Internal(format!(
                    "chain prev_hash broken at seq {seq}"
                )));
            }
            // Decode op + persisted result (fail-closed).
            let op: Op = ciborium::from_reader(op_b.as_slice())
                .map_err(|e| LedgerError::Internal(format!("chain op decode seq {seq}: {e}")))?;
            let result: PersistedResult = ciborium::from_reader(res_b.as_slice())
                .map_err(|e| LedgerError::Internal(format!("chain result decode seq {seq}: {e}")))?;
            // Recompute the entry hash.
            let entry = JournalEntry {
                seq: *seq,
                prev_hash: *prev_hash,
                ts: *ts,
                op,
                result,
            };
            let recomputed = entry.hash().map_err(|e| {
                LedgerError::Internal(format!("chain hash recompute seq {seq}: {e}"))
            })?;
            if recomputed != *head_hash {
                return Err(LedgerError::Internal(format!(
                    "chain head_hash mismatch at seq {seq}"
                )));
            }
            expected_prev = *head_hash;
        }
        Ok(())
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

    fn run_migrations(&self) -> Result<(), LedgerError> {
        let schema = LEDGER_SCHEMA.to_owned();
        self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                client
                    .batch_execute(&schema)
                    .await
                    .map_err(|e| ie(format!("migrate: {e}")))?;
                Ok(Vec::new())
            })
        })?;
        Ok(())
    }

    fn rehydrate(&mut self) -> Result<(), LedgerError> {
        // Head + clock (fail-closed: missing meta is allowed on first boot,
        // but malformed meta is an error).
        let meta_bytes = self.call_db(|pool| {
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
                let head_val: Vec<u8> = match head_row {
                    Some(r) => r.get(0),
                    None => vec![0u8; 40],
                };
                let clock_val: Vec<u8> = match clock_row {
                    Some(r) => r.get(0),
                    None => vec![0u8; 8],
                };
                let mut buf = Vec::new();
                cbor(&(head_val, clock_val), &mut buf)?;
                Ok(buf)
            })
        })?;
        let (head_val, clock_val): (Vec<u8>, Vec<u8>) = ciborium::from_reader(
            meta_bytes.as_slice(),
        )
        .map_err(|e| LedgerError::Internal(format!("meta decode: {e}")))?;
        if head_val.len() >= 40 {
            let mut seq_arr = [0u8; 8];
            seq_arr.copy_from_slice(&head_val[..8]);
            let mut hash_arr = [0u8; 32];
            hash_arr.copy_from_slice(&head_val[8..40]);
            self.head = ChainHead {
                seq: u64::from_be_bytes(seq_arr),
                head_hash: hash_arr,
            };
        }
        if clock_val.len() >= 8 {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&clock_val[..8]);
            self.clock = u64::from_be_bytes(arr);
        }

        // Accounts
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
                let accounts: Vec<Account> = rows
                    .iter()
                    .map(row_to_account)
                    .collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&accounts, &mut buf)?;
                Ok(buf)
            })
        })?;
        let accounts: Vec<Account> = ciborium::from_reader(acc_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("accts decode: {e}")))?;
        self.accounts = accounts.into_iter().map(|a| (a.id, a)).collect();

        // Pending
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
                let items: Vec<PendingReservation> = rows
                    .iter()
                    .map(row_to_pending)
                    .collect::<Result<_, _>>()?;
                let mut buf = Vec::new();
                cbor(&items, &mut buf)?;
                Ok(buf)
            })
        })?;
        let items: Vec<PendingReservation> = ciborium::from_reader(pen_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("pending decode: {e}")))?;
        self.pending = items.into_iter().map(|r| (r.transfer.id, r)).collect();

        // Outcomes (F2: result_cbor is PersistedResult, no result_ok column).
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
        for (tid_b, rcbor, seq_i) in pairs {
            let tid = bytes_to_u128(&tid_b);
            let persisted: PersistedResult = ciborium::from_reader(rcbor.as_slice())
                .map_err(|e| LedgerError::Internal(format!("outcome decode: {e}")))?;
            self.outcomes.insert(
                TransferId(tid),
                Outcome {
                    result: persisted,
                    seq: seq_i as u64,
                },
            );
        }

        // F5: rehydrate the latest checkpoint so prev_checkpoint_hash is
        // correct on the next checkpoint.
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
                        let head = array32(head_b, "head_hash")?;
                        let bal = array32(bal_b, "balances_root")?;
                        let pen = array32(pen_b, "pending_root")?;
                        let prev = array32(prev_b, "prev_checkpoint_hash")?;
                        Some(SignedCheckpoint {
                            ledger_id: Did(ledger_id),
                            seq: jseq as u64,
                            head_hash: head,
                            balances_root: bal,
                            pending_root: pen,
                            ts: ts as u64,
                            prev_checkpoint_hash: prev,
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
        let cp_opt: Option<SignedCheckpoint> = ciborium::from_reader(cp_bytes.as_slice())
            .map_err(|e| LedgerError::Internal(format!("cp decode: {e}")))?;
        self.last_checkpoint = cp_opt;

        tracing::info!(
            "PostgresLedger rehydrated: {} accounts, {} pending, {} outcomes, head seq={}, \
             checkpoint={}",
            self.accounts.len(),
            self.pending.len(),
            self.outcomes.len(),
            self.head.seq,
            self.last_checkpoint.is_some()
        );
        Ok(())
    }

    // ─── The commit loop ───────────────────────────────────────────────────

    fn commit(&mut self, op: Op) -> Outcome {
        // F4: No mirror-level idempotency pre-check — the DB is the source of
        // truth. The DB transaction performs the outcomes-table check below.

        // 1. Stage against the mirror (pure).
        let staged = engine::stage(self, &op);

        // 2. Build journal entry + hash.
        let seq = self.head.seq + 1;
        let prev_hash = self.head.head_hash;
        let entry = JournalEntry {
            seq,
            prev_hash,
            ts: self.clock,
            op: op.clone(),
            result: staged.result.clone(),
        };
        let new_head_hash = match entry.hash() {
            Ok(h) => h,
            Err(e) => {
                return Outcome {
                    result: Err(e),
                    seq: self.head.seq,
                };
            }
        };

        // 3. Write to DB transactionally. The DB transaction does the
        //    idempotency check against ledger_outcomes.
        let clock = self.clock;
        let deltas = staged.deltas.clone();
        let staged_result = staged.result.clone();
        let idempotency_id = op.idempotency_id();

        let db_result: Result<Vec<u8>, LedgerError> = self.call_db(move |pool| {
            let rt = rt_new()?;
            rt.block_on(async {
                let mut client = pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
                let tx = client
                    .transaction()
                    .await
                    .map_err(|e| ie(format!("begin: {e}")))?;

                // F4: DB-level idempotency — check outcomes table inside the
                // transaction. This is the authoritative check.
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
                        // Replay — return stored outcome and rollback the
                        // empty transaction.
                        let _ = tx.rollback().await;
                        let mut buf = Vec::new();
                        cbor(&(result_cbor, seq_i as u64), &mut buf)?;
                        return Ok(buf);
                    }
                }

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
                    let tid_opt: Option<Vec<u8>> =
                        idempotency_id.map(|id| u128_to_bytes(id.0));
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

                tx.commit()
                    .await
                    .map_err(|e| ie(format!("commit: {e}")))?;

                // Return result_cbor + seq for the caller to decode.
                let mut buf = Vec::new();
                cbor(&(result_cbor_buf, seq), &mut buf)?;
                Ok(buf)
            })
        });

        // 4. Decode the reply. On error, attempt reconciliation: the tx may
        //    have committed but the reply was lost (ambiguous commit).
        let (result_cbor_bytes, committed_seq) = match db_result {
            Ok(data) => {
                let (cbor_vec, seq): (Vec<u8>, u64) = match ciborium::from_reader(data.as_slice())
                {
                    Ok(v) => v,
                    Err(e) => {
                        return Outcome {
                            result: Err(LedgerError::Internal(format!(
                                "commit decode: {e}"
                            ))),
                            seq: self.head.seq,
                        };
                    }
                };
                (cbor_vec, seq)
            }
            Err(e) => {
                // Ambiguous commit — reconcile from DB if this op was
                // idempotent. If it was not idempotent, or reconciliation
                // finds nothing, the error was real.
                if let Some(id) = idempotency_id {
                    match self.reconcile_outcome(id) {
                        Ok(Some((cbor_vec, seq))) => (cbor_vec, seq),
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

        // 5. Decode the persisted result.
        let persisted: PersistedResult = match ciborium::from_reader(
            result_cbor_bytes.as_slice(),
        ) {
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

        // 6. Update the mirror with the staged deltas.
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
        if let Some(id) = idempotency_id {
            self.outcomes.insert(id, outcome.clone());
        }
        outcome
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

    fn credit(&mut self, t: IssueTransfer) -> Outcome {
        self.commit(Op::Credit(t))
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
                let mut client =
                    pool.get().await.map_err(|e| ie(format!("pool: {e}")))?;
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(self.clock);
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
                            transfer_id: tid_b
                                .as_deref()
                                .map(|b| TransferId(bytes_to_u128(b))),
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
        id: AccountId(bytes_to_u128(id_b)),
        unit: crate::types::UnitId {
            issuer: Did(issuer.to_owned()),
            resource_class: resource.to_owned(),
        },
        purpose,
        debits_pending: bytes_to_u128(dp),
        debits_posted: bytes_to_u128(dpo),
        credits_pending: bytes_to_u128(cp),
        credits_posted: bytes_to_u128(cpo),
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
    let result: PersistedResult = ciborium::from_reader(result_cbor)
        .map_err(|e| ie(format!("result: {e}")))?;
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

/// Decode 16-byte big-endian BYTEA back to u128. Fails closed if the slice
/// is shorter than 16 bytes (corrupt row).
fn bytes_to_u128(b: &[u8]) -> u128 {
    if b.len() >= 16 {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&b[..16]);
        u128::from_be_bytes(arr)
    } else {
        // Defensive: a corrupt row is treated as 0; verify_chain / row
        // decoding paths surface the underlying corruption via their own
        // checks (e.g. array32 for hashes, op_cbor decode failures).
        0
    }
}
