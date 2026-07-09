//! `MemLedger` — the in-memory reference implementation.
//!
//! Two jobs (plan §3.3):
//! 1. The **WASM-capable** backend: a browser cell / Wanix guest can run a real
//!    ledger for local-first quota. It uses no tokio, no I/O, and no wall clock —
//!    the logical clock is driven explicitly via [`MemLedger::advance_clock`], so
//!    behaviour is fully deterministic and reproducible.
//! 2. The **proptest oracle**: every other backend is equivalence-tested against
//!    it, so its commit loop is the canonical interpretation of `engine::stage`.
//!
//! It keeps *everything* — journal, outcomes, terminal reservations, outbox — in
//! memory forever (outcome-retention pruning is a later work item). The commit
//! loop mirrors the RocksLedger one (idempotency-check → stage → single atomic
//! apply → advance head), so the two stay observably identical.

use std::collections::BTreeMap;

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

/// An in-memory single-writer double-entry ledger.
#[derive(Debug)]
pub struct MemLedger {
    ledger_id: Did,
    accounts: BTreeMap<AccountId, Account>,
    pending: BTreeMap<TransferId, PendingReservation>,
    outcomes: BTreeMap<TransferId, Outcome>,
    journal: Vec<JournalEntry>,
    outbox: BTreeMap<u64, OutboxItem>,
    outbox_next: u64,
    head: ChainHead,
    clock: u64,
    last_checkpoint: Option<SignedCheckpoint>,
}

impl MemLedger {
    /// Create an empty ledger for the given cell identity, logical clock at 0.
    pub fn new(ledger_id: Did) -> Self {
        MemLedger {
            ledger_id,
            accounts: BTreeMap::new(),
            pending: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            journal: Vec::new(),
            outbox: BTreeMap::new(),
            outbox_next: 0,
            head: ChainHead::default(),
            clock: 0,
            last_checkpoint: None,
        }
    }

    /// The cell identity.
    pub fn ledger_id(&self) -> &Did {
        &self.ledger_id
    }

    /// Advance the logical commit clock to at least `to` (monotone — the clock
    /// never regresses). This is the WASM-safe substitute for a wall clock:
    /// tests and the actor drive time explicitly, which is what makes the
    /// expiry-vs-post race deterministic.
    pub fn advance_clock(&mut self, to: u64) {
        self.clock = self.clock.max(to);
    }

    /// The current logical clock.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// The full journal (reference/oracle inspection).
    pub fn journal(&self) -> &[JournalEntry] {
        &self.journal
    }

    /// Iterator over all accounts (for conservation checks in tests).
    pub fn accounts(&self) -> impl Iterator<Item = &Account> {
        self.accounts.values()
    }

    /// The last signed checkpoint, if any.
    pub fn last_checkpoint(&self) -> Option<&SignedCheckpoint> {
        self.last_checkpoint.as_ref()
    }

    // --- the serialized commit loop (plan §2a, in-memory) ---

    /// Apply one op atomically: idempotency-check → stage → apply deltas + append
    /// journal + record outcome + stage outbox, all as one logical commit.
    fn commit(&mut self, op: Op) -> Outcome {
        // 1. Idempotency FIRST (plan §2c): a replay returns the ORIGINAL outcome,
        //    including original errors, without a new journal entry.
        if let Some(id) = op.idempotency_id() {
            if let Some(prior) = self.outcomes.get(&id) {
                return prior.clone();
            }
        }

        // 2. Stage against current state (pure).
        let staged = engine::stage(self, &op);

        // 3. Build the journal entry and its hash BEFORE mutating, so an
        //    (unreachable) encoding failure aborts cleanly with no state change.
        let seq = self.head.seq + 1;
        let entry = JournalEntry {
            seq,
            prev_hash: self.head.head_hash,
            ts: self.clock,
            op: op.clone(),
            result: staged.result.clone(),
        };
        let head_hash = match entry.hash() {
            Ok(h) => h,
            Err(e) => {
                return Outcome {
                    result: Err(e),
                    seq: self.head.seq,
                }
            }
        };

        // 4. Apply deltas + advance the chain (the atomicity unit).
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
        self.journal.push(entry);
        self.head = ChainHead { seq, head_hash };

        let outcome = Outcome {
            result: staged.result,
            seq,
        };

        // 5. Record the outcome (successes AND rejections — plan §2c) and stage a
        //    receipt for settled value (posted spends / issuance, plan §2e).
        if let Some(id) = op.idempotency_id() {
            self.outcomes.insert(id, outcome.clone());
        }
        if matches!(
            outcome.result,
            Ok(TransferResult::Issued) | Ok(TransferResult::Applied { .. })
        ) {
            self.enqueue_outbox(OutboxKind::Receipt, op.idempotency_id(), seq);
        }

        outcome
    }

    fn enqueue_outbox(
        &mut self,
        kind: OutboxKind,
        transfer_id: Option<TransferId>,
        journal_seq: u64,
    ) {
        let seq = OutboxSeq(self.outbox_next);
        self.outbox_next += 1;
        self.outbox.insert(
            seq.0,
            OutboxItem {
                seq,
                kind,
                transfer_id,
                journal_seq,
            },
        );
    }
}

impl engine::StateView for MemLedger {
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

impl LedgerBackend for MemLedger {
    fn open_account(&mut self, spec: AccountSpec) -> Result<Account, LedgerError> {
        let id = spec.account_id()?;
        // Fast path: idempotent return of an existing account, with NO journal
        // growth (the property the idempotency proptest checks for opens).
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
                .ok_or_else(|| LedgerError::Internal("account absent after open".to_owned())),
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
        let balances = balances_root(self.accounts.values())?;
        let pending = pending_root(self.pending.values())?;
        let prev_checkpoint_hash = match &self.last_checkpoint {
            Some(cp) => cp.digest()?,
            None => [0u8; 32],
        };
        let content = CheckpointContent {
            ledger_id: signer.ledger_id(),
            seq: self.head.seq,
            head_hash: self.head.head_hash,
            balances_root: balances,
            pending_root: pending,
            ts: self.clock,
            prev_checkpoint_hash,
        };
        let sig = signer.sign(&content.signing_input()?)?;
        let cp = SignedCheckpoint {
            ledger_id: signer.ledger_id().clone(),
            seq: self.head.seq,
            head_hash: self.head.head_hash,
            balances_root: balances,
            pending_root: pending,
            ts: self.clock,
            prev_checkpoint_hash,
            sig,
        };
        self.last_checkpoint = Some(cp.clone());
        self.enqueue_outbox(OutboxKind::Checkpoint, None, self.head.seq);
        Ok(cp)
    }

    fn tick(&mut self, _signer: &dyn CheckpointSigner) -> Result<TickReport, LedgerError> {
        // Expiry sweep (plan §2b.4): every open reservation whose deadline the
        // logical clock has reached is expired, one journal entry each. (Scheduled
        // checkpoint cadence is a later work item; explicit `checkpoint` covers
        // the skeleton.)
        let now = self.clock;
        let due: Vec<TransferId> = self
            .pending
            .iter()
            .filter(|(_, r)| r.state == PendingState::Pending && now >= r.deadline)
            .map(|(id, _)| *id)
            .collect();
        let mut expired = 0usize;
        for id in due {
            if self.commit(Op::Expire { pending: id }).is_ok() {
                expired += 1;
            }
        }
        Ok(TickReport {
            expired,
            checkpointed: None,
        })
    }

    fn outbox_peek(&self, max: usize) -> Result<Vec<OutboxItem>, LedgerError> {
        Ok(self.outbox.values().take(max).cloned().collect())
    }

    fn outbox_ack(&mut self, up_to: OutboxSeq) -> Result<(), LedgerError> {
        self.outbox.retain(|&seq, _| seq > up_to.0);
        Ok(())
    }

    fn journal_range(&self, from_seq: u64, max: usize) -> Result<Vec<JournalEntry>, LedgerError> {
        Ok(self
            .journal
            .iter()
            .filter(|e| e.seq >= from_seq)
            .take(max)
            .cloned()
            .collect())
    }

    fn head(&self) -> ChainHead {
        self.head
    }
}
